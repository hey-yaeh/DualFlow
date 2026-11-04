"""Export a drift-corrected facial-expression video on the aligned interaction clock.

The authoritative physical interval comes from an immutable Native-reference
trim.  An accepted shared-audio Time Sync artifact maps the external facial
video clock into that trim's Native ego-video PTS clock.  This service samples
facial frames on the *reference* clock (thereby correcting clock drift),
retimes the facial audio by the same affine scale, and writes an explicit
frame-level audit map.  It never re-estimates synchronization or trim bounds.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from oculostudio import __version__
from oculostudio.adapters.native_reference_clock import round_ratio, source_stamp
from oculostudio.adapters.video import frame_timestamps_ns
from oculostudio.domain.models import ArtifactKind, TimeMap
from oculostudio.domain.timegraph import apply_time_map, invert_time_map
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.aligned_clip_export import _trim_inputs
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.audio_video_sync import (
    _probe_streams,
    _resolve_video_input,
    _validate_candidate_sources,
    _validate_role_context,
)
from oculostudio.services.privacy import _output_writer
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import (
    CancellationToken,
    atomic_write_json,
    emit,
    new_id,
    safe_name,
    sha256_file,
    utc_now,
)


ALIGNED_FACIAL_EXPORT_ENGINE = "accepted_audio_timegraph_reference_clock_resample_v1"
ALIGNED_FACIAL_EXPORT_SCHEMA = "oculostudio-aligned-facial-video-1.0"
SUPPORTED_OUTPUT_FPS = (24, 25, 30, 50, 60)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {label}: {exc}") from exc


def _sync_inputs(
    store: ProjectStore,
    *,
    session_id: str,
    time_sync_artifact_id: str,
    expected_native_asset_id: str,
    expected_role: str,
):
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    artifact = session.artifacts.get(time_sync_artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.TIME_SYNC:
        raise ValidationError("Select an accepted shared-audio Time Sync artifact")
    sync_type = str((artifact.parameters or {}).get("sync_type") or (artifact.qc or {}).get("sync_type") or "")
    if sync_type != "shared_audio_video":
        raise ValidationError("The selected Time Sync artifact is not a shared-audio facial↔ego mapping")
    if not bool((artifact.parameters or {}).get("visual_review_confirmed") or (artifact.qc or {}).get("visual_review_confirmed")):
        raise ValidationError("Shared-audio synchronization must be visually reviewed and accepted before facial export")
    if not bool((artifact.qc or {}).get("sync_verified")):
        raise ValidationError("Shared-audio Time Sync is not marked verified")

    candidate_path = find_artifact_file(artifact, names=["audio_video_sync_candidate.json"])
    accepted_path = find_artifact_file(artifact, names=["audio_video_sync_accepted.json"])
    candidate = _read_json(candidate_path, "audio_video_sync_candidate.json")
    accepted = _read_json(accepted_path, "audio_video_sync_accepted.json")
    if candidate.get("session_id") != session_id:
        raise ValidationError("Shared-audio Time Sync belongs to another session")
    role = str(candidate.get("wearer_role_id") or (artifact.parameters or {}).get("wearer_role_id") or "").upper()
    if expected_role and role and role != expected_role:
        raise ValidationError(f"Shared-audio Time Sync is for {role}, not the active role {expected_role}")
    source_id = str(candidate.get("source_asset_id") or "")
    reference_id = str(candidate.get("reference_asset_id") or "")
    if reference_id != expected_native_asset_id:
        raise ValidationError(
            "The accepted shared-audio reference is not the same original Native ego recording used by this aligned clip"
        )
    if source_id not in artifact.source_asset_ids or reference_id not in artifact.source_asset_ids:
        raise ValidationError("Shared-audio artifact lineage does not contain both synchronized video assets")
    _validate_role_context(
        store,
        session_id=session_id,
        source_asset_id=source_id,
        reference_asset_id=reference_id,
        wearer_role_id=expected_role or role or None,
    )
    _validate_candidate_sources(store, candidate)

    mapping_id = str(
        (artifact.parameters or {}).get("time_map_id")
        or (artifact.qc or {}).get("time_map_id")
        or (accepted.get("time_map") or {}).get("id")
        or ""
    )
    mapping = session.time_maps.get(mapping_id)
    if mapping is None:
        raise ValidationError("The accepted shared-audio TimeGraph mapping is missing from this project")
    source_domain = str(candidate.get("source_domain_id") or "")
    reference_domain = str(candidate.get("reference_domain_id") or "")
    if mapping.source_domain_id != source_domain or mapping.target_domain_id != reference_domain:
        raise ValidationError("The accepted TimeGraph mapping no longer matches the shared-audio evidence")
    if not math.isfinite(float(mapping.scale)) or float(mapping.scale) <= 0:
        raise ValidationError("Shared-audio TimeGraph has an invalid clock scale")

    source_video = _resolve_video_input(
        store,
        session_id=session_id,
        asset_id=source_id,
        requested_member=candidate.get("source_video_member"),
    )
    reference_video = _resolve_video_input(
        store,
        session_id=session_id,
        asset_id=reference_id,
        requested_member=candidate.get("reference_video_member"),
    )
    if source_video.stamp != candidate.get("source_stamp"):
        raise ValidationError("Facial video changed after shared-audio synchronization; rerun synchronization")
    if reference_video.stamp != candidate.get("reference_stamp"):
        raise ValidationError("Ego video changed after shared-audio synchronization; rerun synchronization")
    return artifact, candidate, accepted, mapping, source_video, reference_video, role


def _source_clock_cache(
    store: ProjectStore,
    *,
    video,
    ffprobe_path: str | None,
) -> tuple[np.ndarray, int, dict[str, Any], Path]:
    key = hashlib.sha256(
        json.dumps(
            {
                "engine": ALIGNED_FACIAL_EXPORT_ENGINE,
                "asset_id": video.asset_id,
                "source_stamp": video.stamp,
                "member": video.video_member,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:28]
    root = store.cache_dir() / "aligned_facial_video_clock"
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{key}.npz"
    meta_path = root / f"{key}.json"
    if npz_path.is_file() and meta_path.is_file():
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                pts = np.asarray(data["frame_pts_ns"], dtype=np.int64)
                end_ns = int(data["end_media_ns"])
            metadata = _read_json(meta_path, "facial-video clock cache")
            if len(pts) and end_ns > int(pts[-1]):
                return pts, end_ns, metadata, npz_path
        except Exception:
            npz_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    probe = _probe_streams(Path(video.video_path), ffprobe_path=ffprobe_path)
    raw = frame_timestamps_ns(video.video_path, ffprobe_path=ffprobe_path)
    video_start_ns = int(round(float(probe.get("video_start_sec") or 0.0) * 1e9))
    pts = np.asarray(raw, dtype=np.int64) - np.int64(video_start_ns)
    if not len(pts):
        raise ValidationError("Facial video contains no decodable frame timestamps")
    if np.any(np.diff(pts) < 0):
        raise ValidationError("Facial-video frame PTS are not monotonic")
    positive = np.diff(pts)
    positive = positive[positive > 0]
    median_step = int(np.median(positive)) if len(positive) else 33_333_333
    duration_sec = probe.get("video_duration_sec")
    duration_ns = int(round(float(duration_sec) * 1e9)) if duration_sec is not None else 0
    end_ns = max(duration_ns, int(pts[-1]) + max(1, median_step))
    metadata = {
        **probe,
        "source_video_path": str(video.video_path),
        "video_start_ns": int(video_start_ns),
        "frame_count": int(len(pts)),
        "first_relative_pts_ns": int(pts[0]),
        "last_relative_pts_ns": int(pts[-1]),
        "end_media_ns": int(end_ns),
        "median_frame_duration_ns": int(median_step),
        "clock_policy": "ffprobe_frame_pts_minus_video_stream_start",
    }
    pending = npz_path.with_suffix(".npz.partial")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, frame_pts_ns=pts, end_media_ns=np.asarray(end_ns, dtype=np.int64))
    pending.replace(npz_path)
    atomic_write_json(meta_path, metadata)
    return pts, end_ns, metadata, npz_path


def _atempo_chain(factor: float) -> str:
    if not math.isfinite(factor) or factor <= 0:
        raise ValidationError("Audio retiming factor is invalid")
    values: list[float] = []
    remaining = float(factor)
    while remaining < 0.5:
        values.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        values.append(2.0)
        remaining /= 2.0
    values.append(remaining)
    return ",".join(f"atempo={value:.12f}" for value in values)


def _mux_retimed_audio(
    *,
    silent_video: Path,
    source_video: Path,
    output_video: Path,
    source_start_sec: float,
    source_duration_sec: float,
    physical_duration_sec: float,
    output_duration_sec: float,
    audio_tempo_factor: float,
    preserve_audio: bool,
    crf: int,
    preset: str,
    ffmpeg_path: str,
    expected_frames: int,
    cancellation: CancellationToken,
    progress,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video),
    ]
    if preserve_audio:
        # Decode a small guard after the mapped interval. The audited filters
        # trim the retimed signal back to the exact physical/output durations.
        command += [
            "-ss",
            f"{max(0.0, source_start_sec):.9f}",
            "-t",
            f"{max(0.001, source_duration_sec + 0.25):.9f}",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-filter:a",
            (
                f"{_atempo_chain(audio_tempo_factor)},"
                f"atrim=start=0:end={physical_duration_sec:.9f},"
                f"apad=whole_dur={output_duration_sec:.9f},"
                f"atrim=start=0:end={output_duration_sec:.9f},asetpts=N/SR/TB"
            ),
        ]
    else:
        command += ["-map", "0:v:0"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
    ]
    if preserve_audio:
        command += ["-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    pending = output_video.with_name(output_video.stem + ".encoding" + output_video.suffix)
    command += ["-movflags", "+faststart", str(pending)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr, creationflags=flags)
        started = time.monotonic()
        next_notice = started + 2.0
        try:
            while process.poll() is None:
                cancellation.raise_if_cancelled()
                if time.monotonic() - started > 7200:
                    raise ProcessingError("Aligned facial-video encoding exceeded the two-hour safety limit")
                if time.monotonic() >= next_notice:
                    emit(progress, 0.84, f"Encoding aligned facial video · {int(time.monotonic()-started)} s elapsed")
                    next_notice += 2.0
                time.sleep(0.1)
            cancellation.raise_if_cancelled()
            if process.returncode:
                stderr.seek(max(0, stderr.tell() - 16000))
                detail = stderr.read().decode("utf-8", errors="replace")
                raise ProcessingError(f"FFmpeg aligned facial-video encoding failed: {detail}")
            if not pending.is_file() or pending.stat().st_size <= 0:
                raise ProcessingError("FFmpeg returned no aligned facial video")
            check = cv2.VideoCapture(str(pending))
            try:
                encoded = int(check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if not check.isOpened() or encoded != expected_frames:
                    raise ProcessingError(
                        f"Encoded frame count {encoded} differs from aligned facial frame map {expected_frames}"
                    )
            finally:
                check.release()
            os.replace(pending, output_video)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            pending.unlink(missing_ok=True)
    return command


def _decode_reference_clock_video(
    *,
    source_video: Path,
    source_pts_ns: np.ndarray,
    source_end_ns: int,
    mapping: TimeMap,
    reference_start_ns: int,
    reference_end_ns: int,
    parent_frame_map: Path,
    output_fps: int,
    staging: Path,
    output_label: str,
    preserve_audio: bool,
    crf: int,
    preset: str,
    ffmpeg_path: str,
    token: CancellationToken,
    progress,
) -> tuple[Path, Path, dict[str, Any], list[str]]:
    physical_duration_ns = int(reference_end_ns - reference_start_ns)
    frame_count = int(math.ceil(physical_duration_ns * int(output_fps) / 1_000_000_000))
    if frame_count <= 0:
        raise ValidationError("Aligned interaction interval contains no facial-video output frames")
    output_pts = np.asarray(
        [round_ratio(index * 1_000_000_000, int(output_fps)) for index in range(frame_count)],
        dtype=np.int64,
    )
    reference_times = np.int64(reference_start_ns) + output_pts
    inverse = invert_time_map(mapping)
    source_queries = np.asarray(apply_time_map(inverse, reference_times), dtype=np.int64)
    source_start_ns = int(apply_time_map(inverse, int(reference_start_ns)))
    source_end_query_ns = int(apply_time_map(inverse, int(reference_end_ns)))
    if source_end_query_ns <= source_start_ns:
        raise ValidationError("Accepted shared-audio map produced an empty facial-video interval")

    positive = np.diff(source_pts_ns)
    positive = positive[positive > 0]
    median_step = int(np.median(positive)) if len(positive) else 33_333_333
    tolerance = max(50_000_000, int(1.5 * median_step))
    first_pts = int(source_pts_ns[0])
    if source_start_ns < first_pts - tolerance:
        raise ValidationError(
            f"Facial video begins {((first_pts-source_start_ns)/1e9):.3f} s after the aligned interaction start"
        )
    if source_end_query_ns > int(source_end_ns) + tolerance:
        raise ValidationError(
            f"Facial video ends {((source_end_query_ns-int(source_end_ns))/1e9):.3f} s before the aligned interaction end"
        )
    source_queries = np.clip(source_queries, first_pts, int(source_end_ns) - 1)
    source_indices = np.searchsorted(source_pts_ns, source_queries, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(source_pts_ns) - 1).astype(np.int64)
    if np.any(np.diff(source_indices) < 0):
        raise ValidationError("Facial frame selection is not monotonic")

    parent = pd.read_csv(parent_frame_map, low_memory=False)
    required = {"native_frame_index", "native_video_pts_ns"}
    if not required.issubset(parent.columns):
        raise ValidationError("Native-reference trim frame map is missing required PTS columns")
    native_pts = pd.to_numeric(parent["native_video_pts_ns"], errors="raise").astype("int64").to_numpy()
    native_frames = pd.to_numeric(parent["native_frame_index"], errors="raise").astype("int64").to_numpy()
    reference_indices = np.searchsorted(native_pts, reference_times, side="right") - 1
    reference_indices = np.clip(reference_indices, 0, len(native_pts) - 1).astype(np.int64)

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ProcessingError(f"Cannot decode facial-expression video: {source_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise ProcessingError("Facial-video dimensions are unavailable")
    if decoded_count and int(source_indices[-1]) >= decoded_count:
        capture.release()
        raise ValidationError("Facial frame timestamp index exceeds the decodable video")

    safe = safe_name(output_label, fallback="aligned_facial_video")
    silent = staging / "aligned_facial_video_silent.mp4"
    output = staging / f"{safe}.mp4"
    frame_map = staging / "aligned_facial_frame_map.csv"
    writer = _output_writer(silent, width, height, float(output_fps))
    targets: dict[int, list[int]] = {}
    for output_index, source_index in enumerate(source_indices.tolist()):
        targets.setdefault(int(source_index), []).append(int(output_index))
    first_source, last_source = int(source_indices[0]), int(source_indices[-1])
    rows: list[dict[str, Any]] = []
    written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_source)
        for source_index in range(first_source, last_source + 1):
            token.raise_if_cancelled()
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ProcessingError(f"Facial-video decode ended at frame {source_index}")
            for output_index in targets.get(source_index, []):
                writer.write(frame)
                source_frame_start = int(source_pts_ns[source_index])
                source_frame_end = (
                    int(source_pts_ns[source_index + 1])
                    if source_index + 1 < len(source_pts_ns)
                    else int(source_end_ns)
                )
                native_row = int(reference_indices[output_index])
                rows.append(
                    {
                        "output_frame_index": int(output_index),
                        "output_pts_ns": int(output_pts[output_index]),
                        "interaction_clip_time_ns": int(output_pts[output_index]),
                        "reference_ego_media_time_ns": int(reference_times[output_index]),
                        "reference_native_frame_index": int(native_frames[native_row]),
                        "reference_native_frame_pts_ns": int(native_pts[native_row]),
                        "facial_query_time_ns": int(source_queries[output_index]),
                        "facial_source_frame_index": int(source_index),
                        "facial_source_frame_pts_ns": source_frame_start,
                        "facial_source_frame_end_pts_ns": source_frame_end,
                        "facial_selection_error_ns": int(source_queries[output_index] - source_frame_start),
                        "facial_source_frame_duplicated": bool(len(targets.get(source_index, [])) > 1),
                        "facial_boundary_frame": bool(
                            source_frame_start < source_start_ns or source_frame_end > source_end_query_ns
                        ),
                        "time_map_id": mapping.id,
                        "time_map_scale": float(mapping.scale),
                    }
                )
                written += 1
            if source_index % 20 == 0:
                emit(
                    progress,
                    0.16 + 0.50 * (source_index - first_source + 1) / max(1, last_source - first_source + 1),
                    f"Resampling facial frames on aligned clock ({written}/{frame_count})",
                )
    finally:
        capture.release()
        writer.release()
    if written != frame_count:
        raise ProcessingError(f"Wrote {written} aligned facial frames but expected {frame_count}")
    pd.DataFrame(rows).to_csv(frame_map, index=False)

    physical_duration_sec = physical_duration_ns / 1e9
    output_duration_sec = frame_count / float(output_fps)
    source_duration_sec = (source_end_query_ns - source_start_ns) / 1e9
    audio_tempo_factor = source_duration_sec / physical_duration_sec
    emit(progress, 0.70, "Encoding clean drift-corrected facial video")
    command = _mux_retimed_audio(
        silent_video=silent,
        source_video=source_video,
        output_video=output,
        source_start_sec=source_start_ns / 1e9,
        source_duration_sec=source_duration_sec,
        physical_duration_sec=physical_duration_sec,
        output_duration_sec=output_duration_sec,
        audio_tempo_factor=audio_tempo_factor,
        preserve_audio=preserve_audio,
        crf=crf,
        preset=preset,
        ffmpeg_path=ffmpeg_path,
        expected_frames=frame_count,
        cancellation=token,
        progress=progress,
    )
    silent.unlink(missing_ok=True)
    unique = len(set(int(value) for value in source_indices.tolist()))
    stats = {
        "output_fps": int(output_fps),
        "output_frame_count": int(frame_count),
        "output_duration_sec": float(output_duration_sec),
        "physical_interval_duration_sec": float(physical_duration_sec),
        "mapped_source_start_sec": float(source_start_ns / 1e9),
        "mapped_source_end_sec": float(source_end_query_ns / 1e9),
        "mapped_source_duration_sec": float(source_duration_sec),
        "width": int(width),
        "height": int(height),
        "first_facial_source_frame_index": int(source_indices[0]),
        "last_facial_source_frame_index": int(source_indices[-1]),
        "unique_facial_source_frames": int(unique),
        "duplicated_output_frames": int(frame_count - unique),
        "dropped_source_frames_inside_span": int((last_source - first_source + 1) - unique),
        "clock_scale_source_to_reference": float(mapping.scale),
        "clock_drift_ppm": float((mapping.scale - 1.0) * 1e6),
        "audio_tempo_factor": float(audio_tempo_factor),
        "video_drift_corrected": True,
        "audio_drift_corrected": bool(preserve_audio),
        "preserve_audio_requested": bool(preserve_audio),
    }
    return output, frame_map, stats, command


def _report_html(payload: dict[str, Any]) -> str:
    video = payload.get("video") or {}
    sync = payload.get("synchronization") or {}
    return f"""<!doctype html><html><meta charset='utf-8'><title>Aligned facial video</title>
<style>body{{font:15px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#2b2e20}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7d8c8;padding:9px;text-align:left}}th{{background:#eef2d5}}code{{background:#f1f1e9;padding:2px 5px;overflow-wrap:anywhere}}</style>
<h1>OculoStudio v{__version__} — Aligned facial-expression video</h1>
<p>The physical interval is inherited from the immutable C3D-defined Native-reference trim. The accepted shared-audio affine TimeGraph is inverted for every output frame; no start/end or synchronization parameter is re-estimated.</p>
<table><tbody>
<tr><th>Role</th><td>{html.escape(str(payload.get('wearer_role_id') or '—'))}</td></tr>
<tr><th>Facial source</th><td>{html.escape(str(payload.get('source_asset_label') or '—'))}</td></tr>
<tr><th>Ego reference</th><td>{html.escape(str(payload.get('reference_asset_label') or '—'))}</td></tr>
<tr><th>Physical interval</th><td>{payload['reference_window']['duration_sec']:.9f} s</td></tr>
<tr><th>Mapped facial interval</th><td>{video.get('mapped_source_start_sec','—')} → {video.get('mapped_source_end_sec','—')} s</td></tr>
<tr><th>Clock drift</th><td>{video.get('clock_drift_ppm','—')} ppm</td></tr>
<tr><th>Visual A/V correction</th><td>{sync.get('visual_offset_adjustment_ms','—')} ms</td></tr>
<tr><th>Output</th><td>{video.get('output_frame_count','—')} frames @ {video.get('output_fps','—')} fps</td></tr>
<tr><th>Audio retime</th><td>{video.get('audio_tempo_factor','—')}× tempo factor</td></tr>
</tbody></table>
<h2>Frame-level audit</h2><p><code>aligned_facial_frame_map.csv</code> links every output frame to interaction-local time, Native ego frame/PTS, the queried facial time, and the selected original facial frame/PTS.</p>
<h2>Interpretation</h2><ul><li>The output is clean: no QC slate, gaze, face box or AOI overlay is added.</li><li>Video drift is corrected by reference-clock resampling; audio drift is corrected by the same affine scale when audio preservation is enabled.</li><li>The original facial and ego recordings are unchanged.</li></ul></html>"""


def export_aligned_facial_video(
    store: ProjectStore,
    *,
    session_id: str,
    native_trim_artifact_id: str,
    time_sync_artifact_id: str,
    label: str = "Aligned facial video",
    output_fps: int = 30,
    preserve_audio: bool = True,
    crf: int = 18,
    preset: str = "fast",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    if output_fps not in SUPPORTED_OUTPUT_FPS:
        raise ValidationError(f"Output FPS must be one of {SUPPORTED_OUTPUT_FPS}")
    (
        session,
        trim,
        trim_manifest,
        trim_request,
        native,
        native_video_member,
        reference_start_ns,
        reference_end_ns,
        parent_frame_map,
        _tsd_zip,
        _fm_zip,
        trim_role,
    ) = _trim_inputs(store, session_id, native_trim_artifact_id)
    emit(progress, 0.01, "Reading accepted shared-audio TimeGraph and aligned interval")
    sync_artifact, candidate, accepted, mapping, source_video, reference_video, sync_role = _sync_inputs(
        store,
        session_id=session_id,
        time_sync_artifact_id=time_sync_artifact_id,
        expected_native_asset_id=native.id,
        expected_role=trim_role,
    )
    role = trim_role or sync_role
    if reference_video.video_member and str(reference_video.video_member) != str(native_video_member):
        raise ValidationError("Shared-audio synchronization used a different Native scene-video member than the aligned clip")

    encoder = privacy_encoder_preflight(
        ffmpeg_path,
        preserve_audio=preserve_audio,
        preset=preset,
        crf=crf,
        ffprobe_path=ffprobe_path,
    )
    verified_ffmpeg = str(encoder["ffmpeg"]["path"])
    emit(progress, 0.06, "Indexing original facial-video frame PTS")
    source_pts, source_end_ns, source_clock, source_clock_path = _source_clock_cache(
        store,
        video=source_video,
        ffprobe_path=ffprobe_path,
    )

    artifact_id = new_id("artifact")
    category = store.derived_dir(session_id, "aligned_facial_video")
    staging = category / (".partial_" + artifact_id)
    final = category / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    output_label = label.strip() or f"Aligned facial video {role or ''}".strip()
    try:
        video_path, frame_map_path, video_stats, command = _decode_reference_clock_video(
            source_video=Path(source_video.video_path),
            source_pts_ns=source_pts,
            source_end_ns=source_end_ns,
            mapping=mapping,
            reference_start_ns=int(reference_start_ns),
            reference_end_ns=int(reference_end_ns),
            parent_frame_map=parent_frame_map,
            output_fps=int(output_fps),
            staging=staging,
            output_label=output_label,
            preserve_audio=bool(preserve_audio),
            crf=int(crf),
            preset=preset,
            ffmpeg_path=verified_ffmpeg,
            token=token,
            progress=progress,
        )
        source_manifest = staging / "source_audio_video_sync_accepted.json"
        shutil.copy2(find_artifact_file(sync_artifact, names=["audio_video_sync_accepted.json"]), source_manifest)
        source_trim_manifest = staging / "source_trim_manifest.json"
        shutil.copy2(find_artifact_file(trim, names=["trim_manifest.json"]), source_trim_manifest)
        source_native_map = staging / "source_native_frame_map.csv"
        shutil.copy2(parent_frame_map, source_native_map)

        identity = (session.assets[source_video.asset_id].metadata or {}).get("identity") or {}
        payload = {
            "schema_version": ALIGNED_FACIAL_EXPORT_SCHEMA,
            "producer_version": __version__,
            "engine": ALIGNED_FACIAL_EXPORT_ENGINE,
            "artifact_id": artifact_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "time_sync_artifact_id": time_sync_artifact_id,
            "wearer_role_id": role or None,
            "interaction_id": str(identity.get("interaction_id") or "").upper() or None,
            "source_asset_id": source_video.asset_id,
            "source_asset_label": source_video.asset_label,
            "source_asset_kind": source_video.asset_kind,
            "source_video_member": source_video.video_member,
            "reference_asset_id": reference_video.asset_id,
            "reference_asset_label": reference_video.asset_label,
            "reference_video_member": reference_video.video_member,
            "time_map": mapping.model_dump(mode="json"),
            "synchronization": {
                "candidate_id": candidate.get("candidate_id"),
                "confidence": accepted.get("confidence") or candidate.get("confidence"),
                "visual_review_confirmed": accepted.get("visual_review_confirmed") is True,
                "visual_offset_adjustment_ms": float(accepted.get("visual_offset_adjustment_ms") or 0.0),
                "metrics": accepted.get("metrics") or candidate.get("metrics") or {},
            },
            "reference_window": {
                "start_media_ns": str(reference_start_ns),
                "end_media_ns": str(reference_end_ns),
                "duration_ns": str(int(reference_end_ns - reference_start_ns)),
                "duration_sec": float((reference_end_ns - reference_start_ns) / 1e9),
            },
            "video": video_stats,
            "outputs": {
                "aligned_facial_video": str(final / video_path.name),
                "aligned_facial_frame_map": str(final / frame_map_path.name),
                "source_audio_video_sync_accepted": str(final / source_manifest.name),
                "source_trim_manifest": str(final / source_trim_manifest.name),
                "source_native_frame_map": str(final / source_native_map.name),
                "manifest": str(final / "aligned_facial_manifest.json"),
                "report": str(final / "aligned_facial_sync_report.html"),
            },
            "ffmpeg_command": command,
            "source_clock": {
                **source_clock,
                "cache_index_sha256": sha256_file(source_clock_path),
            },
            "source_stamps": {
                "facial_video": source_stamp(session.assets[source_video.asset_id].source_path),
                "ego_reference": source_stamp(session.assets[reference_video.asset_id].source_path),
            },
            "created_at": utc_now(),
        }
        atomic_write_json(staging / "aligned_facial_manifest.json", payload)
        (staging / "aligned_facial_sync_report.html").write_text(_report_html(payload), "utf-8")
        (staging / "README.txt").write_text(
            "OculoStudio aligned facial-expression video. The output is resampled on the accepted ego/reference clock; "
            "use aligned_facial_frame_map.csv for exact source and Native-frame lineage. Original recordings are unchanged.\n",
            "utf-8",
        )
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if final.exists() and artifact_id not in session.artifacts:
            shutil.rmtree(final, ignore_errors=True)
        raise

    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.ALIGNED_FACIAL_VIDEO,
        label=output_label,
        output_root=final,
        source_asset_ids=list(
            dict.fromkeys([source_video.asset_id, reference_video.asset_id, *trim.source_asset_ids])
        ),
        parent_artifact_ids=[native_trim_artifact_id, time_sync_artifact_id],
        parameters={
            "engine": ALIGNED_FACIAL_EXPORT_ENGINE,
            "native_trim_artifact_id": native_trim_artifact_id,
            "time_sync_artifact_id": time_sync_artifact_id,
            "source_asset_id": source_video.asset_id,
            "reference_asset_id": reference_video.asset_id,
            "time_map_id": mapping.id,
            "wearer_role_id": role or None,
            "interaction_id": str(identity.get("interaction_id") or "").upper() or None,
            "output_fps": int(output_fps),
            "preserve_audio": bool(preserve_audio),
            "reference_window": {
                "start_media_ns": str(reference_start_ns),
                "end_media_ns": str(reference_end_ns),
            },
        },
        qc={
            "package_complete": True,
            "sync_verified": True,
            "visual_review_confirmed": True,
            "physical_interval_from_native_trim": True,
            "reference_clock_resampled": True,
            "video_drift_corrected": True,
            "audio_drift_corrected": bool(preserve_audio),
            "source_full_interval_covered": True,
            **video_stats,
        },
        provenance={
            "method": ALIGNED_FACIAL_EXPORT_ENGINE,
            "boundary_source": "immutable_native_reference_trim",
            "clock_source": "accepted_shared_audio_affine_time_map",
            "frame_selection": "reference_output_clock_to_inverse_facial_time_map_to_original_frame_pts",
            "audio_policy": "source_segment_retimed_by_inverse_affine_scale_then_padded_to_cfr_output",
            "arbitrary_dynamic_time_warping_used": False,
            "source_videos_modified": False,
            "synchronization_reestimated": False,
        },
        artifact_id=artifact_id,
    )
    result = {
        "artifact": artifact.model_dump(mode="json"),
        "artifact_id": artifact.id,
        "output_dir": str(final),
        "video": str(final / video_path.name),
        "frame_map": str(final / frame_map_path.name),
        "manifest": str(final / "aligned_facial_manifest.json"),
        "report_html": str(final / "aligned_facial_sync_report.html"),
        "wearer_role_id": role or None,
        "interaction_id": str(identity.get("interaction_id") or "").upper() or None,
        "native_trim_artifact_id": native_trim_artifact_id,
        "time_sync_artifact_id": time_sync_artifact_id,
        "source_asset_id": source_video.asset_id,
        "reference_asset_id": reference_video.asset_id,
        "video_stats": video_stats,
    }

    def remember(project):
        metadata = project.sessions[session_id].metadata
        history = dict(metadata.get("aligned_facial_export_by_trim") or {})
        role_history = dict(history.get(native_trim_artifact_id) or {})
        role_history[time_sync_artifact_id] = result
        history[native_trim_artifact_id] = role_history
        metadata["aligned_facial_export_by_trim"] = history
        metadata["aligned_facial_export_last"] = result

    store.update(remember)
    emit(progress, 1.0, "Aligned facial-expression video is ready")
    return result
