"""Publish a clean aligned ego-video together with the verified TSD/FM subsets.

The Native-reference trim remains the authority for every boundary.  This
service never estimates a new start or end: it reads the immutable trim
manifest, selects source frames from the original NRD ``.time``/PTS clock,
creates an explicit CFR frame map, optionally preserves the source audio, and
copies the already-verified TSD and FaceMapper ZIPs byte-for-byte into one
role-specific aligned package.
"""
from __future__ import annotations

import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from oculostudio import __version__
from oculostudio.adapters.native_reference_clock import build_clock, round_ratio, source_stamp
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.pretrim_overlay import _cache_video_member
from oculostudio.services.privacy import _mux_audio_and_transcode, _output_writer
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, sha256_file, utc_now


ALIGNED_CLIP_EXPORT_ENGINE = "native_pts_aligned_ego_tsd_fm_v1"
ALIGNED_CLIP_EXPORT_SCHEMA = "oculostudio-aligned-clip-package-1.0"
SUPPORTED_OUTPUT_FPS = (24, 25, 30, 50, 60)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {label}: {exc}") from exc


def _trim_inputs(store: ProjectStore, session_id: str, native_trim_artifact_id: str):
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    trim = session.artifacts.get(native_trim_artifact_id)
    if trim is None or trim.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    manifest_path = find_artifact_file(trim, names=["trim_manifest.json"])
    manifest = _read_json(manifest_path, "trim_manifest.json")
    request = dict(manifest.get("request") or trim.parameters or {})
    window = dict(manifest.get("window") or request.get("window") or {})
    try:
        start_ns = int(window["start_media_ns"])
        end_ns = int(window["end_media_ns"])
    except Exception as exc:
        raise ValidationError("Native-reference trim has no valid media-time window") from exc
    if end_ns <= start_ns:
        raise ValidationError("Native-reference trim interval is empty")
    native_id = str(request.get("native_asset_id") or (trim.parameters or {}).get("native_asset_id") or "")
    native = session.assets.get(native_id)
    if native is None or native.kind != "pupil_native":
        raise ValidationError("The Native source recorded by this trim is unavailable")
    if native.id not in trim.source_asset_ids:
        raise ValidationError("The trim lineage does not include the selected Native source")
    video_member = str(request.get("native_video_member") or "").strip()
    if not video_member:
        raise ValidationError("The trim does not record the exact NRD scene-video member")
    expected_native_stamp = (manifest.get("source_stamps") or {}).get("native")
    if expected_native_stamp and source_stamp(native.source_path) != expected_native_stamp:
        raise ValidationError("The original NRD source changed after the verified trim was created")
    manifest_recording_id = str(manifest.get("recording_id") or "")
    if manifest_recording_id and native.recording_id and manifest_recording_id != str(native.recording_id):
        raise ValidationError("The verified trim recording id no longer matches the Native source")
    parent_frame_map = find_artifact_file(trim, names=["native_frame_map.csv"])
    tsd_zip = find_artifact_file(trim, names=["TSD_trimmed.zip"])
    fm_zip = find_artifact_file(trim, names=["FaceMapper_trimmed.zip"])
    role = str(request.get("wearer_role_id") or (trim.parameters or {}).get("wearer_role_id") or "").upper()
    return session, trim, manifest, request, native, video_member, start_ns, end_ns, parent_frame_map, tsd_zip, fm_zip, role


def _decode_aligned_video(
    *,
    store: ProjectStore,
    native,
    video_member: str,
    start_ns: int,
    end_ns: int,
    output_fps: int,
    source_video: Path,
    clock,
    staging: Path,
    label: str,
    preserve_audio: bool,
    crf: int,
    preset: str,
    ffmpeg_path: str,
    token: CancellationToken,
    progress,
) -> tuple[Path, Path, dict[str, Any], list[str]]:
    duration_ns = int(end_ns - start_ns)
    frame_count = int(math.ceil(duration_ns * output_fps / 1_000_000_000))
    if frame_count <= 0:
        raise ValidationError("Aligned ego-video interval contains no output frames")
    output_pts = np.asarray(
        [round_ratio(index * 1_000_000_000, output_fps) for index in range(frame_count)],
        dtype=np.int64,
    )
    requested_times = int(start_ns) + output_pts
    media_ns = np.asarray(clock.media_ns, dtype=np.int64)
    native_indices = np.searchsorted(media_ns, requested_times, side="right") - 1
    native_indices = np.clip(native_indices, 0, len(media_ns) - 1).astype(np.int64)
    if np.any(np.diff(native_indices) < 0):
        raise ValidationError("Native frame selection is not monotonic")

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode Native scene video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Native scene-video dimensions are unavailable")
    if source_frame_count and int(native_indices[-1]) >= source_frame_count:
        cap.release()
        raise ValidationError("Trim frame map references a frame beyond the source ego video")

    safe = safe_name(label, fallback="aligned_ego_video")
    silent = staging / "aligned_ego_video_silent.mp4"
    output_video = staging / f"{safe}.mp4"
    frame_map_path = staging / "aligned_ego_frame_map.csv"
    writer = _output_writer(silent, width, height, float(output_fps))
    targets: dict[int, list[int]] = {}
    for output_index, native_index in enumerate(native_indices.tolist()):
        targets.setdefault(int(native_index), []).append(int(output_index))
    first_source, last_source = int(native_indices[0]), int(native_indices[-1])
    rows: list[dict[str, Any]] = []
    written = 0
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_source)
        for source_index in range(first_source, last_source + 1):
            token.raise_if_cancelled()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ProcessingError(f"Native scene-video decode ended at frame {source_index}")
            for output_index in targets.get(source_index, []):
                writer.write(frame)
                frame_start = int(media_ns[source_index])
                frame_end = int(media_ns[source_index + 1]) if source_index + 1 < len(media_ns) else int(clock.end_media_ns)
                requested = int(requested_times[output_index])
                rows.append({
                    "output_frame_index": int(output_index),
                    "output_pts_ns": int(output_pts[output_index]),
                    "clip_time_ns": int(output_pts[output_index]),
                    "native_frame_index": int(source_index),
                    "native_video_pts_ns": frame_start,
                    "native_frame_end_pts_ns": frame_end,
                    "native_capture_utc_ns": int(clock.capture_ns[source_index]),
                    "requested_native_time_ns": requested,
                    "native_selection_error_ns": int(requested - frame_start),
                    "source_frame_duplicated": bool(len(targets.get(source_index, [])) > 1),
                    "boundary_frame": bool(frame_start < start_ns or frame_end > end_ns),
                })
                written += 1
            if source_index % 20 == 0:
                emit(
                    progress,
                    0.18 + 0.48 * (source_index - first_source + 1) / max(1, last_source - first_source + 1),
                    f"Selecting clean ego frames ({written}/{frame_count})",
                )
    except BaseException:
        raise
    finally:
        cap.release()
        writer.release()
    if written != frame_count:
        raise ProcessingError(f"Wrote {written} aligned ego frames but expected {frame_count}")
    pd.DataFrame(rows).to_csv(frame_map_path, index=False)

    emit(progress, 0.70, "Encoding clean aligned ego video and preserving source audio")
    command = _mux_audio_and_transcode(
        silent,
        source_video,
        output_video,
        source_start_sec=start_ns / 1e9,
        duration_sec=duration_ns / 1e9,
        preserve_audio=preserve_audio,
        crf=crf,
        preset=preset,
        ffmpeg_path=ffmpeg_path,
        expected_frames=frame_count,
        cancellation=token,
        progress=progress,
    )
    silent.unlink(missing_ok=True)
    stats = {
        "output_fps": int(output_fps),
        "output_frame_count": int(frame_count),
        "output_duration_sec": float(frame_count / output_fps),
        "physical_interval_duration_sec": float(duration_ns / 1e9),
        "width": int(width),
        "height": int(height),
        "first_native_frame_index": int(native_indices[0]),
        "last_native_frame_index": int(native_indices[-1]),
        "unique_native_frames": int(len(set(int(value) for value in native_indices))),
        "duplicated_output_frames": int(frame_count - len(set(int(value) for value in native_indices))),
        "dropped_native_frames_inside_span": int((last_source - first_source + 1) - len(set(int(value) for value in native_indices))),
        "preserve_audio_requested": bool(preserve_audio),
    }
    return output_video, frame_map_path, stats, command


def _report_html(payload: dict[str, Any]) -> str:
    output = payload.get("outputs") or {}
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(value))}</td></tr>"
        for name, value in output.items()
        if value
    )
    video = payload.get("video") or {}
    return f"""<!doctype html><html><meta charset='utf-8'><title>Aligned clip package</title>
<style>body{{font:15px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#2b2e20}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7d8c8;padding:9px;text-align:left}}th{{background:#eef2d5}}code{{background:#f1f1e9;padding:2px 5px;overflow-wrap:anywhere}}</style>
<h1>OculoStudio v{__version__} — Aligned ego + TSD + FaceMapper package</h1>
<p>This package reuses the immutable Native-reference trim. No start/end boundary was re-estimated. The ego video is clean (no QC slate or overlay), and the TSD/FaceMapper ZIPs are copied byte-for-byte from the verified trim artifact.</p>
<p><b>Role:</b> {html.escape(str(payload.get('wearer_role_id') or 'unscoped'))}<br><b>Physical interval:</b> {payload['window']['duration_sec']:.9f} s<br><b>Native media:</b> {payload['window']['start_media_ns']} → {payload['window']['end_media_ns']} ns</p>
<table><thead><tr><th>Output</th><th>Path</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Video timing</h2><ul><li>Output: {video.get('output_frame_count','—')} frames at {video.get('output_fps','—')} fps.</li><li>Each output frame has an explicit source Native frame and PTS in <code>aligned_ego_frame_map.csv</code>.</li><li>Audio preservation requested: {video.get('preserve_audio_requested', False)}.</li></ul>
<h2>Scientific interpretation</h2><ul><li>TSD point samples retain their original absolute timestamps.</li><li>FaceMapper geometry follows the same retained Native frame interval.</li><li>The CFR MP4 is for synchronized review and downstream video tools; use its frame map rather than assuming source-frame ordinal equality.</li></ul></html>"""


def export_aligned_clip_package(
    store: ProjectStore,
    *,
    session_id: str,
    native_trim_artifact_id: str,
    label: str = "Aligned clip package",
    include_ego_video: bool = True,
    include_tsd: bool = True,
    include_facemapper: bool = True,
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
    if not any((include_ego_video, include_tsd, include_facemapper)):
        raise ValidationError("Select at least one aligned output")
    if output_fps not in SUPPORTED_OUTPUT_FPS:
        raise ValidationError(f"Output FPS must be one of {SUPPORTED_OUTPUT_FPS}")
    (
        session,
        trim,
        trim_manifest,
        request,
        native,
        video_member,
        start_ns,
        end_ns,
        parent_frame_map,
        tsd_zip,
        fm_zip,
        role,
    ) = _trim_inputs(store, session_id, native_trim_artifact_id)
    duration_ns = int(end_ns - start_ns)
    automatic = bool(request.get("automatic_c3d_defined"))
    emit(progress, 0.01, "Reading verified aligned-clip lineage")

    encoder = None
    verified_ffmpeg = None
    clock = None
    source_video = None
    if include_ego_video:
        encoder = privacy_encoder_preflight(
            ffmpeg_path,
            preserve_audio=preserve_audio,
            preset=preset,
            crf=crf,
            ffprobe_path=ffprobe_path,
        )
        verified_ffmpeg = encoder["ffmpeg"]["path"]
        emit(progress, 0.04, "Loading original NRD .time and MP4 PTS")
        clock = build_clock(
            native.source_path,
            store.cache_dir(),
            video_member=video_member,
            ffprobe_path=ffprobe_path,
            cancellation=token,
            progress=progress,
        )
        if start_ns < 0 or end_ns > int(clock.end_media_ns):
            raise ValidationError("Aligned interval lies outside the selected NRD scene-video part")
        source_video = _cache_video_member(store, native.source_path, video_member, source_stamp(native.source_path))

    artifact_id = new_id("artifact")
    category = store.derived_dir(session_id, "aligned_clip_package")
    staging = category / (".partial_" + artifact_id)
    final = category / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    output_label = label.strip() or f"Aligned clip package {role or ''}".strip()
    outputs: dict[str, str] = {}
    video_stats: dict[str, Any] = {}
    ffmpeg_command: list[str] | None = None
    try:
        if include_tsd:
            token.raise_if_cancelled()
            target = staging / f"{safe_name(output_label, fallback='aligned_clip')}_TSD_aligned.zip"
            shutil.copy2(tsd_zip, target)
            if sha256_file(target) != sha256_file(tsd_zip):
                raise ProcessingError("Aligned TSD copy failed checksum verification")
            outputs["tsd_aligned_zip"] = str(final / target.name)
            emit(progress, 0.09, "Copied verified aligned TSD package")
        if include_facemapper:
            token.raise_if_cancelled()
            target = staging / f"{safe_name(output_label, fallback='aligned_clip')}_FaceMapper_aligned.zip"
            shutil.copy2(fm_zip, target)
            if sha256_file(target) != sha256_file(fm_zip):
                raise ProcessingError("Aligned FaceMapper copy failed checksum verification")
            outputs["facemapper_aligned_zip"] = str(final / target.name)
            emit(progress, 0.13, "Copied verified aligned FaceMapper package")

        parent_map_copy = staging / "native_source_frame_map.csv"
        shutil.copy2(parent_frame_map, parent_map_copy)
        outputs["native_source_frame_map"] = str(final / parent_map_copy.name)
        parent_manifest_copy = staging / "source_trim_manifest.json"
        shutil.copy2(find_artifact_file(trim, names=["trim_manifest.json"]), parent_manifest_copy)
        outputs["source_trim_manifest"] = str(final / parent_manifest_copy.name)

        if include_ego_video:
            assert clock is not None and source_video is not None and verified_ffmpeg is not None
            video_path, frame_map_path, video_stats, ffmpeg_command = _decode_aligned_video(
                store=store,
                native=native,
                video_member=video_member,
                start_ns=start_ns,
                end_ns=end_ns,
                output_fps=output_fps,
                source_video=source_video,
                clock=clock,
                staging=staging,
                label=f"{output_label}_ego",
                preserve_audio=preserve_audio,
                crf=crf,
                preset=preset,
                ffmpeg_path=verified_ffmpeg,
                token=token,
                progress=progress,
            )
            outputs["aligned_ego_video"] = str(final / video_path.name)
            outputs["aligned_ego_frame_map"] = str(final / frame_map_path.name)
            parent_rows = pd.read_csv(parent_frame_map, usecols=["native_frame_index"])
            allowed = set(pd.to_numeric(parent_rows["native_frame_index"], errors="coerce").dropna().astype(int).tolist())
            output_rows = pd.read_csv(frame_map_path, usecols=["native_frame_index"])
            selected = set(pd.to_numeric(output_rows["native_frame_index"], errors="coerce").dropna().astype(int).tolist())
            missing = sorted(selected - allowed)
            if missing:
                raise ValidationError(
                    f"Aligned ego export selected {len(missing)} Native frame(s) outside the verified trim frame map"
                )

        final_manifest = final / "aligned_clip_manifest.json"
        final_report = final / "aligned_clip_report.html"
        payload = {
            "schema_version": ALIGNED_CLIP_EXPORT_SCHEMA,
            "producer_version": __version__,
            "engine": ALIGNED_CLIP_EXPORT_ENGINE,
            "artifact_id": artifact_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "automatic_c3d_defined": automatic,
            "wearer_role_id": role or None,
            "recording_id": trim_manifest.get("recording_id"),
            "native_asset_id": native.id,
            "native_video_member": video_member,
            "window": {
                "start_media_ns": str(start_ns),
                "end_media_ns": str(end_ns),
                "duration_ns": str(duration_ns),
                "duration_sec": float(duration_ns / 1e9),
                "utc_start_ns": str((trim_manifest.get("window") or {}).get("utc_start_ns") or ""),
                "utc_end_ns": str((trim_manifest.get("window") or {}).get("utc_end_ns") or ""),
            },
            "selection": {
                "include_ego_video": bool(include_ego_video),
                "include_tsd": bool(include_tsd),
                "include_facemapper": bool(include_facemapper),
                "output_fps": int(output_fps),
                "preserve_audio": bool(preserve_audio),
            },
            "outputs": outputs,
            "video": video_stats,
            "ffmpeg_command": ffmpeg_command,
            "source_checksums": {
                "parent_trim_manifest_sha256": sha256_file(find_artifact_file(trim, names=["trim_manifest.json"])),
                "parent_native_frame_map_sha256": sha256_file(parent_frame_map),
                "parent_tsd_zip_sha256": sha256_file(tsd_zip),
                "parent_facemapper_zip_sha256": sha256_file(fm_zip),
            },
            "created_at": utc_now(),
        }
        manifest_path = staging / "aligned_clip_manifest.json"
        report_path = staging / "aligned_clip_report.html"
        atomic_write_json(manifest_path, payload)
        report_path.write_text(_report_html(payload), "utf-8")
        (staging / "README.txt").write_text(
            "OculoStudio aligned clip package. The MP4 is a clean CFR review/export video with an explicit frame map. "
            "TSD and FaceMapper ZIPs are byte-identical copies from the verified Native-reference trim.\n",
            "utf-8",
        )
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if final.exists() and artifact_id not in session.artifacts:
            shutil.rmtree(final, ignore_errors=True)
        raise

    source_ids = list(dict.fromkeys([*trim.source_asset_ids, native.id]))
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.ALIGNED_CLIP_PACKAGE,
        label=output_label,
        output_root=final,
        source_asset_ids=source_ids,
        parent_artifact_ids=[native_trim_artifact_id],
        parameters={
            "engine": ALIGNED_CLIP_EXPORT_ENGINE,
            "native_trim_artifact_id": native_trim_artifact_id,
            "wearer_role_id": role or None,
            "automatic_c3d_defined": automatic,
            "include_ego_video": bool(include_ego_video),
            "include_tsd": bool(include_tsd),
            "include_facemapper": bool(include_facemapper),
            "output_fps": int(output_fps),
            "preserve_audio": bool(preserve_audio),
            "native_video_member": video_member,
            "window": {"start_media_ns": str(start_ns), "end_media_ns": str(end_ns)},
        },
        qc={
            "package_complete": True,
            "ego_video_created": bool(include_ego_video),
            "tsd_aligned_included": bool(include_tsd),
            "facemapper_aligned_included": bool(include_facemapper),
            "source_tsd_checksum_preserved": bool(include_tsd),
            "source_facemapper_checksum_preserved": bool(include_facemapper),
            "physical_duration_sec": float(duration_ns / 1e9),
            **video_stats,
        },
        provenance={
            "method": ALIGNED_CLIP_EXPORT_ENGINE,
            "boundary_source": "immutable_native_reference_trim",
            "frame_selection": "constant_fps_output_to_native_display_interval_via_original_pts",
            "source_mp4_modified": False,
            "scientific_tables_recomputed": False,
            "tsd_facemapper_copy_policy": "byte_identical_from_verified_trim",
        },
        artifact_id=artifact_id,
    )
    result = {
        "artifact": artifact.model_dump(mode="json"),
        "artifact_id": artifact.id,
        "output_dir": str(final),
        "video": outputs.get("aligned_ego_video"),
        "frame_map": outputs.get("aligned_ego_frame_map"),
        "tsd_zip": outputs.get("tsd_aligned_zip"),
        "facemapper_zip": outputs.get("facemapper_aligned_zip"),
        "manifest": str(final / "aligned_clip_manifest.json"),
        "report_html": str(final / "aligned_clip_report.html"),
        "wearer_role_id": role or None,
        "native_trim_artifact_id": native_trim_artifact_id,
    }

    def remember(project):
        metadata = project.sessions[session_id].metadata
        history = dict(metadata.get("aligned_clip_export_by_trim") or {})
        history[native_trim_artifact_id] = result
        metadata["aligned_clip_export_by_trim"] = history
        metadata["aligned_clip_export_last"] = result

    store.update(remember)
    emit(progress, 1.0, "Aligned ego video, TSD and FaceMapper package is ready")
    return result
