from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.native_reference_clock import build_clock, native_options, round_ratio
from oculostudio.adapters.pupil import materialized_native_files
from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.video_input_time import parse_native_trim_times
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file, find_video_file
from oculostudio.services.privacy import _mux_audio_and_transcode, _output_writer
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


SUPPORTED_OUTPUT_FPS = (24, 25, 30, 50, 60)
PRIVACY_CLIP_ENGINE = "native-pts-audit-v2"


def _artifact_manifest(artifact, filename: str) -> dict:
    path = find_artifact_file(artifact, names=[filename])
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {filename} from artifact {artifact.id}: {exc}") from exc


def _resolve_time_window(
    *,
    session,
    native_trim_artifact_id: str | None,
    start_timecode: str | None,
    duration_timecode: str | None,
    timecode_fps: int,
) -> tuple[int, int, dict[str, Any], str | None]:
    if native_trim_artifact_id:
        artifact = session.artifacts.get(native_trim_artifact_id)
        if artifact is None or artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
            raise ValidationError("native_trim_artifact_id must reference a Native-reference trim artifact")
        manifest = _artifact_manifest(artifact, "trim_manifest.json")
        window = manifest.get("window") or {}
        try:
            start_ns = int(window["start_media_ns"])
            end_ns = int(window["end_media_ns"])
        except Exception as exc:
            raise ValidationError("Native-reference trim artifact has no valid media-time window") from exc
        request = manifest.get("request") or {}
        interpretation = request.get("timecode_interpretation") or {
            "schema_version": "oculostudio-video-time-input-1.0",
            "start": {"nanoseconds": str(start_ns), "normalized_time": request.get("start_timecode")},
            "duration": {"nanoseconds": str(end_ns - start_ns), "normalized_time": request.get("duration_timecode")},
            "end": {"nanoseconds": str(end_ns)},
            "timecode_fps": request.get("timecode_fps", timecode_fps),
        }
        return start_ns, end_ns, interpretation, native_trim_artifact_id
    if not start_timecode or not duration_timecode:
        raise ValidationError("Enter start + duration, or choose an existing Native-reference trim artifact")
    interpretation = parse_native_trim_times(
        start_timecode,
        duration_timecode,
        timecode_fps=timecode_fps,
    )
    start_ns = int(interpretation["start"]["nanoseconds"])
    end_ns = int(interpretation["end"]["nanoseconds"])
    return start_ns, end_ns, interpretation, None


def _native_source_from_privacy(session, privacy_artifact, explicit_native_asset_id: str | None):
    if explicit_native_asset_id:
        asset = session.assets.get(explicit_native_asset_id)
        if asset is None or asset.kind != "pupil_native":
            raise ValidationError("native_asset_id must reference a Native Recording asset")
        return asset
    candidates = [
        session.assets[asset_id]
        for asset_id in privacy_artifact.source_asset_ids
        if asset_id in session.assets and session.assets[asset_id].kind == "pupil_native"
    ]
    if len(candidates) != 1:
        raise ValidationError("Could not infer one Native source from the Privacy artifact; select it explicitly")
    return candidates[0]


def _video_member_for_window(native_asset, privacy_report: dict, requested_member: str | None) -> str:
    if requested_member:
        return requested_member
    report_member = privacy_report.get("native_video_member")
    if report_member:
        return str(report_member)
    options = native_options(native_asset.source_path)
    if len(options) != 1:
        raise ValidationError("The Native source has multiple scene MP4 parts; select the exact part")
    return str(options[0]["video_member"])


def _privacy_files(artifact):
    video = find_video_file(artifact, preferred_tokens=("privacy",))
    audit = find_artifact_file(artifact, names=["privacy_frame_audit.csv"])
    report = find_artifact_file(artifact, names=["privacy_report.json"])
    try:
        report_data = json.loads(report.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Privacy report cannot be read: {exc}") from exc
    return video, audit, report_data


def _source_video_path(store: ProjectStore, native_asset, video_member: str) -> Path:
    # Privacy processing already materializes this source in normal use. Reuse
    # the project cache; source archives remain unchanged.
    cache = MaterializationCache(store.cache_dir())
    root = cache.materialize(native_asset)
    candidate = root.joinpath(*PurePosixPath(video_member).parts).resolve()
    if root.resolve() not in candidate.parents or not candidate.is_file():
        # Fall back to the unambiguous candidate resolver for one-part assets.
        source_video, _time, _info, _metadata = materialized_native_files(root, native_asset.recording_id)
        return source_video
    return candidate


def create_trimmed_privacy_video(
    store: ProjectStore,
    *,
    session_id: str,
    privacy_artifact_id: str,
    native_asset_id: str | None = None,
    native_trim_artifact_id: str | None = None,
    start_timecode: str | None = None,
    duration_timecode: str | None = None,
    timecode_fps: int = 30,
    output_fps: int = 30,
    native_video_member: str | None = None,
    label: str = "Trimmed privacy video",
    preserve_audio: bool = True,
    crf: int = 20,
    preset: str = "fast",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    if output_fps not in SUPPORTED_OUTPUT_FPS:
        raise ValidationError(f"Output FPS must be one of {SUPPORTED_OUTPUT_FPS}")
    session = store.project.sessions[session_id]
    privacy_artifact = session.artifacts.get(privacy_artifact_id)
    if privacy_artifact is None or privacy_artifact.kind != ArtifactKind.PRIVACY_VIDEO:
        raise ValidationError("Select a full Privacy video artifact")
    start_ns, end_ns, interpretation, trim_parent = _resolve_time_window(
        session=session,
        native_trim_artifact_id=native_trim_artifact_id,
        start_timecode=start_timecode,
        duration_timecode=duration_timecode,
        timecode_fps=timecode_fps,
    )
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        raise ValidationError("Privacy clip duration must be positive")

    emit(progress, 0.01, "Checking Privacy artifact and media tools")
    encoder = privacy_encoder_preflight(
        ffmpeg_path,
        preserve_audio=preserve_audio,
        preset=preset,
        crf=crf,
        ffprobe_path=ffprobe_path,
    )
    verified_ffmpeg = encoder["ffmpeg"]["path"]
    privacy_video, privacy_audit_path, privacy_report = _privacy_files(privacy_artifact)
    native_asset = _native_source_from_privacy(session, privacy_artifact, native_asset_id)
    if trim_parent:
        trim_artifact = session.artifacts.get(trim_parent)
        if trim_artifact is None or native_asset.id not in trim_artifact.source_asset_ids:
            raise ValidationError(
                "The selected Native-reference trim belongs to another Native recording"
            )
    video_member = _video_member_for_window(native_asset, privacy_report, native_video_member)
    emit(progress, 0.04, "Loading actual Native PTS and frame audit")
    clock = build_clock(
        native_asset.source_path,
        store.cache_dir(),
        video_member=video_member,
        ffprobe_path=ffprobe_path,
        cancellation=token,
        progress=progress,
    )
    if start_ns < 0 or end_ns > clock.end_media_ns:
        raise ValidationError(
            f"Requested Privacy clip lies outside the selected NRD part (duration {clock.end_media_ns/1e9:.6f} s)"
        )
    audit = pd.read_csv(privacy_audit_path, low_memory=False)
    required = {"output_frame_index", "native_frame_index", "timestamp_ns"}
    if not required.issubset(audit.columns):
        raise ValidationError(f"Privacy frame audit is missing columns: {sorted(required-set(audit.columns))}")
    audit = audit.sort_values("output_frame_index")
    if audit["output_frame_index"].duplicated().any() or audit["native_frame_index"].duplicated().any():
        raise ValidationError("Privacy frame audit must map output and Native frames one-to-one")
    native_to_privacy = {
        int(row.native_frame_index): int(row.output_frame_index)
        for row in audit[["native_frame_index", "output_frame_index"]].itertuples(index=False)
    }

    # Constant-FPS output: each output PTS is mapped back to the Native display
    # interval, then to the audited privacy frame. Every duplication/drop is
    # explicit in the exported map.
    frame_count = int(math.ceil(duration_ns * output_fps / 1_000_000_000))
    if frame_count <= 0:
        raise ValidationError("Requested Privacy clip contains no output frames")
    output_pts = np.asarray(
        [round_ratio(index * 1_000_000_000, output_fps) for index in range(frame_count)],
        dtype=np.int64,
    )
    native_times = start_ns + output_pts
    native_indices = np.searchsorted(clock.media_ns, native_times, side="right") - 1
    native_indices = np.clip(native_indices, 0, len(clock.media_ns) - 1).astype(int)
    missing = sorted({int(index) for index in native_indices if int(index) not in native_to_privacy})
    if missing:
        privacy_trim_id = str(privacy_report.get("native_trim_artifact_id") or "").strip() or None
        old_engine = not (privacy_report.get("frame_coverage") or {}).get("engine")
        if trim_parent and privacy_trim_id == trim_parent and old_engine:
            preview = ", ".join(str(value) for value in missing[:4])
            suffix = "…" if len(missing) > 4 else ""
            raise ValidationError(
                "This Privacy artifact was created with the legacy boundary-frame policy and omits "
                f"{len(missing)} Native boundary frame(s) ({preview}{suffix}). "
                "Create the Privacy artifact again in OculoStudio v0.20.4, then select the new artifact."
            )
        raise ValidationError(
            f"The selected Privacy artifact does not cover {len(missing)} required Native frames. "
            "Create a full-recording Privacy artifact or choose a covered interval."
        )
    source_indices = np.asarray([native_to_privacy[int(index)] for index in native_indices], dtype=int)
    if np.any(np.diff(source_indices) < 0):
        raise ValidationError("Privacy frame mapping is not monotonic")

    cap = cv2.VideoCapture(str(privacy_video))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode Privacy video: {privacy_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Privacy video dimensions are unavailable")
    if source_frame_count and int(source_indices[-1]) >= source_frame_count:
        cap.release()
        raise ValidationError("Privacy audit references a frame beyond the encoded Privacy video")

    artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "privacy_clip") / (".partial_" + artifact_id)
    final = store.derived_dir(session_id, "privacy_clip") / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    silent = staging / "privacy_clip_silent.mp4"
    output_video = staging / f"{safe_name(label, fallback='privacy_clip')}.mp4"
    frame_map_path = staging / "privacy_trimmed_frame_map.csv"
    report_path = staging / "privacy_clip_report.json"
    writer = _output_writer(silent, width, height, float(output_fps))
    targets_by_source: dict[int, list[int]] = defaultdict(list)
    for output_index, source_index in enumerate(source_indices):
        targets_by_source[int(source_index)].append(output_index)
    first_source, last_source = int(source_indices[0]), int(source_indices[-1])
    written = 0
    rows: list[dict[str, Any]] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_source)
        for source_index in range(first_source, last_source + 1):
            token.raise_if_cancelled()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ProcessingError(f"Privacy video decode ended at frame {source_index}")
            for output_index in targets_by_source.get(source_index, []):
                writer.write(frame)
                native_index = int(native_indices[output_index])
                rows.append({
                    "output_frame_index": output_index,
                    "output_pts_ns": int(output_pts[output_index]),
                    "privacy_source_frame_index": source_index,
                    "native_frame_index": native_index,
                    "native_video_pts_ns": int(clock.media_ns[native_index]),
                    "native_capture_utc_ns": int(clock.capture_ns[native_index]),
                    "clip_time_ns": int(output_pts[output_index]),
                    "requested_native_time_ns": int(native_times[output_index]),
                    "native_selection_error_ns": int(native_times[output_index] - int(clock.media_ns[native_index])),
                })
                written += 1
            if source_index % 20 == 0:
                emit(progress, 0.38 + 0.36 * (source_index-first_source+1) / max(1,last_source-first_source+1),
                     f"Selecting blurred frames ({written}/{frame_count})")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        cap.release()
        writer.release()
    if written != frame_count:
        shutil.rmtree(staging, ignore_errors=True)
        raise ProcessingError(f"Wrote {written} frames but expected {frame_count}")
    pd.DataFrame(rows).to_csv(frame_map_path, index=False)

    source_video = _source_video_path(store, native_asset, video_member)
    try:
        emit(progress, 0.78, "Encoding trimmed Privacy video and preserving audio")
        command = _mux_audio_and_transcode(
            silent,
            source_video,
            output_video,
            source_start_sec=start_ns / 1e9,
            duration_sec=frame_count / output_fps,
            preserve_audio=preserve_audio,
            crf=crf,
            preset=preset,
            ffmpeg_path=verified_ffmpeg,
            expected_frames=frame_count,
            cancellation=token,
            progress=progress,
        )
        silent.unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Store final paths, even though the directory is still staged. This keeps
    # manifests valid after the atomic directory rename.
    final_video = final / output_video.name
    final_frame_map = final / frame_map_path.name
    final_report = final / report_path.name
    report = {
        "schema_version": "oculostudio-privacy-clip-1.0",
        "artifact_id": artifact_id,
        "privacy_artifact_id": privacy_artifact_id,
        "native_trim_artifact_id": trim_parent,
        "native_asset_id": native_asset.id,
        "native_video_member": video_member,
        "time_input": interpretation,
        "window": {"start_media_ns": str(start_ns), "end_media_ns": str(end_ns), "duration_ns": str(duration_ns)},
        "output": {
            "fps": output_fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / output_fps,
            "width": width,
            "height": height,
            "video": str(final_video),
            "frame_map": str(final_frame_map),
        },
        "timing_policy": "constant_fps_frames_selected_by_native_PTS_then_privacy_audit",
        "processing_engine": PRIVACY_CLIP_ENGINE,
        "source_privacy_video": str(privacy_video),
        "ffmpeg_command": command,
        "created_at": utc_now(),
    }
    atomic_write_json(report_path, report)
    try:
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    output_video = final_video
    frame_map_path = final_frame_map
    report_path = final_report
    parent_ids = [privacy_artifact_id] + ([trim_parent] if trim_parent else [])
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.PRIVACY_CLIP,
        label=label,
        output_root=final,
        source_asset_ids=[native_asset.id],
        parent_artifact_ids=parent_ids,
        parameters={
            "time_input": interpretation,
            "output_fps": output_fps,
            "preserve_audio": preserve_audio,
            "native_video_member": video_member,
        },
        qc={
            "output_frame_count": frame_count,
            "output_fps": output_fps,
            "unique_native_frames": int(len(set(int(x) for x in native_indices))),
            "duplicated_output_frames": int(frame_count - len(set(int(x) for x in source_indices))),
            "privacy_source_coverage": True,
        },
        provenance={
            "privacy_artifact_id": privacy_artifact_id,
            "native_trim_artifact_id": trim_parent,
            "frame_selection": "native_PTS_to_privacy_frame_audit",
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Trimmed Privacy video complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "video": str(output_video),
        "frame_map": str(frame_map_path),
        "report": str(report_path),
    }
