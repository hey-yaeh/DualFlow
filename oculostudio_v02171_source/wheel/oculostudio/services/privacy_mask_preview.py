from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.native_reference_clock import build_clock, round_ratio
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.facemapper_filter import filtered_facemapper_root
from oculostudio.services.privacy import (
    FaceBox,
    _mux_audio_and_transcode,
    _output_writer,
    load_face_mapping,
)
from oculostudio.services.privacy_final import (
    _json_file,
    _load_trim_frame_map,
    _source_video_path,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


PRIVACY_MASK_PREVIEW_ENGINE = "accepted-only-motion-repair-preview-v1"


def _draw_box_overlay(
    frame: np.ndarray,
    box: FaceBox,
    *,
    padding_percent: float,
) -> tuple[np.ndarray, tuple[int, int, int], str]:
    height, width = frame.shape[:2]
    clipped = box.clipped(width, height, padding_percent)
    x1, y1, x2, y2 = [int(round(value)) for value in (clipped.x1, clipped.y1, clipped.x2, clipped.y2)]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return frame, (70, 220, 70), "ACCEPTED"
    if box.source == "accepted_gap_repair":
        color, label = (255, 210, 45), "REPAIR · GAP"
    elif box.source == "accepted_edge_repair":
        color, label = (255, 145, 45), "REPAIR · EDGE"
    else:
        color, label = (70, 220, 70), "ACCEPTED"
    tint = frame.copy()
    cv2.rectangle(tint, (x1, y1), (x2, y2), color, -1)
    frame = cv2.addWeighted(tint, 0.14, frame, 0.86, 0.0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame, color, label


def _preview_windows(
    frame_count: int,
    *,
    preview_scope: Literal["sampled", "full"],
    output_fps: int,
    sample_count: int,
    segment_seconds: float,
) -> list[tuple[int, int, str]]:
    if preview_scope == "full":
        return [(0, frame_count, "FULL ACCEPTED + REPAIR MASK PREVIEW")]
    span = max(1, int(round(float(segment_seconds) * output_fps)))
    count = max(3, int(sample_count))
    if frame_count <= span:
        return [(0, frame_count, "FULL ACCEPTED + REPAIR MASK PREVIEW")]
    centres = np.linspace(span / 2, frame_count - span / 2, count)
    windows: list[tuple[int, int, str]] = []
    for ordinal, centre in enumerate(centres, start=1):
        start = max(0, int(round(centre - span / 2)))
        end = min(frame_count, start + span)
        start = max(0, end - span)
        windows.append((start, end, f"SAMPLE {ordinal}/{count}"))
    return windows


def create_privacy_mask_preview(
    store: ProjectStore,
    *,
    session_id: str,
    native_asset_id: str,
    native_trim_artifact_id: str,
    facemapper_filter_artifact_id: str,
    label: str = "Accepted + repair mask preview",
    output_fps: int = 30,
    preview_scope: Literal["sampled", "full"] = "full",
    preview_sample_count: int = 9,
    preview_segment_seconds: float = 6.0,
    padding_percent: float = 0.0,
    gap_fill_frames: int = 3,
    edge_fill_frames: int = 1,
    repair_maximum_center_shift_per_frame: float = 0.85,
    repair_maximum_scale_ratio: float = 1.75,
    repair_minimum_track_observations: int = 3,
    timestamp_tolerance_ms: float = 5.0,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Render the exact accepted-only mask stream used by repaired Privacy.

    Green boxes are original accepted FaceMapper detections. Cyan boxes are
    motion-verified internal repairs and orange boxes are conservative track-
    edge extrapolations. Review candidates are never loaded by this service.
    """

    token = cancellation or CancellationToken.create()
    if preview_scope not in {"sampled", "full"}:
        raise ValidationError("preview_scope must be sampled or full")
    if output_fps < 5 or output_fps > 60:
        raise ValidationError("Preview FPS must be between 5 and 60")

    session = store.project.sessions[session_id]
    native_asset = session.assets.get(native_asset_id)
    if native_asset is None or native_asset.kind != "pupil_native":
        raise ValidationError("Select a Native Recording asset")
    trim_artifact = session.artifacts.get(native_trim_artifact_id)
    if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    filter_artifact = session.artifacts.get(facemapper_filter_artifact_id)
    if filter_artifact is None or filter_artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
        raise ValidationError("Select a FaceMapper QC/filter artifact")
    if str(filter_artifact.parameters.get("native_trim_artifact_id") or "") != native_trim_artifact_id:
        raise ValidationError("The FaceMapper filter belongs to a different Native-reference trim")
    if native_asset_id not in trim_artifact.source_asset_ids:
        raise ValidationError("The Native-reference trim belongs to another Native recording")

    trim_manifest = _json_file(trim_artifact, "trim_manifest.json")
    request = trim_manifest.get("request") or {}
    window = trim_manifest.get("window") or {}
    try:
        start_ns = int(window["start_media_ns"])
        end_ns = int(window["end_media_ns"])
    except Exception as exc:
        raise ValidationError("Native-reference trim has no valid media window") from exc
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        raise ValidationError("Native-reference trim duration must be positive")
    video_member = str(request.get("native_video_member") or "").strip()
    if not video_member:
        raise ValidationError("Native-reference trim does not identify its NRD video member")

    emit(progress, 0.02, "Checking preview encoder and Native frame map")
    encoder = privacy_encoder_preflight(
        ffmpeg_path,
        preserve_audio=False,
        preset="veryfast",
        crf=23,
        ffprobe_path=ffprobe_path,
    )
    clock = build_clock(
        native_asset.source_path,
        store.cache_dir(),
        video_member=video_member,
        ffprobe_path=ffprobe_path,
        cancellation=token,
        progress=progress,
    )
    trim_frame_map = _load_trim_frame_map(trim_artifact)
    retained = set(trim_frame_map["native_frame_index"].astype(int).tolist())
    source_video = _source_video_path(store, native_asset, video_member)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode Native video for Privacy mask preview: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Native video dimensions are unavailable")

    emit(progress, 0.08, "Building accepted-only motion-repaired mask stream")
    mapping = load_face_mapping(
        filtered_facemapper_root(filter_artifact, variant="analysis_strict"),
        recording_id=native_asset.recording_id,
        frame_abs_ns=clock.capture_ns,
        width=width,
        height=height,
        tolerance_ms=timestamp_tolerance_ms,
        gap_fill_frames=gap_fill_frames,
        edge_fill_frames=edge_fill_frames,
        repair_maximum_center_shift_per_frame=repair_maximum_center_shift_per_frame,
        repair_maximum_scale_ratio=repair_maximum_scale_ratio,
        repair_minimum_track_observations=repair_minimum_track_observations,
    )

    full_output_count = int(math.ceil(duration_ns * output_fps / 1_000_000_000))
    output_pts = np.asarray(
        [round_ratio(index * 1_000_000_000, output_fps) for index in range(full_output_count)],
        dtype=np.int64,
    )
    requested_ns = start_ns + output_pts
    native_indices = np.searchsorted(clock.media_ns, requested_ns, side="right") - 1
    native_indices = np.clip(native_indices, 0, len(clock.media_ns) - 1).astype(int)
    outside = [int(value) for value in native_indices if int(value) not in retained]
    if outside:
        cap.release()
        raise ValidationError(
            f"Preview selection contains {len(set(outside))} Native frame(s) outside the trim; re-export the trim"
        )

    windows = _preview_windows(
        full_output_count,
        preview_scope=preview_scope,
        output_fps=output_fps,
        sample_count=preview_sample_count,
        segment_seconds=preview_segment_seconds,
    )
    artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "privacy_mask_preview") / (".partial_" + artifact_id)
    final = store.derived_dir(session_id, "privacy_mask_preview") / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    silent = staging / "accepted_repair_mask_preview_silent.mp4"
    output_video = staging / "accepted_repair_mask_preview.mp4"
    audit_path = staging / "accepted_repair_mask_preview_audit.csv"
    writer = _output_writer(silent, width, height, float(output_fps))
    written = 0
    audit_rows: list[dict[str, Any]] = []
    accepted_draws = 0
    internal_draws = 0
    edge_draws = 0

    try:
        for window_index, (output_start, output_end, window_label) in enumerate(windows):
            token.raise_if_cancelled()
            positions_by_native: dict[int, list[int]] = defaultdict(list)
            for output_index in range(output_start, output_end):
                positions_by_native[int(native_indices[output_index])].append(output_index)
            selected_native = sorted(positions_by_native)
            if not selected_native:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, selected_native[0])
            current = selected_native[0]
            selected_pointer = 0
            while selected_pointer < len(selected_native):
                token.raise_if_cancelled()
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ProcessingError(
                        f"Native video ended before preview frame {selected_native[selected_pointer]}"
                    )
                target_native = selected_native[selected_pointer]
                if current < target_native:
                    current += 1
                    continue
                if current > target_native:
                    raise ProcessingError("Native preview decoder skipped a requested frame")
                boxes = mapping.applied_by_frame.get(current, [])
                rendered = frame.copy()
                for box in boxes:
                    rendered, _color, _label = _draw_box_overlay(
                        rendered,
                        box,
                        padding_percent=padding_percent,
                    )
                    if box.source == "accepted_gap_repair":
                        internal_draws += 1
                    elif box.source == "accepted_edge_repair":
                        edge_draws += 1
                    else:
                        accepted_draws += 1
                cv2.putText(
                    rendered,
                    "Accepted-only Privacy mask: green accepted | cyan gap repair | orange edge repair",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (225, 248, 47),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    rendered,
                    window_label,
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (238, 238, 232),
                    1,
                    cv2.LINE_AA,
                )
                for output_index in positions_by_native[current]:
                    writer.write(rendered)
                    audit_rows.append(
                        {
                            "preview_frame_index": written,
                            "full_output_frame_index": output_index,
                            "output_pts_ns": int(output_pts[output_index]),
                            "native_frame_index": current,
                            "accepted_box_count": sum(
                                box.source not in {"accepted_gap_repair", "accepted_edge_repair"}
                                for box in boxes
                            ),
                            "internal_repair_box_count": sum(
                                box.source == "accepted_gap_repair" for box in boxes
                            ),
                            "edge_repair_box_count": sum(
                                box.source == "accepted_edge_repair" for box in boxes
                            ),
                            "boxes_json": json.dumps(
                                [
                                    {
                                        "x1": round(box.x1, 3),
                                        "y1": round(box.y1, 3),
                                        "x2": round(box.x2, 3),
                                        "y2": round(box.y2, 3),
                                        "source": box.source,
                                        "track_id": box.track_id,
                                    }
                                    for box in boxes
                                ],
                                separators=(",", ":"),
                            ),
                        }
                    )
                    written += 1
                selected_pointer += 1
                current += 1
            emit(
                progress,
                0.15 + 0.65 * (window_index + 1) / max(1, len(windows)),
                "Rendering accepted + repair mask preview",
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        cap.release()
        writer.release()

    if written <= 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise ProcessingError("Privacy mask preview produced no frames")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    _mux_audio_and_transcode(
        silent,
        source_video,
        output_video,
        source_start_sec=0.0,
        duration_sec=written / float(output_fps),
        preserve_audio=False,
        crf=23,
        preset="veryfast",
        ffmpeg_path=encoder["ffmpeg"]["path"],
        expected_frames=written,
        cancellation=token,
        progress=progress,
    )
    silent.unlink(missing_ok=True)

    settings = {
        "candidate_policy": "accepted_repair",
        "padding_percent": padding_percent,
        "gap_fill_frames": gap_fill_frames,
        "edge_fill_frames": edge_fill_frames,
        "repair_maximum_center_shift_per_frame": repair_maximum_center_shift_per_frame,
        "repair_maximum_scale_ratio": repair_maximum_scale_ratio,
        "repair_minimum_track_observations": repair_minimum_track_observations,
        "timestamp_tolerance_ms": timestamp_tolerance_ms,
    }
    report = {
        "schema_version": "oculostudio-privacy-mask-preview-1.0",
        "engine": PRIVACY_MASK_PREVIEW_ENGINE,
        "native_asset_id": native_asset_id,
        "native_trim_artifact_id": native_trim_artifact_id,
        "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
        "preview_scope": preview_scope,
        "preview_fps": output_fps,
        "preview_frame_count": written,
        "full_output_frame_count": full_output_count,
        "settings": settings,
        "counts": {
            "accepted_draws": accepted_draws,
            "internal_repair_draws": internal_draws,
            "edge_repair_draws": edge_draws,
            "unique_internal_repair_boxes": mapping.internal_repair_count,
            "unique_edge_repair_boxes": mapping.edge_repair_count,
        },
        "created_at": utc_now(),
    }
    atomic_write_json(staging / "accepted_repair_mask_preview_report.json", report)
    staging.rename(final)

    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.PRIVACY_MASK_PREVIEW,
        label=label,
        output_root=final,
        source_asset_ids=[native_asset_id],
        parent_artifact_ids=[native_trim_artifact_id, facemapper_filter_artifact_id],
        parameters={
            **settings,
            "native_asset_id": native_asset_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "preview_scope": preview_scope,
            "preview_fps": output_fps,
            "interaction_id": trim_artifact.parameters.get("interaction_id"),
            "role_id": trim_artifact.parameters.get("role_id"),
            "privacy_mask_preview_engine": PRIVACY_MASK_PREVIEW_ENGINE,
        },
        qc={
            "preview_frame_count": written,
            "accepted_draws": accepted_draws,
            "internal_repair_draws": internal_draws,
            "edge_repair_draws": edge_draws,
            "accepted_internal_repair_box_count": mapping.internal_repair_count,
            "accepted_edge_repair_box_count": mapping.edge_repair_count,
        },
        provenance={
            "preview_only": True,
            "review_candidates_used": False,
            "processing_engine": PRIVACY_MASK_PREVIEW_ENGINE,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Accepted-only repair preview ready")
    return {
        "artifact_id": artifact.id,
        "output_dir": str(final),
        "preview_video": str(final / output_video.name),
        "audit_csv": str(final / audit_path.name),
        "report": report,
    }
