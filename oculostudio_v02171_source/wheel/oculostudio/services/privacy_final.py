from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.native_reference_clock import build_clock, round_ratio
from oculostudio.adapters.pupil import materialized_native_files
from oculostudio.adapters.video import probe_video
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.facemapper_filter import filtered_facemapper_root
from oculostudio.services.privacy import (
    PRIVACY_FRAME_COVERAGE_ENGINE,
    FaceMapping,
    HaarFaceVerifier,
    _is_within_sections,
    _merge_boxes,
    _mux_audio_and_transcode,
    _native_trim_frame_interval,
    _output_writer,
    apply_privacy_masks,
    load_face_mapping,
)
from oculostudio.services.privacy_clip import SUPPORTED_OUTPUT_FPS
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


SYNCHRONIZED_PRIVACY_ENGINE = "direct-native-pts-mask-v3-accepted-repair"


def _json_file(artifact, name: str) -> dict[str, Any]:
    path = find_artifact_file(artifact, names=[name])
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {name} from artifact {artifact.id}: {exc}") from exc


def _source_video_path(store: ProjectStore, native_asset, video_member: str) -> Path:
    """Resolve the exact scene member selected by the Native-reference trim."""
    cache = MaterializationCache(store.cache_dir())
    root = cache.materialize(native_asset)
    candidate = root.joinpath(*PurePosixPath(video_member).parts).resolve()
    if root.resolve() in candidate.parents and candidate.is_file():
        return candidate
    # One-part extracted recordings remain supported as a safe fallback.
    source_video, _time, _info, _metadata = materialized_native_files(root, native_asset.recording_id)
    return source_video


def _load_trim_frame_map(trim_artifact) -> pd.DataFrame:
    path = find_artifact_file(trim_artifact, names=["native_frame_map.csv"])
    frame_map = pd.read_csv(path, low_memory=False)
    if "native_frame_index" not in frame_map.columns:
        raise ValidationError("Native-reference trim frame map is missing native_frame_index")
    try:
        indices = pd.to_numeric(frame_map["native_frame_index"], errors="raise").astype("int64")
    except Exception as exc:
        raise ValidationError("Native-reference trim frame indices are invalid") from exc
    if indices.empty:
        raise ValidationError("Native-reference trim contains no retained Native frames")
    if indices.duplicated().any() or np.any(np.diff(indices.to_numpy()) != 1):
        raise ValidationError("Native-reference trim frame map must be unique and contiguous")
    frame_map = frame_map.copy()
    frame_map["native_frame_index"] = indices
    return frame_map


def create_synchronized_privacy_video(
    store: ProjectStore,
    *,
    session_id: str,
    native_asset_id: str,
    native_trim_artifact_id: str,
    facemapper_asset_id: str | None = None,
    facemapper_filter_artifact_id: str | None = None,
    privacy_mask_preview_artifact_id: str | None = None,
    output_fps: int = 30,
    label: str = "Synchronized Privacy video",
    mode: Literal["gaussian", "pixelate", "solid"] = "gaussian",
    strength: int = 30,
    padding_percent: float = 0.0,
    shape: Literal["rectangle", "ellipse"] = "rectangle",
    gap_fill_frames: int = 3,
    edge_fill_frames: int = 1,
    repair_maximum_center_shift_per_frame: float = 0.85,
    repair_maximum_scale_ratio: float = 1.75,
    repair_minimum_track_observations: int = 3,
    timestamp_tolerance_ms: float = 5.0,
    detector_mode: Literal["facemapper", "haar", "hybrid"] = "facemapper",
    candidate_policy: Literal["accepted_review", "accepted_only", "accepted_repair"] = "accepted_repair",
    exact_filter_match: bool = False,
    coverage_policy: Literal["strict", "allow"] = "strict",
    preserve_audio: bool = True,
    crf: int = 20,
    preset: str = "fast",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Create the final trim-matched, constant-FPS Privacy clip in one pass.

    Unlike the legacy workflow, this function never creates an encoded
    ``privacy_video`` and then decodes/re-encodes it into ``privacy_clip``.
    Native PTS choose the final output frames first; masks are applied directly
    to those Native frames and the final audited clip is encoded once.
    """

    token = cancellation or CancellationToken.create()
    if output_fps not in SUPPORTED_OUTPUT_FPS:
        raise ValidationError(f"Output FPS must be one of {SUPPORTED_OUTPUT_FPS}")
    if mode not in {"gaussian", "pixelate", "solid"}:
        raise ValidationError("Privacy mode must be gaussian, pixelate, or solid")
    if shape not in {"rectangle", "ellipse"}:
        raise ValidationError("Mask shape must be rectangle or ellipse")
    if candidate_policy not in {"accepted_review", "accepted_only", "accepted_repair"}:
        raise ValidationError(
            "candidate_policy must be accepted_review, accepted_only, or accepted_repair"
        )
    if exact_filter_match:
        # Exact preview-to-mask mode: only accepted FaceMapper candidates, with
        # no supplemental detector, temporal gap fill, or geometric expansion.
        candidate_policy = "accepted_only"
        detector_mode = "facemapper"
        gap_fill_frames = 0
        edge_fill_frames = 0
        padding_percent = 0.0
    elif candidate_policy == "accepted_repair":
        # Accepted-only temporal repair never reads review candidates and never
        # supplements the track with Haar detections. Only short, motion-
        # plausible holes in accepted tracks are repaired.
        detector_mode = "facemapper"
        gap_fill_frames = max(0, int(gap_fill_frames))
        edge_fill_frames = max(0, int(edge_fill_frames))
        if facemapper_filter_artifact_id is None:
            raise ValidationError(
                "Accepted-only temporal repair requires a FaceMapper QC/filter artifact"
            )
    if edge_fill_frames < 0 or edge_fill_frames > 10:
        raise ValidationError("edge_fill_frames must be between 0 and 10")
    if gap_fill_frames < 0 or gap_fill_frames > 30:
        raise ValidationError("gap_fill_frames must be between 0 and 30")
    if repair_maximum_center_shift_per_frame <= 0 or repair_maximum_center_shift_per_frame > 5:
        raise ValidationError("repair_maximum_center_shift_per_frame must be in (0, 5]")
    if repair_maximum_scale_ratio < 1 or repair_maximum_scale_ratio > 10:
        raise ValidationError("repair_maximum_scale_ratio must be between 1 and 10")
    if repair_minimum_track_observations < 2:
        raise ValidationError("repair_minimum_track_observations must be at least 2")
    if detector_mode in {"facemapper", "hybrid"} and not (
        facemapper_asset_id or facemapper_filter_artifact_id
    ):
        raise ValidationError(
            "Face Mapper or FaceMapper QC/filter input is required for the selected detector mode"
        )

    session = store.project.sessions[session_id]
    native_asset = session.assets.get(native_asset_id)
    if native_asset is None or native_asset.kind != "pupil_native":
        raise ValidationError("native_asset_id must reference a Native Recording asset")
    trim_artifact = session.artifacts.get(native_trim_artifact_id)
    if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    if native_asset_id not in trim_artifact.source_asset_ids:
        raise ValidationError("The selected Native-reference trim belongs to another Native recording")

    trim_manifest = _json_file(trim_artifact, "trim_manifest.json")
    trim_request = trim_manifest.get("request") or {}
    trim_window = trim_manifest.get("window") or {}
    try:
        start_ns = int(trim_window["start_media_ns"])
        end_ns = int(trim_window["end_media_ns"])
        utc_start_ns = int(trim_window["utc_start_ns"])
        utc_end_ns = int(trim_window["utc_end_ns"])
    except Exception as exc:
        raise ValidationError("Native-reference trim artifact has no valid media/UTC window") from exc
    duration_ns = end_ns - start_ns
    if duration_ns <= 0:
        raise ValidationError("Native-reference trim duration must be positive")
    video_member = str(trim_request.get("native_video_member") or "").strip()
    if not video_member:
        raise ValidationError("Native-reference trim does not identify its exact NRD scene-video member")

    # Validate the complete encoder pair before materializing or decoding a large recording.
    token.raise_if_cancelled()
    emit(progress, 0.01, "Checking FFmpeg, ffprobe and final Privacy settings")
    encoder = privacy_encoder_preflight(
        ffmpeg_path,
        preserve_audio=preserve_audio,
        preset=preset,
        crf=crf,
        ffprobe_path=ffprobe_path,
    )
    verified_ffmpeg = encoder["ffmpeg"]["path"]

    token.raise_if_cancelled()
    emit(progress, 0.03, "Loading exact Native PTS selected by the trim")
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
            f"Native-reference trim lies outside the selected NRD part (duration {clock.end_media_ns/1e9:.6f} s)"
        )
    trim_frame_map = _load_trim_frame_map(trim_artifact)
    trim_coverage = _native_trim_frame_interval(trim_artifact, len(clock.media_ns))
    retained_native_indices = set(trim_frame_map["native_frame_index"].astype(int).tolist())

    # Choose the final constant-FPS samples before any masking or encoding.
    frame_count = int(math.ceil(duration_ns * output_fps / 1_000_000_000))
    if frame_count <= 0:
        raise ValidationError("Native-reference trim contains no final output frames")
    output_pts = np.asarray(
        [round_ratio(index * 1_000_000_000, output_fps) for index in range(frame_count)],
        dtype=np.int64,
    )
    requested_media_ns = start_ns + output_pts
    native_indices = np.searchsorted(clock.media_ns, requested_media_ns, side="right") - 1
    native_indices = np.clip(native_indices, 0, len(clock.media_ns) - 1).astype(int)
    outside = sorted({int(index) for index in native_indices if int(index) not in retained_native_indices})
    if outside:
        preview = ", ".join(str(value) for value in outside[:6])
        suffix = "…" if len(outside) > 6 else ""
        raise ValidationError(
            "Final Privacy-frame selection is not contained in the selected Native-reference trim: "
            f"{len(outside)} frame(s) ({preview}{suffix}). Re-export the Native trim before Privacy."
        )
    if np.any(np.diff(native_indices) < 0):
        raise ValidationError("Final Privacy Native-frame selection is not monotonic")

    source_video = _source_video_path(store, native_asset, video_member)
    video_meta = probe_video(source_video, ffprobe_path=ffprobe_path)
    width, height = int(video_meta.get("width") or 0), int(video_meta.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ProcessingError("Native scene video dimensions are unavailable")
    if len(clock.capture_ns) != len(clock.media_ns):
        raise ValidationError("Native video PTS and capture timestamps have different frame counts")

    emit(progress, 0.08, "Loading Privacy-safe FaceMapper candidates")
    mapping = FaceMapping({}, {}, [], 0, 0, 0, [], 0, 0, "")
    source_asset_ids = [native_asset_id]
    parent_artifact_ids = [native_trim_artifact_id]
    face_source_kind = "none"
    if facemapper_filter_artifact_id:
        filter_artifact = session.artifacts.get(facemapper_filter_artifact_id)
        if filter_artifact is None or filter_artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
            raise ValidationError("Select a FaceMapper QC/filter artifact")
        filter_trim_id = str(filter_artifact.parameters.get("native_trim_artifact_id") or "").strip()
        if filter_trim_id and filter_trim_id != native_trim_artifact_id:
            raise ValidationError("The selected FaceMapper filter belongs to a different Native-reference trim")
        filter_variant = (
            "analysis_strict"
            if candidate_policy in {"accepted_only", "accepted_repair"}
            else "privacy_safe"
        )
        mapping = load_face_mapping(
            filtered_facemapper_root(filter_artifact, variant=filter_variant),
            recording_id=native_asset.recording_id,
            frame_abs_ns=clock.capture_ns,
            width=width,
            height=height,
            tolerance_ms=timestamp_tolerance_ms,
            gap_fill_frames=gap_fill_frames,
            edge_fill_frames=edge_fill_frames if candidate_policy == "accepted_repair" else 0,
            repair_maximum_center_shift_per_frame=repair_maximum_center_shift_per_frame,
            repair_maximum_scale_ratio=repair_maximum_scale_ratio,
            repair_minimum_track_observations=repair_minimum_track_observations,
        )
        parent_artifact_ids.append(facemapper_filter_artifact_id)
        face_source_kind = f"filtered_{filter_variant}"
    elif facemapper_asset_id:
        face_asset = session.assets.get(facemapper_asset_id)
        if face_asset is None or face_asset.kind != "face_mapper":
            raise ValidationError("facemapper_asset_id must reference a Face Mapper asset")
        face_root = MaterializationCache(store.cache_dir()).materialize(face_asset)
        mapping = load_face_mapping(
            face_root,
            recording_id=native_asset.recording_id or face_asset.recording_id,
            frame_abs_ns=clock.capture_ns,
            width=width,
            height=height,
            tolerance_ms=timestamp_tolerance_ms,
            gap_fill_frames=gap_fill_frames,
            edge_fill_frames=0,
            repair_maximum_center_shift_per_frame=repair_maximum_center_shift_per_frame,
            repair_maximum_scale_ratio=repair_maximum_scale_ratio,
            repair_minimum_track_observations=repair_minimum_track_observations,
        )
        source_asset_ids.append(facemapper_asset_id)
        face_source_kind = "raw_facemapper"

    if candidate_policy == "accepted_repair":
        preview_artifact = session.artifacts.get(privacy_mask_preview_artifact_id or "")
        if preview_artifact is None or preview_artifact.kind != ArtifactKind.PRIVACY_MASK_PREVIEW:
            raise ValidationError(
                "Render and review a full Accepted + repair mask preview at the final output FPS before creating the final Privacy video"
            )
        preview_parameters = preview_artifact.parameters or {}
        expected_preview = {
            "native_asset_id": native_asset_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "gap_fill_frames": int(gap_fill_frames),
            "edge_fill_frames": int(edge_fill_frames),
            "repair_maximum_center_shift_per_frame": float(
                repair_maximum_center_shift_per_frame
            ),
            "repair_maximum_scale_ratio": float(repair_maximum_scale_ratio),
            "repair_minimum_track_observations": int(repair_minimum_track_observations),
            "timestamp_tolerance_ms": float(timestamp_tolerance_ms),
            "padding_percent": float(padding_percent),
            "preview_scope": "full",
            "preview_fps": int(output_fps),
        }
        mismatches: list[str] = []
        for key, expected in expected_preview.items():
            actual = preview_parameters.get(key)
            if isinstance(expected, float):
                try:
                    equal = abs(float(actual) - expected) <= 1e-9
                except Exception:
                    equal = False
            else:
                equal = actual == expected
            if not equal:
                mismatches.append(key)
        if mismatches:
            raise ValidationError(
                "The accepted+repair preview does not match the current final-mask settings: "
                + ", ".join(mismatches)
                + ". Render the mask preview again."
            )
        parent_artifact_ids.append(preview_artifact.id)

    selected_unique = sorted({int(value) for value in native_indices})
    warnings: list[str] = []
    if candidate_policy == "accepted_only":
        warnings.append(
            "Accepted-only masking excludes review candidates. Confirm the full-length QC preview before publication."
        )
    if candidate_policy == "accepted_repair":
        warnings.append(
            "Accepted-only temporal repair used: review candidates and Haar detections were excluded; "
            "only short motion-plausible gaps in accepted tracks were repaired."
        )
    if exact_filter_match:
        warnings.append(
            "Exact QC-match mode used: accepted candidates only; no Haar supplement, interpolation, or padding."
        )
    if mapping.unmatched_timestamps:
        warnings.append(f"{mapping.unmatched_timestamps} Face Mapper timestamps did not match scene frames")
    if mapping.section_intervals_ns:
        outside_sections = [
            index
            for index in selected_unique
            if _is_within_sections(
                min(max(int(clock.capture_ns[index]), utc_start_ns), utc_end_ns - 1),
                mapping.section_intervals_ns,
            )
            is False
        ]
        if outside_sections:
            warnings.append(
                f"{len(outside_sections)} retained Native frame(s) extend outside Face Mapper section coverage"
            )
    blocking_warnings = [
        row for row in warnings
        if "did not match scene frames" in row or "outside Face Mapper section coverage" in row
    ]
    if coverage_policy == "strict" and blocking_warnings:
        raise ValidationError("Strict privacy policy blocked processing: " + "; ".join(blocking_warnings))

    artifact_id = new_id("artifact")
    category = store.derived_dir(session_id, "privacy_clip")
    staging = category / (".partial_" + artifact_id)
    final = category / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    silent = staging / "privacy_final_silent.mp4"
    output_video = staging / f"{safe_name(label, fallback='privacy_final')}.mp4"
    frame_map_path = staging / "privacy_trimmed_frame_map.csv"
    audit_path = staging / "privacy_frame_audit.csv"
    report_path = staging / "privacy_clip_report.json"
    privacy_report_path = staging / "privacy_report.json"

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        shutil.rmtree(staging, ignore_errors=True)
        raise ProcessingError(f"Cannot decode Native scene video: {source_video}")
    writer = _output_writer(silent, width, height, float(output_fps))
    verifier = HaarFaceVerifier() if detector_mode in {"haar", "hybrid"} else None
    targets_by_native: dict[int, list[int]] = defaultdict(list)
    for output_index, native_index in enumerate(native_indices):
        targets_by_native[int(native_index)].append(output_index)
    first_native, last_native = int(native_indices[0]), int(native_indices[-1])
    frame_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_ordinal = -1
    written = 0
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_native)
        for native_index in range(first_native, last_native + 1):
            token.raise_if_cancelled()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ProcessingError(f"Native scene video decode ended at frame {native_index}")
            output_targets = targets_by_native.get(native_index, [])
            if not output_targets:
                continue
            source_ordinal += 1
            facemapper_boxes = mapping.applied_by_frame.get(native_index, [])
            haar_boxes = verifier.detect(frame) if verifier else []
            if detector_mode == "facemapper":
                boxes = facemapper_boxes
            elif detector_mode == "haar":
                boxes = haar_boxes
            else:
                boxes = _merge_boxes(facemapper_boxes, haar_boxes)
            processed = apply_privacy_masks(
                frame,
                boxes,
                mode=mode,
                strength=strength,
                shape=shape,
                padding_percent=padding_percent,
            )
            for duplicate_ordinal, output_index in enumerate(output_targets):
                writer.write(processed)
                frame_rows.append(
                    {
                        "output_frame_index": int(output_index),
                        "output_pts_ns": int(output_pts[output_index]),
                        "privacy_source_frame_index": int(source_ordinal),
                        "native_frame_index": int(native_index),
                        "native_video_pts_ns": int(clock.media_ns[native_index]),
                        "native_capture_utc_ns": int(clock.capture_ns[native_index]),
                        "clip_time_ns": int(output_pts[output_index]),
                        "requested_native_time_ns": int(requested_media_ns[output_index]),
                        "native_selection_error_ns": int(
                            requested_media_ns[output_index] - int(clock.media_ns[native_index])
                        ),
                        "duplicate_ordinal_for_native": int(duplicate_ordinal),
                        "total_mask_count": int(len(boxes)),
                    }
                )
                written += 1
            section_probe_ns = min(max(int(clock.capture_ns[native_index]), utc_start_ns), utc_end_ns - 1)
            audit_rows.append(
                {
                    "privacy_source_frame_index": int(source_ordinal),
                    "native_frame_index": int(native_index),
                    "native_video_pts_ns": int(clock.media_ns[native_index]),
                    "timestamp_ns": int(clock.capture_ns[native_index]),
                    "output_frame_first": int(output_targets[0]),
                    "output_frame_last": int(output_targets[-1]),
                    "output_occurrences": int(len(output_targets)),
                    "facemapper_raw_face_count": int(len(mapping.raw_by_frame.get(native_index, []))),
                    "facemapper_applied_face_count": int(len(facemapper_boxes)),
                    "haar_face_count": int(len(haar_boxes)),
                    "total_mask_count": int(len(boxes)),
                    "interpolated_mask_count": int(
                        sum(box.source in {"interpolated", "accepted_gap_repair", "accepted_edge_repair"} for box in boxes)
                    ),
                    "accepted_gap_repair_mask_count": int(
                        sum(box.source == "accepted_gap_repair" for box in boxes)
                    ),
                    "accepted_edge_repair_mask_count": int(
                        sum(box.source == "accepted_edge_repair" for box in boxes)
                    ),
                    "within_facemapper_section": _is_within_sections(
                        section_probe_ns, mapping.section_intervals_ns
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
            if source_ordinal % 10 == 0:
                emit(
                    progress,
                    0.12 + 0.62 * written / max(1, frame_count),
                    f"Masking final synchronized frames ({written}/{frame_count})",
                )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        cap.release()
        writer.release()

    if written != frame_count:
        shutil.rmtree(staging, ignore_errors=True)
        raise ProcessingError(f"Wrote {written} synchronized frames but expected {frame_count}")
    frame_rows.sort(key=lambda row: row["output_frame_index"])
    pd.DataFrame(frame_rows).to_csv(frame_map_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    try:
        emit(progress, 0.78, "Encoding final synchronized Privacy video and preserving audio")
        ffmpeg_command = _mux_audio_and_transcode(
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

    unique_native_count = len(selected_unique)
    masked_output_frames = sum(
        int(row["output_occurrences"]) for row in audit_rows if int(row["total_mask_count"]) > 0
    )
    frames_in_section = [row for row in audit_rows if row["within_facemapper_section"] is True]
    section_output_occurrences = sum(int(row["output_occurrences"]) for row in frames_in_section)
    section_masked_occurrences = sum(
        int(row["output_occurrences"])
        for row in frames_in_section
        if int(row["total_mask_count"]) > 0
    )
    qc = {
        "output_frame_count": frame_count,
        "output_fps": output_fps,
        "unique_native_frames": unique_native_count,
        "duplicated_output_frames": int(frame_count - unique_native_count),
        "trim_native_frame_count": int(trim_coverage["frame_count"]),
        "dropped_trim_native_frames": int(trim_coverage["frame_count"] - unique_native_count),
        "frames_with_masks": masked_output_frames,
        "mask_frame_fraction": masked_output_frames / frame_count,
        "frames_in_facemapper_sections": section_output_occurrences,
        "section_frames_with_masks": section_masked_occurrences,
        "section_mask_frame_fraction": (
            section_masked_occurrences / section_output_occurrences
            if section_output_occurrences
            else None
        ),
        "matched_detection_timestamps": mapping.matched_timestamps,
        "unmatched_detection_timestamps": mapping.unmatched_timestamps,
        "invalid_detection_rows": mapping.invalid_rows,
        "interpolated_box_count": mapping.interpolated_count,
        "accepted_internal_repair_box_count": mapping.internal_repair_count,
        "accepted_edge_repair_box_count": mapping.edge_repair_count,
        "track_count": mapping.track_count,
        "candidate_policy": candidate_policy,
        "exact_filter_match": bool(exact_filter_match),
        "supplemental_detector": detector_mode,
        "median_timestamp_match_error_ms": (
            float(np.median(mapping.match_errors_ns) / 1e6) if mapping.match_errors_ns else None
        ),
        "direct_single_pass": True,
        "privacy_source_coverage": True,
        "privacy_limit": (
            "Frames without detected boxes may still contain an undetected face; "
            "visually review before publication."
        ),
    }

    # Store final paths in manifests before the atomic directory rename.
    final_video = final / output_video.name
    final_frame_map = final / frame_map_path.name
    final_audit = final / audit_path.name
    final_report = final / report_path.name
    final_privacy_report = final / privacy_report_path.name
    time_input = trim_request.get("timecode_interpretation") or {
        "schema_version": "oculostudio-video-time-input-1.0",
        "start": {
            "nanoseconds": str(start_ns),
            "normalized_time": trim_request.get("start_timecode"),
        },
        "duration": {
            "nanoseconds": str(duration_ns),
            "normalized_time": trim_request.get("duration_timecode"),
        },
        "end": {"nanoseconds": str(end_ns)},
        "timecode_fps": trim_request.get("timecode_fps", output_fps),
    }
    mask_options = {
        "mode": mode,
        "strength": strength,
        "padding_percent": padding_percent,
        "shape": shape,
        "gap_fill_frames": gap_fill_frames,
        "edge_fill_frames": edge_fill_frames,
        "repair_maximum_center_shift_per_frame": repair_maximum_center_shift_per_frame,
        "repair_maximum_scale_ratio": repair_maximum_scale_ratio,
        "repair_minimum_track_observations": repair_minimum_track_observations,
        "timestamp_tolerance_ms": timestamp_tolerance_ms,
        "detector_mode": detector_mode,
        "candidate_policy": candidate_policy,
        "exact_filter_match": bool(exact_filter_match),
        "coverage_policy": coverage_policy,
        "preserve_audio": preserve_audio,
        "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
        "privacy_mask_preview_artifact_id": privacy_mask_preview_artifact_id,
        "native_trim_artifact_id": native_trim_artifact_id,
        "face_source_kind": face_source_kind,
        "frame_coverage_engine": PRIVACY_FRAME_COVERAGE_ENGINE,
        "synchronized_privacy_engine": SYNCHRONIZED_PRIVACY_ENGINE,
        "output_fps": output_fps,
    }
    report = {
        "schema_version": "oculostudio-privacy-clip-1.1",
        "artifact_id": artifact_id,
        "privacy_artifact_id": None,
        "native_trim_artifact_id": native_trim_artifact_id,
        "native_asset_id": native_asset.id,
        "native_video_member": video_member,
        "time_input": time_input,
        "window": {
            "start_media_ns": str(start_ns),
            "end_media_ns": str(end_ns),
            "duration_ns": str(duration_ns),
            "utc_start_ns": str(utc_start_ns),
            "utc_end_ns": str(utc_end_ns),
        },
        "output": {
            "fps": output_fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / output_fps,
            "width": width,
            "height": height,
            "video": str(final_video),
            "frame_map": str(final_frame_map),
            "privacy_audit": str(final_audit),
        },
        "timing_policy": "constant_fps_direct_native_PTS_masking",
        "processing_engine": SYNCHRONIZED_PRIVACY_ENGINE,
        "source_privacy_video": None,
        "mask_options": mask_options,
        "warnings": warnings,
        "qc": qc,
        "ffmpeg_command": ffmpeg_command,
        "encoder_preflight": encoder,
        "created_at": utc_now(),
    }
    privacy_report = {
        "schema_version": "oculostudio-privacy-report-2.0",
        "native_trim_artifact_id": native_trim_artifact_id,
        "source_asset_ids": source_asset_ids,
        "source_video": str(source_video),
        "output_video": str(final_video),
        "requested_interval_ns": [utc_start_ns, utc_end_ns],
        "actual_media_interval_ns": [start_ns, end_ns],
        "frame_coverage": {
            "engine": PRIVACY_FRAME_COVERAGE_ENGINE,
            "selection_policy": "final_constant_fps_samples_within_native_trim_display_overlap",
            "native_trim": trim_coverage,
        },
        "options": mask_options,
        "warnings": warnings,
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(report_path, report)
    atomic_write_json(privacy_report_path, privacy_report)

    try:
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        artifact = ArtifactStore(store).register(
            session_id=session_id,
            kind=ArtifactKind.PRIVACY_CLIP,
            label=label,
            output_root=final,
            source_asset_ids=source_asset_ids,
            parent_artifact_ids=parent_artifact_ids,
            parameters=mask_options,
            qc=qc,
            provenance={
                "native_trim_artifact_id": native_trim_artifact_id,
                "frame_selection": "native_PTS_direct_to_final_output",
                "single_pass_privacy": True,
                "processing_engine": SYNCHRONIZED_PRIVACY_ENGINE,
                "review_candidates_used": candidate_policy == "accepted_review",
                "accepted_only_seed_source": candidate_policy in {"accepted_only", "accepted_repair"},
                "temporal_repair_applied": candidate_policy == "accepted_repair",
                "mask_preview_verified": bool(
                    candidate_policy != "accepted_repair" or privacy_mask_preview_artifact_id
                ),
            },
            artifact_id=artifact_id,
        )
    except BaseException:
        # Keep the project manifest and derived tree consistent if registration
        # fails after the atomic directory rename.
        shutil.rmtree(final, ignore_errors=True)
        raise
    emit(progress, 1.0, "Synchronized Privacy video complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "video": str(final_video),
        "frame_map": str(final_frame_map),
        "audit_csv": str(final_audit),
        "report": str(final_report),
        "privacy_report": str(final_privacy_report),
    }
