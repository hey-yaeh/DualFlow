from __future__ import annotations

import json
import math
import re
import shutil
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.pupil import materialized_native_files
from oculostudio.domain.models import ArtifactKind, ArtifactRecord
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file, find_video_file
from oculostudio.services.body_primitives import body_geometry_integrity_from_qc
from oculostudio.services.privacy import _mux_audio_and_transcode, _output_writer
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now

ReviewMode = Literal["face", "body", "fusion"]

MANUAL_REVIEW_SCHEMA = "oculostudio-face-aoi-manual-review-1.0"
BODY_MANUAL_REVIEW_SCHEMA = "oculostudio-body-aoi-manual-review-1.0"
MANUAL_REVIEW_ACTIONS = {"accept_auto", "override", "exclude", "defer", "clear"}
MANUAL_REVIEW_RESOLVED_ACTIONS = {"accept_auto", "override", "exclude"}
MANUAL_REVIEW_RELATIONS = {"partner", "self", "other_person"}
MANUAL_REVIEW_REGIONS = {"left_eye", "right_eye", "nose", "mouth", "other_face", "face", "no_face"}
BODY_MANUAL_REVIEW_RELATIONS = {"partner", "self", "other"}
BODY_MANUAL_REVIEW_REGIONS = {
    "head", "neck", "torso", "pelvis",
    "left_arm", "left_upper_arm", "left_forearm", "left_hand",
    "right_arm", "right_upper_arm", "right_forearm", "right_hand",
    "left_leg", "left_thigh", "left_lower_leg", "left_foot",
    "right_leg", "right_thigh", "right_lower_leg", "right_foot",
    "no_body",
}
_MANUAL_REVIEW_LOCK = threading.RLock()


def _require_body_geometry_review_ready(artifact: ArtifactRecord) -> dict[str, Any]:
    report = body_geometry_integrity_from_qc(artifact.qc or {})
    if bool(report.get("review_ready")):
        return report
    failures = report.get("failures") or []
    reason = "; ".join(str(row.get("message") or row.get("code")) for row in failures[:4])
    raise ValidationError(
        "Body AOI human review is blocked because torso/body geometry is not review-ready. "
        + (reason or "Regenerate Body AOI with the current geometry acceptance gate.")
    )


_FACE_FILTER_REASON_EXPLANATIONS = {
    "landmarks_outside_bbox": "One or more facial landmarks fell outside the exported face box.",
    "oversized_bbox_compact_landmarks": "The face box was unusually large while the landmarks occupied only a small cluster.",
    "landmark_cluster_off_center": "The landmark cluster was far from the centre of the face box.",
    "invalid_landmark_vertical_order": "The eyes, nose and mouth were not in a plausible vertical order.",
    "temporal_scale_jump": "The detected face size changed abruptly relative to neighbouring frames.",
    "short_track": "The detection belonged to a track that was too short to trust for analysis.",
    "large_face_requires_review": "The face occupied an unusually large image area and was kept only for review.",
    "implausible_bbox_aspect": "The exported box had an implausible face aspect ratio.",
    "extreme_face_area": "The exported box covered an implausibly large part of the image.",
    "oversized_compact_track": "A repeated large box contained a compact landmark cluster, suggesting a false-positive track.",
    "dominant_bbox_false_positive": "A repeated dominant box was classified as a likely non-face false positive.",
    "invalid_or_unmatched_detection": "The source row could not be converted into a valid frame-aligned face detection.",
}


def _read_csv(artifact: ArtifactRecord, filename: str) -> pd.DataFrame:
    return pd.read_csv(find_artifact_file(artifact, names=[filename]), low_memory=False)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _row_dict(row: pd.Series | dict[str, Any], columns: list[str] | None = None) -> dict[str, Any]:
    source = row if isinstance(row, dict) else row.to_dict()
    if columns is not None:
        source = {key: source.get(key) for key in columns if key in source}
    return {str(key): _jsonable(value) for key, value in source.items()}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read review metadata {path.name}: {exc}") from exc


def _confidence_column(mode: ReviewMode) -> str:
    return {"face": "face_aoi_confidence", "body": "body_confidence", "fusion": "final_confidence"}[mode]


def _region_column(mode: ReviewMode) -> str:
    return {"face": "face_social_region", "body": "body_social_region", "fusion": "final_target_region"}[mode]


def _derive_face_social_region(row: pd.Series) -> str:
    raw = str(row.get("face_region") or "unknown")
    relation = str(row.get("target_relation") or "")
    status = str(row.get("identity_status") or "")
    matched = status in {
        "matched_participant",
        "projection_locked_participant",
        "projection_cluster_inherits_participant",
        "duplicate_inherits_matched_participant",
        "temporal_bridge_matched_participant",
        "temporal_edge_matched_participant",
        "manual_review_override",
        "manual_review_accepted_auto",
    } or status.startswith(("auto_identity_", "auto_projection_", "auto_track_", "auto_fixation_"))
    if relation == "partner":
        if not matched:
            return "partner_face_candidate"
        if raw in {"other_face", "face"}:
            return "partner_face_unspecified"
        return f"partner_{raw}"
    if relation == "self":
        if raw in {"other_face", "face"}:
            return "self_face_unspecified"
        return f"self_{raw}"
    if relation in {"ambiguous_partner", "partner_candidate"}:
        return "partner_face_candidate"
    if relation.startswith("ambiguous_"):
        return "ambiguous_person_face"
    if relation == "other_person":
        return "other_person_face"
    return raw


def _resolved_manual_relation(row: pd.Series, requested: str | None = None) -> str:
    relation = str(requested or row.get("target_relation") or "").strip()
    if relation in MANUAL_REVIEW_RELATIONS:
        return relation
    candidate = str(row.get("candidate_relation") or "").strip()
    if candidate in MANUAL_REVIEW_RELATIONS:
        return candidate
    if "partner" in relation:
        return "partner"
    if "self" in relation:
        return "self"
    return "other_person"


def _entity_column(mode: ReviewMode) -> str:
    return {"face": "face_entity_id", "body": "body_entity_id", "fusion": "final_entity_id"}[mode]


def _status_column(mode: ReviewMode) -> str:
    return {"face": "assignment_status", "body": "assignment_status", "fusion": "fusion_status"}[mode]


def _artifact_for_mode(session, mode: ReviewMode, artifact_id: str) -> ArtifactRecord:
    artifact = session.artifacts.get(artifact_id)
    expected = {
        "face": ArtifactKind.FACE_AOI,
        "body": ArtifactKind.BODY_AOI,
        "fusion": ArtifactKind.TARGET_FUSION,
    }[mode]
    if artifact is None or artifact.kind != expected:
        raise ValidationError(f"Select a {expected.value} artifact for {mode} review")
    return artifact


def _parent_by_kind(session, artifact: ArtifactRecord, kind: ArtifactKind) -> ArtifactRecord | None:
    for parent_id in artifact.parent_artifact_ids:
        parent = session.artifacts.get(parent_id)
        if parent is not None and parent.kind == kind:
            return parent
    return None


def _face_artifact_for_review(session, mode: ReviewMode, artifact: ArtifactRecord) -> ArtifactRecord | None:
    if mode == "face":
        return artifact
    if mode == "fusion":
        return _parent_by_kind(session, artifact, ArtifactKind.FACE_AOI)
    return None


def _body_artifact_for_review(session, mode: ReviewMode, artifact: ArtifactRecord) -> ArtifactRecord | None:
    if mode == "body":
        return artifact
    if mode == "fusion":
        return _parent_by_kind(session, artifact, ArtifactKind.BODY_AOI)
    return None


def _participant_artifact_for_review(session, mode: ReviewMode, artifact: ArtifactRecord) -> ArtifactRecord | None:
    face = _face_artifact_for_review(session, mode, artifact)
    if face is None:
        return None
    requested = face.provenance.get("participant_association_artifact_id")
    candidate = session.artifacts.get(requested) if requested else None
    if candidate is not None and candidate.kind == ArtifactKind.PARTICIPANT_ASSOCIATION:
        return candidate
    return _parent_by_kind(session, face, ArtifactKind.PARTICIPANT_ASSOCIATION)


def _facemapper_filter_artifact_for_review(
    session, mode: ReviewMode, artifact: ArtifactRecord
) -> ArtifactRecord | None:
    face = _face_artifact_for_review(session, mode, artifact)
    if face is None:
        return None
    requested = face.provenance.get("facemapper_filter_artifact_id")
    candidate = session.artifacts.get(requested) if requested else None
    if candidate is not None and candidate.kind == ArtifactKind.FACE_MAPPER_FILTER:
        return candidate
    return _parent_by_kind(session, face, ArtifactKind.FACE_MAPPER_FILTER)


def _native_trim_id(session, mode: ReviewMode, artifact: ArtifactRecord) -> str | None:
    if mode == "face":
        return artifact.provenance.get("native_trim_artifact_id")
    if mode == "body":
        return artifact.provenance.get("native_trim_artifact_id")
    face = _face_artifact_for_review(session, mode, artifact)
    return face.provenance.get("native_trim_artifact_id") if face else None


def _default_video(session, mode: ReviewMode, artifact: ArtifactRecord) -> ArtifactRecord | None:
    face = _face_artifact_for_review(session, mode, artifact)
    if face:
        requested = face.provenance.get("privacy_clip_artifact_id")
        candidate = session.artifacts.get(requested) if requested else None
        if candidate is not None and candidate.kind == ArtifactKind.PRIVACY_CLIP:
            return candidate
    trim_id = _native_trim_id(session, mode, artifact)
    candidates = [
        row for row in session.artifacts.values()
        if row.kind == ArtifactKind.PRIVACY_CLIP
        and (not trim_id or row.provenance.get("native_trim_artifact_id") == trim_id)
    ]
    return candidates[-1] if candidates else None


def _video_artifact(session, mode: ReviewMode, artifact: ArtifactRecord, requested: str | None) -> ArtifactRecord:
    if requested:
        video = session.artifacts.get(requested)
        if video is None or video.kind != ArtifactKind.PRIVACY_CLIP:
            raise ValidationError("AOI Review requires a trimmed Privacy video artifact")
        return video
    video = _default_video(session, mode, artifact)
    if video is None:
        raise ValidationError("No compatible trimmed Privacy video is available for this AOI artifact")
    return video


def _native_asset_for_privacy_clip(session, video_artifact: ArtifactRecord, report: dict[str, Any]):
    native_asset_id = str(report.get("native_asset_id") or "").strip()
    native = session.assets.get(native_asset_id) if native_asset_id else None
    if native is not None and native.kind == "pupil_native":
        return native
    candidates = [
        session.assets.get(asset_id)
        for asset_id in video_artifact.source_asset_ids
        if session.assets.get(asset_id) is not None
        and session.assets[asset_id].kind == "pupil_native"
    ]
    return candidates[0] if len(candidates) == 1 else None


def _unblurred_source_descriptor(
    store: ProjectStore,
    *,
    session_id: str,
    video_artifact: ArtifactRecord,
    frame_map: pd.DataFrame,
    video_info: dict[str, Any],
) -> dict[str, Any]:
    """Describe the original Native scene video for local human review.

    The Privacy clip remains the default and remains the only source used for
    exported QC videos.  The original video is streamed only after the user
    explicitly enables it in the local Review Player.  Frame-to-overlay locking
    uses the audited Native PTS carried by the Privacy clip frame map.
    """
    unavailable = {
        "available": False,
        "local_review_only": True,
        "export_uses_privacy_clip": True,
    }
    report_path = find_artifact_file(
        video_artifact, names=["privacy_clip_report.json"], required=False
    )
    if report_path is None:
        return {**unavailable, "reason": "privacy_clip_report_missing"}
    try:
        report = _load_json(report_path)
    except Exception:
        return {**unavailable, "reason": "privacy_clip_report_unreadable"}
    session = store.project.sessions[session_id]
    native = _native_asset_for_privacy_clip(session, video_artifact, report)
    if native is None:
        return {**unavailable, "reason": "native_source_not_resolved"}
    if "native_video_pts_ns" not in frame_map.columns:
        return {**unavailable, "reason": "native_frame_pts_missing"}
    pts = pd.to_numeric(frame_map["native_video_pts_ns"], errors="coerce")
    if pts.isna().all():
        return {**unavailable, "reason": "native_frame_pts_missing"}
    pts = pts.ffill().bfill().fillna(0).astype("int64")
    window = report.get("window") or {}
    try:
        source_start_ns = int(window.get("start_media_ns"))
    except Exception:
        source_start_ns = int(pts.iloc[0]) if len(pts) else 0
    try:
        source_end_ns = int(window.get("end_media_ns"))
    except Exception:
        fallback_step = int(round(1e9 / max(1e-6, float(video_info.get("fps") or 30.0))))
        source_end_ns = int(pts.iloc[-1]) + fallback_step if len(pts) else source_start_ns
    if source_end_ns <= source_start_ns:
        source_end_ns = source_start_ns + int(round(float(video_info.get("duration_seconds") or 0.0) * 1e9))
    return {
        "available": True,
        "url": f"/api/aoi-review/{session_id}/unblurred-video/{video_artifact.id}",
        "label": "Original Native scene video · unblurred",
        "native_asset_id": native.id,
        "native_video_member": report.get("native_video_member"),
        "frame_pts_ns": [int(value) for value in pts.tolist()],
        "source_start_sec": float(source_start_ns / 1e9),
        "source_end_sec": float(source_end_ns / 1e9),
        "local_duration_sec": float(video_info.get("duration_seconds") or 0.0),
        "original_pixels_preserved": True,
        "reencoded": False,
        "local_review_only": True,
        "export_uses_privacy_clip": True,
        "privacy_notice": (
            "Unblurred video is shown only inside the local annotation player. "
            "Review decisions and exported QC video continue to use the selected Privacy clip."
        ),
    }


def review_unblurred_video_path(
    store: ProjectStore, session_id: str, video_artifact_id: str
) -> Path:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    video_artifact = session.artifacts.get(video_artifact_id)
    if video_artifact is None or video_artifact.kind != ArtifactKind.PRIVACY_CLIP:
        raise ValidationError("Select a trimmed Privacy clip")
    report_path = find_artifact_file(video_artifact, names=["privacy_clip_report.json"])
    report = _load_json(report_path)
    native = _native_asset_for_privacy_clip(session, video_artifact, report)
    if native is None:
        raise ValidationError("The original Native recording used by this Privacy clip is unavailable")
    cache = MaterializationCache(store.cache_dir() / "materialized")
    materialized_root = Path(cache.materialize(native)).resolve()
    member = str(report.get("native_video_member") or "").strip()
    if member:
        candidate = materialized_root.joinpath(*PurePosixPath(member).parts).resolve()
        if (candidate == materialized_root or materialized_root in candidate.parents) and candidate.is_file():
            return candidate
    recording_id = str(report.get("recording_id") or native.recording_id or "")
    video_path, _timestamps, _info, _member = materialized_native_files(
        materialized_root, recording_id
    )
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValidationError("The original Native scene video is unavailable")
    return video_path


def _video_info(video_artifact: ArtifactRecord) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    video = find_video_file(video_artifact, preferred_tokens=("privacy", "trimmed"))
    frame_map = _read_csv(video_artifact, "privacy_trimmed_frame_map.csv")
    report = _load_json(find_artifact_file(video_artifact, names=["privacy_clip_report.json"]))
    required = {"output_frame_index", "output_pts_ns", "native_frame_index", "native_capture_utc_ns"}
    if not required.issubset(frame_map.columns):
        raise ValidationError("Privacy clip frame map is incomplete")
    frame_map = frame_map.sort_values("output_frame_index").reset_index(drop=True)
    fps = float(report.get("output", {}).get("fps") or video_artifact.qc.get("output_fps") or 30.0)
    if fps <= 0:
        raise ValidationError("Privacy clip has an invalid output FPS")
    width = report.get("output", {}).get("width")
    height = report.get("output", {}).get("height")
    if not width or not height:
        cap = cv2.VideoCapture(str(video))
        try:
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or width
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or height
        finally:
            cap.release()
    pts = pd.to_numeric(frame_map["output_pts_ns"], errors="coerce").fillna(0).astype("int64").to_numpy()
    origin = int(pts[0]) if len(pts) else 0
    relative_pts = [int(value - origin) for value in pts]
    info = {
        "fps": fps,
        "frame_count": int(len(frame_map)),
        "duration_seconds": float(len(frame_map) / fps),
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "timeline_origin_ns": origin,
        "frame_pts_ns": relative_pts,
    }
    return video, frame_map, info


def _ray_table_for_body(session, body_artifact: ArtifactRecord) -> pd.DataFrame | None:
    gaze_id = body_artifact.provenance.get("gaze_rays_artifact_id")
    gaze = session.artifacts.get(gaze_id) if gaze_id else _parent_by_kind(session, body_artifact, ArtifactKind.GAZE_RAYS)
    if gaze is None:
        return None
    candidates = [
        Path(file.path) for file in gaze.files
        if Path(file.path).suffix.lower() in {".csv", ".parquet"}
        and "gaze_rays" in Path(file.path).name.lower()
        and Path(file.path).exists()
    ]
    if not candidates:
        return None
    path = candidates[0]
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def _attach_body_review_fields(session, body_artifact: ArtifactRecord, samples: pd.DataFrame) -> pd.DataFrame:
    output = samples.copy()
    needed = {"gaze_x_px", "gaze_y_px"}
    if needed.issubset(output.columns):
        return output
    rays = _ray_table_for_body(session, body_artifact)
    if rays is None:
        return output
    source_col = next((c for c in ["gaze3d_input_row_index", "oculo_source_row_index", "source_row_index"] if c in rays.columns), None)
    body_source_col = "source_gaze_row_index" if "source_gaze_row_index" in output.columns else None
    x_col = next((c for c in ["gaze x [px]", "gaze_x_px", "gaze x", "x"] if c in rays.columns), None)
    y_col = next((c for c in ["gaze y [px]", "gaze_y_px", "gaze y", "y"] if c in rays.columns), None)
    timestamp_col = next((c for c in ["timestamp [ns]", "timestamp_ns", "gaze timestamp [ns]"] if c in rays.columns), None)
    if not x_col or not y_col:
        return output
    keep = [x_col, y_col]
    if source_col and body_source_col:
        keep = [source_col, *keep]
        extra = rays[keep].copy().rename(columns={source_col: body_source_col, x_col: "gaze_x_px", y_col: "gaze_y_px"})
        extra = extra.drop_duplicates(body_source_col, keep="first")
        return output.merge(extra, on=body_source_col, how="left", validate="many_to_one")
    if timestamp_col and "timestamp_ns" in output.columns:
        keep = [timestamp_col, *keep]
        extra = rays[keep].copy().rename(columns={timestamp_col: "timestamp_ns", x_col: "gaze_x_px", y_col: "gaze_y_px"})
        extra = extra.drop_duplicates("timestamp_ns", keep="first")
        return output.merge(extra, on="timestamp_ns", how="left", validate="many_to_one")
    return output


def _nearest_native_frames(timestamp_ns: np.ndarray, frame_map: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reference = pd.to_numeric(frame_map["native_capture_utc_ns"], errors="coerce").astype("int64").to_numpy()
    query = np.asarray(timestamp_ns, dtype=np.int64)
    pos = np.searchsorted(reference, query, side="left")
    right = np.clip(pos, 0, len(reference) - 1)
    left = np.clip(pos - 1, 0, len(reference) - 1)
    left_error = np.abs(query - reference[left])
    right_error = np.abs(reference[right] - query)
    choose_right = right_error < left_error
    index = np.where(choose_right, right, left)
    error = np.where(choose_right, right_error, left_error)
    native = pd.to_numeric(frame_map.iloc[index]["native_frame_index"], errors="coerce").fillna(-1).astype(int).to_numpy()
    return native, error.astype(np.int64)


@dataclass
class ReviewDataset:
    mode: ReviewMode
    artifact: ArtifactRecord
    video_artifact: ArtifactRecord
    samples: pd.DataFrame
    frames: pd.DataFrame
    face_regions: pd.DataFrame
    body_regions: pd.DataFrame
    participant_projections: pd.DataFrame
    face_identities: pd.DataFrame
    video_map: pd.DataFrame
    video_info: dict[str, Any]
    coordinate_space: dict[str, Any]
    video_path: Path
    samples_by_native: dict[int, np.ndarray]
    face_regions_by_native: dict[int, list[dict[str, Any]]]
    body_regions_by_scene: dict[int, list[dict[str, Any]]]
    participant_projections_by_native: dict[int, list[dict[str, Any]]]
    face_identities_by_native: dict[int, list[dict[str, Any]]]
    facemapper_audit_by_native: dict[int, list[dict[str, Any]]]
    role_context: dict[str, Any]
    events: dict[str, list[int]]


_CACHE_LOCK = threading.RLock()
_CACHE: dict[tuple[str, ...], ReviewDataset] = {}


def _file_signature(artifact: ArtifactRecord) -> tuple[str, ...]:
    rows = []
    for item in artifact.files:
        path = Path(item.path)
        if path.exists():
            rows.append(f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
    return tuple(rows)


def _sample_file(artifact: ArtifactRecord, mode: ReviewMode) -> pd.DataFrame:
    name = {"face": "face_aoi_samples.csv", "body": "body_aoi_samples.csv", "fusion": "gaze_targets_full.csv"}[mode]
    return _read_csv(artifact, name)


def _frame_file(artifact: ArtifactRecord, mode: ReviewMode) -> pd.DataFrame:
    name = {"face": "face_aoi_frames.csv", "body": "body_aoi_frames.csv", "fusion": "gaze_targets_frames.csv"}[mode]
    try:
        return _read_csv(artifact, name)
    except Exception:
        return pd.DataFrame()


def _regions_file(artifact: ArtifactRecord | None, filename: str) -> pd.DataFrame:
    if artifact is None:
        return pd.DataFrame()
    try:
        return _read_csv(artifact, filename)
    except Exception:
        return pd.DataFrame()


def _review_coordinate_space(session, mode: ReviewMode, artifact: ArtifactRecord, video_info: dict[str, Any]) -> dict[str, Any]:
    """Describe the pixel space used by gaze, FaceMapper and projected-head rows.

    AOI tables are authored in Native scene pixels.  Privacy clips normally keep
    that resolution, but older/re-encoded clips can be scaled.  The review player
    must therefore scale coordinates explicitly instead of assuming equal sizes.
    """
    video_width = int(video_info.get("width") or 0)
    video_height = int(video_info.get("height") or 0)
    source_width = video_width
    source_height = video_height
    source = "privacy_video_pixels"
    trim_id = _native_trim_id(session, mode, artifact)
    trim = session.artifacts.get(trim_id) if trim_id else None
    if trim is not None and trim.kind == ArtifactKind.NATIVE_REFERENCE_TRIM:
        try:
            manifest_path = find_artifact_file(trim, names=["trim_manifest.json"], required=False)
            if manifest_path is not None:
                manifest = _load_json(manifest_path)
                native = manifest.get("native_reference") or {}
                candidate_width = int(native.get("width") or 0)
                candidate_height = int(native.get("height") or 0)
                if candidate_width > 0 and candidate_height > 0:
                    source_width = candidate_width
                    source_height = candidate_height
                    source = "native_scene_pixels"
        except Exception:
            pass
    scale_x = (video_width / source_width) if video_width > 0 and source_width > 0 else 1.0
    scale_y = (video_height / source_height) if video_height > 0 and source_height > 0 else 1.0
    source_aspect = source_width / source_height if source_width > 0 and source_height > 0 else None
    video_aspect = video_width / video_height if video_width > 0 and video_height > 0 else None
    aspect_error = abs(source_aspect - video_aspect) / source_aspect if source_aspect and video_aspect else 0.0
    return {
        "name": source,
        "width": source_width or None,
        "height": source_height or None,
        "video_width": video_width or None,
        "video_height": video_height or None,
        "scale_x_to_video": float(scale_x),
        "scale_y_to_video": float(scale_y),
        "requires_scaling": bool(abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6),
        "aspect_ratio_error": float(aspect_error),
        "aspect_ratio_compatible": bool(aspect_error <= 0.01),
    }


def _build_dataset(store: ProjectStore, session_id: str, mode: ReviewMode, artifact_id: str,
                   video_artifact_id: str | None, confidence_threshold: float) -> ReviewDataset:
    session = store.project.sessions[session_id]
    artifact = _artifact_for_mode(session, mode, artifact_id)
    video_artifact = _video_artifact(session, mode, artifact, video_artifact_id)
    participant_artifact = _participant_artifact_for_review(session, mode, artifact)
    facemapper_filter_artifact = _facemapper_filter_artifact_for_review(session, mode, artifact)
    cache_key = (session_id, mode, artifact.id, video_artifact.id, f"{confidence_threshold:.6f}",
                 *_file_signature(artifact), *_file_signature(video_artifact),
                 *(_file_signature(participant_artifact) if participant_artifact else ()),
                 *(_file_signature(facemapper_filter_artifact) if facemapper_filter_artifact else ()))
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    samples = _sample_file(artifact, mode)
    frames = _frame_file(artifact, mode)
    face_artifact = _face_artifact_for_review(session, mode, artifact)
    body_artifact = _body_artifact_for_review(session, mode, artifact)
    face_regions = _regions_file(face_artifact, "face_regions.csv")
    body_regions = _regions_file(body_artifact, "body_regions.csv")
    participant_projections = _regions_file(participant_artifact, "participant_head_projection.csv")
    face_identities = _regions_file(participant_artifact, "face_identity_assignment.csv")
    facemapper_audit = _regions_file(facemapper_filter_artifact, "face_filter_audit.csv")
    role_context = _manual_review_role_context(
        artifact=artifact,
        participant_artifact=participant_artifact,
        participant_projections=participant_projections,
        face_identities=face_identities,
        samples=samples,
    )
    if mode == "body":
        samples = _attach_body_review_fields(session, artifact, samples)
    # v0.21.16: Body review uses explicit self/partner social labels while
    # preserving compatibility with older Body AOI artifacts.
    if mode in {"body", "fusion"}:
        if "body_raw_region" not in samples.columns and "body_region" in samples.columns:
            samples["body_raw_region"] = samples["body_region"]
        if "body_coarse_region" not in samples.columns and "body_region" in samples.columns:
            samples["body_coarse_region"] = samples["body_region"]
        if "body_social_region" not in samples.columns:
            relation = samples.get("body_relation", pd.Series("", index=samples.index)).fillna("").astype(str)
            region = samples.get("body_region", pd.Series("unknown", index=samples.index)).fillna("unknown").astype(str)
            samples["body_social_region"] = [
                f"{rel}_{reg}" if rel in {"self", "partner"} and reg not in {"unknown", "no_body", "invalid_gaze", ""} else reg
                for rel, reg in zip(relation, region)
            ]

    if mode == "body":
        role_context = _body_manual_review_role_context(artifact=artifact, samples=samples)

    if mode in {"face", "fusion"} and "face_social_region" not in samples.columns:
        samples["face_social_region"] = [
            _derive_face_social_region(row) for _, row in samples.iterrows()
        ]

    video_path, video_map, video_info = _video_info(video_artifact)
    coordinate_space = _review_coordinate_space(session, mode, artifact, video_info)
    if "native_frame_index" not in samples.columns:
        if "timestamp_ns" not in samples.columns:
            raise ValidationError("AOI sample file has neither native_frame_index nor timestamp_ns")
        numeric = pd.to_numeric(samples["timestamp_ns"], errors="coerce")
        valid = numeric.notna()
        native = np.full(len(samples), -1, dtype=int)
        errors = np.full(len(samples), np.iinfo(np.int64).max, dtype=np.int64)
        if valid.any():
            mapped, error = _nearest_native_frames(numeric.loc[valid].astype("int64").to_numpy(), video_map)
            native[np.flatnonzero(valid.to_numpy())] = mapped
            errors[np.flatnonzero(valid.to_numpy())] = error
        samples["native_frame_index"] = native
        samples["review_video_frame_match_error_ms"] = errors.astype(float) / 1e6
    else:
        samples["native_frame_index"] = pd.to_numeric(samples["native_frame_index"], errors="coerce").fillna(-1).astype(int)

    by_native: dict[int, list[int]] = defaultdict(list)
    for index, native in enumerate(samples["native_frame_index"].to_numpy(dtype=int)):
        if native >= 0:
            by_native[int(native)].append(index)
    samples_by_native = {key: np.asarray(value, dtype=int) for key, value in by_native.items()}

    face_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not face_regions.empty and "scene_frame_index" in face_regions.columns:
        for row in face_regions.to_dict("records"):
            try:
                face_by[int(row["scene_frame_index"])].append(_row_dict(row))
            except Exception:
                continue
    body_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not body_regions.empty and "scene_frame_index" in body_regions.columns:
        for row in body_regions.to_dict("records"):
            try:
                body_by[int(row["scene_frame_index"])].append(_row_dict(row))
            except Exception:
                continue
    projection_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not participant_projections.empty and "native_frame_index" in participant_projections.columns:
        for row in participant_projections.to_dict("records"):
            try:
                projection_by[int(row["native_frame_index"])].append(_row_dict(row))
            except Exception:
                continue
    identity_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not face_identities.empty and "native_frame_index" in face_identities.columns:
        for row in face_identities.to_dict("records"):
            try:
                identity_by[int(row["native_frame_index"])].append(_row_dict(row))
            except Exception:
                continue
    facemapper_audit_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not facemapper_audit.empty and "native_frame_index" in facemapper_audit.columns:
        for row in facemapper_audit.to_dict("records"):
            try:
                frame_value = row.get("native_frame_index")
                if frame_value is None or pd.isna(frame_value):
                    continue
                facemapper_audit_by[int(frame_value)].append(_row_dict(row))
            except Exception:
                continue

    region_col = _region_column(mode)
    confidence_col = _confidence_column(mode)
    status_col = _status_column(mode)
    first_output_for_native: dict[int, int] = {}
    for row in video_map.itertuples(index=False):
        first_output_for_native.setdefault(int(row.native_frame_index), int(row.output_frame_index))
    event_sets: dict[str, set[int]] = {key: set() for key in ["review", "conflict", "low_confidence", "unknown", "invalid"]}
    for native, indices in samples_by_native.items():
        output_frame = first_output_for_native.get(native)
        if output_frame is None:
            continue
        group = samples.iloc[indices]
        review = group.get("review_required", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        if review.any():
            event_sets["review"].add(output_frame)
        statuses = group.get(status_col, pd.Series("", index=group.index)).fillna("").astype(str).str.lower()
        if statuses.str.contains("conflict", regex=False).any():
            event_sets["conflict"].add(output_frame)
        confidence = pd.to_numeric(group.get(confidence_col, pd.Series(np.nan, index=group.index)), errors="coerce")
        regions = group.get(region_col, pd.Series("unknown", index=group.index)).fillna("unknown").astype(str)
        valid_region = ~regions.isin({"unknown", "invalid_gaze", "out_of_frame", "no_face", "no_face_detection", "no_body", ""})
        if ((confidence < confidence_threshold) & valid_region).fillna(False).any():
            event_sets["low_confidence"].add(output_frame)
        if regions.isin({"unknown", "no_face", "no_face_detection", "no_body"}).all():
            event_sets["unknown"].add(output_frame)
        if regions.isin({"invalid_gaze", "out_of_frame"}).any():
            event_sets["invalid"].add(output_frame)
    events = {key: sorted(value) for key, value in event_sets.items()}

    dataset = ReviewDataset(
        mode=mode, artifact=artifact, video_artifact=video_artifact,
        samples=samples, frames=frames, face_regions=face_regions, body_regions=body_regions,
        participant_projections=participant_projections, face_identities=face_identities,
        video_map=video_map, video_info=video_info, coordinate_space=coordinate_space, video_path=video_path,
        samples_by_native=samples_by_native, face_regions_by_native=dict(face_by),
        body_regions_by_scene=dict(body_by),
        participant_projections_by_native=dict(projection_by),
        face_identities_by_native=dict(identity_by),
        facemapper_audit_by_native=dict(facemapper_audit_by),
        role_context=role_context, events=events,
    )
    with _CACHE_LOCK:
        _CACHE.clear() if len(_CACHE) >= 8 else None
        _CACHE[cache_key] = dataset
    return dataset



def _manual_review_path(store: ProjectStore, session_id: str, artifact_id: str) -> Path:
    root = store.annotations_dir(session_id, "face_aoi", artifact_id)
    root.mkdir(parents=True, exist_ok=True)
    return root / "manual_review.json"


def _empty_manual_review_state(session_id: str, artifact_id: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": MANUAL_REVIEW_SCHEMA,
        "session_id": session_id,
        "face_aoi_artifact_id": artifact_id,
        "created_at": now,
        "updated_at": now,
        "decisions": {},
        "finalized_artifact_id": None,
        "finalized_at": None,
    }


def _load_manual_review_state(store: ProjectStore, session_id: str, artifact_id: str) -> dict[str, Any]:
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.FACE_AOI:
        raise ValidationError("Manual review requires a Face AOI artifact")
    path = _manual_review_path(store, session_id, artifact_id)
    if not path.exists():
        return _empty_manual_review_state(session_id, artifact_id)
    state = _load_json(path)
    if state.get("schema_version") != MANUAL_REVIEW_SCHEMA:
        raise ValidationError("The Face AOI manual-review draft uses an unsupported schema")
    if state.get("face_aoi_artifact_id") != artifact_id:
        raise ValidationError("The Face AOI manual-review draft belongs to another artifact")
    if not isinstance(state.get("decisions"), dict):
        state["decisions"] = {}
    return state


def _body_manual_review_path(store: ProjectStore, session_id: str, artifact_id: str) -> Path:
    root = store.annotations_dir(session_id, "body_aoi", artifact_id)
    root.mkdir(parents=True, exist_ok=True)
    return root / "manual_review.json"


def _empty_body_manual_review_state(session_id: str, artifact_id: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": BODY_MANUAL_REVIEW_SCHEMA,
        "session_id": session_id,
        "body_aoi_artifact_id": artifact_id,
        "created_at": now,
        "updated_at": now,
        "decisions": {},
        "finalized_artifact_id": None,
        "finalized_at": None,
    }


def _load_body_manual_review_state(
    store: ProjectStore, session_id: str, artifact_id: str
) -> dict[str, Any]:
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.BODY_AOI:
        raise ValidationError("Manual review requires a Body AOI artifact")
    path = _body_manual_review_path(store, session_id, artifact_id)
    if not path.exists():
        return _empty_body_manual_review_state(session_id, artifact_id)
    state = _load_json(path)
    if state.get("schema_version") != BODY_MANUAL_REVIEW_SCHEMA:
        raise ValidationError("The Body AOI manual-review draft uses an unsupported schema")
    if state.get("body_aoi_artifact_id") != artifact_id:
        raise ValidationError("The Body AOI manual-review draft belongs to another artifact")
    if not isinstance(state.get("decisions"), dict):
        state["decisions"] = {}
    return state


def _sample_uid(row: pd.Series | dict[str, Any]) -> str:
    value = row.get("gaze_sample_uid") if isinstance(row, dict) else row.get("gaze_sample_uid")
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if not text:
        source = row.get("source_gaze_row_index") if isinstance(row, dict) else row.get("source_gaze_row_index")
        native = row.get("native_frame_index") if isinstance(row, dict) else row.get("native_frame_index")
        text = f"native:{native}:source:{source}"
    return text


def _first_output_for_native(data: ReviewDataset) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in data.video_map.itertuples(index=False):
        mapping.setdefault(int(row.native_frame_index), int(row.output_frame_index))
    return mapping



def _body_review_target_key(row: pd.Series | dict[str, Any]) -> tuple[str, str, str] | None:
    def value(name: str) -> str:
        raw = row.get(name) if isinstance(row, dict) else row.get(name)
        return _nonempty_scalar_text(raw).lower()

    relation = value("body_relation")
    entity = _nonempty_scalar_text(row.get("body_entity_id") if isinstance(row, dict) else row.get("body_entity_id"))
    region = value("body_coarse_region") or value("body_region")
    if relation not in {"self", "partner"} or not entity:
        return None
    if region in {"", "no_body", "invalid_gaze", "out_of_frame", "unknown"}:
        return None
    return relation, entity, region


def _body_review_episode_groups(data: ReviewDataset) -> list[dict[str, Any]]:
    """Merge only adjacent, target-stable Body AOI fixation units.

    This changes the number of human decisions, not the scientific samples or
    automatic labels.  A boundary is preserved whenever person, entity, coarse
    body region, self-occlusion state, timing continuity, or fixation-internal
    target agreement is uncertain.
    """
    review_mask = data.samples.get(
        "review_required", pd.Series(False, index=data.samples.index)
    ).fillna(False).astype(bool)
    flagged = data.samples.loc[review_mask].copy()
    if flagged.empty:
        return []
    native_to_output = _first_output_for_native(data)
    base: dict[str, dict[str, Any]] = {}
    for index, row in flagged.iterrows():
        uid = _sample_uid(row)
        fixation = _nonempty_scalar_text(row.get("fixation_id"))
        key = f"fixation:{fixation}" if fixation else f"sample:{uid or index}"
        unit = base.setdefault(key, {
            "base_unit_id": key,
            "fixation_ids": set(),
            "sample_uids": [],
            "output_frames": [],
            "timestamps_ns": [],
            "targets": [],
            "self_occlusion": False,
        })
        if fixation:
            unit["fixation_ids"].add(fixation)
        unit["sample_uids"].append(uid)
        try:
            output_frame = native_to_output.get(int(row.get("native_frame_index")))
        except Exception:
            output_frame = None
        if output_frame is not None:
            unit["output_frames"].append(int(output_frame))
        timestamp = pd.to_numeric(pd.Series([row.get("timestamp_ns")]), errors="coerce").iloc[0]
        if not pd.isna(timestamp):
            unit["timestamps_ns"].append(int(timestamp))
        target = _body_review_target_key(row)
        if target is not None:
            unit["targets"].append(target)
        unit["self_occlusion"] = bool(unit["self_occlusion"] or bool(row.get("self_occlusion")))

    base_rows: list[dict[str, Any]] = []
    for unit in base.values():
        frames = sorted(set(unit["output_frames"]))
        times = sorted(unit["timestamps_ns"])
        target_counts = Counter(unit["targets"])
        target = target_counts.most_common(1)[0][0] if target_counts else None
        target_purity = float(target_counts.get(target, 0) / max(1, sum(target_counts.values()))) if target else 0.0
        base_rows.append({
            **unit,
            "output_frames": frames,
            "first_output_frame": frames[0] if frames else None,
            "last_output_frame": frames[-1] if frames else None,
            "start_timestamp_ns": times[0] if times else None,
            "end_timestamp_ns": times[-1] if times else None,
            "target": target,
            "target_purity": target_purity,
        })
    base_rows.sort(key=lambda row: (
        row["start_timestamp_ns"] is None,
        row["start_timestamp_ns"] if row["start_timestamp_ns"] is not None else 10**30,
        row["first_output_frame"] if row["first_output_frame"] is not None else 10**18,
        row["base_unit_id"],
    ))

    episodes: list[dict[str, Any]] = []
    maximum_gap_ns = int(220e6)
    maximum_span_ns = int(3000e6)
    for unit in base_rows:
        can_merge = False
        if episodes:
            current = episodes[-1]
            current_end = current.get("end_timestamp_ns")
            next_start = unit.get("start_timestamp_ns")
            gap_ns = None if current_end is None or next_start is None else int(next_start) - int(current_end)
            proposed_start = current.get("start_timestamp_ns")
            proposed_end = unit.get("end_timestamp_ns")
            span_ns = None if proposed_start is None or proposed_end is None else int(proposed_end) - int(proposed_start)
            frame_gap = None
            if current.get("last_output_frame") is not None and unit.get("first_output_frame") is not None:
                frame_gap = int(unit["first_output_frame"]) - int(current["last_output_frame"])
            can_merge = bool(
                current.get("target") is not None
                and unit.get("target") == current.get("target")
                and float(current.get("target_purity", 0.0)) == 1.0
                and float(unit.get("target_purity", 0.0)) == 1.0
                and not current.get("self_occlusion")
                and not unit.get("self_occlusion")
                and gap_ns is not None and 0 <= gap_ns <= maximum_gap_ns
                and span_ns is not None and span_ns <= maximum_span_ns
                and (frame_gap is None or 0 <= frame_gap <= 8)
            )
        if not can_merge:
            episodes.append({
                "review_unit_id": f"episode:{len(episodes)+1:06d}",
                "fixation_ids": set(unit["fixation_ids"]),
                "sample_uids": list(unit["sample_uids"]),
                "output_frames": list(unit["output_frames"]),
                "start_timestamp_ns": unit.get("start_timestamp_ns"),
                "end_timestamp_ns": unit.get("end_timestamp_ns"),
                "first_output_frame": unit.get("first_output_frame"),
                "last_output_frame": unit.get("last_output_frame"),
                "target": unit.get("target"),
                "target_purity": float(unit.get("target_purity", 0.0)),
                "self_occlusion": bool(unit.get("self_occlusion")),
                "base_unit_count": 1,
            })
            continue
        current = episodes[-1]
        current["fixation_ids"].update(unit["fixation_ids"])
        current["sample_uids"].extend(unit["sample_uids"])
        current["output_frames"] = sorted(set(current["output_frames"]) | set(unit["output_frames"]))
        current["end_timestamp_ns"] = unit.get("end_timestamp_ns")
        current["last_output_frame"] = unit.get("last_output_frame")
        current["base_unit_count"] += 1

    for episode in episodes:
        episode["sample_uids"] = list(dict.fromkeys(episode["sample_uids"]))
        episode["output_frames"] = sorted(set(episode["output_frames"]))
    return episodes


def _manual_review_summary(data: ReviewDataset, state: dict[str, Any]) -> dict[str, Any]:
    decisions = state.get("decisions") or {}
    review_mask = data.samples.get("review_required", pd.Series(False, index=data.samples.index)).fillna(False).astype(bool)
    flagged = data.samples.loc[review_mask]
    frame_uids: dict[int, list[str]] = defaultdict(list)
    native_to_output = _first_output_for_native(data)
    for _, row in flagged.iterrows():
        try:
            output_frame = native_to_output.get(int(row.get("native_frame_index")))
        except Exception:
            output_frame = None
        if output_frame is not None:
            frame_uids[int(output_frame)].append(_sample_uid(row))
    resolved_uids = {
        str(uid) for uid, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("action") in MANUAL_REVIEW_RESOLVED_ACTIONS
    }
    deferred_uids = {
        str(uid) for uid, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("action") == "defer"
    }
    pending_frames: list[int] = []
    reviewed_frames: list[int] = []
    for frame_index, uids in sorted(frame_uids.items()):
        if uids and all(uid in resolved_uids for uid in uids):
            reviewed_frames.append(frame_index)
        else:
            pending_frames.append(frame_index)
    flagged_uids = {_sample_uid(row) for _, row in flagged.iterrows()}
    resolved_flagged = flagged_uids & resolved_uids
    pending_uids = flagged_uids - resolved_uids
    review_units: dict[str, dict[str, Any]] = {}
    target_column = _region_column(data.mode)
    group_fixations = data.mode == "body"
    if group_fixations:
        for episode in _body_review_episode_groups(data):
            key = str(episode["review_unit_id"])
            target = episode.get("target")
            review_units[key] = {
                "review_unit_id": key,
                "fixation_id": next(iter(sorted(episode["fixation_ids"])), None),
                "fixation_ids": sorted(episode["fixation_ids"]),
                "sample_uids": list(episode["sample_uids"]),
                "output_frames": list(episode["output_frames"]),
                "targets": ["::".join(target)] * len(episode["sample_uids"]) if target else [],
                "base_unit_count": int(episode.get("base_unit_count", 1)),
                "start_timestamp_ns": episode.get("start_timestamp_ns"),
                "end_timestamp_ns": episode.get("end_timestamp_ns"),
            }
    else:
        for _, row in flagged.iterrows():
            uid = _sample_uid(row)
            fixation = _nonempty_scalar_text(row.get("fixation_id"))
            key = f"sample:{uid}"
            unit = review_units.setdefault(key, {
                "review_unit_id": key,
                "fixation_id": fixation,
                "fixation_ids": [fixation] if fixation else [],
                "sample_uids": [],
                "output_frames": [],
                "targets": [],
                "base_unit_count": 1,
                "start_timestamp_ns": None,
                "end_timestamp_ns": None,
            })
            unit["sample_uids"].append(uid)
            try:
                output_frame = native_to_output.get(int(row.get("native_frame_index")))
            except Exception:
                output_frame = None
            if output_frame is not None:
                unit["output_frames"].append(int(output_frame))
            target = _nonempty_scalar_text(row.get(target_column))
            if target:
                unit["targets"].append(target)

    review_unit_rows: list[dict[str, Any]] = []
    for key, unit in review_units.items():
        uids = list(dict.fromkeys(unit["sample_uids"]))
        frames = sorted(set(unit["output_frames"]))
        resolved = bool(uids) and all(uid in resolved_uids for uid in uids)
        pending_count = int(sum(uid not in resolved_uids for uid in uids))
        deferred_count = int(sum(uid in deferred_uids for uid in uids))
        target_counts = Counter(unit["targets"])
        dominant_target = target_counts.most_common(1)[0][0] if target_counts else None
        review_unit_rows.append({
            "review_unit_id": key,
            "fixation_id": unit["fixation_id"],
            "sample_count": int(len(uids)),
            "pending_sample_count": pending_count,
            "deferred_sample_count": deferred_count,
            "resolved": resolved,
            "first_output_frame": frames[0] if frames else None,
            "last_output_frame": frames[-1] if frames else None,
            "representative_output_frame": frames[0] if frames else None,
            "affected_frame_count": int(len(frames)),
            "dominant_target": dominant_target,
            "fixation_ids": list(unit.get("fixation_ids") or []),
            "base_unit_count": int(unit.get("base_unit_count") or 1),
            "start_timestamp_ns": unit.get("start_timestamp_ns"),
            "end_timestamp_ns": unit.get("end_timestamp_ns"),
        })
    review_unit_rows.sort(key=lambda row: (
        row.get("representative_output_frame") is None,
        row.get("representative_output_frame") if row.get("representative_output_frame") is not None else 10**18,
        str(row.get("review_unit_id") or ""),
    ))
    resolved_review_units = [row for row in review_unit_rows if row["resolved"]]
    pending_review_units = [row for row in review_unit_rows if not row["resolved"]]
    pending_representatives = [
        int(row["representative_output_frame"]) for row in pending_review_units
        if row.get("representative_output_frame") is not None
    ]
    reviewed_representatives = [
        int(row["representative_output_frame"]) for row in resolved_review_units
        if row.get("representative_output_frame") is not None
    ]
    episode_merge_count = int(sum(max(0, int(row.get("base_unit_count") or 1) - 1) for row in review_unit_rows))
    if not flagged_uids:
        status = "not_required"
    elif not decisions:
        status = "not_started"
    elif not pending_uids:
        status = "complete"
    else:
        status = "in_progress"
    return {
        "schema_version": state.get("schema_version") or MANUAL_REVIEW_SCHEMA,
        "status": status,
        "total_flagged_sample_count": int(len(flagged_uids)),
        "resolved_sample_count": int(len(resolved_flagged)),
        "pending_sample_count": int(len(pending_uids)),
        "deferred_sample_count": int(len(flagged_uids & deferred_uids)),
        "decision_count": int(len(decisions)),
        "total_flagged_frame_count": int(len(frame_uids)),
        "reviewed_frame_count": int(len(reviewed_frames)),
        "pending_frame_count": int(len(pending_frames)),
        "total_review_unit_count": int(len(review_unit_rows)),
        "resolved_review_unit_count": int(len(resolved_review_units)),
        "pending_review_unit_count": int(len(pending_review_units)),
        "review_unit_policy": (
            "target_stable_episode_when_available_else_fixation"
            if group_fixations and episode_merge_count > 0
            else "fixation_when_available_else_sample" if group_fixations else "sample"
        ),
        "review_episode_merge_count": episode_merge_count,
        "review_units": review_unit_rows,
        "pending_review_units": pending_review_units,
        "pending_review_representative_frames": pending_representatives,
        "reviewed_review_representative_frames": reviewed_representatives,
        "pending_frames": pending_frames,
        "reviewed_frames": reviewed_frames,
        "updated_at": state.get("updated_at"),
        "finalized_artifact_id": state.get("finalized_artifact_id"),
        "finalized_at": state.get("finalized_at"),
    }


def _manual_review_snapshot(row: pd.Series) -> dict[str, Any]:
    fields = [
        "gaze_sample_uid", "source_gaze_row_index", "native_frame_index", "timestamp_ns",
        "face_entity_id", "candidate_entity_id", "candidate_role", "candidate_relation",
        "face_region", "face_social_region", "target_relation", "identity_status",
        "assignment_status", "face_aoi_confidence", "review_required",
    ]
    return _row_dict(row, fields)


def _nonempty_scalar_text(value: Any) -> str | None:
    """Return a stripped scalar string, treating pandas missing values as empty."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None



def _manual_review_role_context(
    *,
    artifact: ArtifactRecord,
    participant_artifact: ArtifactRecord | None,
    participant_projections: pd.DataFrame,
    face_identities: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    """Resolve stable self/partner entity IDs for manual Face AOI review.

    Human review operates on gaze samples, and a flagged sample can legitimately
    have no automatic ``face_entity_id`` or ``candidate_entity_id`` even though
    the participant-association artifact has a unique projected partner.  Keep a
    deterministic role context so selecting relation=partner can resolve to R02
    (or the corresponding entity ID) without asking the reviewer to retype it.
    """
    entities: dict[str, set[str]] = {"self": set(), "partner": set()}
    authoritative_entities: dict[str, set[str]] = {"self": set(), "partner": set()}
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(relation: Any, entity: Any, source: str, *, authoritative: bool = False) -> None:
        relation_text = _nonempty_scalar_text(relation)
        entity_text = _nonempty_scalar_text(entity)
        if relation_text not in entities or not entity_text:
            return
        entities[relation_text].add(entity_text)
        if authoritative:
            authoritative_entities[relation_text].add(entity_text)
        sources[(relation_text, entity_text)].add(source)

    wearer_role_id = None
    if participant_artifact is not None:
        wearer_role_id = (
            _nonempty_scalar_text((participant_artifact.parameters or {}).get("wearer_role_id"))
            or _nonempty_scalar_text((participant_artifact.provenance or {}).get("wearer_role_id"))
        )
        if not wearer_role_id:
            try:
                manifest_path = find_artifact_file(
                    participant_artifact,
                    names=["participant_association_manifest.json"],
                    required=False,
                )
                if manifest_path is not None:
                    wearer_role_id = _nonempty_scalar_text(_load_json(manifest_path).get("wearer_role_id"))
            except Exception:
                wearer_role_id = None

    if not wearer_role_id:
        label_text = " ".join([
            str(artifact.label or ""),
            json.dumps(artifact.provenance or {}, ensure_ascii=True, sort_keys=True),
            json.dumps(artifact.parameters or {}, ensure_ascii=True, sort_keys=True),
        ])
        match = re.search(r"(?:^|[^A-Za-z0-9])(R\d{1,6})(?![A-Za-z0-9])", label_text, re.IGNORECASE)
        wearer_role_id = match.group(1).upper() if match else None

    if not participant_projections.empty:
        for row in participant_projections.to_dict("records"):
            relation = row.get("relation_to_wearer")
            entity = _nonempty_scalar_text(row.get("entity_id")) or _nonempty_scalar_text(row.get("role_id"))
            add(relation, entity, "participant_projection", authoritative=True)

    if not face_identities.empty:
        for row in face_identities.to_dict("records"):
            relation = _nonempty_scalar_text(row.get("target_relation"))
            candidate_relation = _nonempty_scalar_text(row.get("candidate_relation"))
            if relation in {"ambiguous_partner", "partner_candidate"}:
                relation = "partner"
            if relation not in entities and candidate_relation in entities:
                relation = candidate_relation
            entity = (
                _nonempty_scalar_text(row.get("matched_entity_id"))
                or _nonempty_scalar_text(row.get("candidate_entity_id"))
                or _nonempty_scalar_text(row.get("matched_role"))
                or _nonempty_scalar_text(row.get("candidate_role"))
            )
            add(relation, entity, "face_identity")

    if not samples.empty:
        for row in samples.to_dict("records"):
            relation = _nonempty_scalar_text(row.get("target_relation"))
            candidate_relation = _nonempty_scalar_text(row.get("candidate_relation"))
            if relation in {"ambiguous_partner", "partner_candidate"}:
                relation = "partner"
            if relation not in entities and candidate_relation in entities:
                relation = candidate_relation
            entity = (
                _nonempty_scalar_text(row.get("face_entity_id"))
                or _nonempty_scalar_text(row.get("candidate_entity_id"))
                or _nonempty_scalar_text(row.get("candidate_role"))
            )
            add(relation, entity, "face_aoi_samples")

    if wearer_role_id:
        add("self", wearer_role_id, "wearer_role", authoritative=True)

    result: dict[str, Any] = {
        "wearer_role_id": wearer_role_id,
        "self_entity_ids": sorted(entities["self"]),
        "partner_entity_ids": sorted(entities["partner"]),
    }
    for relation in ("self", "partner"):
        all_values = sorted(entities[relation])
        authoritative_values = sorted(authoritative_entities[relation])
        values = authoritative_values or all_values
        unique = values[0] if len(values) == 1 else None
        result[f"{relation}_authoritative_entity_ids"] = authoritative_values
        result[f"{relation}_entity_id"] = unique
        result[f"{relation}_entity_is_unique"] = bool(unique)
        if unique:
            result[f"{relation}_entity_source"] = "+".join(sorted(sources[(relation, unique)]))
        elif len(values) > 1:
            result[f"{relation}_entity_source"] = "ambiguous_multiple_entities"
        else:
            result[f"{relation}_entity_source"] = "unavailable"
    return result


def _manual_review_frame_role_context(
    base_context: dict[str, Any] | None,
    participant_projections: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Prefer the current frame's unique projection over global role context."""
    context = dict(base_context or {})
    frame_entities: dict[str, set[str]] = {"self": set(), "partner": set()}
    for row in participant_projections or []:
        relation = _nonempty_scalar_text(row.get("relation_to_wearer"))
        entity = _nonempty_scalar_text(row.get("entity_id")) or _nonempty_scalar_text(row.get("role_id"))
        if relation in frame_entities and entity:
            frame_entities[relation].add(entity)
    for relation in ("self", "partner"):
        values = sorted(frame_entities[relation])
        context[f"frame_{relation}_entity_ids"] = values
        if len(values) == 1:
            context[f"{relation}_entity_id"] = values[0]
            context[f"{relation}_entity_is_unique"] = True
            context[f"{relation}_entity_source"] = "current_frame_projection"
        elif len(values) > 1:
            context[f"{relation}_entity_id"] = None
            context[f"{relation}_entity_is_unique"] = False
            context[f"{relation}_entity_source"] = "ambiguous_current_frame_projections"
    return context


def _resolve_manual_review_entity(
    row: pd.Series | dict[str, Any],
    *,
    relation: str,
    requested_entity_id: Any = None,
    role_context: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Resolve the entity written by a manual relation/region decision."""
    explicit = _nonempty_scalar_text(requested_entity_id)
    if explicit:
        return explicit, "explicit_entity_id"

    target_relation = _nonempty_scalar_text(row.get("target_relation"))
    candidate_relation = _nonempty_scalar_text(row.get("candidate_relation"))
    normalized_target = "partner" if target_relation in {"ambiguous_partner", "partner_candidate"} else target_relation

    if relation in {"partner", "self"}:
        if candidate_relation == relation or (relation == "partner" and candidate_relation in {"ambiguous_partner", "partner_candidate"}):
            candidate = _nonempty_scalar_text(row.get("candidate_entity_id"))
            if candidate:
                return candidate, "sample_candidate_entity"
            candidate_role = _nonempty_scalar_text(row.get("candidate_role"))
            if candidate_role:
                return candidate_role, "sample_candidate_role"
        if normalized_target == relation:
            matched = _nonempty_scalar_text(row.get("face_entity_id"))
            if matched:
                return matched, "sample_face_entity"
            candidate = _nonempty_scalar_text(row.get("candidate_entity_id"))
            if candidate:
                return candidate, "sample_candidate_entity"
            candidate_role = _nonempty_scalar_text(row.get("candidate_role"))
            if candidate_role:
                return candidate_role, "sample_candidate_role"
        context_entity = _nonempty_scalar_text((role_context or {}).get(f"{relation}_entity_id"))
        if context_entity:
            return context_entity, str((role_context or {}).get(f"{relation}_entity_source") or "participant_role_context")
        return None, "unresolved_relation_entity"

    if relation == "other_person":
        if normalized_target == "other_person":
            entity = _nonempty_scalar_text(row.get("face_entity_id"))
            if entity:
                return entity, "sample_face_entity"
        if candidate_relation == "other_person":
            entity = _nonempty_scalar_text(row.get("candidate_entity_id"))
            if entity:
                return entity, "sample_candidate_entity"
        return None, "other_person_unspecified"

    return None, "relation_without_entity"



def _coarse_body_review_region(region: Any) -> str:
    value = (_nonempty_scalar_text(region) or "unknown").lower()
    mapping = {
        "neck": "head",
        "pelvis": "torso",
        "left_upper_arm": "left_arm",
        "left_forearm": "left_arm",
        "right_upper_arm": "right_arm",
        "right_forearm": "right_arm",
        "left_thigh": "left_leg",
        "left_lower_leg": "left_leg",
        "left_foot": "left_leg",
        "right_thigh": "right_leg",
        "right_lower_leg": "right_leg",
        "right_foot": "right_leg",
    }
    return mapping.get(value, value)


def _body_manual_review_role_context(
    *, artifact: ArtifactRecord, samples: pd.DataFrame
) -> dict[str, Any]:
    """Resolve stable Body C3D entities for self/partner review decisions."""
    parameters = artifact.parameters or {}
    provenance = artifact.provenance or {}
    entities: dict[str, set[str]] = {"self": set(), "partner": set(), "other": set()}
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(relation: Any, entity: Any, source: str) -> None:
        rel = _nonempty_scalar_text(relation)
        ent = _nonempty_scalar_text(entity)
        if rel in entities and ent:
            entities[rel].add(ent)
            sources[(rel, ent)].add(source)

    wearer_role_id = (
        _nonempty_scalar_text(parameters.get("wearer_role_id"))
        or _nonempty_scalar_text(provenance.get("wearer_role_id"))
    )
    partner_role_id = (
        _nonempty_scalar_text(parameters.get("partner_role_id"))
        or _nonempty_scalar_text(provenance.get("partner_role_id"))
    )
    wearer_body_asset_id = (
        _nonempty_scalar_text(parameters.get("wearer_body_asset_id"))
        or _nonempty_scalar_text((artifact.qc or {}).get("wearer_body_asset_id"))
    )
    partner_body_asset_id = (
        _nonempty_scalar_text(parameters.get("partner_body_asset_id"))
        or _nonempty_scalar_text((artifact.qc or {}).get("partner_body_asset_id"))
    )
    add("self", wearer_body_asset_id, "artifact_wearer_body")
    add("partner", partner_body_asset_id, "artifact_partner_body")

    if not samples.empty:
        for row in samples.to_dict("records"):
            add(row.get("body_relation"), row.get("body_entity_id"), "body_aoi_samples")
            add(row.get("first_hit_relation"), row.get("first_hit_entity_id"), "first_hit")
            add(row.get("second_hit_relation"), row.get("second_hit_entity_id"), "second_hit")
            add("partner", row.get("nearest_partner_hit_entity_id"), "nearest_partner_hit")

    result: dict[str, Any] = {
        "wearer_role_id": wearer_role_id,
        "partner_role_id": partner_role_id,
        "wearer_body_asset_id": wearer_body_asset_id,
        "partner_body_asset_id": partner_body_asset_id,
    }
    for relation in ("self", "partner", "other"):
        values = sorted(entities[relation])
        unique = values[0] if len(values) == 1 else None
        result[f"{relation}_entity_ids"] = values
        result[f"{relation}_entity_id"] = unique
        result[f"{relation}_entity_is_unique"] = bool(unique)
        if unique:
            result[f"{relation}_entity_source"] = "+".join(sorted(sources[(relation, unique)]))
        elif len(values) > 1:
            result[f"{relation}_entity_source"] = "ambiguous_multiple_body_entities"
        else:
            result[f"{relation}_entity_source"] = "unavailable"
    return result


def _resolve_body_review_entity(
    row: pd.Series | dict[str, Any],
    *,
    relation: str,
    requested_entity_id: Any = None,
    role_context: dict[str, Any] | None = None,
) -> tuple[str | None, str]:
    explicit = _nonempty_scalar_text(requested_entity_id)
    if explicit:
        return explicit, "explicit_entity_id"
    current_relation = _nonempty_scalar_text(row.get("body_relation"))
    current_entity = _nonempty_scalar_text(row.get("body_entity_id"))
    if current_relation == relation and current_entity:
        return current_entity, "automatic_body_entity"
    if relation == "partner":
        candidate = _nonempty_scalar_text(row.get("nearest_partner_hit_entity_id"))
        if candidate:
            return candidate, "nearest_partner_hit_entity"
    context_entity = _nonempty_scalar_text((role_context or {}).get(f"{relation}_entity_id"))
    if context_entity:
        return context_entity, str((role_context or {}).get(f"{relation}_entity_source") or "body_role_context")
    return None, "unresolved_body_relation_entity"


def _body_manual_review_snapshot(row: pd.Series) -> dict[str, Any]:
    fields = [
        "gaze_sample_uid", "source_gaze_row_index", "native_frame_index", "scene_frame_index", "timestamp_ns",
        "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region", "body_region",
        "body_social_region", "body_confidence", "body_exact_hit", "body_distance_mm",
        "assignment_status", "occlusion_status", "self_occlusion", "review_required",
    ]
    return _row_dict(row, fields)


def _effective_body_review_row(
    row: pd.Series | dict[str, Any],
    decision: dict[str, Any] | None,
    *,
    role_context: dict[str, Any] | None = None,
    output_granularity: str = "coarse_social_regions",
) -> tuple[pd.Series, dict[str, Any]]:
    """Project a saved Body AOI review decision onto one row for live preview."""
    effective = row.copy() if isinstance(row, pd.Series) else pd.Series(dict(row))
    action = str((decision or {}).get("action") or "")
    resolved = action in MANUAL_REVIEW_RESOLVED_ACTIONS
    entity_resolution_source = str((decision or {}).get("entity_resolution_source") or "automatic")
    geometry_consistent = True

    if action == "accept_auto":
        region = _nonempty_scalar_text(effective.get("body_region")) or "no_body"
        relation = _nonempty_scalar_text(effective.get("body_relation"))
        if region not in _BODY_NON_HIT_REGIONS and relation in BODY_MANUAL_REVIEW_RELATIONS:
            entity, entity_resolution_source = _resolve_body_review_entity(
                effective, relation=relation, role_context=role_context
            )
            if entity:
                effective["body_entity_id"] = entity
        effective["assignment_status"] = "manual_review_accepted_auto"
        effective["body_confidence"] = 1.0
        effective["review_required"] = False
    elif action == "exclude":
        entity_resolution_source = "excluded_no_body"
        effective["body_entity_id"] = None
        effective["body_relation"] = None
        effective["body_raw_region"] = "no_body"
        effective["body_coarse_region"] = "no_body"
        effective["body_region"] = "no_body"
        effective["body_social_region"] = "no_body"
        effective["body_confidence"] = 1.0
        effective["body_exact_hit"] = False
        effective["body_distance_mm"] = np.nan
        effective["body_hit_x_mm"] = np.nan
        effective["body_hit_y_mm"] = np.nan
        effective["body_hit_z_mm"] = np.nan
        effective["body_angular_miss_deg"] = np.nan
        effective["self_occlusion"] = False
        effective["occlusion_status"] = "manual_excluded"
        effective["assignment_status"] = "manual_review_excluded"
        effective["body_geometry_backend"] = "manual_review_no_body"
        effective["review_required"] = False
        geometry_consistent = False
    elif action == "override":
        requested_region = _nonempty_scalar_text((decision or {}).get("override_region"))
        raw_region = requested_region or _nonempty_scalar_text(effective.get("body_raw_region")) or _nonempty_scalar_text(effective.get("body_region")) or "no_body"
        coarse_region = _coarse_body_review_region(raw_region)
        requested_relation = _nonempty_scalar_text((decision or {}).get("override_relation"))
        relation = requested_relation or _nonempty_scalar_text(effective.get("body_relation"))
        if coarse_region == "no_body":
            relation = None
            entity = None
            entity_resolution_source = "no_body_no_entity"
            effective["assignment_status"] = "manual_review_no_body"
        else:
            relation = relation if relation in BODY_MANUAL_REVIEW_RELATIONS else "partner"
            entity, entity_resolution_source = _resolve_body_review_entity(
                effective,
                relation=relation,
                requested_entity_id=(decision or {}).get("override_entity_id") or (decision or {}).get("resolved_entity_id"),
                role_context=role_context,
            )
            effective["assignment_status"] = "manual_review_override"
        final_region = raw_region if output_granularity == "detailed_regions" else coarse_region
        automatic_entity = _nonempty_scalar_text(effective.get("body_entity_id"))
        automatic_relation = _nonempty_scalar_text(effective.get("body_relation"))
        automatic_raw = _nonempty_scalar_text(effective.get("body_raw_region")) or _nonempty_scalar_text(effective.get("body_region"))
        geometry_consistent = bool(
            entity == automatic_entity and relation == automatic_relation and raw_region == automatic_raw
        )
        effective["body_entity_id"] = entity
        effective["body_relation"] = relation
        effective["body_raw_region"] = raw_region
        effective["body_coarse_region"] = coarse_region
        effective["body_region"] = final_region
        effective["body_social_region"] = (
            "no_body" if coarse_region == "no_body" else f"{relation}_{final_region}"
        )
        effective["body_confidence"] = 1.0
        effective["review_required"] = False
        if not geometry_consistent:
            effective["body_exact_hit"] = False
            effective["body_distance_mm"] = np.nan
            effective["body_hit_x_mm"] = np.nan
            effective["body_hit_y_mm"] = np.nan
            effective["body_hit_z_mm"] = np.nan
            effective["body_angular_miss_deg"] = np.nan
            effective["self_occlusion"] = False
            effective["occlusion_status"] = "manual_semantic_override"
            effective["body_geometry_backend"] = "manual_review_semantic_override"

    preview = {
        "body_entity_id": _jsonable(effective.get("body_entity_id")),
        "body_relation": _jsonable(effective.get("body_relation")),
        "body_raw_region": _jsonable(effective.get("body_raw_region")),
        "body_coarse_region": _jsonable(effective.get("body_coarse_region")),
        "body_region": _jsonable(effective.get("body_region")),
        "body_social_region": _jsonable(effective.get("body_social_region")),
        "body_confidence": _jsonable(effective.get("body_confidence")),
        "body_exact_hit": _jsonable(effective.get("body_exact_hit")),
        "body_distance_mm": _jsonable(effective.get("body_distance_mm")),
        "body_hit_x_mm": _jsonable(effective.get("body_hit_x_mm")),
        "body_hit_y_mm": _jsonable(effective.get("body_hit_y_mm")),
        "body_hit_z_mm": _jsonable(effective.get("body_hit_z_mm")),
        "body_angular_miss_deg": _jsonable(effective.get("body_angular_miss_deg")),
        "body_geometry_backend": _jsonable(effective.get("body_geometry_backend")),
        "self_occlusion": _jsonable(effective.get("self_occlusion")),
        "assignment_status": _jsonable(effective.get("assignment_status")),
        "occlusion_status": _jsonable(effective.get("occlusion_status")),
        "review_required": _jsonable(effective.get("review_required")),
        "manual_review_action": action,
        "manual_review_resolved": bool(resolved),
        "resolved_entity_id": _jsonable(effective.get("body_entity_id")),
        "entity_resolution_source": entity_resolution_source,
        "geometry_consistent": bool(geometry_consistent),
        "source": (
            "review_override" if action == "override"
            else "review_accept_auto" if action == "accept_auto"
            else "review_excluded" if action == "exclude"
            else "review_deferred" if action == "defer"
            else "automatic"
        ),
    }
    return effective, preview

def _effective_face_review_row(
    row: pd.Series | dict[str, Any],
    decision: dict[str, Any] | None,
    *,
    role_context: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Project a saved review decision onto one Face AOI row for live preview.

    The source row is never mutated.  The transformation intentionally mirrors
    ``apply_face_aoi_manual_review`` so the review player shows the same target
    that will be written into the immutable reviewed artifact.
    """
    source = row.copy() if isinstance(row, pd.Series) else pd.Series(dict(row))
    effective = source.copy()
    action = decision.get("action") if isinstance(decision, dict) else None
    resolved = action in MANUAL_REVIEW_RESOLVED_ACTIONS

    entity_resolution_source = "automatic"
    if action == "accept_auto":
        relation = _resolved_manual_relation(effective)
        region = _nonempty_scalar_text(effective.get("face_region")) or "other_face"
        entity, entity_resolution_source = _resolve_manual_review_entity(
            effective, relation=relation, role_context=role_context
        )
        effective["target_relation"] = relation
        effective["face_region"] = region
        effective["face_entity_id"] = entity
        effective["identity_status"] = "manual_review_accepted_auto"
        effective["assignment_status"] = "manual_review_accepted_auto"
        effective["identity_confidence"] = 1.0
        effective["face_aoi_confidence"] = 1.0
        effective["include_in_primary_analysis"] = (
            relation in {"partner", "self"}
            and region not in {"no_face", "no_face_detection", "invalid_gaze", "out_of_frame"}
        )
        effective["review_required"] = False
        effective["face_social_region"] = _derive_face_social_region(effective)
    elif action == "exclude":
        entity_resolution_source = "excluded_no_entity"
        effective["face_entity_id"] = None
        effective["face_region"] = "no_face"
        effective["face_social_region"] = "no_face"
        effective["target_relation"] = "other_person"
        effective["identity_status"] = "manual_review_excluded"
        effective["assignment_status"] = "manual_review_excluded"
        effective["identity_confidence"] = 1.0
        effective["face_aoi_confidence"] = 1.0
        effective["include_in_primary_analysis"] = False
        effective["review_required"] = False
    elif action == "override":
        relation = _resolved_manual_relation(effective, decision.get("override_relation"))
        region = (
            _nonempty_scalar_text(decision.get("override_region"))
            or _nonempty_scalar_text(effective.get("face_region"))
            or "other_face"
        )
        entity, entity_resolution_source = _resolve_manual_review_entity(
            effective,
            relation=relation,
            requested_entity_id=(
                decision.get("override_entity_id")
                or decision.get("resolved_entity_id")
            ),
            role_context=role_context,
        )
        if region == "no_face":
            relation = "other_person"
            entity = None
            entity_resolution_source = "no_face_no_entity"
            status = "manual_review_no_face"
        else:
            status = "manual_review_override"
        effective["target_relation"] = relation
        effective["face_region"] = region
        effective["face_entity_id"] = entity
        effective["identity_status"] = status
        effective["assignment_status"] = status
        effective["identity_confidence"] = 1.0
        effective["face_aoi_confidence"] = 1.0
        effective["include_in_primary_analysis"] = relation in {"partner", "self"} and region != "no_face"
        effective["review_required"] = False
        effective["face_social_region"] = "no_face" if region == "no_face" else _derive_face_social_region(effective)

    preview = {
        "face_entity_id": _jsonable(effective.get("face_entity_id")),
        "face_region": _jsonable(effective.get("face_region")),
        "face_social_region": _jsonable(effective.get("face_social_region")),
        "target_relation": _jsonable(effective.get("target_relation")),
        "identity_status": _jsonable(effective.get("identity_status")),
        "assignment_status": _jsonable(effective.get("assignment_status")),
        "identity_confidence": _jsonable(effective.get("identity_confidence")),
        "face_aoi_confidence": _jsonable(effective.get("face_aoi_confidence")),
        "include_in_primary_analysis": _jsonable(effective.get("include_in_primary_analysis")),
        "review_required": _jsonable(effective.get("review_required")),
        "manual_review_action": action,
        "manual_review_resolved": bool(resolved),
        "resolved_entity_id": _jsonable(effective.get("face_entity_id")),
        "entity_resolution_source": entity_resolution_source,
        "source": (
            "review_override" if action == "override"
            else "review_accept_auto" if action == "accept_auto"
            else "review_excluded" if action == "exclude"
            else "review_deferred" if action == "defer"
            else "automatic"
        ),
    }
    return effective, preview


def save_face_review_decision(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_id: str,
    video_artifact_id: str | None,
    output_frame_index: int,
    gaze_sample_uid: str | None = None,
    apply_to: Literal["sample", "frame", "fixation", "review_episode"] = "sample",
    action: str,
    override_relation: str | None = None,
    override_entity_id: str | None = None,
    override_region: str | None = None,
    note: str = "",
    reviewer: str = "local reviewer",
    confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    if action not in MANUAL_REVIEW_ACTIONS:
        raise ValidationError(f"Unsupported manual-review action: {action}")
    if apply_to not in {"sample", "frame", "fixation", "review_episode"}:
        raise ValidationError("apply_to must be sample, frame, fixation or review_episode")
    relation = (override_relation or "").strip() or None
    region = (override_region or "").strip() or None
    entity_id = (override_entity_id or "").strip() or None
    if action == "override":
        if relation is not None and relation not in MANUAL_REVIEW_RELATIONS:
            raise ValidationError("Manual Face AOI relation must be partner, self or other_person")
        if region is not None and region not in MANUAL_REVIEW_REGIONS:
            raise ValidationError("Manual Face AOI region is unsupported")
        if relation is None and region is None and entity_id is None:
            raise ValidationError("A correction must change relation, entity or face region")
    data = _build_dataset(store, session_id, "face", artifact_id, video_artifact_id, confidence_threshold)
    resolved_output_frame_index = max(0, min(int(data.video_info["frame_count"]) - 1, int(output_frame_index)))
    group, native_frame, _ = _samples_for_frame(data, resolved_output_frame_index)
    if group.empty:
        raise ValidationError("This video frame has no gaze samples to review")
    uid = (gaze_sample_uid or "").strip()
    current_selected = group.loc[group.apply(_sample_uid, axis=1).eq(uid)] if uid else pd.DataFrame()
    if apply_to == "frame":
        flagged = group.get("review_required", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        selected = group.loc[flagged] if flagged.any() else group
    elif apply_to == "review_episode":
        if current_selected.empty:
            raise ValidationError("Select a gaze sample before applying a decision to its review episode")
        selected_uid = _sample_uid(current_selected.iloc[0])
        episode = next((row for row in _body_review_episode_groups(data) if selected_uid in row.get("sample_uids", [])), None)
        if episode is None:
            raise ValidationError("The selected gaze sample is not part of a pending Body AOI review episode")
        member_uids = set(episode.get("sample_uids") or [])
        selected = data.samples.loc[data.samples.apply(_sample_uid, axis=1).isin(member_uids)].copy()
        if selected.empty:
            raise ValidationError("This Body AOI review episode has no flagged gaze samples")
    elif apply_to == "fixation":
        if current_selected.empty:
            raise ValidationError("Select a gaze sample before applying a decision to its fixation")
        fixation_value = _nonempty_scalar_text(current_selected.iloc[0].get("fixation_id"))
        if not fixation_value:
            raise ValidationError("The selected gaze sample has no fixation ID; use sample or frame scope")
        all_fixations = data.samples.get("fixation_id", pd.Series(None, index=data.samples.index)).map(_nonempty_scalar_text)
        flagged = data.samples.get("review_required", pd.Series(False, index=data.samples.index)).fillna(False).astype(bool)
        selected = data.samples.loc[flagged & all_fixations.eq(fixation_value)].copy()
        if selected.empty:
            raise ValidationError("This fixation has no flagged gaze samples")
    else:
        selected = current_selected
        if selected.empty:
            raise ValidationError("Select a gaze sample before saving the manual review")
    path = _manual_review_path(store, session_id, artifact_id)
    with _MANUAL_REVIEW_LOCK:
        state = _load_manual_review_state(store, session_id, artifact_id)
        decisions = state.setdefault("decisions", {})
        now = utc_now()
        changed: list[str] = []
        resolved_entities: dict[str, str | None] = {}
        native_to_output = _first_output_for_native(data)
        for _, row in selected.iterrows():
            uid = _sample_uid(row)
            row_native_frame = int(pd.to_numeric(row.get("native_frame_index"), errors="coerce"))
            row_output_frame = int(native_to_output.get(row_native_frame, resolved_output_frame_index))
            row_role_context = _manual_review_frame_role_context(
                data.role_context,
                data.participant_projections_by_native.get(row_native_frame, []),
            )
            if action == "clear":
                decisions.pop(uid, None)
                changed.append(uid)
                continue
            resolved_entity_id = entity_id
            entity_resolution_source = "explicit_entity_id" if entity_id else "not_applicable"
            if action == "override":
                resolved_relation = _resolved_manual_relation(row, relation)
                effective_region = region or _nonempty_scalar_text(row.get("face_region")) or "other_face"
                if effective_region == "no_face":
                    resolved_entity_id = None
                    entity_resolution_source = "no_face_no_entity"
                else:
                    resolved_entity_id, entity_resolution_source = _resolve_manual_review_entity(
                        row,
                        relation=resolved_relation,
                        requested_entity_id=entity_id,
                        role_context=row_role_context,
                    )
                    if resolved_relation in {"partner", "self"} and not resolved_entity_id:
                        relation_label = "partner" if resolved_relation == "partner" else "self"
                        raise ValidationError(
                            f"The {relation_label} correction has no uniquely resolved entity. "
                            "Enter Entity ID explicitly before saving."
                        )
            decisions[uid] = {
                "gaze_sample_uid": uid,
                "native_frame_index": int(row_native_frame),
                "output_frame_index": int(row_output_frame),
                "apply_to_scope": apply_to,
                "source_gaze_row_index": _jsonable(row.get("source_gaze_row_index")),
                "action": action,
                "override_relation": relation,
                "override_entity_id": resolved_entity_id,
                "requested_override_entity_id": entity_id,
                "entity_resolution_source": entity_resolution_source,
                "override_region": region,
                "note": str(note or "").strip(),
                "reviewer": str(reviewer or "local reviewer").strip() or "local reviewer",
                "automatic": _manual_review_snapshot(row),
                "updated_at": now,
            }
            resolved_entities[uid] = resolved_entity_id
            changed.append(uid)
        state["updated_at"] = now
        # Any edit after finalisation represents a new review revision.
        if changed:
            state["finalized_artifact_id"] = None
            state["finalized_at"] = None
        atomic_write_json(path, state)
    summary = _manual_review_summary(data, state)
    events = dict(data.events)
    events["pending_review"] = summary["pending_frames"]
    events["pending_review_frames"] = summary["pending_frames"]
    events["reviewed"] = summary["reviewed_frames"]
    return {
        "saved": True,
        "changed_sample_uids": changed,
        "changed_sample_count": int(len(changed)),
        "apply_to_scope": apply_to,
        "resolved_entity_ids": resolved_entities if action != "clear" else {},
        "manual_review": summary,
        "event_frames": events,
    }



def save_body_review_decision(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_id: str,
    video_artifact_id: str | None,
    output_frame_index: int,
    gaze_sample_uid: str | None = None,
    apply_to: Literal["sample", "frame", "fixation"] = "sample",
    action: str,
    override_relation: str | None = None,
    override_entity_id: str | None = None,
    override_region: str | None = None,
    note: str = "",
    reviewer: str = "local reviewer",
    confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    """Persist one or more Body AOI human-review decisions without mutating the source artifact."""
    if action not in MANUAL_REVIEW_ACTIONS:
        raise ValidationError(f"Unsupported manual-review action: {action}")
    if apply_to not in {"sample", "frame", "fixation"}:
        raise ValidationError("apply_to must be sample, frame or fixation")
    session = store.project.sessions[session_id]
    source_artifact = session.artifacts.get(artifact_id)
    if source_artifact is None or source_artifact.kind != ArtifactKind.BODY_AOI:
        raise ValidationError("Manual review requires a Body AOI artifact")
    _require_body_geometry_review_ready(source_artifact)
    relation = (override_relation or "").strip() or None
    region = (override_region or "").strip() or None
    entity_id = (override_entity_id or "").strip() or None
    if action == "override":
        if relation is not None and relation not in BODY_MANUAL_REVIEW_RELATIONS:
            raise ValidationError("Manual Body AOI relation must be partner, self or other")
        if region is not None and region not in BODY_MANUAL_REVIEW_REGIONS:
            raise ValidationError("Manual Body AOI region is unsupported")
        if relation is None and region is None and entity_id is None:
            raise ValidationError("A Body AOI correction must change relation, entity or body region")

    data = _build_dataset(store, session_id, "body", artifact_id, video_artifact_id, confidence_threshold)
    resolved_output_frame_index = max(0, min(int(data.video_info["frame_count"]) - 1, int(output_frame_index)))
    group, _native_frame, _ = _samples_for_frame(data, resolved_output_frame_index)
    if group.empty:
        raise ValidationError("This video frame has no gaze samples to review")
    requested_uid = (gaze_sample_uid or "").strip()
    current_selected = group.loc[group.apply(_sample_uid, axis=1).eq(requested_uid)] if requested_uid else pd.DataFrame()
    if apply_to == "frame":
        flagged = group.get("review_required", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        selected = group.loc[flagged] if flagged.any() else group
    elif apply_to == "fixation":
        if current_selected.empty:
            raise ValidationError("Select a gaze sample before applying a decision to its fixation")
        fixation_value = _nonempty_scalar_text(current_selected.iloc[0].get("fixation_id"))
        if not fixation_value:
            raise ValidationError("The selected gaze sample has no fixation ID; use sample or frame scope")
        all_fixations = data.samples.get("fixation_id", pd.Series(None, index=data.samples.index)).map(_nonempty_scalar_text)
        flagged = data.samples.get("review_required", pd.Series(False, index=data.samples.index)).fillna(False).astype(bool)
        selected = data.samples.loc[flagged & all_fixations.eq(fixation_value)].copy()
        if selected.empty:
            raise ValidationError("This fixation has no flagged gaze samples")
    else:
        selected = current_selected
        if selected.empty:
            raise ValidationError("Select a gaze sample before saving the Body AOI review")

    path = _body_manual_review_path(store, session_id, artifact_id)
    with _MANUAL_REVIEW_LOCK:
        state = _load_body_manual_review_state(store, session_id, artifact_id)
        decisions = state.setdefault("decisions", {})
        now = utc_now()
        changed: list[str] = []
        resolved_entities: dict[str, str | None] = {}
        native_to_output = _first_output_for_native(data)
        for _, row in selected.iterrows():
            uid = _sample_uid(row)
            row_native_frame = int(pd.to_numeric(row.get("native_frame_index"), errors="coerce"))
            row_output_frame = int(native_to_output.get(row_native_frame, resolved_output_frame_index))
            if action == "clear":
                decisions.pop(uid, None)
                changed.append(uid)
                continue
            resolved_entity_id = entity_id
            entity_resolution_source = "explicit_entity_id" if entity_id else "not_applicable"
            if action == "override":
                effective_region = region or _nonempty_scalar_text(row.get("body_raw_region")) or _nonempty_scalar_text(row.get("body_region")) or "no_body"
                coarse = _coarse_body_review_region(effective_region)
                if coarse == "no_body":
                    resolved_entity_id = None
                    entity_resolution_source = "no_body_no_entity"
                else:
                    resolved_relation = relation or _nonempty_scalar_text(row.get("body_relation")) or "partner"
                    if resolved_relation not in BODY_MANUAL_REVIEW_RELATIONS:
                        resolved_relation = "partner"
                    resolved_entity_id, entity_resolution_source = _resolve_body_review_entity(
                        row,
                        relation=resolved_relation,
                        requested_entity_id=entity_id,
                        role_context=data.role_context,
                    )
                    if resolved_relation in {"partner", "self"} and not resolved_entity_id:
                        raise ValidationError(
                            f"The {resolved_relation} Body AOI correction has no uniquely resolved body entity. "
                            "Enter Entity ID explicitly before saving."
                        )
            decisions[uid] = {
                "gaze_sample_uid": uid,
                "native_frame_index": int(row_native_frame),
                "output_frame_index": int(row_output_frame),
                "apply_to_scope": apply_to,
                "source_gaze_row_index": _jsonable(row.get("source_gaze_row_index")),
                "action": action,
                "override_relation": relation,
                "override_entity_id": resolved_entity_id,
                "requested_override_entity_id": entity_id,
                "entity_resolution_source": entity_resolution_source,
                "override_region": region,
                "note": str(note or "").strip(),
                "reviewer": str(reviewer or "local reviewer").strip() or "local reviewer",
                "automatic": _body_manual_review_snapshot(row),
                "updated_at": now,
            }
            resolved_entities[uid] = resolved_entity_id
            changed.append(uid)
        state["updated_at"] = now
        if changed:
            state["finalized_artifact_id"] = None
            state["finalized_at"] = None
        atomic_write_json(path, state)
    summary = _manual_review_summary(data, state)
    events = dict(data.events)
    events["pending_review"] = summary["pending_review_representative_frames"]
    events["pending_review_frames"] = summary["pending_frames"]
    events["reviewed"] = summary["reviewed_review_representative_frames"]
    return {
        "saved": True,
        "changed_sample_uids": changed,
        "changed_sample_count": int(len(changed)),
        "apply_to_scope": apply_to,
        "resolved_entity_ids": resolved_entities if action != "clear" else {},
        "manual_review": summary,
        "event_frames": events,
    }

def _decision_rows(state: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for uid, decision in sorted((state.get("decisions") or {}).items()):
        row = dict(decision) if isinstance(decision, dict) else {"action": str(decision)}
        row["gaze_sample_uid"] = uid
        automatic = row.pop("automatic", None)
        if isinstance(automatic, dict):
            for key, value in automatic.items():
                row[f"automatic_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def apply_face_aoi_manual_review(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_id: str,
    label: str | None = None,
    require_complete: bool = True,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.FACE_AOI:
        raise ValidationError("Select a Face AOI artifact to apply manual review")
    emit(progress, 0.04, "Loading Face AOI samples and manual-review decisions")
    samples = _read_csv(artifact, "face_aoi_samples.csv")
    participant_artifact = _participant_artifact_for_review(session, "face", artifact)
    participant_projections = _regions_file(participant_artifact, "participant_head_projection.csv")
    face_identities = _regions_file(participant_artifact, "face_identity_assignment.csv")
    role_context = _manual_review_role_context(
        artifact=artifact,
        participant_artifact=participant_artifact,
        participant_projections=participant_projections,
        face_identities=face_identities,
        samples=samples,
    )
    projections_by_native: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not participant_projections.empty and "native_frame_index" in participant_projections.columns:
        for projection in participant_projections.to_dict("records"):
            try:
                projections_by_native[int(projection["native_frame_index"])].append(_row_dict(projection))
            except Exception:
                continue
    state = _load_manual_review_state(store, session_id, artifact_id)
    decisions = state.get("decisions") or {}
    review_mask = samples.get("review_required", pd.Series(False, index=samples.index)).fillna(False).astype(bool)
    flagged_uids = {_sample_uid(row) for _, row in samples.loc[review_mask].iterrows()}
    resolved_uids = {
        str(uid) for uid, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("action") in MANUAL_REVIEW_RESOLVED_ACTIONS
    }
    pending = sorted(flagged_uids - resolved_uids)
    if require_complete and pending:
        raise ValidationError(
            f"Manual Face AOI review is incomplete: {len(pending)} flagged gaze samples remain pending"
        )
    if not decisions:
        raise ValidationError("No manual Face AOI review decisions have been saved")
    token.raise_if_cancelled()
    reviewed = samples.copy()
    preserve = [
        "face_entity_id", "face_region", "face_social_region", "target_relation",
        "identity_status", "assignment_status", "face_aoi_confidence",
        "include_in_primary_analysis", "review_required",
    ]
    for column in preserve:
        if column in reviewed.columns and f"automatic_{column}" not in reviewed.columns:
            reviewed[f"automatic_{column}"] = reviewed[column]
    for column, default in [
        ("manual_review_action", ""), ("manual_review_note", ""),
        ("manual_reviewer", ""), ("manual_reviewed_at", ""),
        ("manual_review_source_artifact_id", artifact_id),
        ("manual_review_entity_resolution_source", ""),
    ]:
        if column not in reviewed.columns:
            reviewed[column] = default
    uid_to_index = {_sample_uid(row): index for index, row in reviewed.iterrows()}
    applied = 0
    applied_entity_resolution: dict[str, dict[str, Any]] = {}
    for ordinal, (uid, decision) in enumerate(sorted(decisions.items())):
        token.raise_if_cancelled()
        if not isinstance(decision, dict) or uid not in uid_to_index:
            continue
        action = decision.get("action")
        if action not in MANUAL_REVIEW_RESOLVED_ACTIONS:
            continue
        index = uid_to_index[uid]
        reviewed.at[index, "manual_review_action"] = action
        reviewed.at[index, "manual_review_note"] = decision.get("note") or ""
        reviewed.at[index, "manual_reviewer"] = decision.get("reviewer") or "local reviewer"
        reviewed.at[index, "manual_reviewed_at"] = decision.get("updated_at") or state.get("updated_at") or utc_now()
        reviewed.at[index, "manual_review_source_artifact_id"] = artifact_id
        native_value = reviewed.at[index, "native_frame_index"] if "native_frame_index" in reviewed.columns else None
        try:
            native_index = int(native_value) if native_value is not None and not pd.isna(native_value) else -1
        except Exception:
            native_index = -1
        frame_role_context = _manual_review_frame_role_context(
            role_context, projections_by_native.get(native_index, [])
        )
        entity_resolution_source = "not_applicable"
        if action == "accept_auto":
            relation = _resolved_manual_relation(reviewed.loc[index])
            region = str(reviewed.at[index, "face_region"] or "other_face")
            entity, entity_resolution_source = _resolve_manual_review_entity(
                reviewed.loc[index], relation=relation, role_context=frame_role_context
            )
            reviewed.at[index, "target_relation"] = relation
            reviewed.at[index, "face_entity_id"] = entity
            reviewed.at[index, "identity_status"] = "manual_review_accepted_auto"
            reviewed.at[index, "assignment_status"] = "manual_review_accepted_auto"
            reviewed.at[index, "identity_confidence"] = 1.0
            reviewed.at[index, "face_aoi_confidence"] = 1.0
            reviewed.at[index, "include_in_primary_analysis"] = relation in {"partner", "self"} and region not in {"no_face", "no_face_detection", "invalid_gaze", "out_of_frame"}
            reviewed.at[index, "review_required"] = False
            reviewed.at[index, "face_social_region"] = _derive_face_social_region(reviewed.loc[index])
        elif action == "exclude":
            entity_resolution_source = "excluded_no_entity"
            reviewed.at[index, "face_entity_id"] = None
            reviewed.at[index, "face_region"] = "no_face"
            reviewed.at[index, "face_social_region"] = "no_face"
            reviewed.at[index, "target_relation"] = "other_person"
            reviewed.at[index, "identity_status"] = "manual_review_excluded"
            reviewed.at[index, "assignment_status"] = "manual_review_excluded"
            reviewed.at[index, "face_aoi_confidence"] = 1.0
            reviewed.at[index, "include_in_primary_analysis"] = False
            reviewed.at[index, "review_required"] = False
        elif action == "override":
            relation = _resolved_manual_relation(reviewed.loc[index], decision.get("override_relation"))
            region = str(decision.get("override_region") or reviewed.at[index, "face_region"] or "other_face")
            entity, entity_resolution_source = _resolve_manual_review_entity(
                reviewed.loc[index],
                relation=relation,
                requested_entity_id=(
                    decision.get("override_entity_id")
                    or decision.get("resolved_entity_id")
                ),
                role_context=frame_role_context,
            )
            if region == "no_face":
                relation = "other_person"
                entity = None
                entity_resolution_source = "no_face_no_entity"
                status = "manual_review_no_face"
            else:
                if relation in {"partner", "self"} and not entity:
                    raise ValidationError(
                        f"Saved manual review for {uid} has relation={relation} but no uniquely resolved entity. "
                        "Open the review player and enter Entity ID explicitly."
                    )
                status = "manual_review_override"
            reviewed.at[index, "target_relation"] = relation
            reviewed.at[index, "face_region"] = region
            reviewed.at[index, "face_entity_id"] = entity
            reviewed.at[index, "identity_status"] = status
            reviewed.at[index, "assignment_status"] = status
            reviewed.at[index, "identity_confidence"] = 1.0
            reviewed.at[index, "face_aoi_confidence"] = 1.0
            reviewed.at[index, "include_in_primary_analysis"] = relation in {"partner", "self"} and region != "no_face"
            reviewed.at[index, "review_required"] = False
            reviewed.at[index, "face_social_region"] = "no_face" if region == "no_face" else _derive_face_social_region(reviewed.loc[index])
        reviewed.at[index, "manual_review_entity_resolution_source"] = entity_resolution_source
        applied_entity_resolution[uid] = {
            "resolved_entity_id": _jsonable(reviewed.at[index, "face_entity_id"]),
            "entity_resolution_source": entity_resolution_source,
        }
        applied += 1
        if ordinal % 5000 == 0:
            emit(progress, 0.10 + 0.28 * (ordinal + 1) / max(1, len(decisions)), "Applying manual Face AOI decisions")
    token.raise_if_cancelled()
    trim_id = artifact.provenance.get("native_trim_artifact_id")
    trim_artifact = session.artifacts.get(trim_id) if trim_id else _parent_by_kind(session, artifact, ArtifactKind.NATIVE_REFERENCE_TRIM)
    if trim_artifact is None:
        raise ValidationError("The source Face AOI artifact has no Native-reference trim parent")
    frame_map = pd.read_csv(find_artifact_file(trim_artifact, names=["native_frame_map.csv"]), low_memory=False)
    from oculostudio.services.face_aoi import (
        _clip_statistics_qc_fields,
        _frame_summary as _face_frame_summary,
        _full_clip_partner_face_statistics,
    )
    frames = _face_frame_summary(reviewed, frame_map)
    clip_summary, clip_summary_table = _full_clip_partner_face_statistics(reviewed, frame_map)
    regions = _read_csv(artifact, "face_regions.csv")
    emit(progress, 0.48, "Writing an immutable reviewed Face AOI artifact")
    new_artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "face_aoi") / (".partial_" + new_artifact_id)
    final = store.derived_dir(session_id, "face_aoi") / new_artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        reviewed.to_csv(staging / "face_aoi_samples.csv", index=False)
        frames.to_csv(staging / "face_aoi_frames.csv", index=False)
        regions.to_csv(staging / "face_regions.csv", index=False)
        clip_summary_table.to_csv(staging / "face_aoi_clip_summary.csv", index=False)
        atomic_write_json(staging / "face_aoi_clip_summary.json", clip_summary)

        # Preserve the source artifact's audited automatic-review reduction report.
        # This report describes decisions made before human review; the reviewed
        # artifact then adds the immutable manual-decision ledger alongside it.
        auto_review_summary_csv_written = False
        auto_review_summary_json_written = False
        try:
            source_auto_csv = find_artifact_file(artifact, names=["face_aoi_auto_review_summary.csv"])
            shutil.copy2(source_auto_csv, staging / "face_aoi_auto_review_summary.csv")
            auto_review_summary_csv_written = True
        except Exception:
            pass
        try:
            source_auto_json = find_artifact_file(artifact, names=["face_aoi_auto_review_summary.json"])
            source_auto_summary = _load_json(source_auto_json)
            source_auto_summary["manual_review_lineage"] = {
                "source_face_aoi_artifact_id": artifact_id,
                "manual_review_decision_count": int(len(decisions)),
                "manual_review_applied_sample_count": int(applied),
                "manual_review_pending_sample_count": int(len(pending)),
            }
            atomic_write_json(staging / "face_aoi_auto_review_summary.json", source_auto_summary)
            auto_review_summary_json_written = True
        except Exception:
            pass

        decisions_frame = _decision_rows(state)
        if len(decisions_frame) and "gaze_sample_uid" in decisions_frame.columns:
            decisions_frame["resolved_entity_id"] = decisions_frame["gaze_sample_uid"].map(
                lambda uid: (applied_entity_resolution.get(str(uid)) or {}).get("resolved_entity_id")
            )
            decisions_frame["resolved_entity_resolution_source"] = decisions_frame["gaze_sample_uid"].map(
                lambda uid: (applied_entity_resolution.get(str(uid)) or {}).get("entity_resolution_source")
            )
        decisions_frame.to_csv(staging / "face_aoi_manual_review_decisions.csv", index=False)
        counts = Counter(reviewed.get("face_region", pd.Series(dtype=str)).fillna("unknown").astype(str))
        social_counts = Counter(reviewed.get("face_social_region", reviewed.get("face_region", pd.Series(dtype=str))).fillna("unknown").astype(str))
        clip_qc = _clip_statistics_qc_fields(clip_summary)
        qc = dict(artifact.qc)
        qc.update({
            "manual_review_applied": True,
            "manual_review_source_artifact_id": artifact_id,
            "manual_review_decision_count": int(len(decisions)),
            "manual_review_applied_sample_count": int(applied),
            "manual_review_pending_sample_count": int(len(pending)),
            "review_required_fraction": float(reviewed.get("review_required", pd.Series(False, index=reviewed.index)).fillna(False).astype(bool).mean()) if len(reviewed) else 0.0,
            "primary_analysis_fraction": float(reviewed.get("include_in_primary_analysis", pd.Series(False, index=reviewed.index)).fillna(False).astype(bool).mean()) if len(reviewed) else 0.0,
            "partner_face_fraction": float(reviewed.get("target_relation", pd.Series("", index=reviewed.index)).eq("partner").mean()) if len(reviewed) else 0.0,
            "other_person_face_fraction": float(reviewed.get("target_relation", pd.Series("", index=reviewed.index)).eq("other_person").mean()) if len(reviewed) else 0.0,
            "face_region_counts": dict(counts),
            "face_social_region_counts": dict(social_counts),
            "frame_count": int(len(frames)),
            "overlay_rendered": False,
            "parquet_written": False,
            **clip_qc,
        })
        parameters = dict(artifact.parameters)
        parameters.update({
            "manual_review_applied": True,
            "manual_review_source_artifact_id": artifact_id,
            "manual_review_require_complete": bool(require_complete),
        })
        manifest = {
            "schema_version": "oculostudio-face-aoi-reviewed-1.1",
            "artifact_id": new_artifact_id,
            "source_face_aoi_artifact_id": artifact_id,
            "native_trim_artifact_id": trim_artifact.id,
            "manual_review_schema_version": MANUAL_REVIEW_SCHEMA,
            "manual_review_state": {
                "decision_count": len(decisions),
                "applied_sample_count": applied,
                "pending_sample_count": len(pending),
                "review_completed_at": utc_now(),
            },
            "outputs": {
                "samples_csv": "face_aoi_samples.csv",
                "frames_csv": "face_aoi_frames.csv",
                "regions_csv": "face_regions.csv",
                "clip_summary_csv": "face_aoi_clip_summary.csv",
                "clip_summary_json": "face_aoi_clip_summary.json",
                "auto_review_summary_csv": "face_aoi_auto_review_summary.csv" if auto_review_summary_csv_written else None,
                "auto_review_summary_json": "face_aoi_auto_review_summary.json" if auto_review_summary_json_written else None,
                "review_decisions_csv": "face_aoi_manual_review_decisions.csv",
                "overlay_video": None,
            },
            "parameters": parameters,
            "qc": qc,
            "created_at": utc_now(),
        }
        atomic_write_json(staging / "face_aoi_manifest.json", manifest)
        atomic_write_json(staging / "qc_metrics.json", qc)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    reviewed_label = (label or "").strip() or f"{artifact.label} reviewed"
    result_artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.FACE_AOI,
        label=reviewed_label,
        output_root=final,
        source_asset_ids=artifact.source_asset_ids,
        parent_artifact_ids=[artifact_id],
        parameters=parameters,
        qc=qc,
        provenance={
            **artifact.provenance,
            "manual_review_applied": True,
            "manual_review_source_artifact_id": artifact_id,
            "manual_review_decisions_file": "face_aoi_manual_review_decisions.csv",
            "native_trim_artifact_id": trim_artifact.id,
        },
        artifact_id=new_artifact_id,
    )
    with _MANUAL_REVIEW_LOCK:
        state = _load_manual_review_state(store, session_id, artifact_id)
        state["finalized_artifact_id"] = new_artifact_id
        state["finalized_at"] = utc_now()
        state["updated_at"] = state["finalized_at"]
        atomic_write_json(_manual_review_path(store, session_id, artifact_id), state)
    emit(progress, 1.0, "Reviewed Face AOI artifact created")
    return {
        "artifact": result_artifact.model_dump(mode="json"),
        "samples_csv": str(final / "face_aoi_samples.csv"),
        "frames_csv": str(final / "face_aoi_frames.csv"),
        "clip_summary_csv": str(final / "face_aoi_clip_summary.csv"),
        "clip_summary_json": str(final / "face_aoi_clip_summary.json"),
        "auto_review_summary_csv": str(final / "face_aoi_auto_review_summary.csv") if (final / "face_aoi_auto_review_summary.csv").exists() else None,
        "auto_review_summary_json": str(final / "face_aoi_auto_review_summary.json") if (final / "face_aoi_auto_review_summary.json").exists() else None,
        "decisions_csv": str(final / "face_aoi_manual_review_decisions.csv"),
        "manifest": str(final / "face_aoi_manifest.json"),
    }



def apply_body_aoi_manual_review(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_id: str,
    label: str | None = None,
    require_complete: bool = True,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Create an immutable Body AOI artifact with saved human decisions applied."""
    token = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.BODY_AOI:
        raise ValidationError("Select a Body AOI artifact to apply manual review")
    _require_body_geometry_review_ready(artifact)
    emit(progress, 0.04, "Loading Body AOI samples and manual-review decisions")
    samples = _read_csv(artifact, "body_aoi_samples.csv")
    if "body_raw_region" not in samples.columns and "body_region" in samples.columns:
        samples["body_raw_region"] = samples["body_region"]
    if "body_coarse_region" not in samples.columns and "body_region" in samples.columns:
        samples["body_coarse_region"] = samples["body_region"].map(_coarse_body_review_region)
    if "body_social_region" not in samples.columns:
        relation = samples.get("body_relation", pd.Series("", index=samples.index)).fillna("").astype(str)
        region = samples.get("body_region", pd.Series("unknown", index=samples.index)).fillna("unknown").astype(str)
        samples["body_social_region"] = [
            f"{rel}_{reg}" if rel in BODY_MANUAL_REVIEW_RELATIONS and reg not in _BODY_NON_HIT_REGIONS else reg
            for rel, reg in zip(relation, region)
        ]
    role_context = _body_manual_review_role_context(artifact=artifact, samples=samples)
    state = _load_body_manual_review_state(store, session_id, artifact_id)
    decisions = state.get("decisions") or {}
    review_mask = samples.get("review_required", pd.Series(False, index=samples.index)).fillna(False).astype(bool)
    flagged_uids = {_sample_uid(row) for _, row in samples.loc[review_mask].iterrows()}
    resolved_uids = {
        str(uid) for uid, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("action") in MANUAL_REVIEW_RESOLVED_ACTIONS
    }
    pending = sorted(flagged_uids - resolved_uids)
    if require_complete and pending:
        raise ValidationError(
            f"Manual Body AOI review is incomplete: {len(pending)} flagged gaze samples remain pending"
        )
    if not decisions:
        raise ValidationError("No manual Body AOI review decisions have been saved")

    token.raise_if_cancelled()
    reviewed = samples.copy()
    preserve = [
        "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region", "body_region",
        "body_social_region", "body_confidence", "body_exact_hit", "body_distance_mm",
        "body_hit_x_mm", "body_hit_y_mm", "body_hit_z_mm", "body_angular_miss_deg",
        "body_geometry_backend", "self_occlusion", "occlusion_status", "assignment_status", "review_required",
    ]
    for column in preserve:
        if column in reviewed.columns and f"automatic_{column}" not in reviewed.columns:
            reviewed[f"automatic_{column}"] = reviewed[column]
    reviewed["manual_review_action"] = None
    reviewed["manual_review_note"] = None
    reviewed["manual_review_reviewer"] = None
    reviewed["manual_review_updated_at"] = None
    reviewed["manual_review_apply_to_scope"] = None
    reviewed["manual_review_entity_resolution_source"] = None
    reviewed["manual_review_geometry_consistent"] = None
    output_granularity = str((artifact.parameters or {}).get("output_granularity") or "coarse_social_regions")
    applied = 0
    applied_entity_resolution: dict[str, dict[str, Any]] = {}
    effective_fields = [
        "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region", "body_region",
        "body_social_region", "body_confidence", "body_exact_hit", "body_distance_mm",
        "body_hit_x_mm", "body_hit_y_mm", "body_hit_z_mm", "body_angular_miss_deg",
        "body_geometry_backend", "self_occlusion", "occlusion_status", "assignment_status", "review_required",
    ]
    for ordinal, (index, row) in enumerate(reviewed.iterrows()):
        token.raise_if_cancelled()
        uid = _sample_uid(row)
        decision = decisions.get(uid)
        if not isinstance(decision, dict):
            continue
        effective, preview = _effective_body_review_row(
            row,
            decision,
            role_context=role_context,
            output_granularity=output_granularity,
        )
        for column in effective_fields:
            if column in effective.index:
                reviewed.at[index, column] = effective.get(column)
        reviewed.at[index, "manual_review_action"] = decision.get("action")
        reviewed.at[index, "manual_review_note"] = decision.get("note")
        reviewed.at[index, "manual_review_reviewer"] = decision.get("reviewer")
        reviewed.at[index, "manual_review_updated_at"] = decision.get("updated_at")
        reviewed.at[index, "manual_review_apply_to_scope"] = decision.get("apply_to_scope")
        reviewed.at[index, "manual_review_entity_resolution_source"] = preview.get("entity_resolution_source")
        reviewed.at[index, "manual_review_geometry_consistent"] = preview.get("geometry_consistent")
        applied_entity_resolution[uid] = {
            "resolved_entity_id": preview.get("resolved_entity_id"),
            "entity_resolution_source": preview.get("entity_resolution_source"),
        }
        if decision.get("action") in MANUAL_REVIEW_RESOLVED_ACTIONS:
            applied += 1
        if ordinal % 1000 == 0:
            emit(progress, 0.10 + 0.28 * (ordinal + 1) / max(1, len(reviewed)), "Applying manual Body AOI decisions")

    token.raise_if_cancelled()
    from oculostudio.services.body_aoi import _frame_summary as _body_frame_summary
    frames = _body_frame_summary(reviewed)
    regions = _read_csv(artifact, "body_regions.csv")
    emit(progress, 0.48, "Writing an immutable reviewed Body AOI artifact")
    new_artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "body_aoi", artifact.clip_id or "full") / (".partial_" + new_artifact_id)
    final = store.derived_dir(session_id, "body_aoi", artifact.clip_id or "full") / new_artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        reviewed.to_csv(staging / "body_aoi_samples.csv", index=False)
        frames.to_csv(staging / "body_aoi_frames.csv", index=False)
        regions.to_csv(staging / "body_regions.csv", index=False)
        parquet_files: list[str] = []
        source_has_parquet = any(Path(file.path).suffix.lower() == ".parquet" for file in artifact.files)
        if source_has_parquet:
            try:
                for frame, name in [
                    (reviewed, "body_aoi_samples.parquet"),
                    (frames, "body_aoi_frames.parquet"),
                    (regions, "body_regions.parquet"),
                ]:
                    frame.to_parquet(staging / name, index=False)
                    parquet_files.append(name)
            except Exception:
                parquet_files = []
        decisions_frame = _decision_rows(state)
        if len(decisions_frame) and "gaze_sample_uid" in decisions_frame.columns:
            decisions_frame["resolved_entity_id"] = decisions_frame["gaze_sample_uid"].map(
                lambda uid: (applied_entity_resolution.get(str(uid)) or {}).get("resolved_entity_id")
            )
            decisions_frame["resolved_entity_resolution_source"] = decisions_frame["gaze_sample_uid"].map(
                lambda uid: (applied_entity_resolution.get(str(uid)) or {}).get("entity_resolution_source")
            )
        decisions_frame.to_csv(staging / "body_aoi_manual_review_decisions.csv", index=False)

        regions_series = reviewed.get("body_region", pd.Series("unknown", index=reviewed.index)).fillna("unknown").astype(str)
        raw_regions = reviewed.get("body_raw_region", regions_series).fillna("unknown").astype(str)
        relations = reviewed.get("body_relation", pd.Series("none", index=reviewed.index)).fillna("none").astype(str)
        social = reviewed.get("body_social_region", regions_series).fillna("unknown").astype(str)
        hit_mask = ~regions_series.str.lower().isin(_BODY_NON_HIT_REGIONS)
        qc = dict(artifact.qc)
        qc.update({
            "manual_review_applied": True,
            "manual_review_source_artifact_id": artifact_id,
            "manual_review_decision_count": int(len(decisions)),
            "manual_review_applied_sample_count": int(applied),
            "manual_review_pending_sample_count": int(len(pending)),
            "output_rows": int(len(reviewed)),
            "frame_count": int(len(frames)),
            "review_required_fraction": float(reviewed.get("review_required", pd.Series(False, index=reviewed.index)).fillna(False).astype(bool).mean()) if len(reviewed) else 0.0,
            "body_hit_fraction": float(hit_mask.mean()) if len(reviewed) else 0.0,
            "body_region_counts": dict(Counter(regions_series)),
            "body_raw_region_counts": dict(Counter(raw_regions)),
            "body_social_region_counts": dict(Counter(social)),
            "body_relation_counts": dict(Counter(relations)),
            "self_hit_fraction": float(relations.eq("self").mean()) if len(reviewed) else 0.0,
            "partner_hit_fraction": float(relations.eq("partner").mean()) if len(reviewed) else 0.0,
            "self_occlusion_fraction": float(reviewed.get("self_occlusion", pd.Series(False, index=reviewed.index)).fillna(False).astype(bool).mean()) if len(reviewed) else 0.0,
            "output_granularity": output_granularity,
            "parquet_written": bool(parquet_files),
        })
        parameters = dict(artifact.parameters)
        parameters.update({
            "manual_review_applied": True,
            "manual_review_source_artifact_id": artifact_id,
            "manual_review_require_complete": bool(require_complete),
        })
        manifest = {
            "schema_version": "oculostudio-body-aoi-reviewed-1.0",
            "artifact_id": new_artifact_id,
            "source_body_aoi_artifact_id": artifact_id,
            "manual_review_schema_version": BODY_MANUAL_REVIEW_SCHEMA,
            "manual_review_state": {
                "decision_count": int(len(decisions)),
                "applied_sample_count": int(applied),
                "pending_sample_count": int(len(pending)),
                "review_completed_at": utc_now(),
            },
            "outputs": {
                "samples_csv": "body_aoi_samples.csv",
                "frames_csv": "body_aoi_frames.csv",
                "regions_csv": "body_regions.csv",
                "parquet": parquet_files,
                "review_decisions_csv": "body_aoi_manual_review_decisions.csv",
            },
            "parameters": parameters,
            "qc": qc,
            "created_at": utc_now(),
        }
        atomic_write_json(staging / "body_aoi_manifest.json", manifest)
        atomic_write_json(staging / "qc_metrics.json", qc)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    reviewed_label = (label or "").strip() or f"{artifact.label} reviewed"
    result_artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.BODY_AOI,
        label=reviewed_label,
        output_root=final,
        source_asset_ids=artifact.source_asset_ids,
        parent_artifact_ids=[artifact_id],
        clip_id=artifact.clip_id,
        parameters=parameters,
        qc=qc,
        provenance={
            **artifact.provenance,
            "manual_review_applied": True,
            "manual_review_source_artifact_id": artifact_id,
            "manual_review_decisions_file": "body_aoi_manual_review_decisions.csv",
        },
        artifact_id=new_artifact_id,
    )
    with _MANUAL_REVIEW_LOCK:
        state = _load_body_manual_review_state(store, session_id, artifact_id)
        state["finalized_artifact_id"] = new_artifact_id
        state["finalized_at"] = utc_now()
        state["updated_at"] = state["finalized_at"]
        atomic_write_json(_body_manual_review_path(store, session_id, artifact_id), state)
    emit(progress, 1.0, "Reviewed Body AOI artifact created")
    return {
        "artifact": result_artifact.model_dump(mode="json"),
        "samples_csv": str(final / "body_aoi_samples.csv"),
        "frames_csv": str(final / "body_aoi_frames.csv"),
        "regions_csv": str(final / "body_regions.csv"),
        "decisions_csv": str(final / "body_aoi_manual_review_decisions.csv"),
        "manifest": str(final / "body_aoi_manifest.json"),
    }

def _artifact_identity_codes(artifact: ArtifactRecord) -> dict[str, str]:
    text = " ".join([
        str(artifact.label or ""),
        json.dumps(artifact.provenance or {}, ensure_ascii=True, sort_keys=True),
        json.dumps(artifact.parameters or {}, ensure_ascii=True, sort_keys=True),
    ])
    def code(prefix: str) -> str:
        match = re.search(rf"(?:^|[^A-Za-z0-9])({prefix}\d{{1,6}})(?![A-Za-z0-9])", text, re.IGNORECASE)
        return match.group(1).upper() if match else ""
    return {"session_code": code("S"), "interaction_id": code("I"), "role_id": code("R")}

def review_options(store: ProjectStore, *, session_id: str) -> dict[str, Any]:
    session = store.project.sessions[session_id]
    videos = [row for row in session.artifacts.values() if row.kind == ArtifactKind.PRIVACY_CLIP]
    modes = {}
    for mode, kind in [("face", ArtifactKind.FACE_AOI), ("body", ArtifactKind.BODY_AOI), ("fusion", ArtifactKind.TARGET_FUSION)]:
        rows = []
        for artifact in session.artifacts.values():
            if artifact.kind != kind:
                continue
            default = _default_video(session, mode, artifact)
            trim_id = _native_trim_id(session, mode, artifact)
            compatible = [
                video.id for video in videos
                if not trim_id or video.provenance.get("native_trim_artifact_id") == trim_id
            ]
            identity = _artifact_identity_codes(artifact)
            manual_state = (
                _load_manual_review_state(store, session_id, artifact.id)
                if mode == "face"
                else _load_body_manual_review_state(store, session_id, artifact.id)
                if mode == "body"
                else None
            )
            rows.append({
                "artifact_id": artifact.id,
                "label": artifact.label,
                "kind": artifact.kind,
                "created_at": artifact.created_at,
                "default_video_artifact_id": default.id if default else None,
                "compatible_video_artifact_ids": compatible,
                "native_trim_artifact_id": trim_id,
                "qc": artifact.qc,
                **identity,
                "manual_review_draft": {
                    "decision_count": len((manual_state or {}).get("decisions") or {}),
                    "updated_at": (manual_state or {}).get("updated_at"),
                    "finalized_artifact_id": (manual_state or {}).get("finalized_artifact_id"),
                } if manual_state is not None else None,
            })
        modes[mode] = rows
    return {
        "modes": modes,
        "videos": [{
            "artifact_id": row.id,
            "label": row.label,
            "created_at": row.created_at,
            "native_trim_artifact_id": row.provenance.get("native_trim_artifact_id"),
            "fps": row.qc.get("output_fps"),
            "frame_count": row.qc.get("output_frame_count"),
            **_artifact_identity_codes(row),
        } for row in videos],
    }



_BODY_NON_HIT_REGIONS = {
    "", "none", "nan", "unknown", "no_body", "invalid_gaze", "out_of_frame", "no_gaze",
}


def _body_text_series(samples: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in samples.columns:
        return pd.Series(default, index=samples.index, dtype=object)
    return samples[column].fillna(default).astype(str).str.strip()


def _body_hit_mask(samples: pd.DataFrame) -> pd.Series:
    region = _body_text_series(samples, "body_coarse_region", "")
    fallback = _body_text_series(samples, "body_region", "unknown")
    region = region.mask(region.eq(""), fallback)
    entity = _body_text_series(samples, "body_entity_id", "")
    return ~region.str.lower().isin(_BODY_NON_HIT_REGIONS) & entity.ne("")


def _counter_percentages(counter: Counter, denominator: int) -> dict[str, float]:
    base = max(1, int(denominator))
    return {str(key): float(value) / base for key, value in counter.items()}


def _body_analytics(samples: pd.DataFrame) -> dict[str, Any]:
    """Summarise Body AOI coverage, social relation and anatomical regions.

    Counts stay sample-based because gaze sampling is the scientific unit.  The
    UI presents both counts and percentages and never estimates image-space
    body masks from world-space C3D primitives.
    """
    total = int(len(samples))
    if total == 0:
        return {
            "sample_count": 0,
            "hit_count": 0,
            "hit_fraction": 0.0,
            "region_counts": {},
            "relation_counts": {},
            "social_region_counts": {},
            "per_relation_region_counts": {},
        }

    hit = _body_hit_mask(samples)
    hit_count = int(hit.sum())
    relation = _body_text_series(samples, "body_relation", "none").str.lower()
    coarse = _body_text_series(samples, "body_coarse_region", "")
    fallback = _body_text_series(samples, "body_region", "unknown")
    coarse = coarse.mask(coarse.eq(""), fallback).str.lower()
    social = _body_text_series(samples, "body_social_region", "")
    social = social.mask(social.eq(""), [
        f"{rel}_{reg}" if rel in {"self", "partner"} and reg not in _BODY_NON_HIT_REGIONS else reg
        for rel, reg in zip(relation, coarse)
    ]).str.lower()

    relation_counts = Counter(relation.loc[hit])
    coarse_counts = Counter(coarse.loc[hit])
    social_counts = Counter(social.loc[hit])
    assignment_counts = Counter(_body_text_series(samples, "assignment_status", "unknown"))
    occlusion_counts = Counter(_body_text_series(samples, "occlusion_status", "unknown"))

    per_relation: dict[str, dict[str, int]] = {}
    for rel in ("self", "partner", "other"):
        mask = hit & relation.eq(rel)
        counts = Counter(coarse.loc[mask])
        if counts:
            per_relation[rel] = dict(counts)

    confidence = pd.to_numeric(samples.get("body_confidence", pd.Series(np.nan, index=samples.index)), errors="coerce")
    distance = pd.to_numeric(samples.get("body_distance_mm", pd.Series(np.nan, index=samples.index)), errors="coerce")
    exact = samples.get("body_exact_hit", pd.Series(False, index=samples.index)).fillna(False).astype(bool)
    review = samples.get("review_required", pd.Series(False, index=samples.index)).fillna(False).astype(bool)
    review_before = samples.get(
        "review_required_before_auto_resolution", review
    ).fillna(False).astype(bool)
    auto_resolved = samples.get(
        "auto_review_resolved", pd.Series(False, index=samples.index)
    ).fillna(False).astype(bool)
    auto_sources = Counter(
        _body_text_series(samples.loc[auto_resolved], "auto_review_resolution_source", "")
    ) if auto_resolved.any() else Counter()
    self_occlusion = samples.get("self_occlusion", pd.Series(False, index=samples.index)).fillna(False).astype(bool)
    valid_ray = samples.get("ray_valid", pd.Series(False, index=samples.index)).fillna(False).astype(bool)
    no_body = coarse.str.lower().eq("no_body")
    invalid = coarse.str.lower().isin({"invalid_gaze", "out_of_frame"})
    unknown = coarse.str.lower().isin({"unknown", ""})
    hit_distance = distance.loc[hit & distance.notna()]
    hit_confidence = confidence.loc[hit & confidence.notna()]

    return {
        "sample_count": total,
        "valid_ray_count": int(valid_ray.sum()),
        "valid_ray_fraction": float(valid_ray.mean()),
        "hit_count": hit_count,
        "hit_fraction": float(hit_count / total),
        "no_body_count": int(no_body.sum()),
        "no_body_fraction": float(no_body.mean()),
        "invalid_count": int(invalid.sum()),
        "invalid_fraction": float(invalid.mean()),
        "unknown_count": int(unknown.sum()),
        "unknown_fraction": float(unknown.mean()),
        "review_required_count": int(review.sum()),
        "review_required_fraction": float(review.mean()),
        "review_required_count_before_auto_resolution": int(review_before.sum()),
        "review_required_fraction_before_auto_resolution": float(review_before.mean()),
        "auto_review_resolved_count": int(auto_resolved.sum()),
        "auto_review_resolution_source_counts": dict(auto_sources),
        "review_frame_count": int(samples.loc[review, "scene_frame_index"].nunique()) if "scene_frame_index" in samples.columns else 0,
        "review_fixation_count": int(samples.loc[review, "fixation_id"].dropna().astype(str).nunique()) if "fixation_id" in samples.columns else 0,
        "exact_hit_count": int((hit & exact).sum()),
        "exact_hit_fraction_of_hits": float((hit & exact).sum() / max(1, hit_count)),
        "uncertainty_hit_count": int((hit & ~exact).sum()),
        "self_occlusion_count": int(self_occlusion.sum()),
        "self_occlusion_fraction": float(self_occlusion.mean()),
        "mean_hit_confidence": float(hit_confidence.mean()) if len(hit_confidence) else 0.0,
        "mean_hit_distance_mm": float(hit_distance.mean()) if len(hit_distance) else None,
        "median_hit_distance_mm": float(hit_distance.median()) if len(hit_distance) else None,
        "relation_counts": dict(relation_counts),
        "relation_percentages_of_all_samples": _counter_percentages(relation_counts, total),
        "relation_percentages_of_hits": _counter_percentages(relation_counts, hit_count),
        "region_counts": dict(coarse_counts),
        "region_percentages_of_all_samples": _counter_percentages(coarse_counts, total),
        "region_percentages_of_hits": _counter_percentages(coarse_counts, hit_count),
        "social_region_counts": dict(social_counts),
        "social_region_percentages_of_all_samples": _counter_percentages(social_counts, total),
        "per_relation_region_counts": per_relation,
        "assignment_status_counts": dict(assignment_counts),
        "occlusion_status_counts": dict(occlusion_counts),
    }


def _body_timeline(data: ReviewDataset, *, maximum_bins: int = 180) -> dict[str, Any]:
    """Return a compact whole-clip Body AOI timeline for the review canvas."""
    frame_count = max(1, int(data.video_info.get("frame_count") or 1))
    fps = max(1e-6, float(data.video_info.get("fps") or 30.0))
    bin_count = min(maximum_bins, frame_count)
    bin_count = max(1, bin_count)
    first_output = _first_output_for_native(data)
    frame_values = pd.to_numeric(
        data.samples.get("native_frame_index", pd.Series(np.nan, index=data.samples.index)),
        errors="coerce",
    ).map(first_output)
    valid = frame_values.notna()
    timeline = data.samples.loc[valid].copy()
    timeline["__output_frame"] = frame_values.loc[valid].astype(int)
    timeline["__bin"] = np.minimum(
        bin_count - 1,
        np.floor(timeline["__output_frame"].to_numpy(dtype=float) * bin_count / frame_count).astype(int),
    )
    bins: list[dict[str, Any]] = []
    grouped = {int(key): value for key, value in timeline.groupby("__bin", sort=False)}
    for bin_index in range(bin_count):
        start_frame = int(math.floor(bin_index * frame_count / bin_count))
        end_frame = int(max(start_frame, math.ceil((bin_index + 1) * frame_count / bin_count) - 1))
        group = grouped.get(bin_index, timeline.iloc[0:0])
        if len(group):
            hit = _body_hit_mask(group)
            relation = _body_text_series(group, "body_relation", "none").str.lower()
            coarse = _body_text_series(group, "body_coarse_region", "")
            coarse = coarse.mask(coarse.eq(""), _body_text_series(group, "body_region", "unknown")).str.lower()
            social = _body_text_series(group, "body_social_region", "")
            social = social.mask(social.eq(""), [
                f"{rel}_{reg}" if rel in {"self", "partner"} and reg not in _BODY_NON_HIT_REGIONS else reg
                for rel, reg in zip(relation, coarse)
            ]).str.lower()
            social_counts = Counter(social.loc[hit])
            relation_counts = Counter(relation.loc[hit])
            region_counts = Counter(coarse.loc[hit])
            review = group.get("review_required", pd.Series(False, index=group.index)).fillna(False).astype(bool)
            self_occlusion = group.get("self_occlusion", pd.Series(False, index=group.index)).fillna(False).astype(bool)
            confidence = pd.to_numeric(group.get("body_confidence", pd.Series(np.nan, index=group.index)), errors="coerce")
            dominant_social = social_counts.most_common(1)[0][0] if social_counts else "no_body"
            dominant_relation = relation_counts.most_common(1)[0][0] if relation_counts else "none"
            dominant_region = region_counts.most_common(1)[0][0] if region_counts else "no_body"
            bins.append({
                "index": bin_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": start_frame / fps,
                "end_seconds": min(frame_count / fps, (end_frame + 1) / fps),
                "sample_count": int(len(group)),
                "hit_fraction": float(hit.mean()),
                "review_fraction": float(review.mean()),
                "self_occlusion_fraction": float(self_occlusion.mean()),
                "mean_confidence": float(confidence.fillna(0).mean()),
                "dominant_social_region": dominant_social,
                "dominant_relation": dominant_relation,
                "dominant_region": dominant_region,
            })
        else:
            bins.append({
                "index": bin_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": start_frame / fps,
                "end_seconds": min(frame_count / fps, (end_frame + 1) / fps),
                "sample_count": 0,
                "hit_fraction": 0.0,
                "review_fraction": 0.0,
                "self_occlusion_fraction": 0.0,
                "mean_confidence": 0.0,
                "dominant_social_region": "no_data",
                "dominant_relation": "none",
                "dominant_region": "no_data",
            })
    return {
        "frame_count": frame_count,
        "bin_count": bin_count,
        "bins": bins,
        "encoding": "dominant body target with hit/review/occlusion fractions",
    }


def _body_world_bounds(data: ReviewDataset, *, maximum_rows: int = 6000) -> dict[str, Any] | None:
    """Estimate stable robust world bounds for the dual orthographic inset."""
    points: list[np.ndarray] = []
    maximum_radius_mm = 0.0
    regions = data.body_regions
    if not regions.empty and "geometry_json" in regions.columns:
        if len(regions) > maximum_rows:
            indices = np.linspace(0, len(regions) - 1, maximum_rows, dtype=int)
            geometry_values = regions.iloc[indices]["geometry_json"]
        else:
            geometry_values = regions["geometry_json"]
        for value in geometry_values:
            try:
                geometry = json.loads(value) if isinstance(value, str) else value
                if not isinstance(geometry, dict):
                    continue
                try:
                    radius_value = float(geometry.get("radius_mm") or 0.0)
                    if np.isfinite(radius_value):
                        maximum_radius_mm = max(maximum_radius_mm, radius_value)
                except Exception:
                    pass
                for key in ("center_mm", "start_mm", "end_mm"):
                    candidate = geometry.get(key)
                    if candidate is None:
                        continue
                    point = np.asarray(candidate, dtype=float)
                    if point.shape == (3,) and np.isfinite(point).all():
                        points.append(point)
            except Exception:
                continue
    numeric_columns = [
        ("body_hit_x_mm", "body_hit_y_mm", "body_hit_z_mm"),
        ("ray_origin_x_mm", "ray_origin_y_mm", "ray_origin_z_mm"),
    ]
    sample_rows = data.samples
    if len(sample_rows) > maximum_rows:
        sample_rows = sample_rows.iloc[np.linspace(0, len(sample_rows) - 1, maximum_rows, dtype=int)]
    for columns in numeric_columns:
        if not all(column in sample_rows.columns for column in columns):
            continue
        values = sample_rows.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        points.extend(values)
    if not points:
        return None
    array = np.asarray(points, dtype=float)
    lower = np.nanpercentile(array, 1.0, axis=0)
    upper = np.nanpercentile(array, 99.0, axis=0)
    span = np.maximum(upper - lower, 250.0)
    padding = np.maximum(span * 0.10, max(120.0, maximum_radius_mm * 1.15))
    lower -= padding
    upper += padding
    return {
        "x": {"min": float(lower[0]), "max": float(upper[0])},
        "y": {"min": float(lower[1]), "max": float(upper[1])},
        "z": {"min": float(lower[2]), "max": float(upper[2])},
        "sampled_point_count": int(len(array)),
        "robust_percentiles": [1.0, 99.0],
        "unit": "mm",
    }

def review_manifest(store: ProjectStore, *, session_id: str, mode: ReviewMode, artifact_id: str,
                    video_artifact_id: str | None = None, confidence_threshold: float = 0.65) -> dict[str, Any]:
    data = _build_dataset(store, session_id, mode, artifact_id, video_artifact_id, confidence_threshold)
    samples = data.samples
    region_col = _region_column(mode)
    counts = Counter(samples.get(region_col, pd.Series(dtype=str)).fillna("unknown").astype(str))
    body_analytics = _body_analytics(samples) if mode in {"body", "fusion"} else None
    body_timeline = _body_timeline(data) if mode in {"body", "fusion"} else None
    body_world_bounds = _body_world_bounds(data) if mode in {"body", "fusion"} else None
    manual_review = None
    events = dict(data.events)
    body_geometry_integrity = (
        body_geometry_integrity_from_qc(data.artifact.qc or {})
        if mode == "body" else None
    )
    if mode in {"face", "body"}:
        manual_state = (
            _load_manual_review_state(store, session_id, artifact_id)
            if mode == "face"
            else _load_body_manual_review_state(store, session_id, artifact_id)
        )
        manual_review = _manual_review_summary(data, manual_state)
        if mode == "body":
            events["pending_review"] = manual_review["pending_review_representative_frames"]
            events["reviewed"] = manual_review["reviewed_review_representative_frames"]
        else:
            events["pending_review"] = manual_review["pending_frames"]
            events["reviewed"] = manual_review["reviewed_frames"]
        events["pending_review_frames"] = manual_review["pending_frames"]
    unblurred_source = _unblurred_source_descriptor(
        store,
        session_id=session_id,
        video_artifact=data.video_artifact,
        frame_map=data.video_map,
        video_info=data.video_info,
    )
    return {
        "schema_version": "oculostudio-aoi-review-1.3",
        "mode": mode,
        "artifact": {"id": data.artifact.id, "label": data.artifact.label, "kind": data.artifact.kind},
        "video_artifact": {"id": data.video_artifact.id, "label": data.video_artifact.label},
        "video_url": f"/api/media/artifact/{session_id}/{data.video_artifact.id}",
        "video_sources": {
            "privacy": {
                "available": True,
                "url": f"/api/media/artifact/{session_id}/{data.video_artifact.id}",
                "label": "Privacy clip · blurred",
                "frame_pts_ns": list(data.video_info.get("frame_pts_ns") or []),
                "source_start_sec": 0.0,
                "source_end_sec": float(data.video_info.get("duration_seconds") or 0.0),
                "local_review_only": False,
                "export_uses_privacy_clip": True,
            },
            "unblurred": unblurred_source,
        },
        "video": data.video_info,
        "coordinate_space": data.coordinate_space,
        "sample_count": int(len(samples)),
        "region_counts": dict(counts),
        "body_analytics": body_analytics,
        "body_timeline": body_timeline,
        "body_world_bounds": body_world_bounds,
        "body_geometry_integrity": body_geometry_integrity,
        "body_review_allowed": bool(body_geometry_integrity.get("review_ready")) if body_geometry_integrity else None,
        "event_frames": events,
        "event_counts": {key: len(value) for key, value in events.items()},
        "manual_review": manual_review,
        "manual_review_role_context": data.role_context if mode in {"face", "body"} else None,
        "confidence_threshold": confidence_threshold,
        "available_layers": {
            "gaze_points": "gaze_x_px" in samples.columns and "gaze_y_px" in samples.columns,
            "gaze_trail": "gaze_x_px" in samples.columns and "gaze_y_px" in samples.columns,
            "face_regions": bool(data.face_regions_by_native),
            "participant_heads": bool(data.participant_projections_by_native),
            "face_identity": bool(data.face_identities_by_native),
            "other_person_faces": bool(data.face_identities_by_native),
            "body_world_inset": bool(data.body_regions_by_scene),
            "body_dual_orthographic": bool(data.body_regions_by_scene),
            "body_timeline": mode in {"body", "fusion"} and body_timeline is not None,
            "body_statistics": mode in {"body", "fusion"} and body_analytics is not None,
            "body_target": mode in {"body", "fusion"},
            "final_target": mode == "fusion",
            "confidence": True,
            "review_flags": True,
        },
        "limitations": [
            "Tracked participant heads are camera-projected when a participant-association artifact is available; unmatched Face Mapper detections remain other-person evidence; plausible projected-partner candidates remain explicitly ambiguous and are never silently demoted to other_person.",
            "Body primitives are shown as stable X–Z and Y–Z world-space evidence views; they are not drawn as camera-accurate image masks unless an explicit camera projection exists.",
            "The player is for QC and review. CSV/Parquet artifacts remain the analysis source of truth.",
        ],
    }


def _samples_for_frame(data: ReviewDataset, output_frame_index: int) -> tuple[pd.DataFrame, int, int]:
    row = data.video_map.iloc[output_frame_index]
    native = int(row["native_frame_index"])
    indices = data.samples_by_native.get(native, np.asarray([], dtype=int))
    return data.samples.iloc[indices].copy(), native, int(row["output_pts_ns"])


def _frame_summary_for(data: ReviewDataset, native_frame: int, group: pd.DataFrame) -> dict[str, Any]:
    region_col = _region_column(data.mode)
    confidence_col = _confidence_column(data.mode)
    entity_col = _entity_column(data.mode)
    distribution = Counter(group.get(region_col, pd.Series(dtype=str)).fillna("unknown").astype(str))
    score: dict[tuple[str | None, str], float] = defaultdict(float)
    for _, row in group.iterrows():
        entity = row.get(entity_col)
        entity = str(entity) if entity is not None and not pd.isna(entity) and str(entity) else None
        region = str(row.get(region_col) or "unknown")
        conf = pd.to_numeric(pd.Series([row.get(confidence_col)]), errors="coerce").fillna(0).iloc[0]
        score[(entity, region)] += max(0.05, float(conf))
    primary = max(score, key=score.get) if score else (None, "no_gaze")
    result = {
        "native_frame_index": native_frame,
        "gaze_sample_count": int(len(group)),
        "primary_entity_id": primary[0],
        "primary_region": primary[1],
        "distribution": dict(distribution),
        "mean_confidence": float(pd.to_numeric(group.get(confidence_col, pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if len(group) else 0.0,
        "review_required": bool(group.get("review_required", pd.Series(False, index=group.index)).fillna(False).astype(bool).any()) if len(group) else False,
    }
    if data.mode in {"body", "fusion"}:
        hit = _body_hit_mask(group)
        relation = _body_text_series(group, "body_relation", "none").str.lower()
        coarse = _body_text_series(group, "body_coarse_region", "")
        coarse = coarse.mask(coarse.eq(""), _body_text_series(group, "body_region", "unknown")).str.lower()
        exact = group.get("body_exact_hit", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        self_occlusion = group.get("self_occlusion", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        distance = pd.to_numeric(group.get("body_distance_mm", pd.Series(np.nan, index=group.index)), errors="coerce")
        result["body_metrics"] = {
            "hit_count": int(hit.sum()),
            "hit_fraction": float(hit.mean()) if len(group) else 0.0,
            "relation_distribution": dict(Counter(relation.loc[hit])),
            "coarse_region_distribution": dict(Counter(coarse.loc[hit])),
            "exact_hit_count": int((hit & exact).sum()),
            "uncertainty_hit_count": int((hit & ~exact).sum()),
            "self_occlusion_count": int(self_occlusion.sum()),
            "mean_hit_distance_mm": float(distance.loc[hit & distance.notna()].mean()) if (hit & distance.notna()).any() else None,
        }
    return result


def _face_coverage_diagnostic(data: ReviewDataset, native_frame: int) -> dict[str, Any] | None:
    if data.mode not in {"face", "fusion"}:
        return None
    regions = data.face_regions_by_native.get(native_frame, [])
    projections = data.participant_projections_by_native.get(native_frame, [])
    identities = data.face_identities_by_native.get(native_frame, [])
    audit_rows = data.facemapper_audit_by_native.get(native_frame, [])
    status_counts = Counter(str(row.get("filter_status") or "unknown") for row in audit_rows)
    reason_counts: Counter[str] = Counter()
    for row in audit_rows:
        for reason in str(row.get("rejection_reasons") or "").split(";"):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1
    partner_projection = any(
        str(row.get("relation_to_wearer") or "") == "partner"
        and bool(row.get("projection_valid"))
        and bool(row.get("in_camera_view"))
        for row in projections
    )
    partner_identity_rows = [
        row for row in identities
        if str(row.get("target_relation") or "") in {"partner", "ambiguous_partner"}
        or str(row.get("candidate_relation") or "") == "partner"
    ]
    matched_partner_rows = [
        row for row in partner_identity_rows
        if str(row.get("target_relation") or "") == "partner"
        or (str(row.get("candidate_relation") or "") == "partner"
            and str(row.get("identity_status") or "") in {
                "matched_participant", "projection_locked_participant",
                "overlap_locked_participant", "projection_cluster_inherits_participant",
            })
    ]
    matched_partner = bool(matched_partner_rows)

    def finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    best_identity = max(
        matched_partner_rows or partner_identity_rows,
        key=lambda row: finite_float(row.get("identity_confidence")) or 0.0,
        default=None,
    )
    identity_confidence = finite_float((best_identity or {}).get("identity_confidence"))
    projected_box_iou = finite_float((best_identity or {}).get("projected_box_iou"))
    normalized_center_error = finite_float((best_identity or {}).get("normalized_center_error"))
    overlap_value = (best_identity or {}).get("bbox_overlap_present")
    if overlap_value is None:
        bbox_overlap_present = bool(projected_box_iou is not None and projected_box_iou > 0.0)
    else:
        bbox_overlap_present = bool(overlap_value)
    weak_geometric_agreement = bool(
        best_identity
        and (
            not bbox_overlap_present
            or (identity_confidence is not None and identity_confidence < 0.55)
        )
    )

    face_tolerance_ms = float(data.artifact.parameters.get("face_tolerance_ms") or 0.0)
    recommendation = ""
    if regions:
        if matched_partner and weak_geometric_agreement:
            code = "partner_face_covered_low_geometric_agreement"
            severity = "warning"
            title = "FaceMapper geometry is present, but its alignment with the projected partner head is weak."
            detail = (
                "The source FaceMapper box or landmarks may be tight, shifted or partially clipped even though the identity match passed. "
                "OculoStudio preserves the accepted source geometry instead of silently expanding or moving it."
            )
            recommendation = "Inspect the unblurred frame and FaceMapper QC. If the source box is visibly wrong, repair or regenerate FaceMapper rather than accepting the fine AOI blindly."
        elif matched_partner:
            code = "partner_face_covered"
            severity = "good"
            title = "Partner face is covered by an accepted FaceMapper detection."
            detail = "Fine facial AOIs use the exact accepted FaceMapper box and landmarks for this Native frame."
            recommendation = "No coverage problem is detected for this frame; use manual review only if the source landmarks are visibly misplaced."
        elif partner_projection:
            code = "face_detected_partner_identity_unresolved"
            severity = "warning"
            title = "A face was detected, but it was not securely assigned to the projected partner."
            detail = "The green FaceMapper box either does not overlap the projected partner box, or more than one distinct green face overlaps it and identity is ambiguous."
            recommendation = "Check whether exactly one green face box overlaps the R02 projection. Centre offset and box-size mismatch alone are not rejection reasons."
        else:
            code = "face_detected_without_partner_projection"
            severity = "warning"
            title = "FaceMapper geometry exists, but no valid in-view partner projection supports it."
            detail = "Check ETM calibration, C3D head markers, wearer-role assignment and participant association for this frame."
            recommendation = "Repair the 3D participant projection or explicitly review the entity; the 2D face alone cannot prove that it is the partner."
    elif audit_rows:
        if status_counts.get("accepted", 0):
            code = "accepted_detection_not_mapped_to_aoi"
            severity = "warning"
            title = "FaceMapper accepted a detection, but no Face AOI geometry reached this review frame."
            detail = (
                f"The most likely cause is the Face AOI timestamp gate ({face_tolerance_ms:g} ms), "
                "a frame-index mismatch, or partner-only filtering after identity association."
            )
            recommendation = "Compare Native and FaceMapper timestamps; increase tolerance only if the streams are known to share the same physical start and the measured offset justifies it."
        elif status_counts.get("review", 0):
            code = "analysis_strict_filtered_review_candidate"
            severity = "warning"
            title = "FaceMapper found a candidate, but analysis-strict filtering excluded it."
            detail = "The privacy-safe layer may retain this candidate, while Face AOI analysis uses accepted detections only."
            recommendation = "Open FaceMapper QC and inspect the listed geometry or temporal reason. Accepting the privacy blur does not automatically make the detection valid for scientific AOI analysis."
        else:
            code = "facemapper_detection_rejected"
            severity = "warning"
            title = "FaceMapper exported a detection, but QC rejected it."
            detail = "The source box or five landmarks failed geometry or temporal-consistency checks."
            recommendation = "Inspect the rejection reasons below and regenerate or manually repair the FaceMapper source when the visible face is valid."
    else:
        code = "facemapper_no_detection"
        severity = "warning"
        title = "FaceMapper exported no face detection for this Native frame."
        detail = (
            "Common causes are motion blur, profile view, occlusion, a small or edge-clipped face, "
            "poor illumination, or a brief detector miss. OculoStudio does not invent fine landmarks for missing detections."
        )
        recommendation = "Use the unblurred review toggle to verify the face. If it is clearly visible, rerun or repair FaceMapper; manual gaze labels can be saved, but missing fine geometry should not be fabricated automatically."
    if partner_projection and not regions:
        detail += " A valid 3D partner-head projection is present, so this is primarily a 2D FaceMapper coverage issue rather than loss of the tracked partner."
    reason_details = [
        {
            "reason": reason,
            "count": int(count),
            "explanation": _FACE_FILTER_REASON_EXPLANATIONS.get(
                reason, reason.replace("_", " ").capitalize() + "."
            ),
        }
        for reason, count in reason_counts.most_common()
    ]
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "detail": detail,
        "recommendation": recommendation,
        "face_region_count": int(len(regions)),
        "identity_row_count": int(len(identities)),
        "partner_projection_valid": bool(partner_projection),
        "matched_partner": bool(matched_partner),
        "identity_confidence": identity_confidence,
        "projected_box_iou": projected_box_iou,
        "bbox_overlap_present": bbox_overlap_present,
        "normalized_center_error": normalized_center_error,
        "weak_geometric_agreement": weak_geometric_agreement,
        "filter_status_counts": dict(status_counts),
        "filter_reason_counts": dict(reason_counts),
        "filter_reason_details": reason_details,
        "face_tolerance_ms": face_tolerance_ms,
        "geometry_policy": "exact_source_box_and_landmarks_no_interpolation",
    }


def _review_frame_from_dataset(
    data: ReviewDataset,
    *,
    store: ProjectStore,
    session_id: str,
    artifact_id: str,
    output_frame_index: int,
    sample_index: int = 0,
    trail_frames: int = 6,
    confidence_threshold: float = 0.65,
    manual_state: dict[str, Any] | None = None,
    manual_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = data.mode
    frame_count = int(data.video_info["frame_count"])
    output_frame_index = max(0, min(frame_count - 1, int(output_frame_index)))
    group, native, pts_ns = _samples_for_frame(data, output_frame_index)
    sample_index = max(0, min(max(0, len(group) - 1), int(sample_index)))
    columns = [
        "gaze_sample_uid", "source_gaze_row_index", "timestamp_ns", "clip_time_ns",
        "native_frame_index", "scene_frame_index", "gaze_x_px", "gaze_y_px", "worn",
        "blink_id", "fixation_id", "fm_gaze_on_face", "face_entity_id", "face_track_id",
        "face_region", "face_social_region", "face_aoi_confidence", "green_face_exact_hit",
        "green_face_edge_tolerance_hit", "face_gate_hit_type", "face_edge_distance_px",
        "face_edge_tolerance_px", "face_edge_candidate_count", "face_edge_ambiguous",
        "face_edge_tolerance_policy", "body_entity_id", "body_relation", "body_raw_region",
        "body_coarse_region", "body_region", "body_social_region", "body_confidence",
        "body_distance_mm", "body_exact_hit", "body_angular_miss_deg", "body_hit_x_mm",
        "body_hit_y_mm", "body_hit_z_mm", "ray_valid", "ray_origin_x_mm", "ray_origin_y_mm",
        "ray_origin_z_mm", "ray_direction_x", "ray_direction_y", "ray_direction_z",
        "final_entity_id", "final_target_domain", "final_target_region", "final_confidence",
        "fusion_status", "evidence_sources", "assignment_status", "face_assignment_status",
        "body_assignment_status", "raw_face_entity_id", "raw_face_region", "identity_status",
        "identity_confidence", "target_relation", "candidate_entity_id", "candidate_role", "candidate_relation",
        "include_in_primary_analysis", "participant_association_artifact_id", "candidate_hit_count", "self_occlusion", "occlusion_status",
        "first_hit_entity_id", "first_hit_relation", "first_hit_raw_region", "first_hit_coarse_region",
        "first_hit_social_region", "first_hit_distance_mm", "second_hit_entity_id", "second_hit_relation",
        "second_hit_raw_region", "second_hit_coarse_region", "second_hit_social_region", "second_hit_distance_mm",
        "nearest_partner_hit_entity_id", "nearest_partner_hit_social_region", "nearest_partner_hit_distance_mm",
        "first_hit_primitive_confidence", "second_hit_angular_miss_deg", "second_hit_primitive_confidence",
        "review_required", "review_required_before_auto_resolution", "review_reason_before_auto_resolution",
        "auto_review_resolved", "auto_review_resolution_source", "auto_review_confidence", "auto_review_policy",
    ]
    automatic_summary = _frame_summary_for(data, native, group)
    frame_role_context = (
        data.role_context
        if mode == "body"
        else _manual_review_frame_role_context(
            data.role_context, data.participant_projections_by_native.get(native, [])
        )
    )
    effective_group = group.copy()
    effective_previews: dict[str, dict[str, Any]] = {}
    manual_summary = None
    decisions: dict[str, Any] = {}
    if mode in {"face", "body"}:
        if manual_state is None:
            manual_state = (
                _load_manual_review_state(store, session_id, artifact_id)
                if mode == "face"
                else _load_body_manual_review_state(store, session_id, artifact_id)
            )
        decisions = manual_state.get("decisions") or {}
        if mode == "face":
            preview_fields = [
                "face_entity_id", "face_region", "face_social_region", "target_relation",
                "identity_status", "assignment_status", "identity_confidence",
                "face_aoi_confidence", "include_in_primary_analysis", "review_required",
            ]
        else:
            preview_fields = [
                "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region",
                "body_region", "body_social_region", "body_confidence", "body_exact_hit",
                "body_distance_mm", "body_hit_x_mm", "body_hit_y_mm", "body_hit_z_mm",
                "body_angular_miss_deg", "body_geometry_backend", "self_occlusion",
                "occlusion_status", "assignment_status", "review_required",
            ]
        output_granularity = str((data.artifact.parameters or {}).get("output_granularity") or "coarse_social_regions")
        for row_index, row in group.iterrows():
            uid = _sample_uid(row)
            decision = decisions.get(uid)
            if mode == "face":
                effective_row, preview = _effective_face_review_row(
                    row, decision, role_context=frame_role_context
                )
            else:
                effective_row, preview = _effective_body_review_row(
                    row, decision, role_context=frame_role_context, output_granularity=output_granularity
                )
            effective_previews[uid] = preview
            for field in preview_fields:
                effective_group.at[row_index, field] = effective_row.get(field)
        if manual_summary is None:
            manual_summary = _manual_review_summary(data, manual_state)

    samples = [_row_dict(row, columns) for _, row in group.iterrows()]
    if mode in {"face", "body"}:
        output_granularity = str((data.artifact.parameters or {}).get("output_granularity") or "coarse_social_regions")
        for sample in samples:
            uid = _sample_uid(sample)
            decision = decisions.get(uid)
            if mode == "face":
                preview = effective_previews.get(uid) or _effective_face_review_row(
                    sample, decision, role_context=frame_role_context
                )[1]
            else:
                preview = effective_previews.get(uid) or _effective_body_review_row(
                    sample, decision, role_context=frame_role_context, output_granularity=output_granularity
                )[1]
            sample["manual_review_decision"] = decision
            sample["manual_review_resolved"] = bool(preview.get("manual_review_resolved"))
            for key, value in preview.items():
                sample[f"effective_{key}"] = value
    selected = samples[sample_index] if samples else None

    summary_uses_manual_review = bool(
        mode in {"face", "body"}
        and any(bool(preview.get("manual_review_resolved")) for preview in effective_previews.values())
    )
    effective_summary = (
        _frame_summary_for(data, native, effective_group)
        if mode in {"face", "body"}
        else automatic_summary
    )

    start_frame = max(0, output_frame_index - max(0, int(trail_frames)))
    trail_native = data.video_map.iloc[start_frame:output_frame_index + 1]["native_frame_index"].astype(int).tolist()
    trail_rows = []
    for frame_no, native_index in enumerate(trail_native, start=start_frame):
        indices = data.samples_by_native.get(native_index, np.asarray([], dtype=int))
        for idx in indices:
            row = data.samples.iloc[int(idx)]
            x = _jsonable(row.get("gaze_x_px"))
            y = _jsonable(row.get("gaze_y_px"))
            if x is not None and y is not None:
                trail_rows.append({
                    "output_frame_index": frame_no,
                    "gaze_x_px": x,
                    "gaze_y_px": y,
                    "region": _jsonable(row.get(_region_column(mode))),
                    "confidence": _jsonable(row.get(_confidence_column(mode))),
                })

    scene_frames = sorted({
        int(value) for value in pd.to_numeric(group.get("scene_frame_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
    })
    body_regions = []
    for scene_frame in scene_frames[:4]:
        body_regions.extend(data.body_regions_by_scene.get(scene_frame, []))
    return {
        "mode": mode,
        "output_frame_index": output_frame_index,
        "output_pts_ns": pts_ns,
        "time_seconds": pts_ns / 1e9,
        "native_frame_index": native,
        "native_capture_utc_ns": _jsonable(data.video_map.iloc[output_frame_index].get("native_capture_utc_ns")),
        "sample_index": sample_index,
        "sample_count": len(samples),
        "selected_sample": selected,
        "samples": samples,
        "summary": effective_summary,
        "automatic_summary": automatic_summary if mode in {"face", "body"} else None,
        "summary_uses_manual_review": summary_uses_manual_review,
        "resolved_decision_count_in_frame": int(sum(
            1 for preview in effective_previews.values() if preview.get("manual_review_resolved")
        )),
        "manual_review": manual_summary,
        "manual_review_role_context": frame_role_context if mode in {"face", "body"} else None,
        "manual_review_decision": selected.get("manual_review_decision") if selected else None,
        "face_regions": data.face_regions_by_native.get(native, []),
        "participant_projections": data.participant_projections_by_native.get(native, []),
        "face_identities": data.face_identities_by_native.get(native, []),
        "face_coverage_diagnostic": _face_coverage_diagnostic(data, native),
        "body_regions": body_regions,
        "trail": trail_rows[-120:],
        "confidence_threshold": confidence_threshold,
    }



def _compact_review_frame_from_dataset(
    data: ReviewDataset,
    *,
    store: ProjectStore,
    session_id: str,
    artifact_id: str,
    output_frame_index: int,
    trail_frames: int = 2,
    confidence_threshold: float = 0.65,
    manual_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimal payload needed for real-time AOI playback.

    The paused inspector uses :func:`_review_frame_from_dataset`.  Playback does
    not need the global manual-review summary, face coverage diagnostics, or the
    large inspector audit structure on every presented video frame.  Keeping
    this path deliberately small prevents a 30 fps video from being throttled by
    repeated DOM-sized JSON payloads while preserving the gaze ray, body
    primitives, face geometry and final target overlay.
    """
    mode = data.mode
    frame_count = int(data.video_info["frame_count"])
    output_frame_index = max(0, min(frame_count - 1, int(output_frame_index)))
    group, native, pts_ns = _samples_for_frame(data, output_frame_index)

    # Use the high-rate gaze sample nearest the actual presented-video PTS for
    # the highlighted ray.  The old playback path always chose sample zero (or
    # the first pending review sample), which made the 3D ray appear frozen or
    # jump irregularly even while the video advanced.
    selected_position = 0
    if not group.empty:
        time_values = pd.to_numeric(
            group.get("clip_time_ns", group.get("timestamp_ns", pd.Series(np.nan, index=group.index))),
            errors="coerce",
        )
        valid = time_values.notna()
        if valid.any():
            selected_label = (time_values.loc[valid] - int(pts_ns)).abs().idxmin()
            selected_position = int(group.index.get_loc(selected_label))
    selected_position = max(0, min(max(0, len(group) - 1), selected_position))

    columns = [
        "gaze_sample_uid", "source_gaze_row_index", "timestamp_ns", "clip_time_ns",
        "native_frame_index", "scene_frame_index", "gaze_x_px", "gaze_y_px", "worn",
        "blink_id", "fixation_id", "face_entity_id", "face_region", "face_social_region",
        "face_aoi_confidence", "target_relation", "identity_status", "identity_confidence",
        "candidate_entity_id", "candidate_role", "candidate_relation",
        "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region",
        "body_region", "body_social_region", "body_confidence", "body_distance_mm",
        "body_exact_hit", "body_angular_miss_deg", "body_hit_x_mm", "body_hit_y_mm",
        "body_hit_z_mm", "ray_valid", "ray_origin_x_mm", "ray_origin_y_mm",
        "ray_origin_z_mm", "ray_direction_x", "ray_direction_y", "ray_direction_z",
        "final_entity_id", "final_target_domain", "final_target_region", "final_confidence",
        "fusion_status", "assignment_status", "candidate_hit_count", "self_occlusion",
        "occlusion_status", "first_hit_entity_id", "first_hit_relation",
        "first_hit_raw_region", "first_hit_coarse_region", "first_hit_social_region",
        "first_hit_distance_mm", "second_hit_entity_id", "second_hit_relation",
        "second_hit_raw_region", "second_hit_coarse_region", "second_hit_social_region",
        "second_hit_distance_mm", "nearest_partner_hit_entity_id",
        "nearest_partner_hit_social_region", "nearest_partner_hit_distance_mm",
        "first_hit_primitive_confidence", "second_hit_angular_miss_deg",
        "second_hit_primitive_confidence", "review_required", "auto_review_resolved",
        "auto_review_resolution_source", "auto_review_confidence", "auto_review_policy",
    ]
    samples = [_row_dict(row, columns) for _, row in group.iterrows()]

    # Applying already-saved decisions is inexpensive at the handful of gaze
    # samples in one video frame and keeps playback visually consistent with the
    # paused review state.  We intentionally skip the global progress summary.
    frame_role_context = (
        data.role_context
        if mode == "body"
        else _manual_review_frame_role_context(
            data.role_context, data.participant_projections_by_native.get(native, [])
        )
    )
    decisions: dict[str, Any] = {}
    if mode in {"face", "body"}:
        if manual_state is None:
            manual_state = (
                _load_manual_review_state(store, session_id, artifact_id)
                if mode == "face"
                else _load_body_manual_review_state(store, session_id, artifact_id)
            )
        decisions = manual_state.get("decisions") or {}
        output_granularity = str((data.artifact.parameters or {}).get("output_granularity") or "coarse_social_regions")
        for sample in samples:
            uid = _sample_uid(sample)
            decision = decisions.get(uid)
            if mode == "face":
                _effective_row, preview = _effective_face_review_row(
                    sample, decision, role_context=frame_role_context
                )
            else:
                _effective_row, preview = _effective_body_review_row(
                    sample, decision, role_context=frame_role_context,
                    output_granularity=output_granularity,
                )
            sample["manual_review_decision"] = decision
            sample["manual_review_resolved"] = bool(preview.get("manual_review_resolved"))
            for key, value in preview.items():
                sample[f"effective_{key}"] = value

    selected = samples[selected_position] if samples else None
    playback_sample_keys = {
        "gaze_sample_uid", "gaze_x_px", "gaze_y_px",
        "face_entity_id", "face_region", "face_social_region", "face_aoi_confidence",
        "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region",
        "body_region", "body_social_region", "body_confidence",
        "final_entity_id", "final_target_region", "final_confidence",
        "review_required", "manual_review_decision", "manual_review_resolved",
    }
    playback_samples = []
    for sample in samples:
        compact = {key: value for key, value in sample.items() if key in playback_sample_keys or key.startswith("effective_")}
        playback_samples.append(compact)
    summary = _frame_summary_for(data, native, group)

    start_frame = max(0, output_frame_index - max(0, int(trail_frames)))
    trail_native = data.video_map.iloc[start_frame:output_frame_index + 1]["native_frame_index"].astype(int).tolist()
    trail_rows: list[dict[str, Any]] = []
    for frame_no, native_index in enumerate(trail_native, start=start_frame):
        indices = data.samples_by_native.get(native_index, np.asarray([], dtype=int))
        for idx in indices:
            row = data.samples.iloc[int(idx)]
            x = _jsonable(row.get("gaze_x_px"))
            y = _jsonable(row.get("gaze_y_px"))
            if x is not None and y is not None:
                trail_rows.append({
                    "output_frame_index": frame_no,
                    "gaze_x_px": x,
                    "gaze_y_px": y,
                    "region": _jsonable(row.get(_region_column(mode))),
                    "confidence": _jsonable(row.get(_confidence_column(mode))),
                })

    scene_frames = sorted({
        int(value)
        for value in pd.to_numeric(
            group.get("scene_frame_index", pd.Series(dtype=float)), errors="coerce"
        ).dropna().astype(int)
    })
    body_regions: list[dict[str, Any]] = []
    body_region_keys = {
        "entity_id", "entity_relation", "relation", "region", "coarse_region",
        "geometry_type", "geometry_json", "geometry_provenance", "confidence",
    }
    for scene_frame in scene_frames[:4]:
        for row in data.body_regions_by_scene.get(scene_frame, []):
            body_regions.append({key: value for key, value in row.items() if key in body_region_keys})

    face_region_keys = {
        "region", "face_social_region", "geometry_type", "geometry_json",
        "include_in_primary_analysis",
    }
    projection_keys = {
        "projection_valid", "in_camera_view", "projected_bbox_x1", "projected_bbox_y1",
        "projected_bbox_x2", "projected_bbox_y2", "relation_to_wearer", "role_id", "entity_id",
    }
    identity_keys = {
        "target_relation", "candidate_relation", "face_box_x1", "face_box_y1",
        "face_box_x2", "face_box_y2", "identity_status", "matched_entity_id",
        "candidate_entity_id", "identity_confidence", "review_required",
    }
    compact_face_regions = [
        {key: value for key, value in row.items() if key in face_region_keys}
        for row in data.face_regions_by_native.get(native, [])
    ]
    compact_projections = [
        {key: value for key, value in row.items() if key in projection_keys}
        for row in data.participant_projections_by_native.get(native, [])
    ]
    compact_identities = [
        {key: value for key, value in row.items() if key in identity_keys}
        for row in data.face_identities_by_native.get(native, [])
    ]

    return {
        "schema_version": "oculostudio-aoi-review-playback-frame-1.0",
        "payload_profile": "compact_playback",
        "mode": mode,
        "output_frame_index": output_frame_index,
        "output_pts_ns": pts_ns,
        "time_seconds": pts_ns / 1e9,
        "native_frame_index": native,
        "sample_index": selected_position,
        "sample_count": len(samples),
        "selected_sample": selected,
        "samples": playback_samples,
        "summary": summary,
        "automatic_summary": summary if mode in {"face", "body"} else None,
        "summary_uses_manual_review": False,
        "resolved_decision_count_in_frame": int(sum(
            bool(sample.get("manual_review_resolved")) for sample in samples
        )),
        "manual_review": None,
        "manual_review_role_context": frame_role_context if mode in {"face", "body"} else None,
        "manual_review_decision": selected.get("manual_review_decision") if selected else None,
        "face_regions": compact_face_regions,
        "participant_projections": compact_projections,
        "face_identities": compact_identities,
        "face_coverage_diagnostic": None,
        "body_regions": body_regions,
        "trail": trail_rows[-48:],
        "confidence_threshold": confidence_threshold,
    }

def review_frame(store: ProjectStore, *, session_id: str, mode: ReviewMode, artifact_id: str,
                 video_artifact_id: str | None = None, output_frame_index: int,
                 sample_index: int = 0, trail_frames: int = 6,
                 confidence_threshold: float = 0.65) -> dict[str, Any]:
    data = _build_dataset(store, session_id, mode, artifact_id, video_artifact_id, confidence_threshold)
    manual_state = None
    manual_summary = None
    if mode in {"face", "body"}:
        manual_state = (
            _load_manual_review_state(store, session_id, artifact_id)
            if mode == "face"
            else _load_body_manual_review_state(store, session_id, artifact_id)
        )
        manual_summary = _manual_review_summary(data, manual_state)
    return _review_frame_from_dataset(
        data, store=store, session_id=session_id, artifact_id=artifact_id,
        output_frame_index=output_frame_index, sample_index=sample_index,
        trail_frames=trail_frames, confidence_threshold=confidence_threshold,
        manual_state=manual_state, manual_summary=manual_summary,
    )


def review_frame_window(
    store: ProjectStore,
    *,
    session_id: str,
    mode: ReviewMode,
    artifact_id: str,
    video_artifact_id: str | None = None,
    start_frame_index: int,
    frame_count: int = 24,
    trail_frames: int = 6,
    confidence_threshold: float = 0.65,
    compact_playback: bool = False,
) -> dict[str, Any]:
    """Return a short consecutive frame window for frame-locked playback.

    The dataset and manual-review state are loaded once for the whole window.
    This prevents one HTTP request per presented video frame and avoids stale
    overlays when the UI is playing at 25/30/60 fps.
    """
    data = _build_dataset(store, session_id, mode, artifact_id, video_artifact_id, confidence_threshold)
    total = int(data.video_info["frame_count"])
    start = max(0, min(max(0, total - 1), int(start_frame_index)))
    count = max(1, min(120, int(frame_count)))
    end = min(total, start + count)
    manual_state = None
    manual_summary = None
    if mode in {"face", "body"}:
        manual_state = (
            _load_manual_review_state(store, session_id, artifact_id)
            if mode == "face"
            else _load_body_manual_review_state(store, session_id, artifact_id)
        )
        # Compact playback packets never render the progress inspector.  On a
        # long clip, rebuilding every fixation/review episode for each rolling
        # window can cost more than the 30 fps overlay itself.
        if not compact_playback:
            manual_summary = _manual_review_summary(data, manual_state)
    frames: list[dict[str, Any]] = []
    for frame_index in range(start, end):
        if compact_playback:
            payload = _compact_review_frame_from_dataset(
                data, store=store, session_id=session_id, artifact_id=artifact_id,
                output_frame_index=frame_index,
                trail_frames=trail_frames, confidence_threshold=confidence_threshold,
                manual_state=manual_state,
            )
        else:
            payload = _review_frame_from_dataset(
                data, store=store, session_id=session_id, artifact_id=artifact_id,
                output_frame_index=frame_index, sample_index=0,
                trail_frames=trail_frames, confidence_threshold=confidence_threshold,
                manual_state=manual_state, manual_summary=manual_summary,
            )
            # The manifest already carries the global progress summary. Repeating it
            # in every buffered frame wastes bandwidth on long sessions.
            payload["manual_review"] = None
        frames.append(payload)
    return {
        "schema_version": "oculostudio-aoi-review-window-1.1",
        "payload_profile": "compact_playback" if compact_playback else "full_review",
        "mode": mode,
        "start_frame_index": start,
        "end_frame_index_exclusive": end,
        "frame_count": len(frames),
        "frames": frames,
    }


def _draw_face_regions(frame: np.ndarray, rows: list[dict[str, Any]]) -> None:
    colors = {
        "left_eye": (50, 245, 230), "right_eye": (80, 220, 120), "nose": (80, 190, 255),
        "mouth": (220, 110, 255), "other_face": (180, 190, 80), "face": (180, 190, 80),
    }
    for row in rows:
        if "include_in_primary_analysis" in row and row.get("include_in_primary_analysis") is False:
            continue
        try:
            geometry = row.get("geometry_json")
            geometry = json.loads(geometry) if isinstance(geometry, str) else geometry
            color = colors.get(str(row.get("region")), (50, 245, 230))
            geometry_type = str(row.get("geometry_type") or "ellipse")
            if geometry_type == "polygon" or (isinstance(geometry, dict) and geometry.get("points")):
                points = np.asarray(geometry.get("points") or [], dtype=float)
                if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                    continue
                polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
            else:
                center = (int(round(geometry["center_x"])), int(round(geometry["center_y"])))
                axes = (max(1, int(round(geometry["radius_x"]))), max(1, int(round(geometry["radius_y"]))))
                cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
        except Exception:
            continue


def _draw_participant_association(frame: np.ndarray, payload: dict[str, Any]) -> None:
    for row in payload.get("participant_projections") or []:
        try:
            if not bool(row.get("projection_valid")) or not bool(row.get("in_camera_view")):
                continue
            x1, y1, x2, y2 = [int(round(float(row[key]))) for key in (
                "projected_bbox_x1", "projected_bbox_y1", "projected_bbox_x2", "projected_bbox_y2"
            )]
            relation = str(row.get("relation_to_wearer") or "tracked")
            color = (65, 240, 255) if relation == "partner" else (255, 190, 70)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"projected {row.get('role_id') or row.get('entity_id') or 'participant'} · {relation}"
            cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2.LINE_AA)
        except Exception:
            continue
    for row in payload.get("face_identities") or []:
        try:
            x1, y1, x2, y2 = [int(round(float(row[key]))) for key in (
                "face_box_x1", "face_box_y1", "face_box_x2", "face_box_y2"
            )]
            relation = str(row.get("target_relation") or "other_person")
            status = str(row.get("identity_status") or "unknown")
            matched_statuses = {
                "matched_participant", "duplicate_inherits_matched_participant",
                "temporal_bridge_matched_participant", "temporal_edge_matched_participant",
            }
            if relation == "partner" and status in matched_statuses:
                color = (90, 240, 90)
            elif relation == "self" and status in matched_statuses:
                color = (255, 210, 65)
            elif relation in {"ambiguous_partner", "partner_candidate"} or row.get("candidate_relation") == "partner":
                color = (55, 210, 245)
            elif "ambiguous" in status or bool(row.get("review_required")):
                color = (55, 70, 230)
            else:
                color = (40, 155, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            entity = row.get("matched_entity_id") or row.get("candidate_entity_id") or "other_person"
            confidence = row.get("identity_confidence")
            relation_label = "partner_candidate" if relation in {"ambiguous_partner", "partner_candidate"} else relation
            text = f"{entity} · {relation_label}"
            if confidence is not None and not pd.isna(confidence):
                text += f" · {float(confidence):.2f}"
            cv2.putText(frame, text, (x1, min(frame.shape[0]-8, y2 + 17)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        except Exception:
            continue


def _draw_body_world_inset(frame: np.ndarray, payload: dict[str, Any]) -> None:
    rows = payload.get("body_regions") or []
    sample = payload.get("selected_sample") or {}
    parsed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    points: list[np.ndarray] = []
    maximum_radius_mm = 0.0
    for row in rows:
        try:
            geometry = row.get("geometry_json")
            geometry = json.loads(geometry) if isinstance(geometry, str) else geometry
            if not isinstance(geometry, dict):
                continue
            parsed.append((row, geometry))
            try:
                radius_value = float(geometry.get("radius_mm") or 0.0)
                if np.isfinite(radius_value):
                    maximum_radius_mm = max(maximum_radius_mm, radius_value)
            except Exception:
                pass
            for key in ("center_mm", "start_mm", "end_mm"):
                value = geometry.get(key)
                if value is not None:
                    array = np.asarray(value, dtype=float)
                    if array.shape == (3,) and np.isfinite(array).all():
                        points.append(array)
        except Exception:
            continue

    # Old Body AOI artifacts may contain valid shoulder/hip limb primitives but
    # no central torso row because the original C3D lacked the required chest
    # marker pair.  Draw a clearly provenance-marked visual torso from those
    # observed anchors; new v0.21.49+ artifacts also create the same geometry in
    # the annotation backend so torso ray intersections are scientifically usable.
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row, geometry in parsed:
        entity = str(row.get("entity_id") or "body")
        relation = str(row.get("entity_relation") or row.get("relation") or "other").lower()
        key = (entity, relation)
        group = groups.setdefault(key, {
            "entity": entity, "relation": relation, "has_torso": False,
            "shoulders": [], "hips": [], "necks": [], "heads": [], "confidence": [],
        })
        raw = str(row.get("region") or "").lower()
        if raw == "torso":
            group["has_torso"] = True
        def add(target: str, value: Any) -> None:
            try:
                point = np.asarray(value, dtype=float)
                if point.shape == (3,) and np.isfinite(point).all():
                    group[target].append(point)
            except Exception:
                pass
        if raw in {"left_upper_arm", "right_upper_arm"}:
            add("shoulders", geometry.get("start_mm"))
        if raw in {"left_thigh", "right_thigh"}:
            add("hips", geometry.get("start_mm"))
        if raw == "neck":
            add("necks", geometry.get("end_mm"))
            add("necks", geometry.get("start_mm"))
        if raw == "head":
            add("heads", geometry.get("center_mm"))
        try:
            value = float(row.get("confidence"))
            if np.isfinite(value):
                group["confidence"].append(value)
        except Exception:
            pass
    for group in groups.values():
        if group["has_torso"] or not group["hips"]:
            continue
        lower = np.mean(np.stack(group["hips"]), axis=0)
        if group["shoulders"]:
            upper = np.mean(np.stack(group["shoulders"]), axis=0)
        elif group["necks"]:
            upper = np.mean(np.stack(group["necks"]), axis=0)
        elif group["heads"]:
            head_point = np.mean(np.stack(group["heads"]), axis=0)
            upper = lower + 0.64 * (head_point - lower)
        else:
            continue
        length = float(np.linalg.norm(upper - lower))
        if not (60.0 <= length <= 1200.0):
            continue
        shoulder_width = float(np.linalg.norm(group["shoulders"][0] - group["shoulders"][1])) if len(group["shoulders"]) >= 2 else 0.0
        hip_width = float(np.linalg.norm(group["hips"][0] - group["hips"][1])) if len(group["hips"]) >= 2 else 0.0
        inferred_width = max(shoulder_width, hip_width)
        radius = float(np.clip(0.46 * inferred_width if inferred_width else 185.0, 135.0, 220.0))
        confidence = float(np.clip(np.mean(group["confidence"]) * 0.72 if group["confidence"] else 0.55, 0.35, 0.72))
        synthetic_row = {
            "entity_id": group["entity"],
            "entity_relation": group["relation"],
            "relation": group["relation"],
            "region": "torso",
            "coarse_region": "torso",
            "geometry_provenance": "visual_fallback",
            "geometry_is_derived": True,
            "confidence": confidence,
        }
        synthetic_geometry = {
            "start_mm": upper.tolist(),
            "end_mm": lower.tolist(),
            "radius_mm": radius,
            "visual_fallback": True,
        }
        parsed.append((synthetic_row, synthetic_geometry))
        points.extend([upper, lower])
        maximum_radius_mm = max(maximum_radius_mm, radius)

    origin = None
    hit = None
    try:
        if sample.get("ray_origin_x_mm") is not None:
            origin = np.asarray([
                sample.get("ray_origin_x_mm"), sample.get("ray_origin_y_mm"), sample.get("ray_origin_z_mm")
            ], dtype=float)
            if np.isfinite(origin).all():
                points.append(origin)
            else:
                origin = None
        if sample.get("body_hit_x_mm") is not None:
            hit = np.asarray([
                sample.get("body_hit_x_mm"), sample.get("body_hit_y_mm"), sample.get("body_hit_z_mm")
            ], dtype=float)
            if np.isfinite(hit).all():
                points.append(hit)
            else:
                hit = None
    except Exception:
        origin = hit = None
    if not points:
        return

    inset_w = min(300, max(220, int(frame.shape[1] * 0.24)))
    inset_h = min(190, max(140, int(frame.shape[0] * 0.25)))
    x0 = frame.shape[1] - inset_w - 14
    y0 = frame.shape[0] - inset_h - 14
    x1, y1 = x0 + inset_w, y0 + inset_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (22, 27, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (105, 116, 82), 1)
    cv2.putText(frame, "3D BODY EVIDENCE - WORLD X/Z", (x0 + 10, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 232, 210), 1, cv2.LINE_AA)
    pad = 22
    xs = [float(p[0]) for p in points]
    zs = [float(p[2]) for p in points]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    if max_x - min_x < 1.0:
        min_x -= 1.0
        max_x += 1.0
    if max_z - min_z < 1.0:
        min_z -= 1.0
        max_z += 1.0
    # Account for the physical radius of spheres/capsules so a visible torso
    # surface cannot be clipped by bounds that only contain primitive axes.
    primitive_padding = max(80.0, maximum_radius_mm * 1.15)
    min_x -= primitive_padding
    max_x += primitive_padding
    min_z -= primitive_padding
    max_z += primitive_padding
    plot_x0, plot_y0 = x0 + pad, y0 + 30
    plot_w, plot_h = inset_w - 2 * pad, inset_h - 44
    scale = min(plot_w / (max_x - min_x), plot_h / (max_z - min_z))

    def map_point(point: np.ndarray | list[float]) -> tuple[int, int]:
        point = np.asarray(point, dtype=float)
        return (
            int(round(plot_x0 + (point[0] - min_x) * scale)),
            int(round(plot_y0 + plot_h - (point[2] - min_z) * scale)),
        )

    for i in range(5):
        y = int(round(plot_y0 + i * plot_h / 4))
        cv2.line(frame, (plot_x0, y), (plot_x0 + plot_w, y), (52, 60, 45), 1)
    selected_entity = str(sample.get("body_entity_id") or "")
    selected_raw = str(sample.get("body_raw_region") or sample.get("body_region") or "")
    selected_coarse = str(sample.get("body_coarse_region") or sample.get("body_region") or "")
    # Surface pass: render head and torso/pelvis with their true projected
    # radius.  This keeps the two people readable instead of reducing the
    # torso capsule to the same one-pixel centre-line as a limb.
    surface = frame.copy()
    surface_drawn = False
    for row, geometry in parsed:
        row_raw = str(row.get("region") or "")
        row_coarse = str(row.get("coarse_region") or row_raw)
        if row_coarse not in {"head", "torso"}:
            continue
        color = (94, 211, 244) if row_coarse == "head" else (240, 201, 76)
        radius = max(2, min(54, int(round(float(geometry.get("radius_mm") or 20.0) * scale))))
        if geometry.get("center_mm") is not None:
            cv2.circle(surface, map_point(geometry["center_mm"]), radius, color, -1, cv2.LINE_AA)
            surface_drawn = True
        elif geometry.get("start_mm") is not None and geometry.get("end_mm") is not None:
            cv2.line(
                surface,
                map_point(geometry["start_mm"]),
                map_point(geometry["end_mm"]),
                color,
                max(4, 2 * radius),
                cv2.LINE_AA,
            )
            surface_drawn = True
    if surface_drawn:
        cv2.addWeighted(surface, 0.28, frame, 0.72, 0, frame)

    # Evidence pass: retain crisp primitive axes/outlines and selected-hit
    # emphasis above the translucent body surfaces.
    for row, geometry in parsed:
        row_entity = str(row.get("entity_id") or "")
        row_raw = str(row.get("region") or "")
        row_coarse = str(row.get("coarse_region") or row_raw)
        active = bool(row_entity == selected_entity and (row_raw == selected_raw or row_coarse == selected_coarse))
        relation = str(row.get("entity_relation") or row.get("relation") or "other").lower()
        if active:
            color = (255, 255, 255)
        elif relation == "partner":
            color = (209, 230, 72)
        elif relation == "self":
            color = (85, 191, 255)
        else:
            color = (135, 155, 120)
        thickness = 3 if active else (2 if row_coarse == "torso" else 1)
        if geometry.get("center_mm") is not None:
            center = map_point(geometry["center_mm"])
            radius = max(2, min(54, int(round(float(geometry.get("radius_mm") or 20.0) * scale))))
            cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
        elif geometry.get("start_mm") is not None and geometry.get("end_mm") is not None:
            cv2.line(frame, map_point(geometry["start_mm"]), map_point(geometry["end_mm"]), color, thickness, cv2.LINE_AA)
    if origin is not None and sample.get("ray_direction_x") is not None:
        try:
            direction = np.asarray([
                sample.get("ray_direction_x"), sample.get("ray_direction_y"), sample.get("ray_direction_z")
            ], dtype=float)
            distance = float(sample.get("body_distance_mm") or 800.0)
            if np.isfinite(direction).all():
                cv2.line(frame, map_point(origin), map_point(origin + direction * distance), (95, 230, 105), 2, cv2.LINE_AA)
        except Exception:
            pass
    if hit is not None:
        cv2.circle(frame, map_point(hit), 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_review_frame(frame: np.ndarray, payload: dict[str, Any], mode: ReviewMode,
                       *, draw_face_regions: bool, draw_trail: bool, draw_confidence: bool) -> None:
    if draw_face_regions:
        _draw_face_regions(frame, payload.get("face_regions") or [])
        _draw_participant_association(frame, payload)
    if mode in {"body", "fusion"}:
        _draw_body_world_inset(frame, payload)
    if draw_trail:
        trail = payload.get("trail") or []
        points = [
            (int(round(row["gaze_x_px"])), int(round(row["gaze_y_px"])))
            for row in trail if row.get("gaze_x_px") is not None and row.get("gaze_y_px") is not None
        ]
        for left, right in zip(points, points[1:]):
            cv2.line(frame, left, right, (90, 210, 120), 1, cv2.LINE_AA)
    for sample in payload.get("samples") or []:
        x, y = sample.get("gaze_x_px"), sample.get("gaze_y_px")
        if x is None or y is None:
            continue
        point = (int(round(x)), int(round(y)))
        cv2.circle(frame, point, 5, (35, 255, 110), -1, cv2.LINE_AA)
    summary = payload.get("summary") or {}
    label = f"{summary.get('primary_entity_id') or '—'} · {summary.get('primary_region') or 'no_gaze'}"
    confidence = summary.get("mean_confidence") or 0.0
    cv2.rectangle(frame, (10, 10), (min(frame.shape[1]-10, 520), 72), (24, 27, 21), -1)
    cv2.putText(frame, f"{mode.upper()} AOI: {label}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (225, 248, 47), 2, cv2.LINE_AA)
    if draw_confidence:
        cv2.putText(frame, f"confidence {confidence:.3f} · samples {summary.get('gaze_sample_count',0)}", (20, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 230, 210), 1, cv2.LINE_AA)
    if summary.get("review_required"):
        cv2.rectangle(frame, (frame.shape[1]-210, 12), (frame.shape[1]-12, 50), (40, 55, 190), -1)
        cv2.putText(frame, "REVIEW REQUIRED", (frame.shape[1]-198, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255,255,255), 2, cv2.LINE_AA)


def render_aoi_review_video(
    store: ProjectStore,
    *,
    session_id: str,
    mode: ReviewMode,
    artifact_id: str,
    video_artifact_id: str | None = None,
    label: str = "AOI review video",
    confidence_threshold: float = 0.65,
    draw_face_regions: bool = True,
    draw_trail: bool = True,
    draw_confidence: bool = True,
    preserve_audio: bool = True,
    crf: int = 20,
    preset: str = "fast",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    data = _build_dataset(store, session_id, mode, artifact_id, video_artifact_id, confidence_threshold)
    encoder = privacy_encoder_preflight(ffmpeg_path, preserve_audio=preserve_audio, preset=preset, crf=crf, ffprobe_path=ffprobe_path)
    cap = cv2.VideoCapture(str(data.video_path))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode review video: {data.video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Review video dimensions are unavailable")
    fps = float(data.video_info["fps"])
    frame_count = int(data.video_info["frame_count"])
    artifact_id_out = new_id("artifact")
    staging = store.derived_dir(session_id, "aoi_review_video") / (".partial_" + artifact_id_out)
    final = store.derived_dir(session_id, "aoi_review_video") / artifact_id_out
    staging.mkdir(parents=True, exist_ok=False)
    silent = staging / "review_silent.mp4"
    output = staging / f"{safe_name(label, fallback='aoi_review')}.mp4"
    report_path = staging / "aoi_review_manifest.json"
    writer = _output_writer(silent, width, height, fps)
    rows = []
    try:
        for frame_index in range(frame_count):
            token.raise_if_cancelled()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ProcessingError(f"Review video decode ended at frame {frame_index}")
            payload = review_frame(
                store, session_id=session_id, mode=mode, artifact_id=artifact_id,
                video_artifact_id=data.video_artifact.id, output_frame_index=frame_index,
                sample_index=0, trail_frames=max(1, int(round(fps * 0.20))),
                confidence_threshold=confidence_threshold,
            )
            _draw_review_frame(frame, payload, mode, draw_face_regions=draw_face_regions,
                               draw_trail=draw_trail, draw_confidence=draw_confidence)
            writer.write(frame)
            rows.append({
                "output_frame_index": frame_index,
                "native_frame_index": payload["native_frame_index"],
                "primary_entity_id": payload["summary"]["primary_entity_id"],
                "primary_region": payload["summary"]["primary_region"],
                "mean_confidence": payload["summary"]["mean_confidence"],
                "review_required": payload["summary"]["review_required"],
            })
            if frame_index % 20 == 0:
                emit(progress, 0.05 + 0.78 * (frame_index + 1) / frame_count,
                     f"Rendering {mode} AOI review ({frame_index + 1}/{frame_count})")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        cap.release()
        writer.release()
    pd.DataFrame(rows).to_csv(staging / "aoi_review_frame_map.csv", index=False)
    try:
        _mux_audio_and_transcode(
            silent, data.video_path, output, source_start_sec=0.0,
            duration_sec=frame_count / fps, preserve_audio=preserve_audio,
            crf=crf, preset=preset, ffmpeg_path=encoder["ffmpeg"]["path"],
            expected_frames=frame_count, cancellation=token, progress=progress,
        )
        silent.unlink(missing_ok=True)
        manifest = {
            "schema_version": "oculostudio-aoi-review-video-1.0",
            "artifact_id": artifact_id_out,
            "mode": mode,
            "source_aoi_artifact_id": artifact_id,
            "video_artifact_id": data.video_artifact.id,
            "confidence_threshold": confidence_threshold,
            "layers": {
                "face_regions": draw_face_regions,
                "participant_association": bool(data.participant_projections_by_native or data.face_identities_by_native),
                "trail": draw_trail,
                "confidence": draw_confidence,
                "body_world_inset": mode in {"body", "fusion"},
            },
            "frame_count": frame_count,
            "fps": fps,
            "created_at": utc_now(),
        }
        atomic_write_json(report_path, manifest)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.AOI_REVIEW_VIDEO,
        label=label,
        output_root=final,
        parent_artifact_ids=[artifact_id, data.video_artifact.id],
        source_asset_ids=list(dict.fromkeys([*data.artifact.source_asset_ids, *data.video_artifact.source_asset_ids])),
        parameters={
            "mode": mode, "confidence_threshold": confidence_threshold,
            "draw_face_regions": draw_face_regions, "draw_trail": draw_trail,
            "draw_confidence": draw_confidence,
        },
        qc={
            "output_frame_count": frame_count,
            "output_fps": fps,
            "review_frame_count": len(data.events["review"]),
            "conflict_frame_count": len(data.events["conflict"]),
            "low_confidence_frame_count": len(data.events["low_confidence"]),
        },
        provenance={
            "source_aoi_artifact_id": artifact_id,
            "video_artifact_id": data.video_artifact.id,
            "dynamic_review_equivalent": True,
        },
        artifact_id=artifact_id_out,
    )
    emit(progress, 1.0, "AOI review video complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "video": str(final / output.name),
        "frame_map": str(final / "aoi_review_frame_map.csv"),
        "manifest": str(final / report_path.name),
    }
