from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.semantic import SemanticRegion
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file, find_video_file
from oculostudio.services.face_semantics import (
    FaceInstance,
    Point2D,
    classify_face_point,
    face_instances_to_regions,
    load_face_semantics,
)
from oculostudio.services.privacy import _mux_audio_and_transcode, _output_writer
from oculostudio.services.participant_association import MATCHED_IDENTITY_STATUSES, load_participant_association

FACE_AOI_ENGINE = "track-aware-gaze-timestamp-face-geometry-v14"
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


def _normal(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _column(columns, aliases, *, required=True):
    lookup = {_normal(column): column for column in columns}
    for alias in aliases:
        if _normal(alias) in lookup:
            return lookup[_normal(alias)]
    if required:
        raise ValidationError(f"Face AOI input is missing required column: {aliases[0]}")
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {path.name}: {exc}") from exc


def _trim_root(artifact) -> tuple[Path, dict]:
    manifest = find_artifact_file(artifact, names=["trim_manifest.json"])
    return manifest.parent, _read_json(manifest)


def _bool_value(value) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _gaze_uid(recording_id: str, source_row_index: int, timestamp_ns: int) -> str:
    raw = f"{recording_id}|{source_row_index}|{timestamp_ns}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


def _estimate_rate_hz_ns(values: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype("int64").to_numpy()
    if len(numeric) < 2:
        return None
    numeric = np.unique(numeric)
    if len(numeric) < 2:
        return None
    delta = np.diff(numeric).astype(np.float64)
    delta = delta[delta > 0]
    if not len(delta):
        return None
    median = float(np.median(delta))
    return float(1e9 / median) if median > 0 else None


AUTO_REVIEW_POLICIES = {"off", "conservative", "adaptive", "balanced"}
_AUTO_MATCHED_STATUS_PREFIXES = (
    "auto_identity_",
    "auto_projection_",
    "auto_track_",
    "auto_fixation_",
)
_AUTO_FACE_REGIONS = {
    SemanticRegion.LEFT_EYE.value,
    SemanticRegion.RIGHT_EYE.value,
    SemanticRegion.NOSE.value,
    SemanticRegion.MOUTH.value,
    SemanticRegion.OTHER_FACE.value,
    SemanticRegion.FACE.value,
}
_AUTO_SPECIFIC_FACE_REGIONS = {
    SemanticRegion.LEFT_EYE.value,
    SemanticRegion.RIGHT_EYE.value,
    SemanticRegion.NOSE.value,
    SemanticRegion.MOUTH.value,
}


def _present_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _bool_flag(value: Any) -> bool:
    parsed = _bool_value(value)
    return bool(parsed) if parsed is not None else False


def _is_auto_matched_status(value: Any) -> bool:
    status = _present_text(value) or ""
    return status in MATCHED_IDENTITY_STATUSES or status.startswith(_AUTO_MATCHED_STATUS_PREFIXES)


def _identity_auto_resolution(row: pd.Series, policy: str) -> dict[str, Any] | None:
    """Conservatively accept already-labelled participant identities.

    Participant association intentionally retains a review flag for several
    matched statuses.  Face AOI can clear that flag when the same row carries
    auditable 3D/track evidence.  This never converts an unmatched candidate.
    """
    if policy == "off":
        return None
    status = _present_text(row.get("identity_status")) or ""
    if status not in MATCHED_IDENTITY_STATUSES:
        return None
    relation = _present_text(row.get("target_relation"))
    entity = _present_text(row.get("face_entity_id"))
    region = _present_text(row.get("face_region"))
    if relation not in {"partner", "self"} or not entity or region not in _AUTO_FACE_REGIONS:
        return None
    if int(_safe_float(row.get("fm_on_face_match_count"), 0)) > 1:
        return None
    assignment_status = _present_text(row.get("assignment_status")) or ""
    if assignment_status == "face_geometry_vs_onface_conflict":
        return None

    identity_confidence = float(np.clip(_safe_float(row.get("identity_confidence"), 0.0), 0.0, 1.0))
    hit_confidence = float(np.clip(_safe_float(row.get("face_aoi_confidence"), 0.0), 0.0, 1.0))
    normalized_error = _safe_float(row.get("identity_normalized_center_error"), float("inf"))
    projected_iou = _safe_float(row.get("identity_projected_box_iou"), 0.0)
    bbox_overlap = _bool_flag(row.get("identity_bbox_overlap_present"))
    face_overlap_fraction = _safe_float(row.get("identity_face_box_overlap_fraction"), 0.0)
    duplicate_iou = _safe_float(row.get("identity_duplicate_iou"), 0.0)
    center_inside = _bool_flag(row.get("identity_projected_center_inside_face"))
    face_inside = _bool_flag(row.get("identity_face_center_inside_projection"))
    geometry_strong = _bool_flag(row.get("identity_geometry_strong"))
    track_consistent = _bool_flag(row.get("identity_temporal_track_consistency"))
    candidate_entity = _present_text(row.get("candidate_entity_id"))
    candidate_relation = _present_text(row.get("candidate_relation"))
    candidate_agrees = (not candidate_entity or candidate_entity == entity) and (
        not candidate_relation or candidate_relation == relation
    )
    geometric_support = bool(
        geometry_strong
        or bbox_overlap
        or (center_inside and normalized_error <= (0.95 if policy == "conservative" else 1.05 if policy == "adaptive" else 1.15))
        or (face_inside and projected_iou >= (0.02 if policy == "conservative" else 0.015 if policy == "adaptive" else 0.01))
        or (track_consistent and normalized_error <= (1.05 if policy == "conservative" else 1.15 if policy == "adaptive" else 1.25))
    )

    thresholds = {
        "conservative": {
            "matched_participant": 0.50,
            "projection_locked_participant": 0.30,
            "overlap_locked_participant": 0.68,
            "projection_cluster_inherits_participant": 0.48,
            "duplicate_inherits_matched_participant": 0.50,
            "temporal_bridge_matched_participant": 0.42,
            "temporal_edge_matched_participant": 0.58,
        },
        "adaptive": {
            "matched_participant": 0.46,
            "projection_locked_participant": 0.26,
            "overlap_locked_participant": 0.66,
            "projection_cluster_inherits_participant": 0.42,
            "duplicate_inherits_matched_participant": 0.45,
            "temporal_bridge_matched_participant": 0.38,
            "temporal_edge_matched_participant": 0.52,
        },
        "balanced": {
            "matched_participant": 0.42,
            "projection_locked_participant": 0.22,
            "overlap_locked_participant": 0.64,
            "projection_cluster_inherits_participant": 0.38,
            "duplicate_inherits_matched_participant": 0.40,
            "temporal_bridge_matched_participant": 0.34,
            "temporal_edge_matched_participant": 0.46,
        },
    }[policy]
    threshold = thresholds.get(status)
    if threshold is None or identity_confidence < threshold or hit_confidence < (0.40 if policy == "conservative" else 0.35 if policy == "adaptive" else 0.30):
        return None

    if status == "overlap_locked_participant":
        supported = candidate_agrees and bbox_overlap and face_overlap_fraction > 0.0
    elif status == "temporal_bridge_matched_participant":
        supported = candidate_agrees
    elif status == "temporal_edge_matched_participant":
        supported = candidate_agrees and (geometric_support or track_consistent)
    elif status == "duplicate_inherits_matched_participant":
        supported = candidate_agrees and (duplicate_iou >= (0.58 if policy == "conservative" else 0.50 if policy == "adaptive" else 0.42) or geometric_support)
    elif status == "projection_cluster_inherits_participant":
        supported = candidate_agrees and geometric_support
    else:
        supported = candidate_agrees and geometric_support
    if not supported:
        return None

    confidence = float(np.clip(0.58 * identity_confidence + 0.42 * hit_confidence, 0.0, 0.99))
    return {
        "source": f"identity_{status}",
        "confidence": confidence,
        "identity_status": f"auto_identity_{status}",
        "assignment_status": "participant_face_aoi_hit_auto_identity",
    }


def _direct_green_box_auto_resolution(row: pd.Series, policy: str) -> dict[str, Any] | None:
    """Accept a direct hit inside an identity-resolved FaceMapper green box.

    Once Participant Association has resolved the accepted green face box to a
    recorded participant, the box itself is the authoritative face-membership
    observation.  Low upstream match confidence may remain in the audit fields,
    but it must not force a human decision for a gaze centre that is already
    inside that exact box.  Edge-band hits and genuinely ambiguous identities
    are intentionally excluded from this rule.
    """
    if policy == "off":
        return None
    if not _bool_flag(row.get("green_face_exact_hit")):
        return None
    if _bool_flag(row.get("face_edge_ambiguous")):
        return None
    if not _bool_flag(row.get("include_in_primary_analysis")):
        return None
    relation = _present_text(row.get("target_relation"))
    entity = _present_text(row.get("face_entity_id"))
    region = _present_text(row.get("face_region"))
    status = _present_text(row.get("identity_status")) or ""
    if relation not in {"partner", "self"} or not entity or region not in _AUTO_FACE_REGIONS:
        return None
    if not _is_auto_matched_status(status):
        return None
    face_confidence = float(np.clip(_safe_float(row.get("face_aoi_confidence"), 0.0), 0.0, 1.0))
    identity_confidence = float(np.clip(_safe_float(row.get("identity_confidence"), 0.0), 0.0, 1.0))
    confidence = float(np.clip(max(0.50, 0.58 * face_confidence + 0.42 * identity_confidence), 0.0, 0.99))
    return {
        "source": "direct_green_box_matched_identity",
        "confidence": confidence,
        "identity_status": "auto_identity_direct_green_box_matched_participant",
        "assignment_status": "participant_face_aoi_hit_auto_direct_green_box",
    }


def _valid_fixation_key(value: Any) -> str | None:
    return _present_text(value)


def _auto_resolution_pair(row: pd.Series) -> tuple[str, str] | None:
    relation = _present_text(row.get("target_relation"))
    entity = _present_text(row.get("face_entity_id"))
    if relation not in {"partner", "self"} or not entity:
        return None
    return relation, entity


def _auto_resolution_candidate_compatible(row: pd.Series, relation: str, entity: str) -> bool:
    candidate_relation = _present_text(row.get("candidate_relation"))
    candidate_entity = _present_text(row.get("candidate_entity_id"))
    if candidate_relation and candidate_relation != relation:
        return False
    if candidate_entity and candidate_entity != entity:
        return False
    return True


def _set_auto_resolution(
    frame: pd.DataFrame,
    index: Any,
    *,
    relation: str,
    entity: str,
    region: str,
    confidence: float,
    source: str,
    identity_status: str,
    assignment_status: str,
) -> None:
    frame.at[index, "target_relation"] = relation
    frame.at[index, "face_entity_id"] = entity
    frame.at[index, "face_region"] = region
    frame.at[index, "identity_status"] = identity_status
    frame.at[index, "assignment_status"] = assignment_status
    frame.at[index, "identity_confidence"] = max(
        _safe_float(frame.at[index, "identity_confidence"] if "identity_confidence" in frame.columns else None, 0.0),
        float(confidence),
    )
    frame.at[index, "face_aoi_confidence"] = max(
        _safe_float(frame.at[index, "face_aoi_confidence"], 0.0),
        float(confidence),
    )
    frame.at[index, "include_in_primary_analysis"] = True
    frame.at[index, "review_required"] = False
    frame.at[index, "auto_review_resolved"] = True
    frame.at[index, "auto_review_resolution_source"] = source
    frame.at[index, "auto_review_confidence"] = float(confidence)


def _projection_auto_resolution(row: pd.Series, policy: str) -> dict[str, Any] | None:
    """Deprecated compatibility hook: projections never classify gaze as face.

    Participant-head projections identify which accepted FaceMapper green box
    belongs to which person. They are not gaze AOIs. A sample without a direct
    hit inside an accepted green box must remain ``no_face`` regardless of the
    projection, FaceMapper's exported gaze-on-face flag, or temporal context.
    """
    return None


def _facemapper_on_face_consensus(values: list[bool | None]) -> tuple[bool | None, bool, bool, int]:
    """Collapse duplicate FaceMapper on-face rows without treating copies as conflicts.

    Returns ``(value, conflict, duplicate_consensus, unique_value_count)``.
    Only contradictory non-null values are a true conflict.
    """
    known = [value for value in values if value is not None]
    unique = set(known)
    conflict = len(unique) > 1
    value = next(iter(unique)) if len(unique) == 1 else None
    duplicate_consensus = len(values) > 1 and len(unique) == 1
    return value, conflict, duplicate_consensus, len(unique)


def _unique_projection_candidate(
    x_px: float,
    y_px: float,
    projections: list[dict[str, Any]],
    *,
    target_person_policy: str,
    auto_review_policy: str,
) -> dict[str, Any] | None:
    """Legacy helper retained for artifact-reader compatibility.

    Strict green-box Face AOI generation no longer calls this function. Projected
    participant-head boxes identify green FaceMapper detections; they never
    classify a gaze point as face.
    """
    if not np.isfinite([x_px, y_px]).all() or auto_review_policy == "off":
        return None
    margin_ratio = {
        "conservative": 0.12,
        "adaptive": 0.08,
        "balanced": 0.04,
    }.get(auto_review_policy, 0.12)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for projection in projections:
        if _bool_value(projection.get("projection_valid")) is not True:
            continue
        in_view = _bool_value(projection.get("in_camera_view"))
        if in_view is False:
            continue
        relation = _present_text(projection.get("relation_to_wearer"))
        if relation not in {"partner", "self"}:
            continue
        if target_person_policy == "partner_only" and relation != "partner":
            continue
        entity = _present_text(projection.get("entity_id")) or _present_text(projection.get("role_id"))
        if not entity:
            continue
        try:
            x1 = float(projection.get("projected_bbox_x1"))
            y1 = float(projection.get("projected_bbox_y1"))
            x2 = float(projection.get("projected_bbox_x2"))
            y2 = float(projection.get("projected_bbox_y2"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            continue
        width = x2 - x1
        height = y2 - y1
        left = x1 + width * margin_ratio
        right = x2 - width * margin_ratio
        top = y1 + height * margin_ratio
        bottom = y2 - height * margin_ratio
        if not (left <= x_px <= right and top <= y_px <= bottom):
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        dx = abs(x_px - cx) / max(1.0, width * 0.5)
        dy = abs(y_px - cy) / max(1.0, height * 0.5)
        centrality = float(np.clip(1.0 - max(dx, dy), 0.0, 1.0))
        projection_confidence = float(np.clip(
            _safe_float(projection.get("projection_confidence"), 0.0), 0.0, 1.0
        ))
        confidence = float(np.clip(0.78 * projection_confidence + 0.22 * centrality, 0.0, 1.0))
        key = (relation, entity)
        current = candidates.get(key)
        if current is None or confidence > float(current["confidence"]):
            candidates[key] = {
                "relation": relation,
                "entity_id": entity,
                "role_id": _present_text(projection.get("role_id")),
                "confidence": confidence,
                "projection_confidence": projection_confidence,
                "centrality": centrality,
            }
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _review_reason(row: pd.Series) -> str:
    status = _present_text(row.get("assignment_status")) or "unknown_review_reason"
    if _bool_flag(row.get("fm_on_face_conflict")):
        return "conflicting_facemapper_on_face_rows"
    return status


def _apply_auto_review_reduction(samples: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Reduce review load with auditable temporal evidence only.

    The order is deliberate: direct exact green-box acceptance, strong upstream
    identity evidence, same-track continuity, then same-fixation consensus. Each stage can only
    clear a review flag; invalid/out-of-frame rows and contradictory candidates
    are never promoted.
    """
    if policy not in AUTO_REVIEW_POLICIES:
        raise ValidationError("auto_review_policy must be off, conservative, adaptive or balanced")
    frame = samples.copy()
    if "review_required" not in frame.columns:
        frame["review_required"] = False
    frame["review_required"] = frame["review_required"].fillna(False).astype(bool)
    frame["review_required_before_auto_resolution"] = frame["review_required"].astype(bool)
    frame["review_reason_before_auto_resolution"] = frame.apply(_review_reason, axis=1)
    frame["auto_review_resolved"] = False
    frame["auto_review_resolution_source"] = ""
    frame["auto_review_confidence"] = np.nan
    frame["auto_review_policy"] = policy

    # Geometry is authoritative before any temporal reasoning. If a valid sample
    # is outside every accepted green FaceMapper box, it is deterministically
    # no_face and must not enter Human Review or be promoted by context.
    strict_outside_mask = (
        frame.get("assignment_status", pd.Series("", index=frame.index)).astype(str)
        .eq("outside_all_accepted_facemapper_face_boxes")
        | (
            frame.get("face_region", pd.Series("", index=frame.index)).astype(str).eq("no_face")
            & ~frame.get("raw_face_region", pd.Series(None, index=frame.index)).map(_present_text).notna()
        )
    )
    strict_outside_was_review = strict_outside_mask & frame["review_required"]
    if strict_outside_mask.any():
        frame.loc[strict_outside_mask, "face_region"] = "no_face"
        frame.loc[strict_outside_mask, "face_entity_id"] = None
        frame.loc[strict_outside_mask, "target_relation"] = None
        frame.loc[strict_outside_mask, "identity_status"] = None
        frame.loc[strict_outside_mask, "include_in_primary_analysis"] = False
        frame.loc[strict_outside_mask, "face_aoi_confidence"] = 1.0
        frame.loc[strict_outside_mask, "review_required"] = False
        frame.loc[strict_outside_mask, "assignment_status"] = "outside_all_accepted_facemapper_face_boxes"
    if strict_outside_was_review.any():
        frame.loc[strict_outside_was_review, "auto_review_resolved"] = True
        frame.loc[strict_outside_was_review, "auto_review_resolution_source"] = "strict_green_box_no_face"
        frame.loc[strict_outside_was_review, "auto_review_confidence"] = 1.0

    # 1) Exact gaze centres inside a participant-resolved green FaceMapper box
    # are accepted directly.  The upstream review flag remains auditable in
    # review_required_before_auto_resolution, but no manual decision is needed.
    # The bounded edge band is handled by the later evidence rules.
    if policy != "off":
        for index, row in frame.loc[frame["review_required"]].iterrows():
            direct_resolution = _direct_green_box_auto_resolution(row, policy)
            if direct_resolution is not None:
                _set_auto_resolution(
                    frame,
                    index,
                    relation=str(row.get("target_relation")),
                    entity=str(row.get("face_entity_id")),
                    region=str(row.get("face_region")),
                    confidence=float(direct_resolution["confidence"]),
                    source=str(direct_resolution["source"]),
                    identity_status=str(direct_resolution["identity_status"]),
                    assignment_status=str(direct_resolution["assignment_status"]),
                )
                continue
            projection_resolution = _projection_auto_resolution(row, policy)
            if projection_resolution is not None:
                _set_auto_resolution(
                    frame,
                    index,
                    relation=str(projection_resolution["relation"]),
                    entity=str(projection_resolution["entity"]),
                    region=str(projection_resolution["region"]),
                    confidence=float(projection_resolution["confidence"]),
                    source=str(projection_resolution["source"]),
                    identity_status=str(projection_resolution["identity_status"]),
                    assignment_status=str(projection_resolution["assignment_status"]),
                )
                continue
            resolution = _identity_auto_resolution(row, policy)
            if resolution is None:
                continue
            _set_auto_resolution(
                frame,
                index,
                relation=str(row.get("target_relation")),
                entity=str(row.get("face_entity_id")),
                region=str(row.get("face_region")),
                confidence=float(resolution["confidence"]),
                source=str(resolution["source"]),
                identity_status=str(resolution["identity_status"]),
                assignment_status=str(resolution["assignment_status"]),
            )

    anchor_threshold = {
        "conservative": 0.60,
        "adaptive": 0.55,
        "balanced": 0.50,
        "off": 1.01,
    }[policy]
    max_track_gap_frames = {"conservative": 4, "adaptive": 6, "balanced": 8, "off": 0}[policy]
    max_fixation_gap_ns = int({"conservative": 90, "adaptive": 120, "balanced": 150, "off": 0}[policy] * 1_000_000)
    minimum_fixation_anchors = {"conservative": 3, "adaptive": 2, "balanced": 2, "off": 99}[policy]
    pair_consensus_threshold = {"conservative": 0.86, "adaptive": 0.80, "balanced": 0.74, "off": 1.01}[policy]
    region_consensus_threshold = {"conservative": 0.84, "adaptive": 0.80, "balanced": 0.70, "off": 1.01}[policy]

    def anchor_mask(current: pd.DataFrame) -> pd.Series:
        relation_ok = current.get("target_relation", pd.Series("", index=current.index)).astype(str).isin({"partner", "self"})
        entity_ok = current.get("face_entity_id", pd.Series(None, index=current.index)).map(_present_text).notna()
        region_ok = current.get("face_region", pd.Series("", index=current.index)).astype(str).isin(_AUTO_FACE_REGIONS)
        confidence_ok = pd.to_numeric(current.get("face_aoi_confidence", 0.0), errors="coerce").fillna(0.0) >= anchor_threshold
        primary_ok = current.get("include_in_primary_analysis", pd.Series(False, index=current.index)).fillna(False).astype(bool)
        return (~current["review_required"]) & relation_ok & entity_ok & region_ok & confidence_ok & primary_ok

    # 2) Same FaceMapper track, matching entities on both sides.  This repairs
    # isolated identity dropouts without inventing face geometry.
    if policy != "off" and "face_track_id" in frame.columns and "native_frame_index" in frame.columns:
        anchors = frame.loc[anchor_mask(frame)].copy()
        for track_value, group in frame.loc[frame["review_required"]].groupby("face_track_id", dropna=True, sort=False):
            if _present_text(track_value) is None:
                continue
            track_anchors = anchors.loc[anchors["face_track_id"] == track_value].sort_values("native_frame_index")
            if track_anchors.empty:
                continue
            for index, row in group.sort_values("native_frame_index").iterrows():
                raw_region = _present_text(row.get("raw_face_region"))
                if raw_region not in _AUTO_FACE_REGIONS:
                    continue
                if int(_safe_float(row.get("fm_on_face_match_count"), 0)) > 1:
                    continue
                current_frame = int(_safe_float(row.get("native_frame_index"), -1))
                previous = track_anchors.loc[pd.to_numeric(track_anchors["native_frame_index"], errors="coerce") <= current_frame].tail(1)
                following = track_anchors.loc[pd.to_numeric(track_anchors["native_frame_index"], errors="coerce") >= current_frame].head(1)
                if previous.empty or following.empty:
                    if policy == "conservative":
                        continue
                    nearest = track_anchors.iloc[(pd.to_numeric(track_anchors["native_frame_index"], errors="coerce") - current_frame).abs().argsort()[:1]]
                    if nearest.empty or abs(int(nearest.iloc[0]["native_frame_index"]) - current_frame) > 2:
                        continue
                    previous = following = nearest
                prev_row = previous.iloc[0]
                next_row = following.iloc[0]
                prev_pair = _auto_resolution_pair(prev_row)
                next_pair = _auto_resolution_pair(next_row)
                if prev_pair is None or next_pair is None or prev_pair != next_pair:
                    continue
                relation, entity = prev_pair
                if not _auto_resolution_candidate_compatible(row, relation, entity):
                    continue
                prev_gap = abs(current_frame - int(prev_row["native_frame_index"]))
                next_gap = abs(int(next_row["native_frame_index"]) - current_frame)
                if max(prev_gap, next_gap) > max_track_gap_frames:
                    continue
                confidence = float(np.clip(
                    0.78 * min(_safe_float(prev_row.get("face_aoi_confidence"), 0.0), _safe_float(next_row.get("face_aoi_confidence"), 0.0))
                    + 0.22 * _safe_float(row.get("face_aoi_confidence"), 0.0),
                    0.0,
                    0.97,
                ))
                _set_auto_resolution(
                    frame,
                    index,
                    relation=relation,
                    entity=entity,
                    region=raw_region,
                    confidence=confidence,
                    source="same_track_temporal_bridge",
                    identity_status="auto_track_temporal_bridge_matched_participant",
                    assignment_status="participant_face_aoi_hit_auto_track_bridge",
                )
        # Recompute anchors so track-resolved rows may support fixation-level
        # face identity, but never recursively change another track.
        anchors = frame.loc[anchor_mask(frame)].copy()

        # 3) Same fixation consensus.  Specific eyes/nose/mouth are retained only
        # when the row already has geometry, or (balanced mode) both temporal
        # neighbours agree.  Conservative mode converts missing fine geometry to
        # partner/self face-unspecified rather than fabricating a feature.
        if "fixation_id" in frame.columns and "timestamp_ns" in frame.columns:
            fixation_keys = frame["fixation_id"].map(_valid_fixation_key)
            for fixation_key, group_indices in fixation_keys.dropna().groupby(fixation_keys.dropna()).groups.items():
                group = frame.loc[list(group_indices)].sort_values("timestamp_ns")
                group_anchors = group.loc[anchor_mask(group)]
                if len(group_anchors) < minimum_fixation_anchors:
                    continue
                pair_weights: dict[tuple[str, str], float] = defaultdict(float)
                pair_counts: Counter[tuple[str, str]] = Counter()
                for _, anchor in group_anchors.iterrows():
                    pair = _auto_resolution_pair(anchor)
                    if pair is None:
                        continue
                    weight = max(0.05, _safe_float(anchor.get("face_aoi_confidence"), 0.0))
                    pair_weights[pair] += weight
                    pair_counts[pair] += 1
                if not pair_weights:
                    continue
                pair, pair_weight = max(pair_weights.items(), key=lambda item: item[1])
                total_pair_weight = sum(pair_weights.values())
                pair_share = pair_weight / max(total_pair_weight, 1e-9)
                if pair_counts[pair] < minimum_fixation_anchors or pair_share < pair_consensus_threshold:
                    continue
                relation, entity = pair
                consensus_anchors = group_anchors.loc[
                    (group_anchors["target_relation"].astype(str) == relation)
                    & (group_anchors["face_entity_id"].astype(str) == entity)
                ].sort_values("timestamp_ns")
                if consensus_anchors.empty:
                    continue
                region_weights: dict[str, float] = defaultdict(float)
                for _, anchor in consensus_anchors.iterrows():
                    region = _present_text(anchor.get("face_region"))
                    if region in _AUTO_FACE_REGIONS:
                        region_weights[region] += max(0.05, _safe_float(anchor.get("face_aoi_confidence"), 0.0))
                region_choice = max(region_weights.items(), key=lambda item: item[1])[0] if region_weights else SemanticRegion.OTHER_FACE.value
                region_share = (
                    region_weights.get(region_choice, 0.0) / max(sum(region_weights.values()), 1e-9)
                    if region_weights else 0.0
                )
                for index, row in group.loc[group["review_required"]].iterrows():
                    current_region = _present_text(row.get("face_region"))
                    if current_region in {SemanticRegion.INVALID_GAZE.value, SemanticRegion.OUT_OF_FRAME.value}:
                        continue
                    if not _auto_resolution_candidate_compatible(row, relation, entity):
                        continue
                    if int(_safe_float(row.get("fm_on_face_match_count"), 0)) > 1:
                        continue
                    timestamp = int(_safe_float(row.get("timestamp_ns"), -1))
                    before = consensus_anchors.loc[pd.to_numeric(consensus_anchors["timestamp_ns"], errors="coerce") <= timestamp].tail(1)
                    after = consensus_anchors.loc[pd.to_numeric(consensus_anchors["timestamp_ns"], errors="coerce") >= timestamp].head(1)
                    nearest_gap = min(
                        [abs(int(anchor_ts) - timestamp) for anchor_ts in pd.to_numeric(consensus_anchors["timestamp_ns"], errors="coerce").dropna().astype("int64").tolist()]
                        or [10**18]
                    )
                    raw_region = _present_text(row.get("raw_face_region"))
                    geometry_available = raw_region in _AUTO_FACE_REGIONS
                    if not geometry_available:
                        # Strict green-box rule: temporal/fixation continuity may
                        # resolve identity only for a sample that already has a
                        # direct FaceMapper green-box geometry hit. It cannot turn
                        # an outside-box sample into face_unspecified.
                        continue
                    if nearest_gap > max_fixation_gap_ns:
                        continue
                    resolved_region = raw_region
                    mean_anchor_confidence = float(pd.to_numeric(consensus_anchors["face_aoi_confidence"], errors="coerce").fillna(0.0).mean())
                    confidence = float(np.clip(
                        0.68 * mean_anchor_confidence + 0.22 * pair_share + 0.10 * max(region_share, 0.5),
                        0.0,
                        0.95,
                    ))
                    _set_auto_resolution(
                        frame,
                        index,
                        relation=relation,
                        entity=entity,
                        region=resolved_region,
                        confidence=confidence,
                        source="same_fixation_identity_consensus",
                        identity_status="auto_fixation_consensus_matched_participant",
                        assignment_status="participant_face_aoi_hit_auto_fixation",
                    )

    frame["face_social_region"] = [
        _face_social_region(row.get("face_region"), row.get("target_relation"), row.get("identity_status"))
        for _, row in frame.iterrows()
    ]
    before_mask = frame["review_required_before_auto_resolution"].fillna(False).astype(bool)
    after_mask = frame["review_required"].fillna(False).astype(bool)
    resolved_mask = frame["auto_review_resolved"].fillna(False).astype(bool)
    source_counts = Counter(
        frame.loc[resolved_mask, "auto_review_resolution_source"].fillna("").astype(str)
    )
    before_frames = int(frame.loc[before_mask, "native_frame_index"].nunique()) if "native_frame_index" in frame.columns else 0
    after_frames = int(frame.loc[after_mask, "native_frame_index"].nunique()) if "native_frame_index" in frame.columns else 0
    before_count = int(before_mask.sum())
    after_count = int(after_mask.sum())
    resolved_count = int(resolved_mask.sum())
    before_reason_counts = Counter(
        frame.loc[before_mask, "review_reason_before_auto_resolution"].fillna("unknown_review_reason").astype(str)
    ) if before_count else Counter()
    remaining_reason_counts = Counter(
        frame.loc[after_mask].apply(_review_reason, axis=1).astype(str)
    ) if after_count else Counter()
    report = {
        "schema_version": "oculostudio-face-aoi-auto-review-1.0",
        "policy": policy,
        "review_samples_before": before_count,
        "auto_resolved_samples": resolved_count,
        "review_samples_remaining": after_count,
        "sample_review_reduction_fraction": (
            float((before_count - after_count) / before_count) if before_count else 0.0
        ),
        "review_frames_before": before_frames,
        "review_frames_remaining": after_frames,
        "frame_review_reduction_fraction": (
            float((before_frames - after_frames) / before_frames) if before_frames else 0.0
        ),
        "resolution_source_counts": dict(sorted(source_counts.items())),
        "review_reason_counts_before": dict(sorted(before_reason_counts.items())),
        "remaining_review_reason_counts": dict(sorted(remaining_reason_counts.items())),
        "safety_guards": [
            "never_promote_invalid_or_out_of_frame_gaze",
            "never_override_conflicting_candidate_identity",
            "duplicate_gaze_on_face_rows_are_reviewed_only_when_their_values_conflict",
            "participant_projection_is_identity_gate_only_and_never_a_gaze_aoi",
            "accepted_facemapper_green_box_is_the_authoritative_face_membership_gate",
            "exact_green_box_hits_with_matched_identity_are_auto_accepted",
            "same_track_or_fixation_consensus_never_promotes_outside_box_samples",
            "all auto-resolutions retained in sample-level audit columns",
        ],
    }
    table_rows = [
        {
            "scope": "total",
            "source": "review_samples_before",
            "count": before_count,
            "fraction_of_pre_auto_review": 1.0 if before_count else 0.0,
        },
        {
            "scope": "total",
            "source": "auto_resolved_samples",
            "count": resolved_count,
            "fraction_of_pre_auto_review": float(resolved_count / before_count) if before_count else 0.0,
        },
        {
            "scope": "total",
            "source": "review_samples_remaining",
            "count": after_count,
            "fraction_of_pre_auto_review": float(after_count / before_count) if before_count else 0.0,
        },
    ]
    table_rows.extend({
        "scope": "resolution_source",
        "source": source,
        "count": int(count),
        "fraction_of_pre_auto_review": float(count / before_count) if before_count else 0.0,
    } for source, count in sorted(source_counts.items()))
    table_rows.extend({
        "scope": "review_reason_before",
        "source": source,
        "count": int(count),
        "fraction_of_pre_auto_review": float(count / before_count) if before_count else 0.0,
    } for source, count in sorted(before_reason_counts.items()))
    table_rows.extend({
        "scope": "remaining_review_reason",
        "source": source,
        "count": int(count),
        "fraction_of_pre_auto_review": float(count / before_count) if before_count else 0.0,
    } for source, count in sorted(remaining_reason_counts.items()))
    return frame, report, pd.DataFrame(table_rows)


def _face_social_region(raw_region: str | None, target_relation: str | None, identity_status: str | None) -> str | None:
    """Combine person relation and facial sub-region without ambiguous labels."""
    region = str(raw_region or "").strip()
    relation = str(target_relation or "").strip()
    status = str(identity_status or "").strip()
    if not region:
        return None
    if region in {"invalid_gaze", "out_of_frame", "no_face", "no_face_detection", "unknown"}:
        return region
    matched = status in MATCHED_IDENTITY_STATUSES or status.startswith(_AUTO_MATCHED_STATUS_PREFIXES)
    if relation == "partner":
        if not matched:
            return "partner_face_candidate"
        if region in {"other_face", "face"}:
            return "partner_face_unspecified"
        return f"partner_{region}"
    if relation == "self":
        if region in {"other_face", "face"}:
            return "self_face_unspecified"
        return f"self_{region}"
    if relation in {"ambiguous_partner", "partner_candidate"}:
        return "partner_face_candidate"
    if relation.startswith("ambiguous_") or region in {"unresolved_face_identity", "ambiguous_face_identity"}:
        return "ambiguous_person_face"
    if relation == "other_person":
        return "other_person_face"
    return region


def _finite_xy(identity: dict[str, Any], x_key: str, y_key: str) -> tuple[float, float] | None:
    try:
        x = float(identity.get(x_key))
        y = float(identity.get(y_key))
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x, y]).all():
        return None
    return x, y


def _mirror_point_x(point: Point2D | None, axis_x: float) -> Point2D | None:
    if point is None:
        return None
    return Point2D(2.0 * axis_x - float(point.x), float(point.y))


def _apply_face_orientation(
    instance: FaceInstance,
    *,
    mirror_geometry_x: bool,
    swap_bilateral_labels: bool,
) -> FaceInstance:
    """Apply one canonical geometry/label transform to a FaceMapper face.

    ``mirror_geometry_x`` reflects *all* landmarks inside the detected face box;
    this fixes exports whose landmark coordinates were produced in a mirrored
    local face coordinate system.  ``swap_bilateral_labels`` changes anatomical
    names without moving coordinates.  Keeping the two operations separate is
    essential: a label swap cannot repair a mirrored nose/mouth geometry.
    """
    left_eye = instance.left_eye
    right_eye = instance.right_eye
    nose = instance.nose
    mouth_left = instance.mouth_left
    mouth_right = instance.mouth_right
    if mirror_geometry_x:
        axis_x = 0.5 * (float(instance.x1) + float(instance.x2))
        left_eye = _mirror_point_x(left_eye, axis_x)
        right_eye = _mirror_point_x(right_eye, axis_x)
        nose = _mirror_point_x(nose, axis_x)
        mouth_left = _mirror_point_x(mouth_left, axis_x)
        mouth_right = _mirror_point_x(mouth_right, axis_x)
    if swap_bilateral_labels:
        left_eye, right_eye = right_eye, left_eye
        mouth_left, mouth_right = mouth_right, mouth_left
    return replace(
        instance,
        left_eye=left_eye,
        right_eye=right_eye,
        nose=nose,
        mouth_left=mouth_left,
        mouth_right=mouth_right,
    )


def _orientation_cost(
    instance: FaceInstance,
    identity: dict[str, Any],
    *,
    mirror_geometry_x: bool,
    swap_bilateral_labels: bool,
) -> float | None:
    """Score a landmark-orientation hypothesis against projected head anatomy."""
    if instance.left_eye is None or instance.right_eye is None:
        return None
    left_anchor = _finite_xy(
        identity,
        "projected_anatomical_left_head_x_px",
        "projected_anatomical_left_head_y_px",
    )
    right_anchor = _finite_xy(
        identity,
        "projected_anatomical_right_head_x_px",
        "projected_anatomical_right_head_y_px",
    )
    if left_anchor is None or right_anchor is None:
        return None
    transformed = _apply_face_orientation(
        instance,
        mirror_geometry_x=mirror_geometry_x,
        swap_bilateral_labels=swap_bilateral_labels,
    )
    if transformed.left_eye is None or transformed.right_eye is None:
        return None
    left_eye = np.array([transformed.left_eye.x, transformed.left_eye.y], dtype=float)
    right_eye = np.array([transformed.right_eye.x, transformed.right_eye.y], dtype=float)
    left_ref = np.array(left_anchor, dtype=float)
    right_ref = np.array(right_anchor, dtype=float)
    anchor_sep = max(8.0, float(np.linalg.norm(left_ref - right_ref)))
    eye_sep = max(8.0, float(np.linalg.norm(left_eye - right_eye)))
    scale = max(anchor_sep, eye_sep, 0.25 * (instance.width + instance.height))
    cost = (float(np.linalg.norm(left_eye - left_ref)) + float(np.linalg.norm(right_eye - right_ref))) / scale

    # A mirrored local landmark geometry also moves the nose and mouth to the
    # wrong half of a profile face.  The projected face/head centre provides a
    # weak but useful midline reference that breaks the eye-pair degeneracy.
    head_center = _finite_xy(identity, "projected_head_x_px", "projected_head_y_px")
    if head_center is not None:
        head_x = float(head_center[0])
        if transformed.nose is not None:
            cost += 0.28 * abs(float(transformed.nose.x) - head_x) / max(8.0, instance.width)
        if transformed.mouth_left is not None and transformed.mouth_right is not None:
            mouth_x = 0.5 * (float(transformed.mouth_left.x) + float(transformed.mouth_right.x))
            cost += 0.12 * abs(mouth_x - head_x) / max(8.0, instance.width)
    return float(cost)


def _fallback_label_swap(instances: list[FaceInstance], *, mirror_geometry_x: bool) -> bool:
    deltas: list[float] = []
    for instance in instances:
        transformed = _apply_face_orientation(
            instance,
            mirror_geometry_x=mirror_geometry_x,
            swap_bilateral_labels=False,
        )
        if transformed.left_eye is None or transformed.right_eye is None:
            continue
        delta = float(transformed.left_eye.x - transformed.right_eye.x)
        if np.isfinite(delta) and abs(delta) >= 1.0:
            deltas.append(delta)
    # In a normal scene-camera image the observed person's anatomical left eye
    # is usually image-right.  This is only a fallback when 3D evidence is
    # unavailable; the chosen fallback is recorded and marked for review.
    return bool(deltas and float(np.median(deltas)) < 0.0)


def _lock_partner_orientation_by_track(
    instances_by_frame: dict[int, list[FaceInstance]],
    identity_by_key: dict[tuple[int, int], dict[str, Any]],
    *,
    geometry_policy: Literal["auto_3d", "as_exported", "mirror_x"],
    left_right_policy: Literal["anatomical_auto", "as_exported", "swap_eyes"],
) -> dict[int, dict[str, Any]]:
    if geometry_policy not in {"auto_3d", "as_exported", "mirror_x"}:
        raise ValidationError("face_landmark_geometry_policy must be auto_3d, as_exported or mirror_x")
    by_track: dict[int, list[FaceInstance]] = defaultdict(list)
    evidence: dict[int, dict[tuple[bool, bool], list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame_index, instances in instances_by_frame.items():
        for instance in instances:
            by_track[int(instance.track_id)].append(instance)
            identity = identity_by_key.get((int(frame_index), int(instance.source_row_index)))
            if not identity:
                continue
            relation = str(identity.get("target_relation") or identity.get("candidate_relation") or "")
            if relation not in {"partner", "ambiguous_partner", "partner_candidate"}:
                continue
            mirrors = [False, True] if geometry_policy == "auto_3d" else [geometry_policy == "mirror_x"]
            swaps = [False, True] if left_right_policy == "anatomical_auto" else [left_right_policy == "swap_eyes"]
            for mirror in mirrors:
                for swap in swaps:
                    cost = _orientation_cost(
                        instance,
                        identity,
                        mirror_geometry_x=mirror,
                        swap_bilateral_labels=swap,
                    )
                    if cost is not None and np.isfinite(cost):
                        evidence[int(instance.track_id)][(bool(mirror), bool(swap))].append(float(cost))

    locked: dict[int, dict[str, Any]] = {}
    for track_id, track_instances in by_track.items():
        hypotheses: list[tuple[float, bool, bool, int]] = []
        for (mirror, swap), costs in evidence.get(track_id, {}).items():
            if not costs:
                continue
            robust_cost = float(np.median(np.asarray(costs, dtype=float)))
            hypotheses.append((robust_cost, mirror, swap, len(costs)))
        hypotheses.sort(key=lambda row: row[0])
        if hypotheses:
            best = hypotheses[0]
            second_cost = hypotheses[1][0] if len(hypotheses) > 1 else best[0] + 1.0
            margin = max(0.0, float(second_cost - best[0]))
            confidence = float(np.clip(margin / max(0.15, second_cost), 0.0, 1.0))
            enough = best[3] >= 3 and margin >= 0.025
            if enough:
                locked[track_id] = {
                    "mirror_geometry_x": bool(best[1]),
                    "swap_bilateral_labels": bool(best[2]),
                    "source": "vicon_3d_track_lock",
                    "confidence": confidence,
                    "evidence_count": int(best[3]),
                    "best_cost": float(best[0]),
                    "second_cost": float(second_cost),
                    "verified": True,
                }
                continue
        forced_mirror = geometry_policy == "mirror_x"
        forced_swap = left_right_policy == "swap_eyes"
        if left_right_policy == "anatomical_auto":
            forced_swap = _fallback_label_swap(track_instances, mirror_geometry_x=forced_mirror)
        locked[track_id] = {
            "mirror_geometry_x": bool(forced_mirror),
            "swap_bilateral_labels": bool(forced_swap),
            "source": "fallback_export_geometry" if geometry_policy == "as_exported" else "fallback_unverified",
            "confidence": 0.0,
            "evidence_count": int(hypotheses[0][3]) if hypotheses else 0,
            "best_cost": float(hypotheses[0][0]) if hypotheses else None,
            "second_cost": float(hypotheses[1][0]) if len(hypotheses) > 1 else None,
            "verified": False,
        }
    return locked


def _lerp_point(
    first: Point2D | None,
    second: Point2D | None,
    alpha: float,
    fallback: Point2D | None,
) -> Point2D | None:
    if first is None or second is None:
        return fallback
    return Point2D(
        (1.0 - alpha) * float(first.x) + alpha * float(second.x),
        (1.0 - alpha) * float(first.y) + alpha * float(second.y),
    )


def _face_track_index(
    instances_by_frame: dict[int, list[FaceInstance]],
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[FaceInstance]] = defaultdict(list)
    for instances in instances_by_frame.values():
        for instance in instances:
            grouped[int(instance.track_id)].append(instance)
    output: dict[int, dict[str, Any]] = {}
    for track_id, instances in grouped.items():
        ordered = sorted(
            instances,
            key=lambda item: (int(item.scene_timestamp_ns), int(item.scene_frame_index)),
        )
        output[track_id] = {
            "instances": ordered,
            "timestamps_ns": np.asarray(
                [int(item.scene_timestamp_ns) for item in ordered], dtype=np.int64
            ),
        }
    return output


def _identity_signature_for_instance(
    instance: FaceInstance,
    identity_by_key: dict[tuple[int, int], dict[str, Any]],
) -> tuple[str, str] | None:
    row = identity_by_key.get(
        (int(instance.scene_frame_index), int(instance.source_row_index))
    )
    if not row:
        return None
    status = str(row.get("identity_status") or "")
    entity = _present_text(row.get("matched_entity_id"))
    relation = _present_text(row.get("target_relation"))
    if status not in MATCHED_IDENTITY_STATUSES or entity is None:
        return None
    return entity, relation or ""


def _interpolate_face_instances_for_sample(
    *,
    timestamp_ns: int,
    native_frame_index: int,
    instances_by_frame: dict[int, list[FaceInstance]],
    track_index: dict[int, dict[str, Any]],
    identity_by_key: dict[tuple[int, int], dict[str, Any]],
    target_person_policy: str,
    temporal_policy: str,
    maximum_gap_ms: float,
    maximum_motion_ratio: float,
) -> tuple[list[FaceInstance], dict[int, dict[str, Any]]]:
    """Estimate FaceMapper geometry at one gaze timestamp.

    Video frames remain the only observed image evidence. Interpolation is
    allowed only for the same track, across a short gap, with plausible motion
    and stable participant identity when identity filtering is active.
    """
    base_instances = list(instances_by_frame.get(int(native_frame_index), []))
    metadata: dict[int, dict[str, Any]] = {}
    if temporal_policy == "nearest_frame" or timestamp_ns < 0:
        for item in base_instances:
            metadata[int(item.source_row_index)] = {
                "mode": "nearest_frame",
                "previous_frame_index": int(item.scene_frame_index),
                "next_frame_index": int(item.scene_frame_index),
                "alpha": 0.0,
                "gap_ms": 0.0,
                "confidence": float(item.confidence),
                "reason": "policy_nearest_frame",
            }
        return base_instances, metadata

    output: list[FaceInstance] = []
    for base in base_instances:
        default = {
            "mode": "nearest_frame",
            "previous_frame_index": int(base.scene_frame_index),
            "next_frame_index": int(base.scene_frame_index),
            "alpha": 0.0,
            "gap_ms": 0.0,
            "confidence": float(base.confidence),
            "reason": "no_safe_bracket",
        }
        if int(base.scene_timestamp_ns) == int(timestamp_ns):
            default["reason"] = "exact_observed_frame"
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        track = track_index.get(int(base.track_id))
        if not track:
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        times = track["timestamps_ns"]
        instances = track["instances"]
        position = int(np.searchsorted(times, int(timestamp_ns), side="left"))
        previous_index = max(0, position - 1)
        next_index = min(len(instances) - 1, position)
        previous = instances[previous_index]
        following = instances[next_index]
        if int(previous.scene_timestamp_ns) > timestamp_ns and previous_index > 0:
            previous_index -= 1
            previous = instances[previous_index]
        if int(following.scene_timestamp_ns) < timestamp_ns and next_index + 1 < len(instances):
            next_index += 1
            following = instances[next_index]
        t0 = int(previous.scene_timestamp_ns)
        t1 = int(following.scene_timestamp_ns)
        if not (t0 <= timestamp_ns <= t1) or t1 <= t0:
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        gap_ms = float(t1 - t0) / 1e6
        if gap_ms > float(maximum_gap_ms):
            default["reason"] = "track_gap_too_large"
            default["gap_ms"] = gap_ms
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        if int(following.scene_frame_index) - int(previous.scene_frame_index) > 3:
            default["reason"] = "frame_gap_too_large"
            default["gap_ms"] = gap_ms
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        previous_diagonal = math.hypot(previous.width, previous.height)
        following_diagonal = math.hypot(following.width, following.height)
        scale = max(1.0, 0.5 * (previous_diagonal + following_diagonal))
        center_motion = math.hypot(
            float(following.center.x) - float(previous.center.x),
            float(following.center.y) - float(previous.center.y),
        ) / scale
        width_ratio = following.width / max(previous.width, 1e-6)
        height_ratio = following.height / max(previous.height, 1e-6)
        if (
            center_motion > float(maximum_motion_ratio)
            or not 0.55 <= width_ratio <= 1.80
            or not 0.55 <= height_ratio <= 1.80
        ):
            default["reason"] = "implausible_track_motion"
            default["gap_ms"] = gap_ms
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        if (
            previous.explicit_entity_id
            and following.explicit_entity_id
            and previous.explicit_entity_id != following.explicit_entity_id
        ):
            default["reason"] = "explicit_entity_changed"
            default["gap_ms"] = gap_ms
            output.append(base)
            metadata[int(base.source_row_index)] = default
            continue
        if identity_by_key and target_person_policy != "all_detected":
            previous_identity = _identity_signature_for_instance(previous, identity_by_key)
            following_identity = _identity_signature_for_instance(following, identity_by_key)
            if previous_identity is None or previous_identity != following_identity:
                default["reason"] = "identity_not_stable"
                default["gap_ms"] = gap_ms
                output.append(base)
                metadata[int(base.source_row_index)] = default
                continue
        alpha = float(np.clip((timestamp_ns - t0) / float(t1 - t0), 0.0, 1.0))
        nearest_endpoint = (
            previous if abs(timestamp_ns - t0) <= abs(t1 - timestamp_ns) else following
        )
        if int(nearest_endpoint.scene_frame_index) != int(native_frame_index):
            nearest_endpoint = base
        confidence = float(
            np.clip(
                min(float(previous.confidence), float(following.confidence))
                * max(0.70, 1.0 - 0.18 * center_motion),
                0.0,
                1.0,
            )
        )
        interpolated = FaceInstance(
            scene_frame_index=int(native_frame_index),
            scene_timestamp_ns=int(timestamp_ns),
            track_id=int(base.track_id),
            explicit_entity_id=(
                previous.explicit_entity_id
                if previous.explicit_entity_id == following.explicit_entity_id
                else base.explicit_entity_id
            ),
            x1=(1.0 - alpha) * float(previous.x1) + alpha * float(following.x1),
            y1=(1.0 - alpha) * float(previous.y1) + alpha * float(following.y1),
            x2=(1.0 - alpha) * float(previous.x2) + alpha * float(following.x2),
            y2=(1.0 - alpha) * float(previous.y2) + alpha * float(following.y2),
            left_eye=_lerp_point(previous.left_eye, following.left_eye, alpha, base.left_eye),
            right_eye=_lerp_point(previous.right_eye, following.right_eye, alpha, base.right_eye),
            nose=_lerp_point(previous.nose, following.nose, alpha, base.nose),
            mouth_left=_lerp_point(previous.mouth_left, following.mouth_left, alpha, base.mouth_left),
            mouth_right=_lerp_point(previous.mouth_right, following.mouth_right, alpha, base.mouth_right),
            confidence=confidence,
            source_row_index=int(nearest_endpoint.source_row_index),
        )
        output.append(interpolated)
        metadata[int(interpolated.source_row_index)] = {
            "mode": "track_interpolated",
            "previous_frame_index": int(previous.scene_frame_index),
            "next_frame_index": int(following.scene_frame_index),
            "alpha": alpha,
            "gap_ms": gap_ms,
            "confidence": confidence,
            "reason": "same_track_safe_interpolation",
        }
    return output, metadata

def _frame_summary(samples: pd.DataFrame, frame_map: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        int(frame): rows for frame, rows in samples.groupby("native_frame_index", dropna=False)
        if pd.notna(frame)
    }
    rows = []
    for frame in frame_map.itertuples(index=False):
        native_index = int(frame.native_frame_index)
        sample_rows = grouped.get(native_index)
        if sample_rows is None or len(sample_rows) == 0:
            rows.append({
                "native_frame_index": native_index,
                "source_frame_ordinal_in_window": int(frame.source_frame_ordinal_in_window),
                "clip_time_ns": int(frame.clip_video_pts_ns),
                "gaze_sample_count": 0,
                "valid_gaze_sample_count": 0,
                "primary_face_entity": None,
                "primary_face_region": "no_gaze_sample",
                "face_region_distribution_json": "{}",
                "mean_face_confidence": 0.0,
                "review_required": False,
            })
            continue
        valid = ~sample_rows["face_region"].isin(["invalid_gaze", "out_of_frame"])
        display_column = "face_social_region" if "face_social_region" in sample_rows.columns else "face_region"
        distribution = Counter(sample_rows[display_column].fillna(sample_rows["face_region"]).astype(str))
        score: dict[tuple[str | None, str], float] = defaultdict(float)
        for row in sample_rows.itertuples(index=False):
            entity = row.face_entity_id if isinstance(row.face_entity_id, str) and row.face_entity_id else None
            social = getattr(row, "face_social_region", None) or row.face_region
            score[(entity, str(social))] += max(0.05, float(row.face_aoi_confidence or 0.0))
        primary = max(score, key=score.get) if score else (None, "unknown")
        rows.append({
            "native_frame_index": native_index,
            "source_frame_ordinal_in_window": int(frame.source_frame_ordinal_in_window),
            "clip_time_ns": int(frame.clip_video_pts_ns),
            "gaze_sample_count": int(len(sample_rows)),
            "valid_gaze_sample_count": int(valid.sum()),
            "primary_face_entity": primary[0],
            "primary_face_region": primary[1],
            "face_region_distribution_json": json.dumps(distribution, sort_keys=True, separators=(",", ":")),
            "mean_face_confidence": float(pd.to_numeric(sample_rows["face_aoi_confidence"], errors="coerce").fillna(0).mean()),
            "review_required": bool(sample_rows["review_required"].fillna(False).astype(bool).any()),
        })
    return pd.DataFrame(rows)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _full_clip_partner_face_statistics(
    samples: pd.DataFrame,
    frame_map: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarise partner-face gaze across the complete trimmed clip.

    Two denominators are deliberately reported:

    * all valid gaze samples in the clip, which answers how much of the recorded
      gaze stream lands on the partner's face or a particular facial region;
    * all samples already classified as partner-face hits, which describes the
      distribution within the partner face itself.

    A frame-based companion is also exported so users can audit the result
    without assuming a perfectly constant eye-tracker sampling rate.
    """
    frame = samples.copy()
    if frame.empty:
        frame = pd.DataFrame(index=pd.RangeIndex(0))
    region = frame.get("face_region", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    relation = frame.get("target_relation", pd.Series("", index=frame.index)).fillna("").astype(str)

    valid = ~region.isin({"invalid_gaze", "out_of_frame"})
    if {"gaze_x_px", "gaze_y_px"}.issubset(frame.columns):
        x = pd.to_numeric(frame["gaze_x_px"], errors="coerce")
        y = pd.to_numeric(frame["gaze_y_px"], errors="coerce")
        valid &= x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if "worn" in frame.columns:
        worn = frame["worn"].map(_bool_value)
        valid &= ~worn.eq(False)

    facial_regions = {
        SemanticRegion.LEFT_EYE.value,
        SemanticRegion.RIGHT_EYE.value,
        SemanticRegion.NOSE.value,
        SemanticRegion.MOUTH.value,
        SemanticRegion.OTHER_FACE.value,
        SemanticRegion.FACE.value,
    }
    partner = relation.eq("partner")
    partner_face = valid & partner & region.isin(facial_regions)
    left_eye = partner_face & region.eq(SemanticRegion.LEFT_EYE.value)
    right_eye = partner_face & region.eq(SemanticRegion.RIGHT_EYE.value)
    eyes = left_eye | right_eye
    nose = partner_face & region.eq(SemanticRegion.NOSE.value)
    mouth = partner_face & region.eq(SemanticRegion.MOUTH.value)
    other_face = partner_face & region.isin({SemanticRegion.OTHER_FACE.value, SemanticRegion.FACE.value})

    masks = {
        "partner_face": partner_face,
        "partner_eyes": eyes,
        "partner_left_eye": left_eye,
        "partner_right_eye": right_eye,
        "partner_nose": nose,
        "partner_mouth": mouth,
        "partner_other_face": other_face,
    }
    valid_count = int(valid.sum())
    partner_face_count = int(partner_face.sum())

    sample_based: dict[str, Any] = {
        "denominator": "valid_gaze_samples",
        "denominator_count": valid_count,
    }
    within_partner: dict[str, Any] = {
        "denominator": "partner_face_samples",
        "denominator_count": partner_face_count,
    }
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        count = int(mask.sum())
        sample_based[name] = {
            "count": count,
            "fraction": _safe_fraction(count, valid_count),
        }
        rows.append({
            "scope": "sample_based_full_clip",
            "metric": name,
            "count": count,
            "denominator": "valid_gaze_samples",
            "denominator_count": valid_count,
            "fraction": _safe_fraction(count, valid_count),
        })
        if name != "partner_face":
            within_partner[name] = {
                "count": count,
                "fraction": _safe_fraction(count, partner_face_count),
            }
            rows.append({
                "scope": "within_partner_face",
                "metric": name,
                "count": count,
                "denominator": "partner_face_samples",
                "denominator_count": partner_face_count,
                "fraction": _safe_fraction(count, partner_face_count),
            })

    clip_frame_values = pd.to_numeric(
        frame_map.get("native_frame_index", pd.Series(dtype=float)), errors="coerce"
    ).dropna().astype(int)
    clip_frames = set(clip_frame_values.tolist())
    clip_frame_count = int(len(clip_frame_values))
    sample_native = pd.to_numeric(
        frame.get("native_frame_index", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    valid_gaze_frames = set(sample_native.loc[valid & sample_native.notna()].astype(int).tolist())
    if clip_frames:
        valid_gaze_frames &= clip_frames
    valid_gaze_frame_count = int(len(valid_gaze_frames))
    frame_based: dict[str, Any] = {
        "clip_frame_count": clip_frame_count,
        "valid_gaze_frame_count": valid_gaze_frame_count,
        "denominator": "all_clip_frames",
    }
    for name, mask in masks.items():
        hit_frames = set(sample_native.loc[mask & sample_native.notna()].astype(int).tolist())
        if clip_frames:
            hit_frames &= clip_frames
        count = int(len(hit_frames))
        frame_based[name] = {
            "frame_count": count,
            "fraction_of_all_clip_frames": _safe_fraction(count, clip_frame_count),
            "fraction_of_valid_gaze_frames": _safe_fraction(count, valid_gaze_frame_count),
        }
        rows.append({
            "scope": "frame_based_full_clip",
            "metric": name,
            "count": count,
            "denominator": "all_clip_frames",
            "denominator_count": clip_frame_count,
            "fraction": _safe_fraction(count, clip_frame_count),
        })

    summary = {
        "schema_version": "oculostudio-face-aoi-clip-summary-1.0",
        "sample_based": sample_based,
        "within_partner_face": within_partner,
        "frame_based": frame_based,
    }
    table = pd.DataFrame(rows)
    if len(table):
        table["percent"] = table["fraction"] * 100.0
    else:
        table = pd.DataFrame(columns=[
            "scope", "metric", "count", "denominator", "denominator_count", "fraction", "percent",
        ])
    return summary, table


def _clip_statistics_qc_fields(summary: dict[str, Any]) -> dict[str, Any]:
    sample = summary.get("sample_based") or {}
    within = summary.get("within_partner_face") or {}
    frames = summary.get("frame_based") or {}

    def sample_fraction(name: str) -> float:
        return float((sample.get(name) or {}).get("fraction") or 0.0)

    def sample_count(name: str) -> int:
        return int((sample.get(name) or {}).get("count") or 0)

    def within_fraction(name: str) -> float:
        return float((within.get(name) or {}).get("fraction") or 0.0)

    def frame_fraction(name: str) -> float:
        return float((frames.get(name) or {}).get("fraction_of_all_clip_frames") or 0.0)

    return {
        "valid_gaze_sample_count": int(sample.get("denominator_count") or 0),
        "partner_face_sample_count": sample_count("partner_face"),
        "partner_face_sample_fraction": sample_fraction("partner_face"),
        "partner_eye_sample_count": sample_count("partner_eyes"),
        "partner_eye_sample_fraction": sample_fraction("partner_eyes"),
        "partner_left_eye_sample_fraction": sample_fraction("partner_left_eye"),
        "partner_right_eye_sample_fraction": sample_fraction("partner_right_eye"),
        "partner_nose_sample_count": sample_count("partner_nose"),
        "partner_nose_sample_fraction": sample_fraction("partner_nose"),
        "partner_mouth_sample_count": sample_count("partner_mouth"),
        "partner_mouth_sample_fraction": sample_fraction("partner_mouth"),
        "partner_other_face_sample_fraction": sample_fraction("partner_other_face"),
        "partner_eye_share_within_partner_face": within_fraction("partner_eyes"),
        "partner_nose_share_within_partner_face": within_fraction("partner_nose"),
        "partner_mouth_share_within_partner_face": within_fraction("partner_mouth"),
        "partner_other_face_share_within_partner_face": within_fraction("partner_other_face"),
        "clip_frame_count": int(frames.get("clip_frame_count") or 0),
        "valid_gaze_frame_count": int(frames.get("valid_gaze_frame_count") or 0),
        "partner_face_frame_fraction": frame_fraction("partner_face"),
        "partner_eye_frame_fraction": frame_fraction("partner_eyes"),
        "partner_nose_frame_fraction": frame_fraction("partner_nose"),
        "partner_mouth_frame_fraction": frame_fraction("partner_mouth"),
    }


def _draw_face_overlay(
    *,
    privacy_artifact,
    output_root: Path,
    samples: pd.DataFrame,
    regions: pd.DataFrame,
    label: str,
    preserve_audio: bool,
    crf: int,
    preset: str,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    progress,
    token: CancellationToken,
) -> tuple[Path, Path, dict]:
    # Check the encoder before decoding/rendering a potentially long clip.
    encoder = privacy_encoder_preflight(
        ffmpeg_path,
        preserve_audio=preserve_audio,
        preset=preset,
        crf=crf,
        ffprobe_path=ffprobe_path,
    )
    video = find_video_file(privacy_artifact, preferred_tokens=("privacy",))
    frame_map_path = find_artifact_file(privacy_artifact, names=["privacy_trimmed_frame_map.csv"])
    report_path = find_artifact_file(privacy_artifact, names=["privacy_clip_report.json"])
    report = _read_json(report_path)
    frame_map = pd.read_csv(frame_map_path, low_memory=False)
    required = {"output_frame_index", "native_frame_index"}
    if not required.issubset(frame_map.columns):
        raise ValidationError("Privacy clip frame map is incomplete")
    fps = float(report.get("output", {}).get("fps") or 30.0)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode Privacy clip: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Privacy clip dimensions are unavailable")
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) not in {0, len(frame_map)}:
        cap.release()
        raise ValidationError("Privacy clip video and frame map have different frame counts")

    region_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in regions.to_dict("records"):
        region_by_frame[int(row["scene_frame_index"])].append(row)
    gaze_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in samples.loc[samples["native_frame_index"].notna()].to_dict("records"):
        gaze_by_frame[int(row["native_frame_index"])].append(row)

    silent = output_root / "face_aoi_overlay_silent.mp4"
    overlay_video = output_root / f"{safe_name(label, fallback='face_aoi')}_overlay.mp4"
    overlay_map = output_root / "face_aoi_overlay_frame_map.csv"
    writer = _output_writer(silent, width, height, fps)
    map_rows = []
    try:
        for ordinal, frame_row in enumerate(frame_map.itertuples(index=False)):
            token.raise_if_cancelled()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ProcessingError(f"Privacy clip decode ended at frame {ordinal}")
            native_index = int(frame_row.native_frame_index)
            for region in region_by_frame.get(native_index, []):
                if "include_in_primary_analysis" in region and not bool(region.get("include_in_primary_analysis")):
                    continue
                try:
                    geometry = json.loads(region["geometry_json"])
                    geometry_type = str(region.get("geometry_type") or "ellipse")
                    if geometry_type == "polygon" or geometry.get("points"):
                        points = np.asarray(geometry.get("points") or [], dtype=float)
                        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
                            continue
                        polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [polygon], True, (45, 245, 230), 2, cv2.LINE_AA)
                        center = tuple(np.rint(points.mean(axis=0)).astype(int).tolist())
                    else:
                        center = (int(round(geometry["center_x"])), int(round(geometry["center_y"])))
                        axes = (max(1, int(round(geometry["radius_x"]))), max(1, int(round(geometry["radius_y"]))))
                        cv2.ellipse(frame, center, axes, 0, 0, 360, (45, 245, 230), 2, cv2.LINE_AA)
                    label_text = str(region.get("face_social_region") or region["region"])
                    cv2.putText(frame, label_text, (center[0]+4, center[1]-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45,245,230), 1, cv2.LINE_AA)
                except Exception:
                    continue
            gaze_rows = gaze_by_frame.get(native_index, [])
            for gaze in gaze_rows:
                if np.isfinite([gaze.get("gaze_x_px", np.nan), gaze.get("gaze_y_px", np.nan)]).all():
                    point = (int(round(gaze["gaze_x_px"])), int(round(gaze["gaze_y_px"])))
                    cv2.circle(frame, point, 5, (30, 255, 110), -1, cv2.LINE_AA)
            primary = None
            if gaze_rows:
                counts = Counter(str(row.get("face_social_region") or row.get("face_region")) for row in gaze_rows)
                primary = counts.most_common(1)[0][0]
                cv2.rectangle(frame, (10, 10), (290, 46), (22, 26, 20), -1)
                cv2.putText(frame, f"Face AOI: {primary}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (225, 248, 47), 2, cv2.LINE_AA)
            writer.write(frame)
            mapped = frame_row._asdict()
            mapped.update({
                "overlay_output_frame_index": ordinal,
                "primary_face_region": primary,
                "gaze_sample_count": len(gaze_rows),
            })
            map_rows.append(mapped)
            if ordinal % 20 == 0:
                emit(progress, 0.72 + 0.17 * (ordinal+1)/max(1,len(frame_map)),
                     f"Rendering AOI overlay ({ordinal+1}/{len(frame_map)})")
    finally:
        cap.release()
        writer.release()
    pd.DataFrame(map_rows).to_csv(overlay_map, index=False)
    _mux_audio_and_transcode(
        silent,
        video,
        overlay_video,
        source_start_sec=0.0,
        duration_sec=len(frame_map)/fps,
        preserve_audio=preserve_audio,
        crf=crf,
        preset=preset,
        ffmpeg_path=encoder["ffmpeg"]["path"],
        expected_frames=len(frame_map),
        cancellation=token,
        progress=progress,
    )
    silent.unlink(missing_ok=True)
    return overlay_video, overlay_map, {"fps": fps, "frame_count": len(frame_map), "source_video": str(video)}


def _validate_privacy_clip_matches_trim(privacy_artifact, trim_manifest: dict, trim_frame_map: pd.DataFrame) -> None:
    """Fail closed when a blurred clip does not represent the same NRD cut."""
    privacy_report = _read_json(find_artifact_file(privacy_artifact, names=["privacy_clip_report.json"]))
    privacy_map = pd.read_csv(
        find_artifact_file(privacy_artifact, names=["privacy_trimmed_frame_map.csv"]),
        low_memory=False,
    )
    expected_window = trim_manifest.get("window") or {}
    actual_window = privacy_report.get("window") or {}
    try:
        expected = (int(expected_window["start_media_ns"]), int(expected_window["end_media_ns"]))
        actual = (int(actual_window["start_media_ns"]), int(actual_window["end_media_ns"]))
    except Exception as exc:
        raise ValidationError("Could not verify Privacy clip and Native-reference trim windows") from exc
    if actual != expected:
        raise ValidationError(
            "The selected Privacy clip uses a different NRD start/duration than the selected data trim"
        )
    if "native_frame_index" not in privacy_map.columns or "native_frame_index" not in trim_frame_map.columns:
        raise ValidationError("Privacy/data trim frame maps are missing Native frame indices")
    allowed = set(pd.to_numeric(trim_frame_map["native_frame_index"], errors="coerce").dropna().astype(int))
    used = set(pd.to_numeric(privacy_map["native_frame_index"], errors="coerce").dropna().astype(int))
    if not used or not used.issubset(allowed):
        raise ValidationError("Privacy clip frame mapping is not contained in the selected Native-reference trim")


def create_face_aoi_annotations(
    store: ProjectStore,
    *,
    session_id: str,
    native_trim_artifact_id: str,
    participant_association_artifact_id: str | None = None,
    facemapper_filter_artifact_id: str | None = None,
    target_person_policy: Literal["partner_only", "tracked_participants", "all_detected"] = "all_detected",
    privacy_clip_artifact_id: str | None = None,
    label: str = "Face AOI",
    face_tolerance_ms: float = 8.0,
    face_left_right_policy: Literal["canonical_filter", "anatomical_auto", "as_exported", "swap_eyes"] = "canonical_filter",
    face_landmark_geometry_policy: Literal["as_exported", "auto_3d", "mirror_x"] = "as_exported",
    fine_feature_aoi_scale: float = 1.55,
    eye_feature_aoi_scale: float = 1.35,
    nose_region_mode: Literal["nasal_envelope", "bridge_polygon", "ellipse"] = "nasal_envelope",
    mouth_center_policy: Literal["adaptive_lip_envelope", "anatomical_strict", "speech_sensitive_mouth_jaw", "landmark_midpoint", "lowered_landmark_midpoint", "nose_base_to_face_bottom"] = "adaptive_lip_envelope",
    face_edge_tolerance_policy: Literal["exact", "edge_inclusive", "forgiving"] = "edge_inclusive",
    auto_review_policy: Literal["off", "conservative", "adaptive", "balanced"] = "adaptive",
    face_geometry_time_policy: Literal["track_interpolated", "nearest_frame"] = "track_interpolated",
    face_interpolation_max_gap_ms: float = 75.0,
    face_interpolation_max_motion_ratio: float = 1.25,
    output_parquet: bool = False,
    render_overlay: bool = True,
    preserve_audio: bool = True,
    crf: int = 20,
    preset: str = "fast",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    trim_artifact = session.artifacts.get(native_trim_artifact_id)
    if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    trim_root, manifest = _trim_root(trim_artifact)
    tsd_root = trim_root / "tsd"
    fm_root = trim_root / "face_mapper"
    gaze_path = tsd_root / "gaze.csv"
    if not gaze_path.exists():
        raise ValidationError("Trim artifact contains no tsd/gaze.csv")
    frame_map_path = trim_root / "native_frame_map.csv"
    full_clock_path = trim_root / "native_clock_full.csv"
    if not frame_map_path.exists() or not full_clock_path.exists():
        raise ValidationError("Trim artifact is missing Native frame maps")
    frame_map = pd.read_csv(frame_map_path, low_memory=False)
    full_clock = pd.read_csv(full_clock_path, low_memory=False)
    scene_timestamps = pd.to_numeric(full_clock["native_capture_utc_ns"], errors="raise").astype("int64").to_numpy()
    native_meta = manifest.get("native_reference") or {}
    width = int(native_meta.get("width") or 0)
    height = int(native_meta.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValidationError("Trim manifest has no valid Native scene resolution")
    recording_id = str(manifest.get("recording_id") or "")
    if target_person_policy not in {"partner_only", "tracked_participants", "all_detected"}:
        raise ValidationError("target_person_policy must be partner_only, tracked_participants or all_detected")
    participant_association = None
    identity_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    projections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    association_manifest: dict[str, Any] | None = None
    if participant_association_artifact_id:
        association_artifact = session.artifacts.get(participant_association_artifact_id)
        if association_artifact is None or association_artifact.kind != ArtifactKind.PARTICIPANT_ASSOCIATION:
            raise ValidationError("Select a participant-association artifact")
        projections_df, identities_df, association_manifest = load_participant_association(association_artifact)
        if association_manifest.get("native_trim_artifact_id") != native_trim_artifact_id:
            raise ValidationError(f"Participant association uses trim {association_manifest.get('native_trim_artifact_id')}; Face AOI selected {native_trim_artifact_id}. Regenerate or select the association from the exact same role/trim.")
        for projection_row in projections_df.to_dict("records"):
            try:
                projections_by_frame[int(projection_row["native_frame_index"])].append(projection_row)
            except Exception:
                continue
        for identity_row in identities_df.to_dict("records"):
            try:
                key = (int(identity_row["native_frame_index"]), int(identity_row["face_source_row_index"]))
            except Exception:
                continue
            identity_by_key[key] = identity_row
        participant_association = association_artifact
    elif target_person_policy != "all_detected":
        raise ValidationError("Partner/tracked-only Face AOI requires a participant-association artifact")

    filter_artifact = None
    if facemapper_filter_artifact_id:
        filter_artifact = session.artifacts.get(facemapper_filter_artifact_id)
        if filter_artifact is None or filter_artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
            raise ValidationError("Select a FaceMapper QC/filter artifact")
        if native_trim_artifact_id not in filter_artifact.parent_artifact_ids:
            raise ValidationError("FaceMapper filter and Face AOI must use the same Native-reference trim")
        from oculostudio.services.facemapper_filter import filtered_facemapper_root
        fm_root = filtered_facemapper_root(filter_artifact, variant="analysis_strict")

    try:
        fine_feature_aoi_scale = float(fine_feature_aoi_scale)
    except (TypeError, ValueError) as exc:
        raise ValidationError("fine_feature_aoi_scale must be numeric") from exc
    if not 0.75 <= fine_feature_aoi_scale <= 2.00:
        raise ValidationError("fine_feature_aoi_scale must be between 0.75 and 2.00")
    try:
        eye_feature_aoi_scale = float(eye_feature_aoi_scale)
    except (TypeError, ValueError) as exc:
        raise ValidationError("eye_feature_aoi_scale must be numeric") from exc
    if not 0.75 <= eye_feature_aoi_scale <= 1.75:
        raise ValidationError("eye_feature_aoi_scale must be between 0.75 and 1.75")
    if nose_region_mode not in {"nasal_envelope", "bridge_polygon", "ellipse"}:
        raise ValidationError("nose_region_mode must be nasal_envelope, bridge_polygon or ellipse")
    if mouth_center_policy not in {"adaptive_lip_envelope", "anatomical_strict", "speech_sensitive_mouth_jaw", "landmark_midpoint", "lowered_landmark_midpoint", "nose_base_to_face_bottom"}:
        raise ValidationError("mouth_center_policy must be adaptive_lip_envelope, anatomical_strict, speech_sensitive_mouth_jaw, landmark_midpoint, lowered_landmark_midpoint or nose_base_to_face_bottom")
    if face_edge_tolerance_policy not in {"exact", "edge_inclusive", "forgiving"}:
        raise ValidationError("face_edge_tolerance_policy must be exact, edge_inclusive or forgiving")
    if auto_review_policy not in AUTO_REVIEW_POLICIES:
        raise ValidationError("auto_review_policy must be off, conservative, adaptive or balanced")
    if face_geometry_time_policy not in {"track_interpolated", "nearest_frame"}:
        raise ValidationError(
            "face_geometry_time_policy must be track_interpolated or nearest_frame"
        )
    if not 10.0 <= float(face_interpolation_max_gap_ms) <= 500.0:
        raise ValidationError("face_interpolation_max_gap_ms must be between 10 and 500")
    if not 0.25 <= float(face_interpolation_max_motion_ratio) <= 5.0:
        raise ValidationError(
            "face_interpolation_max_motion_ratio must be between 0.25 and 5.0"
        )
    edge_tolerance_profiles = {
        "exact": {"ratio": 0.0, "minimum_px": 0.0, "maximum_px": 0.0},
        "edge_inclusive": {"ratio": 0.045, "minimum_px": 6.0, "maximum_px": 18.0},
        "forgiving": {"ratio": 0.075, "minimum_px": 8.0, "maximum_px": 26.0},
    }
    edge_profile = edge_tolerance_profiles[face_edge_tolerance_policy]
    aoi_settings = {
        "bounded_features": True,
        "feature_scale": fine_feature_aoi_scale,
        "eye_feature_scale": eye_feature_aoi_scale,
        "nose_region_mode": nose_region_mode,
        "mouth_center_policy": mouth_center_policy,
        "mouth_center_offset_ratio": 0.12,
        # Balanced lip-envelope defaults. These are deliberately independent
        # of the global fine-feature scale so enlarging the eyes/nose cannot
        # silently turn the mouth into a lower-face AOI.
        "mouth_lateral_padding_ratio": 0.15,
        "mouth_top_padding_face_ratio": 0.045,
        "mouth_bottom_padding_face_ratio": 0.095,
        "mouth_max_height_face_ratio": 0.18,
        "mouth_absolute_max_height_face_ratio": 0.22,
        "mouth_max_width_face_ratio": 0.52,
        "mouth_bottom_limit_face_ratio": 0.90,
        "mouth_min_top_below_nose_ratio": 0.015,
        "mouth_strict_radius_x_face_ratio": 0.14,
        "mouth_strict_radius_y_face_ratio": 0.055,
        "nose_base_offset_ratio": 0.30,
        "nasal_envelope_top_y_fraction": 0.16,
        "nasal_envelope_top_halfwidth_iod_ratio": 0.16,
        "nasal_envelope_top_halfwidth_face_ratio": 0.055,
        "nasal_envelope_mid_halfwidth_iod_ratio": 0.22,
        "nasal_envelope_mid_halfwidth_face_ratio": 0.075,
        "nasal_envelope_bottom_halfwidth_iod_ratio": 0.30,
        "nasal_envelope_bottom_halfwidth_face_ratio": 0.12,
        "nasal_envelope_bottom_halfwidth_mouth_ratio": 0.42,
        "face_margin_ratio": 0.02,
        # Coarse face membership can absorb bounded gaze-localisation error at
        # the visible green-box edge. Fine eye/nose/mouth geometry is unchanged.
        "face_edge_tolerance_policy": face_edge_tolerance_policy,
        "face_edge_tolerance_ratio": edge_profile["ratio"],
        "face_edge_tolerance_min_px": edge_profile["minimum_px"],
        "face_edge_tolerance_max_px": edge_profile["maximum_px"],
    }
    nose_geometry_model = {
        "nasal_envelope": "nasal_bridge_alar_envelope",
        "bridge_polygon": "eye_line_to_nose_base_bridge_legacy",
        "ellipse": "bounded_ellipse",
    }[nose_region_mode]
    mouth_geometry_model = {
        "adaptive_lip_envelope": "adaptive_lip_envelope",
        "anatomical_strict": "bounded_lip_ellipse",
        "speech_sensitive_mouth_jaw": "speech_sensitive_mouth_jaw",
        "nose_base_to_face_bottom": "speech_sensitive_mouth_jaw",
        "lowered_landmark_midpoint": "bounded_ellipse",
        "landmark_midpoint": "bounded_ellipse",
    }[mouth_center_policy]
    mouth_semantic_scope = (
        "mouth_and_jaw"
        if mouth_center_policy in {"speech_sensitive_mouth_jaw", "nose_base_to_face_bottom"}
        else "lips"
    )

    emit(progress, 0.04, "Loading FaceMapper landmarks with expanded eyes and balanced bounded facial AOIs")
    canonical_filter_manifest: dict[str, Any] | None = None
    canonical_filter_ready = False
    if filter_artifact is not None:
        manifest_path = find_artifact_file(filter_artifact, names=["filter_manifest.json"], required=False)
        if manifest_path is not None:
            canonical_filter_manifest = _read_json(manifest_path)
            canonical_filter_ready = (
                canonical_filter_manifest.get("canonical_bilateral_label_policy") == "subject_anatomical"
                and canonical_filter_manifest.get("canonical_geometry_policy") == "source_pixels_preserved"
                and not bool(canonical_filter_manifest.get("geometry_mirror_applied"))
            )

    if face_left_right_policy == "canonical_filter":
        if filter_artifact is None:
            raise ValidationError(
                "Canonical anatomical Face AOI requires a v0.21.22+ FaceMapper filter artifact"
            )
        if not canonical_filter_ready:
            raise ValidationError(
                "The selected FaceMapper filter predates canonical bilateral semantics. "
                "Re-run FaceMapper QC/filter in v0.21.22 and select the new filter artifact."
            )
        if face_landmark_geometry_policy != "as_exported":
            raise ValidationError(
                "Canonical FaceMapper filters preserve source pixel geometry; geometry mirroring is not permitted"
            )
        load_lr_policy = "as_exported"
    else:
        # Legacy/raw compatibility path.  Modern workflows should use the
        # canonical filter artifact so bilateral semantics are normalised once.
        load_lr_policy = "as_exported" if participant_association is not None else face_left_right_policy

    face_result = load_face_semantics(
        fm_root,
        recording_id=recording_id or None,
        scene_timestamps_ns=scene_timestamps,
        width=width,
        height=height,
        tolerance_ms=face_tolerance_ms,
        aoi_settings=aoi_settings,
        left_right_policy=load_lr_policy,
    )
    if face_left_right_policy == "canonical_filter":
        orientation_by_track = {
            int(instance.track_id): {
                "mirror_geometry_x": False,
                "swap_bilateral_labels": False,
                "source": "canonical_facemapper_filter",
                "confidence": 1.0,
                "evidence_count": 0,
                "best_cost": None,
                "second_cost": None,
                "verified": True,
            }
            for instances in face_result.instances_by_frame.values()
            for instance in instances
        }
    else:
        orientation_by_track = _lock_partner_orientation_by_track(
            face_result.instances_by_frame,
            identity_by_key,
            geometry_policy=face_landmark_geometry_policy,
            left_right_policy=face_left_right_policy,
        )
    corrected_instances_by_frame: dict[int, list[FaceInstance]] = {}
    geometry_mirror_by_key: dict[tuple[int, int], bool] = {}
    bilateral_swap_by_key: dict[tuple[int, int], bool] = {}
    anatomical_source_by_key: dict[tuple[int, int], str] = {}
    anatomical_confidence_by_key: dict[tuple[int, int], float] = {}
    anatomical_evidence_by_key: dict[tuple[int, int], int] = {}
    anatomical_verified_by_key: dict[tuple[int, int], bool] = {}
    for frame_index, frame_instances in face_result.instances_by_frame.items():
        corrected: list[FaceInstance] = []
        for instance in frame_instances:
            key = (int(frame_index), int(instance.source_row_index))
            locked = orientation_by_track.get(int(instance.track_id), {
                "mirror_geometry_x": face_landmark_geometry_policy == "mirror_x",
                "swap_bilateral_labels": face_left_right_policy == "swap_eyes",
                "source": "untracked_export_geometry",
                "confidence": 0.0,
                "evidence_count": 0,
                "verified": False,
            })
            oriented = _apply_face_orientation(
                instance,
                mirror_geometry_x=bool(locked["mirror_geometry_x"]),
                swap_bilateral_labels=bool(locked["swap_bilateral_labels"]),
            )
            corrected.append(oriented)
            geometry_mirror_by_key[key] = bool(locked["mirror_geometry_x"])
            bilateral_swap_by_key[key] = bool(locked["swap_bilateral_labels"])
            anatomical_source_by_key[key] = str(locked["source"])
            anatomical_confidence_by_key[key] = float(locked.get("confidence") or 0.0)
            anatomical_evidence_by_key[key] = int(locked.get("evidence_count") or 0)
            anatomical_verified_by_key[key] = bool(locked.get("verified"))
        corrected_instances_by_frame[int(frame_index)] = corrected
    # Critical parity fix: rebuild overlay geometries from the corrected face
    # instances.  Merely renaming left/right regions leaves the ellipse centres
    # at their old mirrored coordinates.
    corrected_regions = face_instances_to_regions(corrected_instances_by_frame, aoi_settings=aoi_settings)
    temporal_track_index = _face_track_index(corrected_instances_by_frame)
    face_result.diagnostics["face_landmark_geometry_policy"] = face_landmark_geometry_policy
    face_result.diagnostics["participant_3d_geometry_mirror_count"] = int(sum(geometry_mirror_by_key.values()))
    face_result.diagnostics["participant_3d_bilateral_swap_count"] = int(sum(bilateral_swap_by_key.values()))
    face_result.diagnostics["participant_3d_anatomical_verification_count"] = int(sum(anatomical_verified_by_key.values()))
    face_result.diagnostics["participant_3d_orientation_track_count"] = int(len(orientation_by_track))
    face_result.diagnostics["participant_3d_orientation_tracks"] = {str(key): value for key, value in orientation_by_track.items()}
    face_result.diagnostics["canonical_filter_ready"] = bool(canonical_filter_ready)
    face_result.diagnostics["source_pixel_geometry_preserved"] = bool(face_left_right_policy == "canonical_filter")
    face_result.diagnostics["canonical_bilateral_convention"] = (
        "subject_anatomical" if face_left_right_policy == "canonical_filter" else None
    )
    face_result.diagnostics["canonical_filter_manifest_schema"] = (
        canonical_filter_manifest.get("schema_version") if canonical_filter_manifest else None
    )
    face_result.diagnostics["fine_feature_aoi_scale"] = fine_feature_aoi_scale
    face_result.diagnostics["eye_feature_aoi_scale"] = eye_feature_aoi_scale
    face_result.diagnostics["effective_eye_aoi_scale"] = fine_feature_aoi_scale * eye_feature_aoi_scale
    face_result.diagnostics["nose_region_mode"] = nose_region_mode
    face_result.diagnostics["nose_region_geometry_model"] = nose_geometry_model
    face_result.diagnostics["mouth_center_policy"] = mouth_center_policy
    face_result.diagnostics["mouth_center_offset_ratio"] = aoi_settings["mouth_center_offset_ratio"]
    face_result.diagnostics["mouth_region_geometry_model"] = mouth_geometry_model
    face_result.diagnostics["mouth_semantic_scope"] = mouth_semantic_scope
    face_result.diagnostics["mouth_lateral_padding_ratio"] = aoi_settings["mouth_lateral_padding_ratio"]
    face_result.diagnostics["mouth_max_height_face_ratio"] = aoi_settings["mouth_max_height_face_ratio"]
    face_result.diagnostics["mouth_bottom_limit_face_ratio"] = aoi_settings["mouth_bottom_limit_face_ratio"]
    face_result.diagnostics["aoi_coordinate_source"] = "accepted_facemapper_green_face_box"
    face_result.diagnostics["participant_projection_usage"] = "identity_gate_only"
    face_result.diagnostics["full_face_geometry_model"] = "face_bbox_exact_rectangle_plus_coarse_edge_band"
    face_result.diagnostics["face_membership_policy"] = "accepted_green_box_with_bounded_edge_tolerance"
    face_result.diagnostics["face_edge_tolerance_policy"] = face_edge_tolerance_policy
    face_result.diagnostics["face_edge_tolerance_ratio"] = aoi_settings["face_edge_tolerance_ratio"]
    face_result.diagnostics["face_edge_tolerance_min_px"] = aoi_settings["face_edge_tolerance_min_px"]
    face_result.diagnostics["face_edge_tolerance_max_px"] = aoi_settings["face_edge_tolerance_max_px"]
    face_result.diagnostics["fine_feature_edge_expansion"] = False
    face_result.diagnostics["facemapper_gaze_on_face_usage"] = "diagnostic_only"
    face_result.diagnostics["temporal_face_promotion_outside_green_box"] = False
    face_result.diagnostics["direct_green_box_review_policy"] = "auto_accept_matched_identity"
    face_result.diagnostics["fine_feature_containment_policy"] = "face_bbox_inner_clip"
    face_result.diagnostics["fine_feature_face_margin_ratio"] = aoi_settings["face_margin_ratio"]
    face_result.diagnostics["face_geometry_time_policy"] = face_geometry_time_policy
    face_result.diagnostics["face_interpolation_max_gap_ms"] = float(face_interpolation_max_gap_ms)
    face_result.diagnostics["face_interpolation_max_motion_ratio"] = float(face_interpolation_max_motion_ratio)
    face_result.diagnostics["face_geometry_observation_clock"] = "native_video_frames"
    face_result.diagnostics["face_geometry_classification_clock"] = "gaze_sample_timestamps"
    gaze = pd.read_csv(gaze_path, low_memory=False)
    timestamp_col = _column(gaze.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"])
    x_col = _column(gaze.columns, ["gaze x [px]", "gaze_x_px", "gaze x", "x"])
    y_col = _column(gaze.columns, ["gaze y [px]", "gaze_y_px", "gaze y", "y"])
    frame_col = _column(gaze.columns, ["oculo_nearest_native_frame_index"])
    source_row_col = _column(gaze.columns, ["oculo_source_row_index"], required=False)
    clip_time_col = _column(gaze.columns, ["oculo_clip_video_ns"], required=False)
    worn_col = _column(gaze.columns, ["worn"], required=False)
    blink_col = _column(gaze.columns, ["blink id", "blink_id"], required=False)
    fixation_col = _column(gaze.columns, ["fixation id", "fixation_id"], required=False)
    frame_error_col = _column(gaze.columns, ["oculo_frame_match_error_ns"], required=False)

    # Access timestamp/source-index columns directly. Iterating a mixed numeric
    # pandas row can coerce an epoch-nanosecond int64 to float64 and silently
    # lose tens or hundreds of nanoseconds.
    timestamp_values = pd.to_numeric(gaze[timestamp_col], errors="coerce").astype("Int64")
    source_row_values = (
        pd.to_numeric(gaze[source_row_col], errors="coerce").astype("Int64")
        if source_row_col else None
    )
    native_frame_values = pd.to_numeric(gaze[frame_col], errors="coerce").astype("Int64")
    x_values = pd.to_numeric(gaze[x_col], errors="coerce")
    y_values = pd.to_numeric(gaze[y_col], errors="coerce")
    clip_time_values = (
        pd.to_numeric(gaze[clip_time_col], errors="coerce").astype("Int64")
        if clip_time_col else None
    )
    frame_error_values = pd.to_numeric(gaze[frame_error_col], errors="coerce") if frame_error_col else None

    on_face_by_timestamp: dict[int, list[bool | None]] = defaultdict(list)
    on_face_path = fm_root / "gaze_on_face.csv"
    if on_face_path.exists():
        on_face = pd.read_csv(on_face_path, low_memory=False)
        fm_timestamp_col = _column(on_face.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"])
        fm_flag_col = _column(on_face.columns, ["gaze on face", "gaze_on_face"])
        for timestamp, flag in zip(on_face[fm_timestamp_col], on_face[fm_flag_col]):
            try:
                on_face_by_timestamp[int(timestamp)].append(_bool_value(flag))
            except Exception:
                continue

    emit(progress, 0.22, "Classifying each gaze sample against facial AOIs")
    rows: list[dict[str, Any]] = []
    for ordinal in range(len(gaze)):
        token.raise_if_cancelled()
        if pd.isna(timestamp_values.iloc[ordinal]):
            # Trimmed TSD should normally contain valid timestamps, but preserve
            # a deterministic invalid row instead of crashing or dropping it.
            timestamp = -1
        else:
            timestamp = int(timestamp_values.iloc[ordinal])
        source_row_index = (
            int(source_row_values.iloc[ordinal])
            if source_row_values is not None and pd.notna(source_row_values.iloc[ordinal])
            else ordinal
        )
        native_frame = int(native_frame_values.iloc[ordinal]) if pd.notna(native_frame_values.iloc[ordinal]) else -1
        x = float(x_values.iloc[ordinal]) if pd.notna(x_values.iloc[ordinal]) else np.nan
        y = float(y_values.iloc[ordinal]) if pd.notna(y_values.iloc[ordinal]) else np.nan
        worn = _bool_value(gaze[worn_col].iloc[ordinal]) if worn_col else True
        fm_values = on_face_by_timestamp.get(timestamp, [])
        (
            fm_on_face,
            fm_on_face_conflict,
            fm_on_face_duplicate_consensus,
            fm_on_face_unique_value_count,
        ) = _facemapper_on_face_consensus(fm_values)
        instances, temporal_metadata = _interpolate_face_instances_for_sample(
            timestamp_ns=timestamp,
            native_frame_index=native_frame,
            instances_by_frame=corrected_instances_by_frame,
            track_index=temporal_track_index,
            identity_by_key=identity_by_key,
            target_person_policy=target_person_policy,
            temporal_policy=face_geometry_time_policy,
            maximum_gap_ms=float(face_interpolation_max_gap_ms),
            maximum_motion_ratio=float(face_interpolation_max_motion_ratio),
        )
        edge_eligible_source_rows: set[int] | None = None
        if participant_association is not None and target_person_policy != "all_detected":
            edge_eligible_source_rows = set()
            for instance in instances:
                identity_row = identity_by_key.get((native_frame, int(instance.source_row_index)))
                if not identity_row:
                    continue
                matched_entity = _present_text(identity_row.get("matched_entity_id"))
                identity_state = _present_text(identity_row.get("identity_status")) or ""
                relation = _present_text(identity_row.get("target_relation"))
                matched = identity_state in MATCHED_IDENTITY_STATUSES and matched_entity is not None
                if not matched:
                    continue
                if target_person_policy == "partner_only" and relation == "partner":
                    edge_eligible_source_rows.add(int(instance.source_row_index))
                elif target_person_policy == "tracked_participants" and relation in {"partner", "self"}:
                    edge_eligible_source_rows.add(int(instance.source_row_index))
        hit = classify_face_point(
            x, y, instances, aoi_settings=aoi_settings,
            edge_eligible_source_row_indices=edge_eligible_source_rows,
        )
        selected_temporal = (
            temporal_metadata.get(int(hit.source_row_index)) if hit is not None else None
        )
        candidate_temporal_mode = (
            "track_interpolated"
            if any(row.get("mode") == "track_interpolated" for row in temporal_metadata.values())
            else "nearest_frame"
        )
        edge_hit = bool(hit is not None and hit.gate_hit_type == "green_box_edge_tolerance")
        # Projected participant-head boxes are identity gates for green face
        # boxes only. They are never queried with the gaze point.
        projection_fallback = None
        identity = None
        raw_region = hit.region if hit is not None else None
        raw_entity = hit.entity_id if hit is not None else None
        identity_status = None
        target_relation = None
        identity_confidence = None
        include_in_primary_analysis = False
        review = False
        if hit is not None and identity_by_key:
            identity = identity_by_key.get((native_frame, int(hit.source_row_index)))
        if worn is False:
            region = SemanticRegion.INVALID_GAZE.value
            status = "not_worn"
            confidence = 1.0
            entity = None
            track_id = None
        elif not np.isfinite([x, y]).all() or x < 0 or y < 0 or x >= width or y >= height:
            region = SemanticRegion.OUT_OF_FRAME.value
            status = "out_of_frame"
            confidence = 1.0
            entity = None
            track_id = None
        elif hit is not None:
            track_id = hit.track_id
            if participant_association is None:
                region = hit.region
                status = "face_aoi_edge_tolerance_hit" if edge_hit else "face_aoi_hit"
                confidence = hit.confidence
                entity = hit.entity_id
                include_in_primary_analysis = True
            elif identity is None:
                region = "unresolved_face_identity"
                status = "face_detection_missing_identity_row"
                confidence = hit.confidence * 0.35
                entity = None
                identity_status = "missing_identity_row"
                review = True
            else:
                identity_status = str(identity.get("identity_status") or "unknown_identity")
                target_relation = str(identity.get("target_relation") or "other_person")
                identity_confidence = float(identity.get("identity_confidence") or 0.0)
                entity_value = identity.get("matched_entity_id")
                entity = str(entity_value) if entity_value is not None and str(entity_value) not in {"", "nan", "None"} else None
                matched = identity_status in MATCHED_IDENTITY_STATUSES and entity is not None
                if matched:
                    allowed = (
                        target_person_policy == "all_detected"
                        or target_person_policy == "tracked_participants"
                        or (target_person_policy == "partner_only" and target_relation == "partner")
                    )
                    if allowed:
                        region = hit.region
                        status = "participant_face_aoi_edge_tolerance_hit" if edge_hit else "participant_face_aoi_hit"
                        confidence = hit.confidence * max(0.25, identity_confidence)
                        include_in_primary_analysis = True
                    else:
                        region = "self_face" if target_relation == "self" else "tracked_other_face"
                        status = "excluded_non_partner_face"
                        confidence = hit.confidence * max(0.25, identity_confidence)
                elif target_relation in {"ambiguous_partner", "partner_candidate"} or identity.get("candidate_relation") == "partner":
                    candidate_value = identity.get("candidate_entity_id")
                    entity = str(candidate_value) if candidate_value is not None and str(candidate_value) not in {"", "nan", "None"} else entity
                    region = "ambiguous_face_identity"
                    status = identity_status
                    confidence = hit.confidence * max(0.2, identity_confidence or 0.0)
                    review = True
                elif target_relation == "other_person" or identity_status == "untracked_person":
                    entity = entity or f"other_person_track_{track_id}"
                    if target_person_policy == "all_detected":
                        region = hit.region
                        status = "other_person_face_aoi_edge_tolerance_hit" if edge_hit else "other_person_face_aoi_hit"
                        include_in_primary_analysis = True
                    else:
                        region = "other_person_face"
                        status = "excluded_other_person"
                    confidence = hit.confidence * max(0.25, identity_confidence or 0.5)
                else:
                    region = "ambiguous_face_identity"
                    status = identity_status
                    confidence = hit.confidence * max(0.2, identity_confidence or 0.0)
                    review = True
                review = review or bool(identity.get("review_required", False))
            if edge_hit and bool(hit.edge_ambiguous):
                review = True
                status = f"{status}+ambiguous_green_box_edge_band"
            # FaceMapper's exported gaze_on_face column is retained only as a
            # diagnostic. The accepted green FaceMapper geometry is authoritative.
            if fm_on_face is False:
                status = f"{status}+facemapper_flag_disagrees_with_green_box"
        elif not instances:
            region = "no_face_detection"
            status = "no_face_detection"
            confidence = 0.0
            entity = None
            track_id = None
        else:
            # At least one accepted green face exists in this frame, but this gaze
            # point is outside every accepted green face box. Do not use the light-
            # blue participant projection, the exported FaceMapper on-face flag,
            # or fixation continuity to turn it back into a face label.
            region = "no_face"
            status = "outside_all_accepted_facemapper_face_boxes"
            confidence = 1.0
            entity = None
            track_id = None
            target_relation = None
            identity_status = None
            include_in_primary_analysis = False
            review = False
        face_social_region = _face_social_region(
            raw_region if raw_region is not None else region,
            target_relation,
            identity_status,
        )
        rows.append({
            "recording_id": recording_id,
            "gaze_sample_uid": _gaze_uid(recording_id, source_row_index, timestamp),
            "source_gaze_row_index": source_row_index,
            "timestamp_ns": timestamp,
            "clip_time_ns": (
                int(clip_time_values.iloc[ordinal])
                if clip_time_values is not None and pd.notna(clip_time_values.iloc[ordinal]) else pd.NA
            ),
            "native_frame_index": native_frame,
            "gaze_x_px": x,
            "gaze_y_px": y,
            "worn": worn,
            "blink_id": gaze[blink_col].iloc[ordinal] if blink_col else None,
            "fixation_id": gaze[fixation_col].iloc[ordinal] if fixation_col else None,
            "frame_match_error_ms": (
                float(frame_error_values.iloc[ordinal]) / 1e6
                if frame_error_values is not None and pd.notna(frame_error_values.iloc[ordinal]) else np.nan
            ),
            "face_geometry_time_policy": face_geometry_time_policy,
            "face_geometry_time_mode": (
                selected_temporal.get("mode")
                if selected_temporal is not None else candidate_temporal_mode
            ),
            "face_geometry_observation_clock": "native_video_frames",
            "face_geometry_classification_clock": "gaze_sample_timestamps",
            "face_geometry_previous_frame_index": (
                selected_temporal.get("previous_frame_index")
                if selected_temporal is not None else pd.NA
            ),
            "face_geometry_next_frame_index": (
                selected_temporal.get("next_frame_index")
                if selected_temporal is not None else pd.NA
            ),
            "face_geometry_interpolation_alpha": (
                selected_temporal.get("alpha")
                if selected_temporal is not None else np.nan
            ),
            "face_geometry_interpolation_gap_ms": (
                selected_temporal.get("gap_ms")
                if selected_temporal is not None else np.nan
            ),
            "face_geometry_interpolation_confidence": (
                selected_temporal.get("confidence")
                if selected_temporal is not None else np.nan
            ),
            "face_geometry_interpolation_reason": (
                selected_temporal.get("reason")
                if selected_temporal is not None else "no_selected_face"
            ),
            "fm_gaze_on_face": fm_on_face,
            "fm_on_face_match_count": len(fm_values),
            "fm_on_face_unique_value_count": int(fm_on_face_unique_value_count),
            "fm_on_face_duplicate_consensus": bool(fm_on_face_duplicate_consensus),
            "fm_on_face_conflict": bool(fm_on_face_conflict),
            "green_face_geometry_hit": bool(hit is not None),
            "green_face_exact_hit": bool(hit is not None and hit.gate_hit_type == "exact_green_box"),
            "direct_green_box_review_eligible": bool(
                hit is not None
                and hit.gate_hit_type == "exact_green_box"
                and identity is not None
                and str(identity.get("identity_status") or "") in MATCHED_IDENTITY_STATUSES
                and include_in_primary_analysis
            ),
            "green_face_edge_tolerance_hit": bool(edge_hit),
            "face_gate_hit_type": hit.gate_hit_type if hit is not None else "outside",
            "face_edge_distance_px": float(hit.edge_distance_px) if hit is not None else np.nan,
            "face_edge_tolerance_px": float(hit.edge_tolerance_px) if hit is not None else 0.0,
            "face_edge_candidate_count": int(hit.edge_candidate_count) if hit is not None else 0,
            "face_edge_ambiguous": bool(hit.edge_ambiguous) if hit is not None else False,
            "face_edge_tolerance_policy": face_edge_tolerance_policy,
            "face_membership_policy": "accepted_green_box_with_bounded_edge_tolerance",
            "fm_flag_vs_green_geometry_disagreement": bool(
                fm_on_face is not None and bool(fm_on_face) != bool(hit is not None)
            ),
            "projection_fallback_entity_id": projection_fallback.get("entity_id") if projection_fallback else None,
            "projection_fallback_role_id": projection_fallback.get("role_id") if projection_fallback else None,
            "projection_fallback_relation": projection_fallback.get("relation") if projection_fallback else None,
            "projection_fallback_confidence": projection_fallback.get("confidence") if projection_fallback else None,
            "projection_fallback_centrality": projection_fallback.get("centrality") if projection_fallback else None,
            "raw_face_entity_id": raw_entity,
            "raw_face_region": raw_region,
            "face_entity_id": entity,
            "face_track_id": track_id,
            "face_region": region,
            "face_social_region": face_social_region,
            "face_aoi_confidence": float(confidence),
            "identity_status": identity_status,
            "identity_confidence": identity_confidence,
            "identity_match_reason": identity.get("match_reason") if identity else None,
            "identity_normalized_center_error": identity.get("normalized_center_error") if identity else None,
            "identity_projected_box_iou": identity.get("projected_box_iou") if identity else None,
            "identity_bbox_intersection_area_px2": identity.get("bbox_intersection_area_px2") if identity else None,
            "identity_bbox_overlap_present": identity.get("bbox_overlap_present") if identity else None,
            "identity_projected_box_overlap_fraction": identity.get("projected_box_overlap_fraction") if identity else None,
            "identity_face_box_overlap_fraction": identity.get("face_box_overlap_fraction") if identity else None,
            "identity_overlap_unique_for_projection": identity.get("overlap_unique_for_projection") if identity else None,
            "identity_overlap_unique_for_face": identity.get("overlap_unique_for_face") if identity else None,
            "identity_geometry_strong": identity.get("geometry_strong") if identity else None,
            "identity_projected_center_inside_face": identity.get("projected_center_inside_face") if identity else None,
            "identity_face_center_inside_projection": identity.get("face_center_inside_projection") if identity else None,
            "identity_duplicate_iou": identity.get("duplicate_iou") if identity else None,
            "identity_temporal_track_consistency": identity.get("temporal_track_consistency") if identity else None,
            "upstream_identity_review_required": bool(identity.get("review_required", False)) if identity else False,
            "target_relation": target_relation,
            "candidate_entity_id": identity.get("candidate_entity_id") if identity else None,
            "candidate_role": identity.get("candidate_role") if identity else None,
            "candidate_relation": identity.get("candidate_relation") if identity else None,
            "include_in_primary_analysis": bool(include_in_primary_analysis),
            "target_person_policy": target_person_policy,
            "source_bilateral_convention": (canonical_filter_manifest or {}).get("source_bilateral_label_policy"),
            "canonical_bilateral_convention": (canonical_filter_manifest or {}).get("canonical_bilateral_label_policy"),
            "source_pixel_geometry_preserved": bool(face_left_right_policy == "canonical_filter"),
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "participant_association_artifact_id": participant_association_artifact_id,
            "landmark_geometry_mirror_applied": bool(geometry_mirror_by_key.get((native_frame, int(hit.source_row_index)), False)) if hit is not None else False,
            "bilateral_label_swap_applied": bool(bilateral_swap_by_key.get((native_frame, int(hit.source_row_index)), False)) if hit is not None else False,
            "anatomical_3d_swap_applied": bool(bilateral_swap_by_key.get((native_frame, int(hit.source_row_index)), False)) if hit is not None else False,
            "anatomical_orientation_source": anatomical_source_by_key.get((native_frame, int(hit.source_row_index))) if hit is not None else None,
            "anatomical_orientation_confidence": anatomical_confidence_by_key.get((native_frame, int(hit.source_row_index)), 0.0) if hit is not None else 0.0,
            "anatomical_orientation_evidence_count": anatomical_evidence_by_key.get((native_frame, int(hit.source_row_index)), 0) if hit is not None else 0,
            "anatomical_orientation_verified": anatomical_verified_by_key.get((native_frame, int(hit.source_row_index)), False) if hit is not None else False,
            "assignment_status": status,
            "review_required": bool(review),
        })
        if ordinal % 2000 == 0:
            emit(progress, 0.22 + 0.34*(ordinal+1)/max(1,len(gaze)),
                 f"Classifying face AOIs ({ordinal+1}/{len(gaze)})")
    samples = pd.DataFrame(rows)
    temporal_mode_counts = Counter(
        samples.get("face_geometry_time_mode", pd.Series(dtype=str))
        .fillna("unknown")
        .astype(str)
    )
    face_result.diagnostics["face_geometry_time_mode_counts"] = dict(
        sorted(temporal_mode_counts.items())
    )
    face_result.diagnostics["track_interpolated_sample_fraction"] = (
        float(samples["face_geometry_time_mode"].eq("track_interpolated").mean())
        if len(samples) else 0.0
    )
    samples, auto_review_report, auto_review_table = _apply_auto_review_reduction(samples, auto_review_policy)
    in_scope_frames = set(pd.to_numeric(frame_map["native_frame_index"], errors="coerce").dropna().astype(int))
    if corrected_regions.empty or "scene_frame_index" not in corrected_regions.columns:
        regions = pd.DataFrame(columns=[
            "scene_frame_index", "scene_timestamp_ns", "entity_id", "region",
            "coordinate_space", "geometry_type", "geometry_json", "source",
            "source_asset_id", "confidence", "priority", "face_track_id",
        ])
    else:
        regions = corrected_regions.loc[
            corrected_regions["scene_frame_index"].isin(in_scope_frames)
        ].copy()
    if participant_association is not None and len(regions):
        identity_frame = pd.DataFrame(list(identity_by_key.values()))
        if len(identity_frame):
            identity_columns = [
                "native_frame_index", "face_source_row_index", "matched_entity_id", "matched_role",
                "target_relation", "identity_status", "identity_confidence",
                "candidate_entity_id", "candidate_role", "candidate_relation",
                "include_in_partner_analysis", "review_required",
                "projected_anatomical_left_head_x_px", "projected_anatomical_left_head_y_px",
                "projected_anatomical_right_head_x_px", "projected_anatomical_right_head_y_px",
            ]
            identity_frame = identity_frame[[column for column in identity_columns if column in identity_frame.columns]].copy()
            identity_frame = identity_frame.rename(columns={"native_frame_index": "scene_frame_index"})
            regions = regions.merge(
                identity_frame.drop_duplicates(["scene_frame_index", "face_source_row_index"]),
                on=["scene_frame_index", "face_source_row_index"],
                how="left",
                validate="many_to_one",
            )
            matched_entity = regions.get("matched_entity_id")
            if matched_entity is not None:
                regions["entity_id"] = matched_entity.where(matched_entity.notna(), regions["entity_id"])
            regions["include_in_primary_analysis"] = regions.get("include_in_partner_analysis", False).fillna(False).astype(bool)
            regions["landmark_geometry_mirror_applied"] = [
                bool(geometry_mirror_by_key.get((int(row.get("scene_frame_index")), int(row.get("face_source_row_index"))), False))
                for _, row in regions.iterrows()
            ]
            regions["bilateral_label_swap_applied"] = [
                bool(bilateral_swap_by_key.get((int(row.get("scene_frame_index")), int(row.get("face_source_row_index"))), False))
                for _, row in regions.iterrows()
            ]
            regions["anatomical_orientation_source"] = [
                anatomical_source_by_key.get((int(row.get("scene_frame_index")), int(row.get("face_source_row_index"))))
                for _, row in regions.iterrows()
            ]
            regions["anatomical_orientation_confidence"] = [
                float(anatomical_confidence_by_key.get((int(row.get("scene_frame_index")), int(row.get("face_source_row_index"))), 0.0))
                for _, row in regions.iterrows()
            ]
            regions["anatomical_orientation_verified"] = [
                bool(anatomical_verified_by_key.get((int(row.get("scene_frame_index")), int(row.get("face_source_row_index"))), False))
                for _, row in regions.iterrows()
            ]
            regions["face_social_region"] = [
                _face_social_region(row.get("region"), row.get("target_relation"), row.get("identity_status"))
                for _, row in regions.iterrows()
            ]
            if target_person_policy == "all_detected":
                regions["include_in_primary_analysis"] = True
    summary = _frame_summary(samples, frame_map)
    clip_summary, clip_summary_table = _full_clip_partner_face_statistics(samples, frame_map)

    artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "face_aoi") / (".partial_" + artifact_id)
    final = store.derived_dir(session_id, "face_aoi") / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    samples_csv = staging / "face_aoi_samples.csv"
    summary_csv = staging / "face_aoi_frames.csv"
    regions_csv = staging / "face_regions.csv"
    clip_summary_csv = staging / "face_aoi_clip_summary.csv"
    clip_summary_json = staging / "face_aoi_clip_summary.json"
    auto_review_csv = staging / "face_aoi_auto_review_summary.csv"
    auto_review_json = staging / "face_aoi_auto_review_summary.json"
    overlay = None
    overlay_map = None
    overlay_meta = None
    parent_ids = [native_trim_artifact_id]
    if participant_association_artifact_id:
        parent_ids.append(participant_association_artifact_id)
    if facemapper_filter_artifact_id:
        parent_ids.append(facemapper_filter_artifact_id)
    parquet_files: list[str] = []
    parquet_warning = None
    try:
        samples.to_csv(samples_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        regions.to_csv(regions_csv, index=False)
        clip_summary_table.to_csv(clip_summary_csv, index=False)
        atomic_write_json(clip_summary_json, clip_summary)
        auto_review_table.to_csv(auto_review_csv, index=False)
        atomic_write_json(auto_review_json, auto_review_report)
        if output_parquet:
            try:
                for frame, name in [
                    (samples, "face_aoi_samples.parquet"),
                    (summary, "face_aoi_frames.parquet"),
                    (regions, "face_regions.parquet"),
                ]:
                    path = staging / name
                    frame.to_parquet(path, index=False)
                    parquet_files.append(name)
            except Exception as exc:
                parquet_warning = f"Parquet output was requested but could not be written: {exc}"
                for path in staging.glob("*.parquet"):
                    path.unlink(missing_ok=True)
                parquet_files = []

        if privacy_clip_artifact_id and render_overlay:
            privacy_artifact = session.artifacts.get(privacy_clip_artifact_id)
            if privacy_artifact is None or privacy_artifact.kind != ArtifactKind.PRIVACY_CLIP:
                raise ValidationError("privacy_clip_artifact_id must reference a trimmed Privacy video")
            _validate_privacy_clip_matches_trim(privacy_artifact, manifest, frame_map)
            emit(progress, 0.68, "Rendering Face AOI overlay on blurred video")
            overlay, overlay_map, overlay_meta = _draw_face_overlay(
                privacy_artifact=privacy_artifact,
                output_root=staging,
                samples=samples,
                regions=regions,
                label=label,
                preserve_audio=preserve_audio,
                crf=crf,
                preset=preset,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                progress=progress,
                token=token,
            )
            parent_ids.append(privacy_clip_artifact_id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    counts = Counter(samples["face_region"].astype(str))
    social_counts = Counter(samples["face_social_region"].fillna(samples["face_region"]).astype(str))
    clip_qc = _clip_statistics_qc_fields(clip_summary)
    qc = {
        "input_gaze_rows": int(len(gaze)),
        "output_rows": int(len(samples)),
        "row_count_preserved": bool(len(gaze) == len(samples)),
        "face_aoi_hit_fraction": float(samples["face_region"].isin([
            SemanticRegion.LEFT_EYE.value, SemanticRegion.RIGHT_EYE.value,
            SemanticRegion.NOSE.value, SemanticRegion.MOUTH.value,
            SemanticRegion.OTHER_FACE.value, SemanticRegion.FACE.value,
        ]).mean()) if len(samples) else 0.0,
        "primary_analysis_fraction": float(samples["include_in_primary_analysis"].astype(bool).mean()) if len(samples) else 0.0,
        "partner_face_fraction": float(samples["target_relation"].eq("partner").mean()) if len(samples) else 0.0,
        "other_person_face_fraction": float(samples["target_relation"].eq("other_person").mean()) if len(samples) else 0.0,
        "review_required_fraction": float(samples["review_required"].mean()) if len(samples) else 0.0,
        "auto_review_policy": auto_review_policy,
        "review_samples_before_auto_resolution": int(auto_review_report["review_samples_before"]),
        "auto_resolved_sample_count": int(auto_review_report["auto_resolved_samples"]),
        "review_samples_remaining": int(auto_review_report["review_samples_remaining"]),
        "auto_review_sample_reduction_fraction": float(auto_review_report["sample_review_reduction_fraction"]),
        "review_frames_before_auto_resolution": int(auto_review_report["review_frames_before"]),
        "review_frames_remaining": int(auto_review_report["review_frames_remaining"]),
        "auto_review_frame_reduction_fraction": float(auto_review_report["frame_review_reduction_fraction"]),
        "auto_review_resolution_source_counts": dict(auto_review_report["resolution_source_counts"]),
        "review_reason_counts_before": dict(auto_review_report["review_reason_counts_before"]),
        "remaining_review_reason_counts": dict(auto_review_report["remaining_review_reason_counts"]),
        "facemapper_on_face_duplicate_consensus_sample_count": int(samples.get("fm_on_face_duplicate_consensus", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "facemapper_on_face_conflict_sample_count": int(samples.get("fm_on_face_conflict", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "projection_fallback_candidate_sample_count": int(samples.get("projection_fallback_entity_id", pd.Series(None, index=samples.index)).notna().sum()) if len(samples) else 0,
        "outside_green_face_box_sample_count": int(samples.get("assignment_status", pd.Series("", index=samples.index)).eq("outside_all_accepted_facemapper_face_boxes").sum()) if len(samples) else 0,
        "green_face_exact_hit_sample_count": int(samples.get("green_face_exact_hit", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "direct_green_box_review_eligible_sample_count": int(samples.get("direct_green_box_review_eligible", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "direct_green_box_auto_accepted_sample_count": int(samples.get("auto_review_resolution_source", pd.Series("", index=samples.index)).astype(str).eq("direct_green_box_matched_identity").sum()) if len(samples) else 0,
        "direct_green_box_review_policy": "auto_accept_matched_identity",
        "green_face_edge_tolerance_hit_sample_count": int(samples.get("green_face_edge_tolerance_hit", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "green_face_edge_ambiguous_sample_count": int(samples.get("face_edge_ambiguous", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "face_edge_tolerance_policy": face_edge_tolerance_policy,
        "face_edge_tolerance_ratio": aoi_settings["face_edge_tolerance_ratio"],
        "face_edge_tolerance_min_px": aoi_settings["face_edge_tolerance_min_px"],
        "face_edge_tolerance_max_px": aoi_settings["face_edge_tolerance_max_px"],
        "fm_flag_vs_green_geometry_disagreement_count": int(samples.get("fm_flag_vs_green_geometry_disagreement", pd.Series(False, index=samples.index)).fillna(False).astype(bool).sum()) if len(samples) else 0,
        "face_membership_policy": "accepted_green_box_with_bounded_edge_tolerance",
        "facemapper_gaze_on_face_usage": "diagnostic_only",
        "face_region_counts": dict(counts),
        "face_social_region_counts": dict(social_counts),
        "frame_count": int(len(summary)),
        "face_diagnostics": face_result.diagnostics,
        "face_aoi_engine": FACE_AOI_ENGINE,
        "face_geometry_time_policy": face_geometry_time_policy,
        "face_geometry_observation_clock": "native_video_frames",
        "face_geometry_classification_clock": "gaze_sample_timestamps",
        "face_interpolation_max_gap_ms": float(face_interpolation_max_gap_ms),
        "face_interpolation_max_motion_ratio": float(face_interpolation_max_motion_ratio),
        "face_geometry_time_mode_counts": dict(sorted(temporal_mode_counts.items())),
        "track_interpolated_sample_fraction": (
            float(samples["face_geometry_time_mode"].eq("track_interpolated").mean())
            if len(samples) else 0.0
        ),
        "estimated_gaze_sample_rate_hz": _estimate_rate_hz_ns(timestamp_values),
        "estimated_native_video_rate_hz": _estimate_rate_hz_ns(scene_timestamps),
        "fine_feature_aoi_scale": fine_feature_aoi_scale,
        "eye_feature_aoi_scale": eye_feature_aoi_scale,
        "effective_eye_aoi_scale": fine_feature_aoi_scale * eye_feature_aoi_scale,
        "nose_region_mode": nose_region_mode,
        "nose_region_geometry_model": nose_geometry_model,
        "mouth_center_policy": mouth_center_policy,
        "mouth_center_offset_ratio": aoi_settings["mouth_center_offset_ratio"],
        "mouth_region_geometry_model": mouth_geometry_model,
        "mouth_semantic_scope": mouth_semantic_scope,
        "mouth_lateral_padding_ratio": aoi_settings["mouth_lateral_padding_ratio"],
        "mouth_top_padding_face_ratio": aoi_settings["mouth_top_padding_face_ratio"],
        "mouth_bottom_padding_face_ratio": aoi_settings["mouth_bottom_padding_face_ratio"],
        "mouth_max_height_face_ratio": aoi_settings["mouth_max_height_face_ratio"],
        "mouth_bottom_limit_face_ratio": aoi_settings["mouth_bottom_limit_face_ratio"],
        "aoi_coordinate_source": "accepted_facemapper_green_face_box",
        "participant_projection_usage": "identity_gate_only",
        "face_membership_policy": "accepted_green_box_with_bounded_edge_tolerance",
        "face_edge_tolerance_policy": face_edge_tolerance_policy,
        "face_edge_tolerance_ratio": aoi_settings["face_edge_tolerance_ratio"],
        "face_edge_tolerance_min_px": aoi_settings["face_edge_tolerance_min_px"],
        "face_edge_tolerance_max_px": aoi_settings["face_edge_tolerance_max_px"],
        "fine_feature_edge_expansion": False,
        "facemapper_gaze_on_face_usage": "diagnostic_only",
        "temporal_face_promotion_outside_green_box": False,
        "direct_green_box_review_policy": "auto_accept_matched_identity",
        "full_face_geometry_model": "face_bbox_exact_rectangle_plus_coarse_edge_band",
        "fine_feature_containment_policy": "face_bbox_inner_clip",
        "fine_feature_face_margin_ratio": aoi_settings["face_margin_ratio"],
        "fine_feature_region_rows": int(regions["region"].isin([
            SemanticRegion.LEFT_EYE.value, SemanticRegion.RIGHT_EYE.value,
            SemanticRegion.NOSE.value, SemanticRegion.MOUTH.value,
        ]).sum()) if len(regions) and "region" in regions.columns else 0,
        "fine_feature_contained_fraction": float(
            regions.loc[regions["region"].isin([
                SemanticRegion.LEFT_EYE.value, SemanticRegion.RIGHT_EYE.value,
                SemanticRegion.NOSE.value, SemanticRegion.MOUTH.value,
            ]), "aoi_contained_in_face_bbox"].astype(bool).mean()
        ) if len(regions) and "aoi_contained_in_face_bbox" in regions.columns and regions["region"].isin([
            SemanticRegion.LEFT_EYE.value, SemanticRegion.RIGHT_EYE.value,
            SemanticRegion.NOSE.value, SemanticRegion.MOUTH.value,
        ]).any() else 1.0,
        "participant_3d_geometry_mirror_count": int(sum(geometry_mirror_by_key.values())),
        "participant_3d_bilateral_swap_count": int(sum(bilateral_swap_by_key.values())),
        "overlay_rendered": overlay is not None,
        "parquet_written": bool(parquet_files),
        "parquet_warning": parquet_warning,
        **clip_qc,
    }
    manifest_data = {
        "schema_version": "oculostudio-face-aoi-1.13",
        "face_aoi_engine": FACE_AOI_ENGINE,
        "artifact_id": artifact_id,
        "native_trim_artifact_id": native_trim_artifact_id,
        "participant_association_artifact_id": participant_association_artifact_id,
        "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
        "privacy_clip_artifact_id": privacy_clip_artifact_id,
        "recording_id": recording_id,
        "ontology": "oculostudio-face-aoi-1.13",
        "parameters": {
            "face_tolerance_ms": face_tolerance_ms,
            "face_left_right_policy": face_left_right_policy,
            "face_landmark_geometry_policy": face_landmark_geometry_policy,
            "fine_feature_aoi_scale": fine_feature_aoi_scale,
            "eye_feature_aoi_scale": eye_feature_aoi_scale,
            "effective_eye_aoi_scale": fine_feature_aoi_scale * eye_feature_aoi_scale,
            "nose_region_mode": nose_region_mode,
            "mouth_center_policy": mouth_center_policy,
            "mouth_center_offset_ratio": aoi_settings["mouth_center_offset_ratio"],
            "mouth_region_geometry_model": mouth_geometry_model,
            "mouth_semantic_scope": mouth_semantic_scope,
            "mouth_lateral_padding_ratio": aoi_settings["mouth_lateral_padding_ratio"],
            "mouth_top_padding_face_ratio": aoi_settings["mouth_top_padding_face_ratio"],
            "mouth_bottom_padding_face_ratio": aoi_settings["mouth_bottom_padding_face_ratio"],
            "mouth_max_height_face_ratio": aoi_settings["mouth_max_height_face_ratio"],
            "mouth_bottom_limit_face_ratio": aoi_settings["mouth_bottom_limit_face_ratio"],
            "aoi_coordinate_source": "accepted_facemapper_green_face_box",
            "participant_projection_usage": "identity_gate_only",
            "face_membership_policy": "accepted_green_box_with_bounded_edge_tolerance",
            "face_edge_tolerance_policy": face_edge_tolerance_policy,
            "face_edge_tolerance_ratio": aoi_settings["face_edge_tolerance_ratio"],
            "face_edge_tolerance_min_px": aoi_settings["face_edge_tolerance_min_px"],
            "face_edge_tolerance_max_px": aoi_settings["face_edge_tolerance_max_px"],
            "fine_feature_edge_expansion": False,
            "facemapper_gaze_on_face_usage": "diagnostic_only",
            "temporal_face_promotion_outside_green_box": False,
            "direct_green_box_review_policy": "auto_accept_matched_identity",
            "auto_review_policy": auto_review_policy,
            "face_geometry_time_policy": face_geometry_time_policy,
            "face_interpolation_max_gap_ms": float(face_interpolation_max_gap_ms),
            "face_interpolation_max_motion_ratio": float(face_interpolation_max_motion_ratio),
            "face_geometry_observation_clock": "native_video_frames",
            "face_geometry_classification_clock": "gaze_sample_timestamps",
            "nose_region_geometry_model": nose_geometry_model,
            "full_face_geometry_model": "face_bbox_exact_rectangle_plus_coarse_edge_band",
            "fine_feature_containment_policy": "face_bbox_inner_clip",
            "fine_feature_face_margin_ratio": aoi_settings["face_margin_ratio"],
            "canonical_face_mapper_filter": bool(canonical_filter_ready),
            "canonical_bilateral_convention": "subject_anatomical" if canonical_filter_ready else None,
            "source_pixel_geometry_preserved": bool(canonical_filter_ready),
            "face_aoi_engine": FACE_AOI_ENGINE,
            "target_person_policy": target_person_policy,
            "source_bilateral_convention": (canonical_filter_manifest or {}).get("source_bilateral_label_policy"),
            "canonical_bilateral_convention": (canonical_filter_manifest or {}).get("canonical_bilateral_label_policy"),
            "source_pixel_geometry_preserved": bool(face_left_right_policy == "canonical_filter"),
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "output_parquet": output_parquet,
            "render_overlay": render_overlay,
        },
        "outputs": {
            "samples_csv": "face_aoi_samples.csv",
            "frames_csv": "face_aoi_frames.csv",
            "regions_csv": "face_regions.csv",
            "clip_summary_csv": "face_aoi_clip_summary.csv",
            "clip_summary_json": "face_aoi_clip_summary.json",
            "auto_review_summary_csv": "face_aoi_auto_review_summary.csv",
            "auto_review_summary_json": "face_aoi_auto_review_summary.json",
            "overlay_video": overlay.name if overlay else None,
            "overlay_frame_map": overlay_map.name if overlay_map else None,
            "parquet": parquet_files,
        },
        "qc": qc,
        "overlay": overlay_meta,
        "created_at": utc_now(),
    }
    atomic_write_json(staging / "face_aoi_manifest.json", manifest_data)
    atomic_write_json(staging / "qc_metrics.json", qc)
    try:
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.FACE_AOI,
        label=label,
        output_root=final,
        source_asset_ids=trim_artifact.source_asset_ids,
        parent_artifact_ids=parent_ids,
        parameters=manifest_data["parameters"],
        qc=qc,
        provenance={
            "native_trim_artifact_id": native_trim_artifact_id,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "participant_association_artifact_id": participant_association_artifact_id,
            "privacy_clip_artifact_id": privacy_clip_artifact_id,
            "target_person_policy": target_person_policy,
            "classification_order": "gaze_sample_then_frame_summary",
            "face_left_right_policy": face_left_right_policy,
            "face_landmark_geometry_policy": face_landmark_geometry_policy,
            "fine_feature_aoi_scale": fine_feature_aoi_scale,
            "eye_feature_aoi_scale": eye_feature_aoi_scale,
            "effective_eye_aoi_scale": fine_feature_aoi_scale * eye_feature_aoi_scale,
            "nose_region_mode": nose_region_mode,
            "mouth_center_policy": mouth_center_policy,
            "mouth_center_offset_ratio": aoi_settings["mouth_center_offset_ratio"],
            "mouth_region_geometry_model": mouth_geometry_model,
            "mouth_semantic_scope": mouth_semantic_scope,
            "mouth_lateral_padding_ratio": aoi_settings["mouth_lateral_padding_ratio"],
            "mouth_top_padding_face_ratio": aoi_settings["mouth_top_padding_face_ratio"],
            "mouth_bottom_padding_face_ratio": aoi_settings["mouth_bottom_padding_face_ratio"],
            "mouth_max_height_face_ratio": aoi_settings["mouth_max_height_face_ratio"],
            "mouth_bottom_limit_face_ratio": aoi_settings["mouth_bottom_limit_face_ratio"],
            "aoi_coordinate_source": "accepted_facemapper_green_face_box",
            "participant_projection_usage": "identity_gate_only",
            "face_membership_policy": "accepted_green_box_with_bounded_edge_tolerance",
            "face_edge_tolerance_policy": face_edge_tolerance_policy,
            "face_edge_tolerance_ratio": aoi_settings["face_edge_tolerance_ratio"],
            "face_edge_tolerance_min_px": aoi_settings["face_edge_tolerance_min_px"],
            "face_edge_tolerance_max_px": aoi_settings["face_edge_tolerance_max_px"],
            "fine_feature_edge_expansion": False,
            "facemapper_gaze_on_face_usage": "diagnostic_only",
            "temporal_face_promotion_outside_green_box": False,
            "direct_green_box_review_policy": "auto_accept_matched_identity",
            "auto_review_policy": auto_review_policy,
            "nose_region_geometry_model": nose_geometry_model,
            "full_face_geometry_model": "face_bbox_exact_rectangle_plus_coarse_edge_band",
            "fine_feature_containment_policy": "face_bbox_inner_clip",
            "fine_feature_face_margin_ratio": aoi_settings["face_margin_ratio"],
            "canonical_face_mapper_filter": bool(canonical_filter_ready),
            "canonical_bilateral_convention": "subject_anatomical" if canonical_filter_ready else None,
            "source_pixel_geometry_preserved": bool(canonical_filter_ready),
            "face_aoi_engine": FACE_AOI_ENGINE,
            "face_geometry_time_policy": face_geometry_time_policy,
            "face_geometry_observation_clock": "native_video_frames",
            "face_geometry_classification_clock": "gaze_sample_timestamps",
            "face_interpolation_max_gap_ms": float(face_interpolation_max_gap_ms),
            "face_interpolation_max_motion_ratio": float(face_interpolation_max_motion_ratio),
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Face AOI annotation complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "samples_csv": str(final / samples_csv.name),
        "frames_csv": str(final / summary_csv.name),
        "regions_csv": str(final / regions_csv.name),
        "clip_summary_csv": str(final / clip_summary_csv.name),
        "clip_summary_json": str(final / clip_summary_json.name),
        "auto_review_summary_csv": str(final / auto_review_csv.name),
        "auto_review_summary_json": str(final / auto_review_json.name),
        "overlay_video": str(final / overlay.name) if overlay else None,
        "manifest": str(final / "face_aoi_manifest.json"),
    }
