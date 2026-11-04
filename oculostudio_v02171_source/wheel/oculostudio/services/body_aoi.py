from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.mocap import read_mocap
from oculostudio.adapters.pupil import read_cloud_world_timestamps
from oculostudio.domain.models import ArtifactKind, AssetKind
from oculostudio.domain.semantic import SemanticRegion
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ValidationError
from oculostudio.services.artifact_io import artifact_root, find_artifact_file
from oculostudio.services.body_primitives import (
    DEFAULT_MINIMUM_BODY_FRAME_COVERAGE,
    DEFAULT_MINIMUM_TORSO_COVERAGE,
    assess_body_geometry_integrity,
    build_body_primitives,
    coarse_body_region,
    intersect_frame_rays_ranked,
    intersect_sample_rays_ranked,
    load_body_profile,
)
from oculostudio.services.etm_role_assignment import load_etm_assignment
from oculostudio.services.semantic_targets import (
    _artifact_table_path,
    _find_column,
    _nearest_indices,
    _read_table,
    _resolve_cloud_asset,
    _scene_resolution,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now



def _positive_resolution(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _native_trim_scene_resolution(trim_artifact) -> tuple[int, int, dict[str, Any]] | None:
    """Recover the actual ego-video resolution from a Native-reference trim.

    Native-reference trims intentionally contain data tables rather than a copied
    scene video.  Their manifest nevertheless records the exact selected Native
    MP4 clock metadata, including width and height.  Body AOI uses that metadata
    only for provenance; all body intersections are performed in 3D.
    """
    if trim_artifact is None:
        return None
    manifest_path = find_artifact_file(trim_artifact, names=["trim_manifest.json"], required=False)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except Exception:
        return None
    candidates = [
        manifest.get("native_reference"),
        manifest.get("clock_summary"),
        (manifest.get("preview") or {}).get("native_reference")
        if isinstance(manifest.get("preview"), dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        width = _positive_resolution(candidate.get("width"))
        height = _positive_resolution(candidate.get("height"))
        if width and height:
            return width, height, {
                "source": "native_reference_trim_manifest",
                "path": str(manifest_path),
                "required_for_body_intersection": False,
            }
    return None


def _body_scene_resolution(
    cache: MaterializationCache,
    cloud_asset,
    gaze_artifact,
    trim_artifact,
) -> tuple[int | None, int | None, dict[str, Any]]:
    """Resolve optional scene-size provenance without blocking Body AOI.

    The 3D gaze-ray/body intersection is independent of image width and height.
    Prefer the selected Native-reference trim's exact video metadata.  Otherwise
    retain the legacy calibration/Cloud lookup when available.  If neither source
    contains a resolution, continue with explicit nullable provenance rather than
    rejecting a valid data-only pre-trimmed workflow.
    """
    trim_resolution = _native_trim_scene_resolution(trim_artifact)
    if trim_resolution is not None:
        return trim_resolution
    try:
        width, height, provenance = _scene_resolution(cache, cloud_asset, gaze_artifact)
        return width, height, {**provenance, "required_for_body_intersection": False}
    except ValidationError as exc:
        return None, None, {
            "source": "unavailable_nonblocking",
            "required_for_body_intersection": False,
            "reason": str(exc),
        }

def _gaze_uid(recording_id: str, source_row_index: int, timestamp_ns: int) -> str:
    raw = f"{recording_id}|{source_row_index}|{timestamp_ns}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


def _write_body_geometry_preflight(
    store: ProjectStore,
    *,
    session_id: str,
    gaze_rays_artifact_id: str,
    report: dict[str, Any],
    parameters: dict[str, Any],
) -> Path:
    root = store.root / "diagnostics" / safe_name(session_id) / "body_aoi_geometry"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{safe_name(gaze_rays_artifact_id)}_{new_id('preflight')}.json"
    payload = {
        "schema_version": "oculostudio-body-aoi-geometry-preflight-1.0",
        "session_id": session_id,
        "gaze_rays_artifact_id": gaze_rays_artifact_id,
        "created_at": utc_now(),
        "parameters": parameters,
        "body_geometry_integrity": report,
    }
    atomic_write_json(path, payload)
    return path


def _body_geometry_gate_error(report: dict[str, Any], diagnostic_path: Path, project_root: Path) -> str:
    lines = [
        "Body geometry acceptance failed before gaze-ray intersection.",
    ]
    for entity in report.get("entities") or []:
        if not entity.get("required") or entity.get("status") != "failed":
            continue
        relation = entity.get("entity_relation") or "body"
        entity_id = entity.get("entity_id") or "unknown"
        source = entity.get("source_file_name") or entity.get("source_asset_label") or entity.get("source_asset_id") or "unknown source"
        torso = int(entity.get("torso_primitive_frame_count") or 0)
        expected = int(entity.get("expected_frame_count") or 0)
        any_frames = int(entity.get("frames_with_any_primitive") or 0)
        lines.append(
            f"- {relation} {entity_id} ({source}): torso {torso}/{expected} "
            f"({float(entity.get('torso_coverage_fraction') or 0.0):.1%}); "
            f"any body geometry {any_frames}/{expected} "
            f"({float(entity.get('body_frame_coverage_fraction') or 0.0):.1%})."
        )
        protocol = entity.get("marker_protocol") or {}
        protocol_name = str(protocol.get("name") or "unknown")
        matches = entity.get("landmark_matches") or {}
        chest = matches.get("chest_center") or []
        pelvis = matches.get("pelvis_center") or []
        left_hip = matches.get("left_hip") or []
        right_hip = matches.get("right_hip") or []
        lines.append(
            f"  Marker protocol: {protocol_name}; chest markers={chest or 'none'}; "
            f"pelvis markers={pelvis or 'none'}; left/right hip anchors="
            f"{left_hip or 'none'} / {right_hip or 'none'}."
        )
    first_failures = report.get("failures") or []
    if first_failures:
        lines.append("Reasons: " + "; ".join(str(row.get("message") or row.get("code")) for row in first_failures[:6]))
    try:
        relative = diagnostic_path.relative_to(project_root)
    except ValueError:
        relative = diagnostic_path
    lines.extend([
        "Verify wearer/partner C3D assignment, recording duration and raw Vicon marker labels. OculoStudio maps the OCUMO full-body 10-finger profile (FWT/MWT/BWT waist and FSH/TSH/BSH shoulder clusters) directly from the original C3D.",
        f"Diagnostic report: {relative}",
        "Warn only / Off may create a diagnostic artifact, but invalid geometry cannot enter human review or target fusion.",
    ])
    return "\n".join(lines)


def _frame_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = samples.loc[samples["scene_frame_index"].notna()].groupby("scene_frame_index")
    for frame_index, frame in grouped:
        distribution = Counter(frame["body_region"].fillna("no_body").astype(str))
        relation_distribution = Counter(frame.get("body_relation", pd.Series(index=frame.index, dtype=object)).fillna("none").astype(str))
        score: dict[tuple[str | None, str | None, str], float] = defaultdict(float)
        for row in frame.itertuples(index=False):
            entity = row.body_entity_id if isinstance(row.body_entity_id, str) and row.body_entity_id else None
            relation = getattr(row, "body_relation", None)
            relation = relation if isinstance(relation, str) and relation else None
            region = row.body_region if isinstance(row.body_region, str) and row.body_region else "no_body"
            score[(entity, relation, region)] += max(0.05, float(row.body_confidence or 0.0))
        primary = max(score, key=score.get) if score else (None, None, "no_body")
        rows.append({
            "scene_frame_index": int(frame_index),
            "scene_timestamp_ns": int(pd.to_numeric(frame["scene_timestamp_ns"], errors="coerce").dropna().iloc[0]),
            "gaze_sample_count": int(len(frame)),
            "valid_ray_count": int(frame["ray_valid"].fillna(False).astype(bool).sum()),
            "primary_body_entity": primary[0],
            "primary_body_relation": primary[1],
            "primary_body_region": primary[2],
            "body_region_distribution_json": json.dumps(distribution, sort_keys=True, separators=(",", ":")),
            "body_relation_distribution_json": json.dumps(relation_distribution, sort_keys=True, separators=(",", ":")),
            "self_occlusion_present": bool(frame.get("self_occlusion", pd.Series(False, index=frame.index)).fillna(False).astype(bool).any()),
            "mean_body_confidence": float(pd.to_numeric(frame["body_confidence"], errors="coerce").fillna(0).mean()),
            "review_required": bool(frame["review_required"].fillna(False).astype(bool).any()),
        })
    return pd.DataFrame(rows)


def _selected_frame_times(
    values_sec: np.ndarray,
    nearest: np.ndarray,
    selected_frames: np.ndarray,
    *,
    clock_label: str,
) -> np.ndarray:
    """Aggregate an exact gaze-sample clock to one display time per video frame.

    Body-ray intersection is performed at every gaze-sample timestamp.  This
    helper is used only to build the lower-rate geometry shown in the review
    player, so the median sample time inside each scene frame is appropriate.
    """
    values_sec = np.asarray(values_sec, dtype=float)
    result = np.full(len(selected_frames), np.nan, dtype=float)
    for position, frame_index in enumerate(selected_frames):
        values = values_sec[(nearest == int(frame_index)) & np.isfinite(values_sec)]
        if len(values):
            result[position] = float(np.median(values))
    if not np.isfinite(result).all():
        missing = int((~np.isfinite(result)).sum())
        raise ValidationError(
            f"Could not recover {clock_label} for {missing} selected scene frames"
        )
    return result


def _gaze_sample_c3d_times(rays: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """Return the exact ETM/Vicon C3D time already used to build each gaze ray.

    A TimeGraph-generated Gaze 3D artifact is already in the Vicon world frame
    and records the tracker query time in ``gaze3d_c3d_time_sec``.  Downstream
    Body AOI must reuse that clock directly.  Routing the original Pupil clock
    through a project-level ``global`` domain is both unnecessary and can fail
    in otherwise valid C3D-defined interaction projects.
    """
    column = _find_column(
        rays.columns,
        [
            "gaze3d_c3d_time_sec",
            "gaze3d_tracker_time_sec",
            "c3d_time_sec",
            "tracker_time_sec",
        ],
        required=False,
    )
    if not column:
        return None, None
    values = pd.to_numeric(rays[column], errors="coerce").to_numpy(dtype=float)
    return values, column


def _gaze_sample_local_times(rays: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """Return a co-started local clock carried by every gaze ray.

    This is a backward-compatible fallback for older Gaze 3D artifacts that do
    not expose the absolute C3D query time.  New artifacts should normally use
    :func:`_gaze_sample_c3d_times` instead.
    """
    local_col = _find_column(
        rays.columns,
        [
            "gaze3d_tracker_local_time_sec",
            "tracker_local_time_sec",
            "clip_time_ns",
            "oculo_clip_video_ns",
        ],
        required=False,
    )
    if not local_col:
        return None, None
    local = pd.to_numeric(rays[local_col], errors="coerce").to_numpy(dtype=float)
    if local_col in {"clip_time_ns", "oculo_clip_video_ns"}:
        local = local / 1e9
    return local, local_col


def _estimate_frequency_hz(timestamps_ns: np.ndarray) -> float | None:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    if len(values) < 2:
        return None
    values = np.unique(values)
    if len(values) < 2:
        return None
    delta = np.diff(values).astype(np.float64)
    delta = delta[delta > 0]
    if not len(delta):
        return None
    median = float(np.median(delta))
    return float(1e9 / median) if median > 0 else None


def _normalise_social_hits(
    hits,
    *,
    relation_by_asset: dict[str, str],
    wearer_role_id: str | None,
    partner_role_id: str | None,
    self_near_origin_exclusion_mm: float,
):
    """Return distinct, distance-ordered social-body intersections.

    Overlapping detailed primitives may map to the same coarse social region.
    Only the nearest representative of each entity/coarse-region pair is kept,
    while a nearer self surface is never skipped in favour of a farther partner.
    """
    output = []
    seen: set[tuple[str, str]] = set()
    for hit in hits or []:
        relation = relation_by_asset.get(hit.source_asset_id, "other")
        if wearer_role_id and str(hit.entity_id).upper() == wearer_role_id:
            relation = "self"
        elif partner_role_id and str(hit.entity_id).upper() == partner_role_id:
            relation = "partner"
        if relation == "self" and float(hit.distance_mm) < float(self_near_origin_exclusion_mm):
            continue
        coarse = coarse_body_region(hit.region)
        key = (str(hit.entity_id), coarse)
        if key in seen:
            continue
        seen.add(key)
        output.append((hit, relation, coarse))
    return output


def _hit_columns(prefix: str, item, *, output_granularity: str) -> dict[str, Any]:
    if item is None:
        return {
            f"{prefix}_entity_id": None,
            f"{prefix}_relation": None,
            f"{prefix}_raw_region": None,
            f"{prefix}_coarse_region": None,
            f"{prefix}_region": None,
            f"{prefix}_social_region": None,
            f"{prefix}_distance_mm": np.nan,
            f"{prefix}_x_mm": np.nan,
            f"{prefix}_y_mm": np.nan,
            f"{prefix}_z_mm": np.nan,
            f"{prefix}_confidence": 0.0,
            f"{prefix}_exact": False,
            f"{prefix}_angular_miss_deg": np.nan,
            f"{prefix}_primitive_confidence": 0.0,
            f"{prefix}_source_asset_id": None,
        }
    hit, relation, coarse = item
    selected_region = coarse if output_granularity == "coarse_social_regions" else str(hit.region)
    social_region = f"{relation}_{selected_region}" if relation in {"self", "partner"} else selected_region
    return {
        f"{prefix}_entity_id": hit.entity_id,
        f"{prefix}_relation": relation,
        f"{prefix}_raw_region": str(hit.region),
        f"{prefix}_coarse_region": coarse,
        f"{prefix}_region": selected_region,
        f"{prefix}_social_region": social_region,
        f"{prefix}_distance_mm": float(hit.distance_mm),
        f"{prefix}_x_mm": float(hit.point_mm[0]),
        f"{prefix}_y_mm": float(hit.point_mm[1]),
        f"{prefix}_z_mm": float(hit.point_mm[2]),
        f"{prefix}_confidence": float(hit.confidence),
        f"{prefix}_exact": bool(hit.exact_hit),
        f"{prefix}_angular_miss_deg": float(hit.angular_miss_deg),
        f"{prefix}_primitive_confidence": float(hit.primitive_confidence),
        f"{prefix}_source_asset_id": hit.source_asset_id,
    }



BODY_AUTO_REVIEW_POLICIES = {
    "off",
    "conservative",
    "adaptive",
    "evidence_guided",
    "semantic_guided",
    "balanced",
}


def _body_text_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _body_target_tuple(row: pd.Series) -> tuple[str, str, str] | None:
    relation = _body_text_value(row.get("body_relation")).lower()
    entity = _body_text_value(row.get("body_entity_id"))
    region = _body_text_value(row.get("body_coarse_region"))
    if not region:
        region = _body_text_value(row.get("body_region"))
    region = region.lower()
    if relation not in {"self", "partner"} or not entity:
        return None
    if region in {"", "no_body", "invalid_gaze", "out_of_frame", "unknown"}:
        return None
    return relation, entity, region



def _second_body_target_tuple(row: pd.Series) -> tuple[str, str, str] | None:
    relation = _body_text_value(row.get("second_hit_relation")).lower()
    entity = _body_text_value(row.get("second_hit_entity_id"))
    region = _body_text_value(row.get("second_hit_coarse_region"))
    if not region:
        region = _body_text_value(row.get("second_hit_raw_region"))
    region = region.lower()
    if relation not in {"self", "partner"} or not entity:
        return None
    if region in {"", "no_body", "invalid_gaze", "out_of_frame", "unknown"}:
        return None
    return relation, entity, region


def _same_semantic_surface_candidates(row: pd.Series) -> bool:
    """True when the only competing surface keeps the exported coarse label.

    In coarse Body AOI mode, head/neck, torso/pelvis, or two primitives from the
    same limb can be separate 3D surfaces but the same scientific label.  Such a
    pair is not a human-label ambiguity when person, entity and coarse region all
    agree.  We deliberately require exactly two candidates because additional
    hidden surfaces are not represented in the audit columns.
    """
    count = int(pd.to_numeric(
        pd.Series([row.get("candidate_hit_count")]), errors="coerce"
    ).fillna(0).iloc[0])
    if count != 2:
        return False
    primary = _body_target_tuple(row)
    secondary = _second_body_target_tuple(row)
    return primary is not None and secondary == primary


def _body_review_reason(row: pd.Series) -> str:
    status = (_body_text_value(row.get("assignment_status")) or "unknown").lower()
    if status == "no_scene_frame":
        return "no_scene_frame"
    if bool(row.get("self_occlusion")):
        return "self_occlusion"
    if not bool(row.get("body_exact_hit")) and int(pd.to_numeric(pd.Series([row.get("candidate_hit_count")]), errors="coerce").fillna(0).iloc[0]) > 1:
        return "multiple_uncertainty_candidates"
    if not bool(row.get("body_exact_hit")):
        return "uncertainty_cone_candidate"
    return status or "unknown"


def _valid_fixation_key(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "-1"} else text


def _apply_body_auto_review_reduction(
    samples: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Clear only auditable Body AOI uncertainty cases.

    The automatic target is never changed.  A review flag is cleared only when
    the near-body candidate is geometrically well separated or when matching
    exact hits in the same video frame/fixation provide consistent evidence.
    Self-occlusion, missing scene frames, other-person candidates and close
    competing surfaces remain in the human queue.
    """
    if policy not in BODY_AUTO_REVIEW_POLICIES:
        raise ValidationError(
            "body auto_review_policy must be off, conservative, adaptive, "
            "evidence_guided, semantic_guided or balanced"
        )
    frame = samples.copy()
    frame["review_required"] = frame.get("review_required", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame["review_required_before_auto_resolution"] = frame["review_required"].astype(bool)
    frame["review_reason_before_auto_resolution"] = frame.apply(_body_review_reason, axis=1)
    frame["auto_review_resolved"] = False
    frame["auto_review_resolution_source"] = ""
    frame["auto_review_confidence"] = np.nan
    frame["auto_review_policy"] = policy
    frame["auto_review_evidence_group"] = ""
    frame["auto_review_evidence_sample_count"] = 0
    frame["auto_review_evidence_target_share"] = np.nan
    frame["auto_review_evidence_median_angle_deg"] = np.nan
    frame["auto_review_evidence_p90_angle_deg"] = np.nan
    frame["auto_review_evidence_exact_anchor_count"] = 0

    config = {
        "conservative": {
            "angle": 0.18, "small_angle": 0.10, "min_conf": 0.50,
            "min_primitive": 0.74, "margin": 0.36, "share": 0.88,
            "frame_anchors": 3, "fixation_anchors": 3, "fixation_gap_ms": 90,
            "candidate_fixation": False,
            "candidate_min_samples": 99, "candidate_target_share": 1.01,
            "candidate_body_fraction": 1.01, "candidate_single_share": 1.01,
            "candidate_median_angle": 0.0, "candidate_p90_angle": 0.0,
            "candidate_min_conf": 1.01, "candidate_min_primitive": 1.01,
            "candidate_margin": 99.0, "candidate_max_gap_ms": 0,
            "candidate_max_span_ms": 0,
            "temporal_run": False, "run_min_samples": 99,
            "run_gap_ms": 0, "run_median_angle": 0.0,
            "run_p90_angle": 0.0, "run_min_conf": 1.01,
            "run_min_primitive": 1.01,
        },
        "adaptive": {
            "angle": 0.30, "small_angle": 0.18, "min_conf": 0.42,
            "min_primitive": 0.62, "margin": 0.25, "share": 0.80,
            "frame_anchors": 2, "fixation_anchors": 2, "fixation_gap_ms": 130,
            "candidate_fixation": False,
            "candidate_min_samples": 99, "candidate_target_share": 1.01,
            "candidate_body_fraction": 1.01, "candidate_single_share": 1.01,
            "candidate_median_angle": 0.0, "candidate_p90_angle": 0.0,
            "candidate_min_conf": 1.01, "candidate_min_primitive": 1.01,
            "candidate_margin": 99.0, "candidate_max_gap_ms": 0,
            "candidate_max_span_ms": 0,
            "temporal_run": False, "run_min_samples": 99,
            "run_gap_ms": 0, "run_median_angle": 0.0,
            "run_p90_angle": 0.0, "run_min_conf": 1.01,
            "run_min_primitive": 1.01,
        },
        "evidence_guided": {
            # The per-sample limits remain conservative.  The larger reduction
            # comes from auditable agreement across a complete fixation rather
            # than simply widening the uncertainty cone or deleting flags.
            "angle": 0.36, "small_angle": 0.22, "min_conf": 0.36,
            "min_primitive": 0.58, "margin": 0.22, "share": 0.82,
            "frame_anchors": 1, "fixation_anchors": 1, "fixation_gap_ms": 220,
            "candidate_fixation": True,
            "candidate_min_samples": 3, "candidate_target_share": 0.94,
            "candidate_body_fraction": 0.70, "candidate_single_share": 0.72,
            "candidate_median_angle": 0.52, "candidate_p90_angle": 0.82,
            "candidate_min_conf": 0.27, "candidate_min_primitive": 0.60,
            "candidate_margin": 0.24, "candidate_max_gap_ms": 110,
            "candidate_max_span_ms": 2500,
            "temporal_run": True, "run_min_samples": 4,
            "run_gap_ms": 75, "run_median_angle": 0.34,
            "run_p90_angle": 0.55, "run_min_conf": 0.34,
            "run_min_primitive": 0.66,
        },
        "semantic_guided": {
            # Recommended for coarse social-body labels.  It keeps the v0.21.53
            # fixation evidence limits but recognises that two raw primitives
            # may still represent one identical coarse person/body target.
            "angle": 0.40, "small_angle": 0.23, "min_conf": 0.34,
            "min_primitive": 0.56, "margin": 0.21, "share": 0.80,
            "frame_anchors": 1, "fixation_anchors": 1, "fixation_gap_ms": 240,
            "candidate_fixation": True,
            "candidate_min_samples": 3, "candidate_target_share": 0.92,
            "candidate_body_fraction": 0.68, "candidate_single_share": 0.66,
            "candidate_median_angle": 0.56, "candidate_p90_angle": 0.86,
            "candidate_min_conf": 0.25, "candidate_min_primitive": 0.58,
            "candidate_margin": 0.21, "candidate_max_gap_ms": 120,
            "candidate_max_span_ms": 2600,
            "temporal_run": True, "run_min_samples": 4,
            "run_gap_ms": 80, "run_median_angle": 0.37,
            "run_p90_angle": 0.60, "run_min_conf": 0.31,
            "run_min_primitive": 0.62,
            "temporal_exact_anchor_ms": 240,
        },
        "balanced": {
            "angle": 0.45, "small_angle": 0.28, "min_conf": 0.32,
            "min_primitive": 0.50, "margin": 0.18, "share": 0.74,
            "frame_anchors": 2, "fixation_anchors": 2, "fixation_gap_ms": 170,
            "candidate_fixation": True,
            "candidate_min_samples": 3, "candidate_target_share": 0.88,
            "candidate_body_fraction": 0.62, "candidate_single_share": 0.62,
            "candidate_median_angle": 0.66, "candidate_p90_angle": 1.00,
            "candidate_min_conf": 0.22, "candidate_min_primitive": 0.50,
            "candidate_margin": 0.18, "candidate_max_gap_ms": 150,
            "candidate_max_span_ms": 3000,
            "temporal_run": True, "run_min_samples": 3,
            "run_gap_ms": 95, "run_median_angle": 0.46,
            "run_p90_angle": 0.75, "run_min_conf": 0.25,
            "run_min_primitive": 0.54,
        },
        "off": {
            "angle": 0.0, "small_angle": 0.0, "min_conf": 1.01,
            "min_primitive": 1.01, "margin": 99.0, "share": 1.01,
            "frame_anchors": 99, "fixation_anchors": 99, "fixation_gap_ms": 0,
            "candidate_fixation": False,
            "candidate_min_samples": 99, "candidate_target_share": 1.01,
            "candidate_body_fraction": 1.01, "candidate_single_share": 1.01,
            "candidate_median_angle": 0.0, "candidate_p90_angle": 0.0,
            "candidate_min_conf": 1.01, "candidate_min_primitive": 1.01,
            "candidate_margin": 99.0, "candidate_max_gap_ms": 0,
            "candidate_max_span_ms": 0,
            "temporal_run": False, "run_min_samples": 99,
            "run_gap_ms": 0, "run_median_angle": 0.0,
            "run_p90_angle": 0.0, "run_min_conf": 1.01,
            "run_min_primitive": 1.01,
        },
    }[policy]

    # A fixation that changes person or body region is semantically ambiguous,
    # even when an individual sample has a small angular miss.  The new
    # fixation-aware policies therefore fail closed for the whole group instead
    # of letting an earlier per-sample rule partially empty a conflict.
    conflicted_fixations: set[str] = set()
    if policy in {"evidence_guided", "semantic_guided", "balanced"} and "fixation_id" in frame.columns:
        fixation_keys_for_guard = frame["fixation_id"].map(_valid_fixation_key)
        guarded = frame.loc[fixation_keys_for_guard.notna()].copy()
        guarded["_fixation_guard"] = fixation_keys_for_guard.loc[guarded.index]
        for fixation_key, group in guarded.groupby("_fixation_guard", sort=False):
            targets = {
                target
                for _, row in group.iterrows()
                if (target := _body_target_tuple(row)) is not None
            }
            if len(targets) > 1:
                conflicted_fixations.add(str(fixation_key))

    def resolve(
        index: Any,
        source: str,
        confidence: float,
        *,
        evidence_group: str = "",
        evidence_sample_count: int = 0,
        evidence_target_share: float | None = None,
        evidence_median_angle: float | None = None,
        evidence_p90_angle: float | None = None,
        evidence_exact_anchor_count: int = 0,
    ) -> None:
        frame.at[index, "review_required"] = False
        frame.at[index, "auto_review_resolved"] = True
        frame.at[index, "auto_review_resolution_source"] = source
        frame.at[index, "auto_review_confidence"] = float(np.clip(confidence, 0.0, 0.99))
        frame.at[index, "auto_review_evidence_group"] = evidence_group
        frame.at[index, "auto_review_evidence_sample_count"] = int(evidence_sample_count)
        frame.at[index, "auto_review_evidence_target_share"] = (
            float(evidence_target_share) if evidence_target_share is not None else np.nan
        )
        frame.at[index, "auto_review_evidence_median_angle_deg"] = (
            float(evidence_median_angle) if evidence_median_angle is not None else np.nan
        )
        frame.at[index, "auto_review_evidence_p90_angle_deg"] = (
            float(evidence_p90_angle) if evidence_p90_angle is not None else np.nan
        )
        frame.at[index, "auto_review_evidence_exact_anchor_count"] = int(
            evidence_exact_anchor_count
        )
        status = _body_text_value(frame.at[index, "assignment_status"]) or "body_3d_uncertainty_cone"
        if "_auto_" not in status:
            frame.at[index, "assignment_status"] = f"{status}_auto_{source}"

    def is_uncertainty_candidate(row: pd.Series, *, context_consensus: bool = False) -> bool:
        """Return True only for a genuine, auditable near-body candidate.

        Missing-frame rows and non-body statuses must never be cleared even if
        stale target columns are present.  Temporal consensus is deliberately
        restricted to a single surviving surface candidate; competing body
        surfaces remain a human decision.
        """
        status = _body_text_value(row.get("assignment_status")).lower()
        if not status.startswith("body_3d_uncertainty_cone_"):
            return False
        if bool(row.get("body_exact_hit")) or bool(row.get("self_occlusion")):
            return False
        fixation_key = _valid_fixation_key(row.get("fixation_id"))
        if fixation_key and fixation_key in conflicted_fixations:
            return False
        if _body_target_tuple(row) is None:
            return False
        # A missing video frame prevents a human preview and is intentionally
        # kept pending even though v0.21.52+ can still compute the 3D hit.
        preview_available = row.get("review_preview_available")
        if preview_available is not None and not bool(preview_available):
            return False
        scene_frame = pd.to_numeric(
            pd.Series([row.get("scene_frame_index")]), errors="coerce"
        ).iloc[0]
        if pd.isna(scene_frame):
            return False
        candidate_count = int(pd.to_numeric(
            pd.Series([row.get("candidate_hit_count")]), errors="coerce"
        ).fillna(0).iloc[0])
        if candidate_count < 1:
            return False
        if context_consensus and candidate_count > 1:
            if policy != "semantic_guided" or not _same_semantic_surface_candidates(row):
                return False
        return True

    # 1) A very small angular miss with a clear margin to the next candidate is
    # effectively an edge-inclusive body hit. Hands/feet use tighter limits.
    if policy != "off":
        for index, row in frame.loc[frame["review_required"]].iterrows():
            if not is_uncertainty_candidate(row):
                continue
            angle = pd.to_numeric(pd.Series([row.get("body_angular_miss_deg")]), errors="coerce").iloc[0]
            body_conf = pd.to_numeric(pd.Series([row.get("body_confidence")]), errors="coerce").fillna(0.0).iloc[0]
            primitive_conf = pd.to_numeric(pd.Series([row.get("first_hit_primitive_confidence")]), errors="coerce").fillna(0.0).iloc[0]
            second_angle = pd.to_numeric(pd.Series([row.get("second_hit_angular_miss_deg")]), errors="coerce").iloc[0]
            candidate_count = int(pd.to_numeric(pd.Series([row.get("candidate_hit_count")]), errors="coerce").fillna(0).iloc[0])
            raw = str(row.get("body_raw_region") or "").lower()
            small = raw.endswith("hand") or raw.endswith("foot")
            angle_limit = float(config["small_angle"] if small else config["angle"])
            margin_limit = float(config["margin"] + (0.08 if small else 0.0))
            same_semantic_surface = policy == "semantic_guided" and _same_semantic_surface_candidates(row)
            margin = float("inf") if pd.isna(second_angle) or candidate_count <= 1 or same_semantic_surface else float(second_angle) - float(angle)
            if pd.isna(angle) or float(angle) > angle_limit:
                continue
            if float(body_conf) < float(config["min_conf"]) + (0.04 if small else 0.0):
                continue
            if float(primitive_conf) < float(config["min_primitive"]):
                continue
            if margin < margin_limit:
                continue
            angular_score = max(0.0, 1.0 - float(angle) / max(angle_limit, 1e-9))
            margin_score = 1.0 if not np.isfinite(margin) else min(1.0, max(0.0, margin / max(margin_limit * 2.0, 1e-9)))
            confidence = 0.48 * float(body_conf) + 0.30 * angular_score + 0.22 * margin_score
            resolve(index, "same_coarse_surface" if same_semantic_surface else "near_body_margin", confidence)

    def anchor_rows() -> pd.DataFrame:
        exact = frame.get("body_exact_hit", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        self_occ = frame.get("self_occlusion", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        confidence = pd.to_numeric(frame.get("body_confidence", 0.0), errors="coerce").fillna(0.0)
        return frame.loc[exact & ~self_occ & (confidence >= 0.55)].copy()

    def consensus_target(rows: pd.DataFrame, minimum: int) -> tuple[tuple[str, str, str] | None, float, float]:
        weighted: dict[tuple[str, str, str], float] = defaultdict(float)
        counts: Counter = Counter()
        for _, anchor in rows.iterrows():
            target = _body_target_tuple(anchor)
            if target is None:
                continue
            weight = max(0.05, float(pd.to_numeric(pd.Series([anchor.get("body_confidence")]), errors="coerce").fillna(0.0).iloc[0]))
            weighted[target] += weight
            counts[target] += 1
        if not weighted:
            return None, 0.0, 0.0
        target, weight = max(weighted.items(), key=lambda item: item[1])
        share = weight / max(sum(weighted.values()), 1e-9)
        if counts[target] < minimum or share < float(config["share"]):
            return None, share, 0.0
        mean_conf = float(pd.to_numeric(rows.loc[
            rows.apply(lambda row: _body_target_tuple(row) == target, axis=1), "body_confidence"
        ], errors="coerce").fillna(0.0).mean())
        return target, share, mean_conf

    # 2) Multiple exact high-rate gaze samples in the same video frame provide
    # strong evidence for nearby uncertainty candidates with the same target.
    if policy != "off" and "scene_frame_index" in frame.columns:
        anchors = anchor_rows()
        unresolved = frame.loc[frame["review_required"]]
        for frame_value, group in unresolved.groupby("scene_frame_index", dropna=True, sort=False):
            same = anchors.loc[anchors["scene_frame_index"] == frame_value]
            target, share, mean_conf = consensus_target(same, int(config["frame_anchors"]))
            if target is None:
                continue
            for index, row in group.iterrows():
                if not is_uncertainty_candidate(row, context_consensus=True):
                    continue
                if _body_target_tuple(row) != target:
                    continue
                resolve(index, "same_frame_exact_consensus", 0.72 * mean_conf + 0.28 * share)

    # 3) Same-fixation exact-hit consensus removes repeated per-sample review
    # without allowing temporal context to invent a different body target.
    if policy != "off" and "fixation_id" in frame.columns:
        anchors = anchor_rows()
        fixation_keys = frame["fixation_id"].map(_valid_fixation_key)
        anchor_keys = anchors["fixation_id"].map(_valid_fixation_key) if "fixation_id" in anchors.columns else pd.Series(None, index=anchors.index)
        unresolved = frame.loc[frame["review_required"]].copy()
        unresolved["_fixation_key"] = fixation_keys.loc[unresolved.index]
        for fixation_key, group in unresolved.dropna(subset=["_fixation_key"]).groupby("_fixation_key", sort=False):
            same = anchors.loc[anchor_keys == fixation_key]
            target, share, mean_conf = consensus_target(same, int(config["fixation_anchors"]))
            if target is None or same.empty:
                continue
            anchor_times = pd.to_numeric(same.get("timestamp_ns", pd.Series(np.nan, index=same.index)), errors="coerce").dropna().astype("int64").to_numpy()
            for index, row in group.iterrows():
                if not is_uncertainty_candidate(row, context_consensus=True):
                    continue
                if _body_target_tuple(row) != target:
                    continue
                timestamp = pd.to_numeric(pd.Series([row.get("timestamp_ns")]), errors="coerce").iloc[0]
                if pd.isna(timestamp) or not len(anchor_times):
                    continue
                nearest_gap_ms = float(np.min(np.abs(anchor_times - int(timestamp))) / 1e6)
                if nearest_gap_ms > float(config["fixation_gap_ms"]):
                    continue
                temporal_score = max(0.0, 1.0 - nearest_gap_ms / max(float(config["fixation_gap_ms"]), 1e-9))
                resolve(index, "same_fixation_exact_consensus", 0.60 * mean_conf + 0.25 * share + 0.15 * temporal_score)


    # 3b) Exporters may split one visual fixation into adjacent fixation IDs.
    # Under the semantic policy, a nearby exact hit can anchor a candidate only
    # when every exact body target in the local time window agrees and the
    # candidate has no true coarse-label surface conflict.
    if policy == "semantic_guided" and "timestamp_ns" in frame.columns:
        anchors = anchor_rows().copy()
        if not anchors.empty:
            anchor_times_all = pd.to_numeric(anchors["timestamp_ns"], errors="coerce")
            anchors = anchors.loc[anchor_times_all.notna()].copy()
            anchors["_anchor_time"] = anchor_times_all.loc[anchors.index].astype("int64")
            window_ns = int(float(config.get("temporal_exact_anchor_ms", 0)) * 1e6)
            for index, row in frame.loc[frame["review_required"]].iterrows():
                if not is_uncertainty_candidate(row, context_consensus=True):
                    continue
                target = _body_target_tuple(row)
                timestamp = pd.to_numeric(pd.Series([row.get("timestamp_ns")]), errors="coerce").iloc[0]
                if target is None or pd.isna(timestamp) or window_ns <= 0:
                    continue
                local = anchors.loc[(anchors["_anchor_time"] - int(timestamp)).abs() <= window_ns]
                if local.empty:
                    continue
                local_targets = {
                    candidate for _, anchor in local.iterrows()
                    if (candidate := _body_target_tuple(anchor)) is not None
                }
                if local_targets != {target}:
                    continue
                nearest_gap_ms = float((local["_anchor_time"] - int(timestamp)).abs().min() / 1e6)
                mean_conf = float(pd.to_numeric(local["body_confidence"], errors="coerce").fillna(0.0).mean())
                temporal_score = max(0.0, 1.0 - nearest_gap_ms / max(float(config["temporal_exact_anchor_ms"]), 1e-9))
                resolve(
                    index,
                    "adjacent_exact_anchor",
                    0.70 * mean_conf + 0.30 * temporal_score,
                    evidence_group=f"temporal_anchor:{target[0]}:{target[1]}:{target[2]}",
                    evidence_sample_count=int(len(local)),
                    evidence_target_share=1.0,
                    evidence_exact_anchor_count=int(len(local)),
                )

    def numeric(values: pd.Series | Any) -> pd.Series:
        if isinstance(values, pd.Series):
            return pd.to_numeric(values, errors="coerce")
        return pd.to_numeric(pd.Series(values), errors="coerce")

    def candidate_margin(row: pd.Series) -> float:
        first = numeric([row.get("body_angular_miss_deg")]).iloc[0]
        second = numeric([row.get("second_hit_angular_miss_deg")]).iloc[0]
        count = int(numeric([row.get("candidate_hit_count")]).fillna(0).iloc[0])
        if count <= 1 or pd.isna(second):
            return float("inf")
        if pd.isna(first):
            return float("-inf")
        return float(second) - float(first)

    # 4) A fixation can be internally decisive even when none of its samples
    # crosses the exact primitive boundary.  This is the main v0.21.54 review
    # reduction: the target is never changed, and a complete group is cleared
    # only when its near-body evidence is homogeneous, temporally contiguous,
    # well inside the uncertainty cone and not contradicted by an exact hit.
    if bool(config.get("candidate_fixation")) and "fixation_id" in frame.columns:
        fixation_keys = frame["fixation_id"].map(_valid_fixation_key)
        for fixation_key, indices in fixation_keys.dropna().groupby(
            fixation_keys.dropna(), sort=False
        ).groups.items():
            group = frame.loc[list(indices)].sort_values("timestamp_ns")
            pending = group.loc[group["review_required"]].copy()
            if len(pending) < int(config["candidate_min_samples"]):
                continue
            if pending.get("self_occlusion", pd.Series(False, index=pending.index)).fillna(False).astype(bool).any():
                continue
            if not all(is_uncertainty_candidate(row) for _, row in pending.iterrows()):
                continue

            valid_ray = group.get("ray_valid", pd.Series(True, index=group.index)).fillna(False).astype(bool)
            valid_group = group.loc[valid_ray]
            if valid_group.empty:
                continue
            target_rows = valid_group.loc[
                valid_group.apply(lambda row: _body_target_tuple(row) is not None, axis=1)
            ]
            body_fraction = float(len(target_rows) / max(1, len(valid_group)))
            if body_fraction < float(config["candidate_body_fraction"]):
                continue

            weighted: dict[tuple[str, str, str], float] = defaultdict(float)
            counts: Counter = Counter()
            for _, row in target_rows.iterrows():
                target = _body_target_tuple(row)
                if target is None:
                    continue
                weight = max(
                    0.05,
                    float(numeric([row.get("body_confidence")]).fillna(0.0).iloc[0]),
                )
                weighted[target] += weight
                counts[target] += 1
            if not weighted:
                continue
            target, target_weight = max(weighted.items(), key=lambda item: item[1])
            target_share = float(target_weight / max(sum(weighted.values()), 1e-9))
            if target_share < float(config["candidate_target_share"]):
                continue
            if any(_body_target_tuple(row) != target for _, row in pending.iterrows()):
                continue

            exact_rows = group.loc[
                group.get("body_exact_hit", pd.Series(False, index=group.index)).fillna(False).astype(bool)
            ]
            exact_targets = {
                candidate
                for _, row in exact_rows.iterrows()
                if (candidate := _body_target_tuple(row)) is not None
            }
            if exact_targets and exact_targets != {target}:
                continue
            exact_anchor_count = int(sum(
                _body_target_tuple(row) == target for _, row in exact_rows.iterrows()
            ))

            angles = numeric(pending["body_angular_miss_deg"]).dropna()
            confidences = numeric(pending["body_confidence"]).dropna()
            primitives = numeric(pending.get("first_hit_primitive_confidence", 0.0)).dropna()
            if angles.empty or confidences.empty or primitives.empty:
                continue
            median_angle = float(angles.median())
            p90_angle = float(angles.quantile(0.90))
            median_conf = float(confidences.median())
            median_primitive = float(primitives.median())

            # One exact anchor inside the same fixation is strong geometric
            # evidence and permits a modest relaxation, but never overrides a
            # target conflict or a close competing surface.
            anchor_relax = 1.18 if exact_anchor_count else 1.0
            if median_angle > float(config["candidate_median_angle"]) * anchor_relax:
                continue
            if p90_angle > float(config["candidate_p90_angle"]) * anchor_relax:
                continue
            if median_conf < float(config["candidate_min_conf"]) / anchor_relax:
                continue
            if median_primitive < float(config["candidate_min_primitive"]):
                continue

            candidate_counts = numeric(pending.get("candidate_hit_count", 0)).fillna(0).astype(int)
            margins = pd.Series(
                [candidate_margin(row) for _, row in pending.iterrows()],
                index=pending.index,
                dtype=float,
            )
            semantic_surface = pd.Series(
                [policy == "semantic_guided" and _same_semantic_surface_candidates(row) for _, row in pending.iterrows()],
                index=pending.index,
                dtype=bool,
            )
            acceptable_surface = (candidate_counts <= 1) | semantic_surface | (
                margins >= float(config["candidate_margin"])
            )
            single_share = float(((candidate_counts <= 1) | semantic_surface).mean())
            acceptable_share = float(acceptable_surface.mean())
            if single_share < float(config["candidate_single_share"]) and acceptable_share < 0.92:
                continue
            if not bool(acceptable_surface.all()):
                # Close competing surfaces remain a person-level decision.  We
                # do not partially clear a fixation because that would leave a
                # visually misleading "mostly resolved" group in the queue.
                continue

            timestamps = numeric(group.get("timestamp_ns", np.nan)).dropna().astype("int64")
            if timestamps.empty:
                continue
            timestamp_values = np.sort(timestamps.to_numpy())
            span_ms = float((timestamp_values[-1] - timestamp_values[0]) / 1e6)
            gaps_ms = np.diff(timestamp_values).astype(float) / 1e6
            maximum_gap_ms = float(gaps_ms.max()) if len(gaps_ms) else 0.0
            if span_ms > float(config["candidate_max_span_ms"]):
                continue
            if maximum_gap_ms > float(config["candidate_max_gap_ms"]):
                continue

            angular_score = max(
                0.0,
                1.0
                - median_angle
                / max(float(config["candidate_median_angle"]) * anchor_relax, 1e-9),
            )
            continuity_score = max(
                0.0,
                1.0
                - maximum_gap_ms
                / max(float(config["candidate_max_gap_ms"]), 1e-9),
            )
            confidence = float(np.clip(
                0.30 * median_conf
                + 0.24 * target_share
                + 0.18 * body_fraction
                + 0.16 * angular_score
                + 0.12 * continuity_score,
                0.0,
                0.97,
            ))
            for index in pending.index:
                resolve(
                    index,
                    "semantic_fixation_consensus" if policy == "semantic_guided" else "same_fixation_candidate_consensus",
                    confidence,
                    evidence_group=f"fixation:{fixation_key}",
                    evidence_sample_count=int(len(pending)),
                    evidence_target_share=target_share,
                    evidence_median_angle=median_angle,
                    evidence_p90_angle=p90_angle,
                    evidence_exact_anchor_count=exact_anchor_count,
                )

    # 5) Fixation IDs can be absent or split by an exporter.  Very short,
    # unanimous high-rate runs are allowed to clear only under tighter limits
    # than the fixation rule.  This never crosses a target change, a preview
    # gap, self occlusion or a competing-surface row.
    if bool(config.get("temporal_run")) and "timestamp_ns" in frame.columns:
        unresolved = frame.loc[frame["review_required"]].copy()
        if not unresolved.empty:
            unresolved["_target"] = unresolved.apply(_body_target_tuple, axis=1)
            unresolved["_fixation"] = unresolved.get(
                "fixation_id", pd.Series(None, index=unresolved.index)
            ).map(_valid_fixation_key)
            unresolved = unresolved.loc[unresolved["_fixation"].isna()].sort_values("timestamp_ns")
            runs: list[list[Any]] = []
            current: list[Any] = []
            previous_target = None
            previous_timestamp = None
            for index, row in unresolved.iterrows():
                if not is_uncertainty_candidate(row, context_consensus=True):
                    if current:
                        runs.append(current)
                    current, previous_target, previous_timestamp = [], None, None
                    continue
                target = row["_target"]
                timestamp = numeric([row.get("timestamp_ns")]).iloc[0]
                gap_ms = (
                    float((timestamp - previous_timestamp) / 1e6)
                    if previous_timestamp is not None and not pd.isna(timestamp)
                    else 0.0
                )
                if (
                    current
                    and (
                        target != previous_target
                        or pd.isna(timestamp)
                        or gap_ms < 0
                        or gap_ms > float(config["run_gap_ms"])
                    )
                ):
                    runs.append(current)
                    current = []
                current.append(index)
                previous_target = target
                previous_timestamp = timestamp
            if current:
                runs.append(current)

            for run_number, indices in enumerate(runs):
                if len(indices) < int(config["run_min_samples"]):
                    continue
                run = frame.loc[indices]
                if run.get("self_occlusion", pd.Series(False, index=run.index)).fillna(False).astype(bool).any():
                    continue
                angles = numeric(run["body_angular_miss_deg"]).dropna()
                confidences = numeric(run["body_confidence"]).dropna()
                primitives = numeric(run.get("first_hit_primitive_confidence", 0.0)).dropna()
                if angles.empty or confidences.empty or primitives.empty:
                    continue
                median_angle = float(angles.median())
                p90_angle = float(angles.quantile(0.90))
                if median_angle > float(config["run_median_angle"]):
                    continue
                if p90_angle > float(config["run_p90_angle"]):
                    continue
                if float(confidences.median()) < float(config["run_min_conf"]):
                    continue
                if float(primitives.median()) < float(config["run_min_primitive"]):
                    continue
                confidence = float(np.clip(
                    0.46 * float(confidences.median())
                    + 0.30 * max(0.0, 1.0 - median_angle / max(float(config["run_median_angle"]), 1e-9))
                    + 0.24,
                    0.0,
                    0.94,
                ))
                for index in indices:
                    resolve(
                        index,
                        "short_temporal_candidate_run",
                        confidence,
                        evidence_group=f"temporal_run:{run_number}",
                        evidence_sample_count=int(len(indices)),
                        evidence_target_share=1.0,
                        evidence_median_angle=median_angle,
                        evidence_p90_angle=p90_angle,
                    )

    before = frame["review_required_before_auto_resolution"].fillna(False).astype(bool)
    after = frame["review_required"].fillna(False).astype(bool)
    resolved = frame["auto_review_resolved"].fillna(False).astype(bool)
    before_count, after_count, resolved_count = int(before.sum()), int(after.sum()), int(resolved.sum())
    source_counts = Counter(frame.loc[resolved, "auto_review_resolution_source"].fillna("").astype(str))
    reason_before = Counter(frame.loc[before, "review_reason_before_auto_resolution"].fillna("unknown").astype(str))
    reason_after = Counter(frame.loc[after].apply(_body_review_reason, axis=1).astype(str)) if after_count else Counter()
    before_frames = int(frame.loc[before, "scene_frame_index"].nunique()) if "scene_frame_index" in frame.columns else 0
    after_frames = int(frame.loc[after, "scene_frame_index"].nunique()) if "scene_frame_index" in frame.columns else 0
    before_fixations = int(frame.loc[before, "fixation_id"].map(_valid_fixation_key).dropna().nunique()) if "fixation_id" in frame.columns else 0
    after_fixations = int(frame.loc[after, "fixation_id"].map(_valid_fixation_key).dropna().nunique()) if "fixation_id" in frame.columns else 0

    def review_unit_count(mask: pd.Series) -> int:
        keys: set[str] = set()
        for index, row in frame.loc[mask].iterrows():
            fixation = _valid_fixation_key(row.get("fixation_id"))
            if fixation:
                keys.add(f"fixation:{fixation}")
            else:
                uid = _body_text_value(row.get("gaze_sample_uid"))
                keys.add(f"sample:{uid or index}")
        return int(len(keys))

    before_units = review_unit_count(before)
    after_units = review_unit_count(after)
    report = {
        "schema_version": "oculostudio-body-aoi-auto-review-1.1",
        "policy": policy,
        "policy_thresholds": {key: value for key, value in config.items()},
        "conflicted_fixation_count": int(len(conflicted_fixations)),
        "review_samples_before": before_count,
        "auto_resolved_samples": resolved_count,
        "review_samples_remaining": after_count,
        "sample_review_reduction_fraction": float((before_count - after_count) / before_count) if before_count else 0.0,
        "review_frames_before": before_frames,
        "review_frames_remaining": after_frames,
        "frame_review_reduction_fraction": float((before_frames - after_frames) / before_frames) if before_frames else 0.0,
        "review_fixations_before": before_fixations,
        "review_fixations_remaining": after_fixations,
        "fixation_review_reduction_fraction": float((before_fixations - after_fixations) / before_fixations) if before_fixations else 0.0,
        "review_units_before": before_units,
        "review_units_remaining": after_units,
        "review_unit_reduction_fraction": float((before_units - after_units) / before_units) if before_units else 0.0,
        "resolution_source_counts": dict(sorted(source_counts.items())),
        "review_reason_counts_before": dict(sorted(reason_before.items())),
        "remaining_review_reason_counts": dict(sorted(reason_after.items())),
        "safety_guards": [
            "automatic target relation, entity and body region are never changed",
            "video-frame availability affects preview only and never suppresses 3D Body AOI computation",
            "self-occlusion rows remain in human review",
            "other-person and unresolved-entity candidates remain in human review",
            "near-body acceptance requires angular, confidence and competing-candidate margins",
            "frame and fixation consensus require matching exact-hit targets",
            "candidate-only fixation consensus requires one unchanged social-body target, high target share, temporal continuity, bounded angular miss and no close competing surface",
            "short temporal runs never cross target changes or fixation boundaries",
            "all cleared flags retain sample-level audit columns",
        ],
    }
    rows = [
        {"scope": "total", "source": "review_samples_before", "count": before_count, "fraction_of_pre_auto_review": 1.0 if before_count else 0.0},
        {"scope": "total", "source": "auto_resolved_samples", "count": resolved_count, "fraction_of_pre_auto_review": float(resolved_count / before_count) if before_count else 0.0},
        {"scope": "total", "source": "review_samples_remaining", "count": after_count, "fraction_of_pre_auto_review": float(after_count / before_count) if before_count else 0.0},
        {"scope": "review_units", "source": "review_units_before", "count": before_units, "fraction_of_pre_auto_review": 1.0 if before_units else 0.0},
        {"scope": "review_units", "source": "review_units_remaining", "count": after_units, "fraction_of_pre_auto_review": float(after_units / before_units) if before_units else 0.0},
    ]
    rows.extend({"scope": "resolution_source", "source": key, "count": int(value), "fraction_of_pre_auto_review": float(value / before_count) if before_count else 0.0} for key, value in sorted(source_counts.items()))
    rows.extend({"scope": "remaining_review_reason", "source": key, "count": int(value), "fraction_of_pre_auto_review": float(value / before_count) if before_count else 0.0} for key, value in sorted(reason_after.items()))
    return frame, report, pd.DataFrame(rows)


def _restore_body_review_baseline(samples: pd.DataFrame) -> pd.DataFrame:
    """Restore the immutable pre-auto-review queue from an existing artifact.

    v0.21.49+ stores ``review_required_before_auto_resolution`` for every row.
    Reprocessing therefore does not reinterpret a prior auto-resolution as
    human truth.  Older sample-clock artifacts can fall back to the original
    geometric rule: every valid near-body (non-exact) candidate is reviewable.
    """
    frame = samples.copy()
    if "review_required_before_auto_resolution" in frame.columns:
        baseline = frame["review_required_before_auto_resolution"].fillna(False).astype(bool)
    else:
        exact = frame.get("body_exact_hit", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        target = frame.apply(lambda row: _body_target_tuple(row) is not None, axis=1)
        status = frame.get("assignment_status", pd.Series("", index=frame.index)).fillna("").astype(str)
        baseline = (~exact) & target & status.str.startswith("body_3d_uncertainty_cone_")
    frame["review_required"] = baseline

    # Auto-review appends an audit suffix to the geometric assignment status.
    # Strip only that suffix so the new policy sees the original evidence class.
    if "assignment_status" in frame.columns:
        frame["assignment_status"] = (
            frame["assignment_status"]
            .fillna("")
            .astype(str)
            .str.replace(r"_auto_.*$", "", regex=True)
        )
    return frame


def reprocess_body_aoi_review_reduction(
    store: ProjectStore,
    *,
    session_id: str,
    body_aoi_artifact_id: str,
    auto_review_policy: Literal[
        "off", "conservative", "adaptive", "evidence_guided", "semantic_guided", "balanced"
    ] = "semantic_guided",
    label: str | None = None,
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Create a child Body AOI with a newly evaluated review queue.

    Ray/body intersections and automatic targets are copied unchanged.  Only
    the auditable ``review_required`` decision and its auto-resolution evidence
    are recomputed, so a v0.21.52 sample-clock artifact can be upgraded without rerunning C3D interpolation or 3D intersection.
    """
    token = cancellation or CancellationToken.create()
    if auto_review_policy not in BODY_AUTO_REVIEW_POLICIES:
        raise ValidationError(
            "body auto_review_policy must be off, conservative, adaptive, "
            "evidence_guided, semantic_guided or balanced"
        )
    session = store.project.sessions[session_id]
    source = session.artifacts.get(body_aoi_artifact_id)
    if source is None or source.kind != ArtifactKind.BODY_AOI:
        raise ValidationError("Select an existing Body AOI artifact")
    if str(source.qc.get("body_geometry_clock") or "") != "gaze_sample_timestamp":
        raise ValidationError(
            "Review-only reduction requires a v0.21.52+ gaze-sample-clock Body AOI. "
            "Regenerate older video-frame-clock artifacts first."
        )
    if not bool(source.qc.get("body_geometry_review_ready")):
        raise ValidationError(
            "Review-only reduction is blocked because the source Body AOI did not pass "
            "the torso/body geometry integrity gate."
        )
    manual_draft = (
        store.annotations_dir(session_id, "body_aoi")
        / safe_name(body_aoi_artifact_id) / "manual_review.json"
    )
    if manual_draft.exists():
        try:
            draft_payload = json.loads(manual_draft.read_text("utf-8"))
        except Exception as exc:
            raise ValidationError(
                "The Body AOI manual-review draft could not be read. Repair or remove "
                "the draft before re-evaluating the automatic queue."
            ) from exc
        decisions = draft_payload.get("decisions") if isinstance(draft_payload, dict) else None
        if isinstance(decisions, dict) and decisions:
            raise ValidationError(
                "This Body AOI already has saved manual-review decisions. Finish or clear "
                "that draft before creating a review-reduced child artifact, so human "
                "decisions are never silently discarded."
            )

    emit(progress, 0.08, "Reading the existing Body AOI evidence")
    token.raise_if_cancelled()
    samples_path = find_artifact_file(source, names=["body_aoi_samples.csv"])
    samples = pd.read_csv(samples_path, low_memory=False)
    original_targets = samples[[
        column for column in (
            "body_relation", "body_entity_id", "body_raw_region",
            "body_coarse_region", "body_region", "body_social_region",
            "body_exact_hit", "body_angular_miss_deg", "body_distance_mm",
        ) if column in samples.columns
    ]].copy()

    emit(progress, 0.28, "Restoring the original near-body review queue")
    baseline = _restore_body_review_baseline(samples)
    token.raise_if_cancelled()
    emit(progress, 0.48, "Applying fixation-aware evidence reduction")
    reduced, auto_report, auto_table = _apply_body_auto_review_reduction(
        baseline, auto_review_policy
    )
    token.raise_if_cancelled()

    # A review-reduction pass is never allowed to change semantic or geometric
    # automatic evidence.  Fail closed if a future code change violates this.
    if not original_targets.equals(reduced[original_targets.columns]):
        raise ValidationError(
            "Review reduction attempted to change an automatic Body AOI target; "
            "the derived artifact was not created."
        )

    summary = _frame_summary(reduced)
    artifact_id = new_id("artifact")
    output_root = (
        store.derived_dir(session_id, "body_aoi", source.clip_id or "full")
        / artifact_id
    )
    output_root.mkdir(parents=True, exist_ok=False)
    samples_csv = output_root / "body_aoi_samples.csv"
    frames_csv = output_root / "body_aoi_frames.csv"
    auto_csv = output_root / "body_aoi_auto_review_summary.csv"
    auto_json = output_root / "body_aoi_auto_review_summary.json"
    reduced.to_csv(samples_csv, index=False)
    summary.to_csv(frames_csv, index=False)
    auto_table.to_csv(auto_csv, index=False)
    atomic_write_json(auto_json, auto_report)

    emit(progress, 0.70, "Copying immutable body geometry evidence")
    source_root = artifact_root(source)
    copied_support: dict[str, str] = {}
    for filename in (
        "body_regions.csv",
        "body_aoi_timebase.json",
        "body_geometry_preflight.json",
        "body_geometry_preflight_diagnostic.json",
    ):
        candidate = find_artifact_file(source, names=[filename], required=False)
        if candidate is None:
            continue
        target = output_root / candidate.name
        shutil.copy2(candidate, target)
        copied_support[filename] = str(target)
    if output_parquet:
        try:
            reduced.to_parquet(output_root / "body_aoi_samples.parquet", index=False)
            summary.to_parquet(output_root / "body_aoi_frames.parquet", index=False)
            regions_source = find_artifact_file(source, names=["body_regions.csv"], required=False)
            if regions_source is not None:
                pd.read_csv(regions_source, low_memory=False).to_parquet(
                    output_root / "body_regions.parquet", index=False
                )
        except Exception:
            pass

    qc = dict(source.qc or {})
    qc.update({
        "review_required_fraction": float(reduced["review_required"].mean()) if len(reduced) else 0.0,
        "review_required_fraction_before_auto_resolution": float(
            reduced["review_required_before_auto_resolution"].mean()
        ) if len(reduced) else 0.0,
        "auto_review_policy": auto_review_policy,
        "review_samples_before_auto_resolution": int(auto_report["review_samples_before"]),
        "auto_resolved_sample_count": int(auto_report["auto_resolved_samples"]),
        "review_samples_remaining": int(auto_report["review_samples_remaining"]),
        "auto_review_sample_reduction_fraction": float(auto_report["sample_review_reduction_fraction"]),
        "review_frames_before_auto_resolution": int(auto_report["review_frames_before"]),
        "review_frames_remaining": int(auto_report["review_frames_remaining"]),
        "review_fixations_before_auto_resolution": int(auto_report["review_fixations_before"]),
        "review_fixations_remaining": int(auto_report["review_fixations_remaining"]),
        "review_units_before_auto_resolution": int(auto_report["review_units_before"]),
        "review_units_remaining": int(auto_report["review_units_remaining"]),
        "auto_review_unit_reduction_fraction": float(auto_report["review_unit_reduction_fraction"]),
        "auto_review_resolution_source_counts": dict(auto_report["resolution_source_counts"]),
        "auto_review_conflicted_fixation_count": int(auto_report.get("conflicted_fixation_count", 0)),
        "auto_review_policy_thresholds": dict(auto_report.get("policy_thresholds") or {}),
        "remaining_review_reason_counts": dict(auto_report["remaining_review_reason_counts"]),
        "review_reduction_reprocessed": True,
        "review_reduction_source_artifact_id": body_aoi_artifact_id,
        "review_reduction_engine": "fixation_evidence_v2",
    })
    parameters = dict(source.parameters or {})
    parameters.update({
        "auto_review_policy": auto_review_policy,
        "review_reduction_only": True,
        "source_body_aoi_artifact_id": body_aoi_artifact_id,
    })
    manifest = {
        "schema_version": "oculostudio-body-aoi-2.4",
        "artifact_id": artifact_id,
        "source_body_aoi_artifact_id": body_aoi_artifact_id,
        "review_reduction_only": True,
        "parameters": parameters,
        "outputs": {
            "samples_csv": str(samples_csv),
            "frames_csv": str(frames_csv),
            "auto_review_summary_csv": str(auto_csv),
            "auto_review_summary_json": str(auto_json),
            **copied_support,
        },
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "body_aoi_manifest.json", manifest)
    atomic_write_json(output_root / "qc_metrics.json", qc)
    emit(progress, 0.90, "Registering the reduced Body review artifact")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.BODY_AOI,
        label=label or f"{source.label} · reduced review",
        output_root=output_root,
        source_asset_ids=list(source.source_asset_ids),
        parent_artifact_ids=[body_aoi_artifact_id],
        clip_id=source.clip_id,
        parameters=parameters,
        qc=qc,
        provenance={
            **dict(source.provenance or {}),
            "source_body_aoi_artifact_id": body_aoi_artifact_id,
            "review_reduction_only": True,
            "body_auto_review_policy": auto_review_policy,
            "review_reduction_engine": "fixation_evidence_v2",
            "automatic_body_targets_unchanged": True,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Body AOI review queue reduced")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "samples_csv": str(samples_csv),
        "frames_csv": str(frames_csv),
        "auto_review_summary_csv": str(auto_csv),
        "auto_review_summary_json": str(auto_json),
        "manifest": str(output_root / "body_aoi_manifest.json"),
        "review_units_before": int(auto_report["review_units_before"]),
        "review_units_remaining": int(auto_report["review_units_remaining"]),
    }



def create_body_aoi_annotations(
    store: ProjectStore,
    *,
    session_id: str,
    gaze_rays_artifact_id: str,
    timeseries_asset_id: str | None = None,
    body_asset_ids: list[str] | None = None,
    body_profile_path: str | None = None,
    entity_marker_prefixes: dict[str, str] | None = None,
    etm_assignment_artifact_id: str | None = None,
    native_trim_artifact_id: str | None = None,
    wearer_role_id: str | None = None,
    wearer_body_asset_id: str | None = None,
    partner_body_asset_id: str | None = None,
    include_self_body: bool = True,
    exclude_self_head_neck: bool = True,
    self_near_origin_exclusion_m: float = 0.20,
    first_hit_occlusion: bool = True,
    output_granularity: Literal["coarse_social_regions", "detailed_regions"] = "coarse_social_regions",
    auto_review_policy: Literal[
        "off", "conservative", "adaptive", "evidence_guided", "semantic_guided", "balanced"
    ] = "semantic_guided",
    geometry_acceptance_policy: Literal["required", "warn", "off"] = "required",
    minimum_torso_coverage: float = DEFAULT_MINIMUM_TORSO_COVERAGE,
    minimum_body_frame_coverage: float = DEFAULT_MINIMUM_BODY_FRAME_COVERAGE,
    max_marker_gap_sec: float = 0.1,
    uncertainty_cone_deg: float = 1.5,
    maximum_target_distance_m: float = 10.0,
    scene_frame_tolerance_ms: float = 20.0,
    sample_clock_chunk_size: int = 4096,
    label: str = "Body AOI",
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    body_asset_ids = list(body_asset_ids or [])
    entity_marker_prefixes = dict(entity_marker_prefixes or {})
    session = store.project.sessions[session_id]
    assignment_entity_ids: dict[str, str] = {}
    assignment_rows: dict[str, dict[str, Any]] = {}
    assignment_artifact = None
    if etm_assignment_artifact_id:
        assignment_artifact = session.artifacts.get(etm_assignment_artifact_id)
        if assignment_artifact is None:
            raise ValidationError("Selected ETM-role assignment artifact is unavailable")
        assignment = load_etm_assignment(assignment_artifact)
        assignment_rows = assignment.get("assignments") or {}
        if assignment.get("status") != "resolved":
            raise ValidationError("ETM-role assignment requires review and cannot drive Body AOI automatically")
        for role_id, row in assignment_rows.items():
            body_id = str(row.get("body_asset_id") or "")
            if not body_id:
                continue
            assignment_entity_ids[body_id] = str(role_id)
            if row.get("body_subject_prefix") and body_id not in entity_marker_prefixes:
                entity_marker_prefixes[body_id] = row["body_subject_prefix"]
    gaze_artifact = session.artifacts.get(gaze_rays_artifact_id)
    if gaze_artifact is None or gaze_artifact.kind != ArtifactKind.GAZE_RAYS:
        raise ValidationError("Select a Gaze 3D ray artifact")

    wearer_role_id = str(
        wearer_role_id
        or (gaze_artifact.parameters or {}).get("wearer_role_id")
        or (gaze_artifact.qc or {}).get("wearer_role_id")
        or ""
    ).strip().upper() or None
    partner_role_id: str | None = None
    if assignment_rows:
        if wearer_role_id is None:
            gaze_tracker = str((gaze_artifact.parameters or {}).get("tracker_asset_id") or "")
            candidates = [
                str(role_id) for role_id, row in assignment_rows.items()
                if gaze_tracker and str(row.get("etm_asset_id") or "") == gaze_tracker
            ]
            if len(candidates) == 1:
                wearer_role_id = candidates[0].upper()
        if wearer_role_id is None:
            raise ValidationError("Body AOI needs the gaze wearer role so self and partner can be distinguished")
        if wearer_role_id not in assignment_rows:
            raise ValidationError(
                f"Wearer role {wearer_role_id!r} is not present in the selected ETM-role assignment"
            )
        role_row = assignment_rows[wearer_role_id]
        wearer_body_asset_id = wearer_body_asset_id or str(role_row.get("body_asset_id") or "") or None
        partner_roles = [str(role).upper() for role in assignment_rows if str(role).upper() != wearer_role_id]
        if len(partner_roles) == 1:
            partner_role_id = partner_roles[0]
            partner_body_asset_id = partner_body_asset_id or str(
                assignment_rows[partner_role_id].get("body_asset_id") or ""
            ) or None
        elif partner_body_asset_id:
            partner_role_id = assignment_entity_ids.get(partner_body_asset_id)
        else:
            raise ValidationError(
                "Body AOI currently requires one partner role in the selected dyadic assignment"
            )

    selected_body_ids: list[str] = []
    if include_self_body and wearer_body_asset_id:
        selected_body_ids.append(wearer_body_asset_id)
    if partner_body_asset_id:
        selected_body_ids.append(partner_body_asset_id)
    selected_body_ids.extend(body_asset_ids)
    body_asset_ids = list(dict.fromkeys(value for value in selected_body_ids if value))
    if not include_self_body and wearer_body_asset_id:
        body_asset_ids = [value for value in body_asset_ids if value != wearer_body_asset_id]
    if not body_asset_ids:
        raise ValidationError("Select a partner Body C3D; self Body C3D is optional but recommended")
    if partner_body_asset_id and partner_body_asset_id not in body_asset_ids:
        body_asset_ids.append(partner_body_asset_id)
    if include_self_body and not wearer_body_asset_id:
        raise ValidationError("Include self body is enabled, but no wearer Body C3D could be resolved")
    if output_granularity not in {"coarse_social_regions", "detailed_regions"}:
        raise ValidationError("output_granularity must be coarse_social_regions or detailed_regions")
    if geometry_acceptance_policy not in {"required", "warn", "off"}:
        raise ValidationError("geometry_acceptance_policy must be required, warn or off")
    if not 0.0 <= float(minimum_torso_coverage) <= 1.0:
        raise ValidationError("minimum_torso_coverage must be between 0 and 1")
    if not 0.0 <= float(minimum_body_frame_coverage) <= 1.0:
        raise ValidationError("minimum_body_frame_coverage must be between 0 and 1")
    if self_near_origin_exclusion_m < 0 or self_near_origin_exclusion_m > 2:
        raise ValidationError("self_near_origin_exclusion_m must be between 0 and 2 metres")
    if int(sample_clock_chunk_size) < 128 or int(sample_clock_chunk_size) > 50_000:
        raise ValidationError("sample_clock_chunk_size must be between 128 and 50000")

    relation_by_asset: dict[str, str] = {}
    if wearer_body_asset_id:
        relation_by_asset[wearer_body_asset_id] = "self"
    if partner_body_asset_id:
        relation_by_asset[partner_body_asset_id] = "partner"
    for asset_id in body_asset_ids:
        relation_by_asset.setdefault(asset_id, "other")

    gaze_alignment_mode = str((gaze_artifact.parameters or {}).get("alignment_mode") or "timegraph")
    gaze_native_trim_id = (gaze_artifact.parameters or {}).get("native_trim_artifact_id")
    if native_trim_artifact_id and gaze_native_trim_id and native_trim_artifact_id != gaze_native_trim_id:
        raise ValidationError("Selected Native-reference trim differs from the trim used to generate these gaze rays")
    if not native_trim_artifact_id and gaze_native_trim_id:
        native_trim_artifact_id = str(gaze_native_trim_id)
    # ``gaze_alignment_mode`` describes how Gaze 3D originally obtained the ETM
    # pose.  It must not decide the downstream Body AOI clock.  TimeGraph-made
    # rays still carry the exact C3D query time and can be intersected with the
    # role-resolved body C3Ds without a separate Pupil→global path.
    cloud_asset = _resolve_cloud_asset(session, gaze_artifact, timeseries_asset_id)

    emit(progress, 0.03, "Loading calibrated gaze rays and scene timestamps")
    rays = _read_table(_artifact_table_path(gaze_artifact))
    original_ray_count = len(rays)
    trim_artifact = None
    if native_trim_artifact_id:
        trim_artifact = session.artifacts.get(native_trim_artifact_id)
        if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
            raise ValidationError("native_trim_artifact_id must reference a Native-reference trim artifact")
        trim_gaze_path = find_artifact_file(trim_artifact, names=["gaze.csv"])
        trim_gaze = pd.read_csv(trim_gaze_path, low_memory=False)
        trim_source_col = _find_column(trim_gaze.columns, ["oculo_source_row_index", "source_gaze_row_index", "source_row_index"])
        ray_source_col = _find_column(rays.columns, ["gaze3d_input_row_index", "oculo_source_row_index", "source_row_index"])
        if trim_gaze[trim_source_col].duplicated().any() or rays[ray_source_col].duplicated().any():
            raise ValidationError("Native trim and gaze-ray source-row indices must be unique")
        trim_order = pd.to_numeric(trim_gaze[trim_source_col], errors="raise").astype("int64")
        ray_index = pd.to_numeric(rays[ray_source_col], errors="raise").astype("int64")
        rays = rays.copy()
        rays[ray_source_col] = ray_index
        rays = rays.set_index(ray_source_col, drop=False)
        missing = trim_order.loc[~trim_order.isin(rays.index)]
        if len(missing):
            raise ValidationError(
                f"Gaze-ray artifact is missing {len(missing)} samples required by the selected Native-reference trim. "
                "Generate rays from a scope that covers this trim."
            )
        rays = rays.loc[trim_order.to_numpy()].reset_index(drop=True)
        # Attach the exact Native/clip frame universe and 2D gaze fields used by Face AOI.
        attach_aliases = {
            "native_frame_index": ["oculo_nearest_native_frame_index", "native_frame_index"],
            "clip_time_ns": ["oculo_clip_video_ns", "clip_time_ns"],
            "gaze_x_px": ["gaze x [px]", "gaze_x_px", "gaze x", "x"],
            "gaze_y_px": ["gaze y [px]", "gaze_y_px", "gaze y", "y"],
            "worn": ["worn"],
            "blink_id": ["blink id", "blink_id"],
            "fixation_id": ["fixation id", "fixation_id"],
        }
        for canonical, aliases in attach_aliases.items():
            source = _find_column(trim_gaze.columns, aliases, required=False)
            if source:
                rays[canonical] = trim_gaze[source].to_numpy()
        trim_timestamp_col = _find_column(trim_gaze.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"])
        ray_timestamp_col = _find_column(rays.columns, ["timestamp [ns]", "timestamp_ns", "gaze timestamp [ns]"])
        trim_ts = pd.to_numeric(trim_gaze[trim_timestamp_col], errors="coerce").astype("Int64")
        ray_ts = pd.to_numeric(rays[ray_timestamp_col], errors="coerce").astype("Int64")
        mismatch = (trim_ts.notna() & ray_ts.notna() & (trim_ts != ray_ts)).sum()
        if mismatch:
            raise ValidationError(f"Native trim and gaze-ray timestamps differ for {int(mismatch)} source rows")
    input_count = len(rays)
    timestamp_col = _find_column(rays.columns, ["timestamp [ns]", "timestamp_ns", "gaze timestamp [ns]"])
    ray_valid_col = _find_column(rays.columns, ["gaze3d_ray_valid", "ray_valid"])
    origin_cols = [
        _find_column(rays.columns, ["gaze3d_origin_x_mm", "origin_x_mm"]),
        _find_column(rays.columns, ["gaze3d_origin_y_mm", "origin_y_mm"]),
        _find_column(rays.columns, ["gaze3d_origin_z_mm", "origin_z_mm"]),
    ]
    direction_cols = [
        _find_column(rays.columns, ["gaze3d_direction_x", "direction_x"]),
        _find_column(rays.columns, ["gaze3d_direction_y", "direction_y"]),
        _find_column(rays.columns, ["gaze3d_direction_z", "direction_z"]),
    ]
    source_row_col = _find_column(rays.columns, ["oculo_source_row_index", "gaze3d_input_row_index", "source_row_index"], required=False)
    timestamp_numeric = pd.to_numeric(rays[timestamp_col], errors="coerce")
    timestamp_valid = timestamp_numeric.notna().to_numpy()
    timestamp_ns = np.zeros(input_count, dtype=np.int64)
    timestamp_ns[timestamp_valid] = timestamp_numeric.loc[timestamp_numeric.notna()].astype("int64").to_numpy()
    scene_timestamps = read_cloud_world_timestamps(cloud_asset.source_path)
    nearest_scene, frame_error = _nearest_indices(scene_timestamps, timestamp_ns)
    scene_frame_valid = timestamp_valid & (
        frame_error <= int(round(scene_frame_tolerance_ms * 1e6))
    )
    nearest = nearest_scene.copy()
    nearest[~scene_frame_valid] = -1
    selected_frames = np.unique(nearest[scene_frame_valid])
    selected_timestamps = scene_timestamps[selected_frames]

    cache = MaterializationCache(store.cache_dir())
    width, height, resolution_provenance = _body_scene_resolution(
        cache, cloud_asset, gaze_artifact, trim_artifact
    )
    if width is None or height is None:
        emit(
            progress,
            0.08,
            "Scene-video resolution unavailable; Body AOI will still use gaze-sample timestamps and C3D geometry",
        )

    sample_c3d_sec, sample_c3d_clock_column = _gaze_sample_c3d_times(rays)
    sample_local_sec, sample_local_clock_column = _gaze_sample_local_times(rays)

    trim_parameters = dict((trim_artifact.parameters or {}) if trim_artifact is not None else {})
    assignment_body_ids = {
        str(row.get("body_asset_id") or "")
        for row in assignment_rows.values()
        if str(row.get("body_asset_id") or "")
    }
    trim_body_ids = {
        str(row.get("asset_id") or "")
        for row in (trim_parameters.get("body_intervals") or {}).values()
        if isinstance(row, dict) and str(row.get("asset_id") or "")
    }
    if str(trim_parameters.get("body_asset_id") or ""):
        trim_body_ids.add(str(trim_parameters["body_asset_id"]))
    known_same_vicon_body_ids = assignment_body_ids | trim_body_ids
    selected_body_set = set(body_asset_ids)
    direct_c3d_clock_allowed = bool(
        native_trim_artifact_id
        and sample_c3d_sec is not None
        and known_same_vicon_body_ids
        and selected_body_set.issubset(known_same_vicon_body_ids)
    )
    local_shared_start_allowed = bool(
        native_trim_artifact_id
        and sample_local_sec is not None
        and gaze_alignment_mode == "pretrimmed_shared_start"
    )

    if direct_c3d_clock_allowed:
        body_time_alignment_mode = "gaze_ray_c3d_time"
        selected_c3d_sec = (
            _selected_frame_times(
                sample_c3d_sec,
                nearest,
                selected_frames,
                clock_label="gaze-ray C3D time",
            )
            if len(selected_frames)
            else np.asarray([], dtype=float)
        )
        selected_local_sec = np.asarray([], dtype=float)
    elif local_shared_start_allowed:
        body_time_alignment_mode = "pretrimmed_shared_start"
        selected_local_sec = (
            _selected_frame_times(
                sample_local_sec,
                nearest,
                selected_frames,
                clock_label="co-started local time",
            )
            if len(selected_frames)
            else np.asarray([], dtype=float)
        )
        selected_c3d_sec = np.asarray([], dtype=float)
    else:
        body_time_alignment_mode = "timegraph"
        selected_c3d_sec = np.asarray([], dtype=float)
        selected_local_sec = np.asarray([], dtype=float)

    graph = None
    gaze_global_ns = np.full(input_count, -1, dtype=np.int64)
    scene_global_ns = np.asarray([], dtype=np.int64)
    if body_time_alignment_mode == "timegraph":
        if not cloud_asset.primary_time_domain_id:
            raise ValidationError(
                "Timeseries asset has no primary time domain and the selected gaze rays do not expose a reusable C3D clock"
            )
        graph = TimeGraph.from_session(session)
        valid_timestamp_indices = np.flatnonzero(timestamp_valid)
        try:
            if len(valid_timestamp_indices):
                gaze_global_ns[valid_timestamp_indices] = graph.convert(
                    cloud_asset.primary_time_domain_id,
                    session.global_time_domain_id,
                    timestamp_ns[valid_timestamp_indices],
                ).astype(np.int64)
            if len(selected_timestamps):
                scene_global_ns = graph.convert(
                    cloud_asset.primary_time_domain_id,
                    session.global_time_domain_id,
                    selected_timestamps,
                ).astype(np.int64)
        except Exception as exc:
            raise ValidationError(
                "Body AOI could not reuse an exact gaze-ray C3D clock and no TimeGraph path connects the selected Timeseries to Global Time. "
                "Regenerate Gaze 3D from the matching C3D-defined Native trim, or select the matching trim and Calibration handoff. "
                f"Original error: {exc}"
            ) from exc

    profile = load_body_profile(body_profile_path)
    diagnostics: list[dict[str, Any]] = []
    display_diagnostics: list[dict[str, Any]] = []
    asset_contexts: list[dict[str, Any]] = []
    emit(progress, 0.11, "Preflighting C3D geometry at every gaze-sample timestamp")
    for position, asset_id in enumerate(body_asset_ids):
        token.raise_if_cancelled()
        asset = session.assets.get(asset_id)
        if asset is None or asset.kind not in {AssetKind.C3D, AssetKind.MOCAP_NPZ}:
            raise ValidationError(f"Body asset {asset_id} is not C3D or canonical mocap NPZ")
        body_mocap = read_mocap(asset.source_path)
        if body_time_alignment_mode == "gaze_ray_c3d_time":
            assert sample_c3d_sec is not None
            sample_query_time_sec = sample_c3d_sec.copy()
            display_query_time_sec = selected_c3d_sec.copy()
        elif body_time_alignment_mode == "pretrimmed_shared_start":
            assert sample_local_sec is not None
            sample_query_time_sec = float(body_mocap.time_sec[0]) + sample_local_sec
            display_query_time_sec = (
                float(body_mocap.time_sec[0]) + selected_local_sec
                if len(selected_frames) else np.asarray([], dtype=float)
            )
        else:
            if not asset.primary_time_domain_id:
                raise ValidationError(f"Body asset {asset.label} has no time domain")
            assert graph is not None
            sample_query_time_sec = np.full(input_count, np.nan, dtype=float)
            valid_timestamp_indices = np.flatnonzero(timestamp_valid)
            if len(valid_timestamp_indices):
                query_ns = graph.convert(
                    session.global_time_domain_id,
                    asset.primary_time_domain_id,
                    gaze_global_ns[valid_timestamp_indices],
                ).astype(np.int64)
                sample_query_time_sec[valid_timestamp_indices] = query_ns.astype(np.float64) / 1e9
            if len(selected_frames):
                display_query_ns = graph.convert(
                    session.global_time_domain_id,
                    asset.primary_time_domain_id,
                    scene_global_ns,
                ).astype(np.int64)
                display_query_time_sec = display_query_ns.astype(np.float64) / 1e9
            else:
                display_query_time_sec = np.asarray([], dtype=float)

        finite_query = np.isfinite(sample_query_time_sec)
        if finite_query.any():
            lower = float(body_mocap.time_sec[0]) - float(max_marker_gap_sec)
            upper = float(body_mocap.time_sec[-1]) + float(max_marker_gap_sec)
            query_time_range_coverage = float(
                ((sample_query_time_sec[finite_query] >= lower) & (sample_query_time_sec[finite_query] <= upper)).mean()
            )
        else:
            query_time_range_coverage = 0.0
        if body_time_alignment_mode in {"gaze_ray_c3d_time", "pretrimmed_shared_start"} and query_time_range_coverage < 0.90:
            raise ValidationError(
                f"Body asset {asset.label!r} overlaps only {100.0*query_time_range_coverage:.1f}% of the gaze-ray C3D clock. "
                "Select the role-resolved body C3D from the same interaction/Calibration handoff, or provide an explicit TimeGraph."
            )

        entity_id = (
            assignment_entity_ids.get(asset_id)
            or asset.participant_id
            or safe_name(asset.label, fallback=f"person_{position+1}")
        )
        relation = relation_by_asset.get(asset_id, "other")
        preflight_indices = np.flatnonzero(
            timestamp_valid & np.isfinite(sample_query_time_sec)
        )
        preflight = build_body_primitives(
            body_asset_id=asset_id,
            body_path=asset.source_path,
            entity_id=entity_id,
            scene_frame_indices=preflight_indices,
            scene_timestamps_ns=timestamp_ns[preflight_indices],
            query_time_sec=sample_query_time_sec[preflight_indices],
            profile=profile,
            marker_prefix=entity_marker_prefixes.get(asset_id),
            max_marker_gap_sec=max_marker_gap_sec,
            mocap_data=body_mocap,
            diagnostics_only=True,
            include_frame_regions=False,
        )
        diagnostic = dict(preflight.diagnostics)
        diagnostic["entity_relation"] = relation
        diagnostic["source_asset_label"] = asset.label
        diagnostic["source_file_name"] = Path(asset.source_path).name
        diagnostic["self_head_neck_excluded"] = bool(
            relation == "self" and exclude_self_head_neck
        )
        diagnostic["geometry_time_mode"] = "gaze_sample_timestamp"
        diagnostic["body_time_alignment_mode"] = body_time_alignment_mode
        diagnostic["gaze_ray_c3d_clock_column"] = sample_c3d_clock_column
        diagnostic["gaze_ray_local_clock_column"] = sample_local_clock_column
        diagnostic["query_time_range_coverage"] = query_time_range_coverage
        diagnostic["gaze_sample_count_total"] = int(input_count)
        diagnostic["gaze_timestamp_valid_count"] = int(timestamp_valid.sum())
        diagnostics.append(diagnostic)
        asset_contexts.append({
            "asset_id": asset_id,
            "asset": asset,
            "entity_id": entity_id,
            "relation": relation,
            "mocap": body_mocap,
            "sample_query_time_sec": sample_query_time_sec,
            "display_query_time_sec": display_query_time_sec,
            "diagnostic": diagnostic,
        })
        emit(
            progress,
            0.11 + 0.19 * (position + 1) / len(body_asset_ids),
            f"Preflighted gaze-sample geometry for {entity_id}",
        )

    expected_geometry_relations = ["partner", *(["self"] if include_self_body else [])]
    geometry_integrity = assess_body_geometry_integrity(
        diagnostics,
        policy=geometry_acceptance_policy,
        minimum_torso_coverage=float(minimum_torso_coverage),
        minimum_body_frame_coverage=float(minimum_body_frame_coverage),
        expected_relations=expected_geometry_relations,
    )
    geometry_integrity["coverage_clock"] = "gaze_sample_timestamp"
    geometry_preflight_path: Path | None = None
    if not geometry_integrity.get("gate_passed"):
        geometry_preflight_path = _write_body_geometry_preflight(
            store,
            session_id=session_id,
            gaze_rays_artifact_id=gaze_rays_artifact_id,
            report=geometry_integrity,
            parameters={
                "body_asset_ids": body_asset_ids,
                "wearer_body_asset_id": wearer_body_asset_id,
                "partner_body_asset_id": partner_body_asset_id,
                "wearer_role_id": wearer_role_id,
                "partner_role_id": partner_role_id,
                "entity_marker_prefixes": entity_marker_prefixes,
                "geometry_acceptance_policy": geometry_acceptance_policy,
                "minimum_torso_coverage": float(minimum_torso_coverage),
                "minimum_body_frame_coverage": float(minimum_body_frame_coverage),
                "max_marker_gap_sec": float(max_marker_gap_sec),
                "coverage_clock": "gaze_sample_timestamp",
            },
        )
        geometry_integrity["diagnostic_path"] = str(geometry_preflight_path)
        if geometry_acceptance_policy == "required":
            emit(progress, 0.31, "Body geometry gate blocked at the gaze-sample clock")
            raise ValidationError(
                _body_geometry_gate_error(
                    geometry_integrity, geometry_preflight_path, store.root
                )
            )
        emit(progress, 0.31, "Body geometry gate warning: continuing for diagnosis only")
    else:
        emit(progress, 0.31, "Body geometry gate passed at every gaze-sample timestamp")

    # Build a separate, lower-rate geometry table only for video overlay and
    # human-review navigation.  It never drives the 3D Body AOI intersection.
    primitives_by_frame: dict[int, list] = defaultdict(list)
    region_frames: list[pd.DataFrame] = []
    excluded_self_primitive_count = 0
    if len(selected_frames):
        emit(progress, 0.33, "Building video-frame body geometry for review display")
        for position, context in enumerate(asset_contexts):
            display_build = build_body_primitives(
                body_asset_id=context["asset_id"],
                body_path=context["asset"].source_path,
                entity_id=context["entity_id"],
                scene_frame_indices=selected_frames,
                scene_timestamps_ns=selected_timestamps,
                query_time_sec=context["display_query_time_sec"],
                profile=profile,
                marker_prefix=entity_marker_prefixes.get(context["asset_id"]),
                max_marker_gap_sec=max_marker_gap_sec,
                mocap_data=context["mocap"],
            )
            relation = context["relation"]
            for frame_index, primitives in display_build.primitives_by_frame.items():
                selected_primitives = []
                for primitive in primitives:
                    if (
                        relation == "self"
                        and exclude_self_head_neck
                        and str(primitive.region).lower() in {"head", "neck"}
                    ):
                        excluded_self_primitive_count += 1
                        continue
                    selected_primitives.append(primitive)
                primitives_by_frame[frame_index].extend(selected_primitives)
            if not display_build.frame_regions.empty:
                frame_regions = display_build.frame_regions.copy()
                frame_regions["entity_relation"] = relation
                frame_regions["coarse_region"] = frame_regions["region"].map(
                    coarse_body_region
                )
                frame_regions["geometry_time_mode"] = "video_frame_display_only"
                if relation == "self" and exclude_self_head_neck:
                    frame_regions = frame_regions.loc[
                        ~frame_regions["region"]
                        .astype(str)
                        .str.lower()
                        .isin({"head", "neck"})
                    ].copy()
                region_frames.append(frame_regions)
            display_diagnostic = dict(display_build.diagnostics)
            display_diagnostic["entity_relation"] = relation
            display_diagnostic["geometry_time_mode"] = "video_frame_display_only"
            display_diagnostics.append(display_diagnostic)
            context["diagnostic"]["video_frame_geometry"] = display_diagnostic
            emit(
                progress,
                0.33 + 0.07 * (position + 1) / len(asset_contexts),
                f"Built display geometry for {context['entity_id']}",
            )

    origins = rays[origin_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    directions = rays[direction_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ray_valid = rays[ray_valid_col].fillna(False).astype(bool).to_numpy()
    ranked_hits: list[list] = [[] for _ in range(input_count)]
    intersection_indices = np.flatnonzero(timestamp_valid & ray_valid)
    chunk_size = int(sample_clock_chunk_size)
    emit(progress, 0.41, "Intersecting each gaze ray with body geometry at its exact timestamp")
    for chunk_start in range(0, len(intersection_indices), chunk_size):
        token.raise_if_cancelled()
        indices = intersection_indices[chunk_start:chunk_start + chunk_size]
        primitives_for_samples: list[list] = [[] for _ in range(len(indices))]
        local_position = {int(global_index): position for position, global_index in enumerate(indices)}
        for context in asset_contexts:
            sample_build = build_body_primitives(
                body_asset_id=context["asset_id"],
                body_path=context["asset"].source_path,
                entity_id=context["entity_id"],
                scene_frame_indices=indices,
                scene_timestamps_ns=timestamp_ns[indices],
                query_time_sec=context["sample_query_time_sec"][indices],
                profile=profile,
                marker_prefix=entity_marker_prefixes.get(context["asset_id"]),
                max_marker_gap_sec=max_marker_gap_sec,
                mocap_data=context["mocap"],
                include_frame_regions=False,
            )
            relation = context["relation"]
            for global_index, primitives in sample_build.primitives_by_frame.items():
                selected_primitives = []
                for primitive in primitives:
                    if (
                        relation == "self"
                        and exclude_self_head_neck
                        and str(primitive.region).lower() in {"head", "neck"}
                    ):
                        continue
                    selected_primitives.append(primitive)
                primitives_for_samples[local_position[int(global_index)]].extend(
                    selected_primitives
                )
        chunk_hits = intersect_sample_rays_ranked(
            origins[indices],
            directions[indices],
            primitives_for_samples,
            uncertainty_cone_deg=uncertainty_cone_deg,
            maximum_distance_mm=maximum_target_distance_m * 1000.0,
        )
        for global_index, hit_rows in zip(indices, chunk_hits):
            ranked_hits[int(global_index)] = hit_rows
        completed = chunk_start + len(indices)
        emit(
            progress,
            0.41 + 0.36 * completed / max(1, len(intersection_indices)),
            f"Gaze-sample body intersections ({completed}/{len(intersection_indices)})",
        )

    valid_scene_errors_ms = frame_error[scene_frame_valid].astype(np.float64) / 1e6
    timebase_summary = {
        "schema_version": "oculostudio-body-aoi-timebase-1.0",
        "body_geometry_clock": "gaze_sample_timestamp",
        "body_time_alignment_mode": body_time_alignment_mode,
        "ray_generation_alignment_mode": gaze_alignment_mode,
        "gaze_ray_c3d_clock_column": sample_c3d_clock_column,
        "gaze_ray_local_clock_column": sample_local_clock_column,
        "global_timegraph_required": bool(body_time_alignment_mode == "timegraph"),
        "video_clock_usage": "display_and_human_review_only",
        "gaze_sample_count": int(input_count),
        "valid_gaze_timestamp_count": int(timestamp_valid.sum()),
        "valid_gaze_ray_count": int(ray_valid.sum()),
        "estimated_gaze_sample_rate_hz": _estimate_frequency_hz(
            timestamp_ns[timestamp_valid]
        ),
        "scene_video_frame_count": int(len(scene_timestamps)),
        "estimated_scene_video_rate_hz": _estimate_frequency_hz(scene_timestamps),
        "scene_frame_available_sample_count": int(scene_frame_valid.sum()),
        "scene_frame_available_fraction": float(scene_frame_valid.mean()) if input_count else 0.0,
        "median_gaze_to_video_frame_error_ms": (
            float(np.median(valid_scene_errors_ms)) if len(valid_scene_errors_ms) else None
        ),
        "p95_gaze_to_video_frame_error_ms": (
            float(np.quantile(valid_scene_errors_ms, 0.95)) if len(valid_scene_errors_ms) else None
        ),
        "body_sources": [
            {
                "asset_id": context["asset_id"],
                "label": context["asset"].label,
                "entity_id": context["entity_id"],
                "relation": context["relation"],
                "source_frame_rate_hz": float(context["mocap"].frame_rate),
                "source_frame_count": int(len(context["mocap"].time_sec)),
                "median_query_to_source_sample_error_ms": context["diagnostic"].get(
                    "median_query_to_source_sample_error_ms"
                ),
                "p95_query_to_source_sample_error_ms": context["diagnostic"].get(
                    "p95_query_to_source_sample_error_ms"
                ),
            }
            for context in asset_contexts
        ],
    }

    recording_id = cloud_asset.recording_id or "unknown_recording"
    native_frame_col = _find_column(rays.columns, ["native_frame_index", "oculo_nearest_native_frame_index"], required=False)
    clip_time_col = _find_column(rays.columns, ["clip_time_ns", "oculo_clip_video_ns"], required=False)
    gaze_x_col = _find_column(rays.columns, ["gaze_x_px", "gaze x [px]", "gaze x", "x"], required=False)
    gaze_y_col = _find_column(rays.columns, ["gaze_y_px", "gaze y [px]", "gaze y", "y"], required=False)
    worn_col = _find_column(rays.columns, ["worn"], required=False)
    blink_col = _find_column(rays.columns, ["blink_id", "blink id"], required=False)
    fixation_col = _find_column(rays.columns, ["fixation_id", "fixation id"], required=False)
    rows = []
    near_origin_mm = float(self_near_origin_exclusion_m) * 1000.0
    for ordinal in range(input_count):
        source_row = int(rays[source_row_col].iloc[ordinal]) if source_row_col and pd.notna(rays[source_row_col].iloc[ordinal]) else ordinal
        stamp = int(timestamp_ns[ordinal]) if timestamp_valid[ordinal] else -1
        social_hits = _normalise_social_hits(
            ranked_hits[ordinal],
            relation_by_asset=relation_by_asset,
            wearer_role_id=wearer_role_id,
            partner_role_id=partner_role_id,
            self_near_origin_exclusion_mm=near_origin_mm,
        )
        first_item = social_hits[0] if social_hits else None
        second_item = social_hits[1] if first_hit_occlusion and len(social_hits) > 1 else None
        partner_item = next((item for item in social_hits if item[1] == "partner"), None)
        first_cols = _hit_columns("first_hit", first_item, output_granularity=output_granularity)
        second_cols = _hit_columns("second_hit", second_item, output_granularity=output_granularity)
        partner_cols = _hit_columns("nearest_partner_hit", partner_item, output_granularity=output_granularity)
        first_hit = first_item[0] if first_item else None
        first_relation = first_item[1] if first_item else None
        self_occlusion = bool(
            first_hit_occlusion
            and first_item is not None
            and first_relation == "self"
            and partner_item is not None
            and float(partner_item[0].distance_mm) > float(first_hit.distance_mm)
        )
        if not timestamp_valid[ordinal]:
            status = "invalid_timestamp"
            region = SemanticRegion.INVALID_GAZE.value
            review = False
            occlusion_status = "invalid"
        elif not ray_valid[ordinal]:
            status = "invalid_ray"
            region = SemanticRegion.INVALID_GAZE.value
            review = False
            occlusion_status = "invalid"
        elif first_hit is None:
            status = "valid_ray_no_body_hit"
            region = "no_body"
            review = False
            occlusion_status = "no_body_hit"
        else:
            status = "body_3d_exact" if first_hit.exact_hit else "body_3d_uncertainty_cone"
            status += f"_{first_relation or 'other'}"
            region = first_cols["first_hit_region"]
            review = not first_hit.exact_hit
            if self_occlusion:
                occlusion_status = "self_occludes_partner"
            elif first_relation == "self":
                occlusion_status = "self_visible_first"
            elif first_relation == "partner":
                occlusion_status = "partner_visible_first"
            else:
                occlusion_status = "other_visible_first"
        row = {
            "recording_id": recording_id,
            "gaze_sample_uid": _gaze_uid(recording_id, source_row, stamp),
            "source_gaze_row_index": source_row,
            "timestamp_ns": stamp if stamp >= 0 else pd.NA,
            "clip_time_ns": (
                int(pd.to_numeric(pd.Series([rays[clip_time_col].iloc[ordinal]]), errors="coerce").iloc[0])
                if clip_time_col and pd.notna(rays[clip_time_col].iloc[ordinal]) else pd.NA
            ),
            "native_frame_index": (
                int(pd.to_numeric(pd.Series([rays[native_frame_col].iloc[ordinal]]), errors="coerce").iloc[0])
                if native_frame_col and pd.notna(rays[native_frame_col].iloc[ordinal]) else pd.NA
            ),
            "gaze_x_px": float(rays[gaze_x_col].iloc[ordinal]) if gaze_x_col and pd.notna(rays[gaze_x_col].iloc[ordinal]) else np.nan,
            "gaze_y_px": float(rays[gaze_y_col].iloc[ordinal]) if gaze_y_col and pd.notna(rays[gaze_y_col].iloc[ordinal]) else np.nan,
            "worn": rays[worn_col].iloc[ordinal] if worn_col else None,
            "blink_id": rays[blink_col].iloc[ordinal] if blink_col else None,
            "fixation_id": rays[fixation_col].iloc[ordinal] if fixation_col else None,
            "scene_frame_index": int(nearest[ordinal]) if nearest[ordinal] >= 0 else pd.NA,
            "scene_timestamp_ns": int(scene_timestamps[nearest[ordinal]]) if nearest[ordinal] >= 0 else pd.NA,
            "scene_frame_match_error_ms": float(frame_error[ordinal])/1e6 if scene_frame_valid[ordinal] else np.nan,
            "scene_frame_available": bool(scene_frame_valid[ordinal]),
            "review_preview_available": bool(scene_frame_valid[ordinal]),
            "body_geometry_time_mode": "gaze_sample_timestamp",
            "body_time_alignment_mode": body_time_alignment_mode,
            "body_geometry_query_timestamp_ns": stamp if stamp >= 0 else pd.NA,
            "body_geometry_query_time_sec": (
                float(sample_c3d_sec[ordinal])
                if body_time_alignment_mode == "gaze_ray_c3d_time" and sample_c3d_sec is not None and np.isfinite(sample_c3d_sec[ordinal])
                else (
                    float(sample_local_sec[ordinal])
                    if body_time_alignment_mode == "pretrimmed_shared_start" and sample_local_sec is not None and np.isfinite(sample_local_sec[ordinal])
                    else np.nan
                )
            ),
            "ray_valid": bool(ray_valid[ordinal]),
            "ray_origin_x_mm": origins[ordinal,0],
            "ray_origin_y_mm": origins[ordinal,1],
            "ray_origin_z_mm": origins[ordinal,2],
            "ray_direction_x": directions[ordinal,0],
            "ray_direction_y": directions[ordinal,1],
            "ray_direction_z": directions[ordinal,2],
            "wearer_role_id": wearer_role_id,
            "partner_role_id": partner_role_id,
            "body_entity_id": first_cols["first_hit_entity_id"],
            "body_relation": first_cols["first_hit_relation"],
            "body_raw_region": first_cols["first_hit_raw_region"],
            "body_coarse_region": first_cols["first_hit_coarse_region"],
            "body_region": region,
            "body_social_region": (
                first_cols["first_hit_social_region"]
                if first_cols["first_hit_social_region"] is not None else region
            ),
            "body_distance_mm": first_cols["first_hit_distance_mm"],
            "body_hit_x_mm": first_cols["first_hit_x_mm"],
            "body_hit_y_mm": first_cols["first_hit_y_mm"],
            "body_hit_z_mm": first_cols["first_hit_z_mm"],
            "body_confidence": first_cols["first_hit_confidence"],
            "body_exact_hit": first_cols["first_hit_exact"],
            "body_angular_miss_deg": first_cols["first_hit_angular_miss_deg"],
            "body_geometry_backend": "c3d_articulated_primitives_v6_gaze_sample_clock",
            "candidate_hit_count": int(len(social_hits)),
            "self_occlusion": self_occlusion,
            "occlusion_status": occlusion_status,
            "assignment_status": status,
            "review_required": bool(review),
        }
        row.update(first_cols)
        row.update(second_cols)
        row.update(partner_cols)
        rows.append(row)
    samples = pd.DataFrame(rows)
    samples, auto_review_report, auto_review_table = _apply_body_auto_review_reduction(
        samples, auto_review_policy
    )
    summary = _frame_summary(samples)
    regions = pd.concat(region_frames, ignore_index=True, sort=False) if region_frames else pd.DataFrame()

    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "body_aoi", gaze_artifact.clip_id or "full") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    samples_csv = output_root / "body_aoi_samples.csv"
    frames_csv = output_root / "body_aoi_frames.csv"
    regions_csv = output_root / "body_regions.csv"
    auto_review_csv = output_root / "body_aoi_auto_review_summary.csv"
    auto_review_json = output_root / "body_aoi_auto_review_summary.json"
    timebase_json = output_root / "body_aoi_timebase.json"
    samples.to_csv(samples_csv, index=False)
    summary.to_csv(frames_csv, index=False)
    regions.to_csv(regions_csv, index=False)
    auto_review_table.to_csv(auto_review_csv, index=False)
    atomic_write_json(auto_review_json, auto_review_report)
    atomic_write_json(timebase_json, timebase_summary)
    parquet_files = []
    if output_parquet:
        try:
            for frame, name in [(samples,"body_aoi_samples.parquet"),(summary,"body_aoi_frames.parquet"),(regions,"body_regions.parquet")]:
                path = output_root/name
                frame.to_parquet(path,index=False)
                parquet_files.append(str(path))
        except Exception:
            parquet_files = []
    counts = Counter(samples["body_region"].astype(str))
    relation_counts = Counter(samples["body_relation"].fillna("none").astype(str))
    raw_region_counts = Counter(samples["body_raw_region"].fillna("none").astype(str))
    qc = {
        "source_gaze_ray_rows": int(original_ray_count),
        "input_gaze_rows": input_count,
        "output_rows": len(samples),
        "native_trim_applied": bool(native_trim_artifact_id),
        "alignment_mode": gaze_alignment_mode,
        "body_time_alignment_mode": body_time_alignment_mode,
        "alignment_basis": (
            "Exact C3D query time already stored in each calibrated gaze ray; no project Global Time path required"
            if body_time_alignment_mode == "gaze_ray_c3d_time"
            else (
                "Each gaze ray's tracker-local timestamp + first frame of each co-started body C3D"
                if body_time_alignment_mode == "pretrimmed_shared_start"
                else "Verified TimeGraph mapping through project Global Time"
            )
        ),
        "global_timegraph_required": bool(body_time_alignment_mode == "timegraph"),
        "body_geometry_clock": "gaze_sample_timestamp",
        "video_clock_usage": "display_and_human_review_only",
        "estimated_gaze_sample_rate_hz": timebase_summary.get("estimated_gaze_sample_rate_hz"),
        "estimated_scene_video_rate_hz": timebase_summary.get("estimated_scene_video_rate_hz"),
        "scene_frame_available_fraction": timebase_summary.get("scene_frame_available_fraction"),
        "median_gaze_to_video_frame_error_ms": timebase_summary.get("median_gaze_to_video_frame_error_ms"),
        "p95_gaze_to_video_frame_error_ms": timebase_summary.get("p95_gaze_to_video_frame_error_ms"),
        "timebase_summary": timebase_summary,
        "row_count_preserved": bool(input_count == len(samples)),
        "valid_ray_fraction": float(samples["ray_valid"].mean()) if len(samples) else 0.0,
        "body_hit_fraction": float((~samples["body_region"].isin([
            "no_body", SemanticRegion.INVALID_GAZE.value, SemanticRegion.UNKNOWN.value,
        ])).mean()) if len(samples) else 0.0,
        "review_required_fraction": float(samples["review_required"].mean()) if len(samples) else 0.0,
        "review_required_fraction_before_auto_resolution": float(samples["review_required_before_auto_resolution"].mean()) if len(samples) else 0.0,
        "auto_review_policy": auto_review_policy,
        "review_samples_before_auto_resolution": int(auto_review_report["review_samples_before"]),
        "auto_resolved_sample_count": int(auto_review_report["auto_resolved_samples"]),
        "review_samples_remaining": int(auto_review_report["review_samples_remaining"]),
        "auto_review_sample_reduction_fraction": float(auto_review_report["sample_review_reduction_fraction"]),
        "review_frames_before_auto_resolution": int(auto_review_report["review_frames_before"]),
        "review_frames_remaining": int(auto_review_report["review_frames_remaining"]),
        "review_fixations_before_auto_resolution": int(auto_review_report["review_fixations_before"]),
        "review_fixations_remaining": int(auto_review_report["review_fixations_remaining"]),
        "review_units_before_auto_resolution": int(auto_review_report["review_units_before"]),
        "review_units_remaining": int(auto_review_report["review_units_remaining"]),
        "auto_review_unit_reduction_fraction": float(auto_review_report["review_unit_reduction_fraction"]),
        "auto_review_resolution_source_counts": dict(auto_review_report["resolution_source_counts"]),
        "auto_review_conflicted_fixation_count": int(auto_review_report.get("conflicted_fixation_count", 0)),
        "auto_review_policy_thresholds": dict(auto_review_report.get("policy_thresholds") or {}),
        "remaining_review_reason_counts": dict(auto_review_report["remaining_review_reason_counts"]),
        "body_region_counts": dict(counts),
        "body_raw_region_counts": dict(raw_region_counts),
        "body_relation_counts": dict(relation_counts),
        "self_hit_fraction": float((samples["body_relation"] == "self").mean()) if len(samples) else 0.0,
        "partner_hit_fraction": float((samples["body_relation"] == "partner").mean()) if len(samples) else 0.0,
        "self_occlusion_fraction": float(samples["self_occlusion"].fillna(False).astype(bool).mean()) if len(samples) else 0.0,
        "second_hit_available_fraction": float(samples["second_hit_entity_id"].notna().mean()) if len(samples) else 0.0,
        "wearer_role_id": wearer_role_id,
        "partner_role_id": partner_role_id,
        "wearer_body_asset_id": wearer_body_asset_id,
        "partner_body_asset_id": partner_body_asset_id,
        "include_self_body": bool(include_self_body),
        "exclude_self_head_neck": bool(exclude_self_head_neck),
        "self_near_origin_exclusion_m": float(self_near_origin_exclusion_m),
        "first_hit_occlusion": bool(first_hit_occlusion),
        "output_granularity": output_granularity,
        "excluded_self_head_neck_primitive_count": int(excluded_self_primitive_count),
        "body_diagnostics": diagnostics,
        "body_display_diagnostics": display_diagnostics,
        "body_geometry_integrity": geometry_integrity,
        "body_geometry_gate_passed": bool(geometry_integrity.get("gate_passed")),
        "body_geometry_review_ready": bool(geometry_integrity.get("review_ready")),
        "body_geometry_fusion_ready": bool(geometry_integrity.get("fusion_ready")),
        "body_geometry_preflight_diagnostic_path": str(geometry_preflight_path) if geometry_preflight_path else None,
        "scene_resolution_available": bool(width is not None and height is not None),
        "scene_resolution_required_for_body_intersection": False,
        "scene_frame_required_for_body_intersection": False,
        "scene_resolution_source": resolution_provenance.get("source"),
    }
    manifest = {
        "schema_version": "oculostudio-body-aoi-2.3",
        "artifact_id": artifact_id,
        "gaze_rays_artifact_id": gaze_rays_artifact_id,
        "recording_id": recording_id,
        "parameters": {
            "body_profile_path": body_profile_path,
            "entity_marker_prefixes": entity_marker_prefixes,
            "etm_assignment_artifact_id": etm_assignment_artifact_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "alignment_mode": gaze_alignment_mode,
            "body_time_alignment_mode": body_time_alignment_mode,
            "wearer_role_id": wearer_role_id,
            "partner_role_id": partner_role_id,
            "wearer_body_asset_id": wearer_body_asset_id,
            "partner_body_asset_id": partner_body_asset_id,
            "include_self_body": bool(include_self_body),
            "exclude_self_head_neck": bool(exclude_self_head_neck),
            "self_near_origin_exclusion_m": float(self_near_origin_exclusion_m),
            "first_hit_occlusion": bool(first_hit_occlusion),
            "output_granularity": output_granularity,
            "auto_review_policy": auto_review_policy,
            "geometry_acceptance_policy": geometry_acceptance_policy,
            "minimum_torso_coverage": float(minimum_torso_coverage),
            "minimum_body_frame_coverage": float(minimum_body_frame_coverage),
            "max_marker_gap_sec": max_marker_gap_sec,
            "uncertainty_cone_deg": uncertainty_cone_deg,
            "maximum_target_distance_m": maximum_target_distance_m,
            "scene_frame_tolerance_ms": scene_frame_tolerance_ms,
            "sample_clock_chunk_size": int(sample_clock_chunk_size),
            "body_geometry_clock": "gaze_sample_timestamp",
        },
        "scene_resolution": {"width": width, "height": height, **resolution_provenance},
        "outputs": {
            "samples_csv": str(samples_csv),
            "frames_csv": str(frames_csv),
            "regions_csv": str(regions_csv),
            "auto_review_summary_csv": str(auto_review_csv),
            "auto_review_summary_json": str(auto_review_json),
            "timebase_json": str(timebase_json),
            "geometry_preflight_diagnostic_json": str(geometry_preflight_path) if geometry_preflight_path else None,
            "parquet": parquet_files,
        },
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root/"body_aoi_manifest.json", manifest)
    atomic_write_json(output_root/"qc_metrics.json", qc)
    source_assets = [cloud_asset.id, *body_asset_ids]
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.BODY_AOI,
        label=label,
        output_root=output_root,
        source_asset_ids=list(dict.fromkeys(source_assets)),
        parent_artifact_ids=[artifact_id for artifact_id in [gaze_rays_artifact_id, etm_assignment_artifact_id, native_trim_artifact_id] if artifact_id],
        clip_id=gaze_artifact.clip_id,
        parameters=manifest["parameters"],
        qc=qc,
        provenance={
            "gaze_rays_artifact_id": gaze_rays_artifact_id,
            "body_profile_path": body_profile_path,
            "native_trim_artifact_id": native_trim_artifact_id,
            "alignment_mode": gaze_alignment_mode,
            "geometry_source": "c3d_articulated_primitives_v6_gaze_sample_clock",
            "body_geometry_clock": "gaze_sample_timestamp",
            "video_clock_usage": "display_and_human_review_only",
            "wearer_role_id": wearer_role_id,
            "partner_role_id": partner_role_id,
            "visibility_rule": "strict_first_hit_with_self_occlusion",
            "body_auto_review_policy": auto_review_policy,
            "body_geometry_acceptance_policy": geometry_acceptance_policy,
            "body_geometry_gate_passed": bool(geometry_integrity.get("gate_passed")),
            "minimum_torso_coverage": float(minimum_torso_coverage),
            "minimum_body_frame_coverage": float(minimum_body_frame_coverage),
            "torso_geometry_fallback": "direct raw-Vicon waist/chest clusters, bilateral shoulders/hips, or observed body axis",
        },
        artifact_id=artifact_id,
    )
    emit(progress,1.0,"Body AOI annotation complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "samples_csv": str(samples_csv),
        "frames_csv": str(frames_csv),
        "regions_csv": str(regions_csv),
        "auto_review_summary_csv": str(auto_review_csv),
        "auto_review_summary_json": str(auto_review_json),
        "timebase_json": str(timebase_json),
        "manifest": str(output_root / "body_aoi_manifest.json"),
    }
