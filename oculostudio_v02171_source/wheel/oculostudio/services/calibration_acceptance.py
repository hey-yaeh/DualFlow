from __future__ import annotations

import csv
import html
import json
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.mocap import read_mocap
from oculostudio.adapters.pupil import materialized_cloud_root
from oculostudio.domain.geometry import PoseSeries, make_transform, normalize
from oculostudio.domain.models import ArtifactKind, TimeMap
from oculostudio.domain.timegraph import TimeGraph, invert_time_map
from oculostudio.errors import ProcessingError, TimeGraphError, ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.calibration import CameraIntrinsics, load_calibration_profile
from oculostudio.services.calibration_profiles import resolve_profile_tracker_points
from oculostudio.services.etm_role_assignment import resolve_assignment_for_role
from oculostudio.services.gaze3d import _find_column, _find_gaze_csv, unproject_gaze_pixels
from oculostudio.services.participant_association import create_participant_association, load_participant_association
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


AcceptanceStatus = Literal["accepted", "accepted_limited", "review", "rejected"]


def _profile_file(store: ProjectStore, session_id: str, artifact_id: str) -> Path:
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
        raise ValidationError("Select a transferred calibration-profile artifact")
    candidates = [Path(item.path) for item in artifact.files if Path(item.path).exists()]
    for suffix in (".oculocal", ".g3dcal", ".json"):
        matches = [path for path in candidates if path.suffix.lower() == suffix]
        if matches:
            matches.sort(key=lambda p: ("summary" in p.name.lower(), len(p.parts), str(p).lower()))
            return matches[0]
    raise ValidationError("Calibration profile artifact contains no readable profile")


def _time_path(session, source_domain_id: str, target_domain_id: str) -> list[TimeMap]:
    if source_domain_id == target_domain_id:
        return []
    adjacency: dict[str, list[tuple[TimeMap, TimeMap]]] = {key: [] for key in session.time_domains}
    for original in session.time_maps.values():
        adjacency.setdefault(original.source_domain_id, []).append((original, original))
        adjacency.setdefault(original.target_domain_id, []).append((invert_time_map(original), original))
    queue: deque[tuple[str, list[TimeMap], list[TimeMap]]] = deque([(source_domain_id, [], [])])
    visited = {source_domain_id}
    while queue:
        domain, traversed, originals = queue.popleft()
        for edge, original in adjacency.get(domain, []):
            if edge.target_domain_id in visited:
                continue
            next_traversed = [*traversed, edge]
            next_originals = [*originals, original]
            if edge.target_domain_id == target_domain_id:
                return next_originals
            visited.add(edge.target_domain_id)
            queue.append((edge.target_domain_id, next_traversed, next_originals))
    raise TimeGraphError(f"No time-map path connects '{source_domain_id}' to '{target_domain_id}'")


def _time_gate(session, source_domain_id: str, target_domain_id: str, *, maximum_rms_ms: float) -> dict[str, Any]:
    try:
        path = _time_path(session, source_domain_id, target_domain_id)
        resolved = TimeGraph.from_session(session).resolve(source_domain_id, target_domain_id)
    except TimeGraphError as exc:
        return {
            "status": "failed",
            "path_exists": False,
            "reason": str(exc),
            "methods": [],
            "anchor_count": 0,
            "rms_residual_ms": None,
        }
    provisional = [row for row in path if "provisional" in str(row.method).lower()]
    methods = [row.method for row in path]
    anchor_count = int(sum(max(0, row.quality.anchor_count) for row in path))
    rms_ms = float(resolved.quality.rms_residual_ns / 1e6)
    passed = not provisional and anchor_count > 0 and rms_ms <= maximum_rms_ms
    return {
        "status": "passed" if passed else "failed",
        "path_exists": True,
        "reason": None if passed else (
            "Time path contains provisional mapping" if provisional else
            "Time path has no synchronization anchors" if anchor_count <= 0 else
            f"Composed synchronization RMS {rms_ms:.3f} ms exceeds {maximum_rms_ms:.3f} ms"
        ),
        "methods": methods,
        "mapping_ids": [row.id for row in path],
        "anchor_count": anchor_count,
        "rms_residual_ms": rms_ms,
        "maximum_rms_ms": maximum_rms_ms,
        "contains_provisional": bool(provisional),
    }


def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _head_projection_gate(
    projections: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    minimum_projection_valid_fraction: float,
    minimum_partner_match_frame_fraction: float,
    maximum_median_normalized_center_error: float,
    minimum_median_projected_iou: float,
    maximum_systematic_offset_normalized: float,
    maximum_review_fraction: float,
) -> dict[str, Any]:
    if projections.empty:
        return {"status": "failed", "reason": "No projected participant heads were produced"}
    partner_projection = projections.loc[projections.get("relation_to_wearer", pd.Series(index=projections.index, dtype=str)).eq("partner")].copy()
    if partner_projection.empty:
        return {"status": "failed", "reason": "No partner head projection rows were produced"}
    projection_valid = partner_projection.get("projection_valid", pd.Series(False, index=partner_projection.index)).astype(bool)
    in_view = partner_projection.get("in_camera_view", pd.Series(False, index=partner_projection.index)).astype(bool)
    projection_valid_fraction = float((projection_valid & in_view).mean()) if len(partner_projection) else 0.0

    face_frames = set(pd.to_numeric(identities.get("native_frame_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    partner_rows = identities.loc[
        identities.get("target_relation", pd.Series(index=identities.index, dtype=str)).eq("partner")
        & identities.get("identity_status", pd.Series(index=identities.index, dtype=str)).eq("matched_participant")
    ].copy()
    projected_frames = set(
        pd.to_numeric(partner_projection.loc[projection_valid & in_view, "native_frame_index"], errors="coerce")
        .dropna().astype(int).tolist()
    )
    evaluable_frames = projected_frames & face_frames
    matched_frames = set(pd.to_numeric(partner_rows.get("native_frame_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    partner_match_frame_fraction = float(len(matched_frames & evaluable_frames) / len(evaluable_frames)) if evaluable_frames else 0.0

    norm_error = _safe_numeric(partner_rows.get("normalized_center_error", pd.Series(dtype=float)))
    norm_error = norm_error[np.isfinite(norm_error)]
    iou = _safe_numeric(partner_rows.get("projected_box_iou", pd.Series(dtype=float)))
    iou = iou[np.isfinite(iou)]
    confidence = _safe_numeric(partner_rows.get("identity_confidence", pd.Series(dtype=float)))
    confidence = confidence[np.isfinite(confidence)]
    review = identities.get("review_required", pd.Series(False, index=identities.index)).astype(bool)
    review_fraction = float(review.mean()) if len(review) else 1.0

    dx_norm: list[float] = []
    dy_norm: list[float] = []
    for _, row in partner_rows.iterrows():
        try:
            x1, x2 = float(row["face_box_x1"]), float(row["face_box_x2"])
            y1, y2 = float(row["face_box_y1"]), float(row["face_box_y2"])
            px, py = float(row["projected_head_x_px"]), float(row["projected_head_y_px"])
        except Exception:
            continue
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        dx_norm.append(((x1 + x2) * 0.5 - px) / width)
        dy_norm.append(((y1 + y2) * 0.5 - py) / height)
    systematic_x = float(np.median(dx_norm)) if dx_norm else None
    systematic_y = float(np.median(dy_norm)) if dy_norm else None
    systematic_mag = float(math.hypot(systematic_x, systematic_y)) if systematic_x is not None and systematic_y is not None else None

    median_error = float(np.median(norm_error)) if len(norm_error) else None
    p95_error = float(np.quantile(norm_error, 0.95)) if len(norm_error) else None
    median_iou = float(np.median(iou)) if len(iou) else None
    median_confidence = float(np.median(confidence)) if len(confidence) else None
    checks = {
        "projection_coverage": projection_valid_fraction >= minimum_projection_valid_fraction,
        "partner_match_coverage": partner_match_frame_fraction >= minimum_partner_match_frame_fraction,
        "median_center_error": median_error is not None and median_error <= maximum_median_normalized_center_error,
        "median_iou": median_iou is not None and median_iou >= minimum_median_projected_iou,
        "systematic_offset": systematic_mag is not None and systematic_mag <= maximum_systematic_offset_normalized,
        "review_fraction": review_fraction <= maximum_review_fraction,
    }
    passed = all(checks.values())
    borderline_count = sum(bool(value) for value in checks.values())
    status = "passed" if passed else ("review" if borderline_count >= max(3, len(checks) - 2) else "failed")
    return {
        "status": status,
        "checks": checks,
        "projection_valid_fraction": projection_valid_fraction,
        "evaluable_partner_frame_count": len(evaluable_frames),
        "matched_partner_frame_count": len(matched_frames & evaluable_frames),
        "partner_match_frame_fraction": partner_match_frame_fraction,
        "matched_partner_detection_count": int(len(partner_rows)),
        "median_normalized_center_error": median_error,
        "p95_normalized_center_error": p95_error,
        "median_projected_box_iou": median_iou,
        "median_identity_confidence": median_confidence,
        "systematic_offset_x_normalized": systematic_x,
        "systematic_offset_y_normalized": systematic_y,
        "systematic_offset_magnitude_normalized": systematic_mag,
        "review_required_fraction": review_fraction,
        "thresholds": {
            "minimum_projection_valid_fraction": minimum_projection_valid_fraction,
            "minimum_partner_match_frame_fraction": minimum_partner_match_frame_fraction,
            "maximum_median_normalized_center_error": maximum_median_normalized_center_error,
            "minimum_median_projected_iou": minimum_median_projected_iou,
            "maximum_systematic_offset_normalized": maximum_systematic_offset_normalized,
            "maximum_review_fraction": maximum_review_fraction,
        },
    }


def _known_target_gate(
    store: ProjectStore,
    *,
    session_id: str,
    observations_path: str | None,
    timeseries_asset_id: str,
    tracker_asset_id: str,
    profile: dict[str, Any],
    maximum_timestamp_error_ms: float,
    maximum_median_angular_error_deg: float,
    minimum_pass_fraction: float,
    minimum_observation_count: int,
    maximum_point_to_ray_distance_mm: float,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    if not observations_path:
        return {"status": "unavailable", "reason": "No known-target observations CSV was supplied"}, None
    path = Path(observations_path).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"Known-target observations file does not exist: {path}")
    observations = pd.read_csv(path, low_memory=False)
    required = {"timestamp_ns", "target_x_mm", "target_y_mm", "target_z_mm"}
    if not required.issubset(observations.columns):
        raise ValidationError(
            "Known-target CSV requires timestamp_ns,target_x_mm,target_y_mm,target_z_mm"
        )
    session = store.project.sessions[session_id]
    timeseries = session.assets[timeseries_asset_id]
    tracker = session.assets[tracker_asset_id]
    if not timeseries.primary_time_domain_id or not tracker.primary_time_domain_id:
        raise ValidationError("Known-target validation needs Timeseries and ETM time domains")
    cache = MaterializationCache(store.cache_dir())
    root = materialized_cloud_root(cache.materialize(timeseries), timeseries.recording_id)
    gaze = pd.read_csv(_find_gaze_csv(root), low_memory=False)
    timestamp_col = _find_column(gaze.columns, ["timestamp [ns]", "timestamp_ns"])
    x_col = _find_column(gaze.columns, ["gaze x [px]", "gaze_x_px", "gaze x", "x"])
    y_col = _find_column(gaze.columns, ["gaze y [px]", "gaze_y_px", "gaze y", "y"])
    gaze_ts = pd.to_numeric(gaze[timestamp_col], errors="coerce").to_numpy(dtype=np.float64)
    finite_gaze = np.isfinite(gaze_ts)
    gaze_ts_valid = gaze_ts[finite_gaze].astype(np.int64)
    gaze_indices = np.flatnonzero(finite_gaze)
    if not len(gaze_indices):
        raise ProcessingError("Timeseries gaze.csv contains no valid timestamps")

    tracker_data = read_mocap(tracker.source_path)
    labels = [str(value) for value in (profile.get("tracker") or {}).get("marker_labels") or []]
    template = np.asarray((profile.get("tracker") or {}).get("marker_template_mm"), dtype=float)
    positions, marker_valid, matched_labels, matched_prefix = resolve_profile_tracker_points(labels, tracker_data)
    pose = PoseSeries.from_markers(
        tracker_data.time_sec,
        template,
        positions,
        marker_valid,
        min_markers=3,
        max_fit_rmse_mm=12.0,
    )
    intrinsics = CameraIntrinsics.from_dict(profile["camera_intrinsics"])
    T_tracker_from_camera = np.asarray(profile["T_tracker_from_camera"], dtype=float).reshape(4, 4)
    R_tc = T_tracker_from_camera[:3, :3]
    p_tc = T_tracker_from_camera[:3, 3]
    graph = TimeGraph.from_session(session)

    rows: list[dict[str, Any]] = []
    tolerance_ns = int(round(maximum_timestamp_error_ms * 1e6))
    for obs_index, obs in observations.iterrows():
        timestamp = int(obs["timestamp_ns"])
        insertion = int(np.searchsorted(gaze_ts_valid, timestamp, side="left"))
        candidate_ordinals = [value for value in (insertion - 1, insertion) if 0 <= value < len(gaze_ts_valid)]
        if not candidate_ordinals:
            continue
        best_ordinal = min(candidate_ordinals, key=lambda value: abs(int(gaze_ts_valid[value]) - timestamp))
        source_index = int(gaze_indices[best_ordinal])
        gaze_timestamp = int(gaze_ts_valid[best_ordinal])
        timestamp_error_ns = abs(gaze_timestamp - timestamp)
        row: dict[str, Any] = {
            "observation_index": int(obs_index),
            "label": str(obs.get("label") or f"target_{obs_index}"),
            "timestamp_ns": timestamp,
            "matched_gaze_timestamp_ns": gaze_timestamp,
            "timestamp_error_ms": timestamp_error_ns / 1e6,
            "matched": timestamp_error_ns <= tolerance_ns,
            "target_x_mm": float(obs["target_x_mm"]),
            "target_y_mm": float(obs["target_y_mm"]),
            "target_z_mm": float(obs["target_z_mm"]),
            "matched_tracker_prefix": matched_prefix,
        }
        if timestamp_error_ns > tolerance_ns:
            row.update({"ray_valid": False, "pass": False, "invalid_reason": "timestamp_tolerance"})
            rows.append(row)
            continue
        query_tracker_ns = int(graph.convert(timeseries.primary_time_domain_id, tracker.primary_time_domain_id, gaze_timestamp))
        interp = pose.interpolate(np.asarray([query_tracker_ns / 1e9]), max_gap_sec=0.05)
        x = float(pd.to_numeric(pd.Series([gaze.iloc[source_index][x_col]]), errors="coerce").iloc[0])
        y = float(pd.to_numeric(pd.Series([gaze.iloc[source_index][y_col]]), errors="coerce").iloc[0])
        direction_camera = unproject_gaze_pixels(np.asarray([x]), np.asarray([y]), intrinsics)[0]
        valid = bool(interp.valid[0] and np.isfinite(direction_camera).all())
        if not valid:
            row.update({"ray_valid": False, "pass": False, "invalid_reason": "tracker_pose_or_gaze"})
            rows.append(row)
            continue
        R_wt = interp.rotations[0]
        p_wt = interp.translations[0]
        origin = p_wt + R_wt @ p_tc
        direction = normalize(R_wt @ (R_tc @ direction_camera))
        target = np.asarray([row["target_x_mm"], row["target_y_mm"], row["target_z_mm"]], dtype=float)
        target_vector = target - origin
        distance = float(np.linalg.norm(target_vector))
        target_direction = normalize(target_vector)
        cosine = float(np.clip(np.dot(direction, target_direction), -1.0, 1.0))
        angular_error = float(np.degrees(np.arccos(cosine)))
        point_to_ray = float(np.linalg.norm(np.cross(target_vector, direction)))
        passed = angular_error <= maximum_median_angular_error_deg and point_to_ray <= maximum_point_to_ray_distance_mm
        row.update({
            "ray_valid": True,
            "origin_x_mm": float(origin[0]),
            "origin_y_mm": float(origin[1]),
            "origin_z_mm": float(origin[2]),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
            "target_distance_mm": distance,
            "angular_error_deg": angular_error,
            "point_to_ray_distance_mm": point_to_ray,
            "tracker_fit_rmse_mm": float(interp.fit_rmse_mm[0]) if np.isfinite(interp.fit_rmse_mm[0]) else None,
            "pass": bool(passed),
            "invalid_reason": None,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    valid = result.loc[result.get("ray_valid", pd.Series(False, index=result.index)).astype(bool)].copy() if len(result) else pd.DataFrame()
    if len(valid):
        angular = pd.to_numeric(valid["angular_error_deg"], errors="coerce").dropna()
        distances = pd.to_numeric(valid["point_to_ray_distance_mm"], errors="coerce").dropna()
        pass_fraction = float(valid["pass"].astype(bool).mean())
        median_angular = float(angular.median()) if len(angular) else None
        p95_angular = float(angular.quantile(0.95)) if len(angular) else None
        median_distance = float(distances.median()) if len(distances) else None
    else:
        pass_fraction = 0.0
        median_angular = p95_angular = median_distance = None
    enough = len(valid) >= minimum_observation_count
    passed = bool(
        enough
        and median_angular is not None
        and median_angular <= maximum_median_angular_error_deg
        and pass_fraction >= minimum_pass_fraction
    )
    status = "passed" if passed else ("review" if len(valid) else "failed")
    return {
        "status": status,
        "source_path": str(path),
        "observation_count": int(len(result)),
        "valid_observation_count": int(len(valid)),
        "minimum_observation_count": int(minimum_observation_count),
        "pass_fraction": pass_fraction,
        "minimum_pass_fraction": minimum_pass_fraction,
        "median_angular_error_deg": median_angular,
        "p95_angular_error_deg": p95_angular,
        "maximum_median_angular_error_deg": maximum_median_angular_error_deg,
        "median_point_to_ray_distance_mm": median_distance,
        "maximum_point_to_ray_distance_mm": maximum_point_to_ray_distance_mm,
        "maximum_timestamp_error_ms": maximum_timestamp_error_ms,
        "matched_marker_labels": matched_labels,
        "matched_tracker_prefix": matched_prefix,
    }, result


def _final_status(
    transfer: dict[str, Any],
    time_gate: dict[str, Any],
    head_gate: dict[str, Any],
    known_gate: dict[str, Any],
) -> tuple[AcceptanceStatus, str, bool]:
    if transfer.get("status") != "passed" or time_gate.get("status") != "passed":
        return "rejected", "2d_only", True
    head_status = head_gate.get("status")
    known_status = known_gate.get("status")
    if head_status == "passed" and known_status == "passed":
        return "accepted", "fine_body_and_object", False
    if head_status == "passed" and known_status == "unavailable":
        return "accepted_limited", "body_and_large_object", True
    if known_status == "passed" and head_status in {"review", "failed"}:
        return "review", "coarse_body_and_large_object_pilot", True
    if head_status == "passed" and known_status == "review":
        return "review", "body_and_large_object_pilot", True
    if head_status == "review" and known_status in {"unavailable", "review"}:
        return "review", "coarse_body_only_pilot", True
    return "rejected", "2d_only", True


def _html_report(payload: dict[str, Any]) -> str:
    def value(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return html.escape(str(v))

    sections = []
    for title, key in (
        ("Tracker transfer", "transfer_validation"),
        ("Time synchronization", "time_validation"),
        ("Participant-head projection", "head_projection_validation"),
        ("Known target", "known_target_validation"),
    ):
        row = payload.get(key) or {}
        entries = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{value(v)}</td></tr>"
            for k, v in row.items()
            if not isinstance(v, (dict, list))
        )
        sections.append(f"<section><h2>{html.escape(title)}</h2><table>{entries}</table></section>")
    return f"""<!doctype html><meta charset='utf-8'><title>Transferred calibration acceptance</title>
<style>body{{font:15px/1.55 system-ui;max-width:1100px;margin:36px auto;color:#24251f}}header{{padding:22px;border:1px solid #d8d8cf;border-radius:16px;background:#f8f8f2}}.status{{font-size:28px;font-weight:800}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #e3e3da;padding:7px}}th{{width:38%}}section{{margin:24px 0}}</style>
<header><div class='status'>{html.escape(str(payload['acceptance_status']).upper())}</div><p>Allowed target granularity: <b>{html.escape(str(payload['allowed_target_granularity']))}</b></p><p>Review required: <b>{payload['review_required']}</b></p></header>{''.join(sections)}"""


def run_transferred_calibration_acceptance(
    store: ProjectStore,
    *,
    session_id: str,
    transferred_profile_artifact_id: str,
    native_trim_artifact_id: str,
    etm_assignment_artifact_id: str,
    wearer_role_id: str,
    participant_association_artifact_id: str | None = None,
    known_target_observations_path: str | None = None,
    label: str = "Transferred calibration acceptance",
    maximum_sync_rms_ms: float = 25.0,
    minimum_projection_valid_fraction: float = 0.60,
    minimum_partner_match_frame_fraction: float = 0.55,
    maximum_median_normalized_center_error: float = 0.75,
    minimum_median_projected_iou: float = 0.03,
    maximum_systematic_offset_normalized: float = 0.40,
    maximum_review_fraction: float = 0.40,
    known_target_maximum_timestamp_error_ms: float = 25.0,
    known_target_maximum_median_angular_error_deg: float = 3.0,
    known_target_minimum_pass_fraction: float = 0.67,
    known_target_minimum_observation_count: int = 3,
    known_target_maximum_point_to_ray_distance_mm: float = 80.0,
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    profile_artifact = session.artifacts.get(transferred_profile_artifact_id)
    if profile_artifact is None or profile_artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
        raise ValidationError("Select a transferred calibration profile artifact")
    profile_path = _profile_file(store, session_id, transferred_profile_artifact_id)
    profile = load_calibration_profile(profile_path)
    transfer = profile.get("transfer_validation") or profile_artifact.qc.get("transfer_validation") or {}
    if profile.get("calibration_source") != "transferred_hardware_profile":
        raise ValidationError("Calibration acceptance is intended for a transferred hardware profile")
    if transfer.get("status") != "passed":
        raise ProcessingError("Transferred calibration profile did not pass tracker-shape validation")

    trim = session.artifacts.get(native_trim_artifact_id)
    assignment_artifact = session.artifacts.get(etm_assignment_artifact_id)
    if trim is None or trim.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    if assignment_artifact is None or assignment_artifact.kind != ArtifactKind.ETM_ROLE_ASSIGNMENT:
        raise ValidationError("Select an ETM-role assignment artifact")
    assignment_row = resolve_assignment_for_role(assignment_artifact, wearer_role_id)
    tracker_asset_id = assignment_row["etm_asset_id"]
    tracker_asset = session.assets[tracker_asset_id]

    trim_manifest = json.loads(find_artifact_file(trim, names=["trim_manifest.json"]).read_text("utf-8"))
    request = trim_manifest.get("request") or {}
    timeseries_asset_id = request.get("timeseries_asset_id") or trim.parameters.get("timeseries_asset_id")
    if not timeseries_asset_id or timeseries_asset_id not in session.assets:
        raise ValidationError("Could not resolve Timeseries source from Native-reference trim")
    timeseries = session.assets[timeseries_asset_id]
    if not timeseries.primary_time_domain_id or not tracker_asset.primary_time_domain_id:
        raise ValidationError("Timeseries and ETM assets require primary time domains")

    emit(progress, 0.06, "Checking production TimeGraph path")
    time_gate = _time_gate(
        session,
        timeseries.primary_time_domain_id,
        tracker_asset.primary_time_domain_id,
        maximum_rms_ms=maximum_sync_rms_ms,
    )
    if time_gate["status"] != "passed":
        # Continue to write an auditable rejected acceptance artifact rather than hiding the cause.
        emit(progress, 0.12, "Time synchronization gate failed; recording rejection evidence")

    token.raise_if_cancelled()
    association_artifact_id = participant_association_artifact_id
    association = None
    projections = pd.DataFrame()
    identities = pd.DataFrame()
    association_manifest: dict[str, Any] = {}
    known_rows = None
    if time_gate["status"] == "passed":
        if association_artifact_id:
            association = session.artifacts.get(association_artifact_id)
            if association is None or association.kind != ArtifactKind.PARTICIPANT_ASSOCIATION:
                raise ValidationError("Selected participant-association artifact is unavailable")
        else:
            emit(progress, 0.16, "Running participant-head projection and face association")
            association_result = create_participant_association(
                store,
                session_id=session_id,
                native_trim_artifact_id=native_trim_artifact_id,
                etm_assignment_artifact_id=etm_assignment_artifact_id,
                wearer_role_id=wearer_role_id,
                calibration_profile_artifact_id=transferred_profile_artifact_id,
                target_policy="partner_only",
                output_parquet=output_parquet,
                label=f"{label} · head projection",
                progress=lambda fraction, stage: emit(progress, 0.16 + 0.42 * fraction, stage),
                cancellation=token,
            )
            association_artifact_id = association_result["artifact"]["id"]
            association = store.project.sessions[session_id].artifacts[association_artifact_id]
        projections, identities, association_manifest = load_participant_association(association)
        head_gate = _head_projection_gate(
            projections,
            identities,
            minimum_projection_valid_fraction=minimum_projection_valid_fraction,
            minimum_partner_match_frame_fraction=minimum_partner_match_frame_fraction,
            maximum_median_normalized_center_error=maximum_median_normalized_center_error,
            minimum_median_projected_iou=minimum_median_projected_iou,
            maximum_systematic_offset_normalized=maximum_systematic_offset_normalized,
            maximum_review_fraction=maximum_review_fraction,
        )

        token.raise_if_cancelled()
        emit(progress, 0.62, "Evaluating optional known-target observations")
        known_gate, known_rows = _known_target_gate(
            store,
            session_id=session_id,
            observations_path=known_target_observations_path,
            timeseries_asset_id=timeseries_asset_id,
            tracker_asset_id=tracker_asset_id,
            profile=profile,
            maximum_timestamp_error_ms=known_target_maximum_timestamp_error_ms,
            maximum_median_angular_error_deg=known_target_maximum_median_angular_error_deg,
            minimum_pass_fraction=known_target_minimum_pass_fraction,
            minimum_observation_count=known_target_minimum_observation_count,
            maximum_point_to_ray_distance_mm=known_target_maximum_point_to_ray_distance_mm,
        )
    else:
        head_gate = {"status": "failed", "reason": "Head projection was skipped because the TimeGraph gate failed"}
        known_gate = {"status": "unavailable", "reason": "Known-target validation was skipped because the TimeGraph gate failed"}
    status, granularity, review_required = _final_status(transfer, time_gate, head_gate, known_gate)
    production_ready = status in {"accepted", "accepted_limited"}
    artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "calibration_acceptance") / (".partial_" + artifact_id)
    final = store.derived_dir(session_id, "calibration_acceptance") / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        projections.to_csv(staging / "participant_head_projection.csv", index=False)
        identities.to_csv(staging / "face_identity_assignment.csv", index=False)
        if known_rows is not None:
            known_rows.to_csv(staging / "known_target_validation.csv", index=False)
        payload = {
            "schema_version": "oculostudio-calibration-acceptance-1.0",
            "artifact_id": artifact_id,
            "created_at": utc_now(),
            "acceptance_status": status,
            "production_ready": production_ready,
            "review_required": review_required,
            "allowed_target_granularity": granularity,
            "session_id": session_id,
            "session_label": session.label,
            "interaction_id": transfer.get("interaction_id") or profile_artifact.parameters.get("interaction_id"),
            "wearer_role_id": wearer_role_id,
            "transferred_profile_artifact_id": transferred_profile_artifact_id,
            "calibration_profile_id": profile.get("profile_id"),
            "native_trim_artifact_id": native_trim_artifact_id,
            "etm_assignment_artifact_id": etm_assignment_artifact_id,
            "participant_association_artifact_id": association_artifact_id,
            "timeseries_asset_id": timeseries_asset_id,
            "tracker_asset_id": tracker_asset_id,
            "transfer_validation": transfer,
            "time_validation": time_gate,
            "head_projection_validation": head_gate,
            "known_target_validation": known_gate,
            "participant_association_qc": association_manifest.get("qc") or (association.qc if association else {}),
        }
        atomic_write_json(staging / "calibration_acceptance.json", payload)
        (staging / "calibration_acceptance_report.html").write_text(_html_report(payload), "utf-8")
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    qc = {
        "acceptance_status": status,
        "production_ready": production_ready,
        "review_required": review_required,
        "allowed_target_granularity": granularity,
        "transfer_status": transfer.get("status"),
        "time_status": time_gate.get("status"),
        "head_projection_status": head_gate.get("status"),
        "known_target_status": known_gate.get("status"),
        "tracker_shape_rms_mm": transfer.get("pairwise_distance_rms_mm"),
        "sync_rms_ms": time_gate.get("rms_residual_ms"),
        "partner_match_frame_fraction": head_gate.get("partner_match_frame_fraction"),
        "median_normalized_center_error": head_gate.get("median_normalized_center_error"),
        "median_projected_box_iou": head_gate.get("median_projected_box_iou"),
        "known_target_median_angular_error_deg": known_gate.get("median_angular_error_deg"),
    }
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_ACCEPTANCE,
        label=label,
        output_root=final,
        source_asset_ids=list(dict.fromkeys([timeseries_asset_id, tracker_asset_id, *(association.source_asset_ids if association else [])])),
        parent_artifact_ids=list(dict.fromkeys([
            transferred_profile_artifact_id,
            native_trim_artifact_id,
            etm_assignment_artifact_id,
            *([association_artifact_id] if association_artifact_id else []),
        ])),
        parameters={
            "interaction_id": payload.get("interaction_id"),
            "wearer_role_id": wearer_role_id,
            "known_target_observations_path": known_target_observations_path,
            "maximum_sync_rms_ms": maximum_sync_rms_ms,
        },
        provenance={
            "calibration_profile_id": profile.get("profile_id"),
            "calibration_profile_artifact_id": transferred_profile_artifact_id,
            "participant_association_artifact_id": association_artifact_id,
        },
        qc=qc,
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, f"Transferred calibration acceptance: {status}")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "acceptance": payload,
        "report": str(final / "calibration_acceptance_report.html"),
    }


def load_calibration_acceptance(artifact) -> dict[str, Any]:
    if artifact.kind != ArtifactKind.CALIBRATION_ACCEPTANCE:
        raise ValidationError("Artifact is not a calibration acceptance")
    path = find_artifact_file(artifact, names=["calibration_acceptance.json"])
    return json.loads(path.read_text("utf-8"))
