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
from scipy.optimize import linear_sum_assignment

from oculostudio.adapters.mocap import MocapData, read_mocap
from oculostudio.domain.geometry import PoseSeries, invert_transform, make_transform, normalize, transform_points
from oculostudio.domain.models import ArtifactKind, ArtifactRecord
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ProcessingError, TimeGraphError, ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.calibration import CameraIntrinsics, load_calibration_profile
from oculostudio.services.calibration_profiles import resolve_profile_tracker_points
from oculostudio.services.etm_role_assignment import load_etm_assignment, resolve_assignment_for_role
from oculostudio.services.face_semantics import FaceInstance, load_face_semantics
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


TargetPolicy = Literal["partner_only", "tracked_participants", "all_detected"]


PARTICIPANT_ASSOCIATION_ENGINE = "projected-box-overlap-identity-v4"

MATCHED_IDENTITY_STATUSES = {
    "matched_participant",
    "projection_locked_participant",
    "overlap_locked_participant",
    "projection_cluster_inherits_participant",
    "duplicate_inherits_matched_participant",
    "temporal_bridge_matched_participant",
    "temporal_edge_matched_participant",
}



def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {path.name}: {exc}") from exc


def _trim_root(artifact: ArtifactRecord) -> tuple[Path, dict[str, Any]]:
    manifest = find_artifact_file(artifact, names=["trim_manifest.json"])
    return manifest.parent, _read_json(manifest)


def _profile_path(
    store: ProjectStore,
    session_id: str,
    *,
    profile_artifact_id: str | None,
    profile_path: str | None,
) -> tuple[Path, str | None]:
    if profile_path:
        path = Path(profile_path).expanduser().resolve()
        if not path.exists():
            raise ValidationError(f"Calibration profile does not exist: {path}")
        if path.suffix.lower() not in {".oculocal", ".g3dcal", ".json"}:
            raise ValidationError("Calibration profile must be .oculocal, .g3dcal or compatible JSON")
        return path, None
    if not profile_artifact_id:
        raise ValidationError("A calibration profile artifact or file is required")
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(profile_artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
        raise ValidationError("Selected calibration profile artifact is unavailable")
    candidates = [
        Path(row.path) for row in artifact.files
        if Path(row.path).suffix.lower() in {".oculocal", ".g3dcal", ".json"}
        and Path(row.path).exists()
    ]
    if not candidates:
        raise ValidationError("Calibration artifact contains no supported profile file")
    candidates.sort(key=lambda path: (0 if path.suffix.lower() in {".oculocal", ".g3dcal"} else 1, str(path)))
    return candidates[0], artifact.id


def _suffix(label: str) -> str:
    return str(label).strip().rsplit(":", 1)[-1].upper().replace(" ", "")


def _interpolate_markers(
    data: MocapData,
    marker_labels: list[str],
    query_sec: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, valid = data.select(marker_labels)
    query = np.asarray(query_sec, dtype=float)
    output = np.full((len(query), len(marker_labels), 3), np.nan, dtype=float)
    output_valid = np.zeros((len(query), len(marker_labels)), dtype=bool)
    gap_sec = np.full((len(query), len(marker_labels)), np.nan, dtype=float)
    for marker_index in range(len(marker_labels)):
        source_indices = np.flatnonzero(valid[:, marker_index])
        if not len(source_indices):
            continue
        source_time = data.time_sec[source_indices]
        source_points = points[source_indices, marker_index]
        if len(source_indices) == 1:
            exact = np.isclose(query, source_time[0], atol=1e-9, rtol=0.0)
            output[exact, marker_index] = source_points[0]
            output_valid[exact, marker_index] = True
            gap_sec[exact, marker_index] = 0.0
            continue
        right = np.searchsorted(source_time, query, side="left")
        right = np.clip(right, 0, len(source_time) - 1)
        left = np.maximum(0, right - 1)
        exact = np.isclose(source_time[right], query, atol=1e-9, rtol=0.0)
        left[exact] = right[exact]
        span = source_time[right] - source_time[left]
        inside = np.isfinite(query) & (query >= source_time[0]) & (query <= source_time[-1])
        allowed = inside & (exact | (span <= max_gap_sec))
        if not allowed.any():
            continue
        selected = np.flatnonzero(allowed)
        l = left[selected]
        r = right[selected]
        denominator = source_time[r] - source_time[l]
        alpha = np.zeros(len(selected), dtype=float)
        nonzero = denominator > 0
        alpha[nonzero] = (query[selected][nonzero] - source_time[l][nonzero]) / denominator[nonzero]
        output[selected, marker_index] = (
            source_points[l] * (1.0 - alpha[:, None]) + source_points[r] * alpha[:, None]
        )
        output_valid[selected, marker_index] = True
        gap_sec[selected, marker_index] = span[selected]
    return output, output_valid, gap_sec


def _rescaled_intrinsics(intrinsics: CameraIntrinsics, width: int, height: int) -> tuple[CameraIntrinsics, dict[str, Any]]:
    if not intrinsics.width or not intrinsics.height:
        return CameraIntrinsics(intrinsics.matrix.copy(), intrinsics.distortion.copy(), width, height), {
            "rescaled": False,
            "reason": "profile_resolution_missing",
        }
    sx = width / float(intrinsics.width)
    sy = height / float(intrinsics.height)
    source_aspect = intrinsics.width / float(intrinsics.height)
    target_aspect = width / float(height)
    if abs(source_aspect - target_aspect) / max(source_aspect, 1e-9) > 0.015:
        raise ValidationError(
            "Calibration resolution and Native scene resolution have different aspect ratios; "
            "crop/rotation cannot be inferred safely"
        )
    matrix = intrinsics.matrix.copy().astype(float)
    matrix[0, 0] *= sx
    matrix[0, 2] *= sx
    matrix[1, 1] *= sy
    matrix[1, 2] *= sy
    return CameraIntrinsics(matrix, intrinsics.distortion.copy(), width, height), {
        "rescaled": bool(abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9),
        "source_resolution": [intrinsics.width, intrinsics.height],
        "target_resolution": [width, height],
        "scale_x": sx,
        "scale_y": sy,
    }


def _project_camera_points(points_camera_mm: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points = np.asarray(points_camera_mm, dtype=float).reshape(-1, 3)
    result = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    if valid.any():
        projected, _ = cv2.projectPoints(
            points[valid].astype(np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            intrinsics.matrix.astype(np.float64),
            intrinsics.distortion.astype(np.float64),
        )
        result[valid] = projected.reshape(-1, 2)
    return result


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _head_geometry(
    marker_points_mm: np.ndarray,
    marker_valid: np.ndarray,
    labels: list[str],
) -> dict[str, Any] | None:
    indices = np.flatnonzero(marker_valid & np.isfinite(marker_points_mm).all(axis=1))
    if len(indices) < 3:
        return None
    points = marker_points_mm[indices]
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(np.clip(np.nanmax(distances) + 25.0, 65.0, 145.0))
    by_suffix = {_suffix(labels[index]): marker_points_mm[index] for index in indices}
    front_points = [by_suffix[key] for key in ("LFHD", "RFHD", "ARIEL") if key in by_suffix]
    back_points = [by_suffix[key] for key in ("LBHD", "RBHD") if key in by_suffix]
    front_center = np.mean(front_points, axis=0) if front_points else center
    forward = None
    if front_points and back_points:
        forward = normalize(front_center - np.mean(back_points, axis=0))
        if not np.isfinite(forward).all():
            forward = None
    face_center = front_center if front_points else center
    anatomical_left = by_suffix.get("LFHD")
    anatomical_right = by_suffix.get("RFHD")
    return {
        "center_world_mm": center,
        "face_center_world_mm": face_center,
        "forward_world": forward,
        "radius_mm": radius,
        "marker_count": int(len(indices)),
        "marker_points_world_mm": points,
        "anatomical_left_head_world_mm": anatomical_left,
        "anatomical_right_head_world_mm": anatomical_right,
    }


def _project_head(
    geometry: dict[str, Any],
    T_world_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    T_camera_from_world = invert_transform(T_world_from_camera)
    center_world = np.asarray(geometry["center_world_mm"], dtype=float)
    face_center_world = np.asarray(geometry["face_center_world_mm"], dtype=float)
    center_camera = transform_points(T_camera_from_world, center_world)
    face_camera = transform_points(T_camera_from_world, face_center_world)
    face_px = _project_camera_points(face_camera.reshape(1, 3), intrinsics)[0]
    marker_camera = transform_points(T_camera_from_world, geometry["marker_points_world_mm"])
    marker_px = _project_camera_points(marker_camera, intrinsics)
    depth = float(face_camera[2])
    in_front = bool(np.isfinite(face_camera).all() and depth > 1e-6)
    radius = float(geometry["radius_mm"])
    if in_front and np.isfinite(face_px).all():
        radius_x = max(8.0, float(intrinsics.matrix[0, 0]) * radius / depth * 1.05)
        radius_y = max(10.0, float(intrinsics.matrix[1, 1]) * radius / depth * 1.30)
        finite_marker = marker_px[np.isfinite(marker_px).all(axis=1)]
        if len(finite_marker):
            mx1, my1 = finite_marker.min(axis=0)
            mx2, my2 = finite_marker.max(axis=0)
            radius_x = max(radius_x, 0.75 * max(1.0, mx2 - mx1))
            radius_y = max(radius_y, 0.95 * max(1.0, my2 - my1))
        x1 = float(face_px[0] - radius_x)
        y1 = float(face_px[1] - radius_y * 0.82)
        x2 = float(face_px[0] + radius_x)
        y2 = float(face_px[1] + radius_y * 1.18)
        in_view = bool(x2 >= 0 and y2 >= 0 and x1 < image_width and y1 < image_height)
    else:
        x1 = y1 = x2 = y2 = float("nan")
        in_view = False
    def _project_optional_head_marker(key: str) -> tuple[float | None, float | None]:
        point = geometry.get(key)
        if point is None:
            return None, None
        point_camera = transform_points(T_camera_from_world, np.asarray(point, dtype=float))
        projected = _project_camera_points(np.asarray(point_camera, dtype=float).reshape(1, 3), intrinsics)[0]
        if not np.isfinite(projected).all():
            return None, None
        return float(projected[0]), float(projected[1])

    anatomical_left_x, anatomical_left_y = _project_optional_head_marker("anatomical_left_head_world_mm")
    anatomical_right_x, anatomical_right_y = _project_optional_head_marker("anatomical_right_head_world_mm")

    forward = geometry.get("forward_world")
    facing = None
    if forward is not None and np.isfinite(forward).all():
        camera_world = np.asarray(T_world_from_camera[:3, 3], dtype=float)
        to_camera = normalize(camera_world - center_world)
        if np.isfinite(to_camera).all():
            facing = float(np.clip(np.dot(forward, to_camera), -1.0, 1.0))
    return {
        "head_center_world_x_mm": float(center_world[0]),
        "head_center_world_y_mm": float(center_world[1]),
        "head_center_world_z_mm": float(center_world[2]),
        "head_center_camera_x_mm": float(center_camera[0]),
        "head_center_camera_y_mm": float(center_camera[1]),
        "head_center_camera_z_mm": float(center_camera[2]),
        "projected_head_x_px": float(face_px[0]) if np.isfinite(face_px[0]) else None,
        "projected_head_y_px": float(face_px[1]) if np.isfinite(face_px[1]) else None,
        "projected_head_depth_m": depth / 1000.0 if np.isfinite(depth) else None,
        "projected_bbox_x1": x1 if np.isfinite(x1) else None,
        "projected_bbox_y1": y1 if np.isfinite(y1) else None,
        "projected_bbox_x2": x2 if np.isfinite(x2) else None,
        "projected_bbox_y2": y2 if np.isfinite(y2) else None,
        "projected_anatomical_left_head_x_px": anatomical_left_x,
        "projected_anatomical_left_head_y_px": anatomical_left_y,
        "projected_anatomical_right_head_x_px": anatomical_right_x,
        "projected_anatomical_right_head_y_px": anatomical_right_y,
        "head_facing_score": facing,
        "in_camera_view": in_view,
        "head_marker_count": int(geometry["marker_count"]),
    }


def _face_box(instance: FaceInstance) -> tuple[float, float, float, float]:
    return float(instance.x1), float(instance.y1), float(instance.x2), float(instance.y2)


def _bbox_overlap_metrics(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> dict[str, float | bool]:
    """Return auditable overlap metrics for two axis-aligned boxes.

    Participant identity uses the projected 3D head box only as an identity
    gate.  The FaceMapper box remains the authoritative 2D face geometry for
    all downstream AOIs.  Consequently, a projected box does not need to match
    the FaceMapper box in size or centre: a positive-area intersection is the
    primary identity evidence.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return {
        "bbox_intersection_area_px2": float(intersection),
        "bbox_overlap_present": bool(intersection > 1e-6),
        "projected_box_overlap_fraction": float(intersection / area_a) if area_a > 0 else 0.0,
        "face_box_overlap_fraction": float(intersection / area_b) if area_b > 0 else 0.0,
        "projected_box_iou": float(intersection / union) if union > 0 else 0.0,
    }


def _pair_metrics(projection: dict[str, Any], face: FaceInstance, previous_track: int | None) -> dict[str, float | bool]:
    predicted = (
        float(projection["projected_bbox_x1"]),
        float(projection["projected_bbox_y1"]),
        float(projection["projected_bbox_x2"]),
        float(projection["projected_bbox_y2"]),
    )
    observed = _face_box(face)
    px = float(projection["projected_head_x_px"])
    py = float(projection["projected_head_y_px"])
    center_error = math.hypot(px - face.center.x, py - face.center.y)
    scale = max(12.0, math.hypot(face.width, face.height))
    normalized_error = center_error / scale
    overlap = _bbox_overlap_metrics(predicted, observed)
    iou = float(overlap["projected_box_iou"])
    projected_overlap = float(overlap["projected_box_overlap_fraction"])
    face_overlap = float(overlap["face_box_overlap_fraction"])
    overlap_present = bool(overlap["bbox_overlap_present"])
    predicted_area = max(1.0, (predicted[2] - predicted[0]) * (predicted[3] - predicted[1]))
    observed_area = max(1.0, face.width * face.height)
    area_log_error = abs(math.log(observed_area / predicted_area))
    projected_center_inside_face = bool(observed[0] <= px <= observed[2] and observed[1] <= py <= observed[3])
    face_center_inside_projection = bool(
        predicted[0] <= face.center.x <= predicted[2] and predicted[1] <= face.center.y <= predicted[3]
    )
    facing = projection.get("head_facing_score")
    facing_penalty = 0.0 if facing is None else 0.5 * max(0.0, -float(facing))
    temporal_bonus = 0.18 if previous_track is not None and int(face.track_id) == int(previous_track) else 0.0

    # Identity is overlap-first.  Centre distance and size mismatch are only
    # tie-breakers when more than one distinct FaceMapper face overlaps a
    # projected participant box; they never veto a unique positive overlap.
    overlap_penalty = 1.0 - face_overlap if overlap_present else 2.0
    cost = (
        0.72 * overlap_penalty
        + 0.11 * min(2.5, normalized_error)
        + 0.07 * (1.0 - projected_overlap)
        + 0.05 * min(2.0, area_log_error)
        + 0.02 * facing_penalty
        - temporal_bonus
    )
    center_score = math.exp(-1.15 * min(2.5, normalized_error))
    overlap_score = float(np.clip(face_overlap, 0.0, 1.0))
    projection_coverage_score = float(np.clip(projected_overlap / 0.35, 0.0, 1.0))
    geometry_score = float(np.clip(
        0.78 * overlap_score + 0.12 * projection_coverage_score + 0.10 * center_score,
        0.0,
        1.0,
    ))
    geometry_strong = bool(overlap_present)
    return {
        "center_error_px": float(center_error),
        "normalized_center_error": float(normalized_error),
        "projected_box_iou": float(iou),
        "bbox_intersection_area_px2": float(overlap["bbox_intersection_area_px2"]),
        "bbox_overlap_present": overlap_present,
        "projected_box_overlap_fraction": projected_overlap,
        "face_box_overlap_fraction": face_overlap,
        "area_log_error": float(area_log_error),
        "projected_center_inside_face": projected_center_inside_face,
        "face_center_inside_projection": face_center_inside_projection,
        "temporal_track_consistency": bool(temporal_bonus > 0),
        "geometry_score": geometry_score,
        "geometry_strong": geometry_strong,
        "identity_gate": "positive_bbox_intersection",
        "cost": float(max(0.0, cost)),
    }

def _duplicate_face_metrics(a: FaceInstance, b: FaceInstance) -> dict[str, float | bool]:
    iou = _bbox_iou(_face_box(a), _face_box(b))
    scale = max(12.0, 0.5 * (math.hypot(a.width, a.height) + math.hypot(b.width, b.height)))
    center_distance = math.hypot(a.center.x - b.center.x, a.center.y - b.center.y) / scale
    area_a = max(1.0, a.width * a.height)
    area_b = max(1.0, b.width * b.height)
    area_ratio = max(area_a, area_b) / min(area_a, area_b)
    duplicate = bool(
        (iou >= 0.50 and area_ratio <= 3.0)
        or (center_distance <= 0.24 and area_ratio <= 2.5)
    )
    return {
        "duplicate": duplicate,
        "duplicate_iou": float(iou),
        "duplicate_center_distance": float(center_distance),
        "duplicate_area_ratio": float(area_ratio),
    }


def _projection_supports_face(metrics: dict[str, Any], *, relaxed: bool = False) -> bool:
    """Return whether a projected participant box overlaps a FaceMapper face.

    The light-blue 3D projection is an identity gate, not the AOI geometry.
    A positive-area intersection is therefore sufficient support.  ``relaxed``
    remains in the signature for backward compatibility but intentionally does
    not change the overlap rule.
    """
    return bool(metrics.get("bbox_overlap_present"))

def _same_projection_face_cluster(
    source: FaceInstance,
    candidate: FaceInstance,
    projection_metrics: dict[str, Any],
) -> bool:
    duplicate = _duplicate_face_metrics(source, candidate)
    if duplicate["duplicate"]:
        return True
    # FaceMapper can emit a tight landmark face and a larger face box for the
    # same physical person.  They may have modest IoU but share the same centre
    # and the same projected participant support.
    return bool(
        _projection_supports_face(projection_metrics, relaxed=True)
        and float(duplicate["duplicate_center_distance"]) <= 0.42
        and float(duplicate["duplicate_area_ratio"]) <= 5.0
        and (
            float(duplicate["duplicate_iou"]) >= 0.16
            or bool(projection_metrics.get("projected_center_inside_face"))
        )
    )


def _associate_frame(
    projections: list[dict[str, Any]],
    faces: list[FaceInstance],
    *,
    previous_track_by_role: dict[str, int],
    maximum_normalized_center_error: float,
    minimum_projected_iou: float,
    minimum_assignment_margin: float,
    minimum_identity_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not faces:
        return [], previous_track_by_role
    usable = [row for row in projections if row.get("projection_valid") and row.get("in_camera_view")]
    if not usable:
        return [
            {
                "face": face,
                "matched_entity_id": None,
                "matched_role": None,
                "candidate_entity_id": None,
                "candidate_role": None,
                "candidate_relation": None,
                "target_relation": "other_person",
                "identity_status": "no_participant_projection",
                "identity_confidence": 0.0,
                "include_in_partner_analysis": False,
                "review_required": True,
                "match_reason": "no_participant_projection",
                "metrics": {},
            }
            for face in faces
        ], previous_track_by_role

    costs = np.full((len(usable), len(faces)), 1e6, dtype=float)
    metrics: dict[tuple[int, int], dict[str, Any]] = {}
    for i, projection in enumerate(usable):
        role = str(projection["role_id"])
        for j, face in enumerate(faces):
            row = _pair_metrics(projection, face, previous_track_by_role.get(role))
            # The projected head box is used only to identify which green
            # FaceMapper box belongs to a participant.  Any positive-area
            # overlap is a candidate; the legacy centre/IoU thresholds remain
            # recorded for compatibility but no longer reject a valid overlap.
            candidate = bool(row.get("bbox_overlap_present"))
            row["candidate"] = bool(candidate)
            metrics[(i, j)] = row
            if candidate:
                costs[i, j] = float(row["cost"])

    row_ind, col_ind = linear_sum_assignment(costs)
    accepted_by_face: dict[int, dict[str, Any]] = {}
    updated_tracks = dict(previous_track_by_role)
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if costs[i, j] >= 1e5:
            continue
        projection = usable[i]
        face = faces[j]
        row = metrics[(i, j)]
        selected_cost = float(costs[i, j])
        alternatives: list[float] = []
        # Near-identical FaceMapper boxes are duplicate hypotheses for one
        # physical face, not competing identities.  They must not collapse the
        # assignment margin and turn a clearly projected partner into other.
        for k in range(len(faces)):
            if k == j or costs[i, k] >= 1e5:
                continue
            duplicate = _duplicate_face_metrics(face, faces[k])
            if duplicate["duplicate"]:
                continue
            alternatives.append(float(costs[i, k]) - selected_cost)
        for k in range(len(usable)):
            if k == i or costs[k, j] >= 1e5:
                continue
            alternatives.append(float(costs[k, j]) - selected_cost)
        margin = min(alternatives) if alternatives else 1.0
        projection_confidence = float(np.clip(projection.get("projection_confidence") or 0.0, 0.0, 1.0))
        face_confidence = float(np.clip(face.confidence, 0.0, 1.0))
        margin_factor = float(np.clip(margin / max(minimum_assignment_margin, 1e-9), 0.0, 1.0))
        geometry_confidence = math.exp(-float(row["cost"]))
        # Do not multiply low-but-valid detector/projection confidences to near
        # zero.  Both sources have already passed their own validity gates.
        confidence = float(np.clip(
            geometry_confidence
            * (0.58 + 0.42 * projection_confidence)
            * (0.58 + 0.42 * face_confidence)
            * (0.72 + 0.28 * margin_factor),
            0.0,
            1.0,
        ))
        projection_relation = str(projection["relation_to_wearer"])

        # A unique overlap is decisive identity evidence.  Multiple near-
        # identical FaceMapper boxes count as one physical face; two genuinely
        # distinct overlapping faces remain ambiguous and are sent to review.
        overlapping_faces = [
            k for k in range(len(faces))
            if bool((metrics.get((i, k)) or {}).get("bbox_overlap_present"))
        ]
        projection_unique_face = all(
            k == j or _duplicate_face_metrics(face, faces[k])["duplicate"]
            for k in overlapping_faces
        )
        overlapping_projections = [
            k for k in range(len(usable))
            if bool((metrics.get((k, j)) or {}).get("bbox_overlap_present"))
        ]
        overlapping_entities = {str(usable[k].get("entity_id")) for k in overlapping_projections}
        face_unique_projection = len(overlapping_entities) <= 1
        overlap_ambiguous = bool(not projection_unique_face or not face_unique_projection)
        row["overlap_unique_for_projection"] = bool(projection_unique_face)
        row["overlap_unique_for_face"] = bool(face_unique_projection)
        row["overlap_distinct_face_count"] = int(1 if projection_unique_face and overlapping_faces else len(overlapping_faces))
        row["overlap_distinct_projection_count"] = int(len(overlapping_entities))

        ambiguous = bool(overlap_ambiguous or margin < minimum_assignment_margin)
        geometry_rescue_threshold = max(0.24, minimum_identity_confidence * 0.60)
        geometry_rescue = bool(
            row.get("geometry_strong")
            and not ambiguous
            and confidence >= geometry_rescue_threshold
            and (margin >= minimum_assignment_margin or row.get("temporal_track_consistency"))
        )
        overlap_lock = bool(
            row.get("bbox_overlap_present")
            and not overlap_ambiguous
            and float(np.clip(face.confidence, 0.0, 1.0)) >= 0.01
        )
        if overlap_lock:
            confidence = max(
                confidence,
                float(np.clip(
                    0.70
                    + 0.25 * float(row.get("face_box_overlap_fraction") or 0.0)
                    + (0.05 if row.get("temporal_track_consistency") else 0.0),
                    0.0,
                    1.0,
                )),
            )
        accepted = bool(
            overlap_lock
            or (not overlap_ambiguous and not ambiguous and (confidence >= minimum_identity_confidence or geometry_rescue))
        )
        if accepted:
            identity_status = "overlap_locked_participant" if overlap_lock else "matched_participant"
            matched_role = str(projection["role_id"])
            matched_entity = str(projection["entity_id"])
            target_relation = projection_relation
            updated_tracks[matched_role] = int(face.track_id)
            match_reason = (
                "unique_projected_bbox_overlap" if overlap_lock
                else ("geometry_rescue" if geometry_rescue and confidence < minimum_identity_confidence else "confidence_gate")
            )
        else:
            # A plausible tracked-participant candidate is not an unrelated
            # person.  Preserve its projected identity and relation for review
            # instead of silently demoting it to other_person.
            identity_status = "ambiguous_participant_candidate" if ambiguous else "low_confidence_participant_candidate"
            matched_role = None
            matched_entity = None
            target_relation = f"ambiguous_{projection_relation}"
            match_reason = "assignment_margin" if ambiguous else "identity_confidence"
        accepted_by_face[j] = {
            "face": face,
            "matched_entity_id": matched_entity,
            "matched_role": matched_role,
            "candidate_entity_id": str(projection["entity_id"]),
            "candidate_role": str(projection["role_id"]),
            "candidate_relation": projection_relation,
            "target_relation": target_relation,
            "identity_status": identity_status,
            "identity_confidence": confidence,
            "include_in_partner_analysis": bool(accepted and projection_relation == "partner"),
            "review_required": bool(not accepted),
            "match_reason": match_reason,
            "metrics": {**row, "assignment_margin": float(margin)},
            "projection": projection,
        }

    # FaceMapper can emit overlapping duplicate candidates for the same face.
    # If one candidate has a tracked identity, near-identical candidates in the
    # same frame inherit it instead of being labelled as unrelated people.
    for face_index, face in enumerate(faces):
        if face_index in accepted_by_face:
            continue
        best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for source_index, source in accepted_by_face.items():
            duplicate = _duplicate_face_metrics(face, faces[source_index])
            source_projection = source.get("projection") or {}
            candidate_metrics = _pair_metrics(source_projection, face, None) if source_projection else {}
            if not _same_projection_face_cluster(faces[source_index], face, candidate_metrics):
                continue
            score = float(duplicate["duplicate_iou"] - 0.25 * duplicate["duplicate_center_distance"])
            if best is None or score > best[0]:
                best = (score, source, duplicate)
        if best is None:
            continue
        _, source, duplicate = best
        source_matched = str(source.get("identity_status")) in MATCHED_IDENTITY_STATUSES
        confidence = float(np.clip(
            min(float(source.get("identity_confidence") or 0.0), float(np.clip(face.confidence, 0.0, 1.0)))
            * (0.82 + 0.18 * float(duplicate["duplicate_iou"])),
            0.0,
            1.0,
        ))
        if source_matched:
            accepted_by_face[face_index] = {
                "face": face,
                "matched_entity_id": source.get("matched_entity_id"),
                "matched_role": source.get("matched_role"),
                "candidate_entity_id": source.get("candidate_entity_id") or source.get("matched_entity_id"),
                "candidate_role": source.get("candidate_role") or source.get("matched_role"),
                "candidate_relation": source.get("candidate_relation") or source.get("target_relation"),
                "target_relation": source.get("target_relation"),
                "identity_status": "projection_cluster_inherits_participant",
                "identity_confidence": confidence,
                "include_in_partner_analysis": bool(source.get("include_in_partner_analysis")),
                "review_required": bool(source.get("review_required")),
                "match_reason": "same_frame_duplicate_inheritance",
                "metrics": duplicate,
                "projection": source.get("projection") or {},
            }
        else:
            candidate_relation = source.get("candidate_relation")
            accepted_by_face[face_index] = {
                "face": face,
                "matched_entity_id": None,
                "matched_role": None,
                "candidate_entity_id": source.get("candidate_entity_id"),
                "candidate_role": source.get("candidate_role"),
                "candidate_relation": candidate_relation,
                "target_relation": f"ambiguous_{candidate_relation}" if candidate_relation else "ambiguous_person",
                "identity_status": "duplicate_inherits_participant_candidate",
                "identity_confidence": confidence,
                "include_in_partner_analysis": False,
                "review_required": True,
                "match_reason": "same_frame_duplicate_candidate",
                "metrics": duplicate,
                "projection": source.get("projection") or {},
            }

    # Any remaining face that is directly supported by a unique projected
    # partner must never be silently classified as an unrelated person.  This
    # catches FaceMapper hypotheses that differ in scale from the representative
    # box selected by the Hungarian assignment.
    for face_index, face in enumerate(faces):
        if face_index in accepted_by_face:
            continue
        supported: list[tuple[float, int, dict[str, Any]]] = []
        for projection_index, projection in enumerate(usable):
            relation = str(projection.get("relation_to_wearer") or "")
            row = metrics.get((projection_index, face_index)) or _pair_metrics(
                projection, face, previous_track_by_role.get(str(projection.get("role_id")))
            )
            if relation == "partner" and _projection_supports_face(row, relaxed=True):
                supported.append((float(row.get("cost") or 1e6), projection_index, row))
        if not supported:
            continue
        supported.sort(key=lambda item: item[0])
        _, projection_index, row = supported[0]
        projection = usable[projection_index]
        overlapping_faces_for_projection = [
            k for k in range(len(faces))
            if bool((metrics.get((projection_index, k)) or {}).get("bbox_overlap_present"))
        ]
        projection_unique_face = all(
            k == face_index or _duplicate_face_metrics(face, faces[k])["duplicate"]
            for k in overlapping_faces_for_projection
        )
        overlapping_projection_entities = {
            str(usable[k].get("entity_id"))
            for k in range(len(usable))
            if bool((metrics.get((k, face_index)) or {}).get("bbox_overlap_present"))
        }
        face_unique_projection = len(overlapping_projection_entities) <= 1
        row["overlap_unique_for_projection"] = bool(projection_unique_face)
        row["overlap_unique_for_face"] = bool(face_unique_projection)
        row["overlap_distinct_face_count"] = int(1 if projection_unique_face and overlapping_faces_for_projection else len(overlapping_faces_for_projection))
        row["overlap_distinct_projection_count"] = int(len(overlapping_projection_entities))
        strong = bool(
            _projection_supports_face(row, relaxed=False)
            and projection_unique_face
            and face_unique_projection
        )
        projection_confidence = float(np.clip(projection.get("projection_confidence") or 0.0, 0.0, 1.0))
        face_confidence = float(np.clip(face.confidence, 0.0, 1.0))
        confidence = float(np.clip(
            max(float(row.get("geometry_score") or 0.0), math.exp(-float(row.get("cost") or 5.0)))
            * (0.62 + 0.38 * projection_confidence)
            * (0.62 + 0.38 * face_confidence),
            0.0,
            1.0,
        ))
        matched_role = str(projection["role_id"])
        matched_entity = str(projection["entity_id"])
        existing_for_role = [
            (source_index, source)
            for source_index, source in accepted_by_face.items()
            if str(source.get("matched_role") or "") == matched_role
            and str(source.get("identity_status") or "") in MATCHED_IDENTITY_STATUSES
        ]
        if existing_for_role:
            # A role already has a representative face in this frame.  A second
            # box is only promoted when it is geometrically the same physical
            # face; otherwise retain it as a partner candidate for review.
            strong = bool(strong and any(
                _same_projection_face_cluster(faces[source_index], face, row)
                for source_index, _ in existing_for_role
            ))
        if strong:
            confidence = max(
                confidence,
                float(np.clip(
                    0.70 + 0.25 * float(row.get("face_box_overlap_fraction") or 0.0),
                    0.0,
                    1.0,
                )),
            )
            accepted_by_face[face_index] = {
                "face": face,
                "matched_entity_id": matched_entity,
                "matched_role": matched_role,
                "candidate_entity_id": matched_entity,
                "candidate_role": matched_role,
                "candidate_relation": "partner",
                "target_relation": "partner",
                "identity_status": "overlap_locked_participant",
                "identity_confidence": confidence,
                "include_in_partner_analysis": True,
                "review_required": False,
                "match_reason": "direct_unique_projected_bbox_overlap",
                "metrics": row,
                "projection": projection,
            }
            updated_tracks[matched_role] = int(face.track_id)
        else:
            accepted_by_face[face_index] = {
                "face": face,
                "matched_entity_id": None,
                "matched_role": None,
                "candidate_entity_id": str(projection["entity_id"]),
                "candidate_role": str(projection["role_id"]),
                "candidate_relation": "partner",
                "target_relation": "ambiguous_partner",
                "identity_status": "projected_partner_candidate",
                "identity_confidence": confidence,
                "include_in_partner_analysis": False,
                "review_required": True,
                "match_reason": "direct_projection_candidate",
                "metrics": row,
                "projection": projection,
            }

    output: list[dict[str, Any]] = []
    for face_index, face in enumerate(faces):
        row = accepted_by_face.get(face_index)
        if row is None:
            output.append({
                "face": face,
                "matched_entity_id": f"other_person_track_{face.track_id}",
                "matched_role": None,
                "candidate_entity_id": None,
                "candidate_role": None,
                "candidate_relation": None,
                "target_relation": "other_person",
                "identity_status": "untracked_person",
                "identity_confidence": float(np.clip(face.confidence * 0.5, 0.0, 1.0)),
                "include_in_partner_analysis": False,
                "review_required": False,
                "match_reason": "no_tracked_participant_candidate",
                "metrics": {},
            })
        else:
            output.append(row)
    return output, updated_tracks


def _stabilize_identity_tracks(identities: pd.DataFrame, *, max_bridge_frames: int = 12, max_edge_frames: int = 2) -> pd.DataFrame:
    """Bridge short identity dropouts without converting unrelated faces.

    Only rows on the same FaceMapper track are rescued, and a bridge requires
    matching participant evidence on both sides.  Edge rescue additionally
    requires a current projected-participant candidate.
    """
    if identities.empty or "face_track_id" not in identities.columns:
        return identities
    frame = identities.copy()
    frame["temporal_identity_rescue"] = False
    matched_mask = frame["identity_status"].astype(str).isin(MATCHED_IDENTITY_STATUSES)
    for _, group in frame.groupby("face_track_id", sort=False):
        ordered = group.sort_values("native_frame_index")
        positions = ordered.index.tolist()
        matched_positions = [idx for idx in positions if bool(matched_mask.loc[idx])]
        if not matched_positions:
            continue
        for pos, idx in enumerate(positions):
            if bool(matched_mask.loc[idx]):
                continue
            current_frame = int(frame.at[idx, "native_frame_index"])
            previous = [m for m in matched_positions if int(frame.at[m, "native_frame_index"]) < current_frame]
            following = [m for m in matched_positions if int(frame.at[m, "native_frame_index"]) > current_frame]
            prev_idx = previous[-1] if previous else None
            next_idx = following[0] if following else None
            rescue_source = None
            rescue_status = None
            if prev_idx is not None and next_idx is not None:
                prev_gap = current_frame - int(frame.at[prev_idx, "native_frame_index"])
                next_gap = int(frame.at[next_idx, "native_frame_index"]) - current_frame
                same_entity = str(frame.at[prev_idx, "matched_entity_id"]) == str(frame.at[next_idx, "matched_entity_id"])
                same_relation = str(frame.at[prev_idx, "target_relation"]) == str(frame.at[next_idx, "target_relation"])
                if same_entity and same_relation and prev_gap <= max_bridge_frames and next_gap <= max_bridge_frames:
                    rescue_source = (prev_idx, next_idx)
                    rescue_status = "temporal_bridge_matched_participant"
            if rescue_source is None:
                candidate_relation = str(frame.at[idx, "candidate_relation"]) if "candidate_relation" in frame.columns and pd.notna(frame.at[idx, "candidate_relation"]) else ""
                candidate_entity = str(frame.at[idx, "candidate_entity_id"]) if "candidate_entity_id" in frame.columns and pd.notna(frame.at[idx, "candidate_entity_id"]) else ""
                for neighbour in (prev_idx, next_idx):
                    if neighbour is None:
                        continue
                    gap = abs(current_frame - int(frame.at[neighbour, "native_frame_index"]))
                    if gap > max_edge_frames:
                        continue
                    if candidate_relation and candidate_relation == str(frame.at[neighbour, "target_relation"]):
                        if not candidate_entity or candidate_entity == str(frame.at[neighbour, "matched_entity_id"]):
                            rescue_source = (neighbour,)
                            rescue_status = "temporal_edge_matched_participant"
                            break
            if rescue_source is None:
                continue
            source_indices = list(rescue_source)
            source = source_indices[0]
            entity = frame.at[source, "matched_entity_id"]
            role = frame.at[source, "matched_role"]
            relation = frame.at[source, "target_relation"]
            source_confidences = [
                float(value) if pd.notna(value) and np.isfinite(float(value)) else 0.0
                for value in (frame.at[s, "identity_confidence"] for s in source_indices)
            ]
            current_value = frame.at[idx, "identity_confidence"]
            current_confidence = float(current_value) if pd.notna(current_value) and np.isfinite(float(current_value)) else 0.0
            rescued_confidence = float(np.clip(max(current_confidence, 0.78 * min(source_confidences)), 0.0, 1.0))
            frame.at[idx, "matched_entity_id"] = entity
            frame.at[idx, "matched_role"] = role
            frame.at[idx, "target_relation"] = relation
            frame.at[idx, "identity_status"] = rescue_status
            frame.at[idx, "identity_confidence"] = rescued_confidence
            frame.at[idx, "include_in_partner_analysis"] = bool(str(relation) == "partner")
            frame.at[idx, "review_required"] = True
            frame.at[idx, "match_reason"] = rescue_status
            frame.at[idx, "temporal_identity_rescue"] = True
    return frame


def load_participant_association(artifact: ArtifactRecord) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if artifact.kind != ArtifactKind.PARTICIPANT_ASSOCIATION:
        raise ValidationError("Artifact is not a participant association")
    projections = pd.read_csv(find_artifact_file(artifact, names=["participant_head_projection.csv"]), low_memory=False)
    identities = pd.read_csv(find_artifact_file(artifact, names=["face_identity_assignment.csv"]), low_memory=False)
    manifest = _read_json(find_artifact_file(artifact, names=["participant_association_manifest.json"]))
    return projections, identities, manifest


def create_participant_association(
    store: ProjectStore,
    *,
    session_id: str,
    native_trim_artifact_id: str,
    etm_assignment_artifact_id: str,
    wearer_role_id: str,
    calibration_profile_artifact_id: str | None = None,
    calibration_profile_path: str | None = None,
    facemapper_filter_artifact_id: str | None = None,
    target_policy: TargetPolicy = "partner_only",
    alignment_mode: Literal["timegraph", "pretrimmed_shared_start"] = "pretrimmed_shared_start",
    shared_start_offset_sec: float = 0.0,
    face_tolerance_ms: float = 8.0,
    max_pose_gap_sec: float = 0.05,
    max_head_marker_gap_sec: float = 0.05,
    maximum_normalized_center_error: float = 1.25,
    minimum_projected_iou: float = 0.015,
    minimum_assignment_margin: float = 0.12,
    minimum_identity_confidence: float = 0.42,
    output_parquet: bool = False,
    label: str = "Participant association",
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    if target_policy not in {"partner_only", "tracked_participants", "all_detected"}:
        raise ValidationError("target_policy must be partner_only, tracked_participants or all_detected")
    if alignment_mode not in {"timegraph", "pretrimmed_shared_start"}:
        raise ValidationError("alignment_mode must be timegraph or pretrimmed_shared_start")
    if not math.isfinite(shared_start_offset_sec) or abs(float(shared_start_offset_sec)) > 5.0:
        raise ValidationError("shared_start_offset_sec must be within ±5 seconds")
    session = store.project.sessions[session_id]
    trim_artifact = session.artifacts.get(native_trim_artifact_id)
    if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    assignment_artifact = session.artifacts.get(etm_assignment_artifact_id)
    if assignment_artifact is None or assignment_artifact.kind != ArtifactKind.ETM_ROLE_ASSIGNMENT:
        raise ValidationError("Select a resolved ETM-role assignment artifact")
    assignment = load_etm_assignment(assignment_artifact)
    if assignment.get("status") != "resolved":
        raise ValidationError("ETM-role assignment requires review before participant projection")
    assignments = assignment.get("assignments") or {}
    if wearer_role_id not in assignments:
        raise ValidationError(f"Wearer role {wearer_role_id!r} is not present in the ETM assignment")
    ego_assignment = resolve_assignment_for_role(assignment_artifact, wearer_role_id)
    recommended_profile_artifact_id = ego_assignment.get("recommended_calibration_artifact_id")
    recommended_profile_path = ego_assignment.get("recommended_calibration_profile_path")
    if recommended_profile_artifact_id:
        if calibration_profile_artifact_id and calibration_profile_artifact_id != recommended_profile_artifact_id:
            raise ValidationError(
                f"Calibration profile override does not match the paired handoff for {wearer_role_id} → "
                f"{ego_assignment.get('etm_id') or ego_assignment.get('etm_asset_label') or 'assigned ETM'}"
            )
        if calibration_profile_path:
            raise ValidationError(
                f"External calibration-profile override is blocked because the paired handoff already binds {wearer_role_id}"
            )
        calibration_profile_artifact_id = recommended_profile_artifact_id
        calibration_profile_path = None
        profile_resolution_source = "paired_calibration_handoff"
    elif recommended_profile_path:
        if calibration_profile_artifact_id:
            raise ValidationError(
                f"Calibration profile override does not match the paired handoff for {wearer_role_id}"
            )
        if calibration_profile_path and Path(calibration_profile_path).expanduser().resolve() != Path(recommended_profile_path).expanduser().resolve():
            raise ValidationError(
                f"External calibration-profile override does not match the paired handoff for {wearer_role_id}"
            )
        calibration_profile_path = recommended_profile_path
        profile_resolution_source = "paired_calibration_handoff"
    else:
        profile_resolution_source = "explicit_recovery_override"
    profile_file, resolved_profile_artifact_id = _profile_path(
        store,
        session_id,
        profile_artifact_id=calibration_profile_artifact_id,
        profile_path=calibration_profile_path,
    )
    profile = load_calibration_profile(profile_file)

    trim_root, trim_manifest = _trim_root(trim_artifact)
    frame_map = pd.read_csv(trim_root / "native_frame_map.csv", low_memory=False)
    full_clock = pd.read_csv(trim_root / "native_clock_full.csv", low_memory=False)
    required_frame = {"native_frame_index", "native_capture_utc_ns"}
    if not required_frame.issubset(frame_map.columns):
        raise ValidationError("Native-reference trim frame map is incomplete")
    native_reference = trim_manifest.get("native_reference") or {}
    width = int(native_reference.get("width") or 0)
    height = int(native_reference.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValidationError("Native-reference trim has no valid scene resolution")
    recording_id = str(trim_manifest.get("recording_id") or "")
    request = trim_manifest.get("request") or {}
    timeseries_asset_id = request.get("timeseries_asset_id") or trim_artifact.parameters.get("timeseries_asset_id")
    if not timeseries_asset_id or timeseries_asset_id not in session.assets:
        raise ValidationError("Could not resolve the Timeseries source from the Native-reference trim")
    timeseries_asset = session.assets[timeseries_asset_id]
    if alignment_mode == "timegraph" and not timeseries_asset.primary_time_domain_id:
        raise ValidationError("Timeseries asset has no primary Pupil time domain")

    emit(progress, 0.03, "Loading Face Mapper detections and participant geometry")
    scene_timestamps = pd.to_numeric(full_clock["native_capture_utc_ns"], errors="raise").astype("int64").to_numpy()
    filter_artifact = None
    if facemapper_filter_artifact_id:
        filter_artifact = session.artifacts.get(facemapper_filter_artifact_id)
        if filter_artifact is None or filter_artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
            raise ValidationError("Select a FaceMapper QC/filter artifact")
        if native_trim_artifact_id not in filter_artifact.parent_artifact_ids:
            raise ValidationError("FaceMapper filter and participant association must use the same Native-reference trim")
        from oculostudio.services.facemapper_filter import filtered_facemapper_root
        fm_root = filtered_facemapper_root(filter_artifact, variant="analysis_strict")
    else:
        fm_root = trim_root / "face_mapper"
    face_result = load_face_semantics(
        fm_root,
        recording_id=recording_id or None,
        scene_timestamps_ns=scene_timestamps,
        width=width,
        height=height,
        tolerance_ms=face_tolerance_ms,
        left_right_policy="as_exported",
    )
    frame_numeric = pd.to_numeric(frame_map["native_frame_index"], errors="coerce")
    frame_valid_mask = frame_numeric.notna().to_numpy()
    selected_frames = frame_numeric.loc[frame_valid_mask].astype(int).to_numpy()
    selected_timestamps = (
        pd.to_numeric(frame_map.loc[frame_valid_mask, "native_capture_utc_ns"], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    selected_local_sec = None
    if alignment_mode == "pretrimmed_shared_start":
        local_column = (
            "retained_display_start_ns"
            if "retained_display_start_ns" in frame_map.columns
            else "clip_video_pts_ns"
        )
        if local_column not in frame_map.columns:
            raise ValidationError("Native-reference trim frame map has no local-time column")
        selected_local_ns = pd.to_numeric(
            frame_map.loc[frame_valid_mask, local_column], errors="raise"
        ).to_numpy(dtype=np.float64)
        selected_local_sec = selected_local_ns / 1e9 + float(shared_start_offset_sec)

    etm_asset_id = ego_assignment["etm_asset_id"]
    etm_asset = session.assets.get(etm_asset_id)
    if etm_asset is None:
        raise ValidationError("Assigned ETM asset is unavailable")
    if alignment_mode == "timegraph" and not etm_asset.primary_time_domain_id:
        raise ValidationError("Assigned ETM asset has no primary Vicon time domain")
    etm_data = read_mocap(etm_asset.source_path)
    tracker_labels = [str(value) for value in profile["tracker"]["marker_labels"]]
    tracker_template = np.asarray(profile["tracker"]["marker_template_mm"], dtype=float)
    tracker_positions, tracker_valid, tracker_labels, tracker_prefix = resolve_profile_tracker_points(
        tracker_labels, etm_data
    )
    tracker_pose = PoseSeries.from_markers(
        etm_data.time_sec,
        tracker_template,
        tracker_positions,
        tracker_valid,
        min_markers=3,
        max_fit_rmse_mm=12.0,
    )

    graph = TimeGraph.from_session(session) if alignment_mode == "timegraph" else None
    if alignment_mode == "pretrimmed_shared_start":
        assert selected_local_sec is not None
        etm_query_sec = float(etm_data.time_sec[0]) + selected_local_sec
        etm_timeline_source = "pretrimmed_shared_start"
    else:
        try:
            assert graph is not None
            etm_query_ns = graph.convert(
                timeseries_asset.primary_time_domain_id,
                etm_asset.primary_time_domain_id,
                selected_timestamps,
            ).astype(np.int64)
        except TimeGraphError as exc:
            raise ValidationError(
                "No validated TimeGraph path connects the selected Pupil recording to the assigned ETM C3D"
            ) from exc
        etm_query_sec = etm_query_ns.astype(np.float64) / 1e9
        etm_timeline_source = "timegraph"
    tracker_interp = tracker_pose.interpolate(etm_query_sec, max_gap_sec=max_pose_gap_sec)

    intrinsics_raw = CameraIntrinsics.from_dict(profile["camera_intrinsics"])
    intrinsics, intrinsics_transform = _rescaled_intrinsics(intrinsics_raw, width, height)
    T_tracker_from_camera = np.asarray(profile["T_tracker_from_camera"], dtype=float).reshape(4, 4)
    camera_transforms = np.full((len(selected_frames), 4, 4), np.nan, dtype=float)
    for index in np.flatnonzero(tracker_interp.valid):
        camera_transforms[index] = make_transform(
            tracker_interp.rotations[index], tracker_interp.translations[index]
        ) @ T_tracker_from_camera

    role_rows = []
    for role_id, row in sorted(assignments.items()):
        if target_policy == "partner_only" and role_id == wearer_role_id:
            continue
        body_asset = session.assets.get(row.get("body_asset_id"))
        if body_asset is None:
            raise ValidationError(f"Body asset for role {role_id} is unavailable")
        body_data = read_mocap(body_asset.source_path)
        marker_labels = [str(value) for value in row.get("body_head_marker_labels") or []]
        if len(marker_labels) < 3:
            raise ValidationError(f"Role {role_id} has fewer than three head-marker labels in the ETM assignment")
        if alignment_mode == "pretrimmed_shared_start":
            assert selected_local_sec is not None
            body_query_sec = float(body_data.time_sec[0]) + selected_local_sec
            timeline_source = "pretrimmed_shared_start"
        elif body_asset.primary_time_domain_id:
            try:
                assert graph is not None
                body_query_ns = graph.convert(
                    timeseries_asset.primary_time_domain_id,
                    body_asset.primary_time_domain_id,
                    selected_timestamps,
                ).astype(np.int64)
                body_query_sec = body_query_ns.astype(np.float64) / 1e9
                timeline_source = "timegraph"
            except TimeGraphError:
                if (
                    len(body_data.time_sec) == len(etm_data.time_sec)
                    and abs(body_data.frame_rate - etm_data.frame_rate) <= 1e-9
                    and body_data.first_frame == etm_data.first_frame
                    and np.allclose(body_data.time_sec, etm_data.time_sec, atol=1e-9, rtol=0.0)
                ):
                    body_query_sec = etm_query_sec
                    timeline_source = "shared_vicon_timeline"
                else:
                    raise ValidationError(
                        f"No TimeGraph path connects Pupil time to body C3D for {role_id}, and its Vicon timeline "
                        "does not exactly match the assigned ETM"
                    )
        else:
            if (
                len(body_data.time_sec) == len(etm_data.time_sec)
                and abs(body_data.frame_rate - etm_data.frame_rate) <= 1e-9
                and body_data.first_frame == etm_data.first_frame
                and np.allclose(body_data.time_sec, etm_data.time_sec, atol=1e-9, rtol=0.0)
            ):
                body_query_sec = etm_query_sec
                timeline_source = "shared_vicon_timeline"
            else:
                raise ValidationError(f"Body asset for {role_id} has no usable time domain")
        points, valid, marker_gap = _interpolate_markers(
            body_data,
            marker_labels,
            body_query_sec,
            max_gap_sec=max_head_marker_gap_sec,
        )
        role_rows.append({
            "role_id": str(role_id),
            "entity_id": str(role_id),
            "relation_to_wearer": "self" if role_id == wearer_role_id else "partner",
            "body_asset_id": body_asset.id,
            "body_asset_label": body_asset.label,
            "body_subject_prefix": row.get("body_subject_prefix"),
            "marker_labels": marker_labels,
            "points": points,
            "valid": valid,
            "marker_gap": marker_gap,
            "timeline_source": timeline_source,
        })

    emit(progress, 0.20, "Projecting tracked participant heads into the ego camera")
    projection_rows: list[dict[str, Any]] = []
    projections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame_ordinal, native_frame in enumerate(selected_frames):
        token.raise_if_cancelled()
        camera_valid = bool(tracker_interp.valid[frame_ordinal] and np.isfinite(camera_transforms[frame_ordinal]).all())
        for role in role_rows:
            geometry = _head_geometry(role["points"][frame_ordinal], role["valid"][frame_ordinal], role["marker_labels"])
            row: dict[str, Any] = {
                "native_frame_index": int(native_frame),
                "source_frame_ordinal_in_window": int(frame_ordinal),
                "native_capture_utc_ns": int(selected_timestamps[frame_ordinal]),
                "role_id": role["role_id"],
                "entity_id": role["entity_id"],
                "relation_to_wearer": role["relation_to_wearer"],
                "body_asset_id": role["body_asset_id"],
                "body_subject_prefix": role["body_subject_prefix"],
                "timeline_source": role["timeline_source"],
                "camera_timeline_source": etm_timeline_source,
                "camera_pose_valid": camera_valid,
                "camera_pose_interpolated": bool(tracker_interp.interpolated[frame_ordinal]) if camera_valid else False,
                "camera_pose_gap_sec": float(tracker_interp.interpolation_gap_sec[frame_ordinal]) if camera_valid and np.isfinite(tracker_interp.interpolation_gap_sec[frame_ordinal]) else None,
                "head_pose_valid": geometry is not None,
                "projection_valid": False,
                "projection_confidence": 0.0,
            }
            if geometry is not None and camera_valid:
                projected = _project_head(
                    geometry,
                    camera_transforms[frame_ordinal],
                    intrinsics,
                    image_width=width,
                    image_height=height,
                )
                marker_fraction = min(1.0, geometry["marker_count"] / max(4.0, len(role["marker_labels"])))
                gap_values = role["marker_gap"][frame_ordinal]
                finite_gap = gap_values[np.isfinite(gap_values)]
                gap_penalty = 1.0 if not len(finite_gap) else float(np.clip(1.0 - np.nanmax(finite_gap) / max(max_head_marker_gap_sec, 1e-9) * 0.35, 0.55, 1.0))
                camera_penalty = 0.85 if tracker_interp.interpolated[frame_ordinal] else 1.0
                in_view = bool(projected["in_camera_view"])
                confidence = float(np.clip(marker_fraction * gap_penalty * camera_penalty * (1.0 if in_view else 0.45), 0.0, 1.0))
                row.update(projected)
                row["projection_valid"] = bool(projected["projected_head_depth_m"] is not None and projected["projected_head_depth_m"] > 0)
                row["projection_confidence"] = confidence
            projection_rows.append(row)
            projections_by_frame[int(native_frame)].append(row)
        if frame_ordinal % 200 == 0:
            emit(progress, 0.20 + 0.34 * (frame_ordinal + 1) / max(1, len(selected_frames)), f"Projecting heads ({frame_ordinal + 1}/{len(selected_frames)})")

    emit(progress, 0.57, "Associating Face Mapper detections with projected participants")
    identity_rows: list[dict[str, Any]] = []
    previous_track_by_role: dict[str, int] = {}
    for frame_ordinal, native_frame in enumerate(selected_frames):
        token.raise_if_cancelled()
        faces = face_result.instances_by_frame.get(int(native_frame), [])
        associated, previous_track_by_role = _associate_frame(
            projections_by_frame.get(int(native_frame), []),
            faces,
            previous_track_by_role=previous_track_by_role,
            maximum_normalized_center_error=maximum_normalized_center_error,
            minimum_projected_iou=minimum_projected_iou,
            minimum_assignment_margin=minimum_assignment_margin,
            minimum_identity_confidence=minimum_identity_confidence,
        )
        for result in associated:
            face: FaceInstance = result["face"]
            projection = result.get("projection") or {}
            metrics = result.get("metrics") or {}
            identity_rows.append({
                "native_frame_index": int(native_frame),
                "source_frame_ordinal_in_window": int(frame_ordinal),
                "scene_timestamp_ns": int(face.scene_timestamp_ns),
                "face_source_row_index": int(face.source_row_index),
                "face_track_id": int(face.track_id),
                "face_box_x1": float(face.x1),
                "face_box_y1": float(face.y1),
                "face_box_x2": float(face.x2),
                "face_box_y2": float(face.y2),
                "face_detection_confidence": float(face.confidence),
                "matched_entity_id": result["matched_entity_id"],
                "matched_role": result["matched_role"],
                "candidate_entity_id": result.get("candidate_entity_id"),
                "candidate_role": result.get("candidate_role"),
                "candidate_relation": result.get("candidate_relation"),
                "target_relation": result["target_relation"],
                "identity_status": result["identity_status"],
                "identity_confidence": float(result["identity_confidence"]),
                "include_in_partner_analysis": bool(result["include_in_partner_analysis"]),
                "review_required": bool(result["review_required"]),
                "match_reason": result.get("match_reason"),
                "projected_head_x_px": projection.get("projected_head_x_px"),
                "projected_head_y_px": projection.get("projected_head_y_px"),
                "projected_head_depth_m": projection.get("projected_head_depth_m"),
                "projected_bbox_x1": projection.get("projected_bbox_x1"),
                "projected_bbox_y1": projection.get("projected_bbox_y1"),
                "projected_bbox_x2": projection.get("projected_bbox_x2"),
                "projected_bbox_y2": projection.get("projected_bbox_y2"),
                "projected_anatomical_left_head_x_px": projection.get("projected_anatomical_left_head_x_px"),
                "projected_anatomical_left_head_y_px": projection.get("projected_anatomical_left_head_y_px"),
                "projected_anatomical_right_head_x_px": projection.get("projected_anatomical_right_head_x_px"),
                "projected_anatomical_right_head_y_px": projection.get("projected_anatomical_right_head_y_px"),
                "head_facing_score": projection.get("head_facing_score"),
                "center_error_px": metrics.get("center_error_px"),
                "normalized_center_error": metrics.get("normalized_center_error"),
                "projected_box_iou": metrics.get("projected_box_iou"),
                "bbox_intersection_area_px2": metrics.get("bbox_intersection_area_px2"),
                "bbox_overlap_present": metrics.get("bbox_overlap_present"),
                "projected_box_overlap_fraction": metrics.get("projected_box_overlap_fraction"),
                "face_box_overlap_fraction": metrics.get("face_box_overlap_fraction"),
                "overlap_unique_for_projection": metrics.get("overlap_unique_for_projection"),
                "overlap_unique_for_face": metrics.get("overlap_unique_for_face"),
                "overlap_distinct_face_count": metrics.get("overlap_distinct_face_count"),
                "overlap_distinct_projection_count": metrics.get("overlap_distinct_projection_count"),
                "area_log_error": metrics.get("area_log_error"),
                "assignment_cost": metrics.get("cost"),
                "assignment_margin": metrics.get("assignment_margin"),
                "temporal_track_consistency": metrics.get("temporal_track_consistency"),
                "projected_center_inside_face": metrics.get("projected_center_inside_face"),
                "face_center_inside_projection": metrics.get("face_center_inside_projection"),
                "geometry_score": metrics.get("geometry_score"),
                "geometry_strong": metrics.get("geometry_strong"),
                "duplicate_iou": metrics.get("duplicate_iou"),
                "duplicate_center_distance": metrics.get("duplicate_center_distance"),
            })
        if frame_ordinal % 200 == 0:
            emit(progress, 0.57 + 0.25 * (frame_ordinal + 1) / max(1, len(selected_frames)), f"Associating faces ({frame_ordinal + 1}/{len(selected_frames)})")

    projections = pd.DataFrame(projection_rows)
    partner_projection_mask = pd.Series(False, index=projections.index, dtype=bool)
    if len(projections):
        partner_projection_mask = (
            projections.get("relation_to_wearer", pd.Series("", index=projections.index)).astype(str).eq("partner")
            & projections.get("projection_valid", pd.Series(False, index=projections.index)).fillna(False).astype(bool)
            & projections.get("in_camera_view", pd.Series(False, index=projections.index)).fillna(False).astype(bool)
        )
    partner_projection_in_view_count = int(partner_projection_mask.sum())
    partner_projection_frame_count = int(
        projections.loc[partner_projection_mask, "native_frame_index"].nunique()
    ) if partner_projection_in_view_count else 0
    if target_policy == "partner_only" and partner_projection_in_view_count == 0:
        raise ValidationError(
            f"No valid in-view partner head projections were produced for {wearer_role_id} → "
            f"{ego_assignment.get('etm_id') or ego_assignment.get('etm_asset_label') or 'assigned ETM'}. "
            "The participant-association artifact was not published. Check the role-locked calibration profile, "
            "ETM/body assignment, shared-start trim, and partner head markers."
        )
    identities = _stabilize_identity_tracks(pd.DataFrame(identity_rows))
    artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "participant_association") / (".partial_" + artifact_id)
    final = store.derived_dir(session_id, "participant_association") / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    parquet_files: list[str] = []
    parquet_warning = None
    try:
        projections.to_csv(staging / "participant_head_projection.csv", index=False)
        identities.to_csv(staging / "face_identity_assignment.csv", index=False)
        if output_parquet:
            try:
                projections.to_parquet(staging / "participant_head_projection.parquet", index=False)
                identities.to_parquet(staging / "face_identity_assignment.parquet", index=False)
                parquet_files = ["participant_head_projection.parquet", "face_identity_assignment.parquet"]
            except Exception as exc:
                parquet_warning = f"Parquet output was requested but could not be written: {exc}"
                for path in staging.glob("*.parquet"):
                    path.unlink(missing_ok=True)
        status_counts = dict(pd.Series(identities.get("identity_status", pd.Series(dtype=str))).value_counts())
        matched = identities["identity_status"].astype(str).isin(MATCHED_IDENTITY_STATUSES) if len(identities) else pd.Series(dtype=bool)
        partner = identities["include_in_partner_analysis"].astype(bool) if len(identities) else pd.Series(dtype=bool)
        qc = {
            "frame_count": int(len(selected_frames)),
            "alignment_mode": alignment_mode,
            "shared_start_offset_sec": float(shared_start_offset_sec),
            "etm_timeline_source": etm_timeline_source,
            "projected_role_count": int(len(role_rows)),
            "projection_valid_fraction": float(projections["projection_valid"].mean()) if len(projections) else 0.0,
            "partner_projection_in_view_count": partner_projection_in_view_count,
            "partner_projection_in_view_frame_count": partner_projection_frame_count,
            "partner_projection_in_view_fraction": float(partner_projection_frame_count / max(1, len(selected_frames))),
            "resolved_wearer_role_id": wearer_role_id,
            "resolved_etm_id": ego_assignment.get("etm_id"),
            "resolved_calibration_profile_artifact_id": resolved_profile_artifact_id or calibration_profile_artifact_id,
            "profile_resolution_source": profile_resolution_source,
            "face_detection_count": int(len(identities)),
            "matched_participant_count": int(matched.sum()) if len(identities) else 0,
            "matched_participant_fraction": float(matched.mean()) if len(identities) else 0.0,
            "partner_face_count": int(partner.sum()) if len(identities) else 0,
            "overlap_locked_participant_count": int(identities["identity_status"].eq("overlap_locked_participant").sum()) if len(identities) else 0,
            "bbox_overlap_candidate_count": int(identities.get("bbox_overlap_present", pd.Series(False, index=identities.index)).map(lambda value: bool(value) if pd.notna(value) else False).sum()) if len(identities) else 0,
            "other_person_count": int(identities["target_relation"].eq("other_person").sum()) if len(identities) else 0,
            "partner_candidate_count": int(identities["target_relation"].isin(["ambiguous_partner", "partner_candidate"]).sum()) if len(identities) else 0,
            "duplicate_identity_inheritance_count": int(identities["identity_status"].astype(str).str.startswith("duplicate_").sum()) if len(identities) else 0,
            "temporal_identity_rescue_count": int(identities.get("temporal_identity_rescue", pd.Series(False, index=identities.index)).astype(bool).sum()) if len(identities) else 0,
            "review_required_fraction": float(identities["review_required"].astype(bool).mean()) if len(identities) else 0.0,
            "identity_status_counts": status_counts,
            "face_mapper_diagnostics": face_result.diagnostics,
            "parquet_written": bool(parquet_files),
            "parquet_warning": parquet_warning,
        }
        qc = _json_safe(qc)
        manifest = {
            "schema_version": "oculostudio-participant-association-1.3",
            "identity_engine": PARTICIPANT_ASSOCIATION_ENGINE,
            "artifact_id": artifact_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "etm_assignment_artifact_id": etm_assignment_artifact_id,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "wearer_role_id": wearer_role_id,
            "recording_id": recording_id,
            "timeseries_asset_id": timeseries_asset_id,
            "etm_asset_id": etm_asset_id,
            "calibration_profile_path": str(profile_file),
            "calibration_profile_artifact_id": resolved_profile_artifact_id or calibration_profile_artifact_id,
            "profile_resolution_source": profile_resolution_source,
            "matched_tracker_marker_labels": tracker_labels,
            "matched_tracker_prefix": tracker_prefix,
            "target_policy": target_policy,
            "intrinsics_transform": intrinsics_transform,
            "parameters": {
                "alignment_mode": alignment_mode,
                "shared_start_offset_sec": float(shared_start_offset_sec),
                "face_tolerance_ms": face_tolerance_ms,
                "max_pose_gap_sec": max_pose_gap_sec,
                "max_head_marker_gap_sec": max_head_marker_gap_sec,
                "maximum_normalized_center_error": maximum_normalized_center_error,
                "minimum_projected_iou": minimum_projected_iou,
                "minimum_assignment_margin": minimum_assignment_margin,
                "minimum_identity_confidence": minimum_identity_confidence,
                "identity_gate": "positive_projected_box_face_box_intersection",
                "aoi_geometry_source": "facemapper_face_box_only",
                "multiple_overlap_policy": "review_distinct_faces_choose_duplicate_cluster_as_one",
            },
            "outputs": {
                "participant_head_projection_csv": "participant_head_projection.csv",
                "face_identity_assignment_csv": "face_identity_assignment.csv",
                "parquet": parquet_files,
            },
            "qc": qc,
            "created_at": utc_now(),
        }
        atomic_write_json(staging / "participant_association_manifest.json", _json_safe(manifest))
        atomic_write_json(staging / "qc_metrics.json", _json_safe(qc))
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    source_assets = list(dict.fromkeys([
        *trim_artifact.source_asset_ids,
        *(row["body_asset_id"] for row in role_rows),
        etm_asset_id,
    ]))
    parents = [native_trim_artifact_id, etm_assignment_artifact_id]
    profile_parent = resolved_profile_artifact_id or calibration_profile_artifact_id
    if profile_parent:
        parents.append(profile_parent)
    if facemapper_filter_artifact_id:
        parents.append(facemapper_filter_artifact_id)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.PARTICIPANT_ASSOCIATION,
        label=label,
        output_root=final,
        source_asset_ids=source_assets,
        parent_artifact_ids=list(dict.fromkeys(parents)),
        parameters={
            "wearer_role_id": wearer_role_id,
            "target_policy": target_policy,
            "participant_identity_engine": PARTICIPANT_ASSOCIATION_ENGINE,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "alignment_mode": alignment_mode,
            "shared_start_offset_sec": float(shared_start_offset_sec),
            "face_tolerance_ms": face_tolerance_ms,
            "resolved_etm_id": ego_assignment.get("etm_id"),
            "resolved_calibration_profile_artifact_id": resolved_profile_artifact_id or calibration_profile_artifact_id,
            "profile_resolution_source": profile_resolution_source,
            "native_trim_artifact_id": native_trim_artifact_id,
            "output_parquet": output_parquet,
        },
        qc=qc,
        provenance={
            "native_trim_artifact_id": native_trim_artifact_id,
            "etm_assignment_artifact_id": etm_assignment_artifact_id,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "calibration_profile_artifact_id": profile_parent,
            "timeseries_asset_id": timeseries_asset_id,
            "recording_id": recording_id,
            "participant_identity_engine": PARTICIPANT_ASSOCIATION_ENGINE,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Participant projection and face identity association complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "participant_head_projection": str(final / "participant_head_projection.csv"),
        "face_identity_assignment": str(final / "face_identity_assignment.csv"),
        "manifest": str(final / "participant_association_manifest.json"),
    }
