from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd

from oculostudio.domain.models import ArtifactKind, ArtifactRecord
from oculostudio.adapters.mocap import MocapData, read_mocap
from oculostudio.domain.geometry import PoseSeries
from oculostudio.services.calibration import load_calibration_profile
from oculostudio.errors import ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.etm_role_assignment import _centre_series, load_etm_assignment
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json

ETM_PREVIEW_SCHEMA = "oculostudio-etm-motion-preview-1.0"
GAZE_PREVIEW_SCHEMA = "oculostudio-gaze-ray-preview-1.1"
DYADIC_GAZE_PREVIEW_SCHEMA = "oculostudio-dyadic-gaze-preview-1.0"
MAX_ETM_PREVIEW_SAMPLES = 900
MAX_GAZE_PREVIEW_SAMPLES = 1200
MAX_DYADIC_GAZE_PREVIEW_SAMPLES = 1200


def _preview_cache_path(store: ProjectStore, artifact_id: str, name: str) -> Path:
    root = store.cache_dir() / "previews" / artifact_id
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _rounded_point(point: np.ndarray | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if point is None:
        return None
    values = np.asarray(point, dtype=float).reshape(-1)
    if len(values) != 3 or not np.isfinite(values).all():
        return None
    return [round(float(value), 3) for value in values]


def _bounds(points: list[list[float]]) -> dict[str, list[float]]:
    if not points:
        return {
            "min": [-500.0, -500.0, -500.0],
            "max": [500.0, 500.0, 500.0],
            "centre": [0.0, 0.0, 0.0],
            "span": [1000.0, 1000.0, 1000.0],
        }
    array = np.asarray(points, dtype=float)
    minimum = np.nanmin(array, axis=0)
    maximum = np.nanmax(array, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = np.maximum(span * 0.08, 40.0)
    minimum -= padding
    maximum += padding
    span = maximum - minimum
    centre = (minimum + maximum) / 2.0
    return {
        "min": [round(float(value), 3) for value in minimum],
        "max": [round(float(value), 3) for value in maximum],
        "centre": [round(float(value), 3) for value in centre],
        "span": [round(float(value), 3) for value in span],
    }


def _sample_frames(start: int, end: int, maximum_samples: int) -> np.ndarray:
    count = max(0, int(end) - int(start) + 1)
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    sample_count = min(maximum_samples, count)
    offsets = np.unique(np.rint(np.linspace(0, count - 1, sample_count)).astype(np.int64))
    return offsets + int(start)


def _series_point(series, frame_number: int) -> list[float] | None:
    index = int(frame_number) - int(series.first_frame)
    if index < 0 or index >= len(series.frame_numbers):
        return None
    if not bool(series.valid[index]):
        return None
    return _rounded_point(series.centres_mm[index])


def _build_etm_preview(
    store: ProjectStore,
    session_id: str,
    artifact: ArtifactRecord,
    *,
    maximum_samples: int = MAX_ETM_PREVIEW_SAMPLES,
) -> dict[str, Any]:
    if artifact.kind != ArtifactKind.ETM_ROLE_ASSIGNMENT:
        raise ValidationError("Selected artifact is not an ETM role assignment")
    session = store.project.sessions[session_id]
    assignment = load_etm_assignment(artifact)
    role_rows = assignment.get("role_assets") or []
    etm_rows = assignment.get("etm_assets") or []
    if not role_rows or not etm_rows:
        raise ValidationError("ETM assignment does not contain role/ETM source metadata")

    roles = {}
    for row in role_rows:
        asset_id = str(row.get("asset_id") or "")
        asset = session.assets.get(asset_id)
        if asset is None:
            raise ValidationError(f"Body asset required by the ETM assignment is unavailable: {asset_id}")
        role_id = str(row.get("role_id") or asset.participant_id or asset.label)
        roles[role_id] = _centre_series(
            asset.id,
            asset.label,
            role_id,
            asset.source_path,
            mode="body",
            marker_prefix=row.get("subject_prefix"),
        )

    etms = {}
    for row in etm_rows:
        asset_id = str(row.get("asset_id") or "")
        asset = session.assets.get(asset_id)
        if asset is None:
            raise ValidationError(f"ETM asset required by the assignment is unavailable: {asset_id}")
        etm_id = str(row.get("etm_id") or asset.label)
        etms[etm_id] = _centre_series(
            asset.id,
            asset.label,
            etm_id,
            asset.source_path,
            mode="etm",
            marker_prefix=row.get("tracker_prefix"),
        )

    all_series = [*roles.values(), *etms.values()]
    start = max(int(series.frame_numbers[0]) for series in all_series)
    end = min(int(series.frame_numbers[-1]) for series in all_series)
    if end < start:
        raise ValidationError("Body and ETM C3Ds have no shared Vicon frame interval for preview")
    frames = _sample_frames(start, end, maximum_samples)
    frame_rate = float(assignment.get("frame_rate_hz") or all_series[0].frame_rate)
    assignments = assignment.get("assignments") or {}
    assignment_by_etm = assignment.get("assignment_by_etm") or {}

    samples: list[dict[str, Any]] = []
    all_points: list[list[float]] = []
    first_series = all_series[0]
    for frame_number in frames.tolist():
        role_points = {role_id: _series_point(series, frame_number) for role_id, series in roles.items()}
        etm_points = {etm_id: _series_point(series, frame_number) for etm_id, series in etms.items()}
        for point in [*role_points.values(), *etm_points.values()]:
            if point is not None:
                all_points.append(point)
        distances = {}
        for role_id, row in assignments.items():
            etm_id = str(row.get("etm_id") or "")
            role_point = role_points.get(role_id)
            etm_point = etm_points.get(etm_id)
            if role_point is not None and etm_point is not None:
                distances[role_id] = round(
                    float(np.linalg.norm(np.asarray(role_point) - np.asarray(etm_point))), 3
                )
            else:
                distances[role_id] = None
        time_index = frame_number - int(first_series.first_frame)
        if 0 <= time_index < len(first_series.time_sec):
            time_sec = float(first_series.time_sec[time_index])
        else:
            time_sec = float(frame_number - start) / frame_rate
        samples.append(
            {
                "frame": int(frame_number),
                "time_sec": round(time_sec, 6),
                "roles": role_points,
                "etms": etm_points,
                "assigned_distances_mm": distances,
            }
        )

    role_metadata = {}
    for role_id, series in roles.items():
        row = assignments.get(role_id) or {}
        role_metadata[role_id] = {
            "asset_id": series.asset_id,
            "asset_label": series.label,
            "subject_prefix": series.subject_prefix,
            "assigned_etm_id": row.get("etm_id"),
            "median_head_distance_mm": row.get("median_head_distance_mm"),
            "alternative_head_distance_mm": row.get("alternative_head_distance_mm"),
            "distance_margin_mm": row.get("distance_margin_mm"),
            "confidence": row.get("confidence"),
        }
    etm_metadata = {}
    for etm_id, series in etms.items():
        row = assignment_by_etm.get(etm_id) or {}
        etm_metadata[etm_id] = {
            "asset_id": series.asset_id,
            "asset_label": series.label,
            "tracker_prefix": series.subject_prefix,
            "assigned_role_id": row.get("role_id"),
            "marker_valid_fraction": series.marker_valid_fraction,
        }

    distance_matrix = {}
    for etm_id, role_map in (assignment.get("distance_statistics_mm") or {}).items():
        distance_matrix[etm_id] = {
            role_id: row.get("median_mm") for role_id, row in (role_map or {}).items()
        }

    return {
        "schema_version": ETM_PREVIEW_SCHEMA,
        "artifact_id": artifact.id,
        "artifact_label": artifact.label,
        "status": assignment.get("status"),
        "coordinate_system": "Vicon world coordinates (millimetres)",
        "frame_rate_hz": frame_rate,
        "source_frame_start": int(start),
        "source_frame_end": int(end),
        "source_frame_count": int(end - start + 1),
        "sample_count": len(samples),
        "maximum_preview_samples": int(maximum_samples),
        "roles": role_metadata,
        "etms": etm_metadata,
        "distance_matrix_median_mm": distance_matrix,
        "bounds_mm": _bounds(all_points),
        "samples": samples,
    }


def get_etm_assignment_preview(
    store: ProjectStore,
    session_id: str,
    artifact_id: str,
    *,
    maximum_samples: int = MAX_ETM_PREVIEW_SAMPLES,
) -> dict[str, Any]:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    artifact = session.artifacts.get(artifact_id)
    if artifact is None:
        raise ValidationError("ETM assignment artifact was not found")
    cache = _preview_cache_path(store, artifact_id, "etm_assignment_preview_v1.json")
    if cache.exists():
        try:
            data = json.loads(cache.read_text("utf-8"))
            if data.get("schema_version") == ETM_PREVIEW_SCHEMA:
                return data
        except Exception:
            cache.unlink(missing_ok=True)
    data = _build_etm_preview(store, session_id, artifact, maximum_samples=maximum_samples)
    atomic_write_json(cache, data)
    return data


def _bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"}).to_numpy(dtype=bool)


def _build_gaze_preview(
    artifact: ArtifactRecord,
    *,
    maximum_samples: int = MAX_GAZE_PREVIEW_SAMPLES,
    ray_length_mm: float = 1500.0,
) -> dict[str, Any]:
    if artifact.kind != ArtifactKind.GAZE_RAYS:
        raise ValidationError("Selected artifact is not a Gaze 3D ray artifact")
    csv_path = find_artifact_file(artifact, names=("gaze_rays_world.csv",))
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = [
        "gaze3d_origin_x_mm",
        "gaze3d_origin_y_mm",
        "gaze3d_origin_z_mm",
        "gaze3d_direction_x",
        "gaze3d_direction_y",
        "gaze3d_direction_z",
    ]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValidationError(f"Gaze ray table is missing preview columns: {missing}")
    optional = [
        "gaze3d_ray_valid",
        "gaze3d_input_row_index",
        "gaze3d_c3d_time_sec",
        "gaze3d_tracker_local_time_sec",
        "gaze3d_c3d_frame_float",
        "gaze3d_tracker_pose_interpolated",
        "gaze3d_tracker_fit_rmse_mm",
    ]
    columns = required + [column for column in optional if column in header]
    frame = pd.read_csv(csv_path, usecols=columns)
    origin = frame[["gaze3d_origin_x_mm", "gaze3d_origin_y_mm", "gaze3d_origin_z_mm"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    direction = frame[["gaze3d_direction_x", "gaze3d_direction_y", "gaze3d_direction_z"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(origin).all(axis=1) & np.isfinite(direction).all(axis=1)
    if "gaze3d_ray_valid" in frame:
        valid &= _bool_series(frame["gaze3d_ray_valid"])
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return {
            "schema_version": GAZE_PREVIEW_SCHEMA,
            "artifact_id": artifact.id,
            "artifact_label": artifact.label,
            "coordinate_system": "Vicon world coordinates (millimetres)",
            "source_valid_ray_count": 0,
            "sample_count": 0,
            "ray_length_mm": float(ray_length_mm),
            "bounds_mm": _bounds([]),
            "samples": [],
        }
    sample_count = min(maximum_samples, len(valid_indices))
    selection = np.unique(
        np.rint(np.linspace(0, len(valid_indices) - 1, sample_count)).astype(np.int64)
    )
    selected = valid_indices[selection]
    samples = []
    points: list[list[float]] = []
    for index in selected.tolist():
        o = origin[index]
        d = direction[index]
        norm = float(np.linalg.norm(d))
        if not np.isfinite(norm) or norm <= 1e-12:
            continue
        d = d / norm
        endpoint = o + d * float(ray_length_mm)
        origin_row = _rounded_point(o)
        direction_row = _rounded_point(d)
        endpoint_row = _rounded_point(endpoint)
        if origin_row is None or direction_row is None or endpoint_row is None:
            continue
        points.extend([origin_row, endpoint_row])
        sample = {
            "table_row": int(index),
            "input_row_index": int(frame.iloc[index]["gaze3d_input_row_index"])
            if "gaze3d_input_row_index" in frame and pd.notna(frame.iloc[index]["gaze3d_input_row_index"])
            else int(index),
            "c3d_time_sec": round(float(frame.iloc[index]["gaze3d_c3d_time_sec"]), 6)
            if "gaze3d_c3d_time_sec" in frame and pd.notna(frame.iloc[index]["gaze3d_c3d_time_sec"])
            else None,
            "local_time_sec": round(float(frame.iloc[index]["gaze3d_tracker_local_time_sec"]), 6)
            if "gaze3d_tracker_local_time_sec" in frame and pd.notna(frame.iloc[index]["gaze3d_tracker_local_time_sec"])
            else (
                round(float(frame.iloc[index]["gaze3d_c3d_time_sec"]), 6)
                if "gaze3d_c3d_time_sec" in frame and pd.notna(frame.iloc[index]["gaze3d_c3d_time_sec"])
                else None
            ),
            "c3d_frame_float": round(float(frame.iloc[index]["gaze3d_c3d_frame_float"]), 4)
            if "gaze3d_c3d_frame_float" in frame and pd.notna(frame.iloc[index]["gaze3d_c3d_frame_float"])
            else None,
            "origin_mm": origin_row,
            "direction": direction_row,
            "endpoint_mm": endpoint_row,
            "tracker_pose_interpolated": bool(_bool_series(frame.iloc[[index]]["gaze3d_tracker_pose_interpolated"])[0])
            if "gaze3d_tracker_pose_interpolated" in frame
            else None,
            "tracker_fit_rmse_mm": round(float(frame.iloc[index]["gaze3d_tracker_fit_rmse_mm"]), 4)
            if "gaze3d_tracker_fit_rmse_mm" in frame and pd.notna(frame.iloc[index]["gaze3d_tracker_fit_rmse_mm"])
            else None,
        }
        samples.append(sample)
    return {
        "schema_version": GAZE_PREVIEW_SCHEMA,
        "artifact_id": artifact.id,
        "artifact_label": artifact.label,
        "coordinate_system": "Vicon world coordinates (millimetres)",
        "wearer_role_id": artifact.qc.get("wearer_role_id"),
        "assignment_artifact_id": artifact.qc.get("etm_assignment_artifact_id") or artifact.parameters.get("etm_assignment_artifact_id"),
        "alignment_mode": artifact.qc.get("alignment_mode") or artifact.parameters.get("alignment_mode"),
        "native_trim_artifact_id": artifact.qc.get("native_trim_artifact_id") or artifact.parameters.get("native_trim_artifact_id"),
        "calibration_hardware_id": artifact.qc.get("calibration_hardware_id"),
        "calibration_profile_id": artifact.qc.get("calibration_profile_id"),
        "source_valid_ray_count": int(len(valid_indices)),
        "sample_count": len(samples),
        "maximum_preview_samples": int(maximum_samples),
        "ray_length_mm": float(ray_length_mm),
        "bounds_mm": _bounds(points),
        "samples": samples,
    }


def get_gaze_rays_preview(
    store: ProjectStore,
    session_id: str,
    artifact_id: str,
    *,
    maximum_samples: int = MAX_GAZE_PREVIEW_SAMPLES,
) -> dict[str, Any]:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    artifact = session.artifacts.get(artifact_id)
    if artifact is None:
        raise ValidationError("Gaze 3D artifact was not found")
    cache = _preview_cache_path(store, artifact_id, "gaze_rays_preview_v1.json")
    if cache.exists():
        try:
            data = json.loads(cache.read_text("utf-8"))
            if data.get("schema_version") == GAZE_PREVIEW_SCHEMA:
                return data
        except Exception:
            cache.unlink(missing_ok=True)
    data = _build_gaze_preview(artifact, maximum_samples=maximum_samples)
    atomic_write_json(cache, data)
    return data


def _gaze_artifact_role(artifact: ArtifactRecord) -> str:
    return str(
        artifact.qc.get("wearer_role_id")
        or artifact.parameters.get("wearer_role_id")
        or ""
    ).strip().upper()


def _read_dyadic_gaze_table(artifact: ArtifactRecord) -> dict[str, Any]:
    if artifact.kind != ArtifactKind.GAZE_RAYS:
        raise ValidationError("Dyadic preview requires two Gaze 3D ray artifacts")
    csv_path = find_artifact_file(artifact, names=("gaze_rays_world.csv",))
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = [
        "gaze3d_origin_x_mm", "gaze3d_origin_y_mm", "gaze3d_origin_z_mm",
        "gaze3d_direction_x", "gaze3d_direction_y", "gaze3d_direction_z",
    ]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValidationError(f"Gaze ray table is missing dyadic-preview columns: {missing}")
    time_column = next((column for column in (
        "gaze3d_tracker_local_time_sec",
        "clip_time_sec",
        "gaze3d_c3d_time_sec",
    ) if column in header), None)
    if time_column is None:
        raise ValidationError("Gaze ray table has no compatible local-time column")
    optional = [
        "gaze3d_ray_valid", "gaze3d_input_row_index", "gaze3d_c3d_time_sec",
        "gaze3d_tracker_local_time_sec", "gaze3d_c3d_frame_float",
        "gaze3d_tracker_pose_interpolated", "gaze3d_tracker_fit_rmse_mm",
    ]
    columns = list(dict.fromkeys(required + [time_column] + [column for column in optional if column in header]))
    frame = pd.read_csv(csv_path, usecols=columns, low_memory=False)
    times = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
    origin = frame[["gaze3d_origin_x_mm", "gaze3d_origin_y_mm", "gaze3d_origin_z_mm"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    direction = frame[["gaze3d_direction_x", "gaze3d_direction_y", "gaze3d_direction_z"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(origin).all(axis=1) & np.isfinite(direction).all(axis=1)
    if "gaze3d_ray_valid" in frame:
        valid &= _bool_series(frame["gaze3d_ray_valid"])
    finite_times = times[np.isfinite(times)]
    valid_indices = np.flatnonzero(valid)
    valid_times = times[valid_indices]
    order = np.argsort(valid_times, kind="stable") if len(valid_times) else np.asarray([], dtype=np.int64)
    valid_indices = valid_indices[order]
    valid_times = valid_times[order]
    return {
        "artifact": artifact,
        "frame": frame,
        "time_column": time_column,
        "times": times,
        "finite_times": finite_times,
        "valid_indices": valid_indices,
        "valid_times": valid_times,
        "origin": origin,
        "direction": direction,
        "role": _gaze_artifact_role(artifact),
        "assignment_artifact_id": artifact.qc.get("etm_assignment_artifact_id") or artifact.parameters.get("etm_assignment_artifact_id"),
        "alignment_mode": artifact.qc.get("alignment_mode") or artifact.parameters.get("alignment_mode"),
        "native_trim_artifact_id": artifact.qc.get("native_trim_artifact_id") or artifact.parameters.get("native_trim_artifact_id"),
        "calibration_hardware_id": artifact.qc.get("calibration_hardware_id"),
        "calibration_profile_id": artifact.qc.get("calibration_profile_id"),
    }


def _median_positive_interval(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) < 2:
        return None
    delta = np.diff(unique)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if not len(delta):
        return None
    return float(np.median(delta))


def _nearest_valid_row(data: dict[str, Any], target_time: float, tolerance_sec: float, ray_length_mm: float) -> dict[str, Any]:
    valid_times = data["valid_times"]
    valid_indices = data["valid_indices"]
    if not len(valid_times):
        return {"available": False, "reason": "no valid rays in source artifact"}
    right = int(np.searchsorted(valid_times, target_time, side="left"))
    candidates = [index for index in (right - 1, right) if 0 <= index < len(valid_times)]
    if not candidates:
        return {"available": False, "reason": "no sample near common time"}
    nearest = min(candidates, key=lambda index: abs(float(valid_times[index]) - target_time))
    source_index = int(valid_indices[nearest])
    sample_time = float(valid_times[nearest])
    delta_sec = abs(sample_time - target_time)
    if delta_sec > tolerance_sec:
        return {
            "available": False,
            "reason": f"no valid sample within {tolerance_sec * 1000.0:.1f} ms",
            "nearest_sample_time_sec": round(sample_time, 6),
            "delta_ms": round(delta_sec * 1000.0, 3),
        }
    direction = np.asarray(data["direction"][source_index], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        return {"available": False, "reason": "invalid direction vector"}
    direction /= norm
    origin = np.asarray(data["origin"][source_index], dtype=float)
    endpoint = origin + direction * float(ray_length_mm)
    frame = data["frame"]
    row = frame.iloc[source_index]
    result = {
        "available": True,
        "sample_time_sec": round(sample_time, 6),
        "delta_ms": round((sample_time - target_time) * 1000.0, 3),
        "table_row": source_index,
        "input_row_index": int(row["gaze3d_input_row_index"])
        if "gaze3d_input_row_index" in frame and pd.notna(row.get("gaze3d_input_row_index"))
        else source_index,
        "c3d_time_sec": round(float(row["gaze3d_c3d_time_sec"]), 6)
        if "gaze3d_c3d_time_sec" in frame and pd.notna(row.get("gaze3d_c3d_time_sec"))
        else None,
        "c3d_frame_float": round(float(row["gaze3d_c3d_frame_float"]), 4)
        if "gaze3d_c3d_frame_float" in frame and pd.notna(row.get("gaze3d_c3d_frame_float"))
        else None,
        "origin_mm": _rounded_point(origin),
        "direction": _rounded_point(direction),
        "endpoint_mm": _rounded_point(endpoint),
        "tracker_pose_interpolated": bool(_bool_series(frame.iloc[[source_index]]["gaze3d_tracker_pose_interpolated"])[0])
        if "gaze3d_tracker_pose_interpolated" in frame
        else None,
        "tracker_fit_rmse_mm": round(float(row["gaze3d_tracker_fit_rmse_mm"]), 4)
        if "gaze3d_tracker_fit_rmse_mm" in frame and pd.notna(row.get("gaze3d_tracker_fit_rmse_mm"))
        else None,
    }
    return result


def _build_dyadic_gaze_preview(
    artifact_a: ArtifactRecord,
    artifact_b: ArtifactRecord,
    *,
    maximum_samples: int = MAX_DYADIC_GAZE_PREVIEW_SAMPLES,
    pairing_tolerance_ms: float | None = None,
    ray_length_mm: float = 1500.0,
) -> dict[str, Any]:
    a = _read_dyadic_gaze_table(artifact_a)
    b = _read_dyadic_gaze_table(artifact_b)
    if not a["role"] or not b["role"]:
        raise ValidationError("Both gaze artifacts must record their wearer roles")
    if a["role"] == b["role"]:
        raise ValidationError(f"Dyadic preview requires different roles; both artifacts are {a['role']}")
    assignment_a = str(a["assignment_artifact_id"] or "")
    assignment_b = str(b["assignment_artifact_id"] or "")
    if assignment_a and assignment_b and assignment_a != assignment_b:
        raise ValidationError("The two gaze artifacts use different Calibration handoffs / ETM assignments")
    alignment_a = str(a["alignment_mode"] or "")
    alignment_b = str(b["alignment_mode"] or "")
    if alignment_a and alignment_b and alignment_a != alignment_b:
        raise ValidationError("The two gaze artifacts use different time-alignment modes")
    if not len(a["finite_times"]) or not len(b["finite_times"]):
        raise ValidationError("One of the gaze artifacts has no finite local-time samples")
    common_start = max(float(np.nanmin(a["finite_times"])), float(np.nanmin(b["finite_times"])))
    common_end = min(float(np.nanmax(a["finite_times"])), float(np.nanmax(b["finite_times"])))
    if not np.isfinite(common_start) or not np.isfinite(common_end) or common_end <= common_start:
        raise ValidationError("The two gaze artifacts have no shared local-time interval")
    interval_a = _median_positive_interval(a["finite_times"])
    interval_b = _median_positive_interval(b["finite_times"])
    if pairing_tolerance_ms is None:
        intervals = [value for value in (interval_a, interval_b) if value is not None]
        adaptive = max(intervals) * 2.5 if intervals else 0.025
        tolerance_sec = float(np.clip(adaptive, 0.012, 0.050))
    else:
        tolerance_sec = float(pairing_tolerance_ms) / 1000.0
        if not np.isfinite(tolerance_sec) or tolerance_sec <= 0 or tolerance_sec > 0.5:
            raise ValidationError("Pairing tolerance must be between 0 and 500 ms")
    duration = common_end - common_start
    nominal_count = max(2, int(np.floor(duration * 30.0)) + 1)
    sample_count = min(maximum_samples, nominal_count)
    timeline = np.linspace(common_start, common_end, sample_count, dtype=float)
    roles = {a["role"]: a, b["role"]: b}
    ordered_roles = sorted(roles, key=lambda value: (0 if value == "R01" else 1 if value == "R02" else 2, value))
    samples: list[dict[str, Any]] = []
    all_points: list[list[float]] = []
    pair_deltas: list[float] = []
    available_counts = {role: 0 for role in ordered_roles}
    pair_count = 0
    for target_time in timeline.tolist():
        role_samples: dict[str, Any] = {}
        available_times = []
        for role in ordered_roles:
            row = _nearest_valid_row(roles[role], target_time, tolerance_sec, ray_length_mm)
            role_samples[role] = row
            if row.get("available"):
                available_counts[role] += 1
                available_times.append(float(row["sample_time_sec"]))
                if row.get("origin_mm") is not None:
                    all_points.append(row["origin_mm"])
                if row.get("endpoint_mm") is not None:
                    all_points.append(row["endpoint_mm"])
        both_valid = len(available_times) == len(ordered_roles) == 2
        pair_delta_ms = None
        if both_valid:
            pair_count += 1
            pair_delta_ms = abs(available_times[0] - available_times[1]) * 1000.0
            pair_deltas.append(pair_delta_ms)
        samples.append({
            "time_sec": round(float(target_time), 6),
            "roles": role_samples,
            "both_valid": both_valid,
            "pair_delta_ms": round(float(pair_delta_ms), 3) if pair_delta_ms is not None else None,
        })
    metadata = {}
    for role in ordered_roles:
        data = roles[role]
        artifact = data["artifact"]
        metadata[role] = {
            "artifact_id": artifact.id,
            "artifact_label": artifact.label,
            "source_valid_ray_count": int(len(data["valid_indices"])),
            "source_row_count": int(len(data["times"])),
            "preview_available_count": int(available_counts[role]),
            "preview_available_fraction": float(available_counts[role] / max(1, len(samples))),
            "assignment_artifact_id": data["assignment_artifact_id"],
            "alignment_mode": data["alignment_mode"],
            "native_trim_artifact_id": data["native_trim_artifact_id"],
            "calibration_hardware_id": data["calibration_hardware_id"],
            "calibration_profile_id": data["calibration_profile_id"],
            "time_column": data["time_column"],
        }
    return {
        "schema_version": DYADIC_GAZE_PREVIEW_SCHEMA,
        "coordinate_system": "Vicon world coordinates (millimetres)",
        "alignment": "Shared Native-trim local time" if alignment_a == "pretrimmed_shared_start" else (alignment_a or "compatible local time"),
        "roles": metadata,
        "role_order": ordered_roles,
        "assignment_artifact_id": assignment_a or assignment_b or None,
        "common_start_sec": round(common_start, 6),
        "common_end_sec": round(common_end, 6),
        "common_overlap_duration_sec": round(duration, 6),
        "pairing_tolerance_ms": round(tolerance_sec * 1000.0, 3),
        "preview_sample_rate_hz": round((len(samples) - 1) / duration, 6) if duration > 0 and len(samples) > 1 else None,
        "sample_count": len(samples),
        "common_valid_pair_count": int(pair_count),
        "common_valid_pair_fraction": float(pair_count / max(1, len(samples))),
        "median_pair_delta_ms": round(float(np.median(pair_deltas)), 3) if pair_deltas else None,
        "p95_pair_delta_ms": round(float(np.percentile(pair_deltas, 95)), 3) if pair_deltas else None,
        "ray_length_mm": float(ray_length_mm),
        "bounds_mm": _bounds(all_points),
        "samples": samples,
    }


def get_dyadic_gaze_rays_preview(
    store: ProjectStore,
    session_id: str,
    artifact_a_id: str,
    artifact_b_id: str,
    *,
    maximum_samples: int = MAX_DYADIC_GAZE_PREVIEW_SAMPLES,
    pairing_tolerance_ms: float | None = None,
) -> dict[str, Any]:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    artifact_a = session.artifacts.get(artifact_a_id)
    artifact_b = session.artifacts.get(artifact_b_id)
    if artifact_a is None or artifact_b is None:
        raise ValidationError("Both Gaze 3D artifacts are required for dyadic preview")
    key = hashlib.sha256(
        json.dumps({
            "a": artifact_a.id, "b": artifact_b.id,
            "a_created": artifact_a.created_at, "b_created": artifact_b.created_at,
            "maximum_samples": int(maximum_samples),
            "pairing_tolerance_ms": pairing_tolerance_ms,
            "schema": DYADIC_GAZE_PREVIEW_SCHEMA,
        }, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    cache_root = store.cache_dir() / "previews" / "dyadic_gaze"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"dyadic_gaze_{key}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text("utf-8"))
            if data.get("schema_version") == DYADIC_GAZE_PREVIEW_SCHEMA:
                return data
        except Exception:
            cache.unlink(missing_ok=True)
    data = _build_dyadic_gaze_preview(
        artifact_a, artifact_b, maximum_samples=maximum_samples,
        pairing_tolerance_ms=pairing_tolerance_ms,
    )
    atomic_write_json(cache, data)
    return data


CALIBRATION_PREVIEW_SCHEMA = "oculostudio-calibration-motion-preview-1.0"
MAX_CALIBRATION_PREVIEW_SAMPLES = 600
MAX_CALIBRATION_WORLD_MARKERS = 36


def _local_time(data: MocapData) -> np.ndarray:
    values = np.asarray(data.time_sec, dtype=float)
    if not len(values):
        return values
    return values - float(values[0])


def _nearest_indices(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    if not len(times):
        return np.zeros(len(query), dtype=np.int64)
    right = np.searchsorted(times, query, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(0, right - 1)
    choose_right = np.abs(times[right] - query) < np.abs(query - times[left])
    return np.where(choose_right, right, left).astype(np.int64)


def _marker_subset(data: MocapData, limit: int | None = None) -> np.ndarray:
    count = len(data.labels)
    if not limit or count <= limit:
        return np.arange(count, dtype=np.int64)
    score = np.asarray(data.valid, dtype=bool).sum(axis=0)
    selected = np.argsort(score)[::-1][: int(limit)]
    return np.sort(selected.astype(np.int64))


def _calibration_source_preview(
    asset,
    data: MocapData,
    query_time: np.ndarray,
    *,
    role: str,
    marker_limit: int | None = None,
) -> tuple[dict[str, Any], list[list[list[float] | None]], list[list[float] | None], list[list[float]]]:
    local = _local_time(data)
    indices = _nearest_indices(local, query_time)
    marker_indices = _marker_subset(data, marker_limit)
    labels = [data.labels[int(index)] for index in marker_indices]
    marker_samples: list[list[list[float] | None]] = []
    centre_samples: list[list[float] | None] = []
    all_points: list[list[float]] = []
    valid_counts: list[int] = []
    frame_numbers: list[int] = []
    for index in indices.tolist():
        rows: list[list[float] | None] = []
        valid_points: list[np.ndarray] = []
        for marker_index in marker_indices.tolist():
            if bool(data.valid[index, marker_index]) and np.isfinite(data.points_mm[index, marker_index]).all():
                point = _rounded_point(data.points_mm[index, marker_index])
                rows.append(point)
                if point is not None:
                    valid_points.append(np.asarray(point, dtype=float))
                    all_points.append(point)
            else:
                rows.append(None)
        marker_samples.append(rows)
        valid_counts.append(len(valid_points))
        frame_numbers.append(int(data.first_frame) + int(index))
        if valid_points:
            centre = _rounded_point(np.mean(np.asarray(valid_points), axis=0))
            centre_samples.append(centre)
            if centre is not None:
                all_points.append(centre)
        else:
            centre_samples.append(None)
    finite_fraction = float(np.asarray(data.valid[:, marker_indices], dtype=bool).mean()) if len(marker_indices) else 0.0
    metadata = {
        "asset_id": asset.id,
        "asset_label": asset.label,
        "role": role,
        "marker_labels": labels,
        "marker_count": len(labels),
        "source_marker_count": len(data.labels),
        "frame_count": int(data.frame_count),
        "frame_rate_hz": float(data.frame_rate),
        "duration_sec": round(float(local[-1]) if len(local) else 0.0, 6),
        "valid_marker_fraction": round(finite_fraction, 6),
        "sample_frame_numbers": frame_numbers,
        "sample_valid_marker_counts": valid_counts,
    }
    return metadata, marker_samples, centre_samples, all_points


def _calibration_profile_path(artifact: ArtifactRecord) -> Path | None:
    candidates = [Path(row.path) for row in artifact.files if Path(row.path).exists()]
    for suffix in (".oculocal", ".g3dcal", ".json"):
        matches = [path for path in candidates if path.suffix.lower() == suffix]
        if matches:
            matches.sort(key=lambda value: ("summary" in value.name.lower(), len(value.parts), str(value).lower()))
            return matches[0]
    return None


def _profile_overlay(
    profile_artifact: ArtifactRecord | None,
    profile: dict[str, Any] | None,
    tracker_assets: list,
    tracker_data: dict[str, MocapData],
    query_time: np.ndarray,
) -> tuple[dict[str, Any] | None, list[dict[str, Any] | None], list[list[float]]]:
    if profile_artifact is None or not profile:
        return None, [None for _ in query_time], []
    tracker = profile.get("tracker") or {}
    labels = list(tracker.get("marker_labels") or [])
    template = np.asarray(tracker.get("marker_template_mm") or [], dtype=float)
    transform = np.asarray(profile.get("T_tracker_from_camera") or [], dtype=float)
    if template.ndim != 2 or template.shape[1:] != (3,) or len(template) < 3 or transform.size != 16:
        return {
            "artifact_id": profile_artifact.id,
            "artifact_label": profile_artifact.label,
            "available": False,
            "reason": "Profile lacks a complete tracker template or camera transform",
        }, [None for _ in query_time], []
    transform = transform.reshape(4, 4)
    source_id = str(tracker.get("source_asset_id") or (profile.get("sources") or {}).get("tracker_asset_id") or "")
    selected_asset = next((row for row in tracker_assets if row.id == source_id), None)
    if selected_asset is None:
        label_set = {str(value).strip().replace(" ", "").upper() for value in labels}
        for row in tracker_assets:
            data = tracker_data.get(row.id)
            if data is None:
                continue
            available = {str(value).strip().replace(" ", "").upper() for value in data.labels}
            if label_set and label_set.issubset(available):
                selected_asset = row
                break
    if selected_asset is None:
        return {
            "artifact_id": profile_artifact.id,
            "artifact_label": profile_artifact.label,
            "available": False,
            "reason": "Select the tracker C3D used by this profile to place the camera in world coordinates",
            "hardware_id": profile.get("hardware_id"),
            "mount_version": profile.get("mount_version"),
        }, [None for _ in query_time], []
    data = tracker_data[selected_asset.id]
    try:
        positions, valid = data.select(labels)
    except Exception as error:
        return {
            "artifact_id": profile_artifact.id,
            "artifact_label": profile_artifact.label,
            "available": False,
            "reason": f"Tracker markers required by the profile are unavailable: {error}",
            "tracker_asset_id": selected_asset.id,
        }, [None for _ in query_time], []
    pose = PoseSeries.from_markers(
        _local_time(data), template, positions, valid,
        min_markers=3, max_fit_rmse_mm=20.0,
    ).interpolate(query_time, max_gap_sec=0.2)
    R_tc = transform[:3, :3]
    p_tc = transform[:3, 3]
    axis_length = 90.0
    samples: list[dict[str, Any] | None] = []
    points: list[list[float]] = []
    valid_count = 0
    for index in range(len(query_time)):
        if not bool(pose.valid[index]):
            samples.append(None)
            continue
        R_wt = pose.rotations[index]
        p_wt = pose.translations[index]
        origin = R_wt @ p_tc + p_wt
        R_wc = R_wt @ R_tc
        axes = {
            "x": origin + R_wc[:, 0] * axis_length,
            "y": origin + R_wc[:, 1] * axis_length,
            "z": origin + R_wc[:, 2] * axis_length,
        }
        origin_row = _rounded_point(origin)
        axes_row = {key: _rounded_point(value) for key, value in axes.items()}
        if origin_row is not None:
            points.append(origin_row)
        points.extend([value for value in axes_row.values() if value is not None])
        samples.append({
            "origin_mm": origin_row,
            "axes_mm": axes_row,
            "fit_rmse_mm": round(float(pose.fit_rmse_mm[index]), 4) if np.isfinite(pose.fit_rmse_mm[index]) else None,
            "interpolated": bool(pose.interpolated[index]),
            "marker_count": int(pose.marker_count[index]),
        })
        valid_count += 1
    qc = profile_artifact.qc or {}
    metadata = {
        "artifact_id": profile_artifact.id,
        "artifact_label": profile_artifact.label,
        "available": valid_count > 0,
        "tracker_asset_id": selected_asset.id,
        "tracker_asset_label": selected_asset.label,
        "hardware_id": profile.get("hardware_id") or profile_artifact.parameters.get("hardware_id"),
        "hardware_display_name": profile.get("hardware_display_name") or qc.get("hardware_display_name"),
        "mount_version": profile.get("mount_version") or profile_artifact.parameters.get("mount_version"),
        "camera_origin_tracker_mm": _rounded_point(p_tc),
        "valid_pose_samples": valid_count,
        "sample_count": len(query_time),
        "direct_reprojection_rms_px_median": qc.get("direct_reprojection_rms_px_median"),
        "direct_reprojection_rms_px_p90": qc.get("direct_reprojection_rms_px_p90"),
        "robust_translation_residual_mm_median": qc.get("translation_residual_mm_median") or qc.get("translation_residual_median_mm"),
        "robust_rotation_residual_deg_median": qc.get("rotation_residual_deg_median") or qc.get("rotation_residual_median_deg"),
        "selected_tracker_asset_id": qc.get("selected_tracker_asset_id") or source_id,
        "calibration_source": profile.get("calibration_source") or profile_artifact.parameters.get("calibration_source"),
    }
    return metadata, samples, points


def get_calibration_motion_preview(
    store: ProjectStore,
    session_id: str,
    *,
    board_asset_id: str | None = None,
    tracker_asset_ids: list[str] | None = None,
    world_asset_id: str | None = None,
    profile_artifact_id: str | None = None,
    maximum_samples: int = MAX_CALIBRATION_PREVIEW_SAMPLES,
) -> dict[str, Any]:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    tracker_asset_ids = [str(value) for value in (tracker_asset_ids or []) if str(value)]
    source_ids = [value for value in [board_asset_id, *tracker_asset_ids, world_asset_id] if value]
    if not source_ids:
        return {
            "schema_version": CALIBRATION_PREVIEW_SCHEMA,
            "session_id": session_id,
            "sample_count": 0,
            "duration_sec": 0.0,
            "sources": {},
            "profile": None,
            "bounds_mm": _bounds([]),
            "samples": [],
            "message": "Select a calibration-board C3D and at least one tracker candidate.",
        }
    missing = [asset_id for asset_id in source_ids if asset_id not in session.assets]
    if missing:
        raise ValidationError(f"Calibration preview assets are unavailable: {missing}")
    assets_by_id = {asset_id: session.assets[asset_id] for asset_id in source_ids}
    wrong_scope = [asset.label for asset in assets_by_id.values()
                   if str((asset.metadata or {}).get("asset_scope") or "interaction") != "calibration"]
    if wrong_scope:
        raise ValidationError(
            "Calibration preview only accepts Calibration Session Registry sources: " + ", ".join(wrong_scope)
        )
    set_ids = {str((asset.metadata or {}).get("calibration_set_id") or "") for asset in assets_by_id.values()}
    if len(set_ids) > 1:
        raise ValidationError("Calibration preview sources belong to different calibration sets")
    profile_artifact = session.artifacts.get(profile_artifact_id) if profile_artifact_id else None
    if profile_artifact_id and (profile_artifact is None or profile_artifact.kind != ArtifactKind.CALIBRATION_PROFILE):
        raise ValidationError("Selected calibration preview profile is unavailable")
    key_rows = []
    for asset_id in source_ids:
        row = assets_by_id[asset_id]
        key_rows.append((asset_id, row.source_sha256, row.size_bytes, row.metadata.get("frame_count"), row.metadata.get("duration_sec")))
    key_rows.append((profile_artifact_id, getattr(profile_artifact, "created_at", None)))
    digest = hashlib.sha256(json.dumps(key_rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
    cache_root = store.cache_dir() / "previews" / "calibration"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"calibration_motion_{digest}_{int(maximum_samples)}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text("utf-8"))
            if cached.get("schema_version") == CALIBRATION_PREVIEW_SCHEMA:
                return cached
        except Exception:
            cache_path.unlink(missing_ok=True)

    loaded: dict[str, MocapData] = {}
    for asset_id, asset in assets_by_id.items():
        if asset.kind not in {"c3d", "mocap_npz"}:
            raise ValidationError(f"Calibration preview requires mocap assets: {asset.label}")
        loaded[asset_id] = read_mocap(asset.source_path)
    durations = []
    for data in loaded.values():
        local = _local_time(data)
        if len(local):
            durations.append(float(local[-1]))
    duration = min(durations) if durations else 0.0
    sample_count = min(max(2, int(maximum_samples)), max(2, int(round(duration * 20.0)) + 1)) if duration > 0 else 1
    query_time = np.linspace(0.0, max(0.0, duration), sample_count, dtype=float)
    sources: dict[str, Any] = {}
    source_samples: dict[str, list[list[list[float] | None]]] = {}
    source_centres: dict[str, list[list[float] | None]] = {}
    all_points: list[list[float]] = []

    if board_asset_id:
        asset = assets_by_id[board_asset_id]
        meta, marker_rows, centres, points = _calibration_source_preview(asset, loaded[board_asset_id], query_time, role="board")
        sources[board_asset_id] = meta
        source_samples[board_asset_id] = marker_rows
        source_centres[board_asset_id] = centres
        all_points.extend(points)
    for tracker_index, tracker_id in enumerate(tracker_asset_ids):
        asset = assets_by_id[tracker_id]
        meta, marker_rows, centres, points = _calibration_source_preview(asset, loaded[tracker_id], query_time, role=f"tracker_{tracker_index + 1}")
        sources[tracker_id] = meta
        source_samples[tracker_id] = marker_rows
        source_centres[tracker_id] = centres
        all_points.extend(points)
    if world_asset_id:
        asset = assets_by_id[world_asset_id]
        meta, marker_rows, centres, points = _calibration_source_preview(
            asset, loaded[world_asset_id], query_time, role="world", marker_limit=MAX_CALIBRATION_WORLD_MARKERS,
        )
        sources[world_asset_id] = meta
        source_samples[world_asset_id] = marker_rows
        source_centres[world_asset_id] = centres
        all_points.extend(points)

    profile = None
    profile_data = None
    if profile_artifact is not None:
        profile_path = _calibration_profile_path(profile_artifact)
        if profile_path is not None:
            try:
                profile_data = load_calibration_profile(profile_path)
            except Exception:
                profile_data = None
    tracker_assets = [assets_by_id[asset_id] for asset_id in tracker_asset_ids]
    tracker_data = {asset_id: loaded[asset_id] for asset_id in tracker_asset_ids}
    profile, camera_samples, profile_points = _profile_overlay(
        profile_artifact, profile_data, tracker_assets, tracker_data, query_time,
    )
    all_points.extend(profile_points)

    samples = []
    for index, time_sec in enumerate(query_time.tolist()):
        sample_sources = {}
        for asset_id in source_samples:
            sample_sources[asset_id] = {
                "markers_mm": source_samples[asset_id][index],
                "centre_mm": source_centres[asset_id][index],
                "valid_marker_count": sources[asset_id]["sample_valid_marker_counts"][index],
                "frame": sources[asset_id]["sample_frame_numbers"][index],
            }
        samples.append({
            "time_sec": round(float(time_sec), 6),
            "sources": sample_sources,
            "camera": camera_samples[index] if index < len(camera_samples) else None,
        })
    result = {
        "schema_version": CALIBRATION_PREVIEW_SCHEMA,
        "session_id": session_id,
        "coordinate_system": "Calibration-take Vicon coordinates (millimetres)",
        "alignment": "Each selected calibration C3D is shown on its own take-local time axis starting at zero.",
        "duration_sec": round(float(duration), 6),
        "sample_count": len(samples),
        "maximum_preview_samples": int(maximum_samples),
        "sources": sources,
        "profile": profile,
        "bounds_mm": _bounds(all_points),
        "samples": samples,
    }
    atomic_write_json(cache_path, result)
    return result
