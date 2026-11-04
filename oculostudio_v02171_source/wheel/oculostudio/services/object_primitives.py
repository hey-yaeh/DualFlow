from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oculostudio.adapters.mocap import MocapData, read_mocap
from oculostudio.errors import ValidationError


@dataclass(frozen=True)
class ObjectPrimitive:
    scene_frame_index: int
    scene_timestamp_ns: int
    object_id: str
    object_category: str
    source_asset_id: str
    shape: str
    confidence: float
    center_mm: np.ndarray
    axes: np.ndarray
    half_lengths_mm: np.ndarray
    radius_mm: float

    def geometry_json(self) -> str:
        if self.shape == "sphere":
            payload = {
                "shape": "sphere",
                "center_mm": self.center_mm.tolist(),
                "radius_mm": float(self.radius_mm),
            }
        else:
            payload = {
                "shape": "obb",
                "center_mm": self.center_mm.tolist(),
                "axes": self.axes.tolist(),
                "half_lengths_mm": self.half_lengths_mm.tolist(),
                "bounding_radius_mm": float(self.radius_mm),
            }
        return json.dumps(payload, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ObjectHit:
    object_id: str
    object_category: str
    distance_mm: float
    point_mm: np.ndarray
    confidence: float
    exact_hit: bool
    angular_miss_deg: float
    primitive_confidence: float
    source_asset_id: str


@dataclass
class ObjectPrimitiveBuildResult:
    primitives_by_frame: dict[int, list[ObjectPrimitive]]
    frame_regions: pd.DataFrame
    diagnostics: dict[str, Any]


def _key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _prefix_tail(label: str) -> tuple[str, str]:
    raw = str(label).strip()
    if ":" in raw:
        prefix, tail = raw.rsplit(":", 1)
        return prefix.strip(), tail.strip()
    return "", raw


def _interpolate_marker(
    data: MocapData,
    marker_index: int,
    query_sec: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_valid = data.valid[:, marker_index] & np.isfinite(data.points_mm[:, marker_index]).all(axis=1)
    source_indices = np.flatnonzero(source_valid)
    output = np.full((len(query_sec), 3), np.nan, dtype=float)
    valid = np.zeros(len(query_sec), dtype=bool)
    if len(source_indices) == 0:
        return output, valid
    times = data.time_sec[source_indices]
    points = data.points_mm[source_indices, marker_index]
    if len(source_indices) == 1:
        exact = np.isclose(query_sec, times[0], atol=1e-9, rtol=0)
        output[exact] = points[0]
        valid[exact] = True
        return output, valid
    positions = np.searchsorted(times, query_sec, side="left")
    right = np.clip(positions, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    exact = np.isclose(query_sec, times[right], atol=1e-9, rtol=0)
    left[exact] = right[exact]
    in_range = (query_sec >= times[0]) & (query_sec <= times[-1])
    gaps = times[right] - times[left]
    allowed = in_range & (exact | (gaps <= max_gap_sec))
    denom = np.where(gaps > 0, gaps, 1.0)
    alpha = np.clip((query_sec - times[left]) / denom, 0.0, 1.0)
    output[allowed] = points[left[allowed]] * (1.0 - alpha[allowed, None]) + points[right[allowed]] * alpha[allowed, None]
    valid[allowed] = True
    return output, valid


def _load_profile(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"schema_version": "oculostudio-object-profile-1.0", "objects": []}
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read object profile {source}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        raise ValidationError("Object profile requires an 'objects' list")
    return payload


def _auto_groups(labels: list[str], fallback_id: str) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for label in labels:
        prefix, _tail = _prefix_tail(label)
        if prefix:
            groups.setdefault(prefix, []).append(label)
    usable = [
        {"object_id": prefix, "label": prefix, "marker_prefix": prefix, "geometry": "auto_obb"}
        for prefix, rows in sorted(groups.items()) if len(rows) >= 3
    ]
    if usable:
        return usable
    if len(labels) >= 3:
        return [{"object_id": fallback_id, "label": fallback_id, "markers": labels, "geometry": "auto_obb"}]
    return []


def _resolve_marker_indices(labels: list[str], definition: dict[str, Any]) -> list[int]:
    requested = {_key(x) for x in definition.get("markers", []) if str(x).strip()}
    prefix = str(definition.get("marker_prefix") or "").strip()
    prefix_key = _key(prefix)
    indices: list[int] = []
    for index, label in enumerate(labels):
        subject, tail = _prefix_tail(label)
        if requested and (_key(label) in requested or _key(tail) in requested):
            indices.append(index)
        elif prefix_key and (_key(subject) == prefix_key or _key(subject).endswith(prefix_key) or prefix_key.endswith(_key(subject))):
            indices.append(index)
    return sorted(set(indices))


def _fit_primitive(points: np.ndarray, *, geometry: str, padding_mm: float, radius_mm: float | None,
                   dimensions_mm: list[float] | None, minimum_half_extent_mm: float) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
    center = np.mean(points, axis=0)
    centered = points - center
    if geometry == "sphere" or len(points) < 3:
        radius = float(radius_mm if radius_mm is not None else np.max(np.linalg.norm(centered, axis=1)) + padding_mm)
        return "sphere", center, np.eye(3), np.array([radius, radius, radius]), radius
    covariance = centered.T @ centered / max(1, len(points) - 1)
    _values, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = centered @ axes
    if dimensions_mm is not None:
        dims = np.asarray(dimensions_mm, dtype=float)
        if dims.shape != (3,) or np.any(dims <= 0):
            raise ValidationError("Object dimensions_mm must contain three positive values")
        half = dims / 2.0
    else:
        half = np.max(np.abs(local), axis=0) + padding_mm
        half = np.maximum(half, minimum_half_extent_mm)
    radius = float(np.linalg.norm(half))
    return "obb", center, axes, half, radius


def build_object_primitives(
    *,
    object_asset_id: str,
    object_path: str | Path,
    scene_frame_indices: np.ndarray,
    scene_timestamps_ns: np.ndarray,
    query_time_sec: np.ndarray,
    profile_path: str | Path | None = None,
    object_overrides: dict[str, str] | None = None,
    max_marker_gap_sec: float = 0.1,
    default_padding_mm: float = 25.0,
) -> ObjectPrimitiveBuildResult:
    data = read_mocap(object_path)
    profile = _load_profile(profile_path)
    fallback = Path(object_path).stem
    definitions = list(profile.get("objects") or _auto_groups(data.labels, fallback))
    if not definitions:
        raise ValidationError("No object marker groups could be resolved. Supply a profile with object IDs and markers/prefixes.")
    object_overrides = object_overrides or {}
    primitives_by_frame: dict[int, list[ObjectPrimitive]] = {}
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for definition in definitions:
        indices = _resolve_marker_indices(data.labels, definition)
        minimum_markers = int(definition.get("minimum_markers", 3))
        if len(indices) < minimum_markers:
            diagnostics.append({"object_id": definition.get("object_id"), "status": "insufficient_markers", "resolved_markers": len(indices)})
            continue
        interpolated = []
        valid_masks = []
        for marker_index in indices:
            points, valid = _interpolate_marker(data, marker_index, query_time_sec, max_gap_sec=max_marker_gap_sec)
            interpolated.append(points)
            valid_masks.append(valid)
        point_stack = np.stack(interpolated, axis=1)
        valid_stack = np.stack(valid_masks, axis=1)
        raw_id = str(definition.get("object_id") or definition.get("label") or "object")
        object_id = str(object_overrides.get(raw_id, raw_id))
        category = str(definition.get("category") or definition.get("label") or raw_id)
        geometry = str(definition.get("geometry") or "auto_obb").lower()
        padding = float(definition.get("padding_mm", default_padding_mm))
        radius_override = definition.get("radius_mm")
        dimensions = definition.get("dimensions_mm")
        minimum_half_extent = float(definition.get("minimum_half_extent_mm", 10.0))
        built_count = 0
        for position, frame_index in enumerate(scene_frame_indices):
            valid_indices = np.flatnonzero(valid_stack[position])
            if len(valid_indices) < minimum_markers:
                continue
            points = point_stack[position, valid_indices]
            shape, center, axes, half, radius = _fit_primitive(
                points,
                geometry="sphere" if geometry == "sphere" else "auto_obb",
                padding_mm=padding,
                radius_mm=float(radius_override) if radius_override is not None else None,
                dimensions_mm=dimensions,
                minimum_half_extent_mm=minimum_half_extent,
            )
            confidence = float(min(1.0, len(valid_indices) / max(minimum_markers, len(indices))))
            primitive = ObjectPrimitive(
                scene_frame_index=int(frame_index),
                scene_timestamp_ns=int(scene_timestamps_ns[position]),
                object_id=object_id,
                object_category=category,
                source_asset_id=object_asset_id,
                shape=shape,
                confidence=confidence,
                center_mm=center,
                axes=axes,
                half_lengths_mm=half,
                radius_mm=radius,
            )
            primitives_by_frame.setdefault(int(frame_index), []).append(primitive)
            rows.append({
                "scene_frame_index": int(frame_index),
                "scene_timestamp_ns": int(scene_timestamps_ns[position]),
                "object_entity_id": object_id,
                "object_category": category,
                "source_asset_id": object_asset_id,
                "shape": shape,
                "marker_count": int(len(valid_indices)),
                "primitive_confidence": confidence,
                "geometry_json": primitive.geometry_json(),
            })
            built_count += 1
        diagnostics.append({
            "object_id": object_id,
            "object_category": category,
            "status": "ok" if built_count else "no_valid_frames",
            "resolved_marker_count": len(indices),
            "resolved_marker_labels": [data.labels[i] for i in indices],
            "built_frame_count": built_count,
            "geometry": geometry,
        })
    if not primitives_by_frame:
        raise ValidationError("Object geometry could not be built for any requested scene frame")
    return ObjectPrimitiveBuildResult(
        primitives_by_frame=primitives_by_frame,
        frame_regions=pd.DataFrame(rows),
        diagnostics={"asset_id": object_asset_id, "objects": diagnostics, "frame_rate": data.frame_rate},
    )


def _ray_sphere(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> float | None:
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    discriminant = b * b - c
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    candidates = [-b - root, -b + root]
    positive = [value for value in candidates if value >= 0]
    return min(positive) if positive else None


def _ray_obb(origin: np.ndarray, direction: np.ndarray, primitive: ObjectPrimitive) -> float | None:
    local_o = primitive.axes.T @ (origin - primitive.center_mm)
    local_d = primitive.axes.T @ direction
    t_min = -float("inf")
    t_max = float("inf")
    for axis in range(3):
        half = float(primitive.half_lengths_mm[axis])
        if abs(local_d[axis]) < 1e-12:
            if local_o[axis] < -half or local_o[axis] > half:
                return None
            continue
        t1 = (-half - local_o[axis]) / local_d[axis]
        t2 = (half - local_o[axis]) / local_d[axis]
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, float(t1))
        t_max = min(t_max, float(t2))
        if t_min > t_max:
            return None
    if t_max < 0:
        return None
    return max(0.0, t_min)


def _angular_miss(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> tuple[float, float]:
    vector = center - origin
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-9:
        return 0.0, 0.0
    angle = math.degrees(math.acos(float(np.clip(np.dot(direction, vector / distance), -1.0, 1.0))))
    extent = math.degrees(math.asin(float(np.clip(radius / max(distance, radius), 0.0, 1.0))))
    return max(0.0, angle - extent), distance


def intersect_frame_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    primitives: list[ObjectPrimitive],
    *,
    uncertainty_cone_deg: float,
    maximum_distance_mm: float,
) -> list[ObjectHit | None]:
    output: list[ObjectHit | None] = []
    for origin, direction in zip(origins, directions):
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(origin).all() or not np.isfinite(direction).all() or norm <= 0:
            output.append(None)
            continue
        direction = direction / norm
        exact: list[ObjectHit] = []
        candidates: list[ObjectHit] = []
        for primitive in primitives:
            distance = _ray_sphere(origin, direction, primitive.center_mm, primitive.radius_mm) if primitive.shape == "sphere" else _ray_obb(origin, direction, primitive)
            if distance is not None and distance <= maximum_distance_mm:
                point = origin + direction * distance
                exact.append(ObjectHit(
                    object_id=primitive.object_id,
                    object_category=primitive.object_category,
                    distance_mm=float(distance),
                    point_mm=point,
                    confidence=float(primitive.confidence),
                    exact_hit=True,
                    angular_miss_deg=0.0,
                    primitive_confidence=primitive.confidence,
                    source_asset_id=primitive.source_asset_id,
                ))
                continue
            miss, center_distance = _angular_miss(origin, direction, primitive.center_mm, primitive.radius_mm)
            if miss <= uncertainty_cone_deg and center_distance <= maximum_distance_mm:
                confidence = primitive.confidence * max(0.0, 1.0 - miss / max(uncertainty_cone_deg, 1e-9)) * 0.65
                candidates.append(ObjectHit(
                    object_id=primitive.object_id,
                    object_category=primitive.object_category,
                    distance_mm=center_distance,
                    point_mm=origin + direction * center_distance,
                    confidence=float(confidence),
                    exact_hit=False,
                    angular_miss_deg=float(miss),
                    primitive_confidence=primitive.confidence,
                    source_asset_id=primitive.source_asset_id,
                ))
        if exact:
            output.append(min(exact, key=lambda hit: hit.distance_mm))
        elif candidates:
            output.append(min(candidates, key=lambda hit: (hit.angular_miss_deg, hit.distance_mm)))
        else:
            output.append(None)
    return output
