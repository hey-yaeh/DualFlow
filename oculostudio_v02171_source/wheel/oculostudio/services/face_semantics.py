from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from oculostudio.adapters.pupil import find_facemapper_files
from oculostudio.domain.semantic import SemanticRegion
from oculostudio.errors import ValidationError


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class EllipseAOI:
    region: str
    center: Point2D
    radius_x: float
    radius_y: float
    confidence: float
    requested_radius_x: float | None = None
    requested_radius_y: float | None = None
    geometry_clipped: bool = False
    containment_policy: str | None = None
    feature_scale: float = 1.0

    def normalized_distance(self, x: float, y: float) -> float:
        if self.radius_x <= 0 or self.radius_y <= 0:
            return float("inf")
        return math.sqrt(
            ((x - self.center.x) / self.radius_x) ** 2
            + ((y - self.center.y) / self.radius_y) ** 2
        )


def _point_on_segment(
    x: float,
    y: float,
    start: Point2D,
    end: Point2D,
    *,
    tolerance: float = 1e-6,
) -> bool:
    cross = (x - start.x) * (end.y - start.y) - (y - start.y) * (end.x - start.x)
    if abs(cross) > tolerance:
        return False
    dot = (x - start.x) * (end.x - start.x) + (y - start.y) * (end.y - start.y)
    if dot < -tolerance:
        return False
    length_sq = (end.x - start.x) ** 2 + (end.y - start.y) ** 2
    return dot <= length_sq + tolerance


def _point_in_polygon(x: float, y: float, vertices: tuple[Point2D, ...]) -> bool:
    """Boundary-inclusive ray-casting test for a simple 2D polygon."""
    if len(vertices) < 3:
        return False
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if _point_on_segment(x, y, previous, current):
            return True
        crosses = (current.y > y) != (previous.y > y)
        if crosses:
            denominator = previous.y - current.y
            if abs(denominator) < 1e-12:
                denominator = 1e-12
            crossing_x = (previous.x - current.x) * (y - current.y) / denominator + current.x
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class PolygonAOI:
    """A polygonal facial AOI used for extended regions such as the nose bridge."""

    region: str
    vertices: tuple[Point2D, ...]
    confidence: float
    requested_vertices: tuple[Point2D, ...] | None = None
    geometry_clipped: bool = False
    containment_policy: str | None = None
    feature_scale: float = 1.0
    geometry_model: str = "polygon"

    @property
    def center(self) -> Point2D:
        if not self.vertices:
            return Point2D(0.0, 0.0)
        return Point2D(
            float(sum(point.x for point in self.vertices) / len(self.vertices)),
            float(sum(point.y for point in self.vertices) / len(self.vertices)),
        )

    def normalized_distance(self, x: float, y: float) -> float:
        if not _point_in_polygon(float(x), float(y), self.vertices):
            return float("inf")
        center = self.center
        maximum = max(
            (math.hypot(point.x - center.x, point.y - center.y) for point in self.vertices),
            default=0.0,
        )
        if maximum <= 1e-9:
            return 0.0
        return min(1.0, math.hypot(float(x) - center.x, float(y) - center.y) / maximum)


@dataclass(frozen=True)
class FaceInstance:
    scene_frame_index: int
    scene_timestamp_ns: int
    track_id: int
    explicit_entity_id: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    left_eye: Point2D | None
    right_eye: Point2D | None
    nose: Point2D | None
    mouth_left: Point2D | None
    mouth_right: Point2D | None
    confidence: float
    source_row_index: int

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def entity_id(self) -> str:
        return self.explicit_entity_id or f"face_track_{self.track_id}"


@dataclass(frozen=True)
class FaceHit:
    entity_id: str
    entity_is_explicit: bool
    track_id: int
    region: str
    confidence: float
    normalized_distance: float
    scene_frame_index: int
    source_row_index: int
    gate_hit_type: str = "exact_green_box"
    edge_distance_px: float = 0.0
    edge_tolerance_px: float = 0.0
    edge_candidate_count: int = 0
    edge_ambiguous: bool = False


@dataclass
class FaceSemanticResult:
    instances_by_frame: dict[int, list[FaceInstance]]
    frame_regions: pd.DataFrame
    diagnostics: dict[str, Any]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _column(columns, aliases: list[str], *, required: bool = False) -> str | None:
    mapping = {_normalize(column): column for column in columns}
    for alias in aliases:
        key = _normalize(alias)
        if key in mapping:
            return mapping[key]
    if required:
        raise ValidationError(f"Face Mapper table is missing required column: {aliases[0]}")
    return None


def _point_columns(columns, names: list[str]) -> tuple[str | None, str | None]:
    x_aliases: list[str] = []
    y_aliases: list[str] = []
    for name in names:
        x_aliases.extend([
            f"{name} x [px]", f"{name}_x_px", f"{name} x", f"{name}_x",
            f"{name} center x [px]", f"{name}_center_x_px",
        ])
        y_aliases.extend([
            f"{name} y [px]", f"{name}_y_px", f"{name} y", f"{name}_y",
            f"{name} center y [px]", f"{name}_center_y_px",
        ])
    return _column(columns, x_aliases), _column(columns, y_aliases)


def _point(row: pd.Series, x_col: str | None, y_col: str | None, width: int, height: int) -> Point2D | None:
    if not x_col or not y_col:
        return None
    try:
        x, y = float(row[x_col]), float(row[y_col])
    except Exception:
        return None
    if not np.isfinite([x, y]).all():
        return None
    if max(abs(x), abs(y)) <= 2.0:
        x *= width
        y *= height
    return Point2D(x, y)


def _infer_anatomical_swap(
    frame: pd.DataFrame,
    left_eye_cols: tuple[str | None, str | None],
    right_eye_cols: tuple[str | None, str | None],
    width: int,
    height: int,
) -> tuple[bool, dict[str, Any]]:
    """Infer whether exported bilateral labels use image-left/image-right.

    OculoStudio's canonical labels are anatomical: ``left_eye`` means the
    observed person's own left eye.  In a normal (non-mirrored) scene camera
    that eye appears on the image-right side.  Some FaceMapper exports instead
    name landmarks by image side.  A robust median over the recording lets us
    normalize those exports once without frame-by-frame label flicker.
    """
    if not all(left_eye_cols) or not all(right_eye_cols) or frame.empty:
        return False, {
            "evidence_count": 0,
            "median_exported_left_minus_right_x_px": None,
            "reason": "insufficient_bilateral_landmarks",
        }
    deltas: list[float] = []
    step = max(1, len(frame) // 4000)
    for _, row in frame.iloc[::step].iterrows():
        left = _point(row, *left_eye_cols, width, height)
        right = _point(row, *right_eye_cols, width, height)
        if left is None or right is None:
            continue
        delta = float(left.x - right.x)
        if np.isfinite(delta) and abs(delta) >= 1.0:
            deltas.append(delta)
    if len(deltas) < 3:
        return False, {
            "evidence_count": len(deltas),
            "median_exported_left_minus_right_x_px": float(np.median(deltas)) if deltas else None,
            "reason": "insufficient_bilateral_landmarks",
        }
    median_delta = float(np.median(deltas))
    # Exported 'left' lying image-left means viewer/image-side naming and must
    # be swapped to obtain the observed person's anatomical left/right.
    return median_delta < 0.0, {
        "evidence_count": len(deltas),
        "median_exported_left_minus_right_x_px": median_delta,
        "reason": "median_eye_x_order",
    }


def _nearest_indices(reference_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    if len(reference) == 0:
        return np.full(len(query), -1, dtype=int), np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
    pos = np.searchsorted(reference, query, side="left")
    right = np.clip(pos, 0, len(reference) - 1)
    left = np.clip(pos - 1, 0, len(reference) - 1)
    left_error = np.abs(query - reference[left])
    right_error = np.abs(reference[right] - query)
    choose_right = right_error < left_error
    index = np.where(choose_right, right, left)
    error = np.where(choose_right, right_error, left_error)
    return index.astype(int), error.astype(np.int64)


def _iou(a: FaceInstance, b: FaceInstance) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(a: FaceInstance, b: FaceInstance) -> float:
    scale = max(1.0, 0.5 * (a.width + a.height + b.width + b.height))
    return math.hypot(a.center.x - b.center.x, a.center.y - b.center.y) / scale


def _assign_tracks(instances: list[FaceInstance], maximum_frame_gap: int = 4) -> list[FaceInstance]:
    by_frame: dict[int, list[FaceInstance]] = {}
    for item in instances:
        by_frame.setdefault(item.scene_frame_index, []).append(item)
    next_track = 0
    active: dict[int, FaceInstance] = {}
    active_frame: dict[int, int] = {}
    output: list[FaceInstance] = []
    for frame_index in sorted(by_frame):
        rows = by_frame[frame_index]
        candidates: list[tuple[float, int, int]] = []
        for row_index, row in enumerate(rows):
            for track_id, previous in active.items():
                if frame_index - active_frame[track_id] > maximum_frame_gap:
                    continue
                overlap = _iou(row, previous)
                distance = _center_distance(row, previous)
                if overlap >= 0.04 or distance <= 1.45:
                    candidates.append(((1.0 - overlap) + 0.35 * distance, row_index, track_id))
        assigned_rows: set[int] = set()
        assigned_tracks: set[int] = set()
        row_to_track: dict[int, int] = {}
        for _, row_index, track_id in sorted(candidates):
            if row_index in assigned_rows or track_id in assigned_tracks:
                continue
            assigned_rows.add(row_index)
            assigned_tracks.add(track_id)
            row_to_track[row_index] = track_id
        for row_index, row in enumerate(rows):
            track_id = row_to_track.get(row_index)
            if track_id is None:
                track_id = next_track
                next_track += 1
            tracked = replace(row, track_id=track_id)
            active[track_id] = tracked
            active_frame[track_id] = frame_index
            output.append(tracked)
    return output


def _bounded_feature_aoi(
    *,
    region: str,
    center: Point2D,
    desired_radius_x: float,
    desired_radius_y: float,
    instance: FaceInstance,
    confidence: float,
    feature_scale: float,
    margin_ratio: float,
) -> EllipseAOI:
    """Return an axis-aligned feature ellipse fully contained in the face box.

    FaceMapper landmarks occasionally lie very near (or slightly outside) a
    detector boundary.  Enlarging an ellipse around such a point without a
    containment step makes the AOI spill onto hair, clothing, or background.
    This helper first clamps the centre into a small inner face rectangle, then
    clips each radius to the available distance from that centre to the four
    inner boundaries.  The same geometry is used for classification, tables,
    live review, and rendered overlays.
    """
    width = max(0.0, float(instance.width))
    height = max(0.0, float(instance.height))
    margin_ratio = float(np.clip(margin_ratio, 0.0, 0.20))
    margin_x = min(width * margin_ratio, max(0.0, width * 0.45))
    margin_y = min(height * margin_ratio, max(0.0, height * 0.45))
    left = float(instance.x1) + margin_x
    right = float(instance.x2) - margin_x
    top = float(instance.y1) + margin_y
    bottom = float(instance.y2) - margin_y
    if right <= left:
        left, right = float(instance.x1), float(instance.x2)
    if bottom <= top:
        top, bottom = float(instance.y1), float(instance.y2)

    # A half-pixel minimum keeps the geometry valid even for very small boxes,
    # while still guaranteeing containment.
    half_width = max(0.0, 0.5 * (right - left))
    half_height = max(0.0, 0.5 * (bottom - top))
    minimum_rx = min(0.5, half_width)
    minimum_ry = min(0.5, half_height)
    cx_low, cx_high = left + minimum_rx, right - minimum_rx
    cy_low, cy_high = top + minimum_ry, bottom - minimum_ry
    cx = float(np.clip(center.x, cx_low, cx_high)) if cx_high >= cx_low else 0.5 * (left + right)
    cy = float(np.clip(center.y, cy_low, cy_high)) if cy_high >= cy_low else 0.5 * (top + bottom)

    max_rx = max(0.0, min(cx - left, right - cx))
    max_ry = max(0.0, min(cy - top, bottom - cy))
    radius_x = min(max(0.0, float(desired_radius_x)), max_rx)
    radius_y = min(max(0.0, float(desired_radius_y)), max_ry)
    # Keep a valid ellipse whenever the face box has any usable extent.
    if max_rx > 0.0:
        radius_x = max(minimum_rx, radius_x)
    if max_ry > 0.0:
        radius_y = max(minimum_ry, radius_y)

    clipped = (
        abs(cx - float(center.x)) > 1e-6
        or abs(cy - float(center.y)) > 1e-6
        or radius_x + 1e-6 < float(desired_radius_x)
        or radius_y + 1e-6 < float(desired_radius_y)
    )
    return EllipseAOI(
        region=region,
        center=Point2D(cx, cy),
        radius_x=radius_x,
        radius_y=radius_y,
        confidence=confidence,
        requested_radius_x=float(desired_radius_x),
        requested_radius_y=float(desired_radius_y),
        geometry_clipped=bool(clipped),
        containment_policy="face_bbox_inner_clip",
        feature_scale=float(feature_scale),
    )


def _inner_face_bounds(instance: FaceInstance, margin_ratio: float) -> tuple[float, float, float, float]:
    width = max(0.0, float(instance.width))
    height = max(0.0, float(instance.height))
    margin_ratio = float(np.clip(margin_ratio, 0.0, 0.20))
    margin_x = min(width * margin_ratio, max(0.0, width * 0.45))
    margin_y = min(height * margin_ratio, max(0.0, height * 0.45))
    left = float(instance.x1) + margin_x
    right = float(instance.x2) - margin_x
    top = float(instance.y1) + margin_y
    bottom = float(instance.y2) - margin_y
    if right <= left:
        left, right = float(instance.x1), float(instance.x2)
    if bottom <= top:
        top, bottom = float(instance.y1), float(instance.y2)
    return left, right, top, bottom


def _bounded_polygon_aoi(
    *,
    region: str,
    vertices: list[Point2D] | tuple[Point2D, ...],
    instance: FaceInstance,
    confidence: float,
    feature_scale: float,
    margin_ratio: float,
    geometry_model: str,
) -> PolygonAOI:
    """Clamp every polygon vertex to the inner face rectangle."""
    requested = tuple(Point2D(float(point.x), float(point.y)) for point in vertices)
    left, right, top, bottom = _inner_face_bounds(instance, margin_ratio)
    clipped_vertices = tuple(
        Point2D(
            float(np.clip(point.x, left, right)),
            float(np.clip(point.y, top, bottom)),
        )
        for point in requested
    )
    clipped = any(
        abs(before.x - after.x) > 1e-6 or abs(before.y - after.y) > 1e-6
        for before, after in zip(requested, clipped_vertices)
    )
    return PolygonAOI(
        region=region,
        vertices=clipped_vertices,
        requested_vertices=requested,
        confidence=confidence,
        geometry_clipped=bool(clipped),
        containment_policy=("face_bbox_exact_clip" if float(margin_ratio) <= 0.0 else "face_bbox_inner_clip"),
        feature_scale=float(feature_scale),
        geometry_model=geometry_model,
    )


def _nose_base_y(instance: FaceInstance, settings: dict[str, Any]) -> float:
    """Estimate the horizontal lower boundary of the nose region.

    Five-point FaceMapper data exposes a nose point and mouth corners rather
    than a full nose contour. The boundary is placed just below the nose point
    using the nose-to-mouth distance, then clamped inside the accepted green
    FaceMapper face box. In the balanced facial model, the lip envelope starts
    lower than this boundary and the intervening pixels remain ``other_face``.
    """
    margin_ratio = float(np.clip(settings.get("face_margin_ratio", 0.02), 0.0, 0.20))
    left, right, top, bottom = _inner_face_bounds(instance, margin_ratio)
    height = max(1.0, bottom - top)
    ratio = float(np.clip(settings.get("nose_base_offset_ratio", 0.22), 0.0, 0.60))

    mouth_y = None
    if instance.mouth_left is not None and instance.mouth_right is not None:
        mouth_y = 0.5 * (float(instance.mouth_left.y) + float(instance.mouth_right.y))

    if instance.nose is not None:
        nose_y = float(instance.nose.y)
        if mouth_y is not None and mouth_y > nose_y:
            candidate = nose_y + ratio * (mouth_y - nose_y)
        else:
            candidate = nose_y + 0.055 * height
    elif mouth_y is not None and instance.left_eye is not None and instance.right_eye is not None:
        eye_y = 0.5 * (float(instance.left_eye.y) + float(instance.right_eye.y))
        candidate = eye_y + 0.62 * max(1.0, mouth_y - eye_y)
    elif mouth_y is not None:
        candidate = mouth_y - 0.18 * height
    else:
        candidate = top + 0.58 * height

    # Preserve a meaningful nose and lower-face area even under noisy landmarks.
    return float(np.clip(candidate, top + 0.20 * height, bottom - 0.16 * height))


def _lower_face_mouth_vertices(
    instance: FaceInstance,
    *,
    settings: dict[str, Any],
) -> list[Point2D]:
    """Cover the face from the nose baseline to the green face-box bottom."""
    # The user-facing definition is the entire lower portion of the accepted
    # green FaceMapper face box: from the shared nose baseline to the box bottom.
    # Unlike fine landmark ellipses, this semantic partition uses the exact box
    # left/right/bottom boundaries rather than an inner safety margin.
    left, right, bottom = float(instance.x1), float(instance.x2), float(instance.y2)
    nose_base = _nose_base_y(instance, settings)
    return [
        Point2D(left, nose_base),
        Point2D(right, nose_base),
        Point2D(right, bottom),
        Point2D(left, bottom),
    ]


def _adaptive_lip_envelope_vertices(
    instance: FaceInstance,
    *,
    settings: dict[str, Any],
) -> list[Point2D] | None:
    """Build a bounded, mouth-corner-aligned lip envelope.

    FaceMapper provides mouth corners but not upper/lower lip contours.  The
    balanced model therefore expands the corner span laterally by 15%, adds a
    modest asymmetric vertical envelope (more space below for an open mouth),
    and caps the result well above the chin.  The global eye/nose feature scale
    is intentionally ignored. Missing mouth landmarks produce no synthetic
    mouth AOI; the accepted green face box then falls back to ``other_face``.
    """
    if instance.mouth_left is None or instance.mouth_right is None:
        return None
    left_point = np.array([instance.mouth_left.x, instance.mouth_left.y], dtype=float)
    right_point = np.array([instance.mouth_right.x, instance.mouth_right.y], dtype=float)
    vector = right_point - left_point
    span = float(np.linalg.norm(vector))
    if not np.isfinite(span) or span < 1.0:
        return None
    axis = vector / span
    normal = np.array([-axis[1], axis[0]], dtype=float)
    if normal[1] < 0.0:
        normal = -normal

    width = max(1.0, float(instance.width))
    height = max(1.0, float(instance.height))
    lateral_ratio = float(np.clip(settings.get("mouth_lateral_padding_ratio", 0.15), 0.0, 0.50))
    lateral = span * lateral_ratio
    requested_width = span + 2.0 * lateral
    maximum_width = width * float(np.clip(settings.get("mouth_max_width_face_ratio", 0.52), 0.20, 0.90))
    if requested_width > maximum_width:
        lateral = max(0.0, 0.5 * (maximum_width - span))

    top_padding = max(1.0, height * float(np.clip(settings.get("mouth_top_padding_face_ratio", 0.045), 0.01, 0.20)), span * 0.15)
    bottom_padding = max(1.0, height * float(np.clip(settings.get("mouth_bottom_padding_face_ratio", 0.095), 0.02, 0.25)), span * 0.35)
    requested_height = top_padding + bottom_padding
    maximum_height = height * float(np.clip(settings.get("mouth_max_height_face_ratio", 0.18), 0.08, 0.22))
    absolute_maximum = height * float(np.clip(settings.get("mouth_absolute_max_height_face_ratio", 0.22), 0.10, 0.30))
    maximum_height = min(maximum_height, absolute_maximum)
    if requested_height > maximum_height and requested_height > 1e-9:
        scale = maximum_height / requested_height
        top_padding *= scale
        bottom_padding *= scale

    left_anchor = left_point - axis * lateral
    right_anchor = right_point + axis * lateral
    vertices = [
        left_anchor - normal * top_padding,
        right_anchor - normal * top_padding,
        right_anchor + normal * bottom_padding,
        left_anchor + normal * bottom_padding,
    ]

    # Keep the lip region below the nose baseline and leave a visible chin band
    # for ``other_face``.  This is a semantic guard, not only box clipping.
    nose_floor = _nose_base_y(instance, settings) + height * float(np.clip(settings.get("mouth_min_top_below_nose_ratio", 0.015), 0.0, 0.10))
    bottom_limit = float(instance.y1) + height * float(np.clip(settings.get("mouth_bottom_limit_face_ratio", 0.90), 0.70, 0.98))
    current_top = min(float(point[1]) for point in vertices)
    current_bottom = max(float(point[1]) for point in vertices)
    if current_top < nose_floor:
        shift = nose_floor - current_top
        vertices = [point + np.array([0.0, shift], dtype=float) for point in vertices]
        current_bottom += shift
    if current_bottom > bottom_limit:
        shift = current_bottom - bottom_limit
        vertices = [point - np.array([0.0, shift], dtype=float) for point in vertices]

    inner_left, inner_right, inner_top, inner_bottom = _inner_face_bounds(
        instance, float(np.clip(settings.get("face_margin_ratio", 0.02), 0.0, 0.20))
    )
    lower_y = max(inner_top, nose_floor)
    upper_y = min(inner_bottom, bottom_limit)
    if upper_y <= lower_y + 1.0:
        return None
    output = [
        Point2D(
            float(np.clip(point[0], inner_left, inner_right)),
            float(np.clip(point[1], lower_y, upper_y)),
        )
        for point in vertices
    ]
    # Reject collapsed/noisy polygons instead of inventing a mouth label.
    area = 0.0
    for index, point in enumerate(output):
        nxt = output[(index + 1) % len(output)]
        area += point.x * nxt.y - nxt.x * point.y
    if abs(area) * 0.5 < 4.0:
        return None
    return output


def _strict_lip_ellipse(
    instance: FaceInstance,
    *,
    settings: dict[str, Any],
) -> tuple[Point2D, float, float] | None:
    """Compact landmark-centred lip ellipse independent of feature scale."""
    if instance.mouth_left is None or instance.mouth_right is None:
        return None
    center = Point2D(
        0.5 * (float(instance.mouth_left.x) + float(instance.mouth_right.x)),
        0.5 * (float(instance.mouth_left.y) + float(instance.mouth_right.y)),
    )
    span = float(math.hypot(
        instance.mouth_right.x - instance.mouth_left.x,
        instance.mouth_right.y - instance.mouth_left.y,
    ))
    radius_x = max(3.0, 0.58 * span, instance.width * float(settings.get("mouth_strict_radius_x_face_ratio", 0.14)))
    radius_y = max(3.0, 0.18 * span, instance.height * float(settings.get("mouth_strict_radius_y_face_ratio", 0.055)))
    return center, radius_x, radius_y


def _nose_bridge_vertices(
    instance: FaceInstance,
    *,
    feature_scale: float,
    settings: dict[str, Any],
) -> list[Point2D] | None:
    """Legacy eye-line-to-base nose polygon retained for reproducibility."""
    if instance.left_eye is None or instance.right_eye is None:
        return None
    eye_a = np.array([instance.left_eye.x, instance.left_eye.y], dtype=float)
    eye_b = np.array([instance.right_eye.x, instance.right_eye.y], dtype=float)
    eye_vector = eye_b - eye_a
    eye_span = float(np.linalg.norm(eye_vector))
    if not np.isfinite(eye_span) or eye_span < 1.0:
        return None
    axis = eye_vector / eye_span
    eye_mid = 0.5 * (eye_a + eye_b)

    top_expand = eye_span * float(settings.get("nose_bridge_top_expand_ratio", 0.04)) * min(feature_scale, 1.8)
    top_a = eye_a - axis * top_expand
    top_b = eye_b + axis * top_expand

    if instance.nose is not None:
        lower_center_x = float(instance.nose.x)
    elif instance.mouth_left is not None and instance.mouth_right is not None:
        lower_center_x = 0.5 * (float(instance.mouth_left.x) + float(instance.mouth_right.x))
    else:
        lower_center_x = float(eye_mid[0])
    lower_center = np.array([lower_center_x, _nose_base_y(instance, settings)], dtype=float)

    mouth_span = 0.0
    if instance.mouth_left is not None and instance.mouth_right is not None:
        mouth_span = float(math.hypot(
            instance.mouth_right.x - instance.mouth_left.x,
            instance.mouth_right.y - instance.mouth_left.y,
        ))
    lower_half_width = max(
        eye_span * float(settings.get("nose_bridge_bottom_halfwidth_iod_ratio", 0.14)),
        mouth_span * float(settings.get("nose_bridge_bottom_halfwidth_mouth_ratio", 0.20)),
    ) * min(max(feature_scale, 0.75), 1.8)
    nose_base_y = float(lower_center[1])
    bottom_a = np.array([lower_center_x - lower_half_width, nose_base_y], dtype=float)
    bottom_b = np.array([lower_center_x + lower_half_width, nose_base_y], dtype=float)
    return [
        Point2D(float(top_a[0]), float(top_a[1])),
        Point2D(float(top_b[0]), float(top_b[1])),
        Point2D(float(bottom_b[0]), float(bottom_b[1])),
        Point2D(float(bottom_a[0]), float(bottom_a[1])),
    ]


def _nasal_envelope_vertices(
    instance: FaceInstance,
    *,
    feature_scale: float,
    settings: dict[str, Any],
) -> list[Point2D] | None:
    """Build one continuous bridge-to-alae polygon without an hourglass waist.

    Five-point FaceMapper landmarks provide eye centres, one nose point and
    mouth corners, but no nostril contour.  The recommended nasal envelope is
    therefore a monotonic six-vertex polygon: narrow below the inner-eye line,
    wider around the nose point, and widest at the estimated nose base.  This
    covers both nasal alae while avoiding the legacy eye-wide top / narrow-base
    shape that visually resembled an ``8`` or hourglass.
    """
    if instance.left_eye is None or instance.right_eye is None:
        return None
    eye_a = np.array([instance.left_eye.x, instance.left_eye.y], dtype=float)
    eye_b = np.array([instance.right_eye.x, instance.right_eye.y], dtype=float)
    eye_span = float(np.linalg.norm(eye_b - eye_a))
    if not np.isfinite(eye_span) or eye_span < 1.0:
        return None
    eye_mid = 0.5 * (eye_a + eye_b)
    height = max(1.0, float(instance.height))
    width = max(1.0, float(instance.width))

    if instance.nose is not None:
        nose_x = float(instance.nose.x)
        nose_y = float(instance.nose.y)
    else:
        nose_x = float(eye_mid[0])
        nose_y = float(eye_mid[1] + 0.50 * max(1.0, _nose_base_y(instance, settings) - eye_mid[1]))
    base_y = float(_nose_base_y(instance, settings))
    if base_y <= float(eye_mid[1]) + 1.0:
        return None

    top_fraction = float(np.clip(settings.get("nasal_envelope_top_y_fraction", 0.16), 0.05, 0.40))
    top_y = float(eye_mid[1] + top_fraction * (base_y - float(eye_mid[1])))
    mid_y = float(np.clip(nose_y, top_y + 1.0, base_y - 1.0))

    scale = min(max(float(feature_scale), 0.75), 1.80)
    top_half = max(
        eye_span * float(settings.get("nasal_envelope_top_halfwidth_iod_ratio", 0.16)),
        width * float(settings.get("nasal_envelope_top_halfwidth_face_ratio", 0.055)),
    ) * scale
    mid_half = max(
        top_half,
        eye_span * float(settings.get("nasal_envelope_mid_halfwidth_iod_ratio", 0.22)) * scale,
        width * float(settings.get("nasal_envelope_mid_halfwidth_face_ratio", 0.075)) * scale,
    )
    mouth_span = 0.0
    if instance.mouth_left is not None and instance.mouth_right is not None:
        mouth_span = float(math.hypot(
            instance.mouth_right.x - instance.mouth_left.x,
            instance.mouth_right.y - instance.mouth_left.y,
        ))
    bottom_half = max(
        mid_half,
        eye_span * float(settings.get("nasal_envelope_bottom_halfwidth_iod_ratio", 0.30)) * scale,
        width * float(settings.get("nasal_envelope_bottom_halfwidth_face_ratio", 0.12)) * scale,
        mouth_span * float(settings.get("nasal_envelope_bottom_halfwidth_mouth_ratio", 0.42)),
    )

    # Gradually follow the observed nose centre while keeping all widths
    # monotonic (top <= middle <= base).  The horizontal base explicitly spans
    # both nostril/ala regions rather than collapsing to a narrow nose-tip box.
    bridge_x = float(0.65 * eye_mid[0] + 0.35 * nose_x)
    mid_x = float(0.25 * eye_mid[0] + 0.75 * nose_x)
    return [
        Point2D(bridge_x - top_half, top_y),
        Point2D(bridge_x + top_half, top_y),
        Point2D(mid_x + mid_half, mid_y),
        Point2D(nose_x + bottom_half, base_y),
        Point2D(nose_x - bottom_half, base_y),
        Point2D(mid_x - mid_half, mid_y),
    ]

def _face_aois(
    instance: FaceInstance,
    settings: dict[str, Any] | None = None,
) -> list[EllipseAOI | PolygonAOI]:
    settings = settings or {}
    face_conf = float(np.clip(instance.confidence, 0.0, 1.0))
    if instance.left_eye and instance.right_eye:
        iod = math.hypot(
            instance.left_eye.x - instance.right_eye.x,
            instance.left_eye.y - instance.right_eye.y,
        )
    else:
        iod = max(8.0, instance.width * 0.34)
    iod = max(8.0, iod)

    bounded_features = bool(settings.get("bounded_features", False))
    feature_scale = float(np.clip(settings.get("feature_scale", 1.0), 0.50, 2.00))
    eye_feature_scale = float(np.clip(settings.get("eye_feature_scale", 1.0), 0.75, 1.75))
    nose_region_mode = str(settings.get("nose_region_mode", "ellipse"))
    mouth_center_policy = str(settings.get("mouth_center_policy", "adaptive_lip_envelope"))
    mouth_center_offset_ratio = float(np.clip(settings.get("mouth_center_offset_ratio", 0.12), 0.0, 0.35))
    face_margin_ratio = float(np.clip(settings.get("face_margin_ratio", 0.02), 0.0, 0.20))

    if bounded_features:
        # Preserve the landmark-relative model, add face-relative floors for
        # distant faces, and allow an additional eye-specific expansion.  Every
        # resulting geometry is still clipped to the FaceMapper face box.
        effective_eye_scale = feature_scale * eye_feature_scale
        eye_rx = max(3.0, iod * settings.get("eye_radius_x_ratio", 0.18), instance.width * 0.075) * effective_eye_scale
        eye_ry = max(3.0, iod * settings.get("eye_radius_y_ratio", 0.14), instance.height * 0.045) * effective_eye_scale
        nose_rx = max(3.0, iod * settings.get("nose_radius_x_ratio", 0.16), instance.width * 0.070) * feature_scale
        nose_ry = max(4.0, iod * settings.get("nose_radius_y_ratio", 0.22), instance.height * 0.090) * feature_scale
        mouth_ry = max(4.0, iod * settings.get("mouth_radius_y_ratio", 0.16), instance.height * 0.060) * feature_scale
    else:
        # Exact legacy geometry for callers that do not opt into the new Face
        # AOI sizing policy (participant association, pre-trim diagnostics, etc.).
        eye_rx = max(3.0, iod * settings.get("eye_radius_x_ratio", 0.18))
        eye_ry = max(3.0, iod * settings.get("eye_radius_y_ratio", 0.14))
        nose_rx = max(3.0, iod * settings.get("nose_radius_x_ratio", 0.16))
        nose_ry = max(4.0, iod * settings.get("nose_radius_y_ratio", 0.22))
        mouth_ry = max(4.0, iod * settings.get("mouth_radius_y_ratio", 0.16))

    rows: list[EllipseAOI | PolygonAOI] = []

    def add_feature(
        region: str,
        center: Point2D,
        rx: float,
        ry: float,
        confidence: float,
        *,
        audit_scale: float | None = None,
    ) -> None:
        if bounded_features:
            rows.append(_bounded_feature_aoi(
                region=region,
                center=center,
                desired_radius_x=rx,
                desired_radius_y=ry,
                instance=instance,
                confidence=confidence,
                feature_scale=float(audit_scale if audit_scale is not None else feature_scale),
                margin_ratio=face_margin_ratio,
            ))
        else:
            rows.append(EllipseAOI(region, center, rx, ry, confidence))

    if instance.left_eye:
        add_feature(
            SemanticRegion.LEFT_EYE.value,
            instance.left_eye,
            eye_rx,
            eye_ry,
            face_conf,
            audit_scale=effective_eye_scale if bounded_features else 1.0,
        )
    if instance.right_eye:
        add_feature(
            SemanticRegion.RIGHT_EYE.value,
            instance.right_eye,
            eye_rx,
            eye_ry,
            face_conf,
            audit_scale=effective_eye_scale if bounded_features else 1.0,
        )
    if bounded_features and nose_region_mode in {"nasal_envelope", "bridge_polygon"}:
        if nose_region_mode == "nasal_envelope":
            nose_vertices = _nasal_envelope_vertices(
                instance,
                feature_scale=feature_scale,
                settings=settings,
            )
            nose_geometry_model = "nasal_bridge_alar_envelope"
        else:
            nose_vertices = _nose_bridge_vertices(
                instance,
                feature_scale=feature_scale,
                settings=settings,
            )
            nose_geometry_model = "eye_line_to_nose_base_bridge_legacy"
        if nose_vertices:
            rows.append(_bounded_polygon_aoi(
                region=SemanticRegion.NOSE.value,
                vertices=nose_vertices,
                instance=instance,
                confidence=face_conf * 0.96,
                feature_scale=feature_scale,
                margin_ratio=face_margin_ratio,
                geometry_model=nose_geometry_model,
            ))
        elif instance.nose:
            add_feature(SemanticRegion.NOSE.value, instance.nose, nose_rx, nose_ry, face_conf * 0.96)
    elif instance.nose:
        add_feature(SemanticRegion.NOSE.value, instance.nose, nose_rx, nose_ry, face_conf * 0.96)
    if bounded_features and mouth_center_policy == "adaptive_lip_envelope":
        mouth_vertices = _adaptive_lip_envelope_vertices(instance, settings=settings)
        if mouth_vertices:
            rows.append(_bounded_polygon_aoi(
                region=SemanticRegion.MOUTH.value,
                vertices=mouth_vertices,
                instance=instance,
                confidence=face_conf * 0.97,
                feature_scale=1.0,
                margin_ratio=face_margin_ratio,
                geometry_model="adaptive_lip_envelope",
            ))
    elif bounded_features and mouth_center_policy == "anatomical_strict":
        strict_geometry = _strict_lip_ellipse(instance, settings=settings)
        if strict_geometry is not None:
            center, radius_x, radius_y = strict_geometry
            add_feature(
                SemanticRegion.MOUTH.value,
                center,
                radius_x,
                radius_y,
                face_conf * 0.97,
                audit_scale=1.0,
            )
    elif bounded_features and mouth_center_policy in {"speech_sensitive_mouth_jaw", "nose_base_to_face_bottom"}:
        rows.append(_bounded_polygon_aoi(
            region=SemanticRegion.MOUTH.value,
            vertices=_lower_face_mouth_vertices(instance, settings=settings),
            instance=instance,
            confidence=face_conf * 0.97,
            feature_scale=1.0,
            margin_ratio=0.0,
            geometry_model="speech_sensitive_mouth_jaw",
        ))
    elif instance.mouth_left and instance.mouth_right:
        raw_center = Point2D(
            (instance.mouth_left.x + instance.mouth_right.x) / 2.0,
            (instance.mouth_left.y + instance.mouth_right.y) / 2.0,
        )
        # Legacy ellipse modes are retained for reproducibility.
        center = raw_center
        if bounded_features and mouth_center_policy == "lowered_landmark_midpoint":
            if instance.nose is not None:
                nose_to_mouth_y = max(0.0, float(raw_center.y - instance.nose.y))
                downward_shift = min(
                    max(1.0, nose_to_mouth_y * mouth_center_offset_ratio),
                    max(1.0, instance.height * 0.035),
                )
            else:
                downward_shift = max(1.0, instance.height * 0.025)
            center = Point2D(raw_center.x, raw_center.y + downward_shift)
        mouth_width = math.hypot(
            instance.mouth_left.x - instance.mouth_right.x,
            instance.mouth_left.y - instance.mouth_right.y,
        )
        mouth_rx = max(
            4.0,
            mouth_width * settings.get("mouth_width_multiplier", 0.68),
            instance.width * 0.130 if bounded_features else 0.0,
        ) * (feature_scale if bounded_features else 1.0)
        add_feature(SemanticRegion.MOUTH.value, center, mouth_rx, mouth_ry, face_conf * 0.97)
    elif instance.nose and mouth_center_policy in {"landmark_midpoint", "lowered_landmark_midpoint"}:
        fallback_offset = 0.44 if bounded_features and mouth_center_policy == "lowered_landmark_midpoint" else 0.38
        center = Point2D(instance.nose.x, instance.nose.y + fallback_offset * iod)
        mouth_rx = max(4.0, iod * 0.27, instance.width * 0.130 if bounded_features else 0.0)
        if bounded_features:
            mouth_rx *= feature_scale
        add_feature(SemanticRegion.MOUTH.value, center, mouth_rx, mouth_ry, face_conf * 0.72)

    # Preserve exact legacy full-face geometry for non-Face-AOI callers. In
    # bounded Face-AOI mode, the accepted green FaceMapper rectangle is the
    # *authoritative* face-membership gate. Fine features remain bounded inside
    # it, while the catch-all other-face region covers the complete detector
    # box, including its boundary. A gaze point outside this rectangle can never
    # be promoted to a face by projection, FaceMapper flags, or temporal context.
    if bounded_features:
        left, right, top, bottom = (
            float(instance.x1), float(instance.x2),
            float(instance.y1), float(instance.y2),
        )
        rows.append(_bounded_polygon_aoi(
            region=SemanticRegion.OTHER_FACE.value,
            vertices=[
                Point2D(left, top),
                Point2D(right, top),
                Point2D(right, bottom),
                Point2D(left, bottom),
            ],
            instance=instance,
            confidence=face_conf * 0.88,
            feature_scale=1.0,
            margin_ratio=0.0,
            geometry_model="face_bbox_exact_rectangle",
        ))
    else:
        face_rx = max(4.0, instance.width * settings.get("face_radius_x_ratio", 0.50))
        face_ry = max(4.0, instance.height * settings.get("face_radius_y_ratio", 0.52))
        rows.append(EllipseAOI(
            SemanticRegion.OTHER_FACE.value,
            instance.center,
            face_rx,
            face_ry,
            face_conf * 0.88,
            requested_radius_x=face_rx,
            requested_radius_y=face_ry,
            geometry_clipped=False,
            containment_policy=None,
            feature_scale=1.0,
        ))
    return rows


def face_instances_to_regions(
    instances_by_frame: dict[int, list[FaceInstance]],
    *,
    aoi_settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build display/analysis regions from the *current* face geometry.

    Face AOI may canonicalize or mirror landmark geometry after loading the
    FaceMapper table.  Rebuilding the region table from those corrected
    instances prevents the overlay from showing stale, pre-correction
    coordinates while the sample classifier uses corrected coordinates.
    """
    rows: list[dict[str, Any]] = []
    for frame_index in sorted(instances_by_frame):
        for item in instances_by_frame[frame_index]:
            for aoi in _face_aois(item, aoi_settings):
                if isinstance(aoi, PolygonAOI):
                    center = aoi.center
                    geometry_type = "polygon"
                    geometry = {
                        "points": [[point.x, point.y] for point in aoi.vertices],
                        "center_x": center.x,
                        "center_y": center.y,
                        "geometry_model": aoi.geometry_model,
                    }
                    requested_radius_x = None
                    requested_radius_y = None
                    requested_geometry = {
                        "points": [[point.x, point.y] for point in (aoi.requested_vertices or aoi.vertices)],
                        "geometry_model": aoi.geometry_model,
                    }
                    contained = all(
                        item.x1 - 1e-6 <= point.x <= item.x2 + 1e-6
                        and item.y1 - 1e-6 <= point.y <= item.y2 + 1e-6
                        for point in aoi.vertices
                    )
                    vertex_count = len(aoi.vertices)
                    geometry_model = aoi.geometry_model
                else:
                    geometry_type = "ellipse"
                    geometry = {
                        "center_x": aoi.center.x,
                        "center_y": aoi.center.y,
                        "radius_x": aoi.radius_x,
                        "radius_y": aoi.radius_y,
                    }
                    requested_radius_x = aoi.requested_radius_x
                    requested_radius_y = aoi.requested_radius_y
                    requested_geometry = {
                        "center_x": aoi.center.x,
                        "center_y": aoi.center.y,
                        "radius_x": aoi.requested_radius_x,
                        "radius_y": aoi.requested_radius_y,
                    }
                    contained = bool(
                        aoi.center.x - aoi.radius_x >= item.x1 - 1e-6
                        and aoi.center.x + aoi.radius_x <= item.x2 + 1e-6
                        and aoi.center.y - aoi.radius_y >= item.y1 - 1e-6
                        and aoi.center.y + aoi.radius_y <= item.y2 + 1e-6
                    )
                    vertex_count = 0
                    geometry_model = "bounded_ellipse" if aoi.containment_policy else "ellipse"
                rows.append({
                    "scene_frame_index": item.scene_frame_index,
                    "scene_timestamp_ns": item.scene_timestamp_ns,
                    "entity_id": item.entity_id,
                    "region": aoi.region,
                    "coordinate_space": "image_px",
                    "geometry_type": geometry_type,
                    "geometry_json": json.dumps(geometry, separators=(",", ":"), allow_nan=False),
                    "requested_geometry_json": json.dumps(requested_geometry, separators=(",", ":"), allow_nan=False),
                    "requested_radius_x": requested_radius_x,
                    "requested_radius_y": requested_radius_y,
                    "aoi_geometry_clipped": bool(aoi.geometry_clipped),
                    "aoi_containment_policy": aoi.containment_policy,
                    "aoi_feature_scale": float(aoi.feature_scale),
                    "aoi_geometry_model": geometry_model,
                    "aoi_polygon_vertex_count": int(vertex_count),
                    "face_bbox_x1": float(item.x1),
                    "face_bbox_y1": float(item.y1),
                    "face_bbox_x2": float(item.x2),
                    "face_bbox_y2": float(item.y2),
                    "aoi_contained_in_face_bbox": contained,
                    "source": "face_mapper_landmarks" if aoi.region != SemanticRegion.OTHER_FACE.value else "face_mapper_face_box",
                    "source_asset_id": None,
                    "confidence": aoi.confidence,
                    "priority": 100 if aoi.region != SemanticRegion.OTHER_FACE.value else 60,
                    "face_track_id": item.track_id,
                    "face_source_row_index": item.source_row_index,
                })
    return pd.DataFrame(rows)


def load_face_semantics(
    root: Path,
    *,
    recording_id: str | None,
    scene_timestamps_ns: np.ndarray,
    width: int,
    height: int,
    tolerance_ms: float = 8.0,
    aoi_settings: dict[str, Any] | None = None,
    left_right_policy: Literal["anatomical_auto", "as_exported", "swap_eyes"] = "anatomical_auto",
) -> FaceSemanticResult:
    if left_right_policy not in {"anatomical_auto", "as_exported", "swap_eyes"}:
        raise ValidationError("left_right_policy must be anatomical_auto, as_exported or swap_eyes")
    files = find_facemapper_files(root)
    if not files["detections"]:
        raise ValidationError("Face Mapper asset contains no face_detections.csv or face_positions.csv")
    source = root / files["detections"][0]
    frame = pd.read_csv(source, low_memory=False)
    timestamp_col = _column(frame.columns, ["timestamp [ns]", "timestamp_ns", "world timestamp [ns]"], required=True)
    recording_col = _column(frame.columns, ["recording id", "recording_id"])
    if recording_col and recording_id:
        frame = frame.loc[frame[recording_col].astype(str) == str(recording_id)].copy()
    x1_col = _column(frame.columns, ["p1 x [px]", "p1_x_px", "x1", "left"], required=True)
    y1_col = _column(frame.columns, ["p1 y [px]", "p1_y_px", "y1", "top"], required=True)
    x2_col = _column(frame.columns, ["p2 x [px]", "p2_x_px", "x2", "right"], required=True)
    y2_col = _column(frame.columns, ["p2 y [px]", "p2_y_px", "y2", "bottom"], required=True)
    confidence_col = _column(frame.columns, ["confidence", "face confidence", "score"])
    entity_col = _column(frame.columns, ["participant id", "participant_id", "person id", "person_id", "entity id", "entity_id"])
    left_eye_cols = _point_columns(frame.columns, ["left eye", "left_eye", "eye left"])
    right_eye_cols = _point_columns(frame.columns, ["right eye", "right_eye", "eye right"])
    nose_cols = _point_columns(frame.columns, ["nose", "nose tip", "nose_tip"])
    mouth_left_cols = _point_columns(frame.columns, ["left mouth corner", "mouth left", "left mouth", "mouth_left"])
    mouth_right_cols = _point_columns(frame.columns, ["right mouth corner", "mouth right", "right mouth", "mouth_right"])

    auto_swap = False
    orientation_diagnostics: dict[str, Any] = {
        "evidence_count": 0,
        "median_exported_left_minus_right_x_px": None,
        "reason": "policy_not_auto",
    }
    if left_right_policy == "anatomical_auto":
        auto_swap, orientation_diagnostics = _infer_anatomical_swap(
            frame, left_eye_cols, right_eye_cols, width, height
        )
    apply_bilateral_swap = left_right_policy == "swap_eyes" or auto_swap

    numeric_timestamps = pd.to_numeric(frame[timestamp_col], errors="coerce")
    valid_timestamp = numeric_timestamps.notna()
    frame = frame.loc[valid_timestamp].copy()
    query = numeric_timestamps.loc[valid_timestamp].astype("int64").to_numpy()
    nearest, error = _nearest_indices(np.asarray(scene_timestamps_ns, dtype=np.int64), query)
    accepted = error <= int(round(tolerance_ms * 1e6))
    instances: list[FaceInstance] = []
    invalid_rows = 0
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        if not accepted[ordinal]:
            continue
        try:
            coords = [float(row[x1_col]), float(row[y1_col]), float(row[x2_col]), float(row[y2_col])]
        except Exception:
            invalid_rows += 1
            continue
        if not np.isfinite(coords).all():
            invalid_rows += 1
            continue
        if max(abs(value) for value in coords) <= 2.0:
            coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            invalid_rows += 1
            continue
        confidence = 1.0
        if confidence_col:
            try:
                confidence = float(row[confidence_col])
            except Exception:
                pass
        explicit_entity = None
        if entity_col and pd.notna(row[entity_col]):
            explicit_entity = str(row[entity_col]).strip() or None
        frame_index = int(nearest[ordinal])
        left_eye = _point(row, *left_eye_cols, width, height)
        right_eye = _point(row, *right_eye_cols, width, height)
        mouth_left = _point(row, *mouth_left_cols, width, height)
        mouth_right = _point(row, *mouth_right_cols, width, height)
        if apply_bilateral_swap:
            left_eye, right_eye = right_eye, left_eye
            mouth_left, mouth_right = mouth_right, mouth_left
        instances.append(FaceInstance(
            scene_frame_index=frame_index,
            scene_timestamp_ns=int(scene_timestamps_ns[frame_index]),
            track_id=-1,
            explicit_entity_id=explicit_entity,
            x1=x1, y1=y1, x2=x2, y2=y2,
            left_eye=left_eye,
            right_eye=right_eye,
            nose=_point(row, *nose_cols, width, height),
            mouth_left=mouth_left,
            mouth_right=mouth_right,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            source_row_index=int(row.name) if isinstance(row.name, (int, np.integer)) else ordinal,
        ))
    instances = _assign_tracks(instances)
    by_frame: dict[int, list[FaceInstance]] = {}
    for item in instances:
        by_frame.setdefault(item.scene_frame_index, []).append(item)
    regions = face_instances_to_regions(by_frame, aoi_settings=aoi_settings)
    landmark_rows = int((regions.get("region", pd.Series(dtype=str)) != SemanticRegion.OTHER_FACE.value).sum()) if len(regions) else 0
    diagnostics = {
        "source_file": str(source),
        "input_row_count": int(len(frame)),
        "accepted_face_rows": int(len(instances)),
        "unmatched_timestamp_rows": int((~accepted).sum()),
        "invalid_geometry_rows": int(invalid_rows),
        "frame_count_with_faces": int(len(by_frame)),
        "track_count": int(len({item.track_id for item in instances})),
        "fine_landmark_region_rows": int(landmark_rows),
        "left_right_policy": left_right_policy,
        "left_right_output_convention": "anatomical_subject" if left_right_policy != "as_exported" else "as_exported",
        "left_right_swap_applied": bool(apply_bilateral_swap),
        "left_right_orientation": orientation_diagnostics,
        "aoi_settings": dict(aoi_settings or {}),
        "fine_feature_aoi_scale": float((aoi_settings or {}).get("feature_scale", 1.0)),
        "fine_feature_eye_scale": float((aoi_settings or {}).get("eye_feature_scale", 1.0)),
        "fine_feature_nose_region_mode": str((aoi_settings or {}).get("nose_region_mode", "ellipse")),
        "fine_feature_containment_policy": (
            "face_bbox_inner_clip" if (aoi_settings or {}).get("bounded_features") else "legacy_unbounded"
        ),
        "landmark_columns": {
            "left_eye": list(left_eye_cols),
            "right_eye": list(right_eye_cols),
            "nose": list(nose_cols),
            "mouth_left": list(mouth_left_cols),
            "mouth_right": list(mouth_right_cols),
        },
    }
    return FaceSemanticResult(
        instances_by_frame=by_frame,
        frame_regions=regions,
        diagnostics=diagnostics,
    )


def _face_edge_tolerance_px(instance: FaceInstance, settings: dict[str, Any] | None) -> float:
    """Return the bounded uncertainty band around an accepted green face box.

    The visible green rectangle remains the anatomical face reference.  The band
    exists only to absorb gaze localisation / display-rounding error at its edge;
    it never enlarges the eye, nose, or mouth AOIs.
    """
    settings = settings or {}
    policy = str(settings.get("face_edge_tolerance_policy", "exact"))
    if policy == "exact":
        return 0.0
    ratio = float(settings.get("face_edge_tolerance_ratio", 0.0))
    minimum = float(settings.get("face_edge_tolerance_min_px", 0.0))
    maximum = float(settings.get("face_edge_tolerance_max_px", max(minimum, 0.0)))
    smaller_dimension = max(0.0, min(float(instance.width), float(instance.height)))
    tolerance = max(minimum, ratio * smaller_dimension)
    return float(np.clip(tolerance, 0.0, max(0.0, maximum)))


def _outside_face_box_distance_px(x: float, y: float, instance: FaceInstance) -> float:
    """Chebyshev distance to the exact green rectangle (0 on/inside boundary)."""
    dx = max(float(instance.x1) - float(x), 0.0, float(x) - float(instance.x2))
    dy = max(float(instance.y1) - float(y), 0.0, float(y) - float(instance.y2))
    return float(max(dx, dy))


def classify_face_point(
    x_px: float,
    y_px: float,
    instances: list[FaceInstance],
    *,
    aoi_settings: dict[str, Any] | None = None,
    edge_eligible_source_row_indices: set[int] | None = None,
) -> FaceHit | None:
    if not np.isfinite([x_px, y_px]).all():
        return None
    settings = aoi_settings or {}
    candidates: list[tuple[float, int, FaceHit]] = []
    region_priority = {
        SemanticRegion.LEFT_EYE.value: 0,
        SemanticRegion.RIGHT_EYE.value: 0,
        SemanticRegion.MOUTH.value: 1,
        SemanticRegion.NOSE.value: 2,
        SemanticRegion.OTHER_FACE.value: 3,
    }
    for instance in instances:
        for aoi in _face_aois(instance, settings):
            distance = aoi.normalized_distance(x_px, y_px)
            if distance > 1.0:
                continue
            proximity = max(0.0, 1.0 - distance)
            confidence = float(np.clip(aoi.confidence * (0.72 + 0.28 * proximity), 0.0, 1.0))
            hit = FaceHit(
                entity_id=instance.entity_id,
                entity_is_explicit=instance.explicit_entity_id is not None,
                track_id=instance.track_id,
                region=aoi.region,
                confidence=confidence,
                normalized_distance=float(distance),
                scene_frame_index=instance.scene_frame_index,
                source_row_index=instance.source_row_index,
                gate_hit_type="exact_green_box",
            )
            candidates.append((distance, region_priority[aoi.region], hit))
    if candidates:
        return min(candidates, key=lambda row: (row[1], row[0]))[2]

    # A point whose centre is only a few pixels outside the rendered green line
    # can still represent a face fixation because gaze localisation, video scaling,
    # and line thickness are not exact.  Apply a bounded coarse-face band only
    # after all exact eye/nose/mouth/face tests failed.
    if not bool(settings.get("bounded_features", False)):
        return None
    edge_candidates: list[tuple[float, float, FaceHit]] = []
    eligible = edge_eligible_source_row_indices
    for instance in instances:
        if eligible is not None and int(instance.source_row_index) not in eligible:
            continue
        tolerance = _face_edge_tolerance_px(instance, settings)
        if tolerance <= 0.0:
            continue
        edge_distance = _outside_face_box_distance_px(x_px, y_px, instance)
        if edge_distance <= 0.0 or edge_distance > tolerance + 1e-9:
            continue
        edge_fraction = float(np.clip(edge_distance / tolerance, 0.0, 1.0))
        center_scale = max(1.0, 0.5 * (float(instance.width) + float(instance.height)))
        center_distance = math.hypot(float(x_px) - instance.center.x, float(y_px) - instance.center.y) / center_scale
        confidence = float(np.clip(instance.confidence * (0.82 - 0.24 * edge_fraction), 0.0, 0.90))
        hit = FaceHit(
            entity_id=instance.entity_id,
            entity_is_explicit=instance.explicit_entity_id is not None,
            track_id=instance.track_id,
            region=SemanticRegion.OTHER_FACE.value,
            confidence=confidence,
            normalized_distance=1.0 + edge_fraction,
            scene_frame_index=instance.scene_frame_index,
            source_row_index=instance.source_row_index,
            gate_hit_type="green_box_edge_tolerance",
            edge_distance_px=edge_distance,
            edge_tolerance_px=tolerance,
        )
        edge_candidates.append((edge_fraction, center_distance, hit))
    if not edge_candidates:
        return None
    edge_candidates.sort(key=lambda row: (row[0], row[1]))
    best = edge_candidates[0][2]
    ambiguous = False
    if len(edge_candidates) > 1:
        first = edge_candidates[0]
        second = edge_candidates[1]
        # Two expanded edge bands are treated as ambiguous only when their
        # normalised edge distances are nearly indistinguishable.
        ambiguous = (second[0] - first[0]) < 0.12 and abs(second[1] - first[1]) < 0.10
    return replace(
        best,
        edge_candidate_count=len(edge_candidates),
        edge_ambiguous=bool(ambiguous),
    )
