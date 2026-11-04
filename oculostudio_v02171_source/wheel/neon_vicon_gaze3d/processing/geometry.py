from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class GeometryError(RuntimeError):
    pass


def normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return np.divide(vector, norm, out=np.full_like(vector, np.nan), where=norm > 0)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    points = np.asarray(points, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_fit(
    template: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if valid is None:
        valid = np.isfinite(template).all(axis=1) & np.isfinite(observed).all(axis=1)
    else:
        valid = (
            np.asarray(valid, dtype=bool)
            & np.isfinite(template).all(axis=1)
            & np.isfinite(observed).all(axis=1)
        )
    count = int(valid.sum())
    if count < 3:
        raise GeometryError("At least three visible markers are required for a rigid fit")
    source = template[valid]
    target = observed[valid]
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    predicted = source @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    return rotation, translation, rmse, count


def build_marker_template(
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    iterations: int = 3,
) -> np.ndarray:
    """Build a stable local marker template using generalized Procrustes."""

    positions = np.asarray(positions, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    complete = np.flatnonzero(valid.sum(axis=1) >= 3)
    if len(complete) == 0:
        raise GeometryError("No frame contains at least three valid markers")
    best = complete[np.argmax(valid[complete].sum(axis=1))]
    initial = positions[best].copy()
    center = np.nanmean(initial[valid[best]], axis=0)
    template = initial - center
    # Missing initial markers are recovered from other frames during iteration.
    for _ in range(iterations):
        recovered: list[list[np.ndarray]] = [[] for _ in range(positions.shape[1])]
        for frame in complete:
            mask = valid[frame] & np.isfinite(template).all(axis=1)
            if mask.sum() < 3:
                continue
            try:
                rotation, translation, _, _ = rigid_fit(template, positions[frame], mask)
            except GeometryError:
                continue
            local = (positions[frame] - translation) @ rotation
            for marker in np.flatnonzero(valid[frame]):
                recovered[marker].append(local[marker])
        for marker, samples in enumerate(recovered):
            if samples:
                template[marker] = np.median(np.asarray(samples), axis=0)
        template -= np.nanmean(template, axis=0)
    if not np.isfinite(template).all():
        raise GeometryError("Could not recover all marker positions in the template")
    return template


def make_planar_board_template(
    median_world_points: np.ndarray,
    *,
    top_left_index: int,
    top_right_index: int,
    bottom_left_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return board-local coordinates and T_world_from_board for a reference pose."""

    points = np.asarray(median_world_points, dtype=float)
    origin = points[top_left_index]
    x_axis = normalize(points[top_right_index] - origin)
    down_hint = points[bottom_left_index] - origin
    normal = normalize(np.cross(x_axis, down_hint))
    y_axis = normalize(np.cross(normal, x_axis))
    if np.dot(y_axis, down_hint) < 0:
        y_axis *= -1
        normal *= -1
    rotation_world_from_board = np.column_stack([x_axis, y_axis, normal])
    local = (points - origin) @ rotation_world_from_board
    # Marker centres should be coplanar.  Retain the small measured Z offset for
    # fidelity rather than forcing exactly zero.
    transform = make_transform(rotation_world_from_board, origin)
    return local, transform


@dataclass
class PoseSeries:
    time_sec: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    valid: np.ndarray
    fit_rmse_mm: np.ndarray
    marker_count: np.ndarray

    @classmethod
    def from_markers(
        cls,
        time_sec: np.ndarray,
        template: np.ndarray,
        positions: np.ndarray,
        marker_valid: np.ndarray,
        *,
        min_markers: int = 3,
        max_fit_rmse_mm: float | None = None,
    ) -> "PoseSeries":
        frame_count = len(time_sec)
        rotations = np.full((frame_count, 3, 3), np.nan, dtype=float)
        translations = np.full((frame_count, 3), np.nan, dtype=float)
        rmse = np.full(frame_count, np.nan, dtype=float)
        count = marker_valid.sum(axis=1).astype(int)
        valid_pose = np.zeros(frame_count, dtype=bool)
        for frame in range(frame_count):
            if count[frame] < min_markers:
                continue
            try:
                rotation, translation, error, used = rigid_fit(
                    template, positions[frame], marker_valid[frame]
                )
            except GeometryError:
                continue
            if max_fit_rmse_mm is not None and error > max_fit_rmse_mm:
                continue
            rotations[frame] = rotation
            translations[frame] = translation
            rmse[frame] = error
            count[frame] = used
            valid_pose[frame] = True
        return cls(
            time_sec=np.asarray(time_sec, dtype=float),
            rotations=rotations,
            translations=translations,
            valid=valid_pose,
            fit_rmse_mm=rmse,
            marker_count=count,
        )

    def interpolate(
        self, query_time_sec: np.ndarray, *, max_gap_sec: float = 0.1
    ) -> "InterpolatedPose":
        query = np.asarray(query_time_sec, dtype=float)
        valid_indices = np.flatnonzero(self.valid)
        n = len(query)
        rotations = np.full((n, 3, 3), np.nan, dtype=float)
        translations = np.full((n, 3), np.nan, dtype=float)
        pose_valid = np.zeros(n, dtype=bool)
        interpolated = np.zeros(n, dtype=bool)
        gap = np.full(n, np.nan, dtype=float)
        rmse = np.full(n, np.nan, dtype=float)
        marker_count = np.zeros(n, dtype=int)
        if len(valid_indices) < 2:
            return InterpolatedPose(
                query, rotations, translations, pose_valid, interpolated, gap, rmse, marker_count
            )
        valid_time = self.time_sec[valid_indices]
        slerp = Slerp(valid_time, Rotation.from_matrix(self.rotations[valid_indices]))
        inside = (query >= valid_time[0]) & (query <= valid_time[-1]) & np.isfinite(query)
        inside_indices = np.flatnonzero(inside)
        if len(inside_indices):
            qi = query[inside_indices]
            right_pos = np.searchsorted(valid_time, qi, side="left")
            right_pos = np.clip(right_pos, 0, len(valid_time) - 1)
            left_pos = np.maximum(0, right_pos - 1)
            exact = np.isclose(valid_time[right_pos], qi, rtol=0.0, atol=1e-10)
            left_pos[exact] = right_pos[exact]
            left_t = valid_time[left_pos]
            right_t = valid_time[right_pos]
            local_gap = right_t - left_t
            local_gap[exact] = 0.0
            allowed = exact | (local_gap <= max_gap_sec)
            selected = inside_indices[allowed]
            if len(selected):
                selected_time = query[selected]
                rotations[selected] = slerp(selected_time).as_matrix()
                for axis in range(3):
                    translations[selected, axis] = np.interp(
                        selected_time,
                        valid_time,
                        self.translations[valid_indices, axis],
                    )
                pose_valid[selected] = True
                interpolated[selected] = ~exact[allowed]
                gap[selected] = local_gap[allowed]
                nearest_right = right_pos[allowed]
                nearest_left = left_pos[allowed]
                choose_right = np.abs(valid_time[nearest_right] - selected_time) < np.abs(
                    selected_time - valid_time[nearest_left]
                )
                nearest = np.where(choose_right, nearest_right, nearest_left)
                source_indices = valid_indices[nearest]
                rmse[selected] = self.fit_rmse_mm[source_indices]
                marker_count[selected] = self.marker_count[source_indices]
        return InterpolatedPose(
            query,
            rotations,
            translations,
            pose_valid,
            interpolated,
            gap,
            rmse,
            marker_count,
        )


@dataclass
class InterpolatedPose:
    time_sec: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    valid: np.ndarray
    interpolated: np.ndarray
    interpolation_gap_sec: np.ndarray
    fit_rmse_mm: np.ndarray
    marker_count: np.ndarray

    def transforms(self) -> np.ndarray:
        transforms = np.full((len(self.time_sec), 4, 4), np.nan, dtype=float)
        transforms[:, 3, 3] = 1.0
        transforms[:, :3, :3] = self.rotations
        transforms[:, :3, 3] = self.translations
        return transforms


def average_transforms(
    transforms: np.ndarray,
    *,
    translation_outlier_mm: float = 20.0,
    rotation_outlier_deg: float = 5.0,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.asarray(transforms, dtype=float)
    valid = np.isfinite(transforms).all(axis=(1, 2))
    if valid.sum() == 0:
        raise GeometryError("No finite transforms to average")
    mask = valid.copy()
    for _ in range(iterations):
        rotation = Rotation.from_matrix(transforms[mask, :3, :3]).mean()
        translation = np.median(transforms[mask, :3, 3], axis=0)
        rotation_error = (
            rotation.inv() * Rotation.from_matrix(transforms[:, :3, :3])
        ).magnitude()
        rotation_error = np.degrees(rotation_error)
        translation_error = np.linalg.norm(transforms[:, :3, 3] - translation, axis=1)
        new_mask = valid & (rotation_error <= rotation_outlier_deg) & (
            translation_error <= translation_outlier_mm
        )
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    mean_rotation = Rotation.from_matrix(transforms[mask, :3, :3]).mean().as_matrix()
    mean_translation = np.median(transforms[mask, :3, 3], axis=0)
    return make_transform(mean_rotation, mean_translation), mask


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = Rotation.from_matrix(rotation_a).inv() * Rotation.from_matrix(rotation_b)
    return float(np.degrees(delta.magnitude()))


def angular_speed_from_rotations(rotations: np.ndarray, time_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)
    if len(rotations) < 2:
        return np.empty(0), np.empty(0)
    delta = Rotation.from_matrix(rotations[:-1]).inv() * Rotation.from_matrix(rotations[1:])
    dt = np.diff(time_sec)
    speed = np.divide(delta.magnitude(), dt, out=np.full_like(dt, np.nan), where=dt > 0)
    mid = (time_sec[:-1] + time_sec[1:]) / 2.0
    return mid, speed


def geometric_median(
    points: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 256,
) -> np.ndarray:
    """Return the Euclidean geometric median of finite 3-D points.

    The geometric median is substantially less sensitive to a few unstable PnP
    translations than a component-wise mean.  It also returns an actual vector,
    unlike averaging only the candidate vector lengths.
    """

    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise GeometryError("No finite points for geometric median")
    estimate = np.median(points, axis=0)
    for _ in range(max_iterations):
        distance = np.linalg.norm(points - estimate, axis=1)
        coincident = np.flatnonzero(distance <= tolerance)
        if len(coincident):
            return points[int(coincident[0])].copy()
        weights = 1.0 / np.maximum(distance, tolerance)
        updated = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) <= tolerance:
            return updated
        estimate = updated
    return estimate


def robust_average_transforms(
    transforms: np.ndarray,
    *,
    minimum_translation_gate_mm: float = 5.0,
    minimum_rotation_gate_deg: float = 0.5,
    maximum_translation_gate_mm: float = 50.0,
    maximum_rotation_gate_deg: float = 10.0,
    mad_multiplier: float = 3.5,
    iterations: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robustly aggregate repeated estimates of one fixed SE(3) transform.

    Translation is estimated with a geometric median and rotation with the
    intrinsic SciPy rotation mean.  Candidate outliers are rejected using
    data-adaptive MAD gates.  Crucially, this function never optimizes camera
    translation against reprojection error, avoiding centimetre-scale drift in
    distant, nearly static planar calibration recordings.
    """

    transforms = np.asarray(transforms, dtype=float)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise GeometryError("Transforms must have shape (N, 4, 4)")
    finite = np.isfinite(transforms).all(axis=(1, 2))
    if int(finite.sum()) < 3:
        raise GeometryError("At least three finite transforms are required")
    mask = finite.copy()

    def robust_scale(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return 1.4826 * mad

    translation_gate = float(maximum_translation_gate_mm)
    rotation_gate = float(maximum_rotation_gate_deg)
    for _ in range(iterations):
        indices = np.flatnonzero(mask)
        if len(indices) < 3:
            break
        center_t = geometric_median(transforms[indices, :3, 3])
        center_r = Rotation.from_matrix(transforms[indices, :3, :3]).mean()
        translation_error = np.linalg.norm(
            transforms[:, :3, 3] - center_t, axis=1
        )
        rotation_error = np.degrees(
            (center_r.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()
        )
        selected_t = translation_error[mask]
        selected_r = rotation_error[mask]
        translation_gate = min(
            maximum_translation_gate_mm,
            max(
                minimum_translation_gate_mm,
                float(np.median(selected_t))
                + mad_multiplier * robust_scale(selected_t),
            ),
        )
        rotation_gate = min(
            maximum_rotation_gate_deg,
            max(
                minimum_rotation_gate_deg,
                float(np.median(selected_r))
                + mad_multiplier * robust_scale(selected_r),
            ),
        )
        updated = finite & (translation_error <= translation_gate) & (
            rotation_error <= rotation_gate
        )
        if int(updated.sum()) < 3:
            break
        if np.array_equal(updated, mask):
            mask = updated
            break
        mask = updated

    indices = np.flatnonzero(mask)
    center_t = geometric_median(transforms[indices, :3, 3])
    center_r_object = Rotation.from_matrix(transforms[indices, :3, :3]).mean()
    center_r = center_r_object.as_matrix()
    translation_error = np.linalg.norm(transforms[:, :3, 3] - center_t, axis=1)
    rotation_error = np.degrees(
        (center_r_object.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()
    )
    candidate_distance = np.linalg.norm(transforms[:, :3, 3], axis=1)

    def stats(values: np.ndarray, selected: np.ndarray | None = None) -> dict:
        values = np.asarray(values, dtype=float)
        if selected is not None:
            values = values[selected]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {"n": 0}
        return {
            "n": int(len(values)),
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    metrics = {
        "estimator": "robust_multiframe_se3_geometric_median_v2",
        "candidate_count": int(finite.sum()),
        "inlier_count": int(mask.sum()),
        "inlier_fraction": float(mask.sum() / finite.sum()),
        "translation_gate_mm": float(translation_gate),
        "rotation_gate_deg": float(rotation_gate),
        "camera_origin_tracker_mm": center_t.tolist(),
        "profile_transform_distance_from_tracker_centroid_mm": float(
            np.linalg.norm(center_t)
        ),
        "candidate_camera_distance_mm_all": stats(candidate_distance, finite),
        "candidate_camera_distance_mm_inliers": stats(candidate_distance, mask),
        "candidate_translation_residual_mm": stats(translation_error, mask),
        "candidate_rotation_residual_deg": stats(rotation_error, mask),
        "translation_component_std_mm_inliers": np.std(
            transforms[mask, :3, 3], axis=0
        ).tolist(),
    }
    return make_transform(center_r, center_t), mask, metrics
