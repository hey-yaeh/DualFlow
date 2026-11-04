from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from oculostudio.errors import ValidationError


def normalize(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return np.divide(vector, norm, out=np.full_like(vector, np.nan), where=norm > 1e-12)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def rigid_fit(template: np.ndarray, observed: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, int]:
    template = np.asarray(template, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if template.shape != observed.shape or template.ndim != 2 or template.shape[1] != 3:
        raise ValidationError("Rigid-fit arrays must have matching shape (N, 3)")
    mask = np.isfinite(template).all(axis=1) & np.isfinite(observed).all(axis=1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if int(mask.sum()) < 3:
        raise ValidationError("At least three valid non-collinear markers are required")
    x = template[mask]
    y = observed[mask]
    x_center = x.mean(axis=0)
    y_center = y.mean(axis=0)
    covariance = (x - x_center).T @ (y - y_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = y_center - rotation @ x_center
    fitted = x @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((fitted - y) ** 2, axis=1))))
    return rotation, translation, rmse, int(mask.sum())


def build_marker_template(points: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValidationError("Marker points must have shape (frames, markers, 3)")
    if valid is None:
        valid = np.isfinite(points).all(axis=2)
    valid = np.asarray(valid, dtype=bool)
    counts = valid.sum(axis=1)
    if counts.max(initial=0) < 3:
        raise ValidationError("No frame contains at least three valid markers")
    best = int(np.argmax(counts))
    reference = points[best].copy()
    center = np.nanmean(np.where(valid[best, :, None], reference, np.nan), axis=0)
    return reference - center


def geometric_median(points: np.ndarray, tolerance: float = 1e-8, max_iterations: int = 256) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValidationError("No finite points are available")
    estimate = np.median(points, axis=0)
    for _ in range(max_iterations):
        distance = np.linalg.norm(points - estimate, axis=1)
        if np.any(distance <= tolerance):
            return points[int(np.argmin(distance))].copy()
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
    transforms = np.asarray(transforms, dtype=float)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValidationError("Transforms must have shape (N, 4, 4)")
    finite = np.isfinite(transforms).all(axis=(1, 2))
    if int(finite.sum()) < 3:
        raise ValidationError("At least three finite transform candidates are required")
    mask = finite.copy()

    def robust_scale(values: np.ndarray) -> float:
        median = float(np.median(values))
        return 1.4826 * float(np.median(np.abs(values - median)))

    translation_gate = maximum_translation_gate_mm
    rotation_gate = maximum_rotation_gate_deg
    for _ in range(iterations):
        idx = np.flatnonzero(mask)
        if len(idx) < 3:
            break
        center_t = geometric_median(transforms[idx, :3, 3])
        center_r = Rotation.from_matrix(transforms[idx, :3, :3]).mean()
        t_error = np.linalg.norm(transforms[:, :3, 3] - center_t, axis=1)
        r_error = np.degrees(
            (center_r.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()
        )
        selected_t = t_error[mask]
        selected_r = r_error[mask]
        translation_gate = min(
            maximum_translation_gate_mm,
            max(minimum_translation_gate_mm, float(np.median(selected_t)) + mad_multiplier * robust_scale(selected_t)),
        )
        rotation_gate = min(
            maximum_rotation_gate_deg,
            max(minimum_rotation_gate_deg, float(np.median(selected_r)) + mad_multiplier * robust_scale(selected_r)),
        )
        new_mask = finite & (t_error <= translation_gate) & (r_error <= rotation_gate)
        if int(new_mask.sum()) < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    idx = np.flatnonzero(mask)
    center_t = geometric_median(transforms[idx, :3, 3])
    center_r = Rotation.from_matrix(transforms[idx, :3, :3]).mean()
    center = make_transform(center_r.as_matrix(), center_t)
    t_error = np.linalg.norm(transforms[:, :3, 3] - center_t, axis=1)
    r_error = np.degrees(
        (center_r.inv() * Rotation.from_matrix(transforms[:, :3, :3])).magnitude()
    )
    metrics = {
        "candidate_count": int(len(transforms)),
        "accepted_count": int(mask.sum()),
        "translation_gate_mm": float(translation_gate),
        "rotation_gate_deg": float(rotation_gate),
        "translation_median_error_mm": float(np.median(t_error[mask])),
        "translation_max_error_mm": float(np.max(t_error[mask])),
        "rotation_median_error_deg": float(np.median(r_error[mask])),
        "rotation_max_error_deg": float(np.max(r_error[mask])),
    }
    return center, mask, metrics


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
        valid: np.ndarray | None = None,
        *,
        min_markers: int = 3,
        max_fit_rmse_mm: float | None = 10.0,
    ) -> "PoseSeries":
        time_sec = np.asarray(time_sec, dtype=float)
        positions = np.asarray(positions, dtype=float)
        if valid is None:
            valid = np.isfinite(positions).all(axis=2)
        valid = np.asarray(valid, dtype=bool)
        n = len(time_sec)
        rotations = np.full((n, 3, 3), np.nan)
        translations = np.full((n, 3), np.nan)
        pose_valid = np.zeros(n, dtype=bool)
        rmse = np.full(n, np.nan)
        count = np.zeros(n, dtype=int)
        for frame in range(n):
            if int(valid[frame].sum()) < min_markers:
                continue
            try:
                rotation, translation, error, used = rigid_fit(template, positions[frame], valid[frame])
            except ValidationError:
                continue
            if max_fit_rmse_mm is not None and error > max_fit_rmse_mm:
                continue
            rotations[frame] = rotation
            translations[frame] = translation
            rmse[frame] = error
            count[frame] = used
            pose_valid[frame] = True
        return cls(time_sec, rotations, translations, pose_valid, rmse, count)

    def interpolate(self, query_time_sec: np.ndarray, *, max_gap_sec: float = 0.1) -> "InterpolatedPose":
        query = np.asarray(query_time_sec, dtype=float)
        valid_indices = np.flatnonzero(self.valid)
        n = len(query)
        rotations = np.full((n, 3, 3), np.nan)
        translations = np.full((n, 3), np.nan)
        pose_valid = np.zeros(n, dtype=bool)
        interpolated = np.zeros(n, dtype=bool)
        gaps = np.full(n, np.nan)
        rmse = np.full(n, np.nan)
        marker_count = np.zeros(n, dtype=int)
        if len(valid_indices) == 0:
            return InterpolatedPose(query, rotations, translations, pose_valid, interpolated, gaps, rmse, marker_count)
        valid_time = self.time_sec[valid_indices]
        if len(valid_indices) == 1:
            exact = np.isclose(query, valid_time[0], atol=1e-9, rtol=0.0)
            rotations[exact] = self.rotations[valid_indices[0]]
            translations[exact] = self.translations[valid_indices[0]]
            pose_valid[exact] = True
            return InterpolatedPose(query, rotations, translations, pose_valid, interpolated, gaps, rmse, marker_count)
        slerp = Slerp(valid_time, Rotation.from_matrix(self.rotations[valid_indices]))
        inside_indices = np.flatnonzero(np.isfinite(query) & (query >= valid_time[0]) & (query <= valid_time[-1]))
        if len(inside_indices):
            qi = query[inside_indices]
            right = np.searchsorted(valid_time, qi, side="left")
            right = np.clip(right, 0, len(valid_time) - 1)
            left = np.maximum(0, right - 1)
            exact = np.isclose(valid_time[right], qi, atol=1e-10, rtol=0.0)
            left[exact] = right[exact]
            local_gap = valid_time[right] - valid_time[left]
            local_gap[exact] = 0.0
            allowed = exact | (local_gap <= max_gap_sec)
            selected = inside_indices[allowed]
            if len(selected):
                selected_time = query[selected]
                rotations[selected] = slerp(selected_time).as_matrix()
                for axis in range(3):
                    translations[selected, axis] = np.interp(
                        selected_time, valid_time, self.translations[valid_indices, axis]
                    )
                pose_valid[selected] = True
                interpolated[selected] = ~exact[allowed]
                gaps[selected] = local_gap[allowed]
                nearest_left = left[allowed]
                nearest_right = right[allowed]
                choose_right = np.abs(valid_time[nearest_right] - selected_time) < np.abs(selected_time - valid_time[nearest_left])
                nearest = np.where(choose_right, nearest_right, nearest_left)
                source = valid_indices[nearest]
                rmse[selected] = self.fit_rmse_mm[source]
                marker_count[selected] = self.marker_count[source]
        return InterpolatedPose(query, rotations, translations, pose_valid, interpolated, gaps, rmse, marker_count)


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
