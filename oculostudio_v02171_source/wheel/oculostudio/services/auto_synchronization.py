from __future__ import annotations

import io
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags

from neon_vicon_gaze3d.processing.geometry import angular_speed_from_rotations
from oculostudio.adapters.archive import SourceView
from oculostudio.adapters.mocap import read_mocap
from oculostudio.adapters.pupil import find_cloud_candidates
from oculostudio.domain.geometry import PoseSeries, build_marker_template
from oculostudio.domain.models import ArtifactKind, TimeAnchor
from oculostudio.domain.timegraph import apply_time_map, fit_time_map
from oculostudio.errors import ValidationError
from oculostudio.services.etm_role_assignment import resolve_assignment_for_role
from oculostudio.services.synchronization import fit_and_store_time_map
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


AUTO_SYNC_ENGINE = "pupil_imu_vicon_angular_velocity_v3_precision_audited"


@dataclass(frozen=True)
class UniformSignal:
    grid_sec: np.ndarray
    signature: np.ndarray
    raw_speed: np.ndarray
    valid: np.ndarray
    sample_rate_hz: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GlobalAlignment:
    lag_samples: int
    delta_sec: float
    correlation: float
    second_correlation: float
    peak_margin: float
    overlap_sec: float
    candidate_count: int


def _normalise_column(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _find_column(columns, aliases: tuple[str, ...], *, required: bool = True) -> str | None:
    mapping = {_normalise_column(column): str(column) for column in columns}
    for alias in aliases:
        key = _normalise_column(alias)
        if key in mapping:
            return mapping[key]
    if required:
        raise ValidationError(f"Automatic synchronization could not find column {aliases[0]!r}")
    return None


def _pupil_imu_member(view: SourceView, recording_id: str | None) -> str:
    candidates = find_cloud_candidates(view)
    if recording_id:
        matches = [candidate for candidate in candidates if candidate.recording_id == recording_id]
        if matches:
            candidates = matches
    names = set(view.names())
    candidate_members: list[str] = []
    for candidate in candidates:
        member = f"{candidate.root}/imu.csv" if candidate.root else "imu.csv"
        if member in names and member not in candidate_members:
            candidate_members.append(member)
    if len(candidate_members) == 1:
        return candidate_members[0]
    if len(candidate_members) > 1:
        raise ValidationError(
            "The selected Timeseries source contains multiple Pupil recordings with imu.csv. "
            "Set or re-import the recording identity so automatic synchronization can fail closed instead of choosing one by archive order."
        )
    imu = [name for name in names if PurePosixPath(name).name.lower() == "imu.csv"]
    if len(imu) == 1:
        return imu[0]
    if not imu:
        raise ValidationError(
            "Automatic Pupil↔Vicon synchronization needs imu.csv in the selected Timeseries export. "
            "Use manual shared-event anchors when IMU data is unavailable."
        )
    raise ValidationError(
        f"The selected Timeseries source contains {len(imu)} imu.csv files and the recording could not be resolved uniquely."
    )


def _read_pupil_gyro(asset) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with SourceView(asset.source_path) as view:
        member = _pupil_imu_member(view, asset.recording_id)
        frame = pd.read_csv(io.StringIO(view.read_text(member)), low_memory=False)
    timestamp_col = _find_column(
        frame.columns,
        ("timestamp [ns]", "timestamp_ns", "timestamp ns", "timestamp"),
    )
    x_col = _find_column(
        frame.columns,
        ("gyro x [deg/s]", "gyro_x_deg_s", "gyro x deg/s", "gyro x [rad/s]", "gyro_x_rad_s"),
    )
    y_col = _find_column(
        frame.columns,
        ("gyro y [deg/s]", "gyro_y_deg_s", "gyro y deg/s", "gyro y [rad/s]", "gyro_y_rad_s"),
    )
    z_col = _find_column(
        frame.columns,
        ("gyro z [deg/s]", "gyro_z_deg_s", "gyro z deg/s", "gyro z [rad/s]", "gyro_z_rad_s"),
    )
    values = frame[[timestamp_col, x_col, y_col, z_col]].copy()
    for column in values.columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna().sort_values(timestamp_col)
    if values.empty:
        raise ValidationError("imu.csv contains no finite timestamped gyroscope samples")
    # Pupil exports can occasionally contain repeated timestamps. Averaging the
    # vectors preserves the physical sample while keeping interpolation monotonic.
    values = values.groupby(timestamp_col, as_index=False)[[x_col, y_col, z_col]].mean()
    timestamps = values[timestamp_col].to_numpy(dtype=np.int64)
    gyro = values[[x_col, y_col, z_col]].to_numpy(dtype=float)
    units_are_radians = all("rad" in _normalise_column(column) for column in (x_col, y_col, z_col))
    if not units_are_radians:
        gyro = np.radians(gyro)
    speed = np.linalg.norm(gyro, axis=1)
    finite = np.isfinite(speed)
    timestamps = timestamps[finite]
    speed = speed[finite]
    if len(timestamps) < 100:
        raise ValidationError("imu.csv contains too few finite gyroscope samples for automatic synchronization")
    dt = np.diff(timestamps.astype(np.float64)) / 1e9
    positive = dt[dt > 0]
    rate = float(1.0 / np.median(positive)) if len(positive) else float("nan")
    return timestamps, speed, {
        "member": member,
        "sample_count": int(len(timestamps)),
        "sample_rate_hz": rate,
        "timestamp_start_ns": int(timestamps[0]),
        "timestamp_end_ns": int(timestamps[-1]),
        "duration_sec": float((timestamps[-1] - timestamps[0]) / 1e9),
        "gyro_units": "rad/s" if units_are_radians else "deg/s converted to rad/s",
    }


def _label_prefix(label: str) -> str | None:
    text = str(label).strip()
    if ":" not in text:
        return None
    prefix = text.rsplit(":", 1)[0].strip()
    return prefix or None


def _fallback_tracker_labels(labels: list[str]) -> list[str]:
    groups: dict[str | None, list[str]] = {}
    for label in labels:
        groups.setdefault(_label_prefix(label), []).append(label)
    viable = [(len(rows), prefix, rows) for prefix, rows in groups.items() if len(rows) >= 3]
    if not viable:
        raise ValidationError("The selected tracker motion contains fewer than three usable markers")
    viable.sort(key=lambda row: row[0], reverse=True)
    best = viable[0]
    tied = [row for row in viable if row[0] == best[0]]
    if len(tied) > 1:
        names = [row[1] or "<unprefixed>" for row in tied]
        raise ValidationError(
            "Automatic synchronization cannot choose a unique tracker marker group. "
            f"Use an ETM-role assignment or resolve one of these prefixes: {names}."
        )
    return list(best[2])


def _read_tracker_angular_speed(
    tracker_asset,
    *,
    tracker_marker_labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = read_mocap(tracker_asset.source_path)
    selected_labels = list(tracker_marker_labels or _fallback_tracker_labels(data.labels))
    positions, marker_valid = data.select(selected_labels)
    template = build_marker_template(positions, marker_valid)
    pose = PoseSeries.from_markers(
        data.time_sec,
        template,
        positions,
        marker_valid,
        min_markers=3,
        max_fit_rmse_mm=15.0,
    )
    valid_index = np.flatnonzero(pose.valid)
    if len(valid_index) < 20:
        raise ValidationError("The ETM tracker has too few valid rigid-body poses for automatic synchronization")
    valid_time = pose.time_sec[valid_index]
    valid_rotations = pose.rotations[valid_index]
    time_mid, speed = angular_speed_from_rotations(valid_rotations, valid_time)
    gaps = np.diff(valid_time)
    maximum_pair_gap = max(0.1, 4.0 / max(float(data.frame_rate), 1.0))
    finite = np.isfinite(speed) & (gaps > 0) & (gaps <= maximum_pair_gap)
    time_mid = time_mid[finite]
    speed = speed[finite]
    if len(time_mid) < 20:
        raise ValidationError("The ETM tracker motion is too fragmented for automatic synchronization")
    return time_mid, speed, {
        "sample_count": int(len(time_mid)),
        "frame_rate_hz": float(data.frame_rate),
        "first_frame": int(data.first_frame),
        "time_start_sec": float(time_mid[0]),
        "time_end_sec": float(time_mid[-1]),
        "duration_sec": float(time_mid[-1] - time_mid[0]),
        "tracker_marker_labels": selected_labels,
        "pose_valid_fraction": float(pose.valid.mean()),
        "median_fit_rmse_mm": float(np.nanmedian(pose.fit_rmse_mm[pose.valid])),
    }


def _nearest_distance(grid: np.ndarray, sample_time: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sample_time, grid, side="left")
    right = np.clip(pos, 0, len(sample_time) - 1)
    left = np.clip(pos - 1, 0, len(sample_time) - 1)
    return np.minimum(np.abs(grid - sample_time[left]), np.abs(grid - sample_time[right]))


def _uniform_motion_signature(
    time_sec: np.ndarray,
    speed: np.ndarray,
    *,
    sample_rate_hz: float,
    maximum_gap_sec: float | None = None,
) -> UniformSignal:
    time_sec = np.asarray(time_sec, dtype=float)
    speed = np.asarray(speed, dtype=float)
    finite = np.isfinite(time_sec) & np.isfinite(speed)
    time_sec = time_sec[finite]
    speed = speed[finite]
    if len(time_sec) < 20:
        raise ValidationError("Motion signal has too few finite samples")
    order = np.argsort(time_sec)
    time_sec = time_sec[order]
    speed = speed[order]
    unique, inverse = np.unique(time_sec, return_inverse=True)
    if len(unique) != len(time_sec):
        sums = np.bincount(inverse, weights=speed)
        counts = np.bincount(inverse)
        speed = sums / np.maximum(counts, 1)
        time_sec = unique
    dt = 1.0 / float(sample_rate_hz)
    grid = np.arange(time_sec[0], time_sec[-1] + 0.25 * dt, dt, dtype=float)
    if len(grid) < 100:
        raise ValidationError("Motion recording is too short for automatic synchronization")
    raw = np.interp(grid, time_sec, speed)
    source_dt = np.diff(time_sec)
    source_dt = source_dt[source_dt > 0]
    if maximum_gap_sec is None:
        maximum_gap_sec = max(0.12, 4.0 * float(np.median(source_dt))) if len(source_dt) else 0.12
    valid = _nearest_distance(grid, time_sec) <= float(maximum_gap_sec)
    finite_raw = raw[valid]
    if len(finite_raw) < 100:
        raise ValidationError("Motion signal has insufficient continuous coverage")
    lower = float(np.nanpercentile(finite_raw, 0.2))
    upper = float(np.nanpercentile(finite_raw, 99.8))
    clipped = np.clip(raw, lower, max(upper, lower + 1e-9))
    transformed = np.log1p(np.maximum(clipped, 0.0))
    narrow_sigma = max(0.8, 0.055 * sample_rate_hz)
    broad_sigma = max(2.0, 0.75 * sample_rate_hz)
    smooth = gaussian_filter1d(transformed, narrow_sigma, mode="nearest")
    baseline = gaussian_filter1d(transformed, broad_sigma, mode="nearest")
    signature = smooth - baseline
    centre = float(np.nanmedian(signature[valid]))
    mad = 1.4826 * float(np.nanmedian(np.abs(signature[valid] - centre)))
    scale = mad if mad > 1e-8 else float(np.nanstd(signature[valid]))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValidationError("The recording contains insufficient rotational motion for automatic synchronization")
    signature = (signature - centre) / scale
    signature[~valid] = np.nan
    dynamic_range = float(np.nanpercentile(finite_raw, 95) - np.nanpercentile(finite_raw, 50))
    return UniformSignal(
        grid_sec=grid,
        signature=signature,
        raw_speed=raw,
        valid=valid,
        sample_rate_hz=float(sample_rate_hz),
        metadata={
            "start_sec": float(grid[0]),
            "end_sec": float(grid[-1]),
            "duration_sec": float(grid[-1] - grid[0]),
            "valid_fraction": float(valid.mean()),
            "maximum_gap_sec": float(maximum_gap_sec),
            "speed_median_rad_s": float(np.nanmedian(finite_raw)),
            "speed_p95_rad_s": float(np.nanpercentile(finite_raw, 95)),
            "speed_dynamic_range_rad_s": dynamic_range,
            "signature_scale": float(scale),
        },
    )


def _masked_correlation(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    y_valid = np.isfinite(y)
    x_valid = np.isfinite(x)
    y0 = np.where(y_valid, y, 0.0)
    x0 = np.where(x_valid, x, 0.0)
    if y_valid.any():
        y0[y_valid] -= float(np.mean(y0[y_valid]))
    if x_valid.any():
        x0[x_valid] -= float(np.mean(x0[x_valid]))
    numerator = correlate(y0, x0, mode="full", method="fft")
    overlap = correlate(y_valid.astype(float), x_valid.astype(float), mode="full", method="fft")
    y_energy = correlate(y0 * y0, x_valid.astype(float), mode="full", method="fft")
    x_energy = correlate(y_valid.astype(float), x0 * x0, mode="full", method="fft")
    denominator = np.sqrt(np.maximum(y_energy * x_energy, 0.0))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 1e-10,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    lags = correlation_lags(len(y), len(x), mode="full")
    return correlation, lags.astype(np.int64), overlap


def _global_alignment(
    pupil: UniformSignal,
    tracker: UniformSignal,
    *,
    maximum_offset_sec: float,
    minimum_overlap_sec: float,
    peak_exclusion_sec: float = 1.5,
) -> GlobalAlignment:
    if not math.isclose(pupil.sample_rate_hz, tracker.sample_rate_hz, rel_tol=1e-9):
        raise ValidationError("Internal synchronization signals use different sample rates")
    correlation, lags, overlap = _masked_correlation(tracker.signature, pupil.signature)
    rate = pupil.sample_rate_hz
    lag_sec = lags / rate
    admissible = (
        np.isfinite(correlation)
        & (np.abs(lag_sec) <= float(maximum_offset_sec))
        & (overlap >= float(minimum_overlap_sec) * rate)
    )
    candidates = np.flatnonzero(admissible)
    if not len(candidates):
        raise ValidationError(
            "No Pupil/Vicon overlap remains inside the automatic synchronization search range. "
            "Increase the maximum offset or use manual anchors."
        )
    best = int(candidates[np.nanargmax(correlation[candidates])])
    best_lag = int(lags[best])
    separated = candidates[np.abs(lags[candidates] - best_lag) >= int(round(peak_exclusion_sec * rate))]
    second_corr = float(np.nanmax(correlation[separated])) if len(separated) else -1.0
    return GlobalAlignment(
        lag_samples=best_lag,
        delta_sec=float(best_lag / rate),
        correlation=float(correlation[best]),
        second_correlation=second_corr,
        peak_margin=float(correlation[best] - second_corr),
        overlap_sec=float(overlap[best] / rate),
        candidate_count=int(len(candidates)),
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(a) & np.isfinite(b)
    count = int(finite.sum())
    if count < 5:
        return float("nan"), count
    aa = a[finite] - float(np.mean(a[finite]))
    bb = b[finite] - float(np.mean(b[finite]))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-10:
        return float("nan"), count
    return float(np.dot(aa, bb) / denom), count



def _refine_lag_peak(
    correlations: dict[int, float],
    best_residual: int,
    *,
    sample_rate_hz: float,
) -> dict[str, float | None]:
    """Refine an integer lag with a local quadratic peak and report precision.

    The previous implementation exposed only integer lags on a 50 Hz grid.  A
    real drift smaller than 20 ms across the recording therefore collapsed to
    exactly zero.  The quadratic vertex is not treated as a new observation;
    it is an interpolation of the independent local correlation peak, and its
    finite precision is carried into the drift confidence interval.
    """
    y0 = float(correlations[best_residual])
    ym = correlations.get(best_residual - 1)
    yp = correlations.get(best_residual + 1)
    delta = 0.0
    refined = y0
    curvature = None
    if ym is not None and yp is not None:
        ym = float(ym)
        yp = float(yp)
        denominator = ym - 2.0 * y0 + yp
        if np.isfinite(denominator) and denominator < -1e-10:
            delta = float(np.clip(0.5 * (ym - yp) / denominator, -0.5, 0.5))
            refined = float(y0 - 0.25 * (ym - yp) * delta)
            curvature = float(max(0.0, y0 - 0.5 * (ym + yp)))

    # Estimate peak width from a small correlation drop around the local peak.
    # Even a very sharp peak retains a quarter-grid uncertainty floor; this is
    # what prevents a co-start-trimmed recording from being reported as having
    # mathematically exact zero drift.
    drop = max(0.005, min(0.02, 0.01 * max(abs(y0), 1.0)))
    threshold = y0 - drop
    left = best_residual
    right = best_residual
    while (left - 1) in correlations and float(correlations[left - 1]) >= threshold:
        left -= 1
    while (right + 1) in correlations and float(correlations[right + 1]) >= threshold:
        right += 1
    half_width_samples = max(0.25, 0.5 * float(right - left))
    grid_ms = 1000.0 / float(sample_rate_hz)
    uncertainty_ms = max(0.25 * grid_ms, half_width_samples * grid_ms)
    return {
        "fractional_residual_samples": float(best_residual + delta),
        "subsample_shift_samples": float(delta),
        "refined_correlation": float(min(1.0, refined)),
        "peak_curvature": curvature,
        "lag_grid_ms": float(grid_ms),
        "lag_uncertainty_ms": float(uncertainty_ms),
        "peak_width_samples": float(2.0 * half_width_samples),
    }


def _window_drift_diagnostics(windows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in windows if row.get("accepted")]
    if len(accepted) < 2:
        return {
            "window_time_span_sec": 0.0,
            "window_local_offset_range_ms": None,
            "window_drift_ppm": None,
            "drift_ci95_ppm": None,
            "drift_resolution_ppm": None,
            "offset_ci95_ms": None,
            "median_lag_uncertainty_ms": None,
            "drift_significant": False,
        }
    x = np.asarray([float(row["pupil_rel_sec"]) for row in accepted], dtype=float)
    y = np.asarray([float(row["target_rel_sec"]) - float(row["pupil_rel_sec"]) for row in accepted], dtype=float)
    sigma = np.asarray([
        max(float(row.get("lag_uncertainty_ms") or 0.0) / 1000.0, 1e-6)
        for row in accepted
    ], dtype=float)
    x_ref = float(np.median(x))
    design = np.column_stack([np.ones_like(x), x - x_ref])
    weights = 1.0 / np.maximum(sigma * sigma, 1e-12)
    normal = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(normal)
    beta = covariance @ (design.T @ (weights * y))
    slope = float(beta[1])
    slope_se = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    drift_ci95 = 1.96 * slope_se * 1e6
    # Offset uncertainty at local time zero, not just at the centred reference.
    zero_vector = np.asarray([1.0, -x_ref], dtype=float)
    offset_var = float(zero_vector @ covariance @ zero_vector)
    offset_ci95_ms = 1.96 * math.sqrt(max(offset_var, 0.0)) * 1000.0
    span = float(np.ptp(x))
    median_uncertainty_ms = float(np.median(sigma) * 1000.0)
    drift_resolution = (
        float(2.0 * median_uncertainty_ms / max(span, 1e-9) * 1000.0)
        if span > 0
        else None
    )
    return {
        "window_time_span_sec": span,
        "window_local_offset_range_ms": float(np.ptp(y) * 1000.0),
        "window_drift_ppm": slope * 1e6,
        "drift_ci95_ppm": float(drift_ci95),
        "drift_resolution_ppm": drift_resolution,
        "offset_ci95_ms": float(offset_ci95_ms),
        "median_lag_uncertainty_ms": median_uncertainty_ms,
        "drift_significant": bool(abs(slope * 1e6) > drift_ci95),
    }


def _leave_one_out_window_validation(
    windows: list[dict[str, Any]],
    *,
    tracker_start_sec: float,
) -> dict[str, Any]:
    accepted_indices = [index for index, row in enumerate(windows) if row.get("accepted")]
    residuals: list[float] = []
    if len(accepted_indices) < 3:
        return {
            "holdout_window_count": 0,
            "cross_validated_median_residual_ms": None,
            "cross_validated_p95_residual_ms": None,
            "cross_validated_max_abs_residual_ms": None,
        }
    for held_index in accepted_indices:
        training = [index for index in accepted_indices if index != held_index]
        anchors = []
        for index in training:
            row = windows[index]
            anchors.append(TimeAnchor(
                source_ns=int(row["source_ns"]),
                target_ns=int(round((tracker_start_sec + float(row["target_rel_sec"])) * 1e9)),
                label=f"LOO training window {int(row['window_index'])}",
                weight=max(0.05, min(1.0, float(row["correlation"]) ** 2)),
            ))
        if len(anchors) < 2:
            continue
        mapping = fit_time_map("loo_source", "loo_target", anchors, method="auto_imu_leave_one_out")
        held = windows[held_index]
        predicted = int(apply_time_map(mapping, int(held["source_ns"])))
        observed = int(round((tracker_start_sec + float(held["target_rel_sec"])) * 1e9))
        residual_ms = float((observed - predicted) / 1e6)
        held["cross_validated_residual_ms"] = residual_ms
        held["validation_mode"] = "leave_one_window_out"
        residuals.append(residual_ms)
    absolute = np.abs(np.asarray(residuals, dtype=float))
    return {
        "holdout_window_count": int(len(residuals)),
        "cross_validated_median_residual_ms": float(np.median(absolute)) if len(absolute) else None,
        "cross_validated_p95_residual_ms": float(np.percentile(absolute, 95)) if len(absolute) else None,
        "cross_validated_max_abs_residual_ms": float(np.max(absolute)) if len(absolute) else None,
    }

def _window_matches(
    pupil: UniformSignal,
    tracker: UniformSignal,
    alignment: GlobalAlignment,
    *,
    pupil_start_ns: int,
) -> list[dict[str, Any]]:
    rate = pupil.sample_rate_hz
    lag = int(alignment.lag_samples)
    overlap_start = max(0, -lag)
    overlap_end = min(len(pupil.signature), len(tracker.signature) - lag)
    overlap_samples = overlap_end - overlap_start
    if overlap_samples <= int(rate * 5):
        return []
    overlap_sec = overlap_samples / rate
    window_sec = min(60.0, max(14.0, overlap_sec / 6.0))
    window = max(int(round(window_sec * rate)), int(rate * 8))
    stride = max(int(round(window / 2)), 1)
    residual_radius = max(int(round(1.0 * rate)), 1)
    exclusion = max(int(round(0.12 * rate)), 1)
    starts = list(range(overlap_start, max(overlap_start + 1, overlap_end - window + 1), stride))
    final_start = max(overlap_start, overlap_end - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts, start=1):
        end = min(overlap_end, start + window)
        if end - start < int(rate * 5):
            continue
        x = pupil.signature[start:end]
        candidates: list[tuple[int, float, int]] = []
        correlations: dict[int, float] = {}
        for residual in range(-residual_radius, residual_radius + 1):
            y_start = start + lag + residual
            y_end = y_start + len(x)
            if y_start < 0 or y_end > len(tracker.signature):
                continue
            corr, count = _pearson(x, tracker.signature[y_start:y_end])
            if np.isfinite(corr) and count >= int(rate * 4):
                correlations[int(residual)] = float(corr)
                candidates.append((residual, corr, count))
        if not candidates:
            continue
        candidates.sort(key=lambda row: row[1], reverse=True)
        best_residual, best_corr, best_count = candidates[0]
        refinement = _refine_lag_peak(correlations, int(best_residual), sample_rate_hz=rate)
        # Peak width is a descriptive ambiguity measure, not the standard error
        # of a lag estimated from thousands of samples.  Convert it to an
        # effective estimator precision and retain a quarter-grid floor.
        peak_half_width_ms = float(refinement["lag_uncertainty_ms"] or (1000.0 / rate))
        refinement["peak_half_width_ms"] = peak_half_width_ms
        refinement["lag_uncertainty_ms"] = max(
            0.25 * float(refinement["lag_grid_ms"] or (1000.0 / rate)),
            peak_half_width_ms / math.sqrt(max(float(best_count), 1.0)),
        )
        fractional_residual = float(refinement["fractional_residual_samples"])
        refined_corr = float(refinement["refined_correlation"])
        separated = [row for row in candidates[1:] if abs(row[0] - best_residual) >= exclusion]
        second = max((row[1] for row in separated), default=-1.0)
        centre_index = int(round((start + end - 1) / 2))
        pupil_rel_sec = centre_index / rate
        target_rel_sec = pupil_rel_sec + (lag + fractional_residual) / rate
        source_ns = int(round(pupil_start_ns + pupil_rel_sec * 1e9))
        peak_margin = float(refined_corr - second)
        accepted = bool(
            refined_corr >= 0.22
            and peak_margin >= 0.005
            and float(refinement["lag_uncertainty_ms"] or 0.0) <= 250.0
        )
        rows.append(
            {
                "window_index": index,
                "pupil_rel_sec": float(pupil_rel_sec),
                "target_rel_sec": float(target_rel_sec),
                "source_ns": source_ns,
                "global_lag_ms": float(lag / rate * 1000.0),
                "integer_residual_lag_samples": int(best_residual),
                "residual_lag_samples": fractional_residual,
                "subsample_shift_samples": float(refinement["subsample_shift_samples"] or 0.0),
                "residual_lag_ms": float(fractional_residual / rate * 1000.0),
                "local_offset_sec": float(target_rel_sec - pupil_rel_sec),
                "local_offset_ms": float((target_rel_sec - pupil_rel_sec) * 1000.0),
                "lag_grid_ms": float(refinement["lag_grid_ms"] or (1000.0 / rate)),
                "lag_uncertainty_ms": float(refinement["lag_uncertainty_ms"] or (1000.0 / rate)),
                "peak_width_samples": float(refinement["peak_width_samples"] or 0.0),
                "peak_half_width_ms": float(refinement["peak_half_width_ms"] or 0.0),
                "peak_curvature": refinement["peak_curvature"],
                "correlation": refined_corr,
                "integer_peak_correlation": float(best_corr),
                "second_correlation": float(second),
                "peak_margin": peak_margin,
                "valid_sample_count": int(best_count),
                "window_duration_sec": float((end - start) / rate),
                "accepted": accepted,
            }
        )
    accepted = [row for row in rows if row["accepted"]]
    if len(accepted) >= 3:
        residual = np.asarray([row["residual_lag_ms"] for row in accepted], dtype=float)
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median)))
        # Keep a physical floor tied to the estimator precision rather than the
        # old fixed 80 ms gate.
        precision_floor = float(np.median([row["lag_uncertainty_ms"] for row in accepted]))
        gate = max(4.5 * precision_floor, 4.5 * mad)
        for row in rows:
            if row["accepted"] and abs(float(row["residual_lag_ms"]) - median) > gate:
                row["accepted"] = False
                row["rejection_reason"] = "local lag outlier beyond precision-aware gate"
    return rows

def _anchors_from_windows(
    windows: list[dict[str, Any]],
    *,
    tracker_start_sec: float,
    alignment: GlobalAlignment,
    pupil_start_ns: int,
    pupil_duration_sec: float,
) -> list[dict[str, Any]]:
    accepted = [row for row in windows if row.get("accepted")]
    anchors: list[dict[str, Any]] = []
    for row in accepted:
        target_sec = tracker_start_sec + float(row["target_rel_sec"])
        anchors.append(
            {
                "source_ns": int(row["source_ns"]),
                "target_ns": int(round(target_sec * 1e9)),
                "label": f"Automatic IMU motion window {int(row['window_index'])}",
                "weight": max(0.05, min(1.0, float(row["correlation"]) ** 2)),
            }
        )
    if not anchors:
        pupil_rel = max(0.0, min(float(pupil_duration_sec), float(pupil_duration_sec) / 2.0))
        target_sec = tracker_start_sec + pupil_rel + alignment.delta_sec
        anchors.append(
            {
                "source_ns": int(round(pupil_start_ns + pupil_rel * 1e9)),
                "target_ns": int(round(target_sec * 1e9)),
                "label": "Automatic IMU global offset candidate",
                "weight": max(0.05, min(1.0, alignment.correlation**2)),
            }
        )
    anchors.sort(key=lambda row: row["source_ns"])
    return anchors


def _mapping_metrics(
    mapping,
    *,
    pupil_start_ns: int,
    tracker_start_sec: float,
    pupil: UniformSignal,
    tracker: UniformSignal,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    pupil_source_ns = pupil_start_ns + np.rint((pupil.grid_sec - pupil.grid_sec[0]) * 1e9).astype(np.int64)
    mapped_sec = apply_time_map(mapping, pupil_source_ns).astype(np.float64) / 1e9
    tracker_query = mapped_sec - tracker_start_sec + tracker.grid_sec[0]
    aligned = np.interp(
        tracker_query,
        tracker.grid_sec,
        np.where(np.isfinite(tracker.signature), tracker.signature, 0.0),
        left=np.nan,
        right=np.nan,
    )
    tracker_valid_interp = np.interp(
        tracker_query,
        tracker.grid_sec,
        tracker.valid.astype(float),
        left=0.0,
        right=0.0,
    ) >= 0.999
    aligned[~tracker_valid_interp] = np.nan
    aligned_corr, aligned_count = _pearson(pupil.signature, aligned)
    accepted = [row for row in windows if row.get("accepted")]
    in_sample_residual_ms = []
    for row in accepted:
        predicted = int(apply_time_map(mapping, int(row["source_ns"])))
        observed = int(round((tracker_start_sec + float(row["target_rel_sec"])) * 1e9))
        residual = float((observed - predicted) / 1e6)
        row["in_sample_residual_ms"] = residual
        in_sample_residual_ms.append(residual)
    cv = _leave_one_out_window_validation(windows, tracker_start_sec=tracker_start_sec)
    drift_diagnostics = _window_drift_diagnostics(windows)
    in_sample_abs = np.abs(np.asarray(in_sample_residual_ms, dtype=float))
    cv_p95 = cv.get("cross_validated_p95_residual_ms")
    cv_median = cv.get("cross_validated_median_residual_ms")
    cv_max = cv.get("cross_validated_max_abs_residual_ms")
    # Backward-compatible residual keys now mean independent holdout residuals
    # whenever enough distributed windows exist.  This prevents fitting and
    # scoring the same anchors from producing an artificial exact zero.
    primary_median = cv_median if cv_median is not None else (float(np.median(in_sample_abs)) if len(in_sample_abs) else None)
    primary_p95 = cv_p95 if cv_p95 is not None else (float(np.percentile(in_sample_abs, 95)) if len(in_sample_abs) else None)
    primary_max = cv_max if cv_max is not None else (float(np.max(in_sample_abs)) if len(in_sample_abs) else None)
    offset_at_start_sec = float(apply_time_map(mapping, pupil_start_ns)) / 1e9 - tracker_start_sec
    residual_resolution = drift_diagnostics.get("median_lag_uncertainty_ms")
    return {
        "aligned_correlation": float(aligned_corr) if np.isfinite(aligned_corr) else None,
        "aligned_sample_count": int(aligned_count),
        "estimated_offset_sec": offset_at_start_sec,
        "estimated_drift_ppm": float((mapping.scale - 1.0) * 1e6),
        "window_count": int(len(windows)),
        "matched_window_count": int(len(accepted)),
        "median_window_correlation": float(np.median([row["correlation"] for row in accepted])) if accepted else None,
        "median_window_peak_margin": float(np.median([row["peak_margin"] for row in accepted])) if accepted else None,
        "in_sample_median_residual_ms": float(np.median(in_sample_abs)) if len(in_sample_abs) else None,
        "in_sample_p95_residual_ms": float(np.percentile(in_sample_abs, 95)) if len(in_sample_abs) else None,
        "in_sample_max_abs_residual_ms": float(np.max(in_sample_abs)) if len(in_sample_abs) else None,
        "median_residual_ms": primary_median,
        "p95_residual_ms": primary_p95,
        "maximum_abs_residual_ms": primary_max,
        "residual_validation_mode": "leave_one_window_out" if int(cv.get("holdout_window_count") or 0) else "in_sample_fallback",
        "residual_resolution_ms": residual_resolution,
        "residual_below_resolution": bool(
            primary_p95 is not None
            and residual_resolution is not None
            and float(primary_p95) < float(residual_resolution)
        ),
        **cv,
        **drift_diagnostics,
    }

def _confidence(
    *,
    alignment: GlobalAlignment,
    mapping_metrics: dict[str, Any],
    overlap_sec: float,
    pupil_motion: dict[str, Any],
    tracker_motion: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    aligned = mapping_metrics.get("aligned_correlation")
    matched = int(mapping_metrics.get("matched_window_count") or 0)
    holdout = int(mapping_metrics.get("holdout_window_count") or 0)
    median_window = mapping_metrics.get("median_window_correlation")
    p95 = mapping_metrics.get("p95_residual_ms")
    residual_resolution = float(mapping_metrics.get("residual_resolution_ms") or 0.0)
    drift = abs(float(mapping_metrics.get("estimated_drift_ppm") or 0.0))
    drift_ci95 = mapping_metrics.get("drift_ci95_ppm")
    window_span = float(mapping_metrics.get("window_time_span_sec") or 0.0)
    required_windows = 3 if overlap_sec < 45.0 else 5 if overlap_sec < 120.0 else 6
    p95_limit = max(35.0, 3.0 * residual_resolution)
    high_checks = {
        "global correlation": alignment.correlation >= 0.35,
        "unique peak": alignment.peak_margin >= 0.025,
        "aligned correlation": aligned is not None and aligned >= 0.45,
        "matched windows": matched >= required_windows,
        "independent holdout windows": holdout >= max(3, required_windows - 1),
        "window correlation": median_window is not None and median_window >= 0.34,
        "cross-validated residual": p95 is not None and p95 <= p95_limit,
        "drift precision": drift_ci95 is not None and float(drift_ci95) <= 1200.0,
        "distributed time span": window_span >= min(max(20.0, 0.35 * overlap_sec), 180.0),
        "clock drift": drift <= 5000.0,
        "overlap": overlap_sec >= 12.0,
        "Pupil motion": float(pupil_motion.get("speed_dynamic_range_rad_s") or 0.0) > 0.02,
        "Vicon motion": float(tracker_motion.get("speed_dynamic_range_rad_s") or 0.0) > 0.02,
    }
    if all(high_checks.values()):
        reasons.append("Independent local motion windows support one affine Pupil↔Vicon map across the recording")
        if mapping_metrics.get("residual_below_resolution"):
            reasons.append(
                "Holdout residual is below the local-lag estimator resolution; zero is not interpreted as exact physical error"
            )
        if not mapping_metrics.get("drift_significant"):
            reasons.append("Estimated drift is statistically indistinguishable from zero at the reported 95% interval")
        return "HIGH", reasons
    for name, passed in high_checks.items():
        if not passed:
            reasons.append(f"High-confidence gate not met: {name}")
    review = (
        alignment.correlation >= 0.20
        and alignment.peak_margin >= 0.008
        and aligned is not None
        and aligned >= 0.25
        and overlap_sec >= 8.0
        and matched >= 2
        and holdout >= 1
        and drift <= 15000.0
        and (drift_ci95 is None or float(drift_ci95) <= 5000.0)
    )
    if review:
        return "REVIEW", reasons
    reasons.insert(0, "Automatic IMU↔Vicon alignment is not sufficiently identifiable")
    return "FAILED", reasons

def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    data = dict(candidate)
    data["anchors"] = [
        {
            **anchor,
            "source_ns": str(anchor["source_ns"]),
            "target_ns": str(anchor["target_ns"]),
        }
        for anchor in candidate.get("anchors", [])
    ]
    # Window rows are compact and useful for audit/review; keep them, but do not
    # send full sampled signals through JSON.
    data.pop("signal_rows", None)
    data.pop("cache_directory", None)
    return data


def _candidate_root(store: ProjectStore, candidate_id: str) -> Path:
    root = store.cache_dir() / "automatic_time_sync" / safe_name(candidate_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_candidate_cache(store: ProjectStore, candidate: dict[str, Any]) -> Path:
    root = _candidate_root(store, candidate["candidate_id"])
    atomic_write_json(root / "automatic_time_sync_candidate.json", candidate)
    windows = pd.DataFrame(candidate.get("windows") or [])
    windows.to_csv(root / "automatic_time_sync_windows.csv", index=False)
    signals = pd.DataFrame(candidate.get("signal_rows") or [])
    signals.to_csv(root / "automatic_time_sync_alignment.csv", index=False)
    return root


def _load_candidate(store: ProjectStore, candidate_id: str) -> tuple[dict[str, Any], Path]:
    root = _candidate_root(store, candidate_id)
    path = root / "automatic_time_sync_candidate.json"
    if not path.exists():
        raise ValidationError("Automatic synchronization candidate has expired or is unavailable")
    return json.loads(path.read_text("utf-8")), root


def _register_sync_artifact(
    store: ProjectStore,
    *,
    session_id: str,
    candidate: dict[str, Any],
    candidate_root: Path,
    mapping,
    accepted_by_user: bool,
) -> Any:
    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "time_sync", artifact_id)
    for path in candidate_root.iterdir():
        if path.is_file():
            shutil.copy2(path, output_root / path.name)
    metrics = dict(candidate.get("metrics") or {})
    matched_windows = int(metrics.get("matched_window_count") or 0)
    anchor_count = int(metrics.get("anchor_count") or len(candidate.get("anchors") or []))
    distributed_drift_validated = bool(matched_windows >= 2 and anchor_count >= 2)
    precision_validated = bool(
        int(metrics.get("holdout_window_count") or 0) >= 3
        and metrics.get("drift_ci95_ppm") is not None
        and metrics.get("residual_validation_mode") == "leave_one_window_out"
        and metrics.get("precision_audit_version")
    )
    formal_mapping_accepted = bool(
        candidate.get("confidence") == "HIGH"
        or (candidate.get("confidence") == "REVIEW" and accepted_by_user)
    )
    production_ready = bool(formal_mapping_accepted and distributed_drift_validated and precision_validated)
    accepted = {
        "schema_version": "oculostudio-automatic-time-sync-1.2",
        "engine": AUTO_SYNC_ENGINE,
        "candidate_id": candidate["candidate_id"],
        "confidence": candidate["confidence"],
        "accepted_by_user": bool(accepted_by_user),
        "wearer_role_id": candidate.get("wearer_role_id"),
        "assignment_artifact_id": candidate.get("assignment_artifact_id"),
        "source_asset_id": candidate.get("source_asset_id"),
        "target_asset_id": candidate.get("target_asset_id"),
        "ego_video_used": False,
        "distributed_drift_validated": distributed_drift_validated,
        "precision_validated": precision_validated,
        "formal_mapping_accepted": formal_mapping_accepted,
        "production_ready": production_ready,
        "time_map": mapping.model_dump(mode="json"),
        "metrics": metrics,
        "reasons": candidate["reasons"],
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "automatic_time_sync_accepted.json", accepted)
    session = store.project.sessions[session_id]
    source = session.assets[candidate["source_asset_id"]]
    target = session.assets[candidate["target_asset_id"]]
    return ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.TIME_SYNC,
        label=(
            f"Automatic time sync · {candidate.get('wearer_role_id')} · {source.label} ↔ {target.label}"
            if candidate.get("wearer_role_id")
            else f"Automatic time sync · {source.label} ↔ {target.label}"
        ),
        output_root=output_root,
        source_asset_ids=[source.id, target.id],
        parent_artifact_ids=[candidate["assignment_artifact_id"]] if candidate.get("assignment_artifact_id") else [],
        parameters={
            "engine": AUTO_SYNC_ENGINE,
            "source_domain_id": candidate["source_domain_id"],
            "target_domain_id": candidate["target_domain_id"],
            "source_asset_id": source.id,
            "target_asset_id": target.id,
            "candidate_id": candidate["candidate_id"],
            "time_map_id": mapping.id,
            "wearer_role_id": candidate.get("wearer_role_id"),
            "assignment_artifact_id": candidate.get("assignment_artifact_id"),
            "accepted_by_user": bool(accepted_by_user),
            "ego_video_used": False,
        },
        qc={
            "confidence": candidate["confidence"],
            "sync_verified": formal_mapping_accepted,
            "distributed_drift_validated": distributed_drift_validated,
            "precision_validated": precision_validated,
            "production_ready": production_ready,
            "accepted_by_user": bool(accepted_by_user),
            "wearer_role_id": candidate.get("wearer_role_id"),
            "time_map_id": mapping.id,
            "ego_video_used": False,
            **metrics,
        },
        provenance={
            "method": mapping.method,
            "automatic_signal": "Pupil IMU gyroscope magnitude ↔ Vicon ETM rigid-body angular velocity",
            "clock_estimation_inputs": ["Pupil imu.csv", "ETM wearer C3D/NPZ"],
            "ego_video_used": False,
            "role_assignment_artifact_id": candidate.get("assignment_artifact_id"),
        },
        artifact_id=artifact_id,
    )



def _safe_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def get_automatic_time_sync_preview(
    store: ProjectStore,
    *,
    session_id: str,
    candidate_id: str | None = None,
    artifact_id: str | None = None,
    maximum_samples: int = 1800,
) -> dict[str, Any]:
    if bool(candidate_id) == bool(artifact_id):
        raise ValidationError("Provide exactly one synchronization candidate or accepted artifact")
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    if candidate_id:
        candidate, root = _load_candidate(store, candidate_id)
        accepted_payload = None
    else:
        artifact = session.artifacts.get(str(artifact_id))
        if artifact is None or artifact.kind != ArtifactKind.TIME_SYNC:
            raise ValidationError("Selected synchronization artifact is unavailable")
        paths = {Path(row.path).name: Path(row.path) for row in artifact.files if Path(row.path).exists()}
        candidate_path = paths.get("automatic_time_sync_candidate.json")
        if candidate_path is None:
            raise ValidationError("This synchronization artifact predates auditable preview evidence")
        candidate = json.loads(candidate_path.read_text("utf-8"))
        root = candidate_path.parent
        accepted_path = paths.get("automatic_time_sync_accepted.json")
        accepted_payload = json.loads(accepted_path.read_text("utf-8")) if accepted_path else None
    signal_path = root / "automatic_time_sync_alignment.csv"
    window_path = root / "automatic_time_sync_windows.csv"
    signals = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    windows = pd.read_csv(window_path) if window_path.exists() else pd.DataFrame(candidate.get("windows") or [])
    maximum_samples = int(max(100, min(int(maximum_samples), 10000)))
    if len(signals) > maximum_samples:
        indices = np.unique(np.rint(np.linspace(0, len(signals) - 1, maximum_samples)).astype(np.int64))
        signals = signals.iloc[indices]
    signal_rows = [
        {str(key): _safe_json_scalar(value) for key, value in row.items()}
        for row in signals.to_dict(orient="records")
    ]
    window_rows = [
        {str(key): _safe_json_scalar(value) for key, value in row.items()}
        for row in windows.to_dict(orient="records")
    ]
    return {
        "candidate_id": candidate.get("candidate_id"),
        "artifact_id": artifact_id,
        "confidence": candidate.get("confidence"),
        "source_asset_label": candidate.get("source_asset_label"),
        "target_asset_label": candidate.get("target_asset_label"),
        "wearer_role_id": candidate.get("wearer_role_id"),
        "metrics": candidate.get("metrics") or {},
        "reasons": candidate.get("reasons") or [],
        "signals": signal_rows,
        "windows": window_rows,
        "accepted": accepted_payload,
        "ego_video_used": False,
    }

def accept_automatic_time_sync(
    store: ProjectStore,
    *,
    session_id: str,
    candidate_id: str,
    accepted_by_user: bool = True,
) -> dict[str, Any]:
    candidate, root = _load_candidate(store, candidate_id)
    if candidate.get("session_id") != session_id:
        raise ValidationError("Automatic synchronization candidate belongs to another session")
    if candidate.get("confidence") == "FAILED":
        raise ValidationError("A failed automatic synchronization candidate cannot be accepted")
    method = "auto_imu_reviewed_robust_affine" if accepted_by_user else "auto_imu_high_robust_affine"
    mapping = fit_and_store_time_map(
        store,
        session_id=session_id,
        source_domain_id=candidate["source_domain_id"],
        target_domain_id=candidate["target_domain_id"],
        anchors=candidate["anchors"],
        method=method,
    )
    artifact = _register_sync_artifact(
        store,
        session_id=session_id,
        candidate=candidate,
        candidate_root=root,
        mapping=mapping,
        accepted_by_user=accepted_by_user,
    )
    candidate["accepted_time_map_id"] = mapping.id
    candidate["accepted_artifact_id"] = artifact.id
    atomic_write_json(root / "automatic_time_sync_candidate.json", candidate)
    return {
        "status": "saved",
        "confidence": candidate["confidence"],
        "candidate": _public_candidate(candidate),
        "time_map": mapping.model_dump(mode="json"),
        "artifact": artifact.model_dump(mode="json"),
    }


def automatic_pupil_vicon_sync(
    store: ProjectStore,
    *,
    session_id: str,
    timeseries_asset_id: str,
    tracker_asset_id: str | None = None,
    etm_assignment_artifact_id: str | None = None,
    wearer_role_id: str | None = None,
    maximum_offset_sec: float = 120.0,
    sample_rate_hz: float = 120.0,
    save_if_high: bool = True,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    timeseries_asset = session.assets.get(timeseries_asset_id)
    if timeseries_asset is None:
        raise ValidationError("Selected Timeseries asset is unavailable")
    assignment_row = None
    assignment_artifact = None
    if etm_assignment_artifact_id:
        assignment_artifact = session.artifacts.get(etm_assignment_artifact_id)
        if assignment_artifact is None:
            raise ValidationError("Selected ETM-role assignment artifact is unavailable")
        if not wearer_role_id:
            raise ValidationError("Select the wearer role for automatic synchronization")
        assignment_row = resolve_assignment_for_role(assignment_artifact, wearer_role_id)
        assigned_tracker_id = assignment_row.get("etm_asset_id")
        if tracker_asset_id and tracker_asset_id != assigned_tracker_id:
            raise ValidationError("Selected tracker conflicts with the ETM-role assignment")
        tracker_asset_id = assigned_tracker_id
        requested_role = str(wearer_role_id or "").upper()
        identity = (timeseries_asset.metadata or {}).get("identity") or {}
        declared_role = str(identity.get("role_id") or (timeseries_asset.metadata or {}).get("role_id") or "").upper()
        if declared_role and declared_role != requested_role:
            raise ValidationError(
                f"Selected Pupil Timeseries is declared as {declared_role}, not {requested_role}; choose the role-matched imu.csv source"
            )
        assignment_interaction = str((assignment_artifact.parameters or {}).get("interaction_id") or "").upper()
        source_interaction = str(identity.get("interaction_id") or (timeseries_asset.metadata or {}).get("interaction_id") or "").upper()
        if assignment_interaction and source_interaction and assignment_interaction != source_interaction:
            raise ValidationError(
                "Selected Pupil Timeseries belongs to a different interaction than the wearer assignment"
            )
    if not tracker_asset_id:
        raise ValidationError("Select a tracker or ETM-role assignment for automatic synchronization")
    tracker_asset = session.assets.get(tracker_asset_id)
    if tracker_asset is None:
        raise ValidationError("Resolved ETM tracker asset is unavailable")
    source_domain_id = timeseries_asset.primary_time_domain_id
    target_domain_id = tracker_asset.primary_time_domain_id
    if not source_domain_id or not target_domain_id:
        raise ValidationError("Timeseries and tracker assets need primary time domains")
    if source_domain_id == target_domain_id:
        raise ValidationError("Timeseries and tracker already use the same clock")
    if maximum_offset_sec <= 0 or maximum_offset_sec > 3600:
        raise ValidationError("Maximum automatic synchronization offset must be between 0 and 3600 seconds")
    if sample_rate_hz < 20 or sample_rate_hz > 240:
        raise ValidationError("Automatic synchronization sample rate must be between 20 and 240 Hz")

    emit(progress, 0.05, "Reading Pupil IMU gyroscope")
    pupil_ns, pupil_speed, pupil_source_meta = _read_pupil_gyro(timeseries_asset)
    cancellation.raise_if_cancelled()
    emit(progress, 0.20, "Reconstructing ETM rigid-body rotation")
    tracker_time, tracker_speed, tracker_source_meta = _read_tracker_angular_speed(
        tracker_asset,
        tracker_marker_labels=(assignment_row or {}).get("tracker_marker_labels"),
    )
    cancellation.raise_if_cancelled()

    pupil_rel = (pupil_ns - int(pupil_ns[0])).astype(np.float64) / 1e9
    tracker_rel = tracker_time - float(tracker_time[0])
    emit(progress, 0.38, "Building comparable angular-motion signatures")
    pupil_signal = _uniform_motion_signature(pupil_rel, pupil_speed, sample_rate_hz=sample_rate_hz)
    tracker_signal = _uniform_motion_signature(tracker_rel, tracker_speed, sample_rate_hz=sample_rate_hz)
    minimum_overlap = min(30.0, max(8.0, 0.25 * min(pupil_signal.metadata["duration_sec"], tracker_signal.metadata["duration_sec"])))
    emit(progress, 0.52, "Searching Pupil↔Vicon offset")
    alignment = _global_alignment(
        pupil_signal,
        tracker_signal,
        maximum_offset_sec=maximum_offset_sec,
        minimum_overlap_sec=minimum_overlap,
    )
    cancellation.raise_if_cancelled()
    emit(progress, 0.68, "Estimating drift from distributed motion windows")
    windows = _window_matches(
        pupil_signal,
        tracker_signal,
        alignment,
        pupil_start_ns=int(pupil_ns[0]),
    )
    anchors = _anchors_from_windows(
        windows,
        tracker_start_sec=float(tracker_time[0]),
        alignment=alignment,
        pupil_start_ns=int(pupil_ns[0]),
        pupil_duration_sec=float(pupil_rel[-1]),
    )
    model_anchors = [TimeAnchor.model_validate(anchor) for anchor in anchors]
    mapping_preview = fit_time_map(
        source_domain_id,
        target_domain_id,
        model_anchors,
        method="auto_imu_candidate",
    )
    map_metrics = _mapping_metrics(
        mapping_preview,
        pupil_start_ns=int(pupil_ns[0]),
        tracker_start_sec=float(tracker_time[0]),
        pupil=pupil_signal,
        tracker=tracker_signal,
        windows=windows,
    )
    overlap_sec = float(alignment.overlap_sec)
    confidence, reasons = _confidence(
        alignment=alignment,
        mapping_metrics=map_metrics,
        overlap_sec=overlap_sec,
        pupil_motion=pupil_signal.metadata,
        tracker_motion=tracker_signal.metadata,
    )
    metrics = {
        "engine": AUTO_SYNC_ENGINE,
        "global_correlation": alignment.correlation,
        "second_best_correlation": alignment.second_correlation,
        "global_peak_margin": alignment.peak_margin,
        "global_offset_delta_sec": alignment.delta_sec,
        "usable_overlap_sec": overlap_sec,
        "search_maximum_offset_sec": float(maximum_offset_sec),
        "sample_rate_hz": float(sample_rate_hz),
        "anchor_count": int(len(anchors)),
        "preview_rms_residual_ms": float(mapping_preview.quality.rms_residual_ns / 1e6),
        "preview_max_abs_residual_ms": float(mapping_preview.quality.max_abs_residual_ns / 1e6),
        **map_metrics,
        "pupil_signal": {**pupil_source_meta, **pupil_signal.metadata},
        "tracker_signal": {**tracker_source_meta, **tracker_signal.metadata},
        "relative_start_alignment_sec": float(alignment.delta_sec),
        "co_start_prior_detected": bool(abs(float(alignment.delta_sec)) <= max(0.25, 2.0 / float(sample_rate_hz))),
        "precision_audit_version": "subsample-lag+leave-one-window-out+drift-ci-v1",
    }
    candidate_id = new_id("autosync")
    # Decimate aligned traces to a compact audit file. This is enough for a
    # future plot/inspection view without copying the original recordings.
    stride = max(1, int(math.ceil(len(pupil_signal.grid_sec) / 12000)))
    source_ns_grid = int(pupil_ns[0]) + np.rint(pupil_signal.grid_sec * 1e9).astype(np.int64)
    mapped_sec = apply_time_map(mapping_preview, source_ns_grid).astype(np.float64) / 1e9
    tracker_query_rel = mapped_sec - float(tracker_time[0])
    tracker_interp = np.interp(
        tracker_query_rel,
        tracker_signal.grid_sec,
        np.where(np.isfinite(tracker_signal.signature), tracker_signal.signature, 0.0),
        left=np.nan,
        right=np.nan,
    )
    signal_rows = [
        {
            "pupil_rel_sec": float(pupil_signal.grid_sec[i]),
            "pupil_timestamp_ns": int(source_ns_grid[i]),
            "pupil_signature": float(pupil_signal.signature[i]) if np.isfinite(pupil_signal.signature[i]) else None,
            "mapped_vicon_time_sec": float(mapped_sec[i]),
            "vicon_signature": float(tracker_interp[i]) if np.isfinite(tracker_interp[i]) else None,
            "signature_difference": float(pupil_signal.signature[i] - tracker_interp[i]) if np.isfinite(pupil_signal.signature[i]) and np.isfinite(tracker_interp[i]) else None,
        }
        for i in range(0, len(pupil_signal.grid_sec), stride)
    ]
    candidate = {
        "schema_version": "oculostudio-auto-sync-candidate-1.1",
        "candidate_id": candidate_id,
        "session_id": session_id,
        "source_asset_id": timeseries_asset.id,
        "source_asset_label": timeseries_asset.label,
        "target_asset_id": tracker_asset.id,
        "target_asset_label": tracker_asset.label,
        "source_domain_id": source_domain_id,
        "target_domain_id": target_domain_id,
        "assignment_artifact_id": etm_assignment_artifact_id,
        "wearer_role_id": wearer_role_id,
        "confidence": confidence,
        "anchors": anchors,
        "windows": windows,
        "metrics": metrics,
        "reasons": reasons,
        "signal_rows": signal_rows,
        "created_at": utc_now(),
    }
    root = _write_candidate_cache(store, candidate)
    emit(progress, 0.92, f"Automatic synchronization confidence: {confidence}")
    if confidence == "HIGH" and save_if_high:
        result = accept_automatic_time_sync(
            store,
            session_id=session_id,
            candidate_id=candidate_id,
            accepted_by_user=False,
        )
        result["message"] = "High-confidence Pupil IMU↔Vicon mapping was saved automatically."
        emit(progress, 1.0, "Automatic synchronization saved")
        return result
    status = "review_required" if confidence in {"HIGH", "REVIEW"} else "failed"
    emit(progress, 1.0, "Automatic synchronization candidate ready")
    return {
        "status": status,
        "confidence": confidence,
        "candidate": _public_candidate(candidate),
        "message": (
            "Automatic motion alignment found a plausible mapping that requires explicit user confirmation."
            if confidence in {"HIGH", "REVIEW"}
            else "Automatic motion alignment was not identifiable; use manual shared-event anchors."
        ),
    }
