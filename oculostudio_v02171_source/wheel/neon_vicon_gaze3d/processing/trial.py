from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

from ..io.archive import sha256_file
from ..io.c3d_reader import C3DData, read_c3d
from ..io.profile import CalibrationProfile
from ..io.pupil_recording import TimeseriesRecording, unproject_gaze_pixels
from .geometry import (
    PoseSeries,
    angular_speed_from_rotations,
    make_transform,
    transform_points,
)
from .sync import SyncMapping


class TrialProcessingError(RuntimeError):
    pass


@dataclass
class TrialRequest:
    timeseries_source: Path
    c3d_path: Path
    calibration_profile_path: Path
    sync_mapping: SyncMapping
    output_directory: Path
    scene_source: Path | None = None
    tracker_label_mapping: dict[str, str] | None = None
    canonical_participant_id: str = ""
    output_name: str = ""
    max_pose_interpolation_gap_sec: float = 0.1
    max_tracker_fit_rmse_mm: float = 5.0
    write_parquet: bool = True
    allow_profile_mismatch: bool = False
    allow_legacy_profile: bool = False
    include_raw_identity_metadata: bool = False


@dataclass
class TrialResult:
    output_csv: Path
    output_parquet: Path | None
    sync_json: Path
    manifest_json: Path
    qc_metrics_json: Path
    qc_report_html: Path
    qc_files: list[Path]
    row_count: int
    valid_ray_count: int


def _log(callback: Callable[[str], None] | None, text: str) -> None:
    if callback:
        callback(text)


def _report_progress(
    callback: Callable[[float, str], None] | None,
    value: float,
    text: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), text)


def _privacy_safe_recording_info(info: dict) -> dict:
    """Whitelist non-name recording fields for privacy-safe manifests."""

    allowed = (
        "recording_id",
        "start_time",
        "duration",
        "duration_ns",
        "module_serial_number",
        "frame_name",
        "pipeline_version",
        "app_version",
        "data_format_version",
    )
    return {key: info.get(key) for key in allowed if key in info}


def _privacy_safe_input_paths(request: TrialRequest) -> dict:
    if request.include_raw_identity_metadata:
        return {
            "timeseries_source": str(request.timeseries_source),
            "scene_source": str(request.scene_source) if request.scene_source else None,
            "c3d": str(request.c3d_path),
            "calibration_profile": str(request.calibration_profile_path),
        }
    return {
        "timeseries_source": "<redacted; see input_sha256>",
        "scene_source": (
            "<redacted; see input_sha256>" if request.scene_source else None
        ),
        "c3d": "<redacted; see input_sha256>",
        "calibration_profile": "<redacted; see input_sha256>",
    }


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    result = []
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            result.append(np.linalg.norm(points[first] - points[second]))
    return np.asarray(result)


def _tracker_shape_check(
    profile_template: np.ndarray,
    trial_positions: np.ndarray,
    trial_valid: np.ndarray,
) -> dict:
    reference = _pairwise_distances(profile_template)
    complete = np.flatnonzero(trial_valid.all(axis=1))
    if len(complete) == 0:
        return {"complete_frames": 0, "median_pairwise_error_mm": None, "p95_pairwise_error_mm": None}
    if len(complete) > 200:
        complete = complete[
            np.linspace(0, len(complete) - 1, 200).round().astype(int)
        ]
    errors = []
    for frame in complete:
        current = _pairwise_distances(trial_positions[frame])
        errors.extend(np.abs(current - reference))
    error = np.asarray(errors)
    return {
        "complete_frames": int(len(complete)),
        "median_pairwise_error_mm": float(np.median(error)),
        "p95_pairwise_error_mm": float(np.percentile(error, 95)),
        "max_pairwise_error_mm": float(np.max(error)),
    }


def _apply_calibrated_camera_transform(
    tracker_rotations_world: np.ndarray,
    tracker_translations_world_mm: np.ndarray,
    R_tracker_from_camera: np.ndarray,
    camera_origin_tracker_mm: np.ndarray,
    camera_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the calibrated camera-to-tracker transform to camera-space rays.

    This is the core profile-dependent step used by the Timeseries exporter.
    The ray origin is the calibrated Neon scene-camera optical centre expressed
    in Vicon world coordinates; the direction is the calibrated camera-space
    gaze direction rotated into the same world frame.
    """

    tracker_rotations_world = np.asarray(tracker_rotations_world, dtype=float)
    tracker_translations_world_mm = np.asarray(
        tracker_translations_world_mm, dtype=float
    )
    R_tracker_from_camera = np.asarray(R_tracker_from_camera, dtype=float)
    camera_origin_tracker_mm = np.asarray(
        camera_origin_tracker_mm, dtype=float
    ).reshape(3)
    camera_directions = np.asarray(camera_directions, dtype=float)
    if tracker_rotations_world.shape[0] != camera_directions.shape[0]:
        raise ValueError("Pose and gaze-direction sample counts differ")
    R_world_from_camera = np.einsum(
        "nij,jk->nik", tracker_rotations_world, R_tracker_from_camera
    )
    origins_world_mm = (
        np.einsum(
            "nij,j->ni", tracker_rotations_world, camera_origin_tracker_mm
        )
        + tracker_translations_world_mm
    )
    directions_world = np.einsum(
        "nij,nj->ni", R_world_from_camera, camera_directions
    )
    norms = np.linalg.norm(directions_world, axis=1, keepdims=True)
    finite = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-12)
    directions_world[finite] /= norms[finite]
    directions_world[~finite] = np.nan
    return origins_world_mm, directions_world


def _imu_sync_metrics(
    recording: TimeseriesRecording,
    pose: PoseSeries,
    sync: SyncMapping,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    if recording.imu is None:
        return {"available": False, "reason": "imu.csv not present"}, None
    required = {
        "timestamp [ns]",
        "gyro x [deg/s]",
        "gyro y [deg/s]",
        "gyro z [deg/s]",
    }
    if not required.issubset(recording.imu.columns):
        return {"available": False, "reason": "imu.csv columns missing"}, None
    valid_pose = np.flatnonzero(pose.valid)
    if len(valid_pose) < 10:
        return {"available": False, "reason": "insufficient valid tracker poses"}, None
    tracker_t, tracker_speed = angular_speed_from_rotations(
        pose.rotations[valid_pose], pose.time_sec[valid_pose]
    )
    imu_t = sync.pupil_ns_to_c3d_sec(
        recording.imu["timestamp [ns]"].to_numpy(dtype=np.int64)
    )
    gyro = recording.imu[
        ["gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]"]
    ].to_numpy(dtype=float)
    imu_speed = np.linalg.norm(np.radians(gyro), axis=1)
    start = max(float(np.nanmin(tracker_t)), float(np.nanmin(imu_t)))
    end = min(float(np.nanmax(tracker_t)), float(np.nanmax(imu_t)))
    if end - start < 2.0:
        return {"available": False, "reason": "less than 2 seconds overlap"}, None
    rate = 100.0
    grid = np.arange(start, end, 1.0 / rate)
    tracker_interp = np.interp(grid, tracker_t, tracker_speed)
    imu_interp = np.interp(grid, imu_t, imu_speed)
    tracker_norm = tracker_interp - np.mean(tracker_interp)
    imu_norm = imu_interp - np.mean(imu_interp)
    denominator = np.std(tracker_norm) * np.std(imu_norm)
    if denominator < 1e-8:
        return {"available": False, "reason": "insufficient rotational motion"}, None
    zero_corr = float(np.corrcoef(tracker_interp, imu_interp)[0, 1])
    correlation = correlate(imu_norm, tracker_norm, mode="full", method="fft")
    lags = correlation_lags(len(imu_norm), len(tracker_norm), mode="full")
    allowed = np.abs(lags / rate) <= 0.5
    selected = np.flatnonzero(allowed)
    best = selected[np.argmax(correlation[selected])]
    residual_lag_sec = float(lags[best] / rate)
    shifted_grid = grid + residual_lag_sec
    imu_shifted = np.interp(grid, shifted_grid, imu_interp, left=np.nan, right=np.nan)
    finite = np.isfinite(imu_shifted)
    shifted_corr = (
        float(np.corrcoef(tracker_interp[finite], imu_shifted[finite])[0, 1])
        if finite.sum() > 10
        else None
    )
    metrics = {
        "available": True,
        "overlap_start_sec": start,
        "overlap_end_sec": end,
        "zero_lag_correlation": zero_corr,
        "best_residual_lag_ms": residual_lag_sec * 1000.0,
        "best_lag_correlation": shifted_corr,
    }
    return metrics, (grid, tracker_interp, imu_interp)


def process_trial(
    request: TrialRequest,
    *,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> TrialResult:
    output = Path(request.output_directory)
    qc_dir = output / "qc"
    output.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    _report_progress(progress, 0.01, "Loading calibrated camera-to-tracker profile")
    profile = CalibrationProfile.load(request.calibration_profile_path)
    profile_sha256 = sha256_file(request.calibration_profile_path)
    participant_id = str(request.canonical_participant_id).strip()
    if not participant_id:
        raise TrialProcessingError(
            "Enter an anonymized canonical participant ID (for example P01). "
            "The exporter never falls back to the raw Pupil wearer name."
        )
    if not profile.is_robust_v2_profile and not request.allow_legacy_profile:
        raise TrialProcessingError(
            "This calibration profile was created by the legacy v1 estimator, which could "
            "move camera translation by several centimetres in distant planar recordings. "
            "Rebuild the profile with v0.2.0 or later, or enable the explicit legacy-profile override "
            "only for comparison/audit."
        )
    _report_progress(progress, 0.06, "Opening Pupil Labs Timeseries data")
    _log(log, "Opening Pupil Labs Timeseries data...")
    with TimeseriesRecording.open(
        request.timeseries_source, request.scene_source
    ) as recording:
        trial_serial = str(
            recording.info.get("module_serial_number", recording.intrinsics.serial_number)
        )
        if (
            profile.module_serial_number
            and trial_serial
            and trial_serial != str(profile.module_serial_number)
            and not request.allow_profile_mismatch
        ):
            raise TrialProcessingError(
                "The Timeseries Neon module serial does not match the calibration profile: "
                f"trial={trial_serial}, profile={profile.module_serial_number}. "
                "Enable the explicit mismatch override only after verifying the mount identity."
            )
        matrix_difference = float(np.max(np.abs(recording.intrinsics.matrix - profile.scene_camera_matrix)))
        distortion_a = np.asarray(recording.intrinsics.distortion, dtype=float).reshape(-1)
        distortion_b = np.asarray(profile.scene_distortion_coefficients, dtype=float).reshape(-1)
        if distortion_a.shape == distortion_b.shape:
            distortion_difference = float(np.max(np.abs(distortion_a - distortion_b)))
        else:
            distortion_difference = float("inf")
        if (matrix_difference > 1e-3 or distortion_difference > 1e-5) and not request.allow_profile_mismatch:
            raise TrialProcessingError(
                "The Timeseries scene-camera intrinsics differ from the calibration profile. "
                f"K max difference={matrix_difference:.6g}, distortion max difference={distortion_difference:.6g}."
            )
        _report_progress(progress, 0.15, "Profile validated; reading interaction C3D")
        _log(log, "Reading Vicon C3D and reconstructing the eye-tracker rigid body...")
        c3d = read_c3d(request.c3d_path)
        trial_labels = [
            (request.tracker_label_mapping or {}).get(label, label)
            for label in profile.tracker_marker_labels
        ]
        positions, marker_valid = c3d.select(trial_labels)
        shape_check = _tracker_shape_check(
            profile.tracker_template_mm, positions, marker_valid
        )
        shape_p95 = shape_check.get("p95_pairwise_error_mm")
        if (
            shape_p95 is not None
            and float(shape_p95) > 5.0
            and not request.allow_profile_mismatch
        ):
            raise TrialProcessingError(
                f"Tracker marker geometry differs from the calibration profile (p95 pairwise error {float(shape_p95):.2f} mm). "
                "This usually means the wrong eye tracker, a changed marker mount, or a label-order mismatch."
            )
        _report_progress(progress, 0.25, "Reconstructing the eye-tracker rigid-body pose")
        pose = PoseSeries.from_markers(
            c3d.time_sec,
            profile.tracker_template_mm,
            positions,
            marker_valid,
            min_markers=3,
            max_fit_rmse_mm=request.max_tracker_fit_rmse_mm,
        )
        gaze = recording.gaze.copy()
        required_columns = {"timestamp [ns]", "gaze x [px]", "gaze y [px]"}
        if not required_columns.issubset(gaze.columns):
            raise TrialProcessingError(
                f"gaze.csv is missing columns: {sorted(required_columns - set(gaze.columns))}"
            )
        _report_progress(progress, 0.38, "Mapping Timeseries timestamps into the Vicon timebase")
        timestamps_ns = gaze["timestamp [ns]"].to_numpy(dtype=np.int64)
        c3d_time = request.sync_mapping.pupil_ns_to_c3d_sec(timestamps_ns)
        in_c3d_time_range = (c3d_time >= c3d.time_sec[0]) & (c3d_time <= c3d.time_sec[-1])
        _report_progress(progress, 0.45, "Interpolating tracker pose to every gaze timestamp")
        _log(log, "Interpolating the 120 Hz eye-tracker pose to every Timeseries gaze sample...")
        interpolated = pose.interpolate(
            c3d_time, max_gap_sec=request.max_pose_interpolation_gap_sec
        )
        _report_progress(progress, 0.53, "Unprojecting 2D gaze into scene-camera directions")
        gaze_xy = gaze[["gaze x [px]", "gaze y [px]"]].to_numpy(dtype=float)
        camera_directions = unproject_gaze_pixels(gaze_xy, recording.intrinsics)
        sample_count = len(gaze)
        origins = np.full((sample_count, 3), np.nan, dtype=float)
        directions = np.full((sample_count, 3), np.nan, dtype=float)
        valid = interpolated.valid & np.isfinite(camera_directions).all(axis=1)
        if "worn" in gaze.columns:
            valid &= gaze["worn"].fillna(0).to_numpy(dtype=float) > 0.5
        indices = np.flatnonzero(valid)
        _report_progress(
            progress,
            0.60,
            "Applying the calibrated camera-to-tracker transform to all valid gaze samples",
        )
        if len(indices):
            transformed_origins, transformed_directions = (
                _apply_calibrated_camera_transform(
                    interpolated.rotations[indices],
                    interpolated.translations[indices],
                    profile.R_tracker_from_camera,
                    profile.camera_origin_tracker_mm,
                    camera_directions[indices],
                )
            )
            origins[indices] = transformed_origins
            directions[indices] = transformed_directions
        ray_valid = valid & np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1)
        _log(log, f"Generated {int(ray_valid.sum()):,} valid rays for {sample_count:,} Timeseries rows.")

        _report_progress(progress, 0.69, "Building the full row-preserving output table")
        gaze["gaze3d_input_row_index"] = np.arange(sample_count, dtype=np.int64)
        recording_start_ns = int(recording.info.get("start_time", timestamps_ns[0] if sample_count else 0))
        first_gaze_ns = int(timestamps_ns[0]) if sample_count else recording_start_ns
        gaze["gaze3d_time_from_recording_start_sec"] = (timestamps_ns.astype(np.float64) - float(recording_start_ns)) / 1e9
        gaze["gaze3d_time_from_first_gaze_sec"] = (timestamps_ns.astype(np.float64) - float(first_gaze_ns)) / 1e9
        gaze["gaze3d_c3d_time_sec"] = c3d_time
        gaze["gaze3d_c3d_frame_float"] = c3d_time * c3d.frame_rate_hz + c3d.first_frame
        gaze["gaze3d_in_c3d_time_range"] = in_c3d_time_range
        gaze["gaze3d_origin_x_mm"] = origins[:, 0]
        gaze["gaze3d_origin_y_mm"] = origins[:, 1]
        gaze["gaze3d_origin_z_mm"] = origins[:, 2]
        gaze["gaze3d_direction_x"] = directions[:, 0]
        gaze["gaze3d_direction_y"] = directions[:, 1]
        gaze["gaze3d_direction_z"] = directions[:, 2]
        gaze["gaze3d_tracker_pose_valid"] = interpolated.valid
        gaze["gaze3d_tracker_pose_interpolated"] = interpolated.interpolated
        gaze["gaze3d_tracker_interpolation_gap_sec"] = interpolated.interpolation_gap_sec
        gaze["gaze3d_tracker_fit_rmse_mm"] = interpolated.fit_rmse_mm
        gaze["gaze3d_tracker_marker_count"] = interpolated.marker_count
        gaze["gaze3d_ray_valid"] = ray_valid
        gaze["gaze3d_calibration_profile_id"] = profile.profile_id
        gaze["gaze3d_calibration_profile_format_version"] = int(profile.format_version)
        gaze["gaze3d_calibration_estimator_id"] = profile.calibration_estimator_id
        gaze["gaze3d_calibrated_transform_applied"] = True
        gaze["gaze3d_calibration_profile_sha256"] = profile_sha256
        gaze["gaze3d_camera_origin_tracker_x_mm"] = float(
            profile.camera_origin_tracker_mm[0]
        )
        gaze["gaze3d_camera_origin_tracker_y_mm"] = float(
            profile.camera_origin_tracker_mm[1]
        )
        gaze["gaze3d_camera_origin_tracker_z_mm"] = float(
            profile.camera_origin_tracker_mm[2]
        )
        gaze["gaze3d_sync_method"] = request.sync_mapping.method
        gaze["gaze3d_sync_scale"] = request.sync_mapping.scale
        gaze["gaze3d_sync_offset_sec"] = request.sync_mapping.offset_sec
        gaze["gaze3d_sync_uncertainty_ms"] = request.sync_mapping.uncertainty_ms
        gaze["gaze3d_canonical_participant_id"] = participant_id
        gaze["gaze3d_raw_identity_metadata_included"] = bool(
            request.include_raw_identity_metadata
        )
        gaze["gaze3d_pupil_wearer_name_raw"] = (
            str(recording.info.get("wearer_name", ""))
            if request.include_raw_identity_metadata
            else ""
        )
        gaze["gaze3d_c3d_tracker_labels"] = ";".join(trial_labels)

        recording_id = str(recording.info.get("recording_id", "")).strip()
        recording_name = _safe_name(
            request.output_name
            or (f"recording_{recording_id[:12]}" if recording_id else "recording")
        )
        participant = _safe_name(participant_id)
        stem = f"{recording_name}_{participant}_gaze3d_timeseries"
        output_csv = output / f"{stem}.csv"
        _report_progress(progress, 0.76, "Writing full 3D gaze-ray CSV")
        gaze.to_csv(output_csv, index=False)
        output_parquet: Path | None = None
        if request.write_parquet:
            try:
                _report_progress(progress, 0.82, "Writing Parquet output")
                output_parquet = output / f"{stem}.parquet"
                gaze.to_parquet(output_parquet, index=False)
            except Exception as exc:
                output_parquet = None
                _log(log, f"Parquet export skipped: {exc}")

        _report_progress(progress, 0.86, "Computing synchronization and tracker QC metrics")
        sync_json = request.sync_mapping.save(output / "sync_mapping.json")
        imu_metrics, imu_plot_data = _imu_sync_metrics(
            recording, pose, request.sync_mapping
        )
        overlap_rows = int(in_c3d_time_range.sum())
        overlap_valid_fraction = (
            float(ray_valid[in_c3d_time_range].mean()) if overlap_rows else 0.0
        )
        if not ray_valid.any() or overlap_rows == 0:
            qc_status = "FAIL"
        elif overlap_valid_fraction >= 0.90:
            qc_status = "PASS"
        else:
            qc_status = "WARN"
        qc_metrics = {
            "status": qc_status,
            "timeseries_rows": sample_count,
            "valid_ray_rows": int(ray_valid.sum()),
            "valid_ray_fraction": float(ray_valid.mean()) if sample_count else 0.0,
            "rows_mapped_inside_c3d_time_range": overlap_rows,
            "valid_ray_fraction_inside_c3d_time_range": overlap_valid_fraction,
            "rows_outside_c3d_time_range_retained_with_invalid_ray": int((~in_c3d_time_range).sum()),
            "calibration_applied": {
                "profile_id": profile.profile_id,
                "profile_sha256": profile_sha256,
                "profile_format_version": int(profile.format_version),
                "estimator_id": profile.calibration_estimator_id,
                "tracker_marker_labels_in_profile": list(profile.tracker_marker_labels),
                "camera_origin_tracker_mm": profile.camera_origin_tracker_mm.tolist(),
                "R_tracker_from_camera": profile.R_tracker_from_camera.tolist(),
                "T_tracker_from_camera": profile.T_tracker_from_camera.tolist(),
                "camera_to_tracker_centroid_distance_mm": float(
                    np.linalg.norm(profile.camera_origin_tracker_mm)
                ),
            },
            "c3d": {
                "frame_rate_hz": c3d.frame_rate_hz,
                "frame_count": c3d.frame_count,
                "duration_sec": c3d.duration_sec,
            },
            "tracker": {
                "labels": trial_labels,
                "valid_pose_frame_fraction": float(pose.valid.mean()),
                "fit_rmse_mm": _stats(pose.fit_rmse_mm),
                "shape_check": shape_check,
            },
            "profile_match": {
                "profile_module_serial": profile.module_serial_number,
                "trial_module_serial": str(
                    recording.info.get("module_serial_number", recording.intrinsics.serial_number)
                ),
                "module_serial_match": str(
                    recording.info.get("module_serial_number", recording.intrinsics.serial_number)
                )
                == str(profile.module_serial_number),
                "profile_frame_name": profile.frame_name,
                "trial_frame_name": str(recording.info.get("frame_name", "")),
                "scene_camera_matrix_max_abs_difference": matrix_difference,
                "scene_distortion_max_abs_difference": distortion_difference,
            },
            "sync": request.sync_mapping.to_dict(),
            "imu_sync_validation": imu_metrics,
            "scene_video_available": recording.scene.available,
        }
        qc_files = _make_trial_qc(
            recording=recording,
            c3d=c3d,
            pose=pose,
            gaze=gaze,
            origins=origins,
            directions=directions,
            ray_valid=ray_valid,
            imu_plot_data=imu_plot_data,
            imu_metrics=imu_metrics,
            qc_dir=qc_dir,
            progress=lambda value, text: _report_progress(
                progress, 0.88 + 0.08 * value, text
            ),
        )
        qc_metrics_json = output / "qc_metrics.json"
        qc_metrics_json.write_text(
            json.dumps(qc_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        recording_info_for_manifest = (
            dict(recording.info)
            if request.include_raw_identity_metadata
            else _privacy_safe_recording_info(dict(recording.info))
        )

        manifest = {
            "format": "NeonVicon-Gaze3D-ProcessingManifest",
            "format_version": 1,
            "recording_info": recording_info_for_manifest,
            "calibration_profile": profile.audit_summary(
                include_raw_identity_metadata=request.include_raw_identity_metadata,
                canonical_participant_id_override=participant_id,
            ),
            "inputs": _privacy_safe_input_paths(request),
            "input_sha256": {
                "timeseries_source": sha256_file(request.timeseries_source)
                if request.timeseries_source.is_file()
                else None,
                "scene_source": sha256_file(request.scene_source)
                if request.scene_source and request.scene_source.is_file()
                else None,
                "c3d": sha256_file(request.c3d_path),
                "calibration_profile": sha256_file(request.calibration_profile_path),
            },
            "outputs": {
                "csv": output_csv.name,
                "parquet": output_parquet.name if output_parquet else None,
                "sync": sync_json.name,
                "qc_report": "qc_report.html",
            },
            "calibrated_transform_used": {
                "profile_id": profile.profile_id,
                "profile_sha256": profile_sha256,
                "profile_format_version": int(profile.format_version),
                "calibration_estimator_id": profile.calibration_estimator_id,
                "T_tracker_from_camera": profile.T_tracker_from_camera.tolist(),
                "camera_origin_tracker_mm": profile.camera_origin_tracker_mm.tolist(),
                "R_tracker_from_camera": profile.R_tracker_from_camera.tolist(),
            },
            "parameters": {
                "max_pose_interpolation_gap_sec": request.max_pose_interpolation_gap_sec,
                "max_tracker_fit_rmse_mm": request.max_tracker_fit_rmse_mm,
                "no_trim_applied": True,
                "allow_profile_mismatch": request.allow_profile_mismatch,
                "include_raw_identity_metadata": request.include_raw_identity_metadata,
                "output_name": request.output_name,
                "row_preservation": "one output row per input gaze.csv row, in original order",
                "ray_origin_definition": "Neon scene-camera optical center in Vicon global coordinates",
                "ray_direction_definition": "unit gaze direction in Vicon global coordinates",
            },
        }
        _report_progress(progress, 0.97, "Writing processing manifest and HTML QC report")
        manifest_json = output / "processing_manifest.json"
        manifest_json.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        qc_report = output / "qc_report.html"
        qc_report.write_text(
            _render_trial_report(qc_metrics, qc_files, output_csv, output_parquet),
            encoding="utf-8",
        )
        _report_progress(progress, 1.0, "Full calibrated 3D gaze-ray export complete")
        return TrialResult(
            output_csv=output_csv,
            output_parquet=output_parquet,
            sync_json=sync_json,
            manifest_json=manifest_json,
            qc_metrics_json=qc_metrics_json,
            qc_report_html=qc_report,
            qc_files=qc_files,
            row_count=sample_count,
            valid_ray_count=int(ray_valid.sum()),
        )


def _make_trial_qc(
    *,
    recording: TimeseriesRecording,
    c3d: C3DData,
    pose: PoseSeries,
    gaze: pd.DataFrame,
    origins: np.ndarray,
    directions: np.ndarray,
    ray_valid: np.ndarray,
    imu_plot_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    imu_metrics: dict,
    qc_dir: Path,
    progress: Callable[[float, str], None] | None = None,
) -> list[Path]:
    files: list[Path] = []
    _report_progress(progress, 0.02, "Rendering tracker-pose QC")
    fig, first = plt.subplots(figsize=(11, 4))
    first.plot(c3d.time_sec, pose.marker_count, label="Visible markers")
    first.set_xlabel("C3D time (s)")
    first.set_ylabel("Marker count")
    second = first.twinx()
    second.plot(c3d.time_sec, pose.fit_rmse_mm, alpha=0.7, label="Rigid-fit RMSE")
    second.set_ylabel("Rigid-fit RMSE (mm)")
    first.set_title("Eye-tracker marker coverage and rigid-body fit")
    first.grid(True, alpha=0.3)
    fig.tight_layout()
    path = qc_dir / "tracker_pose_quality.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path)
    _report_progress(progress, 0.20, "Rendering ray-validity timeline")

    fig, axis = plt.subplots(figsize=(11, 3.5))
    time = gaze["gaze3d_c3d_time_sec"].to_numpy(dtype=float)
    axis.plot(time, ray_valid.astype(float), linewidth=0.7)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Mapped C3D time (s)")
    axis.set_ylabel("Ray valid")
    axis.set_title("Full Timeseries output coverage (no trim applied)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = qc_dir / "ray_validity_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path)
    _report_progress(progress, 0.38, "Rendering 3D gaze-ray preview")

    valid_indices = np.flatnonzero(ray_valid)
    if len(valid_indices):
        chosen = valid_indices[
            np.linspace(0, len(valid_indices) - 1, min(80, len(valid_indices))).round().astype(int)
        ]
        fig = plt.figure(figsize=(9, 7))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(
            origins[chosen, 0], origins[chosen, 1], origins[chosen, 2], label="Scene-camera trajectory"
        )
        for index in chosen[:: max(1, len(chosen) // 30)]:
            end = origins[index] + directions[index] * 500.0
            axis.plot(
                [origins[index, 0], end[0]],
                [origins[index, 1], end[1]],
                [origins[index, 2], end[2]],
                alpha=0.5,
            )
        axis.set_xlabel("Vicon X (mm)")
        axis.set_ylabel("Vicon Y (mm)")
        axis.set_zlabel("Vicon Z (mm)")
        axis.set_title("Decimated 3D gaze-ray preview (500 mm segments)")
        _equal_3d(axis, np.vstack([origins[chosen], origins[chosen] + directions[chosen] * 500.0]))
        fig.tight_layout()
        path = qc_dir / "gaze_rays_3d_preview.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

    _report_progress(progress, 0.58, "Rendering scene-video gaze overlay")
    if recording.scene.available and len(valid_indices):
        selected = valid_indices[
            np.linspace(0, len(valid_indices) - 1, min(6, len(valid_indices))).round().astype(int)
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.reshape(-1)
        for axis, index in zip(axes, selected):
            timestamp = int(gaze.iloc[index]["timestamp [ns]"])
            try:
                frame, frame_timestamp, section, delta = recording.scene.read_nearest(timestamp)
                x = float(gaze.iloc[index]["gaze x [px]"])
                y = float(gaze.iloc[index]["gaze y [px]"])
                cv2.circle(frame, (int(round(x)), int(round(y))), 18, (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"gaze row {index} | frame delta {delta / 1e6:.2f} ms",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axis.set_title(f"C3D t={gaze.iloc[index]['gaze3d_c3d_time_sec']:.2f}s")
            except Exception as exc:
                axis.text(0.05, 0.5, str(exc), wrap=True)
            axis.axis("off")
        for axis in axes[len(selected) :]:
            axis.axis("off")
        fig.suptitle("Scene-video gaze overlay sanity check")
        fig.tight_layout()
        path = qc_dir / "scene_gaze_overlay_contact_sheet.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)

    _report_progress(progress, 0.80, "Rendering IMU–Vicon synchronization QC")
    if imu_plot_data is not None:
        grid, tracker_speed, imu_speed = imu_plot_data
        fig, axis = plt.subplots(figsize=(11, 4))
        axis.plot(grid, tracker_speed, label="Vicon tracker angular speed")
        axis.plot(grid, imu_speed, label="Neon IMU gyro magnitude", alpha=0.8)
        axis.set_xlabel("C3D time (s)")
        axis.set_ylabel("Angular speed (rad/s)")
        title = "IMU–Vicon synchronization validation"
        if imu_metrics.get("available"):
            title += (
                f" | zero-lag r={imu_metrics.get('zero_lag_correlation', float('nan')):.3f}"
                f" | residual lag={imu_metrics.get('best_residual_lag_ms', float('nan')):.1f} ms"
            )
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        path = qc_dir / "imu_vicon_sync_validation.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)
    _report_progress(progress, 1.0, "QC visualizations complete")
    return files


def _equal_3d(axis, points: np.ndarray) -> None:
    minimum = np.nanmin(points, axis=0)
    maximum = np.nanmax(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(maximum - minimum) / 2.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _safe_name(text: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    ).strip("_")
    return result or "recording"


def _render_trial_report(
    metrics: dict, qc_files: list[Path], csv_path: Path, parquet_path: Path | None
) -> str:
    images = "".join(
        f'<section><h3>{path.stem.replace("_", " ").title()}</h3><img src="qc/{path.name}" alt="{path.name}"></section>'
        for path in qc_files
    )
    parquet = parquet_path.name if parquet_path else "Not written (optional dependency unavailable)"
    calibration = metrics.get("calibration_applied", {})
    calibration_summary = (
        f"Profile <code>{calibration.get('profile_id', 'unknown')}</code> "
        f"(<code>{calibration.get('estimator_id', 'unknown')}</code>) was applied. "
        f"Camera origin in tracker coordinates: "
        f"<code>{calibration.get('camera_origin_tracker_mm', [])}</code> mm."
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>3D Gaze QC</title>
<style>body{{font-family:Arial,sans-serif;max-width:1150px;margin:30px auto;line-height:1.45}}img{{max-width:100%;border:1px solid #ccc}}pre{{background:#f5f5f5;padding:12px;overflow:auto}}</style></head>
<body><h1>Neon–Vicon Timeseries 3D Gaze QC</h1>
<p><strong>CSV:</strong> {csv_path.name}</p><p><strong>Parquet:</strong> {parquet}</p>
<p><strong>No trim was applied.</strong> Output row order and row count follow the input <code>gaze.csv</code>.</p>
<p><strong>Calibrated 6-DoF transform applied:</strong> {calibration_summary}</p>
<p>For every valid sample, the exporter computes <code>R_world_camera = R_world_tracker × R_tracker_camera</code>, then transforms the calibrated camera optical centre and gaze direction into Vicon world coordinates.</p>
<h2>Metrics</h2><pre>{json.dumps(metrics, ensure_ascii=False, indent=2)}</pre>
<h2>Visual checks</h2>{images}</body></html>"""
