from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from ..io.archive import sha256_file
from ..io.c3d_reader import C3DData, read_c3d
from ..io.profile import CalibrationProfile
from ..io.pupil_recording import NativeRecording, unproject_gaze_pixels
from .geometry import (
    PoseSeries,
    robust_average_transforms,
    build_marker_template,
    invert_transform,
    make_planar_board_template,
    make_transform,
    normalize,
    rigid_fit,
    transform_points,
)


class CalibrationError(RuntimeError):
    pass


CORNER_ORDER = ("TL", "TR", "BR", "BL")


@dataclass
class CalibrationRequest:
    pupil_calibration_source: Path
    tracker_c3d_path: Path
    board_c3d_path: Path
    tracker_marker_labels: list[str]
    board_corner_labels: dict[str, dict[str, str]]
    outer_corner_labels: dict[str, str]
    canonical_participant_id: str
    display_name: str
    output_directory: Path
    head_c3d_path: Path | None = None
    head_marker_labels: list[str] | None = None
    sample_every_n_frames: int = 2
    max_detection_frames: int = 400
    apriltag_ids: tuple[int, ...] = (0, 1, 2, 3)
    max_pose_gap_sec: float = 0.1
    tracker_identity_audit: dict | None = None
    include_raw_identity_metadata: bool = False


@dataclass
class DetectionFrame:
    frame_index: int
    timestamp_ns: int
    scene_time_sec: float
    object_points_mm: np.ndarray
    image_points_px: np.ndarray
    direct_T_camera_from_board: np.ndarray
    direct_rms_px: float
    detection_mode: str


@dataclass
class CalibrationResult:
    profile: CalibrationProfile
    profile_path: Path
    qc_files: list[Path]
    report_path: Path
    metrics_path: Path


def _log(callback: Callable[[str], None] | None, text: str) -> None:
    if callback:
        callback(text)


def _privacy_safe_identity_audit(audit: dict | None) -> dict | None:
    """Return an audit summary that contains no actor/wearer names or local paths."""

    if not audit:
        return audit
    selected_index = audit.get("selected_candidate_index")
    selected_hash = None
    candidate_summaries: list[dict] = []
    for candidate in audit.get("candidates", []):
        summary = {
            "candidate_index": candidate.get("candidate_index"),
            "sha256": candidate.get("sha256"),
            "marker_labels": list(candidate.get("marker_labels", [])),
            "frame_count": candidate.get("frame_count"),
            "frame_rate_hz": candidate.get("frame_rate_hz"),
            "valid_centroid_frames": candidate.get("valid_centroid_frames"),
            "camera_distance_mm": dict(candidate.get("camera_distance_mm", {})),
            "camera_proximity_score_mm": candidate.get("camera_proximity_score_mm"),
            "camera_frames_under_250mm_fraction": candidate.get(
                "camera_frames_under_250mm_fraction"
            ),
            "camera_board_mapping_variant": candidate.get(
                "camera_board_mapping_variant"
            ),
            "camera_direct_pnp_rms_px": dict(
                candidate.get("camera_direct_pnp_rms_px", {})
            ),
        }
        candidate_summaries.append(summary)
        if candidate.get("candidate_index") == selected_index:
            selected_hash = candidate.get("sha256")
    return {
        "privacy_redacted": True,
        "identity_evidence_type": audit.get("identity_evidence_type", ""),
        "method": audit.get("method", ""),
        "confidence": audit.get("confidence", ""),
        "selected_candidate_index": selected_index,
        "selected_tracker_sha256": selected_hash,
        "selected_tracker_labels": list(audit.get("selected_tracker_labels", [])),
        "board_mapping_source": audit.get("board_mapping_source", ""),
        "recommended_board_corner_labels": dict(
            audit.get("recommended_board_corner_labels", {})
        ),
        "recommended_outer_corner_labels": dict(
            audit.get("recommended_outer_corner_labels", {})
        ),
        "head_qc_available": bool(audit.get("head_qc_available", False)),
        "candidates": candidate_summaries,
        "warnings": [
            "Raw Pupil wearer identity, C3D subject names, and local file paths were redacted."
        ],
    }


def _report_progress(
    callback: Callable[[float, str], None] | None,
    value: float,
    text: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), text)


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CalibrationError(f"Could not read video frame {frame_index}")
        return frame
    finally:
        capture.release()


def _detect_apriltags(
    recording: NativeRecording,
    *,
    apriltag_ids: tuple[int, ...],
    object_points_by_tag: dict[int, np.ndarray],
    sample_every_n_frames: int,
    max_detection_frames: int,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[DetectionFrame], int]:
    """Detect the four board tags with an automatic full-resolution fallback.

    Large, nearby tags are first detected on an 800-pixel-wide image for speed.
    If any required tag is missing, the same sampled frame is immediately retried
    at native resolution. This is important for distant-tag recordings where
    the reduced image can miss tags that remain detectable in the native frame.
    """

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    capture = cv2.VideoCapture(str(recording.scene_video_path))
    detections: list[DetectionFrame] = []
    all_four_count = 0
    total_frames = min(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(recording.scene_timestamps_ns)
    )

    def detect_and_refine(gray: np.ndarray, scale: float) -> dict[int, np.ndarray]:
        if scale < 1.0:
            image = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            image = gray
        corners, ids, _ = detector.detectMarkers(image)
        found: dict[int, np.ndarray] = {}
        if ids is None:
            return found
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.005,
        )
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            if scale < 1.0:
                points = points / scale
            refined = cv2.cornerSubPix(
                gray,
                points.reshape(-1, 1, 2),
                (5, 5),
                (-1, -1),
                criteria,
            ).reshape(4, 2)
            found[int(marker_id)] = refined.astype(float)
        return found

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame_index >= total_frames:
                break
            if frame_index % max(1, sample_every_n_frames) != 0:
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coarse_scale = min(1.0, 800.0 / float(gray.shape[1]))
            found = detect_and_refine(gray, coarse_scale)
            detection_mode = "downscaled"
            if not all(marker_id in found for marker_id in apriltag_ids):
                found = detect_and_refine(gray, 1.0)
                detection_mode = "full_resolution_fallback"
            if all(marker_id in found for marker_id in apriltag_ids):
                all_four_count += 1
                object_points: list[np.ndarray] = []
                for marker_id in apriltag_ids:
                    object_points.extend(object_points_by_tag[marker_id])
                object_array = np.asarray(object_points, dtype=float)
                # OpenCV may cyclically rotate or reflect the canonical tag
                # corner order relative to the physical TL/TR/BR/BL labels.
                permutations: list[tuple[int, int, int, int]] = []
                base = np.arange(4)
                for reverse in (False, True):
                    order = base[::-1] if reverse else base
                    for shift in range(4):
                        permutation = tuple(int(x) for x in np.roll(order, -shift))
                        if permutation not in permutations:
                            permutations.append(permutation)
                best_solution = None
                for permutation in permutations:
                    image_points: list[np.ndarray] = []
                    for marker_id in apriltag_ids:
                        image_points.extend(found[marker_id][list(permutation)])
                    candidate_image = np.asarray(image_points, dtype=float)
                    ok_pnp, rvec, tvec = cv2.solvePnP(
                        object_array,
                        candidate_image,
                        recording.intrinsics.matrix,
                        recording.intrinsics.distortion,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if not ok_pnp:
                        continue
                    projected, _ = cv2.projectPoints(
                        object_array,
                        rvec,
                        tvec,
                        recording.intrinsics.matrix,
                        recording.intrinsics.distortion,
                    )
                    error = projected.reshape(-1, 2) - candidate_image
                    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                    if best_solution is None or rms < best_solution[0]:
                        best_solution = (rms, candidate_image, rvec, tvec)
                if best_solution is not None:
                    rms, image_array, rvec, tvec = best_solution
                    rotation, _ = cv2.Rodrigues(rvec)
                    T_camera_from_board = make_transform(rotation, tvec.reshape(3))
                    detections.append(
                        DetectionFrame(
                            frame_index=frame_index,
                            timestamp_ns=int(recording.scene_timestamps_ns[frame_index]),
                            scene_time_sec=float(
                                (
                                    int(recording.scene_timestamps_ns[frame_index])
                                    - int(recording.scene_timestamps_ns[0])
                                )
                                / 1e9
                            ),
                            object_points_mm=object_array,
                            image_points_px=image_array,
                            direct_T_camera_from_board=T_camera_from_board,
                            direct_rms_px=rms,
                            detection_mode=detection_mode,
                        )
                    )
            if frame_index % 30 == 0:
                _report_progress(
                    progress,
                    frame_index / max(1, total_frames),
                    f"Detecting AprilTags ({frame_index}/{total_frames} frames)",
                )
            frame_index += 1
    finally:
        capture.release()
    if len(detections) > max_detection_frames:
        indices = (
            np.linspace(0, len(detections) - 1, max_detection_frames)
            .round()
            .astype(int)
        )
        detections = [detections[index] for index in indices]
    _report_progress(
        progress,
        1.0,
        f"AprilTag detection complete ({len(detections)} usable sampled frames)",
    )
    return detections, all_four_count

def _prepare_board(
    board: C3DData,
    board_corner_labels: dict[str, dict[str, str]],
    outer_corner_labels: dict[str, str],
    apriltag_ids: tuple[int, ...],
) -> tuple[list[str], np.ndarray, dict[int, np.ndarray], np.ndarray, PoseSeries]:
    ordered_labels: list[str] = []
    for marker_id in apriltag_ids:
        mapping = board_corner_labels[str(marker_id)]
        ordered_labels.extend(mapping[corner] for corner in CORNER_ORDER)
    all_needed = list(dict.fromkeys(ordered_labels + list(outer_corner_labels.values())))
    positions, valid = board.select(all_needed)
    if not np.all(valid.sum(axis=0) > 0):
        missing = [all_needed[i] for i in np.flatnonzero(valid.sum(axis=0) == 0)]
        raise CalibrationError(f"Board markers never visible: {missing}")
    median_world = np.nanmedian(np.where(valid[..., None], positions, np.nan), axis=0)
    label_to_index = {label: i for i, label in enumerate(all_needed)}
    local_all, _ = make_planar_board_template(
        median_world,
        top_left_index=label_to_index[outer_corner_labels["TL"]],
        top_right_index=label_to_index[outer_corner_labels["TR"]],
        bottom_left_index=label_to_index[outer_corner_labels["BL"]],
    )
    outer_tl_local = local_all[label_to_index[outer_corner_labels["TL"]]].copy()
    local_all -= outer_tl_local
    object_points_by_tag: dict[int, np.ndarray] = {}
    for marker_id in apriltag_ids:
        labels = [board_corner_labels[str(marker_id)][corner] for corner in CORNER_ORDER]
        object_points_by_tag[marker_id] = np.asarray(
            [local_all[label_to_index[label]] for label in labels], dtype=float
        )
    # Use every selected point as the board rigid template.
    board_template = local_all
    board_pose = PoseSeries.from_markers(
        board.time_sec, board_template, positions, valid, min_markers=4
    )
    outer_local = np.asarray(
        [
            local_all[label_to_index[outer_corner_labels[key]]]
            for key in ("TL", "TR", "BR", "BL")
        ]
    )
    return all_needed, board_template, object_points_by_tag, outer_local, board_pose


def _prepare_tracker(
    tracker: C3DData, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PoseSeries]:
    positions, valid = tracker.select(labels)
    template = build_marker_template(positions, valid)
    pose = PoseSeries.from_markers(
        tracker.time_sec, template, positions, valid, min_markers=3
    )
    return template, positions, valid, pose


def _transform_candidates(
    detections: list[DetectionFrame],
    offset_sec: float,
    tracker_pose: PoseSeries,
    board_pose: PoseSeries,
    max_gap_sec: float,
) -> np.ndarray:
    scene_times = np.asarray([d.scene_time_sec for d in detections])
    query = scene_times + offset_sec
    tracker = tracker_pose.interpolate(query, max_gap_sec=max_gap_sec)
    board = board_pose.interpolate(query, max_gap_sec=max_gap_sec)
    valid = tracker.valid & board.valid
    transforms: list[np.ndarray] = []
    for index in np.flatnonzero(valid):
        T_world_from_tracker = make_transform(
            tracker.rotations[index], tracker.translations[index]
        )
        T_world_from_board = make_transform(board.rotations[index], board.translations[index])
        T_camera_from_board = detections[index].direct_T_camera_from_board
        T_world_from_camera = T_world_from_board @ invert_transform(T_camera_from_board)
        T_tracker_from_camera = invert_transform(T_world_from_tracker) @ T_world_from_camera
        transforms.append(T_tracker_from_camera)
    if not transforms:
        return np.empty((0, 4, 4))
    return np.asarray(transforms)


def _offset_dispersion_cost_nearest(
    detections: list[DetectionFrame],
    offset_sec: float,
    tracker_pose: PoseSeries,
    board_pose: PoseSeries,
    *,
    max_frames: int = 60,
) -> float:
    """Fast temporal-alignment score based on all-frame transform consistency.

    Do not reject transform outliers in this temporal search: a wrong offset can
    otherwise look artificially good after inconsistent frames are discarded.
    Robust rejection is applied only after the time basin has been selected.
    """

    scene_times = np.asarray([d.scene_time_sec for d in detections])
    query = scene_times + offset_sec
    rate = 1.0 / float(np.median(np.diff(tracker_pose.time_sec)))
    indices = np.rint(query * rate).astype(int)
    in_range = (indices >= 0) & (indices < len(tracker_pose.time_sec))
    rows = np.flatnonzero(in_range)
    if len(rows) < 10:
        return float("inf")
    if len(rows) > max_frames:
        selection = np.linspace(0, len(rows) - 1, max_frames).round().astype(int)
        rows = rows[selection]
    transforms: list[np.ndarray] = []
    for row in rows:
        frame = indices[row]
        if not (tracker_pose.valid[frame] and board_pose.valid[frame]):
            continue
        T_w_t = make_transform(
            tracker_pose.rotations[frame], tracker_pose.translations[frame]
        )
        T_w_b = make_transform(
            board_pose.rotations[frame], board_pose.translations[frame]
        )
        T_w_c = T_w_b @ invert_transform(
            detections[row].direct_T_camera_from_board
        )
        transforms.append(invert_transform(T_w_t) @ T_w_c)
    if len(transforms) < 10:
        return float("inf")
    transforms_array = np.asarray(transforms)
    center_t = np.median(transforms_array[:, :3, 3], axis=0)
    center_r = Rotation.from_matrix(transforms_array[:, :3, :3]).mean()
    translation_error = np.linalg.norm(
        transforms_array[:, :3, 3] - center_t, axis=1
    )
    rotation_error = np.degrees(
        (
            center_r.inv()
            * Rotation.from_matrix(transforms_array[:, :3, :3])
        ).magnitude()
    )
    cost = (
        float(np.median(translation_error))
        + 0.25 * float(np.percentile(translation_error, 90))
        + 10.0 * float(np.median(rotation_error))
        + 2.0 * float(np.percentile(rotation_error, 90))
    )
    overlap_penalty = 0.5 / math.sqrt(len(transforms_array))
    return cost + overlap_penalty

def _offset_cost_continuous(
    detections: list[DetectionFrame],
    offset_sec: float,
    tracker_pose: PoseSeries,
    board_pose: PoseSeries,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    max_gap_sec: float,
) -> float:
    scene_times = np.asarray([d.scene_time_sec for d in detections])
    query = scene_times + offset_sec
    tracker = tracker_pose.interpolate(query, max_gap_sec=max_gap_sec)
    board = board_pose.interpolate(query, max_gap_sec=max_gap_sec)
    valid = tracker.valid & board.valid
    if valid.sum() < 10:
        return float("inf")
    transforms: list[np.ndarray] = []
    for row in np.flatnonzero(valid):
        T_w_t = make_transform(tracker.rotations[row], tracker.translations[row])
        T_w_b = make_transform(board.rotations[row], board.translations[row])
        T_w_c = T_w_b @ invert_transform(detections[row].direct_T_camera_from_board)
        transforms.append(invert_transform(T_w_t) @ T_w_c)
    mean_transform, _, _ = robust_average_transforms(
        np.asarray(transforms),
        maximum_translation_gate_mm=80.0,
        maximum_rotation_gate_deg=20.0,
    )
    errors: list[float] = []
    for row in np.flatnonzero(valid):
        T_w_t = make_transform(tracker.rotations[row], tracker.translations[row])
        T_w_b = make_transform(board.rotations[row], board.translations[row])
        projected = _project_frame(
            detections[row].object_points_mm,
            T_w_b,
            T_w_t,
            mean_transform,
            camera_matrix,
            distortion,
        )
        delta = projected - detections[row].image_points_px
        errors.append(float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))))
    return float(np.sqrt(np.mean(np.square(errors)))) + 0.25 / math.sqrt(len(errors))


def _search_offset(
    detections: list[DetectionFrame],
    tracker_pose: PoseSeries,
    board_pose: PoseSeries,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    max_gap_sec: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate calibration-only time offset without moving camera extrinsics.

    Several low-dispersion temporal basins are retained and compared by actual
    fixed-transform reprojection.  This avoids choosing a short but internally
    quiet overlap from an otherwise wrong part of the recording.
    """

    scene = np.asarray([d.scene_time_sec for d in detections])
    max_c3d = min(tracker_pose.time_sec[-1], board_pose.time_sec[-1])
    minimum = -float(scene.max())
    maximum = float(max_c3d - scene.min())
    coarse = np.arange(minimum, maximum + 0.050, 0.050)
    coarse_cost = np.asarray(
        [
            _offset_dispersion_cost_nearest(
                detections,
                value,
                tracker_pose,
                board_pose,
                max_frames=48,
            )
            for value in coarse
        ]
    )
    finite_indices = np.flatnonzero(np.isfinite(coarse_cost))
    if not len(finite_indices):
        raise CalibrationError(
            "Could not find temporal overlap between calibration video and C3D"
        )

    # Retain distinct temporal basins.  A wrong offset can occasionally have a
    # low dispersion over a short quiet segment, so the global minimum alone is
    # not trusted.
    basin_centres: list[float] = []
    for index in finite_indices[np.argsort(coarse_cost[finite_indices])]:
        value = float(coarse[index])
        if all(abs(value - existing) >= 0.35 for existing in basin_centres):
            basin_centres.append(value)
        if len(basin_centres) >= 6:
            break

    basin_candidates: list[tuple[float, float]] = []
    for centre in basin_centres:
        medium = np.arange(centre - 0.120, centre + 0.1201, 0.004)
        medium_cost = np.asarray(
            [
                _offset_dispersion_cost_nearest(
                    detections,
                    value,
                    tracker_pose,
                    board_pose,
                    max_frames=96,
                )
                for value in medium
            ]
        )
        if not np.isfinite(medium_cost).any():
            continue
        candidate = float(medium[np.nanargmin(medium_cost)])
        reprojection_cost = _offset_cost_continuous(
            detections,
            candidate,
            tracker_pose,
            board_pose,
            camera_matrix,
            distortion,
            max_gap_sec,
        )
        if np.isfinite(reprojection_cost):
            basin_candidates.append((float(reprojection_cost), candidate))
    if not basin_candidates:
        raise CalibrationError(
            "Temporal candidate basins were found, but none produced a valid fixed-transform reprojection"
        )
    basin_candidates.sort(key=lambda item: item[0])
    best = float(basin_candidates[0][1])

    fine = np.arange(best - 0.030, best + 0.0301, 0.001)
    fine_cost = np.asarray(
        [
            _offset_cost_continuous(
                detections,
                value,
                tracker_pose,
                board_pose,
                camera_matrix,
                distortion,
                max_gap_sec,
            )
            for value in fine
        ]
    )
    if np.isfinite(fine_cost).any():
        best = float(fine[np.nanargmin(fine_cost)])
    return best, fine, fine_cost

def _project_frame(
    object_points_board: np.ndarray,
    T_world_from_board: np.ndarray,
    T_world_from_tracker: np.ndarray,
    T_tracker_from_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    T_world_from_camera = T_world_from_tracker @ T_tracker_from_camera
    T_camera_from_world = invert_transform(T_world_from_camera)
    world_points = transform_points(T_world_from_board, object_points_board)
    camera_points = transform_points(T_camera_from_world, world_points)
    projected, _ = cv2.projectPoints(
        camera_points,
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


def _fixed_reprojection_residuals(
    detections: list[DetectionFrame],
    offset_sec: float,
    tracker_pose: PoseSeries,
    board_pose: PoseSeries,
    T_tracker_from_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    scene_times = np.asarray([d.scene_time_sec for d in detections], dtype=float)
    query = scene_times + float(offset_sec)
    tracker = tracker_pose.interpolate(query, max_gap_sec=max_gap_sec)
    board = board_pose.interpolate(query, max_gap_sec=max_gap_sec)
    point_residuals: list[np.ndarray] = []
    frame_rms = np.full(len(detections), np.nan, dtype=float)
    for index, detection in enumerate(detections):
        if not (tracker.valid[index] and board.valid[index]):
            continue
        T_w_t = make_transform(tracker.rotations[index], tracker.translations[index])
        T_w_b = make_transform(board.rotations[index], board.translations[index])
        projected = _project_frame(
            detection.object_points_mm,
            T_w_b,
            T_w_t,
            T_tracker_from_camera,
            camera_matrix,
            distortion,
        )
        residual = projected - detection.image_points_px
        point_residuals.append(residual)
        frame_rms[index] = float(
            np.sqrt(np.mean(np.sum(residual**2, axis=1)))
        )
    if not point_residuals:
        return np.empty((0, 2), dtype=float), frame_rms
    return np.concatenate(point_residuals, axis=0), frame_rms

def _percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }




def _plot_tracker_identity_audit(audit: dict | None, output_directory: Path) -> Path | None:
    """Render either head-distance or direct camera-to-tracker identity evidence."""

    if not audit or not audit.get("candidates"):
        return None
    candidates = list(audit.get("candidates") or [])
    evidence_type = str(audit.get("identity_evidence_type", "head_marker_proximity"))
    selected = audit.get("selected_tracker_filename") or "none"
    confidence = audit.get("confidence") or "unknown"
    wearer = audit.get("wearer_name_raw") or "unknown"

    if evidence_type == "apriltag_camera_to_tracker_proximity":
        labels: list[str] = []
        medians: list[float] = []
        p05: list[float] = []
        p95: list[float] = []
        for row, candidate in enumerate(candidates):
            summary = candidate.get("camera_distance_mm") or {}
            if not summary.get("n", 0):
                continue
            labels.append(str(candidate.get("filename") or f"Candidate {row + 1}"))
            medians.append(float(summary["median"]))
            p05.append(float(summary["p05"]))
            p95.append(float(summary["p95"]))
        if not medians:
            return None
        values = np.asarray(medians, dtype=float)
        lower = values - np.asarray(p05, dtype=float)
        upper = np.asarray(p95, dtype=float) - values
        figure, axis = plt.subplots(figsize=(max(8.0, 2.5 + 2.0 * len(labels)), 5.2))
        x = np.arange(len(labels), dtype=float)
        bars = axis.bar(x, values, yerr=np.vstack([lower, upper]), capsize=6)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.set_ylabel("Camera optical centre → tracker centroid [mm]\nmedian with p05–p95]")
        axis.set_title(
            "Automatic tracker identity: AprilTag camera fallback\n"
            f"wearer={wearer} · selected={selected} · confidence={confidence}",
            fontsize=11,
        )
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_directory / "tracker_identity_camera_distance.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        return path

    subjects = list(audit.get("available_c3d_subjects") or [])
    if not subjects:
        return None
    values = np.full((len(candidates), len(subjects)), np.nan, dtype=float)
    labels: list[str] = []
    for row, candidate in enumerate(candidates):
        labels.append(str(candidate.get("filename") or f"Candidate {row + 1}"))
        distances = candidate.get("subject_distance_mm") or {}
        for col, subject in enumerate(subjects):
            summary = distances.get(subject) or {}
            if summary.get("n", 0) and summary.get("median") is not None:
                values[row, col] = float(summary["median"])
    if not np.isfinite(values).any():
        return None

    figure, axis = plt.subplots(figsize=(max(8.5, 2.5 + 1.6 * len(subjects)), 5.2))
    x = np.arange(len(subjects), dtype=float)
    width = min(0.36, 0.8 / max(1, len(candidates)))
    offsets = (np.arange(len(candidates)) - (len(candidates) - 1) / 2.0) * width
    for row, (label, offset) in enumerate(zip(labels, offsets)):
        bars = axis.bar(x + offset, values[row], width=width, label=label)
        for bar, value in zip(bars, values[row]):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(value),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    axis.set_xticks(x, subjects)
    axis.set_ylabel("Median tracker-centroid → head-centre distance [mm]")
    axis.set_title(
        "Automatic tracker identity: labelled head markers\n"
        f"wearer={wearer} · selected={selected} · confidence={confidence}",
        fontsize=11,
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path = output_directory / "tracker_identity_head_distance.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path

def _calibration_qc(
    *,
    request: CalibrationRequest,
    recording: NativeRecording,
    detections: list[DetectionFrame],
    tracker_pose: PoseSeries,
    tracker_template_mm: np.ndarray,
    board_pose: PoseSeries,
    T_tracker_from_camera: np.ndarray,
    offset_sec: float,
    outer_local: np.ndarray,
    fine_offsets: np.ndarray,
    fine_cost: np.ndarray,
    output_directory: Path,
    candidate_transforms: np.ndarray | None = None,
    candidate_scene_times_sec: np.ndarray | None = None,
    candidate_inlier_mask: np.ndarray | None = None,
    estimator_metrics: dict | None = None,
    head_time_sec: np.ndarray | None = None,
    head_centers_mm: np.ndarray | None = None,
    head_valid: np.ndarray | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[dict, list[Path]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    _report_progress(progress, 0.01, "Interpolating calibration poses for QC")
    scene_times = np.asarray([d.scene_time_sec for d in detections])
    query = scene_times + offset_sec
    tracker = tracker_pose.interpolate(query, max_gap_sec=request.max_pose_gap_sec)
    board = board_pose.interpolate(query, max_gap_sec=request.max_pose_gap_sec)
    head_query = None
    head_query_valid = None
    if head_time_sec is not None and head_centers_mm is not None and head_valid is not None and np.any(head_valid):
        valid_indices = np.flatnonzero(head_valid)
        head_query = np.column_stack([
            np.interp(query, head_time_sec[valid_indices], head_centers_mm[valid_indices, component], left=np.nan, right=np.nan)
            for component in range(3)
        ])
        head_query_valid = np.isfinite(head_query).all(axis=1)
    frame_rms = np.full(len(detections), np.nan)
    projected_all: list[np.ndarray | None] = []
    optical_angles: list[float] = []
    camera_board_distances: list[float] = []
    camera_head_distances: list[float] = []
    camera_forward_projections: list[float] = []
    for index, detection in enumerate(detections):
        if not (tracker.valid[index] and board.valid[index]):
            projected_all.append(None)
            continue
        T_w_t = make_transform(tracker.rotations[index], tracker.translations[index])
        T_w_b = make_transform(board.rotations[index], board.translations[index])
        projected = _project_frame(
            detection.object_points_mm,
            T_w_b,
            T_w_t,
            T_tracker_from_camera,
            recording.intrinsics.matrix,
            recording.intrinsics.distortion,
        )
        projected_all.append(projected)
        error = projected - detection.image_points_px
        frame_rms[index] = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        T_w_c = T_w_t @ T_tracker_from_camera
        board_center_world = transform_points(T_w_b, outer_local).mean(axis=0)
        vector = board_center_world - T_w_c[:3, 3]
        camera_board_distances.append(float(np.linalg.norm(vector)))
        optical = T_w_c[:3, 2]
        cosine = np.clip(np.dot(normalize(optical), normalize(vector)), -1.0, 1.0)
        optical_angles.append(float(np.degrees(np.arccos(cosine))))
        if head_query is not None and head_query_valid is not None and head_query_valid[index]:
            head_center = head_query[index]
            camera_offset = T_w_c[:3, 3] - head_center
            forward = normalize(board_center_world - head_center)
            camera_head_distances.append(float(np.linalg.norm(camera_offset)))
            camera_forward_projections.append(float(np.dot(camera_offset, forward)))

    _report_progress(progress, 0.10, "Computing fixed-transform reprojection errors")
    qc_files: list[Path] = []
    # Reprojection overlay contact sheet.
    usable = [i for i, item in enumerate(projected_all) if item is not None]
    if usable:
        _report_progress(progress, 0.14, "Rendering AprilTag reprojection contact sheet")
        chosen = np.linspace(0, len(usable) - 1, min(6, len(usable))).round().astype(int)
        chosen = [usable[i] for i in chosen]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes_flat = axes.reshape(-1)
        for sheet_index, (axis, index) in enumerate(zip(axes_flat, chosen)):
            _report_progress(
                progress,
                0.15 + 0.14 * (sheet_index / max(1, len(chosen))),
                f"Reading calibration scene frame {sheet_index + 1}/{len(chosen)} for QC",
            )
            frame = _read_video_frame(recording.scene_video_path, detections[index].frame_index)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axis.imshow(frame_rgb)
            axis.scatter(
                detections[index].image_points_px[:, 0],
                detections[index].image_points_px[:, 1],
                s=16,
                label="Detected",
            )
            axis.scatter(
                projected_all[index][:, 0],
                projected_all[index][:, 1],
                s=20,
                marker="x",
                label="Projected",
            )
            axis.set_title(
                f"Frame {detections[index].frame_index} | RMS {frame_rms[index]:.2f}px"
            )
            axis.axis("off")
        for axis in axes_flat[len(chosen) :]:
            axis.axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        path = output_directory / "calibration_reprojection_overlay.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        qc_files.append(path)

    _report_progress(progress, 0.31, "Rendering reprojection-error timeline")
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(scene_times, frame_rms)
    axis.set_xlabel("Neon scene time (s)")
    axis.set_ylabel("Reprojection RMS (px)")
    axis.set_title("Fixed camera-to-tracker transform reprojection")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_directory / "calibration_reprojection_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    qc_files.append(path)

    _report_progress(progress, 0.40, "Rendering calibration time-alignment QC")
    fig, axis = plt.subplots(figsize=(10, 4))
    finite = np.isfinite(fine_cost)
    axis.plot(fine_offsets[finite], fine_cost[finite])
    axis.axvline(offset_sec, linestyle="--", label=f"Selected {offset_sec:.6f}s")
    axis.set_xlabel("C3D time = scene time + offset (s)")
    axis.set_ylabel("Rigid-transform dispersion cost")
    axis.set_title("Calibration temporal alignment")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = output_directory / "calibration_time_alignment.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    qc_files.append(path)

    if candidate_transforms is not None and len(candidate_transforms):
        _report_progress(progress, 0.49, "Rendering camera-to-glasses distance stability")
        candidate_transforms = np.asarray(candidate_transforms, dtype=float)
        distances = np.linalg.norm(candidate_transforms[:, :3, 3], axis=1)
        if candidate_scene_times_sec is None or len(candidate_scene_times_sec) != len(distances):
            candidate_scene_times_sec = np.arange(len(distances), dtype=float)
            x_label = "Candidate index"
        else:
            candidate_scene_times_sec = np.asarray(candidate_scene_times_sec, dtype=float)
            x_label = "Neon scene time (s)"
        if candidate_inlier_mask is None or len(candidate_inlier_mask) != len(distances):
            candidate_inlier_mask = np.ones(len(distances), dtype=bool)
        else:
            candidate_inlier_mask = np.asarray(candidate_inlier_mask, dtype=bool)
        fig, axis = plt.subplots(figsize=(10, 4.5))
        axis.scatter(
            candidate_scene_times_sec[~candidate_inlier_mask],
            distances[~candidate_inlier_mask],
            marker="x",
            s=24,
            label="Rejected candidate",
        )
        axis.scatter(
            candidate_scene_times_sec[candidate_inlier_mask],
            distances[candidate_inlier_mask],
            s=18,
            label="Inlier candidate",
        )
        profile_distance = float(np.linalg.norm(T_tracker_from_camera[:3, 3]))
        inlier_distances = distances[candidate_inlier_mask]
        axis.axhline(
            profile_distance,
            linestyle="-",
            label=f"Profile transform norm {profile_distance:.2f} mm",
        )
        if len(inlier_distances):
            axis.axhline(
                float(np.median(inlier_distances)),
                linestyle="--",
                label=f"Per-frame median {np.median(inlier_distances):.2f} mm",
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Camera → six-marker centroid distance (mm)")
        axis.set_title("Camera-to-glasses distance stability")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        path = output_directory / "calibration_camera_distance_stability.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        qc_files.append(path)

    # 3-D geometry at the median usable frame.
    if usable:
        _report_progress(progress, 0.58, "Rendering 3-D calibration geometry")
        index = usable[len(usable) // 2]
        T_w_t = make_transform(tracker.rotations[index], tracker.translations[index])
        T_w_b = make_transform(board.rotations[index], board.translations[index])
        T_w_c = T_w_t @ T_tracker_from_camera
        board_world = transform_points(T_w_b, outer_local)
        tracker_world = transform_points(T_w_t, tracker_template_mm)
        fig = plt.figure(figsize=(9, 7))
        axis = fig.add_subplot(111, projection="3d")
        axis.scatter(tracker_world[:, 0], tracker_world[:, 1], tracker_world[:, 2], label="Tracker markers")
        axis.scatter(board_world[:, 0], board_world[:, 1], board_world[:, 2], label="Board corners")
        camera = T_w_c[:3, 3]
        axis.scatter([camera[0]], [camera[1]], [camera[2]], marker="^", s=70, label="Scene camera")
        geometry_points = [tracker_world, board_world, camera.reshape(1, 3)]
        if head_query is not None and head_query_valid is not None and head_query_valid[index]:
            head_center = head_query[index]
            axis.scatter([head_center[0]], [head_center[1]], [head_center[2]], marker="o", s=70, label="Head-marker centre")
            axis.plot([head_center[0], camera[0]], [head_center[1], camera[1]], [head_center[2], camera[2]], linestyle="--", label="Head → camera")
            geometry_points.append(head_center.reshape(1, 3))
        length = 150.0
        for component, label in enumerate(("Camera X", "Camera Y", "Camera Z")):
            endpoint = camera + length * T_w_c[:3, component]
            axis.plot([camera[0], endpoint[0]], [camera[1], endpoint[1]], [camera[2], endpoint[2]], label=label)
        axis.set_xlabel("Vicon X (mm)")
        axis.set_ylabel("Vicon Y (mm)")
        axis.set_zlabel("Vicon Z (mm)")
        axis.set_title("Calibration geometry")
        axis.legend(fontsize=8)
        _set_equal_3d(axis, np.vstack(geometry_points))
        fig.tight_layout()
        path = output_directory / "calibration_geometry.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        qc_files.append(path)

    gaze_metrics: dict = {"available": False}
    if recording.gaze_timestamps_ns is not None and recording.gaze_xy_px is not None:
        _report_progress(progress, 0.67, "Interpolating calibration gaze poses")
        gaze_scene_time = (
            recording.gaze_timestamps_ns.astype(np.float64)
            - float(recording.scene_timestamps_ns[0])
        ) / 1e9
        gaze_query = gaze_scene_time + offset_sec
        tracker_gaze = tracker_pose.interpolate(
            gaze_query, max_gap_sec=request.max_pose_gap_sec
        )
        board_gaze = board_pose.interpolate(
            gaze_query, max_gap_sec=request.max_pose_gap_sec
        )
        camera_dirs = unproject_gaze_pixels(recording.gaze_xy_px, recording.intrinsics)
        valid = (
            tracker_gaze.valid
            & board_gaze.valid
            & np.isfinite(camera_dirs).all(axis=1)
        )
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices):
            _report_progress(
                progress, 0.75, f"Intersecting {len(valid_indices):,} calibration rays with the board"
            )
            R_w_t = tracker_gaze.rotations[valid_indices]
            t_w_t = tracker_gaze.translations[valid_indices]
            R_w_b = board_gaze.rotations[valid_indices]
            t_w_b = board_gaze.translations[valid_indices]
            R_t_c = T_tracker_from_camera[:3, :3]
            t_t_c = T_tracker_from_camera[:3, 3]
            origin_world = np.einsum("nij,j->ni", R_w_t, t_t_c) + t_w_t
            R_w_c = np.einsum("nij,jk->nik", R_w_t, R_t_c)
            direction_world = np.einsum(
                "nij,nj->ni", R_w_c, camera_dirs[valid_indices]
            )
            R_b_w = np.transpose(R_w_b, (0, 2, 1))
            origin_board = np.einsum(
                "nij,nj->ni", R_b_w, origin_world - t_w_b
            )
            direction_board = np.einsum(
                "nij,nj->ni", R_b_w, direction_world
            )
            finite_direction = np.abs(direction_board[:, 2]) >= 1e-9
            distance = np.full(len(valid_indices), np.nan, dtype=float)
            distance[finite_direction] = (
                -origin_board[finite_direction, 2]
                / direction_board[finite_direction, 2]
            )
            points = origin_board + distance[:, None] * direction_board
            finite_hit = np.isfinite(points).all(axis=1) & np.isfinite(distance)
            points = points[finite_hit]
            distance = distance[finite_hit]
            if len(points):
                hit_array = points[:, :2]
                in_front = distance > 0
                x_min, y_min = np.min(outer_local[:, :2], axis=0)
                x_max, y_max = np.max(outer_local[:, :2], axis=0)
                inside = (
                    in_front
                    & (hit_array[:, 0] >= x_min)
                    & (hit_array[:, 0] <= x_max)
                    & (hit_array[:, 1] >= y_min)
                    & (hit_array[:, 1] <= y_max)
                )
                gaze_metrics = {
                    "available": True,
                    "samples": int(len(hit_array)),
                    "fraction_intersects_in_front": float(np.mean(in_front)),
                    "fraction_inside_outer_rectangle": float(np.mean(inside)),
                    "hit_x_mm": _percentiles(hit_array[:, 0]),
                    "hit_y_mm": _percentiles(hit_array[:, 1]),
                }
                _report_progress(progress, 0.84, "Rendering calibration gaze-board intersections")
                fig, axis = plt.subplots(figsize=(7, 7))
                axis.scatter(hit_array[:, 0], hit_array[:, 1], s=4, alpha=0.4)
                polygon = np.vstack([outer_local[:, :2], outer_local[0, :2]])
                axis.plot(polygon[:, 0], polygon[:, 1], linewidth=2)
                axis.set_aspect("equal", adjustable="box")
                axis.invert_yaxis()
                axis.set_xlabel("Board local X (mm)")
                axis.set_ylabel("Board local Y (mm)")
                axis.set_title("Calibration gaze-ray intersections with board")
                axis.grid(True, alpha=0.3)
                fig.tight_layout()
                path = output_directory / "calibration_gaze_hits_board.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                qc_files.append(path)

    _report_progress(progress, 0.94, "Finalising calibration QC metrics")
    metrics = {
        "fixed_reprojection_rms_px": _percentiles(frame_rms),
        "direct_pnp_rms_px": _percentiles(
            np.asarray([d.direct_rms_px for d in detections])
        ),
        "camera_origin_tracker_mm": T_tracker_from_camera[:3, 3].tolist(),
        "tracker_marker_centroid_definition": (
            "The local tracker template is centred at the arithmetic centroid of the six "
            "selected glasses markers."
        ),
        "camera_distance_from_tracker_centroid_mm": float(
            np.linalg.norm(T_tracker_from_camera[:3, 3])
        ),
        "camera_to_individual_tracker_markers_mm": {
            label: float(
                np.linalg.norm(
                    tracker_template_mm[index] - T_tracker_from_camera[:3, 3]
                )
            )
            for index, label in enumerate(request.tracker_marker_labels)
        },
        "camera_distance_definitions": {
            "profile_transform_vector_norm_mm": float(
                np.linalg.norm(T_tracker_from_camera[:3, 3])
            ),
            "meaning": (
                "The profile value is the norm of the single robust 3-D camera-origin "
                "vector used to generate rays. Per-frame candidate distance summaries "
                "are reported separately and must not be substituted for the vector."
            ),
            "per_frame_candidate_distance_mm": (
                (estimator_metrics or {}).get(
                    "candidate_camera_distance_mm_inliers", {"n": 0}
                )
            ),
        },
        "robust_multiframe_estimator": estimator_metrics or {},
        "camera_optical_axis_to_board_center_deg": _percentiles(
            np.asarray(optical_angles)
        ),
        "camera_to_board_distance_mm": _percentiles(
            np.asarray(camera_board_distances)
        ),
        "camera_head_check": {
            "available": bool(camera_head_distances),
            "camera_to_head_marker_center_mm": _percentiles(np.asarray(camera_head_distances)),
            "camera_forward_projection_toward_board_mm": _percentiles(np.asarray(camera_forward_projections)),
            "fraction_camera_in_front_of_head_toward_board": float(np.mean(np.asarray(camera_forward_projections) > 0)) if camera_forward_projections else None,
        },
        "gaze_board_check": gaze_metrics,
    }
    _report_progress(progress, 1.0, "Calibration QC complete")
    return metrics, qc_files


def _set_equal_3d(axis, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float)
    minimum = np.nanmin(points, axis=0)
    maximum = np.nanmax(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(maximum - minimum) / 2.0
    if radius <= 0:
        radius = 1.0
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def run_workflow_b_calibration(
    request: CalibrationRequest,
    *,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> CalibrationResult:
    output = Path(request.output_directory)
    qc_dir = output / "qc"
    output.mkdir(parents=True, exist_ok=True)
    _report_progress(progress, 0.01, "Opening calibration recording and C3D files")
    _log(log, "Opening calibration recording and C3D files...")
    with NativeRecording.open(request.pupil_calibration_source) as recording:
        tracker = read_c3d(request.tracker_c3d_path)
        board = read_c3d(request.board_c3d_path)
        _report_progress(progress, 0.07, "Preparing tracker and AprilTag-board geometry")
        head_time_sec = None
        head_centers_mm = None
        head_valid = None
        if request.head_c3d_path and request.head_marker_labels:
            head_c3d = read_c3d(request.head_c3d_path)
            if abs(head_c3d.frame_rate_hz - tracker.frame_rate_hz) > 1e-6 or head_c3d.frame_count != tracker.frame_count:
                raise CalibrationError("Optional head-check C3D must have the same frame rate and frame count as the tracker C3D")
            head_positions, head_marker_valid = head_c3d.select(request.head_marker_labels)
            counts = head_marker_valid.sum(axis=1)
            head_centers_mm = np.nanmean(np.where(head_marker_valid[..., None], head_positions, np.nan), axis=1)
            head_valid = (counts >= 3) & np.isfinite(head_centers_mm).all(axis=1)
            head_time_sec = head_c3d.time_sec
        if abs(tracker.frame_rate_hz - board.frame_rate_hz) > 1e-6:
            raise CalibrationError("Tracker and board C3D frame rates differ")
        if tracker.frame_count != board.frame_count:
            raise CalibrationError("Tracker and board C3D frame counts differ")
        _report_progress(progress, 0.11, "Reconstructing calibration rigid bodies")
        tracker_template, tracker_positions, tracker_valid, tracker_pose = _prepare_tracker(
            tracker, request.tracker_marker_labels
        )
        (
            board_labels,
            board_template,
            object_points_by_tag,
            outer_local,
            board_pose,
        ) = _prepare_board(
            board,
            request.board_corner_labels,
            request.outer_corner_labels,
            request.apriltag_ids,
        )
        _log(log, "Detecting AprilTags in the Neon calibration scene video...")
        detections, all_four_count = _detect_apriltags(
            recording,
            apriltag_ids=request.apriltag_ids,
            object_points_by_tag=object_points_by_tag,
            sample_every_n_frames=request.sample_every_n_frames,
            max_detection_frames=request.max_detection_frames,
            progress=lambda value, text: _report_progress(
                progress, 0.15 + 0.38 * value, text
            ),
        )
        if len(detections) < 20:
            raise CalibrationError(
                f"Only {len(detections)} usable AprilTag frames were detected; at least 20 are required"
            )
        _log(log, f"Detected all four AprilTags in {len(detections)} sampled frames.")
        _report_progress(progress, 0.55, "Estimating calibration video-to-C3D time offset")
        _log(log, "Estimating the calibration recording's Neon-to-C3D time offset...")
        initial_offset, fine_offsets, fine_cost = _search_offset(
            detections,
            tracker_pose,
            board_pose,
            recording.intrinsics.matrix,
            recording.intrinsics.distortion,
            request.max_pose_gap_sec,
        )
        initial_query = np.asarray([d.scene_time_sec for d in detections]) + initial_offset
        initial_tracker = tracker_pose.interpolate(
            initial_query, max_gap_sec=request.max_pose_gap_sec
        )
        initial_board = board_pose.interpolate(
            initial_query, max_gap_sec=request.max_pose_gap_sec
        )
        fit_mask = initial_tracker.valid & initial_board.valid
        fit_detections = [
            detection for detection, keep in zip(detections, fit_mask) if keep
        ]
        if len(fit_detections) < 10:
            raise CalibrationError(
                "Fewer than ten AprilTag frames overlap the calibration C3D"
            )
        _report_progress(progress, 0.64, "Estimating per-frame camera-to-tracker transforms")
        candidates = _transform_candidates(
            fit_detections,
            initial_offset,
            tracker_pose,
            board_pose,
            request.max_pose_gap_sec,
        )
        _report_progress(progress, 0.70, "Robustly aggregating the calibrated 6-DoF transform")
        final_transform, candidate_inlier_mask, estimator_metrics = robust_average_transforms(
            candidates,
            maximum_translation_gate_mm=50.0,
            maximum_rotation_gate_deg=10.0,
        )
        final_offset = float(initial_offset)
        point_residual, fixed_frame_rms = _fixed_reprojection_residuals(
            fit_detections,
            final_offset,
            tracker_pose,
            board_pose,
            final_transform,
            recording.intrinsics.matrix,
            recording.intrinsics.distortion,
            request.max_pose_gap_sec,
        )
        point_error = np.linalg.norm(point_residual, axis=1)
        fixed_rms = (
            float(np.sqrt(np.mean(point_error**2)))
            if len(point_error)
            else float("nan")
        )
        _log(log, f"Calibration-only scene-to-C3D offset: {final_offset:.6f} s")
        _log(
            log,
            f"Robust multiframe transform: {int(candidate_inlier_mask.sum())}/{len(candidates)} inliers; "
            f"camera-to-six-marker-centroid vector norm "
            f"{np.linalg.norm(final_transform[:3, 3]):.3f} mm.",
        )
        candidate_distance_stats = estimator_metrics[
            "candidate_camera_distance_mm_inliers"
        ]
        _log(
            log,
            "Per-frame camera-distance candidates: "
            f"median {candidate_distance_stats.get('median', float('nan')):.3f} mm, "
            f"mean {candidate_distance_stats.get('mean', float('nan')):.3f} mm.",
        )
        _log(log, f"Fixed-transform point RMS: {fixed_rms:.3f} px")

        _report_progress(progress, 0.77, "Generating calibration QC visualizations")
        qc_metrics, qc_files = _calibration_qc(
            request=request,
            recording=recording,
            detections=detections,
            tracker_pose=tracker_pose,
            tracker_template_mm=tracker_template,
            board_pose=board_pose,
            T_tracker_from_camera=final_transform,
            offset_sec=final_offset,
            outer_local=outer_local,
            fine_offsets=fine_offsets,
            fine_cost=fine_cost,
            output_directory=qc_dir,
            candidate_transforms=candidates,
            candidate_scene_times_sec=np.asarray(
                [d.scene_time_sec for d in fit_detections], dtype=float
            ),
            candidate_inlier_mask=candidate_inlier_mask,
            estimator_metrics=estimator_metrics,
            head_time_sec=head_time_sec,
            head_centers_mm=head_centers_mm,
            head_valid=head_valid,
            progress=lambda value, text: _report_progress(
                progress, 0.77 + 0.13 * value, text
            ),
        )
        identity_audit = (
            request.tracker_identity_audit
            if request.include_raw_identity_metadata
            else _privacy_safe_identity_audit(request.tracker_identity_audit)
        )
        identity_plot = _plot_tracker_identity_audit(identity_audit, qc_dir)
        if identity_plot is not None:
            qc_files.insert(0, identity_plot)
        warnings: list[str] = []
        camera_board_median = qc_metrics.get("camera_to_board_distance_mm", {}).get("median")
        if camera_board_median and camera_board_median > 1000:
            warnings.append(
                "Calibration board was more than 1 m from the scene camera; validate fine-grained AOI accuracy independently."
            )
        if fixed_rms > 4.0:
            warnings.append("Fixed-transform reprojection RMS exceeds 4 px.")
        distance_stats = estimator_metrics.get(
            "candidate_camera_distance_mm_inliers", {}
        )
        distance_span = (
            float(distance_stats.get("p95", 0.0))
            - float(distance_stats.get("p05", 0.0))
            if distance_stats.get("n", 0)
            else 0.0
        )
        if camera_board_median and camera_board_median > 1000 and distance_span > 10.0:
            warnings.append(
                "WEAK_TRANSLATION_CONSTRAINT: the board is distant and per-frame camera translations span "
                f"{distance_span:.1f} mm between p05 and p95. The robust transform is usable for pilot rays, "
                "but fine-grained facial AOI accuracy requires an independent known-target validation."
            )
        head_check = qc_metrics.get("camera_head_check", {})
        if head_check.get("available") and head_check.get("fraction_camera_in_front_of_head_toward_board", 0.0) < 0.95:
            warnings.append("The optional head-marker check did not consistently place the scene camera in front of the head toward the calibration board.")
        gaze_check = qc_metrics.get("gaze_board_check", {})
        if gaze_check.get("available") and gaze_check.get(
            "fraction_inside_outer_rectangle", 0.0
        ) < 0.8:
            warnings.append("Fewer than 80% of calibration gaze rays hit the board rectangle.")

        if request.include_raw_identity_metadata:
            source_files = {
                "pupil_calibration": str(request.pupil_calibration_source),
                "tracker_c3d": str(request.tracker_c3d_path),
                "board_c3d": str(request.board_c3d_path),
                "head_c3d": str(request.head_c3d_path) if request.head_c3d_path else None,
                "head_marker_labels": request.head_marker_labels,
            }
        else:
            source_files = {
                "pupil_calibration": "<redacted; see source_sha256>",
                "tracker_c3d": "<redacted; see source_sha256>",
                "board_c3d": "<redacted; see source_sha256>",
                "head_c3d": (
                    "<redacted; see source_sha256>" if request.head_c3d_path else None
                ),
                "head_marker_labels": None,
            }

        metadata = {
            "workflow": "B_precise_person_specific",
            "calibration_estimator_id": "robust_multiframe_se3_geometric_median_v2",
            "calibration_estimator_policy": (
                "Direct per-frame PnP/Vicon transforms are robustly aggregated. "
                "Camera translation is not jointly optimized against reprojection error."
            ),
            "calibration_scene_to_c3d_offset_sec": final_offset,
            "calibration_offset_scope": "calibration_recording_only_do_not_reuse_for_trials",
            "board_corner_labels": request.board_corner_labels,
            "outer_corner_labels": request.outer_corner_labels,
            "board_marker_labels_used": board_labels,
            "privacy": {
                "raw_identity_metadata_included": bool(
                    request.include_raw_identity_metadata
                )
            },
            "source_files": source_files,
            "source_sha256": {
                "pupil_calibration": sha256_file(request.pupil_calibration_source)
                if request.pupil_calibration_source.is_file()
                else None,
                "tracker_c3d": sha256_file(request.tracker_c3d_path),
                "board_c3d": sha256_file(request.board_c3d_path),
                "head_c3d": sha256_file(request.head_c3d_path) if request.head_c3d_path and request.head_c3d_path.is_file() else None,
            },
            "tracker_identity_detection": identity_audit,
        }
        metrics = {
            "apriltag": {
                "sampled_usable_frames": len(detections),
                "sampled_frames_with_all_tags": all_four_count,
                "downscaled_detection_frames": int(
                    sum(d.detection_mode == "downscaled" for d in detections)
                ),
                "full_resolution_fallback_frames": int(
                    sum(
                        d.detection_mode == "full_resolution_fallback"
                        for d in detections
                    )
                ),
                "direct_pnp_rms_px": _percentiles(
                    np.asarray([d.direct_rms_px for d in detections])
                ),
            },
            "fixed_fit": {
                "estimator": estimator_metrics.get("estimator"),
                "unconstrained_joint_transform_optimization_used": False,
                "point_rms_px": fixed_rms,
                "point_error_px": _percentiles(point_error),
                "per_frame_rms_px": _percentiles(fixed_frame_rms),
                "calibration_scene_to_c3d_offset_sec": final_offset,
                "candidate_transform_statistics": estimator_metrics,
            },
            "tracker": {
                "valid_pose_fraction": float(np.mean(tracker_pose.valid)),
                "fit_rmse_mm": _percentiles(tracker_pose.fit_rmse_mm),
            },
            "board": {
                "valid_pose_fraction": float(np.mean(board_pose.valid)),
                "fit_rmse_mm": _percentiles(board_pose.fit_rmse_mm),
                "outer_width_mm": float(
                    (np.linalg.norm(outer_local[1] - outer_local[0])
                     + np.linalg.norm(outer_local[2] - outer_local[3]))
                    / 2.0
                ),
                "outer_height_mm": float(
                    (np.linalg.norm(outer_local[3] - outer_local[0])
                     + np.linalg.norm(outer_local[2] - outer_local[1]))
                    / 2.0
                ),
            },
            "quality_control": qc_metrics,
        }
        _report_progress(progress, 0.91, "Building calibrated profile and provenance metadata")
        profile = CalibrationProfile.create(
            display_name=request.display_name,
            canonical_participant_id=request.canonical_participant_id,
            module_serial_number=str(
                recording.info.get("module_serial_number", recording.intrinsics.serial_number)
            ),
            frame_name=str(recording.info.get("frame_name", "")),
            source_recording_id=str(recording.info.get("recording_id", "")),
            tracker_marker_labels=request.tracker_marker_labels,
            tracker_template_mm=tracker_template,
            T_tracker_from_camera=final_transform,
            scene_camera_matrix=recording.intrinsics.matrix,
            scene_distortion_coefficients=recording.intrinsics.distortion,
            metadata=metadata,
            metrics=metrics,
            warnings=warnings,
        )
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in request.display_name
        ).strip("_") or "calibration"
        _report_progress(progress, 0.96, "Writing calibration profile and report")
        profile_path = profile.save(output / f"{safe_name}.g3dcal", qc_files)
        metrics_path = output / "calibration_metrics.json"
        metrics_path.write_text(
            json.dumps(profile.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if identity_audit is not None:
            (output / "tracker_identity_detection.json").write_text(
                json.dumps(identity_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        report_path = output / "calibration_report.html"
        report_path.write_text(
            _render_calibration_report(profile, qc_files), encoding="utf-8"
        )
        _log(log, f"Calibration profile written: {profile_path}")
        _report_progress(progress, 1.0, "Calibration profile complete")
        return CalibrationResult(
            profile=profile,
            profile_path=profile_path,
            qc_files=qc_files,
            report_path=report_path,
            metrics_path=metrics_path,
        )


def _render_calibration_report(profile: CalibrationProfile, qc_files: list[Path]) -> str:
    warnings = "".join(f"<li>{item}</li>" for item in profile.warnings) or "<li>None</li>"
    identity = profile.metadata.get("tracker_identity_detection") or {}
    if identity:
        if identity.get("privacy_redacted"):
            identity_summary = (
                "<p><strong>Raw wearer/subject identity:</strong> redacted</p>"
                f"<p><strong>Selected tracker candidate:</strong> {identity.get('selected_candidate_index', 'not recorded')}</p>"
                f"<p><strong>Tracker identity confidence:</strong> {identity.get('confidence', 'not recorded')}</p>"
            )
        else:
            identity_summary = (
                f"<p><strong>Pupil wearer:</strong> {identity.get('wearer_name_raw', 'not recorded')}</p>"
                f"<p><strong>Matched C3D subject:</strong> {identity.get('matched_c3d_subject', 'not recorded')}</p>"
                f"<p><strong>Automatically selected tracker:</strong> {identity.get('selected_tracker_filename', 'not recorded')}</p>"
                f"<p><strong>Tracker identity confidence:</strong> {identity.get('confidence', 'not recorded')}</p>"
            )
    else:
        identity_summary = "<p><strong>Tracker identity audit:</strong> not embedded in this profile.</p>"
    images = "".join(
        f'<section><h3>{path.stem.replace("_", " ").title()}</h3><img src="qc/{path.name}" alt="{path.name}"></section>'
        for path in qc_files
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Calibration QC</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:30px auto;line-height:1.45}}img{{max-width:100%;border:1px solid #ccc}}code,pre{{background:#f4f4f4;padding:4px}}.pass{{color:#176b2c}}.warn{{color:#9a5a00}}</style></head>
<body><h1>Neon–Vicon Workflow B Calibration QC</h1>
<p><strong>Profile:</strong> {profile.display_name}</p>
<p><strong>Participant:</strong> {profile.canonical_participant_id}</p>
<p><strong>Module serial:</strong> {profile.module_serial_number}</p>
{identity_summary}
<p><strong>Fixed-transform RMS:</strong> {profile.metrics['fixed_fit']['point_rms_px']:.3f} px</p>
<p><strong>Robust profile camera-to-six-marker centroid distance:</strong> {profile.metrics['quality_control']['camera_distance_definitions']['profile_transform_vector_norm_mm']:.3f} mm</p>
<p><strong>Per-frame candidate median distance:</strong> {profile.metrics['fixed_fit']['candidate_transform_statistics']['candidate_camera_distance_mm_inliers'].get('median', float('nan')):.3f} mm</p>
<h2>Warnings</h2><ul>{warnings}</ul>
<h2>Metrics</h2><pre>{json.dumps(profile.metrics, ensure_ascii=False, indent=2)}</pre>
<h2>Visual checks</h2>{images}</body></html>"""
