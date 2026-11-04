from __future__ import annotations

"""Automatic association of an anonymous eye-tracker C3D with a Pupil wearer.

Two audited routes are supported:

1. **Labelled-head route** — discover C3D subjects from ARIEL/LFHD/LBHD/RFHD/
   RBHD, match the Pupil wearer name to a subject, then select the tracker
   cluster physically attached to that head.
2. **AprilTag camera route** — when body markers are exported only as anonymous
   ``*`` trajectories, reconstruct the Pupil scene-camera optical centre from
   the calibration board and select the tracker cluster lying at that camera.
   A 16-marker planar 4×4 board mapping can be inferred automatically.

No file-name assumption (``1`` or ``2``) is used.  All evidence and distances
are serialisable for profile provenance and audit.
"""

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..io.archive import sha256_file
from ..io.c3d_reader import C3DData, read_c3d
from ..io.pupil_recording import NativeRecording


STANDARD_HEAD_SUFFIXES = ("ARIEL", "LFHD", "LBHD", "RFHD", "RBHD")


class TrackerIdentityError(RuntimeError):
    pass


def _report_progress(
    callback: Callable[[float, str], None] | None,
    value: float,
    text: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), text)


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def name_similarity(left: str, right: str) -> float:
    """Return a conservative name/alias similarity in [0, 1]."""

    a = _normalise_name(left)
    b = _normalise_name(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    base = SequenceMatcher(None, a, b).ratio()
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 3 and longer.startswith(shorter):
        return max(base, 0.92)
    if len(shorter) >= 3 and shorter in longer:
        return max(base, 0.88)
    if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
        return max(base, 0.86)
    return float(base)


def _marker_sort_key(label: str) -> tuple[str, int, str]:
    match = re.search(r"(?:^|[:_\- ])M[_\- ]?(\d+)$", label, flags=re.IGNORECASE)
    number = int(match.group(1)) if match else 10**9
    prefix = label[: match.start()] if match else label
    return prefix.casefold(), number, label.casefold()


def infer_tracker_marker_labels(c3d: C3DData) -> list[str]:
    """Infer the marker group representing one eye-tracker rigid cluster."""

    if not c3d.labels:
        raise TrackerIdentityError(f"No point labels in {c3d.path}")

    groups: dict[str, list[str]] = {}
    for label in c3d.labels:
        if ":" in label:
            prefix = label.rsplit(":", 1)[0]
        else:
            marker_match = re.match(r"(.+?)[_\- ]?M[_\- ]?\d+$", label, re.IGNORECASE)
            prefix = marker_match.group(1) if marker_match else "__all__"
        groups.setdefault(prefix, []).append(label)

    scored: list[tuple[float, str, list[str]]] = []
    for prefix, labels in groups.items():
        marker_like = [
            label
            for label in labels
            if re.search(r"(?:^|[:_\- ])M[_\- ]?\d+$", label, re.IGNORECASE)
        ]
        selected = marker_like if len(marker_like) >= 3 else labels
        score = 0.0
        norm_prefix = _normalise_name(prefix)
        if "eyetracker" in norm_prefix or norm_prefix.startswith("tracker"):
            score += 100.0
        if 3 <= len(selected) <= 12:
            score += 30.0
        if len(selected) == 6:
            score += 20.0
        score += min(len(selected), 20) / 100.0
        scored.append((score, prefix, sorted(selected, key=_marker_sort_key)))

    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    best_score, _, labels = scored[0]
    if best_score < 30.0 and 3 <= len(c3d.labels) <= 12:
        labels = sorted(c3d.labels, key=_marker_sort_key)
    if len(labels) < 3:
        raise TrackerIdentityError(
            f"Could not infer at least three tracker markers from {c3d.path.name}"
        )
    return labels


def discover_head_subjects(
    world: C3DData,
    *,
    suffixes: tuple[str, ...] = STANDARD_HEAD_SUFFIXES,
    minimum_markers: int = 3,
) -> dict[str, list[str]]:
    suffix_set = {suffix.upper() for suffix in suffixes}
    grouped: dict[str, dict[str, str]] = {}
    for label in world.labels:
        if ":" not in label:
            continue
        subject, suffix = label.split(":", 1)
        suffix_upper = suffix.upper()
        if suffix_upper in suffix_set:
            grouped.setdefault(subject, {})[suffix_upper] = label

    result: dict[str, list[str]] = {}
    for subject, mapping in grouped.items():
        labels = [mapping[suffix] for suffix in suffixes if suffix in mapping]
        if len(labels) >= minimum_markers:
            result[subject] = labels
    return result


def _aligned_slices(left: C3DData, right: C3DData) -> tuple[slice, slice, int]:
    left_start = int(left.first_frame)
    left_stop = left_start + left.frame_count
    right_start = int(right.first_frame)
    right_stop = right_start + right.frame_count
    start = max(left_start, right_start)
    stop = min(left_stop, right_stop)
    if stop <= start:
        raise TrackerIdentityError(
            f"No overlapping C3D frames between {left.path.name} and {right.path.name}"
        )
    return (
        slice(start - left_start, stop - left_start),
        slice(start - right_start, stop - right_start),
        stop - start,
    )


def _centroid(
    points: np.ndarray,
    valid: np.ndarray,
    minimum_markers: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a frame-wise marker centroid without empty-slice warnings."""

    counts = valid.sum(axis=1).astype(float)
    summed = np.where(valid[..., None], points, 0.0).sum(axis=1)
    center = np.full((points.shape[0], 3), np.nan, dtype=float)
    nonempty = counts > 0
    center[nonempty] = summed[nonempty] / counts[nonempty, None]
    center_valid = (
        (counts >= float(minimum_markers))
        & np.isfinite(center).all(axis=1)
    )
    return center, center_valid


def _distance_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
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


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise TrackerIdentityError("Could not construct a stable board coordinate axis")
    return vector / norm


def _median_marker_positions(c3d: C3DData) -> dict[str, np.ndarray]:
    values = np.where(c3d.valid[..., None], c3d.points_mm, np.nan)
    median = np.nanmedian(values, axis=0)
    result: dict[str, np.ndarray] = {}
    for label, point in zip(c3d.labels, median):
        if np.isfinite(point).all():
            result[label] = np.asarray(point, dtype=float)
    return result


def _mapping_from_grid(grid: np.ndarray) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    grid = np.asarray(grid, dtype=object)
    tags = {
        "0": {"TL": str(grid[0, 0]), "TR": str(grid[0, 1]), "BR": str(grid[1, 1]), "BL": str(grid[1, 0])},
        "1": {"TL": str(grid[0, 2]), "TR": str(grid[0, 3]), "BR": str(grid[1, 3]), "BL": str(grid[1, 2])},
        "2": {"TL": str(grid[2, 0]), "TR": str(grid[2, 1]), "BR": str(grid[3, 1]), "BL": str(grid[3, 0])},
        "3": {"TL": str(grid[2, 2]), "TR": str(grid[2, 3]), "BR": str(grid[3, 3]), "BL": str(grid[3, 2])},
    }
    outer = {
        "TL": str(grid[0, 0]),
        "TR": str(grid[0, 3]),
        "BR": str(grid[3, 3]),
        "BL": str(grid[3, 0]),
    }
    return tags, outer


def infer_planar_4x4_board_mapping_candidates(
    board: C3DData,
) -> list[tuple[str, dict[str, dict[str, str]], dict[str, str]]]:
    """Infer all eight orientation hypotheses for a rigid 4×4 marker grid.

    The grouping is geometry-only and therefore does not assume that Vicon X/Y/Z
    corresponds to image left/right/up/down.  The eight rotations/reflections are
    scored later using the Pupil AprilTag image and tracker proximity.
    """

    if board.point_count != 16:
        raise TrackerIdentityError(
            f"Automatic board-grid inference requires exactly 16 markers; {board.path.name} has {board.point_count}."
        )
    positions_by_label = _median_marker_positions(board)
    if len(positions_by_label) != 16:
        raise TrackerIdentityError(
            "All 16 calibration-board markers must have at least one valid sample for automatic grid inference."
        )
    labels = list(board.labels)
    positions = np.asarray([positions_by_label[label] for label in labels], dtype=float)
    centered = positions - positions.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 3 or singular_values[1] <= 1e-6:
        raise TrackerIdentityError("The calibration-board markers do not span a stable plane")
    if singular_values[2] / singular_values[1] > 0.08:
        raise TrackerIdentityError(
            "The 16 calibration-board markers are not sufficiently planar for automatic 4×4 inference."
        )
    coordinates = centered @ axes[:2].T
    row_order = np.argsort(coordinates[:, 0])
    grid = np.empty((4, 4), dtype=object)
    row_centres: list[float] = []
    within_row_spread: list[float] = []
    for row_index in range(4):
        indices = row_order[row_index * 4 : (row_index + 1) * 4]
        row_centres.append(float(np.mean(coordinates[indices, 0])))
        within_row_spread.append(float(np.ptp(coordinates[indices, 0])))
        indices = indices[np.argsort(coordinates[indices, 1])]
        grid[row_index, :] = [labels[int(index)] for index in indices]
    row_gap = float(np.min(np.diff(np.sort(row_centres))))
    if row_gap <= max(5.0, 2.0 * max(within_row_spread)):
        raise TrackerIdentityError(
            "Could not separate the 16 calibration-board markers into four stable grid rows."
        )

    variants: list[tuple[str, dict[str, dict[str, str]], dict[str, str]]] = []
    seen: set[tuple[str, ...]] = set()
    for rotation_index in range(4):
        rotated = np.rot90(grid, rotation_index)
        for reflected, candidate_grid in (
            (False, rotated),
            (True, np.fliplr(rotated)),
        ):
            key = tuple(str(value) for value in candidate_grid.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            name = f"auto_grid_rot{rotation_index * 90}" + ("_mirror" if reflected else "")
            tags, outer = _mapping_from_grid(candidate_grid)
            variants.append((name, tags, outer))
    return variants


def _mapping_labels_exist(
    board: C3DData,
    tags: dict[str, dict[str, str]] | None,
    outer: dict[str, str] | None,
) -> bool:
    if not tags or not outer:
        return False
    try:
        labels = [tags[str(marker_id)][corner] for marker_id in (0, 1, 2, 3) for corner in ("TL", "TR", "BR", "BL")]
        labels.extend(outer[corner] for corner in ("TL", "TR", "BR", "BL"))
    except Exception:
        return False
    return all(str(label) in board.labels for label in labels)


def _detect_raw_apriltag_frames(
    recording: NativeRecording,
    *,
    sample_every_n_frames: int = 4,
    max_frames: int = 160,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    capture = cv2.VideoCapture(str(recording.scene_video_path))
    total = min(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(recording.scene_timestamps_ns),
    )
    detected: list[dict] = []

    def detect(gray: np.ndarray, scale: float) -> dict[int, np.ndarray]:
        image = (
            cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else gray
        )
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            return {}
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.005,
        )
        found: dict[int, np.ndarray] = {}
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
        while frame_index < total:
            if frame_index % 30 == 0:
                _report_progress(
                    progress,
                    frame_index / max(1, total),
                    f"Detecting AprilTags in calibration video ({frame_index}/{total} frames)",
                )
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % max(1, int(sample_every_n_frames)) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scale = min(1.0, 800.0 / float(gray.shape[1]))
                found = detect(gray, scale)
                if not all(marker_id in found for marker_id in (0, 1, 2, 3)):
                    found = detect(gray, 1.0)
                if all(marker_id in found for marker_id in (0, 1, 2, 3)):
                    detected.append(
                        {
                            "frame_index": int(frame_index),
                            "timestamp_ns": int(recording.scene_timestamps_ns[frame_index]),
                            "scene_time_sec": float(
                                (
                                    int(recording.scene_timestamps_ns[frame_index])
                                    - int(recording.scene_timestamps_ns[0])
                                )
                                / 1e9
                            ),
                            "corners": {marker_id: found[marker_id] for marker_id in (0, 1, 2, 3)},
                        }
                    )
            frame_index += 1
    finally:
        capture.release()
    if len(detected) > max_frames:
        indices = np.linspace(0, len(detected) - 1, max_frames).round().astype(int)
        detected = [detected[int(index)] for index in indices]
    _report_progress(progress, 1.0, f"Detected {len(detected)} usable AprilTag frames")
    return detected


def _solve_camera_positions_for_mapping(
    *,
    board: C3DData,
    tags: dict[str, dict[str, str]],
    outer: dict[str, str],
    raw_frames: list[dict],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = _median_marker_positions(board)
    required = [tags[str(marker_id)][corner] for marker_id in (0, 1, 2, 3) for corner in ("TL", "TR", "BR", "BL")]
    required.extend(outer[corner] for corner in ("TL", "TR", "BR", "BL"))
    missing = [label for label in required if label not in positions]
    if missing:
        raise TrackerIdentityError(f"Board mapping refers to unavailable markers: {sorted(set(missing))}")

    origin = positions[outer["TL"]]
    x_axis = _normalise_vector(positions[outer["TR"]] - origin)
    y_seed = positions[outer["BL"]] - origin
    y_axis = _normalise_vector(y_seed - np.dot(y_seed, x_axis) * x_axis)
    z_axis = _normalise_vector(np.cross(x_axis, y_axis))
    y_axis = _normalise_vector(np.cross(z_axis, x_axis))
    rotation_world_from_board = np.column_stack([x_axis, y_axis, z_axis])
    T_world_from_board = _make_transform(rotation_world_from_board, origin)

    local: dict[str, np.ndarray] = {
        label: rotation_world_from_board.T @ (point - origin)
        for label, point in positions.items()
    }
    object_points = np.asarray(
        [
            local[tags[str(marker_id)][corner]]
            for marker_id in (0, 1, 2, 3)
            for corner in ("TL", "TR", "BR", "BL")
        ],
        dtype=float,
    )

    permutations: list[tuple[int, int, int, int]] = []
    base = np.arange(4)
    for reverse in (False, True):
        order = base[::-1] if reverse else base
        for shift in range(4):
            permutation = tuple(int(value) for value in np.roll(order, -shift))
            if permutation not in permutations:
                permutations.append(permutation)

    camera_positions: list[np.ndarray] = []
    pnp_rms: list[float] = []
    for frame in raw_frames:
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for permutation in permutations:
            image_points = np.asarray(
                [
                    point
                    for marker_id in (0, 1, 2, 3)
                    for point in frame["corners"][marker_id][list(permutation)]
                ],
                dtype=float,
            )
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or float(tvec.reshape(3)[2]) <= 0:
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )
            error = projected.reshape(-1, 2) - image_points
            rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
            if best is None or rms < best[0]:
                best = (rms, rvec.reshape(3), tvec.reshape(3))
        if best is None:
            continue
        rms, rvec, tvec = best
        rotation_camera_from_board, _ = cv2.Rodrigues(rvec)
        T_camera_from_board = _make_transform(rotation_camera_from_board, tvec)
        T_world_from_camera = T_world_from_board @ _invert_transform(T_camera_from_board)
        camera_positions.append(T_world_from_camera[:3, 3])
        pnp_rms.append(rms)
    return np.asarray(camera_positions, dtype=float), np.asarray(pnp_rms, dtype=float)


def _tracker_centroid_trajectory(
    tracker: C3DData,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    points, valid = tracker.select(labels)
    counts = valid.sum(axis=1)
    summed = np.where(valid[..., None], points, 0.0).sum(axis=1)
    centers = np.full((tracker.frame_count, 3), np.nan, dtype=float)
    mask = counts >= 3
    centers[mask] = summed[mask] / counts[mask, None]
    mask &= np.isfinite(centers).all(axis=1)
    return centers, mask


def _detect_matching_tracker_by_board_camera_proximity(
    *,
    pupil_native_source: str | Path,
    tracker_candidate_paths: list[Path],
    board_c3d_path: str | Path,
    canonical_participant_id: str,
    wearer_name: str,
    wearer_id: str,
    recording_id: str,
    module_serial: str,
    board_corner_labels: dict[str, dict[str, str]] | None,
    outer_corner_labels: dict[str, str] | None,
    sample_every_n_frames: int,
    max_board_mapping_frames: int,
    log: Callable[[str], None] | None,
    progress: Callable[[float, str], None] | None = None,
) -> "TrackerIdentityResult":
    _report_progress(progress, 0.15, "Reading AprilTag board and tracker C3D candidates")
    board = read_c3d(board_c3d_path)
    mapping_candidates: list[
        tuple[str, dict[str, dict[str, str]], dict[str, str]]
    ] = []
    if _mapping_labels_exist(board, board_corner_labels, outer_corner_labels):
        mapping_candidates.append(
            (
                "user_supplied_mapping",
                {str(key): dict(value) for key, value in board_corner_labels.items()},
                dict(outer_corner_labels),
            )
        )
    inferred = infer_planar_4x4_board_mapping_candidates(board)
    existing_keys = {
        tuple(tags[str(marker_id)][corner] for marker_id in (0, 1, 2, 3) for corner in ("TL", "TR", "BR", "BL"))
        for _, tags, _ in mapping_candidates
    }
    for name, tags, outer in inferred:
        key = tuple(tags[str(marker_id)][corner] for marker_id in (0, 1, 2, 3) for corner in ("TL", "TR", "BR", "BL"))
        if key not in existing_keys:
            mapping_candidates.append((name, tags, outer))
            existing_keys.add(key)

    candidate_rows: list[TrackerCandidateIdentity] = []
    tracker_centers: list[np.ndarray] = []
    for candidate_index, path in enumerate(tracker_candidate_paths, start=1):
        _report_progress(
            progress,
            0.18 + 0.08 * candidate_index / max(1, len(tracker_candidate_paths)),
            f"Reading tracker candidate {candidate_index}/{len(tracker_candidate_paths)}",
        )
        tracker = read_c3d(path)
        labels = infer_tracker_marker_labels(tracker)
        centers, center_valid = _tracker_centroid_trajectory(tracker, labels)
        valid_centers = centers[center_valid]
        if len(valid_centers) < 10:
            raise TrackerIdentityError(
                f"Tracker candidate {path.name} has fewer than ten valid centroid frames."
            )
        tracker_centers.append(valid_centers)
        prefix = labels[0].rsplit(":", 1)[0] if ":" in labels[0] else path.stem
        candidate_rows.append(
            TrackerCandidateIdentity(
                candidate_index=candidate_index,
                path=str(path),
                filename=path.name,
                sha256=sha256_file(path) if path.is_file() else None,
                marker_labels=labels,
                marker_prefix=prefix,
                frame_count=tracker.frame_count,
                frame_rate_hz=tracker.frame_rate_hz,
                first_frame=tracker.first_frame,
                valid_centroid_frames=int(center_valid.sum()),
            )
        )

    with NativeRecording.open(pupil_native_source) as recording:
        raw_frames = _detect_raw_apriltag_frames(
            recording,
            sample_every_n_frames=max(1, int(sample_every_n_frames)),
            max_frames=max(10, int(max_board_mapping_frames)),
            progress=lambda value, text: _report_progress(
                progress, 0.28 + 0.40 * value, text
            ),
        )
        if len(raw_frames) < 10:
            raise TrackerIdentityError(
                f"Only {len(raw_frames)} sampled frames contained all four AprilTags; at least ten are required for board-based tracker identity."
            )
        if log:
            log(
                f"No labelled head subjects were found. Using {len(raw_frames)} AprilTag frames to locate the Pupil scene camera directly in Vicon space."
            )
        best_by_candidate: list[dict | None] = [None] * len(candidate_rows)
        for mapping_index, (mapping_name, tags, outer) in enumerate(mapping_candidates, start=1):
            _report_progress(
                progress,
                0.70 + 0.22 * mapping_index / max(1, len(mapping_candidates)),
                f"Testing board orientation {mapping_index}/{len(mapping_candidates)}",
            )
            try:
                camera_positions, pnp_rms = _solve_camera_positions_for_mapping(
                    board=board,
                    tags=tags,
                    outer=outer,
                    raw_frames=raw_frames,
                    camera_matrix=recording.intrinsics.matrix,
                    distortion=recording.intrinsics.distortion,
                )
            except Exception as exc:
                if log:
                    log(f"Board mapping {mapping_name} was skipped: {exc}")
                continue
            if len(camera_positions) < 10:
                continue
            pnp_summary = _distance_summary(pnp_rms)
            pnp_median = float(pnp_summary.get("median", float("inf")))
            for candidate_index, centers in enumerate(tracker_centers):
                tree = cKDTree(centers)
                nearest_distance, _ = tree.query(camera_positions, k=1)
                summary = _distance_summary(nearest_distance)
                if not summary.get("n", 0):
                    continue
                median = float(summary["median"])
                p25 = float(np.percentile(nearest_distance, 25))
                fraction_under_250 = float(np.mean(nearest_distance < 250.0))
                # Distance dominates. PnP RMS only breaks board-orientation ties.
                score = p25 + 0.25 * median + 4.0 * pnp_median
                if fraction_under_250 < 0.25:
                    score += 500.0 * (0.25 - fraction_under_250) / 0.25
                evidence = {
                    "score": float(score),
                    "mapping_name": mapping_name,
                    "tags": tags,
                    "outer": outer,
                    "distance_summary": summary,
                    "fraction_under_250": fraction_under_250,
                    "pnp_summary": pnp_summary,
                    "camera_position_count": int(len(camera_positions)),
                }
                current = best_by_candidate[candidate_index]
                if current is None or float(evidence["score"]) < float(current["score"]):
                    best_by_candidate[candidate_index] = evidence

    if not any(item is not None for item in best_by_candidate):
        raise TrackerIdentityError(
            "AprilTags were detected, but no board orientation produced a usable camera position for either tracker candidate."
        )

    for candidate, evidence in zip(candidate_rows, best_by_candidate):
        if evidence is None:
            continue
        candidate.camera_distance_mm = dict(evidence["distance_summary"])
        candidate.camera_proximity_score_mm = float(evidence["score"])
        candidate.camera_frames_under_250mm_fraction = float(evidence["fraction_under_250"])
        candidate.camera_board_mapping_variant = str(evidence["mapping_name"])
        candidate.camera_direct_pnp_rms_px = dict(evidence["pnp_summary"])
        if log:
            log(
                f"Candidate {candidate.candidate_index} ({candidate.filename}): camera-to-tracker median "
                f"{float(candidate.camera_distance_mm.get('median', float('nan'))):.1f} mm; "
                f"p05–p95 {float(candidate.camera_distance_mm.get('p05', float('nan'))):.1f}–"
                f"{float(candidate.camera_distance_mm.get('p95', float('nan'))):.1f} mm; "
                f"mapping {candidate.camera_board_mapping_variant}."
            )

    finite = [
        (float(item["score"]), index)
        for index, item in enumerate(best_by_candidate)
        if item is not None and np.isfinite(float(item["score"]))
    ]
    finite.sort()
    if not finite:
        raise TrackerIdentityError("No finite camera-to-tracker identity score was produced")
    best_score, selected_index = finite[0]
    selected_evidence = best_by_candidate[selected_index]
    assert selected_evidence is not None
    second_score = finite[1][0] if len(finite) > 1 else float("inf")
    second_index = finite[1][1] if len(finite) > 1 else None
    best_median = float(selected_evidence["distance_summary"]["median"])
    second_median = (
        float(best_by_candidate[second_index]["distance_summary"]["median"])
        if second_index is not None and best_by_candidate[second_index] is not None
        else float("inf")
    )
    margin = second_median - best_median
    ratio = second_median / max(best_median, 1e-9)
    fraction_under_250 = float(selected_evidence["fraction_under_250"])
    if len(candidate_rows) == 1:
        confidence = "SINGLE_CANDIDATE"
    elif best_median < 180.0 and margin >= 150.0 and ratio >= 2.0 and fraction_under_250 >= 0.50:
        confidence = "HIGH"
    elif best_median < 250.0 and margin >= 80.0 and ratio >= 1.5 and fraction_under_250 >= 0.30:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    selected = candidate_rows[selected_index]
    reasons = [
        "The full world C3D had no labelled participant head markers, so head-proximity identity was unavailable.",
        "The AprilTag board and Pupil scene image located the scene-camera optical centre directly in Vicon space.",
        f"Selected tracker median optical-centre distance was {best_median:.3f} mm.",
        f"Alternative tracker median distance was {second_median:.3f} mm (margin {margin:.3f} mm; ratio {ratio:.3f}).",
        f"Selected board mapping hypothesis was {selected_evidence['mapping_name']}.",
    ]
    warnings = [
        "No labelled C3D head markers were available; camera-in-front-of-head QC will be disabled for this calibration profile."
    ]
    if confidence == "LOW":
        warnings.append(
            "Camera-to-tracker proximity did not clearly separate the two candidates. Review the evidence and use a manual override only after independent verification."
        )
    if not _mapping_labels_exist(board, board_corner_labels, outer_corner_labels):
        warnings.append(
            "The board mapping currently typed in the GUI did not match this board C3D. A 4×4 mapping was inferred from board geometry and will replace it."
        )
    if log:
        log(
            f"Selected {selected.filename} using AprilTag camera-to-tracker proximity with {confidence} confidence."
        )

    _report_progress(progress, 1.0, "Tracker identity detection complete")
    return TrackerIdentityResult(
        wearer_name_raw=wearer_name,
        wearer_id=wearer_id,
        recording_id=recording_id,
        module_serial_number=module_serial,
        canonical_participant_id=canonical_participant_id,
        recommended_canonical_participant_id=(canonical_participant_id or wearer_name),
        available_c3d_subjects=[],
        matched_c3d_subject=None,
        subject_name_match_score=0.0,
        subject_match_source="apriltag_board_camera_proximity_no_labeled_head_markers",
        canonical_name_conflict=False,
        selected_candidate_index=selected_index + 1,
        selected_tracker_path=selected.path,
        selected_tracker_filename=selected.filename,
        selected_tracker_labels=selected.marker_labels,
        confidence=confidence,
        method="apriltag_scene_camera_to_tracker_spatial_proximity",
        reasons=reasons,
        warnings=warnings,
        candidates=candidate_rows,
        identity_evidence_type="apriltag_camera_to_tracker_proximity",
        board_mapping_source=str(selected_evidence["mapping_name"]),
        recommended_board_corner_labels={
            str(key): dict(value) for key, value in selected_evidence["tags"].items()
        },
        recommended_outer_corner_labels=dict(selected_evidence["outer"]),
        head_qc_available=False,
    )


@dataclass
class TrackerCandidateIdentity:
    candidate_index: int
    path: str
    filename: str
    sha256: str | None
    marker_labels: list[str]
    marker_prefix: str
    frame_count: int
    frame_rate_hz: float
    first_frame: int
    valid_centroid_frames: int
    subject_distance_mm: dict[str, dict[str, float | int]] = field(default_factory=dict)
    nearest_subject: str | None = None
    nearest_subject_distance_mm: float | None = None
    globally_assigned_subject: str | None = None
    camera_distance_mm: dict[str, float | int] = field(default_factory=dict)
    camera_proximity_score_mm: float | None = None
    camera_frames_under_250mm_fraction: float | None = None
    camera_board_mapping_variant: str | None = None
    camera_direct_pnp_rms_px: dict[str, float | int] = field(default_factory=dict)


@dataclass
class TrackerIdentityResult:
    wearer_name_raw: str
    wearer_id: str
    recording_id: str
    module_serial_number: str
    canonical_participant_id: str
    recommended_canonical_participant_id: str
    available_c3d_subjects: list[str]
    matched_c3d_subject: str | None
    subject_name_match_score: float
    subject_match_source: str
    canonical_name_conflict: bool
    selected_candidate_index: int | None
    selected_tracker_path: str | None
    selected_tracker_filename: str | None
    selected_tracker_labels: list[str]
    confidence: str
    method: str
    reasons: list[str]
    warnings: list[str]
    candidates: list[TrackerCandidateIdentity]
    identity_evidence_type: str = "head_marker_proximity"
    board_mapping_source: str = ""
    recommended_board_corner_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    recommended_outer_corner_labels: dict[str, str] = field(default_factory=dict)
    head_qc_available: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def selected_path(self) -> Path:
        if not self.selected_tracker_path:
            raise TrackerIdentityError("No tracker candidate was selected")
        return Path(self.selected_tracker_path)


def _match_one_name(
    query: str,
    subjects: list[str],
) -> tuple[str | None, float, float]:
    scored = sorted(
        ((name_similarity(query, subject), subject) for subject in subjects),
        key=lambda item: (-item[0], item[1].casefold()),
    )
    if not scored:
        return None, 0.0, 0.0
    best_score, best_subject = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 0.60 or (len(scored) > 1 and best_score - runner_up < 0.08):
        return None, float(best_score), float(runner_up)
    return best_subject, float(best_score), float(runner_up)


def _best_subject_match(
    wearer_name: str,
    canonical_participant_id: str,
    subjects: list[str],
) -> tuple[str | None, float, float, str, bool, str | None, float]:
    """Match the Native wearer to a C3D subject.

    The raw Pupil wearer name is authoritative when it is present and
    unambiguous.  The user-entered canonical participant ID is a fallback, not
    a competing query. This prevents a stale GUI value from overriding the
    wearer stored in the Native recording.
    """

    wearer_subject, wearer_score, wearer_runner = _match_one_name(
        wearer_name, subjects
    ) if wearer_name else (None, 0.0, 0.0)
    canonical_subject, canonical_score, _ = _match_one_name(
        canonical_participant_id, subjects
    ) if canonical_participant_id else (None, 0.0, 0.0)

    if wearer_subject is not None:
        conflict = bool(
            canonical_subject is not None
            and canonical_subject != wearer_subject
        )
        return (
            wearer_subject,
            float(wearer_score),
            float(wearer_runner),
            "pupil_wearer_name",
            conflict,
            canonical_subject,
            float(canonical_score),
        )
    if canonical_subject is not None:
        return (
            canonical_subject,
            float(canonical_score),
            0.0,
            "canonical_participant_id_fallback",
            False,
            canonical_subject,
            float(canonical_score),
        )
    best_score = max(float(wearer_score), float(canonical_score))
    runner = float(wearer_runner)
    return None, best_score, runner, "unresolved", False, canonical_subject, float(canonical_score)


def detect_matching_tracker(
    *,
    pupil_native_source: str | Path,
    tracker_candidate_paths: list[str | Path],
    world_c3d_path: str | Path | None = None,
    canonical_participant_id: str = "",
    board_c3d_path: str | Path | None = None,
    board_corner_labels: dict[str, dict[str, str]] | None = None,
    outer_corner_labels: dict[str, str] | None = None,
    sample_every_n_frames: int = 4,
    max_board_mapping_frames: int = 48,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> TrackerIdentityResult:
    """Select the tracker C3D spatially attached to the Native wearer.

    Labelled head markers are preferred.  If they are absent and an AprilTag
    board C3D is supplied, the detector reconstructs the scene-camera optical
    centre and compares it directly with both tracker clusters.
    """

    _report_progress(progress, 0.02, "Reading Pupil recording metadata")
    paths = [Path(path) for path in tracker_candidate_paths if str(path).strip()]
    if not paths:
        raise TrackerIdentityError("At least one tracker C3D candidate is required")
    if len({path.resolve() for path in paths}) != len(paths):
        raise TrackerIdentityError("The same tracker C3D was selected more than once")

    with NativeRecording.open(pupil_native_source) as recording:
        info = dict(recording.info)
        wearer_name = str(info.get("wearer_name", "")).strip()
        wearer_id = str(info.get("wearer_id", info.get("wearer_uuid", ""))).strip()
        recording_id = str(info.get("recording_id", "")).strip()
        module_serial = str(
            info.get("module_serial_number", recording.intrinsics.serial_number)
        ).strip()

    _report_progress(progress, 0.08, "Reading world C3D and discovering head subjects")
    if log:
        log(
            f"Pupil wearer: {wearer_name or '<not recorded>'}; "
            f"recording {recording_id or '<unknown>'}; module {module_serial or '<unknown>'}."
        )

    world = read_c3d(world_c3d_path) if world_c3d_path else None
    subjects = discover_head_subjects(world) if world is not None else {}
    if not subjects:
        if board_c3d_path is None:
            world_name = Path(world_c3d_path).name if world_c3d_path else "<not supplied>"
            raise TrackerIdentityError(
                "No labelled C3D subjects with at least three standard head markers "
                f"({', '.join(STANDARD_HEAD_SUFFIXES)}) were found in {world_name}. "
                "Select the AprilTag board C3D so the app can use its camera-to-tracker fallback, "
                "or use a verified manual override."
            )
        return _detect_matching_tracker_by_board_camera_proximity(
            pupil_native_source=pupil_native_source,
            tracker_candidate_paths=paths,
            board_c3d_path=board_c3d_path,
            canonical_participant_id=canonical_participant_id,
            wearer_name=wearer_name,
            wearer_id=wearer_id,
            recording_id=recording_id,
            module_serial=module_serial,
            board_corner_labels=board_corner_labels,
            outer_corner_labels=outer_corner_labels,
            sample_every_n_frames=sample_every_n_frames,
            max_board_mapping_frames=max_board_mapping_frames,
            log=log,
            progress=progress,
        )

    (
        matched_subject,
        match_score,
        runner_score,
        subject_match_source,
        canonical_name_conflict,
        canonical_matched_subject,
        canonical_match_score,
    ) = _best_subject_match(
        wearer_name,
        canonical_participant_id,
        sorted(subjects),
    )
    if log:
        if matched_subject:
            log(
                f"Matched '{wearer_name or canonical_participant_id}' to C3D subject "
                f"'{matched_subject}' using {subject_match_source} "
                f"(name score {match_score:.3f})."
            )
        else:
            log(
                "Could not unambiguously match the Pupil wearer name to a C3D subject; "
                f"best name score was {match_score:.3f}."
            )

    _report_progress(progress, 0.25, "Computing participant head-centre trajectories")
    subject_centers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for subject, labels in subjects.items():
        points, valid = world.select(labels)
        subject_centers[subject] = _centroid(points, valid, minimum_markers=3)

    candidates: list[TrackerCandidateIdentity] = []
    distance_matrix = np.full((len(paths), len(subjects)), np.nan, dtype=float)
    subject_names = sorted(subjects)

    for candidate_index, path in enumerate(paths, start=1):
        _report_progress(
            progress,
            0.32 + 0.52 * candidate_index / max(1, len(paths)),
            f"Comparing tracker candidate {candidate_index}/{len(paths)} with head trajectories",
        )
        tracker = read_c3d(path)
        labels = infer_tracker_marker_labels(tracker)
        positions, valid = tracker.select(labels)
        tracker_center, tracker_center_valid = _centroid(positions, valid, minimum_markers=3)
        tracker_slice, world_slice, overlap = _aligned_slices(tracker, world)
        tracker_center_overlap = tracker_center[tracker_slice]
        tracker_valid_overlap = tracker_center_valid[tracker_slice]

        subject_distance: dict[str, dict[str, float | int]] = {}
        for subject_index, subject in enumerate(subject_names):
            head_center, head_valid = subject_centers[subject]
            head_center_overlap = head_center[world_slice]
            head_valid_overlap = head_valid[world_slice]
            mask = tracker_valid_overlap & head_valid_overlap
            if np.any(mask):
                distance = np.linalg.norm(
                    tracker_center_overlap[mask] - head_center_overlap[mask], axis=1
                )
                summary = _distance_summary(distance)
                distance_matrix[candidate_index - 1, subject_index] = float(
                    summary.get("median", np.nan)
                )
            else:
                summary = {"n": 0}
            summary["overlap_frame_count"] = int(overlap)
            subject_distance[subject] = summary

        finite = [
            (float(values["median"]), subject)
            for subject, values in subject_distance.items()
            if values.get("n", 0) and "median" in values
        ]
        finite.sort(key=lambda item: item[0])
        nearest_distance, nearest_subject = finite[0] if finite else (None, None)
        prefix = labels[0].rsplit(":", 1)[0] if ":" in labels[0] else Path(path).stem
        candidates.append(
            TrackerCandidateIdentity(
                candidate_index=candidate_index,
                path=str(path),
                filename=path.name,
                sha256=sha256_file(path) if path.is_file() else None,
                marker_labels=labels,
                marker_prefix=prefix,
                frame_count=tracker.frame_count,
                frame_rate_hz=tracker.frame_rate_hz,
                first_frame=tracker.first_frame,
                valid_centroid_frames=int(tracker_center_valid.sum()),
                subject_distance_mm=subject_distance,
                nearest_subject=nearest_subject,
                nearest_subject_distance_mm=(
                    float(nearest_distance) if nearest_distance is not None else None
                ),
            )
        )
        if log:
            rendered = ", ".join(
                f"{subject}: {values.get('median', float('nan')):.1f} mm"
                for subject, values in subject_distance.items()
                if values.get("n", 0)
            )
            log(f"Candidate {candidate_index} ({path.name}) head distances — {rendered}")

    # Global one-to-one assignment is used when at least two candidates/subjects exist.
    finite_cost = np.where(np.isfinite(distance_matrix), distance_matrix, 1e9)
    if finite_cost.size and np.isfinite(distance_matrix).any():
        rows, cols = linear_sum_assignment(finite_cost)
        for row, col in zip(rows, cols):
            if finite_cost[row, col] < 1e8:
                candidates[int(row)].globally_assigned_subject = subject_names[int(col)]

    selected_index: int | None = None
    reasons: list[str] = []
    warnings: list[str] = []
    if canonical_name_conflict:
        warnings.append(
            "The user-entered canonical participant ID matched a different C3D subject "
            f"('{canonical_matched_subject}', score {canonical_match_score:.3f}) than the "
            f"Pupil Native wearer ('{matched_subject}'). The Native wearer match was used "
            "for tracker selection."
        )
    confidence = "AMBIGUOUS"
    method = "wearer_name_plus_tracker_to_head_spatial_proximity"

    if len(candidates) == 1:
        selected_index = 0
        confidence = "SINGLE_CANDIDATE"
        reasons.append("Only one tracker C3D candidate was supplied.")
    elif matched_subject:
        target_col = subject_names.index(matched_subject)
        target_distances = distance_matrix[:, target_col]
        finite_rows = np.flatnonzero(np.isfinite(target_distances))
        if len(finite_rows):
            ordered = finite_rows[np.argsort(target_distances[finite_rows])]
            selected_index = int(ordered[0])
            best = float(target_distances[selected_index])
            second = float(target_distances[int(ordered[1])]) if len(ordered) > 1 else float("inf")
            margin = second - best
            ratio = second / max(best, 1e-9)
            assigned = candidates[selected_index].globally_assigned_subject
            if best < 300.0 and margin >= 100.0 and ratio >= 1.5 and match_score >= 0.70:
                confidence = "HIGH"
            elif best < 350.0 and margin >= 50.0 and ratio >= 1.25:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            reasons.extend(
                [
                    f"Subject match source was {subject_match_source}; matched C3D subject "
                    f"'{matched_subject}' with score {match_score:.3f}.",
                    f"Selected tracker median distance to '{matched_subject}' was {best:.3f} mm.",
                    f"Alternative tracker median distance was {second:.3f} mm (margin {margin:.3f} mm; ratio {ratio:.3f}).",
                ]
            )
            if assigned and assigned != matched_subject:
                warnings.append(
                    f"Global tracker-to-subject assignment linked the selected tracker to '{assigned}', "
                    f"not the name-matched subject '{matched_subject}'."
                )
                confidence = "AMBIGUOUS"
                selected_index = None
        else:
            warnings.append(
                f"No candidate had valid overlapping frames with subject '{matched_subject}'."
            )
    else:
        warnings.append(
            "The Pupil wearer name could not be mapped unambiguously to a C3D subject. "
            "Use a manual override or correct the canonical participant ID."
        )

    if selected_index is not None and confidence in {"LOW", "AMBIGUOUS"}:
        warnings.append(
            "Tracker selection evidence is weak. Review the distance table before building a production profile."
        )
    if selected_index is None:
        selected_path = None
        selected_filename = None
        selected_labels: list[str] = []
    else:
        selected = candidates[selected_index]
        selected_path = selected.path
        selected_filename = selected.filename
        selected_labels = selected.marker_labels
        if log:
            log(
                f"Selected {selected.filename} ({selected.marker_prefix}) with {confidence} confidence."
            )

    if not wearer_name:
        warnings.append(
            "wearer.json did not provide a wearer name; canonical participant text was used where possible."
        )
    if matched_subject is None and match_score > 0:
        warnings.append(
            f"Best subject-name score {match_score:.3f} was not sufficiently separated from runner-up {runner_score:.3f}."
        )

    _report_progress(progress, 1.0, "Tracker identity detection complete")
    return TrackerIdentityResult(
        wearer_name_raw=wearer_name,
        wearer_id=wearer_id,
        recording_id=recording_id,
        module_serial_number=module_serial,
        canonical_participant_id=canonical_participant_id,
        recommended_canonical_participant_id=(
            wearer_name or canonical_participant_id or matched_subject or ""
        ),
        available_c3d_subjects=subject_names,
        matched_c3d_subject=matched_subject,
        subject_name_match_score=match_score,
        subject_match_source=subject_match_source,
        canonical_name_conflict=canonical_name_conflict,
        selected_candidate_index=(selected_index + 1 if selected_index is not None else None),
        selected_tracker_path=selected_path,
        selected_tracker_filename=selected_filename,
        selected_tracker_labels=selected_labels,
        confidence=confidence,
        method=method,
        reasons=reasons,
        warnings=warnings,
        candidates=candidates,
        identity_evidence_type="head_marker_proximity",
        board_mapping_source="user_supplied_or_not_required",
        recommended_board_corner_labels={},
        recommended_outer_corner_labels={},
        head_qc_available=True,
    )


def render_tracker_identity_text(result: TrackerIdentityResult) -> str:
    lines = [
        "PUPIL NATIVE RECORDING",
        f"wearer_name={result.wearer_name_raw or '<missing>'}",
        f"wearer_id={result.wearer_id or '<missing>'}",
        f"recording_id={result.recording_id or '<missing>'}",
        f"module_serial={result.module_serial_number or '<missing>'}",
        "",
        "TRACKER IDENTITY RESULT",
        f"evidence_type={result.identity_evidence_type}",
        f"method={result.method}",
        f"matched_subject={result.matched_c3d_subject or '<not available>'}",
        f"subject_match_source={result.subject_match_source}",
        f"subject_name_score={result.subject_name_match_score:.3f}",
        f"canonical_name_conflict={result.canonical_name_conflict}",
        f"recommended_canonical_participant_id={result.recommended_canonical_participant_id or '<none>'}",
        f"confidence={result.confidence}",
        f"selected_candidate={result.selected_candidate_index or '<none>'}",
        f"selected_file={result.selected_tracker_filename or '<none>'}",
        f"selected_labels={', '.join(result.selected_tracker_labels) or '<none>'}",
        f"head_qc_available={result.head_qc_available}",
    ]
    if result.recommended_board_corner_labels:
        lines.extend(
            [
                "",
                "RECOMMENDED APRILTAG BOARD MAPPING",
                f"mapping_source={result.board_mapping_source}",
            ]
        )
        for marker_id in ("0", "1", "2", "3"):
            mapping = result.recommended_board_corner_labels.get(marker_id, {})
            lines.append(
                f"Tag {marker_id}: TL={mapping.get('TL')}, TR={mapping.get('TR')}, "
                f"BR={mapping.get('BR')}, BL={mapping.get('BL')}"
            )
        outer = result.recommended_outer_corner_labels
        lines.append(
            f"Outer: TL={outer.get('TL')}, TR={outer.get('TR')}, "
            f"BR={outer.get('BR')}, BL={outer.get('BL')}"
        )

    if result.identity_evidence_type == "apriltag_camera_to_tracker_proximity":
        lines.extend(["", "CAMERA OPTICAL CENTRE → TRACKER CLUSTER DISTANCES"])
        for candidate in result.candidates:
            lines.append(
                f"Candidate {candidate.candidate_index}: {candidate.filename} "
                f"[{candidate.marker_prefix}]"
            )
            summary = candidate.camera_distance_mm
            if summary.get("n", 0):
                lines.append(
                    f"  median={float(summary['median']):.3f} mm, "
                    f"p05={float(summary['p05']):.3f}, p95={float(summary['p95']):.3f}, "
                    f"mean={float(summary['mean']):.3f}, n={int(summary['n'])}"
                )
                lines.append(
                    f"  fraction_under_250mm={float(candidate.camera_frames_under_250mm_fraction or 0.0):.3f}; "
                    f"score={float(candidate.camera_proximity_score_mm or float('nan')):.3f}; "
                    f"board_mapping={candidate.camera_board_mapping_variant or '<none>'}"
                )
                if candidate.camera_direct_pnp_rms_px.get("n", 0):
                    lines.append(
                        f"  direct_PnP_RMS_median={float(candidate.camera_direct_pnp_rms_px['median']):.3f} px"
                    )
            else:
                lines.append("  no valid camera-to-tracker evidence")
    else:
        lines.extend(["", "CANDIDATE DISTANCES TO HEAD CENTRES"])
        for candidate in result.candidates:
            lines.append(
                f"Candidate {candidate.candidate_index}: {candidate.filename} "
                f"[{candidate.marker_prefix}]"
            )
            for subject, summary in candidate.subject_distance_mm.items():
                if summary.get("n", 0):
                    lines.append(
                        f"  {subject}: median={float(summary['median']):.3f} mm, "
                        f"p05={float(summary['p05']):.3f}, p95={float(summary['p95']):.3f}, "
                        f"n={int(summary['n'])}"
                    )
                else:
                    lines.append(f"  {subject}: no valid overlap")
            lines.append(
                f"  nearest_subject={candidate.nearest_subject or '<none>'}; "
                f"global_assignment={candidate.globally_assigned_subject or '<none>'}"
            )
    if result.reasons:
        lines.extend(["", "REASONS", *[f"- {item}" for item in result.reasons]])
    if result.warnings:
        lines.extend(["", "WARNINGS", *[f"- {item}" for item in result.warnings]])
    return "\n".join(lines)

