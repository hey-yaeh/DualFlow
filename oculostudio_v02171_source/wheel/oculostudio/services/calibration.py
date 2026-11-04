from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.mocap import MocapData, read_mocap
from oculostudio.adapters.pupil import materialized_native_files, read_camera_intrinsics
from oculostudio.domain.geometry import (
    PoseSeries,
    build_marker_template,
    invert_transform,
    make_transform,
    robust_average_transforms,
    transform_points,
)
from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.services.hardware_registration import resolve_measured_registration
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, sha256_file, utc_now

CORNER_ORDER = ("TL", "TR", "BR", "BL")


@dataclass(frozen=True)
class CameraIntrinsics:
    matrix: np.ndarray
    distortion: np.ndarray
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        matrix = np.asarray(data.get("matrix") or data.get("camera_matrix"), dtype=float).reshape(3, 3)
        distortion = np.asarray(data.get("distortion") or data.get("distortion_coefficients") or [], dtype=float).reshape(-1)
        resolution = data.get("resolution") or [data.get("width"), data.get("height")]
        width = int(resolution[0]) if resolution and resolution[0] else None
        height = int(resolution[1]) if resolution and len(resolution) > 1 and resolution[1] else None
        return cls(matrix, distortion, width, height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist(),
            "distortion": self.distortion.tolist(),
            "resolution": [self.width, self.height],
        }


@dataclass(frozen=True)
class BoardDefinition:
    family: str
    tag_corners_mm: dict[int, np.ndarray]
    marker_template_mm: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "BoardDefinition":
        data = json.loads(Path(path).read_text("utf-8"))
        tags_raw = data.get("tag_corners_mm") or data.get("tags")
        marker_raw = data.get("marker_template_mm") or data.get("board_marker_template_mm")
        if not isinstance(tags_raw, dict) or not tags_raw:
            raise ValidationError("Board definition needs tag_corners_mm")
        if not isinstance(marker_raw, dict) or len(marker_raw) < 3:
            raise ValidationError("Board definition needs at least three marker_template_mm entries")
        tags = {int(key): np.asarray(value, dtype=float).reshape(4, 3) for key, value in tags_raw.items()}
        markers = {str(key): np.asarray(value, dtype=float).reshape(3) for key, value in marker_raw.items()}
        return cls(str(data.get("family") or "tag36h11"), tags, markers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "tag_corners_mm": {str(key): value.tolist() for key, value in self.tag_corners_mm.items()},
            "marker_template_mm": {key: value.tolist() for key, value in self.marker_template_mm.items()},
        }


@dataclass
class TagObservation:
    frame_index: int
    timestamp_ns: int
    object_points_mm: np.ndarray
    image_points_px: np.ndarray
    T_camera_from_board: np.ndarray
    direct_rms_px: float


def _dictionary_for_family(family: str):
    if not hasattr(cv2, "aruco"):
        raise ProcessingError("This OpenCV build does not include the aruco module")
    mapping = {
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    }
    key = family.lower().replace("apriltag", "tag").replace("_", "")
    for label, constant in mapping.items():
        if key == label.replace("_", ""):
            return cv2.aruco.getPredefinedDictionary(constant)
    raise ValidationError(f"Unsupported AprilTag family: {family}")


def _corner_permutations() -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    base = np.arange(4)
    for reverse in (False, True):
        order = base[::-1] if reverse else base
        for shift in range(4):
            candidate = tuple(int(value) for value in np.roll(order, -shift))
            if candidate not in result:
                result.append(candidate)
    return result


def detect_apriltag_observations(
    video_path: str | Path,
    scene_timestamps_ns: np.ndarray,
    intrinsics: CameraIntrinsics,
    board: BoardDefinition,
    *,
    sample_every_n_frames: int = 5,
    max_observations: int = 250,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> list[TagObservation]:
    cancellation = cancellation or CancellationToken.create()
    dictionary = _dictionary_for_family(board.family)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    required_ids = sorted(board.tag_corners_mm)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ProcessingError(f"Cannot open calibration video: {video_path}")
    total_frames = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or len(scene_timestamps_ns)), len(scene_timestamps_ns))
    observations: list[TagObservation] = []
    try:
        frame_index = 0
        while frame_index < total_frames:
            cancellation.raise_if_cancelled()
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % max(1, sample_every_n_frames):
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                found = {int(marker_id): np.asarray(marker_corners, dtype=float).reshape(4, 2) for marker_corners, marker_id in zip(corners, ids.reshape(-1))}
                if all(marker_id in found for marker_id in required_ids):
                    object_points = np.concatenate([board.tag_corners_mm[marker_id] for marker_id in required_ids], axis=0)
                    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
                    for permutation in _corner_permutations():
                        image_points = np.concatenate([found[marker_id][list(permutation)] for marker_id in required_ids], axis=0)
                        ok_pnp, rvec, tvec = cv2.solvePnP(
                            object_points,
                            image_points,
                            intrinsics.matrix,
                            intrinsics.distortion,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        if not ok_pnp:
                            continue
                        projected, _ = cv2.projectPoints(
                            object_points,
                            rvec,
                            tvec,
                            intrinsics.matrix,
                            intrinsics.distortion,
                        )
                        error = projected.reshape(-1, 2) - image_points
                        rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                        if best is None or rms < best[0]:
                            best = (rms, image_points, rvec, tvec)
                    if best is not None:
                        rms, image_points, rvec, tvec = best
                        rotation, _ = cv2.Rodrigues(rvec)
                        observations.append(
                            TagObservation(
                                frame_index=frame_index,
                                timestamp_ns=int(scene_timestamps_ns[frame_index]),
                                object_points_mm=object_points,
                                image_points_px=image_points,
                                T_camera_from_board=make_transform(rotation, tvec.reshape(3)),
                                direct_rms_px=rms,
                            )
                        )
            if frame_index % 60 == 0:
                emit(progress, frame_index / max(1, total_frames), f"Detecting AprilTags ({frame_index}/{total_frames})")
            frame_index += 1
    finally:
        capture.release()
    if len(observations) > max_observations:
        indices = np.linspace(0, len(observations) - 1, max_observations).round().astype(int)
        observations = [observations[index] for index in indices]
    if len(observations) < 3:
        raise ProcessingError(
            f"Only {len(observations)} usable AprilTag frames were found; at least three are required"
        )
    emit(progress, 1.0, f"Detected {len(observations)} usable AprilTag frames")
    return observations


def _pose_from_definition(data: MocapData, marker_template: dict[str, np.ndarray]) -> tuple[PoseSeries, list[str], np.ndarray]:
    labels = list(marker_template)
    positions, valid = data.select(labels)
    template = np.asarray([marker_template[label] for label in labels], dtype=float)
    pose = PoseSeries.from_markers(data.time_sec, template, positions, valid, min_markers=3, max_fit_rmse_mm=15.0)
    return pose, labels, template


def _tracker_pose(data: MocapData, labels: list[str] | None) -> tuple[PoseSeries, list[str], np.ndarray]:
    if labels:
        positions, valid = data.select(labels)
        used_labels = labels
    else:
        valid_count = data.valid.sum(axis=0)
        indices = np.argsort(valid_count)[::-1]
        indices = [int(index) for index in indices if valid_count[index] > 0][:8]
        if len(indices) < 3:
            raise ValidationError("Tracker C3D has fewer than three usable marker trajectories")
        used_labels = [data.labels[index] for index in indices]
        positions, valid = data.points_mm[:, indices], data.valid[:, indices]
    template = build_marker_template(positions, valid)
    pose = PoseSeries.from_markers(data.time_sec, template, positions, valid, min_markers=3, max_fit_rmse_mm=12.0)
    return pose, used_labels, template


def _camera_world_positions(
    observations: list[TagObservation],
    board_pose: PoseSeries,
    board_query_sec: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    interpolated = board_pose.interpolate(board_query_sec, max_gap_sec=max_gap_sec)
    positions = np.full((len(observations), 3), np.nan)
    for index in np.flatnonzero(interpolated.valid):
        T_world_from_board = make_transform(interpolated.rotations[index], interpolated.translations[index])
        T_world_from_camera = T_world_from_board @ invert_transform(observations[index].T_camera_from_board)
        positions[index] = T_world_from_camera[:3, 3]
    return positions, interpolated.valid


def _rank_tracker_candidates(
    tracker_candidates: list[tuple[str, MocapData, PoseSeries, list[str], np.ndarray]],
    observations: list[TagObservation],
    camera_world_positions: np.ndarray,
    camera_valid: np.ndarray,
    tracker_query_by_asset: dict[str, np.ndarray],
    *,
    max_gap_sec: float,
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for asset_id, _data, pose, labels, _template in tracker_candidates:
        interpolated = pose.interpolate(tracker_query_by_asset[asset_id], max_gap_sec=max_gap_sec)
        valid = camera_valid & interpolated.valid
        distance = np.linalg.norm(camera_world_positions - interpolated.translations, axis=1)
        values = distance[valid]
        ranking.append(
            {
                "asset_id": asset_id,
                "marker_labels": labels,
                "valid_frames": int(valid.sum()),
                "median_camera_to_tracker_mm": float(np.median(values)) if len(values) else float("inf"),
                "p90_camera_to_tracker_mm": float(np.percentile(values, 90)) if len(values) else float("inf"),
            }
        )
    return sorted(ranking, key=lambda row: row["median_camera_to_tracker_mm"])


def create_calibration_profile(
    store: ProjectStore,
    *,
    session_id: str,
    native_asset_id: str,
    tracker_asset_ids: list[str],
    board_asset_id: str,
    board_definition_path: str,
    intrinsics_path: str | None = None,
    participant_id: str = "P01",
    profile_name: str = "Calibration",
    tracker_marker_labels: dict[str, list[str]] | None = None,
    sample_every_n_frames: int = 5,
    max_pose_gap_sec: float = 0.1,
    minimum_identity_margin_mm: float = 10.0,
    hardware_id: str | None = None,
    mount_version: str | None = None,
    valid_sessions: list[str] | None = None,
    auto_register_hardware: bool = True,
    auto_profile_name: bool = True,
    allow_hardware_id_override: bool = False,
    output_g3dcal: bool = True,
    calibration_set_id: str | None = None,
    calibration_revision_id: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    if not tracker_asset_ids:
        raise ValidationError("At least one tracker C3D/NPZ asset is required")
    session = store.project.sessions[session_id]
    cache = MaterializationCache(store.cache_dir())
    native_asset = session.assets[native_asset_id]
    native_root = cache.materialize(native_asset)
    video_path, time_path, _, _ = materialized_native_files(native_root, native_asset.recording_id)
    scene_timestamps = np.fromfile(time_path, dtype="<i8").astype(np.int64, copy=False)
    if intrinsics_path:
        intrinsics_data = json.loads(Path(intrinsics_path).read_text("utf-8"))
    else:
        intrinsics_data = read_camera_intrinsics(native_asset.source_path)
        if intrinsics_data is None:
            raise ValidationError("Camera intrinsics were not found; provide an intrinsics JSON file")
    intrinsics = CameraIntrinsics.from_dict(intrinsics_data)
    board = BoardDefinition.load(board_definition_path)
    emit(progress, 0.02, "Reading calibration motion capture")
    board_asset = session.assets[board_asset_id]
    board_data = read_mocap(board_asset.source_path)
    board_pose, board_labels, board_template = _pose_from_definition(board_data, board.marker_template_mm)
    tracker_candidates = []
    for asset_id in tracker_asset_ids:
        asset = session.assets[asset_id]
        data = read_mocap(asset.source_path)
        pose, labels, template = _tracker_pose(data, (tracker_marker_labels or {}).get(asset_id))
        tracker_candidates.append((asset_id, data, pose, labels, template))
    cancellation.raise_if_cancelled()
    observations = detect_apriltag_observations(
        video_path,
        scene_timestamps,
        intrinsics,
        board,
        sample_every_n_frames=sample_every_n_frames,
        progress=lambda f, s: emit(progress, 0.05 + 0.35 * f, s),
        cancellation=cancellation,
    )
    graph = TimeGraph.from_session(session)
    if not native_asset.primary_time_domain_id or not board_asset.primary_time_domain_id:
        raise ValidationError("Calibration assets are missing time domains")
    observation_times = np.asarray([observation.timestamp_ns for observation in observations], dtype=np.int64)
    board_query_ns = graph.convert(native_asset.primary_time_domain_id, board_asset.primary_time_domain_id, observation_times)
    board_query_sec = np.asarray(board_query_ns, dtype=np.float64) / 1e9
    camera_world, camera_valid = _camera_world_positions(
        observations, board_pose, board_query_sec, max_gap_sec=max_pose_gap_sec
    )
    tracker_query_by_asset: dict[str, np.ndarray] = {}
    for asset_id, *_ in tracker_candidates:
        asset = session.assets[asset_id]
        if not asset.primary_time_domain_id:
            raise ValidationError(f"Tracker asset has no time domain: {asset.label}")
        query_ns = graph.convert(native_asset.primary_time_domain_id, asset.primary_time_domain_id, observation_times)
        tracker_query_by_asset[asset_id] = np.asarray(query_ns, dtype=np.float64) / 1e9
    ranking = _rank_tracker_candidates(
        tracker_candidates,
        observations,
        camera_world,
        camera_valid,
        tracker_query_by_asset,
        max_gap_sec=max_pose_gap_sec,
    )
    if not ranking or not np.isfinite(ranking[0]["median_camera_to_tracker_mm"]):
        raise ProcessingError("No tracker candidate overlaps the calibration observations")
    if len(ranking) > 1:
        margin = ranking[1]["median_camera_to_tracker_mm"] - ranking[0]["median_camera_to_tracker_mm"]
        if margin < minimum_identity_margin_mm:
            raise ProcessingError(
                f"Tracker identity is ambiguous: distance margin {margin:.2f} mm is below {minimum_identity_margin_mm:.2f} mm"
            )
    selected_id = ranking[0]["asset_id"]
    selected = next(row for row in tracker_candidates if row[0] == selected_id)
    _, selected_data, selected_pose, selected_labels, selected_template = selected
    selected_asset = session.assets[selected_id]
    registration = resolve_measured_registration(
        session=session,
        selected_tracker_label=selected_asset.label or Path(selected_asset.source_path).name,
        marker_labels=selected_labels,
        calibration_wearer_id=participant_id,
        requested_hardware_id=hardware_id,
        mount_version=mount_version,
        valid_sessions=valid_sessions or [session.label],
        requested_profile_name=profile_name,
        auto_register_hardware=auto_register_hardware,
        auto_profile_name=auto_profile_name,
        allow_hardware_id_override=allow_hardware_id_override,
    )
    hardware_id = registration["hardware_id"]
    mount_version = registration["mount_version"]
    valid_sessions = registration["valid_sessions"]
    profile_name = registration["profile_name"]
    selected_interpolated = selected_pose.interpolate(
        tracker_query_by_asset[selected_id], max_gap_sec=max_pose_gap_sec
    )
    board_interpolated = board_pose.interpolate(board_query_sec, max_gap_sec=max_pose_gap_sec)
    valid = selected_interpolated.valid & board_interpolated.valid
    candidates: list[np.ndarray] = []
    candidate_indices: list[int] = []
    for index in np.flatnonzero(valid):
        T_world_from_tracker = make_transform(
            selected_interpolated.rotations[index], selected_interpolated.translations[index]
        )
        T_world_from_board = make_transform(
            board_interpolated.rotations[index], board_interpolated.translations[index]
        )
        T_world_from_camera = T_world_from_board @ invert_transform(observations[index].T_camera_from_board)
        candidates.append(invert_transform(T_world_from_tracker) @ T_world_from_camera)
        candidate_indices.append(int(index))
    if len(candidates) < 3:
        raise ProcessingError("Fewer than three synchronized tracker/board/tag observations are valid")
    emit(progress, 0.65, "Robustly aggregating camera-to-tracker transforms")
    transform, accepted_mask, robust_metrics = robust_average_transforms(np.asarray(candidates))
    accepted_observation_indices = [candidate_indices[i] for i in np.flatnonzero(accepted_mask)]
    direct_rms = np.asarray([observations[index].direct_rms_px for index in accepted_observation_indices])
    profile_id = new_id("cal")
    qc = {
        **robust_metrics,
        "apriltag_observation_count": len(observations),
        "synchronized_candidate_count": len(candidates),
        "direct_reprojection_rms_px_median": float(np.median(direct_rms)),
        "direct_reprojection_rms_px_p90": float(np.percentile(direct_rms, 90)),
        "tracker_ranking": ranking,
        "selected_tracker_asset_id": selected_id,
        "camera_origin_tracker_mm": transform[:3, 3].tolist(),
        "camera_origin_norm_mm": float(np.linalg.norm(transform[:3, 3])),
        "hardware_id": hardware_id,
        "hardware_display_name": registration["hardware_display_name"],
        "mount_version": mount_version,
        "hardware_registration_status": registration["registration_status"],
        "hardware_registration_revision": registration["registration_revision"],
    }
    output_root = store.derived_dir(session_id, "calibrations", profile_id)
    output_path = output_root / f"{safe_name(profile_name)}.oculocal"
    profile = {
        "format": "OculoStudio Calibration Profile",
        "format_version": "1.0",
        "profile_id": profile_id,
        "profile_name": profile_name,
        "participant_id": participant_id,
        "hardware_id": hardware_id,
        "hardware_display_name": registration["hardware_display_name"],
        "legacy_hardware_aliases": registration["legacy_aliases"],
        "hardware_registration": registration,
        "mount_version": mount_version,
        "valid_sessions": [str(value).strip() for value in (valid_sessions or []) if str(value).strip()],
        "calibration_source": "session_measured_apriltag",
        "T_tracker_from_camera": transform.tolist(),
        "R_tracker_from_camera": transform[:3, :3].tolist(),
        "camera_origin_tracker_mm": transform[:3, 3].tolist(),
        "camera_intrinsics": intrinsics.to_dict(),
        "tracker": {
            "source_asset_id": selected_id,
            "marker_labels": selected_labels,
            "marker_template_mm": selected_template.tolist(),
        },
        "board": {
            "source_asset_id": board_asset_id,
            "marker_labels": board_labels,
            "definition": board.to_dict(),
        },
        "synchronization": {
            "native_time_domain_id": native_asset.primary_time_domain_id,
            "tracker_time_domain_id": session.assets[selected_id].primary_time_domain_id,
            "board_time_domain_id": board_asset.primary_time_domain_id,
        },
        "sources": {
            "native_asset_id": native_asset_id,
            "native_sha256": native_asset.source_sha256,
            "tracker_asset_id": selected_id,
            "tracker_sha256": session.assets[selected_id].source_sha256,
            "board_asset_id": board_asset_id,
            "board_sha256": board_asset.source_sha256,
            "board_definition_path": str(Path(board_definition_path).resolve()),
            "board_definition_sha256": sha256_file(board_definition_path),
        },
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(output_path, profile)
    portable_path = None
    if output_g3dcal:
        from oculostudio.services.calibration_profiles import write_g3dcal_profile

        portable_path = output_root / f"{safe_name(profile_name)}.g3dcal"
        write_g3dcal_profile(profile, portable_path)
    atomic_write_json(output_root / "calibration_qc.json", qc)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=profile_name,
        output_root=output_root,
        source_asset_ids=[native_asset_id, selected_id, board_asset_id],
        qc=qc,
        parameters={
            "participant_id": participant_id,
            "sample_every_n_frames": sample_every_n_frames,
            "max_pose_gap_sec": max_pose_gap_sec,
            "minimum_identity_margin_mm": minimum_identity_margin_mm,
            "hardware_id": hardware_id,
            "hardware_display_name": registration["hardware_display_name"],
            "mount_version": mount_version,
            "valid_sessions": [str(value).strip() for value in (valid_sessions or []) if str(value).strip()],
            "calibration_source": "session_measured_apriltag",
            "auto_register_hardware": bool(auto_register_hardware),
            "auto_profile_name": bool(auto_profile_name),
            "allow_hardware_id_override": bool(allow_hardware_id_override),
            "hardware_registration_status": registration["registration_status"],
            "hardware_registration_revision": registration["registration_revision"],
            "supersedes_profile_artifact_id": registration["supersedes_profile_artifact_id"],
            "legacy_hardware_aliases": registration["legacy_aliases"],
            "output_g3dcal": output_g3dcal,
            "calibration_set_id": calibration_set_id,
            "calibration_revision_id": calibration_revision_id,
        },
    )
    calibration_set_registration = None
    if calibration_set_id:
        from oculostudio.services.calibration_sets import register_calibration_profile_ref

        calibration_set_registration = register_calibration_profile_ref(
            store,
            calibration_set_id=calibration_set_id,
            source_session_id=session_id,
            profile_artifact_id=artifact.id,
            revision_id=calibration_revision_id,
        )
    emit(progress, 1.0, "Calibration profile complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "profile_path": str(output_path),
        "g3dcal_path": str(portable_path) if portable_path else None,
        "hardware_registration": registration,
        "profile": profile,
        "calibration_set_registration": calibration_set_registration,
    }


def load_calibration_profile(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".g3dcal":
        import io
        import zipfile

        try:
            with zipfile.ZipFile(path) as archive:
                metadata = json.loads(archive.read("profile.json"))
                arrays = np.load(io.BytesIO(archive.read("arrays.npz")))
                tracker_template = np.asarray(arrays["tracker_template_mm"], dtype=float)
                transform = np.asarray(arrays["T_tracker_from_camera"], dtype=float)
                camera_matrix = np.asarray(arrays["scene_camera_matrix"], dtype=float)
                distortion = np.asarray(arrays["scene_distortion_coefficients"], dtype=float)
        except Exception as exc:
            raise ValidationError(f"Could not read .g3dcal profile {path}: {exc}") from exc
        labels = list(metadata.get("tracker_marker_labels") or [])
        if len(labels) != len(tracker_template):
            raise ValidationError(".g3dcal tracker labels and template have different lengths")
        data = {
            "format": metadata.get("format", "NeonVicon-Gaze3D-CalibrationProfile"),
            "format_version": metadata.get("format_version"),
            "profile_id": metadata.get("profile_id"),
            "profile_name": metadata.get("display_name") or path.stem,
            "participant_id": metadata.get("canonical_participant_id"),
            "hardware_id": (metadata.get("metadata") or {}).get("hardware_id"),
            "hardware_display_name": (metadata.get("metadata") or {}).get("hardware_display_name"),
            "legacy_hardware_aliases": (metadata.get("metadata") or {}).get("legacy_hardware_aliases") or [],
            "hardware_registration": (metadata.get("metadata") or {}).get("hardware_registration") or {},
            "mount_version": (metadata.get("metadata") or {}).get("mount_version"),
            "valid_sessions": (metadata.get("metadata") or {}).get("valid_sessions") or [],
            "calibration_source": (metadata.get("metadata") or {}).get("calibration_source") or "imported_g3dcal",
            "source_profile_id": (metadata.get("metadata") or {}).get("source_profile_id"),
            "transfer_validation": (metadata.get("metadata") or {}).get("transfer_validation") or {},
            "T_tracker_from_camera": transform.tolist(),
            "R_tracker_from_camera": transform[:3, :3].tolist(),
            "camera_origin_tracker_mm": transform[:3, 3].tolist(),
            "camera_intrinsics": {
                "matrix": camera_matrix.tolist(),
                "distortion": distortion.tolist(),
                "resolution": [None, None],
            },
            "tracker": {
                "source_asset_id": None,
                "marker_labels": labels,
                "marker_template_mm": tracker_template.tolist(),
            },
            "metadata": metadata.get("metadata") or {},
            "warnings": metadata.get("warnings") or [],
            "qc": metadata.get("metrics") or {},
            "source_profile_path": str(path),
        }
    else:
        data = json.loads(path.read_text("utf-8"))
    required = ["T_tracker_from_camera", "camera_intrinsics", "tracker"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValidationError(f"Calibration profile is missing fields: {missing}")
    np.asarray(data["T_tracker_from_camera"], dtype=float).reshape(4, 4)
    return data
