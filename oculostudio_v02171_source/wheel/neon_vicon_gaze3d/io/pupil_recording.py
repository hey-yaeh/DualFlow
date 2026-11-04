from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import tempfile
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .archive import ArchiveError, MaterializedSource, find_unique


CALIBRATION_DTYPE = np.dtype(
    [
        ("version", "u1"),
        ("serial", "S6"),
        ("scene_camera_matrix", "(3,3)<f8"),
        ("scene_distortion_coefficients", "(8,)<f8"),
        ("scene_extrinsics_affine_matrix", "(4,4)<f8"),
        ("right_camera_matrix", "(3,3)<f8"),
        ("right_distortion_coefficients", "(8,)<f8"),
        ("right_extrinsics_affine_matrix", "(4,4)<f8"),
        ("left_camera_matrix", "(3,3)<f8"),
        ("left_distortion_coefficients", "(8,)<f8"),
        ("left_extrinsics_affine_matrix", "(4,4)<f8"),
        ("crc", "<u4"),
    ]
)


class PupilRecordingError(RuntimeError):
    pass


@dataclass
class CameraIntrinsics:
    matrix: np.ndarray
    distortion: np.ndarray
    serial_number: str


@dataclass
class SceneSection:
    section_id: str
    video_path: Path
    timestamps_ns: np.ndarray

    def frame_count_video(self) -> int:
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    def read_frame(self, local_index: int) -> np.ndarray:
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(local_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise PupilRecordingError(
                    f"Could not decode frame {local_index} from {self.video_path.name}"
                )
            return frame
        finally:
            capture.release()


@dataclass
class SceneIndex:
    sections: list[SceneSection]

    @property
    def available(self) -> bool:
        return bool(self.sections)

    @property
    def total_timestamp_rows(self) -> int:
        return sum(len(section.timestamps_ns) for section in self.sections)

    def timestamp_bounds(self) -> tuple[int, int] | None:
        if not self.sections:
            return None
        values = [section.timestamps_ns for section in self.sections if len(section.timestamps_ns)]
        if not values:
            return None
        return int(min(v[0] for v in values)), int(max(v[-1] for v in values))

    def nearest_frame(self, timestamp_ns: int) -> tuple[SceneSection, int, int]:
        if not self.sections:
            raise PupilRecordingError("No scene video is available")
        best: tuple[int, SceneSection, int] | None = None
        for section in self.sections:
            ts = section.timestamps_ns
            if len(ts) == 0:
                continue
            index = int(np.searchsorted(ts, timestamp_ns))
            candidates = [max(0, min(len(ts) - 1, index))]
            if index > 0:
                candidates.append(index - 1)
            for candidate in candidates:
                delta = abs(int(ts[candidate]) - int(timestamp_ns))
                if best is None or delta < best[0]:
                    best = (delta, section, candidate)
        if best is None:
            raise PupilRecordingError("Scene index has no frame timestamps")
        return best[1], best[2], best[0]

    def read_nearest(self, timestamp_ns: int) -> tuple[np.ndarray, int, str, int]:
        section, index, delta = self.nearest_frame(timestamp_ns)
        frame = section.read_frame(index)
        return frame, int(section.timestamps_ns[index]), section.section_id, delta


@dataclass
class TimeseriesRecording:
    root: Path
    info: dict
    gaze: pd.DataFrame
    world_timestamps: pd.DataFrame
    intrinsics: CameraIntrinsics
    imu: pd.DataFrame | None
    scene: SceneIndex
    data_source_path: Path
    scene_source_path: Path | None
    _sources: list[MaterializedSource]

    @classmethod
    def open(
        cls,
        data_source: str | Path,
        scene_source: str | Path | None = None,
        *,
        temp_parent: str | Path | None = None,
    ) -> "TimeseriesRecording":
        sources = [MaterializedSource(data_source, temp_parent)]
        if scene_source:
            scene_path = Path(scene_source)
            if scene_path.resolve() != Path(data_source).resolve():
                sources.append(MaterializedSource(scene_source, temp_parent))
        try:
            data_root = sources[0].root
            info_path = find_unique(data_root, "info.json")
            gaze_path = find_unique(data_root, "gaze.csv")
            camera_path = find_unique(data_root, "scene_camera.json")
            world_path = find_unique(data_root, "world_timestamps.csv")
            assert info_path and gaze_path and camera_path and world_path
            recording_root = info_path.parent
            info = json.loads(info_path.read_text(encoding="utf-8"))
            wearer_path = find_unique(data_root, "wearer.json", required=False)
            if wearer_path:
                wearer = json.loads(wearer_path.read_text(encoding="utf-8"))
                info.setdefault("wearer_name", str(wearer.get("name", "")))
                info.setdefault("wearer_id", str(wearer.get("uuid", info.get("wearer_id", ""))))
            gaze = pd.read_csv(gaze_path)
            world = pd.read_csv(world_path)
            camera = json.loads(camera_path.read_text(encoding="utf-8"))
            distortion = camera.get("distortion_coefficients", camera.get("dist_coefs"))
            if distortion is None:
                raise PupilRecordingError("scene_camera.json has no distortion coefficients")
            intrinsics = CameraIntrinsics(
                matrix=np.asarray(camera["camera_matrix"], dtype=float),
                distortion=np.asarray(distortion, dtype=float).reshape(-1),
                serial_number=str(camera.get("serial_number", info.get("module_serial_number", ""))),
            )
            imu_path = find_unique(data_root, "imu.csv", required=False)
            imu = pd.read_csv(imu_path) if imu_path else None
            roots = [source.root for source in sources]
            # A separate scene source may be either the matching Cloud scene
            # export or the Native recording.  Native video must use its own
            # .time file; it must never be paired by row number with
            # world_timestamps.csv.
            scene = SceneIndex([])
            for candidate_root in roots[1:] + roots[:1]:
                native_time = find_unique(
                    candidate_root, "Neon Scene Camera v1 ps1.time", required=False
                )
                native_video = find_unique(
                    candidate_root, "Neon Scene Camera v1 ps1.mp4", required=False
                )
                if native_time and native_video:
                    native_ts = np.fromfile(native_time, dtype="<u8")
                    scene = SceneIndex(
                        [SceneSection("native_fallback", native_video, native_ts)]
                    )
                    break
            if not scene.available:
                scene = _build_timeseries_scene_index(roots, world)
            return cls(
                root=recording_root,
                info=info,
                gaze=gaze,
                world_timestamps=world,
                intrinsics=intrinsics,
                imu=imu,
                scene=scene,
                data_source_path=Path(data_source),
                scene_source_path=Path(scene_source) if scene_source else None,
                _sources=sources,
            )
        except Exception:
            for source in sources:
                source.close()
            raise

    def close(self) -> None:
        for source in self._sources:
            source.close()
        self._sources.clear()

    def __enter__(self) -> "TimeseriesRecording":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class NativeRecording:
    root: Path
    info: dict
    intrinsics: CameraIntrinsics
    scene_timestamps_ns: np.ndarray
    scene_video_path: Path
    gaze_timestamps_ns: np.ndarray | None
    gaze_xy_px: np.ndarray | None
    _source: MaterializedSource

    @classmethod
    def open(
        cls, source: str | Path, *, temp_parent: str | Path | None = None
    ) -> "NativeRecording":
        materialized = MaterializedSource(source, temp_parent)
        try:
            root = materialized.root
            info_path = find_unique(root, "info.json")
            calibration_path = find_unique(root, "calibration.bin")
            scene_time_path = find_unique(root, "Neon Scene Camera v1 ps1.time")
            scene_video_path = find_unique(root, "Neon Scene Camera v1 ps1.mp4")
            assert info_path and calibration_path and scene_time_path and scene_video_path
            recording_root = info_path.parent
            info = json.loads(info_path.read_text(encoding="utf-8"))
            wearer_path = find_unique(root, "wearer.json", required=False)
            if wearer_path:
                wearer = json.loads(wearer_path.read_text(encoding="utf-8"))
                info.setdefault("wearer_name", str(wearer.get("name", "")))
                info.setdefault("wearer_id", str(wearer.get("uuid", info.get("wearer_id", ""))))
            calib = np.fromfile(calibration_path, dtype=CALIBRATION_DTYPE, count=1)
            if len(calib) != 1:
                raise PupilRecordingError("Could not read calibration.bin")
            row = calib[0]
            intrinsics = CameraIntrinsics(
                matrix=np.asarray(row["scene_camera_matrix"], dtype=float),
                distortion=np.asarray(row["scene_distortion_coefficients"], dtype=float),
                serial_number=row["serial"].decode("ascii", errors="ignore").strip("\x00"),
            )
            scene_ts = np.fromfile(scene_time_path, dtype="<u8")
            gaze_time_path = find_unique(root, "gaze ps1.time", required=False)
            gaze_raw_path = find_unique(root, "gaze ps1.raw", required=False)
            gaze_ts: np.ndarray | None = None
            gaze_xy: np.ndarray | None = None
            if gaze_time_path and gaze_raw_path:
                gaze_ts = np.fromfile(gaze_time_path, dtype="<u8")
                raw = np.fromfile(gaze_raw_path, dtype="<f4")
                if raw.size % 2 == 0:
                    gaze_xy = raw.reshape(-1, 2).astype(float)
                    count = min(len(gaze_ts), len(gaze_xy))
                    gaze_ts = gaze_ts[:count]
                    gaze_xy = gaze_xy[:count]
            return cls(
                root=recording_root,
                info=info,
                intrinsics=intrinsics,
                scene_timestamps_ns=scene_ts,
                scene_video_path=scene_video_path,
                gaze_timestamps_ns=gaze_ts,
                gaze_xy_px=gaze_xy,
                _source=materialized,
            )
        except Exception:
            materialized.close()
            raise

    @property
    def scene(self) -> SceneIndex:
        return SceneIndex(
            [
                SceneSection(
                    section_id="native",
                    video_path=self.scene_video_path,
                    timestamps_ns=self.scene_timestamps_ns,
                )
            ]
        )

    def close(self) -> None:
        self._source.close()

    def __enter__(self) -> "NativeRecording":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _build_timeseries_scene_index(
    roots: Iterable[Path], world: pd.DataFrame
) -> SceneIndex:
    if "section id" not in world.columns or "timestamp [ns]" not in world.columns:
        raise PupilRecordingError("world_timestamps.csv has unexpected columns")
    mp4s: list[Path] = []
    for root in roots:
        mp4s.extend(root.rglob("*.mp4"))
    sections: list[SceneSection] = []
    for section_id, rows in world.groupby("section id", sort=False):
        sid = str(section_id)
        candidates = [path for path in mp4s if path.name.startswith(sid)]
        if not candidates:
            short = sid.split("-")[0]
            candidates = [path for path in mp4s if path.name.startswith(short)]
        if not candidates and len(set(world["section id"].astype(str))) == 1 and len(mp4s) == 1:
            candidates = [mp4s[0]]
        if not candidates:
            continue
        candidates.sort(key=lambda path: len(path.parts))
        timestamps = rows["timestamp [ns]"].to_numpy(dtype=np.int64)
        sections.append(
            SceneSection(
                section_id=sid,
                video_path=candidates[0],
                timestamps_ns=timestamps,
            )
        )
    return SceneIndex(sections)


def unproject_gaze_pixels(
    gaze_xy_px: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    gaze_xy_px = np.asarray(gaze_xy_px, dtype=float)
    directions = np.full((len(gaze_xy_px), 3), np.nan, dtype=float)
    valid = np.isfinite(gaze_xy_px).all(axis=1)
    if not valid.any():
        return directions
    undistorted = cv2.undistortPoints(
        gaze_xy_px[valid].reshape(-1, 1, 2),
        intrinsics.matrix,
        intrinsics.distortion,
    ).reshape(-1, 2)
    rays = np.column_stack([undistorted, np.ones(len(undistorted))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    directions[valid] = rays
    return directions
