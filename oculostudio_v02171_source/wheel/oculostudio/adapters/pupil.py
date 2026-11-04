from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import SourceView
from oculostudio.errors import AssetError, ValidationError


SCENE_VIDEO_PATTERN = re.compile(r"neon scene camera.*\.mp4$", re.I)
SCENE_TIME_PATTERN = re.compile(r"neon scene camera.*\.time$", re.I)


@dataclass(frozen=True)
class NativeCandidate:
    root: str
    info_name: str
    video_name: str
    time_name: str
    recording_id: str | None
    info: dict[str, Any]


@dataclass(frozen=True)
class CloudCandidate:
    root: str
    world_timestamps_name: str
    gaze_name: str
    info_name: str | None
    scene_video_name: str | None
    recording_id: str | None
    info: dict[str, Any]


def _parent(name: str) -> str:
    value = PurePosixPath(name)
    return value.parent.as_posix() if value.parent.as_posix() != "." else ""


def _join(root: str, basename: str) -> str:
    return f"{root}/{basename}" if root else basename


def _recording_id(info: dict[str, Any]) -> str | None:
    for key in ("recording_id", "recording id", "id"):
        value = info.get(key)
        if value:
            return str(value)
    recording = info.get("recording")
    if isinstance(recording, dict):
        return str(recording.get("id") or recording.get("recording_id") or "") or None
    return None


def find_native_candidates(source: str | Path | SourceView) -> list[NativeCandidate]:
    owns = not isinstance(source, SourceView)
    view = SourceView(source) if owns else source
    try:
        names = view.names()
        by_root: dict[str, list[str]] = {}
        for name in names:
            by_root.setdefault(_parent(name), []).append(name)
        rows: list[NativeCandidate] = []
        for root, members in by_root.items():
            info_names = [n for n in members if PurePosixPath(n).name.lower() == "info.json"]
            videos = [n for n in members if SCENE_VIDEO_PATTERN.search(PurePosixPath(n).name)]
            times = [n for n in members if SCENE_TIME_PATTERN.search(PurePosixPath(n).name)]
            if not (info_names and videos and times):
                continue
            try:
                info = view.read_json(info_names[0])
            except Exception:
                info = {}
            rows.append(
                NativeCandidate(root, info_names[0], videos[0], times[0], _recording_id(info), info)
            )
        return rows
    finally:
        if owns:
            view.close()


def find_cloud_candidates(source: str | Path | SourceView) -> list[CloudCandidate]:
    owns = not isinstance(source, SourceView)
    view = SourceView(source) if owns else source
    try:
        names = view.names()
        by_root: dict[str, list[str]] = {}
        for name in names:
            by_root.setdefault(_parent(name), []).append(name)
        rows: list[CloudCandidate] = []
        for root, members in by_root.items():
            world = [n for n in members if PurePosixPath(n).name.lower() == "world_timestamps.csv"]
            gaze = [n for n in members if PurePosixPath(n).name.lower() == "gaze.csv"]
            if not (world and gaze):
                continue
            info_names = [n for n in members if PurePosixPath(n).name.lower() == "info.json"]
            videos = [n for n in members if PurePosixPath(n).suffix.lower() == ".mp4" and "scene" in PurePosixPath(n).name.lower()]
            info = {}
            if info_names:
                try:
                    info = view.read_json(info_names[0])
                except Exception:
                    pass
            rows.append(
                CloudCandidate(
                    root=root,
                    world_timestamps_name=world[0],
                    gaze_name=gaze[0],
                    info_name=info_names[0] if info_names else None,
                    scene_video_name=videos[0] if videos else None,
                    recording_id=_recording_id(info),
                    info=info,
                )
            )
        return rows
    finally:
        if owns:
            view.close()


def read_native_scene_timestamps(source: str | Path, candidate_index: int = 0) -> np.ndarray:
    with SourceView(source) as view:
        candidates = find_native_candidates(view)
        if not candidates:
            raise AssetError("No Native recording with info.json, scene video, and scene .time was found")
        candidate = candidates[candidate_index]
        raw = view.read_bytes(candidate.time_name)
    if len(raw) % 8:
        raise AssetError("Native scene .time size is not divisible by 8 bytes")
    timestamps = np.frombuffer(raw, dtype="<i8").astype(np.int64, copy=True)
    if len(timestamps) == 0:
        raise AssetError("Native scene .time is empty")
    return timestamps


def _timestamp_column(columns: Iterable[str]) -> str:
    normalized = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in columns}
    for key in ("timestampns", "worldtimestampns", "timestamp"):
        if key in normalized:
            return normalized[key]
    if len(normalized) == 1:
        return next(iter(normalized.values()))
    raise ValidationError("Could not determine a timestamp column")


def read_cloud_world_timestamps(source: str | Path, candidate_index: int = 0) -> np.ndarray:
    with SourceView(source) as view:
        candidates = find_cloud_candidates(view)
        if not candidates:
            raise AssetError("No Timeseries recording with world_timestamps.csv and gaze.csv was found")
        text = view.read_text(candidates[candidate_index].world_timestamps_name)
    frame = pd.read_csv(io.StringIO(text))
    column = _timestamp_column(frame.columns)
    values = pd.to_numeric(frame[column], errors="coerce").dropna().astype("int64").to_numpy()
    if len(values) == 0:
        raise AssetError("world_timestamps.csv contains no valid timestamps")
    return values


def find_facemapper_files(source: str | Path | SourceView) -> dict[str, list[str]]:
    owns = not isinstance(source, SourceView)
    view = SourceView(source) if owns else source
    try:
        names = view.names()
        detections = [
            n for n in names if PurePosixPath(n).name.lower() in {"face_detections.csv", "face_positions.csv"}
        ]
        sections = [n for n in names if PurePosixPath(n).name.lower() == "sections.csv"]
        gaze = [n for n in names if "gaze" in PurePosixPath(n).name.lower() and n.lower().endswith(".csv")]
        fixations = [n for n in names if "fixation" in PurePosixPath(n).name.lower() and n.lower().endswith(".csv")]
        return {"detections": detections, "sections": sections, "gaze": gaze, "fixations": fixations}
    finally:
        if owns:
            view.close()


def read_camera_intrinsics(source: str | Path) -> dict[str, Any] | None:
    """Read common Pupil scene-camera JSON representations."""
    with SourceView(source) as view:
        candidates = [
            n for n in view.names() if PurePosixPath(n).name.lower() in {"scene_camera.json", "scene-camera.json", "camera.json"}
        ]
        for name in candidates:
            try:
                data = view.read_json(name)
            except Exception:
                continue
            matrix = data.get("camera_matrix") or data.get("matrix") or data.get("intrinsics")
            distortion = data.get("distortion_coefficients") or data.get("distortion") or []
            resolution = data.get("resolution") or [data.get("width"), data.get("height")]
            if isinstance(matrix, dict):
                fx, fy = matrix.get("fx"), matrix.get("fy")
                cx, cy = matrix.get("cx"), matrix.get("cy")
                if all(v is not None for v in (fx, fy, cx, cy)):
                    matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            if matrix is not None:
                return {"matrix": matrix, "distortion": distortion, "resolution": resolution, "source_member": name}
    return None


def materialized_native_files(root: Path, recording_id: str | None = None) -> tuple[Path, Path, Path, dict]:
    candidates = find_native_candidates(root)
    if recording_id:
        matches = [c for c in candidates if c.recording_id == recording_id]
        if matches:
            candidates = matches
    if len(candidates) != 1:
        raise AssetError(f"Expected one Native recording, found {len(candidates)}")
    candidate = candidates[0]
    return root / candidate.video_name, root / candidate.time_name, root / candidate.info_name, candidate.info


def materialized_cloud_root(root: Path, recording_id: str | None = None) -> Path:
    candidates = find_cloud_candidates(root)
    if recording_id:
        matches = [c for c in candidates if c.recording_id == recording_id]
        if matches:
            candidates = matches
    if len(candidates) != 1:
        raise AssetError(f"Expected one Timeseries recording, found {len(candidates)}")
    return root / candidates[0].root
