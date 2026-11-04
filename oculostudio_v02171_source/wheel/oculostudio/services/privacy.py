from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import time
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache, SourceView
from oculostudio.adapters.pupil import find_facemapper_files, materialized_native_files
from oculostudio.adapters.video import probe_video
from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, find_executable, new_id, safe_name, utc_now


PRIVACY_FRAME_COVERAGE_ENGINE = "native-trim-display-overlap-v2"


@dataclass(frozen=True)
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "facemapper"
    track_id: int | None = None
    confidence: float | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def clipped(self, width: int, height: int, padding_percent: float = 0.0) -> "FaceBox":
        px = self.width * padding_percent / 200.0
        py = self.height * padding_percent / 200.0
        return replace(
            self,
            x1=max(0.0, self.x1 - px),
            y1=max(0.0, self.y1 - py),
            x2=min(float(width), self.x2 + px),
            y2=min(float(height), self.y2 + py),
        )


@dataclass
class FaceMapping:
    raw_by_frame: dict[int, list[FaceBox]]
    applied_by_frame: dict[int, list[FaceBox]]
    section_intervals_ns: list[tuple[int, int]]
    matched_timestamps: int
    unmatched_timestamps: int
    invalid_rows: int
    match_errors_ns: list[int]
    interpolated_count: int
    track_count: int
    source_file: str
    internal_repair_count: int = 0
    edge_repair_count: int = 0


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _column(columns, aliases, *, required=True):
    mapping = {_normalize(c): c for c in columns}
    for alias in aliases:
        if _normalize(alias) in mapping:
            return mapping[_normalize(alias)]
    if required:
        raise ValidationError(f"Face table is missing a required column: {aliases[0]}")
    return None


def _nearest_index(values: np.ndarray, query: int) -> tuple[int, int]:
    pos = int(np.searchsorted(values, query, side="left"))
    candidates = []
    if pos < len(values):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return -1, 2**63 - 1
    index = min(candidates, key=lambda i: abs(int(values[i]) - query))
    return index, abs(int(values[index]) - query)


def _iou(left: FaceBox, right: FaceBox) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(left: FaceBox, right: FaceBox) -> float:
    lx, ly = left.center
    rx, ry = right.center
    scale = max(1.0, 0.5 * (left.width + right.width + left.height + right.height))
    return math.hypot(lx - rx, ly - ry) / scale


def _scale_ratio(left: FaceBox, right: FaceBox) -> float:
    """Return a symmetric face-box scale ratio.

    Width and height are considered independently so a badly stretched box is
    rejected even when its area happens to remain similar.
    """

    ratios = (
        left.width / max(right.width, 1e-9),
        right.width / max(left.width, 1e-9),
        left.height / max(right.height, 1e-9),
        right.height / max(left.height, 1e-9),
    )
    return float(max(ratios))


def _repair_transition_is_plausible(
    left_frame: int,
    left_box: FaceBox,
    right_frame: int,
    right_box: FaceBox,
    *,
    maximum_center_shift_per_frame: float,
    maximum_scale_ratio: float,
) -> bool:
    frame_delta = int(right_frame) - int(left_frame)
    if frame_delta <= 0:
        return False
    if left_box.width <= 1 or left_box.height <= 1 or right_box.width <= 1 or right_box.height <= 1:
        return False
    motion_per_frame = _center_distance(left_box, right_box) / frame_delta
    return bool(
        motion_per_frame <= maximum_center_shift_per_frame
        and _scale_ratio(left_box, right_box) <= maximum_scale_ratio
    )


def _interpolate_box(
    left_box: FaceBox,
    right_box: FaceBox,
    alpha: float,
    *,
    source: str,
    track_id: int,
) -> FaceBox:
    return FaceBox(
        x1=(1 - alpha) * left_box.x1 + alpha * right_box.x1,
        y1=(1 - alpha) * left_box.y1 + alpha * right_box.y1,
        x2=(1 - alpha) * left_box.x2 + alpha * right_box.x2,
        y2=(1 - alpha) * left_box.y2 + alpha * right_box.y2,
        source=source,
        track_id=track_id,
        confidence=None,
    )


def _extrapolate_box(
    anchor_frame: int,
    anchor_box: FaceBox,
    neighbour_frame: int,
    neighbour_box: FaceBox,
    target_frame: int,
    *,
    source: str,
    track_id: int,
) -> FaceBox | None:
    delta = int(neighbour_frame) - int(anchor_frame)
    if delta == 0:
        return None
    factor = (int(target_frame) - int(anchor_frame)) / delta
    return FaceBox(
        x1=anchor_box.x1 + factor * (neighbour_box.x1 - anchor_box.x1),
        y1=anchor_box.y1 + factor * (neighbour_box.y1 - anchor_box.y1),
        x2=anchor_box.x2 + factor * (neighbour_box.x2 - anchor_box.x2),
        y2=anchor_box.y2 + factor * (neighbour_box.y2 - anchor_box.y2),
        source=source,
        track_id=track_id,
        confidence=None,
    )


def assign_tracks_and_fill_gaps(
    boxes_by_frame: dict[int, list[FaceBox]],
    *,
    gap_fill_frames: int = 3,
    edge_fill_frames: int = 0,
    iou_threshold: float = 0.05,
    center_distance_threshold: float = 1.8,
    maximum_center_shift_per_frame: float = 0.85,
    maximum_scale_ratio: float = 1.75,
    minimum_track_observations_for_edge_repair: int = 3,
    frame_min: int = 0,
    frame_max: int | None = None,
) -> tuple[dict[int, list[FaceBox]], int, int, int]:
    """Assign accepted detections to tracks and repair very short drop-outs.

    The repair stage is deliberately independent of FaceMapper ``review``
    candidates: callers that want accepted-only repair pass only the
    analysis-strict detections. Internal gaps are interpolated only when the
    two accepted endpoints have plausible motion and scale. Optional leading
    and trailing repair uses constant-velocity extrapolation for at most a few
    frames and requires a stable accepted track.
    """

    gap_fill_frames = max(0, int(gap_fill_frames))
    edge_fill_frames = max(0, int(edge_fill_frames))
    if maximum_center_shift_per_frame <= 0:
        raise ValidationError("maximum_center_shift_per_frame must be positive")
    if maximum_scale_ratio < 1:
        raise ValidationError("maximum_scale_ratio must be at least 1")
    observations: dict[int, list[tuple[int, FaceBox]]] = defaultdict(list)
    last: dict[int, tuple[int, FaceBox]] = {}
    next_track = 0
    assigned: dict[int, list[FaceBox]] = defaultdict(list)
    for frame_index in sorted(boxes_by_frame):
        candidates = boxes_by_frame[frame_index]
        active = {
            track_id: value
            for track_id, value in last.items()
            if frame_index - value[0] <= gap_fill_frames + 1
        }
        possible: list[tuple[float, int, int]] = []
        for box_index, box in enumerate(candidates):
            for track_id, (previous_frame, previous) in active.items():
                overlap = _iou(box, previous)
                distance = _center_distance(box, previous)
                frame_delta = max(1, int(frame_index) - int(previous_frame))
                # A fixed association radius fragments the very tracks this
                # feature is meant to repair when the head moves quickly over
                # a two- or three-frame dropout.  Allow exactly the same
                # per-frame motion budget used by the repair plausibility
                # gate, while retaining the legacy radius for adjacent frames.
                allowed_distance = max(
                    float(center_distance_threshold),
                    float(maximum_center_shift_per_frame) * frame_delta,
                )
                scale_is_plausible = _scale_ratio(box, previous) <= maximum_scale_ratio
                if scale_is_plausible and (overlap >= iou_threshold or distance <= allowed_distance):
                    distance_per_frame = distance / frame_delta
                    possible.append(
                        ((1.0 - overlap) + 0.35 * distance_per_frame, box_index, track_id)
                    )
        used_boxes: set[int] = set()
        used_tracks: set[int] = set()
        box_to_track: dict[int, int] = {}
        for _, box_index, track_id in sorted(possible):
            if box_index in used_boxes or track_id in used_tracks:
                continue
            used_boxes.add(box_index)
            used_tracks.add(track_id)
            box_to_track[box_index] = track_id
        for box_index, box in enumerate(candidates):
            track_id = box_to_track.get(box_index)
            if track_id is None:
                track_id = next_track
                next_track += 1
            tracked = replace(box, track_id=track_id)
            assigned[frame_index].append(tracked)
            observations[track_id].append((frame_index, tracked))
            last[track_id] = (frame_index, tracked)

    internal_repaired = 0
    for track_id, rows in observations.items():
        rows = sorted(rows)
        for (left_frame, left_box), (right_frame, right_box) in zip(rows, rows[1:]):
            gap = right_frame - left_frame - 1
            if gap <= 0 or gap > gap_fill_frames:
                continue
            if not _repair_transition_is_plausible(
                left_frame,
                left_box,
                right_frame,
                right_box,
                maximum_center_shift_per_frame=maximum_center_shift_per_frame,
                maximum_scale_ratio=maximum_scale_ratio,
            ):
                continue
            for offset in range(1, gap + 1):
                alpha = offset / (gap + 1)
                box = _interpolate_box(
                    left_box,
                    right_box,
                    alpha,
                    source="accepted_gap_repair",
                    track_id=track_id,
                )
                assigned[left_frame + offset].append(box)
                internal_repaired += 1

    edge_repaired = 0
    if edge_fill_frames > 0:
        if frame_max is None:
            frame_max = max(boxes_by_frame, default=frame_min)
        for track_id, rows in observations.items():
            rows = sorted(rows)
            if len(rows) < max(2, int(minimum_track_observations_for_edge_repair)):
                continue

            # Leading edge: use the first two accepted observations to predict
            # at most a couple of immediately preceding frames.
            first_frame, first_box = rows[0]
            second_frame, second_box = rows[1]
            if _repair_transition_is_plausible(
                first_frame,
                first_box,
                second_frame,
                second_box,
                maximum_center_shift_per_frame=maximum_center_shift_per_frame,
                maximum_scale_ratio=maximum_scale_ratio,
            ):
                for offset in range(1, min(edge_fill_frames, first_frame - frame_min) + 1):
                    target = first_frame - offset
                    box = _extrapolate_box(
                        first_frame,
                        first_box,
                        second_frame,
                        second_box,
                        target,
                        source="accepted_edge_repair",
                        track_id=track_id,
                    )
                    if box is None or box.width <= 1 or box.height <= 1:
                        continue
                    if any(_iou(box, existing) >= 0.75 for existing in assigned.get(target, [])):
                        continue
                    assigned[target].append(box)
                    edge_repaired += 1

            # Trailing edge: use the last two accepted observations to predict
            # immediately following frames. The same motion/scale gates apply.
            previous_frame, previous_box = rows[-2]
            last_frame, last_box = rows[-1]
            if _repair_transition_is_plausible(
                previous_frame,
                previous_box,
                last_frame,
                last_box,
                maximum_center_shift_per_frame=maximum_center_shift_per_frame,
                maximum_scale_ratio=maximum_scale_ratio,
            ):
                for offset in range(1, min(edge_fill_frames, int(frame_max) - last_frame) + 1):
                    target = last_frame + offset
                    box = _extrapolate_box(
                        last_frame,
                        last_box,
                        previous_frame,
                        previous_box,
                        target,
                        source="accepted_edge_repair",
                        track_id=track_id,
                    )
                    if box is None or box.width <= 1 or box.height <= 1:
                        continue
                    if any(_iou(box, existing) >= 0.75 for existing in assigned.get(target, [])):
                        continue
                    assigned[target].append(box)
                    edge_repaired += 1

    return (
        {frame: sorted(boxes, key=lambda b: (b.track_id or -1, b.x1)) for frame, boxes in assigned.items()},
        next_track,
        internal_repaired,
        edge_repaired,
    )


def _read_sections(root: Path, recording_id: str | None) -> list[tuple[int, int]]:
    candidates = list(root.rglob("sections.csv"))
    if not candidates:
        return []
    frame = pd.read_csv(candidates[0], low_memory=False)
    rid = _column(frame.columns, ["recording id", "recording_id"], required=False)
    start = _column(frame.columns, ["section start time [ns]", "section_start_time_ns", "start timestamp [ns]"], required=False)
    end = _column(frame.columns, ["section end time [ns]", "section_end_time_ns", "end timestamp [ns]"], required=False)
    if start is None or end is None:
        return []
    if rid and recording_id:
        frame = frame.loc[frame[rid].astype(str) == str(recording_id)]
    intervals: list[tuple[int, int]] = []
    for _, row in frame.iterrows():
        try:
            left, right = int(row[start]), int(row[end])
        except Exception:
            continue
        if right > left:
            intervals.append((left, right))
    merged: list[tuple[int, int]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def load_face_mapping(
    root: Path,
    *,
    recording_id: str | None,
    frame_abs_ns: np.ndarray,
    width: int,
    height: int,
    tolerance_ms: float = 5.0,
    gap_fill_frames: int = 3,
    edge_fill_frames: int = 0,
    repair_maximum_center_shift_per_frame: float = 0.85,
    repair_maximum_scale_ratio: float = 1.75,
    repair_minimum_track_observations: int = 3,
) -> FaceMapping:
    files = find_facemapper_files(root)
    if not files["detections"]:
        raise ValidationError("Face Mapper source contains no face_detections.csv or face_positions.csv")
    source = root / files["detections"][0]
    frame = pd.read_csv(source, low_memory=False)
    rid_col = _column(frame.columns, ["recording id", "recording_id"], required=False)
    timestamp_col = _column(frame.columns, ["timestamp [ns]", "timestamp_ns"])
    x1_col = _column(frame.columns, ["p1 x [px]", "p1_x_px", "x1", "left"])
    y1_col = _column(frame.columns, ["p1 y [px]", "p1_y_px", "y1", "top"])
    x2_col = _column(frame.columns, ["p2 x [px]", "p2_x_px", "x2", "right"])
    y2_col = _column(frame.columns, ["p2 y [px]", "p2_y_px", "y2", "bottom"])
    confidence_col = _column(frame.columns, ["confidence", "score", "face confidence"], required=False)
    if rid_col and recording_id:
        frame = frame.loc[frame[rid_col].astype(str) == str(recording_id)]
    tolerance_ns = int(round(tolerance_ms * 1_000_000))
    grouped: dict[int, list[FaceBox]] = defaultdict(list)
    matched_timestamps: set[int] = set()
    unmatched_timestamps: set[int] = set()
    match_errors: list[int] = []
    invalid = 0
    for _, row in frame.iterrows():
        try:
            timestamp = int(row[timestamp_col])
            coords = [float(row[x1_col]), float(row[y1_col]), float(row[x2_col]), float(row[y2_col])]
        except Exception:
            invalid += 1
            continue
        if max(abs(v) for v in coords) <= 2.0:
            coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
        x1, y1, x2, y2 = coords
        if not all(np.isfinite(coords)) or x2 <= x1 or y2 <= y1:
            invalid += 1
            continue
        index, error = _nearest_index(frame_abs_ns, timestamp)
        if index < 0 or error > tolerance_ns:
            unmatched_timestamps.add(timestamp)
            continue
        confidence = None
        if confidence_col is not None:
            try:
                confidence = float(row[confidence_col])
            except Exception:
                pass
        grouped[index].append(FaceBox(x1, y1, x2, y2, confidence=confidence))
        matched_timestamps.add(timestamp)
        match_errors.append(error)
    applied, tracks, internal_repaired, edge_repaired = assign_tracks_and_fill_gaps(
        grouped,
        gap_fill_frames=gap_fill_frames,
        edge_fill_frames=edge_fill_frames,
        maximum_center_shift_per_frame=repair_maximum_center_shift_per_frame,
        maximum_scale_ratio=repair_maximum_scale_ratio,
        minimum_track_observations_for_edge_repair=repair_minimum_track_observations,
        frame_min=0,
        frame_max=max(0, len(frame_abs_ns) - 1),
    )
    return FaceMapping(
        raw_by_frame=dict(grouped),
        applied_by_frame=applied,
        section_intervals_ns=_read_sections(root, recording_id),
        matched_timestamps=len(matched_timestamps),
        unmatched_timestamps=len(unmatched_timestamps),
        invalid_rows=invalid,
        match_errors_ns=match_errors,
        interpolated_count=internal_repaired + edge_repaired,
        track_count=tracks,
        source_file=str(source),
        internal_repair_count=internal_repaired,
        edge_repair_count=edge_repaired,
    )



def _apply_box(frame: np.ndarray, box: FaceBox, *, mode: str, strength: int, shape: str) -> None:
    height, width = frame.shape[:2]
    clipped = box.clipped(width, height)
    x1, y1, x2, y2 = [int(round(value)) for value in (clipped.x1, clipped.y1, clipped.x2, clipped.y2)]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if mode == "gaussian":
        kernel = max(3, int(strength) * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        transformed = cv2.GaussianBlur(region, (kernel, kernel), sigmaX=0)
    elif mode == "pixelate":
        block = max(2, int(strength))
        small_width = max(1, region.shape[1] // block)
        small_height = max(1, region.shape[0] // block)
        small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        transformed = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    elif mode == "solid":
        transformed = np.zeros_like(region)
    else:
        raise ValidationError(f"Unsupported privacy mode: {mode}")
    if shape == "rectangle":
        region[:] = transformed
    elif shape == "ellipse":
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        center = (region.shape[1] // 2, region.shape[0] // 2)
        axes = (max(1, region.shape[1] // 2), max(1, region.shape[0] // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        region[:] = np.rint(alpha * transformed + (1.0 - alpha) * region).astype(np.uint8)
    else:
        raise ValidationError(f"Unsupported mask shape: {shape}")


def apply_privacy_masks(
    frame: np.ndarray,
    boxes: list[FaceBox],
    *,
    mode: str,
    strength: int,
    shape: str,
    padding_percent: float,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for box in boxes:
        _apply_box(
            output,
            box.clipped(width, height, padding_percent),
            mode=mode,
            strength=strength,
            shape=shape,
        )
    return output


class HaarFaceVerifier:
    def __init__(self):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ProcessingError("OpenCV Haar face detector could not be loaded")

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(36, 36),
        )
        return [
            FaceBox(float(x), float(y), float(x + w), float(y + h), source="opencv_haar")
            for x, y, w, h in detections
        ]


def _merge_boxes(primary: list[FaceBox], secondary: list[FaceBox], *, duplicate_iou: float = 0.35) -> list[FaceBox]:
    result = list(primary)
    for candidate in secondary:
        if any(_iou(candidate, existing) >= duplicate_iou for existing in result):
            continue
        result.append(candidate)
    return result


def _output_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise ProcessingError("OpenCV could not create a temporary video writer")


def _mux_audio_and_transcode(
    silent_video: Path,
    source_video: Path,
    output_video: Path,
    *,
    source_start_sec: float,
    duration_sec: float,
    preserve_audio: bool,
    crf: int,
    preset: str,
    ffmpeg_path: str | None = None,
    expected_frames: int | None = None,
    cancellation: CancellationToken | None = None,
    progress=None,
) -> list[str]:
    cancellation = cancellation or CancellationToken.create()
    cancellation.raise_if_cancelled()
    ffmpeg = ffmpeg_path or find_executable("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video),
    ]
    if preserve_audio:
        command += [
            "-ss",
            f"{source_start_sec:.9f}",
            "-t",
            f"{duration_sec:.9f}",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
        ]
    else:
        command += ["-map", "0:v:0"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
    ]
    if preserve_audio:
        # Audio input already has a bounded -t. -shortest can discard the last
        # masked frame when audio is a few milliseconds shorter than video.
        command += ["-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    # Never expose a partially encoded file under the final video name.
    pending = output_video.with_name(output_video.stem + ".encoding" + output_video.suffix)
    command += ["-movflags", "+faststart", str(pending)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr, creationflags=flags)
        started = time.monotonic()
        next_notice = started + 2
        try:
            while process.poll() is None:
                cancellation.raise_if_cancelled()
                if time.monotonic() - started > 7200:
                    raise ProcessingError("Privacy encoding timed out after two hours")
                if time.monotonic() >= next_notice:
                    emit(progress, 0.84, f"Encoding with verified FFmpeg · {int(time.monotonic()-started)} s elapsed")
                    next_notice += 2
                time.sleep(0.1)
            cancellation.raise_if_cancelled()
            if process.returncode != 0:
                stderr.seek(max(0, stderr.tell() - 12000))
                detail = stderr.read().decode("utf-8", errors="replace")
                raise ProcessingError(f"FFmpeg privacy-video encoding failed: {detail}")
            if not pending.is_file() or pending.stat().st_size == 0:
                raise ProcessingError("FFmpeg returned no encoded video")
            if expected_frames is not None:
                check = cv2.VideoCapture(str(pending))
                try:
                    encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if not check.isOpened() or encoded_frames != expected_frames:
                        raise ProcessingError(f"Encoded frame count {encoded_frames} differs from masked-frame audit {expected_frames}; no completed artifact will be published")
                finally:
                    check.release()
            os.replace(pending, output_video)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            pending.unlink(missing_ok=True)
    return command


def _is_within_sections(timestamp_ns: int, intervals: list[tuple[int, int]]) -> bool | None:
    if not intervals:
        return None
    return any(left <= timestamp_ns < right for left, right in intervals)


def _native_trim_frame_interval(trim_artifact, frame_count: int) -> dict[str, Any]:
    """Return the exact contiguous Native-frame coverage recorded by a trim.

    Native-reference trims use display-interval overlap semantics.  Converting
    their UTC boundaries back with ``searchsorted`` drops the leading frame
    whenever the requested start lies inside a displayed frame.  The exported
    ``native_frame_map.csv`` is therefore the authoritative source here.
    """
    try:
        frame_map_path = find_artifact_file(trim_artifact, names=["native_frame_map.csv"])
        frame_map = pd.read_csv(frame_map_path, low_memory=False)
    except Exception as exc:
        raise ValidationError(
            "The Native-reference trim has no readable native_frame_map.csv; re-export the trim"
        ) from exc
    if "native_frame_index" not in frame_map.columns:
        raise ValidationError("Native-reference trim frame map is missing native_frame_index")
    try:
        indices = pd.to_numeric(frame_map["native_frame_index"], errors="raise").astype("int64").to_numpy()
    except Exception as exc:
        raise ValidationError("Native-reference trim frame indices are invalid") from exc
    if not len(indices):
        raise ValidationError("Native-reference trim contains no retained Native frames")
    if np.any(np.diff(indices) != 1):
        raise ValidationError("Native-reference trim frame map must be strictly contiguous")
    start_frame = int(indices[0])
    end_frame = int(indices[-1]) + 1
    if start_frame < 0 or end_frame > int(frame_count):
        raise ValidationError(
            "Native-reference trim frame map lies outside the selected Native scene video"
        )
    boundary_frames = 0
    if "boundary_frame" in frame_map.columns:
        boundary_frames = int(
            frame_map["boundary_frame"].astype(str).str.strip().str.lower().eq("true").sum()
        )
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": int(end_frame - start_frame),
        "boundary_frame_count": boundary_frames,
        "frame_map": str(frame_map_path),
        "policy": "native_reference_trim_display_overlap",
    }


def create_privacy_video(
    store: ProjectStore,
    *,
    session_id: str,
    native_asset_id: str,
    facemapper_asset_id: str | None = None,
    facemapper_filter_artifact_id: str | None = None,
    native_trim_artifact_id: str | None = None,
    clip_id: str | None = None,
    mode: Literal["gaussian", "pixelate", "solid"] = "gaussian",
    strength: int = 30,
    padding_percent: float = 0.0,
    shape: Literal["rectangle", "ellipse"] = "rectangle",
    gap_fill_frames: int = 3,
    edge_fill_frames: int = 1,
    repair_maximum_center_shift_per_frame: float = 0.85,
    repair_maximum_scale_ratio: float = 1.75,
    repair_minimum_track_observations: int = 3,
    timestamp_tolerance_ms: float = 5.0,
    detector_mode: Literal["facemapper", "haar", "hybrid"] = "facemapper",
    candidate_policy: Literal["accepted_review", "accepted_only", "accepted_repair"] = "accepted_review",
    exact_filter_match: bool = False,
    coverage_policy: Literal["strict", "allow"] = "strict",
    preserve_audio: bool = True,
    crf: int = 20,
    preset: str = "fast",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    if mode not in {"gaussian", "pixelate", "solid"}:
        raise ValidationError("Privacy mode must be gaussian, pixelate, or solid")
    if shape not in {"rectangle", "ellipse"}:
        raise ValidationError("Mask shape must be rectangle or ellipse")
    if candidate_policy not in {"accepted_review", "accepted_only", "accepted_repair"}:
        raise ValidationError(
            "candidate_policy must be accepted_review, accepted_only, or accepted_repair"
        )
    if exact_filter_match:
        candidate_policy = "accepted_only"
        detector_mode = "facemapper"
        gap_fill_frames = 0
        edge_fill_frames = 0
        padding_percent = 0.0
    elif candidate_policy == "accepted_repair":
        detector_mode = "facemapper"
        if not facemapper_filter_artifact_id:
            raise ValidationError(
                "Accepted-only temporal repair requires a FaceMapper QC/filter artifact"
            )
    if detector_mode in {"facemapper", "hybrid"} and not (facemapper_asset_id or facemapper_filter_artifact_id):
        raise ValidationError("Face Mapper or FaceMapper QC/filter input is required for the selected detector mode")
    # Validate the encoder BEFORE materializing a large ZIP or masking frames.
    cancellation.raise_if_cancelled()
    emit(progress, 0.01, "Checking FFmpeg and H.264/AAC encoder")
    encoder_info = privacy_encoder_preflight(ffmpeg_path, preserve_audio=preserve_audio,
                                            preset=preset, crf=crf, ffprobe_path=ffprobe_path)
    verified_ffmpeg = encoder_info["ffmpeg"]["path"]
    cancellation.raise_if_cancelled()
    emit(progress, 0.03, "Encoder verified; loading Native recording")
    session = store.project.sessions[session_id]
    native_asset = session.assets[native_asset_id]
    cache = MaterializationCache(store.cache_dir())
    native_root = cache.materialize(native_asset)
    source_video, scene_time_path, _, _ = materialized_native_files(native_root, native_asset.recording_id)
    timestamps = np.fromfile(scene_time_path, dtype="<i8").astype(np.int64, copy=False)
    if len(timestamps) == 0:
        raise ProcessingError("Native scene timestamp file is empty")
    video_meta = probe_video(source_video, ffprobe_path=ffprobe_path)
    width, height = int(video_meta["width"]), int(video_meta["height"])
    if width <= 0 or height <= 0:
        raise ProcessingError("Scene video dimensions are unavailable")
    frame_count = min(len(timestamps), int(video_meta.get("frame_count") or len(timestamps)))
    timestamps = timestamps[:frame_count]
    mapping = FaceMapping({}, {}, [], 0, 0, 0, [], 0, 0, "")
    source_asset_ids = [native_asset_id]
    parent_artifact_ids: list[str] = []
    face_source_kind = "none"
    resolved_native_trim_artifact_id = native_trim_artifact_id
    if facemapper_filter_artifact_id:
        filter_artifact = session.artifacts.get(facemapper_filter_artifact_id)
        if filter_artifact is None or filter_artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
            raise ValidationError("Select a FaceMapper QC/filter artifact")
        filter_trim_id = str(filter_artifact.parameters.get("native_trim_artifact_id") or "").strip() or None
        if resolved_native_trim_artifact_id and filter_trim_id and resolved_native_trim_artifact_id != filter_trim_id:
            raise ValidationError(
                "The selected FaceMapper filter belongs to a different Native-reference trim"
            )
        if resolved_native_trim_artifact_id is None:
            resolved_native_trim_artifact_id = filter_trim_id
        from oculostudio.services.facemapper_filter import filtered_facemapper_root
        filter_variant = (
            "analysis_strict"
            if candidate_policy in {"accepted_only", "accepted_repair"}
            else "privacy_safe"
        )
        facemapper_root = filtered_facemapper_root(filter_artifact, variant=filter_variant)
        mapping = load_face_mapping(
            facemapper_root,
            recording_id=native_asset.recording_id,
            frame_abs_ns=timestamps,
            width=width,
            height=height,
            tolerance_ms=timestamp_tolerance_ms,
            gap_fill_frames=gap_fill_frames,
            edge_fill_frames=edge_fill_frames if candidate_policy == "accepted_repair" else 0,
            repair_maximum_center_shift_per_frame=repair_maximum_center_shift_per_frame,
            repair_maximum_scale_ratio=repair_maximum_scale_ratio,
            repair_minimum_track_observations=repair_minimum_track_observations,
        )
        parent_artifact_ids.append(facemapper_filter_artifact_id)
        face_source_kind = f"filtered_{filter_variant}"
    elif facemapper_asset_id:
        facemapper_asset = session.assets[facemapper_asset_id]
        facemapper_root = cache.materialize(facemapper_asset)
        mapping = load_face_mapping(
            facemapper_root,
            recording_id=native_asset.recording_id or facemapper_asset.recording_id,
            frame_abs_ns=timestamps,
            width=width,
            height=height,
            tolerance_ms=timestamp_tolerance_ms,
            gap_fill_frames=gap_fill_frames,
            edge_fill_frames=0,
            repair_maximum_center_shift_per_frame=repair_maximum_center_shift_per_frame,
            repair_maximum_scale_ratio=repair_maximum_scale_ratio,
            repair_minimum_track_observations=repair_minimum_track_observations,
        )
        source_asset_ids.append(facemapper_asset_id)
        face_source_kind = "raw_facemapper"
    if resolved_native_trim_artifact_id and clip_id:
        raise ValidationError("Choose either a Native-reference trim or a Global-Time clip, not both")
    trim_manifest: dict[str, Any] | None = None
    trim_frame_coverage: dict[str, Any] | None = None
    if resolved_native_trim_artifact_id:
        trim_artifact = session.artifacts.get(resolved_native_trim_artifact_id)
        if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
            raise ValidationError("Select a Native-reference trim artifact")
        if native_asset_id not in trim_artifact.source_asset_ids:
            raise ValidationError("The selected Native-reference trim belongs to another Native recording")
        try:
            trim_manifest_path = find_artifact_file(trim_artifact, names=["trim_manifest.json"])
            trim_manifest = json.loads(trim_manifest_path.read_text("utf-8"))
            trim_window = trim_manifest.get("window") or {}
            start_ns = int(trim_window["utc_start_ns"])
            end_ns = int(trim_window["utc_end_ns"])
        except Exception as exc:
            raise ValidationError("The Native-reference trim has no valid absolute-time window") from exc
        trim_frame_coverage = _native_trim_frame_interval(trim_artifact, frame_count)
        clip_label = trim_artifact.label
        if resolved_native_trim_artifact_id not in parent_artifact_ids:
            parent_artifact_ids.append(resolved_native_trim_artifact_id)
    elif clip_id:
        clip = session.clips[clip_id]
        if not native_asset.primary_time_domain_id:
            raise ValidationError("Native asset has no Pupil clock time domain")
        graph = TimeGraph.from_session(session)
        start_ns = int(graph.convert(session.global_time_domain_id, native_asset.primary_time_domain_id, clip.start_global_ns))
        end_ns = int(graph.convert(session.global_time_domain_id, native_asset.primary_time_domain_id, clip.end_global_ns))
        clip_label = clip.label
    else:
        start_ns = int(timestamps[0])
        median_interval = int(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 33_333_333
        end_ns = int(timestamps[-1] + median_interval)
        clip_label = "full_recording"
    if trim_frame_coverage is not None:
        start_frame = int(trim_frame_coverage["start_frame"])
        end_frame = int(trim_frame_coverage["end_frame"])
        frame_selection_policy = "native_reference_trim_display_overlap"
    else:
        start_frame = int(np.searchsorted(timestamps, start_ns, side="left"))
        end_frame = int(np.searchsorted(timestamps, end_ns, side="left"))
        end_frame = min(end_frame, frame_count)
        frame_selection_policy = "capture_timestamp_half_open"
    if end_frame <= start_frame:
        raise ValidationError("Privacy interval does not intersect the scene video")
    actual_start_ns = int(timestamps[start_frame])
    interval_ns = int(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 33_333_333
    actual_end_ns = int(timestamps[end_frame]) if end_frame < len(timestamps) else int(timestamps[-1] + interval_ns)
    # A Native trim can start inside the presentation interval of its first
    # frame.  Test the retained part of each boundary frame rather than the
    # frame's earlier capture timestamp; otherwise strict coverage rejects the
    # exact boundary frame that the trim intentionally retained.
    clip_inside_sections = all(
        _is_within_sections(
            min(max(int(timestamps[index]), int(start_ns)), int(end_ns) - 1),
            mapping.section_intervals_ns,
        ) is not False
        for index in range(start_frame, end_frame)
    )
    warnings: list[str] = []
    if candidate_policy == "accepted_repair":
        warnings.append(
            "Accepted-only temporal repair used: review candidates and Haar detections were excluded."
        )
    elif candidate_policy == "accepted_only":
        warnings.append("Accepted-only masking excludes review candidates.")
    if mapping.unmatched_timestamps:
        warnings.append(f"{mapping.unmatched_timestamps} Face Mapper timestamps did not match scene frames")
    if mapping.section_intervals_ns and not clip_inside_sections:
        warnings.append("The requested clip extends outside Face Mapper section coverage")
    blocking_warnings = [
        warning
        for warning in warnings
        if "did not match scene frames" in warning
        or "outside Face Mapper section coverage" in warning
    ]
    if coverage_policy == "strict" and blocking_warnings:
        raise ValidationError(
            "Strict privacy policy blocked processing: " + "; ".join(blocking_warnings)
        )

    artifact_id = new_id("artifact")
    interval_key = resolved_native_trim_artifact_id or clip_id or "full"
    output_root = store.derived_dir(session_id, "privacy", interval_key) / artifact_id / safe_name(native_asset.label)
    output_root.mkdir(parents=True, exist_ok=False)
    temp_video = output_root / "privacy_silent_intermediate.mp4"
    output_video = output_root / f"{safe_name(clip_label)}_{mode}.mp4"
    audit_path = output_root / "privacy_frame_audit.csv"
    report_path = output_root / "privacy_report.json"
    fps = 1e9 / interval_ns if interval_ns > 0 else float(video_meta.get("nominal_fps") or 30.0)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ProcessingError(f"Cannot decode scene video: {source_video}")
    writer = _output_writer(temp_video, width, height, fps)
    verifier = HaarFaceVerifier() if detector_mode in {"haar", "hybrid"} else None
    audit_rows: list[dict[str, Any]] = []
    total = end_frame - start_frame
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for output_index, native_index in enumerate(range(start_frame, end_frame)):
            cancellation.raise_if_cancelled()
            ok, frame = capture.read()
            if not ok or frame is None:
                warnings.append(f"Video decode ended early at native frame {native_index}")
                break
            facemapper_boxes = mapping.applied_by_frame.get(native_index, [])
            haar_boxes = verifier.detect(frame) if verifier else []
            if detector_mode == "facemapper":
                boxes = facemapper_boxes
            elif detector_mode == "haar":
                boxes = haar_boxes
            else:
                boxes = _merge_boxes(facemapper_boxes, haar_boxes)
            processed = apply_privacy_masks(
                frame,
                boxes,
                mode=mode,
                strength=strength,
                shape=shape,
                padding_percent=padding_percent,
            )
            writer.write(processed)
            raw_count = len(mapping.raw_by_frame.get(native_index, []))
            audit_rows.append(
                {
                    "output_frame_index": output_index,
                    "native_frame_index": native_index,
                    "timestamp_ns": int(timestamps[native_index]),
                    "facemapper_raw_face_count": raw_count,
                    "facemapper_applied_face_count": len(facemapper_boxes),
                    "haar_face_count": len(haar_boxes),
                    "total_mask_count": len(boxes),
                    "interpolated_mask_count": sum(box.source == "interpolated" for box in boxes),
                    "within_facemapper_section": _is_within_sections(int(timestamps[native_index]), mapping.section_intervals_ns),
                    "boxes_json": json.dumps(
                        [
                            {
                                "x1": round(box.x1, 3),
                                "y1": round(box.y1, 3),
                                "x2": round(box.x2, 3),
                                "y2": round(box.y2, 3),
                                "source": box.source,
                                "track_id": box.track_id,
                            }
                            for box in boxes
                        ],
                        separators=(",", ":"),
                    ),
                }
            )
            if output_index % 10 == 0:
                emit(progress, 0.08 + 0.72 * (output_index + 1) / total, f"Applying privacy masks ({output_index + 1}/{total})")
    finally:
        capture.release()
        writer.release()
    if not audit_rows:
        raise ProcessingError("No output frames were generated")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    actual_output_frames = len(audit_rows)
    output_duration_sec = actual_output_frames / fps
    source_start_sec = (actual_start_ns - int(timestamps[0])) / 1e9
    emit(progress, 0.84, "Encoding video and preserving audio")
    command = _mux_audio_and_transcode(
        temp_video,
        source_video,
        output_video,
        source_start_sec=source_start_sec,
        duration_sec=output_duration_sec,
        preserve_audio=preserve_audio,
        crf=crf,
        preset=preset,
        ffmpeg_path=verified_ffmpeg,
        expected_frames=actual_output_frames,
        cancellation=cancellation,
        progress=progress,
    )
    temp_video.unlink(missing_ok=True)
    frames_with_masks = sum(row["total_mask_count"] > 0 for row in audit_rows)
    frames_in_section = [row for row in audit_rows if row["within_facemapper_section"] is True]
    frames_in_section_with_masks = sum(row["total_mask_count"] > 0 for row in frames_in_section)
    qc = {
        "output_frame_count": actual_output_frames,
        "frames_with_masks": frames_with_masks,
        "mask_frame_fraction": frames_with_masks / actual_output_frames,
        "frames_in_facemapper_sections": len(frames_in_section),
        "section_frames_with_masks": frames_in_section_with_masks,
        "section_mask_frame_fraction": (
            frames_in_section_with_masks / len(frames_in_section) if frames_in_section else None
        ),
        "matched_detection_timestamps": mapping.matched_timestamps,
        "unmatched_detection_timestamps": mapping.unmatched_timestamps,
        "invalid_detection_rows": mapping.invalid_rows,
        "interpolated_box_count": mapping.interpolated_count,
        "accepted_internal_repair_box_count": mapping.internal_repair_count,
        "accepted_edge_repair_box_count": mapping.edge_repair_count,
        "track_count": mapping.track_count,
        "candidate_policy": candidate_policy,
        "exact_filter_match": bool(exact_filter_match),
        "median_timestamp_match_error_ms": (
            float(np.median(mapping.match_errors_ns) / 1e6) if mapping.match_errors_ns else None
        ),
        "privacy_limit": "Frames without detected boxes may still contain an undetected face; visually review before publication.",
    }
    report = {
        "schema_version": "oculostudio-privacy-report-1.0",
        "source_asset_ids": source_asset_ids,
        "clip_id": clip_id,
        "native_trim_artifact_id": resolved_native_trim_artifact_id,
        "source_video": str(source_video),
        "output_video": str(output_video),
        "requested_interval_ns": [start_ns, end_ns],
        "actual_interval_ns": [actual_start_ns, actual_end_ns],
        "frame_interval": [start_frame, end_frame],
        "frame_coverage": {
            "engine": PRIVACY_FRAME_COVERAGE_ENGINE,
            "selection_policy": frame_selection_policy,
            "native_trim": trim_frame_coverage,
        },
        "options": {
            "mode": mode,
            "strength": strength,
            "padding_percent": padding_percent,
            "shape": shape,
            "gap_fill_frames": gap_fill_frames,
            "edge_fill_frames": edge_fill_frames,
            "repair_maximum_center_shift_per_frame": repair_maximum_center_shift_per_frame,
            "repair_maximum_scale_ratio": repair_maximum_scale_ratio,
            "repair_minimum_track_observations": repair_minimum_track_observations,
            "timestamp_tolerance_ms": timestamp_tolerance_ms,
            "detector_mode": detector_mode,
            "candidate_policy": candidate_policy,
            "exact_filter_match": bool(exact_filter_match),
            "coverage_policy": coverage_policy,
            "preserve_audio": preserve_audio,
            "facemapper_filter_artifact_id": facemapper_filter_artifact_id,
            "native_trim_artifact_id": resolved_native_trim_artifact_id,
            "interval_source": (
                "native_reference_trim" if resolved_native_trim_artifact_id
                else "global_time_clip" if clip_id
                else "full_recording"
            ),
            "face_source_kind": face_source_kind,
            "frame_coverage_engine": PRIVACY_FRAME_COVERAGE_ENGINE,
        },
        "warnings": warnings,
        "qc": qc,
        "ffmpeg_command": command,
        "encoder_preflight": encoder_info,
        "encoded_pending_path_renamed_to": str(output_video),
        "created_at": utc_now(),
    }
    atomic_write_json(report_path, report)
    emit(progress, 0.96, "Registering privacy artifact")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.PRIVACY_VIDEO,
        label=f"{clip_label} — privacy video",
        output_root=output_root,
        source_asset_ids=source_asset_ids,
        parent_artifact_ids=parent_artifact_ids,
        clip_id=clip_id,
        parameters=report["options"],
        qc=qc,
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Privacy video complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "video": str(output_video),
        "audit_csv": str(audit_path),
        "report": report,
    }
