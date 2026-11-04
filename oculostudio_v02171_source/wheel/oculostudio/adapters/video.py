from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oculostudio.errors import ProcessingError
from oculostudio.util import find_executable


def probe_video(path: str | Path, ffprobe_path: str | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        ffprobe = find_executable("ffprobe", ffprobe_path)
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
        data = json.loads(subprocess.check_output(command, text=True, encoding="utf-8", errors="replace"))
        video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

        def parse_rate(value: str | None) -> float | None:
            if not value or value in {"0/0", "N/A"}:
                return None
            try:
                left, right = value.split("/", 1)
                return float(left) / float(right)
            except Exception:
                return None

        duration = video.get("duration") or data.get("format", {}).get("duration")
        return {
            "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
            "duration_sec": float(duration) if duration not in (None, "N/A") else None,
            "nominal_fps": parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
            "time_base": video.get("time_base"),
            "frame_count": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None,
            "video_codec": video.get("codec_name"),
            "has_audio": audio is not None,
            "audio_codec": audio.get("codec_name") if audio else None,
            "probe_backend": "ffprobe",
        }
    except Exception:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ProcessingError(f"Cannot open video: {path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "duration_sec": count / fps if fps > 0 else None,
                "nominal_fps": fps or None,
                "time_base": None,
                "frame_count": count or None,
                "video_codec": None,
                "has_audio": None,
                "audio_codec": None,
                "probe_backend": "opencv",
            }
        finally:
            capture.release()


def frame_timestamps_ns(path: str | Path, ffprobe_path: str | None = None) -> np.ndarray:
    ffprobe = find_executable("ffprobe", ffprobe_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,pts_time",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True, encoding="utf-8", errors="replace"))
    values: list[int] = []
    for frame in data.get("frames", []):
        raw = frame.get("best_effort_timestamp_time") or frame.get("pts_time") or frame.get("pkt_pts_time")
        if raw is None:
            continue
        values.append(int(round(float(raw) * 1_000_000_000)))
    if not values:
        raise ProcessingError(f"No video frame timestamps were found in {path}")
    return np.asarray(values, dtype=np.int64)


def make_thumbnail(path: str | Path, output: str | Path, *, at_sec: float = 0.0, width: int = 1280) -> Path:
    path = Path(path)
    output = Path(output)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProcessingError(f"Cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, at_sec) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise ProcessingError(f"Cannot decode a thumbnail from {path}")
        if frame.shape[1] > width:
            scale = width / frame.shape[1]
            frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))))
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame):
            raise ProcessingError(f"Cannot write thumbnail: {output}")
        return output
    finally:
        capture.release()
