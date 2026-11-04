"""Explicit video-editor input times. Never used to synthesize NRD frame PTS.

Four-field colon timecodes use a declared integer editing timebase, non-drop.
Seconds and fractional seconds retain nanosecond precision. Epoch timestamps
are not accepted here: this module works only in the selected MP4's local time.
"""
from __future__ import annotations

import re
from decimal import Decimal

from oculostudio.errors import ValidationError

SUPPORTED_TIMECODE_FPS = (24, 25, 30, 50, 60)
DEFAULT_TIMECODE_FPS = 30
NS = 1_000_000_000
MAX_NS = (1 << 63) - 1
FRAME_RE = re.compile(r"([0-9]+):([0-9]{2}):([0-9]{2}):([0-9]{2})")
SECONDS_RE = re.compile(r"[0-9]+(?::[0-9]{1,2}){0,2}(?:\.[0-9]{1,9})?")


def validate_timecode_fps(fps: int) -> int:
    if type(fps) is not int or fps not in SUPPORTED_TIMECODE_FPS:
        raise ValidationError(
            "Timecode FPS must be an integer: 24, 25, 30, 50 or 60. "
            "29.97/59.94 and drop-frame notation are not supported in this release."
        )
    return fps


def format_input_time(ns: int) -> str:
    whole, fraction = divmod(int(ns), NS)
    hours, rest = divmod(whole, 3600)
    minutes, seconds = divmod(rest, 60)
    digits = f"{fraction:09d}".rstrip("0").ljust(3, "0")
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{digits}"


def parse_editor_time(value: str, *, timecode_fps: int = DEFAULT_TIMECODE_FPS,
                      positive: bool = False) -> dict:
    """Return a JSON-safe interpretation; nanoseconds are decimal strings.

    Only the FF field is converted using timecode_fps. A value such as 49.21
    is still 49.21 seconds, never 49 seconds + 21 frames. Fractions of a
    nanosecond in FF/fps are rounded to nearest integer (half up).
    """
    fps = validate_timecode_fps(timecode_fps)
    if not isinstance(value, str):
        raise ValidationError("Time input must be a string, not a numeric timestamp")
    text = value.strip()
    if not text or len(text) > 64:
        raise ValidationError("Enter a video start / duration (maximum 64 characters)")
    if ";" in text:
        raise ValidationError("Drop-frame ';' timecodes are not supported. Use integer-FPS HH:MM:SS:FF or seconds.")
    match = FRAME_RE.fullmatch(text)
    frames = None
    if match:
        hours, minutes, seconds, frames = map(int, match.groups())
        if minutes >= 60 or seconds >= 60:
            raise ValidationError("Minutes and seconds in HH:MM:SS:FF must be 00–59")
        if frames >= fps:
            raise ValidationError(f"At {fps} fps the last frame field must be 00–{fps-1:02d}; received {frames:02d}")
        # Pure integer division, no binary floating point or average source FPS.
        n = (hours * 3600 + minutes * 60 + seconds) * NS + (2 * frames * NS + fps) // (2 * fps)
        notation = "HH:MM:SS:FF"
    elif SECONDS_RE.fullmatch(text):
        parts = text.split(":")
        if len(parts) > 1 and any(Decimal(p) >= 60 for p in parts[1:]):
            raise ValidationError("Minutes/seconds fields after a colon must be less than 60")
        # The largest accepted time fits int64 ns, so explicit integer parsing
        # avoids Decimal context precision loss even for a rejected huge input.
        whole, dot, frac = parts[-1].partition(".")
        total_seconds = 0
        for p in [*parts[:-1], whole]:
            total_seconds = total_seconds * 60 + int(p)
        n = total_seconds * NS + (int(frac.ljust(9, "0")) if dot else 0)
        notation = {1: "seconds", 2: "MM:SS.fraction", 3: "HH:MM:SS.fraction"}[len(parts)]
    else:
        raise ValidationError("Use seconds, MM:SS.mmm, HH:MM:SS.mmm or HH:MM:SS:FF. Frame fields need two digits, e.g. 00:00:49:21.")
    if n > MAX_NS:
        raise ValidationError("Video time exceeds the supported int64 nanosecond range")
    if positive and n <= 0:
        raise ValidationError("Retained duration must be greater than zero")
    return {"input": value, "notation": notation, "nanoseconds": str(n),
            "normalized_time": format_input_time(n),
            "timecode_fps_used": fps if frames is not None else None,
            "frame_field": frames}


def parse_video_time(value: str, *, positive: bool = False,
                     timecode_fps: int = DEFAULT_TIMECODE_FPS) -> int:
    """Backwards-compatible integer-ns API, now supporting frame timecodes."""
    return int(parse_editor_time(value, timecode_fps=timecode_fps, positive=positive)["nanoseconds"])


def parse_native_trim_times(start_timecode: str, duration_timecode: str, *,
                            timecode_fps: int = DEFAULT_TIMECODE_FPS) -> dict:
    start = parse_editor_time(start_timecode, timecode_fps=timecode_fps)
    duration = parse_editor_time(duration_timecode, timecode_fps=timecode_fps, positive=True)
    end = int(start["nanoseconds"]) + int(duration["nanoseconds"])
    if end > MAX_NS:
        raise ValidationError("Start plus duration exceeds the supported time range")
    return {"schema_version": "oculostudio-video-time-input-1.0",
            "timecode_fps": timecode_fps, "timecode_convention": "non_drop_integer_fps",
            "start": start, "duration": duration,
            "end": {"nanoseconds": str(end), "normalized_time": format_input_time(end)},
            "rounding": "nearest_nanosecond_half_up",
            "uses_frame_timecode": start["frame_field"] is not None or duration["frame_field"] is not None,
            "nrd_clock_source": "actual_MP4_PTS_and_companion_time_file",
            "source_index_checked": False,
            "notice": "Input-format conversion only. NRD duration, recording IDs and source-frame matching are checked by Preview alignment & counts."}
