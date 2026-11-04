from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from oculostudio.errors import ValidationError

NANOSECONDS = Decimal("1000000000")


def parse_timecode_ns(value: str | int | float, *, frame_rate: float = 30.0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int((Decimal(str(value)) * NANOSECONDS).to_integral_value(rounding=ROUND_HALF_UP))
    text = str(value).strip()
    if not text:
        raise ValidationError("Timecode is empty")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return int((Decimal(text) * NANOSECONDS).to_integral_value(rounding=ROUND_HALF_UP))
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes = Decimal(parts[0])
            seconds = Decimal(parts[1])
            total = minutes * 60 + seconds
        elif len(parts) == 3:
            hours = Decimal(parts[0])
            minutes = Decimal(parts[1])
            seconds = Decimal(parts[2])
            total = hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 4:
            hours, minutes, seconds, frames = [int(part) for part in parts]
            if frame_rate <= 0:
                raise ValidationError("Frame rate must be positive")
            if not (0 <= minutes < 60 and 0 <= seconds < 60 and 0 <= frames < int(frame_rate + 0.999999)):
                raise ValidationError(f"Invalid frame timecode: {text}")
            total = Decimal(hours * 3600 + minutes * 60 + seconds) + Decimal(frames) / Decimal(str(frame_rate))
        else:
            raise ValidationError(
                "Timecode must be seconds, MM:SS.mmm, HH:MM:SS.mmm, or HH:MM:SS:FF"
            )
    except (InvalidOperation, ValueError) as exc:
        raise ValidationError(f"Invalid timecode: {text}") from exc
    if total < 0:
        raise ValidationError("Timecode cannot be negative")
    return int((total * NANOSECONDS).to_integral_value(rounding=ROUND_HALF_UP))


def format_timecode_ns(value_ns: int, *, decimals: int = 3) -> str:
    sign = "-" if value_ns < 0 else ""
    value_ns = abs(int(value_ns))
    whole, fraction = divmod(value_ns, 1_000_000_000)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    if decimals <= 0:
        return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
    fraction_text = f"{fraction:09d}"[:decimals]
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction_text}"
