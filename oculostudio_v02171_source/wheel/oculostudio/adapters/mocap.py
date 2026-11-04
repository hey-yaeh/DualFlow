from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Any

import numpy as np

from oculostudio.errors import DependencyError, ProcessingError, ValidationError


_C3D_BLOCK_BYTES = 512
_C3D_PROCESSOR_INTEL = 84
_C3D_PROCESSOR_DEC = 85
_C3D_PROCESSOR_MIPS = 86


@dataclass
class MocapData:
    path: Path
    time_sec: np.ndarray
    labels: list[str]
    points_mm: np.ndarray
    valid: np.ndarray
    frame_rate: float
    first_frame: int = 0
    metadata: dict | None = None

    @property
    def frame_count(self) -> int:
        return int(len(self.time_sec))

    def select(self, labels: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        requested = list(labels)
        normalized = {_normalize_label(label): index for index, label in enumerate(self.labels)}
        indices: list[int] = []
        missing: list[str] = []
        for label in requested:
            key = _normalize_label(label)
            if key not in normalized:
                missing.append(label)
            else:
                indices.append(normalized[key])
        if missing:
            raise ValidationError(f"Mocap markers are missing: {missing}")
        return self.points_mm[:, indices], self.valid[:, indices]


def _normalize_label(label: str) -> str:
    return str(label).strip().replace(" ", "").upper()


@dataclass(frozen=True)
class _C3DParameter:
    group_id: int
    name: str
    type_code: int
    dimensions: tuple[int, ...]
    raw: bytes
    processor_type: int


@dataclass(frozen=True)
class _C3DLayout:
    processor_type: int
    point_count: int
    analog_values_per_frame: int
    frame_rate: float
    scale: float
    data_offset: int
    bytes_per_sample: int
    bytes_per_frame: int
    frame_count: int
    first_frame: int
    last_frame: int
    labels: tuple[str, ...]
    units: str
    frame_count_source: str
    header_first_frame: int
    header_last_frame: int
    point_frames_parameter: int | None
    file_frame_capacity: int
    trailing_bytes: int
    long_frame_recovered: bool
    parameter_processor_name: str


def _processor_name(processor_type: int) -> str:
    return {
        _C3D_PROCESSOR_INTEL: "Intel/IEEE little-endian",
        _C3D_PROCESSOR_DEC: "DEC/VAX little-endian",
        _C3D_PROCESSOR_MIPS: "MIPS/IEEE big-endian",
    }.get(processor_type, f"unknown ({processor_type})")


def _integer_endian(processor_type: int) -> str:
    if processor_type in {_C3D_PROCESSOR_INTEL, _C3D_PROCESSOR_DEC}:
        return "<"
    if processor_type == _C3D_PROCESSOR_MIPS:
        return ">"
    raise ProcessingError(f"Unsupported C3D processor type: {processor_type}")


def _vax_f32_array(raw: bytes | np.ndarray) -> np.ndarray:
    """Convert DEC/VAX F-floating bytes to native IEEE float32.

    C3D processor type 85 stores 32-bit floating-point values in VAX F format.
    The exponent is biased two steps above IEEE-754 and the fraction bytes use
    the VAX word order. This vectorised conversion is used for both parameters
    and the point data section.
    """

    array = (np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else np.asarray(raw, dtype=np.uint8)).reshape(-1, 4)
    b0 = array[:, 0].astype(np.uint32)
    b1 = array[:, 1].astype(np.uint32)
    b2 = array[:, 2].astype(np.uint32)
    b3 = array[:, 3].astype(np.uint32)
    sign = (b1 >> 7) & 0x1
    exponent = ((b1 & 0x7F) << 1) | (b0 >> 7)
    fraction = ((b0 & 0x7F) << 16) | (b3 << 8) | b2
    ieee_exponent = np.where(exponent >= 2, exponent - 2, 0).astype(np.uint32)
    bits = (sign << 31) | (ieee_exponent << 23) | fraction
    values = bits.astype(np.uint32).view(np.float32)
    values = values.copy()
    values[exponent == 0] = 0.0
    return values


def _decode_float32(raw: bytes | np.ndarray, processor_type: int) -> np.ndarray:
    if len(raw) % 4:
        raise ProcessingError("Invalid C3D floating-point byte count")
    if processor_type == _C3D_PROCESSOR_DEC:
        return _vax_f32_array(raw)
    dtype = "<f4" if processor_type == _C3D_PROCESSOR_INTEL else ">f4"
    return np.frombuffer(raw, dtype=dtype).astype(np.float32, copy=True)


def _decode_int16(raw: bytes, processor_type: int, *, unsigned: bool = False) -> np.ndarray:
    dtype = (_integer_endian(processor_type) + ("u2" if unsigned else "i2"))
    return np.frombuffer(raw, dtype=dtype).copy()


def _decode_uint32(raw: bytes, processor_type: int) -> int:
    if len(raw) < 4:
        raise ProcessingError("Invalid 32-bit C3D parameter")
    return int(struct.unpack(_integer_endian(processor_type) + "I", raw[:4])[0])


def _read_parameter_records(path: Path, header: bytes) -> tuple[int, dict[tuple[str, str], _C3DParameter]]:
    parameter_block = int(header[0])
    if parameter_block <= 0:
        raise ProcessingError("Invalid C3D parameter-section pointer")
    parameter_offset = (parameter_block - 1) * _C3D_BLOCK_BYTES
    with path.open("rb") as stream:
        stream.seek(parameter_offset)
        section_header = stream.read(4)
        if len(section_header) != 4:
            raise ProcessingError("C3D parameter header is truncated")
        processor_type = int(section_header[3])
        endian = _integer_endian(processor_type)
        stream.seek(0, 2)
        file_size = stream.tell()
        # Header word 9 is duplicated POINT:DATA_START. It is safe for DEC and
        # Intel because integers share little-endian storage; MIPS is big-endian.
        data_block = struct.unpack(endian + "H", header[16:18])[0]
        data_offset = (int(data_block) - 1) * _C3D_BLOCK_BYTES if data_block else file_size
        limit = min(file_size, data_offset if data_offset > parameter_offset else file_size)
        position = parameter_offset + 4
        groups: dict[int, str] = {}
        pending: list[_C3DParameter] = []
        visited: set[int] = set()
        while position + 2 <= limit and position not in visited:
            visited.add(position)
            stream.seek(position)
            prefix = stream.read(2)
            if len(prefix) < 2:
                break
            name_length = struct.unpack("b", prefix[:1])[0]
            group_id = struct.unpack("b", prefix[1:2])[0]
            if name_length == 0:
                break
            width = abs(int(name_length))
            name_bytes = stream.read(width)
            if len(name_bytes) != width:
                break
            name = name_bytes.decode("latin-1", errors="replace").strip().upper()
            offset_position = stream.tell()
            offset_raw = stream.read(2)
            if len(offset_raw) != 2:
                break
            next_offset = int(struct.unpack(endian + "h", offset_raw)[0])
            next_position = offset_position + next_offset
            payload_position = stream.tell()
            if group_id < 0:
                description_length_raw = stream.read(1)
                if description_length_raw:
                    description_length = int(description_length_raw[0])
                    stream.seek(description_length, 1)
                groups[-group_id] = name
            elif group_id > 0:
                type_raw = stream.read(1)
                dimension_count_raw = stream.read(1)
                if not type_raw or not dimension_count_raw:
                    break
                type_code = struct.unpack("b", type_raw)[0]
                dimension_count = int(dimension_count_raw[0])
                dimensions = tuple(int(value) for value in stream.read(dimension_count))
                element_count = 1
                for dimension in dimensions:
                    element_count *= int(dimension)
                data_length = abs(type_code) * element_count
                raw_value = stream.read(data_length)
                pending.append(
                    _C3DParameter(
                        group_id=int(group_id),
                        name=name,
                        type_code=int(type_code),
                        dimensions=dimensions,
                        raw=raw_value,
                        processor_type=processor_type,
                    )
                )
            if next_offset <= 0 or next_position <= position or next_position > limit:
                # Some writer implementations leave an inaccurate parameter-block
                # count but the record chain itself remains valid. Stop safely when
                # a record does not point forward into the parameter region.
                break
            position = next_position

    records: dict[tuple[str, str], _C3DParameter] = {}
    for parameter in pending:
        group_name = groups.get(parameter.group_id, f"GROUP_{parameter.group_id}")
        records[(group_name.upper(), parameter.name.upper())] = parameter
    return processor_type, records


def _parameter(records: dict[tuple[str, str], _C3DParameter], group: str, name: str) -> _C3DParameter | None:
    return records.get((group.upper(), name.upper()))


def _parameter_strings(parameter: _C3DParameter | None) -> list[str]:
    if parameter is None or parameter.type_code != -1:
        return []
    raw = parameter.raw
    dimensions = parameter.dimensions
    if not dimensions:
        return [raw.decode("latin-1", errors="replace").strip(" \x00")]
    width = max(1, int(dimensions[0]))
    count = max(1, len(raw) // width)
    return [
        raw[index * width : (index + 1) * width].decode("latin-1", errors="replace").strip(" \x00")
        for index in range(count)
    ]


def _parameter_uint16_values(parameter: _C3DParameter | None) -> list[int]:
    if parameter is None:
        return []
    if parameter.type_code == 2:
        return [int(value) for value in _decode_int16(parameter.raw, parameter.processor_type, unsigned=True)]
    if parameter.type_code == 1:
        return [int(value) for value in parameter.raw]
    return []


def _parameter_float_values(parameter: _C3DParameter | None) -> list[float]:
    if parameter is None:
        return []
    if parameter.type_code == 4:
        return [float(value) for value in _decode_float32(parameter.raw, parameter.processor_type)]
    if parameter.type_code == 2:
        return [float(value) for value in _decode_int16(parameter.raw, parameter.processor_type, unsigned=False)]
    if parameter.type_code == 1:
        return [float(value) for value in parameter.raw]
    return []


def _parameter_scalar_int(parameter: _C3DParameter | None, default: int | None = None) -> int | None:
    if parameter is None:
        return default
    if parameter.type_code == 2 and len(parameter.raw) >= 2:
        return int(_decode_int16(parameter.raw[:2], parameter.processor_type, unsigned=True)[0])
    if parameter.type_code == 1 and parameter.raw:
        return int(parameter.raw[0])
    if parameter.type_code == 4 and len(parameter.raw) >= 4:
        value = float(_decode_float32(parameter.raw[:4], parameter.processor_type)[0])
        if np.isfinite(value):
            return int(round(value))
    return default


def _parameter_scalar_float(parameter: _C3DParameter | None, default: float | None = None) -> float | None:
    values = _parameter_float_values(parameter)
    return float(values[0]) if values else default


def _trial_field_value(parameter: _C3DParameter | None) -> int | None:
    values = _parameter_uint16_values(parameter)
    if len(values) < 2:
        return None
    # The official C3D documentation describes the high word multiplier as
    # 65535. Some software uses the conventional 65536. The caller validates
    # both possibilities against the data-section capacity.
    return int(values[0] + values[1] * 65535)


def _trial_field_value_65536(parameter: _C3DParameter | None) -> int | None:
    values = _parameter_uint16_values(parameter)
    if len(values) < 2:
        return None
    return int(values[0] + values[1] * 65536)


def _raw_uint32_parameter(parameter: _C3DParameter | None) -> int | None:
    if parameter is None or len(parameter.raw) < 4:
        return None
    return _decode_uint32(parameter.raw, parameter.processor_type)


def _build_c3d_layout(path: Path) -> _C3DLayout:
    with path.open("rb") as stream:
        header = stream.read(_C3D_BLOCK_BYTES)
    if len(header) < 24 or header[1] != 0x50:
        raise ProcessingError("The file is not a supported C3D point file")
    processor_type, records = _read_parameter_records(path, header)
    endian = _integer_endian(processor_type)

    point_count = _parameter_scalar_int(_parameter(records, "POINT", "USED"))
    if not point_count:
        point_count = int(struct.unpack(endian + "H", header[2:4])[0])
    if not point_count or point_count <= 0:
        raise ProcessingError("C3D POINT:USED is missing or invalid")

    scale = _parameter_scalar_float(_parameter(records, "POINT", "SCALE"))
    if scale is None:
        scale = float(_decode_float32(header[12:16], processor_type)[0])
    frame_rate = _parameter_scalar_float(_parameter(records, "POINT", "RATE"))
    if frame_rate is None:
        frame_rate = float(_decode_float32(header[20:24], processor_type)[0])
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ProcessingError("C3D POINT:RATE is missing or invalid")

    data_block = _parameter_scalar_int(_parameter(records, "POINT", "DATA_START"))
    if not data_block:
        data_block = int(struct.unpack(endian + "H", header[16:18])[0])
    if not data_block or data_block <= 0:
        raise ProcessingError("C3D POINT:DATA_START is missing or invalid")
    data_offset = (int(data_block) - 1) * _C3D_BLOCK_BYTES

    header_analog_values = int(struct.unpack(endian + "H", header[4:6])[0])
    analog_values_per_frame = header_analog_values
    if analog_values_per_frame <= 0:
        analog_used = _parameter_scalar_int(_parameter(records, "ANALOG", "USED"), 0) or 0
        analog_rate = _parameter_scalar_float(_parameter(records, "ANALOG", "RATE"), 0.0) or 0.0
        ratio = int(round(analog_rate / frame_rate)) if analog_used and analog_rate else 0
        analog_values_per_frame = max(0, int(analog_used) * max(0, ratio))

    bytes_per_sample = 4 if float(scale) < 0 else 2
    point_bytes_per_frame = int(point_count) * 4 * bytes_per_sample
    bytes_per_frame = point_bytes_per_frame + int(analog_values_per_frame) * bytes_per_sample
    if bytes_per_frame <= 0:
        raise ProcessingError("C3D frame layout is invalid")
    data_bytes = max(0, path.stat().st_size - data_offset)
    file_frame_capacity = data_bytes // bytes_per_frame
    trailing_bytes = data_bytes % bytes_per_frame
    if file_frame_capacity <= 0:
        raise ProcessingError("C3D data section contains no complete frames")

    labels: list[str] = []
    for parameter_name in ("LABELS", "LABELS2"):
        labels.extend(value for value in _parameter_strings(_parameter(records, "POINT", parameter_name)) if value)
    if len(labels) < point_count:
        labels.extend(f"POINT_{index + 1}" for index in range(len(labels), point_count))
    labels = labels[:point_count]
    units_values = _parameter_strings(_parameter(records, "POINT", "UNITS"))
    units = units_values[0] if units_values else "mm"

    header_first = int(struct.unpack(endian + "H", header[6:8])[0])
    header_last = int(struct.unpack(endian + "H", header[8:10])[0])
    point_frames = _parameter_scalar_int(_parameter(records, "POINT", "FRAMES"))

    candidates: list[tuple[str, int, int]] = []
    hom_first = _raw_uint32_parameter(_parameter(records, "POINT", "HOM_FIRST_FRAME"))
    hom_last = _raw_uint32_parameter(_parameter(records, "POINT", "HOM_LAST_FRAME"))
    if hom_first is not None and hom_last is not None and hom_last >= hom_first:
        candidates.append(("POINT:HOM_FIRST_FRAME/HOM_LAST_FRAME", hom_first, hom_last))

    trial_start_param = _parameter(records, "TRIAL", "ACTUAL_START_FIELD")
    trial_end_param = _parameter(records, "TRIAL", "ACTUAL_END_FIELD")
    for source, start, end in (
        ("TRIAL:ACTUAL_START_FIELD/ACTUAL_END_FIELD", _trial_field_value(trial_start_param), _trial_field_value(trial_end_param)),
        ("TRIAL fields (65536-word interpretation)", _trial_field_value_65536(trial_start_param), _trial_field_value_65536(trial_end_param)),
    ):
        if start is not None and end is not None and end >= start:
            candidates.append((source, start, end))

    long_frames = _parameter_scalar_int(_parameter(records, "POINT", "LONG_FRAMES"))
    if long_frames and long_frames > 0:
        candidates.append(("POINT:LONG_FRAMES", header_first, header_first + int(long_frames) - 1))

    selected_source = "POINT:FRAMES"
    selected_first = header_first
    selected_count = int(point_frames or 0)

    # Prefer explicit 32-bit vendor/standard ranges when they agree with the
    # physical data section. Vicon DEC files may store POINT:FRAMES modulo
    # 65536 while retaining the real range in POINT:HOM_*.
    plausible_ranges = [
        (source, first, last)
        for source, first, last in candidates
        if 0 < (last - first + 1) <= file_frame_capacity
    ]
    if plausible_ranges:
        source, first, last = min(
            plausible_ranges,
            key=lambda item: abs(file_frame_capacity - (item[2] - item[1] + 1)),
        )
        selected_source = source
        selected_first = int(first)
        selected_count = int(last - first + 1)
    elif selected_count <= 0 or selected_count > file_frame_capacity:
        selected_source = "data-section byte count"
        selected_count = int(file_frame_capacity)
        selected_first = int(header_first)
    elif file_frame_capacity > selected_count:
        difference = int(file_frame_capacity - selected_count)
        if difference > 0 and difference % 65536 == 0:
            selected_source = "data-section byte count (16-bit frame-count wrap recovered)"
            selected_count = int(file_frame_capacity)
            selected_first = int(header_first)

    # Do not count all-zero storage padding as a final frame when the explicit
    # POINT:FRAMES value is otherwise plausible (a common C3D writer pattern).
    if selected_count == file_frame_capacity and point_frames and 0 < point_frames < file_frame_capacity:
        difference = file_frame_capacity - int(point_frames)
        if difference < 4 and difference % 65536 != 0:
            selected_source = "POINT:FRAMES"
            selected_count = int(point_frames)
            selected_first = int(header_first)

    if selected_count <= 0 or selected_count > file_frame_capacity:
        raise ProcessingError(
            f"C3D frame count {selected_count} is incompatible with the data-section capacity {file_frame_capacity}"
        )
    selected_last = int(selected_first + selected_count - 1)
    long_frame_recovered = bool(
        selected_count > 65535
        or (point_frames is not None and selected_count != int(point_frames))
        or "HOM_" in selected_source
        or "ACTUAL_" in selected_source
        or "LONG_FRAMES" in selected_source
    )
    return _C3DLayout(
        processor_type=processor_type,
        point_count=int(point_count),
        analog_values_per_frame=int(analog_values_per_frame),
        frame_rate=float(frame_rate),
        scale=float(scale),
        data_offset=int(data_offset),
        bytes_per_sample=int(bytes_per_sample),
        bytes_per_frame=int(bytes_per_frame),
        frame_count=int(selected_count),
        first_frame=int(selected_first),
        last_frame=int(selected_last),
        labels=tuple(labels),
        units=str(units or "mm"),
        frame_count_source=selected_source,
        header_first_frame=int(header_first),
        header_last_frame=int(header_last),
        point_frames_parameter=int(point_frames) if point_frames is not None else None,
        file_frame_capacity=int(file_frame_capacity),
        trailing_bytes=int(trailing_bytes),
        long_frame_recovered=long_frame_recovered,
        parameter_processor_name=_processor_name(processor_type),
    )


def _read_c3d_internal(path: Path) -> MocapData:
    layout = _build_c3d_layout(path)
    point_bytes_per_frame = layout.point_count * 4 * layout.bytes_per_sample
    raw_frames = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=layout.data_offset,
        shape=(layout.frame_count, layout.bytes_per_frame),
    )
    raw_points = np.ascontiguousarray(raw_frames[:, :point_bytes_per_frame])
    if layout.bytes_per_sample == 4:
        values = _decode_float32(raw_points.reshape(-1, 4), layout.processor_type).reshape(
            layout.frame_count, layout.point_count, 4
        )
        points = values[:, :, :3].astype(float, copy=False)
        residual = values[:, :, 3].astype(float, copy=False)
        valid = np.isfinite(points).all(axis=2) & np.isfinite(residual) & (residual >= 0)
    else:
        dtype = _integer_endian(layout.processor_type) + "i2"
        values = np.frombuffer(raw_points, dtype=dtype).reshape(layout.frame_count, layout.point_count, 4)
        points = values[:, :, :3].astype(float) * abs(layout.scale)
        residual_word = values[:, :, 3].astype(np.int32)
        valid = np.isfinite(points).all(axis=2) & (residual_word >= 0)

    # Normalize coordinates to millimetres because all downstream geometry
    # contracts use mm, while preserving the original POINT:UNITS metadata.
    unit_key = layout.units.strip().lower()
    unit_to_mm = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimetre": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "metre": 1000.0,
        "in": 25.4,
        "inch": 25.4,
    }.get(unit_key, 1.0)
    points = np.asarray(points, dtype=float) * unit_to_mm
    time_sec = (np.arange(layout.frame_count, dtype=float) + layout.first_frame) / layout.frame_rate
    metadata = {
        "format": "c3d",
        "units": "mm",
        "units_original": layout.units,
        "unit_to_mm": unit_to_mm,
        "reader_backend": "oculostudio_c3d_core_v2",
        "processor_type": layout.processor_type,
        "processor_format": layout.parameter_processor_name,
        "frame_count_source": layout.frame_count_source,
        "point_frames_parameter": layout.point_frames_parameter,
        "header_first_frame": layout.header_first_frame,
        "header_last_frame": layout.header_last_frame,
        "actual_first_frame": layout.first_frame,
        "actual_last_frame": layout.last_frame,
        "file_frame_capacity": layout.file_frame_capacity,
        "data_offset_bytes": layout.data_offset,
        "bytes_per_frame": layout.bytes_per_frame,
        "trailing_bytes": layout.trailing_bytes,
        "analog_values_per_frame": layout.analog_values_per_frame,
        "long_frame_recovered": layout.long_frame_recovered,
    }
    return MocapData(
        path=path,
        time_sec=time_sec,
        labels=list(layout.labels),
        points_mm=points,
        valid=np.asarray(valid, dtype=bool),
        frame_rate=layout.frame_rate,
        first_frame=layout.first_frame,
        metadata=metadata,
    )


def _read_c3d_ezc3d(path: Path) -> MocapData:
    try:
        import ezc3d  # type: ignore
    except Exception as exc:
        raise DependencyError("The optional ezc3d fallback is unavailable") from exc
    data = ezc3d.c3d(str(path))
    labels = [str(label).strip() for label in data["parameters"]["POINT"]["LABELS"]["value"]]
    points_raw = np.asarray(data["data"]["points"], dtype=float)
    points = np.transpose(points_raw[:3], (2, 1, 0))
    residual = np.transpose(points_raw[3], (1, 0))
    valid = np.isfinite(points).all(axis=2) & np.isfinite(residual) & (residual >= 0)
    rate = float(data["parameters"]["POINT"]["RATE"]["value"][0])
    first_frame = int(data["header"]["points"].get("first_frame", 0))
    time_sec = (np.arange(points.shape[0], dtype=float) + first_frame) / rate
    return MocapData(
        path=path,
        time_sec=time_sec,
        labels=labels,
        points_mm=points,
        valid=valid,
        frame_rate=rate,
        first_frame=first_frame,
        metadata={"format": "c3d", "units": "mm", "reader_backend": "ezc3d_fallback"},
    )


def read_mocap(path: str | Path) -> MocapData:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            time_sec = np.asarray(archive["time_sec"], dtype=float)
            labels = [str(x) for x in archive["labels"].tolist()]
            points = np.asarray(archive["points_mm"], dtype=float)
            valid = np.asarray(archive["valid"], dtype=bool) if "valid" in archive else np.isfinite(points).all(axis=2)
            frame_rate = float(archive["frame_rate"]) if "frame_rate" in archive else float(1.0 / np.median(np.diff(time_sec)))
        return MocapData(path, time_sec, labels, points, valid, frame_rate, metadata={"format": "oculostudio_npz"})
    if path.suffix.lower() != ".c3d":
        raise ProcessingError(f"Unsupported motion-capture format: {path.suffix}")
    internal_error: Exception | None = None
    try:
        return _read_c3d_internal(path)
    except Exception as exc:
        internal_error = exc
    try:
        fallback = _read_c3d_ezc3d(path)
        fallback.metadata = {
            **(fallback.metadata or {}),
            "internal_reader_warning": f"{type(internal_error).__name__}: {internal_error}",
        }
        return fallback
    except Exception as fallback_error:
        raise ProcessingError(
            "Could not decode the C3D file with the built-in reader or ezc3d fallback. "
            f"Built-in error: {internal_error}. Fallback error: {fallback_error}"
        ) from fallback_error


def probe_mocap(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "format": path.suffix.lower().lstrip("."),
        "reader_available": True,
    }
    try:
        data = read_mocap(path)
        duration = (data.time_sec[-1] - data.time_sec[0]) if len(data.time_sec) > 1 else 0.0
        metadata = dict(data.metadata or {})
        result.update(
            {
                "frame_count": data.frame_count,
                "frame_rate": data.frame_rate,
                "duration_sec": float(duration),
                "nominal_duration_sec": float(data.frame_count / data.frame_rate) if data.frame_rate else None,
                "marker_count": len(data.labels),
                "marker_labels_preview": data.labels[:40],
                "valid_sample_fraction": float(np.mean(data.valid)) if data.valid.size else 0.0,
                **metadata,
            }
        )
        if metadata.get("long_frame_recovered"):
            wrapped = metadata.get("point_frames_parameter")
            result["long_frame_notice"] = (
                f"Recovered {data.frame_count:,} C3D frames from the 32-bit frame range"
                + (f"; POINT:FRAMES contained the wrapped value {wrapped:,}." if isinstance(wrapped, int) else ".")
            )
    except DependencyError as exc:
        result["reader_available"] = False
        result["warning"] = str(exc)
    return result
