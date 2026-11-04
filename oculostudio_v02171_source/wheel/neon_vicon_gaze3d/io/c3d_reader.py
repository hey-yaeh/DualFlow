from __future__ import annotations

"""Small, dependency-free C3D point reader.

The implementation intentionally focuses on the subset required by the app:
POINT labels, 3-D point samples, validity/residual, frame range, frame rate and
units.  It supports Intel, DEC/VAX and MIPS processor encodings and both
floating-point and scaled-integer point storage.
"""

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np

PROCESSOR_INTEL = 84
PROCESSOR_DEC = 85
PROCESSOR_MIPS = 86


class C3DError(RuntimeError):
    pass


def _dec_to_ieee_bytes(data: bytes) -> np.ndarray:
    """Convert packed VAX/DEC F-floating values to IEEE float32.

    This follows the conventional C3D DEC conversion used by established C3D
    readers.  It is vectorised and suitable for full marker arrays.
    """

    src = np.frombuffer(data, dtype=np.uint8)
    if len(src) % 4:
        raise C3DError("DEC float buffer length is not divisible by four")
    dst = np.empty(len(src), dtype=np.uint8)
    dst[0::4] = src[2::4]
    dst[1::4] = src[3::4]
    dst[2::4] = src[0::4]
    # Reshuffle and decrement the VAX exponent.  Keep exact zero as zero.
    dst[3::4] = src[1::4] + (np.bitwise_and(src[1::4], 0x7F) == 0) - 1
    return np.frombuffer(dst.tobytes(), dtype=np.float32)


def _read_f32(data: bytes, processor: int) -> float:
    if processor == PROCESSOR_DEC:
        return float(_dec_to_ieee_bytes(data[:4])[0])
    endian = ">" if processor == PROCESSOR_MIPS else "<"
    return float(struct.unpack(endian + "f", data[:4])[0])


@dataclass(frozen=True)
class _Parameter:
    bytes_per_element: int
    dimensions: tuple[int, ...]
    data: bytes
    description: str


@dataclass
class C3DData:
    path: Path
    labels: list[str]
    points_mm: np.ndarray  # (frame, point, xyz)
    valid: np.ndarray  # (frame, point)
    residual: np.ndarray  # (frame, point), native residual channel
    frame_rate_hz: float
    first_frame: int
    units_original: str
    unit_to_mm: float
    parameters: dict[tuple[str, str], Any]

    @property
    def frame_count(self) -> int:
        return int(self.points_mm.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.points_mm.shape[1])

    @property
    def duration_sec(self) -> float:
        if self.frame_count <= 1:
            return 0.0
        return (self.frame_count - 1) / self.frame_rate_hz

    @property
    def time_sec(self) -> np.ndarray:
        return np.arange(self.frame_count, dtype=float) / self.frame_rate_hz

    def index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise C3DError(f"Marker label not found: {label}") from exc

    def select(self, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        indices = [self.index(label) for label in labels]
        return self.points_mm[:, indices, :], self.valid[:, indices]


class C3DReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise C3DError(f"C3D file does not exist: {self.path}")
        self._raw = self.path.read_bytes()
        if len(self._raw) < 512:
            raise C3DError("File is too short to be a C3D file")

    def read(self) -> C3DData:
        header = self._raw[:512]
        parameter_block = int(header[0])
        if header[1] != 80:
            raise C3DError(f"Invalid C3D magic byte: {header[1]}")
        parameter_offset = (parameter_block - 1) * 512
        if parameter_offset + 4 > len(self._raw):
            raise C3DError("Invalid C3D parameter block")
        processor = int(self._raw[parameter_offset + 3])
        if processor not in {PROCESSOR_INTEL, PROCESSOR_DEC, PROCESSOR_MIPS}:
            raise C3DError(f"Unsupported C3D processor type: {processor}")
        endian = ">" if processor == PROCESSOR_MIPS else "<"

        header_point_count = struct.unpack_from(endian + "H", header, 2)[0]
        header_first_frame = struct.unpack_from(endian + "H", header, 6)[0]
        header_last_frame = struct.unpack_from(endian + "H", header, 8)[0]
        header_scale = _read_f32(header[12:16], processor)
        header_data_block = struct.unpack_from(endian + "H", header, 16)[0]
        header_analog_per_frame = struct.unpack_from(endian + "H", header, 18)[0]
        header_rate = _read_f32(header[20:24], processor)

        groups, params = self._parse_parameters(
            parameter_offset=parameter_offset,
            data_block=header_data_block,
            processor=processor,
            endian=endian,
        )

        def pvalue(group: str, name: str, default: Any = None) -> Any:
            key = (group.upper(), name.upper())
            if key not in params:
                return default
            return self._decode_parameter(params[key], processor, endian)

        point_count = int(pvalue("POINT", "USED", header_point_count))
        frame_rate = float(pvalue("POINT", "RATE", header_rate))
        scale = float(pvalue("POINT", "SCALE", header_scale))
        data_block = int(pvalue("POINT", "DATA_START", header_data_block))
        frame_count = int(
            pvalue(
                "POINT",
                "FRAMES",
                max(0, header_last_frame - header_first_frame + 1),
            )
        )

        labels: list[str] = []
        suffix = ""
        while True:
            key = ("POINT", "LABELS" + suffix)
            if key not in params:
                break
            value = self._decode_parameter(params[key], processor, endian)
            if isinstance(value, str):
                labels.append(value)
            else:
                labels.extend(str(item) for item in value)
            suffix = "2" if suffix == "" else str(int(suffix) + 1)
        labels = [label.strip() for label in labels[:point_count]]
        if len(labels) < point_count:
            labels.extend(f"POINT_{i + 1}" for i in range(len(labels), point_count))

        units_raw = pvalue("POINT", "UNITS", "mm")
        if isinstance(units_raw, list):
            units_raw = units_raw[0] if units_raw else "mm"
        units = str(units_raw).strip().lower()
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
        }.get(units, 1.0)

        points, residual, valid = self._read_points(
            processor=processor,
            endian=endian,
            data_block=data_block,
            point_count=point_count,
            frame_count=frame_count,
            scale=scale,
            analog_values_per_point_frame=int(header_analog_per_frame),
        )
        points_mm = points * unit_to_mm

        decoded_params: dict[tuple[str, str], Any] = {}
        for key, param in params.items():
            try:
                decoded_params[key] = self._decode_parameter(param, processor, endian)
            except Exception:
                # Nonessential vendor parameters should not make point reading fail.
                continue
        decoded_params[("_META", "GROUPS")] = groups
        decoded_params[("_META", "PROCESSOR")] = processor

        return C3DData(
            path=self.path,
            labels=labels,
            points_mm=points_mm,
            valid=valid,
            residual=residual,
            frame_rate_hz=frame_rate,
            first_frame=int(header_first_frame),
            units_original=units,
            unit_to_mm=unit_to_mm,
            parameters=decoded_params,
        )

    def _parse_parameters(
        self,
        *,
        parameter_offset: int,
        data_block: int,
        processor: int,
        endian: str,
    ) -> tuple[dict[int, tuple[str, str]], dict[tuple[str, str], _Parameter]]:
        raw = self._raw
        pos = parameter_offset + 4
        # Parameters can occupy many blocks.  The reliable upper bound is the
        # beginning of the point-data section, not byte 2 of the parameter header.
        end = min((data_block - 1) * 512, len(raw))
        groups: dict[int, tuple[str, str]] = {}
        params_by_gid: list[tuple[int, str, _Parameter]] = []

        while pos + 2 <= end:
            name_length = struct.unpack_from("b", raw, pos)[0]
            pos += 1
            if name_length == 0:
                # Some writers add zero padding before the data block.
                break
            group_id = struct.unpack_from("b", raw, pos)[0]
            pos += 1
            length = abs(name_length)
            if pos + length + 2 > end:
                raise C3DError("Truncated C3D parameter record")
            name = raw[pos : pos + length].decode("latin-1", errors="replace")
            pos += length
            offset_field_pos = pos
            next_offset = struct.unpack_from(endian + "h", raw, pos)[0]
            pos += 2
            record_end = offset_field_pos + next_offset if next_offset > 0 else end

            if group_id < 0:
                if pos >= end:
                    break
                desc_len = int(raw[pos])
                pos += 1
                description = raw[pos : pos + desc_len].decode(
                    "latin-1", errors="replace"
                )
                groups[-group_id] = (name, description)
            else:
                if pos + 2 > end:
                    raise C3DError("Truncated C3D parameter payload")
                bytes_per_element = struct.unpack_from("b", raw, pos)[0]
                pos += 1
                dimension_count = int(raw[pos])
                pos += 1
                dimensions = tuple(int(x) for x in raw[pos : pos + dimension_count])
                pos += dimension_count
                total_bytes = abs(bytes_per_element)
                for dimension in dimensions:
                    total_bytes *= dimension
                data = bytes(raw[pos : pos + total_bytes])
                pos += total_bytes
                if pos >= end:
                    description = ""
                else:
                    desc_len = int(raw[pos])
                    pos += 1
                    description = raw[pos : pos + desc_len].decode(
                        "latin-1", errors="replace"
                    )
                params_by_gid.append(
                    (
                        group_id,
                        name,
                        _Parameter(
                            bytes_per_element=bytes_per_element,
                            dimensions=dimensions,
                            data=data,
                            description=description,
                        ),
                    )
                )
            if next_offset > 0:
                pos = record_end
            else:
                break

        params: dict[tuple[str, str], _Parameter] = {}
        for gid, name, param in params_by_gid:
            group_name = groups.get(gid, (str(gid), ""))[0]
            params[(group_name.upper(), name.upper())] = param
        return groups, params

    @staticmethod
    def _decode_parameter(
        param: _Parameter, processor: int, endian: str
    ) -> Any:
        bpe = param.bytes_per_element
        dims = param.dimensions
        data = param.data
        if bpe == -1:
            if not dims:
                return data.decode("latin-1", errors="replace").rstrip("\x00 ")
            string_length = dims[0]
            count = int(np.prod(dims[1:])) if len(dims) > 1 else 1
            return [
                data[i * string_length : (i + 1) * string_length]
                .decode("latin-1", errors="replace")
                .rstrip("\x00 ")
                for i in range(count)
            ]
        if bpe == 1:
            values = np.frombuffer(data, dtype=np.int8)
        elif bpe == 2:
            values = np.frombuffer(data, dtype=np.dtype(endian + "i2"))
        elif bpe == 4:
            if processor == PROCESSOR_DEC:
                values = _dec_to_ieee_bytes(data)
            else:
                values = np.frombuffer(data, dtype=np.dtype(endian + "f4"))
        else:
            raise C3DError(f"Unsupported C3D parameter element size: {bpe}")
        if not dims:
            return values[0].item() if len(values) else None
        shaped = values.reshape(tuple(reversed(dims)))
        return shaped.copy()

    def _read_points(
        self,
        *,
        processor: int,
        endian: str,
        data_block: int,
        point_count: int,
        frame_count: int,
        scale: float,
        analog_values_per_point_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = (data_block - 1) * 512
        if scale < 0:
            point_bytes_per_frame = point_count * 4 * 4
            analog_bytes_per_frame = analog_values_per_point_frame * 4
            frame_bytes = point_bytes_per_frame + analog_bytes_per_frame
            needed = frame_count * frame_bytes
            if start + needed > len(self._raw):
                available_frames = max(0, (len(self._raw) - start) // frame_bytes)
                frame_count = min(frame_count, available_frames)
                needed = frame_count * frame_bytes
            if analog_bytes_per_frame == 0:
                point_raw = self._raw[start : start + needed]
                if processor == PROCESSOR_DEC:
                    values = _dec_to_ieee_bytes(point_raw)
                else:
                    dtype = np.dtype((">" if processor == PROCESSOR_MIPS else "<") + "f4")
                    values = np.frombuffer(point_raw, dtype=dtype)
                values = values.reshape(frame_count, point_count, 4)
            else:
                values = np.empty((frame_count, point_count, 4), dtype=np.float32)
                for frame in range(frame_count):
                    offset = start + frame * frame_bytes
                    chunk = self._raw[offset : offset + point_bytes_per_frame]
                    if processor == PROCESSOR_DEC:
                        decoded = _dec_to_ieee_bytes(chunk)
                    else:
                        dtype = np.dtype(
                            (">" if processor == PROCESSOR_MIPS else "<") + "f4"
                        )
                        decoded = np.frombuffer(chunk, dtype=dtype)
                    values[frame] = decoded.reshape(point_count, 4)
            points = values[:, :, :3].astype(float)
            residual = values[:, :, 3].astype(float)
            valid = np.isfinite(points).all(axis=2) & (residual >= 0)
        else:
            point_bytes_per_frame = point_count * 4 * 2
            analog_bytes_per_frame = analog_values_per_point_frame * 2
            frame_bytes = point_bytes_per_frame + analog_bytes_per_frame
            available_frames = max(0, (len(self._raw) - start) // frame_bytes)
            frame_count = min(frame_count, available_frames)
            values = np.empty((frame_count, point_count, 4), dtype=np.int16)
            dtype = np.dtype(endian + "i2")
            for frame in range(frame_count):
                offset = start + frame * frame_bytes
                chunk = self._raw[offset : offset + point_bytes_per_frame]
                values[frame] = np.frombuffer(chunk, dtype=dtype).reshape(point_count, 4)
            points = values[:, :, :3].astype(float) * scale
            packed = values[:, :, 3]
            valid = packed >= 0
            residual = np.where(valid, (packed & 0xFF).astype(float) * abs(scale), -1.0)

        points[~valid] = np.nan
        return points, residual, valid


def read_c3d(path: str | Path) -> C3DData:
    """Read C3D through OculoStudio's unified long-frame-safe core.

    The legacy calibration package keeps its historical C3DData contract, while
    all frame-count, DEC/VAX and point decoding is delegated to the same reader
    used by import, preview, assignment, Gaze 3D and AOI services.
    """

    from oculostudio.adapters.mocap import read_mocap

    source = read_mocap(path)
    valid = np.asarray(source.valid, dtype=bool)
    points = np.asarray(source.points_mm, dtype=float).copy()
    points[~valid] = np.nan
    residual = np.where(valid, 0.0, -1.0)
    metadata = dict(source.metadata or {})
    units = str(metadata.get("units") or "mm")
    return C3DData(
        path=Path(path),
        labels=list(source.labels),
        points_mm=points,
        valid=valid,
        residual=residual,
        frame_rate_hz=float(source.frame_rate),
        first_frame=int(source.first_frame),
        units_original=units,
        unit_to_mm=1.0,
        parameters={("_META", "OCULOSTUDIO"): metadata},
    )
