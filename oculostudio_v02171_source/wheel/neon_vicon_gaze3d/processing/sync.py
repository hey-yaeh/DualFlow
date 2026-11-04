from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import subprocess
from typing import Callable, Iterable

import numpy as np
from scipy.signal import find_peaks

from ..io.c3d_reader import C3DData
from ..io.pupil_recording import SceneIndex


class SyncError(RuntimeError):
    pass


def _report_progress(
    callback: Callable[[float, str], None] | None,
    value: float,
    text: str,
) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), text)


def _decode_audio_mono(video_path: Path, sample_rate: int) -> np.ndarray:
    """Decode an MP4 audio track with PyAV, falling back to FFmpeg.

    PyAV is installed by the standard app requirements and avoids requiring a
    separate FFmpeg executable on most Windows systems.
    """
    try:
        import av  # type: ignore

        chunks: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            streams = [stream for stream in container.streams if stream.type == "audio"]
            if not streams:
                return np.empty(0, dtype=np.float32)
            stream = streams[0]
            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=sample_rate
            )
            for frame in container.decode(stream):
                converted = resampler.resample(frame)
                if converted is None:
                    continue
                if not isinstance(converted, list):
                    converted = [converted]
                for output in converted:
                    chunks.append(np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1))
            flushed = resampler.resample(None)
            if flushed is not None:
                if not isinstance(flushed, list):
                    flushed = [flushed]
                for output in flushed:
                    chunks.append(np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    except Exception:
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_executable = "ffmpeg"
        command = [
            ffmpeg_executable, "-v", "error", "-i", str(video_path), "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
        ]
        try:
            result = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(result.stdout, dtype="<f4")


@dataclass
class SyncAnchor:
    pupil_timestamp_ns: int
    c3d_time_sec: float
    label: str = "clap"
    source: str = "manual"


@dataclass
class SyncMapping:
    reference_pupil_timestamp_ns: int
    scale: float
    offset_sec: float
    anchors: list[SyncAnchor]
    method: str
    uncertainty_ms: float | None = None
    notes: str = ""

    def pupil_ns_to_c3d_sec(self, timestamps_ns: np.ndarray | int) -> np.ndarray:
        values = np.asarray(timestamps_ns, dtype=np.float64)
        relative = (values - float(self.reference_pupil_timestamp_ns)) / 1e9
        return self.scale * relative + self.offset_sec

    def c3d_sec_to_pupil_ns(self, time_sec: np.ndarray | float) -> np.ndarray:
        values = np.asarray(time_sec, dtype=float)
        relative = (values - self.offset_sec) / self.scale
        return np.rint(relative * 1e9 + self.reference_pupil_timestamp_ns).astype(np.int64)

    def to_dict(self) -> dict:
        return {
            "format": "NeonVicon-Gaze3D-SyncMapping",
            "format_version": 1,
            "reference_pupil_timestamp_ns": int(self.reference_pupil_timestamp_ns),
            "scale": float(self.scale),
            "offset_sec": float(self.offset_sec),
            "equation": "c3d_time_sec = scale * ((pupil_timestamp_ns - reference_pupil_timestamp_ns) / 1e9) + offset_sec",
            "anchors": [asdict(anchor) for anchor in self.anchors],
            "method": self.method,
            "uncertainty_ms": self.uncertainty_ms,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SyncMapping":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            reference_pupil_timestamp_ns=int(data["reference_pupil_timestamp_ns"]),
            scale=float(data["scale"]),
            offset_sec=float(data["offset_sec"]),
            anchors=[SyncAnchor(**item) for item in data.get("anchors", [])],
            method=str(data.get("method", "loaded")),
            uncertainty_ms=data.get("uncertainty_ms"),
            notes=str(data.get("notes", "")),
        )


def fit_sync_mapping(
    anchors: Iterable[SyncAnchor],
    *,
    reference_pupil_timestamp_ns: int | None = None,
) -> SyncMapping:
    anchors = list(anchors)
    if not anchors:
        raise SyncError("At least one synchronization anchor is required")
    if reference_pupil_timestamp_ns is None:
        reference_pupil_timestamp_ns = int(anchors[0].pupil_timestamp_ns)
    x = np.asarray(
        [
            (anchor.pupil_timestamp_ns - reference_pupil_timestamp_ns) / 1e9
            for anchor in anchors
        ],
        dtype=float,
    )
    y = np.asarray([anchor.c3d_time_sec for anchor in anchors], dtype=float)
    if len(anchors) == 1:
        scale = 1.0
        offset = float(y[0] - x[0])
        method = "single_anchor_offset_only"
        uncertainty = None
    else:
        design = np.column_stack([x, np.ones_like(x)])
        scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - (scale * x + offset)
        uncertainty = float(np.sqrt(np.mean(residual**2)) * 1000.0)
        method = "multi_anchor_offset_and_drift"
    return SyncMapping(
        reference_pupil_timestamp_ns=int(reference_pupil_timestamp_ns),
        scale=float(scale),
        offset_sec=float(offset),
        anchors=anchors,
        method=method,
        uncertainty_ms=uncertainty,
    )


@dataclass
class AudioPeak:
    timestamp_ns: int
    section_id: str
    video_time_sec: float
    score: float


def detect_audio_clap_candidates(
    scene: SceneIndex,
    *,
    max_candidates: int = 10,
    sample_rate: int = 16000,
    progress: Callable[[float, str], None] | None = None,
) -> list[AudioPeak]:
    candidates: list[AudioPeak] = []
    sections = list(scene.sections)
    for section_index, section in enumerate(sections, start=1):
        _report_progress(
            progress,
            (section_index - 1) / max(1, len(sections)),
            f"Decoding Scene Video audio section {section_index}/{len(sections)}",
        )
        audio = _decode_audio_mono(section.video_path, sample_rate)
        if len(audio) < sample_rate:
            continue
        envelope = np.abs(audio)
        window = max(1, int(sample_rate * 0.004))
        kernel = np.ones(window, dtype=float) / window
        envelope = np.convolve(envelope, kernel, mode="same")
        threshold = float(np.percentile(envelope, 99.5))
        prominence = max(float(np.std(envelope) * 5.0), threshold * 0.25)
        peaks, props = find_peaks(
            envelope,
            height=threshold,
            prominence=prominence,
            distance=int(sample_rate * 0.35),
        )
        if len(peaks) == 0:
            continue
        prominences = props.get("prominences", envelope[peaks])
        order = np.argsort(prominences)[::-1][:max_candidates]
        first_timestamp = int(section.timestamps_ns[0])
        for index in order:
            sample = int(peaks[index])
            time_sec = sample / sample_rate
            candidates.append(
                AudioPeak(
                    timestamp_ns=int(round(first_timestamp + time_sec * 1e9)),
                    section_id=section.section_id,
                    video_time_sec=float(time_sec),
                    score=float(prominences[index]),
                )
            )

        # Add temporally diverse raw-impulse candidates.  A physical clap can
        # be quiet when the scene-camera microphone is far away, so a global
        # amplitude ranking alone may omit it.  Keeping the strongest two
        # impulses per 5-second bin makes the GUI useful for manual visual
        # confirmation without silently choosing a peak.
        raw = np.abs(audio)
        raw_peaks, raw_props = find_peaks(
            raw,
            prominence=max(0.002, float(np.std(raw) * 0.20)),
            distance=max(1, int(sample_rate * 0.10)),
        )
        raw_prominence = raw_props.get("prominences", raw[raw_peaks])
        bin_samples = int(sample_rate * 5.0)
        if len(raw_peaks):
            for bin_start in range(0, len(audio), bin_samples):
                in_bin = np.flatnonzero(
                    (raw_peaks >= bin_start) & (raw_peaks < bin_start + bin_samples)
                )
                if not len(in_bin):
                    continue
                local_order = in_bin[np.argsort(raw_prominence[in_bin])[::-1][:2]]
                for raw_index in local_order:
                    sample = int(raw_peaks[raw_index])
                    time_sec = sample / sample_rate
                    candidates.append(
                        AudioPeak(
                            timestamp_ns=int(round(first_timestamp + time_sec * 1e9)),
                            section_id=section.section_id,
                            video_time_sec=float(time_sec),
                            score=float(raw_prominence[raw_index]),
                        )
                    )

    # Deduplicate candidates closer than 50 ms, retaining the higher score.
    candidates.sort(key=lambda item: item.score, reverse=True)
    deduplicated: list[AudioPeak] = []
    for candidate in candidates:
        if any(
            existing.section_id == candidate.section_id
            and abs(existing.video_time_sec - candidate.video_time_sec) < 0.050
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    _report_progress(progress, 1.0, "Scene Video audio candidate detection complete")
    return deduplicated[:max_candidates]


@dataclass
class ViconClapCandidate:
    frame_index: int
    frame_number: int
    c3d_time_sec: float
    centroid_distance_mm: float
    score: float


def detect_vicon_clap_candidates(
    c3d: C3DData,
    top_labels: list[str],
    bottom_labels: list[str],
    *,
    max_candidates: int = 10,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[ViconClapCandidate], np.ndarray]:
    _report_progress(progress, 0.05, "Reading Vicon clapperboard marker clusters")
    top, top_valid = c3d.select(top_labels)
    bottom, bottom_valid = c3d.select(bottom_labels)
    top_count = top_valid.sum(axis=1)
    bottom_count = bottom_valid.sum(axis=1)
    top_center = np.nanmean(np.where(top_valid[..., None], top, np.nan), axis=1)
    bottom_center = np.nanmean(np.where(bottom_valid[..., None], bottom, np.nan), axis=1)
    distance = np.linalg.norm(top_center - bottom_center, axis=1)
    usable = (top_count >= 3) & (bottom_count >= 3) & np.isfinite(distance)
    filled = distance.copy()
    if usable.any():
        valid_indices = np.flatnonzero(usable)
        filled[~usable] = np.interp(
            np.flatnonzero(~usable), valid_indices, distance[valid_indices]
        )
    else:
        raise SyncError("Clapperboard marker clusters have no valid overlapping frames")
    window = max(3, int(round(c3d.frame_rate_hz * 0.025)))
    if window % 2 == 0:
        window += 1
    smoothed = np.convolve(filled, np.ones(window) / window, mode="same")
    dynamic_range = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5))
    peaks, props = find_peaks(
        -smoothed,
        prominence=max(0.5, dynamic_range * 0.03),
        distance=max(1, int(c3d.frame_rate_hz * 0.25)),
    )
    if len(peaks) == 0:
        peaks = np.asarray([int(np.nanargmin(smoothed))])
        prominences = np.asarray([1.0])
    else:
        prominences = props.get("prominences", np.ones(len(peaks)))
    candidates: list[ViconClapCandidate] = []
    for peak, prominence in zip(peaks, prominences):
        # Reward a sharp closing motion in addition to geometric proximity.
        left = max(0, peak - int(c3d.frame_rate_hz * 0.08))
        closing = max(0.0, float(smoothed[left] - smoothed[peak]))
        score = float(prominence + closing)
        candidates.append(
            ViconClapCandidate(
                frame_index=int(peak),
                frame_number=int(c3d.first_frame + peak),
                c3d_time_sec=float(peak / c3d.frame_rate_hz),
                centroid_distance_mm=float(smoothed[peak]),
                score=score,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    _report_progress(progress, 1.0, "Vicon clapperboard candidate detection complete")
    return candidates[:max_candidates], distance
