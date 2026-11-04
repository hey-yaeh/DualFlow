from __future__ import annotations

"""Robust shared-audio synchronization for simultaneously recorded videos.

The service aligns an external facial-expression video (source clock) to an
Ego/NRD scene video (reference clock) without assuming identical starts or
identical complete soundtracks.  It uses a microphone-robust amplitude/onset
signature, distributed local matches, affine drift fitting, leave-one-window-
out validation, and explicit user acceptance.  A Pupil Native reference also
publishes a Native-media-PTS -> Pupil-recording-clock bridge so the external
video can join the existing TimeGraph.
"""

import hashlib
from copy import deepcopy
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfilt

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.native_reference_clock import (
    build_clock,
    open_member,
    select_native,
    source_stamp,
)
from oculostudio.adapters.pupil import find_cloud_candidates
from oculostudio.domain.models import ArtifactKind, TimeAnchor
from oculostudio.domain.timegraph import apply_time_map, fit_time_map
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import resolve_media_tool
from oculostudio.services.auto_synchronization import (
    GlobalAlignment,
    _global_alignment,
    _pearson,
    _refine_lag_peak,
)
from oculostudio.services.synchronization import fit_and_store_time_map
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import (
    CancellationToken,
    atomic_write_json,
    emit,
    new_id,
    safe_name,
    utc_now,
)


AUDIO_VIDEO_SYNC_ENGINE = "shared_audio_video_affine_v1_distributed_holdout"
AUDIO_DECODE_RATE_HZ = 8_000
AUDIO_FEATURE_RATE_HZ = 100.0


@dataclass(frozen=True)
class VideoInput:
    asset_id: str
    asset_label: str
    asset_kind: str
    source_path: str
    video_path: str
    video_member: str | None
    media_domain_id: str
    capture_domain_id: str | None
    stamp: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AudioSignal:
    grid_sec: np.ndarray
    signature: np.ndarray
    raw_energy: np.ndarray
    valid: np.ndarray
    sample_rate_hz: float
    metadata: dict[str, Any]


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _domain_by_role(session, asset, role: str) -> str | None:
    for domain_id in asset.time_domain_ids:
        domain = session.time_domains.get(domain_id)
        if domain and str((domain.metadata or {}).get("role") or "") == role:
            return domain.id
    return None


def _cache_member(store: ProjectStore, asset, member: str, stamp: dict[str, Any]) -> Path:
    suffix = Path(member).suffix.lower() or ".mp4"
    key = hashlib.sha256(
        json.dumps(
            {
                "source": str(Path(asset.source_path).resolve()),
                "asset_id": asset.id,
                "member": member,
                "stamp": stamp,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:28]
    target = store.cache_dir() / "audio_video_sync_sources" / f"{key}{suffix}"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".partial")
    pending.unlink(missing_ok=True)
    with open_member(asset.source_path, member) as source, pending.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    pending.replace(target)
    return target


def _asset_identity(asset) -> dict[str, str]:
    raw = (asset.metadata or {}).get("identity") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        "session_id": str(raw.get("session_id") or "").upper(),
        "interaction_id": str(raw.get("interaction_id") or "").upper(),
        "role_id": str(raw.get("role_id") or "").upper(),
    }


def _validate_role_context(store: ProjectStore, *, session_id: str, source_asset_id: str,
                           reference_asset_id: str, wearer_role_id: str | None) -> None:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    source = session.assets.get(source_asset_id)
    reference = session.assets.get(reference_asset_id)
    if source is None or reference is None:
        raise ValidationError("Selected video asset is unavailable")
    source_identity = _asset_identity(source)
    reference_identity = _asset_identity(reference)
    requested_role = str(wearer_role_id or "").upper()
    for label, identity in (("Facial-expression", source_identity), ("Ego/reference", reference_identity)):
        if requested_role and identity.get("role_id") and identity["role_id"] != requested_role:
            raise ValidationError(
                f"{label} video is assigned to {identity['role_id']}, not requested wearer {requested_role}"
            )
    if (
        source_identity.get("interaction_id")
        and reference_identity.get("interaction_id")
        and source_identity["interaction_id"] != reference_identity["interaction_id"]
    ):
        raise ValidationError("Facial-expression and ego videos belong to different interactions")
    if (
        source_identity.get("role_id")
        and reference_identity.get("role_id")
        and source_identity["role_id"] != reference_identity["role_id"]
    ):
        raise ValidationError("Facial-expression and ego videos belong to different wearer roles")


def _resolve_video_input(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    requested_member: str | None = None,
) -> VideoInput:
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    asset = session.assets.get(asset_id)
    if asset is None:
        raise ValidationError("Selected video asset is unavailable")
    kind = str(asset.kind)
    stamp = source_stamp(asset.source_path)
    member: str | None = None
    capture_domain_id: str | None = None
    metadata: dict[str, Any] = {}

    if kind == "video":
        path = Path(asset.source_path).expanduser().resolve()
        if not path.is_file():
            raise ValidationError("Selected video source is not a readable file")
        media_domain_id = asset.primary_time_domain_id
    elif kind == "pupil_native":
        selected = select_native(asset.source_path, requested_member)
        member = str(selected["video_member"])
        path = _cache_member(store, asset, member, stamp)
        media_domain_id = _domain_by_role(session, asset, "native_scene_pts")
        capture_domain_id = _domain_by_role(session, asset, "pupil_clock") or asset.primary_time_domain_id
        metadata = {
            "recording_id": selected.get("recording_id"),
            "time_member": selected.get("time_member"),
            "info_member": selected.get("info_member"),
        }
    elif kind == "pupil_cloud":
        root = MaterializationCache(store.cache_dir()).materialize(asset)
        candidates = find_cloud_candidates(root)
        if asset.recording_id:
            matched = [row for row in candidates if row.recording_id == asset.recording_id]
            if matched:
                candidates = matched
        candidates = [row for row in candidates if row.scene_video_name]
        if len(candidates) != 1:
            raise ValidationError(
                "The selected Pupil Cloud source does not resolve to one scene video; choose the exact recording first"
            )
        candidate = candidates[0]
        path = (root / candidate.scene_video_name).resolve()
        member = candidate.scene_video_name
        media_domain_id = _domain_by_role(session, asset, "cloud_scene_pts")
        capture_domain_id = _domain_by_role(session, asset, "pupil_clock") or asset.primary_time_domain_id
        metadata = {"recording_id": candidate.recording_id, "scene_video_member": candidate.scene_video_name}
    else:
        raise ValidationError("Shared-audio synchronization accepts Video, Pupil Native, or Pupil Cloud scene-video assets")

    if not media_domain_id:
        raise ValidationError("Selected video asset has no media-PTS time domain")
    if not path.is_file():
        raise ValidationError(f"Resolved video file is unavailable: {path}")
    return VideoInput(
        asset_id=asset.id,
        asset_label=asset.label,
        asset_kind=kind,
        source_path=asset.source_path,
        video_path=str(path),
        video_member=member,
        media_domain_id=media_domain_id,
        capture_domain_id=capture_domain_id,
        stamp=stamp,
        metadata=metadata,
    )


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "N/A"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _probe_streams(path: Path, *, ffprobe_path: str | None = None) -> dict[str, Any]:
    ffprobe = resolve_media_tool("ffprobe", ffprobe_path)
    command = [
        ffprobe.path,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            creationflags=_flags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProcessingError(f"Could not inspect video/audio streams: {exc}") from exc
    if proc.returncode:
        raise ProcessingError(f"ffprobe failed while inspecting audio: {proc.stderr[-2000:]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProcessingError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams") or []
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    audio = next((row for row in streams if row.get("codec_type") == "audio"), None)
    if video is None:
        raise ValidationError("Selected source contains no video stream")
    if audio is None:
        raise ValidationError("Selected source contains no audio stream; shared-audio synchronization cannot run")
    format_row = payload.get("format") or {}
    video_start = _float(video.get("start_time"), 0.0) or 0.0
    audio_start = _float(audio.get("start_time"), 0.0) or 0.0
    duration = _float(audio.get("duration"), None)
    if duration is None:
        duration = _float(format_row.get("duration"), None)
    return {
        "ffprobe_path": ffprobe.path,
        "ffprobe_version": ffprobe.version,
        "video_stream_index": int(video.get("index", 0)),
        "audio_stream_index": int(audio.get("index", 0)),
        "video_start_sec": float(video_start),
        "audio_start_sec": float(audio_start),
        "audio_to_video_offset_sec": float(audio_start - video_start),
        "audio_duration_sec": duration,
        "video_duration_sec": _float(video.get("duration"), _float(format_row.get("duration"), None)),
        "audio_sample_rate_hz": int(audio.get("sample_rate") or 0) or None,
        "audio_channels": int(audio.get("channels") or 0) or None,
        "audio_codec": audio.get("codec_name"),
        "video_codec": video.get("codec_name"),
    }


def _decode_audio(
    store: ProjectStore,
    video: VideoInput,
    *,
    decode_rate_hz: int,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    progress=None,
    cancellation: CancellationToken,
) -> tuple[np.ndarray, dict[str, Any]]:
    probe = _probe_streams(Path(video.video_path), ffprobe_path=ffprobe_path)
    ffmpeg = resolve_media_tool("ffmpeg", ffmpeg_path, sibling_hint=probe["ffprobe_path"])
    key = hashlib.sha256(
        json.dumps(
            {
                "engine": AUDIO_VIDEO_SYNC_ENGINE,
                "asset_id": video.asset_id,
                "source_stamp": video.stamp,
                "member": video.video_member,
                "decode_rate_hz": int(decode_rate_hz),
                "ffmpeg": ffmpeg.version,
                "audio_stream_index": probe["audio_stream_index"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:28]
    root = store.cache_dir() / "audio_video_sync_audio"
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / f"{key}.f32le"
    meta_path = root / f"{key}.json"
    if raw_path.is_file() and meta_path.is_file() and raw_path.stat().st_size >= decode_rate_hz * 4:
        metadata = json.loads(meta_path.read_text("utf-8"))
        samples = np.fromfile(raw_path, dtype="<f4")
        if int(metadata.get("sample_count") or 0) == len(samples):
            return samples.astype(np.float64, copy=False), metadata

    pending = raw_path.with_suffix(raw_path.suffix + ".partial")
    pending.unlink(missing_ok=True)
    command = [
        ffmpeg.path,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        video.video_path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(decode_rate_hz)),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        str(pending),
    ]
    emit(progress, 0.08, f"Decoding mono audio · {video.asset_label}")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=_flags(),
    )
    started = time.monotonic()
    try:
        while proc.poll() is None:
            cancellation.raise_if_cancelled()
            if time.monotonic() - started > 7200:
                raise ProcessingError("Audio extraction exceeded the two-hour safety limit")
            time.sleep(0.1)
        stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")
        if proc.returncode:
            raise ProcessingError(f"ffmpeg audio extraction failed ({proc.returncode}): {stderr[-3000:]}")
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        pending.unlink(missing_ok=True)
        raise
    if not pending.is_file() or pending.stat().st_size < decode_rate_hz * 4:
        pending.unlink(missing_ok=True)
        raise ValidationError("Decoded audio is shorter than one second")
    if pending.stat().st_size % 4:
        pending.unlink(missing_ok=True)
        raise ProcessingError("Decoded float audio has a truncated sample")
    pending.replace(raw_path)
    sample_count = raw_path.stat().st_size // 4
    metadata = {
        **probe,
        "ffmpeg_path": ffmpeg.path,
        "ffmpeg_version": ffmpeg.version,
        "decoded_sample_rate_hz": int(decode_rate_hz),
        "sample_count": int(sample_count),
        "decoded_duration_sec": float(sample_count / decode_rate_hz),
        "raw_cache_path": str(raw_path),
    }
    atomic_write_json(meta_path, metadata)
    samples = np.fromfile(raw_path, dtype="<f4")
    return samples.astype(np.float64, copy=False), metadata


def _robust_z(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(values) & mask
    if int(finite.sum()) < 20:
        finite = np.isfinite(values)
    centre = float(np.nanmedian(values[finite]))
    scale = 1.4826 * float(np.nanmedian(np.abs(values[finite] - centre)))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(values[finite]))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return (values - centre) / scale, centre, scale


def _audio_signature(
    samples: np.ndarray,
    metadata: dict[str, Any],
    *,
    decode_rate_hz: int,
    feature_rate_hz: float,
) -> AudioSignal:
    values = np.asarray(samples, dtype=np.float64)
    if len(values) < decode_rate_hz * 3:
        raise ValidationError("Audio is too short for automatic synchronization")
    finite = np.isfinite(values)
    if not finite.all():
        values = np.where(finite, values, 0.0)
    values -= float(np.mean(values))
    amplitude = float(np.percentile(np.abs(values), 99.5))
    if not math.isfinite(amplitude) or amplitude < 1e-6:
        raise ValidationError("Audio contains insufficient non-silent signal")
    values = np.clip(values / amplitude, -3.0, 3.0)
    nyquist = decode_rate_hz / 2.0
    high = min(3600.0, nyquist * 0.94)
    if high > 120.0:
        sos = butter(4, [70.0 / nyquist, high / nyquist], btype="bandpass", output="sos")
        values = sosfilt(sos, values)

    hop = max(1, int(round(decode_rate_hz / feature_rate_hz)))
    window = max(hop * 2, int(round(0.040 * decode_rate_hz)))
    energy = np.sqrt(np.maximum(uniform_filter1d(values * values, size=window, mode="nearest"), 0.0))
    derivative = np.diff(values, prepend=values[0])
    derivative_energy = np.sqrt(
        np.maximum(uniform_filter1d(derivative * derivative, size=window, mode="nearest"), 0.0)
    )
    centres = np.arange(window // 2, len(values), hop, dtype=np.int64)
    if len(centres) < int(feature_rate_hz * 3):
        raise ValidationError("Audio has too few feature frames")
    rms = energy[centres]
    drms = derivative_energy[centres]
    log_rms = np.log(np.maximum(rms, 1e-8))
    log_drms = np.log(np.maximum(drms, 1e-8))

    low = float(np.percentile(log_rms, 15))
    high_level = float(np.percentile(log_rms, 90))
    dynamic = high_level - low
    if not math.isfinite(dynamic) or dynamic < 0.12:
        raise ValidationError("Audio energy is too constant to identify a reliable shared pattern")
    threshold = low + 0.08 * dynamic
    active = log_rms >= threshold
    active = maximum_filter1d(active.astype(np.uint8), size=max(1, int(round(0.30 * feature_rate_hz)))) > 0
    if float(active.mean()) < 0.04:
        threshold = low + 0.03 * dynamic
        active = log_rms >= threshold
        active = maximum_filter1d(active.astype(np.uint8), size=max(1, int(round(0.30 * feature_rate_hz)))) > 0

    rms_hp = log_rms - gaussian_filter1d(log_rms, sigma=max(1.0, 1.25 * feature_rate_hz), mode="nearest")
    drms_hp = log_drms - gaussian_filter1d(log_drms, sigma=max(1.0, 0.75 * feature_rate_hz), mode="nearest")
    onset = np.maximum(np.gradient(log_rms), 0.0)
    onset = gaussian_filter1d(onset, sigma=max(0.8, 0.018 * feature_rate_hz), mode="nearest")
    rms_z, _, rms_scale = _robust_z(rms_hp, active)
    drms_z, _, drms_scale = _robust_z(drms_hp, active)
    onset_z, _, onset_scale = _robust_z(onset, active)
    signature = 0.52 * rms_z + 0.34 * drms_z + 0.14 * onset_z
    signature, signature_centre, signature_scale = _robust_z(signature, active)
    audio_offset = float(metadata.get("audio_to_video_offset_sec") or 0.0)
    grid = audio_offset + centres.astype(np.float64) / float(decode_rate_hz)
    valid = active & np.isfinite(signature) & (grid >= 0.0)
    video_duration = _float(metadata.get("video_duration_sec"), None)
    if video_duration is not None:
        valid &= grid <= float(video_duration) + (0.5 / float(feature_rate_hz))
    signature = signature.astype(float)
    signature[~valid] = np.nan
    return AudioSignal(
        grid_sec=grid,
        signature=signature,
        raw_energy=rms,
        valid=valid,
        sample_rate_hz=float(feature_rate_hz),
        metadata={
            **metadata,
            "feature_rate_hz": float(feature_rate_hz),
            "feature_count": int(len(grid)),
            "feature_start_video_sec": float(grid[0]),
            "feature_end_video_sec": float(grid[-1]),
            "feature_duration_sec": float(grid[-1] - grid[0]),
            "active_fraction": float(valid.mean()),
            "log_energy_dynamic_range": float(dynamic),
            "rms_component_scale": float(rms_scale),
            "derivative_component_scale": float(drms_scale),
            "onset_component_scale": float(onset_scale),
            "signature_centre": float(signature_centre),
            "signature_scale": float(signature_scale),
        },
    )


def _audio_windows(
    source: AudioSignal,
    reference: AudioSignal,
    alignment: GlobalAlignment,
) -> list[dict[str, Any]]:
    rate = float(source.sample_rate_hz)
    if not math.isclose(rate, float(reference.sample_rate_hz), rel_tol=1e-9):
        raise ValidationError("Internal audio feature rates differ")
    lag = int(alignment.lag_samples)
    overlap_start = max(0, -lag)
    overlap_end = min(len(source.signature), len(reference.signature) - lag)
    overlap_samples = overlap_end - overlap_start
    if overlap_samples <= int(rate * 6):
        return []
    overlap_sec = overlap_samples / rate
    window_sec = min(42.0, max(14.0, overlap_sec / 9.0))
    window = max(int(round(window_sec * rate)), int(rate * 10))
    stride = max(int(round(window * 0.45)), 1)
    residual_radius = max(int(round(2.0 * rate)), 1)
    exclusion = max(int(round(0.25 * rate)), 1)
    starts = list(range(overlap_start, max(overlap_start + 1, overlap_end - window + 1), stride))
    final_start = max(overlap_start, overlap_end - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts, start=1):
        end = min(overlap_end, start + window)
        if end - start < int(rate * 6):
            continue
        x = source.signature[start:end]
        candidates: list[tuple[int, float, int]] = []
        correlations: dict[int, float] = {}
        for residual in range(-residual_radius, residual_radius + 1):
            y_start = start + lag + residual
            y_end = y_start + len(x)
            if y_start < 0 or y_end > len(reference.signature):
                continue
            corr, count = _pearson(x, reference.signature[y_start:y_end])
            if np.isfinite(corr) and count >= max(int(rate * 3.0), int(0.12 * len(x))):
                correlations[int(residual)] = float(corr)
                candidates.append((int(residual), float(corr), int(count)))
        if not candidates:
            rows.append({
                "window_index": index,
                "source_video_sec": float(source.grid_sec[int((start + end - 1) // 2)]),
                "accepted": False,
                "rejection_reason": "insufficient common active audio",
                "window_duration_sec": float((end - start) / rate),
            })
            continue
        candidates.sort(key=lambda item: item[1], reverse=True)
        best_residual, best_corr, best_count = candidates[0]
        refinement = _refine_lag_peak(correlations, best_residual, sample_rate_hz=rate)
        peak_half_width_ms = float(refinement["lag_uncertainty_ms"] or (1000.0 / rate))
        uncertainty_ms = max(
            0.25 * float(refinement["lag_grid_ms"] or (1000.0 / rate)),
            peak_half_width_ms / math.sqrt(max(float(best_count), 1.0)),
        )
        fractional_residual = float(refinement["fractional_residual_samples"])
        refined_corr = float(refinement["refined_correlation"])
        separated = [item for item in candidates[1:] if abs(item[0] - best_residual) >= exclusion]
        second = max((item[1] for item in separated), default=-1.0)
        centre_index = int(round((start + end - 1) / 2))
        reference_index = centre_index + lag + fractional_residual
        source_video_sec = float(source.grid_sec[centre_index])
        reference_video_sec = float(reference.grid_sec[0] + reference_index / rate)
        peak_margin = float(refined_corr - second)
        source_activity = float(np.mean(np.isfinite(x)))
        y_nominal_start = start + lag + best_residual
        y_nominal_end = y_nominal_start + len(x)
        reference_activity = float(np.mean(np.isfinite(reference.signature[y_nominal_start:y_nominal_end])))
        accepted = bool(
            refined_corr >= 0.24
            and peak_margin >= 0.010
            and uncertainty_ms <= 120.0
            and source_activity >= 0.10
            and reference_activity >= 0.10
        )
        reason = None
        if not accepted:
            if refined_corr < 0.24:
                reason = "weak common audio pattern"
            elif peak_margin < 0.010:
                reason = "ambiguous repeated audio pattern"
            elif uncertainty_ms > 120.0:
                reason = "local lag is imprecise"
            else:
                reason = "insufficient active audio"
        rows.append({
            "window_index": index,
            "source_video_sec": source_video_sec,
            "reference_video_sec": reference_video_sec,
            "source_ns": int(round(source_video_sec * 1e9)),
            "target_ns": int(round(reference_video_sec * 1e9)),
            "global_lag_ms": float(lag / rate * 1000.0),
            "integer_residual_lag_samples": int(best_residual),
            "residual_lag_samples": fractional_residual,
            "residual_lag_ms": float(fractional_residual / rate * 1000.0),
            "local_offset_sec": float(reference_video_sec - source_video_sec),
            "local_offset_ms": float((reference_video_sec - source_video_sec) * 1000.0),
            "lag_grid_ms": float(refinement["lag_grid_ms"] or (1000.0 / rate)),
            "lag_uncertainty_ms": float(uncertainty_ms),
            "peak_half_width_ms": peak_half_width_ms,
            "peak_curvature": refinement["peak_curvature"],
            "correlation": refined_corr,
            "integer_peak_correlation": float(best_corr),
            "second_correlation": float(second),
            "peak_margin": peak_margin,
            "valid_sample_count": int(best_count),
            "source_activity_fraction": source_activity,
            "reference_activity_fraction": reference_activity,
            "window_duration_sec": float((end - start) / rate),
            "accepted": accepted,
            "rejection_reason": reason,
        })

    accepted_rows = [row for row in rows if row.get("accepted")]
    if len(accepted_rows) >= 3:
        offsets = np.asarray([float(row["local_offset_ms"]) for row in accepted_rows], dtype=float)
        times = np.asarray([float(row["source_video_sec"]) for row in accepted_rows], dtype=float)
        # Remove only gross outliers from a preliminary affine trend.  Different
        # commands/noises are expected, so no unmatched window is forced into the fit.
        x = times - float(np.median(times))
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, offsets, rcond=None)
        residual = offsets - design @ beta
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median)))
        precision = float(np.median([float(row["lag_uncertainty_ms"]) for row in accepted_rows]))
        gate = max(35.0, 4.5 * precision, 4.5 * mad)
        for row in rows:
            if not row.get("accepted"):
                continue
            expected = float(beta[0] + beta[1] * (float(row["source_video_sec"]) - float(np.median(times))))
            if abs(float(row["local_offset_ms"]) - expected) > gate:
                row["accepted"] = False
                row["rejection_reason"] = "local lag is inconsistent with the distributed affine trend"
    return rows


def _window_diagnostics(windows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in windows if row.get("accepted")]
    if len(accepted) < 2:
        return {
            "window_time_span_sec": 0.0,
            "window_local_offset_range_ms": None,
            "window_drift_ppm": None,
            "drift_ci95_ppm": None,
            "offset_ci95_ms": None,
            "median_lag_uncertainty_ms": None,
            "drift_significant": False,
        }
    x = np.asarray([float(row["source_video_sec"]) for row in accepted], dtype=float)
    y = np.asarray([float(row["reference_video_sec"]) - float(row["source_video_sec"]) for row in accepted], dtype=float)
    sigma = np.asarray([max(float(row.get("lag_uncertainty_ms") or 0.0) / 1000.0, 1e-6) for row in accepted])
    x_ref = float(np.median(x))
    design = np.column_stack([np.ones_like(x), x - x_ref])
    weights = 1.0 / np.maximum(sigma * sigma, 1e-12)
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(normal)
    beta = covariance @ (design.T @ (weights * y))
    slope = float(beta[1])
    slope_se = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    zero = np.asarray([1.0, -x_ref], dtype=float)
    offset_var = float(zero @ covariance @ zero)
    return {
        "window_time_span_sec": float(np.ptp(x)),
        "window_local_offset_range_ms": float(np.ptp(y) * 1000.0),
        "window_drift_ppm": float(slope * 1e6),
        "drift_ci95_ppm": float(1.96 * slope_se * 1e6),
        "offset_ci95_ms": float(1.96 * math.sqrt(max(offset_var, 0.0)) * 1000.0),
        "median_lag_uncertainty_ms": float(np.median(sigma) * 1000.0),
        "drift_significant": bool(abs(slope) > 1.96 * slope_se),
    }


def _leave_one_out(windows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_indices = [index for index, row in enumerate(windows) if row.get("accepted")]
    residuals: list[float] = []
    if len(accepted_indices) < 3:
        return {
            "holdout_window_count": 0,
            "cross_validated_median_residual_ms": None,
            "cross_validated_p95_residual_ms": None,
            "cross_validated_max_abs_residual_ms": None,
        }
    for held_index in accepted_indices:
        anchors = []
        for index in accepted_indices:
            if index == held_index:
                continue
            row = windows[index]
            anchors.append(TimeAnchor(
                source_ns=int(row["source_ns"]),
                target_ns=int(row["target_ns"]),
                label=f"Shared-audio LOO window {int(row['window_index'])}",
                weight=max(0.05, min(1.0, float(row["correlation"]) ** 2)),
            ))
        mapping = fit_time_map("audio_source", "audio_reference", anchors, method="shared_audio_leave_one_out")
        held = windows[held_index]
        predicted = int(apply_time_map(mapping, int(held["source_ns"])))
        residual_ms = float((int(held["target_ns"]) - predicted) / 1e6)
        held["cross_validated_residual_ms"] = residual_ms
        held["validation_mode"] = "leave_one_window_out"
        residuals.append(residual_ms)
    absolute = np.abs(np.asarray(residuals, dtype=float))
    return {
        "holdout_window_count": int(len(absolute)),
        "cross_validated_median_residual_ms": float(np.median(absolute)),
        "cross_validated_p95_residual_ms": float(np.percentile(absolute, 95)),
        "cross_validated_max_abs_residual_ms": float(np.max(absolute)),
    }


def _anchors(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in windows:
        if not row.get("accepted"):
            continue
        rows.append({
            "source_ns": int(row["source_ns"]),
            "target_ns": int(row["target_ns"]),
            "label": f"Shared-audio window {int(row['window_index'])}",
            "weight": max(0.05, min(1.0, float(row["correlation"]) ** 2)),
        })
    rows.sort(key=lambda row: row["source_ns"])
    return rows


def _mapping_metrics(
    mapping,
    *,
    source: AudioSignal,
    reference: AudioSignal,
    windows: list[dict[str, Any]],
    alignment: GlobalAlignment,
) -> dict[str, Any]:
    source_ns = np.rint(source.grid_sec * 1e9).astype(np.int64)
    mapped_sec = apply_time_map(mapping, source_ns).astype(np.float64) / 1e9
    reference_interp = np.interp(
        mapped_sec,
        reference.grid_sec,
        np.where(np.isfinite(reference.signature), reference.signature, 0.0),
        left=np.nan,
        right=np.nan,
    )
    reference_valid = np.interp(
        mapped_sec,
        reference.grid_sec,
        reference.valid.astype(float),
        left=0.0,
        right=0.0,
    ) >= 0.999
    reference_interp[~reference_valid] = np.nan
    aligned_corr, aligned_count = _pearson(source.signature, reference_interp)
    accepted = [row for row in windows if row.get("accepted")]
    in_sample = []
    for row in accepted:
        predicted = int(apply_time_map(mapping, int(row["source_ns"])))
        residual_ms = float((int(row["target_ns"]) - predicted) / 1e6)
        row["in_sample_residual_ms"] = residual_ms
        in_sample.append(residual_ms)
    in_sample_abs = np.abs(np.asarray(in_sample, dtype=float))
    cv = _leave_one_out(windows)
    diagnostics = _window_diagnostics(windows)
    p95 = cv.get("cross_validated_p95_residual_ms")
    if p95 is None and len(in_sample_abs):
        p95 = float(np.percentile(in_sample_abs, 95))
    median = cv.get("cross_validated_median_residual_ms")
    if median is None and len(in_sample_abs):
        median = float(np.median(in_sample_abs))
    maximum = cv.get("cross_validated_max_abs_residual_ms")
    if maximum is None and len(in_sample_abs):
        maximum = float(np.max(in_sample_abs))
    span = float(diagnostics.get("window_time_span_sec") or 0.0)
    overlap = float(alignment.overlap_sec)
    coverage = span / overlap if overlap > 0 else 0.0
    true_global_offset = float(reference.grid_sec[0] - source.grid_sec[0] + alignment.delta_sec)
    return {
        "aligned_correlation": float(aligned_corr) if np.isfinite(aligned_corr) else None,
        "aligned_sample_count": int(aligned_count),
        "estimated_offset_sec": float(apply_time_map(mapping, 0) / 1e9),
        "coarse_global_offset_sec": true_global_offset,
        "estimated_drift_ppm": float((mapping.scale - 1.0) * 1e6),
        "window_count": int(len(windows)),
        "matched_window_count": int(len(accepted)),
        "rejected_window_count": int(len(windows) - len(accepted)),
        "median_window_correlation": float(np.median([row["correlation"] for row in accepted])) if accepted else None,
        "median_window_peak_margin": float(np.median([row["peak_margin"] for row in accepted])) if accepted else None,
        "in_sample_median_residual_ms": float(np.median(in_sample_abs)) if len(in_sample_abs) else None,
        "in_sample_p95_residual_ms": float(np.percentile(in_sample_abs, 95)) if len(in_sample_abs) else None,
        "in_sample_max_abs_residual_ms": float(np.max(in_sample_abs)) if len(in_sample_abs) else None,
        "median_residual_ms": median,
        "p95_residual_ms": p95,
        "maximum_abs_residual_ms": maximum,
        "residual_validation_mode": "leave_one_window_out" if int(cv.get("holdout_window_count") or 0) else "in_sample_fallback",
        "matched_time_coverage_fraction": float(max(0.0, min(1.0, coverage))),
        **cv,
        **diagnostics,
    }


def _confidence(
    *,
    alignment: GlobalAlignment,
    metrics: dict[str, Any],
    source: AudioSignal,
    reference: AudioSignal,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    overlap = float(alignment.overlap_sec)
    matched = int(metrics.get("matched_window_count") or 0)
    holdout = int(metrics.get("holdout_window_count") or 0)
    required = 3 if overlap < 45 else 5 if overlap < 150 else 6
    p95 = metrics.get("p95_residual_ms")
    uncertainty = float(metrics.get("median_lag_uncertainty_ms") or 0.0)
    residual_limit = max(35.0, 3.5 * uncertainty)
    drift = abs(float(metrics.get("estimated_drift_ppm") or 0.0))
    high_checks = {
        "global audio correlation": alignment.correlation >= 0.30,
        "unique coarse peak": alignment.peak_margin >= 0.020,
        "aligned audio correlation": metrics.get("aligned_correlation") is not None and float(metrics["aligned_correlation"]) >= 0.40,
        "distributed matched windows": matched >= required,
        "independent holdout windows": holdout >= max(3, required - 1),
        "local audio correlation": metrics.get("median_window_correlation") is not None and float(metrics["median_window_correlation"]) >= 0.32,
        "holdout residual": p95 is not None and float(p95) <= residual_limit,
        "time coverage": float(metrics.get("matched_time_coverage_fraction") or 0.0) >= 0.55,
        "source audio activity": float(source.metadata.get("active_fraction") or 0.0) >= 0.05,
        "reference audio activity": float(reference.metadata.get("active_fraction") or 0.0) >= 0.05,
        "clock drift": drift <= 5000.0,
        "usable overlap": overlap >= 12.0,
    }
    if all(high_checks.values()):
        reasons.append("Distributed common-audio windows support one affine video-clock map across the recording")
        if int(metrics.get("rejected_window_count") or 0):
            reasons.append("Low-agreement, ambiguous, or affine-inconsistent windows were excluded from the fit")
        else:
            reasons.append("All accepted windows supported the same lag; short one-sided sounds did not move the fitted map")
        reasons.append("Audio establishes physical co-recording time; start/middle/end visual A/V review remains required")
        return "HIGH", reasons
    for name, passed in high_checks.items():
        if not passed:
            reasons.append(f"High-confidence gate not met: {name}")
    review = bool(
        alignment.correlation >= 0.18
        and alignment.peak_margin >= 0.006
        and metrics.get("aligned_correlation") is not None
        and float(metrics["aligned_correlation"]) >= 0.24
        and overlap >= 8.0
        and matched >= 2
        and holdout >= 1
        and drift <= 15000.0
    )
    if review:
        reasons.append("A plausible common-audio mapping was found, but visual confirmation is required before acceptance")
        return "REVIEW", reasons
    reasons.insert(0, "Shared audio is not sufficiently identifiable across the selected videos")
    return "FAILED", reasons


def _candidate_root(store: ProjectStore, candidate_id: str) -> Path:
    root = store.cache_dir() / "audio_video_sync" / safe_name(candidate_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    result["anchors"] = [
        {**row, "source_ns": str(row["source_ns"]), "target_ns": str(row["target_ns"])}
        for row in candidate.get("anchors") or []
    ]
    result.pop("signal_rows", None)
    return result


def _write_candidate(store: ProjectStore, candidate: dict[str, Any]) -> Path:
    root = _candidate_root(store, candidate["candidate_id"])
    atomic_write_json(root / "audio_video_sync_candidate.json", candidate)
    pd.DataFrame(candidate.get("windows") or []).to_csv(root / "audio_video_sync_windows.csv", index=False)
    pd.DataFrame(candidate.get("signal_rows") or []).to_csv(root / "audio_video_sync_alignment.csv", index=False)
    return root


def _load_candidate(store: ProjectStore, candidate_id: str) -> tuple[dict[str, Any], Path]:
    root = _candidate_root(store, candidate_id)
    path = root / "audio_video_sync_candidate.json"
    if not path.is_file():
        raise ValidationError("Shared-audio synchronization candidate has expired or is unavailable")
    return json.loads(path.read_text("utf-8")), root


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def automatic_shared_audio_video_sync(
    store: ProjectStore,
    *,
    session_id: str,
    source_video_asset_id: str,
    reference_video_asset_id: str,
    source_video_member: str | None = None,
    reference_video_member: str | None = None,
    wearer_role_id: str | None = None,
    maximum_offset_sec: float = 300.0,
    decode_rate_hz: int = AUDIO_DECODE_RATE_HZ,
    feature_rate_hz: float = AUDIO_FEATURE_RATE_HZ,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    if source_video_asset_id == reference_video_asset_id:
        raise ValidationError("Source facial video and reference ego video must be different assets")
    if maximum_offset_sec <= 0 or maximum_offset_sec > 7200:
        raise ValidationError("Maximum audio offset search must be between 0 and 7200 seconds")
    if decode_rate_hz < 4000 or decode_rate_hz > 48000:
        raise ValidationError("Audio decode rate must be between 4 kHz and 48 kHz")
    if feature_rate_hz < 25 or feature_rate_hz > 200:
        raise ValidationError("Audio feature rate must be between 25 and 200 Hz")
    _validate_role_context(
        store,
        session_id=session_id,
        source_asset_id=source_video_asset_id,
        reference_asset_id=reference_video_asset_id,
        wearer_role_id=wearer_role_id,
    )

    emit(progress, 0.02, "Resolving facial-expression and ego video sources")
    source_input = _resolve_video_input(
        store,
        session_id=session_id,
        asset_id=source_video_asset_id,
        requested_member=source_video_member,
    )
    reference_input = _resolve_video_input(
        store,
        session_id=session_id,
        asset_id=reference_video_asset_id,
        requested_member=reference_video_member,
    )
    if source_input.media_domain_id == reference_input.media_domain_id:
        raise ValidationError("The two videos already use the same media clock")
    token.raise_if_cancelled()

    source_samples, source_decode = _decode_audio(
        store,
        source_input,
        decode_rate_hz=decode_rate_hz,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        progress=progress,
        cancellation=token,
    )
    emit(progress, 0.18, "Decoding reference ego-video audio")
    reference_samples, reference_decode = _decode_audio(
        store,
        reference_input,
        decode_rate_hz=decode_rate_hz,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        progress=progress,
        cancellation=token,
    )
    token.raise_if_cancelled()

    emit(progress, 0.34, "Building microphone-robust audio pattern signatures")
    source_signal = _audio_signature(
        source_samples,
        source_decode,
        decode_rate_hz=decode_rate_hz,
        feature_rate_hz=feature_rate_hz,
    )
    reference_signal = _audio_signature(
        reference_samples,
        reference_decode,
        decode_rate_hz=decode_rate_hz,
        feature_rate_hz=feature_rate_hz,
    )
    minimum_overlap = min(
        45.0,
        max(8.0, 0.20 * min(source_signal.metadata["feature_duration_sec"], reference_signal.metadata["feature_duration_sec"])),
    )
    emit(progress, 0.50, "Searching the shared acoustic offset")
    alignment = _global_alignment(
        source_signal,
        reference_signal,
        maximum_offset_sec=maximum_offset_sec,
        minimum_overlap_sec=minimum_overlap,
        peak_exclusion_sec=2.0,
    )
    token.raise_if_cancelled()

    emit(progress, 0.66, "Matching distributed common-audio windows and rejecting non-shared instructions")
    windows = _audio_windows(source_signal, reference_signal, alignment)
    anchors = _anchors(windows)
    if len(anchors) < 2:
        confidence = "FAILED"
        reasons = [
            "Fewer than two distributed common-audio windows survived quality gates",
            "The TimeGraph was not modified",
        ]
        metrics = {
            "engine": AUDIO_VIDEO_SYNC_ENGINE,
            "global_correlation": alignment.correlation,
            "second_best_correlation": alignment.second_correlation,
            "global_peak_margin": alignment.peak_margin,
            "usable_overlap_sec": alignment.overlap_sec,
            "window_count": len(windows),
            "matched_window_count": len(anchors),
            "source_audio": source_signal.metadata,
            "reference_audio": reference_signal.metadata,
        }
        preview_map = None
    else:
        mapping_preview = fit_time_map(
            source_input.media_domain_id,
            reference_input.media_domain_id,
            [TimeAnchor.model_validate(row) for row in anchors],
            method="shared_audio_candidate",
        )
        metrics = {
            "engine": AUDIO_VIDEO_SYNC_ENGINE,
            "global_correlation": alignment.correlation,
            "second_best_correlation": alignment.second_correlation,
            "global_peak_margin": alignment.peak_margin,
            "global_lag_samples": alignment.lag_samples,
            "global_lag_sec_relative_to_audio_starts": alignment.delta_sec,
            "usable_overlap_sec": alignment.overlap_sec,
            "search_maximum_offset_sec": float(maximum_offset_sec),
            "decode_rate_hz": int(decode_rate_hz),
            "feature_rate_hz": float(feature_rate_hz),
            "anchor_count": int(len(anchors)),
            "preview_rms_residual_ms": float(mapping_preview.quality.rms_residual_ns / 1e6),
            "preview_max_abs_residual_ms": float(mapping_preview.quality.max_abs_residual_ns / 1e6),
            **_mapping_metrics(
                mapping_preview,
                source=source_signal,
                reference=reference_signal,
                windows=windows,
                alignment=alignment,
            ),
            "source_audio": source_signal.metadata,
            "reference_audio": reference_signal.metadata,
            "precision_audit_version": "audio-feature-subsample+leave-one-window-out+drift-ci-v1",
        }
        confidence, reasons = _confidence(
            alignment=alignment,
            metrics=metrics,
            source=source_signal,
            reference=reference_signal,
        )
        preview_map = mapping_preview.model_dump(mode="json")

    candidate_id = new_id("audiosync")
    signal_rows: list[dict[str, Any]] = []
    if preview_map:
        mapping_preview = fit_time_map(
            source_input.media_domain_id,
            reference_input.media_domain_id,
            [TimeAnchor.model_validate(row) for row in anchors],
            method="shared_audio_candidate",
        )
        stride = max(1, int(math.ceil(len(source_signal.grid_sec) / 12000)))
        source_ns_grid = np.rint(source_signal.grid_sec * 1e9).astype(np.int64)
        mapped_reference_sec = apply_time_map(mapping_preview, source_ns_grid).astype(np.float64) / 1e9
        reference_interp = np.interp(
            mapped_reference_sec,
            reference_signal.grid_sec,
            np.where(np.isfinite(reference_signal.signature), reference_signal.signature, 0.0),
            left=np.nan,
            right=np.nan,
        )
        signal_rows = [
            {
                "source_video_sec": float(source_signal.grid_sec[index]),
                "source_signature": float(source_signal.signature[index]) if np.isfinite(source_signal.signature[index]) else None,
                "mapped_reference_video_sec": float(mapped_reference_sec[index]),
                "reference_signature": float(reference_interp[index]) if np.isfinite(reference_interp[index]) else None,
            }
            for index in range(0, len(source_signal.grid_sec), stride)
        ]

    candidate = {
        "schema_version": "oculostudio-audio-video-sync-candidate-1.0",
        "engine": AUDIO_VIDEO_SYNC_ENGINE,
        "candidate_id": candidate_id,
        "session_id": session_id,
        "wearer_role_id": str(wearer_role_id or "").upper() or None,
        "source_asset_id": source_input.asset_id,
        "source_asset_label": source_input.asset_label,
        "source_asset_kind": source_input.asset_kind,
        "source_video_member": source_input.video_member,
        "source_domain_id": source_input.media_domain_id,
        "source_stamp": source_input.stamp,
        "reference_asset_id": reference_input.asset_id,
        "reference_asset_label": reference_input.asset_label,
        "reference_asset_kind": reference_input.asset_kind,
        "reference_video_member": reference_input.video_member,
        "reference_domain_id": reference_input.media_domain_id,
        "reference_capture_domain_id": reference_input.capture_domain_id,
        "reference_stamp": reference_input.stamp,
        "confidence": confidence,
        "anchors": anchors,
        "windows": windows,
        "metrics": metrics,
        "reasons": reasons,
        "preview_time_map": preview_map,
        "signal_rows": signal_rows,
        "visual_review_required": True,
        "created_at": utc_now(),
    }
    _write_candidate(store, candidate)
    emit(progress, 1.0, f"Shared-audio synchronization candidate: {confidence}")
    return {
        "status": "review_required" if confidence in {"HIGH", "REVIEW"} else "failed",
        "confidence": confidence,
        "candidate": _public_candidate(candidate),
        "message": (
            "A distributed shared-audio mapping is ready. Review start, middle and end video alignment before accepting."
            if confidence in {"HIGH", "REVIEW"}
            else "Shared audio could not identify a safe affine mapping. No TimeGraph was changed."
        ),
    }


def get_audio_video_sync_preview(
    store: ProjectStore,
    *,
    session_id: str,
    candidate_id: str | None = None,
    artifact_id: str | None = None,
    maximum_samples: int = 2200,
) -> dict[str, Any]:
    if bool(candidate_id) == bool(artifact_id):
        raise ValidationError("Provide exactly one shared-audio candidate or accepted artifact")
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    accepted = None
    if candidate_id:
        candidate, root = _load_candidate(store, candidate_id)
    else:
        artifact = session.artifacts.get(str(artifact_id))
        if artifact is None or artifact.kind != ArtifactKind.TIME_SYNC:
            raise ValidationError("Selected shared-audio synchronization artifact is unavailable")
        paths = {Path(row.path).name: Path(row.path) for row in artifact.files if Path(row.path).exists()}
        candidate_path = paths.get("audio_video_sync_candidate.json")
        if candidate_path is None:
            raise ValidationError("This Time Sync artifact does not contain shared-audio evidence")
        candidate = json.loads(candidate_path.read_text("utf-8"))
        root = candidate_path.parent
        accepted_path = paths.get("audio_video_sync_accepted.json")
        accepted = json.loads(accepted_path.read_text("utf-8")) if accepted_path else None
    if candidate.get("session_id") != session_id:
        raise ValidationError("Shared-audio synchronization belongs to another session")
    signals = pd.read_csv(root / "audio_video_sync_alignment.csv") if (root / "audio_video_sync_alignment.csv").exists() else pd.DataFrame()
    windows = pd.read_csv(root / "audio_video_sync_windows.csv") if (root / "audio_video_sync_windows.csv").exists() else pd.DataFrame(candidate.get("windows") or [])
    maximum_samples = int(max(100, min(int(maximum_samples), 10000)))
    if len(signals) > maximum_samples:
        indices = np.unique(np.rint(np.linspace(0, len(signals) - 1, maximum_samples)).astype(np.int64))
        signals = signals.iloc[indices]
    return {
        "candidate_id": candidate.get("candidate_id"),
        "artifact_id": artifact_id,
        "confidence": candidate.get("confidence"),
        "wearer_role_id": candidate.get("wearer_role_id"),
        "source_asset_id": candidate.get("source_asset_id"),
        "source_asset_label": candidate.get("source_asset_label"),
        "reference_asset_id": candidate.get("reference_asset_id"),
        "reference_asset_label": candidate.get("reference_asset_label"),
        "metrics": candidate.get("metrics") or {},
        "reasons": candidate.get("reasons") or [],
        "preview_time_map": candidate.get("preview_time_map"),
        "signals": [
            {str(key): _safe_json(value) for key, value in row.items()}
            for row in signals.to_dict(orient="records")
        ],
        "windows": [
            {str(key): _safe_json(value) for key, value in row.items()}
            for row in windows.to_dict(orient="records")
        ],
        "accepted": accepted,
        "visual_review_required": True,
    }


def _validate_candidate_sources(store: ProjectStore, candidate: dict[str, Any]) -> None:
    session = store.project.sessions[candidate["session_id"]]
    for prefix in ("source", "reference"):
        asset = session.assets.get(candidate[f"{prefix}_asset_id"])
        if asset is None:
            raise ValidationError(f"{prefix.title()} video asset is no longer registered")
        if source_stamp(asset.source_path) != candidate.get(f"{prefix}_stamp"):
            raise ValidationError(f"{prefix.title()} video source changed after synchronization; rerun shared-audio matching")


def _native_pts_bridge(
    store: ProjectStore,
    *,
    session_id: str,
    candidate: dict[str, Any],
    ffprobe_path: str | None,
) -> tuple[Any | None, dict[str, Any] | None]:
    if candidate.get("reference_asset_kind") != "pupil_native":
        return None, None
    media_domain_id = candidate.get("reference_domain_id")
    capture_domain_id = candidate.get("reference_capture_domain_id")
    if not media_domain_id or not capture_domain_id or media_domain_id == capture_domain_id:
        return None, None
    session = store.project.sessions[session_id]
    asset = session.assets[candidate["reference_asset_id"]]
    clock = build_clock(
        asset.source_path,
        store.cache_dir(),
        video_member=candidate.get("reference_video_member"),
        ffprobe_path=ffprobe_path,
    )
    count = min(17, len(clock.media_ns))
    indices = np.unique(np.rint(np.linspace(0, len(clock.media_ns) - 1, count)).astype(np.int64))
    anchors = [
        {
            "source_ns": int(clock.media_ns[index]),
            "target_ns": int(clock.capture_ns[index]),
            "label": f"Native scene frame {int(index)} media PTS ↔ capture time",
            "weight": 1.0,
        }
        for index in indices
    ]
    mapping = fit_and_store_time_map(
        store,
        session_id=session_id,
        source_domain_id=media_domain_id,
        target_domain_id=capture_domain_id,
        anchors=anchors,
        method="native_scene_pts_to_pupil_capture_affine",
    )
    return mapping, {
        "anchor_count": len(anchors),
        "rms_residual_ms": float(mapping.quality.rms_residual_ns / 1e6),
        "max_abs_residual_ms": float(mapping.quality.max_abs_residual_ns / 1e6),
    }


def accept_audio_video_sync(
    store: ProjectStore,
    *,
    session_id: str,
    candidate_id: str,
    visual_offset_adjustment_ms: float = 0.0,
    visual_review_confirmed: bool = False,
    ffprobe_path: str | None = None,
) -> dict[str, Any]:
    candidate, root = _load_candidate(store, candidate_id)
    if candidate.get("session_id") != session_id:
        raise ValidationError("Shared-audio synchronization candidate belongs to another session")
    if candidate.get("confidence") == "FAILED":
        raise ValidationError("A failed shared-audio synchronization candidate cannot be accepted")
    if not visual_review_confirmed:
        raise ValidationError(
            "Review start, middle and end in the split-video player and confirm visual alignment before publishing this TimeGraph"
        )
    if not math.isfinite(float(visual_offset_adjustment_ms)) or abs(float(visual_offset_adjustment_ms)) > 2000:
        raise ValidationError("Visual A/V correction must be within ±2000 ms")
    _validate_candidate_sources(store, candidate)
    correction_ns = int(round(float(visual_offset_adjustment_ms) * 1e6))
    corrected_anchors = [
        {**row, "target_ns": int(row["target_ns"]) + correction_ns}
        for row in candidate.get("anchors") or []
    ]
    if len(corrected_anchors) < 2:
        raise ValidationError("Shared-audio candidate has too few accepted anchors")

    # Acceptance replaces at most two direct maps.  Preserve their exact prior
    # records so an artifact-writing or Native-bridge failure cannot leave a
    # half-published TimeGraph.
    snapshot = store.snapshot().sessions[session_id]
    pairs = {(candidate["source_domain_id"], candidate["reference_domain_id"])}
    reference_capture = candidate.get("reference_capture_domain_id")
    if reference_capture and reference_capture != candidate.get("reference_domain_id"):
        pairs.add((candidate["reference_domain_id"], reference_capture))
    prior_maps = [
        deepcopy(row) for row in snapshot.time_maps.values()
        if (row.source_domain_id, row.target_domain_id) in pairs
    ]
    artifact_id = new_id("artifact")
    output_root: Path | None = None
    mapping = None
    bridge = None
    artifact = None
    try:
        mapping = fit_and_store_time_map(
            store,
            session_id=session_id,
            source_domain_id=candidate["source_domain_id"],
            target_domain_id=candidate["reference_domain_id"],
            anchors=corrected_anchors,
            method="shared_audio_reviewed_affine",
        )
        bridge, bridge_qc = _native_pts_bridge(
            store,
            session_id=session_id,
            candidate=candidate,
            ffprobe_path=ffprobe_path,
        )

        output_root = store.derived_dir(session_id, "time_sync", artifact_id)
        for path in root.iterdir():
            if path.is_file():
                shutil.copy2(path, output_root / path.name)
        metrics = dict(candidate.get("metrics") or {})
        accepted = {
            "schema_version": "oculostudio-audio-video-sync-accepted-1.0",
            "engine": AUDIO_VIDEO_SYNC_ENGINE,
            "candidate_id": candidate_id,
            "confidence": candidate.get("confidence"),
            "wearer_role_id": candidate.get("wearer_role_id"),
            "source_asset_id": candidate.get("source_asset_id"),
            "reference_asset_id": candidate.get("reference_asset_id"),
            "visual_review_confirmed": True,
            "visual_offset_adjustment_ms": float(visual_offset_adjustment_ms),
            "audio_mapping_accepted": True,
            "visual_validation_status": "confirmed",
            "time_map": mapping.model_dump(mode="json"),
            "reference_pts_bridge_time_map": bridge.model_dump(mode="json") if bridge else None,
            "reference_pts_bridge_qc": bridge_qc,
            "metrics": metrics,
            "reasons": candidate.get("reasons") or [],
            "created_at": utc_now(),
        }
        atomic_write_json(output_root / "audio_video_sync_accepted.json", accepted)
        session = store.project.sessions[session_id]
        source = session.assets[candidate["source_asset_id"]]
        reference = session.assets[candidate["reference_asset_id"]]
        artifact = ArtifactStore(store).register(
            session_id=session_id,
            kind=ArtifactKind.TIME_SYNC,
            label=f"Shared-audio video sync · {source.label} → {reference.label}",
            output_root=output_root,
            source_asset_ids=[source.id, reference.id],
            parameters={
                "engine": AUDIO_VIDEO_SYNC_ENGINE,
                "sync_type": "shared_audio_video",
                "source_asset_id": source.id,
                "reference_asset_id": reference.id,
                "source_domain_id": candidate["source_domain_id"],
                "reference_domain_id": candidate["reference_domain_id"],
                "candidate_id": candidate_id,
                "time_map_id": mapping.id,
                "reference_pts_bridge_time_map_id": bridge.id if bridge else None,
                "wearer_role_id": candidate.get("wearer_role_id"),
                "visual_offset_adjustment_ms": float(visual_offset_adjustment_ms),
                "visual_review_confirmed": True,
            },
            qc={
                "confidence": candidate.get("confidence"),
                "sync_verified": True,
                "production_ready": True,
                "sync_type": "shared_audio_video",
                "audio_mapping_accepted": True,
                "visual_review_confirmed": True,
                "visual_review_required": False,
                "visual_offset_adjustment_ms": float(visual_offset_adjustment_ms),
                "time_map_id": mapping.id,
                "reference_pts_bridge_time_map_id": bridge.id if bridge else None,
                "reference_pts_bridge_qc": bridge_qc,
                **metrics,
            },
            provenance={
                "method": mapping.method,
                "automatic_signal": "Distributed shared audio-pattern matching between simultaneously recorded videos",
                "clock_estimation_inputs": ["source video audio track", "reference ego-video audio track"],
                "non_shared_audio_policy": "low-agreement or affine-inconsistent windows are excluded; short one-sided sound may be tolerated when the remaining window uniquely supports the same lag",
                "arbitrary_dynamic_time_warping_used": False,
                "visual_av_latency_note": "Device-internal audio/video latency can remain after audio alignment; start/middle/end visual confirmation is mandatory.",
            },
            artifact_id=artifact_id,
        )
        candidate["accepted_time_map_id"] = mapping.id
        candidate["accepted_artifact_id"] = artifact.id
        candidate["visual_offset_adjustment_ms"] = float(visual_offset_adjustment_ms)
        candidate["visual_review_confirmed"] = True
        atomic_write_json(root / "audio_video_sync_candidate.json", candidate)
    except BaseException:
        def rollback(project):
            session = project.sessions[session_id]
            for map_id, row in list(session.time_maps.items()):
                if (row.source_domain_id, row.target_domain_id) in pairs:
                    del session.time_maps[map_id]
            for row in prior_maps:
                session.time_maps[row.id] = row
            session.artifacts.pop(artifact_id, None)
        store.update(rollback)
        if output_root is not None:
            shutil.rmtree(output_root, ignore_errors=True)
        raise

    return {
        "status": "saved",
        "candidate": _public_candidate(candidate),
        "time_map": mapping.model_dump(mode="json"),
        "reference_pts_bridge_time_map": bridge.model_dump(mode="json") if bridge else None,
        "artifact": artifact.model_dump(mode="json"),
    }


def audio_video_sync_video_path(
    store: ProjectStore,
    *,
    session_id: str,
    candidate_id: str,
    side: str,
) -> Path:
    candidate, _root = _load_candidate(store, candidate_id)
    if candidate.get("session_id") != session_id:
        raise ValidationError("Shared-audio synchronization candidate belongs to another session")
    if side not in {"source", "reference"}:
        raise ValidationError("Video side must be source or reference")
    video = _resolve_video_input(
        store,
        session_id=session_id,
        asset_id=candidate[f"{side}_asset_id"],
        requested_member=candidate.get(f"{side}_video_member"),
    )
    if video.stamp != candidate.get(f"{side}_stamp"):
        raise ValidationError(f"{side.title()} video changed after synchronization; rerun matching")
    return Path(video.video_path)
