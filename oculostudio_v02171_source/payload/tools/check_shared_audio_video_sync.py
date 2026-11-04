"""Installed-package validation for robust shared-audio video synchronization."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

import oculostudio
from oculostudio.adapters.video import frame_timestamps_ns
from oculostudio.errors import ValidationError
from oculostudio.services import (
    accept_audio_video_sync,
    add_asset,
    audio_video_sync_video_path,
    automatic_shared_audio_video_sync,
    get_audio_video_sync_preview,
)
from oculostudio.storage import ProjectStore
from oculostudio.util import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "VALIDATION_v0_21_61_SHARED_AUDIO_VIDEO_SYNC.json"


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _instruction(duration: float, sample_rate: int, seed: int, base: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(round(duration * sample_rate)), dtype=float) / sample_rate
    envelope = (0.25 + 0.75 * (np.sin(2 * np.pi * 3.7 * time) > -0.35))
    envelope *= np.sin(np.pi * np.clip(time / max(duration, 1e-6), 0, 1)) ** 0.5
    signal = 0.18 * envelope * (
        np.sin(2 * np.pi * base * time)
        + 0.5 * np.sin(2 * np.pi * 2.2 * base * time + 0.5)
    )
    signal += 0.015 * rng.standard_normal(len(time))
    return np.tanh(signal * 2.0)


def _shared_physical_signal(duration: float, sample_rate: int) -> np.ndarray:
    rng = np.random.default_rng(48291)
    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=float) / sample_rate
    signal = np.zeros(count, dtype=float)
    coarse = rng.uniform(0, 1, int(duration * 100) + 1)
    for start in np.arange(0, duration, 4.0):
        if rng.random() >= 0.82:
            continue
        length = float(rng.uniform(1.0, 3.4))
        first = int(start * sample_rate)
        last = min(count, int((start + length) * sample_rate))
        local = time[first:last] - start
        fundamental = float(rng.uniform(90, 220))
        modulation = 0.45 + 0.35 * np.sin(
            2 * np.pi * rng.uniform(2.5, 6.0) * local + rng.uniform(0, 2 * np.pi)
        )
        carrier = (
            np.sin(2 * np.pi * fundamental * local)
            + 0.55 * np.sin(2 * np.pi * 2.07 * fundamental * local + 0.2)
            + 0.25 * np.sin(2 * np.pi * 3.18 * fundamental * local + 0.8)
        )
        envelope = np.sin(np.pi * np.clip(local / length, 0, 1)) ** 0.7
        syllables = 0.30 + 0.70 * (
            np.sin(2 * np.pi * rng.uniform(2, 5) * local + rng.uniform(0, 2 * np.pi)) > -0.2
        )
        signal[first:last] += 0.12 * envelope * modulation * syllables * carrier
    locations = rng.choice(np.arange(1, duration - 1, 0.25), size=70, replace=False)
    for second in locations:
        first = int(second * sample_rate)
        length = min(int(0.09 * sample_rate), count - first)
        local = np.arange(length, dtype=float) / sample_rate
        signal[first:first + length] += (
            float(rng.uniform(0.05, 0.22))
            * np.exp(-local * rng.uniform(28, 75))
            * np.sin(2 * np.pi * rng.uniform(220, 2200) * local + rng.uniform(0, 2 * np.pi))
        )
    room_noise = rng.standard_normal(count)
    band = butter(3, [80 / (sample_rate / 2), 3300 / (sample_rate / 2)], btype="bandpass", output="sos")
    room_noise = sosfilt(band, room_noise)
    noise_envelope = np.interp(time, np.linspace(0, duration, len(coarse)), coarse)
    signal += 0.012 * (0.2 + noise_envelope) * room_noise
    return np.tanh(signal * 2.4)


def _device_track(
    physical: np.ndarray,
    *,
    physical_duration: float,
    sample_rate: int,
    intro_sec: float,
    clock_scale: float,
    outro_sec: float,
    seed: int,
    source: bool,
) -> np.ndarray:
    duration = intro_sec + physical_duration / clock_scale + outro_sec
    rng = np.random.default_rng(seed)
    local_time = np.arange(int(round(duration * sample_rate)), dtype=float) / sample_rate
    physical_time = clock_scale * (local_time - intro_sec)
    track = np.zeros_like(local_time)
    valid = (physical_time >= 0) & (physical_time < physical_duration)
    physical_grid = np.arange(len(physical), dtype=float) / sample_rate
    track[valid] = np.interp(physical_time[valid], physical_grid, physical)
    if source:
        colour = butter(3, [120 / (sample_rate / 2), 3000 / (sample_rate / 2)], btype="bandpass", output="sos")
        track = np.tanh(sosfilt(colour, track) * 1.02)
    else:
        colour = butter(3, [70 / (sample_rate / 2), 2400 / (sample_rate / 2)], btype="bandpass", output="sos")
        track = np.tanh(sosfilt(colour, track) * 1.16)
    track += 0.006 * rng.standard_normal(len(track))
    # Different pre-interaction instructions exist only on one device.
    opening = _instruction(min(intro_sec * 0.72 + 0.8, 3.0), sample_rate, seed + 100, 145 if source else 235)
    first = int(0.25 * sample_rate)
    track[first:first + len(opening)] += opening[: max(0, len(track) - first)]
    # A second one-sided instruction replaces a short middle interval.  The
    # affine mapping must remain correct even though the soundtracks are not identical.
    physical_centre = 25.0 if source else 52.0
    centre = int((intro_sec + physical_centre / clock_scale) * sample_rate)
    half = int(2.3 * sample_rate)
    lo, hi = max(0, centre - half), min(len(track), centre + half)
    unique = _instruction((hi - lo) / sample_rate, sample_rate, seed + 200, 310 if source else 420)
    track[lo:hi] = unique[: hi - lo] + 0.003 * rng.standard_normal(hi - lo)
    return np.clip(track, -0.98, 0.98)


def _write_video(root: Path, name: str, samples: np.ndarray, sample_rate: int, colour: str) -> Path:
    wav = root / f"{name}.wav"
    wavfile.write(wav, sample_rate, (samples * 32767).astype(np.int16))
    duration = len(samples) / sample_rate
    video = root / f"{name}.mp4"
    _run([
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:r=30:d={duration:.9f}",
        "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(video),
    ])
    return video


def _native_zip(root: Path, ego_video: Path) -> Path:
    pts = frame_timestamps_ns(ego_video)
    epoch = 1_771_244_884_733_000_000
    capture = (epoch + pts).astype("<u8")
    recording_id = "synthetic-shared-audio-native"
    archive = root / "S001_I01_R01_NRD_untrimmed.zip"
    video_member = "Neon Scene Camera v1 ps1.mp4"
    time_member = "Neon Scene Camera v1 ps1.time"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr("info.json", json.dumps({
            "recording_id": recording_id,
            "start_time": epoch,
            "duration": int(pts[-1] - pts[0] + 33_333_333),
        }))
        output.writestr(time_member, capture.tobytes())
        output.write(ego_video, video_member)
    return archive


def _independent_video(root: Path, sample_rate: int, duration: int = 32) -> Path:
    rng = np.random.default_rng(99631)
    time = np.arange(sample_rate * duration, dtype=float) / sample_rate
    envelope = np.interp(time, np.linspace(0, duration, 420), rng.random(420))
    samples = 0.16 * envelope * rng.standard_normal(len(time))
    for second in rng.uniform(1, duration - 1, 30):
        first = int(second * sample_rate)
        length = min(int(0.10 * sample_rate), len(samples) - first)
        local = np.arange(length, dtype=float) / sample_rate
        samples[first:first + length] += 0.42 * np.exp(-36 * local) * np.sin(
            2 * np.pi * rng.uniform(170, 1900) * local
        )
    return _write_video(root, "independent", np.tanh(samples), sample_rate, "0x7a3e3e")


def _fixture(root: Path) -> dict:
    sample_rate = 48_000
    physical_duration = 78.0
    source_intro, reference_intro = 4.2, 1.1
    source_scale, reference_scale = 1.0004, 1.0
    physical = _shared_physical_signal(physical_duration, sample_rate)
    facial = _device_track(
        physical, physical_duration=physical_duration, sample_rate=sample_rate,
        intro_sec=source_intro, clock_scale=source_scale, outro_sec=2.3, seed=1, source=True,
    )
    ego = _device_track(
        physical, physical_duration=physical_duration, sample_rate=sample_rate,
        intro_sec=reference_intro, clock_scale=reference_scale, outro_sec=3.1, seed=2, source=False,
    )
    facial_video = _write_video(root, "facial_expression", facial, sample_rate, "0x4c5a78")
    ego_video = _write_video(root, "ego", ego, sample_rate, "0x6b5a44")
    native = _native_zip(root, ego_video)
    independent = _independent_video(root, sample_rate)
    return {
        "facial_video": facial_video,
        "ego_video": ego_video,
        "native_zip": native,
        "independent_video": independent,
        "expected_scale": source_scale / reference_scale,
        "expected_offset_sec": reference_intro - (source_scale / reference_scale) * source_intro,
    }


def main() -> int:
    checks: dict[str, bool] = {}
    report: dict = {
        "schema_version": "oculostudio-release-validation-1.0",
        "release": "OculoStudio v0.21.71 — Shared-Audio Video Synchronization",
        "checks": checks,
    }
    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    package = Path(oculostudio.__file__).resolve().parent
    service_text = (package / "services" / "audio_video_sync.py").read_text("utf-8")
    server_text = (package / "server.py").read_text("utf-8")
    app_text = (package / "static" / "app.js").read_text("utf-8")
    checks.update({
        "single_affine_mapping_not_dtw": "arbitrary_dynamic_time_warping_used" in service_text,
        "distributed_window_matching": "_audio_windows" in service_text,
        "leave_one_window_out_validation": "_leave_one_out" in service_text,
        "visual_review_required_by_backend": "Review start, middle and end" in service_text,
        "job_route_present": '"/api/jobs/audio-video-sync"' in server_text,
        "accept_route_present": '"/api/time-maps/audio-video/accept"' in server_text,
        "range_video_route_present": '"/api/media/audio-video-sync/{session_id}/{candidate_id}/{side}"' in server_text,
        "calibration_panel_present": "Shared-audio facial ↔ ego video sync" in app_text,
        "split_video_review_present": "avsync-video-grid" in app_text,
        "start_middle_end_review_present": all(text in app_text for text in ("Review start", "Review middle", "Review end")),
    })

    with tempfile.TemporaryDirectory(prefix="oculo_shared_audio_sync_") as temp:
        temp_root = Path(temp)
        fixture = _fixture(temp_root)
        source_before = sha256_file(fixture["facial_video"])
        native_before = sha256_file(fixture["native_zip"])
        store = ProjectStore.create(temp_root / "project", "Shared audio validation", session_label="S001")
        session_id = store.project.selected_session_id
        facial_asset = add_asset(
            store, session_id=session_id, path=fixture["facial_video"],
            label="S001_I01_R01_FacialExpression.mp4", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01", "media_role": "facial_expression_video"}},
        )
        native_asset = add_asset(
            store, session_id=session_id, path=fixture["native_zip"],
            label="S001_I01_R01_NRD_untrimmed.zip", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01", "media_role": "ego_video"}},
        )
        independent_asset = add_asset(
            store, session_id=session_id, path=fixture["independent_video"],
            label="S001_I01_R01_UnrelatedVideo.mp4", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01"}},
        )
        maps_before = len(store.project.sessions[session_id].time_maps)
        role_mismatch_blocked = False
        try:
            automatic_shared_audio_video_sync(
                store,
                session_id=session_id,
                source_video_asset_id=facial_asset.id,
                reference_video_asset_id=native_asset.id,
                wearer_role_id="R02",
                maximum_offset_sec=15,
            )
        except ValidationError:
            role_mismatch_blocked = True
        checks["role_mismatch_is_rejected_before_matching"] = role_mismatch_blocked
        checks["role_mismatch_does_not_publish_map"] = len(store.project.sessions[session_id].time_maps) == maps_before
        result = automatic_shared_audio_video_sync(
            store,
            session_id=session_id,
            source_video_asset_id=facial_asset.id,
            reference_video_asset_id=native_asset.id,
            wearer_role_id="R01",
            maximum_offset_sec=15,
        )
        candidate = result["candidate"]
        metrics = candidate["metrics"]
        checks.update({
            "shared_audio_candidate_is_reviewable": result["confidence"] in {"HIGH", "REVIEW"},
            "candidate_does_not_publish_time_map": len(store.project.sessions[session_id].time_maps) == maps_before,
            "offset_recovered_within_120ms": abs(float(metrics["estimated_offset_sec"]) - fixture["expected_offset_sec"]) <= 0.12,
            "clock_scale_recovered": abs(float(candidate["preview_time_map"]["scale"]) - fixture["expected_scale"]) <= 0.0012,
            "distributed_anchors_at_least_five": int(metrics.get("matched_window_count") or 0) >= 5,
            "leave_one_out_windows_at_least_three": int(metrics.get("holdout_window_count") or 0) >= 3,
            "holdout_p95_below_35ms": float(metrics.get("p95_residual_ms") or 1e9) <= 35.0,
            "matched_span_covers_recording": float(metrics.get("matched_time_coverage_fraction") or 0.0) >= 0.55,
            "one_sided_instruction_did_not_bias_mapping": abs(float(metrics["estimated_offset_sec"]) - fixture["expected_offset_sec"]) <= 0.12,
        })
        blocked = False
        try:
            accept_audio_video_sync(
                store, session_id=session_id, candidate_id=candidate["candidate_id"],
                visual_review_confirmed=False,
            )
        except ValidationError:
            blocked = True
        checks["acceptance_blocked_without_visual_review"] = blocked
        checks["blocked_acceptance_does_not_publish_time_map"] = len(store.project.sessions[session_id].time_maps) == maps_before

        maps_before_registration_failure = {
            key: value.model_dump(mode="json")
            for key, value in store.project.sessions[session_id].time_maps.items()
        }
        registration_failure_rolled_back = False
        try:
            with patch("oculostudio.services.audio_video_sync.ArtifactStore.register", side_effect=RuntimeError("synthetic registration failure")):
                accept_audio_video_sync(
                    store,
                    session_id=session_id,
                    candidate_id=candidate["candidate_id"],
                    visual_review_confirmed=True,
                    visual_offset_adjustment_ms=12.0,
                )
        except RuntimeError:
            after_failure = {
                key: value.model_dump(mode="json")
                for key, value in store.project.sessions[session_id].time_maps.items()
            }
            registration_failure_rolled_back = after_failure == maps_before_registration_failure
        checks["failed_artifact_registration_rolls_back_timegraph"] = registration_failure_rolled_back

        accepted = accept_audio_video_sync(
            store,
            session_id=session_id,
            candidate_id=candidate["candidate_id"],
            visual_review_confirmed=True,
            visual_offset_adjustment_ms=12.0,
        )
        checks.update({
            "reviewed_time_map_published": accepted["time_map"] is not None,
            "native_pts_to_pupil_bridge_published": accepted["reference_pts_bridge_time_map"] is not None,
            "time_sync_artifact_registered": accepted["artifact"]["kind"] == "time_sync",
            "visual_correction_audited": math.isclose(float(accepted["artifact"]["parameters"]["visual_offset_adjustment_ms"]), 12.0),
        })
        preview = get_audio_video_sync_preview(
            store, session_id=session_id, artifact_id=accepted["artifact"]["id"], maximum_samples=420,
        )
        checks.update({
            "accepted_evidence_reopens": preview.get("accepted", {}).get("visual_review_confirmed") is True,
            "preview_is_bounded": 0 < len(preview.get("signals") or []) <= 420,
            "source_media_resolves": audio_video_sync_video_path(store, session_id=session_id, candidate_id=candidate["candidate_id"], side="source").is_file(),
            "native_reference_media_resolves": audio_video_sync_video_path(store, session_id=session_id, candidate_id=candidate["candidate_id"], side="reference").is_file(),
        })
        negative_maps_before = len(store.project.sessions[session_id].time_maps)
        negative = automatic_shared_audio_video_sync(
            store,
            session_id=session_id,
            source_video_asset_id=facial_asset.id,
            reference_video_asset_id=independent_asset.id,
            wearer_role_id="R01",
            maximum_offset_sec=10,
        )
        checks.update({
            "unrelated_audio_fails_closed": negative["confidence"] == "FAILED",
            "failed_candidate_does_not_publish_map": len(store.project.sessions[session_id].time_maps) == negative_maps_before,
            "source_video_unchanged": sha256_file(fixture["facial_video"]) == source_before,
            "native_zip_unchanged": sha256_file(fixture["native_zip"]) == native_before,
        })
        report["measured"] = {
            "expected_scale": fixture["expected_scale"],
            "recovered_scale": candidate["preview_time_map"]["scale"],
            "expected_offset_sec": fixture["expected_offset_sec"],
            "recovered_offset_sec": metrics.get("estimated_offset_sec"),
            "recovered_drift_ppm": metrics.get("estimated_drift_ppm"),
            "global_correlation": metrics.get("global_correlation"),
            "aligned_correlation": metrics.get("aligned_correlation"),
            "matched_windows": metrics.get("matched_window_count"),
            "rejected_windows": metrics.get("rejected_window_count"),
            "holdout_p95_residual_ms": metrics.get("p95_residual_ms"),
            "matched_time_coverage_fraction": metrics.get("matched_time_coverage_fraction"),
            "negative_confidence": negative.get("confidence"),
            "negative_global_correlation": negative.get("candidate", {}).get("metrics", {}).get("global_correlation"),
        }

    failed = sorted(name for name, passed in checks.items() if not passed)
    report["passed_count"] = sum(bool(value) for value in checks.values())
    report["check_count"] = len(checks)
    report["failed"] = failed
    report["all_pass"] = not failed
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
