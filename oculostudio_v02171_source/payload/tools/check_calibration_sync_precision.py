from __future__ import annotations

import json
import math

import numpy as np

from oculostudio.domain.models import TimeAnchor
from oculostudio.domain.timegraph import fit_time_map
from oculostudio.services.auto_synchronization import (
    _anchors_from_windows,
    _confidence,
    _global_alignment,
    _mapping_metrics,
    _uniform_motion_signature,
    _window_matches,
)


def motion(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    signal = 0.05 * np.maximum(0.0, np.sin(0.17 * t + 0.4))
    centres = np.asarray([8, 22, 41, 63, 87, 115, 146, 183, 221, 264, 305, 343], dtype=float)
    widths = np.asarray([0.7, 1.2, 0.8, 1.7, 0.9, 1.4, 1.0, 1.6, 0.8, 1.2, 1.5, 0.9])
    amplitudes = np.asarray([1.2, 2.0, 1.5, 2.4, 1.7, 2.1, 1.3, 2.5, 1.8, 2.2, 1.6, 2.3])
    for centre, width, amplitude in zip(centres, widths, amplitudes):
        signal += amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)
    return signal


def run_case(scale: float) -> dict:
    duration = 360.0
    pupil_t = np.arange(0.0, duration, 1.0 / 200.0)
    tracker_t = np.arange(0.0, scale * duration + 1.0, 1.0 / 120.0)
    pupil_speed = motion(pupil_t)
    equivalent = tracker_t / scale
    tracker_speed = motion(np.clip(equivalent, 0.0, duration))
    pupil = _uniform_motion_signature(pupil_t, pupil_speed, sample_rate_hz=120.0)
    tracker = _uniform_motion_signature(tracker_t, tracker_speed, sample_rate_hz=120.0)
    alignment = _global_alignment(pupil, tracker, maximum_offset_sec=2.0, minimum_overlap_sec=60.0)
    start_ns = 1_800_000_000_000_000_000
    windows = _window_matches(pupil, tracker, alignment, pupil_start_ns=start_ns)
    anchors = _anchors_from_windows(
        windows,
        tracker_start_sec=0.0,
        alignment=alignment,
        pupil_start_ns=start_ns,
        pupil_duration_sec=duration,
    )
    mapping = fit_time_map(
        "pupil",
        "vicon",
        [TimeAnchor.model_validate(row) for row in anchors],
        method="precision_test",
    )
    metrics = _mapping_metrics(
        mapping,
        pupil_start_ns=start_ns,
        tracker_start_sec=0.0,
        pupil=pupil,
        tracker=tracker,
        windows=windows,
    )
    metrics.update({
        "global_correlation": alignment.correlation,
        "global_peak_margin": alignment.peak_margin,
        "usable_overlap_sec": alignment.overlap_sec,
    })
    confidence, reasons = _confidence(
        alignment=alignment,
        mapping_metrics=metrics,
        overlap_sec=alignment.overlap_sec,
        pupil_motion=pupil.metadata,
        tracker_motion=tracker.metadata,
    )
    return {"metrics": metrics, "confidence": confidence, "reasons": reasons, "windows": windows}


def main() -> None:
    drift_case = run_case(1.000060)
    zero_case = run_case(1.0)
    drift = drift_case["metrics"]
    zero = zero_case["metrics"]
    assert drift_case["confidence"] == "HIGH", drift_case["reasons"]
    assert abs(float(drift["estimated_drift_ppm"]) - 60.0) < 35.0
    assert int(drift["holdout_window_count"]) >= 6
    assert 0.0 < float(drift["drift_ci95_ppm"]) < 250.0
    assert float(drift["residual_resolution_ms"]) > 0.0
    assert drift["residual_validation_mode"] == "leave_one_window_out"

    # A truly zero-drift synthetic case may estimate zero, but it must never be
    # represented as infinite precision.  The confidence interval and local-lag
    # resolution remain strictly positive.
    assert zero_case["confidence"] == "HIGH", zero_case["reasons"]
    assert abs(float(zero["estimated_drift_ppm"])) < 35.0
    assert float(zero["drift_ci95_ppm"]) > 0.0
    assert float(zero["residual_resolution_ms"]) > 0.0
    assert int(zero["holdout_window_count"]) >= 6

    print(json.dumps({
        "ok": True,
        "drift_case": {key: drift.get(key) for key in (
            "estimated_drift_ppm", "drift_ci95_ppm", "p95_residual_ms",
            "residual_resolution_ms", "holdout_window_count", "window_time_span_sec",
        )},
        "zero_case": {key: zero.get(key) for key in (
            "estimated_drift_ppm", "drift_ci95_ppm", "p95_residual_ms",
            "residual_resolution_ms", "holdout_window_count", "residual_below_resolution",
        )},
        "checks": [
            "60 ppm drift recovered from co-start-like signals",
            "local lags refined below one sample",
            "leave-one-window-out residuals used for QC",
            "zero drift retains a finite 95% interval",
            "zero residual retains a finite estimator resolution",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
