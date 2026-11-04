"""Validation for OculoStudio v0.21.71 Body Geometry Integrity Gate."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import oculostudio
from oculostudio.errors import ValidationError
from oculostudio.services.aoi_review import _require_body_geometry_review_ready
from oculostudio.services.body_primitives import (
    assess_body_geometry_integrity,
    body_geometry_integrity_from_qc,
    build_body_primitives,
    load_body_profile,
)


def _synthetic_body_npz(root: Path) -> Path:
    labels = [
        "P01:LFHD", "P01:RFHD", "P01:LSHO", "P01:RSHO",
        "P01:LELB", "P01:RELB", "P01:LWRI", "P01:RWRI",
        "P01:LHIP", "P01:RHIP", "P01:LKNE", "P01:RKNE",
        "P01:LANK", "P01:RANK",
    ]
    anchors = {
        "LFHD": (-80, 0, 1750), "RFHD": (80, 0, 1750),
        "LSHO": (-220, 0, 1450), "RSHO": (220, 0, 1450),
        "LELB": (-430, 0, 1200), "RELB": (430, 0, 1200),
        "LWRI": (-520, 0, 950), "RWRI": (520, 0, 950),
        "LHIP": (-140, 0, 950), "RHIP": (140, 0, 950),
        "LKNE": (-140, 0, 520), "RKNE": (140, 0, 520),
        "LANK": (-140, 0, 100), "RANK": (140, 0, 100),
    }
    points = np.zeros((6, len(labels), 3), dtype=float)
    for frame in range(points.shape[0]):
        for marker, label in enumerate(labels):
            points[frame, marker] = np.asarray(anchors[label.split(":")[-1]]) + [frame * 4, 0, 0]
    path = root / "body_without_central_torso_markers.npz"
    np.savez(
        path,
        time_sec=np.arange(points.shape[0], dtype=float) / 100.0,
        labels=np.asarray(labels, dtype=object),
        points_mm=points,
        valid=np.ones((points.shape[0], len(labels)), dtype=bool),
        frame_rate=100.0,
    )
    return path


def _diagnostic(*, relation: str, torso: int = 100, body: int = 100, pelvis: int = 100) -> dict:
    return {
        "entity_id": f"{relation}_body",
        "entity_relation": relation,
        "source_asset_id": f"asset_{relation}",
        "source_asset_label": f"{relation.title()} C3D",
        "scene_frame_count": 100,
        "frames_with_any_primitive": body,
        "torso_primitive_frame_count": torso,
        "pelvis_primitive_frame_count": pelvis,
        "torso_direct_frame_count": 20 if torso else 0,
        "torso_derived_frame_count": max(0, torso - 20) if torso else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    checks: dict[str, bool] = {}

    passed = assess_body_geometry_integrity(
        [_diagnostic(relation="partner"), _diagnostic(relation="self")],
        expected_relations=["partner", "self"],
    )
    checks["valid_self_partner_geometry_passes"] = bool(
        passed["gate_passed"] and passed["review_ready"] and passed["fusion_ready"]
    )

    missing = assess_body_geometry_integrity(
        [_diagnostic(relation="partner", torso=0, pelvis=0)],
        expected_relations=["partner"],
    )
    checks["missing_torso_fails"] = bool(
        not missing["gate_passed"]
        and any(row.get("code") == "torso_missing" for row in missing["failures"])
    )

    low_coverage = assess_body_geometry_integrity(
        [_diagnostic(relation="partner", torso=72, body=75, pelvis=72)],
        expected_relations=["partner"],
    )
    checks["coverage_thresholds_fail_closed"] = bool(
        not low_coverage["gate_passed"]
        and {row.get("code") for row in low_coverage["failures"]}
        >= {"body_timeline_coverage_below_threshold", "torso_coverage_below_threshold"}
    )

    warned = assess_body_geometry_integrity(
        [_diagnostic(relation="partner", torso=0, pelvis=0)],
        policy="warn",
        expected_relations=["partner"],
    )
    checks["diagnostic_policy_never_unlocks_review"] = bool(
        warned["enforcement_outcome"] == "continued_with_warning"
        and not warned["review_ready"]
        and not warned["fusion_ready"]
    )

    legacy = body_geometry_integrity_from_qc({})
    checks["legacy_artifact_without_diagnostics_is_blocked"] = bool(
        legacy["legacy"] and not legacy["review_ready"]
    )

    try:
        _require_body_geometry_review_ready(SimpleNamespace(qc={}))
        checks["manual_review_guard_rejects_invalid_geometry"] = False
    except ValidationError:
        checks["manual_review_guard_rejects_invalid_geometry"] = True

    valid_qc = {
        "body_geometry_clock": "gaze_sample_timestamp",
        "include_self_body": True,
        "body_diagnostics": [_diagnostic(relation="partner"), _diagnostic(relation="self")],
    }
    accepted = _require_body_geometry_review_ready(SimpleNamespace(qc=valid_qc))
    checks["manual_review_guard_accepts_verified_geometry"] = bool(accepted["review_ready"])

    with tempfile.TemporaryDirectory(prefix="oculo_geometry_gate_") as temp:
        temp_root = Path(temp)
        body_path = _synthetic_body_npz(temp_root)
        frames = np.arange(6, dtype=np.int64)
        result = build_body_primitives(
            body_asset_id="body_asset",
            body_path=body_path,
            entity_id="P01",
            scene_frame_indices=frames,
            scene_timestamps_ns=frames * 10_000_000,
            query_time_sec=frames.astype(float) / 100.0,
            profile=load_body_profile(),
            marker_prefix="P01",
            max_marker_gap_sec=0.1,
        )
        diagnostics = result.diagnostics
        checks["fallback_creates_real_torso"] = diagnostics.get("torso_primitive_frame_count") == 6
        checks["fallback_creates_real_pelvis"] = diagnostics.get("pelvis_primitive_frame_count") == 6
        checks["fallback_provenance_is_counted"] = bool(
            diagnostics.get("torso_derived_frame_count") == 6
            and diagnostics.get("geometry_provenance_counts")
        )
        generated = assess_body_geometry_integrity(
            [{**diagnostics, "entity_relation": "partner"}],
            expected_relations=["partner"],
        )
        checks["fallback_geometry_passes_acceptance"] = bool(generated["gate_passed"])

    package_root = Path(oculostudio.__file__).resolve().parent
    body_source = (package_root / "services" / "body_aoi.py").read_text(encoding="utf-8")
    fusion_source = (package_root / "services" / "target_fusion.py").read_text(encoding="utf-8")
    review_source = (package_root / "services" / "aoi_review.py").read_text(encoding="utf-8")
    server_source = (package_root / "server.py").read_text(encoding="utf-8")
    app_source = (package_root / "static" / "app.js").read_text(encoding="utf-8")
    review_js = (package_root / "static" / "aoi_review.js").read_text(encoding="utf-8")
    index_source = (package_root / "static" / "index.html").read_text(encoding="utf-8")

    checks["gate_runs_before_ray_intersection"] = (
        body_source.index("assess_body_geometry_integrity(")
        < body_source.index("origins = rays[origin_cols]")
    )
    checks["failed_gate_writes_preflight_diagnostic"] = all(
        token in body_source
        for token in ("_write_body_geometry_preflight", "diagnostic_path", "Body geometry gate blocked")
    )
    checks["api_exposes_gate_policy_and_thresholds"] = all(
        token in server_source
        for token in ("geometry_acceptance_policy", "minimum_torso_coverage", "minimum_body_frame_coverage")
    )
    checks["target_fusion_is_guarded"] = "not fusion-ready because torso/body geometry failed" in fusion_source
    checks["body_manual_review_is_guarded"] = "Body AOI human review is blocked" in review_source
    checks["fixation_level_review_units_packaged"] = all(
        token in review_source
        for token in ("review_unit_policy", "pending_review_representative_frames", "fixation_when_available_else_sample")
    )
    checks["frontend_has_acceptance_controls"] = all(
        token in app_source
        for token in ("body-aoi-geometry-policy", "body-aoi-min-torso", "body-aoi-min-body")
    )
    checks["frontend_blocks_invalid_review_and_fusion"] = all(
        token in app_source
        for token in ("Geometry blocked · regenerate", "bodyArtifactReviewReady", "bodyArtifactFusionReady")
    )
    checks["review_player_shows_integrity_banner"] = all(
        token in review_js
        for token in ("body-review-geometry-gate", "Human review blocked: torso geometry invalid", "body_review_allowed")
    )
    checks["body_mode_opens_representative_pending_group"] = "['face','body'].includes(state.mode)" in review_js
    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    checks["cache_busting_version"] = "app.js?v=0.21.71" in index_source and "aoi_review.js?v=0.21.71" in index_source

    failed = [name for name, passed_check in checks.items() if not passed_check]
    report = {
        "release": "OculoStudio v0.21.71",
        "feature": "Body Geometry Integrity Gate and fixation-level Body AOI review",
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": failed,
        "valid_report": passed,
        "missing_torso_report": missing,
        "low_coverage_report": low_coverage,
        "ok": not failed,
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
