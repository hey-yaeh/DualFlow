"""Targeted validation for OculoStudio v0.21.71 torso fallback and Body review reduction."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import oculostudio
from oculostudio.services import aoi_review
from oculostudio.services.body_aoi import _apply_body_auto_review_reduction
from oculostudio.services.body_primitives import build_body_primitives, load_body_profile


def _body_npz(root: Path) -> Path:
    labels = [
        "P01:LFHD", "P01:RFHD", "P01:LSHO", "P01:RSHO",
        "P01:LELB", "P01:RELB", "P01:LWRI", "P01:RWRI",
        "P01:LHIP", "P01:RHIP", "P01:LKNE", "P01:RKNE",
        "P01:LANK", "P01:RANK",
    ]
    base = {
        "LFHD": (-80, 0, 1750), "RFHD": (80, 0, 1750),
        "LSHO": (-220, 0, 1450), "RSHO": (220, 0, 1450),
        "LELB": (-430, 0, 1200), "RELB": (430, 0, 1200),
        "LWRI": (-520, 0, 950), "RWRI": (520, 0, 950),
        "LHIP": (-140, 0, 950), "RHIP": (140, 0, 950),
        "LKNE": (-140, 0, 520), "RKNE": (140, 0, 520),
        "LANK": (-140, 0, 100), "RANK": (140, 0, 100),
    }
    points = np.zeros((4, len(labels), 3), dtype=float)
    for frame_index in range(4):
        for marker_index, label in enumerate(labels):
            points[frame_index, marker_index] = np.asarray(base[label.split(":")[-1]], dtype=float) + [frame_index * 5, 0, 0]
    path = root / "body_without_central_torso_markers.npz"
    np.savez(
        path,
        time_sec=np.arange(4, dtype=float) / 100.0,
        labels=np.asarray(labels, dtype=object),
        points_mm=points,
        valid=np.ones((4, len(labels)), dtype=bool),
        frame_rate=100.0,
    )
    return path


def _review_row(
    index: int,
    *,
    exact: bool = False,
    review: bool = True,
    angle: float = 0.20,
    second_angle: float = 1.0,
    candidate_count: int = 1,
    frame_index: float = 10,
    fixation_id: str | None = "F1",
    self_occlusion: bool = False,
    status: str | None = None,
) -> dict:
    return {
        "source_gaze_row_index": index,
        "timestamp_ns": 1_000_000_000 + index * 5_000_000,
        "scene_frame_index": frame_index,
        "fixation_id": fixation_id,
        "body_relation": "partner",
        "body_entity_id": "P02",
        "body_coarse_region": "torso",
        "body_region": "torso",
        "body_raw_region": "torso",
        "body_exact_hit": exact,
        "body_angular_miss_deg": 0.0 if exact else angle,
        "body_confidence": 0.85,
        "first_hit_primitive_confidence": 0.80,
        "second_hit_angular_miss_deg": second_angle,
        "candidate_hit_count": candidate_count,
        "self_occlusion": self_occlusion,
        "review_required": review,
        "assignment_status": status or ("body_3d_exact_partner" if exact else "body_3d_uncertainty_cone_partner"),
    }


def _visual_rows() -> list[dict]:
    def capsule(entity: str, relation: str, region: str, start: list[float], end: list[float], radius: float) -> dict:
        return {
            "entity_id": entity,
            "entity_relation": relation,
            "region": region,
            "coarse_region": region,
            "confidence": 0.85,
            "geometry_json": json.dumps({"start_mm": start, "end_mm": end, "radius_mm": radius}),
        }

    def sphere(entity: str, relation: str, region: str, center: list[float], radius: float) -> dict:
        return {
            "entity_id": entity,
            "entity_relation": relation,
            "region": region,
            "coarse_region": region,
            "confidence": 0.85,
            "geometry_json": json.dumps({"center_mm": center, "radius_mm": radius}),
        }

    rows: list[dict] = []
    for x, entity, relation, y in [(-320, "R01", "self", 900), (320, "R02", "partner", 1500)]:
        rows.extend([
            sphere(entity, relation, "head", [x, y, 1750], 110),
            capsule(entity, relation, "neck", [x, y, 1750], [x, y, 1480], 62),
            capsule(entity, relation, "left_upper_arm", [x - 210, y, 1450], [x - 420, y, 1180], 62),
            capsule(entity, relation, "right_upper_arm", [x + 210, y, 1450], [x + 420, y, 1180], 62),
            capsule(entity, relation, "left_thigh", [x - 130, y, 930], [x - 140, y, 500], 92),
            capsule(entity, relation, "right_thigh", [x + 130, y, 930], [x + 140, y, 500], 92),
        ])
    return rows


def main() -> int:
    package_root = Path(oculostudio.__file__).resolve().parent
    js = (package_root / "static" / "aoi_review.js").read_text("utf-8")
    app_js = (package_root / "static" / "app.js").read_text("utf-8")
    index = (package_root / "static" / "index.html").read_text("utf-8")
    checks: dict[str, bool] = {}
    report: dict[str, object] = {
        "release": f"OculoStudio v{oculostudio.__version__}",
        "feature": "Torso geometry fallback and auditable Body AOI review reduction",
        "checks": checks,
    }

    with tempfile.TemporaryDirectory(prefix="oculo_2149_body_") as temporary:
        root = Path(temporary)
        result = build_body_primitives(
            body_asset_id="body_asset",
            body_path=_body_npz(root),
            entity_id="P01",
            scene_frame_indices=np.arange(4),
            scene_timestamps_ns=np.arange(4, dtype=np.int64) * 10_000_000,
            query_time_sec=np.arange(4, dtype=float) / 100.0,
            profile=load_body_profile(),
            marker_prefix="P01",
            max_marker_gap_sec=0.1,
        )
        region_counts = result.frame_regions.groupby("region").size().to_dict()
        diagnostics = result.diagnostics
        checks["torso_created_without_central_markers"] = region_counts.get("torso") == 4
        checks["pelvis_created_from_bilateral_hips"] = region_counts.get("pelvis") == 4
        checks["fallback_provenance_recorded"] = (
            diagnostics.get("landmark_fallback_counts", {}).get("bilateral_shoulders", 0) >= 4
            and diagnostics.get("landmark_fallback_counts", {}).get("bilateral_hips", 0) == 4
        )
        checks["torso_coverage_diagnostic"] = diagnostics.get("torso_primitive_frame_count") == 4

    rows = [
        _review_row(0, exact=True, review=False),
        _review_row(1, exact=True, review=False),
        _review_row(2, angle=0.22, second_angle=1.0),
        _review_row(3, angle=0.80, second_angle=0.90, candidate_count=2),
        _review_row(4, angle=0.10, second_angle=1.0, self_occlusion=True),
        _review_row(5, angle=0.80, second_angle=1.2, frame_index=11, fixation_id="F1"),
        _review_row(6, angle=0.10, status="no_scene_frame", frame_index=np.nan, fixation_id=None),
    ]
    reviewed, auto_report, _table = _apply_body_auto_review_reduction(pd.DataFrame(rows), "adaptive")
    checks["clear_margin_candidate_auto_accepted"] = bool(reviewed.loc[2, "auto_review_resolved"])
    checks["same_fixation_candidate_auto_accepted"] = bool(reviewed.loc[5, "auto_review_resolved"])
    checks["competing_surfaces_remain_pending"] = bool(reviewed.loc[3, "review_required"])
    checks["self_occlusion_remains_pending"] = bool(reviewed.loc[4, "review_required"])
    checks["missing_scene_frame_remains_pending"] = bool(reviewed.loc[6, "review_required"])
    checks["automatic_target_unchanged"] = all(
        reviewed.loc[index, ["body_relation", "body_entity_id", "body_region"]].tolist()
        == ["partner", "P02", "torso"]
        for index in reviewed.index
    )
    checks["sample_level_audit_columns"] = all(column in reviewed.columns for column in (
        "review_required_before_auto_resolution", "review_reason_before_auto_resolution",
        "auto_review_resolved", "auto_review_resolution_source", "auto_review_confidence", "auto_review_policy",
    ))

    frame = np.full((720, 1280, 3), 15, dtype=np.uint8)
    aoi_review._draw_body_world_inset(frame, {"body_regions": _visual_rows(), "selected_sample": {}})
    torso_like = (
        (frame[:, :, 0].astype(int) > frame[:, :, 1].astype(int) + 3)
        & (frame[:, :, 1].astype(int) > frame[:, :, 2].astype(int) + 4)
    )
    checks["old_artifact_visual_torso_fallback"] = int(torso_like.sum()) > 300

    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    checks["cache_busting_version"] = "aoi_review.js?v=0.21.71" in index and "app.js?v=0.21.71" in index
    checks["live_visual_fallback_packaged"] = "augmentVisualTorsoFallback" in js and "visual_fallback" in js
    checks["near_body_language_packaged"] = "near-body candidate" in js and "near-body · auto accepted" in js
    checks["body_policy_selector_packaged"] = "body-aoi-auto-review" in app_js and "Adaptive · recommended" in app_js
    checks["body_review_summary_packaged"] = "Body AOI · torso geometry + review reduction" in app_js and "Torso primitive coverage" in app_js

    report.update({
        "region_counts": region_counts,
        "geometry_diagnostics": diagnostics,
        "auto_review": auto_report,
        "visual_torso_like_pixels": int(torso_like.sum()),
    })
    failed = [name for name, passed in checks.items() if not passed]
    report["failed"] = failed
    report["ok"] = not failed
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
