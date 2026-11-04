"""Targeted validation for OculoStudio v0.21.71 Body AOI visual analytics."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import oculostudio
from oculostudio.services import aoi_review


def main() -> int:
    samples = pd.DataFrame(
        [
            {
                "native_frame_index": 0,
                "ray_valid": True,
                "body_relation": "partner",
                "body_coarse_region": "head",
                "body_social_region": "partner_head",
                "body_entity_id": "P02",
                "body_confidence": 0.95,
                "body_distance_mm": 1480.0,
                "body_exact_hit": True,
                "review_required": False,
                "self_occlusion": False,
                "assignment_status": "body_hit",
                "occlusion_status": "partner_visible_first",
                "body_hit_x_mm": 100.0,
                "body_hit_y_mm": 1580.0,
                "body_hit_z_mm": 2250.0,
                "ray_origin_x_mm": 0.0,
                "ray_origin_y_mm": 1600.0,
                "ray_origin_z_mm": 0.0,
            },
            {
                "native_frame_index": 1,
                "ray_valid": True,
                "body_relation": "self",
                "body_coarse_region": "right_hand",
                "body_social_region": "self_right_hand",
                "body_entity_id": "P01",
                "body_confidence": 0.72,
                "body_distance_mm": 410.0,
                "body_exact_hit": False,
                "review_required": True,
                "self_occlusion": True,
                "assignment_status": "body_hit_uncertainty",
                "occlusion_status": "self_visible_first",
                "body_hit_x_mm": -180.0,
                "body_hit_y_mm": 1120.0,
                "body_hit_z_mm": 420.0,
                "ray_origin_x_mm": 0.0,
                "ray_origin_y_mm": 1600.0,
                "ray_origin_z_mm": 0.0,
            },
            {
                "native_frame_index": 2,
                "ray_valid": True,
                "body_relation": "none",
                "body_coarse_region": "no_body",
                "body_social_region": "no_body",
                "body_confidence": 0.0,
                "body_exact_hit": False,
                "review_required": False,
                "self_occlusion": False,
                "assignment_status": "no_body",
                "occlusion_status": "none",
                "ray_origin_x_mm": 0.0,
                "ray_origin_y_mm": 1600.0,
                "ray_origin_z_mm": 0.0,
            },
        ]
    )
    body_regions = pd.DataFrame(
        [
            {
                "geometry_json": json.dumps(
                    {"center_mm": [100, 1500, 2200], "start_mm": [100, 1450, 2200], "end_mm": [100, 1650, 2200]}
                )
            },
            {
                "geometry_json": json.dumps(
                    {"center_mm": [-160, 1100, 450], "start_mm": [-250, 1120, 400], "end_mm": [-80, 1100, 500]}
                )
            },
        ]
    )
    data = SimpleNamespace(
        samples=samples,
        body_regions=body_regions,
        video_map=pd.DataFrame(
            {
                "native_frame_index": [0, 1, 2],
                "output_frame_index": [0, 1, 2],
            }
        ),
        video_info={"frame_count": 3, "fps": 30.0},
    )

    analytics = aoi_review._body_analytics(samples)
    timeline = aoi_review._body_timeline(data)
    bounds = aoi_review._body_world_bounds(data)

    static_root = Path(aoi_review.__file__).resolve().parents[1] / "static"
    js = (static_root / "aoi_review.js").read_text(encoding="utf-8")
    css = (static_root / "styles.css").read_text(encoding="utf-8")
    index = (static_root / "index.html").read_text(encoding="utf-8")
    source = Path(aoi_review.__file__).read_text(encoding="utf-8")

    checks = {
        "installed_version": oculostudio.__version__ == "0.21.71",
        "manifest_schema_1_3": '"oculostudio-aoi-review-1.3"' in source,
        "clip_hit_fraction": abs(float(analytics["hit_fraction"]) - (2.0 / 3.0)) < 1e-9,
        "self_partner_counts": analytics["relation_counts"] == {"partner": 1, "self": 1},
        "region_counts": analytics["region_counts"] == {"head": 1, "right_hand": 1},
        "occlusion_and_review_metrics": analytics["self_occlusion_count"] == 1 and analytics["review_required_count"] == 1,
        "timeline_bins": timeline["bin_count"] == 3 and len(timeline["bins"]) == 3,
        "timeline_target_semantics": timeline["bins"][0]["dominant_relation"] == "partner" and timeline["bins"][1]["dominant_relation"] == "self",
        "stable_world_bounds": bool(bounds and bounds["sampled_point_count"] >= 4 and bounds["unit"] == "mm"),
        "dual_orthographic_ui": all(token in js for token in ("drawWorldInset", "data-body-view", "X–Z", "Y–Z")),
        "timeline_ui": all(token in js for token in ("drawBodyTimeline", "body-aoi-timeline", "bodyTimelineHover")),
        "body_statistics_ui": all(token in js for token in ("renderBodyOverview", "bodyMapSvg", "Body-region proportions")),
        "body_visual_styles": all(token in css for token in (".body-aoi-overview", ".body-aoi-timeline-panel", ".aoi-review-world-tabs")),
        "cache_busting_version": "aoi_review.js?v=0.21.71" in index and "styles.css?v=0.21.71" in index,
    }
    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"release": "OculoStudio v0.21.71", "checks": checks, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
