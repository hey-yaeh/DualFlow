"""Targeted v0.21.71 green-box facial partition and full-clip statistics validation."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

import oculostudio
from oculostudio.server import FaceAOIJobRequest
from oculostudio.services.face_aoi import (
    FACE_AOI_ENGINE,
    _clip_statistics_qc_fields,
    _full_clip_partner_face_statistics,
    create_face_aoi_annotations,
)
from oculostudio.services.face_semantics import (
    EllipseAOI,
    FaceInstance,
    Point2D,
    PolygonAOI,
    _face_aois,
    classify_face_point,
    face_instances_to_regions,
)

NEW_SETTINGS = {
    "bounded_features": True,
    "feature_scale": 1.55,
    "eye_feature_scale": 1.35,
    "nose_region_mode": "nasal_envelope",
    "mouth_center_policy": "adaptive_lip_envelope",
    "mouth_lateral_padding_ratio": 0.15,
    "mouth_top_padding_face_ratio": 0.045,
    "mouth_bottom_padding_face_ratio": 0.095,
    "mouth_max_height_face_ratio": 0.18,
    "mouth_absolute_max_height_face_ratio": 0.22,
    "mouth_max_width_face_ratio": 0.52,
    "mouth_bottom_limit_face_ratio": 0.90,
    "mouth_min_top_below_nose_ratio": 0.015,
    "mouth_strict_radius_x_face_ratio": 0.14,
    "mouth_strict_radius_y_face_ratio": 0.055,
    "nose_base_offset_ratio": 0.30,
    "nasal_envelope_top_y_fraction": 0.16,
    "nasal_envelope_top_halfwidth_iod_ratio": 0.16,
    "nasal_envelope_top_halfwidth_face_ratio": 0.055,
    "nasal_envelope_mid_halfwidth_iod_ratio": 0.22,
    "nasal_envelope_mid_halfwidth_face_ratio": 0.075,
    "nasal_envelope_bottom_halfwidth_iod_ratio": 0.30,
    "nasal_envelope_bottom_halfwidth_face_ratio": 0.12,
    "nasal_envelope_bottom_halfwidth_mouth_ratio": 0.42,
    "face_margin_ratio": 0.02,
}
PREVIOUS_SETTINGS = {
    "bounded_features": True,
    "feature_scale": 1.35,
    "eye_feature_scale": 1.0,
    "nose_region_mode": "ellipse",
    "mouth_center_policy": "landmark_midpoint",
    "face_margin_ratio": 0.02,
}
FINE_REGIONS = {"left_eye", "right_eye", "nose", "mouth"}


def contained(aoi: EllipseAOI | PolygonAOI, face: FaceInstance, tolerance: float = 1e-6) -> bool:
    if isinstance(aoi, PolygonAOI):
        return all(
            face.x1 - tolerance <= point.x <= face.x2 + tolerance
            and face.y1 - tolerance <= point.y <= face.y2 + tolerance
            for point in aoi.vertices
        )
    return (
        aoi.center.x - aoi.radius_x >= face.x1 - tolerance
        and aoi.center.x + aoi.radius_x <= face.x2 + tolerance
        and aoi.center.y - aoi.radius_y >= face.y1 - tolerance
        and aoi.center.y + aoi.radius_y <= face.y2 + tolerance
    )


def approx(value: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(value), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    checks: dict[str, bool] = {}

    face = FaceInstance(
        scene_frame_index=10,
        scene_timestamp_ns=1_000_000_000,
        track_id=7,
        explicit_entity_id="R02",
        x1=100.0,
        y1=80.0,
        x2=240.0,
        y2=250.0,
        left_eye=Point2D(195.0, 132.0),
        right_eye=Point2D(145.0, 133.0),
        nose=Point2D(170.0, 174.0),
        mouth_left=Point2D(150.0, 211.0),
        mouth_right=Point2D(195.0, 211.0),
        confidence=0.96,
        source_row_index=4,
    )
    previous = {aoi.region: aoi for aoi in _face_aois(face, PREVIOUS_SETTINGS)}
    expanded = {aoi.region: aoi for aoi in _face_aois(face, NEW_SETTINGS)}

    checks["application_version"] = oculostudio.__version__ == "0.21.71"
    checks["engine_version"] = (
        FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
    )
    checks["four_fine_regions_present"] = FINE_REGIONS.issubset(expanded)
    checks["eyes_are_substantially_larger"] = all(
        isinstance(expanded[name], EllipseAOI)
        and isinstance(previous[name], EllipseAOI)
        and expanded[name].radius_x > previous[name].radius_x * 1.25
        and expanded[name].radius_y > previous[name].radius_y * 1.25
        for name in ("left_eye", "right_eye")
    )
    checks["nose_is_nasal_bridge_alar_envelope"] = (
        isinstance(expanded["nose"], PolygonAOI)
        and expanded["nose"].geometry_model == "nasal_bridge_alar_envelope"
        and len(expanded["nose"].vertices) == 6
    )
    if isinstance(expanded["nose"], PolygonAOI) and len(expanded["nose"].vertices) == 6:
        nose_widths = (
            expanded["nose"].vertices[1].x - expanded["nose"].vertices[0].x,
            expanded["nose"].vertices[2].x - expanded["nose"].vertices[5].x,
            expanded["nose"].vertices[3].x - expanded["nose"].vertices[4].x,
        )
        checks["nose_not_hourglass"] = nose_widths[0] <= nose_widths[1] <= nose_widths[2]
        checks["nose_alae_span_is_wider_than_bridge"] = nose_widths[2] >= 1.6 * nose_widths[0]
    else:
        checks["nose_not_hourglass"] = False
        checks["nose_alae_span_is_wider_than_bridge"] = False
    checks["mouth_is_adaptive_lip_envelope"] = (
        isinstance(expanded["mouth"], PolygonAOI)
        and expanded["mouth"].geometry_model == "adaptive_lip_envelope"
        and len(expanded["mouth"].vertices) == 4
    )
    if isinstance(expanded["mouth"], PolygonAOI):
        mouth_top = min(point.y for point in expanded["mouth"].vertices)
        mouth_bottom = max(point.y for point in expanded["mouth"].vertices)
        mouth_left = min(point.x for point in expanded["mouth"].vertices)
        mouth_right = max(point.x for point in expanded["mouth"].vertices)
        checks["mouth_height_bounded"] = mouth_bottom - mouth_top <= face.height * 0.18 + 1e-6
        checks["mouth_width_bounded"] = mouth_right - mouth_left <= face.width * 0.52 + 1e-6
        checks["mouth_leaves_chin_for_other_face"] = (
            mouth_bottom <= face.y1 + 0.90 * face.height + 1e-6
            and classify_face_point(face.center.x, face.y1 + 0.95 * face.height, [face], aoi_settings=NEW_SETTINGS).region == "other_face"
        )
    else:
        checks["mouth_height_bounded"] = False
        checks["mouth_width_bounded"] = False
        checks["mouth_leaves_chin_for_other_face"] = False
    checks["all_fine_regions_contained"] = all(contained(expanded[name], face) for name in FINE_REGIONS)
    checks["effective_eye_scale_is_audited"] = all(
        isinstance(expanded[name], EllipseAOI)
        and approx(expanded[name].feature_scale, 1.55 * 1.35)
        for name in ("left_eye", "right_eye")
    )

    # The bridge and bilateral alar probes are outside the previous small nose
    # ellipse but inside the new monotonic nasal envelope.
    bridge_probe_x = 170.0
    bridge_probe_y = 150.0
    old_hit = classify_face_point(
        bridge_probe_x,
        bridge_probe_y,
        [face],
        aoi_settings=PREVIOUS_SETTINGS,
    )
    new_hit = classify_face_point(
        bridge_probe_x,
        bridge_probe_y,
        [face],
        aoi_settings=NEW_SETTINGS,
    )
    checks["classification_uses_nasal_envelope"] = (
        new_hit is not None
        and new_hit.region == "nose"
        and (old_hit is None or old_hit.region != "nose")
    )
    alar_y = max(point.y for point in expanded["nose"].vertices) - 1.0
    alar_left_x = min(point.x for point in expanded["nose"].vertices if abs(point.y - max(p.y for p in expanded["nose"].vertices)) < 1e-6) + 2.0
    alar_right_x = max(point.x for point in expanded["nose"].vertices if abs(point.y - max(p.y for p in expanded["nose"].vertices)) < 1e-6) - 2.0
    checks["classification_covers_both_nasal_alae"] = (
        classify_face_point(alar_left_x, alar_y, [face], aoi_settings=NEW_SETTINGS).region == "nose"
        and classify_face_point(alar_right_x, alar_y, [face], aoi_settings=NEW_SETTINGS).region == "nose"
    )

    checks["eye_priority_beats_overlapping_nose"] = (
        classify_face_point(
            face.left_eye.x,
            face.left_eye.y,
            [face],
            aoi_settings=NEW_SETTINGS,
        ).region
        == "left_eye"
    )
    mouth_center = Point2D(
        (face.mouth_left.x + face.mouth_right.x) / 2.0,
        (face.mouth_left.y + face.mouth_right.y) / 2.0,
    )
    checks["mouth_priority_beats_overlapping_nose"] = (
        classify_face_point(
            mouth_center.x,
            mouth_center.y,
            [face],
            aoi_settings=NEW_SETTINGS,
        ).region
        == "mouth"
    )

    # Noisy landmarks must still be clipped to the detector face rectangle.
    edge = FaceInstance(
        scene_frame_index=11,
        scene_timestamp_ns=1_033_333_333,
        track_id=8,
        explicit_entity_id=None,
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=140.0,
        left_eye=Point2D(3.0, 17.0),
        right_eye=Point2D(105.0, 25.0),
        nose=Point2D(115.0, 85.0),
        mouth_left=Point2D(15.0, 138.0),
        mouth_right=Point2D(105.0, 145.0),
        confidence=0.82,
        source_row_index=5,
    )
    edge_aois = _face_aois(edge, NEW_SETTINGS)
    checks["edge_landmarks_are_clipped"] = all(contained(aoi, edge) for aoi in edge_aois)
    checks["edge_clipping_is_audited"] = any(
        aoi.geometry_clipped for aoi in edge_aois if aoi.region in FINE_REGIONS
    )

    regions = face_instances_to_regions({face.scene_frame_index: [face]}, aoi_settings=NEW_SETTINGS)
    nose_row = regions.loc[regions["region"].eq("nose")].iloc[0]
    checks["region_table_has_polygon_audit"] = {
        "geometry_type",
        "geometry_json",
        "requested_geometry_json",
        "aoi_geometry_model",
        "aoi_polygon_vertex_count",
        "aoi_geometry_clipped",
        "aoi_containment_policy",
        "aoi_feature_scale",
        "aoi_contained_in_face_bbox",
    }.issubset(regions.columns)
    checks["nose_polygon_serialized"] = (
        nose_row["geometry_type"] == "polygon"
        and nose_row["aoi_geometry_model"] == "nasal_bridge_alar_envelope"
        and int(nose_row["aoi_polygon_vertex_count"]) == 6
        and len(json.loads(nose_row["geometry_json"])["points"]) == 6
    )
    checks["region_table_reports_full_containment"] = bool(regions["aoi_contained_in_face_bbox"].all())

    # Full-clip sample and frame statistics. There are eight valid gaze samples:
    # five strict partner-face hits, including two eye hits, one nose, one mouth,
    # and one unspecified/other face. Invalid and unworn samples are excluded.
    samples = pd.DataFrame([
        {"native_frame_index": 0, "gaze_x_px": 10.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "partner", "face_region": "left_eye"},
        {"native_frame_index": 1, "gaze_x_px": 11.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "partner", "face_region": "right_eye"},
        {"native_frame_index": 2, "gaze_x_px": 12.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "partner", "face_region": "nose"},
        {"native_frame_index": 3, "gaze_x_px": 13.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "partner", "face_region": "mouth"},
        {"native_frame_index": 4, "gaze_x_px": 14.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "partner", "face_region": "other_face"},
        {"native_frame_index": 5, "gaze_x_px": 15.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "unknown", "face_region": "no_face"},
        {"native_frame_index": 6, "gaze_x_px": 16.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "self", "face_region": "mouth"},
        {"native_frame_index": 7, "gaze_x_px": 17.0, "gaze_y_px": 10.0, "worn": True, "target_relation": "other_person", "face_region": "face"},
        {"native_frame_index": 8, "gaze_x_px": np.nan, "gaze_y_px": 10.0, "worn": True, "target_relation": "partner", "face_region": "invalid_gaze"},
        {"native_frame_index": 9, "gaze_x_px": 19.0, "gaze_y_px": 10.0, "worn": False, "target_relation": "partner", "face_region": "mouth"},
    ])
    frame_map = pd.DataFrame({"native_frame_index": list(range(10))})
    clip_summary, clip_table = _full_clip_partner_face_statistics(samples, frame_map)
    qc_fields = _clip_statistics_qc_fields(clip_summary)
    checks["clip_stats_valid_denominator"] = qc_fields["valid_gaze_sample_count"] == 8
    checks["clip_stats_partner_face_fraction"] = (
        qc_fields["partner_face_sample_count"] == 5
        and approx(qc_fields["partner_face_sample_fraction"], 5 / 8)
    )
    checks["clip_stats_feature_fractions"] = (
        approx(qc_fields["partner_eye_sample_fraction"], 2 / 8)
        and approx(qc_fields["partner_nose_sample_fraction"], 1 / 8)
        and approx(qc_fields["partner_mouth_sample_fraction"], 1 / 8)
    )
    checks["clip_stats_within_partner_face"] = (
        approx(qc_fields["partner_eye_share_within_partner_face"], 2 / 5)
        and approx(qc_fields["partner_nose_share_within_partner_face"], 1 / 5)
        and approx(qc_fields["partner_mouth_share_within_partner_face"], 1 / 5)
        and approx(qc_fields["partner_other_face_share_within_partner_face"], 1 / 5)
    )
    checks["clip_stats_frame_fraction"] = (
        qc_fields["clip_frame_count"] == 10
        and qc_fields["valid_gaze_frame_count"] == 8
        and approx(qc_fields["partner_face_frame_fraction"], 5 / 10)
    )
    checks["clip_stats_export_table_complete"] = (
        {"sample_based_full_clip", "within_partner_face", "frame_based_full_clip"}
        .issubset(set(clip_table["scope"]))
        and "percent" in clip_table.columns
    )

    request = FaceAOIJobRequest(session_id="session", native_trim_artifact_id="trim")
    checks["api_defaults"] = (
        request.fine_feature_aoi_scale == 1.55
        and request.eye_feature_aoi_scale == 1.35
        and request.nose_region_mode == "nasal_envelope"
        and request.mouth_center_policy == "adaptive_lip_envelope"
    )
    invalid_cases = [
        {"fine_feature_aoi_scale": 2.01},
        {"eye_feature_aoi_scale": 1.76},
        {"nose_region_mode": "triangle"},
    ]
    invalid_rejected = []
    for overrides in invalid_cases:
        try:
            FaceAOIJobRequest(session_id="session", native_trim_artifact_id="trim", **overrides)
        except PydanticValidationError:
            invalid_rejected.append(True)
        else:
            invalid_rejected.append(False)
    checks["api_rejects_invalid_geometry_settings"] = all(invalid_rejected)

    package_root = Path(oculostudio.__file__).resolve().parent
    app_js = (package_root / "static" / "app.js").read_text("utf-8")
    review_js = (package_root / "static" / "aoi_review.js").read_text("utf-8")
    checks["ui_has_expanded_geometry_controls"] = all(
        token in app_js
        for token in (
            'id="face-aoi-feature-scale"',
            'value="1.55" selected',
            'id="face-aoi-eye-scale"',
            'value="1.35" selected',
            'id="face-aoi-nose-mode"',
            'value="nasal_envelope" selected',
            "eye_feature_aoi_scale:Number($('#face-aoi-eye-scale')?.value||1.35)",
            "nose_region_mode:$('#face-aoi-nose-mode')?.value||'nasal_envelope'",
            'value="adaptive_lip_envelope" selected',
            "mouth_center_policy:$('#face-aoi-mouth-center')?.value||'adaptive_lip_envelope'",
        )
    )
    checks["ui_has_full_clip_statistics"] = all(
        token in app_js
        for token in (
            "Partner-face gaze proportions",
            "partner_face_sample_fraction",
            "partner_eye_share_within_partner_face",
            "partner_nose_share_within_partner_face",
            "partner_mouth_share_within_partner_face",
            "Full-clip partner-face gaze",
        )
    )
    checks["live_review_draws_polygons"] = all(
        token in review_js for token in ("==='polygon'", "ctx.lineTo", "ctx.closePath")
    )

    service_source = inspect.getsource(create_face_aoi_annotations)
    checks["single_geometry_drives_hits_regions_and_exports"] = all(
        token in service_source
        for token in (
            "aoi_settings=aoi_settings",
            "face_instances_to_regions(corrected_instances_by_frame, aoi_settings=aoi_settings)",
            "edge_eligible_source_row_indices=edge_eligible_source_rows",
            '"clip_summary_csv": "face_aoi_clip_summary.csv"',
            '"clip_summary_json": "face_aoi_clip_summary.json"',
        )
    )

    report = {
        "application_version": oculostudio.__version__,
        "feature": "Green-box eyes, nasal bridge/alae envelope, balanced lips and complete-clip partner-face statistics",
        "defaults": {
            "fine_feature_aoi_scale": 1.55,
            "eye_feature_aoi_scale": 1.35,
            "effective_eye_scale": 1.55 * 1.35,
            "nose_region_mode": "nasal_envelope",
            "mouth_region_mode": "adaptive_lip_envelope",
            "containment_policy": "face_bbox_inner_clip",
        },
        "checks": checks,
        "all_passed": all(checks.values()),
        "geometry_example": {
            "previous_eye_radius": {
                "x": previous["left_eye"].radius_x,
                "y": previous["left_eye"].radius_y,
            },
            "expanded_eye_radius": {
                "x": expanded["left_eye"].radius_x,
                "y": expanded["left_eye"].radius_y,
            },
            "nose_polygon": [[point.x, point.y] for point in expanded["nose"].vertices],
            "mouth_polygon": [[point.x, point.y] for point in expanded["mouth"].vertices],
        },
        "statistics_example": clip_summary,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
