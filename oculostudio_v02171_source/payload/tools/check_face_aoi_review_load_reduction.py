"""Targeted validation for OculoStudio v0.21.71 review-load reduction."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

import oculostudio
from oculostudio.server import FaceAOIJobRequest, FaceAOIReviewDecisionRequest
from oculostudio.services.face_aoi import (
    FACE_AOI_ENGINE,
    _apply_auto_review_reduction,
    _facemapper_on_face_consensus,
    _unique_projection_candidate,
)
from oculostudio.services.face_semantics import (
    FaceInstance,
    Point2D,
    _face_aois,
    classify_face_point,
)


def main() -> int:
    checks: dict[str, bool] = {}

    value, conflict, duplicate, unique_count = _facemapper_on_face_consensus([True, True, None])
    checks["duplicate_on_face_rows_are_consensus_not_conflict"] = bool(
        value is True and not conflict and duplicate and unique_count == 1
    )
    value2, conflict2, duplicate2, unique_count2 = _facemapper_on_face_consensus([True, False])
    checks["contradictory_on_face_rows_remain_conflict"] = bool(
        value2 is None and conflict2 and not duplicate2 and unique_count2 == 2
    )

    projection = {
        "native_frame_index": 10,
        "projection_valid": True,
        "in_camera_view": True,
        "relation_to_wearer": "partner",
        "entity_id": "R02",
        "role_id": "R02",
        "projected_bbox_x1": 100,
        "projected_bbox_y1": 100,
        "projected_bbox_x2": 300,
        "projected_bbox_y2": 360,
        "projection_confidence": 0.82,
    }
    candidate = _unique_projection_candidate(
        200,
        220,
        [projection],
        target_person_policy="partner_only",
        auto_review_policy="adaptive",
    )
    checks["unique_partner_projection_candidate"] = bool(
        candidate and candidate["entity_id"] == "R02" and candidate["relation"] == "partner"
    )
    ambiguous = _unique_projection_candidate(
        200,
        220,
        [projection, {**projection, "entity_id": "R03", "role_id": "R03"}],
        target_person_policy="partner_only",
        auto_review_policy="adaptive",
    )
    checks["multiple_projection_candidates_are_not_guessed"] = ambiguous is None

    row = {
        "review_required": True,
        "fm_gaze_on_face": True,
        "fm_on_face_conflict": False,
        "projection_fallback_relation": "partner",
        "projection_fallback_entity_id": "R02",
        "projection_fallback_confidence": 0.76,
        "raw_face_region": None,
        "face_region": "no_face",
        "face_entity_id": None,
        "target_relation": None,
        "candidate_entity_id": None,
        "candidate_relation": None,
        "face_aoi_confidence": 0.55,
        "identity_status": "missing_identity_row",
        "assignment_status": "outside_all_accepted_facemapper_face_boxes",
        "include_in_primary_analysis": False,
        "native_frame_index": 10,
        "fm_on_face_match_count": 2,
        "fixation_id": "fix-1",
        "timestamp_ns": 10_000_000,
    }
    resolved, report, table = _apply_auto_review_reduction(pd.DataFrame([row]), "adaptive")
    checks["strict_green_box_gate_overrides_projection_and_fm_flag"] = bool(
        not resolved.loc[0, "review_required"]
        and pd.isna(resolved.loc[0, "face_entity_id"])
        and resolved.loc[0, "face_region"] == "no_face"
        and resolved.loc[0, "face_social_region"] == "no_face"
        and resolved.loc[0, "assignment_status"] == "outside_all_accepted_facemapper_face_boxes"
        and resolved.loc[0, "auto_review_resolution_source"] == "strict_green_box_no_face"
        and not bool(resolved.loc[0, "include_in_primary_analysis"])
    )
    checks["review_reason_audit_preserves_original_reason"] = bool(
        report["review_reason_counts_before"] == {"outside_all_accepted_facemapper_face_boxes": 1}
        and report["remaining_review_reason_counts"] == {}
        and "remaining_review_reason" not in set(table.loc[table["count"] > 0, "scope"])
    )
    conservative, _, _ = _apply_auto_review_reduction(
        pd.DataFrame([{**row, "projection_fallback_confidence": 0.99}]), "conservative"
    )
    checks["projection_confidence_cannot_turn_outside_gaze_into_face"] = bool(
        not conservative.loc[0, "review_required"]
        and conservative.loc[0, "face_region"] == "no_face"
        and pd.isna(conservative.loc[0, "face_entity_id"])
    )
    invalid, _, _ = _apply_auto_review_reduction(
        pd.DataFrame([{**row, "face_region": "invalid_gaze", "assignment_status": "not_worn"}]),
        "adaptive",
    )
    checks["invalid_gaze_never_auto_promoted"] = bool(invalid.loc[0, "review_required"])

    instance = FaceInstance(
        scene_frame_index=0,
        scene_timestamp_ns=0,
        track_id=1,
        explicit_entity_id="R02",
        x1=100,
        y1=100,
        x2=300,
        y2=400,
        left_eye=Point2D(230, 180),
        right_eye=Point2D(170, 180),
        nose=Point2D(200, 240),
        mouth_left=Point2D(175, 300),
        mouth_right=Point2D(225, 300),
        confidence=0.9,
        source_row_index=0,
    )
    settings = {
        "bounded_features": True,
        "feature_scale": 1.55,
        "eye_feature_scale": 1.35,
        "nose_region_mode": "bridge_polygon",
        "face_margin_ratio": 0.02,
        "mouth_center_offset_ratio": 0.12,
    }
    legacy_aois = _face_aois(instance, {**settings, "mouth_center_policy": "landmark_midpoint"})
    lowered_aois = _face_aois(instance, {**settings, "mouth_center_policy": "lowered_landmark_midpoint"})
    legacy_mouth = next(aoi for aoi in legacy_aois if aoi.region == "mouth")
    lowered_mouth = next(aoi for aoi in lowered_aois if aoi.region == "mouth")
    full_face = next(aoi for aoi in lowered_aois if aoi.region == "other_face")
    checks["mouth_aoi_is_shifted_down_not_up"] = bool(
        lowered_mouth.center.y > legacy_mouth.center.y
        and lowered_mouth.center.y + lowered_mouth.radius_y <= instance.y2 + 1e-6
    )
    checks["full_face_gate_uses_exact_green_face_rectangle"] = bool(
        getattr(full_face, "geometry_model", "") == "face_bbox_exact_rectangle"
    )
    corner_hit = classify_face_point(108, 108, [instance], aoi_settings={**settings, "mouth_center_policy": "lowered_landmark_midpoint"})
    checks["face_box_corner_no_longer_becomes_unnecessary_review"] = bool(
        corner_hit and corner_hit.region == "other_face"
    )

    request = FaceAOIJobRequest(session_id="S001", native_trim_artifact_id="trim")
    checks["api_defaults_to_adaptive_and_balanced_lips"] = bool(
        request.auto_review_policy == "adaptive"
        and request.mouth_center_policy == "adaptive_lip_envelope"
    )
    fixation_request = FaceAOIReviewDecisionRequest(
        video_artifact_id="privacy",
        output_frame_index=1,
        gaze_sample_uid="g1",
        apply_to="fixation",
        action="accept_auto",
    )
    checks["api_accepts_fixation_batch_review"] = fixation_request.apply_to == "fixation"
    try:
        FaceAOIJobRequest(
            session_id="S001",
            native_trim_artifact_id="trim",
            auto_review_policy="unsafe",  # type: ignore[arg-type]
        )
        checks["api_rejects_unknown_auto_policy"] = False
    except PydanticValidationError:
        checks["api_rejects_unknown_auto_policy"] = True

    root = Path(oculostudio.__file__).parent
    app_js = (root / "static" / "app.js").read_text("utf-8")
    review_js = (root / "static" / "aoi_review.js").read_text("utf-8")
    checks["ui_exposes_adaptive_policy_and_mouth_position"] = all(
        token in app_js
        for token in (
            'value="adaptive" selected',
            'id="face-aoi-mouth-center"',
            "mouth_center_policy:$('#face-aoi-mouth-center')",
            "remaining_review_reason_counts",
        )
    )
    checks["ui_exposes_fixation_batch_scope"] = all(
        token in review_js
        for token in (
            '<option value="fixation">All flagged samples in this fixation</option>',
            "pending_review_unit_count",
        )
    )
    checks["version_and_engine_updated"] = bool(
        oculostudio.__version__ == "0.21.71"
        and FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
    )

    output = {
        "application_version": oculostudio.__version__,
        "feature": "Strict green-box review reduction and balanced lip partition",
        "checks": checks,
        "all_passed": all(checks.values()),
        "example_report": report,
        "mouth_centres": {
            "legacy_y": legacy_mouth.center.y,
            "lowered_y": lowered_mouth.center.y,
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
