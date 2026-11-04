"""Targeted validation for v0.21.71 nasal envelope and direct FM-hit acceptance."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.server import FaceAOIJobRequest
from oculostudio.services.face_aoi import FACE_AOI_ENGINE, _apply_auto_review_reduction
from oculostudio.services.face_semantics import (
    FaceInstance,
    Point2D,
    PolygonAOI,
    _face_aois,
    classify_face_point,
)


def settings() -> dict:
    return {
        "bounded_features": True,
        "feature_scale": 1.55,
        "eye_feature_scale": 1.35,
        "nose_region_mode": "nasal_envelope",
        "mouth_center_policy": "adaptive_lip_envelope",
        "nose_base_offset_ratio": 0.30,
        "nasal_envelope_top_y_fraction": 0.16,
        "nasal_envelope_top_halfwidth_iod_ratio": 0.16,
        "nasal_envelope_top_halfwidth_face_ratio": 0.055,
        "nasal_envelope_mid_halfwidth_iod_ratio": 0.22,
        "nasal_envelope_mid_halfwidth_face_ratio": 0.075,
        "nasal_envelope_bottom_halfwidth_iod_ratio": 0.30,
        "nasal_envelope_bottom_halfwidth_face_ratio": 0.12,
        "nasal_envelope_bottom_halfwidth_mouth_ratio": 0.42,
        "mouth_lateral_padding_ratio": 0.15,
        "mouth_top_padding_face_ratio": 0.045,
        "mouth_bottom_padding_face_ratio": 0.095,
        "mouth_max_height_face_ratio": 0.18,
        "mouth_absolute_max_height_face_ratio": 0.22,
        "mouth_max_width_face_ratio": 0.52,
        "mouth_bottom_limit_face_ratio": 0.90,
        "mouth_min_top_below_nose_ratio": 0.015,
        "face_margin_ratio": 0.02,
    }


def face() -> FaceInstance:
    return FaceInstance(
        scene_frame_index=0,
        scene_timestamp_ns=0,
        track_id=7,
        explicit_entity_id="R02",
        x1=100.0,
        y1=100.0,
        x2=300.0,
        y2=400.0,
        left_eye=Point2D(164.0, 184.0),
        right_eye=Point2D(236.0, 184.0),
        nose=Point2D(200.0, 256.0),
        mouth_left=Point2D(172.0, 310.0),
        mouth_right=Point2D(228.0, 310.0),
        confidence=0.90,
        source_row_index=17,
    )


def review_row(**updates) -> dict:
    row = {
        "review_required": True,
        "assignment_status": "participant_face_aoi_hit",
        "green_face_exact_hit": True,
        "green_face_edge_tolerance_hit": False,
        "face_edge_ambiguous": False,
        "include_in_primary_analysis": True,
        "target_relation": "partner",
        "face_entity_id": "R02",
        "face_region": "other_face",
        "raw_face_region": "other_face",
        "identity_status": "matched_participant",
        "face_aoi_confidence": 0.48,
        "identity_confidence": 0.22,
        "candidate_entity_id": "R02",
        "candidate_relation": "partner",
        "native_frame_index": 10,
        "timestamp_ns": 1_000_000,
        "fixation_id": "fix-1",
        "face_track_id": 7,
        "fm_on_face_match_count": 1,
    }
    row.update(updates)
    return row


def main() -> int:
    checks: dict[str, bool] = {}
    instance = face()
    nose = next(aoi for aoi in _face_aois(instance, settings()) if aoi.region == "nose")
    checks["single_six_vertex_nasal_envelope"] = bool(
        isinstance(nose, PolygonAOI)
        and nose.geometry_model == "nasal_bridge_alar_envelope"
        and len(nose.vertices) == 6
    )
    if isinstance(nose, PolygonAOI) and len(nose.vertices) == 6:
        top_width = nose.vertices[1].x - nose.vertices[0].x
        mid_width = nose.vertices[2].x - nose.vertices[5].x
        base_width = nose.vertices[3].x - nose.vertices[4].x
        checks["monotonic_width_no_hourglass"] = 0 < top_width <= mid_width <= base_width
        checks["alar_base_materially_wider_than_bridge"] = base_width >= 1.6 * top_width
        alar_y = nose.vertices[3].y - 1.0
        checks["left_ala_classifies_nose"] = (
            classify_face_point(nose.vertices[4].x + 2.0, alar_y, [instance], aoi_settings=settings()).region
            == "nose"
        )
        checks["right_ala_classifies_nose"] = (
            classify_face_point(nose.vertices[3].x - 2.0, alar_y, [instance], aoi_settings=settings()).region
            == "nose"
        )
    else:
        checks["monotonic_width_no_hourglass"] = False
        checks["alar_base_materially_wider_than_bridge"] = False
        checks["left_ala_classifies_nose"] = False
        checks["right_ala_classifies_nose"] = False

    direct, direct_report, _ = _apply_auto_review_reduction(
        pd.DataFrame([review_row()]), "adaptive"
    )
    checks["exact_fm_hit_auto_accepted"] = bool(
        not direct.loc[0, "review_required"]
        and direct.loc[0, "auto_review_resolved"]
        and direct.loc[0, "auto_review_resolution_source"] == "direct_green_box_matched_identity"
        and direct.loc[0, "face_entity_id"] == "R02"
        and direct_report["review_samples_remaining"] == 0
    )
    ambiguous, _, _ = _apply_auto_review_reduction(
        pd.DataFrame([
            review_row(
                identity_status="low_confidence_participant_candidate",
                target_relation="ambiguous_partner",
                face_entity_id=None,
                include_in_primary_analysis=False,
                candidate_entity_id="R02",
                assignment_status="low_confidence_participant_candidate",
            )
        ]),
        "adaptive",
    )
    checks["identity_ambiguity_still_requires_review"] = bool(ambiguous.loc[0, "review_required"])
    edge, _, _ = _apply_auto_review_reduction(
        pd.DataFrame([
            review_row(
                green_face_exact_hit=False,
                green_face_edge_tolerance_hit=True,
                identity_status="low_confidence_participant_candidate",
                target_relation="ambiguous_partner",
                face_entity_id=None,
                include_in_primary_analysis=False,
                assignment_status="low_confidence_participant_candidate",
            )
        ]),
        "adaptive",
    )
    checks["edge_band_not_mistaken_for_direct_fm_hit"] = bool(edge.loc[0, "review_required"])

    request = FaceAOIJobRequest(session_id="session", native_trim_artifact_id="trim")
    checks["api_defaults"] = bool(
        request.nose_region_mode == "nasal_envelope"
        and request.auto_review_policy == "adaptive"
    )
    package_root = Path(oculostudio.__file__).resolve().parent
    app_js = (package_root / "static" / "app.js").read_text("utf-8")
    index_html = (package_root / "static" / "index.html").read_text("utf-8")
    checks["ui_defaults_and_explanation"] = all(
        token in app_js
        for token in (
            'value="nasal_envelope" selected',
            "Nasal bridge + both alae",
            "directly inside an identity-resolved green FaceMapper box is accepted automatically",
            "direct_green_box_auto_accepted_sample_count",
        )
    )
    checks["version_engine_cache"] = bool(
        oculostudio.__version__ == "0.21.71"
        and FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
        and "app.js?v=0.21.71" in index_html
    )

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "release": "OculoStudio v0.21.71",
        "feature": "Nasal bridge/alae envelope and direct green-box auto acceptance",
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
        "all_passed": not failed,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
