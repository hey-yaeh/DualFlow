"""Targeted validation for v0.21.71 overlap identity and balanced facial AOIs."""
from __future__ import annotations

import json
from pathlib import Path

import oculostudio
from oculostudio.server import FaceAOIJobRequest
from oculostudio.services.face_aoi import FACE_AOI_ENGINE
from oculostudio.services.face_semantics import (
    EllipseAOI,
    FaceInstance,
    Point2D,
    PolygonAOI,
    _face_aois,
    classify_face_point,
)
from oculostudio.services.participant_association import (
    PARTICIPANT_ASSOCIATION_ENGINE,
    _associate_frame,
    _pair_metrics,
)


def make_face(x1: float, y1: float, x2: float, y2: float, track: int = 1, *, mouth: bool = True) -> FaceInstance:
    width = x2 - x1
    height = y2 - y1
    return FaceInstance(
        scene_frame_index=0,
        scene_timestamp_ns=0,
        track_id=track,
        explicit_entity_id=None,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        left_eye=Point2D(x1 + 0.32 * width, y1 + 0.28 * height),
        right_eye=Point2D(x1 + 0.68 * width, y1 + 0.28 * height),
        nose=Point2D(x1 + 0.50 * width, y1 + 0.52 * height),
        mouth_left=Point2D(x1 + 0.36 * width, y1 + 0.70 * height) if mouth else None,
        mouth_right=Point2D(x1 + 0.64 * width, y1 + 0.70 * height) if mouth else None,
        confidence=0.90,
        source_row_index=track,
    )


def projection(x1=100.0, y1=100.0, x2=220.0, y2=220.0, entity="R02") -> dict:
    return {
        "projection_valid": True,
        "in_camera_view": True,
        "role_id": entity,
        "entity_id": entity,
        "relation_to_wearer": "partner",
        "projected_bbox_x1": x1,
        "projected_bbox_y1": y1,
        "projected_bbox_x2": x2,
        "projected_bbox_y2": y2,
        "projected_head_x_px": 0.5 * (x1 + x2),
        "projected_head_y_px": 0.5 * (y1 + y2),
        "projection_confidence": 0.20,
        "head_facing_score": 0.0,
    }


def associate(projections: list[dict], faces: list[FaceInstance]) -> list[dict]:
    return _associate_frame(
        projections,
        faces,
        previous_track_by_role={},
        maximum_normalized_center_error=1.25,
        minimum_projected_iou=0.015,
        minimum_assignment_margin=0.08,
        minimum_identity_confidence=0.50,
    )[0]


def settings(policy: str = "adaptive_lip_envelope", feature_scale: float = 1.55) -> dict:
    return {
        "bounded_features": True,
        "feature_scale": feature_scale,
        "eye_feature_scale": 1.35,
        "nose_region_mode": "nasal_envelope",
        "mouth_center_policy": policy,
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


def main() -> int:
    checks: dict[str, bool] = {}

    p = projection()
    face = make_face(215, 145, 285, 225)
    metrics = _pair_metrics(p, face, None)
    result = associate([p], [face])[0]
    checks["positive_overlap_detected"] = bool(metrics["bbox_overlap_present"])
    checks["small_overlap_not_vetoed_by_center"] = (
        result["identity_status"] == "overlap_locked_participant"
        and result["matched_entity_id"] == "R02"
        and result["target_relation"] == "partner"
        and result["review_required"] is False
    )
    no_overlap = associate([p], [make_face(221, 221, 281, 281)])[0]
    checks["no_overlap_not_partner"] = no_overlap["target_relation"] == "other_person"
    ambiguous = associate([p], [make_face(110, 115, 160, 180, 1), make_face(175, 120, 218, 182, 2)])
    checks["multiple_distinct_overlap_protected"] = (
        len(ambiguous) == 2
        and all(row["review_required"] for row in ambiguous)
        and all(row["target_relation"] == "ambiguous_partner" for row in ambiguous)
    )

    green_face = make_face(100, 100, 300, 400)
    aois = _face_aois(green_face, settings())
    nose = next(aoi for aoi in aois if aoi.region == "nose")
    mouth = next(aoi for aoi in aois if aoi.region == "mouth")
    eyes = [aoi for aoi in aois if aoi.region in {"left_eye", "right_eye"}]
    checks["nose_is_bridge_to_alae_envelope"] = (
        isinstance(nose, PolygonAOI)
        and nose.geometry_model == "nasal_bridge_alar_envelope"
        and len(nose.vertices) == 6
    )
    if isinstance(nose, PolygonAOI) and len(nose.vertices) == 6:
        widths = (
            nose.vertices[1].x - nose.vertices[0].x,
            nose.vertices[2].x - nose.vertices[5].x,
            nose.vertices[3].x - nose.vertices[4].x,
        )
        checks["nose_width_increases_toward_alae"] = widths[0] <= widths[1] <= widths[2]
        checks["both_nasal_alae_are_covered"] = (
            classify_face_point(nose.vertices[4].x + 2.0, nose.vertices[4].y - 1.0, [green_face], aoi_settings=settings()).region == "nose"
            and classify_face_point(nose.vertices[3].x - 2.0, nose.vertices[3].y - 1.0, [green_face], aoi_settings=settings()).region == "nose"
        )
    else:
        checks["nose_width_increases_toward_alae"] = False
        checks["both_nasal_alae_are_covered"] = False
    checks["mouth_is_balanced_lip_polygon"] = isinstance(mouth, PolygonAOI) and mouth.geometry_model == "adaptive_lip_envelope"
    if isinstance(mouth, PolygonAOI):
        mouth_top = min(point.y for point in mouth.vertices)
        mouth_bottom = max(point.y for point in mouth.vertices)
        mouth_left = min(point.x for point in mouth.vertices)
        mouth_right = max(point.x for point in mouth.vertices)
        checks["mouth_height_is_bounded"] = (mouth_bottom - mouth_top) <= green_face.height * 0.18 + 1e-6
        checks["mouth_width_is_bounded"] = (mouth_right - mouth_left) <= green_face.width * 0.52 + 1e-6
        checks["mouth_stops_above_chin"] = mouth_bottom <= green_face.y1 + 0.90 * green_face.height + 1e-6 and mouth_bottom < green_face.y2
        mouth_y = 0.5 * (green_face.mouth_left.y + green_face.mouth_right.y)
        checks["mouth_landmark_classifies_as_mouth"] = classify_face_point(200.0, mouth_y, [green_face], aoi_settings=settings()).region == "mouth"
        checks["chin_classifies_as_other_face"] = classify_face_point(200.0, green_face.y1 + 0.95 * green_face.height, [green_face], aoi_settings=settings()).region == "other_face"
    checks["eyes_remain_bilateral_ellipses"] = len(eyes) == 2 and all(isinstance(aoi, EllipseAOI) for aoi in eyes)

    # Mouth geometry must not grow when only eye/nose feature scale changes.
    mouth_large = next(aoi for aoi in _face_aois(green_face, settings(feature_scale=1.75)) if aoi.region == "mouth")
    checks["mouth_independent_of_global_feature_scale"] = (
        isinstance(mouth, PolygonAOI) and isinstance(mouth_large, PolygonAOI)
        and [(round(p.x, 6), round(p.y, 6)) for p in mouth.vertices]
        == [(round(p.x, 6), round(p.y, 6)) for p in mouth_large.vertices]
    )

    # Missing mouth landmarks must not synthesize a mouth AOI.
    no_landmarks = make_face(100, 100, 300, 400, mouth=False)
    no_landmark_aois = _face_aois(no_landmarks, settings())
    checks["missing_mouth_landmarks_do_not_invent_mouth"] = not any(aoi.region == "mouth" for aoi in no_landmark_aois)
    checks["missing_mouth_landmarks_fall_back_to_other_face"] = classify_face_point(200, 310, [no_landmarks], aoi_settings=settings()).region == "other_face"

    broad = next(aoi for aoi in _face_aois(green_face, settings("speech_sensitive_mouth_jaw")) if aoi.region == "mouth")
    checks["speech_sensitive_mode_retained_explicitly"] = (
        isinstance(broad, PolygonAOI)
        and broad.geometry_model == "speech_sensitive_mouth_jaw"
        and abs(max(p.y for p in broad.vertices) - green_face.y2) < 1e-6
    )

    request = FaceAOIJobRequest(session_id="session", native_trim_artifact_id="trim")
    checks["api_default_balanced_lips_and_nose"] = (request.mouth_center_policy == "adaptive_lip_envelope" and request.nose_region_mode == "nasal_envelope")
    checks["version_and_engines"] = (
        oculostudio.__version__ == "0.21.71"
        and PARTICIPANT_ASSOCIATION_ENGINE == "projected-box-overlap-identity-v4"
        and FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
    )

    package_root = Path(oculostudio.__file__).resolve().parent
    app_js = (package_root / "static" / "app.js").read_text("utf-8")
    index = (package_root / "static" / "index.html").read_text("utf-8")
    checks["ui_describes_overlap_identity"] = all(token in app_js for token in (
        "Role-locked partner-head projection",
        "light-blue projected partner-head box is an identity gate only",
        "genuinely ambiguous identities remain reviewable",
    ))
    checks["ui_defaults_to_balanced_lips"] = all(token in app_js for token in (
        'value="nasal_envelope" selected',
        "Nasal bridge + both alae",
        'value="adaptive_lip_envelope" selected',
        "Balanced lips · recommended",
        "remaining lower face is other_face",
        "Speech-sensitive mouth/jaw · broad",
    ))
    checks["cache_busting_version"] = "app.js?v=0.21.71" in index and "styles.css?v=0.21.71" in index

    failed = [name for name, value in checks.items() if not value]
    report = {
        "release": "OculoStudio v0.21.71",
        "feature": "Overlap identity gate, nasal bridge/alae envelope and balanced lip/other-face partition",
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
