"""Targeted validation for v0.21.71 edge-inclusive green face membership."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind
from oculostudio.server import FaceAOIJobRequest
from oculostudio.services.face_aoi import FACE_AOI_ENGINE, create_face_aoi_annotations
from oculostudio.services.face_semantics import FaceInstance, Point2D, classify_face_point
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def make_face(*, source_row: int = 7, explicit: str | None = "R02") -> FaceInstance:
    return FaceInstance(
        scene_frame_index=0,
        scene_timestamp_ns=0,
        track_id=2,
        explicit_entity_id=explicit,
        x1=100.0,
        y1=100.0,
        x2=300.0,
        y2=400.0,
        left_eye=Point2D(170.0, 180.0),
        right_eye=Point2D(230.0, 180.0),
        nose=Point2D(200.0, 245.0),
        mouth_left=Point2D(175.0, 305.0),
        mouth_right=Point2D(225.0, 305.0),
        confidence=0.95,
        source_row_index=source_row,
    )


def settings(policy: str = "edge_inclusive") -> dict:
    profiles = {
        "exact": (0.0, 0.0, 0.0),
        "edge_inclusive": (0.045, 6.0, 18.0),
        "forgiving": (0.075, 8.0, 26.0),
    }
    ratio, minimum, maximum = profiles[policy]
    return {
        "bounded_features": True,
        "feature_scale": 1.55,
        "eye_feature_scale": 1.35,
        "nose_region_mode": "bridge_polygon",
        "mouth_center_policy": "adaptive_lip_envelope",
        "face_margin_ratio": 0.02,
        "face_edge_tolerance_policy": policy,
        "face_edge_tolerance_ratio": ratio,
        "face_edge_tolerance_min_px": minimum,
        "face_edge_tolerance_max_px": maximum,
    }


def build_end_to_end(checks: dict[str, bool]) -> None:
    with tempfile.TemporaryDirectory(prefix="oculostudio_edge_face_") as temp:
        store = ProjectStore.create(Path(temp) / "project", "Edge face validation", session_label="S001")
        session_id = store.project.selected_session_id
        artifacts = ArtifactStore(store)
        trim_root = store.derived_dir(session_id, "native_reference_trim", "edge_trim")
        (trim_root / "tsd").mkdir(parents=True, exist_ok=True)
        (trim_root / "face_mapper").mkdir(parents=True, exist_ok=True)
        timestamps = [1_000_000_000 + i * 33_333_333 for i in range(4)]
        pd.DataFrame({
            "native_frame_index": range(4),
            "source_frame_ordinal_in_window": range(4),
            "clip_video_pts_ns": [i * 33_333_333 for i in range(4)],
            "native_capture_utc_ns": timestamps,
        }).to_csv(trim_root / "native_frame_map.csv", index=False)
        pd.DataFrame({
            "native_frame_index": range(4),
            "native_capture_utc_ns": timestamps,
        }).to_csv(trim_root / "native_clock_full.csv", index=False)
        (trim_root / "trim_manifest.json").write_text(json.dumps({
            "recording_id": "rec-edge",
            "native_reference": {"width": 640, "height": 480},
        }), encoding="utf-8")
        pd.DataFrame([
            {"timestamp [ns]": timestamps[0], "gaze x [px]": 300.0, "gaze y [px]": 220.0,
             "oculo_nearest_native_frame_index": 0, "oculo_source_row_index": 0, "worn": True},
            {"timestamp [ns]": timestamps[1], "gaze x [px]": 305.0, "gaze y [px]": 220.0,
             "oculo_nearest_native_frame_index": 1, "oculo_source_row_index": 1, "worn": True},
            {"timestamp [ns]": timestamps[2], "gaze x [px]": 325.0, "gaze y [px]": 220.0,
             "oculo_nearest_native_frame_index": 2, "oculo_source_row_index": 2, "worn": True},
            {"timestamp [ns]": timestamps[3], "gaze x [px]": 200.0, "gaze y [px]": 220.0,
             "oculo_nearest_native_frame_index": 3, "oculo_source_row_index": 3, "worn": True},
        ]).to_csv(trim_root / "tsd" / "gaze.csv", index=False)
        detections = []
        for i in range(3):
            detections.append({
                "timestamp [ns]": timestamps[i], "recording id": "rec-edge",
                "p1 x [px]": 100.0, "p1 y [px]": 100.0,
                "p2 x [px]": 300.0, "p2 y [px]": 400.0,
                "left eye x [px]": 230.0, "left eye y [px]": 180.0,
                "right eye x [px]": 170.0, "right eye y [px]": 180.0,
                "nose x [px]": 200.0, "nose y [px]": 245.0,
                "left mouth corner x [px]": 175.0, "left mouth corner y [px]": 305.0,
                "right mouth corner x [px]": 225.0, "right mouth corner y [px]": 305.0,
                "confidence": 0.95,
            })
        pd.DataFrame(detections).to_csv(trim_root / "face_mapper" / "face_detections.csv", index=False)
        pd.DataFrame([
            {"timestamp [ns]": ts, "gaze on face": True} for ts in timestamps
        ]).to_csv(trim_root / "face_mapper" / "gaze_on_face.csv", index=False)
        trim = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
            label="S001_I01_R01_trim",
            output_root=trim_root,
        )

        association_root = store.derived_dir(session_id, "participant_association", "edge_association")
        projections = []
        identities = []
        for i in range(3):
            projections.append({
                "native_frame_index": i,
                "projection_valid": True,
                "in_camera_view": True,
                "entity_id": "R02",
                "role_id": "R02",
                "relation_to_wearer": "partner",
                "projected_bbox_x1": 95.0,
                "projected_bbox_y1": 90.0,
                "projected_bbox_x2": 310.0,
                "projected_bbox_y2": 410.0,
            })
            identities.append({
                "native_frame_index": i,
                "face_source_row_index": i,
                "face_track_id": 0,
                "matched_entity_id": "R02",
                "matched_role": "R02",
                "target_relation": "partner",
                "identity_status": "overlap_locked_participant",
                "identity_confidence": 0.92,
                "candidate_entity_id": "R02",
                "candidate_role": "R02",
                "candidate_relation": "partner",
                "include_in_partner_analysis": True,
                "review_required": False,
                "bbox_overlap_present": True,
                "face_box_overlap_fraction": 1.0,
            })
        pd.DataFrame(projections).to_csv(association_root / "participant_head_projection.csv", index=False)
        pd.DataFrame(identities).to_csv(association_root / "face_identity_assignment.csv", index=False)
        (association_root / "participant_association_manifest.json").write_text(json.dumps({
            "native_trim_artifact_id": trim.id,
            "wearer_role_id": "R01",
            "role_context": {"wearer_role_id": "R01", "partner_role_ids": ["R02"]},
        }), encoding="utf-8")
        association = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.PARTICIPANT_ASSOCIATION,
            label="S001_I01_R01_participant_association",
            output_root=association_root,
            parent_artifact_ids=[trim.id],
        )
        result = create_face_aoi_annotations(
            store,
            session_id=session_id,
            native_trim_artifact_id=trim.id,
            participant_association_artifact_id=association.id,
            target_person_policy="partner_only",
            face_left_right_policy="as_exported",
            face_edge_tolerance_policy="edge_inclusive",
            render_overlay=False,
            output_parquet=False,
            label="edge_face_aoi",
        )
        samples = pd.read_csv(result["samples_csv"], low_memory=False).set_index("source_gaze_row_index")
        checks["exact_boundary_is_r02_face"] = (
            samples.loc[0, "face_entity_id"] == "R02"
            and samples.loc[0, "face_social_region"] == "partner_face_unspecified"
            and bool(samples.loc[0, "green_face_exact_hit"])
        )
        checks["near_edge_is_r02_face"] = (
            samples.loc[1, "face_entity_id"] == "R02"
            and samples.loc[1, "face_social_region"] == "partner_face_unspecified"
            and bool(samples.loc[1, "green_face_edge_tolerance_hit"])
            and samples.loc[1, "face_gate_hit_type"] == "green_box_edge_tolerance"
            and samples.loc[1, "assignment_status"] == "participant_face_aoi_edge_tolerance_hit"
            and not bool(samples.loc[1, "review_required"])
        )
        checks["far_outside_remains_no_face"] = (
            samples.loc[2, "face_region"] == "no_face"
            and pd.isna(samples.loc[2, "face_entity_id"])
            and not bool(samples.loc[2, "green_face_geometry_hit"])
        )
        checks["missing_detection_remains_distinct"] = samples.loc[3, "face_region"] == "no_face_detection"
        manifest = json.loads((Path(result["samples_csv"]).parent / "face_aoi_manifest.json").read_text("utf-8"))
        checks["manifest_audits_edge_policy"] = (
            manifest["schema_version"] == "oculostudio-face-aoi-1.13"
            and manifest["parameters"]["face_edge_tolerance_policy"] == "edge_inclusive"
            and manifest["parameters"]["fine_feature_edge_expansion"] is False
        )


def main() -> int:
    checks: dict[str, bool] = {}
    face = make_face()
    exact = classify_face_point(300.0, 220.0, [face], aoi_settings=settings(), edge_eligible_source_row_indices={7})
    edge = classify_face_point(305.0, 220.0, [face], aoi_settings=settings(), edge_eligible_source_row_indices={7})
    far = classify_face_point(320.0, 220.0, [face], aoi_settings=settings(), edge_eligible_source_row_indices={7})
    strict = classify_face_point(305.0, 220.0, [face], aoi_settings=settings("exact"), edge_eligible_source_row_indices={7})
    ineligible = classify_face_point(305.0, 220.0, [face], aoi_settings=settings(), edge_eligible_source_row_indices=set())
    checks["exact_boundary_is_inclusive"] = exact is not None and exact.gate_hit_type == "exact_green_box"
    checks["bounded_edge_band_recovers_hit"] = (
        edge is not None
        and edge.region == "other_face"
        and edge.gate_hit_type == "green_box_edge_tolerance"
        and 0 < edge.edge_distance_px <= edge.edge_tolerance_px
    )
    checks["edge_band_is_coarse_face_only"] = edge is not None and edge.region == "other_face"
    checks["far_point_is_not_promoted"] = far is None
    checks["exact_policy_reproduces_old_gate"] = strict is None
    checks["partner_policy_restricts_edge_candidates"] = ineligible is None

    request = FaceAOIJobRequest(session_id="session", native_trim_artifact_id="trim")
    checks["api_default_is_edge_inclusive"] = request.face_edge_tolerance_policy == "edge_inclusive"
    checks["version_and_engine"] = (
        oculostudio.__version__ == "0.21.71"
        and FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
    )
    package_root = Path(oculostudio.__file__).resolve().parent
    app_js = (package_root / "static" / "app.js").read_text("utf-8")
    review_js = (package_root / "static" / "aoi_review.js").read_text("utf-8")
    index = (package_root / "static" / "index.html").read_text("utf-8")
    checks["ui_exposes_edge_policy"] = all(token in app_js for token in (
        'id="face-aoi-edge-tolerance"',
        'value="edge_inclusive" selected',
        'face_edge_tolerance_policy',
        'Eye, nose and mouth AOIs are not enlarged',
    ))
    checks["review_ui_replaces_ambiguous_dash_labels"] = all(token in review_js for token in (
        "function displayFaceEntity",
        "No face",
        "Face not detected",
        "Unresolved",
    ))
    checks["cache_busting_version"] = "app.js?v=0.21.71" in index and "styles.css?v=0.21.71" in index
    build_end_to_end(checks)

    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, value in checks.items() if not value]
    report = {
        "release": "OculoStudio v0.21.71",
        "feature": "Edge-inclusive green FaceMapper gate with explicit R02 entity labels",
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
        "all_passed": not failed,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
