"""End-to-end validation for v0.21.71 strict green-face gate and environment fallback."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind
from oculostudio.services.face_aoi import FACE_AOI_ENGINE, create_face_aoi_annotations
from oculostudio.services.target_fusion import _face_body_decision, _fuse_row, fuse_face_and_body_aoi
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def main() -> int:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="oculostudio_strict_green_gate_") as temp:
        project_root = Path(temp) / "project"
        store = ProjectStore.create(project_root, "Strict green gate validation", session_label="S001")
        session_id = store.project.selected_session_id
        artifacts = ArtifactStore(store)

        trim_root = store.derived_dir(session_id, "native_reference_trim", "strict_gate_trim")
        (trim_root / "tsd").mkdir(parents=True, exist_ok=True)
        (trim_root / "face_mapper").mkdir(parents=True, exist_ok=True)
        timestamps = [1_000_000_000, 1_033_333_333, 1_066_666_666]
        pd.DataFrame({
            "native_frame_index": [0, 1, 2],
            "source_frame_ordinal_in_window": [0, 1, 2],
            "clip_video_pts_ns": [0, 33_333_333, 66_666_666],
            "native_capture_utc_ns": timestamps,
        }).to_csv(trim_root / "native_frame_map.csv", index=False)
        pd.DataFrame({
            "native_frame_index": [0, 1, 2],
            "native_capture_utc_ns": timestamps,
        }).to_csv(trim_root / "native_clock_full.csv", index=False)
        (trim_root / "trim_manifest.json").write_text(json.dumps({
            "recording_id": "rec-1",
            "native_reference": {"width": 640, "height": 480},
        }), encoding="utf-8")

        # Frame 0: gaze inside green face box.
        # Frame 1: gaze outside green face box, but FaceMapper gaze_on_face=True.
        # Frame 2: no accepted green face detection at all.
        gaze = pd.DataFrame([
            {"timestamp [ns]": timestamps[0], "gaze x [px]": 200.0, "gaze y [px]": 200.0,
             "oculo_nearest_native_frame_index": 0, "oculo_source_row_index": 0, "worn": True},
            {"timestamp [ns]": timestamps[1], "gaze x [px]": 420.0, "gaze y [px]": 200.0,
             "oculo_nearest_native_frame_index": 1, "oculo_source_row_index": 1, "worn": True},
            {"timestamp [ns]": timestamps[2], "gaze x [px]": 420.0, "gaze y [px]": 200.0,
             "oculo_nearest_native_frame_index": 2, "oculo_source_row_index": 2, "worn": True},
        ])
        gaze.to_csv(trim_root / "tsd" / "gaze.csv", index=False)

        detections = pd.DataFrame([
            {"timestamp [ns]": timestamps[0], "recording id": "rec-1",
             "p1 x [px]": 100.0, "p1 y [px]": 100.0, "p2 x [px]": 300.0, "p2 y [px]": 360.0,
             "left eye x [px]": 230.0, "left eye y [px]": 165.0,
             "right eye x [px]": 170.0, "right eye y [px]": 165.0,
             "nose x [px]": 200.0, "nose y [px]": 225.0,
             "left mouth corner x [px]": 175.0, "left mouth corner y [px]": 285.0,
             "right mouth corner x [px]": 225.0, "right mouth corner y [px]": 285.0,
             "confidence": 0.95},
            {"timestamp [ns]": timestamps[1], "recording id": "rec-1",
             "p1 x [px]": 100.0, "p1 y [px]": 100.0, "p2 x [px]": 300.0, "p2 y [px]": 360.0,
             "left eye x [px]": 230.0, "left eye y [px]": 165.0,
             "right eye x [px]": 170.0, "right eye y [px]": 165.0,
             "nose x [px]": 200.0, "nose y [px]": 225.0,
             "left mouth corner x [px]": 175.0, "left mouth corner y [px]": 285.0,
             "right mouth corner x [px]": 225.0, "right mouth corner y [px]": 285.0,
             "confidence": 0.95},
        ])
        detections.to_csv(trim_root / "face_mapper" / "face_detections.csv", index=False)
        pd.DataFrame([
            {"timestamp [ns]": ts, "gaze on face": True} for ts in timestamps
        ]).to_csv(trim_root / "face_mapper" / "gaze_on_face.csv", index=False)

        trim = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
            label="S001_I01_R01_trim",
            output_root=trim_root,
        )
        face_result = create_face_aoi_annotations(
            store,
            session_id=session_id,
            native_trim_artifact_id=trim.id,
            target_person_policy="all_detected",
            face_left_right_policy="as_exported",
            auto_review_policy="adaptive",
            render_overlay=False,
            output_parquet=False,
            label="strict_green_face_aoi",
        )
        face_artifact_id = face_result["artifact"]["id"]
        face_samples = pd.read_csv(Path(face_result["samples_csv"]), low_memory=False)
        by_source = face_samples.set_index("source_gaze_row_index")

        checks["inside_green_box_is_face"] = str(by_source.loc[0, "face_region"]) in {
            "left_eye", "right_eye", "nose", "mouth", "other_face"
        }
        checks["outside_green_box_is_no_face_even_when_fm_true"] = (
            by_source.loc[1, "face_region"] == "no_face"
            and by_source.loc[1, "assignment_status"] == "outside_all_accepted_facemapper_face_boxes"
            and bool(by_source.loc[1, "fm_gaze_on_face"])
            and not bool(by_source.loc[1, "review_required"])
            and not bool(by_source.loc[1, "include_in_primary_analysis"])
            and pd.isna(by_source.loc[1, "face_entity_id"])
        )
        checks["no_detection_remains_distinct"] = (
            by_source.loc[2, "face_region"] == "no_face_detection"
            and by_source.loc[2, "assignment_status"] == "no_face_detection"
        )
        checks["projection_fallback_is_disabled"] = face_samples["projection_fallback_entity_id"].isna().all()
        checks["strict_gate_audit_columns"] = all(
            column in face_samples.columns for column in (
                "green_face_geometry_hit", "face_membership_policy",
                "fm_flag_vs_green_geometry_disagreement",
            )
        ) and bool(by_source.loc[1, "fm_flag_vs_green_geometry_disagreement"])

        body_root = store.derived_dir(session_id, "body_aoi", "strict_gate_body")
        body_samples = pd.DataFrame([
            {
                "gaze_sample_uid": row.gaze_sample_uid,
                "source_gaze_row_index": int(row.source_gaze_row_index),
                "timestamp_ns": int(row.timestamp_ns),
                "scene_frame_index": int(row.native_frame_index),
                "body_entity_id": None,
                "body_relation": None,
                "body_raw_region": None,
                "body_coarse_region": None,
                "body_region": "no_body",
                "body_social_region": "no_body",
                "body_confidence": 0.0,
                "body_exact_hit": False,
                "candidate_hit_count": 0,
                "self_occlusion": False,
                "occlusion_status": "no_body_hit",
                "assignment_status": "valid_ray_no_body_hit",
                "review_required": False,
                "ray_valid": True,
            }
            for row in face_samples.itertuples(index=False)
        ])
        body_samples.to_csv(body_root / "body_aoi_samples.csv", index=False)
        pd.DataFrame().to_csv(body_root / "body_aoi_frames.csv", index=False)
        pd.DataFrame().to_csv(body_root / "body_regions.csv", index=False)
        body = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.BODY_AOI,
            label="strict_gate_body_aoi",
            output_root=body_root,
            parent_artifact_ids=[trim.id],
            qc={
                "body_geometry_clock": "gaze_sample_timestamp",
                "include_self_body": False,
                "body_diagnostics": [{
                    "entity_id": "partner",
                    "entity_relation": "partner",
                    "scene_frame_count": 3,
                    "frames_with_any_primitive": 3,
                    "torso_primitive_frame_count": 3,
                    "pelvis_primitive_frame_count": 3,
                    "torso_direct_frame_count": 3,
                    "torso_derived_frame_count": 0,
                }],
            },
        )

        fusion = fuse_face_and_body_aoi(
            store,
            session_id=session_id,
            face_aoi_artifact_id=face_artifact_id,
            body_aoi_artifact_id=body.id,
            label="strict_gate_fusion",
        )
        fused = pd.read_csv(fusion["samples_csv"], low_memory=False).set_index("source_gaze_row_index")
        checks["no_face_plus_no_body_becomes_environment"] = (
            fused.loc[1, "final_target_domain"] == "environment"
            and fused.loc[1, "final_target_region"] == "environment"
            and fused.loc[1, "fusion_status"] == "strict_no_face_and_no_body_hit_environment"
            and not bool(fused.loc[1, "review_required"])
        )
        checks["no_face_detection_plus_no_body_stays_unknown"] = (
            fused.loc[2, "final_target_domain"] == "unknown"
            and fused.loc[2, "final_target_region"] == "unknown"
        )
        checks["face_hit_not_relabelled_environment"] = fused.loc[0, "final_target_domain"] == "face"

        alias_base = {
            "face_region": "no_face",
            "face_social_region": "no_face",
            "face_aoi_confidence": 1.0,
            "face_review_required": False,
            "body_confidence": 0.0,
            "body_review_required": False,
        }
        checks["no_body_part_alias_becomes_environment"] = (
            _face_body_decision(pd.Series({**alias_base, "body_region": "no_body_part"}))["domain"]
            == "environment"
        )
        checks["no_body_hit_alias_becomes_environment"] = (
            _face_body_decision(pd.Series({**alias_base, "body_region": "no_body_hit"}))["domain"]
            == "environment"
        )
        object_override = _fuse_row(pd.Series({
            **alias_base,
            "body_region": "no_body",
            "object_entity_id": "cup_1",
            "object_region": "object",
            "object_category": "cup",
            "object_confidence": 0.95,
            "object_distance_mm": 900.0,
            "object_exact_hit": True,
            "object_review_required": False,
        }))
        checks["valid_object_hit_overrides_environment"] = (
            object_override["final_target_domain"] == "object"
            and object_override["final_target_region"] == "cup"
        )

        manifest = json.loads((Path(face_result["samples_csv"]).parent / "face_aoi_manifest.json").read_text("utf-8"))
        checks["manifest_records_strict_policy"] = (
            manifest["schema_version"] == "oculostudio-face-aoi-1.13"
            and manifest["parameters"]["face_membership_policy"] == "accepted_green_box_with_bounded_edge_tolerance"
            and manifest["parameters"]["participant_projection_usage"] == "identity_gate_only"
            and manifest["parameters"]["temporal_face_promotion_outside_green_box"] is False
        )

    checks["version_and_engine"] = (
        oculostudio.__version__ == "0.21.71"
        and FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
    )
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "release": "OculoStudio v0.21.71",
        "feature": "Strict accepted-green-box face membership, balanced facial AOIs and environment fallback",
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
