"""Targeted Body AOI manual-review regression validation for OculoStudio v0.21.71."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind
from oculostudio.services.aoi_review import (
    apply_body_aoi_manual_review,
    review_frame,
    review_frame_window,
    review_manifest,
    review_options,
    save_body_review_decision,
)
from oculostudio.services.body_aoi import _frame_summary
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def main() -> int:
    report: dict[str, object] = {
        "application_version": oculostudio.__version__,
        "feature": "Body AOI human review",
        "checks": {},
    }
    checks: dict[str, bool] = report["checks"]  # type: ignore[assignment]

    with tempfile.TemporaryDirectory(prefix="oculostudio_body_review_") as temp:
        root = Path(temp) / "project"
        store = ProjectStore.create(root, "Body AOI review validation", session_label="Session S001")
        session_id = store.project.selected_session_id
        artifacts = ArtifactStore(store)

        trim_root = store.derived_dir(session_id, "native_reference_trim", "trim_test")
        frame_map = pd.DataFrame([
            {"native_frame_index": 10, "source_frame_ordinal_in_window": 0, "clip_video_pts_ns": 0, "native_capture_utc_ns": 1_000_000_000},
            {"native_frame_index": 11, "source_frame_ordinal_in_window": 1, "clip_video_pts_ns": 33_333_333, "native_capture_utc_ns": 1_033_333_333},
            {"native_frame_index": 12, "source_frame_ordinal_in_window": 2, "clip_video_pts_ns": 66_666_666, "native_capture_utc_ns": 1_066_666_666},
        ])
        frame_map.to_csv(trim_root / "native_frame_map.csv", index=False)
        (trim_root / "trim_manifest.json").write_text(
            json.dumps({"native_reference": {"width": 1280, "height": 960}}), encoding="utf-8"
        )
        trim = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
            label="S001_I01_R01 trim",
            output_root=trim_root,
        )

        video_root = store.derived_dir(session_id, "privacy_clip", "privacy_test")
        media_probe = Path(oculostudio.__file__).resolve().parent / "resources" / "media_probe_test.mp4"
        shutil.copy2(media_probe, video_root / "S001_I01_R01_privacy_trimmed.mp4")
        pd.DataFrame([
            {"output_frame_index": 0, "output_pts_ns": 0, "native_frame_index": 10, "native_capture_utc_ns": 1_000_000_000},
            {"output_frame_index": 1, "output_pts_ns": 33_333_333, "native_frame_index": 11, "native_capture_utc_ns": 1_033_333_333},
            {"output_frame_index": 2, "output_pts_ns": 66_666_666, "native_frame_index": 12, "native_capture_utc_ns": 1_066_666_666},
        ]).to_csv(video_root / "privacy_trimmed_frame_map.csv", index=False)
        (video_root / "privacy_clip_report.json").write_text(
            json.dumps({"output": {"fps": 30, "width": 640, "height": 480}}), encoding="utf-8"
        )
        video = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.PRIVACY_CLIP,
            label="S001_I01_R01 privacy",
            output_root=video_root,
            provenance={"native_trim_artifact_id": trim.id},
            qc={"output_fps": 30, "output_frame_count": 3},
        )

        samples = pd.DataFrame([
            {
                "gaze_sample_uid": "g0", "source_gaze_row_index": 0,
                "timestamp_ns": 1_000_000_000, "clip_time_ns": 0,
                "native_frame_index": 10, "scene_frame_index": 10,
                "scene_timestamp_ns": 1_000_000_000,
                "gaze_x_px": 250.0, "gaze_y_px": 190.0,
                "fixation_id": "fix-shared", "ray_valid": True,
                "body_entity_id": "body_partner", "body_relation": "partner",
                "body_raw_region": "left_forearm", "body_coarse_region": "left_arm",
                "body_region": "left_arm", "body_social_region": "partner_left_arm",
                "body_confidence": 0.47, "body_exact_hit": False, "body_distance_mm": 1280.0,
                "body_hit_x_mm": 80.0, "body_hit_y_mm": 1450.0, "body_hit_z_mm": 650.0,
                "body_angular_miss_deg": 1.9, "body_geometry_backend": "capsule_uncertainty",
                "assignment_status": "uncertain_cone_hit", "occlusion_status": "partner_candidate",
                "self_occlusion": False, "review_required": True,
                "first_hit_entity_id": "body_partner", "first_hit_relation": "partner",
                "second_hit_entity_id": None, "second_hit_relation": None,
                "nearest_partner_hit_entity_id": "body_partner",
            },
            {
                "gaze_sample_uid": "g1", "source_gaze_row_index": 1,
                "timestamp_ns": 1_033_333_333, "clip_time_ns": 33_333_333,
                "native_frame_index": 11, "scene_frame_index": 11,
                "scene_timestamp_ns": 1_033_333_333,
                "gaze_x_px": 310.0, "gaze_y_px": 230.0,
                "fixation_id": "fix-shared", "ray_valid": True,
                "body_entity_id": "body_self", "body_relation": "self",
                "body_raw_region": "torso", "body_coarse_region": "torso",
                "body_region": "torso", "body_social_region": "self_torso",
                "body_confidence": 0.52, "body_exact_hit": True, "body_distance_mm": 510.0,
                "body_hit_x_mm": 0.0, "body_hit_y_mm": 1120.0, "body_hit_z_mm": 260.0,
                "body_angular_miss_deg": 0.0, "body_geometry_backend": "capsule_exact",
                "assignment_status": "self_first_hit", "occlusion_status": "self_occlusion",
                "self_occlusion": True, "review_required": True,
                "first_hit_entity_id": "body_self", "first_hit_relation": "self",
                "second_hit_entity_id": "body_partner", "second_hit_relation": "partner",
                "nearest_partner_hit_entity_id": "body_partner",
            },
            {
                "gaze_sample_uid": "g2", "source_gaze_row_index": 2,
                "timestamp_ns": 1_066_666_666, "clip_time_ns": 66_666_666,
                "native_frame_index": 12, "scene_frame_index": 12,
                "scene_timestamp_ns": 1_066_666_666,
                "gaze_x_px": 360.0, "gaze_y_px": 140.0,
                "fixation_id": "fix-other", "ray_valid": True,
                "body_entity_id": "body_partner", "body_relation": "partner",
                "body_raw_region": "head", "body_coarse_region": "head",
                "body_region": "head", "body_social_region": "partner_head",
                "body_confidence": 0.96, "body_exact_hit": True, "body_distance_mm": 1180.0,
                "body_hit_x_mm": 25.0, "body_hit_y_mm": 1710.0, "body_hit_z_mm": 620.0,
                "body_angular_miss_deg": 0.0, "body_geometry_backend": "sphere_exact",
                "assignment_status": "accepted", "occlusion_status": "clear",
                "self_occlusion": False, "review_required": False,
                "first_hit_entity_id": "body_partner", "first_hit_relation": "partner",
                "second_hit_entity_id": None, "second_hit_relation": None,
                "nearest_partner_hit_entity_id": "body_partner",
            },
        ])
        body_root = store.derived_dir(session_id, "body_aoi", "body_test")
        samples.to_csv(body_root / "body_aoi_samples.csv", index=False)
        _frame_summary(samples).to_csv(body_root / "body_aoi_frames.csv", index=False)
        pd.DataFrame([
            {"scene_frame_index": 10, "entity_id": "body_partner", "relation_to_wearer": "partner", "region": "torso", "primitive_type": "capsule", "x1_mm": -180, "y1_mm": 900, "z1_mm": 600, "x2_mm": -160, "y2_mm": 1500, "z2_mm": 620, "radius_mm": 210},
            {"scene_frame_index": 10, "entity_id": "body_partner", "relation_to_wearer": "partner", "region": "left_forearm", "primitive_type": "capsule", "x1_mm": -240, "y1_mm": 1300, "z1_mm": 620, "x2_mm": -420, "y2_mm": 1040, "z2_mm": 620, "radius_mm": 55},
            {"scene_frame_index": 11, "entity_id": "body_self", "relation_to_wearer": "self", "region": "torso", "primitive_type": "capsule", "x1_mm": 0, "y1_mm": 850, "z1_mm": 200, "x2_mm": 0, "y2_mm": 1450, "z2_mm": 240, "radius_mm": 200},
            {"scene_frame_index": 11, "entity_id": "body_partner", "relation_to_wearer": "partner", "region": "torso", "primitive_type": "capsule", "x1_mm": -160, "y1_mm": 900, "z1_mm": 620, "x2_mm": -150, "y2_mm": 1500, "z2_mm": 640, "radius_mm": 210},
            {"scene_frame_index": 12, "entity_id": "body_partner", "relation_to_wearer": "partner", "region": "head", "primitive_type": "sphere", "center_x_mm": -150, "center_y_mm": 1740, "center_z_mm": 630, "radius_mm": 125},
        ]).to_csv(body_root / "body_regions.csv", index=False)
        (body_root / "body_aoi_manifest.json").write_text(json.dumps({"schema_version": "test"}), encoding="utf-8")
        body = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.BODY_AOI,
            label="S001_I01_R01 Body AOI",
            output_root=body_root,
            parent_artifact_ids=[trim.id, video.id],
            parameters={
                "wearer_role_id": "R01", "partner_role_id": "R02",
                "wearer_body_asset_id": "body_self", "partner_body_asset_id": "body_partner",
                "output_granularity": "coarse_social_regions",
            },
            provenance={
                "native_trim_artifact_id": trim.id,
                "privacy_clip_artifact_id": video.id,
                "wearer_role_id": "R01", "partner_role_id": "R02",
            },
            qc={
                "body_geometry_clock": "gaze_sample_timestamp",
                "review_required_fraction": 2 / 3,
                "review_required_count": 2,
                "output_rows": 3,
                "body_hit_fraction": 1.0,
                "wearer_body_asset_id": "body_self",
                "partner_body_asset_id": "body_partner",
                "include_self_body": True,
                "body_diagnostics": [
                    {"entity_id": "body_partner", "entity_relation": "partner", "source_asset_id": "body_partner", "scene_frame_count": 3, "frames_with_any_primitive": 3, "torso_primitive_frame_count": 3, "pelvis_primitive_frame_count": 3, "torso_direct_frame_count": 3, "torso_derived_frame_count": 0},
                    {"entity_id": "body_self", "entity_relation": "self", "source_asset_id": "body_self", "scene_frame_count": 3, "frames_with_any_primitive": 3, "torso_primitive_frame_count": 3, "pelvis_primitive_frame_count": 3, "torso_direct_frame_count": 3, "torso_derived_frame_count": 0},
                ],
            },
        )

        options = review_options(store, session_id=session_id)
        body_option = next(row for row in options["modes"]["body"] if row["artifact_id"] == body.id)
        checks["options_expose_body_manual_review"] = (
            body_option["manual_review_draft"]["decision_count"] == 0
            and body_option["qc"]["review_required_count"] == 2
            and body_option["default_video_artifact_id"] == video.id
        )

        manifest = review_manifest(
            store, session_id=session_id, mode="body", artifact_id=body.id, video_artifact_id=video.id
        )
        checks["pending_body_queue"] = (
            manifest["manual_review"]["pending_sample_count"] == 2
            and manifest["manual_review"]["pending_frame_count"] == 2
            and manifest["manual_review"]["pending_review_unit_count"] == 1
            and manifest["manual_review"]["review_unit_policy"] == "fixation_when_available_else_sample"
            and manifest["event_frames"]["pending_review"] == [0]
            and manifest["event_frames"]["pending_review_frames"] == [0, 1]
        )
        checks["body_role_context"] = (
            manifest["manual_review_role_context"]["partner_entity_id"] == "body_partner"
            and manifest["manual_review_role_context"]["self_entity_id"] == "body_self"
        )
        frame0 = review_frame(
            store, session_id=session_id, mode="body", artifact_id=body.id,
            video_artifact_id=video.id, output_frame_index=0,
        )
        checks["frame_selects_flagged_body_sample"] = frame0["selected_sample"]["gaze_sample_uid"] == "g0"
        window = review_frame_window(
            store, session_id=session_id, mode="body", artifact_id=body.id,
            video_artifact_id=video.id, start_frame_index=0, frame_count=3,
        )
        checks["frame_window_has_body_review_state"] = (
            window["frame_count"] == 3
            and all("manual_review" in row for row in window["frames"])
        )

        deferred = save_body_review_decision(
            store, session_id=session_id, artifact_id=body.id, video_artifact_id=video.id,
            output_frame_index=0, gaze_sample_uid="g0", action="defer", reviewer="validator",
        )
        checks["defer_remains_pending"] = (
            deferred["manual_review"]["pending_sample_count"] == 2
            and deferred["manual_review"]["deferred_sample_count"] == 1
        )

        corrected = save_body_review_decision(
            store, session_id=session_id, artifact_id=body.id, video_artifact_id=video.id,
            output_frame_index=0, gaze_sample_uid="g0", action="override",
            override_relation="partner", override_region="right_hand",
            note="confirmed partner right hand", reviewer="validator",
        )
        checks["override_resolves_partner_entity"] = corrected["resolved_entity_ids"]["g0"] == "body_partner"
        corrected_preview = review_frame(
            store, session_id=session_id, mode="body", artifact_id=body.id,
            video_artifact_id=video.id, output_frame_index=0,
        )
        selected = corrected_preview["selected_sample"]
        checks["body_override_live_preview"] = (
            selected["effective_body_entity_id"] == "body_partner"
            and selected["effective_body_relation"] == "partner"
            and selected["effective_body_region"] == "right_hand"
            and selected["effective_body_social_region"] == "partner_right_hand"
            and selected["effective_body_confidence"] == 1.0
            and selected["effective_manual_review_resolved"] is True
            and selected["effective_body_exact_hit"] is False
        )
        checks["body_frame_summary_live_preview"] = (
            corrected_preview["summary_uses_manual_review"] is True
            and corrected_preview["summary"]["primary_region"] == "partner_right_hand"
            and corrected_preview["automatic_summary"]["primary_region"] == "partner_left_arm"
        )

        excluded = save_body_review_decision(
            store, session_id=session_id, artifact_id=body.id, video_artifact_id=video.id,
            output_frame_index=1, gaze_sample_uid="g1", action="exclude",
            note="not a body target", reviewer="validator",
        )
        excluded_preview = review_frame(
            store, session_id=session_id, mode="body", artifact_id=body.id,
            video_artifact_id=video.id, output_frame_index=1,
        )
        checks["no_body_live_preview"] = (
            excluded_preview["selected_sample"]["effective_body_region"] == "no_body"
            and excluded_preview["selected_sample"]["effective_body_social_region"] == "no_body"
            and excluded_preview["selected_sample"]["effective_body_entity_id"] is None
        )
        checks["review_completion_gate"] = (
            excluded["manual_review"]["status"] == "complete"
            and excluded["manual_review"]["pending_sample_count"] == 0
        )

        source_before = pd.read_csv(body_root / "body_aoi_samples.csv")
        result = apply_body_aoi_manual_review(
            store, session_id=session_id, artifact_id=body.id,
            label="S001_I01_R01 Body AOI reviewed", require_complete=True,
        )
        reviewed_id = result["artifact"]["id"]
        reviewed_artifact = store.project.sessions[session_id].artifacts[reviewed_id]
        reviewed_csv = next(Path(row.path) for row in reviewed_artifact.files if Path(row.path).name == "body_aoi_samples.csv")
        reviewed = pd.read_csv(reviewed_csv)
        source_after = pd.read_csv(body_root / "body_aoi_samples.csv")
        g0 = reviewed.loc[reviewed["gaze_sample_uid"] == "g0"].iloc[0]
        g1 = reviewed.loc[reviewed["gaze_sample_uid"] == "g1"].iloc[0]

        checks["source_body_aoi_immutable"] = source_before.equals(source_after)
        checks["reviewed_body_lineage"] = reviewed_artifact.parent_artifact_ids == [body.id]
        checks["override_written_to_reviewed_artifact"] = (
            g0["body_entity_id"] == "body_partner"
            and g0["body_relation"] == "partner"
            and g0["body_region"] == "right_hand"
            and g0["body_social_region"] == "partner_right_hand"
            and g0["manual_review_action"] == "override"
            and not bool(g0["review_required"])
            and pd.isna(g0["body_hit_x_mm"])
            and g0["automatic_body_region"] == "left_arm"
        )
        checks["exclude_written_to_reviewed_artifact"] = (
            g1["body_region"] == "no_body"
            and g1["body_social_region"] == "no_body"
            and pd.isna(g1["body_entity_id"])
            and g1["manual_review_action"] == "exclude"
            and not bool(g1["review_required"])
            and g1["automatic_body_region"] == "torso"
        )
        checks["reviewed_body_outputs"] = all(Path(result[key]).is_file() for key in ["frames_csv", "regions_csv", "decisions_csv", "manifest"])
        checks["reviewed_body_qc"] = (
            reviewed_artifact.qc.get("manual_review_applied") is True
            and int(reviewed_artifact.qc.get("manual_review_pending_sample_count", -1)) == 0
            and abs(float(reviewed_artifact.qc.get("review_required_fraction", 1.0))) < 1e-9
        )

        # Verify fixation scope applies a single decision to all flagged samples.
        batch = save_body_review_decision(
            store, session_id=session_id, artifact_id=body.id, video_artifact_id=video.id,
            output_frame_index=0, gaze_sample_uid="g0", apply_to="fixation",
            action="accept_auto", reviewer="validator",
        )
        checks["fixation_scope_updates_flagged_body_samples"] = (
            batch["changed_sample_count"] == 2
            and set(batch["changed_sample_uids"]) == {"g0", "g1"}
            and batch["manual_review"]["pending_review_unit_count"] == 0
        )

        js = (Path(oculostudio.__file__).resolve().parent / "static" / "aoi_review.js").read_text("utf-8")
        app_js = (Path(oculostudio.__file__).resolve().parent / "static" / "app.js").read_text("utf-8")
        checks["body_review_entry_ui"] = "Open Body review" in js and "Human review · Body AOI" in js
        checks["body_review_editor_ui"] = all(text in js for text in ["Mark no body", "Create reviewed Body AOI", "body-manual-review"])
        checks["body_review_property_and_job_entries"] = (
            "open-body-human-review-side" in app_js
            and "Open Body Human Review" in app_js
            and "Review flagged Body AOI" in app_js
        )

        report["source_artifact_id"] = body.id
        report["reviewed_artifact_id"] = reviewed_id
        report["all_passed"] = all(checks.values())

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
