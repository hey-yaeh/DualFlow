"""Targeted v0.21.71 Face AOI role-resolved manual review regression validation.

Creates a disposable project, saves resumable review decisions, applies them to
an immutable reviewed Face AOI artifact, and verifies lineage and output tables.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind
from oculostudio.services.aoi_review import (
    apply_face_aoi_manual_review,
    review_frame,
    review_frame_window,
    review_manifest,
    review_options,
    save_face_review_decision,
)
from oculostudio.services.face_aoi import _frame_summary
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def main() -> int:
    report: dict[str, object] = {
        "application_version": oculostudio.__version__,
        "feature": "Face AOI role-resolved manual review",
        "checks": {},
    }
    checks: dict[str, bool] = report["checks"]  # type: ignore[assignment]

    with tempfile.TemporaryDirectory(prefix="oculostudio_face_review_") as temp:
        root = Path(temp) / "project"
        store = ProjectStore.create(root, "Face AOI review validation", session_label="Session S001")
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

        participant_root = store.derived_dir(session_id, "participant_association", "participant_test")
        pd.DataFrame([
            {
                "native_frame_index": native, "role_id": "R02", "entity_id": "R02",
                "relation_to_wearer": "partner", "projection_valid": True,
                "projected_bbox_x1": 700, "projected_bbox_y1": 120,
                "projected_bbox_x2": 980, "projected_bbox_y2": 430,
                "projection_confidence": 0.95,
            }
            for native in (10, 11, 12)
        ]).to_csv(participant_root / "participant_head_projection.csv", index=False)
        pd.DataFrame(columns=[
            "native_frame_index", "matched_entity_id", "matched_role",
            "candidate_entity_id", "candidate_role", "candidate_relation",
            "target_relation", "identity_status", "identity_confidence",
        ]).to_csv(participant_root / "face_identity_assignment.csv", index=False)
        (participant_root / "participant_association_manifest.json").write_text(
            json.dumps({"schema_version": "test", "wearer_role_id": "R01"}), encoding="utf-8"
        )
        participant = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.PARTICIPANT_ASSOCIATION,
            label="S001_I01_R01 participant association",
            output_root=participant_root,
            parent_artifact_ids=[trim.id],
            parameters={"wearer_role_id": "R01", "target_policy": "partner_only"},
            provenance={"native_trim_artifact_id": trim.id},
        )

        samples = pd.DataFrame([
            {
                "gaze_sample_uid": "g0", "source_gaze_row_index": 0,
                "timestamp_ns": 1_000_000_000, "clip_time_ns": 0,
                "native_frame_index": 10, "scene_frame_index": 10,
                "gaze_x_px": 200, "gaze_y_px": 180, "worn": True, "fixation_id": "fix-shared",
                "face_entity_id": None, "candidate_entity_id": None,
                "candidate_role": None, "candidate_relation": None,
                "face_region": "face", "face_social_region": "other_person_face",
                "face_aoi_confidence": 0.42, "assignment_status": "on_face_without_fine_aoi",
                "identity_status": "unmatched_person", "identity_confidence": 0.0,
                "target_relation": "other_person", "include_in_primary_analysis": False,
                "review_required": True,
            },
            {
                "gaze_sample_uid": "g1", "source_gaze_row_index": 1,
                "timestamp_ns": 1_033_333_333, "clip_time_ns": 33_333_333,
                "native_frame_index": 11, "scene_frame_index": 11,
                "gaze_x_px": 300, "gaze_y_px": 210, "worn": True, "fixation_id": "fix-shared",
                "face_entity_id": "P02", "candidate_entity_id": "P02",
                "candidate_role": "R02", "candidate_relation": "partner",
                "face_region": "mouth", "face_social_region": "partner_face_candidate",
                "face_aoi_confidence": 0.51, "assignment_status": "ambiguous_candidate",
                "identity_status": "ambiguous_candidate", "identity_confidence": 0.51,
                "target_relation": "ambiguous_partner", "include_in_primary_analysis": False,
                "review_required": True,
            },
            {
                "gaze_sample_uid": "g2", "source_gaze_row_index": 2,
                "timestamp_ns": 1_066_666_666, "clip_time_ns": 66_666_666,
                "native_frame_index": 12, "scene_frame_index": 12,
                "gaze_x_px": 100, "gaze_y_px": 120, "worn": True, "fixation_id": "fix-other",
                "face_entity_id": "P02", "candidate_entity_id": "P02",
                "candidate_role": "R02", "candidate_relation": "partner",
                "face_region": "nose", "face_social_region": "partner_nose",
                "face_aoi_confidence": 0.94, "assignment_status": "matched_participant",
                "identity_status": "matched_participant", "identity_confidence": 0.94,
                "target_relation": "partner", "include_in_primary_analysis": True,
                "review_required": False,
            },
        ])
        face_root = store.derived_dir(session_id, "face_aoi", "face_test")
        samples.to_csv(face_root / "face_aoi_samples.csv", index=False)
        _frame_summary(samples, frame_map).to_csv(face_root / "face_aoi_frames.csv", index=False)
        pd.DataFrame([
            {"scene_frame_index": 10, "region": "left_eye", "geometry_json": '{"center_x":200,"center_y":180,"radius_x":20,"radius_y":10}'},
            {"scene_frame_index": 11, "region": "mouth", "geometry_json": '{"center_x":300,"center_y":210,"radius_x":25,"radius_y":12}'},
        ]).to_csv(face_root / "face_regions.csv", index=False)
        pd.DataFrame([
            {"scope": "total", "source": "review_samples_before", "count": 2, "fraction_of_pre_auto_review": 1.0},
            {"scope": "total", "source": "auto_resolved_samples", "count": 0, "fraction_of_pre_auto_review": 0.0},
            {"scope": "total", "source": "review_samples_remaining", "count": 2, "fraction_of_pre_auto_review": 1.0},
        ]).to_csv(face_root / "face_aoi_auto_review_summary.csv", index=False)
        (face_root / "face_aoi_auto_review_summary.json").write_text(json.dumps({
            "schema_version": "oculostudio-face-aoi-auto-review-1.0",
            "policy": "conservative",
            "review_samples_before": 2,
            "auto_resolved_samples": 0,
            "review_samples_remaining": 2,
        }), encoding="utf-8")
        (face_root / "face_aoi_manifest.json").write_text(json.dumps({"schema_version": "test"}), encoding="utf-8")
        face = artifacts.register(
            session_id=session_id,
            kind=ArtifactKind.FACE_AOI,
            label="S001_I01_R01 Face AOI",
            output_root=face_root,
            parent_artifact_ids=[trim.id, participant.id, video.id],
            provenance={
                "native_trim_artifact_id": trim.id,
                "participant_association_artifact_id": participant.id,
                "privacy_clip_artifact_id": video.id,
            },
            qc={"review_required_fraction": 2 / 3},
        )

        options = review_options(store, session_id=session_id)
        face_option = next(row for row in options["modes"]["face"] if row["artifact_id"] == face.id)
        video_option = next(row for row in options["videos"] if row["artifact_id"] == video.id)
        checks["role_context_detected"] = face_option["role_id"] == "R01" and video_option["role_id"] == "R01"
        checks["compatible_privacy_clip"] = video.id in face_option["compatible_video_artifact_ids"]

        review_js = (Path(oculostudio.__file__).resolve().parent / "static" / "aoi_review.js").read_text("utf-8")
        checks["empty_override_client_guard"] = "Choose a correction" in review_js and "Save correction requires at least one changed field" in review_js
        checks["pending_sample_auto_selection"] = "firstPendingSampleIndex" in review_js and "Select the flagged gaze sample" in review_js
        checks["live_reviewed_sample_preview_ui"] = "Effective reviewed target" in review_js and "automatic:" in review_js
        checks["reviewed_frame_summary_ui"] = "Reviewed frame primary target" in review_js and "Review draft applied" in review_js
        checks["source_immutability_guidance"] = "The source artifact remains unchanged until Create reviewed Face AOI" in review_js
        checks["frame_locked_stale_overlay_guard"] = (
            "SYNCING OVERLAY" in review_js
            and "payloadFrame===Number(expectedFrame)" in review_js
            and "overlaySynced" in review_js
        )
        checks["buffered_playback_window_ui"] = (
            "frame-window" in review_js
            and "playbackWindowSize" in review_js
            and "ensurePlaybackReady" in review_js
        )
        checks["exact_pts_mapping_ui"] = (
            "frame_pts_ns" in review_js
            and "mediaTimeForFrame" in review_js
            and "frameFromMediaTime" in review_js
        )
        checks["coordinate_scaling_ui"] = (
            "coordinate_space" in review_js
            and "scaleX=rect.w/Math.max(1,sourceWidth)" in review_js
            and "scaleY=rect.h/Math.max(1,sourceHeight)" in review_js
        )
        checks["pause_locked_exact_seek_ui"] = (
            "lockPausedPresentedFrame" in review_js
            and "LOCKING PAUSED FRAME" in review_js
            and "lastPresentedFrame" in review_js
            and "video.currentTime=exactTime" in review_js
        )
        checks["pause_navigation_race_guard_ui"] = (
            "pendingExplicitSeekFrame" in review_js
            and "seekReviewFrame" in review_js
            and "never snap back to the old frame" in review_js
        )
        checks["role_entity_suggestion_ui"] = (
            "suggestedEntityForRelation" in review_js
            and "Auto-resolved" in review_js
            and "entityResolutionSource" in review_js
        )

        manifest = review_manifest(
            store, session_id=session_id, mode="face", artifact_id=face.id, video_artifact_id=video.id
        )
        checks["pending_queue"] = (
            manifest["manual_review"]["pending_frame_count"] == 2
            and manifest["event_frames"]["pending_review"] == [0, 1]
        )
        checks["exact_video_pts_manifest"] = manifest["video"]["frame_pts_ns"] == [0, 33_333_333, 66_666_666]
        checks["coordinate_space_scaling_manifest"] = (
            manifest["coordinate_space"]["width"] == 1280
            and manifest["coordinate_space"]["height"] == 960
            and manifest["coordinate_space"]["video_width"] == 640
            and manifest["coordinate_space"]["video_height"] == 480
            and abs(manifest["coordinate_space"]["scale_x_to_video"] - 0.5) < 1e-9
            and abs(manifest["coordinate_space"]["scale_y_to_video"] - 0.5) < 1e-9
        )
        checks["unique_partner_role_context"] = (
            manifest["manual_review_role_context"]["wearer_role_id"] == "R01"
            and manifest["manual_review_role_context"]["partner_entity_id"] == "R02"
        )
        window = review_frame_window(
            store, session_id=session_id, mode="face", artifact_id=face.id,
            video_artifact_id=video.id, start_frame_index=0, frame_count=3, trail_frames=1,
        )
        checks["frame_window_backend"] = (
            window["frame_count"] == 3
            and [row["output_frame_index"] for row in window["frames"]] == [0, 1, 2]
            and all(row["manual_review"] is None for row in window["frames"])
        )
        frame = review_frame(
            store, session_id=session_id, mode="face", artifact_id=face.id,
            video_artifact_id=video.id, output_frame_index=0,
        )
        checks["frame_sample_selection"] = frame["selected_sample"]["gaze_sample_uid"] == "g0"

        deferred = save_face_review_decision(
            store, session_id=session_id, artifact_id=face.id, video_artifact_id=video.id,
            output_frame_index=0, gaze_sample_uid="g0", action="defer", reviewer="validator",
        )
        checks["defer_remains_pending"] = (
            deferred["manual_review"]["pending_sample_count"] == 2
            and deferred["manual_review"]["deferred_sample_count"] == 1
        )

        corrected = save_face_review_decision(
            store, session_id=session_id, artifact_id=face.id, video_artifact_id=video.id,
            output_frame_index=0, gaze_sample_uid="g0", action="override",
            override_relation="partner", override_entity_id=None, override_region="mouth",
            note="confirmed partner mouth", reviewer="validator",
        )
        checks["save_resolves_unique_partner_entity"] = (
            corrected["resolved_entity_ids"]["g0"] == "R02"
        )
        corrected_preview = review_frame(
            store, session_id=session_id, mode="face", artifact_id=face.id,
            video_artifact_id=video.id, output_frame_index=0,
        )
        corrected_sample = corrected_preview["selected_sample"]
        checks["corrected_sample_updates_immediately"] = (
            corrected_sample["effective_face_entity_id"] == "R02"
            and corrected_sample["effective_face_social_region"] == "partner_mouth"
            and corrected_sample["effective_target_relation"] == "partner"
            and corrected_sample["effective_face_aoi_confidence"] == 1.0
            and corrected_sample["effective_manual_review_resolved"] is True
            and corrected_sample["effective_entity_resolution_source"] == "explicit_entity_id"
        )
        checks["corrected_frame_summary_updates_immediately"] = (
            corrected_preview["summary_uses_manual_review"] is True
            and corrected_preview["summary"]["primary_entity_id"] == "R02"
            and corrected_preview["summary"]["primary_region"] == "partner_mouth"
            and corrected_preview["automatic_summary"]["primary_region"] == "other_person_face"
        )

        # Simulate a v0.21.29 draft where relation/region were saved but Entity ID
        # remained null. The new release must repair the preview and final artifact
        # without asking the reviewer to repeat the decision.
        manual_path = root / "annotations" / session_id / "face_aoi" / face.id / "manual_review.json"
        legacy_state = json.loads(manual_path.read_text("utf-8"))
        legacy_state["decisions"]["g0"]["override_entity_id"] = None
        legacy_state["decisions"]["g0"].pop("entity_resolution_source", None)
        manual_path.write_text(json.dumps(legacy_state, indent=2), encoding="utf-8")
        legacy_preview = review_frame(
            store, session_id=session_id, mode="face", artifact_id=face.id,
            video_artifact_id=video.id, output_frame_index=0,
        )
        legacy_sample = legacy_preview["selected_sample"]
        checks["legacy_blank_entity_draft_recovers_r02"] = (
            legacy_sample["effective_face_entity_id"] == "R02"
            and legacy_sample["effective_face_social_region"] == "partner_mouth"
            and legacy_sample["effective_entity_resolution_source"] == "current_frame_projection"
        )
        accepted = save_face_review_decision(
            store, session_id=session_id, artifact_id=face.id, video_artifact_id=video.id,
            output_frame_index=1, gaze_sample_uid="g1", action="accept_auto",
            note="automatic mouth target accepted", reviewer="validator",
        )
        accepted_preview = review_frame(
            store, session_id=session_id, mode="face", artifact_id=face.id,
            video_artifact_id=video.id, output_frame_index=1,
        )
        checks["accepted_auto_preview_matches_final_semantics"] = (
            accepted_preview["selected_sample"]["effective_face_social_region"] == "partner_mouth"
            and accepted_preview["selected_sample"]["effective_target_relation"] == "partner"
            and accepted_preview["summary"]["primary_region"] == "partner_mouth"
        )
        checks["resumable_decisions"] = corrected["manual_review"]["pending_sample_count"] == 1
        checks["review_completion_gate"] = (
            accepted["manual_review"]["status"] == "complete"
            and accepted["manual_review"]["pending_sample_count"] == 0
        )

        source_before = pd.read_csv(face_root / "face_aoi_samples.csv")
        result = apply_face_aoi_manual_review(
            store, session_id=session_id, artifact_id=face.id,
            label="S001_I01_R01 Face AOI reviewed", require_complete=True,
        )
        reviewed_id = result["artifact"]["id"]
        reviewed_artifact = store.project.sessions[session_id].artifacts[reviewed_id]
        reviewed_csv = next(Path(row.path) for row in reviewed_artifact.files if Path(row.path).name == "face_aoi_samples.csv")
        reviewed = pd.read_csv(reviewed_csv)
        source_after = pd.read_csv(face_root / "face_aoi_samples.csv")
        g0 = reviewed.loc[reviewed["gaze_sample_uid"] == "g0"].iloc[0]
        g1 = reviewed.loc[reviewed["gaze_sample_uid"] == "g1"].iloc[0]

        checks["source_immutable"] = source_before.equals(source_after)
        checks["new_artifact_lineage"] = reviewed_artifact.parent_artifact_ids == [face.id]
        checks["override_applied"] = (
            g0["target_relation"] == "partner"
            and g0["face_entity_id"] == "R02"
            and g0["face_social_region"] == "partner_mouth"
            and g0["manual_review_action"] == "override"
            and g0["manual_review_entity_resolution_source"] == "current_frame_projection"
            and not bool(g0["review_required"])
        )
        checks["accept_auto_promoted"] = (
            g1["target_relation"] == "partner"
            and g1["face_social_region"] == "partner_mouth"
            and g1["manual_review_action"] == "accept_auto"
            and not bool(g1["review_required"])
        )
        checks["live_preview_matches_reviewed_artifact"] = (
            legacy_sample["effective_face_entity_id"] == g0["face_entity_id"]
            and legacy_sample["effective_face_social_region"] == g0["face_social_region"]
            and accepted_preview["selected_sample"]["effective_face_social_region"] == g1["face_social_region"]
        )
        checks["decision_audit_csv"] = Path(result["decisions_csv"]).is_file()
        checks["reviewed_frame_table"] = Path(result["frames_csv"]).is_file()
        checks["reviewed_clip_summary_files"] = (
            Path(result["clip_summary_csv"]).is_file()
            and Path(result["clip_summary_json"]).is_file()
        )
        checks["auto_review_audit_preserved"] = (
            Path(result["auto_review_summary_csv"]).is_file()
            and Path(result["auto_review_summary_json"]).is_file()
        )
        preserved_auto_summary = json.loads(Path(result["auto_review_summary_json"]).read_text("utf-8"))
        checks["auto_review_audit_has_manual_lineage"] = (
            preserved_auto_summary.get("policy") == "conservative"
            and preserved_auto_summary.get("manual_review_lineage", {}).get("source_face_aoi_artifact_id") == face.id
            and preserved_auto_summary.get("manual_review_lineage", {}).get("manual_review_applied_sample_count") == 2
        )
        checks["reviewed_clip_stats_recomputed"] = (
            int(reviewed_artifact.qc.get("valid_gaze_sample_count", 0)) == 3
            and int(reviewed_artifact.qc.get("partner_face_sample_count", 0)) == 3
            and abs(float(reviewed_artifact.qc.get("partner_face_sample_fraction", 0.0)) - 1.0) < 1e-9
            and abs(float(reviewed_artifact.qc.get("partner_nose_sample_fraction", 0.0)) - (1 / 3)) < 1e-9
            and abs(float(reviewed_artifact.qc.get("partner_mouth_sample_fraction", 0.0)) - (2 / 3)) < 1e-9
        )

        fixation_batch = save_face_review_decision(
            store, session_id=session_id, artifact_id=face.id, video_artifact_id=video.id,
            output_frame_index=0, gaze_sample_uid="g0", apply_to="fixation",
            action="accept_auto", reviewer="validator",
        )
        checks["fixation_batch_updates_all_flagged_samples"] = (
            fixation_batch.get("changed_sample_count") == 2
            and set(fixation_batch.get("changed_sample_uids") or []) == {"g0", "g1"}
            and fixation_batch.get("apply_to_scope") == "fixation"
            and fixation_batch["manual_review"].get("pending_review_unit_count") == 0
        )

        report["source_artifact_id"] = face.id
        report["reviewed_artifact_id"] = reviewed_id
        report["all_passed"] = all(checks.values())

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
