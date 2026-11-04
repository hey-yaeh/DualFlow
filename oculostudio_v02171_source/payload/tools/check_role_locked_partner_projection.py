"""Portable release check for OculoStudio v0.21.71 role-locked partner projection."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import oculostudio
from oculostudio.demo import create_demo_project
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ValidationError
from oculostudio.services.face_aoi import create_face_aoi_annotations
from oculostudio.services.native_reference_trim import export_native_reference_trim, preview_native_reference_trim
from oculostudio.services.participant_association import create_participant_association
from oculostudio.services import participant_association as association_module
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json


def main() -> int:
    checks: dict[str, bool] = {"installed_version": oculostudio.__version__ == "0.21.71"}
    with tempfile.TemporaryDirectory(prefix="oculostudio_02164_role_lock_") as temp:
        root = Path(temp) / "project"
        base = create_demo_project(root, run_processing=False)
        store = ProjectStore.open(root)
        sid = base["session_id"]
        assets = base["assets"]
        profile_id = base["profile_artifact_id"]
        head = ["P02:LFHD", "P02:RFHD", "P02:LBHD", "P02:RBHD"]

        assignment_root = store.derived_dir(sid, "etm_assignment", "role_lock_check")
        assignment_root.mkdir(parents=True, exist_ok=True)
        assignment_payload = {
            "schema_version": "oculostudio-etm-role-assignment-1.0",
            "status": "resolved",
            "assignments": {
                "R01": {
                    "role_id": "R01", "body_asset_id": assets["body"],
                    "body_asset_label": "R01 body", "body_head_marker_labels": head,
                    "etm_asset_id": "unused-r01-etm", "etm_id": "ETM-B", "confidence": "HIGH",
                },
                "R02": {
                    "role_id": "R02", "body_asset_id": "unused-r02-body",
                    "body_asset_label": "R02 body", "body_head_marker_labels": head,
                    "etm_asset_id": assets["tracker"], "etm_asset_label": "ETM-A tracker",
                    "etm_id": "ETM-A", "tracker_prefix": "tracker",
                    "recommended_calibration_artifact_id": profile_id, "confidence": "HIGH",
                },
            },
        }
        atomic_write_json(assignment_root / "etm_role_assignment.json", assignment_payload)
        assignment = ArtifactStore(store).register(
            session_id=sid, kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
            label="S001_I01 paired Calibration handoff", output_root=assignment_root,
            source_asset_ids=[assets["tracker"], assets["body"]],
            parameters={"calibration_phase": "handoff", "interaction_id": "I01"},
            qc={"status": "resolved", "handoff_complete": True,
                "assignments_summary": assignment_payload["assignments"]},
            provenance={"synthetic": True},
        )

        def make_trim(start: str, label: str) -> dict:
            preview = preview_native_reference_trim(
                store, session_id=sid, native_asset_id=assets["native"],
                timeseries_asset_id=assets["cloud"], facemapper_asset_id=assets["facemapper"],
                start_timecode=start, duration_timecode="00:00:03:00", timecode_fps=30,
                label=label, face_tolerance_ms=8.0,
            )
            return export_native_reference_trim(store, session_id=sid, preview_id=preview["preview_id"])

        trim_r01 = make_trim("00:00:00:00", "S001_I01_C3D_aligned_clip - R01")
        trim_r02 = make_trim("00:00:02:00", "S001_I01_C3D_aligned_clip - R02")

        wrong_profile_blocked = False
        try:
            create_participant_association(
                store, session_id=sid, native_trim_artifact_id=trim_r02["artifact_id"],
                etm_assignment_artifact_id=assignment.id, wearer_role_id="R02",
                calibration_profile_artifact_id="artifact_wrong_profile", target_policy="partner_only",
                label="must fail wrong profile",
            )
        except ValidationError as exc:
            wrong_profile_blocked = "does not match the paired handoff" in str(exc)
        checks["wrong_role_profile_blocked"] = wrong_profile_blocked

        association = create_participant_association(
            store, session_id=sid, native_trim_artifact_id=trim_r02["artifact_id"],
            etm_assignment_artifact_id=assignment.id, wearer_role_id="R02",
            target_policy="partner_only", minimum_identity_confidence=0.25,
            label="S001_I01_R02_participant_association",
        )
        artifact = store.project.sessions[sid].artifacts[association["artifact"]["id"]]
        projections = pd.read_csv(association["participant_head_projection"])
        valid = projections[
            projections["relation_to_wearer"].astype(str).eq("partner")
            & projections["projection_valid"].astype(bool)
            & projections["in_camera_view"].astype(bool)
        ]
        checks.update({
            "r02_resolves_etm_a": artifact.qc.get("resolved_etm_id") == "ETM-A",
            "profile_from_paired_handoff": artifact.qc.get("profile_resolution_source") == "paired_calibration_handoff",
            "valid_in_view_partner_projection": len(valid) > 0,
            "projection_count_audited": int(artifact.qc.get("partner_projection_in_view_count", 0)) == len(valid),
        })

        face = create_face_aoi_annotations(
            store, session_id=sid, native_trim_artifact_id=trim_r02["artifact_id"],
            participant_association_artifact_id=artifact.id, target_person_policy="partner_only",
            label="S001_I01_R02_face_aoi", render_overlay=False,
            face_left_right_policy="as_exported",
        )
        checks["same_trim_face_aoi_created"] = bool(face["artifact"]["id"])

        mismatch_blocked = False
        try:
            create_face_aoi_annotations(
                store, session_id=sid, native_trim_artifact_id=trim_r01["artifact_id"],
                participant_association_artifact_id=artifact.id, target_person_policy="partner_only",
                label="must fail trim mismatch", render_overlay=False,
                face_left_right_policy="as_exported",
            )
        except ValidationError as exc:
            mismatch_blocked = "exact same role/trim" in str(exc)
        checks["mismatched_trim_blocked"] = mismatch_blocked

        before = set(store.project.sessions[sid].artifacts)
        original = association_module._project_head

        def out_of_view(*args, **kwargs):
            row = original(*args, **kwargs)
            row["in_camera_view"] = False
            return row

        zero_blocked = False
        try:
            with patch("oculostudio.services.participant_association._project_head", side_effect=out_of_view):
                create_participant_association(
                    store, session_id=sid, native_trim_artifact_id=trim_r02["artifact_id"],
                    etm_assignment_artifact_id=assignment.id, wearer_role_id="R02",
                    target_policy="partner_only", minimum_identity_confidence=0.25,
                    label="must not publish zero projection",
                )
        except ValidationError as exc:
            zero_blocked = "No valid in-view partner head projections" in str(exc)
        store = ProjectStore.open(root)
        checks["zero_projection_blocked"] = zero_blocked
        checks["zero_projection_publishes_nothing"] = before == set(store.project.sessions[sid].artifacts)

        package = Path(oculostudio.__file__).resolve().parent
        app_js = (package / "static" / "app.js").read_text("utf-8")
        workspace_js = (package / "static" / "interaction_workspace.js").read_text("utf-8")
        checks.update({
            "role_locked_card_present": "Role-locked partner-head projection" in app_js,
            "profile_override_hidden_by_default": 'id="association-profile" type="hidden"' in app_js,
            "exact_trim_filter_present": "recordUsesNativeTrim" in app_js,
            "role_switch_does_not_autofill_profile": "'association-profile',true" not in workspace_js,
            "zero_projection_ui_explained": "zero valid in-view partner projections" in app_js,
        })

    failed = [name for name, value in checks.items() if not value]
    report = {
        "release": "OculoStudio v0.21.71 — Role-Locked Partner Projection",
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
