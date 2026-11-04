from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from oculostudio.domain.models import AssetKind, AssetRecord, ArtifactKind
from oculostudio.services.calibration_sets import (
    create_calibration_set,
    register_calibration_asset_refs,
)
from oculostudio.services.calibration_set_profile_builder import (
    build_calibration_set_profiles,
    calibration_set_profile_build_status,
)
import oculostudio.services.calibration_set_profile_builder as profile_builder_module
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def add_source(store: ProjectStore, session_id: str, root: Path, name: str, kind: AssetKind, role: str, participant=None, hardware=None):
    path = root / name
    path.write_bytes(b"fixture")
    asset = AssetRecord(
        label=name,
        kind=kind,
        source_path=str(path),
        participant_id=participant,
        metadata={
            "asset_scope": "calibration",
            "calibration_role": role,
            "hardware_hint": hardware,
        },
    )
    def mutate(project):
        project.sessions[session_id].assets[asset.id] = asset
    store.update(mutate)
    return asset


FORWARDED_BOARD_MAPPINGS = []


def fake_profile_creator(store: ProjectStore, *, session_id: str, native_asset_id: str, tracker_asset_ids: list[str], participant_id: str, profile_name: str, hardware_id: str, mount_version: str, **kwargs):
    FORWARDED_BOARD_MAPPINGS.append({
        "hardware_id": hardware_id,
        "board_corner_labels": kwargs.get("board_corner_labels"),
        "outer_corner_labels": kwargs.get("outer_corner_labels"),
        "board_mapping_source": kwargs.get("board_mapping_source"),
    })
    out = store.derived_dir(session_id, "calibration_profile") / profile_name
    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / f"{profile_name}.json"
    profile = {
        "schema_version": "oculostudio-test-profile-1.0",
        "hardware_id": hardware_id,
        "hardware_display_name": hardware_id.replace("_", "-"),
        "mount_version": mount_version,
        "participant_id": participant_id,
        "native_asset_id": native_asset_id,
        "tracker_asset_id": tracker_asset_ids[0],
        "T_tracker_from_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "camera_intrinsics": {"matrix": [[1000,0,800],[0,1000,600],[0,0,1]], "distortion": [0,0,0,0,0,0,0,0], "resolution": [1600,1200]},
        "tracker": {"source_asset_id": tracker_asset_ids[0], "marker_labels": ["M1","M2","M3"], "marker_template_mm": [[0,0,0],[10,0,0],[0,10,0]]},
        "metadata": {"hardware_id": hardware_id, "mount_version": mount_version},
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=profile_name,
        output_root=out,
        source_asset_ids=[native_asset_id, tracker_asset_ids[0]],
        parameters={
            "hardware_id": hardware_id,
            "hardware_display_name": hardware_id.replace("_", "-"),
            "mount_version": mount_version,
            "participant_id": participant_id,
            "calibration_source": "measured_apriltag",
        },
        qc={
            "hardware_id": hardware_id,
            "hardware_display_name": hardware_id.replace("_", "-"),
            "participant_id": participant_id,
        },
    )
    return {
        "artifact": artifact.model_dump(mode="json"),
        "profile_path": str(profile_path),
        "g3dcal_path": str(profile_path),
        "profile": profile,
        "hardware_registration": {"hardware_id": hardware_id},
    }


def run():
    checks = {}
    with tempfile.TemporaryDirectory(prefix="oculo_calset_profile_") as td:
        root = Path(td) / "project"
        store = ProjectStore.create(
            root,
            "OCUMO S004-S007",
            session_labels=["S004", "S005", "S006", "S007"],
            settings={"folder_layout_version": 3, "initial_calibration_set": {"enabled": False}},
        )
        snapshot = store.snapshot()
        session_ids = list(snapshot.sessions)
        owner = session_ids[0]
        calset = create_calibration_set(
            store,
            code="CAL_S004_S007",
            label="Calibration S004-S007",
            applicable_session_ids=[owner],
            owner_session_id=owner,
            purpose="combined",
        )
        set_id = calset["id"]

        sources = [
            add_source(store, owner, root, "P01_calibration_S004-S007.zip", AssetKind.PUPIL_NATIVE, "pupil_calibration", "P01"),
            add_source(store, owner, root, "P03_calibration_S004-S007.zip", AssetKind.PUPIL_NATIVE, "pupil_calibration", "P03"),
            add_source(store, owner, root, "calibration_board_S004-S007.c3d", AssetKind.C3D, "calibration_board"),
            add_source(store, owner, root, "eye_tracker_A_S004-S007.c3d", AssetKind.C3D, "tracker_candidate", hardware="ETM_A"),
            add_source(store, owner, root, "eye_tracker_B_S004-S007.c3d", AssetKind.C3D, "tracker_candidate", hardware="ETM_B"),
            add_source(store, owner, root, "markers_world_S004-S007.c3d", AssetKind.C3D, "world_reference"),
        ]
        register_calibration_asset_refs(
            store,
            calibration_set_id=set_id,
            source_session_id=owner,
            asset_ids=[asset.id for asset in sources],
        )

        class FakeBoard:
            labels = [f"M{i:02d}" for i in range(16)]
            point_count = 16
            frame_count = 100
            frame_rate_hz = 120.0

        profile_builder_module.legacy_read_c3d = lambda path: FakeBoard()
        fake_tags = {
            "0": {"TL": "M00", "TR": "M01", "BR": "M05", "BL": "M04"},
            "1": {"TL": "M02", "TR": "M03", "BR": "M07", "BL": "M06"},
            "2": {"TL": "M08", "TR": "M09", "BR": "M13", "BL": "M12"},
            "3": {"TL": "M10", "TR": "M11", "BR": "M15", "BL": "M14"},
        }
        fake_outer = {"TL": "M00", "TR": "M03", "BR": "M15", "BL": "M12"}
        fake_definition = {
            "schema_version": "oculostudio-apriltag-board-definition-1.0",
            "mapping_source": "test_combined_identity_and_board_geometry",
            "mapping_variant": "test_grid_rot0",
            "board_corner_labels": fake_tags,
            "outer_corner_labels": fake_outer,
            "definition_path": str(root / "board_definition.json"),
        }

        status = calibration_set_profile_build_status(store, calibration_set_id=set_id)
        checks["source_inventory_complete"] = status["source_complete"] and status["can_build"]
        checks["current_interaction_participant_not_required"] = status["contract"]["current_interaction_participant_required"] is False
        checks["two_calibration_actors_found"] = sorted(x["participant_id"] for x in status["pupil_recordings"]) == ["P01", "P03"]

        # Simulate the important real-world condition: P01 calibration actor used ETM-B,
        # P03 calibration actor used ETM-A; current interaction R02 could still be P02.
        def fake_pairing(*, native_asset, participant_id, tracker_items, **kwargs):
            hardware = "ETM_B" if participant_id == "P01" else "ETM_A"
            tracker = next(item for item in tracker_items if item["ref"].hardware_hint == hardware)
            return {
                "participant_id": participant_id,
                "native_asset_id": native_asset.id,
                "native_label": native_asset.label,
                "selected_tracker_asset_id": tracker["asset"].id,
                "selected_tracker_label": tracker["asset"].label,
                "hardware_id": hardware,
                "hardware_display_name": hardware.replace("_", "-"),
                "confidence": "HIGH",
                "identity": {"selected_tracker_filename": tracker["asset"].label, "confidence": "HIGH"},
                "logs": [],
            }

        with patch("oculostudio.services.calibration_set_profile_builder._preflight_pairing", side_effect=fake_pairing), \
             patch("oculostudio.services.calibration_set_profile_builder._resolve_shared_board_definition", return_value=fake_definition), \
             patch("oculostudio.services.calibration_set_profile_builder.create_legacy_gaze3d_calibration_profile", side_effect=fake_profile_creator):
            result = build_calibration_set_profiles(
                store,
                session_id=owner,
                calibration_set_id=set_id,
                minimum_tracker_identity_confidence="MEDIUM",
            )

        final = calibration_set_profile_build_status(store, calibration_set_id=set_id)
        checks["build_completed"] = result["status"] == "completed"
        checks["both_hardware_slots_ready"] = final["ready"] and not final["missing_hardware"]
        checks["p01_mapped_to_etm_b"] = any(x["participant_id"] == "P01" and x["hardware_id"] == "ETM_B" for x in result["created_profiles"])
        checks["p03_mapped_to_etm_a"] = any(x["participant_id"] == "P03" and x["hardware_id"] == "ETM_A" for x in result["created_profiles"])
        checks["resolved_board_mapping_forwarded_to_both_estimators"] = (
            len(FORWARDED_BOARD_MAPPINGS) >= 2
            and all(row["board_corner_labels"] == fake_tags for row in FORWARDED_BOARD_MAPPINGS[:2])
            and all(row["outer_corner_labels"] == fake_outer for row in FORWARDED_BOARD_MAPPINGS[:2])
            and all(row["board_mapping_source"] == fake_definition["mapping_source"] for row in FORWARDED_BOARD_MAPPINGS[:2])
        )

        # Apply the ready Set to all Sessions. Source aliases are mocked because
        # these test fixtures are intentionally tiny, while profile files are real.
        from oculostudio.services.calibration_sets import set_calibration_sessions
        def fake_add_asset(store, *, session_id, path, label=None, participant_id=None, metadata_extra=None, **kwargs):
            asset = AssetRecord(label=label or Path(path).name, kind=AssetKind.UNKNOWN, source_path=str(path), participant_id=participant_id, metadata=metadata_extra or {})
            store.update(lambda project: project.sessions[session_id].assets.__setitem__(asset.id, asset))
            return asset
        with patch("oculostudio.services.calibration_sets.add_asset", side_effect=fake_add_asset):
            set_calibration_sessions(store, calibration_set_id=set_id, session_ids=session_ids)
        project = store.snapshot()
        # Each assigned Session must contain traceable profile artifacts after materialization.
        for sid in session_ids:
            profiles = [a for a in project.sessions[sid].artifacts.values() if str(a.kind) == ArtifactKind.CALIBRATION_PROFILE.value]
            hardware = {str(a.parameters.get("hardware_id")) for a in profiles}
            checks[f"session_{project.sessions[sid].label}_has_both_profiles"] = {"ETM_A", "ETM_B"}.issubset(hardware)

        # Atomic failure check: two calibration actors mapping to the same physical ETM
        # must not mutate a fresh revision with a half-pair.
        from oculostudio.services.calibration_sets import create_calibration_revision
        fresh = create_calibration_revision(store, calibration_set_id=set_id, inherit_active=False, activate=True)
        fresh_revision = fresh["active_revision_id"]
        # copy only the source refs into fresh revision for this test
        def copy_sources(model):
            row = model.calibration_sets[set_id]
            previous = sorted(row.revisions.values(), key=lambda r:r.revision_number)[-2]
            row.revisions[fresh_revision].source_assets = list(previous.source_assets)
        store.update(copy_sources)

        def duplicate_pairing(*, native_asset, participant_id, tracker_items, **kwargs):
            tracker = next(item for item in tracker_items if item["ref"].hardware_hint == "ETM_A")
            return {
                "participant_id": participant_id,
                "native_asset_id": native_asset.id,
                "native_label": native_asset.label,
                "selected_tracker_asset_id": tracker["asset"].id,
                "selected_tracker_label": tracker["asset"].label,
                "hardware_id": "ETM_A",
                "hardware_display_name": "ETM-A",
                "confidence": "HIGH",
                "identity": {},
                "logs": [],
            }

        failed = False
        with patch("oculostudio.services.calibration_set_profile_builder._preflight_pairing", side_effect=duplicate_pairing), \
             patch("oculostudio.services.calibration_set_profile_builder._resolve_shared_board_definition", return_value=fake_definition), \
             patch("oculostudio.services.calibration_set_profile_builder.create_legacy_gaze3d_calibration_profile", side_effect=fake_profile_creator):
            try:
                build_calibration_set_profiles(store, session_id=owner, calibration_set_id=set_id, revision_id=fresh_revision)
            except Exception:
                failed = True
        rev = store.snapshot().calibration_sets[set_id].revisions[fresh_revision]
        checks["ambiguous_duplicate_hardware_fails"] = failed
        checks["failed_pair_does_not_commit_half_profile"] = len(rev.profiles) == 0

        # Recovery check: older role-detail workflows could write valid measured
        # artifacts and fail before attaching them to the Calibration Set revision.
        # A new revision with two explicitly tagged unbound profiles must recover
        # without invoking the estimator again.
        recovery = create_calibration_revision(store, calibration_set_id=set_id, inherit_active=False, activate=True)
        recovery_revision = recovery["active_revision_id"]
        def copy_recovery_sources(model):
            row = model.calibration_sets[set_id]
            previous = sorted(row.revisions.values(), key=lambda r:r.revision_number)[-2]
            row.revisions[recovery_revision].source_assets = list(previous.source_assets)
        store.update(copy_recovery_sources)

        by_name = {asset.label: asset for asset in store.snapshot().sessions[owner].assets.values()}
        recover_artifacts = []
        for hardware, participant, tracker_name in [
            ("ETM_A", "P03", "eye_tracker_A_S004-S007.c3d"),
            ("ETM_B", "P01", "eye_tracker_B_S004-S007.c3d"),
        ]:
            native_name = f"{participant}_calibration_S004-S007.zip"
            result_unbound = fake_profile_creator(
                store, session_id=owner, native_asset_id=by_name[native_name].id,
                tracker_asset_ids=[by_name[tracker_name].id], participant_id=participant,
                profile_name=f"RECOVER_{hardware}_{recovery_revision}",
                hardware_id=hardware, mount_version="mount_v1",
            )
            artifact_id = result_unbound["artifact"]["id"]
            def tag_recoverable(model, aid=artifact_id, hw=hardware):
                artifact = model.sessions[owner].artifacts[aid]
                artifact.parameters.update({
                    "calibration_set_id": "CAL_S004_S007",
                    "calibration_set_record_id": set_id,
                    "calibration_revision_id": recovery_revision,
                    "hardware_id": hw,
                    "calibration_set_profile_build_status": "candidate",
                })
                artifact.qc.update({
                    "calibration_set_id": "CAL_S004_S007",
                    "calibration_set_record_id": set_id,
                    "calibration_revision_id": recovery_revision,
                    "hardware_id": hw,
                })
            store.update(tag_recoverable)
            recover_artifacts.append(artifact_id)

        with patch(
            "oculostudio.services.calibration_set_profile_builder.create_legacy_gaze3d_calibration_profile",
            side_effect=AssertionError("recovery should not recompute profiles"),
        ):
            recovered_result = build_calibration_set_profiles(
                store, session_id=owner, calibration_set_id=set_id, revision_id=recovery_revision
            )
        recovered_status = calibration_set_profile_build_status(
            store, calibration_set_id=set_id, revision_id=recovery_revision
        )
        checks["unbound_profiles_recovered_without_recompute"] = (
            recovered_result["status"] == "ready"
            and len(recovered_result.get("recovered_profiles") or []) == 2
            and recovered_status["ready"]
        )
        checks["recovered_revision_has_both_hardware_profiles"] = {
            ref.hardware_id for ref in store.snapshot().calibration_sets[set_id].revisions[recovery_revision].profiles
        } == {"ETM_A", "ETM_B"}

        # Static UI contract checks.
        import oculostudio as _installed_oculo
        appjs = (Path(_installed_oculo.__file__).resolve().parent / 'static' / 'app.js').read_text('utf-8')
        checks["ui_has_project_level_builder"] = "openCalibrationSetProfileBuilder" in appjs
        checks["ui_explains_actor_wearer_difference"] = "calibration actor does not have to equal the current wearer" in appjs.lower()
        checks["ui_filters_active_calibration_set_profiles"] = "calibrationProfilesForActiveSet" in appjs
        checks["ui_old_manual_route_is_recovery_only"] = "Advanced manual profile estimation / recovery only" in appjs
        checks["ui_exposes_board_geometry_state"] = "Shared AprilTag board geometry" in appjs
        checks["ui_explains_independent_evidence_fusion"] = "Independent evidence fusion" in appjs

        report = {
            "schema_version": "oculostudio-calibration-set-profile-builder-check-1.0",
            "checks": checks,
            "passed": sum(bool(v) for v in checks.values()),
            "total": len(checks),
            "all_pass": all(checks.values()),
            "created_profiles": result["created_profiles"],
            "final_status": final,
        }
        print(json.dumps(report, indent=2))
        if not report["all_pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    run()
