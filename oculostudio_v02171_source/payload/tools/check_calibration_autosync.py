from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from oculostudio.domain.models import AssetKind, AssetRecord, ArtifactKind, TimeDomain
from oculostudio.errors import ValidationError
from oculostudio.services.auto_synchronization import automatic_pupil_vicon_sync, get_automatic_time_sync_preview
from oculostudio.services.etm_role_assignment import bind_calibration_profiles_to_assignment
from oculostudio.services.gaze3d import generate_gaze_rays
from oculostudio.storage import ProjectStore
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.util import atomic_write_json, new_id


def motion(t: np.ndarray, phase: float = 0.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    events = [
        (7.0, 1.8, 0.9), (17.0, 2.4, 1.4), (31.0, 1.4, 1.0),
        (45.0, 2.1, 1.2), (61.0, 1.7, 1.5), (78.0, 2.5, 1.1),
        (94.0, 1.5, 1.3), (111.0, 2.2, 1.0),
    ]
    for c, a, w in events:
        y += a * np.exp(-0.5 * ((t - (c + phase)) / w) ** 2)
    y += 0.14 * np.maximum(0.0, np.sin(0.41 * t + 0.3 + phase))
    return y


def build_tracker(path: Path, *, prefix: str, template: np.ndarray, offset: float, scale: float, phase: float) -> None:
    rate = 120.0
    start = 36.75
    duration = 125.0
    tv_rel = np.arange(0.0, offset + scale * duration + 4.0, 1.0 / rate)
    tv = start + tv_rel
    equivalent = (tv_rel - offset) / scale
    speed = np.where((equivalent >= 0.0) & (equivalent <= duration), motion(np.clip(equivalent, 0.0, duration), phase), 0.0)
    theta = np.cumsum(speed) / rate
    points = np.empty((len(tv), len(template), 3), dtype=float)
    for i, angle in enumerate(theta):
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        points[i] = template @ rotation.T + np.asarray([420.0, 110.0, 1650.0])
    labels = np.asarray([f"{prefix}:M{i+1}" for i in range(len(template))])
    np.savez_compressed(
        path,
        time_sec=tv,
        labels=labels,
        points_mm=points,
        valid=np.ones(points.shape[:2], dtype=bool),
        frame_rate=np.asarray(rate),
    )


def make_project(root: Path):
    store = ProjectStore.create(root / "project", "Calibration auto-sync integration", session_label="S001")
    sid = store.project.selected_session_id
    assert sid
    src = root / "sources"
    src.mkdir(parents=True, exist_ok=True)

    role_specs = {
        # User's verified assignment: R01 wears ETM-B, R02 wears ETM-A.
        "R01": {"etm": "ETM_B", "offset": 2.85, "scale": 1.00014, "phase": 0.0},
        "R02": {"etm": "ETM_A", "offset": -1.75, "scale": 0.99988, "phase": 0.65},
    }
    templates = {
        "ETM_A": np.asarray([[-37, -14, 0], [33, -12, 2], [-28, 23, 8], [31, 26, 7], [-10, -3, 36], [17, 8, 41]], float),
        "ETM_B": np.asarray([[-31, -16, 1], [39, -10, 0], [-23, 27, 9], [36, 20, 8], [-5, -8, 31], [21, 11, 45]], float),
    }

    asset_ids: dict[str, dict[str, str]] = {}
    tracker_assets: dict[str, str] = {}
    tracker_labels: dict[str, list[str]] = {}

    def add_asset(project, asset: AssetRecord, domain: TimeDomain):
        session = project.sessions[sid]
        session.time_domains[domain.id] = domain
        session.assets[asset.id] = asset

    for role_index, (role, spec) in enumerate(role_specs.items()):
        ts_dir = src / f"S001_I01_{role}_TSD"
        ts_dir.mkdir()
        rate = 200.0
        duration = 125.0
        tp = np.arange(0.0, duration, 1.0 / rate)
        start_ns = 1_800_000_000_000_000_000 + role_index * 10_000_000_000
        timestamps = start_ns + np.rint(tp * 1e9).astype(np.int64)
        wz = np.degrees(motion(tp, float(spec["phase"])))
        pd.DataFrame({
            "timestamp [ns]": timestamps,
            "gyro x [deg/s]": np.zeros_like(wz),
            "gyro y [deg/s]": np.zeros_like(wz),
            "gyro z [deg/s]": wz,
        }).to_csv(ts_dir / "imu.csv", index=False)

        ts_id = new_id("asset")
        td_id = new_id("time")
        ts_asset = AssetRecord(
            id=ts_id,
            label=f"S001_I01_{role}_TSD",
            kind=AssetKind.PUPIL_CLOUD,
            source_path=str(ts_dir),
            primary_time_domain_id=td_id,
            time_domain_ids=[td_id],
            participant_id=f"P0{role_index+1}",
            metadata={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": role}},
        )
        ts_domain = TimeDomain(id=td_id, label=f"{role} Pupil clock", clock_type="absolute", asset_id=ts_id)
        store.update(lambda project, a=ts_asset, d=ts_domain: add_asset(project, a, d))
        asset_ids[role] = {"timeseries": ts_id}

    for etm_index, etm in enumerate(("ETM_A", "ETM_B")):
        path = src / f"S001_I01_{etm}.npz"
        spec_role = next(role for role, spec in role_specs.items() if spec["etm"] == etm)
        spec = role_specs[spec_role]
        build_tracker(path, prefix=etm, template=templates[etm], offset=float(spec["offset"]), scale=float(spec["scale"]), phase=float(spec["phase"]))
        asset_id = new_id("asset")
        domain_id = new_id("time")
        asset = AssetRecord(
            id=asset_id,
            label=f"S001_I01_{etm}",
            kind=AssetKind.MOCAP_NPZ,
            source_path=str(path),
            primary_time_domain_id=domain_id,
            time_domain_ids=[domain_id],
            metadata={"identity": {"session_id": "S001", "interaction_id": "I01"}, "hardware_id": etm},
        )
        domain = TimeDomain(id=domain_id, label=f"{etm} Vicon clock", clock_type="relative", asset_id=asset_id)
        store.update(lambda project, a=asset, d=domain: add_asset(project, a, d))
        tracker_assets[etm] = asset_id
        tracker_labels[etm] = [f"{etm}:M{i+1}" for i in range(len(templates[etm]))]

    assignment_root = store.derived_dir(sid, "etm_role_assignment", "integration")
    assignment_root.mkdir(parents=True, exist_ok=True)
    assignments = {}
    for role, spec in role_specs.items():
        etm = str(spec["etm"])
        assignments[role] = {
            "role_id": role,
            "participant_id": "P01" if role == "R01" else "P02",
            "body_asset_id": f"body_{role}",
            "body_asset_label": f"S001_I01_{role}_body",
            "body_subject_prefix": "James" if role == "R01" else "Yuan",
            "body_head_marker_labels": ["LFHD", "RFHD", "LBHD", "RBHD"],
            "etm_asset_id": tracker_assets[etm],
            "etm_asset_label": f"S001_I01_{etm}",
            "etm_id": etm,
            "tracker_prefix": etm,
            "tracker_marker_labels": tracker_labels[etm],
            "median_head_distance_mm": 72.0,
            "confidence": "HIGH",
        }
    assignment_payload = {
        "schema_version": "oculostudio-etm-role-assignment-1.0",
        "status": "resolved",
        "method": "integration_fixture",
        "assignments": assignments,
        "assignment_by_etm": {row["etm_id"]: row for row in assignments.values()},
        "warnings": [],
    }
    atomic_write_json(assignment_root / "etm_role_assignment.json", assignment_payload)
    assignment = ArtifactStore(store).register(
        session_id=sid,
        kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
        label="S001 I01 resolved ETM wearer assignment",
        output_root=assignment_root,
        source_asset_ids=list(tracker_assets.values()),
        parameters={"calibration_phase": "wearer_assignment", "interaction_id": "I01"},
        qc={"status": "resolved", "assignments_summary": assignments},
    )

    profile_ids = []
    for etm in ("ETM_A", "ETM_B"):
        root_profile = store.derived_dir(sid, "calibration_profile", etm)
        root_profile.mkdir(parents=True, exist_ok=True)
        profile = {
            "format": "oculostudio-calibration-profile-1.0",
            "profile_id": f"profile_{etm.lower()}",
            "profile_name": f"{etm} measured profile",
            "participant_id": "P01" if etm == "ETM_B" else "P02",
            "hardware_id": etm,
            "mount_version": "mount_v1",
            "calibration_source": "integration_fixture",
            "tracker": {
                "marker_labels": tracker_labels[etm],
                "marker_template_mm": templates[etm].tolist(),
            },
            "T_tracker_from_camera": np.eye(4).tolist(),
            "camera_intrinsics": {"matrix": [[900,0,800],[0,900,600],[0,0,1]], "distortion": [0,0,0,0,0]},
        }
        atomic_write_json(root_profile / f"{etm}_profile.json", profile)
        artifact = ArtifactStore(store).register(
            session_id=sid,
            kind=ArtifactKind.CALIBRATION_PROFILE,
            label=f"{etm} measured profile",
            output_root=root_profile,
            parameters={"hardware_id": etm, "mount_version": "mount_v1", "calibration_source": "integration_fixture"},
            qc={"hardware_id": etm, "calibration_source": "integration_fixture"},
        )
        profile_ids.append(artifact.id)

    return store, sid, assignment, profile_ids, asset_ids


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="oculo_cal_sync_integration_") as temp:
        store, sid, assignment, profile_ids, asset_ids = make_project(Path(temp))
        sync_ids = []
        for role in ("R01", "R02"):
            result = automatic_pupil_vicon_sync(
                store,
                session_id=sid,
                timeseries_asset_id=asset_ids[role]["timeseries"],
                etm_assignment_artifact_id=assignment.id,
                wearer_role_id=role,
                maximum_offset_sec=10.0,
                sample_rate_hz=120.0,
                save_if_high=True,
            )
            assert result["status"] == "saved", (role, result["status"], result["candidate"]["confidence"], result["candidate"].get("reasons"))
            artifact = result["artifact"]
            assert artifact["qc"]["production_ready"] is True
            assert artifact["qc"]["distributed_drift_validated"] is True
            assert artifact["qc"]["precision_validated"] is True
            assert artifact["qc"]["holdout_window_count"] >= 3
            assert artifact["qc"]["residual_validation_mode"] == "leave_one_window_out"
            assert artifact["qc"]["drift_ci95_ppm"] > 0
            assert artifact["qc"]["residual_resolution_ms"] > 0
            assert artifact["qc"]["ego_video_used"] is False
            assert artifact["parameters"]["wearer_role_id"] == role
            assert artifact["parameters"]["assignment_artifact_id"] == assignment.id
            expected_drift = 140.0 if role == "R01" else -120.0
            assert abs(float(artifact["qc"]["estimated_drift_ppm"]) - expected_drift) < 80.0
            preview = get_automatic_time_sync_preview(
                store, session_id=sid, artifact_id=artifact["id"], maximum_samples=400
            )
            assert preview["ego_video_used"] is False
            assert len(preview["signals"]) >= 100
            assert len(preview["windows"]) >= 3
            assert all("local_offset_ms" in row for row in preview["windows"] if row.get("accepted"))
            sync_ids.append(artifact["id"])

        # Explicit role metadata prevents accidentally synchronizing R01 to R02's Pupil clock.
        try:
            automatic_pupil_vicon_sync(
                store,
                session_id=sid,
                timeseries_asset_id=asset_ids["R02"]["timeseries"],
                etm_assignment_artifact_id=assignment.id,
                wearer_role_id="R01",
                maximum_offset_sec=10.0,
                sample_rate_hz=120.0,
                save_if_high=True,
            )
        except ValidationError as exc:
            assert "declared as R02" in str(exc)
        else:
            raise AssertionError("Role-mismatched Pupil Timeseries was not rejected")

        # Fail closed when only one role is synchronized.
        try:
            bind_calibration_profiles_to_assignment(
                store,
                session_id=sid,
                assignment_artifact_id=assignment.id,
                calibration_profile_artifact_ids=profile_ids,
                time_sync_artifact_ids=sync_ids[:1],
                require_time_sync=True,
                require_distributed_drift=True,
            )
        except ValidationError as exc:
            assert "needs an accepted" in str(exc)
        else:
            raise AssertionError("Handoff unexpectedly accepted a missing role synchronization")

        # A legacy/video-derived synchronization artifact cannot satisfy the no-video Calibration contract.
        source_sync = store.project.sessions[sid].artifacts[sync_ids[0]]
        video_root = store.derived_dir(sid, "time_sync", "video_derived_test")
        video_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(video_root / "video_sync.json", {"ego_video_used": True})
        video_sync = ArtifactStore(store).register(
            session_id=sid,
            kind=ArtifactKind.TIME_SYNC,
            label="legacy video-derived sync",
            output_root=video_root,
            source_asset_ids=list(source_sync.source_asset_ids),
            parent_artifact_ids=[assignment.id],
            parameters={**source_sync.parameters, "ego_video_used": True},
            qc={**source_sync.qc, "ego_video_used": True},
        )
        try:
            bind_calibration_profiles_to_assignment(
                store,
                session_id=sid,
                assignment_artifact_id=assignment.id,
                calibration_profile_artifact_ids=profile_ids,
                time_sync_artifact_ids=[video_sync.id, sync_ids[1]],
                require_time_sync=True,
                require_distributed_drift=True,
            )
        except ValidationError as exc:
            assert "ego-video-derived synchronization is not accepted" in str(exc)
        else:
            raise AssertionError("Video-derived synchronization unexpectedly satisfied Calibration")

        handoff = bind_calibration_profiles_to_assignment(
            store,
            session_id=sid,
            assignment_artifact_id=assignment.id,
            calibration_profile_artifact_ids=profile_ids,
            time_sync_artifact_ids=sync_ids,
            require_time_sync=True,
            require_distributed_drift=True,
            maximum_profile_shape_rms_mm=1.0,
            minimum_profile_shape_margin_mm=0.1,
            label="S001 I01 calibrated + synchronized handoff",
        )
        artifact = handoff["artifact"]
        assert artifact["qc"]["handoff_complete"] is True
        assert artifact["qc"]["time_sync_complete"] is True
        assert artifact["qc"]["time_sync_role_count"] == 2
        assert artifact["qc"]["time_sync_ego_video_used"] is False
        by_role = handoff["result"]["time_synchronization"]["by_role"]
        assert set(by_role) == {"R01", "R02"}
        assert all(row["production_ready"] for row in by_role.values())
        assert all(row["ego_video_used"] is False for row in by_role.values())
        for role, row in handoff["result"]["assignments"].items():
            assert row["recommended_time_map_id"]
            assert row["recommended_time_sync_artifact_id"]
            assert row["recommended_calibration_artifact_id"]

        # Legacy profile-only handoffs are no longer sufficient for Native-trim Gaze 3D.
        legacy_root = store.derived_dir(sid, "calibration_handoff", "legacy_profile_only")
        legacy_root.mkdir(parents=True, exist_ok=True)
        legacy_payload = json.loads(json.dumps(handoff["result"]))
        legacy_payload["schema_version"] = "oculostudio-calibration-handoff-1.0"
        legacy_payload.pop("time_synchronization", None)
        for row in legacy_payload["assignments"].values():
            row.pop("recommended_time_sync_artifact_id", None)
            row.pop("recommended_time_map_id", None)
            row.pop("time_sync_check", None)
        atomic_write_json(legacy_root / "etm_role_assignment.json", legacy_payload)
        legacy_handoff = ArtifactStore(store).register(
            session_id=sid,
            kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
            label="legacy profile-only handoff",
            output_root=legacy_root,
            source_asset_ids=assignment.source_asset_ids,
            parent_artifact_ids=[assignment.id, *profile_ids],
            parameters={
                "calibration_phase": "handoff",
                "binding_source_assignment_artifact_id": assignment.id,
                "calibration_profile_artifact_ids": profile_ids,
            },
            qc={"status": "resolved", "handoff_complete": True, "assignments_summary": legacy_payload["assignments"]},
        )
        try:
            generate_gaze_rays(
                store,
                session_id=sid,
                timeseries_asset_id=asset_ids["R01"]["timeseries"],
                etm_assignment_artifact_id=legacy_handoff.id,
                wearer_role_id="R01",
                scope="native_trim",
                native_trim_artifact_id="not-needed-for-early-guard",
                alignment_mode="timegraph",
            )
        except ValidationError as exc:
            assert "legacy Calibration handoff" in str(exc)
        else:
            raise AssertionError("Legacy profile-only handoff unexpectedly entered Native-trim Gaze3D")

        # Gaze3D must not silently use shared-start once synchronized Calibration handoff exists.
        try:
            generate_gaze_rays(
                store,
                session_id=sid,
                timeseries_asset_id=asset_ids["R01"]["timeseries"],
                etm_assignment_artifact_id=artifact["id"],
                wearer_role_id="R01",
                scope="native_trim",
                native_trim_artifact_id="not-needed-for-early-guard",
                alignment_mode="pretrimmed_shared_start",
            )
        except ValidationError as exc:
            assert "shared-start alignment is not allowed" in str(exc), str(exc)
        else:
            raise AssertionError("Gaze3D did not block shared-start bypass")

        # Sync artifacts tied to another wearer-assignment revision cannot be reused.
        other_root = store.derived_dir(sid, "etm_role_assignment", "alternate")
        other_root.mkdir(parents=True, exist_ok=True)
        source_payload = json.loads(Path(assignment.files[0].path).read_text("utf-8"))
        atomic_write_json(other_root / "etm_role_assignment.json", source_payload)
        alternate = ArtifactStore(store).register(
            session_id=sid,
            kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
            label="alternate assignment revision",
            output_root=other_root,
            source_asset_ids=assignment.source_asset_ids,
            parameters={"calibration_phase": "wearer_assignment"},
            qc={"status": "resolved"},
        )
        try:
            bind_calibration_profiles_to_assignment(
                store,
                session_id=sid,
                assignment_artifact_id=alternate.id,
                calibration_profile_artifact_ids=profile_ids,
                time_sync_artifact_ids=sync_ids,
                require_time_sync=True,
            )
        except ValidationError as exc:
            assert "different wearer assignment" in str(exc)
        else:
            raise AssertionError("Handoff reused synchronization from another assignment revision")

        summary = {
            "ok": True,
            "assignment_artifact_id": assignment.id,
            "time_sync_artifact_ids": sync_ids,
            "handoff_artifact_id": artifact["id"],
            "roles": by_role,
            "checks": [
                "two role-specific Pupil IMU↔ETM mappings created without ego video",
                "R01→ETM-B and R02→ETM-A assignment preserved",
                "role-mismatched Pupil Timeseries blocked",
                "missing role synchronization blocked",
                "video-derived synchronization blocked",
                "distributed drift required",
                "sub-sample local lags and independent leave-one-window-out residuals required",
                "drift reported with a finite non-zero 95% interval",
                "accepted synchronization preview evidence is readable without ego video",
                "calibration profiles and TimeGraphs bound into one immutable handoff",
                "legacy profile-only handoff blocked",
                "Gaze3D shared-start bypass blocked",
                "stale assignment synchronization blocked",
            ],
        }
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
