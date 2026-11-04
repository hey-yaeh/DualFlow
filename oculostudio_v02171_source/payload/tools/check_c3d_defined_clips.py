from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from oculostudio import __version__
from oculostudio.domain.models import ArtifactKind, TimeMap, TimeMapQuality
from oculostudio.errors import ValidationError
from oculostudio.services import add_asset, create_c3d_defined_aligned_clips
from oculostudio.storage import ProjectStore
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.util import atomic_write_json, new_id


def _table(columns, rows) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def _make_role(data: Path, role: str, base_ns: int, recording_id: str) -> tuple[Path, Path, Path]:
    duration_sec = 2.0
    frame_count = 60
    video = data / f"{role}_scene.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("Synthetic MP4 encoder is unavailable")
    for index in range(frame_count):
        image = np.full((120, 160, 3), [36, 48, 58], dtype=np.uint8)
        cv2.circle(image, (40 + index % 80, 60), 14, (170, 210, 180), -1)
        cv2.putText(image, f"{role} {index}", (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (245, 245, 245), 1)
        writer.write(image)
    writer.release()
    times = np.asarray([base_ns + round(index / 30.0 * 1e9) for index in range(frame_count)], dtype="<i8")
    duration_ns = int(duration_sec * 1e9)
    info = json.dumps({"recording_id": recording_id, "start_time": base_ns, "duration": duration_ns})
    nrd = data / f"S001_I01_{role}_NRD.zip"
    tsd = data / f"S001_I01_{role}_TSD.zip"
    face = data / f"S001_I01_{role}_FaceMapper.zip"
    with zipfile.ZipFile(nrd, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("recording/info.json", info)
        archive.writestr("recording/Neon Scene Camera v1 ps1.mp4", video.read_bytes())
        archive.writestr("recording/Neon Scene Camera v1 ps1.time", times.tobytes())
    video.unlink()
    gaze_times = [base_ns + index * 5_000_000 for index in range(int(duration_sec * 200))]
    section_columns = ["section id", "recording id", "section start time [ns]", "section end time [ns]"]
    gaze_columns = ["section id", "recording id", "timestamp [ns]", "gaze x [px]", "gaze y [px]", "worn"]
    with zipfile.ZipFile(tsd, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("recording/info.json", info)
        archive.writestr("recording/gaze.csv", _table(gaze_columns, [[f"{role}-section", recording_id, stamp, 80, 55, 1] for stamp in gaze_times]))
        archive.writestr("recording/imu.csv", _table(["recording id", "timestamp [ns]", "gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]"], [[recording_id, stamp, 0.1, 0.2, 0.3] for stamp in gaze_times]))
        archive.writestr("recording/world_timestamps.csv", _table(["recording id", "timestamp [ns]"], [[recording_id, int(stamp)] for stamp in times]))
        archive.writestr("sections.csv", _table(section_columns, [[f"{role}-section", recording_id, base_ns, base_ns + duration_ns]]))
    with zipfile.ZipFile(face, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sections.csv", _table(section_columns, [[f"{role}-face", recording_id, base_ns, base_ns + duration_ns]]))
        archive.writestr("gaze_on_face.csv", _table(["section id", "recording id", "timestamp [ns]", "gaze on face"], [[f"{role}-face", recording_id, stamp, "True"] for stamp in gaze_times]))
        archive.writestr("face_detections.csv", _table(["recording id", "timestamp [ns]", "p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"], [[recording_id, int(stamp), 55, 35, 105, 90] for stamp in times]))
    return nrd, tsd, face


def _make_mocap(path: Path, *, start_sec: float = 10.5, duration_sec: float = 1.0, rate_hz: float = 120.0) -> None:
    count = int(round(duration_sec * rate_hz))
    time_sec = start_sec + np.arange(count, dtype=float) / rate_hz
    labels = np.asarray(["A", "B", "C"], dtype=object)
    points = np.zeros((count, 3, 3), dtype=float)
    points[:, :, 2] = 1000.0
    valid = np.ones((count, 3), dtype=bool)
    np.savez(path, time_sec=time_sec, labels=labels, points_mm=points, valid=valid, frame_rate=np.asarray(rate_hz))


def _set_identity(store: ProjectStore, session_id: str, asset_id: str, role: str) -> None:
    def mutate(project):
        asset = project.sessions[session_id].assets[asset_id]
        asset.metadata["identity"] = {
            "session_id": "S001",
            "interaction_id": "I01",
            "role_id": role,
            "participant_id": "P01" if role == "R01" else "P02",
            "source": "release_self_check",
        }
    store.update(mutate)


def _fixture() -> tuple[tempfile.TemporaryDirectory, ProjectStore, str, object, dict[str, dict[str, str]]]:
    temp = tempfile.TemporaryDirectory(prefix="oculostudio_c3d_defined_clip_check_")
    base = Path(temp.name)
    inputs = base / "inputs"
    inputs.mkdir()
    store = ProjectStore.create(base / "project", "C3D-defined clips release check")
    session_id = store.project.selected_session_id
    role_assets: dict[str, dict[str, str]] = {}
    bases = {"R01": 1_800_000_000_000_000_000, "R02": 1_900_000_000_000_000_000}
    for role in ("R01", "R02"):
        nrd, tsd, face = _make_role(inputs, role, bases[role], f"recording-{role}")
        role_assets[role] = {}
        for key, path, label in (("native", nrd, "NRD"), ("tsd", tsd, "TSD"), ("face", face, "FaceMapper")):
            asset = add_asset(store, session_id=session_id, path=path, label=f"S001_I01_{role}_{label}", checksum=True)
            role_assets[role][key] = asset.id
            _set_identity(store, session_id, asset.id, role)
    body_ids: dict[str, str] = {}
    etm_ids: dict[str, str] = {}
    for role in ("R01", "R02"):
        path = inputs / f"S001_I01_{role}_body.npz"
        _make_mocap(path)
        asset = add_asset(store, session_id=session_id, path=path, label=f"S001_I01_{role}_body", checksum=True)
        body_ids[role] = asset.id
        _set_identity(store, session_id, asset.id, role)
    for token, role in (("A", "R02"), ("B", "R01")):
        path = inputs / f"S001_I01_ETM_{token}.npz"
        _make_mocap(path)
        asset = add_asset(store, session_id=session_id, path=path, label=f"S001_I01_ETM_{token}", checksum=True)
        etm_ids[token] = asset.id
        _set_identity(store, session_id, asset.id, role)
    sync_ids: dict[str, str] = {}
    map_ids: dict[str, str] = {}
    for role, token in (("R01", "B"), ("R02", "A")):
        snapshot = store.project.sessions[session_id]
        tsd = snapshot.assets[role_assets[role]["tsd"]]
        etm = snapshot.assets[etm_ids[token]]
        scale = 1.000004 if role == "R01" else 0.999993
        mapping = TimeMap(
            id=new_id("map"), source_domain_id=tsd.primary_time_domain_id, target_domain_id=etm.primary_time_domain_id,
            scale=scale, source_origin_ns=bases[role], target_origin_ns=10_000_000_000,
            offset_ns=10_000_000_000 - scale * bases[role], method="release_precision_audit",
            quality=TimeMapQuality(anchor_count=18, rms_residual_ns=900_000, max_abs_residual_ns=1_800_000, median_abs_residual_ns=500_000),
        )
        store.update(lambda project, row=mapping: project.sessions[session_id].time_maps.__setitem__(row.id, row))
        sync_root = store.derived_dir(session_id, "time_sync", new_id("syncroot"))
        atomic_write_json(sync_root / "accepted.json", {"role": role, "time_map_id": mapping.id})
        sync = ArtifactStore(store).register(
            session_id=session_id, kind=ArtifactKind.TIME_SYNC, label=f"Automatic time sync {role}", output_root=sync_root,
            source_asset_ids=[tsd.id, etm.id], parameters={"source_asset_id": tsd.id, "target_asset_id": etm.id, "wearer_role_id": role},
            qc={"production_ready": True, "precision_validated": True, "ego_video_used": False, "estimated_drift_ppm": (scale - 1.0) * 1e6},
        )
        sync_ids[role] = sync.id
        map_ids[role] = mapping.id
    handoff_root = store.derived_dir(session_id, "etm_role_assignment", new_id("handoffroot"))
    atomic_write_json(handoff_root / "handoff.json", {"ok": True})
    summary = {
        "R01": {"body_asset_id": body_ids["R01"], "etm_asset_id": etm_ids["B"], "recommended_time_sync_artifact_id": sync_ids["R01"], "recommended_time_map_id": map_ids["R01"], "etm_id": "ETM-B"},
        "R02": {"body_asset_id": body_ids["R02"], "etm_asset_id": etm_ids["A"], "recommended_time_sync_artifact_id": sync_ids["R02"], "recommended_time_map_id": map_ids["R02"], "etm_id": "ETM-A"},
    }
    handoff = ArtifactStore(store).register(
        session_id=session_id, kind=ArtifactKind.ETM_ROLE_ASSIGNMENT, label="S001 I01 synchronized Calibration handoff", output_root=handoff_root,
        source_asset_ids=[*body_ids.values(), *etm_ids.values()], parameters={"calibration_phase": "handoff", "require_time_sync": True},
        qc={"handoff_complete": True, "time_sync_complete": True, "assignments_summary": summary},
    )
    role_inputs = {
        role: {"native_asset_id": role_assets[role]["native"], "timeseries_asset_id": role_assets[role]["tsd"], "facemapper_asset_id": role_assets[role]["face"]}
        for role in ("R01", "R02")
    }
    return temp, store, session_id, handoff, role_inputs


def main() -> None:
    assert __version__ == "0.21.71", __version__
    temp, store, session_id, handoff, role_inputs = _fixture()
    try:
        result = create_c3d_defined_aligned_clips(
            store, session_id=session_id, calibration_handoff_artifact_id=handoff.id, role_inputs=role_inputs,
            label_prefix="Release check aligned clip", boundary_tolerance_ms=20.0,
        )
        session = store.project.sessions[session_id]
        pair = session.artifacts[result["artifact_id"]]
        assert pair.kind == ArtifactKind.ALIGNED_CLIP_SET
        assert pair.qc["all_roles_created"] is True
        assert pair.qc["source_video_reencoded"] is False
        assert len(result["roles"]) == 2
        assert float(pair.qc["maximum_start_boundary_error_ms"]) < 20.0
        assert float(pair.qc["maximum_end_boundary_error_ms"]) < 20.0
        for role in ("R01", "R02"):
            trim_id = result["roles"][role]["trim_artifact_id"]
            trim = session.artifacts[trim_id]
            assert trim.parameters["window_source"] == "common_body_c3d_overlap"
            assert trim.parameters["automatic_c3d_defined"] is True
            assert trim.provenance["source_mp4_reencoded"] is False
        mismatch_temp, mismatch_store, mismatch_session, mismatch_handoff, mismatch_inputs = _fixture()
        try:
            mismatch_inputs["R01"]["timeseries_asset_id"] = mismatch_inputs["R02"]["timeseries_asset_id"]
            before = len(mismatch_store.project.sessions[mismatch_session].artifacts)
            rejected = False
            try:
                create_c3d_defined_aligned_clips(
                    mismatch_store, session_id=mismatch_session, calibration_handoff_artifact_id=mismatch_handoff.id, role_inputs=mismatch_inputs,
                )
            except ValidationError:
                rejected = True
            after = len(mismatch_store.project.sessions[mismatch_session].artifacts)
            assert rejected and before == after
        finally:
            mismatch_temp.cleanup()
        print(json.dumps({
            "ok": True,
            "version": __version__,
            "common_duration_sec": result["common_vicon_window"]["duration_sec"],
            "role_trim_artifacts": {role: result["roles"][role]["trim_artifact_id"] for role in ("R01", "R02")},
            "maximum_start_boundary_error_ms": pair.qc["maximum_start_boundary_error_ms"],
            "maximum_end_boundary_error_ms": pair.qc["maximum_end_boundary_error_ms"],
            "source_video_reencoded": pair.qc["source_video_reencoded"],
            "mismatched_sync_source_rejected_transactionally": True,
        }, indent=2))
    finally:
        temp.cleanup()


if __name__ == "__main__":
    main()
