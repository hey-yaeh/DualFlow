from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from oculostudio.domain.models import AssetKind, AssetRecord, ArtifactKind
from oculostudio.services.etm_role_assignment import create_etm_role_assignment
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def _write_npz(path: Path, labels: list[str], points: np.ndarray, valid: np.ndarray, fps: float = 120.0) -> None:
    np.savez(
        path,
        time_sec=np.arange(len(points), dtype=float) / fps,
        labels=np.asarray(labels, dtype=object),
        points_mm=points,
        valid=valid,
        frame_rate=np.asarray(fps),
    )


def _body(path: Path, prefix: str, centre: np.ndarray, frames: int = 720) -> None:
    labels = [f"{prefix}:LFHD", f"{prefix}:LBHD", f"{prefix}:RFHD", f"{prefix}:RBHD"]
    head_shape = np.array([[-45,55,15],[-42,-45,5],[45,55,15],[42,-45,5]], dtype=float)
    t = np.arange(frames) / 120.0
    trajectory = np.column_stack([110*np.sin(t*.22), 35*np.sin(t*.39), 12*np.cos(t*.31)])
    points = centre[None,None,:] + trajectory[:,None,:] + head_shape[None,:,:]
    _write_npz(path, labels, points, np.ones(points.shape[:2], bool))


def _etm(path: Path, prefix: str, head_centre: np.ndarray, shape: np.ndarray, frames: int = 720) -> list[str]:
    labels = [f"{prefix}:M_{i+1}" for i in range(len(shape))]
    t = np.arange(frames) / 120.0
    trajectory = np.column_stack([110*np.sin(t*.22), 35*np.sin(t*.39), 12*np.cos(t*.31)])
    points = head_centre[None,None,:] + trajectory[:,None,:] + np.array([0,0,110])[None,None,:] + shape[None,:,:]
    valid = np.ones(points.shape[:2], bool)
    valid[80,:] = False
    valid[230:233,:] = False
    _write_npz(path, labels, points, valid)
    return labels


def _profile(store: ProjectStore, sid: str, label: str, participant: str, labels: list[str], shape: np.ndarray):
    root = store.derived_dir(sid, "calibrations", label)
    root.mkdir(parents=True, exist_ok=True)
    profile = root / f"{label}.oculocal"
    profile.write_text(json.dumps({
        "profile_id": label,
        "profile_name": label,
        "participant_id": participant,
        "T_tracker_from_camera": np.eye(4).tolist(),
        "camera_intrinsics": {"matrix": [[600,0,320],[0,600,180],[0,0,1]], "distortion": [0,0,0,0,0], "width":640, "height":360},
        "tracker": {"source_asset_id": None, "marker_labels": labels, "marker_template_mm": (shape-shape.mean(axis=0)).tolist()},
    }, indent=2), "utf-8")
    return ArtifactStore(store).register(
        session_id=sid,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=label,
        output_root=root,
        qc={"accepted_count": 180, "reprojection_rms_px": 0.9},
    )


def build(output: Path, *, overwrite: bool = False) -> dict:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "synthetic_inputs"
    inputs.mkdir(exist_ok=True)
    store = ProjectStore.create(output, "ETM wearer-resolution demo", session_label="Session S001")
    sid = store.project.selected_session_id
    assert sid
    r01 = inputs / "S001_I04_R01_C3D.npz"
    r02 = inputs / "S001_I04_R02_C3D.npz"
    etm_a = inputs / "S001_I04_ETM_A.npz"
    etm_b = inputs / "S001_I04_ETM_B.npz"
    c1 = np.array([0.,0.,1600.])
    c2 = np.array([1700.,0.,1600.])
    _body(r01, "Role01", c1)
    _body(r02, "Role02", c2)
    shape_a = np.array([[-32,-18,0],[-15,25,5],[8,-24,-2],[30,18,2],[-4,4,30],[12,3,-28]],float)
    shape_b = np.array([[-75,-10,0],[-10,70,10],[65,-30,-8],[45,55,4],[-32,23,42],[15,-35,-48]],float)
    labels_a = _etm(etm_a, "eye_tracker_1", c2, shape_a)
    labels_b = _etm(etm_b, "eye_tracker_2", c1, shape_b)
    records = {}
    for label,path,participant,count in [
        ("S001_I04_R01_C3D",r01,"R01",4),
        ("S001_I04_R02_C3D",r02,"R02",4),
        ("S001_I04_ETM_A",etm_a,None,6),
        ("S001_I04_ETM_B",etm_b,None,6),
    ]:
        record=AssetRecord(label=label,kind=AssetKind.MOCAP_NPZ,source_path=str(path),participant_id=participant,metadata={"marker_count":count,"frame_count":720,"frame_rate":120})
        records[label]=record
    store.update(lambda p: p.sessions[sid].assets.update({r.id:r for r in records.values()}))
    p1=_profile(store,sid,"eye_tracker_2 calibration","P01",labels_b,shape_b)
    p2=_profile(store,sid,"eye_tracker_1 calibration","P02",labels_a,shape_a)
    result=create_etm_role_assignment(
        store,
        session_id=sid,
        body_asset_ids=[records["S001_I04_R01_C3D"].id,records["S001_I04_R02_C3D"].id],
        etm_asset_ids=[records["S001_I04_ETM_A"].id,records["S001_I04_ETM_B"].id],
        role_labels={records["S001_I04_R01_C3D"].id:"R01",records["S001_I04_R02_C3D"].id:"R02"},
        calibration_profile_artifact_ids=[p1.id,p2.id],
        label="S001_I04 ETM wearer assignment",
    )
    return {"project":str(output),"session_id":sid,"assignment":result}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    print(json.dumps(build(Path(args.output).resolve(),overwrite=args.overwrite),indent=2,default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
