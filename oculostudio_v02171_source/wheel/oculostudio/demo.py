from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from oculostudio.domain.models import ArtifactKind, Clip
from oculostudio.services.assets import add_asset
from oculostudio.services.clipping import run_multimodal_clip
from oculostudio.services.gaze3d import generate_gaze_rays
from oculostudio.services.privacy import create_privacy_video
from oculostudio.services.semantic_targets import create_semantic_gaze_targets
from oculostudio.services.qc import export_qc_report
from oculostudio.services.synchronization import build_native_cloud_frame_map, fit_and_store_time_map
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json, new_id, utc_now


def _write_demo_video(path: Path, *, width: int, height: int, fps: float, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {path}")
    try:
        for index in range(frames):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            # Warm room background.
            image[:] = (44, 49, 39)
            cv2.rectangle(image, (0, int(height * .72)), (width, height), (62, 70, 79), -1)
            cv2.rectangle(image, (22, 28), (165, 250), (34, 38, 30), -1)
            for y in (65, 125, 185):
                cv2.line(image, (35, y), (150, y), (78, 76, 63), 5)
            cv2.circle(image, (int(width * .70), 70), 38, (97, 162, 218), -1)
            # Participant A.
            x = int(width * .43 + 4 * math.sin(index / 35))
            cv2.circle(image, (x, int(height * .39)), 47, (126, 164, 191), -1)
            cv2.ellipse(image, (x, int(height * .62)), (86, 105), 0, 180, 360, (171, 179, 183), -1)
            cv2.circle(image, (x - 16, int(height * .38)), 3, (35, 31, 29), -1)
            cv2.circle(image, (x + 16, int(height * .38)), 3, (35, 31, 29), -1)
            cv2.ellipse(image, (x, int(height * .43)), (14, 6), 0, 0, 180, (73, 83, 120), 2)
            # Participant B in foreground.
            cv2.circle(image, (int(width * .88), int(height * .38)), 63, (90, 126, 157), -1)
            cv2.ellipse(image, (int(width * .91), int(height * .77)), (110, 150), 0, 180, 360, (44, 48, 51), -1)
            # Synthetic gaze overlay and face box.
            box = (x - 56, int(height * .26), x + 56, int(height * .52))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (65, 250, 240), 2)
            gaze = (int(width * (.52 + .16 * math.sin(index / 45))), int(height * (.36 + .04 * math.cos(index / 31))))
            cv2.line(image, (x, int(height * .38)), gaze, (65, 250, 240), 2)
            cv2.circle(image, gaze, 6, (65, 250, 240), -1)
            cv2.putText(image, "OculoStudio synthetic session", (22, height - 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (225, 231, 220), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()


def _write_mocap(path: Path, *, duration_sec: float, fps: float = 120.0) -> dict[str, Any]:
    frames = int(round(duration_sec * fps))
    time_sec = np.arange(frames, dtype=float) / fps
    labels = np.asarray(["TRK_A", "TRK_B", "TRK_C", "TRK_D"], dtype=object)
    template = np.asarray([[-32, -22, 0], [32, -22, 0], [32, 22, 0], [-32, 22, 0]], dtype=float)
    points = np.empty((frames, len(labels), 3), dtype=float)
    for i, t in enumerate(time_sec):
        yaw = .12 * math.sin(t * .65)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        translation = np.asarray([180 * math.sin(t * .18), 40 * math.cos(t * .23), 1520 + 15 * math.sin(t)], dtype=float)
        points[i] = template @ rotation.T + translation
    valid = np.ones(points.shape[:2], dtype=bool)
    np.savez(path, time_sec=time_sec, labels=labels, points_mm=points, valid=valid, frame_rate=fps)
    return {"labels": labels.tolist(), "template": template.tolist()}




def _write_body_actor_mocap(
    path: Path,
    *,
    duration_sec: float,
    width: int,
    height: int,
    fps: float = 120.0,
) -> dict[str, Any]:
    """Write a synthetic participant whose head lies on the synthetic gaze path.

    Coordinates follow the same tracker/camera convention used by the demo
    calibration. The file deliberately uses a Vicon-style subject prefix so
    the multi-person safety logic and marker-prefix handling remain realistic.
    """
    frames = int(round(duration_sec * fps))
    time_sec = np.arange(frames, dtype=float) / fps
    labels = [
        "P02:LFHD", "P02:RFHD", "P02:LBHD", "P02:RBHD",
        "P02:C7", "P02:CLAV", "P02:STRN", "P02:T10",
        "P02:LASI", "P02:RASI", "P02:LPSI", "P02:RPSI",
        "P02:LSHO", "P02:LELB", "P02:LWRA", "P02:LFIN",
        "P02:RSHO", "P02:RELB", "P02:RWRA", "P02:RFIN",
        "P02:LKNE", "P02:LANK", "P02:LTOE",
        "P02:RKNE", "P02:RANK", "P02:RTOE",
    ]
    points = np.empty((frames, len(labels), 3), dtype=float)
    camera_origin_tracker = np.asarray([26.0, 4.0, 12.0], dtype=float)
    fx = fy = 520.0
    cx, cy = width / 2.0, height / 2.0

    for i, t in enumerate(time_sec):
        yaw = .12 * math.sin(t * .65)
        c, ss = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray([[c, -ss, 0], [ss, c, 0], [0, 0, 1]], dtype=float)
        tracker_translation = np.asarray(
            [180 * math.sin(t * .18), 40 * math.cos(t * .23), 1520 + 15 * math.sin(t)],
            dtype=float,
        )
        camera_origin = rotation @ camera_origin_tracker + tracker_translation
        scene_frame = t * 30.0
        face_center_x = width * .43 + 4 * math.sin(scene_frame / 35.0)
        left_eye_x = face_center_x - 16.0
        left_eye_y = height * .38
        direction_camera = np.asarray(
            [(left_eye_x - cx) / fx, (left_eye_y - cy) / fy, 1.0],
            dtype=float,
        )
        direction_camera /= np.linalg.norm(direction_camera)
        direction_world = rotation @ direction_camera
        head = camera_origin + direction_world * 1800.0
        right = rotation @ np.asarray([1.0, 0.0, 0.0])
        down = rotation @ np.asarray([0.0, 1.0, 0.0])
        forward = rotation @ np.asarray([0.0, 0.0, 1.0])

        neck = head + 125 * down
        chest = head + 310 * down
        pelvis = head + 650 * down
        left_shoulder = chest - 180 * right
        right_shoulder = chest + 180 * right
        left_elbow = left_shoulder + 235 * down - 35 * right
        right_elbow = right_shoulder + 235 * down + 35 * right
        left_wrist = left_elbow + 230 * down
        right_wrist = right_elbow + 230 * down
        left_hand = left_wrist + 85 * down
        right_hand = right_wrist + 85 * down
        left_hip = pelvis - 105 * right
        right_hip = pelvis + 105 * right
        left_knee = left_hip + 410 * down
        right_knee = right_hip + 410 * down
        left_ankle = left_knee + 400 * down
        right_ankle = right_knee + 400 * down
        left_toe = left_ankle + 120 * forward + 35 * down
        right_toe = right_ankle + 120 * forward + 35 * down

        marker_positions = {
            "P02:LFHD": head - 36 * right - 18 * down,
            "P02:RFHD": head + 36 * right - 18 * down,
            "P02:LBHD": head - 34 * right + 22 * down,
            "P02:RBHD": head + 34 * right + 22 * down,
            "P02:C7": neck - 18 * down,
            "P02:CLAV": chest - 24 * right - 12 * down,
            "P02:STRN": chest + 24 * right + 12 * down,
            "P02:T10": chest + 48 * down,
            "P02:LASI": left_hip,
            "P02:RASI": right_hip,
            "P02:LPSI": left_hip + 35 * down,
            "P02:RPSI": right_hip + 35 * down,
            "P02:LSHO": left_shoulder,
            "P02:LELB": left_elbow,
            "P02:LWRA": left_wrist,
            "P02:LFIN": left_hand,
            "P02:RSHO": right_shoulder,
            "P02:RELB": right_elbow,
            "P02:RWRA": right_wrist,
            "P02:RFIN": right_hand,
            "P02:LKNE": left_knee,
            "P02:LANK": left_ankle,
            "P02:LTOE": left_toe,
            "P02:RKNE": right_knee,
            "P02:RANK": right_ankle,
            "P02:RTOE": right_toe,
        }
        points[i] = np.asarray([marker_positions[label] for label in labels], dtype=float)

    valid = np.ones(points.shape[:2], dtype=bool)
    np.savez(
        path,
        time_sec=time_sec,
        labels=np.asarray(labels, dtype=object),
        points_mm=points,
        valid=valid,
        frame_rate=fps,
    )
    return {"labels": labels, "subject_prefix": "P02"}


def create_demo_project(output: str | Path, *, overwrite: bool = False, run_processing: bool = True) -> dict[str, Any]:
    root = Path(output).expanduser().resolve()
    if root.exists() and overwrite:
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Demo output is not empty: {root}")
    # Create the clean project before materializing synthetic source assets.
    # ProjectStore.create intentionally rejects non-empty targets, so creating
    # demo_assets first would make all bundled demo launchers fail.
    store = ProjectStore.create(root, "OculoStudio Synthetic Demo", session_label="Session S001")
    assets_root = root / "demo_assets"
    native_root = assets_root / "Native Recording" / "recording"
    cloud_root = assets_root / "Timeseries Data + Scene Video" / "recording"
    face_root = assets_root / "Face Mapper"
    mocap_root = assets_root / "Vicon"
    for path in (native_root, cloud_root, face_root, mocap_root):
        path.mkdir(parents=True, exist_ok=True)

    recording_id = "demo-recording-001"
    width, height, fps, duration = 640, 360, 30.0, 10.0
    frame_count = int(duration * fps)
    base_ns = 1_800_000_000_000_000_000
    scene_timestamps = base_ns + np.rint(np.arange(frame_count) / fps * 1e9).astype(np.int64)
    native_video = native_root / "Neon Scene Camera v1.mp4"
    _write_demo_video(native_video, width=width, height=height, fps=fps, frames=frame_count)
    scene_timestamps.astype("<i8").tofile(native_root / "Neon Scene Camera v1.time")
    atomic_write_json(native_root / "info.json", {"recording_id": recording_id, "name": "Synthetic OculoStudio demo"})
    atomic_write_json(native_root / "scene_camera.json", {
        "camera_matrix": [[520.0, 0.0, width/2], [0.0, 520.0, height/2], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0, 0, 0, 0, 0], "resolution": [width, height],
    })

    # Cloud scene has two leading placeholder rows; the frame-map test should
    # recover that offset from absolute timestamps rather than literal row index.
    cloud_world = np.concatenate([scene_timestamps[0] - np.asarray([2,1],dtype=np.int64)*int(1e9/fps), scene_timestamps])
    pd.DataFrame({"timestamp [ns]": cloud_world}).to_csv(cloud_root / "world_timestamps.csv", index=False)
    gaze_count = int(duration * 60)
    gaze_t = base_ns + np.rint(np.arange(gaze_count) / 60.0 * 1e9).astype(np.int64)
    gaze_time_sec = np.arange(gaze_count, dtype=float) / 60.0
    gaze_scene_frame = gaze_time_sec * fps
    gaze_face_center_x = width * .43 + 4 * np.sin(gaze_scene_frame / 35.0)
    pd.DataFrame({
        "timestamp [ns]": gaze_t,
        "gaze x [px]": gaze_face_center_x - 16.0 + 1.2 * np.sin(np.arange(gaze_count) / 19.0),
        "gaze y [px]": height * .38 + 1.0 * np.cos(np.arange(gaze_count) / 23.0),
        "worn": np.ones(gaze_count,dtype=int),
        "gaze valid": np.ones(gaze_count,dtype=int),
    }).to_csv(cloud_root / "gaze.csv", index=False)
    pd.DataFrame({
        "fixation id": [1,2,3],
        "start timestamp [ns]": [base_ns+int(.8e9),base_ns+int(3.2e9),base_ns+int(6.7e9)],
        "end timestamp [ns]": [base_ns+int(2.1e9),base_ns+int(5.5e9),base_ns+int(8.9e9)],
        "duration [ms]": [1300,2300,2200],
    }).to_csv(cloud_root / "fixations.csv", index=False)
    pd.DataFrame({"timestamp [ns]": [base_ns+int(2.5e9),base_ns+int(6.1e9)], "duration [ms]": [120,95]}).to_csv(cloud_root / "blinks.csv", index=False)
    atomic_write_json(cloud_root / "info.json", {"recording_id": recording_id, "name": "Synthetic Timeseries"})
    shutil.copy2(native_video, cloud_root / "scene video.mp4")

    face_rows = []
    for index in range(frame_count):
        x = int(width * .43 + 4 * math.sin(index / 35))
        face_rows.append({
            "recording id": recording_id,
            "participant id": "P02",
            "timestamp [ns]": int(scene_timestamps[index]),
            "p1 x [px]": x-56, "p1 y [px]": int(height*.26),
            "p2 x [px]": x+56, "p2 y [px]": int(height*.52),
            "left eye x [px]": x-16, "left eye y [px]": height*.38,
            "right eye x [px]": x+16, "right eye y [px]": height*.38,
            "nose x [px]": x, "nose y [px]": height*.405,
            "left mouth corner x [px]": x-13, "left mouth corner y [px]": height*.43,
            "right mouth corner x [px]": x+13, "right mouth corner y [px]": height*.43,
            "confidence": .99,
        })
    pd.DataFrame(face_rows).to_csv(face_root / "face_detections.csv", index=False)
    pd.DataFrame({
        "recording id": [recording_id],
        "section start time [ns]": [int(scene_timestamps[0])],
        "section end time [ns]": [int(scene_timestamps[-1]+round(1e9/fps))],
    }).to_csv(face_root / "sections.csv", index=False)

    mocap_path = mocap_root / "interaction_tracker.npz"
    marker = _write_mocap(mocap_path, duration_sec=duration)
    body_path = mocap_root / "participant_P02_body.npz"
    body_marker = _write_body_actor_mocap(
        body_path,
        duration_sec=duration,
        width=width,
        height=height,
    )
    densepose_path = assets_root / "DensePose Export" / "densepose_results.csv"
    densepose_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "scene_frame_index": np.arange(frame_count, dtype=int),
        "entity_id": ["P02"] * frame_count,
        "body_part": ["head"] * frame_count,
        "confidence": np.full(frame_count, 0.86),
    }).to_csv(densepose_path, index=False)

    sid = store.project.selected_session_id
    assert sid is not None
    native = add_asset(store, session_id=sid, path=native_root.parent, label="Native Recording", participant_id="P01")
    cloud = add_asset(store, session_id=sid, path=cloud_root.parent, label="Timeseries + Scene Video", participant_id="P01")
    face = add_asset(store, session_id=sid, path=face_root, label="Face Mapper", participant_id="P01")
    tracker = add_asset(store, session_id=sid, path=mocap_path, label="World C3D · tracker proxy", participant_id="P01")
    body = add_asset(store, session_id=sid, path=body_path, label="P02 body mocap", participant_id="P02")
    densepose = add_asset(
        store,
        session_id=sid,
        path=densepose_path,
        label="DensePose export · imported evidence only",
        participant_id="P02",
    )

    # Explicitly connect every independent clock to Global Session Time.
    domain_pairs = [
        (native.primary_time_domain_id, [(base_ns, 0), (base_ns+int(duration*1e9), int(duration*1e9))]),
        (face.primary_time_domain_id, [(base_ns, 0), (base_ns+int(duration*1e9), int(duration*1e9))]),
        (tracker.primary_time_domain_id, [(0, 0), (int(duration*1e9), int(duration*1e9))]),
        (body.primary_time_domain_id, [(0, 0), (int(duration*1e9), int(duration*1e9))]),
    ]
    seen = set()
    for domain_id, anchors in domain_pairs:
        if not domain_id or domain_id in seen:
            continue
        seen.add(domain_id)
        fit_and_store_time_map(
            store, session_id=sid, source_domain_id=domain_id,
            target_domain_id=store.project.sessions[sid].global_time_domain_id,
            anchors=[{"source_ns":a,"target_ns":b,"label":f"Demo {i+1}","weight":1} for i,(a,b) in enumerate(anchors)],
        )

    clip = Clip(label="Clip R02", start_global_ns=int(2e9), end_global_ns=int(7e9), notes="Synthetic regression interval")
    store.update(lambda p: p.sessions[sid].clips.__setitem__(clip.id, clip))

    profile_id = new_id("cal")
    profile_dir = store.derived_dir(sid, "calibrations", profile_id)
    profile_path = profile_dir / "Synthetic_Neon_to_Tracker.oculocal"
    transform = np.eye(4)
    transform[:3, 3] = [26.0, 4.0, 12.0]
    profile = {
        "format": "OculoStudio Calibration Profile", "format_version": "1.0",
        "profile_id": profile_id, "profile_name": "Synthetic Neon to Tracker",
        "participant_id": "P01", "T_tracker_from_camera": transform.tolist(),
        "R_tracker_from_camera": transform[:3,:3].tolist(),
        "camera_origin_tracker_mm": transform[:3,3].tolist(),
        "camera_intrinsics": {"matrix": [[520,0,width/2],[0,520,height/2],[0,0,1]], "distortion": [0,0,0,0,0], "resolution": [width,height]},
        "tracker": {"source_asset_id": tracker.id, "marker_labels": marker["labels"], "marker_template_mm": marker["template"]},
        "board": {"source_asset_id": "synthetic", "marker_labels": [], "definition": {}},
        "synchronization": {"native_time_domain_id": native.primary_time_domain_id, "tracker_time_domain_id": tracker.primary_time_domain_id},
        "sources": {"synthetic": True},
        "qc": {"accepted_count": 120, "direct_reprojection_rms_px_median": .42, "synthetic_reference": True},
        "created_at": utc_now(),
    }
    atomic_write_json(profile_path, profile)
    atomic_write_json(profile_dir / "calibration_qc.json", profile["qc"])
    profile_artifact = ArtifactStore(store).register(
        session_id=sid, kind=ArtifactKind.CALIBRATION_PROFILE, label="Synthetic Neon to Tracker",
        output_root=profile_dir, source_asset_ids=[native.id, tracker.id], qc=profile["qc"], provenance={"synthetic": True},
    )

    results: dict[str, Any] = {}
    if run_processing:
        results["frame_map"] = build_native_cloud_frame_map(store, session_id=sid, native_asset_id=native.id, cloud_asset_id=cloud.id, maximum_error_ms=2.0)
        results["clip"] = run_multimodal_clip(store, session_id=sid, clip_id=clip.id, asset_ids=[native.id, cloud.id, face.id])
        results["privacy"] = create_privacy_video(
            store, session_id=sid, native_asset_id=native.id, facemapper_asset_id=face.id,
            clip_id=clip.id, mode="gaussian", detector_mode="facemapper", coverage_policy="strict",
            preserve_audio=False,
        )
        results["gaze3d"] = generate_gaze_rays(
            store, session_id=sid, timeseries_asset_id=cloud.id, tracker_asset_id=tracker.id,
            profile_artifact_id=profile_artifact.id, scope="full", max_pose_gap_sec=.05,
        )
        gaze_artifact_id = results["gaze3d"]["artifact"]["id"]
        results["semantic_targets"] = create_semantic_gaze_targets(
            store,
            session_id=sid,
            gaze_rays_artifact_id=gaze_artifact_id,
            timeseries_asset_id=cloud.id,
            facemapper_asset_id=face.id,
            body_asset_ids=[body.id],
            entity_marker_prefixes={body.id: body_marker["subject_prefix"]},
            densepose_asset_id=densepose.id,
            scene_frame_tolerance_ms=18.0,
            face_tolerance_ms=20.0,
            uncertainty_cone_deg=1.5,
            region_frame_policy="all_scene_frames",
        )
        results["qc"] = export_qc_report(store, sid)

    return {
        "project": str(store.manifest_path), "session_id": sid,
        "assets": {
            "native": native.id,
            "cloud": cloud.id,
            "facemapper": face.id,
            "tracker": tracker.id,
            "body": body.id,
            "densepose": densepose.id,
        },
        "clip_id": clip.id, "profile_artifact_id": profile_artifact.id,
        "processing_complete": bool(run_processing),
        "artifact_count": len(store.project.sessions[sid].artifacts),
        "semantic_targets": results.get("semantic_targets"),
        "report": results.get("qc",{}).get("html"),
    }
