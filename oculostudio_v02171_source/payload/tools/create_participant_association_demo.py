from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from oculostudio.demo import create_demo_project
from oculostudio.domain.models import ArtifactKind, Clip
from oculostudio.services.aoi_review import render_aoi_review_video
from oculostudio.services.body_aoi import create_body_aoi_annotations
from oculostudio.services.face_aoi import create_face_aoi_annotations
from oculostudio.services.gaze3d import generate_gaze_rays
from oculostudio.services.native_reference_trim import export_native_reference_trim, preview_native_reference_trim
from oculostudio.services.participant_association import create_participant_association
from oculostudio.services.privacy import create_privacy_video
from oculostudio.services.privacy_clip import create_trimmed_privacy_video
from oculostudio.services.target_fusion import fuse_face_and_body_aoi
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json


def _assignment(store: ProjectStore, sid: str, base: dict, output: Path):
    assets = base["assets"]
    root = output / "derived" / sid / "etm_assignment" / "clip" / "participant_demo_assignment"
    root.mkdir(parents=True, exist_ok=True)
    head = ["P02:LFHD", "P02:RFHD", "P02:LBHD", "P02:RBHD"]
    payload = {
        "schema_version": "oculostudio-etm-role-assignment-1.0",
        "status": "resolved",
        "assignments": {
            "R01": {
                "role_id": "R01", "body_asset_id": "unused-ego-body",
                "body_head_marker_labels": head, "etm_asset_id": assets["tracker"],
                "etm_id": "ETM_A", "tracker_prefix": "tracker",
                "recommended_calibration_artifact_id": base["profile_artifact_id"],
                "confidence": "HIGH",
            },
            "R02": {
                "role_id": "R02", "body_asset_id": assets["body"],
                "body_subject_prefix": "P02", "body_head_marker_labels": head,
                "etm_asset_id": "unused-partner-etm", "etm_id": "ETM_B",
                "tracker_prefix": "tracker_2", "confidence": "HIGH",
            },
        },
    }
    atomic_write_json(root / "etm_role_assignment.json", payload)
    return ArtifactStore(store).register(
        session_id=sid, kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
        label="S001_I01 ETM role assignment", output_root=root,
        source_asset_ids=[assets["tracker"], assets["body"]],
        qc={"status": "resolved"}, provenance={"synthetic": True},
    )


def build(output: Path, *, overwrite: bool = False) -> dict:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    base = create_demo_project(output, run_processing=False)
    store = ProjectStore.open(output)
    sid = base["session_id"]
    assets = base["assets"]

    # Add an untracked staff face so the identity layer visibly distinguishes it.
    face_asset = store.project.sessions[sid].assets[assets["facemapper"]]
    face_csv = Path(face_asset.source_path) / "face_detections.csv"
    faces = pd.read_csv(face_csv)
    staff = faces.copy()
    staff["participant id"] = "staff"
    for column in [c for c in staff.columns if "x [px]" in c]:
        staff[column] = pd.to_numeric(staff[column], errors="coerce") + 220
    staff["p1 x [px]"] = staff["p1 x [px]"].clip(0, 620)
    staff["p2 x [px]"] = staff["p2 x [px]"].clip(0, 638)
    pd.concat([faces, staff], ignore_index=True).to_csv(face_csv, index=False)

    preview = preview_native_reference_trim(
        store, session_id=sid, native_asset_id=assets["native"],
        timeseries_asset_id=assets["cloud"], facemapper_asset_id=assets["facemapper"],
        start_timecode="00:00:02:00", duration_timecode="00:00:05:00",
        timecode_fps=30, label="S001_I01_R01", face_tolerance_ms=8.0,
    )
    trim = export_native_reference_trim(store, session_id=sid, preview_id=preview["preview_id"])
    privacy_full = create_privacy_video(
        store, session_id=sid, native_asset_id=assets["native"],
        facemapper_asset_id=assets["facemapper"], mode="gaussian",
        detector_mode="facemapper", coverage_policy="allow", preserve_audio=False,
    )
    privacy_clip = create_trimmed_privacy_video(
        store, session_id=sid, privacy_artifact_id=privacy_full["artifact"]["id"],
        native_trim_artifact_id=trim["artifact_id"], output_fps=30,
        label="S001_I01_R01_privacy_trimmed", preserve_audio=False,
    )
    assignment = _assignment(store, sid, base, output)
    association = create_participant_association(
        store, session_id=sid, native_trim_artifact_id=trim["artifact_id"],
        etm_assignment_artifact_id=assignment.id, wearer_role_id="R01",
        calibration_profile_artifact_id=base["profile_artifact_id"],
        target_policy="partner_only", minimum_identity_confidence=.25,
        label="S001_I01_R01 participant association",
    )
    face = create_face_aoi_annotations(
        store, session_id=sid, native_trim_artifact_id=trim["artifact_id"],
        participant_association_artifact_id=association["artifact"]["id"],
        target_person_policy="partner_only",
        privacy_clip_artifact_id=privacy_clip["artifact"]["id"],
        label="S001_I01_R01 partner Face AOI", render_overlay=True,
        preserve_audio=False,
        face_left_right_policy="as_exported",
    )
    analysis_clip = Clip(label="S001_I01_R01_3D", start_global_ns=2_000_000_000, end_global_ns=7_000_000_000)
    store.update(lambda p: p.sessions[sid].clips.__setitem__(analysis_clip.id, analysis_clip))
    rays = generate_gaze_rays(
        store, session_id=sid, timeseries_asset_id=assets["cloud"],
        tracker_asset_id=assets["tracker"], profile_artifact_id=base["profile_artifact_id"],
        scope="clip", clip_id=analysis_clip.id, max_pose_gap_sec=.05,
    )
    body = create_body_aoi_annotations(
        store, session_id=sid, gaze_rays_artifact_id=rays["artifact"]["id"],
        timeseries_asset_id=assets["cloud"], body_asset_ids=[assets["body"]],
        entity_marker_prefixes={assets["body"]: "P02"}, scene_frame_tolerance_ms=20.0,
        native_trim_artifact_id=trim["artifact_id"], label="S001_I01_R01_body_aoi",
        include_self_body=False,
    )
    fusion = fuse_face_and_body_aoi(
        store, session_id=sid, face_aoi_artifact_id=face["artifact"]["id"],
        body_aoi_artifact_id=body["artifact"]["id"], label="S001_I01_R01_gaze_targets_full",
    )
    review = render_aoi_review_video(
        store, session_id=sid, mode="face", artifact_id=face["artifact"]["id"],
        video_artifact_id=privacy_clip["artifact"]["id"],
        label="S001_I01_R01_partner_identity_review", preserve_audio=False,
    )
    return {**base, "native_trim": trim, "privacy_clip": privacy_clip,
            "etm_assignment": assignment.model_dump(mode="json"),
            "participant_association": association, "face_aoi": face,
            "body_aoi": body, "fusion": fusion, "review_video": review}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.output).resolve(), overwrite=args.overwrite), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
