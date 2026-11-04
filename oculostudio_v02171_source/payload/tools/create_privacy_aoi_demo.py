from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from oculostudio.demo import create_demo_project
from oculostudio.domain.models import Clip
from oculostudio.services.aoi_review import render_aoi_review_video
from oculostudio.services.body_aoi import create_body_aoi_annotations
from oculostudio.services.face_aoi import create_face_aoi_annotations
from oculostudio.services.gaze3d import generate_gaze_rays
from oculostudio.services.native_reference_trim import export_native_reference_trim, preview_native_reference_trim
from oculostudio.services.privacy import create_privacy_video
from oculostudio.services.privacy_clip import create_trimmed_privacy_video
from oculostudio.services.target_fusion import fuse_face_and_body_aoi
from oculostudio.storage.project_store import ProjectStore


def build(output: Path, *, overwrite: bool = False) -> dict:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    base = create_demo_project(output, run_processing=False)
    store = ProjectStore.open(output)
    sid = base["session_id"]
    assets = base["assets"]

    preview = preview_native_reference_trim(
        store,
        session_id=sid,
        native_asset_id=assets["native"],
        timeseries_asset_id=assets["cloud"],
        facemapper_asset_id=assets["facemapper"],
        start_timecode="00:00:02:00",
        duration_timecode="00:00:05:00",
        timecode_fps=30,
        label="S001_I01_R01",
        face_tolerance_ms=8.0,
    )
    trim = export_native_reference_trim(store, session_id=sid, preview_id=preview["preview_id"])
    privacy_full = create_privacy_video(
        store,
        session_id=sid,
        native_asset_id=assets["native"],
        facemapper_asset_id=assets["facemapper"],
        mode="gaussian",
        detector_mode="facemapper",
        coverage_policy="strict",
        preserve_audio=False,
    )
    privacy_clip = create_trimmed_privacy_video(
        store,
        session_id=sid,
        privacy_artifact_id=privacy_full["artifact"]["id"],
        native_trim_artifact_id=trim["artifact_id"],
        output_fps=30,
        label="S001_I01_R01_privacy_trimmed",
        preserve_audio=False,
    )
    face = create_face_aoi_annotations(
        store,
        session_id=sid,
        native_trim_artifact_id=trim["artifact_id"],
        privacy_clip_artifact_id=privacy_clip["artifact"]["id"],
        label="S001_I01_R01_face_aoi",
        render_overlay=True,
        preserve_audio=False,
        face_left_right_policy="as_exported",
    )
    analysis_clip = Clip(label="S001_I01_R01_3D", start_global_ns=2_000_000_000, end_global_ns=7_000_000_000)
    store.update(lambda p: p.sessions[sid].clips.__setitem__(analysis_clip.id, analysis_clip))
    rays = generate_gaze_rays(
        store,
        session_id=sid,
        timeseries_asset_id=assets["cloud"],
        tracker_asset_id=assets["tracker"],
        profile_artifact_id=base["profile_artifact_id"],
        scope="clip",
        clip_id=analysis_clip.id,
        max_pose_gap_sec=0.05,
    )
    body = create_body_aoi_annotations(
        store,
        session_id=sid,
        gaze_rays_artifact_id=rays["artifact"]["id"],
        timeseries_asset_id=assets["cloud"],
        body_asset_ids=[assets["body"]],
        partner_body_asset_id=assets["body"],
        include_self_body=False,
        entity_marker_prefixes={assets["body"]: "P02"},
        scene_frame_tolerance_ms=20.0,
        native_trim_artifact_id=trim["artifact_id"],
        label="S001_I01_R01_body_aoi",
    )
    fusion = fuse_face_and_body_aoi(
        store,
        session_id=sid,
        face_aoi_artifact_id=face["artifact"]["id"],
        body_aoi_artifact_id=body["artifact"]["id"],
        label="S001_I01_R01_gaze_targets_full",
    )
    review_video = render_aoi_review_video(
        store,
        session_id=sid,
        mode="fusion",
        artifact_id=fusion["artifact"]["id"],
        video_artifact_id=privacy_clip["artifact"]["id"],
        label="S001_I01_R01_fused_review",
        preserve_audio=False,
    )
    return {
        **base,
        "native_trim": trim,
        "privacy_full": privacy_full,
        "privacy_clip": privacy_clip,
        "face_aoi": face,
        "body_aoi": body,
        "fusion": fusion,
        "review_video": review_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(Path(args.output).resolve(), overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
