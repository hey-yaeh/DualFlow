"""End-to-end validation for OculoStudio v0.21.71 AOI timebase separation."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.demo import create_demo_project
from oculostudio.domain.models import ArtifactKind
from oculostudio.services.body_aoi import create_body_aoi_annotations
from oculostudio.services.face_aoi import create_face_aoi_annotations
from oculostudio.services.gaze3d import generate_gaze_rays
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def _body_end_to_end(root: Path) -> tuple[dict[str, bool], dict]:
    base = create_demo_project(root, run_processing=False)
    store = ProjectStore.open(root)
    sid = base["session_id"]
    assets = base["assets"]
    rays = generate_gaze_rays(
        store,
        session_id=sid,
        timeseries_asset_id=assets["cloud"],
        tracker_asset_id=assets["tracker"],
        profile_artifact_id=base["profile_artifact_id"],
        scope="clip",
        clip_id=base["clip_id"],
        max_pose_gap_sec=0.05,
    )
    result = create_body_aoi_annotations(
        store,
        session_id=sid,
        gaze_rays_artifact_id=rays["artifact"]["id"],
        timeseries_asset_id=assets["cloud"],
        body_asset_ids=[assets["body"]],
        partner_body_asset_id=assets["body"],
        include_self_body=False,
        entity_marker_prefixes={assets["body"]: "P02"},
        native_trim_artifact_id=None,
        sample_clock_chunk_size=512,
        output_parquet=False,
        label="v0.21.71 Body sample-clock end-to-end",
    )
    output_root = Path(result["samples_csv"]).parent
    samples = pd.read_csv(result["samples_csv"], low_memory=False)
    manifest = json.loads((output_root / "body_aoi_manifest.json").read_text("utf-8"))
    timebase = json.loads((output_root / "body_aoi_timebase.json").read_text("utf-8"))
    qc = manifest["qc"]
    body_sources = timebase.get("body_sources") or []
    checks = {
        "body_rows_preserved": len(samples) == 300,
        "body_uses_gaze_sample_clock": bool(
            samples["body_geometry_time_mode"].eq("gaze_sample_timestamp").all()
            and qc.get("body_geometry_clock") == "gaze_sample_timestamp"
        ),
        "body_has_one_query_time_per_gaze_row": bool(
            samples["body_geometry_query_timestamp_ns"].notna().all()
            and samples["body_geometry_query_timestamp_ns"].nunique() == len(samples)
        ),
        "body_video_clock_is_display_only": timebase.get("video_clock_usage") == "display_and_human_review_only",
        "body_timebase_reports_distinct_rates": bool(
            55.0 <= float(timebase.get("estimated_gaze_sample_rate_hz") or 0.0) <= 65.0
            and 29.0 <= float(timebase.get("estimated_scene_video_rate_hz") or 0.0) <= 31.0
            and body_sources
            and 119.0 <= float(body_sources[0].get("source_frame_rate_hz") or 0.0) <= 121.0
        ),
        "body_geometry_gate_is_sample_clock_ready": bool(
            qc.get("body_geometry_gate_passed")
            and qc.get("body_geometry_review_ready")
            and qc.get("body_geometry_fusion_ready")
        ),
        "body_exact_partner_intersection_present": bool(
            samples["assignment_status"].astype(str).eq("body_3d_exact_partner").any()
        ),
        "body_scene_frames_are_preview_metadata": bool(
            "scene_frame_available" in samples.columns
            and "review_preview_available" in samples.columns
            and qc.get("scene_frame_required_for_body_intersection") is False
        ),
    }
    details = {
        "sample_count": int(len(samples)),
        "gaze_rate_hz": timebase.get("estimated_gaze_sample_rate_hz"),
        "video_rate_hz": timebase.get("estimated_scene_video_rate_hz"),
        "body_source_rates_hz": [row.get("source_frame_rate_hz") for row in body_sources],
        "assignment_status_counts": samples["assignment_status"].value_counts(dropna=False).to_dict(),
    }
    return checks, details


def _face_end_to_end(root: Path) -> tuple[dict[str, bool], dict]:
    store = ProjectStore.create(root, "Face gaze-time interpolation validation", session_label="S001")
    sid = store.project.selected_session_id
    artifacts = ArtifactStore(store)
    trim_root = store.derived_dir(sid, "native_reference_trim", "face_clock")
    (trim_root / "tsd").mkdir(parents=True, exist_ok=True)
    (trim_root / "face_mapper").mkdir(parents=True, exist_ok=True)

    frame_ts = [1_000_000_000, 1_033_333_333, 1_066_666_666]
    pd.DataFrame({
        "native_frame_index": [0, 1, 2],
        "source_frame_ordinal_in_window": [0, 1, 2],
        "clip_video_pts_ns": [0, 33_333_333, 66_666_666],
        "native_capture_utc_ns": frame_ts,
    }).to_csv(trim_root / "native_frame_map.csv", index=False)
    pd.DataFrame({
        "native_frame_index": [0, 1, 2],
        "native_capture_utc_ns": frame_ts,
    }).to_csv(trim_root / "native_clock_full.csv", index=False)
    (trim_root / "trim_manifest.json").write_text(json.dumps({
        "recording_id": "rec-1",
        "native_reference": {"width": 640, "height": 480},
    }), "utf-8")

    sample_ts = [1_000_000_000, 1_016_666_666, 1_033_333_333, 1_049_999_999, 1_066_666_666]
    nearest = [0, 0, 1, 1, 2]
    gaze_x = [150.0, 170.0, 190.0, 210.0, 230.0]
    pd.DataFrame([
        {
            "timestamp [ns]": timestamp,
            "gaze x [px]": x,
            "gaze y [px]": 190.0,
            "oculo_nearest_native_frame_index": frame,
            "oculo_source_row_index": index,
            "worn": True,
        }
        for index, (timestamp, x, frame) in enumerate(zip(sample_ts, gaze_x, nearest))
    ]).to_csv(trim_root / "tsd" / "gaze.csv", index=False)

    detections = []
    for timestamp, x1 in zip(frame_ts, [100.0, 140.0, 180.0]):
        detections.append({
            "timestamp [ns]": timestamp,
            "recording id": "rec-1",
            "participant id": "P02",
            "p1 x [px]": x1, "p1 y [px]": 100.0,
            "p2 x [px]": x1 + 100.0, "p2 y [px]": 260.0,
            "left eye x [px]": x1 + 68.0, "left eye y [px]": 145.0,
            "right eye x [px]": x1 + 32.0, "right eye y [px]": 145.0,
            "nose x [px]": x1 + 50.0, "nose y [px]": 190.0,
            "left mouth corner x [px]": x1 + 36.0, "left mouth corner y [px]": 225.0,
            "right mouth corner x [px]": x1 + 64.0, "right mouth corner y [px]": 225.0,
            "confidence": 0.95,
        })
    pd.DataFrame(detections).to_csv(trim_root / "face_mapper" / "face_detections.csv", index=False)
    pd.DataFrame([
        {"timestamp [ns]": timestamp, "gaze on face": True} for timestamp in sample_ts
    ]).to_csv(trim_root / "face_mapper" / "gaze_on_face.csv", index=False)

    trim = artifacts.register(
        session_id=sid,
        kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
        label="Face clock trim",
        output_root=trim_root,
    )
    result = create_face_aoi_annotations(
        store,
        session_id=sid,
        native_trim_artifact_id=trim.id,
        target_person_policy="all_detected",
        face_left_right_policy="as_exported",
        face_geometry_time_policy="track_interpolated",
        render_overlay=False,
        output_parquet=False,
        label="v0.21.71 Face gaze-time end-to-end",
    )
    output_root = Path(result["samples_csv"]).parent
    samples = pd.read_csv(result["samples_csv"], low_memory=False)
    manifest = json.loads((output_root / "face_aoi_manifest.json").read_text("utf-8"))
    qc = manifest["qc"]
    mid = samples.loc[samples["source_gaze_row_index"].isin([1, 3])]
    checks = {
        "face_rows_preserved": len(samples) == 5,
        "face_observations_remain_video_frame_based": bool(samples["face_geometry_observation_clock"].eq("native_video_frames").all()),
        "face_classification_uses_gaze_timestamps": bool(samples["face_geometry_classification_clock"].eq("gaze_sample_timestamps").all()),
        "face_observed_frames_are_not_mislabeled_as_interpolated": bool(
            samples.loc[samples["source_gaze_row_index"].isin([0, 2, 4]), "face_geometry_time_mode"]
            .eq("nearest_frame").all()
        ),
        "face_midframe_samples_are_track_interpolated": bool(mid["face_geometry_time_mode"].eq("track_interpolated").all()),
        "face_interpolation_endpoints_and_alpha_recorded": bool(
            mid["face_geometry_previous_frame_index"].notna().all()
            and mid["face_geometry_next_frame_index"].notna().all()
            and mid["face_geometry_interpolation_alpha"].between(0.49, 0.51).all()
        ),
        "face_interpolation_retains_feature_hits": bool(mid["face_region"].eq("nose").all()),
        "face_manifest_records_policy": bool(
            manifest["parameters"].get("face_geometry_time_policy") == "track_interpolated"
            and qc.get("face_geometry_time_policy") == "track_interpolated"
        ),
        "face_qc_reports_time_modes": bool(
            int((qc.get("face_geometry_time_mode_counts") or {}).get("track_interpolated", 0)) >= 2
            and float(qc.get("track_interpolated_sample_fraction") or 0.0) > 0.0
        ),
    }
    details = {
        "sample_count": int(len(samples)),
        "time_mode_counts": samples["face_geometry_time_mode"].value_counts(dropna=False).to_dict(),
        "interpolated_fraction": qc.get("track_interpolated_sample_fraction"),
        "gaze_rate_hz": qc.get("estimated_gaze_sample_rate_hz"),
        "video_rate_hz": qc.get("estimated_native_video_rate_hz"),
    }
    return checks, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    checks: dict[str, bool] = {"application_version": oculostudio.__version__ == "0.21.71"}
    details: dict = {}
    with tempfile.TemporaryDirectory(prefix="oculo_v02152_e2e_") as temporary:
        base = Path(temporary)
        body_checks, body_details = _body_end_to_end(base / "body_project")
        face_checks, face_details = _face_end_to_end(base / "face_project")
        checks.update(body_checks)
        checks.update(face_checks)
        details["body"] = body_details
        details["face"] = face_details
    failed = [name for name, passed in checks.items() if not bool(passed)]
    report = {
        "release": "OculoStudio v0.21.71 — Gaze-Sample Clock AOI",
        "feature": "End-to-end separation of AOI computation clocks from video display clocks",
        "checks": checks,
        "details": details,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
        "ok": not failed,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", "utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
