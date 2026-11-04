"""Validation for OculoStudio v0.21.71 gaze-sample-clock AOI processing."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import oculostudio
from oculostudio.adapters.mocap import read_mocap
from oculostudio.server import BodyAOIJobRequest, FaceAOIJobRequest
from oculostudio.services.body_aoi import create_body_aoi_annotations
from oculostudio.services.body_primitives import (
    BodyPrimitive,
    _paired_primitive_candidates,
    _primitive_candidates,
    body_geometry_integrity_from_qc,
    build_body_primitives,
    intersect_frame_rays_ranked,
    intersect_sample_rays_ranked,
    load_body_profile,
)
from oculostudio.services.face_aoi import (
    FACE_AOI_ENGINE,
    _face_track_index,
    _interpolate_face_instances_for_sample,
)
from oculostudio.services.face_semantics import FaceInstance, Point2D


def _synthetic_raw_vicon_npz(root: Path, frame_count: int = 240) -> Path:
    labels = [
        "James:C7", "James:T10", "James:LT10", "James:RT10", "James:CLAV", "James:STRN",
        "James:LFWT", "James:LMWT", "James:LBWT", "James:RFWT", "James:RMWT", "James:RBWT",
        "James:LFSH", "James:LTSH", "James:LBSH", "James:RFSH", "James:RTSH", "James:RBSH",
        "James:LFHD", "James:RFHD", "James:LBHD", "James:RBHD",
        "James:LELB", "James:LBEL", "James:RELB", "James:RBEL",
        "James:LIWR", "James:LOWR", "James:RIWR", "James:ROWR",
        "James:LIHAND", "James:LOHAND", "James:RIHAND", "James:ROHAND",
        "James:LTHI", "James:LKNE", "James:LKNU", "James:LSHN", "James:LANK", "James:LHEL", "James:LTOE",
        "James:RTHI", "James:RKNE", "James:RKNU", "James:RSHN", "James:RANK", "James:RHEL", "James:RTOE",
    ]
    anchors = {
        "C7": (0, 35, 1540), "T10": (0, 20, 1220), "LT10": (-90, 15, 1240), "RT10": (90, 15, 1240),
        "CLAV": (0, -15, 1490), "STRN": (0, -45, 1320),
        "LFWT": (-135, 0, 980), "LMWT": (-145, 10, 925), "LBWT": (-125, 35, 940),
        "RFWT": (135, 0, 980), "RMWT": (145, 10, 925), "RBWT": (125, 35, 940),
        "LFSH": (-210, -10, 1460), "LTSH": (-225, 10, 1450), "LBSH": (-205, 35, 1435),
        "RFSH": (210, -10, 1460), "RTSH": (225, 10, 1450), "RBSH": (205, 35, 1435),
        "LFHD": (-70, -15, 1740), "RFHD": (70, -15, 1740), "LBHD": (-65, 40, 1720), "RBHD": (65, 40, 1720),
        "LELB": (-420, 0, 1240), "LBEL": (-405, 25, 1230), "RELB": (420, 0, 1240), "RBEL": (405, 25, 1230),
        "LIWR": (-505, -5, 1010), "LOWR": (-520, 20, 1000), "RIWR": (505, -5, 1010), "ROWR": (520, 20, 1000),
        "LIHAND": (-535, -15, 950), "LOHAND": (-545, 25, 945), "RIHAND": (535, -15, 950), "ROHAND": (545, 25, 945),
        "LTHI": (-135, 5, 850), "LKNE": (-140, 0, 520), "LKNU": (-145, 25, 520), "LSHN": (-140, 5, 300), "LANK": (-140, 0, 105), "LHEL": (-140, 55, 55), "LTOE": (-140, -125, 35),
        "RTHI": (135, 5, 850), "RKNE": (140, 0, 520), "RKNU": (145, 25, 520), "RSHN": (140, 5, 300), "RANK": (140, 0, 105), "RHEL": (140, 55, 55), "RTOE": (140, -125, 35),
    }
    times = np.arange(frame_count, dtype=float) / 120.0
    points = np.zeros((frame_count, len(labels), 3), dtype=float)
    for frame, t in enumerate(times):
        displacement = np.array([180.0 * t, 22.0 * math.sin(2.0 * math.pi * t), 0.0])
        for marker, label in enumerate(labels):
            points[frame, marker] = np.asarray(anchors[label.split(":")[-1]], dtype=float) + displacement
    path = root / "raw_vicon_fullbody_10fingers.npz"
    np.savez(
        path,
        time_sec=times,
        labels=np.asarray(labels, dtype=object),
        points_mm=points,
        valid=np.ones((frame_count, len(labels)), dtype=bool),
        frame_rate=120.0,
    )
    return path


def _face(frame: int, timestamp: int, source: int, x1: float, *, entity: str = "P02") -> FaceInstance:
    return FaceInstance(
        scene_frame_index=frame,
        scene_timestamp_ns=timestamp,
        track_id=7,
        explicit_entity_id=entity,
        x1=x1,
        y1=100.0,
        x2=x1 + 100.0,
        y2=220.0,
        left_eye=Point2D(x1 + 68.0, 140.0),
        right_eye=Point2D(x1 + 32.0, 140.0),
        nose=Point2D(x1 + 50.0, 166.0),
        mouth_left=Point2D(x1 + 36.0, 192.0),
        mouth_right=Point2D(x1 + 64.0, 192.0),
        confidence=0.92,
        source_row_index=source,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--real-c3d", default="")
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["application_version"] = oculostudio.__version__ == "0.21.71"
    checks["face_engine_version"] = FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"

    body_request = BodyAOIJobRequest(
        session_id="S001", gaze_rays_artifact_id="rays", body_asset_ids=["body"]
    )
    face_request = FaceAOIJobRequest(session_id="S001", native_trim_artifact_id="trim")
    checks["body_api_sample_clock_chunk_default"] = body_request.sample_clock_chunk_size == 4096
    checks["face_api_track_interpolation_default"] = face_request.face_geometry_time_policy == "track_interpolated"
    checks["face_api_interpolation_limits"] = (
        face_request.face_interpolation_max_gap_ms == 75.0
        and face_request.face_interpolation_max_motion_ratio == 1.25
    )

    rng = np.random.default_rng(20261101)
    count = 1500
    origins = rng.normal(size=(count, 3)) * 100.0
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    primitives: list[BodyPrimitive] = []
    for index in range(count):
        start = rng.normal(size=3) * 450.0 + np.array([0.0, 0.0, 1200.0])
        shape = "sphere" if index % 9 == 0 else "capsule"
        end = start.copy() if shape == "sphere" else start + rng.normal(size=3) * 220.0
        primitives.append(BodyPrimitive(
            scene_frame_index=index,
            scene_timestamp_ns=index * 5_000_000,
            entity_id="P02",
            source_asset_id="partner_body",
            region="torso",
            shape=shape,
            radius_mm=float(rng.uniform(25.0, 190.0)),
            priority=100,
            confidence=0.9,
            p0_mm=start,
            p1_mm=end,
        ))
    paired = _paired_primitive_candidates(origins, directions, primitives)
    maximum_errors = [0.0, 0.0, 0.0]
    parity = True
    for index, primitive in enumerate(primitives):
        legacy = _primitive_candidates(origins[index:index+1], directions[index:index+1], primitive)
        parity &= bool(legacy[0][0]) == bool(paired[0][index])
        maximum_errors[0] = max(maximum_errors[0], abs(float(legacy[1][0]) - float(paired[1][index])))
        maximum_errors[1] = max(maximum_errors[1], abs(float(legacy[2][0]) - float(paired[2][index])))
        maximum_errors[2] = max(maximum_errors[2], float(np.linalg.norm(legacy[3][0] - paired[3][index])))
    checks["paired_intersection_matches_legacy_math"] = parity and max(maximum_errors) < 1e-7
    details["paired_intersection_max_errors"] = {
        "distance_mm": maximum_errors[0], "angular_deg": maximum_errors[1], "point_mm": maximum_errors[2]
    }

    moving = [
        BodyPrimitive(0, 0, "P02", "partner", "torso", "sphere", 80.0, 100, 1.0, np.array([0.0, 0.0, 1000.0]), np.array([0.0, 0.0, 1000.0])),
        BodyPrimitive(1, 5_000_000, "P02", "partner", "torso", "sphere", 80.0, 100, 1.0, np.array([250.0, 0.0, 1000.0]), np.array([250.0, 0.0, 1000.0])),
    ]
    dynamic_origins = np.array([[0.0, 0.0, 0.0], [250.0, 0.0, 0.0]])
    dynamic_directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    sample_hits = intersect_sample_rays_ranked(dynamic_origins, dynamic_directions, [[moving[0]], [moving[1]]])
    legacy_reuse = intersect_frame_rays_ranked(dynamic_origins, dynamic_directions, [moving[0]])
    checks["sample_clock_tracks_moving_body"] = (
        all(rows and rows[0].exact_hit for rows in sample_hits)
        and bool(legacy_reuse[0])
        and not bool(legacy_reuse[1])
    )

    with tempfile.TemporaryDirectory(prefix="oculo_v02152_") as temporary:
        root = Path(temporary)
        body_path = _synthetic_raw_vicon_npz(root)
        mocap = read_mocap(body_path)
        query = np.arange(float(mocap.time_sec[0]), float(mocap.time_sec[-1]) + 1e-12, 1.0 / 200.0)
        query = query[query <= mocap.time_sec[-1]]
        query_indices = np.arange(len(query), dtype=np.int64)
        started = time.perf_counter()
        preflight = build_body_primitives(
            body_asset_id="raw_body",
            body_path=body_path,
            entity_id="R01",
            scene_frame_indices=query_indices,
            scene_timestamps_ns=np.round(query * 1e9).astype(np.int64),
            query_time_sec=query,
            profile=load_body_profile(),
            marker_prefix="James",
            max_marker_gap_sec=0.1,
            mocap_data=mocap,
            diagnostics_only=True,
            include_frame_regions=False,
        )
        preflight_seconds = time.perf_counter() - started
        diagnostic = preflight.diagnostics
        checks["synthetic_120hz_c3d_preflighted_at_200hz"] = (
            diagnostic.get("source_frame_rate_hz") == 120.0
            and diagnostic.get("scene_frame_count") == len(query)
        )
        checks["raw_vicon_marker_protocol_named_correctly"] = (
            diagnostic.get("marker_protocol", {}).get("name") == "vicon_fullbody_10fingers"
        )
        checks["sample_clock_torso_coverage_complete"] = (
            diagnostic.get("torso_primitive_frame_count") == len(query)
            and diagnostic.get("pelvis_primitive_frame_count") == len(query)
            and diagnostic.get("torso_direct_frame_count") == len(query)
        )
        checks["sample_clock_interpolation_error_bounded_by_120hz"] = (
            float(diagnostic.get("p95_query_to_source_sample_error_ms") or 999.0) <= 4.2
        )
        details["synthetic_preflight"] = {
            "query_count": len(query),
            "source_rate_hz": diagnostic.get("source_frame_rate_hz"),
            "p95_query_to_source_sample_error_ms": diagnostic.get("p95_query_to_source_sample_error_ms"),
            "seconds": preflight_seconds,
        }

    valid_qc = {
        "include_self_body": False,
        "body_geometry_clock": "gaze_sample_timestamp",
        "body_diagnostics": [{
            "entity_id": "R02", "entity_relation": "partner", "scene_frame_count": 100,
            "frames_with_any_primitive": 100, "torso_primitive_frame_count": 100,
            "pelvis_primitive_frame_count": 100, "torso_direct_frame_count": 100,
            "torso_derived_frame_count": 0,
        }],
    }
    valid_report = body_geometry_integrity_from_qc(valid_qc)
    legacy_report = body_geometry_integrity_from_qc({k: v for k, v in valid_qc.items() if k != "body_geometry_clock"})
    checks["sample_clock_body_artifact_review_ready"] = bool(valid_report.get("review_ready"))
    checks["legacy_video_clock_body_artifact_blocked"] = (
        not bool(legacy_report.get("review_ready"))
        and any(row.get("code") == "legacy_video_frame_body_geometry_clock" for row in legacy_report.get("failures", []))
    )

    first = _face(10, 1_000_000_000, 100, 100.0)
    second = _face(11, 1_033_333_333, 101, 120.0)
    by_frame = {10: [first], 11: [second]}
    track_index = _face_track_index(by_frame)
    matched = "matched_participant"
    identity = {
        (10, 100): {"identity_status": matched, "matched_entity_id": "P02", "target_relation": "partner"},
        (11, 101): {"identity_status": matched, "matched_entity_id": "P02", "target_relation": "partner"},
    }
    middle = 1_016_666_666
    estimated, metadata = _interpolate_face_instances_for_sample(
        timestamp_ns=middle, native_frame_index=10, instances_by_frame=by_frame,
        track_index=track_index, identity_by_key=identity, target_person_policy="partner_only",
        temporal_policy="track_interpolated", maximum_gap_ms=75.0, maximum_motion_ratio=1.25,
    )
    checks["safe_face_track_interpolates_at_gaze_time"] = (
        len(estimated) == 1
        and metadata.get(100, {}).get("mode") == "track_interpolated"
        and abs(estimated[0].x1 - 110.0) < 1e-3
    )
    identity_changed = {**identity, (11, 101): {"identity_status": matched, "matched_entity_id": "P03", "target_relation": "partner"}}
    _, changed_meta = _interpolate_face_instances_for_sample(
        timestamp_ns=middle, native_frame_index=10, instances_by_frame=by_frame,
        track_index=track_index, identity_by_key=identity_changed, target_person_policy="partner_only",
        temporal_policy="track_interpolated", maximum_gap_ms=75.0, maximum_motion_ratio=1.25,
    )
    checks["face_identity_change_falls_back"] = changed_meta.get(100, {}).get("reason") == "identity_not_stable"
    _, nearest_meta = _interpolate_face_instances_for_sample(
        timestamp_ns=middle, native_frame_index=10, instances_by_frame=by_frame,
        track_index=track_index, identity_by_key=identity, target_person_policy="partner_only",
        temporal_policy="nearest_frame", maximum_gap_ms=75.0, maximum_motion_ratio=1.25,
    )
    checks["face_nearest_frame_legacy_policy_preserved"] = nearest_meta.get(100, {}).get("reason") == "policy_nearest_frame"

    package_root = Path(oculostudio.__file__).resolve().parent
    body_source = (package_root / "services" / "body_aoi.py").read_text("utf-8")
    face_source = (package_root / "services" / "face_aoi.py").read_text("utf-8")
    fusion_source = (package_root / "services" / "target_fusion.py").read_text("utf-8")
    app_source = (package_root / "static" / "app.js").read_text("utf-8")
    index_source = (package_root / "static" / "index.html").read_text("utf-8")
    create_body_source = inspect.getsource(create_body_aoi_annotations)
    checks["body_intersection_no_longer_uses_video_frame_batches"] = "intersect_frame_rays_ranked(" not in create_body_source
    checks["body_missing_scene_frame_no_longer_blocks_hit"] = 'status = "no_scene_frame"' not in create_body_source
    checks["body_timebase_export_packaged"] = all(token in body_source for token in (
        "body_aoi_timebase.json", "gaze_sample_timestamp", "display_and_human_review_only"
    ))
    checks["face_temporal_provenance_packaged"] = all(token in face_source for token in (
        "face_geometry_previous_frame_index", "face_geometry_interpolation_alpha", "identity_not_stable"
    ))
    checks["fusion_preserves_temporal_fields"] = all(token in fusion_source for token in (
        "body_geometry_time_mode", "face_geometry_time_mode", "gaze_sample_timestamp"
    ))
    checks["ui_exposes_face_time_model"] = all(token in app_source for token in (
        "face-aoi-time-policy", "Track-aware gaze-time interpolation", "face_interpolation_max_gap_ms"
    ))
    checks["ui_explains_body_sample_clock"] = all(token in app_source for token in (
        "One body geometry query per gaze timestamp", "video frames are used only for review display"
    ))
    checks["cache_busting_version"] = "app.js?v=0.21.71" in index_source and "aoi_review.js?v=0.21.71" in index_source

    if args.real_c3d:
        real_path = Path(args.real_c3d)
        if real_path.exists():
            started = time.perf_counter()
            data = read_mocap(real_path)
            query = np.linspace(float(data.time_sec[0]), float(data.time_sec[-1]), 112_669)
            real = build_body_primitives(
                body_asset_id="real_R01", body_path=real_path, entity_id="R01",
                scene_frame_indices=np.arange(len(query), dtype=np.int64),
                scene_timestamps_ns=np.round(query * 1e9).astype(np.int64),
                query_time_sec=query, profile=load_body_profile(), max_marker_gap_sec=0.1,
                mocap_data=data, diagnostics_only=True, include_frame_regions=False,
            ).diagnostics
            elapsed = time.perf_counter() - started
            checks["real_r01_raw_c3d_120hz"] = math.isclose(float(data.frame_rate), 120.0, abs_tol=1e-9)
            checks["real_r01_112669_sample_torso_coverage"] = (
                real.get("torso_primitive_frame_count") == len(query)
                and real.get("pelvis_primitive_frame_count") == len(query)
            )
            checks["real_r01_protocol_detected"] = real.get("marker_protocol", {}).get("name") == "vicon_fullbody_10fingers"
            details["real_r01"] = {
                "path": str(real_path), "c3d_frames": len(data.time_sec), "c3d_rate_hz": data.frame_rate,
                "query_count": len(query), "torso_count": real.get("torso_primitive_frame_count"),
                "pelvis_count": real.get("pelvis_primitive_frame_count"),
                "p95_query_error_ms": real.get("p95_query_to_source_sample_error_ms"),
                "seconds_including_read": elapsed,
            }
        else:
            checks["real_r01_file_available"] = False

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "release": "OculoStudio v0.21.71 — Gaze-Sample Clock AOI",
        "ok": not failed,
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed": failed,
        "checks": checks,
        "details": details,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
