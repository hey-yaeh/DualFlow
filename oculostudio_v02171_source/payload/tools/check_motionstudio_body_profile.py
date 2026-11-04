"""Validate OculoStudio v0.21.71 raw Vicon full-body 10-finger mapping."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import oculostudio
from oculostudio.services.body_primitives import (
    assess_body_geometry_integrity,
    build_body_primitives,
    detect_marker_protocol,
    intersect_frame_rays_ranked,
    load_body_profile,
)

MARKERS = [
    "LFWT", "LMWT", "LBWT", "RFWT", "RMWT", "RBWT",
    "C7", "T10", "LT10", "RT10", "CLAV", "STRN",
    "ARIEL", "LFHD", "LBHD", "RFHD", "RBHD",
    "LFSH", "LTSH", "LBSH", "LUPA", "LELB", "LBEL", "LFRM", "LIWR", "LOWR", "LOHAND", "LIHAND",
    "RFSH", "RTSH", "RBSH", "RUPA", "RELB", "RBEL", "RFRM", "RIWR", "ROWR", "ROHAND", "RIHAND",
    "LTHI", "LKNE", "LKNU", "LSHN", "LANK", "LHEL", "LMT5", "LMT1", "LTOE",
    "RTHI", "RKNE", "RKNU", "RSHN", "RANK", "RHEL", "RMT5", "RMT1", "RTOE",
]

ANCHORS = {
    "LFWT": (-120, 0, 1000), "LMWT": (-100, 0, 1020), "LBWT": (-110, -40, 1000),
    "RFWT": (120, 0, 1000), "RMWT": (100, 0, 1020), "RBWT": (110, -40, 1000),
    "C7": (0, -40, 1480), "T10": (0, -50, 1250), "LT10": (-70, -45, 1260), "RT10": (70, -45, 1260),
    "CLAV": (0, 20, 1460), "STRN": (0, 40, 1320),
    "ARIEL": (0, 0, 1800), "LFHD": (-70, 35, 1740), "LBHD": (-70, -35, 1740),
    "RFHD": (70, 35, 1740), "RBHD": (70, -35, 1740),
    "LFSH": (-215, 30, 1450), "LTSH": (-220, 0, 1470), "LBSH": (-215, -30, 1450),
    "LUPA": (-330, 0, 1320), "LELB": (-430, 25, 1200), "LBEL": (-430, -25, 1200),
    "LFRM": (-500, 0, 1080), "LIWR": (-550, 20, 950), "LOWR": (-550, -20, 950),
    "LOHAND": (-610, -15, 850), "LIHAND": (-610, 15, 850),
    "RFSH": (215, 30, 1450), "RTSH": (220, 0, 1470), "RBSH": (215, -30, 1450),
    "RUPA": (330, 0, 1320), "RELB": (430, 25, 1200), "RBEL": (430, -25, 1200),
    "RFRM": (500, 0, 1080), "RIWR": (550, 20, 950), "ROWR": (550, -20, 950),
    "ROHAND": (610, -15, 850), "RIHAND": (610, 15, 850),
    "LTHI": (-120, 0, 780), "LKNE": (-120, 20, 550), "LKNU": (-120, -20, 550),
    "LSHN": (-120, 0, 320), "LANK": (-120, 0, 100), "LHEL": (-120, -80, 50),
    "LMT5": (-150, 100, 40), "LMT1": (-90, 110, 40), "LTOE": (-120, 170, 35),
    "RTHI": (120, 0, 780), "RKNE": (120, 20, 550), "RKNU": (120, -20, 550),
    "RSHN": (120, 0, 320), "RANK": (120, 0, 100), "RHEL": (120, -80, 50),
    "RMT5": (150, 100, 40), "RMT1": (90, 110, 40), "RTOE": (120, 170, 35),
}


def make_npz(root: Path, separator: str) -> tuple[Path, str | None]:
    subject = "James"
    labels = [f"{subject}{separator}{marker}" for marker in MARKERS]
    frame_count = 8
    points = np.zeros((frame_count, len(labels), 3), dtype=float)
    for frame in range(frame_count):
        for index, marker in enumerate(MARKERS):
            points[frame, index] = np.asarray(ANCHORS[marker], dtype=float) + [frame * 2.0, 0.0, 0.0]
    path = root / f"production10fingers_{'colon' if separator == ':' else 'underscore'}.npz"
    np.savez(
        path,
        time_sec=np.arange(frame_count, dtype=float) / 120.0,
        labels=np.asarray(labels, dtype=object),
        points_mm=points,
        valid=np.ones((frame_count, len(labels)), dtype=bool),
        frame_rate=120.0,
    )
    return path, subject if separator == ":" else None


def build(path: Path, prefix: str | None):
    frames = np.arange(8, dtype=np.int64)
    return build_body_primitives(
        body_asset_id="body_asset",
        body_path=path,
        entity_id="R01",
        scene_frame_indices=frames,
        scene_timestamps_ns=frames * 8_333_333,
        query_time_sec=frames.astype(float) / 120.0,
        profile=load_body_profile(),
        marker_prefix=prefix,
        max_marker_gap_sec=0.1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    checks: dict[str, bool] = {}
    profile = load_body_profile()
    landmarks = profile.landmarks

    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    checks["profile_names_raw_vicon_fullbody"] = all(token in profile.name.lower() for token in ("vicon", "full-body"))
    checks["pelvis_has_all_six_waist_markers"] = set(
        ["LFWT", "LMWT", "LBWT", "RFWT", "RMWT", "RBWT"]
    ).issubset(set(landmarks["pelvis_center"].markers))
    checks["left_shoulder_cluster_mapped"] = set(["LFSH", "LTSH", "LBSH"]).issubset(
        set(landmarks["left_shoulder"].markers)
    )
    checks["right_shoulder_cluster_mapped"] = set(["RFSH", "RTSH", "RBSH"]).issubset(
        set(landmarks["right_shoulder"].markers)
    )
    checks["wrist_hand_knee_aliases_mapped"] = all(
        marker in {m for row in landmarks.values() for m in row.markers}
        for marker in ["LIWR", "LOWR", "RIWR", "ROWR", "LOHAND", "LIHAND", "ROHAND", "RIHAND", "LKNU", "RKNU"]
    )

    with tempfile.TemporaryDirectory(prefix="oculo_v02151_marker_profile_") as temp:
        root = Path(temp)
        colon_path, colon_prefix = make_npz(root, ":")
        colon = build(colon_path, colon_prefix)
        d = colon.diagnostics
        checks["protocol_detected"] = d.get("marker_protocol", {}).get("name") == "vicon_fullbody_10fingers"
        checks["protocol_confidence_high"] = float(d.get("marker_protocol", {}).get("confidence") or 0.0) >= 0.95
        checks["torso_all_frames"] = d.get("torso_primitive_frame_count") == 8
        checks["pelvis_all_frames"] = d.get("pelvis_primitive_frame_count") == 8
        checks["torso_is_direct_marker_geometry"] = d.get("torso_direct_frame_count") == 8 and d.get("torso_derived_frame_count") == 0
        checks["complete_upper_limbs"] = all(d.get("primitive_region_counts", {}).get(region) == 8 for region in ["left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm"])
        checks["complete_hands"] = all(d.get("primitive_region_counts", {}).get(region) == 8 for region in ["left_hand", "right_hand"])
        checks["waist_markers_recorded"] = len(d.get("landmark_matches", {}).get("pelvis_center") or []) >= 6
        checks["valid_frame_counts_recorded"] = d.get("landmark_valid_frame_counts", {}).get("pelvis_center") == 8
        integrity = assess_body_geometry_integrity([{**d, "entity_relation": "partner"}], expected_relations=["partner"])
        checks["geometry_gate_passes"] = bool(integrity.get("gate_passed") and integrity.get("review_ready"))
        checks["integrity_retains_protocol"] = integrity.get("entities", [{}])[0].get("marker_protocol", {}).get("name") == "vicon_fullbody_10fingers"
        hits = intersect_frame_rays_ranked(
            np.asarray([[0.0, -2000.0, 1200.0]]),
            np.asarray([[0.0, 1.0, 0.0]]),
            colon.primitives_by_frame[0],
        )
        checks["torso_participates_in_exact_ray_intersection"] = bool(hits[0] and hits[0][0].exact_hit and hits[0][0].region == "torso")

        underscore_path, underscore_prefix = make_npz(root, "_")
        underscore = build(underscore_path, underscore_prefix)
        checks["underscore_exporter_prefix_supported"] = underscore.diagnostics.get("torso_primitive_frame_count") == 8

    package_root = Path(oculostudio.__file__).resolve().parent
    body_source = (package_root / "services" / "body_aoi.py").read_text(encoding="utf-8")
    app_source = (package_root / "static" / "app.js").read_text(encoding="utf-8")
    review_source = (package_root / "static" / "aoi_review.js").read_text(encoding="utf-8")
    checks["body_aoi_provenance_is_v6_sample_clock"] = "c3d_articulated_primitives_v6_gaze_sample_clock" in body_source
    checks["gate_error_reports_marker_protocol"] = "Marker protocol:" in body_source
    checks["gaze_targets_shows_marker_protocol"] = "unknown marker profile" in app_source
    checks["body_review_shows_marker_protocol"] = "unknown marker profile" in review_source

    failed = [name for name, value in checks.items() if not value]
    report = {
        "release": "OculoStudio v0.21.71",
        "feature": "Raw Vicon full-body 10-finger marker mapping and torso recovery",
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": failed,
        "ok": not failed,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
