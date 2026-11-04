"""Installed-package validation for v0.21.71 formal aligned clip export."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind
from oculostudio.services import create_c3d_defined_aligned_clips, export_aligned_clip_package
from oculostudio.util import sha256_file

from check_c3d_defined_clips import _fixture


def _file(artifact, name: str) -> Path:
    return next(Path(row.path) for row in artifact.files if Path(row.path).name == name)


def main() -> int:
    checks: dict[str, bool] = {}
    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    package = Path(oculostudio.__file__).resolve().parent
    js = (package / "static" / "native_trim.js").read_text("utf-8")
    server = (package / "server.py").read_text("utf-8")
    service = (package / "services" / "aligned_clip_export.py").read_text("utf-8")
    checks.update({
        "right_panel_export_button": 'id="native-export-aligned-package"' in js,
        "central_export_button": "native-export-auto-package" in js and "Export aligned ego + TSD + FM" in js,
        "formal_video_player": 'id="native-aligned-package-player"' in js,
        "three_output_controls": all(token in js for token in ("packageVideo", "packageTsd", "packageFm")),
        "preserve_audio_control": "packageAudio" in js and "Preserve source ego-video audio" in js,
        "api_endpoint_wired": "/api/jobs/aligned-clip-export" in js and '@app.post("/api/jobs/aligned-clip-export")' in server,
        "artifact_kind_wired": "ALIGNED_CLIP_PACKAGE" in service and "aligned_clip_package" in server,
        "no_boundary_reestimation": "immutable_native_reference_trim" in service,
        "byte_identical_data_copy": "byte_identical_from_verified_trim" in service,
    })

    temp, store, session_id, handoff, role_inputs = _fixture()
    try:
        clips = create_c3d_defined_aligned_clips(
            store,
            session_id=session_id,
            calibration_handoff_artifact_id=handoff.id,
            role_inputs=role_inputs,
            label_prefix="Aligned export release check",
            boundary_tolerance_ms=20.0,
        )
        session = store.project.sessions[session_id]
        role_results: dict[str, dict] = {}
        for role in ("R01", "R02"):
            trim_id = clips["roles"][role]["trim_artifact_id"]
            result = export_aligned_clip_package(
                store,
                session_id=session_id,
                native_trim_artifact_id=trim_id,
                label=f"S001 I01 {role} aligned",
                include_ego_video=True,
                include_tsd=True,
                include_facemapper=True,
                output_fps=30,
                preserve_audio=True,
            )
            role_results[role] = result
            artifact = store.project.sessions[session_id].artifacts[result["artifact_id"]]
            trim = store.project.sessions[session_id].artifacts[trim_id]
            checks[f"{role.lower()}_artifact_kind"] = artifact.kind == ArtifactKind.ALIGNED_CLIP_PACKAGE
            checks[f"{role.lower()}_parent_lineage"] = trim_id in artifact.parent_artifact_ids
            checks[f"{role.lower()}_video_exists"] = bool(result["video"]) and Path(result["video"]).is_file() and Path(result["video"]).stat().st_size > 0
            checks[f"{role.lower()}_frame_map_exists"] = bool(result["frame_map"]) and Path(result["frame_map"]).is_file()
            checks[f"{role.lower()}_tsd_exists"] = bool(result["tsd_zip"]) and Path(result["tsd_zip"]).is_file()
            checks[f"{role.lower()}_fm_exists"] = bool(result["facemapper_zip"]) and Path(result["facemapper_zip"]).is_file()
            parent_tsd = _file(trim, "TSD_trimmed.zip")
            parent_fm = _file(trim, "FaceMapper_trimmed.zip")
            checks[f"{role.lower()}_tsd_checksum_preserved"] = sha256_file(result["tsd_zip"]) == sha256_file(parent_tsd)
            checks[f"{role.lower()}_fm_checksum_preserved"] = sha256_file(result["facemapper_zip"]) == sha256_file(parent_fm)
            parent_map = pd.read_csv(_file(trim, "native_frame_map.csv"))
            output_map = pd.read_csv(result["frame_map"])
            checks[f"{role.lower()}_frame_map_monotonic"] = bool(output_map["native_frame_index"].is_monotonic_increasing)
            checks[f"{role.lower()}_frames_within_verified_trim"] = set(output_map["native_frame_index"].astype(int)).issubset(set(parent_map["native_frame_index"].astype(int)))
            checks[f"{role.lower()}_frame_count_matches_video"] = False
            capture = cv2.VideoCapture(str(result["video"]))
            try:
                checks[f"{role.lower()}_frame_count_matches_video"] = (
                    capture.isOpened()
                    and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) == len(output_map)
                )
            finally:
                capture.release()
            checks[f"{role.lower()}_clean_formal_artifact"] = artifact.provenance.get("source_mp4_modified") is False and artifact.qc.get("package_complete") is True
        checks["roles_do_not_share_artifact"] = role_results["R01"]["artifact_id"] != role_results["R02"]["artifact_id"]
        checks["roles_do_not_share_video"] = role_results["R01"]["video"] != role_results["R02"]["video"]

        # Explicitly verify a data-only export does not require or invent a video.
        trim_id = clips["roles"]["R01"]["trim_artifact_id"]
        data_only = export_aligned_clip_package(
            store,
            session_id=session_id,
            native_trim_artifact_id=trim_id,
            label="R01 aligned data only",
            include_ego_video=False,
            include_tsd=True,
            include_facemapper=True,
            output_fps=30,
            preserve_audio=False,
        )
        checks["data_only_supported"] = data_only["video"] is None and Path(data_only["tsd_zip"]).is_file() and Path(data_only["facemapper_zip"]).is_file()

        failed = [name for name, ok in checks.items() if not ok]
        report = {
            "release": "OculoStudio v0.21.71 — Aligned Ego + TSD + FaceMapper Export",
            "checks": checks,
            "failed": failed,
            "role_outputs": {
                role: {
                    "artifact_id": row["artifact_id"],
                    "video_bytes": Path(row["video"]).stat().st_size,
                    "frame_rows": len(pd.read_csv(row["frame_map"])),
                    "tsd_sha256": sha256_file(row["tsd_zip"]),
                    "facemapper_sha256": sha256_file(row["facemapper_zip"]),
                }
                for role, row in role_results.items()
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if failed else 0
    finally:
        temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
