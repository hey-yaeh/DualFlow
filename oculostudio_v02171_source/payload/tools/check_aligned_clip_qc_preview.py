"""Installed-package validation for v0.21.71 aligned clip QC preview."""
from __future__ import annotations

import json
from pathlib import Path

import oculostudio
from oculostudio.services import create_c3d_defined_aligned_clips
from oculostudio.services.pretrim_overlay import render_pretrim_overlay_preview

from check_c3d_defined_clips import _fixture


def main() -> int:
    checks: dict[str, bool] = {}
    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    package = Path(oculostudio.__file__).resolve().parent
    js = (package / "static" / "native_trim.js").read_text("utf-8")
    server = (package / "server.py").read_text("utf-8")
    service = (package / "services" / "pretrim_overlay.py").read_text("utf-8")
    checks.update({
        "right_panel_generate_button": 'id="native-properties-overlay-button"' in js and "Generate QC preview" in js,
        "central_review_button": "native-review-auto-preview" in js and "Review aligned clip video" in js,
        "open_preview_button": 'id="native-properties-open-overlay"' in js,
        "automatic_endpoint_wired": "/api/jobs/aligned-clip-qc-preview" in js and '@app.post("/api/jobs/aligned-clip-qc-preview")' in server,
        "aligned_job_operation": '"aligned_clip_qc_preview"' in server,
        "per_preview_role_history": "native_reference_overlay_by_preview" in js and "native_reference_overlay_by_preview" in service,
        "aligned_overlay_semantics": 'preview_kind = "aligned_clip_qc"' in service and 'preview_label=("ALIGNED CLIP QC"' in service,
        "manual_recovery_preserved": "Manual recovery trim" in js and "/api/jobs/native-trim-overlay-preview" in js,
    })

    temp, store, session_id, handoff, role_inputs = _fixture()
    try:
        clips = create_c3d_defined_aligned_clips(
            store,
            session_id=session_id,
            calibration_handoff_artifact_id=handoff.id,
            role_inputs=role_inputs,
            label_prefix="Aligned QC release check",
            boundary_tolerance_ms=20.0,
        )
        artifact_count_before = len(store.project.sessions[session_id].artifacts)
        previews: dict[str, dict] = {}
        for role in ("R01", "R02"):
            trim_id = clips["roles"][role]["trim_artifact_id"]
            trim = store.project.sessions[session_id].artifacts[trim_id]
            preview_id = str(trim.parameters.get("preview_id") or "")
            checks[f"{role.lower()}_automatic_plan_available"] = bool(preview_id)
            preview = render_pretrim_overlay_preview(
                store,
                session_id=session_id,
                preview_id=preview_id,
                scope="representative",
                segment_seconds=1.0,
                sample_count=3,
                output_fps=30,
                include_face_boxes=True,
                include_landmarks=True,
                include_aoi=True,
                include_gaze=True,
                left_right_policy="anatomical_auto",
            )
            previews[role] = preview
            checks[f"{role.lower()}_preview_kind"] = preview.get("preview_kind") == "aligned_clip_qc"
            checks[f"{role.lower()}_automatic_flag"] = preview.get("automatic_c3d_defined") is True
            checks[f"{role.lower()}_video_exists"] = Path(preview["video_path"]).is_file() and Path(preview["video_path"]).stat().st_size > 0
            checks[f"{role.lower()}_not_artifact"] = preview.get("registered_as_artifact") is False
            checks[f"{role.lower()}_gaze_mapped"] = int(preview.get("gaze_diagnostics", {}).get("mapped", 0)) > 0
        checks["role_preview_ids_are_distinct"] = previews["R01"]["preview_id"] != previews["R02"]["preview_id"]
        checks["role_overlay_ids_are_distinct"] = previews["R01"]["overlay_id"] != previews["R02"]["overlay_id"]
        history = store.project.sessions[session_id].metadata.get("native_reference_overlay_by_preview", {})
        checks["both_roles_restorable"] = all(previews[role]["preview_id"] in history for role in ("R01", "R02"))
        checks["no_scientific_artifact_created"] = len(store.project.sessions[session_id].artifacts) == artifact_count_before
        checks["source_video_not_reencoded"] = all(
            store.project.sessions[session_id].artifacts[clips["roles"][role]["trim_artifact_id"]].provenance.get("source_mp4_reencoded") is False
            for role in ("R01", "R02")
        )
        failed = [name for name, ok in checks.items() if not ok]
        report = {
            "release": "OculoStudio v0.21.71 — Aligned Clip QC Preview",
            "checks": checks,
            "failed": failed,
            "role_previews": {
                role: {
                    "preview_id": previews[role]["preview_id"],
                    "overlay_id": previews[role]["overlay_id"],
                    "video_bytes": Path(previews[role]["video_path"]).stat().st_size,
                    "gaze_mapped": previews[role]["gaze_diagnostics"]["mapped"],
                }
                for role in ("R01", "R02")
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if failed else 0
    finally:
        temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
