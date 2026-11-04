"""Installed-package validation for formal aligned facial-expression video export."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import oculostudio
from oculostudio.adapters.native_reference_clock import build_clock, source_stamp
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ValidationError
from oculostudio.services import (
    accept_audio_video_sync,
    add_asset,
    automatic_shared_audio_video_sync,
    export_aligned_facial_video,
)
from oculostudio.storage import ProjectStore
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.util import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "VALIDATION_v0_21_63_ALIGNED_FACIAL_VIDEO.json"


def _load_shared_fixture_module():
    path = Path(__file__).resolve().parent / "check_shared_audio_video_sync.py"
    spec = importlib.util.spec_from_file_location("shared_audio_fixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _register_trim(store: ProjectStore, session_id: str, native, *, video_member: str, start_sec: float, end_sec: float):
    clock = build_clock(native.source_path, store.cache_dir(), video_member=video_member)
    start_ns, end_ns = int(round(start_sec * 1e9)), int(round(end_sec * 1e9))
    frame_ends = np.r_[clock.media_ns[1:], clock.end_media_ns]
    selected = np.flatnonzero((clock.media_ns < end_ns) & (frame_ends > start_ns))
    artifact_id = "artifact_trim_facial_export"
    output = store.derived_dir(session_id, "native_reference_trim") / artifact_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({
        "native_frame_index": selected.astype(int),
        "native_video_pts_ns": clock.media_ns[selected].astype(np.int64),
        "native_frame_end_pts_ns": frame_ends[selected].astype(np.int64),
        "native_capture_utc_ns": clock.capture_ns[selected].astype(np.int64),
    }).to_csv(output / "native_frame_map.csv", index=False)
    for name in ("TSD_trimmed.zip", "FaceMapper_trimmed.zip"):
        with zipfile.ZipFile(output / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("README.txt", "synthetic aligned subset")
    request = {
        "native_asset_id": native.id,
        "timeseries_asset_id": native.id,
        "facemapper_asset_id": native.id,
        "native_video_member": video_member,
        "wearer_role_id": "R01",
        "interaction_id": "I01",
        "label": "S001 I01 R01 automatic aligned clip",
        "automatic_c3d_defined": True,
        "window_source": "common_body_c3d_overlap",
    }
    manifest = {
        "schema_version": "synthetic-trim-1.0",
        "recording_id": native.recording_id,
        "request": request,
        "window": {"start_media_ns": str(start_ns), "end_media_ns": str(end_ns)},
        "source_stamps": {"native": source_stamp(native.source_path)},
    }
    atomic_write_json(output / "trim_manifest.json", manifest)
    (output / "trim_report.html").write_text("<html>synthetic verified trim</html>", "utf-8")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
        label=request["label"],
        output_root=output,
        source_asset_ids=[native.id],
        parameters={**request, "window": manifest["window"]},
        qc={"common_duration_sec": end_sec - start_sec},
        artifact_id=artifact_id,
    )
    return artifact


def _probe(path: Path) -> dict:
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def main() -> int:
    checks: dict[str, bool] = {}
    report: dict = {
        "schema_version": "oculostudio-release-validation-1.0",
        "release": "OculoStudio v0.21.71 — Aligned Facial-Expression Video Export",
        "checks": checks,
    }
    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    package = Path(oculostudio.__file__).resolve().parent
    service_text = (package / "services" / "aligned_facial_export.py").read_text("utf-8")
    server_text = (package / "server.py").read_text("utf-8")
    trim_js = (package / "static" / "native_trim.js").read_text("utf-8")
    guided_js = (package / "static" / "interaction_workspace.js").read_text("utf-8")
    checks.update({
        "artifact_kind_registered": "ALIGNED_FACIAL_VIDEO" in (package / "domain" / "models.py").read_text("utf-8"),
        "service_exported": "export_aligned_facial_video" in (package / "services" / "__init__.py").read_text("utf-8"),
        "api_route_present": '@app.post("/api/jobs/aligned-facial-video-export")' in server_text,
        "trim_panel_present": "4 · Export aligned facial video" in trim_js,
        "formal_player_present": 'id="native-aligned-facial-player"' in trim_js,
        "accepted_sync_required_in_ui": "visual-verified shared-audio" in trim_js,
        "guided_optional_path_present": "Export drift-corrected aligned facial video" in guided_js,
        "no_sync_reestimation": "synchronization_reestimated" in service_text,
        "reference_clock_resampling": "reference_output_clock_to_inverse_facial_time_map" in service_text,
        "arbitrary_dtw_forbidden": "arbitrary_dynamic_time_warping_used" in service_text,
        "audio_retimed_by_affine_scale": "atempo" in service_text and "audio_tempo_factor" in service_text,
    })

    fixture_module = _load_shared_fixture_module()
    with tempfile.TemporaryDirectory(prefix="oculo_aligned_facial_export_") as temp:
        temp_root = Path(temp)
        fixture_root = temp_root / "fixture"
        fixture_root.mkdir()
        fixture = fixture_module._fixture(fixture_root)
        store = ProjectStore.create(temp_root / "project", "Aligned facial validation", session_label="S001")
        session_id = store.project.selected_session_id
        facial = add_asset(
            store, session_id=session_id, path=fixture["facial_video"],
            label="S001_I01_R01_FV.mp4", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01", "media_role": "facial_expression_video"}},
        )
        native = add_asset(
            store, session_id=session_id, path=fixture["native_zip"],
            label="S001_I01_R01_NRD_untrimmed.zip", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01", "media_role": "ego_video"}},
        )
        source_before, native_before = sha256_file(fixture["facial_video"]), sha256_file(fixture["native_zip"])
        candidate = automatic_shared_audio_video_sync(
            store, session_id=session_id, source_video_asset_id=facial.id,
            reference_video_asset_id=native.id, wearer_role_id="R01", maximum_offset_sec=15,
        )["candidate"]
        accepted = accept_audio_video_sync(
            store, session_id=session_id, candidate_id=candidate["candidate_id"],
            visual_review_confirmed=True, visual_offset_adjustment_ms=0.0,
        )
        sync_id = accepted["artifact"]["id"]
        trim = _register_trim(
            store, session_id, native, video_member=candidate["reference_video_member"], start_sec=6.0, end_sec=56.0,
        )
        result = export_aligned_facial_video(
            store, session_id=session_id, native_trim_artifact_id=trim.id,
            time_sync_artifact_id=sync_id, label="S001_I01_R01_FV_aligned",
            output_fps=30, preserve_audio=True,
        )
        artifact = store.project.sessions[session_id].artifacts[result["artifact_id"]]
        frame_map = pd.read_csv(result["frame_map"])
        manifest = json.loads(Path(result["manifest"]).read_text("utf-8"))
        video = Path(result["video"])
        probe = _probe(video)
        stream_types = {row.get("codec_type") for row in probe.get("streams") or []}
        capture = cv2.VideoCapture(str(video))
        try:
            encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            encoded_duration = encoded_frames / float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        finally:
            capture.release()
        expected_frames = 1500
        mapping_scale = float(accepted["time_map"]["scale"])
        checks.update({
            "formal_artifact_created": artifact.kind == ArtifactKind.ALIGNED_FACIAL_VIDEO,
            "dual_parent_lineage": set(artifact.parent_artifact_ids) == {trim.id, sync_id},
            "clean_mp4_exists": video.is_file() and video.stat().st_size > 0,
            "frame_map_exists": Path(result["frame_map"]).is_file(),
            "manifest_exists": Path(result["manifest"]).is_file(),
            "report_exists": Path(result["report_html"]).is_file(),
            "output_frame_count_exact": len(frame_map) == expected_frames == encoded_frames,
            "output_duration_matches_physical_interval": abs(encoded_duration - 50.0) <= 1 / 30 + 0.01,
            "reference_time_monotonic": frame_map["reference_ego_media_time_ns"].is_monotonic_increasing,
            "facial_source_frame_monotonic": frame_map["facial_source_frame_index"].is_monotonic_increasing,
            "frame_map_has_dual_lineage": all(name in frame_map.columns for name in (
                "reference_native_frame_index", "reference_native_frame_pts_ns", "facial_source_frame_index", "facial_source_frame_pts_ns",
            )),
            "video_clock_drift_corrected": manifest["video"]["video_drift_corrected"] is True,
            "audio_clock_drift_corrected": manifest["video"]["audio_drift_corrected"] is True,
            "audio_stream_preserved": "audio" in stream_types,
            "video_stream_present": "video" in stream_types,
            "tempo_matches_inverse_clock_scale": abs(float(manifest["video"]["audio_tempo_factor"]) - (1.0 / mapping_scale)) < 2e-7,
            "physical_bounds_inherited": artifact.qc.get("physical_interval_from_native_trim") is True,
            "sync_not_reestimated": artifact.provenance.get("synchronization_reestimated") is False,
            "no_dynamic_warping": artifact.provenance.get("arbitrary_dynamic_time_warping_used") is False,
            "source_facial_video_unchanged": sha256_file(fixture["facial_video"]) == source_before,
            "source_native_zip_unchanged": sha256_file(fixture["native_zip"]) == native_before,
        })

        # A data mapping accepted for another Native ego source must not be reused.
        wrong_reference_blocked = False
        fake_root = store.derived_dir(session_id, "native_reference_trim") / "artifact_wrong_reference"
        fake_root.mkdir()
        for row in trim.files:
            src = Path(row.path)
            if src.is_file():
                (fake_root / src.name).write_bytes(src.read_bytes())
        wrong_manifest = json.loads((fake_root / "trim_manifest.json").read_text("utf-8"))
        wrong_manifest["request"]["native_asset_id"] = "missing_native_asset"
        atomic_write_json(fake_root / "trim_manifest.json", wrong_manifest)
        try:
            wrong = ArtifactStore(store).register(
                session_id=session_id, kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
                label="Wrong reference trim", output_root=fake_root,
                source_asset_ids=[native.id], parameters={**wrong_manifest["request"], "window": wrong_manifest["window"]},
                qc={}, artifact_id="artifact_wrong_reference",
            )
            export_aligned_facial_video(
                store, session_id=session_id, native_trim_artifact_id=wrong.id,
                time_sync_artifact_id=sync_id, label="should fail", preserve_audio=False,
            )
        except ValidationError:
            wrong_reference_blocked = True
        checks["reference_lineage_mismatch_fails_closed"] = wrong_reference_blocked
        report["measured"] = {
            "clock_scale": mapping_scale,
            "clock_drift_ppm": manifest["video"]["clock_drift_ppm"],
            "mapped_source_start_sec": manifest["video"]["mapped_source_start_sec"],
            "mapped_source_end_sec": manifest["video"]["mapped_source_end_sec"],
            "audio_tempo_factor": manifest["video"]["audio_tempo_factor"],
            "output_frame_count": len(frame_map),
            "output_video_bytes": video.stat().st_size,
        }

    failed = sorted(name for name, passed in checks.items() if not passed)
    report["passed_count"] = sum(bool(value) for value in checks.values())
    report["check_count"] = len(checks)
    report["failed"] = failed
    report["all_pass"] = not failed
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
