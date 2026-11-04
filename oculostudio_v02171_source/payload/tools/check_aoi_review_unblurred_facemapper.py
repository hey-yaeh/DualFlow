"""Targeted validation for the local unblurred AOI review toggle and FaceMapper diagnostics."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind, AssetKind, AssetRecord
from oculostudio.server import app, runtime
from oculostudio.services.aoi_review import (
    review_frame,
    review_manifest,
    review_unblurred_video_path,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not start: {url}")


def _build_project(root: Path) -> dict[str, str]:
    store = ProjectStore.create(root, "AOI local review validation", session_label="Session S001")
    session_id = store.project.selected_session_id
    artifacts = ArtifactStore(store)

    native_root = root / "sources" / "S001_I01_R01_NRD_untrimmed"
    native_root.mkdir(parents=True)
    media_probe = Path(oculostudio.__file__).resolve().parent / "resources" / "media_probe_test.mp4"
    shutil.copy2(media_probe, native_root / "world.mp4")
    native = AssetRecord(
        label="S001_I01_R01 Native",
        kind=AssetKind.PUPIL_NATIVE,
        source_path=str(native_root),
        recording_id="review-toggle-test",
        participant_id="P01",
    )
    store.update(lambda project: project.sessions[session_id].assets.__setitem__(native.id, native))

    trim_root = store.derived_dir(session_id, "native_reference_trim", "trim_review_toggle")
    pd.DataFrame([
        {"native_frame_index": 10, "source_frame_ordinal_in_window": 0, "clip_video_pts_ns": 0, "native_capture_utc_ns": 1_000_000_000},
        {"native_frame_index": 11, "source_frame_ordinal_in_window": 1, "clip_video_pts_ns": 33_333_333, "native_capture_utc_ns": 1_033_333_333},
        {"native_frame_index": 12, "source_frame_ordinal_in_window": 2, "clip_video_pts_ns": 66_666_666, "native_capture_utc_ns": 1_066_666_666},
    ]).to_csv(trim_root / "native_frame_map.csv", index=False)
    (trim_root / "trim_manifest.json").write_text(json.dumps({
        "recording_id": "review-toggle-test",
        "native_reference": {"width": 32, "height": 32},
    }), "utf-8")
    trim = artifacts.register(
        session_id=session_id,
        kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
        label="S001_I01_R01 trim",
        output_root=trim_root,
        source_asset_ids=[native.id],
    )

    video_root = store.derived_dir(session_id, "privacy_clip", "privacy_review_toggle")
    shutil.copy2(media_probe, video_root / "S001_I01_R01_privacy_trimmed.mp4")
    pd.DataFrame([
        {"output_frame_index": 0, "output_pts_ns": 0, "native_frame_index": 10, "native_capture_utc_ns": 1_000_000_000, "native_video_pts_ns": 0},
        {"output_frame_index": 1, "output_pts_ns": 33_333_333, "native_frame_index": 11, "native_capture_utc_ns": 1_033_333_333, "native_video_pts_ns": 33_333_333},
        {"output_frame_index": 2, "output_pts_ns": 66_666_666, "native_frame_index": 12, "native_capture_utc_ns": 1_066_666_666, "native_video_pts_ns": 66_666_666},
    ]).to_csv(video_root / "privacy_trimmed_frame_map.csv", index=False)
    (video_root / "privacy_clip_report.json").write_text(json.dumps({
        "native_asset_id": native.id,
        "native_video_member": "world.mp4",
        "recording_id": "review-toggle-test",
        "window": {"start_media_ns": 0, "end_media_ns": 100_000_000},
        "output": {"fps": 30, "width": 32, "height": 32},
    }), "utf-8")
    video = artifacts.register(
        session_id=session_id,
        kind=ArtifactKind.PRIVACY_CLIP,
        label="S001_I01_R01 privacy",
        output_root=video_root,
        source_asset_ids=[native.id],
        parent_artifact_ids=[trim.id],
        provenance={"native_trim_artifact_id": trim.id},
        qc={"output_fps": 30, "output_frame_count": 3},
    )

    filter_root = store.derived_dir(session_id, "facemapper_filter", "filter_review_toggle")
    pd.DataFrame([
        {
            "source_face_row_index": 0,
            "timestamp_ns": 1_000_000_000,
            "native_frame_index": 10,
            "track_id": 1,
            "filter_status": "review",
            "rejection_reasons": "temporal_scale_jump;landmark_cluster_off_center",
            "keep_privacy_safe": True,
            "keep_analysis_strict": False,
        },
        {
            "source_face_row_index": 1,
            "timestamp_ns": 1_066_666_666,
            "native_frame_index": 12,
            "track_id": 1,
            "filter_status": "accepted",
            "rejection_reasons": "",
            "keep_privacy_safe": True,
            "keep_analysis_strict": True,
        },
    ]).to_csv(filter_root / "face_filter_audit.csv", index=False)
    (filter_root / "filter_manifest.json").write_text(json.dumps({"schema_version": "test"}), "utf-8")
    filter_artifact = artifacts.register(
        session_id=session_id,
        kind=ArtifactKind.FACE_MAPPER_FILTER,
        label="S001 FaceMapper analysis-strict",
        output_root=filter_root,
        parent_artifact_ids=[trim.id],
    )

    participant_root = store.derived_dir(session_id, "participant_association", "participant_review_toggle")
    pd.DataFrame([
        {
            "native_frame_index": frame,
            "role_id": "R02",
            "entity_id": "R02",
            "relation_to_wearer": "partner",
            "projection_valid": True,
            "in_camera_view": True,
            "projected_bbox_x1": 11,
            "projected_bbox_y1": 5,
            "projected_bbox_x2": 25,
            "projected_bbox_y2": 25,
            "projection_confidence": 0.95,
        }
        for frame in (10, 11, 12)
    ]).to_csv(participant_root / "participant_head_projection.csv", index=False)
    pd.DataFrame([
        {
            "native_frame_index": 12,
            "matched_entity_id": "R02",
            "matched_role": "R02",
            "candidate_entity_id": "R02",
            "candidate_role": "R02",
            "candidate_relation": "partner",
            "target_relation": "partner",
            "identity_status": "matched_participant",
            "identity_confidence": 0.92,
            "projected_box_iou": 0.31,
            "normalized_center_error": 0.18,
        }
    ]).to_csv(participant_root / "face_identity_assignment.csv", index=False)
    (participant_root / "participant_association_manifest.json").write_text(json.dumps({
        "schema_version": "test",
        "wearer_role_id": "R01",
    }), "utf-8")
    participant = artifacts.register(
        session_id=session_id,
        kind=ArtifactKind.PARTICIPANT_ASSOCIATION,
        label="S001_I01_R01 participant association",
        output_root=participant_root,
        parent_artifact_ids=[trim.id, filter_artifact.id],
        parameters={"wearer_role_id": "R01", "target_policy": "partner_only"},
        provenance={
            "native_trim_artifact_id": trim.id,
            "facemapper_filter_artifact_id": filter_artifact.id,
        },
    )

    face_root = store.derived_dir(session_id, "face_aoi", "face_review_toggle")
    samples = pd.DataFrame([
        {
            "gaze_sample_uid": "g0", "source_gaze_row_index": 0,
            "timestamp_ns": 1_000_000_000, "clip_time_ns": 0,
            "native_frame_index": 10, "scene_frame_index": 10,
            "gaze_x_px": 15, "gaze_y_px": 15, "worn": True,
            "face_entity_id": None, "face_region": "no_face",
            "face_social_region": "no_face", "face_aoi_confidence": 0.0,
            "assignment_status": "no_face_detection", "identity_status": "no_face_detection",
            "identity_confidence": 0.0, "target_relation": "none",
            "include_in_primary_analysis": False, "review_required": True,
        },
        {
            "gaze_sample_uid": "g1", "source_gaze_row_index": 1,
            "timestamp_ns": 1_033_333_333, "clip_time_ns": 33_333_333,
            "native_frame_index": 11, "scene_frame_index": 11,
            "gaze_x_px": 16, "gaze_y_px": 16, "worn": True,
            "face_entity_id": None, "face_region": "no_face",
            "face_social_region": "no_face", "face_aoi_confidence": 0.0,
            "assignment_status": "no_face_detection", "identity_status": "no_face_detection",
            "identity_confidence": 0.0, "target_relation": "none",
            "include_in_primary_analysis": False, "review_required": True,
        },
        {
            "gaze_sample_uid": "g2", "source_gaze_row_index": 2,
            "timestamp_ns": 1_066_666_666, "clip_time_ns": 66_666_666,
            "native_frame_index": 12, "scene_frame_index": 12,
            "gaze_x_px": 16, "gaze_y_px": 13, "worn": True,
            "face_entity_id": "R02", "candidate_entity_id": "R02",
            "candidate_role": "R02", "candidate_relation": "partner",
            "face_region": "left_eye", "face_social_region": "partner_left_eye",
            "face_aoi_confidence": 0.92, "assignment_status": "matched_participant",
            "identity_status": "matched_participant", "identity_confidence": 0.92,
            "target_relation": "partner", "include_in_primary_analysis": True,
            "review_required": False,
        },
    ])
    samples.to_csv(face_root / "face_aoi_samples.csv", index=False)
    pd.DataFrame([
        {"output_frame_index": 0, "native_frame_index": 10, "gaze_sample_count": 1, "primary_entity_id": None, "primary_region": "no_face", "mean_confidence": 0.0, "review_required": True},
        {"output_frame_index": 1, "native_frame_index": 11, "gaze_sample_count": 1, "primary_entity_id": None, "primary_region": "no_face", "mean_confidence": 0.0, "review_required": True},
        {"output_frame_index": 2, "native_frame_index": 12, "gaze_sample_count": 1, "primary_entity_id": "R02", "primary_region": "partner_left_eye", "mean_confidence": 0.92, "review_required": False},
    ]).to_csv(face_root / "face_aoi_frames.csv", index=False)
    pd.DataFrame([
        {
            "scene_frame_index": 12,
            "face_entity_id": "R02",
            "region": "face",
            "geometry_type": "bbox",
            "x1": 11, "y1": 5, "x2": 25, "y2": 25,
            "geometry_json": json.dumps({"x1": 11, "y1": 5, "x2": 25, "y2": 25}),
        },
        {
            "scene_frame_index": 12,
            "face_entity_id": "R02",
            "region": "left_eye",
            "geometry_type": "ellipse",
            "geometry_json": json.dumps({"center_x": 16, "center_y": 13, "radius_x": 3, "radius_y": 2}),
        },
    ]).to_csv(face_root / "face_regions.csv", index=False)
    (face_root / "face_aoi_manifest.json").write_text(json.dumps({"schema_version": "test"}), "utf-8")
    face = artifacts.register(
        session_id=session_id,
        kind=ArtifactKind.FACE_AOI,
        label="S001_I01_R01 Face AOI",
        output_root=face_root,
        parent_artifact_ids=[trim.id, filter_artifact.id, participant.id, video.id],
        parameters={"face_tolerance_ms": 8.0},
        provenance={
            "native_trim_artifact_id": trim.id,
            "privacy_clip_artifact_id": video.id,
            "facemapper_filter_artifact_id": filter_artifact.id,
            "participant_association_artifact_id": participant.id,
        },
    )
    return {
        "session_id": session_id,
        "face_artifact_id": face.id,
        "video_artifact_id": video.id,
        "native_video": str(native_root / "world.mp4"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "application_version": oculostudio.__version__,
        "feature": "Local unblurred AOI review and per-frame FaceMapper coverage diagnostics",
        "checks": {},
        "page_errors": [],
        "console_errors": [],
    }
    checks: dict[str, bool] = report["checks"]  # type: ignore[assignment]

    with tempfile.TemporaryDirectory(prefix="oculo_unblurred_review_") as temp:
        root = Path(temp) / "project"
        ids = _build_project(root)
        store = ProjectStore.open(root)
        manifest = review_manifest(
            store,
            session_id=ids["session_id"],
            mode="face",
            artifact_id=ids["face_artifact_id"],
            video_artifact_id=ids["video_artifact_id"],
        )
        unblurred = manifest["video_sources"]["unblurred"]
        checks["manifest_has_privacy_and_unblurred_sources"] = (
            manifest["video_sources"]["privacy"]["available"] is True
            and unblurred["available"] is True
        )
        checks["unblurred_is_local_review_only"] = (
            unblurred["local_review_only"] is True
            and unblurred["export_uses_privacy_clip"] is True
            and manifest["video_sources"]["privacy"]["export_uses_privacy_clip"] is True
        )
        checks["native_pts_drive_unblurred_frame_lock"] = unblurred["frame_pts_ns"] == [0, 33_333_333, 66_666_666]
        checks["original_video_resolves_without_copying_into_artifact"] = (
            review_unblurred_video_path(store, ids["session_id"], ids["video_artifact_id"]).resolve()
            == Path(ids["native_video"]).resolve()
        )

        frame0 = review_frame(
            store, session_id=ids["session_id"], mode="face",
            artifact_id=ids["face_artifact_id"], video_artifact_id=ids["video_artifact_id"],
            output_frame_index=0,
        )["face_coverage_diagnostic"]
        frame1 = review_frame(
            store, session_id=ids["session_id"], mode="face",
            artifact_id=ids["face_artifact_id"], video_artifact_id=ids["video_artifact_id"],
            output_frame_index=1,
        )["face_coverage_diagnostic"]
        frame2 = review_frame(
            store, session_id=ids["session_id"], mode="face",
            artifact_id=ids["face_artifact_id"], video_artifact_id=ids["video_artifact_id"],
            output_frame_index=2,
        )["face_coverage_diagnostic"]
        checks["analysis_strict_filter_reason_is_visible"] = (
            frame0["code"] == "analysis_strict_filtered_review_candidate"
            and frame0["partner_projection_valid"] is True
            and any(row["reason"] == "temporal_scale_jump" and row["explanation"] for row in frame0["filter_reason_details"])
        )
        checks["detector_miss_is_distinguished_from_lost_partner"] = (
            frame1["code"] == "facemapper_no_detection"
            and frame1["partner_projection_valid"] is True
            and "2D FaceMapper coverage issue" in frame1["detail"]
        )
        checks["accepted_partner_geometry_is_reported"] = (
            frame2["code"] == "partner_face_covered"
            and frame2["matched_partner"] is True
            and abs(float(frame2["projected_box_iou"]) - 0.31) < 1e-9
        )
        report["diagnostic_examples"] = {"strict_filter": frame0, "detector_miss": frame1, "covered": frame2}

        js = (Path(oculostudio.__file__).resolve().parent / "static" / "aoi_review.js").read_text("utf-8")
        css = (Path(oculostudio.__file__).resolve().parent / "static" / "styles.css").read_text("utf-8")
        checks["ui_has_one_click_blur_toggle"] = all(token in js for token in (
            "aoi-review-blur-toggle", "Blur off · original", "switchVideoSource", "aoi-review-unblurred-warning",
        ))
        checks["source_switch_waits_for_exact_seek"] = "seekVideoForSourceSwitch" in js and "Timed out while locking the selected review frame" in js
        checks["ui_explains_facemapper_coverage"] = all(token in js for token in (
            "FaceMapper coverage", "What to check:", "filter_reason_details", "2D/3D IoU",
        ))
        checks["unblurred_mode_is_visually_explicit"] = all(token in css for token in (
            ".aoi-review-blur-toggle.active", ".aoi-review-unblurred-warning", ".aoi-review-video-wrap.showing-unblurred",
        ))
        checks["review_export_remains_privacy_clip"] = "video_artifact_id:state.videoArtifactId" in js

        import uvicorn
        runtime.open(root)
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait(url)
            request = urllib.request.Request(
                f"{url}/api/aoi-review/{ids['session_id']}/unblurred-video/{ids['video_artifact_id']}",
                headers={"Range": "bytes=0-31"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = response.read()
                checks["unblurred_route_supports_no_store_byte_ranges"] = (
                    response.status == 206
                    and response.headers.get("Cache-Control") == "no-store"
                    and response.headers.get("Accept-Ranges") == "bytes"
                    and len(payload) == 32
                )

            if not args.no_browser:
                from playwright.sync_api import sync_playwright

                with sync_playwright() as pw:
                    browser = pw.chromium.launch(
                        headless=True,
                        executable_path=shutil.which("chromium") or shutil.which("google-chrome"),
                        args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"],
                    )
                    page = browser.new_page(viewport={"width": 1760, "height": 1040})
                    page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                    page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                    static = Path(oculostudio.__file__).resolve().parent / "static"
                    html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                    html = re.sub(r"<link[^>]+rel=\"stylesheet\"[^>]*>", "", html)
                    page.set_content(html)
                    page.add_style_tag(content=(static / "styles.css").read_text("utf-8"))

                    def relay(path: str, method: str, headers: dict, body: str | None):
                        if not path.startswith("/api/"):
                            raise ValueError("Only local OculoStudio API calls are relayed")
                        request = urllib.request.Request(
                            url + path,
                            method=method,
                            headers=headers or {},
                            data=body.encode() if body is not None else None,
                        )
                        try:
                            with urllib.request.urlopen(request, timeout=180) as response:
                                return {
                                    "status": response.status,
                                    "body": response.read().decode("utf-8"),
                                    "type": response.headers.get("Content-Type", "application/json"),
                                }
                        except urllib.error.HTTPError as error:
                            return {
                                "status": error.code,
                                "body": error.read().decode("utf-8"),
                                "type": error.headers.get("Content-Type", "application/json"),
                            }

                    page.expose_function("relayStudioHTTP", relay)
                    page.evaluate(
                        """() => { window.fetch = async (path, init={}) => {
                        const r = await window.relayStudioHTTP(String(path), init.method || 'GET', init.headers || {}, init.body || null);
                        return new Response(r.body, {status:r.status, headers:{'Content-Type':r.type}});
                    }}"""
                    )

                    def route_local(route):
                        relative = route.request.url.split("studio.local", 1)[-1]
                        headers = {}
                        if route.request.headers.get("range"):
                            headers["Range"] = route.request.headers["range"]
                        request = urllib.request.Request(url + relative, headers=headers)
                        try:
                            with urllib.request.urlopen(request, timeout=180) as response:
                                response_headers = {
                                    key: value
                                    for key, value in response.headers.items()
                                    if key.lower() in {
                                        "content-type", "content-length", "content-range", "accept-ranges", "cache-control",
                                    }
                                }
                                route.fulfill(status=response.status, body=response.read(), headers=response_headers)
                        except urllib.error.HTTPError as error:
                            route.fulfill(status=error.code, body=error.read())

                    page.route("http://studio.local/**", route_local)
                    page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                    for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                        page.add_script_tag(content=(static / name).read_text("utf-8"))
                    page.evaluate("appState.page='targets'; document.dispatchEvent(new Event('DOMContentLoaded'))")
                    page.wait_for_selector('[data-page="targets"]')
                    page.click('[data-page="targets"]')
                    page.wait_for_selector("#aoi-review-shell")
                    page.wait_for_function("document.querySelector('#aoi-review-video-element')?.readyState >= 1", timeout=60_000)
                    page.wait_for_function("document.querySelector('#aoi-review-summary')?.textContent.includes('Video frame')", timeout=60_000)
                    page.click("#aoi-review-next-frame")
                    page.wait_for_function("document.querySelector('#aoi-review-summary')?.textContent.includes('11')", timeout=20_000)
                    before = page.evaluate("""() => ({
                      source: document.querySelector('#aoi-review-video-element').currentSrc,
                      frame: document.querySelector('#aoi-review-summary').innerText,
                      button: document.querySelector('#aoi-review-blur-toggle').textContent,
                    })""")
                    page.click("#aoi-review-blur-toggle")
                    page.wait_for_function("document.querySelector('#aoi-review-blur-toggle')?.classList.contains('active')", timeout=20_000)
                    page.wait_for_function("document.querySelector('#aoi-review-video-element')?.currentSrc.includes('/unblurred-video/')", timeout=20_000)
                    unblurred_state = page.evaluate("""() => ({
                      source: document.querySelector('#aoi-review-video-element').currentSrc,
                      frame: document.querySelector('#aoi-review-summary').innerText,
                      button: document.querySelector('#aoi-review-blur-toggle').textContent,
                      warningHidden: document.querySelector('#aoi-review-unblurred-warning').hidden,
                      diagnostic: document.querySelector('.face-coverage-diagnostic')?.innerText || '',
                    })""")
                    page.click("#aoi-review-blur-toggle")
                    page.wait_for_function("!document.querySelector('#aoi-review-blur-toggle')?.classList.contains('active')", timeout=20_000)
                    page.wait_for_function("document.querySelector('#aoi-review-video-element')?.currentSrc.includes('/api/media/artifact/')", timeout=20_000)
                    after = page.evaluate("""() => ({
                      source: document.querySelector('#aoi-review-video-element').currentSrc,
                      frame: document.querySelector('#aoi-review-summary').innerText,
                      button: document.querySelector('#aoi-review-blur-toggle').textContent,
                    })""")
                    checks["browser_toggle_switches_only_video_source"] = (
                        "/api/media/artifact/" in before["source"]
                        and "/unblurred-video/" in unblurred_state["source"]
                        and "/api/media/artifact/" in after["source"]
                        and "Blur off" in unblurred_state["button"]
                        and unblurred_state["warningHidden"] is False
                    )
                    checks["browser_toggle_preserves_review_frame"] = (
                        "Native frame\n11" in before["frame"]
                        and "Native frame\n11" in unblurred_state["frame"]
                        and "Native frame\n11" in after["frame"]
                    )
                    checks["browser_shows_current_facemapper_reason"] = (
                        "FaceMapper exported no face detection" in unblurred_state["diagnostic"]
                        and "2D FaceMapper coverage issue" in unblurred_state["diagnostic"]
                    )
                    if args.output:
                        output = Path(args.output).resolve()
                        output.parent.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(output), full_page=False)
                        report["screenshot"] = str(output)
                    browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    unexpected_console = [row for row in report["console_errors"] if "favicon" not in row.lower()]
    checks["browser_has_no_page_errors"] = not report["page_errors"]
    checks["browser_has_no_unexpected_console_errors"] = not unexpected_console
    report["all_passed"] = all(checks.values())
    if args.output:
        report_path = Path(args.output).resolve().with_suffix(".validation.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
        report["validation_report"] = str(report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
