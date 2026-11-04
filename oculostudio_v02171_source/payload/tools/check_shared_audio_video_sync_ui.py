"""Live Chromium UI validation for v0.21.71 shared-audio video sync."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str) -> None:
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("OculoStudio test server did not start")


def _relay_media(route, base_url: str) -> None:
    relative = route.request.url.split("studio.local", 1)[-1]
    headers = {}
    if route.request.headers.get("range"):
        headers["Range"] = route.request.headers["range"]
    request = urllib.request.Request(base_url + relative, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_headers = {
                key: value for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length", "content-range", "accept-ranges"}
            }
            route.fulfill(status=response.status, body=response.read(), headers=response_headers)
    except urllib.error.HTTPError as error:
        route.fulfill(status=error.code, body=error.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs" / "OculoStudio_v0_21_61_SharedAudioVideoSync.png"),
    )
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from check_shared_audio_video_sync import _fixture
    from oculostudio.services import add_asset, automatic_shared_audio_video_sync
    from oculostudio.storage import ProjectStore
    from oculostudio.server import app, runtime

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "release": "0.21.71",
        "page_errors": [],
        "console_errors": [],
        "checks": {},
    }
    with tempfile.TemporaryDirectory(prefix="oculo_audio_video_sync_ui_") as temp:
        temp_root = Path(temp)
        fixture = _fixture(temp_root)
        store = ProjectStore.create(temp_root / "project", "Shared audio UI", session_label="S001")
        session_id = store.project.selected_session_id
        facial = add_asset(
            store, session_id=session_id, path=fixture["facial_video"],
            label="S001_I01_R01_FacialExpression.mp4", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01", "media_role": "facial_expression_video"}},
        )
        native = add_asset(
            store, session_id=session_id, path=fixture["native_zip"],
            label="S001_I01_R01_NRD_untrimmed.zip", participant_id="P01",
            metadata_extra={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": "R01", "media_role": "ego_video"}},
        )
        candidate = automatic_shared_audio_video_sync(
            store,
            session_id=session_id,
            source_video_asset_id=facial.id,
            reference_video_asset_id=native.id,
            wearer_role_id="R01",
            maximum_offset_sec=15,
        )["candidate"]
        runtime.open(temp_root / "project")
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait(url)
            with sync_playwright() as pw:
                options = {"headless": True, "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    options["executable_path"] = executable
                browser = pw.chromium.launch(**options)
                page = browser.new_page(viewport={"width": 1880, "height": 1180})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                static = Path(__import__("oculostudio").__file__).resolve().parent / "static"
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html)
                html = html.replace("<head>", '<head><base href="http://studio.local/">', 1)
                page.route("http://studio.local/**", lambda route: _relay_media(route, url))
                page.set_content(html)
                for css_name in ("styles.css", "motionstudio_bridge.css", "interaction_workspace.css"):
                    css_path = static / css_name
                    if css_path.exists():
                        page.add_style_tag(content=css_path.read_text("utf-8"))

                def relay_api(path: str, method: str, headers: dict, body: str | None):
                    request = urllib.request.Request(
                        url + path,
                        method=method,
                        headers=headers or {},
                        data=body.encode() if body is not None else None,
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            return {"status": response.status, "body": response.read().decode("utf-8"), "type": response.headers.get("Content-Type", "application/json")}
                    except urllib.error.HTTPError as error:
                        return {"status": error.code, "body": error.read().decode("utf-8"), "type": error.headers.get("Content-Type", "application/json")}

                page.expose_function("relayStudioHTTP", relay_api)
                page.evaluate("""() => { window.fetch = async (path, init={}) => {
                  const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);
                  return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}});
                }}""")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js", "interaction_workspace.js", "motionstudio_bridge.js"):
                    script = static / name
                    if script.exists():
                        page.add_script_tag(content=script.read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector("#ui-mode-select")
                report["checks"].update({
                    "mode_selector_is_explicit_dropdown": page.locator("#ui-mode-select option").all_text_contents() == ["Guided workflow", "Expert tools"],
                    "guided_mode_is_initially_explicit": page.locator("#ui-mode-select").input_value() == "guided",
                })
                page.select_option("#ui-mode-select", "expert")
                page.wait_for_selector('[data-page="calibration"]')
                page.click('[data-page="calibration"]')
                page.wait_for_selector(".shared-audio-sync-panel")
                report["checks"].update({
                    "expert_mode_selection_is_explicit": page.locator("#ui-mode-select").input_value() == "expert",
                    "calibration_opens_in_paired_summary": page.locator('[data-calibration-view="paired"].active').count() == 1,
                    "role_detail_views_are_read_only_choices": page.locator('[data-calibration-view="R01"]').count() >= 1 and page.locator('[data-calibration-view="R02"]').count() >= 1,
                    "panel_visible_in_calibration": page.locator(".shared-audio-sync-panel").count() == 1,
                    "source_video_selected": page.locator("#avsync-source").input_value() == facial.id,
                    "native_ego_selected": page.locator("#avsync-reference").input_value() == native.id,
                    "run_button_enabled": page.locator("#run-audio-video-sync").is_enabled(),
                    "scientific_contract_visible": page.get_by_text("Scientific acceptance contract", exact=True).count() == 1,
                })
                page.locator(".shared-audio-sync-panel").scroll_into_view_if_needed()
                page.wait_for_timeout(250)
                panel_output = output.with_name(output.stem + "_Panel.png")
                page.screenshot(path=str(panel_output), full_page=False)
                page.evaluate(
                    "candidateId => openAudioVideoSyncReview({candidateId})",
                    candidate["candidate_id"],
                )
                page.wait_for_selector(".avsync-modal")
                page.wait_for_selector("#avsync-source-video")
                page.wait_for_selector("#avsync-reference-video")
                page.wait_for_timeout(2500)
                source_duration = page.locator("#avsync-source-video").evaluate("video => video.duration")
                reference_duration = page.locator("#avsync-reference-video").evaluate("video => video.duration")
                report["checks"].update({
                    "split_video_review_visible": page.locator(".avsync-video-grid video").count() == 2,
                    "three_review_positions_visible": page.locator("[data-avsync-position]").count() == 3,
                    "visual_correction_control_visible": page.locator("#avsync-visual-correction").count() == 1,
                    "visual_confirmation_required": page.locator("#avsync-visual-confirm").count() == 1 and not page.locator("#avsync-visual-confirm").is_checked(),
                    "accept_button_visible": page.locator("#avsync-accept").count() == 1,
                    "source_video_streams": isinstance(source_duration, (int, float)) and source_duration > 1,
                    "reference_video_streams": isinstance(reference_duration, (int, float)) and reference_duration > 1,
                    "audio_trace_canvas_visible": page.locator("#avsync-chart").count() == 1,
                })
                page.screenshot(path=str(output), full_page=False)
                browser.close()
            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
            report["checks"]["page_errors_absent"] = not report["page_errors"]
            report["checks"]["console_errors_absent"] = not unexpected
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()
    failed = sorted(name for name, passed in report["checks"].items() if not passed)
    report["passed_count"] = sum(bool(value) for value in report["checks"].values())
    report["check_count"] = len(report["checks"])
    report["failed"] = failed
    report["all_pass"] = not failed
    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
