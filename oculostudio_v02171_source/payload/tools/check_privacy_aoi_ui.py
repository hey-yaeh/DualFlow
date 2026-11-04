"""Reproducible v0.8 Privacy/Face/Body/Fusion UI regression check.

Creates a disposable synthetic project and uses the packaged HTML/CSS/JS with a
live local FastAPI backend. ``--transport python`` is only a transparent relay
for restricted test environments; no API response or video is mocked.
"""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("OculoStudio test server did not start")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "python"], default="http")
    parser.add_argument(
        "--privacy-output",
        default=str(ROOT / "docs" / "OculoStudio_v0_8_0_privacy_player.png"),
    )
    parser.add_argument(
        "--targets-output",
        default=str(ROOT / "docs" / "OculoStudio_v0_8_0_face_body_fusion.png"),
    )
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from create_privacy_aoi_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.server import app, runtime

    report: dict = {
        "release": "0.8.0",
        "transport": args.transport,
        "page_errors": [],
        "console_errors": [],
        "checks": [],
    }
    privacy_output = Path(args.privacy_output).resolve()
    targets_output = Path(args.targets_output).resolve()
    privacy_output.parent.mkdir(parents=True, exist_ok=True)
    targets_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="oculo_v080_ui_") as temp:
        project = Path(temp) / "project"
        build(project, overwrite=True)
        runtime.open(project)
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait(url)
            verify_ui_assets(url)
            with sync_playwright() as pw:
                options = {"headless": True, "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    options["executable_path"] = executable
                browser = pw.chromium.launch(**options)
                page = browser.new_page(viewport={"width": 1680, "height": 1050})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on(
                    "console",
                    lambda message: report["console_errors"].append(message.text)
                    if message.type == "error"
                    else None,
                )

                if args.transport == "http":
                    page.goto(url, wait_until="networkidle")
                else:
                    static = ROOT / "src" / "oculostudio" / "static"
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
                            with urllib.request.urlopen(request, timeout=120) as response:
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
                            with urllib.request.urlopen(request, timeout=120) as response:
                                response_headers = {
                                    key: value
                                    for key, value in response.headers.items()
                                    if key.lower() in {
                                        "content-type",
                                        "content-length",
                                        "content-range",
                                        "accept-ranges",
                                    }
                                }
                                route.fulfill(
                                    status=response.status,
                                    body=response.read(),
                                    headers=response_headers,
                                )
                        except urllib.error.HTTPError as error:
                            route.fulfill(status=error.code, body=error.read())

                    page.route("http://studio.local/**", route_local)
                    page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                    for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                        page.add_script_tag(content=(static / name).read_text("utf-8"))
                    page.evaluate("appState.page='privacy'; document.dispatchEvent(new Event('DOMContentLoaded'))")

                page.wait_for_selector('[data-page="privacy"]')
                page.click('[data-page="privacy"]')
                page.wait_for_selector("#artifact-video-player")
                # Show the trimmed blurred video on Privacy, not the later AOI overlay.
                privacy_clip_value = page.evaluate("""() => {
                    const option=[...document.querySelectorAll('#artifact-video-select option')]
                      .find(item => item.textContent.toLowerCase().includes('privacy clip'));
                    return option ? option.value : null;
                }""")
                if privacy_clip_value:
                    page.select_option('#artifact-video-select', privacy_clip_value)
                page.wait_for_function(
                    "document.querySelector('#artifact-video-player').readyState >= 1",
                    timeout=60_000,
                )
                privacy_video = page.locator("#artifact-video-player")
                report["privacy_player"] = page.evaluate(
                    """() => { const v=document.querySelector('#artifact-video-player'); return {
                    readyState:v.readyState,duration:v.duration,width:v.videoWidth,height:v.videoHeight,
                    source:v.currentSrc}; }"""
                )
                assert report["privacy_player"]["duration"] > 0
                report["checks"].append("Privacy page loaded a real generated MP4")
                page.screenshot(path=str(privacy_output))

                page.click('[data-page="targets"]')
                page.wait_for_selector("#aoi-review-video-element")
                page.wait_for_function(
                    "document.querySelector('#aoi-review-video-element').readyState >= 1",
                    timeout=60_000,
                )
                page.wait_for_function(
                    "document.querySelector('#aoi-review-summary')?.textContent.includes('Video frame')",
                    timeout=60_000,
                )
                report["targets_player"] = page.evaluate(
                    """() => { const v=document.querySelector('#aoi-review-video-element'); return {
                    readyState:v.readyState,duration:v.duration,width:v.videoWidth,height:v.videoHeight,
                    source:v.currentSrc,mode:document.querySelector('#aoi-review-mode').value}; }"""
                )
                assert report["targets_player"]["duration"] > 0
                assert page.locator(".target-step.complete").count() == 3
                report["checks"].append("Interactive AOI Review Player loaded and all three evidence/fusion steps rendered")
                report["document_scroll_width"] = page.evaluate("document.documentElement.scrollWidth")
                assert report["document_scroll_width"] <= 1680
                page.screenshot(path=str(targets_output))
                browser.close()

            unexpected = [
                message
                for message in report["console_errors"]
                if "favicon" not in message.lower()
            ]
            assert not report["page_errors"], report["page_errors"]
            assert not unexpected, unexpected
            report["ok"] = True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    validation_path = targets_output.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
