"""Reproducible OculoStudio AOI Review Player regression UI check.

Builds a disposable project with Face AOI, Body AOI, fused targets, a trimmed
Privacy MP4 and a pre-rendered review MP4. It then opens the packaged UI against
a live local FastAPI backend and exercises dynamic frame/sample navigation.
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
        "--output",
        default=str(ROOT / "docs" / "OculoStudio_v0_21_46_AOI_Review_Player.png"),
    )
    args = parser.parse_args()

    import uvicorn
    import oculostudio
    from playwright.sync_api import sync_playwright
    from create_privacy_aoi_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.server import app, runtime

    report: dict = {
        "release": "0.21.71",
        "transport": args.transport,
        "page_errors": [],
        "console_errors": [],
        "checks": [],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="oculo_v02144_review_") as temp:
        project = Path(temp) / "project"
        demo = build(project, overwrite=True)
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
                page = browser.new_page(viewport={"width": 1780, "height": 1080})
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
                                        "content-type", "content-length", "content-range", "accept-ranges",
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
                page.wait_for_function("document.querySelector('#aoi-review-video-element').readyState >= 1", timeout=60_000)
                page.wait_for_function("document.querySelector('#aoi-review-summary')?.textContent.includes('Video frame')", timeout=60_000)

                # Prefer fusion mode and exercise sample/frame navigation.
                page.select_option("#aoi-review-mode", "fusion")
                page.wait_for_timeout(500)
                page.wait_for_function("document.querySelector('#aoi-review-video-element').readyState >= 1", timeout=60_000)
                page.click("#aoi-review-next-frame")
                page.wait_for_timeout(250)
                page.click("#aoi-review-next-sample")
                page.wait_for_timeout(250)
                page.click('[data-review-layer="trail"]')
                page.click('[data-review-layer="trail"]')

                state = page.evaluate(
                    """() => ({
                      mode: document.querySelector('#aoi-review-mode').value,
                      artifact: document.querySelector('#aoi-review-artifact').value,
                      video: document.querySelector('#aoi-review-video').value,
                      readyState: document.querySelector('#aoi-review-video-element').readyState,
                      duration: document.querySelector('#aoi-review-video-element').duration,
                      samplePosition: document.querySelector('#aoi-review-sample-position').textContent,
                      summary: document.querySelector('#aoi-review-summary').innerText,
                      reviewButtons: document.querySelectorAll('[data-review-nav]').length,
                      layerToggles: document.querySelectorAll('[data-review-layer]').length,
                      existingReviewVideos: document.querySelectorAll('.aoi-review-exports .map-row').length,
                      scrollWidth: document.documentElement.scrollWidth,
                    })"""
                )
                assert state["mode"] == "fusion"
                assert state["artifact"]
                assert state["video"]
                assert state["readyState"] >= 1
                assert state["duration"] > 0
                assert state["reviewButtons"] >= 4
                assert state["layerToggles"] >= 7
                assert "Video frame" in state["summary"]
                assert state["existingReviewVideos"] >= 1
                assert state["scrollWidth"] <= 1780
                report["player"] = state
                report["checks"].extend([
                    "Fusion AOI artifact loaded against a real trimmed Privacy MP4",
                    "Frame and within-frame gaze-sample navigation rendered",
                    "Review/conflict/low-confidence navigation controls rendered",
                    "Dynamic layer controls rendered",
                    "Pre-rendered review MP4 artifact listed",
                ])
                page.screenshot(path=str(output), full_page=False)
                browser.close()

            unexpected = [message for message in report["console_errors"] if "favicon" not in message.lower()]
            assert not report["page_errors"], report["page_errors"]
            assert not unexpected, unexpected
            report["ok"] = True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    validation_path = output.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
