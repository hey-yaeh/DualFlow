"""Live validation for OculoStudio v0.21.71 AOI review playback.

The check proves that the decoded video, 2D overlay, 3D body inset and gaze ray
advance together while the heavy inspector is frozen, and that the full review
frame is restored after pause.
"""
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

import oculostudio
import uvicorn
from playwright.sync_api import sync_playwright

from create_privacy_aoi_demo import build
from oculostudio.server import app, runtime


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _wait(base_url: str) -> None:
    for _ in range(300):
        try:
            if urllib.request.urlopen(base_url + "/api/health", timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("OculoStudio validation server did not start")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"release": "0.21.71", "checks": {}, "page_errors": [], "console_errors": []}

    with tempfile.TemporaryDirectory(prefix="oculo2154play_") as temp_dir:
        project = Path(temp_dir) / "project"
        build(project, overwrite=True)
        runtime.open(project)
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _wait(base_url)
        try:
            with sync_playwright() as playwright:
                launch = {"headless": True, "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    launch["executable_path"] = executable
                browser = playwright.chromium.launch(**launch)
                page = browser.new_page(viewport={"width": 1780, "height": 1080})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)

                static = Path(oculostudio.__file__).resolve().parent / "static"
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text())
                html = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html)
                page.set_content(html)
                page.add_style_tag(content=(static / "styles.css").read_text())

                def relay(path: str, method: str, headers: dict, body: str | None) -> dict:
                    request = urllib.request.Request(
                        base_url + path, method=method, headers=headers or {},
                        data=body.encode() if body else None,
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            return {
                                "status": response.status,
                                "body": response.read().decode(),
                                "type": response.headers.get("Content-Type", "application/json"),
                            }
                    except urllib.error.HTTPError as error:
                        return {
                            "status": error.code,
                            "body": error.read().decode(),
                            "type": error.headers.get("Content-Type", "application/json"),
                        }

                page.expose_function("relayStudioHTTP", relay)
                page.evaluate("""() => { window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}""")

                def route_local(route) -> None:
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

                page.route("http://studio.local/**", route_local)
                page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                    page.add_script_tag(content=(static / name).read_text())
                page.evaluate("appState.page='targets'; document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector("#aoi-review-shell")
                page.select_option("#aoi-review-mode", "body")
                page.wait_for_function("document.querySelector('#aoi-review-video-element').readyState>=1", timeout=60000)
                page.wait_for_function("document.querySelector('#aoi-review-summary')?.textContent.includes('Video frame')", timeout=60000)

                before = page.evaluate("""() => ({time:document.querySelector('#aoi-review-video-element').currentTime,button:document.querySelector('#aoi-review-play').textContent,status:document.querySelector('#aoi-review-playback-status').textContent,overlay:document.querySelector('#aoi-review-overlay').dataset.overlayFrame||'',world:document.querySelector('#aoi-review-world').dataset.overlayFrame||'',ray:document.querySelector('#aoi-review-world').dataset.rayDirection||''})""")
                page.click("#aoi-review-play")
                page.wait_for_timeout(1600)
                during = page.evaluate("""() => ({time:document.querySelector('#aoi-review-video-element').currentTime,paused:document.querySelector('#aoi-review-video-element').paused,button:document.querySelector('#aoi-review-play').textContent,status:document.querySelector('#aoi-review-playback-status').textContent,overlay:document.querySelector('#aoi-review-overlay').dataset.overlayFrame||'',presented:document.querySelector('#aoi-review-overlay').dataset.presentedFrame||'',world:document.querySelector('#aoi-review-world').dataset.overlayFrame||'',ray:document.querySelector('#aoi-review-world').dataset.rayDirection||'',frameReadout:document.querySelector('#aoi-live-video-frame')?.textContent||''})""")
                page.screenshot(path=args.output_dir / "OculoStudio_v0_21_54_LivePlayback_Preview.png", full_page=True)
                page.click("#aoi-review-play")
                page.wait_for_timeout(700)
                after = page.evaluate("""() => ({paused:document.querySelector('#aoi-review-video-element').paused,button:document.querySelector('#aoi-review-play').textContent,status:document.querySelector('#aoi-review-playback-status').textContent,summary:document.querySelector('#aoi-review-summary').textContent,payload:document.querySelector('#aoi-review-world').dataset.overlayFrame||''})""")
                page.screenshot(path=args.output_dir / "OculoStudio_v0_21_54_ReviewPlayback_Preview.png", full_page=True)
                browser.close()

                report.update({"before": before, "during": during, "after": after})
                report["checks"] = {
                    "application_version": oculostudio.__version__ == "0.21.71",
                    "play_button_visible_before": before["button"] == "▶ Play",
                    "play_button_never_ellipsis": during["button"] == "Ⅱ Pause" and during["button"] != "…",
                    "video_advances_near_real_time": during["time"] - before["time"] > 0.9,
                    "video_is_playing": not during["paused"],
                    "live_status_visible": "Playing" in during["status"],
                    "overlay_frame_advances": during["overlay"] not in {"", before["overlay"]},
                    "world_frame_tracks_overlay": during["world"] == during["overlay"] and during["world"] != "",
                    "live_frame_readout_tracks_overlay": during["frameReadout"] == during["overlay"],
                    "gaze_ray_has_direction": bool(during["ray"]) and during["ray"] != ",,",
                    "gaze_ray_changes_with_video": bool(during["ray"]) and during["ray"] != before["ray"],
                    "pause_button_restored": after["paused"] and after["button"] == "▶ Play",
                    "full_inspector_restored_after_pause": "Video frame" in after["summary"],
                    "no_page_errors": not report["page_errors"],
                    "no_console_errors": not report["console_errors"],
                }
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    report["passed"] = sum(bool(value) for value in report["checks"].values())
    report["total"] = len(report["checks"])
    report["ok"] = report["passed"] == report["total"]
    report_path = args.output_dir / "VALIDATION_v0_21_54_REALTIME_PLAYBACK.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
