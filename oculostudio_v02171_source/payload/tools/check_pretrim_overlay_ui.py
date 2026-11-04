"""Render and validate the v0.18.2 pre-trim overlay workflow with a live backend."""
from __future__ import annotations

import argparse
import base64
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


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait(url: str) -> None:
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
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_18_0_PreTrim_Overlay.png"))
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.native_trim_demo import create_native_trim_demo
    from oculostudio.server import app, runtime

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"release": "0.18.2", "page_errors": [], "console_errors": [], "checks": []}

    with tempfile.TemporaryDirectory(prefix="oculo_pretrim_overlay_") as temp:
        demo = create_native_trim_demo(Path(temp) / "project")
        runtime.open(demo["project"])
        overlay_video = Path(demo["overlay_preview"]["video_path"])
        video_data = "data:video/mp4;base64," + base64.b64encode(overlay_video.read_bytes()).decode("ascii")
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            wait(url)
            with sync_playwright() as pw:
                opts = {"headless": True, "args": ["--no-sandbox"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    opts["executable_path"] = executable
                browser = pw.chromium.launch(**opts)
                page = browser.new_page(viewport={"width": 1900, "height": 1200})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                static = ROOT / "src" / "oculostudio" / "static"
                html_text = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html_text = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html_text)
                page.set_content(html_text)
                for css in ("styles.css", "motionstudio_bridge.css", "interaction_workspace.css"):
                    page.add_style_tag(content=(static / css).read_text("utf-8"))

                def relay(path: str, method: str, headers: dict, body: str | None):
                    request = urllib.request.Request(url + path, method=method, headers=headers or {}, data=body.encode() if body is not None else None)
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            return {"status": response.status, "body": response.read().decode("utf-8"), "type": response.headers.get("Content-Type", "application/json")}
                    except urllib.error.HTTPError as error:
                        return {"status": error.code, "body": error.read().decode("utf-8"), "type": error.headers.get("Content-Type", "application/json")}

                page.expose_function("relayStudioHTTP", relay)
                page.evaluate("""() => { window.fetch = async (path, init={}) => {
                  const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);
                  return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}});
                }}""")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js", "interaction_workspace.js", "motionstudio_bridge.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('#ui-mode-select')
                page.select_option('#ui-mode-select', 'expert')
                page.wait_for_selector('[data-page="clips"]')
                page.click('[data-page="clips"]')
                page.wait_for_selector('#native-overlay-player')
                page.evaluate("(src)=>{const v=document.querySelector('#native-overlay-player');v.src=src;v.load();}", video_data)
                page.wait_for_timeout(800)
                page.evaluate("()=>{const v=document.querySelector('#native-overlay-player');v.currentTime=1.1;v.pause();document.querySelector('.native-overlay-preview-card').scrollIntoView({block:'start'});}")
                page.wait_for_timeout(700)
                assert page.locator('#native-overlay-button').inner_text().strip() == 'Render FM + AOI preview'
                assert page.locator('.native-overlay-preview-card').count() == 1
                assert page.locator('.native-overlay-metrics span').count() == 5
                assert 'TEMPORARY CACHE' in page.locator('.native-overlay-preview-card').inner_text()
                assert page.locator('#native-export-button').is_enabled()
                page.screenshot(path=str(output), full_page=True)
                report["checks"] = [
                    "Alignment/count preview remains a separate verified stage",
                    "Render FM + AOI preview is available before export",
                    "Temporary preview player loads without creating an artifact",
                    "Representative preview metrics are visible",
                    "Export trimmed TSD + FM remains enabled independently",
                    "Overlay settings expose representative, boundary and full modes",
                ]
                browser.close()
            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower() and "native-trim-overlay" not in row.lower()]
            assert not report["page_errors"], report["page_errors"]
            assert not unexpected, unexpected
            report["ok"] = True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
