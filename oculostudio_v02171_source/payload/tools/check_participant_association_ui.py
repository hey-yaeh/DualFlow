"""Render the v0.10 participant-head projection and face-identity workflow."""
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


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url + "/api/health", timeout=1).status == 200:
                return
        except Exception:
            time.sleep(.1)
    raise RuntimeError("test server did not start")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["http", "python"], default="python")
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_10_0_Participant_Association.png"))
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from create_participant_association_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.server import app, runtime

    report = {"release": "0.10.0", "transport": args.transport, "page_errors": [], "console_errors": [], "checks": []}
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oculo_v010_participant_") as temp:
        project = Path(temp) / "project"; build(project, overwrite=True); runtime.open(project)
        port = _free_port(); url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True); thread.start()
        try:
            _wait(url); verify_ui_assets(url)
            with sync_playwright() as pw:
                opts = {"headless": True, "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]}
                exe = shutil.which("chromium") or shutil.which("google-chrome")
                if exe: opts["executable_path"] = exe
                browser = pw.chromium.launch(**opts)
                page = browser.new_page(viewport={"width": 1840, "height": 1080})
                page.on("pageerror", lambda e: report["page_errors"].append(str(e)))
                page.on("console", lambda m: report["console_errors"].append(m.text) if m.type == "error" else None)
                if args.transport == "http":
                    page.goto(url, wait_until="networkidle")
                else:
                    static = ROOT / "src/oculostudio/static"
                    html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                    html = re.sub(r"<link[^>]+rel=\"stylesheet\"[^>]*>", "", html)
                    page.set_content(html); page.add_style_tag(content=(static / "styles.css").read_text("utf-8"))
                    def relay(path, method, headers, body):
                        request = urllib.request.Request(url + path, method=method, headers=headers or {}, data=body.encode() if body else None)
                        try:
                            with urllib.request.urlopen(request, timeout=180) as response:
                                return {"status": response.status, "body": response.read().decode(), "type": response.headers.get("Content-Type", "application/json")}
                        except urllib.error.HTTPError as error:
                            return {"status": error.code, "body": error.read().decode(), "type": error.headers.get("Content-Type", "application/json")}
                    page.expose_function("relayStudioHTTP", relay)
                    page.evaluate("""() => { window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}""")
                    def route_local(route):
                        relative = route.request.url.split("studio.local", 1)[-1]
                        headers = {"Range": route.request.headers["range"]} if route.request.headers.get("range") else {}
                        request = urllib.request.Request(url + relative, headers=headers)
                        try:
                            with urllib.request.urlopen(request, timeout=180) as response:
                                keep = {k:v for k,v in response.headers.items() if k.lower() in {"content-type","content-length","content-range","accept-ranges"}}
                                route.fulfill(status=response.status, body=response.read(), headers=keep)
                        except urllib.error.HTTPError as error:
                            route.fulfill(status=error.code, body=error.read())
                    page.route("http://studio.local/**", route_local)
                    page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                    for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                        page.add_script_tag(content=(static / name).read_text("utf-8"))
                    page.evaluate("appState.page='targets'; document.dispatchEvent(new Event('DOMContentLoaded'))")

                page.wait_for_selector('[data-page="targets"]'); page.click('[data-page="targets"]')
                page.wait_for_selector("#run-participant-association")
                page.wait_for_selector("#aoi-review-shell")
                page.select_option("#aoi-review-mode", "face")
                page.wait_for_function("document.querySelector('#aoi-review-video-element').readyState >= 1", timeout=60_000)
                page.wait_for_function("(document.querySelector('#aoi-review-summary')?.textContent||'').includes('Video frame')", timeout=60_000)
                # Ensure identity evidence is actually visible in the current payload/inspector.
                page.wait_for_function("(document.querySelector('#aoi-review-summary')?.textContent||'').includes('partner')", timeout=60_000)
                state = page.evaluate("""() => ({
                  associationCard: [...document.querySelectorAll('.target-step strong')].some(x=>x.textContent.includes('Participant association')),
                  participantLayer: !!document.querySelector('[data-review-layer="participant"]'),
                  identityLayer: !!document.querySelector('[data-review-layer="identity"]'),
                  otherLayer: !!document.querySelector('[data-review-layer="other"]'),
                  summary: document.querySelector('#aoi-review-summary').innerText,
                  layerCount: document.querySelectorAll('[data-review-layer]').length,
                  scrollWidth: document.documentElement.scrollWidth,
                })""")
                assert state["associationCard"]
                assert state["participantLayer"] and state["identityLayer"] and state["otherLayer"]
                assert "partner" in state["summary"] and "other_person" in state["summary"]
                assert state["layerCount"] >= 10
                assert state["scrollWidth"] <= 1840
                report["ui"] = state
                report["checks"] = [
                    "Participant association stage rendered",
                    "Projected-participant, face-identity and other-person layers rendered",
                    "Partner and staff/other-person evidence visible in the live AOI player",
                    "Packaged UI fit within desktop viewport",
                ]
                page.screenshot(path=str(output), full_page=False)
                browser.close()
            unexpected = [m for m in report["console_errors"] if "favicon" not in m.lower()]
            assert not report["page_errors"], report["page_errors"]
            assert not unexpected, unexpected
            report["ok"] = True
        finally:
            server.should_exit = True; thread.join(timeout=10); runtime.close()
    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
