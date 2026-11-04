"""Live UI validation and preview for OculoStudio v0.21.71 geometry acceptance."""
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


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait(url: str) -> None:
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_21_50_BodyGeometryIntegrity_Preview.png"))
    parser.add_argument("--blocked-output", default=str(ROOT / "docs" / "OculoStudio_v0_21_50_BodyGeometryBlocked_Preview.png"))
    args = parser.parse_args()

    import uvicorn
    import oculostudio
    from create_privacy_aoi_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.domain.models import ArtifactKind
    from oculostudio.server import app, runtime
    from oculostudio.services.body_primitives import assess_body_geometry_integrity
    from playwright.sync_api import sync_playwright

    output = Path(args.output)
    blocked_output = Path(args.blocked_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    blocked_output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "release": "0.21.71",
        "feature": "Body geometry acceptance UI",
        "checks": {},
        "page_errors": [],
        "console_errors": [],
    }
    checks = report["checks"]

    with tempfile.TemporaryDirectory(prefix="oculo_geometry_integrity_ui_") as temp:
        project = Path(temp) / "project"
        build(project, overwrite=True)
        store = runtime.open(project)
        session = store.selected_session()
        body = next(row for row in session.artifacts.values() if row.kind == ArtifactKind.BODY_AOI)

        port = free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            wait(url)
            verify_ui_assets(url)
            static = Path(oculostudio.__file__).resolve().parent / "static"
            with sync_playwright() as playwright:
                launch = {"headless": True, "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    launch["executable_path"] = executable
                browser = playwright.chromium.launch(**launch)
                page = browser.new_page(viewport={"width": 1720, "height": 1100})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html)
                page.set_content(html)
                page.add_style_tag(content=(static / "styles.css").read_text("utf-8"))

                def relay(path, method, headers, body_text):
                    request = urllib.request.Request(
                        url + path,
                        method=method,
                        headers=headers or {},
                        data=body_text.encode() if body_text is not None else None,
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            return {"status": response.status, "body": response.read().decode(), "type": response.headers.get("Content-Type", "application/json")}
                    except urllib.error.HTTPError as error:
                        return {"status": error.code, "body": error.read().decode(), "type": error.headers.get("Content-Type", "application/json")}

                page.expose_function("relayStudioHTTP", relay)
                page.evaluate("""() => { window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}""")

                def route_local(route):
                    relative = route.request.url.split("studio.local", 1)[-1]
                    headers = {}
                    if route.request.headers.get("range"):
                        headers["Range"] = route.request.headers["range"]
                    request = urllib.request.Request(url + relative, headers=headers)
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            pass_headers = {key: value for key, value in response.headers.items() if key.lower() in {"content-type", "content-length", "content-range", "accept-ranges"}}
                            route.fulfill(status=response.status, body=response.read(), headers=pass_headers)
                    except urllib.error.HTTPError as error:
                        route.fulfill(status=error.code, body=error.read())

                page.route("http://studio.local/**", route_local)
                page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("appState.page='targets'; document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector(".body-geometry-integrity-card", timeout=60_000)
                page.locator(".body-geometry-integrity-card").scroll_into_view_if_needed()
                page.wait_for_timeout(500)
                passed_text = page.locator(".body-geometry-integrity-card").inner_text()
                passed_upper = passed_text.upper()
                checks["passed_card_visible"] = "PASSED" in passed_upper and "TORSO" in passed_upper and "PARTNER" in passed_upper
                checks["passed_artifact_review_entry_enabled"] = page.locator("#open-body-human-review-side").count() == 1
                page.screenshot(path=str(output), full_page=False)

                body.qc["body_geometry_integrity"] = assess_body_geometry_integrity(
                    [{
                        "entity_id": "P02",
                        "entity_relation": "partner",
                        "source_asset_id": "participant_P02_body",
                        "source_asset_label": "P02 body mocap",
                        "source_file_name": "participant_P02_body.npz",
                        "scene_frame_count": 150,
                        "frames_with_any_primitive": 150,
                        "torso_primitive_frame_count": 0,
                        "pelvis_primitive_frame_count": 0,
                        "torso_direct_frame_count": 0,
                        "torso_derived_frame_count": 0,
                    }],
                    expected_relations=["partner"],
                )
                store.save()
                page.click("#refresh-state")
                page.wait_for_function("document.querySelector('.body-geometry-integrity-card')?.innerText.includes('BLOCKED')", timeout=60_000)
                page.locator(".body-geometry-integrity-card").scroll_into_view_if_needed()
                page.wait_for_timeout(500)
                blocked_text = page.locator(".body-geometry-integrity-card").inner_text()
                checks["blocked_card_visible"] = "BLOCKED" in blocked_text and "produced no torso primitive" in blocked_text
                checks["blocked_artifact_review_entry_disabled"] = page.locator("#open-body-human-review-side").count() == 0 and "Body AOI blocked" in page.locator(".human-review-property-card.body").inner_text()
                checks["blocked_artifact_removed_from_fusion"] = page.locator("#fusion-body option").count() == 1
                page.screenshot(path=str(blocked_output), full_page=False)
                checks["responsive_without_horizontal_overflow"] = page.evaluate("document.documentElement.scrollWidth <= 1720")
                browser.close()

            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
            checks["no_page_errors"] = not report["page_errors"]
            checks["no_console_errors"] = not unexpected
            checks["installed_version"] = oculostudio.__version__ == "0.21.71"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    failed = [name for name, value in checks.items() if not value]
    report["failed"] = failed
    report["ok"] = not failed
    validation_path = output.with_suffix(".validation.json")
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
