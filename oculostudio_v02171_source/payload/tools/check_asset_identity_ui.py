"""Render and validate the v0.13.1 asset identity workflow with a live backend."""
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


def build_project(root: Path):
    from oculostudio.domain.models import AssetKind, AssetRecord
    from oculostudio.services.asset_identity import set_initial_asset_identity
    from oculostudio.storage import ProjectStore

    store = ProjectStore.create(root, "OCUMO S001 Processing", session_label="S001")
    sid = store.project.selected_session_id
    assert sid

    def mutate(project):
        session = project.sessions[sid]
        rows = [
            ("S001_I01_R01_NRD_untrimmed.zip", AssetKind.PUPIL_NATIVE, "P01"),
            ("S001_I01_R01_TSD_untrimmed.zip", AssetKind.PUPIL_CLOUD, "P01"),
            ("S001_R01_FM_untrimmed.zip", AssetKind.FACE_MAPPER, "P01"),
            ("S001_I01_R02_C3D.c3d", AssetKind.C3D, "P02"),
            ("S001_I01_ETM_A.c3d", AssetKind.C3D, None),
        ]
        for label, kind, participant in rows:
            asset = AssetRecord(label=label, kind=kind, source_path=str(root / "sources" / label))
            session.assets[asset.id] = asset
            set_initial_asset_identity(session, asset, participant, source="import_user")

    store.update(mutate)
    return store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_13_1_Asset_Identity.png"))
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.server import app, runtime

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"release": "0.13.1", "page_errors": [], "console_errors": [], "checks": []}

    with tempfile.TemporaryDirectory(prefix="oculo_identity_ui_") as temp:
        project = Path(temp) / "project"
        build_project(project)
        runtime.open(project)
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
                page = browser.new_page(viewport={"width": 1780, "height": 1080})
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
                page.wait_for_selector('[data-page="project"]')
                page.select_option('#ui-mode-select', 'expert')  # switch to Expert Mode navigation
                page.wait_for_selector('[data-page="data"]')
                page.click('[data-page="data"]')
                page.wait_for_selector('.asset-table')
                page.locator('.asset-inspect').first.click()
                page.wait_for_selector('#edit-selected-identity')
                text = page.locator('.info-card').last.inner_text()
                assert "Participant ID" in text and "P01" in text
                assert "Interaction" in text and "I01" in text
                assert "Role" in text and "R01" in text
                assert page.locator('text=Unassigned').count() >= 1
                page.screenshot(path=str(output), full_page=False)
                page.locator('#edit-selected-identity').click()
                page.wait_for_selector('#identity-save')
                assert page.locator('#identity-participant').input_value() == "P01"
                page.locator('#identity-participant').fill("P01A")
                page.locator('#identity-save').click()
                page.wait_for_timeout(500)
                assert "P01A" in page.locator('.info-card').last.inner_text()
                browser.close()
                report["checks"] = [
                    "Participant ID appears in Inspector",
                    "Session / Interaction / Role appear in Inspector",
                    "Unassigned ETM identity remains visible",
                    "Identity can be edited without re-import",
                ]
            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
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
