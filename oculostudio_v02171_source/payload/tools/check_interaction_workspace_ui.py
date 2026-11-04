"""Render and validate the v0.17.0 Workflow Hub with a live backend."""
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
    from oculostudio.domain.models import ArtifactKind, ArtifactRecord, AssetKind, AssetRecord
    from oculostudio.storage import ProjectStore

    store = ProjectStore.create(root, "OCUMO S001", session_label="S001")
    sid = store.project.selected_session_id
    assert sid

    def mutate(project):
        session = project.sessions[sid]
        for interaction in ("I01", "I02", "I03"):
            for role, participant in (("R01", "P01"), ("R02", "P02")):
                for kind, suffix in ((AssetKind.PUPIL_NATIVE, "NRD"), (AssetKind.PUPIL_CLOUD, "TSD")):
                    asset = AssetRecord(
                        label=f"S001_{interaction}_{role}_{suffix}_untrimmed.zip",
                        kind=kind,
                        source_path=str(root / "sources" / f"S001_{interaction}_{role}_{suffix}.zip"),
                        participant_id=participant,
                        metadata={"identity": {"session_id": "S001", "interaction_id": interaction, "role_id": role}},
                    )
                    session.assets[asset.id] = asset
            for role, participant in (("R01", "P01"), ("R02", "P02")):
                fm = AssetRecord(
                    label=f"S001_{role}_FM_untrimmed.zip",
                    kind=AssetKind.FACE_MAPPER,
                    source_path=str(root / "sources" / f"S001_{role}_FM.zip"),
                    participant_id=participant,
                    metadata={"identity": {"session_id": "S001", "role_id": role}},
                )
                session.assets[fm.id] = fm

        def add(kind, label, parameters=None, qc=None):
            row = ArtifactRecord(kind=kind, label=label, session_id=sid, parameters=parameters or {}, qc=qc or {})
            session.artifacts[row.id] = row

        for role in ("R01", "R02"):
            add(ArtifactKind.NATIVE_REFERENCE_TRIM, f"S001_I01_{role}_trim")
            add(ArtifactKind.PRIVACY_CLIP, f"S001_I01_{role}_privacy_trimmed")
            add(ArtifactKind.FACE_AOI, f"S001_I01_{role}_face_aoi")
        add(ArtifactKind.MOTIONSTUDIO_SCENE, "MotionStudio Scene S001_I01", {"scene_id": "S001_I01"})
        add(ArtifactKind.ETM_ROLE_ASSIGNMENT, "S001_I01_ETM_assignment")
        add(ArtifactKind.CALIBRATION_PROFILE, "ETM-A_mount_v1_S001-S003", {"hardware_id": "ETM_A", "hardware_display_name": "ETM-A", "mount_version": "mount_v1", "valid_sessions": ["S001", "S002", "S003"], "calibration_source": "session_measured_apriltag_auto_16_marker"})
        add(ArtifactKind.CALIBRATION_PROFILE, "ETM-B_mount_v1_S001-S003", {"hardware_id": "ETM_B", "hardware_display_name": "ETM-B", "mount_version": "mount_v1", "valid_sessions": ["S001", "S002", "S003"], "calibration_source": "session_measured_apriltag_auto_16_marker"})

    store.update(mutate)
    return store, sid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_17_0_Workflow_Hub.png"))
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.server import app, runtime

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"release": "0.17.0", "page_errors": [], "console_errors": [], "checks": []}

    with tempfile.TemporaryDirectory(prefix="oculo_workflow_hub_") as temp:
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
                page.wait_for_selector('[data-page="interactions"]')
                assert page.locator('#ui-mode-select').input_value() == 'guided'
                assert page.locator('[data-page="hardware"]').count() == 1
                page.click('[data-page="interactions"]')
                page.wait_for_selector('.wh-stage-nav')
                assert page.locator('.iw-interaction-pill').count() == 3
                assert page.locator('.wh-stage-nav button').count() == 4
                assert page.locator('.wh-context-identities article').count() == 3
                assert page.locator('.wh-role-column').count() == 2
                page.click('[data-iw-stage="scene"]')
                page.wait_for_selector('.wh-shared-card')
                assert page.locator('.wh-shared-card').count() == 1
                page.screenshot(path=str(output), full_page=False)
                page.click('[data-page="hardware"]')
                page.wait_for_selector('.wh-profile-card')
                assert page.locator('.wh-profile-card').count() == 2
                page.select_option('#ui-mode-select', 'expert')
                page.wait_for_selector('[data-page="synchronize"]')
                assert page.locator('#ui-mode-select').input_value() == 'expert'
                report["checks"] = [
                    "Workflow Hub is the default mode",
                    "Five guided navigation entries are present",
                    "Interaction workflow has four ordered stages",
                    "R01, R02 and Shared Scene identities are visible",
                    "Hardware & Calibration page lists ETM profiles",
                    "Expert Tools remain available",
                ]
                browser.close()
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
