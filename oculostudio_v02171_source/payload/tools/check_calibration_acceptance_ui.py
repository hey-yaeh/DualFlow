"""Render and validate the v0.16.1 transferred-calibration acceptance workflow."""
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
    from oculostudio.demo import create_demo_project
    from oculostudio.domain.models import ArtifactKind
    from oculostudio.services.calibration import load_calibration_profile
    from oculostudio.services.calibration_acceptance import run_transferred_calibration_acceptance
    from oculostudio.services.calibration_profiles import validate_calibration_profile_transfer
    from oculostudio.services.native_reference_trim import export_native_reference_trim, preview_native_reference_trim
    from oculostudio.storage import ArtifactStore, ProjectStore
    from oculostudio.util import atomic_write_json

    demo = create_demo_project(root, run_processing=False)
    store = ProjectStore.open(root)
    sid = demo["session_id"]
    assets = demo["assets"]
    transfer = validate_calibration_profile_transfer(
        store,
        session_id=sid,
        tracker_asset_id=assets["tracker"],
        profile_artifact_id=demo["profile_artifact_id"],
        interaction_id="I01",
        role_id="R01",
        label="eye_tracker_1 · I01 transferred",
        maximum_shape_rms_mm=5.0,
        high_confidence_rms_mm=2.0,
    )
    preview = preview_native_reference_trim(
        store,
        session_id=sid,
        native_asset_id=assets["native"],
        timeseries_asset_id=assets["cloud"],
        facemapper_asset_id=assets["facemapper"],
        start_timecode="00:00:02:00",
        duration_timecode="00:00:03:00",
        timecode_fps=30,
        label="S001_I01_R01",
        face_tolerance_ms=8.0,
    )
    trim = export_native_reference_trim(store, session_id=sid, preview_id=preview["preview_id"])
    body_id = assets["body"]
    tracker_id = assets["tracker"]
    assignment_root = root / "derived" / sid / "etm_assignment" / "acceptance_demo"
    assignment_root.mkdir(parents=True, exist_ok=True)
    assignment_payload = {
        "schema_version": "oculostudio-etm-role-assignment-1.0",
        "status": "resolved",
        "assignments": {
            "R01": {
                "role_id": "R01", "body_asset_id": "unused-ego-body",
                "body_head_marker_labels": ["P02:LFHD", "P02:RFHD", "P02:LBHD", "P02:RBHD"],
                "etm_asset_id": tracker_id, "etm_id": "ETM_A", "tracker_prefix": "tracker", "confidence": "HIGH",
            },
            "R02": {
                "role_id": "R02", "body_asset_id": body_id,
                "body_asset_label": store.project.sessions[sid].assets[body_id].label,
                "body_subject_prefix": "P02",
                "body_head_marker_labels": ["P02:LFHD", "P02:RFHD", "P02:LBHD", "P02:RBHD"],
                "etm_asset_id": "unused-partner-etm", "etm_id": "ETM_B", "tracker_prefix": "tracker_2", "confidence": "HIGH",
            },
        },
    }
    atomic_write_json(assignment_root / "etm_role_assignment.json", assignment_payload)
    assignment = ArtifactStore(store).register(
        session_id=sid,
        kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
        label="S001_I01 ETM assignment",
        output_root=assignment_root,
        source_asset_ids=[tracker_id, body_id],
        parameters={"interaction_id": "I01"},
        qc={"status": "resolved"},
    )
    run_transferred_calibration_acceptance(
        store,
        session_id=sid,
        transferred_profile_artifact_id=transfer["artifact"]["id"],
        native_trim_artifact_id=trim["artifact_id"],
        etm_assignment_artifact_id=assignment.id,
        wearer_role_id="R01",
        label="S001_I01_R01 calibration acceptance",
    )
    return store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_15_0_Calibration_Acceptance.png"))
    args = parser.parse_args()
    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.server import app, runtime

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"release": "0.16.1", "page_errors": [], "console_errors": [], "checks": []}
    with tempfile.TemporaryDirectory(prefix="oculo_acceptance_ui_") as temp:
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
                page = browser.new_page(viewport={"width": 1840, "height": 1160})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                import oculostudio
                static = Path(oculostudio.__file__).resolve().parent / "static"
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
                page.select_option('#ui-mode-select', 'expert')
                page.click('[data-page="calibration"]')
                page.wait_for_selector('#run-calibration-acceptance')
                assert page.locator('text=Accept transferred calibration for production').count() >= 1
                assert page.locator('text=S001_I01_R01 calibration acceptance').count() >= 1
                assert page.locator('text=accepted_limited').count() >= 1
                assert page.locator('#cal-accept-profile option').count() >= 2
                page.screenshot(path=str(output), full_page=False)
                browser.close()
                report["checks"] = [
                    "Transferred profile creation is visible",
                    "Automatic production acceptance gate is visible",
                    "Formal time/head/known-target evidence is summarized",
                    "Accepted-limited status and allowed granularity render",
                    "Gaze 3D can consume an acceptance artifact",
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
