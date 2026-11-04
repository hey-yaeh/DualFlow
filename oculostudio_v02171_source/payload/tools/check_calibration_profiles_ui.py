"""Render and validate the v0.16.1 calibration-profile workflow with a live backend."""
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

import numpy as np

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
    from oculostudio.domain.models import AssetKind, AssetRecord, ArtifactKind
    from oculostudio.services.calibration_profiles import import_calibration_profile, write_g3dcal_profile
    from oculostudio.storage import ProjectStore
    from oculostudio.storage.artifact_store import ArtifactStore

    store = ProjectStore.create(root, "OCUMO S001–S003 Calibration Library", session_label="S001")
    sid = store.project.selected_session_id
    assert sid
    source = root / "sources"
    source.mkdir(parents=True, exist_ok=True)
    labels = [f"eye_tracker_1:M_{index}" for index in range(1, 7)]
    template = np.asarray([
        [-35.0, -12.0, 0.0], [34.0, -11.0, 1.0], [-29.0, 21.0, 6.0],
        [31.0, 24.0, 7.0], [-12.0, -3.0, 31.0], [15.0, 5.0, 39.0],
    ])
    profile = {
        "format": "oculostudio-calibration-profile-1.0",
        "profile_id": "cal_eye_tracker_1_mount_v1",
        "profile_name": "eye_tracker_1 · mount_v1 · S001–S003",
        "hardware_id": "eye_tracker_1",
        "mount_version": "mount_v1",
        "valid_sessions": ["S001", "S002", "S003"],
        "participant_id": "P02",
        "calibration_source": "session_measured_apriltag",
        "tracker": {"marker_labels": labels, "marker_template_mm": template.tolist()},
        "T_tracker_from_camera": np.eye(4).tolist(),
        "camera_intrinsics": {"matrix": [[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]], "distortion": [0,0,0,0,0]},
        "qc": {"reprojection_rms_px": 0.94},
        "warnings": [],
    }
    g3dcal = write_g3dcal_profile(profile, source / "eye_tracker_1_mount_v1_S001-S003.g3dcal")
    import_calibration_profile(
        store,
        session_id=sid,
        profile_path=str(g3dcal),
        label="eye_tracker_1 · mount_v1 · S001–S003",
        hardware_id="eye_tracker_1",
        mount_version="mount_v1",
        valid_sessions=["S001", "S002", "S003"],
    )
    # Older measured artifact without Hardware ID, used to validate the
    # no-recalculation migration action.
    unregistered = dict(profile)
    unregistered["profile_id"] = "cal_old_p01"
    unregistered["profile_name"] = "P01_calibration_S001-S003_auto"
    unregistered["participant_id"] = "P01"
    unregistered["hardware_id"] = None
    unregistered["calibration_source"] = "session_measured_apriltag_auto_16_marker"
    unregistered["tracker"] = dict(unregistered["tracker"])
    unregistered["tracker"]["marker_labels"] = [f"eye_tracker_2:M_{index}" for index in range(1, 7)]
    old_root = root / "old_measured"
    old_root.mkdir(parents=True, exist_ok=True)
    write_g3dcal_profile(unregistered, old_root / "P01_calibration_S001-S003_auto.g3dcal")
    ArtifactStore(store).register(
        session_id=sid,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label="P01_calibration_S001-S003_auto",
        output_root=old_root,
        qc={
            "calibration_source": "session_measured_apriltag_auto_16_marker",
            "selected_tracker_filename": "eye_tracker_B_S001-S003.c3d",
        },
        parameters={
            "calibration_source": "session_measured_apriltag_auto_16_marker",
            "participant_id": "P01",
            "hardware_id": None,
            "mount_version": "mount_v1",
            "valid_sessions": ["S001", "S002", "S003"],
        },
    )
    # The transfer panel only needs a selectable ETM asset for rendering.
    tracker_path = source / "S004_I01_ETM_A.npz"
    points = np.repeat(template[None, :, :], 20, axis=0)
    np.savez_compressed(tracker_path, points=points, labels=np.asarray(labels), fps=np.asarray(120.0))

    def mutate(project):
        session = project.sessions[sid]
        asset = AssetRecord(
            label="S004_I01_ETM_A.npz",
            kind=AssetKind.MOCAP_NPZ,
            source_path=str(tracker_path),
            metadata={"format": "mocap_npz", "marker_count": 6, "frame_count": 20, "frame_rate": 120.0},
        )
        session.assets[asset.id] = asset

    store.update(mutate)
    return store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_16_1_Auto_Hardware_Registration.png"))
    args = parser.parse_args()

    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.server import app, runtime

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"release": "0.16.1", "page_errors": [], "console_errors": [], "checks": []}

    with tempfile.TemporaryDirectory(prefix="oculo_calibration_ui_") as temp:
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
                page.select_option('#ui-mode-select', 'expert')
                page.wait_for_selector('[data-page="calibration"]')
                page.click('[data-page="calibration"]')
                page.wait_for_selector('#run-calibration-import')
                assert page.locator('text=Import completed profile').count() >= 1
                assert page.locator('text=Register an existing measured result').count() >= 1
                assert page.locator('#run-calibration-register-measured').is_enabled()
                assert page.locator('text=Auto-register selected tracker as ETM-A / ETM-B').count() >= 1
                assert page.locator('#cal-auto-register').get_attribute('data-on') == 'true'
                assert page.locator('text=Recover a session without calibration').count() >= 1
                assert page.locator('text=Estimate from raw calibration recording').count() >= 1
                assert page.locator('text=Auto-infer legacy 16-marker').count() >= 1
                assert page.locator('#cal-board-mode').input_value() == 'auto_legacy_16_marker'
                assert page.locator('text=eye_tracker_1 · mount_v1 · S001–S003').count() >= 1
                assert page.locator('#cal-transfer-profile option').count() >= 2
                assert page.locator('#cal-transfer-tracker option').count() >= 2
                page.locator('#cal-auto-register').scroll_into_view_if_needed()
                page.wait_for_timeout(250)
                page.screenshot(path=str(output), full_page=False)
                browser.close()
                report["checks"] = [
                    "Existing .g3dcal/.oculocal import is visible",
                    "Existing measured results can be registered without recalibration",
                    "Measured calibration defaults to automatic ETM-A/ETM-B registration",
                    "Missing-session transfer validation is visible",
                    "Raw AprilTag re-estimation remains available",
                    "Legacy 16-marker auto-board mode is visible and selected",
                    "Hardware profile library renders imported profile",
                    "Current ETM asset is selectable for recovery",
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
