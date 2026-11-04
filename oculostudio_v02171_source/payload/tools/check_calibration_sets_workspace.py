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

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait(url: str) -> None:
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("OculoStudio test server did not start")


def profile_payload(hardware: str, x_mm: float) -> dict:
    return {
        "profile_id": f"profile_{hardware.lower().replace('-', '_')}",
        "profile_name": f"{hardware} calibration-set profile",
        "participant_id": None,
        "hardware_id": hardware,
        "hardware_display_name": hardware,
        "mount_version": "mount_v1",
        "valid_sessions": ["S004", "S005", "S006", "S007"],
        "calibration_source": "measured_apriltag",
        "T_tracker_from_camera": [
            [1.0, 0.0, 0.0, x_mm],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "camera_intrinsics": {
            "matrix": [[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0] * 8,
            "resolution": [1600, 1200],
        },
        "tracker": {
            "marker_labels": [f"{hardware}:M1", f"{hardware}:M2", f"{hardware}:M3"],
            "marker_template_mm": [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]],
        },
        "qc": {"accepted_count": 12, "median_reprojection_error_px": 0.5},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wizard-output", default=str(ROOT / "docs" / "OculoStudio_v0_21_67_CalibrationSetWizard.png"))
    parser.add_argument("--data-output", default=str(ROOT / "docs" / "OculoStudio_v0_21_67_DataCalibrationSets.png"))
    parser.add_argument("--manager-output", default=str(ROOT / "docs" / "OculoStudio_v0_21_67_CalibrationSetManager.png"))
    parser.add_argument("--json", default=str(ROOT / "docs" / "OculoStudio_v0_21_67_CalibrationSets_Validation.json"))
    args = parser.parse_args()
    outputs = [Path(args.wizard_output).resolve(), Path(args.data_output).resolve(), Path(args.manager_output).resolve(), Path(args.json).resolve()]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    import uvicorn
    from playwright.sync_api import sync_playwright
    import oculostudio
    from oculostudio.server import app, runtime
    from oculostudio.services.calibration_profiles import import_calibration_profile, write_g3dcal_profile
    from oculostudio.services.calibration_registry import import_calibration_batch
    from oculostudio.services.calibration_sets import list_calibration_sets

    report = {
        "release": "OculoStudio v0.21.71 — Project Calibration Sets",
        "application_version": oculostudio.__version__,
        "feature": "Project-level revisioned Calibration Sets assigned to multiple Sessions without duplicating raw recordings",
        "checks": {},
        "page_errors": [],
        "console_errors": [],
    }
    checks = report["checks"]
    checks["installed_version_0_21_70"] = oculostudio.__version__ == "0.21.71"

    with tempfile.TemporaryDirectory(prefix="oculo_02167_calsets_") as temp:
        temp_root = Path(temp)
        parent = temp_root / "projects"
        parent.mkdir()
        source_dir = temp_root / "calibration_source"
        source_dir.mkdir()
        board_json = source_dir / "AprilTag_board_definition.json"
        board_json.write_text(json.dumps({"tag_corners_mm": {"0": [[0,0,0],[10,0,0],[10,10,0],[0,10,0]]}, "marker_template_mm": {"B1": [0,0,0]}}), encoding="utf-8")

        port = free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            wait(url)
            static = Path(oculostudio.__file__).resolve().parent / "static"
            with sync_playwright() as pw:
                opts = {"headless": True, "args": ["--no-sandbox"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    opts["executable_path"] = executable
                browser = pw.chromium.launch(**opts)
                page = browser.new_page(viewport={"width": 1760, "height": 1100})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html)
                page.set_content(html)
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
                page.evaluate("""() => { window.fetch = async (path, init={}) => { const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null); return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}}); }}""")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "gaze3d_preview.js", "calibration_preview.js", "app.js", "interaction_workspace.js", "motionstudio_bridge.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")

                page.wait_for_selector("#welcome-create")
                page.click("#welcome-create")
                page.fill("#create-parent", str(parent))
                page.fill("#create-folder", "OCUMO_S004_S007")
                page.fill("#create-name", "OCUMO S004–S007")
                page.click("#create-next")
                page.fill("#create-sessions-spec", "S004-S007")
                page.fill("#create-interactions-spec", "I01-I02")
                page.click("#create-next")
                page.wait_for_selector('[data-project-panel="3"]:not([hidden])')
                checks["wizard_has_calibration_set_fields"] = page.locator("#create-calibration-code").count() == 1 and page.locator("#create-calibration-label").count() == 1
                checks["wizard_assigns_all_four_sessions"] = page.locator("#create-calibration-session-chips span").count() == 4
                checks["wizard_folder_tree_uses_calibration_sets"] = "calibration_sets/CAL_S004_S007" in page.locator("#create-folder-tree").inner_text()
                page.screenshot(path=str(outputs[0]), full_page=False)
                page.click("#create-submit")
                page.wait_for_selector(".wh-session-card", timeout=30000)

                store = runtime.require_store()
                sets = list_calibration_sets(store)
                checks["one_initial_calibration_set_created"] = len(sets) == 1
                checks["set_bound_to_four_sessions"] = len(sets[0]["applicable_session_ids"]) == 4
                checks["folder_layout_version_three"] = store.snapshot().settings.get("folder_layout_version") == 3
                checks["calibration_set_revision_folder_exists"] = (Path(store.root) / "calibration_sets" / "CAL_S004_S007" / "revisions" / "rev_001").is_dir()

                owner = sets[0]["owner_session_id"]
                imported = import_calibration_batch(store, session_id=owner, paths=[source_dir], calibration_set_id="CAL_S004_S007")
                checks["one_canonical_source_registered"] = imported["imported_count"] == 1
                snapshot = store.snapshot()
                source_paths = []
                source_asset_ids = []
                for session in snapshot.sessions.values():
                    cal_assets = [asset for asset in session.assets.values() if asset.metadata.get("asset_scope") == "calibration"]
                    checks[f"{session.label}_has_inherited_source_reference"] = len(cal_assets) == 1
                    if cal_assets:
                        source_paths.append(cal_assets[0].source_path)
                        source_asset_ids.append(cal_assets[0].id)
                checks["raw_calibration_bytes_not_duplicated"] = len(set(source_paths)) == 1 and source_paths[0] == str(board_json.resolve())
                checks["session_alias_records_are_distinct"] = len(set(source_asset_ids)) == 4
                checks["calibration_sources_create_no_time_maps"] = all(len(session.time_maps) == 0 for session in snapshot.sessions.values())

                for hardware, x_mm in (("ETM-A", 10.0), ("ETM-B", -10.0)):
                    profile_path = temp_root / f"{hardware}.g3dcal"
                    write_g3dcal_profile(profile_payload(hardware, x_mm), profile_path)
                    import_calibration_profile(store, session_id=owner, profile_path=str(profile_path), label=f"{hardware} set profile", hardware_id=hardware, mount_version="mount_v1", valid_sessions=["S004","S005","S006","S007"], calibration_set_id=sets[0]["id"], calibration_revision_id=sets[0]["active_revision_id"])
                sets = list_calibration_sets(store)
                checks["two_profiles_attached_to_revision"] = sets[0]["profile_count"] == 2 and sets[0]["status"] == "ready"
                snapshot = store.snapshot()
                for session in snapshot.sessions.values():
                    profile_count = sum(1 for artifact in session.artifacts.values() if str(artifact.kind).endswith("calibration_profile"))
                    checks[f"{session.label}_inherits_two_profiles"] = profile_count == 2

                page.evaluate("refreshState({quiet:false,fullRender:true})")
                page.wait_for_timeout(500)
                page.evaluate("navigate('data')")
                page.wait_for_selector("#data-calibration-workspace")
                checks["data_page_separates_calibration_sets"] = page.locator(".calibration-set-card").count() == 1
                checks["data_page_shows_session_inheritance"] = "Inherited by S004" in page.locator(".calibration-set-card").inner_text()
                checks["interaction_table_excludes_calibration_sources"] = page.locator(".data-table-card tbody tr").count() == 0
                page.screenshot(path=str(outputs[1]), full_page=False)

                page.click("#data-manage-calibration-sets")
                page.wait_for_selector(".project-manager")
                checks["manager_lists_revisioned_set"] = page.locator(".calibration-set-manager-card").count() == 1
                checks["manager_has_four_session_assignments"] = page.locator(".calset-session-check:checked").count() == 4
                checks["manager_exposes_new_revision"] = page.locator(".calset-new-revision").count() == 1
                checks["manager_exposes_profile_attachment"] = page.locator(".calset-attach-profile").count() == 1
                page.screenshot(path=str(outputs[2]), full_page=False)

                page.click(".calset-new-revision")
                page.wait_for_selector(".project-manager")
                page.wait_for_timeout(400)
                sets = list_calibration_sets(store)
                checks["new_revision_created_and_activated"] = sets[0]["revision_count"] == 2 and sets[0]["active_revision_number"] == 2
                checks["revision_inherits_sources_and_profiles"] = sets[0]["source_count"] == 1 and sets[0]["profile_count"] == 2
                page.locator(".modal-close").click()
                browser.close()

            manifest = json.loads((parent / "OCUMO_S004_S007" / "project.oculostudio.json").read_text("utf-8"))
            checks["manifest_persists_project_level_calibration_sets"] = len(manifest.get("calibration_sets", {})) == 1
            checks["session_bindings_persist_revision"] = all(bool(session.get("metadata", {}).get("calibration_binding", {}).get("revision_id")) for session in manifest["sessions"].values())
            checks["source_file_remains_outside_project"] = not any(path.name == board_json.name for path in (parent / "OCUMO_S004_S007").rglob("*"))
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    checks["page_errors_absent"] = not report["page_errors"]
    checks["console_errors_absent"] = not report["console_errors"]
    report["passed_count"] = sum(value is True for value in checks.values())
    report["check_count"] = len(checks)
    report["failed"] = [key for key, value in checks.items() if value is not True]
    report["all_pass"] = not report["failed"] and not report["page_errors"] and not report["console_errors"]
    outputs[3].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
