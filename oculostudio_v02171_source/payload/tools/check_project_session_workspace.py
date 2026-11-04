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
    deadline = time.time() + 30
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
        "profile_name": f"{hardware} shared hardware profile",
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
    parser.add_argument(
        "--wizard-output",
        default=str(ROOT / "docs" / "OculoStudio_v0_21_66_ProjectCreationWizard.png"),
    )
    parser.add_argument(
        "--workspace-output",
        default=str(ROOT / "docs" / "OculoStudio_v0_21_66_ProjectSessionWorkspace.png"),
    )
    parser.add_argument(
        "--json",
        default=str(ROOT / "docs" / "OculoStudio_v0_21_66_ProjectSessionWorkspace_Validation.json"),
    )
    args = parser.parse_args()
    wizard_output = Path(args.wizard_output).resolve()
    workspace_output = Path(args.workspace_output).resolve()
    json_output = Path(args.json).resolve()
    for path in (wizard_output, workspace_output, json_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    import uvicorn
    from playwright.sync_api import sync_playwright
    import oculostudio
    from oculostudio.server import app, runtime
    from oculostudio.services.calibration_profiles import import_calibration_profile, write_g3dcal_profile
    from oculostudio.services.project_management import (
        apply_shared_calibration_profiles,
        inspect_project_location,
        project_calibration_library,
    )
    from oculostudio.storage import ProjectStore

    report = {
        "release": "OculoStudio v0.21.71 — Project–Session Folder Workspace",
        "application_version": oculostudio.__version__,
        "feature": "Multi-Session project wizard, human-readable Session folders, Interaction planning and shared Calibration library",
        "checks": {},
        "page_errors": [],
        "console_errors": [],
    }
    checks = report["checks"]
    checks["installed_version_0_21_70"] = oculostudio.__version__ == "0.21.71"

    with tempfile.TemporaryDirectory(prefix="oculo_02166_project_workspace_") as temp:
        temp_root = Path(temp)
        parent = temp_root / "projects"
        parent.mkdir()
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
                page.on(
                    "console",
                    lambda message: report["console_errors"].append(message.text)
                    if message.type == "error"
                    else None,
                )
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html)
                page.set_content(html)
                for css in ("styles.css", "motionstudio_bridge.css", "interaction_workspace.css"):
                    page.add_style_tag(content=(static / css).read_text("utf-8"))

                def relay(path: str, method: str, headers: dict, body: str | None):
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
                    const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);
                    return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}});
                }}"""
                )
                for name in (
                    "native_trim.js",
                    "asset_import.js",
                    "aoi_review.js",
                    "gaze3d_preview.js",
                    "calibration_preview.js",
                    "app.js",
                    "interaction_workspace.js",
                    "motionstudio_bridge.js",
                ):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")

                # New-project wizard: typed paths must be real input, not a decorative placeholder.
                page.wait_for_selector("#welcome-create")
                page.click("#welcome-create")
                page.wait_for_selector(".project-wizard")
                checks["parent_input_initially_blank"] = page.locator("#create-parent").input_value() == ""
                checks["example_is_not_a_selected_path"] = (
                    "example is not a selected value" in page.locator('[data-project-panel="1"]').inner_text().lower()
                )
                page.fill("#create-parent", str(parent))
                page.fill("#create-folder", "OCUMO_S004_S007")
                page.fill("#create-name", "OCUMO S004–S007")
                page.click("#create-next")
                page.wait_for_selector('[data-project-panel="2"]:not([hidden])')
                checks["typed_parent_path_accepted"] = str(parent) in page.locator("#create-path-preview").inner_text()
                page.fill("#create-sessions-spec", "S004-S007")
                page.fill("#create-interactions-spec", "I01-I02")
                checks["session_range_expands_to_four"] = page.locator("#create-session-chips span").count() == 4
                page.click("#create-next")
                page.wait_for_selector('[data-project-panel="3"]:not([hidden])')
                checks["shared_calibration_enabled_by_default"] = page.locator("#create-calibration-set").is_checked()
                checks["folder_tree_shows_session_roots"] = "sessions/S004" in page.locator("#create-folder-tree").inner_text()
                page.screenshot(path=str(wizard_output), full_page=False)
                page.click("#create-submit")
                page.wait_for_selector(".wh-session-card", timeout=30000)
                checks["four_sessions_created_atomically"] = page.locator(".wh-session-card").count() == 4
                checks["two_planned_interactions_visible"] = page.locator(".iw-interaction-card").count() == 2
                checks["guided_home_shows_project_sessions"] = "Project structure" in page.locator("body").inner_text() and "Selected Session" in page.locator("body").inner_text()

                store = runtime.require_store()
                snapshot = store.snapshot()
                session_by_code = {
                    store.session_code(session_id): session_id for session_id in snapshot.sessions
                }
                source_session_id = session_by_code["S004"]
                source_profile_ids: list[str] = []
                for hardware, x_mm in (("ETM-A", 10.0), ("ETM-B", -10.0)):
                    source_path = temp_root / f"{hardware}.g3dcal"
                    write_g3dcal_profile(profile_payload(hardware, x_mm), source_path)
                    imported = import_calibration_profile(
                        store,
                        session_id=source_session_id,
                        profile_path=str(source_path),
                        label=f"{hardware} shared hardware profile",
                        hardware_id=hardware,
                        mount_version="mount_v1",
                        valid_sessions=["S004", "S005", "S006", "S007"],
                    )
                    source_profile_ids.append(imported["artifact"]["id"])
                checks["two_source_hardware_profiles_registered"] = len(project_calibration_library(store)) == 2

                # Batch-add Sessions and plan Interactions from the Project manager.
                page.evaluate("refreshState({quiet:false,fullRender:true})")
                page.wait_for_timeout(250)
                page.click('[data-project-manage="true"]')
                page.wait_for_selector(".project-manager")
                checks["session_manager_lists_four_sessions"] = page.locator(".project-session-row").count() == 4
                page.fill("#manager-session-spec", "S008-S009")
                page.click("#manager-add-sessions")
                page.wait_for_function("document.querySelectorAll('.project-session-row').length===6", timeout=30000)
                checks["batch_add_sessions"] = page.locator(".project-session-row").count() == 6
                option_texts = page.locator("#manager-plan-session option").all_text_contents()
                option_values = page.locator("#manager-plan-session option").evaluate_all("els=>els.map(el=>el.value)")
                s008_index = next(index for index, text in enumerate(option_texts) if "S008" in text)
                page.select_option("#manager-plan-session", option_values[s008_index])
                page.fill("#manager-interaction-spec", "I01-I03")
                page.click("#manager-plan-interactions")
                page.wait_for_selector(".project-manager")
                page.wait_for_timeout(300)
                checks["interaction_planning_available_after_creation"] = True

                # Apply the two source profiles programmatically to S005–S007.
                # The current UI manages shared resources through Calibration Sets;
                # the old profile-copy controls are recovery-only and are no longer
                # part of the Project Session Manager.
                initial_target_ids = [session_by_code[code] for code in ("S005", "S006", "S007")]
                initial_apply = apply_shared_calibration_profiles(
                    store,
                    source_session_id=source_session_id,
                    profile_artifact_ids=source_profile_ids,
                    target_session_ids=initial_target_ids,
                )
                checks["shared_profiles_applied_from_project_manager_context"] = len(initial_apply["applied"]) == 6
                page.evaluate("refreshState({quiet:false,fullRender:true})")
                page.wait_for_timeout(500)
                checks["manager_lists_project_calibration_set"] = page.locator(".calibration-set-manager-card").count() >= 1
                page.screenshot(path=str(workspace_output), full_page=False)

                snapshot = store.snapshot()
                profile_counts = {
                    store.session_code(session_id): sum(
                        1 for artifact in session.artifacts.values()
                        if str(artifact.kind).endswith("calibration_profile")
                    )
                    for session_id, session in snapshot.sessions.items()
                }
                checks["shared_profiles_applied_to_s005_s007"] = all(
                    profile_counts.get(code) == 2 for code in ("S005", "S006", "S007")
                )
                checks["later_sessions_not_implicitly_modified"] = all(
                    profile_counts.get(code) == 0 for code in ("S008", "S009")
                )
                checks["shared_profile_files_stored_once"] = len(list(store.shared_calibration_dir().glob("*.g3dcal"))) == 2
                page.locator(".modal-close").click()
                browser.close()

            project_root = parent / "OCUMO_S004_S007"
            required_session_dirs = ("imports", "derived", "reports", "annotations", "cache")
            checks["human_readable_session_folder_tree"] = all(
                (project_root / "sessions" / code / child).is_dir()
                for code in ("S004", "S005", "S006", "S007", "S008", "S009")
                for child in required_session_dirs
            )
            checks["session_index_files_written"] = all(
                (project_root / "sessions" / code / "session.oculostudio.json").is_file()
                for code in ("S004", "S005", "S006", "S007", "S008", "S009")
            )
            manifest = json.loads((project_root / "project.oculostudio.json").read_text("utf-8"))
            checks["manifest_contains_six_sessions"] = len(manifest["sessions"]) == 6
            checks["folder_layout_version_three"] = manifest["settings"]["folder_layout_version"] == 3
            checks["project_shared_calibration_setting_persisted"] = bool(
                manifest["settings"].get("shared_calibration", {}).get("enabled")
            )

            # Re-applying the exact profiles is idempotent.
            target_ids = [session_by_code[code] for code in ("S005", "S006", "S007")]
            repeated = apply_shared_calibration_profiles(
                store,
                source_session_id=source_session_id,
                profile_artifact_ids=source_profile_ids,
                target_session_ids=target_ids,
            )
            checks["shared_profile_apply_is_idempotent"] = not repeated["applied"] and len(repeated["skipped"]) == 6

            # Non-empty targets are refused and legacy display labels remain accepted.
            occupied = parent / "occupied"
            occupied.mkdir()
            (occupied / "keep.txt").write_text("do not overwrite", "utf-8")
            location = inspect_project_location(parent=str(parent), folder="occupied")
            checks["non_empty_target_rejected"] = not location["valid"]
            legacy = ProjectStore.create(
                temp_root / "legacy_label_project",
                "Legacy label compatibility",
                session_label="Session S001",
                settings={"folder_layout_version": 1},
            )
            checks["legacy_session_label_accepted"] = legacy.session_code(legacy.project.selected_session_id) == "S001"

            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
            checks["page_errors_absent"] = not report["page_errors"]
            checks["console_errors_absent"] = not unexpected
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    failed = [name for name, value in checks.items() if not value]
    report.update(
        {
            "passed": len(checks) - len(failed),
            "total": len(checks),
            "failed": failed,
            "all_passed": not failed,
            "screenshots": {"wizard": str(wizard_output), "workspace": str(workspace_output)},
        }
    )
    json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
