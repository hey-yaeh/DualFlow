"""Live browser regression for Workflow Hub return and paired-role sequencing."""
from __future__ import annotations

import json
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
    import uvicorn
    from playwright.sync_api import sync_playwright

    from oculostudio.domain.models import ArtifactKind, ArtifactRecord, AssetKind, AssetRecord
    from oculostudio.server import app, runtime
    from oculostudio.storage import ProjectStore

    report = {"release": "0.18.2", "checks": [], "page_errors": [], "console_errors": []}

    with tempfile.TemporaryDirectory(prefix="oculo_workflow_return_") as temp:
        root = Path(temp) / "project"
        store = ProjectStore.create(root, "Workflow return", session_label="S001")
        sid = store.project.selected_session_id
        assert sid

        def mutate(project):
            session = project.sessions[sid]
            for role, participant in (("R01", "P01"), ("R02", "P02")):
                rows = [
                    AssetRecord(
                        label=f"S001_I01_{role}_NRD.zip",
                        kind=AssetKind.PUPIL_NATIVE,
                        source_path=f"D:/S001_I01_{role}_NRD.zip",
                        participant_id=participant,
                        metadata={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": role}},
                    ),
                    AssetRecord(
                        label=f"S001_I01_{role}_TSD.zip",
                        kind=AssetKind.PUPIL_CLOUD,
                        source_path=f"D:/S001_I01_{role}_TSD.zip",
                        participant_id=participant,
                        metadata={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": role}},
                    ),
                    AssetRecord(
                        label=f"S001_{role}_FM.zip",
                        kind=AssetKind.FACE_MAPPER,
                        source_path=f"D:/S001_{role}_FM.zip",
                        participant_id=participant,
                        metadata={"identity": {"session_id": "S001", "role_id": role}},
                    ),
                ]
                session.assets.update({row.id: row for row in rows})
            trim = ArtifactRecord(
                kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
                label="S001_I01_R01_trim",
                session_id=sid,
                parameters={"session_id": "S001", "interaction_id": "I01", "role_id": "R01"},
            )
            session.artifacts[trim.id] = trim
            session.metadata.update({"selected_interaction_id": "I01", "selected_role_id": "R01", "selected_workflow": "face"})

        store.update(mutate)
        runtime.open(root / "project.oculostudio.json")

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
                page = browser.new_page(viewport={"width": 1800, "height": 1100})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                static = ROOT / "src" / "oculostudio" / "static"
                import re
                html_text = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html_text = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html_text)
                page.set_content(html_text)
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
                page.evaluate("""() => { window.fetch = async (path, init={}) => {
                  const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);
                  return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}});
                }}""")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js", "interaction_workspace.js", "motionstudio_bridge.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector("#ui-mode-select")

                # The stage-grouped scheduler must pick R02 trim after R01 trim.
                label = page.evaluate("window.OculoInteractionWorkspace.nextRequiredStep().label")
                assert label == "R02: Create retained clip", label
                report["checks"].append("Paired-role scheduler advances to R02 retained clip after R01 trim")

                # Enter Expert Tools / Clips, then use the ribbon that used to be inert.
                page.select_option("#ui-mode-select", "expert")
                page.wait_for_selector('[data-page="clips"]')
                page.click('[data-page="clips"]')
                page.wait_for_selector("[data-iw-back]")
                assert page.locator("[data-iw-back]").is_visible()
                page.click("[data-iw-back]")
                page.wait_for_selector('[data-page="interactions"].active')
                page.wait_for_selector("[data-iw-next]")
                assert "R02: Create retained clip" in page.locator("body").inner_text()
                assert page.evaluate("window.OculoInteractionWorkspace.mode()") == "guided"
                report["checks"].append("Workflow Hub ribbon button returns from Expert Clips to Guided Interactions")
                report["checks"].append("Return refreshes workflow state and displays the R02 next action")
                screenshot = ROOT / "docs" / "OculoStudio_v0_18_2_Workflow_Return.png"
                page.screenshot(path=str(screenshot), full_page=True)
                report["screenshot"] = str(screenshot.relative_to(ROOT))

                browser.close()

            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
            assert not report["page_errors"], report["page_errors"]
            assert not unexpected, unexpected
            report["ok"] = True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    output = ROOT / "docs" / "OculoStudio_v0_18_2_WORKFLOW_RETURN_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
