"""Validate the v0.21.24 dependency-aware artifact cleanup workspace."""
from __future__ import annotations

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
    import oculostudio
    from oculostudio.domain.models import ArtifactKind
    from oculostudio.server import app, runtime
    from oculostudio.storage.artifact_store import ArtifactStore
    from oculostudio.storage.project_store import ProjectStore

    package_root = Path(oculostudio.__file__).resolve().parent
    static = package_root / "static"
    report: dict = {"release": oculostudio.__version__, "checks": [], "page_errors": [], "console_errors": []}
    with tempfile.TemporaryDirectory(prefix="oculo_cleanup_ui_") as temp:
        project_root = Path(temp) / "project"
        store = ProjectStore.create(project_root, "Artifact Cleanup UI", session_label="S001")
        sid = store.project.selected_session_id
        assert sid
        artifacts = ArtifactStore(store)

        old_root = store.derived_dir(sid, "trim", "old")
        (old_root / "trim.csv").write_text("old", "utf-8")
        old = artifacts.register(session_id=sid, kind=ArtifactKind.NATIVE_REFERENCE_TRIM, label="S001_I01_R01_trim", output_root=old_root)

        new_root = store.derived_dir(sid, "trim", "new")
        (new_root / "trim.csv").write_text("new", "utf-8")
        new = artifacts.register(session_id=sid, kind=ArtifactKind.NATIVE_REFERENCE_TRIM, label="S001_I01_R01_trim", output_root=new_root)

        filter_root = store.derived_dir(sid, "face_mapper_filter", "filter")
        (filter_root / "filter.csv").write_text("filter", "utf-8")
        filtered = artifacts.register(
            session_id=sid,
            kind=ArtifactKind.FACE_MAPPER_FILTER,
            label="S001_I01_R01_filter",
            output_root=filter_root,
            parent_artifact_ids=[new.id],
            parameters={"native_trim_artifact_id": new.id},
        )

        privacy_root = store.derived_dir(sid, "privacy", "privacy")
        (privacy_root / "privacy.mp4").write_bytes(b"video")
        artifacts.register(
            session_id=sid,
            kind=ArtifactKind.PRIVACY_CLIP,
            label="S001_I01_R01_privacy",
            output_root=privacy_root,
            parent_artifact_ids=[filtered.id],
        )

        missing_root = store.derived_dir(sid, "face_aoi", "missing")
        (missing_root / "face.csv").write_text("missing", "utf-8")
        artifacts.register(session_id=sid, kind=ArtifactKind.FACE_AOI, label="S001_I01_R01_face_aoi_old", output_root=missing_root)
        (missing_root / "face.csv").unlink()

        orphan_root = store.derived_dir(sid, "face_aoi", "abandoned_run")
        (orphan_root / "partial.tmp").write_text("partial", "utf-8")

        runtime._store = store
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            wait(url)
            with sync_playwright() as pw:
                options = {"headless": True, "args": ["--no-sandbox"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    options["executable_path"] = executable
                browser = pw.chromium.launch(**options)
                page = browser.new_page(viewport={"width": 1645, "height": 920})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html = re.sub(r'<link[^>]+rel="stylesheet"[^>]*>', "", html)
                page.set_content(html)
                page.add_style_tag(content=(static / "styles.css").read_text("utf-8"))

                def relay(path: str, method: str, headers: dict, body: str | None):
                    request = urllib.request.Request(url + path, method=method, headers=headers or {}, data=body.encode() if body else None)
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            return {"status": response.status, "body": response.read().decode(), "type": response.headers.get("Content-Type", "application/json")}
                    except urllib.error.HTTPError as error:
                        return {"status": error.code, "body": error.read().decode(), "type": error.headers.get("Content-Type", "application/json")}

                page.expose_function("relayStudioHTTP", relay)
                page.evaluate("""() => { window.fetch = async (path, init={}) => { const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null); return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}}); }}""")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('[data-page="qc"]')
                page.click('[data-page="qc"]')
                page.wait_for_selector('#artifact-cleanup-root .cleanup-summary-grid')
                page.click('#cleanup-scan-orphans')
                page.wait_for_function("document.querySelectorAll('.orphan-row').length >= 1")
                state = page.evaluate("""() => ({
                  cleanupRows: document.querySelectorAll('.cleanup-row').length,
                  suggested: document.querySelectorAll('.cleanup-badge.suggested').length,
                  blocked: document.querySelectorAll('.cleanup-badge.blocked').length,
                  orphanRows: document.querySelectorAll('.orphan-row').length,
                  overflow: document.documentElement.scrollWidth - window.innerWidth,
                })""")
                assert state["cleanupRows"] >= 5
                assert state["suggested"] >= 2
                assert state["blocked"] >= 1
                assert state["orphanRows"] >= 1
                assert state["overflow"] <= 1
                report["workspace"] = state
                report["checks"] = [
                    "Dependency-aware cleanup table rendered",
                    "Duplicate and missing-file suggestions rendered",
                    "Blocked upstream lineage rendered",
                    "Unreferenced folder scan rendered",
                    "No page-level horizontal overflow",
                ]
                unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
                assert not report["page_errors"], report["page_errors"]
                assert not unexpected, unexpected
                browser.close()
            report["ok"] = True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime._store = None
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
