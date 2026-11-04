"""Validate v0.21.71 Results & QC role/stage grouping with a live backend."""
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    import oculostudio
    import uvicorn
    from playwright.sync_api import sync_playwright
    from create_results_workspace_demo import build
    from oculostudio.server import app, runtime

    static = Path(oculostudio.__file__).resolve().parent / "static"
    report: dict = {
        "release": oculostudio.__version__,
        "feature": "Results & QC grouped artifact lists",
        "page_errors": [],
        "console_errors": [],
    }
    with tempfile.TemporaryDirectory(prefix="oculo_results_grouped_") as temp:
        project = Path(temp) / "project"
        build(project, overwrite=True)
        runtime.open(project)
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait(url)
            with sync_playwright() as pw:
                options = {"headless": True, "args": ["--no-sandbox"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    options["executable_path"] = executable
                browser = pw.chromium.launch(**options)
                page = browser.new_page(viewport={"width": 1780, "height": 1080})
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
                page.add_style_tag(content=(static / "styles.css").read_text("utf-8"))

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
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('[data-page="qc"]')
                page.click('[data-page="qc"]')
                page.wait_for_selector(".result-role-group")
                page.wait_for_selector(".cleanup-role-group")

                state = page.evaluate(
                    """() => ({
                      browserGroups:[...document.querySelectorAll('.result-role-group>summary b')].map(x=>x.textContent),
                      cleanupGroups:[...document.querySelectorAll('.cleanup-role-group>summary b')].map(x=>x.textContent),
                      packageGroups:[...document.querySelectorAll('.result-package-role-group>summary b')].map(x=>x.textContent),
                      browserStages:document.querySelectorAll('.result-stage-heading').length,
                      cleanupStages:document.querySelectorAll('.cleanup-stage-row').length,
                      packageStages:document.querySelectorAll('.result-package-stage-label').length,
                      artifactRows:document.querySelectorAll('[data-result-artifact]').length,
                      cleanupRows:document.querySelectorAll('.cleanup-row').length,
                      packageChecks:document.querySelectorAll('.result-package-check').length,
                      width:document.documentElement.scrollWidth,
                    })"""
                )
                expected = {"R01", "R02", "Shared / session"}
                assert set(state["browserGroups"]) == expected, state
                assert set(state["cleanupGroups"]) == expected, state
                assert set(state["packageGroups"]) == expected, state
                assert state["browserStages"] >= 8
                assert state["cleanupStages"] >= 8
                assert state["packageStages"] >= 8
                assert state["artifactRows"] >= 10
                assert state["cleanupRows"] >= 10
                assert state["packageChecks"] >= 10
                assert state["width"] <= 1780

                page.click('[data-result-group-scope="browser"][data-result-group-action="collapse"]')
                assert page.locator(".result-role-group[open]").count() == 0
                page.click('[data-result-group-scope="browser"][data-result-group-action="expand"]')
                assert page.locator(".result-role-group[open]").count() == 3

                page.locator(".cleanup-role-group").first.locator(".cleanup-check").first.check()
                page.wait_for_timeout(100)
                assert page.locator(".cleanup-footer strong").inner_text() == "1"

                if args.output:
                    output = Path(args.output).resolve()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    page.evaluate("window.scrollTo(0,0)")
                    page.screenshot(path=str(output), full_page=False)
                browser.close()

                unexpected = [m for m in report["console_errors"] if "favicon" not in m.lower()]
                assert not report["page_errors"], report["page_errors"]
                assert not unexpected, unexpected
                report["workspace"] = state
                report["checks"] = [
                    "R01, R02 and Shared groups rendered in all three lists",
                    "Workflow-stage subgroups rendered",
                    "Expand/collapse controls work",
                    "Artifact and export rows remain present across collapsed groups",
                    "Cleanup selection still works",
                    "No page-level horizontal overflow",
                ]
                report["ok"] = True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
