"""Live UI validation for OculoStudio v0.21.71 Human Review entry restore.

The test opens a disposable project in Body AOI mode, verifies that Human review
is still discoverable, then uses that entry to open the matching Face AOI queue
and jump to the first pending review frame. Body AOI source files are hashed
before and after to prove that the navigation does not mutate them.
"""
from __future__ import annotations

import argparse
import hashlib
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _artifact_file(artifact, name: str) -> Path:
    for item in artifact.files:
        path = Path(item.path)
        if path.name == name:
            return path
    raise RuntimeError(f"Artifact {artifact.id} has no {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs" / "OculoStudio_v0_21_50_HumanReviewEntryRegression.png"),
    )
    args = parser.parse_args()

    import uvicorn
    import oculostudio
    from create_privacy_aoi_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.domain.models import ArtifactKind
    from oculostudio.server import app, runtime
    from oculostudio.services.aoi_review import review_manifest

    from playwright.sync_api import sync_playwright

    report: dict[str, object] = {
        "release": "0.21.71",
        "feature": "Persistent Face AOI Human review entry from Body AOI",
        "page_errors": [],
        "console_errors": [],
        "checks": {},
    }
    checks: dict[str, bool] = report["checks"]  # type: ignore[assignment]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="oculo_v02147_human_review_") as temp:
        project = Path(temp) / "project"
        build(project, overwrite=True)
        store = runtime.open(project)
        session = store.selected_session()
        face = next(row for row in session.artifacts.values() if row.kind == ArtifactKind.FACE_AOI)
        body = next(row for row in session.artifacts.values() if row.kind == ArtifactKind.BODY_AOI)
        video = next(row for row in session.artifacts.values() if row.kind == ArtifactKind.PRIVACY_CLIP)

        samples_path = _artifact_file(face, "face_aoi_samples.csv")
        body_samples_path = _artifact_file(body, "body_aoi_samples.csv")
        samples = pd.read_csv(samples_path, low_memory=False)
        flagged_indices = list(samples.index[:3])
        for column in ("identity_status", "candidate_entity_id", "candidate_role", "candidate_relation", "target_relation"):
            samples[column] = samples[column].astype(object)
        samples.loc[flagged_indices, "review_required"] = True
        samples.loc[flagged_indices, "review_required_before_auto_resolution"] = True
        samples.loc[flagged_indices, "review_reason_before_auto_resolution"] = "validation_ambiguous_candidate"
        samples.loc[flagged_indices, "assignment_status"] = "ambiguous_candidate"
        samples.loc[flagged_indices, "identity_status"] = "ambiguous_candidate"
        samples.loc[flagged_indices, "include_in_primary_analysis"] = False
        samples.loc[flagged_indices, "candidate_entity_id"] = "P02"
        samples.loc[flagged_indices, "candidate_role"] = "R02"
        samples.loc[flagged_indices, "candidate_relation"] = "partner"
        samples.loc[flagged_indices, "target_relation"] = "ambiguous_partner"
        samples.to_csv(samples_path, index=False)
        face.qc.update({
            "review_required_fraction": float(len(flagged_indices) / len(samples)),
            "review_samples_before_auto_resolution": len(flagged_indices),
            "review_samples_remaining": len(flagged_indices),
            "review_frames_before_auto_resolution": 2,
            "review_frames_remaining": 2,
        })
        store.save()

        manifest = review_manifest(
            store,
            session_id=session.id,
            mode="face",
            artifact_id=face.id,
            video_artifact_id=video.id,
        )
        first_pending = int(manifest["event_frames"]["pending_review"][0])
        body_hash_before = _sha256(body_samples_path)

        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            _wait(url)
            verify_ui_assets(url)
            static = Path(oculostudio.__file__).resolve().parent / "static"
            with sync_playwright() as pw:
                options = {"headless": True, "args": ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    options["executable_path"] = executable
                browser = pw.chromium.launch(**options)
                page = browser.new_page(viewport={"width": 1660, "height": 1050})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))  # type: ignore[index]
                page.on(
                    "console",
                    lambda message: report["console_errors"].append(message.text)  # type: ignore[index]
                    if message.type == "error"
                    else None,
                )

                # This environment blocks Chromium localhost navigation. Load the
                # packaged shell and relay every API/media request to the live local server.
                html = re.sub(r"<script[^>]+src=[^>]+></script>", "", (static / "index.html").read_text("utf-8"))
                html = re.sub(r"<link[^>]+rel=\"stylesheet\"[^>]*>", "", html)
                page.set_content(html)
                page.add_style_tag(content=(static / "styles.css").read_text("utf-8"))

                def relay(path: str, method: str, headers: dict, body_text: str | None):
                    if not path.startswith("/api/"):
                        raise ValueError("Only local OculoStudio API calls are relayed")
                    request = urllib.request.Request(
                        url + path,
                        method=method,
                        headers=headers or {},
                        data=body_text.encode() if body_text is not None else None,
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
                    const r = await window.relayStudioHTTP(String(path), init.method || 'GET', init.headers || {}, init.body || null);
                    return new Response(r.body, {status:r.status, headers:{'Content-Type':r.type}});
                }}"""
                )

                def route_local(route):
                    relative = route.request.url.split("studio.local", 1)[-1]
                    headers = {}
                    if route.request.headers.get("range"):
                        headers["Range"] = route.request.headers["range"]
                    request = urllib.request.Request(url + relative, headers=headers)
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            response_headers = {
                                key: value
                                for key, value in response.headers.items()
                                if key.lower() in {"content-type", "content-length", "content-range", "accept-ranges"}
                            }
                            route.fulfill(status=response.status, body=response.read(), headers=response_headers)
                    except urllib.error.HTTPError as error:
                        route.fulfill(status=error.code, body=error.read())

                page.route("http://studio.local/**", route_local)
                page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "app.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("appState.page='targets'; document.dispatchEvent(new Event('DOMContentLoaded'))")

                page.wait_for_selector("#aoi-review-shell")
                page.select_option("#aoi-review-mode", "body")
                page.wait_for_function("value => document.querySelector('#aoi-review-artifact')?.value === value", arg=body.id)
                page.wait_for_function("document.querySelector('#aoi-review-shell')?.classList.contains('body-review-active')")

                before = page.evaluate(
                    """() => {
                      const entry=document.querySelector('#aoi-human-review-entry');
                      const button=document.querySelector('#aoi-review-open-human-review');
                      const status=document.querySelector('#aoi-human-review-status');
                      return {
                        mode:document.querySelector('#aoi-review-mode')?.value,
                        artifact:document.querySelector('#aoi-review-artifact')?.value,
                        entryVisible:Boolean(entry && getComputedStyle(entry).display!=='none' && entry.getBoundingClientRect().height>0),
                        buttonVisible:Boolean(button && getComputedStyle(button).display!=='none' && button.getBoundingClientRect().height>0),
                        buttonDisabled:Boolean(button?.disabled),
                        buttonText:button?.textContent?.trim(),
                        status:status?.textContent?.trim(),
                      };
                    }"""
                )
                checks["entry_visible_in_body_mode"] = bool(before["entryVisible"] and before["buttonVisible"])
                checks["entry_enabled_with_face_artifact"] = not before["buttonDisabled"]
                checks["pending_count_is_truthful"] = "3 flagged" in before["status"] and "3 pending" in page.locator("#aoi-human-review-count").inner_text()
                checks["body_mode_was_active_before_open"] = before["mode"] == "body" and before["artifact"] == body.id

                page.click("#aoi-review-open-human-review")
                page.wait_for_function("document.querySelector('#aoi-review-mode')?.value === 'face'")
                page.wait_for_function("value => document.querySelector('#aoi-review-artifact')?.value === value", arg=face.id)
                page.wait_for_function("document.querySelector('#face-manual-review')?.hidden === false")
                page.wait_for_function("document.querySelector('#face-review-progress')?.textContent.includes('0 / 3')")
                page.wait_for_function(
                    "document.querySelector('#aoi-review-summary')?.textContent.includes('Video frame')",
                    timeout=60_000,
                )
                page.wait_for_timeout(350)

                after = page.evaluate(
                    """() => ({
                      mode:document.querySelector('#aoi-review-mode')?.value,
                      artifact:document.querySelector('#aoi-review-artifact')?.value,
                      manualHidden:document.querySelector('#face-manual-review')?.hidden,
                      progress:document.querySelector('#face-review-progress')?.textContent?.trim(),
                      status:document.querySelector('#face-review-status')?.textContent?.trim(),
                      badge:document.querySelector('#aoi-review-badge')?.textContent?.trim(),
                      currentFrame:Number((document.querySelector('#aoi-review-summary')?.innerText.match(/Video frame\\s+(\\d+)/)||[])[1]),
                      scrollTop:document.scrollingElement?.scrollTop||0,
                      panelTop:document.querySelector('#face-manual-review')?.getBoundingClientRect().top,
                      pageWidth:document.documentElement.scrollWidth,
                    })"""
                )
                checks["opens_matching_face_artifact"] = after["mode"] == "face" and after["artifact"] == face.id
                checks["manual_panel_is_open"] = after["manualHidden"] is False and "0 / 3" in after["progress"]
                checks["jumps_to_first_pending_frame"] = after["currentFrame"] == first_pending
                checks["manual_queue_reports_pending"] = after["status"] in {"not started", "in progress"}
                checks["responsive_without_horizontal_overflow"] = int(after["pageWidth"]) <= 1660

                page.screenshot(path=str(output), full_page=False)
                report["before_click"] = before
                report["after_click"] = after
                report["first_pending_frame"] = first_pending
                browser.close()

            body_hash_after = _sha256(body_samples_path)
            checks["body_aoi_source_unchanged"] = body_hash_before == body_hash_after
            checks["installed_version"] = oculostudio.__version__ == "0.21.71"
            static_root = Path(oculostudio.__file__).resolve().parent / "static"
            aoi_js = (static_root / "aoi_review.js").read_text("utf-8")
            app_js = (static_root / "app.js").read_text("utf-8")
            workspace_js = (static_root / "interaction_workspace.js").read_text("utf-8")
            checks["persistent_player_entry_is_packaged"] = all(token in aoi_js for token in (
                "aoi-human-review-entry", "openHumanReview", "faceReviewNeedsAttention",
            ))
            checks["expert_panel_and_job_entries_are_packaged"] = all(token in app_js for token in (
                "open-face-human-review-side", "Open Face Human Review", "Review flagged Face AOI",
            ))
            checks["guided_workflow_entry_is_packaged"] = all(token in workspace_js for token in (
                "data-iw-face-review", "Human review required", "Open Face AOI human review",
            ))
            unexpected = [message for message in report["console_errors"] if "favicon" not in message.lower()]  # type: ignore[index]
            checks["no_page_errors"] = not report["page_errors"]  # type: ignore[index]
            checks["no_console_errors"] = not unexpected
            failed = [name for name, passed in checks.items() if not passed]
            report["failed"] = failed
            report["ok"] = not failed
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    validation = output.with_suffix(".validation.json")
    validation.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
