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
    raise RuntimeError("server did not start")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_21_66_ContextLockedAOIReview.png"))
    parser.add_argument("--json", default=str(ROOT / "docs" / "OculoStudio_v0_21_66_ContextLockedAOIReview_Validation.json"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output = Path(args.json).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)

    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.domain.models import ArtifactKind, ArtifactRecord, AssetKind, AssetRecord
    from oculostudio.server import app, runtime
    from oculostudio.storage import ProjectStore

    report = {
        "release": "0.21.71",
        "feature": "Context-locked AOI review gating",
        "checks": {},
        "page_errors": [],
        "console_errors": [],
    }
    checks = report["checks"]

    with tempfile.TemporaryDirectory(prefix="oculo_02165_context_lock_") as temp:
        root = Path(temp) / "project"
        store = ProjectStore.create(root, "Context lock", session_label="S001")
        sid = store.project.selected_session_id
        assert sid

        def mutate(project):
            session = project.sessions[sid]
            # Give the interaction index explicit R01/R02 source identities.
            for role, participant in (("R01", "P01"), ("R02", "P02")):
                for kind, suffix in ((AssetKind.PUPIL_NATIVE, "NRD"), (AssetKind.PUPIL_CLOUD, "TSD"), (AssetKind.FACE_MAPPER, "FM")):
                    row = AssetRecord(
                        label=f"S001_I01_{role}_{suffix}.zip",
                        kind=kind,
                        source_path=f"D:/S001_I01_{role}_{suffix}.zip",
                        participant_id=participant,
                        metadata={"identity": {"session_id": "S001", "interaction_id": "I01", "role_id": role}},
                    )
                    session.assets[row.id] = row

            r01_trim = ArtifactRecord(
                kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
                label="S001_I01_C3D_aligned_clip - R01",
                session_id=sid,
                parameters={"interaction_id": "I01", "role_id": "R01"},
                created_at="2026-11-02T10:00:00+00:00",
            )
            r02_old_trim = ArtifactRecord(
                kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
                label="S001_I01_R02_old_trim",
                session_id=sid,
                parameters={"interaction_id": "I01", "role_id": "R02"},
                created_at="2026-11-02T10:01:00+00:00",
            )
            r02_current_trim = ArtifactRecord(
                kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
                label="S001_I01_C3D_aligned_clip - R02",
                session_id=sid,
                parameters={"interaction_id": "I01", "role_id": "R02"},
                created_at="2026-11-02T10:05:00+00:00",
            )
            for row in (r01_trim, r02_old_trim, r02_current_trim):
                session.artifacts[row.id] = row

            # R01 current artifacts must never leak into the active R02 view.
            r01_face = ArtifactRecord(
                kind=ArtifactKind.FACE_AOI,
                label="S001_I01_R01_face_aoi",
                session_id=sid,
                parent_artifact_ids=[r01_trim.id],
                parameters={"interaction_id": "I01", "wearer_role_id": "R01", "native_trim_artifact_id": r01_trim.id},
                provenance={"native_trim_artifact_id": r01_trim.id, "wearer_role_id": "R01"},
                qc={"review_samples_remaining": 1391, "face_aoi_hit_fraction": 0.506},
                created_at="2026-11-02T10:02:00+00:00",
            )
            r01_body = ArtifactRecord(
                kind=ArtifactKind.BODY_AOI,
                label="S001_I01_R01_body_aoi",
                session_id=sid,
                parent_artifact_ids=[r01_trim.id],
                parameters={"interaction_id": "I01", "wearer_role_id": "R01", "native_trim_artifact_id": r01_trim.id},
                provenance={"native_trim_artifact_id": r01_trim.id, "wearer_role_id": "R01"},
                qc={"review_samples_remaining": 15889, "review_units_remaining": 1451, "body_hit_fraction": 0.543,
                    "body_geometry_gate_passed": True, "body_geometry_review_ready": True, "body_geometry_fusion_ready": True},
                created_at="2026-11-02T10:03:00+00:00",
            )
            # Same role but obsolete trim must also stay hidden.
            r02_old_face = ArtifactRecord(
                kind=ArtifactKind.FACE_AOI,
                label="S001_I01_R02_face_aoi_old_trim",
                session_id=sid,
                parent_artifact_ids=[r02_old_trim.id],
                parameters={"interaction_id": "I01", "wearer_role_id": "R02", "native_trim_artifact_id": r02_old_trim.id},
                provenance={"native_trim_artifact_id": r02_old_trim.id, "wearer_role_id": "R02"},
                qc={"review_samples_remaining": 333, "face_aoi_hit_fraction": 0.444},
                created_at="2026-11-02T10:04:00+00:00",
            )
            r02_old_body = ArtifactRecord(
                kind=ArtifactKind.BODY_AOI,
                label="S001_I01_R02_body_aoi_old_trim",
                session_id=sid,
                parent_artifact_ids=[r02_old_trim.id],
                parameters={"interaction_id": "I01", "wearer_role_id": "R02", "native_trim_artifact_id": r02_old_trim.id},
                provenance={"native_trim_artifact_id": r02_old_trim.id, "wearer_role_id": "R02"},
                qc={"review_samples_remaining": 222, "review_units_remaining": 20, "body_hit_fraction": 0.4,
                    "body_geometry_gate_passed": True, "body_geometry_review_ready": True, "body_geometry_fusion_ready": True},
                created_at="2026-11-02T10:04:30+00:00",
            )
            for row in (r01_face, r01_body, r02_old_face, r02_old_body):
                session.artifacts[row.id] = row

            session.metadata.update({"selected_interaction_id": "I01", "selected_role_id": "R02", "selected_workflow": "scene"})

        store.update(mutate)
        runtime.open(root)

        port = free_port()
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            wait(url)
            import oculostudio
            static = Path(oculostudio.__file__).resolve().parent / "static"
            with sync_playwright() as pw:
                opts = {"headless": True, "args": ["--no-sandbox"]}
                executable = shutil.which("chromium") or shutil.which("google-chrome")
                if executable:
                    opts["executable_path"] = executable
                browser = pw.chromium.launch(**opts)
                page = browser.new_page(viewport={"width": 1680, "height": 1050})
                page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
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
                            return {"status": response.status, "body": response.read().decode("utf-8"), "type": response.headers.get("Content-Type", "application/json")}
                    except urllib.error.HTTPError as error:
                        return {"status": error.code, "body": error.read().decode("utf-8"), "type": error.headers.get("Content-Type", "application/json")}

                page.expose_function("relayStudioHTTP", relay)
                page.evaluate("""() => { window.fetch = async (path, init={}) => {
                    const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);
                    return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}});
                }}""")
                for name in ("native_trim.js", "asset_import.js", "aoi_review.js", "gaze3d_preview.js", "calibration_preview.js", "app.js", "interaction_workspace.js", "motionstudio_bridge.js"):
                    page.add_script_tag(content=(static / name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector("#ui-mode-select")
                page.select_option("#ui-mode-select", "expert")
                page.wait_for_selector('[data-page="targets"]')
                page.click('[data-page="targets"]')
                page.wait_for_selector(".human-review-property-entry")
                page.wait_for_timeout(300)

                body_text = page.locator("body").inner_text()
                checks["active_context_is_r02"] = "I01 / R02" in body_text or "R02" in page.locator(".target-context-lock").inner_text()
                checks["context_lock_names_exact_trim"] = "Context-locked evidence" in body_text and "R02" in page.locator(".target-context-lock").inner_text()
                checks["no_face_review_button_before_current_artifact"] = page.locator("#open-face-human-review-side").count() == 0
                checks["no_body_review_button_before_current_artifact"] = page.locator("#open-body-human-review-side").count() == 0
                checks["face_placeholder_requires_r02"] = "Generate R02 Face AOI first" in page.locator(".human-review-property-card.face").inner_text()
                checks["body_placeholder_requires_r02"] = "Generate R02 Body AOI first" in page.locator(".human-review-property-card.body").inner_text()
                checks["r01_pending_counts_do_not_leak"] = "1391" not in body_text and "15889" not in body_text
                checks["obsolete_r02_trim_counts_do_not_leak"] = "333" not in body_text and "222" not in body_text
                pipeline = page.locator(".target-pipeline").inner_text()
                checks["pipeline_face_not_run"] = "Face AOI · 2D" in pipeline and "Not run" in pipeline
                checks["pipeline_body_not_run"] = "Body AOI · self + partner" in pipeline and pipeline.count("Not run") >= 2
                checks["review_player_face_disabled"] = page.locator("#aoi-review-open-human-review").is_disabled()
                checks["review_player_body_disabled"] = page.locator("#aoi-review-open-body-human-review").is_disabled()
                checks["fusion_face_has_no_cross_context_option"] = page.locator("#fusion-face option").count() == 1
                checks["fusion_body_has_no_cross_context_option"] = page.locator("#fusion-body option").count() == 1

                page.screenshot(path=str(output), full_page=True)
                report["screenshot"] = str(output)
                browser.close()

            unexpected = [row for row in report["console_errors"] if "favicon" not in row.lower()]
            checks["page_errors_absent"] = not report["page_errors"]
            checks["console_errors_absent"] = not unexpected
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            runtime.close()

    failed = [name for name, value in checks.items() if not value]
    report.update({"passed": len(checks) - len(failed), "total": len(checks), "failed": failed, "all_passed": not failed})
    print(json.dumps(report, indent=2, ensure_ascii=False))
    json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), 'utf-8')
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
