"""Live Chromium UI validation for aligned facial-expression video export."""
from __future__ import annotations

import argparse
import importlib.util
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(url: str) -> None:
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("OculoStudio test server did not start")


def _relay(route, base_url: str) -> None:
    relative = route.request.url.split("studio.local", 1)[-1]
    headers = {}
    if route.request.headers.get("range"):
        headers["Range"] = route.request.headers["range"]
    req = urllib.request.Request(base_url + relative, method=route.request.method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            response_headers = {
                key: value for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length", "content-range", "accept-ranges"}
            }
            route.fulfill(status=response.status, body=response.read(), headers=response_headers)
    except urllib.error.HTTPError as error:
        route.fulfill(status=error.code, body=error.read())


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "docs" / "OculoStudio_v0_21_63_AlignedFacialVideoExport.png"))
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    panel_output = output.with_name(output.stem + "_Controls.png")
    report = {"release":"0.21.71","page_errors":[],"console_errors":[],"checks":{}}

    import uvicorn
    from playwright.sync_api import sync_playwright
    from oculostudio.services import add_asset, automatic_shared_audio_video_sync, accept_audio_video_sync, export_aligned_facial_video
    from oculostudio.storage import ProjectStore
    from oculostudio.server import app, runtime

    shared = _load("shared_fixture_ui", Path(__file__).parent / "check_shared_audio_video_sync.py")
    export_check = _load("aligned_export_fixture_ui", Path(__file__).parent / "check_aligned_facial_video_export.py")

    with tempfile.TemporaryDirectory(prefix="oculo_aligned_facial_ui_") as temp:
        temp_root=Path(temp); fixture_root=temp_root/"fixture"; fixture_root.mkdir()
        fixture=shared._fixture(fixture_root)
        store=ProjectStore.create(temp_root/"project","Aligned facial UI",session_label="S001")
        sid=store.project.selected_session_id
        facial=add_asset(store,session_id=sid,path=fixture["facial_video"],label="S001_I01_R01_FV.mp4",participant_id="P01",metadata_extra={"identity":{"session_id":"S001","interaction_id":"I01","role_id":"R01","media_role":"facial_expression_video"}})
        native=add_asset(store,session_id=sid,path=fixture["native_zip"],label="S001_I01_R01_NRD_untrimmed.zip",participant_id="P01",metadata_extra={"identity":{"session_id":"S001","interaction_id":"I01","role_id":"R01","media_role":"ego_video"}})
        candidate=automatic_shared_audio_video_sync(store,session_id=sid,source_video_asset_id=facial.id,reference_video_asset_id=native.id,wearer_role_id="R01",maximum_offset_sec=15)["candidate"]
        accepted=accept_audio_video_sync(store,session_id=sid,candidate_id=candidate["candidate_id"],visual_review_confirmed=True,visual_offset_adjustment_ms=0.0)
        trim=export_check._register_trim(store,sid,native,video_member=candidate["reference_video_member"],start_sec=6.0,end_sec=24.0)
        aligned=export_aligned_facial_video(store,session_id=sid,native_trim_artifact_id=trim.id,time_sync_artifact_id=accepted["artifact"]["id"],label="S001_I01_R01_FV_aligned",output_fps=30,preserve_audio=True)
        runtime.open(temp_root/"project")
        port=_free_port(); base=f"http://127.0.0.1:{port}"
        server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port=port,log_level="error"))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        try:
            _wait(base)
            with sync_playwright() as pw:
                opts={"headless":True,"args":["--no-sandbox","--autoplay-policy=no-user-gesture-required"]}
                executable=shutil.which("chromium") or shutil.which("google-chrome")
                if executable:opts["executable_path"]=executable
                browser=pw.chromium.launch(**opts)
                page=browser.new_page(viewport={"width":1880,"height":1180})
                page.on("pageerror",lambda error:report["page_errors"].append(str(error)))
                page.on("console",lambda message:report["console_errors"].append(message.text) if message.type=="error" else None)
                static=Path(__import__("oculostudio").__file__).resolve().parent/"static"
                html=re.sub(r"<script[^>]+src=[^>]+></script>","",(static/"index.html").read_text("utf-8"))
                html=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>',"",html).replace("<head>",'<head><base href="http://studio.local/">',1)
                page.route("http://studio.local/**",lambda route:_relay(route,base))
                page.set_content(html)
                for css_name in ("styles.css","motionstudio_bridge.css","interaction_workspace.css"):
                    if (static/css_name).exists(): page.add_style_tag(content=(static/css_name).read_text("utf-8"))
                def relay_http(path,method,headers,body):
                    req=urllib.request.Request(base+path,method=method,headers=headers or {},data=body.encode() if body is not None else None)
                    try:
                        with urllib.request.urlopen(req,timeout=240) as response:return {"status":response.status,"body":response.read().decode("utf-8"),"type":response.headers.get("Content-Type","application/json")}
                    except urllib.error.HTTPError as error:return {"status":error.code,"body":error.read().decode("utf-8"),"type":error.headers.get("Content-Type","application/json")}
                page.expose_function("relayStudioHTTP",relay_http)
                page.evaluate("""() => { window.fetch = async (path, init={}) => { const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null); return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}}); }}""")
                for name in ("native_trim.js","asset_import.js","aoi_review.js","gaze3d_preview.js","calibration_preview.js","app.js","interaction_workspace.js","motionstudio_bridge.js"):
                    if (static/name).exists(): page.add_script_tag(content=(static/name).read_text("utf-8"))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector("#ui-mode-select")
                page.select_option("#ui-mode-select","expert")
                page.wait_for_selector('[data-page="clips"]')
                page.click('[data-page="clips"]')
                page.wait_for_selector("#native-export-aligned-facial")
                page.wait_for_timeout(800)
                selected=page.locator("#native-facial-sync").input_value()
                report["checks"].update({
                    "aligned_facial_export_section_visible":page.get_by_text("4 · Export aligned facial video",exact=True).count()==1,
                    "accepted_time_sync_auto_selected":selected==accepted["artifact"]["id"],
                    "export_button_enabled":page.locator("#native-export-aligned-facial").is_enabled(),
                    "audio_preservation_checked":page.locator('[data-native-facial-check="facialAudio"]').is_checked(),
                    "formal_player_visible":page.locator("#native-aligned-facial-player").count()==1,
                    "open_latest_button_visible":page.locator("#native-properties-open-facial").count()==1,
                    "central_automatic_card_has_facial_action":page.locator(".native-export-auto-facial").count()==1,
                    "timing_section_renumbered":page.get_by_text("5 · Timing & export rules",exact=True).count()==1,
                })
                page.locator("#native-export-aligned-facial").scroll_into_view_if_needed();page.wait_for_timeout(300)
                page.screenshot(path=str(panel_output),full_page=False)
                page.locator("#native-aligned-facial-card").scroll_into_view_if_needed();page.wait_for_timeout(600)
                page.wait_for_function("document.querySelector(\"#native-aligned-facial-player\") && document.querySelector(\"#native-aligned-facial-player\").readyState >= 1", timeout=15000)
                duration=page.locator("#native-aligned-facial-player").evaluate("video=>video.duration")
                report["checks"]["formal_video_streams"]=isinstance(duration,(int,float)) and duration>10
                page.screenshot(path=str(output),full_page=False)
                browser.close()
            unexpected=[row for row in report["console_errors"] if "favicon" not in row.lower()]
            report["checks"]["page_errors_absent"]=not report["page_errors"]
            report["checks"]["console_errors_absent"]=not unexpected
        finally:
            server.should_exit=True;thread.join(timeout=10);runtime.close()
    failed=sorted(name for name,ok in report["checks"].items() if not ok)
    report["passed_count"]=sum(bool(v) for v in report["checks"].values());report["check_count"]=len(report["checks"]);report["failed"]=failed;report["all_pass"]=not failed
    output.with_suffix(".validation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),"utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if not failed else 1

if __name__=="__main__":raise SystemExit(main())
