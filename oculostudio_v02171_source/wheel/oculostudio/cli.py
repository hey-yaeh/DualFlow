from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import uvicorn

from oculostudio import __version__
from oculostudio.media_tools import media_tools_status, privacy_encoder_preflight, save_tool_settings
from oculostudio.media_runtime import complete_preflight, probe_preflight, install_windows_runtime
from oculostudio.errors import OculoStudioError
from oculostudio.adapters.probe import probe_asset
from oculostudio.desktop import launch
from oculostudio.services.qc import export_qc_report
from oculostudio.storage.project_store import ProjectStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oculostudio",
        description="OculoStudio CleanCore — one coherent project, time graph and processing pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"OculoStudio {__version__}")
    sub = parser.add_subparsers(dest="command")

    launch_p = sub.add_parser("launch", help="Launch the desktop application")
    launch_p.add_argument("--project", help="Project folder or project.oculostudio.json")
    launch_p.add_argument("--browser", action="store_true", help="Use the system browser instead of pywebview")
    launch_p.add_argument("--port", type=int)
    launch_p.add_argument("--debug", action="store_true")

    serve_p = sub.add_parser("serve", help="Serve the local UI/API without opening a window")
    serve_p.add_argument("--project")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8766)
    serve_p.add_argument("--reload", action="store_true")

    create_p = sub.add_parser("create", help="Create a new project")
    create_p.add_argument("root")
    create_p.add_argument("--name")
    create_p.add_argument("--session", default="Session S001")

    inspect_p = sub.add_parser("inspect", help="Probe one source asset")
    inspect_p.add_argument("path")
    inspect_p.add_argument("--no-checksum", action="store_true")

    report_p = sub.add_parser("report", help="Export a session QC report")
    report_p.add_argument("project")
    report_p.add_argument("--session")

    demo_p = sub.add_parser("demo", help="Create a self-contained synthetic demo project")
    demo_p.add_argument("output", nargs="?", default="OculoStudio_Demo")
    demo_p.add_argument("--overwrite", action="store_true")

    native_p = sub.add_parser("native-trim", help="Trim TSD/FM using a Native video start and retained duration")
    native_p.add_argument("--project", required=True, help="New or existing OculoStudio project folder")
    native_p.add_argument("--native", required=True)
    native_p.add_argument("--tsd", required=True)
    native_p.add_argument("--facemapper", required=True)
    native_p.add_argument("--start", required=True, help="Timecode read in the original Native MP4")
    native_p.add_argument("--duration", required=True, help="Retained length measured from the cut start")
    native_p.add_argument("--timecode-fps", type=int, choices=[24,25,30,50,60], default=30,
                          help="Editing timebase for HH:MM:SS:FF only (default: 30); does not change NRD PTS")
    native_p.add_argument("--part", default=None, help="Exact scene MP4 archive member for multipart NRD")
    native_p.add_argument("--label", default="Native data clip")
    native_p.add_argument("--ffprobe", default=None)
    native_p.add_argument("--preview-only", action="store_true")
    native_p.add_argument("--event-policy", choices=["clip","inside"], default="clip")
    ndemo = sub.add_parser("demo-native-trim", help="Create a synthetic 2D-only Native trim demo")
    ndemo.add_argument("output")

    media_p = sub.add_parser("media-tools", help="Check/save machine-local FFmpeg and ffprobe paths")
    media_p.add_argument("--ffmpeg", help="Executable or containing bin folder")
    media_p.add_argument("--ffprobe", help="Executable or containing bin folder")
    media_p.add_argument("--save", action="store_true", help="Save supplied paths; empty string restores auto-discovery")
    media_p.add_argument("--test-privacy", action="store_true", help="Actually encode a tiny H.264/AAC MP4")
    media_p.add_argument("--require-ffprobe", action="store_true", help="Run actual MP4 frame/JSON probe test")
    media_p.add_argument("--install-windows", action="store_true", help="Explicitly install complete pair if absent; may download")
    media_p.add_argument("--archive", help="Offline pinned upstream essentials ZIP (advanced)")
    sub.add_parser("check", help="Check runtime dependencies and clean-core invariants")
    return parser


def _runtime_check() -> dict:
    media = media_tools_status()
    optional = {}
    for name in ("webview", "ezc3d", "pyarrow"):
        try:
            __import__(name)
            optional[name] = True
        except Exception:
            optional[name] = False
    static_root = Path(__file__).with_name("static")
    source_root = Path(__file__).parent
    # CleanCore may document migration from previous applications, but it must
    # not ship their source trees or import bridge/engine namespaces. Check the
    # executable package structure rather than flagging explanatory text.
    prohibited_path_parts = {"integrated_engines", "legacy_bridge", "legacy", "old_gui"}
    legacy_hits = []
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root)
        if prohibited_path_parts.intersection(relative.parts):
            legacy_hits.append({"path": str(relative), "reason": "prohibited CleanCore path"})
    from oculostudio.server import app as server_app

    route_paths = [getattr(route, "path", None) for route in server_app.routes]
    static_files_present = all(
        (static_root / name).exists() for name in ("index.html", "styles.css", "app.js", "native_trim.js", "asset_import.js")
    )
    static_routes_present = "/" in route_paths and "/static" in route_paths
    return {
        "app": "OculoStudio",
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": media["tools"]["ffmpeg"].get("path"),
        "ffprobe": media["tools"]["ffprobe"].get("path"),
        "media_tools": media,
        "static_ui": static_files_present and static_routes_present,
        "static_files_present": static_files_present,
        "static_routes_present": static_routes_present,
        "registered_route_paths": [path for path in route_paths if path in {"/", "/static"}],
        "optional": optional,
        "builtin_c3d_reader": True,
        "long_c3d_recovery": True,
        "legacy_source_or_bridge_hits": legacy_hits,
        "clean_core": not legacy_hits,
        "semantic_targets": True,
        "native_reference_data_trim": True,
        "pretrim_overlay_preview": True,
        "raw_facemapper_source_preview": True,
        "facemapper_preview_variants": ["raw_mapper", "canonical_mapper", "filter_qc", "privacy_mask"],
        "pretrim_overlay_registered_as_artifact": False,
        "native_trim_uses_global_timegraph": False,
        "privacy_clip_from_audited_frame_map": True,
        "face_aoi_sample_and_frame_outputs": True,
        "body_aoi_independent_evidence": True,
        "etm_role_resolution": True,
        "etm_assignment_fail_closed": True,
        "target_fusion_join_key": "gaze_sample_uid",
        "artifact_video_byte_range_streaming": True,
        "densepose_mode": "external_results_import_only",
        "detectron2_required": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "launch"
    if command == "media-tools":
        try:
            if args.install_windows:
                installed = install_windows_runtime(archive_path=args.archive,
                    progress=lambda f,s: print(f"[{f:.0%}] {s}", file=sys.stderr, flush=True))
            else:
                installed = None
            if args.save:
                save_tool_settings({k: v for k, v in {"ffmpeg": args.ffmpeg, "ffprobe": args.ffprobe}.items() if v is not None})
            report = media_tools_status()
            if args.test_privacy:
                report["privacy_preflight"] = privacy_encoder_preflight(args.ffmpeg, ffprobe_path=args.ffprobe)
            if args.require_ffprobe:
                report["probe_preflight"] = probe_preflight(args.ffprobe, ffmpeg_path=args.ffmpeg)
            if installed is not None:
                report["installation"] = installed
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        except OculoStudioError as exc:
            print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            return 1
    if command == "launch":
        return launch(project_path=getattr(args, "project", None), browser=getattr(args, "browser", False), port=getattr(args, "port", None), debug=getattr(args, "debug", False))
    if command == "serve":
        from oculostudio.server import app, runtime
        if args.project:
            runtime.open(args.project)
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
        return 0
    if command == "create":
        root = Path(args.root).expanduser().resolve()
        store = ProjectStore.create(root, args.name or root.name, session_label=args.session)
        print(store.manifest_path)
        return 0
    if command == "inspect":
        print(json.dumps(probe_asset(args.path, checksum=not args.no_checksum), indent=2, ensure_ascii=False))
        return 0
    if command == "report":
        store = ProjectStore.open(args.project)
        session_id = args.session or store.project.selected_session_id
        if not session_id:
            parser.error("The project has no selected session")
        result = export_qc_report(store, session_id)
        print(result["html"])
        return 0
    if command == "demo":
        from oculostudio.demo import create_demo_project
        result = create_demo_project(args.output, overwrite=args.overwrite)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if command == "demo-native-trim":
        from oculostudio.native_trim_demo import create_native_trim_demo
        print(json.dumps(create_native_trim_demo(args.output),indent=2,ensure_ascii=False))
        return 0
    if command == "native-trim":
        from oculostudio.services import add_asset
        from oculostudio.services.native_reference_trim import preview_native_reference_trim, export_native_reference_trim
        root=Path(args.project).expanduser().resolve()
        store=ProjectStore.open(root) if (root/"project.oculostudio.json").exists() else ProjectStore.create(root,"Native-reference data clips")
        sid=store.project.selected_session_id
        ids={}
        for role,path in [("native",args.native),("tsd",args.tsd),("face_mapper",args.facemapper)]:
            canonical=str(Path(path).expanduser().resolve())
            a=next((a for a in store.project.sessions[sid].assets.values() if a.source_path==canonical),None)
            if a is None:a=add_asset(store,session_id=sid,path=canonical,label=role,checksum=False)
            ids[role]=a.id
        progress=lambda p,s: print(f"{100*p:.0f}% {s}",file=sys.stderr,flush=True)
        preview=preview_native_reference_trim(store,session_id=sid,native_asset_id=ids['native'],timeseries_asset_id=ids['tsd'],facemapper_asset_id=ids['face_mapper'],start_timecode=args.start,duration_timecode=args.duration,timecode_fps=args.timecode_fps,label=args.label,native_video_member=args.part,interval_policy=args.event_policy,ffprobe_path=args.ffprobe,progress=progress)
        result=preview if args.preview_only else export_native_reference_trim(store,session_id=sid,preview_id=preview['preview_id'],progress=progress)
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return 0
    if command == "check":
        result = _runtime_check()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["clean_core"] and result["static_ui"] else 1
    parser.print_help()
    return 2


def serve_entry() -> int:
    return main(["serve", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
