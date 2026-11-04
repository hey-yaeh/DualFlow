"""Installed-package check for OculoStudio v0.21.71 Privacy preview controls."""
from __future__ import annotations

import json
from pathlib import Path

import oculostudio


def main() -> int:
    root = Path(oculostudio.__file__).resolve().parent
    app_js = (root / "static" / "app.js").read_text("utf-8")
    css = (root / "static" / "styles.css").read_text("utf-8")
    index = (root / "static" / "index.html").read_text("utf-8")
    checks = {
        "installed_version": oculostudio.__version__ == "0.21.71",
        "cache_busting_version": "app.js?v=0.21.71" in index and "styles.css?v=0.21.71" in index,
        "privacy_preview_defaults_to_auto": "privacyPreviewMode: 'auto'" in app_js,
        "dedicated_blurred_privacy_tab": "Blurred Privacy" in app_js and "data-privacy-preview-mode=\"privacy\"" in app_js,
        "final_privacy_outputs_are_separate": "const privacyOutputs=[...full,...clipsOut]" in app_js,
        "auxiliary_outputs_are_separate": "const auxiliaryOutputs=[...maskPreviews,...faceOverlays]" in app_js,
        "svg_media_icons": "function mediaControlIcon(name)" in app_js and "facemapper-control-button" in app_js,
        "fixed_button_geometry": "width:32px; height:32px; min-width:32px; min-height:32px" in css,
        "responsive_control_layout": ".facemapper-live-controls { display:flex" in css and ".facemapper-live-seek { flex:1 1 220px" in css,
    }
    report = {
        "release": oculostudio.__version__,
        "feature": "Dedicated blurred Privacy preview and stable FaceMapper playback controls",
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
