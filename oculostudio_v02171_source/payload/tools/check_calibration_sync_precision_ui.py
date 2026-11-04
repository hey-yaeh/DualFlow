from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_ROOT = Path(__file__).resolve().parents[3] / "wheel_v2156"
# When installed from the package, fall back to the imported module location.
try:
    import oculostudio
    MODULE_ROOT = Path(oculostudio.__file__).resolve().parent
except Exception:
    MODULE_ROOT = WHEEL_ROOT / "oculostudio"

app = (MODULE_ROOT / "static" / "app.js").read_text("utf-8")
css = (MODULE_ROOT / "static" / "styles.css").read_text("utf-8")
server = (MODULE_ROOT.parent / "oculostudio" / "server.py").read_text("utf-8") if (MODULE_ROOT.parent / "oculostudio" / "server.py").exists() else (MODULE_ROOT / "server.py").read_text("utf-8")
gate = (MODULE_ROOT / "services" / "etm_role_assignment.py").read_text("utf-8")
sync = (MODULE_ROOT / "services" / "auto_synchronization.py").read_text("utf-8")

checks = {
    "version_0_21_58": '0.21.71' in (MODULE_ROOT / '__init__.py').read_text('utf-8'),
    "preview_button": "Preview synchronization evidence" in app,
    "preview_api_called": "/api/time-maps/auto-preview/" in app,
    "preview_endpoint": '@app.get("/api/time-maps/auto-preview/{session_id}")' in server,
    "drift_ci_display": "drift_ci95_ppm" in app and "95%" in app,
    "precision_limited_display": "Precision-limited · not exact zero" in app,
    "co_start_explanation": "Co-start prior detected" in app,
    "holdout_display": "Matched / holdout windows" in app,
    "trace_chart": "drawAutoSyncTrace" in app,
    "offset_chart": "drawAutoSyncOffsets" in app,
    "modal_css": ".sync-audit-modal" in css,
    "subsample_engine": "v3_precision_audited" in sync and "_refine_lag_peak" in sync,
    "leave_one_out": "_leave_one_out_window_validation" in sync and "leave_one_window_out" in sync,
    "finite_drift_interval": "drift_ci95_ppm" in sync and "drift_resolution_ppm" in sync,
    "old_artifact_blocked": "independent precision audit is missing" in gate,
    "production_requires_precision": "precision_validated" in gate and "holdout_window_count" in gate,
}
result = {"release": "0.21.71", "ok": all(checks.values()), "checks": checks}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ok"] else 1)
