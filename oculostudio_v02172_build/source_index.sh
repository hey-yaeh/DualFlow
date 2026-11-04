#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-oculostudio_v02171_source/wheel/oculostudio}
for file in static/app.js static/asset_import.js static/styles.css server.py services/asset_import.py services/interactions.py services/assets.py storage/project_store.py adapters/probe.py adapters/pupil.py; do
  echo "===== $file ====="
  grep -nE 'function dataView|function dataProperties|bind.*Data|run-import|asset-import|Import current Interaction|Calibration Data|Interaction Data|UploadStartRequest|ImportPathsRequest|api/assets|imports/uploads|app.mount|facemapper|FaceMapper|function render\(|function bind|add_asset\(|probe_asset|class Upload' "$ROOT/$file" | head -n 400 || true
done
