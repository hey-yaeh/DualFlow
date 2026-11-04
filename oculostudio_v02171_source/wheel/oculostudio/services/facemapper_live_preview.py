"""Live FaceMapper overlay data for the original Native scene video.

The Privacy workspace previously rendered annotated MP4 proxies with OpenCV and
then transcoded them again.  That changed frame timing, colour, sample aspect
ratio and sometimes apparent image scale.  This module instead exposes the
original Native scene video unchanged and a compact frame-indexed overlay
manifest.  The browser draws boxes and landmarks on a canvas at presentation
time using the retained Native PTS clock.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oculostudio import __version__
from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.pupil import materialized_native_files
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json

_SCHEMA = "oculostudio-facemapper-live-overlay-1.0"
_CACHE_REVISION = "live-overlay-v1"


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _column(columns, aliases, *, required: bool = True) -> str | None:
    lookup = {_norm(column): column for column in columns}
    for alias in aliases:
        match = lookup.get(_norm(alias))
        if match is not None:
            return match
    if required:
        raise ValidationError(f"FaceMapper live preview is missing required column: {aliases[0]}")
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:  # pragma: no cover - error text is user-facing
        raise ValidationError(f"Could not read {path.name}: {exc}") from exc


def _filter_context(store: ProjectStore, session_id: str, artifact_id: str):
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
        raise ValidationError("Select a FaceMapper QC/filter artifact")
    filter_manifest_path = find_artifact_file(artifact, names=["filter_manifest.json"])
    filter_root = filter_manifest_path.parent
    filter_manifest = _read_json(filter_manifest_path)
    trim_id = str(filter_manifest.get("native_trim_artifact_id") or artifact.parameters.get("native_trim_artifact_id") or "")
    trim = session.artifacts.get(trim_id)
    if trim is None or trim.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("The FaceMapper filter no longer resolves its Native-reference trim")
    trim_manifest_path = find_artifact_file(trim, names=["trim_manifest.json"])
    trim_root = trim_manifest_path.parent
    trim_manifest = _read_json(trim_manifest_path)
    return session, artifact, filter_root, filter_manifest, trim, trim_root, trim_manifest


def facemapper_live_video_path(store: ProjectStore, session_id: str, artifact_id: str) -> Path:
    session, artifact, _root, manifest, _trim, _trim_root, _trim_manifest = _filter_context(
        store, session_id, artifact_id
    )
    source_ids = manifest.get("source_asset_ids") or {}
    native_id = str(source_ids.get("native") or "")
    native = session.assets.get(native_id)
    if native is None:
        # Compatibility with older filter artifacts that only retained the
        # source-asset list on the artifact record.
        native = next((session.assets.get(asset_id) for asset_id in artifact.source_asset_ids
                       if session.assets.get(asset_id) and session.assets[asset_id].kind == "pupil_native"), None)
    if native is None:
        raise ValidationError("Could not resolve the Native recording used by this FaceMapper filter")
    recording_id = str(manifest.get("recording_id") or native.recording_id or "")
    cache = MaterializationCache(store.cache_dir() / "materialized")
    root = cache.materialize(native)
    video_path, _timestamps, _info, _member = materialized_native_files(root, recording_id)
    if not Path(video_path).exists():
        raise ValidationError("The original Native scene video is unavailable")
    return Path(video_path)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _pixel_pair(row: pd.Series, x_col: str, y_col: str, width: int, height: int) -> list[float] | None:
    x = _finite_float(row.get(x_col))
    y = _finite_float(row.get(y_col))
    if x is None or y is None:
        return None
    if max(abs(x), abs(y)) <= 2.0:
        x *= width
        y *= height
    return [round(x, 3), round(y, 3)]


def _bbox(row: pd.Series, cols: dict[str, str], width: int, height: int) -> list[float] | None:
    values = [_finite_float(row.get(cols[key])) for key in ("x1", "y1", "x2", "y2")]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = [float(value) for value in values]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _source_row_for_audit(
    raw: pd.DataFrame,
    source_index: int,
    timestamp_ns: int | None,
    raw_timestamps: np.ndarray,
) -> pd.Series | None:
    if 0 <= source_index < len(raw):
        row = raw.iloc[source_index]
        if timestamp_ns is None or not len(raw_timestamps):
            return row
        candidate = raw_timestamps[source_index]
        if np.isfinite(candidate) and abs(int(candidate) - int(timestamp_ns)) <= 1_000_000:
            return row
    if timestamp_ns is None or not len(raw_timestamps):
        return None
    finite = np.isfinite(raw_timestamps)
    if not finite.any():
        return None
    valid_indices = np.flatnonzero(finite)
    valid_times = raw_timestamps[finite].astype(np.int64)
    pos = int(np.searchsorted(valid_times, int(timestamp_ns), side="left"))
    options = []
    if pos < len(valid_times):
        options.append(pos)
    if pos > 0:
        options.append(pos - 1)
    if not options:
        return None
    best = min(options, key=lambda idx: abs(int(valid_times[idx]) - int(timestamp_ns)))
    if abs(int(valid_times[best]) - int(timestamp_ns)) > 30_000_000:
        return None
    return raw.iloc[int(valid_indices[best])]


def _build_live_manifest(store: ProjectStore, session_id: str, artifact_id: str) -> dict[str, Any]:
    _session, artifact, root, filter_manifest, _trim, trim_root, trim_manifest = _filter_context(
        store, session_id, artifact_id
    )
    raw_path = root / "source_face_detections_raw.csv"
    audit_path = root / "face_filter_audit.csv"
    frame_map_path = trim_root / "native_frame_map.csv"
    if not raw_path.exists() or not audit_path.exists() or not frame_map_path.exists():
        raise ValidationError(
            "This FaceMapper filter predates live source inspection. Re-run FaceMapper QC/filter once."
        )
    raw = pd.read_csv(raw_path, low_memory=False)
    audit = pd.read_csv(audit_path, low_memory=False)
    frame_map = pd.read_csv(frame_map_path, low_memory=False)
    image = filter_manifest.get("image") or {}
    width, height = int(image.get("width") or 0), int(image.get("height") or 0)
    if width <= 0 or height <= 0:
        native_reference = trim_manifest.get("native_reference") or {}
        width = int(native_reference.get("width") or 0)
        height = int(native_reference.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValidationError("FaceMapper filter has no valid source-video dimensions")

    cols = {
        "timestamp": _column(raw.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"]),
        "x1": _column(raw.columns, ["p1 x [px]", "p1_x_px", "x1", "left"]),
        "y1": _column(raw.columns, ["p1 y [px]", "p1_y_px", "y1", "top"]),
        "x2": _column(raw.columns, ["p2 x [px]", "p2_x_px", "x2", "right"]),
        "y2": _column(raw.columns, ["p2 y [px]", "p2_y_px", "y2", "bottom"]),
        "left_eye_x": _column(raw.columns, ["eye left x [px]", "left eye x [px]", "left_eye_x_px"]),
        "left_eye_y": _column(raw.columns, ["eye left y [px]", "left eye y [px]", "left_eye_y_px"]),
        "right_eye_x": _column(raw.columns, ["eye right x [px]", "right eye x [px]", "right_eye_x_px"]),
        "right_eye_y": _column(raw.columns, ["eye right y [px]", "right eye y [px]", "right_eye_y_px"]),
        "nose_x": _column(raw.columns, ["nose x [px]", "nose_x_px"]),
        "nose_y": _column(raw.columns, ["nose y [px]", "nose_y_px"]),
        "mouth_left_x": _column(raw.columns, ["mouth left x [px]", "left mouth x [px]", "left mouth corner x [px]", "mouth_left_x_px"]),
        "mouth_left_y": _column(raw.columns, ["mouth left y [px]", "left mouth y [px]", "left mouth corner y [px]", "mouth_left_y_px"]),
        "mouth_right_x": _column(raw.columns, ["mouth right x [px]", "right mouth x [px]", "right mouth corner x [px]", "mouth_right_x_px"]),
        "mouth_right_y": _column(raw.columns, ["mouth right y [px]", "right mouth y [px]", "right mouth corner y [px]", "mouth_right_y_px"]),
    }
    raw_timestamps = pd.to_numeric(raw[cols["timestamp"]], errors="coerce").to_numpy(dtype=float)
    swap = bool(filter_manifest.get("bilateral_label_swap_applied"))
    detections: dict[str, list[dict[str, Any]]] = {}
    detection_count = 0
    for _, audit_row in audit.iterrows():
        frame_value = _finite_float(audit_row.get("native_frame_index"))
        if frame_value is None:
            continue
        frame_index = int(round(frame_value))
        source_value = _finite_float(audit_row.get("source_face_row_index"))
        source_index = int(round(source_value)) if source_value is not None else -1
        timestamp_value = _finite_float(audit_row.get("timestamp_ns"))
        timestamp_ns = int(round(timestamp_value)) if timestamp_value is not None else None
        source_row = _source_row_for_audit(raw, source_index, timestamp_ns, raw_timestamps)
        if source_row is None:
            continue
        box = _bbox(source_row, cols, width, height)
        if box is None:
            continue
        source_landmarks = {
            "left_eye": _pixel_pair(source_row, cols["left_eye_x"], cols["left_eye_y"], width, height),
            "right_eye": _pixel_pair(source_row, cols["right_eye_x"], cols["right_eye_y"], width, height),
            "nose": _pixel_pair(source_row, cols["nose_x"], cols["nose_y"], width, height),
            "mouth_left": _pixel_pair(source_row, cols["mouth_left_x"], cols["mouth_left_y"], width, height),
            "mouth_right": _pixel_pair(source_row, cols["mouth_right_x"], cols["mouth_right_y"], width, height),
        }
        source_landmarks = {key: value for key, value in source_landmarks.items() if value is not None}
        canonical = dict(source_landmarks)
        if swap:
            canonical["left_eye"], canonical["right_eye"] = source_landmarks.get("right_eye"), source_landmarks.get("left_eye")
            canonical["mouth_left"], canonical["mouth_right"] = source_landmarks.get("mouth_right"), source_landmarks.get("mouth_left")
            canonical = {key: value for key, value in canonical.items() if value is not None}
        reasons = [token for token in str(audit_row.get("rejection_reasons") or "").split(";") if token]
        row = {
            "track_id": int(round(_finite_float(audit_row.get("track_id")) or -1)),
            "status": str(audit_row.get("filter_status") or "accepted"),
            "box": box,
            "source": source_landmarks,
            "canonical": canonical,
            "reasons": reasons,
        }
        detections.setdefault(str(frame_index), []).append(row)
        detection_count += 1

    required_frame_cols = [
        "native_frame_index", "native_video_pts_ns", "clip_video_pts_ns",
        "retained_display_start_ns", "retained_display_end_ns",
    ]
    for column in required_frame_cols:
        if column not in frame_map.columns:
            raise ValidationError(f"Native frame map is missing {column}")
    frame_map = frame_map.dropna(subset=["native_frame_index", "native_video_pts_ns"]).copy()
    native_indices = pd.to_numeric(frame_map["native_frame_index"], errors="coerce").fillna(-1).astype(int).tolist()
    source_times = (pd.to_numeric(frame_map["native_video_pts_ns"], errors="coerce").fillna(0) / 1e9).round(9).tolist()
    local_times = (pd.to_numeric(frame_map["clip_video_pts_ns"], errors="coerce").fillna(0) / 1e9).round(9).tolist()
    end_local = (pd.to_numeric(frame_map["retained_display_end_ns"], errors="coerce").fillna(0) / 1e9).round(9).tolist()
    if not native_indices:
        raise ValidationError("Native-reference trim has no retained frames")
    trim_window = trim_manifest.get("window") or {}
    start_sec = float(int(trim_window.get("start_media_ns") or round(source_times[0] * 1e9))) / 1e9
    end_sec = float(int(trim_window.get("end_media_ns") or round((source_times[-1] + max(0.0, end_local[-1] - local_times[-1])) * 1e9))) / 1e9
    duration_sec = max(0.0, end_sec - start_sec)
    return {
        "schema_version": _SCHEMA,
        "producer_version": __version__,
        "cache_revision": _CACHE_REVISION,
        "artifact_id": artifact.id,
        "artifact_label": artifact.label,
        "native_trim_artifact_id": filter_manifest.get("native_trim_artifact_id"),
        "recording_id": filter_manifest.get("recording_id"),
        "image": {"width": width, "height": height},
        "video": {
            "source_start_sec": round(start_sec, 9),
            "source_end_sec": round(end_sec, 9),
            "local_duration_sec": round(duration_sec, 9),
            "original_pixels_preserved": True,
            "reencoded": False,
        },
        "timeline": {
            "native_frame_indices": native_indices,
            "source_times_sec": source_times,
            "local_times_sec": local_times,
            "local_frame_end_sec": end_local,
            "frame_count": len(native_indices),
        },
        "detections_by_native_frame": detections,
        "semantics": {
            "source_bilateral_convention": filter_manifest.get("source_bilateral_label_policy"),
            "canonical_bilateral_convention": "subject_anatomical",
            "bilateral_label_swap_applied": swap,
            "geometry_mirror_applied": False,
        },
        "counts": {
            "detection_count": detection_count,
            "frames_with_detections": len(detections),
        },
    }


def facemapper_live_manifest_path(store: ProjectStore, session_id: str, artifact_id: str) -> Path:
    cache_root = store.cache_dir(artifact_id)
    output = cache_root / f"facemapper_live_overlay_{_CACHE_REVISION}.json"
    if not output.exists():
        manifest = _build_live_manifest(store, session_id, artifact_id)
        atomic_write_json(output, manifest)
    return output
