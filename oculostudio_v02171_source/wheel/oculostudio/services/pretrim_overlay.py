"""Temporary Face Mapper / AOI / gaze overlay before publishing a Native-reference trim.

The result lives only in the project cache.  It is deliberately not registered as
an immutable Artifact, because its purpose is to let the user change the proposed
start/duration and preview again before exporting trimmed TSD + Face Mapper data.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.native_reference_clock import load_clock, open_member, source_stamp
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.face_semantics import classify_face_point, load_face_semantics
from oculostudio.services.privacy import _mux_audio_and_transcode, _output_writer
from oculostudio.storage import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, sha256_file, utc_now

_SCOPE = Literal["boundaries", "representative", "distributed", "full"]
_LEFT_RIGHT = Literal["anatomical_auto", "as_exported", "swap_eyes"]
_ID = re.compile(r"[a-z0-9_\-]+")


REGION_COLORS = {
    "left_eye": (255, 205, 50),
    "right_eye": (230, 135, 40),
    "nose": (55, 155, 255),
    "mouth": (220, 70, 210),
    "other_face": (70, 235, 230),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _column(columns, aliases, *, required: bool = True):
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
    if required:
        raise ValidationError(f"Preview input is missing required column: {aliases[0]}")
    return None


def _load_plan(store: ProjectStore, session_id: str, preview_id: str) -> dict:
    if not re.fullmatch(r"nativeplan_[a-f0-9]{12}", preview_id):
        raise ValidationError("Invalid Native trim preview id")
    path = store.cache_dir() / "native_trim_plans" / f"{preview_id}.json"
    if not path.is_file():
        raise ValidationError("Alignment preview not found; run Preview alignment & counts first")
    plan = json.loads(path.read_text("utf-8"))
    if plan.get("project_id") != store.project.id or plan.get("session_id") != session_id:
        raise ValidationError("Alignment preview belongs to another project/session")
    for role, source in plan["sources"].items():
        if source_stamp(source) != plan["source_stamps"][role]:
            raise ValidationError(f"{role} source changed after alignment preview; run it again")
    if sha256_file(plan["clock_index"]) != plan["clock_index_sha256"]:
        raise ValidationError("Native clock cache changed after alignment preview")
    return plan


def _remember_overlay(store: ProjectStore, session_id: str, preview_id: str, request: dict, result: dict) -> None:
    """Persist only lightweight cache metadata, keyed by the exact trim plan.

    R01 and R02 have different Native clocks and therefore different preview IDs.
    Keeping a small per-preview history prevents one role's QC preview from
    replacing the other role's player when the user switches roles or restarts
    OculoStudio. The rendered video itself remains disposable project cache.
    """
    entry = {
        "preview_id": preview_id,
        "request": request,
        "result": result,
        "updated_at": utc_now(),
    }

    def mutate(project):
        metadata = project.sessions[session_id].metadata
        history = dict(metadata.get("native_reference_overlay_by_preview") or {})
        history.pop(preview_id, None)
        history[preview_id] = entry
        while len(history) > 24:
            history.pop(next(iter(history)))
        metadata["native_reference_overlay_by_preview"] = history
        metadata["native_reference_overlay_last"] = entry

    store.update(mutate)


def _cache_video_member(store: ProjectStore, source: str, member: str, stamp: dict) -> Path:
    suffix = Path(member).suffix.lower() or ".mp4"
    key = hashlib.sha256(
        json.dumps({"source": str(Path(source).resolve()), "member": member, "stamp": stamp}, sort_keys=True).encode()
    ).hexdigest()[:24]
    target = store.cache_dir() / "native_trim_overlay_sources" / f"{key}{suffix}"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".partial")
    pending.unlink(missing_ok=True)
    with open_member(source, member) as src, pending.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    pending.replace(target)
    return target


def _segments(start: int, end: int, scope: _SCOPE, seconds: float, sample_count: int = 7) -> list[dict]:
    """Build non-overlapping preview windows.

    ``full`` deliberately ignores ``seconds``: it means the entire candidate
    interval, not one over-sized sample.  ``distributed`` places an evenly
    spaced set of short windows across the whole candidate to expose temporal
    drift that a start/middle/end preview can miss.
    """
    if scope not in {"boundaries", "representative", "distributed", "full"}:
        raise ValidationError("Preview scope must be boundaries, representative, distributed or full")
    duration = end - start
    if duration <= 0:
        raise ValidationError("Preview interval is empty")
    if scope == "full":
        return [{"label": "FULL", "start_media_ns": int(start), "end_media_ns": int(end)}]
    if not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise ValidationError("Preview segment length must be between 1 and 60 seconds")
    if not 3 <= int(sample_count) <= 12:
        raise ValidationError("Distributed preview sample count must be between 3 and 12")
    span = int(round(seconds * 1e9))
    if duration <= span:
        rows = [("FULL", start, end)]
    elif scope == "boundaries":
        rows = [("START", start, min(end, start + span)), ("END", max(start, end - span), end)]
    elif scope == "representative":
        mid = start + duration // 2
        rows = [
            ("START", start, min(end, start + span)),
            ("MIDDLE", max(start, mid - span // 2), min(end, mid + span // 2)),
            ("END", max(start, end - span), end),
        ]
    else:
        # Centre short samples uniformly from the start boundary to the end
        # boundary.  The first and final samples remain boundary anchored.
        count = int(sample_count)
        centres = np.linspace(start + span / 2, end - span / 2, count)
        rows = []
        for idx, centre in enumerate(centres):
            left = max(start, int(round(centre - span / 2)))
            right = min(end, left + span)
            left = max(start, right - span)
            label = f"SAMPLE {idx + 1}/{count}"
            rows.append((label, left, right))
    merged: list[list] = []
    for label, left, right in rows:
        if merged and left <= merged[-1][2]:
            merged[-1][0] = merged[-1][0] + "+" + label
            merged[-1][2] = max(merged[-1][2], right)
        else:
            merged.append([label, left, right])
    return [{"label": label, "start_media_ns": int(left), "end_media_ns": int(right)} for label, left, right in merged]


def _nearest_indices(values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(values, queries, side="left")
    pos = np.clip(pos, 0, len(values) - 1)
    left = np.maximum(0, pos - 1)
    choose_left = np.abs(values[left] - queries) <= np.abs(values[pos] - queries)
    return np.where(choose_left, left, pos).astype(np.int64)


def _load_gaze(plan: dict, clock, width: int, height: int, tolerance_ms: float) -> tuple[dict[int, list[dict]], dict]:
    member = plan["selection"].get("gaze_member")
    if not member:
        return {}, {"rows": 0, "mapped": 0, "invalid": 0}
    with open_member(plan["sources"]["tsd"], member) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        frame = pd.read_csv(text, low_memory=False)
        text.detach()
    rid_col = _column(frame.columns, ["recording id", "recording_id"], required=False)
    ts_col = _column(frame.columns, ["timestamp [ns]", "timestamp_ns", "world timestamp [ns]"])
    x_col = _column(frame.columns, ["gaze x [px]", "gaze_x_px", "gaze x", "x"])
    y_col = _column(frame.columns, ["gaze y [px]", "gaze_y_px", "gaze y", "y"])
    if rid_col:
        frame = frame.loc[frame[rid_col].astype(str) == str(plan["recording_id"])].copy()
    timestamps = pd.to_numeric(frame[ts_col], errors="coerce")
    xs = pd.to_numeric(frame[x_col], errors="coerce")
    ys = pd.to_numeric(frame[y_col], errors="coerce")
    valid = timestamps.notna() & xs.notna() & ys.notna()
    invalid = int((~valid).sum())
    timestamps = timestamps.loc[valid].astype("int64").to_numpy()
    xs = xs.loc[valid].astype(float).to_numpy()
    ys = ys.loc[valid].astype(float).to_numpy()
    normalized = (np.abs(xs) <= 2.0) & (np.abs(ys) <= 2.0)
    xs[normalized] *= width
    ys[normalized] *= height
    in_window = (timestamps >= int(plan["window"]["utc_start_ns"])) & (timestamps < int(plan["window"]["utc_end_ns"]))
    timestamps, xs, ys = timestamps[in_window], xs[in_window], ys[in_window]
    if not len(timestamps):
        return {}, {"rows": int(valid.sum()), "mapped": 0, "invalid": invalid}
    indices = _nearest_indices(np.asarray(clock.capture_ns, dtype=np.int64), timestamps)
    errors = np.abs(np.asarray(clock.capture_ns, dtype=np.int64)[indices] - timestamps)
    accepted = errors <= int(round(tolerance_ms * 1e6))
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for timestamp, x, y, index, error, ok in zip(timestamps, xs, ys, indices, errors, accepted):
        if not ok or not np.isfinite([x, y]).all():
            continue
        by_frame[int(index)].append({"timestamp_ns": int(timestamp), "x": float(x), "y": float(y), "error_ns": int(error)})
    return dict(by_frame), {
        "rows": int(valid.sum()), "window_rows": int(len(timestamps)), "mapped": int(accepted.sum()),
        "invalid": invalid, "median_frame_error_ms": float(np.median(errors[accepted]) / 1e6) if accepted.any() else None,
    }


def _draw_overlay(frame: np.ndarray, frame_index: int, source_time_ns: int, segment_label: str,
                  face_result, gaze_by_frame: dict[int, list[dict]], *, include_face_boxes: bool,
                  include_landmarks: bool, include_aoi: bool, include_gaze: bool,
                  preview_label: str = "PRE-TRIM PREVIEW") -> dict:
    instances = face_result.instances_by_frame.get(int(frame_index), [])
    region_rows = []
    if include_aoi and len(face_result.frame_regions):
        region_rows = face_result.frame_regions.loc[
            face_result.frame_regions["scene_frame_index"] == int(frame_index)
        ].to_dict("records")
    if include_face_boxes:
        for item in instances:
            cv2.rectangle(frame, (int(round(item.x1)), int(round(item.y1))),
                          (int(round(item.x2)), int(round(item.y2))), (80, 245, 160), 2, cv2.LINE_AA)
            cv2.putText(frame, f"face {item.track_id}", (int(item.x1), max(18, int(item.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 245, 160), 1, cv2.LINE_AA)
    if include_landmarks:
        for item in instances:
            points = [
                (item.left_eye, "L", REGION_COLORS["left_eye"]),
                (item.right_eye, "R", REGION_COLORS["right_eye"]),
                (item.nose, "N", REGION_COLORS["nose"]),
                (item.mouth_left, "M", REGION_COLORS["mouth"]),
                (item.mouth_right, "M", REGION_COLORS["mouth"]),
            ]
            for point, label, color in points:
                if point is None:
                    continue
                center = (int(round(point.x)), int(round(point.y)))
                cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
                cv2.putText(frame, label, (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, color, 1, cv2.LINE_AA)
    for row in region_rows:
        try:
            geometry = json.loads(row["geometry_json"])
            region = str(row["region"])
            color = REGION_COLORS.get(region, (70, 235, 230))
            center = (int(round(geometry["center_x"])), int(round(geometry["center_y"])))
            axes = (max(1, int(round(geometry["radius_x"]))), max(1, int(round(geometry["radius_y"]))))
            cv2.ellipse(frame, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.putText(frame, region, (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1, cv2.LINE_AA)
        except Exception:
            continue
    hits = Counter()
    gaze_rows = gaze_by_frame.get(int(frame_index), []) if include_gaze else []
    for gaze in gaze_rows:
        point = (int(round(gaze["x"])), int(round(gaze["y"])))
        cv2.circle(frame, point, 7, (45, 255, 90), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 11, (20, 60, 30), 2, cv2.LINE_AA)
        if include_aoi:
            hit = classify_face_point(gaze["x"], gaze["y"], instances)
            if hit:
                hits[hit.region] += 1
    primary = hits.most_common(1)[0][0] if hits else None
    # A dark, compact status bar makes every sampled segment auditable.
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (25, 29, 22), -1)
    sec = source_time_ns / 1e9
    status = f"{preview_label} | {segment_label} | source {sec:0.3f}s | native frame {frame_index}"
    cv2.putText(frame, status, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 244, 226), 1, cv2.LINE_AA)
    detail = f"faces {len(instances)} | gaze {len(gaze_rows)} | AOI {primary or 'none'} | temporary cache only"
    cv2.putText(frame, detail, (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 245, 55), 1, cv2.LINE_AA)
    return {"faces": len(instances), "gaze": len(gaze_rows), "primary_aoi": primary}


def render_pretrim_overlay_preview(
    store: ProjectStore, *, session_id: str, preview_id: str,
    scope: _SCOPE = "representative", segment_seconds: float = 8.0, sample_count: int = 7, output_fps: int = 30,
    include_face_boxes: bool = True, include_landmarks: bool = True,
    include_aoi: bool = True, include_gaze: bool = True,
    left_right_policy: _LEFT_RIGHT = "anatomical_auto", face_tolerance_ms: float = 8.0,
    gaze_frame_tolerance_ms: float = 20.0, ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None, progress=None,
    cancellation: CancellationToken | None = None,
) -> dict:
    token = cancellation or CancellationToken.create()
    if output_fps not in {15, 24, 25, 30, 50, 60}:
        raise ValidationError("Overlay preview FPS must be 15, 24, 25, 30, 50 or 60")
    if left_right_policy not in {"anatomical_auto", "as_exported", "swap_eyes"}:
        raise ValidationError("left_right_policy must be anatomical_auto, as_exported or swap_eyes")
    plan = _load_plan(store, session_id, preview_id)
    automatic_c3d_defined = bool(plan.get("request", {}).get("automatic_c3d_defined"))
    preview_kind = "aligned_clip_qc" if automatic_c3d_defined else "pretrim_candidate"
    request = {
        "scope": scope, "segment_seconds": float(segment_seconds), "sample_count": int(sample_count), "output_fps": int(output_fps),
        "include_face_boxes": bool(include_face_boxes), "include_landmarks": bool(include_landmarks),
        "include_aoi": bool(include_aoi), "include_gaze": bool(include_gaze),
        "left_right_policy": left_right_policy, "face_tolerance_ms": float(face_tolerance_ms),
        "gaze_frame_tolerance_ms": float(gaze_frame_tolerance_ms),
    }
    signature = hashlib.sha256(json.dumps({"preview_id": preview_id, **request}, sort_keys=True).encode()).hexdigest()[:16]
    overlay_id = f"overlay_{signature}"
    root = store.cache_dir() / "native_trim_overlay" / preview_id / overlay_id
    video_out = root / "pretrim_overlay_preview.mp4"
    metadata_path = root / "pretrim_overlay_preview.json"
    if video_out.is_file() and metadata_path.is_file():
        payload = json.loads(metadata_path.read_text("utf-8"))
        payload.update({
            "video_path": str(video_out), "metadata_path": str(metadata_path),
            "output_dir": str(root), "cache_reused": True,
            "preview_kind": preview_kind, "automatic_c3d_defined": automatic_c3d_defined,
        })
        _remember_overlay(store, session_id, preview_id, request, payload)
        return payload
    root.mkdir(parents=True, exist_ok=True)
    emit(progress, 0.02, "Preparing aligned clip QC preview" if automatic_c3d_defined else "Preparing temporary pre-trim overlay preview")
    encoder = privacy_encoder_preflight(ffmpeg_path, preserve_audio=False, preset="veryfast", crf=23,
                                        ffprobe_path=ffprobe_path)
    ffmpeg = encoder["ffmpeg"]["path"]
    clock = load_clock(plan["clock_index"])
    start, end = int(plan["window"]["start_media_ns"]), int(plan["window"]["end_media_ns"])
    segments = _segments(start, end, scope, segment_seconds, sample_count)
    source_video = _cache_video_member(store, plan["sources"]["native"],
                                       plan["selection"]["native"]["video_member"], plan["source_stamps"]["native"])
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode Native scene video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Native scene-video dimensions are unavailable")
    session = store.snapshot().sessions[session_id]
    fm_asset = session.assets[plan["asset_ids"]["face_mapper"]]
    fm_root = MaterializationCache(store.cache_dir() / "materialized").materialize(fm_asset)
    emit(progress, 0.08, "Loading Face Mapper landmarks and preliminary AOI geometry")
    face_result = load_face_semantics(
        fm_root, recording_id=plan["recording_id"], scene_timestamps_ns=np.asarray(clock.capture_ns, dtype=np.int64),
        width=width, height=height, tolerance_ms=face_tolerance_ms, left_right_policy=left_right_policy,
    )
    gaze_by_frame, gaze_diagnostics = _load_gaze(plan, clock, width, height, gaze_frame_tolerance_ms) if include_gaze else ({}, {"rows": 0, "mapped": 0, "invalid": 0})
    silent = root / "pretrim_overlay_preview_silent.mp4"
    writer = _output_writer(silent, width, height, float(output_fps))
    frame_stats = Counter()
    segment_results = []
    total_sample_frames = sum(max(1, int(math.ceil((row["end_media_ns"] - row["start_media_ns"]) / 1e9 * output_fps))) for row in segments)
    total_with_slates = total_sample_frames + len(segments) * max(1, output_fps // 2)
    written = 0
    try:
        for segment_number, segment in enumerate(segments, 1):
            token.raise_if_cancelled()
            # Insert a short slate to make non-contiguous representative samples explicit.
            for _ in range(max(1, output_fps // 2)):
                slate = np.zeros((height, width, 3), dtype=np.uint8)
                title = f"{segment['label']} SAMPLE {segment_number}/{len(segments)}"
                cv2.putText(slate, title, (max(20, width // 12), height // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.7, width / 1200), (225, 248, 47), 2, cv2.LINE_AA)
                subtitle = ("Aligned clip QC preview — verified C3D-defined interval"
                            if automatic_c3d_defined else
                            "Temporary pre-trim overlay — no data has been exported")
                cv2.putText(slate, subtitle,
                            (max(20, width // 12), height // 2 + 32), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.45, width / 1900), (230, 235, 225), 1, cv2.LINE_AA)
                writer.write(slate); written += 1
            step_ns = 1e9 / output_fps
            times = np.arange(segment["start_media_ns"], segment["end_media_ns"], step_ns, dtype=np.float64).astype(np.int64)
            if not len(times):
                times = np.asarray([segment["start_media_ns"]], dtype=np.int64)
            indices = _nearest_indices(np.asarray(clock.media_ns, dtype=np.int64), times)
            first, last = int(indices.min()), int(indices.max())
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            position = 0
            current_frame = first
            rendered = 0
            while position < len(indices):
                token.raise_if_cancelled()
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ProcessingError(f"Native video decode ended while reading frame {current_frame}")
                while position < len(indices) and int(indices[position]) == current_frame:
                    output_frame = frame.copy()
                    stat = _draw_overlay(output_frame, current_frame, int(clock.media_ns[current_frame]), segment["label"],
                                         face_result, gaze_by_frame, include_face_boxes=include_face_boxes,
                                         include_landmarks=include_landmarks, include_aoi=include_aoi, include_gaze=include_gaze,
                                         preview_label=("ALIGNED CLIP QC" if automatic_c3d_defined else "PRE-TRIM PREVIEW"))
                    frame_stats["frames"] += 1
                    frame_stats["face_instances"] += stat["faces"]
                    frame_stats["gaze_samples"] += stat["gaze"]
                    if stat["primary_aoi"]:
                        frame_stats[f"aoi_{stat['primary_aoi']}"] += 1
                    writer.write(output_frame)
                    rendered += 1; written += 1; position += 1
                    if written % 20 == 0:
                        emit(progress, 0.18 + 0.67 * written / max(1, total_with_slates),
                             f"Rendering temporary overlay ({written}/{total_with_slates})")
                current_frame += 1
                if current_frame > last and position < len(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[position])); current_frame = int(indices[position])
            segment_results.append({**segment, "output_frames": rendered,
                                    "source_start_sec": segment["start_media_ns"] / 1e9,
                                    "source_end_sec": segment["end_media_ns"] / 1e9})
    finally:
        cap.release(); writer.release()
    emit(progress, 0.88, "Encoding browser-compatible temporary preview")
    _mux_audio_and_transcode(
        silent, source_video, video_out, source_start_sec=0.0,
        duration_sec=written / float(output_fps), preserve_audio=False,
        crf=23, preset="veryfast", ffmpeg_path=ffmpeg, expected_frames=written,
        cancellation=token, progress=progress,
    )
    silent.unlink(missing_ok=True)
    metadata = {
        "schema_version": "oculostudio-pretrim-overlay-preview-1.1",
        "temporary": True,
        "registered_as_artifact": False,
        "preview_kind": preview_kind,
        "automatic_c3d_defined": automatic_c3d_defined,
        "session_id": session_id,
        "preview_id": preview_id,
        "overlay_id": overlay_id,
        "recording_id": plan["recording_id"],
        "request": request,
        "candidate_window": plan["window"],
        "segments": segment_results,
        "video_width": width,
        "video_height": height,
        "output_fps": output_fps,
        "output_frame_count": written,
        "output_duration_sec": written / float(output_fps),
        "face_mapper_diagnostics": face_result.diagnostics,
        "gaze_diagnostics": gaze_diagnostics,
        "overlay_counts": dict(frame_stats),
        "notice": (
            "Temporary QC cache preview of the already published C3D-defined aligned interval; source files and immutable artifacts are unchanged."
            if automatic_c3d_defined else
            "Temporary cache preview only. Export trimmed TSD + FM to create an immutable processing result."
        ),
        "video_path": str(video_out),
        "metadata_path": str(metadata_path),
        "output_dir": str(root),
        "cache_reused": False,
    }
    atomic_write_json(metadata_path, metadata)
    _remember_overlay(store, session_id, preview_id, request, metadata)
    emit(progress, 1.0, "Aligned clip QC preview is ready" if automatic_c3d_defined else "Pre-trim FM / AOI / gaze overlay preview is ready")
    return metadata


def pretrim_overlay_video_path(store: ProjectStore, session_id: str, preview_id: str, overlay_id: str) -> Path:
    # Validate the plan as well as path components before exposing a cached file.
    _load_plan(store, session_id, preview_id)
    if not _ID.fullmatch(overlay_id) or not overlay_id.startswith("overlay_"):
        raise ValidationError("Invalid overlay preview id")
    path = store.cache_dir() / "native_trim_overlay" / preview_id / overlay_id / "pretrim_overlay_preview.mp4"
    if not path.is_file():
        raise ValidationError("Temporary overlay preview no longer exists; render it again")
    return path
