from __future__ import annotations

import io
import json
import math
import shutil
import zipfile
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.pupil import materialized_native_files, materialized_cloud_root
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.media_tools import privacy_encoder_preflight
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.facemapper_canonical import (
    BilateralColumns,
    SourceBilateralConvention,
    canonical_landmarks,
    canonicalize_bilateral_semantics,
)
from oculostudio.services.privacy import _mux_audio_and_transcode, _output_writer
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name


Policy = Literal["privacy_safe", "analysis_strict"]


@dataclass
class Candidate:
    row_index: int
    frame_index: int
    timestamp_ns: int
    x1: float
    y1: float
    x2: float
    y2: float
    area_ratio: float
    aspect_ratio: float
    landmark_span_x_ratio: float
    landmark_span_y_ratio: float
    interocular_ratio: float
    landmark_in_box_fraction: float
    landmark_center_offset: float
    vertical_order_ok: bool
    landmarks: dict[str, tuple[float, float]] | None = None
    track_id: int = -1
    status: str = "accepted"
    reasons: list[str] | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _column(columns, aliases, *, required: bool = True):
    lookup = {_norm(c): c for c in columns}
    for alias in aliases:
        key = _norm(alias)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValidationError(f"Face Mapper filter input is missing required column: {aliases[0]}")
    return None


def _nearest_indices(reference_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    if not len(reference):
        return np.full(len(query), -1, dtype=int), np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
    pos = np.searchsorted(reference, query, side="left")
    right = np.clip(pos, 0, len(reference) - 1)
    left = np.clip(pos - 1, 0, len(reference) - 1)
    left_error = np.abs(query - reference[left])
    right_error = np.abs(reference[right] - query)
    use_right = right_error < left_error
    return np.where(use_right, right, left).astype(int), np.where(use_right, right_error, left_error).astype(np.int64)


def _iou(a: Candidate, b: Candidate) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(a: Candidate, b: Candidate) -> float:
    ax, ay = a.center
    bx, by = b.center
    scale = max(1.0, 0.5 * (a.width + a.height + b.width + b.height))
    return math.hypot(ax - bx, ay - by) / scale


def _assign_tracks(candidates: list[Candidate], maximum_frame_gap: int = 4) -> None:
    by_frame: dict[int, list[Candidate]] = defaultdict(list)
    for item in candidates:
        by_frame[item.frame_index].append(item)
    active: dict[int, Candidate] = {}
    active_frame: dict[int, int] = {}
    next_track = 0
    for frame_index in sorted(by_frame):
        rows = by_frame[frame_index]
        possible: list[tuple[float, int, int]] = []
        for row_index, row in enumerate(rows):
            for track_id, previous in active.items():
                if frame_index - active_frame[track_id] > maximum_frame_gap:
                    continue
                overlap = _iou(row, previous)
                distance = _center_distance(row, previous)
                size_ratio = max(row.area_ratio, previous.area_ratio) / max(1e-9, min(row.area_ratio, previous.area_ratio))
                if overlap >= 0.04 or (distance <= 1.45 and size_ratio <= 4.0):
                    possible.append(((1.0 - overlap) + 0.35 * distance + 0.08 * abs(math.log(size_ratio)), row_index, track_id))
        used_rows: set[int] = set()
        used_tracks: set[int] = set()
        row_to_track: dict[int, int] = {}
        for _, row_index, track_id in sorted(possible):
            if row_index in used_rows or track_id in used_tracks:
                continue
            used_rows.add(row_index)
            used_tracks.add(track_id)
            row_to_track[row_index] = track_id
        for row_index, row in enumerate(rows):
            track_id = row_to_track.get(row_index)
            if track_id is None:
                track_id = next_track
                next_track += 1
            row.track_id = track_id
            active[track_id] = row
            active_frame[track_id] = frame_index


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read {path.name}: {exc}") from exc


def _trim_inputs(store: ProjectStore, session_id: str, artifact_id: str):
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("Select a Native-reference trim artifact")
    manifest_path = find_artifact_file(artifact, names=["trim_manifest.json"])
    root = manifest_path.parent
    manifest = _read_json(manifest_path)
    request = manifest.get("request") or {}
    ids = {
        "native": request.get("native_asset_id"),
        "timeseries": request.get("timeseries_asset_id"),
        "face_mapper": request.get("facemapper_asset_id"),
    }
    for role, asset_id in ids.items():
        if not asset_id or asset_id not in session.assets:
            raise ValidationError(f"Could not resolve the {role} source from the Native-reference trim")
    return artifact, root, manifest, {k: session.assets[v] for k, v in ids.items()}


def _point(row: pd.Series, x_col: str, y_col: str, width: int, height: int) -> tuple[float, float] | None:
    try:
        x, y = float(row[x_col]), float(row[y_col])
    except Exception:
        return None
    if not np.isfinite([x, y]).all():
        return None
    if max(abs(x), abs(y)) <= 2.0:
        x *= width
        y *= height
    return x, y


def _candidate_from_row(
    row: pd.Series,
    *,
    row_index: int,
    frame_index: int,
    timestamp_ns: int,
    width: int,
    height: int,
    cols: dict[str, str],
) -> Candidate | None:
    try:
        coords = [float(row[cols[k]]) for k in ("x1", "y1", "x2", "y2")]
    except Exception:
        return None
    if not np.isfinite(coords).all():
        return None
    if max(abs(v) for v in coords) <= 2.0:
        coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
    x1, y1, x2, y2 = coords
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    points = []
    for name in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
        point = _point(row, cols[f"{name}_x"], cols[f"{name}_y"], width, height)
        if point is not None:
            points.append((name, point))
    if points:
        xs = [p[1][0] for p in points]
        ys = [p[1][1] for p in points]
        span_x = (max(xs) - min(xs)) / max(1.0, bw)
        span_y = (max(ys) - min(ys)) / max(1.0, bh)
        in_box = sum(x1 <= x <= x2 and y1 <= y <= y2 for _, (x, y) in points) / len(points)
        mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
        center_offset = math.hypot(mean_x - (x1 + x2) * 0.5, mean_y - (y1 + y2) * 0.5) / max(1.0, math.hypot(bw, bh))
    else:
        span_x = span_y = in_box = 0.0
        center_offset = 1.0
    named = dict(points)
    if "left_eye" in named and "right_eye" in named:
        interocular = math.hypot(named["left_eye"][0] - named["right_eye"][0], named["left_eye"][1] - named["right_eye"][1]) / max(1.0, bw)
        eye_y = (named["left_eye"][1] + named["right_eye"][1]) * 0.5
    else:
        interocular = 0.0
        eye_y = -math.inf
    nose_y = named.get("nose", (0.0, math.inf))[1]
    mouth_y_values = [named[key][1] for key in ("mouth_left", "mouth_right") if key in named]
    mouth_y = float(np.mean(mouth_y_values)) if mouth_y_values else math.inf
    vertical_order = bool(eye_y < nose_y < mouth_y)
    return Candidate(
        row_index=row_index,
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        area_ratio=(bw * bh) / max(1.0, float(width * height)),
        aspect_ratio=bw / max(1.0, bh),
        landmark_span_x_ratio=float(span_x),
        landmark_span_y_ratio=float(span_y),
        interocular_ratio=float(interocular),
        landmark_in_box_fraction=float(in_box),
        landmark_center_offset=float(center_offset),
        vertical_order_ok=vertical_order,
        landmarks={name: (float(point[0]), float(point[1])) for name, point in points},
        reasons=[],
    )


def _classify_candidates(
    candidates: list[Candidate],
    *,
    large_box_area_ratio: float,
    maximum_area_ratio: float,
    minimum_landmark_span_x_ratio: float,
    minimum_landmark_span_y_ratio: float,
    minimum_interocular_ratio: float,
    minimum_landmark_in_box_fraction: float,
    maximum_landmark_center_offset: float,
    maximum_scale_jump_ratio: float,
    minimum_track_frames: int,
) -> dict[int, dict[str, Any]]:
    by_frame: dict[int, list[Candidate]] = defaultdict(list)
    by_track: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate.frame_index].append(candidate)
        by_track[candidate.track_id].append(candidate)
    track_stats: dict[int, dict[str, Any]] = {}
    for track_id, rows in by_track.items():
        area = np.asarray([r.area_ratio for r in rows], dtype=float)
        track_stats[track_id] = {
            "frame_count": len({r.frame_index for r in rows}),
            "row_count": len(rows),
            "median_area_ratio": float(np.median(area)) if len(area) else 0.0,
            "p95_area_ratio": float(np.percentile(area, 95)) if len(area) else 0.0,
            "median_landmark_span_x_ratio": float(np.median([r.landmark_span_x_ratio for r in rows])) if rows else 0.0,
            "median_landmark_span_y_ratio": float(np.median([r.landmark_span_y_ratio for r in rows])) if rows else 0.0,
        }
    for candidate in candidates:
        reasons: list[str] = []
        review: list[str] = []
        if candidate.area_ratio >= maximum_area_ratio:
            reasons.append("extreme_face_area")
        if candidate.aspect_ratio < 0.30 or candidate.aspect_ratio > 2.8:
            reasons.append("implausible_bbox_aspect")
        if candidate.landmark_in_box_fraction < minimum_landmark_in_box_fraction:
            reasons.append("landmarks_outside_bbox")
        compact = (
            candidate.landmark_span_x_ratio < minimum_landmark_span_x_ratio
            or candidate.landmark_span_y_ratio < minimum_landmark_span_y_ratio
            or candidate.interocular_ratio < minimum_interocular_ratio
        )
        if candidate.area_ratio >= large_box_area_ratio and compact:
            reasons.append("oversized_bbox_compact_landmarks")
        if candidate.landmark_center_offset > maximum_landmark_center_offset and compact:
            reasons.append("landmark_cluster_off_center")
        if not candidate.vertical_order_ok and candidate.area_ratio >= large_box_area_ratio:
            reasons.append("invalid_landmark_vertical_order")
        peers = by_frame[candidate.frame_index]
        if len(peers) > 1:
            peer_areas = sorted((p.area_ratio for p in peers if p is not candidate), reverse=True)
            if peer_areas and candidate.area_ratio / max(1e-9, peer_areas[0]) >= 4.0 and compact:
                reasons.append("dominant_bbox_false_positive")
        stats = track_stats[candidate.track_id]
        median_area = float(stats["median_area_ratio"])
        if median_area > 0 and candidate.area_ratio / median_area > maximum_scale_jump_ratio:
            review.append("temporal_scale_jump")
        if stats["frame_count"] < minimum_track_frames:
            review.append("short_track")
        if candidate.area_ratio >= large_box_area_ratio and not compact:
            review.append("large_face_requires_review")
        # A repeatedly oversized track with compact landmarks is stronger evidence
        # than any one frame, and catches the recurrent giant boxes seen in R02.
        if (
            stats["median_area_ratio"] >= large_box_area_ratio
            and (
                stats["median_landmark_span_x_ratio"] < minimum_landmark_span_x_ratio
                or stats["median_landmark_span_y_ratio"] < minimum_landmark_span_y_ratio
            )
        ):
            reasons.append("oversized_compact_track")
        if reasons:
            candidate.status = "rejected"
            candidate.reasons = sorted(set(reasons + review))
        elif review:
            candidate.status = "review"
            candidate.reasons = sorted(set(review))
        else:
            candidate.status = "accepted"
            candidate.reasons = []
    return track_stats


def _copy_or_filter_sections(source_root: Path, destination: Path, recording_id: str) -> None:
    source = source_root / "sections.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source, low_memory=False)
    rid = _column(frame.columns, ["recording id", "recording_id"], required=False)
    if rid:
        frame = frame.loc[frame[rid].astype(str) == str(recording_id)].copy()
    frame.to_csv(destination / "sections.csv", index=False)


def _recompute_gaze_tables(
    *,
    tsd_root: Path,
    accepted_by_frame: dict[int, list[Candidate]],
    scene_timestamps: np.ndarray,
    output_root: Path,
    recording_id: str,
    width: int,
    height: int,
) -> dict[str, int]:
    gaze_path = tsd_root / "gaze.csv"
    if not gaze_path.exists():
        return {"gaze_rows": 0, "gaze_on_face_rows": 0, "fixation_rows": 0}
    gaze = pd.read_csv(gaze_path, low_memory=False)
    ts_col = _column(gaze.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"])
    x_col = _column(gaze.columns, ["gaze x [px]", "gaze_x_px", "gaze x", "x"])
    y_col = _column(gaze.columns, ["gaze y [px]", "gaze_y_px", "gaze y", "y"])
    rid_col = _column(gaze.columns, ["recording id", "recording_id"], required=False)
    section_col = _column(gaze.columns, ["section id", "section_id"], required=False)
    fixation_col = _column(gaze.columns, ["fixation id", "fixation_id"], required=False)
    timestamps = pd.to_numeric(gaze[ts_col], errors="coerce")
    xs = pd.to_numeric(gaze[x_col], errors="coerce")
    ys = pd.to_numeric(gaze[y_col], errors="coerce")
    valid = timestamps.notna() & xs.notna() & ys.notna()
    indices, errors = _nearest_indices(scene_timestamps, timestamps.fillna(0).astype("int64").to_numpy())
    flags = []
    for ordinal in range(len(gaze)):
        if not valid.iloc[ordinal] or errors[ordinal] > 20_000_000:
            flags.append(False)
            continue
        x, y = float(xs.iloc[ordinal]), float(ys.iloc[ordinal])
        if max(abs(x), abs(y)) <= 2.0:
            x *= width
            y *= height
        boxes = accepted_by_frame.get(int(indices[ordinal]), [])
        flags.append(any(box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2 for box in boxes))
    out = pd.DataFrame({
        "section id": gaze[section_col].astype(str) if section_col else "",
        "recording id": gaze[rid_col].astype(str) if rid_col else recording_id,
        "timestamp [ns]": timestamps.astype("Int64"),
        "gaze on face": flags,
    })
    out.to_csv(output_root / "gaze_on_face.csv", index=False)
    fixation_rows = 0
    source_fix = output_root.parent / "source_fixations_on_face.csv"
    if source_fix.exists():
        fix = pd.read_csv(source_fix, low_memory=False)
        start_col = _column(fix.columns, ["start timestamp [ns]", "start_timestamp_ns"], required=False)
        end_col = _column(fix.columns, ["end timestamp [ns]", "end_timestamp_ns"], required=False)
        flag_col = _column(fix.columns, ["fixation on face", "fixation_on_face"], required=False)
        if start_col and end_col and flag_col:
            gaze_ts = timestamps.fillna(-1).astype("int64").to_numpy()
            flag_array = np.asarray(flags, dtype=bool)
            computed = []
            for _, row in fix.iterrows():
                try:
                    left, right = int(row[start_col]), int(row[end_col])
                    mask = (gaze_ts >= left) & (gaze_ts <= right)
                    computed.append(bool(flag_array[mask].mean() >= 0.5) if mask.any() else False)
                except Exception:
                    computed.append(False)
            fix[flag_col] = computed
            fix.to_csv(output_root / "fixations_on_face.csv", index=False)
            fixation_rows = len(fix)
    return {"gaze_rows": len(gaze), "gaze_on_face_rows": len(out), "fixation_rows": fixation_rows}


def _package_variant(
    *,
    source_root: Path,
    tsd_root: Path,
    output_root: Path,
    detections: pd.DataFrame,
    keep_mask: np.ndarray,
    recording_id: str,
    accepted_by_frame: dict[int, list[Candidate]],
    scene_timestamps: np.ndarray,
    width: int,
    height: int,
    policy: Policy,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    filtered = detections.loc[keep_mask].copy()
    filtered.to_csv(output_root / "face_detections.csv", index=False)
    _copy_or_filter_sections(source_root, output_root, recording_id)
    source_fix = source_root / "fixations_on_face.csv"
    if source_fix.exists():
        shutil.copy2(source_fix, output_root.parent / "source_fixations_on_face.csv")
    counts = _recompute_gaze_tables(
        tsd_root=tsd_root,
        accepted_by_frame=accepted_by_frame,
        scene_timestamps=scene_timestamps,
        output_root=output_root,
        recording_id=recording_id,
        width=width,
        height=height,
    )
    (output_root / "OCULOSTUDIO_FILTERED_FACE_MAPPER.txt").write_text(
        "OculoStudio FaceMapper QC/Filter result. Original Face Mapper data remain in the parent Native-reference trim.\n"
        f"Policy: {policy}. See ../face_filter_audit.csv and ../filter_manifest.json.\n",
        "utf-8",
    )
    return {"rows": len(filtered), **counts}


def _zip_folder(folder: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder).as_posix())


def _draw_raw_facemapper_candidate(frame: np.ndarray, candidate: Candidate) -> None:
    """Draw the FaceMapper row exactly as exported.

    This diagnostic layer deliberately performs no filtering, bilateral-name
    canonicalisation or horizontal mirroring.  It exists so users can compare
    the source FaceMapper geometry against later QC and Face-AOI results.
    """
    box_color = (245, 245, 235)
    cv2.rectangle(
        frame,
        (int(round(candidate.x1)), int(round(candidate.y1))),
        (int(round(candidate.x2)), int(round(candidate.y2))),
        box_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"SOURCE T{candidate.track_id}",
        (int(round(candidate.x1)), max(18, int(round(candidate.y1)) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        box_color,
        1,
        cv2.LINE_AA,
    )
    _draw_source_landmarks(frame, candidate, label_suffix="[src]")


def _draw_source_landmarks(
    frame: np.ndarray,
    candidate: Candidate,
    *,
    label_suffix: str = "[src]",
    draw_guides: bool = True,
) -> int:
    """Draw the five FaceMapper source landmarks without canonicalisation.

    This helper is intentionally shared by the exact-source and filtered-QC
    previews.  Filtering may change candidate status, but it must never hide or
    silently move the source landmark geometry that users are auditing.
    """
    landmarks = candidate.landmarks or {}
    styles = {
        "left_eye": ((80, 245, 255), f"LE{label_suffix}"),
        "right_eye": ((255, 120, 230), f"RE{label_suffix}"),
        "nose": ((70, 235, 245), "N"),
        "mouth_left": ((70, 180, 255), f"ML{label_suffix}"),
        "mouth_right": ((255, 190, 80), f"MR{label_suffix}"),
    }
    if draw_guides:
        for first, second, guide_color in (
            ("left_eye", "right_eye", (150, 150, 110)),
            ("mouth_left", "mouth_right", (150, 120, 150)),
        ):
            if first in landmarks and second in landmarks:
                p1 = tuple(int(round(v)) for v in landmarks[first])
                p2 = tuple(int(round(v)) for v in landmarks[second])
                cv2.line(frame, p1, p2, guide_color, 1, cv2.LINE_AA)
    drawn = 0
    for name, point in landmarks.items():
        if name not in styles:
            continue
        color, label = styles[name]
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 8, (20, 22, 18), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def _draw_canonical_landmarks(
    frame: np.ndarray,
    candidate: Candidate,
    *,
    bilateral_label_swap_applied: bool,
    draw_guides: bool = True,
) -> int:
    """Draw anatomical labels while preserving the exact source coordinates."""
    landmarks = canonical_landmarks(
        candidate.landmarks,
        bilateral_label_swap_applied=bilateral_label_swap_applied,
    )
    styles = {
        "left_eye": ((80, 245, 255), "L-eye[subject]"),
        "right_eye": ((255, 120, 230), "R-eye[subject]"),
        "nose": ((70, 235, 245), "N"),
        "mouth_left": ((70, 180, 255), "L-mouth[subject]"),
        "mouth_right": ((255, 190, 80), "R-mouth[subject]"),
    }
    if draw_guides:
        for first, second, guide_color in (
            ("left_eye", "right_eye", (150, 150, 110)),
            ("mouth_left", "mouth_right", (150, 120, 150)),
        ):
            if first in landmarks and second in landmarks:
                p1 = tuple(int(round(v)) for v in landmarks[first])
                p2 = tuple(int(round(v)) for v in landmarks[second])
                cv2.line(frame, p1, p2, guide_color, 1, cv2.LINE_AA)
    drawn = 0
    for name, point in landmarks.items():
        if name not in styles:
            continue
        color, label = styles[name]
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(frame, center, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 8, (20, 22, 18), 1, cv2.LINE_AA)
        cv2.putText(
            frame, label, (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def _draw_canonical_facemapper_candidate(
    frame: np.ndarray,
    candidate: Candidate,
    *,
    bilateral_label_swap_applied: bool,
) -> int:
    box_color = (225, 248, 47)
    cv2.rectangle(
        frame,
        (int(round(candidate.x1)), int(round(candidate.y1))),
        (int(round(candidate.x2)), int(round(candidate.y2))),
        box_color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"CANON T{candidate.track_id}",
        (int(round(candidate.x1)), max(18, int(round(candidate.y1)) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA,
    )
    return _draw_canonical_landmarks(
        frame, candidate,
        bilateral_label_swap_applied=bilateral_label_swap_applied,
    )


def _draw_filter_candidate(
    frame: np.ndarray,
    candidate: Candidate,
    *,
    bilateral_label_swap_applied: bool,
) -> int:
    """Draw QC status with canonical anatomical landmark semantics."""
    color = (
        (70, 220, 70)
        if candidate.status == "accepted"
        else (0, 190, 255)
        if candidate.status == "review"
        else (40, 40, 235)
    )
    thickness = 3 if candidate.status == "rejected" else 2
    cv2.rectangle(
        frame,
        (int(round(candidate.x1)), int(round(candidate.y1))),
        (int(round(candidate.x2)), int(round(candidate.y2))),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{candidate.status.upper()} T{candidate.track_id}",
        (int(round(candidate.x1)), max(18, int(round(candidate.y1)) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    return _draw_canonical_landmarks(
        frame, candidate,
        bilateral_label_swap_applied=bilateral_label_swap_applied,
    )


def _render_preview(
    *,
    source_video: Path,
    frame_map: pd.DataFrame,
    candidates_by_frame: dict[int, list[Candidate]],
    output_path: Path,
    raw_output_path: Path | None,
    canonical_output_path: Path | None,
    bilateral_label_swap_applied: bool,
    render_raw_source: bool,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    sample_count: int,
    segment_seconds: float,
    output_fps: int,
    preview_scope: Literal["sampled", "full"],
    token: CancellationToken,
    progress,
) -> dict[str, Any]:
    encoder = privacy_encoder_preflight(
        ffmpeg_path,
        preserve_audio=False,
        preset="veryfast",
        crf=23,
        ffprobe_path=ffprobe_path,
    )
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ProcessingError(f"Cannot decode Native video for FaceMapper filter preview: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ProcessingError("Native video dimensions are unavailable")
    native_frames = (
        pd.to_numeric(frame_map["native_frame_index"], errors="coerce")
        .dropna()
        .astype(int)
        .to_numpy()
    )
    if not len(native_frames):
        cap.release()
        raise ValidationError("Native-reference trim contains no retained frames")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    span_frames = max(1, int(round(segment_seconds * source_fps)))
    if preview_scope == "full":
        windows = [("FULL RETAINED INTERVAL", int(native_frames[0]), int(native_frames[-1]) + 1)]
    else:
        centres = np.linspace(
            native_frames[0] + span_frames / 2,
            native_frames[-1] - span_frames / 2,
            max(3, sample_count),
        )
        windows = []
        for idx, centre in enumerate(centres):
            left = max(int(native_frames[0]), int(round(centre - span_frames / 2)))
            right = min(int(native_frames[-1]) + 1, left + span_frames)
            left = max(int(native_frames[0]), right - span_frames)
            windows.append((f"SAMPLE {idx+1}/{len(centres)}", left, right))

    filtered_silent = output_path.with_name(output_path.stem + "_silent.mp4")
    filtered_writer = _output_writer(filtered_silent, width, height, float(output_fps))
    raw_silent = None
    raw_writer = None
    if render_raw_source and raw_output_path is not None:
        raw_silent = raw_output_path.with_name(raw_output_path.stem + "_silent.mp4")
        raw_writer = _output_writer(raw_silent, width, height, float(output_fps))
    canonical_silent = None
    canonical_writer = None
    if render_raw_source and canonical_output_path is not None:
        canonical_silent = canonical_output_path.with_name(canonical_output_path.stem + "_silent.mp4")
        canonical_writer = _output_writer(canonical_silent, width, height, float(output_fps))

    written = 0
    summary = Counter()
    raw_summary = Counter()
    canonical_summary = Counter()
    try:
        for window_index, (label, left, right) in enumerate(windows):
            token.raise_if_cancelled()
            if preview_scope != "full":
                for _ in range(max(1, output_fps // 2)):
                    filtered_slate = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(
                        filtered_slate,
                        label,
                        (max(20, width // 12), height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (225, 248, 47),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        filtered_slate,
                        "FILTER QC + CANONICAL ANATOMICAL LANDMARKS | green accepted | amber review | red rejected",
                        (max(20, width // 12), height // 2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (230, 235, 225),
                        1,
                        cv2.LINE_AA,
                    )
                    filtered_writer.write(filtered_slate)
                    if raw_writer is not None:
                        raw_slate = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(raw_slate, label, (max(20, width // 12), height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (225, 248, 47), 2, cv2.LINE_AA)
                        cv2.putText(raw_slate, "RAW FACEMAPPER | exact exported coordinates and labels", (max(20, width // 12), height // 2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 235, 225), 1, cv2.LINE_AA)
                        raw_writer.write(raw_slate)
                    if canonical_writer is not None:
                        canonical_slate = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(canonical_slate, label, (max(20, width // 12), height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (225, 248, 47), 2, cv2.LINE_AA)
                        cv2.putText(canonical_slate, "CANONICAL ANATOMICAL | SAME PIXELS, BILATERAL NAMES ONLY", (max(20, width // 12), height // 2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 235, 225), 1, cv2.LINE_AA)
                        canonical_writer.write(canonical_slate)
                    written += 1

            target_frames = np.arange(
                left,
                right,
                max(1, int(round(source_fps / output_fps))),
                dtype=int,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frames[0]))
            current = int(target_frames[0])
            pos = 0
            while pos < len(target_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                while pos < len(target_frames) and int(target_frames[pos]) == current:
                    candidates = candidates_by_frame.get(current, [])
                    filtered_frame = frame.copy()
                    frame_landmarks = 0
                    for candidate in candidates:
                        frame_landmarks += _draw_filter_candidate(filtered_frame, candidate, bilateral_label_swap_applied=bilateral_label_swap_applied)
                        summary[candidate.status] += 1
                    summary["landmarks"] += frame_landmarks
                    cv2.putText(
                        filtered_frame,
                        "FILTER QC + CANONICAL ANATOMICAL LANDMARKS | green accepted | amber review | red rejected",
                        (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (225, 248, 47),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        filtered_frame,
                        "PIXEL COORDINATES UNCHANGED | bilateral semantics normalised to subject left/right",
                        (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (235, 235, 230),
                        1,
                        cv2.LINE_AA,
                    )
                    if preview_scope == "full":
                        elapsed = max(
                            0.0,
                            (current - int(native_frames[0])) / max(source_fps, 1e-6),
                        )
                        cv2.putText(
                            filtered_frame,
                            f"FULL QC {elapsed:0.1f}s",
                            (16, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (235, 235, 230),
                            1,
                            cv2.LINE_AA,
                        )
                    filtered_writer.write(filtered_frame)

                    if raw_writer is not None:
                        raw_frame = frame.copy()
                        for candidate in candidates:
                            _draw_raw_facemapper_candidate(raw_frame, candidate)
                            raw_summary["detections"] += 1
                            raw_summary["landmarks"] += len(candidate.landmarks or {})
                        cv2.putText(
                            raw_frame,
                            "RAW FACEMAPPER | exact source geometry and source bilateral names",
                            (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.56,
                            (225, 248, 47),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            raw_frame,
                            "NO FILTER | NO MIRROR | LE/RE/ML/MR ARE SOURCE-EXPORTED NAMES",
                            (16, 56),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.49,
                            (235, 235, 230),
                            1,
                            cv2.LINE_AA,
                        )
                        if preview_scope == "full":
                            elapsed = max(
                                0.0,
                                (current - int(native_frames[0])) / max(source_fps, 1e-6),
                            )
                            cv2.putText(
                                raw_frame,
                                f"FULL RAW {elapsed:0.1f}s",
                                (16, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.50,
                                (235, 235, 230),
                                1,
                                cv2.LINE_AA,
                            )
                        raw_writer.write(raw_frame)
                    if canonical_writer is not None:
                        canonical_frame = frame.copy()
                        for candidate in candidates:
                            canonical_summary["detections"] += 1
                            canonical_summary["landmarks"] += _draw_canonical_facemapper_candidate(
                                canonical_frame, candidate,
                                bilateral_label_swap_applied=bilateral_label_swap_applied,
                            )
                        cv2.putText(canonical_frame, "CANONICAL FACEMAPPER | source pixels preserved", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 248, 47), 2, cv2.LINE_AA)
                        cv2.putText(canonical_frame, "ONLY bilateral semantic names converted to subject anatomical left/right", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 230), 1, cv2.LINE_AA)
                        if preview_scope == "full":
                            elapsed = max(0.0, (current - int(native_frames[0])) / max(source_fps, 1e-6))
                            cv2.putText(canonical_frame, f"FULL CANONICAL {elapsed:0.1f}s", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 230), 1, cv2.LINE_AA)
                        canonical_writer.write(canonical_frame)
                    written += 1
                    pos += 1
                current += 1
            emit(
                progress,
                0.72 + 0.13 * (window_index + 1) / max(1, len(windows)),
                "Rendering raw FaceMapper and filtered QC previews",
            )
    finally:
        cap.release()
        filtered_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        if canonical_writer is not None:
            canonical_writer.release()

    _mux_audio_and_transcode(
        filtered_silent,
        source_video,
        output_path,
        source_start_sec=0.0,
        duration_sec=written / float(output_fps),
        preserve_audio=False,
        crf=23,
        preset="veryfast",
        ffmpeg_path=encoder["ffmpeg"]["path"],
        expected_frames=written,
        cancellation=token,
        progress=progress,
    )
    filtered_silent.unlink(missing_ok=True)

    raw_rendered = False
    if raw_silent is not None and raw_output_path is not None:
        emit(progress, 0.92, "Encoding exact raw FaceMapper source preview")
        _mux_audio_and_transcode(
            raw_silent,
            source_video,
            raw_output_path,
            source_start_sec=0.0,
            duration_sec=written / float(output_fps),
            preserve_audio=False,
            crf=23,
            preset="veryfast",
            ffmpeg_path=encoder["ffmpeg"]["path"],
            expected_frames=written,
            cancellation=token,
            progress=progress,
        )
        raw_silent.unlink(missing_ok=True)
        raw_rendered = True

    canonical_rendered = False
    if canonical_silent is not None and canonical_output_path is not None:
        emit(progress, 0.95, "Encoding canonical anatomical FaceMapper preview")
        _mux_audio_and_transcode(
            canonical_silent, source_video, canonical_output_path,
            source_start_sec=0.0, duration_sec=written / float(output_fps),
            preserve_audio=False, crf=23, preset="veryfast",
            ffmpeg_path=encoder["ffmpeg"]["path"], expected_frames=written,
            cancellation=token, progress=progress,
        )
        canonical_silent.unlink(missing_ok=True)
        canonical_rendered = True

    return {
        "scope": preview_scope,
        "sample_count": len(windows) if preview_scope != "full" else None,
        "output_frames": written,
        "output_fps": output_fps,
        "draw_counts": dict(summary),
        "retained_first_native_frame": int(native_frames[0]),
        "retained_last_native_frame": int(native_frames[-1]),
        "filtered_qc_preview": {
            "filename": output_path.name,
            "geometry": "source_pixels_preserved",
            "bilateral_labels": "subject_anatomical",
            "bilateral_label_swap_applied": bool(bilateral_label_swap_applied),
            "geometry_mirror_applied": False,
            "filter_status_overlay": True,
            "landmark_overlay": True,
            "draw_counts": dict(summary),
        },
        "canonical_source_preview": {
            "rendered": canonical_rendered,
            "filename": canonical_output_path.name if canonical_output_path is not None else None,
            "geometry": "exact_source_pixels",
            "bilateral_labels": "subject_anatomical",
            "bilateral_label_swap_applied": bool(bilateral_label_swap_applied),
            "geometry_mirror_applied": False,
            "filtering_applied": False,
            "draw_counts": dict(canonical_summary),
        },
        "raw_source_preview": {
            "rendered": raw_rendered,
            "filename": raw_output_path.name if raw_output_path is not None else None,
            "geometry": "exact_source_export",
            "bilateral_labels": "exact_source_export",
            "filtering_applied": False,
            "canonicalisation_applied": False,
            "draw_counts": dict(raw_summary),
        },
    }

def _html_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    reasons = "".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in sorted(manifest["rejection_reasons"].items())) or "<tr><td>None</td><td>0</td></tr>"
    return f'''<!doctype html><html><meta charset="utf-8"><title>FaceMapper QC & Filter</title>
<style>body{{font:15px/1.55 system-ui;max-width:1050px;margin:35px auto;color:#262a20}}.card{{border:1px solid #d8d8c8;border-radius:16px;padding:18px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}code{{background:#f2f2e9;padding:2px 5px}}</style>
<h1>OculoStudio — FaceMapper QC & Filter</h1><p>Filtering occurs after the Native-reference trim and before Privacy. Original Face Mapper rows remain unchanged in the parent artifact.</p>
<div class="card"><b>Recording:</b> <code>{manifest['recording_id']}</code><br><b>Total detections:</b> {counts['total']}<br><b>Accepted:</b> {counts['accepted']}<br><b>Review (kept for privacy):</b> {counts['review']}<br><b>Rejected:</b> {counts['rejected']}<br><b>Source bilateral convention:</b> {manifest.get('source_bilateral_label_policy','—')}<br><b>Canonical convention:</b> {manifest.get('canonical_bilateral_label_policy','—')}<br><b>Bilateral names exchanged:</b> {manifest.get('bilateral_label_swap_applied',False)}<br><b>Geometry mirrored:</b> {manifest.get('geometry_mirror_applied',False)}</div>
<h2>Canonical semantics</h2><p>The exact source pixel coordinates are preserved. OculoStudio converts only bilateral landmark names to the observed person's anatomical left/right. The immutable source table and canonical table are both retained.</p>
<h2>Policies</h2><ul><li><b>Privacy-safe</b>: accepted + review candidates. Rejects only high-confidence false positives.</li><li><b>Analysis-strict</b>: accepted candidates only.</li></ul>
<h2>Rejection reasons</h2><table><tr><th>Reason</th><th>Count</th></tr>{reasons}</table>
<h2>Important</h2><p>This step filters implausible FaceMapper candidates; it does not identify the conversation partner. Participant Association later performs 3-D partner/self/other identity assignment.</p></html>'''


def create_filtered_facemapper(
    store: ProjectStore,
    *,
    session_id: str,
    native_trim_artifact_id: str,
    label: str = "Filtered Face Mapper",
    large_box_area_ratio: float = 0.075,
    maximum_area_ratio: float = 0.45,
    minimum_landmark_span_x_ratio: float = 0.16,
    minimum_landmark_span_y_ratio: float = 0.22,
    minimum_interocular_ratio: float = 0.10,
    minimum_landmark_in_box_fraction: float = 0.80,
    maximum_landmark_center_offset: float = 0.36,
    maximum_scale_jump_ratio: float = 3.5,
    minimum_track_frames: int = 3,
    render_preview: bool = True,
    render_source_preview: bool = True,
    source_bilateral_convention: SourceBilateralConvention = "pupil_image_side",
    preview_sample_count: int = 9,
    preview_segment_seconds: float = 6.0,
    preview_fps: int = 15,
    preview_scope: Literal["sampled", "full"] = "sampled",
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    if preview_scope not in {"sampled", "full"}:
        raise ValidationError("preview_scope must be sampled or full")
    if not 0.01 <= large_box_area_ratio <= 0.5:
        raise ValidationError("large_box_area_ratio must be between 0.01 and 0.5")
    artifact, trim_root, manifest, sources = _trim_inputs(store, session_id, native_trim_artifact_id)
    recording_id = str(manifest.get("recording_id") or "")
    native_meta = manifest.get("native_reference") or {}
    width = int(native_meta.get("width") or 0)
    height = int(native_meta.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValidationError("Trim manifest has no valid Native video dimensions")
    cache = MaterializationCache(store.cache_dir() / "materialized")
    fm_source_root = cache.materialize(sources["face_mapper"])
    tsd_source_root = materialized_cloud_root(cache.materialize(sources["timeseries"]), recording_id)
    native_root = cache.materialize(sources["native"])
    source_video, _, _, _ = materialized_native_files(native_root, recording_id)
    detection_path = next((p for p in fm_source_root.rglob("face_detections.csv")), None)
    if detection_path is None:
        detection_path = next((p for p in fm_source_root.rglob("face_positions.csv")), None)
    if detection_path is None:
        raise ValidationError("Face Mapper source contains no face_detections.csv or face_positions.csv")
    detections = pd.read_csv(detection_path, low_memory=False)
    cols = {
        "rid": _column(detections.columns, ["recording id", "recording_id"], required=False),
        "timestamp": _column(detections.columns, ["timestamp [ns]", "timestamp_ns"]),
        "x1": _column(detections.columns, ["p1 x [px]", "p1_x_px", "x1", "left"]),
        "y1": _column(detections.columns, ["p1 y [px]", "p1_y_px", "y1", "top"]),
        "x2": _column(detections.columns, ["p2 x [px]", "p2_x_px", "x2", "right"]),
        "y2": _column(detections.columns, ["p2 y [px]", "p2_y_px", "y2", "bottom"]),
        "left_eye_x": _column(detections.columns, ["eye left x [px]", "left eye x [px]", "left_eye_x_px"]),
        "left_eye_y": _column(detections.columns, ["eye left y [px]", "left eye y [px]", "left_eye_y_px"]),
        "right_eye_x": _column(detections.columns, ["eye right x [px]", "right eye x [px]", "right_eye_x_px"]),
        "right_eye_y": _column(detections.columns, ["eye right y [px]", "right eye y [px]", "right_eye_y_px"]),
        "nose_x": _column(detections.columns, ["nose x [px]", "nose_x_px"]),
        "nose_y": _column(detections.columns, ["nose y [px]", "nose_y_px"]),
        "mouth_left_x": _column(detections.columns, ["mouth left x [px]", "left mouth x [px]", "left mouth corner x [px]", "mouth left corner x [px]", "mouth_left_x_px"]),
        "mouth_left_y": _column(detections.columns, ["mouth left y [px]", "left mouth y [px]", "left mouth corner y [px]", "mouth left corner y [px]", "mouth_left_y_px"]),
        "mouth_right_x": _column(detections.columns, ["mouth right x [px]", "right mouth x [px]", "right mouth corner x [px]", "mouth right corner x [px]", "mouth_right_x_px"]),
        "mouth_right_y": _column(detections.columns, ["mouth right y [px]", "right mouth y [px]", "right mouth corner y [px]", "mouth right corner y [px]", "mouth_right_y_px"]),
    }
    if cols["rid"]:
        selected = detections.loc[detections[cols["rid"]].astype(str) == recording_id].copy()
    else:
        selected = detections.copy()
    emit(progress, 0.05, "Loading FaceMapper candidates and Native frame clock")
    full_clock = pd.read_csv(trim_root / "native_clock_full.csv", low_memory=False)
    scene_timestamps = pd.to_numeric(full_clock["native_capture_utc_ns"], errors="raise").astype("int64").to_numpy()
    timestamps = pd.to_numeric(selected[cols["timestamp"]], errors="coerce")
    valid_timestamp = timestamps.notna()
    selected = selected.loc[valid_timestamp].copy()
    timestamps = timestamps.loc[valid_timestamp].astype("int64").to_numpy()
    bilateral_columns = BilateralColumns.from_mapping(cols)
    canonical_selected, bilateral_audit = canonicalize_bilateral_semantics(
        selected, columns=bilateral_columns, source_convention=source_bilateral_convention,
    )
    frame_indices, frame_errors = _nearest_indices(scene_timestamps, timestamps)
    candidates: list[Candidate] = []
    invalid_rows: list[int] = []
    for ordinal, ((source_index, row), frame_index, frame_error, timestamp) in enumerate(zip(selected.iterrows(), frame_indices, frame_errors, timestamps)):
        token.raise_if_cancelled()
        candidate = _candidate_from_row(row, row_index=int(source_index), frame_index=int(frame_index), timestamp_ns=int(timestamp), width=width, height=height, cols=cols)
        if candidate is None or frame_error > 30_000_000:
            invalid_rows.append(int(source_index))
            continue
        candidates.append(candidate)
    _assign_tracks(candidates)
    track_stats = _classify_candidates(
        candidates,
        large_box_area_ratio=large_box_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
        minimum_landmark_span_x_ratio=minimum_landmark_span_x_ratio,
        minimum_landmark_span_y_ratio=minimum_landmark_span_y_ratio,
        minimum_interocular_ratio=minimum_interocular_ratio,
        minimum_landmark_in_box_fraction=minimum_landmark_in_box_fraction,
        maximum_landmark_center_offset=maximum_landmark_center_offset,
        maximum_scale_jump_ratio=maximum_scale_jump_ratio,
        minimum_track_frames=minimum_track_frames,
    )
    emit(progress, 0.28, "Classifying FaceMapper tracks and repeated false-positive boxes")
    by_source_index = {c.row_index: c for c in candidates}
    audit_rows = []
    reason_counts = Counter()
    for source_index, row in selected.iterrows():
        candidate = by_source_index.get(int(source_index))
        if candidate is None:
            status, reasons = "rejected", ["invalid_or_unmatched_detection"]
            frame_index = None
            track_id = None
            metrics = {}
        else:
            status, reasons = candidate.status, candidate.reasons or []
            frame_index = candidate.frame_index
            track_id = candidate.track_id
            metrics = {
                "bbox_area_ratio": candidate.area_ratio,
                "bbox_aspect_ratio": candidate.aspect_ratio,
                "landmark_span_x_ratio": candidate.landmark_span_x_ratio,
                "landmark_span_y_ratio": candidate.landmark_span_y_ratio,
                "interocular_ratio": candidate.interocular_ratio,
                "landmark_in_box_fraction": candidate.landmark_in_box_fraction,
                "landmark_center_offset": candidate.landmark_center_offset,
                "vertical_order_ok": candidate.vertical_order_ok,
            }
        reason_counts.update(reasons)
        audit_rows.append({
            "source_face_row_index": int(source_index),
            "timestamp_ns": int(row[cols["timestamp"]]) if pd.notna(row[cols["timestamp"]]) else None,
            "native_frame_index": frame_index,
            "track_id": track_id,
            "filter_status": status,
            "rejection_reasons": ";".join(reasons),
            "keep_privacy_safe": status in {"accepted", "review"},
            "keep_analysis_strict": status == "accepted",
            "source_bilateral_convention": bilateral_audit["resolved_source_convention"],
            "canonical_bilateral_convention": "subject_anatomical",
            "bilateral_label_swap_applied": bool(bilateral_audit["bilateral_label_swap_applied"]),
            "geometry_mirror_applied": False,
            **metrics,
        })
    audit = pd.DataFrame(audit_rows)
    status_by_index = dict(zip(audit["source_face_row_index"], audit["filter_status"]))
    privacy_mask = selected.index.to_series().map(lambda idx: status_by_index.get(int(idx)) in {"accepted", "review"}).to_numpy()
    strict_mask = selected.index.to_series().map(lambda idx: status_by_index.get(int(idx)) == "accepted").to_numpy()
    safe_by_frame: dict[int, list[Candidate]] = defaultdict(list)
    strict_by_frame: dict[int, list[Candidate]] = defaultdict(list)
    all_by_frame: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        all_by_frame[candidate.frame_index].append(candidate)
        if candidate.status in {"accepted", "review"}:
            safe_by_frame[candidate.frame_index].append(candidate)
        if candidate.status == "accepted":
            strict_by_frame[candidate.frame_index].append(candidate)
    artifact_id = new_id("artifact")
    staging = store.derived_dir(session_id, "facemapper_filter") / (".partial_" + artifact_id)
    final = store.derived_dir(session_id, "facemapper_filter") / artifact_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        audit.to_csv(staging / "face_filter_audit.csv", index=False)
        tracks = []
        for track_id, stats in sorted(track_stats.items()):
            rows = [c for c in candidates if c.track_id == track_id]
            counts = Counter(c.status for c in rows)
            tracks.append({"track_id": track_id, **stats, **{f"{key}_count": value for key, value in counts.items()}})
        pd.DataFrame(tracks).to_csv(staging / "face_filter_tracks.csv", index=False)
        # Preserve the exact source FaceMapper table for audit and for comparing
        # source geometry/labels against later canonical Face AOI outputs.
        shutil.copy2(detection_path, staging / "source_face_detections_raw.csv")
        canonical_selected.to_csv(staging / "face_detections_canonical.csv", index=False)
        atomic_write_json(staging / "bilateral_semantics_audit.json", bilateral_audit)
        # Keep a copy of the source fixation table for reproducible recomputation.
        source_fix = next((p for p in fm_source_root.rglob("fixations_on_face.csv")), None)
        if source_fix is not None:
            shutil.copy2(source_fix, staging / "source_fixations_on_face.csv")
        safe_stats = _package_variant(
            source_root=fm_source_root, tsd_root=tsd_source_root, output_root=staging / "privacy_safe",
            detections=canonical_selected, keep_mask=privacy_mask, recording_id=recording_id,
            accepted_by_frame=safe_by_frame, scene_timestamps=scene_timestamps,
            width=width, height=height, policy="privacy_safe",
        )
        strict_stats = _package_variant(
            source_root=fm_source_root, tsd_root=tsd_source_root, output_root=staging / "analysis_strict",
            detections=canonical_selected, keep_mask=strict_mask, recording_id=recording_id,
            accepted_by_frame=strict_by_frame, scene_timestamps=scene_timestamps,
            width=width, height=height, policy="analysis_strict",
        )
        _zip_folder(staging / "privacy_safe", staging / "FaceMapper_privacy_safe.zip")
        _zip_folder(staging / "analysis_strict", staging / "FaceMapper_analysis_strict.zip")
        frame_map = pd.read_csv(trim_root / "native_frame_map.csv", low_memory=False)
        preview_info = None
        if render_preview:
            emit(progress, 0.66, "Rendering accepted/review/rejected FaceMapper QC preview")
            preview_info = _render_preview(
                source_video=source_video,
                frame_map=frame_map,
                candidates_by_frame=all_by_frame,
                output_path=staging / "facemapper_filter_qc_preview.mp4",
                raw_output_path=(staging / "facemapper_raw_mapper_preview.mp4") if render_source_preview else None,
                canonical_output_path=(staging / "facemapper_canonical_mapper_preview.mp4") if render_source_preview else None,
                bilateral_label_swap_applied=bool(bilateral_audit["bilateral_label_swap_applied"]),
                render_raw_source=bool(render_source_preview),
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                sample_count=preview_sample_count,
                segment_seconds=preview_segment_seconds,
                output_fps=preview_fps,
                preview_scope=preview_scope,
                token=token,
                progress=progress,
            )
        counts = Counter(c.status for c in candidates)
        counts["invalid"] += len(invalid_rows)
        source_landmark_count = int(sum(len(candidate.landmarks or {}) for candidate in candidates))
        manifest_out = {
            "schema_version": "oculostudio-facemapper-filter-1.2",
            "native_trim_artifact_id": native_trim_artifact_id,
            "recording_id": recording_id,
            "source_asset_ids": {key: value.id for key, value in sources.items()},
            "image": {"width": width, "height": height},
            "counts": {
                "total": int(len(selected)),
                "accepted": int(counts["accepted"]),
                "review": int(counts["review"]),
                "rejected": int(counts["rejected"] + counts["invalid"]),
            },
            "rejection_reasons": dict(reason_counts),
            "thresholds": {
                "large_box_area_ratio": large_box_area_ratio,
                "maximum_area_ratio": maximum_area_ratio,
                "minimum_landmark_span_x_ratio": minimum_landmark_span_x_ratio,
                "minimum_landmark_span_y_ratio": minimum_landmark_span_y_ratio,
                "minimum_interocular_ratio": minimum_interocular_ratio,
                "minimum_landmark_in_box_fraction": minimum_landmark_in_box_fraction,
                "maximum_landmark_center_offset": maximum_landmark_center_offset,
                "maximum_scale_jump_ratio": maximum_scale_jump_ratio,
                "minimum_track_frames": minimum_track_frames,
            },
            "privacy_safe": safe_stats,
            "analysis_strict": strict_stats,
            "preview": preview_info,
            "preview_scope": preview_scope,
            "raw_source_preview_rendered": bool(preview_info and preview_info.get("raw_source_preview", {}).get("rendered")),
            "canonical_source_preview_rendered": bool(preview_info and preview_info.get("canonical_source_preview", {}).get("rendered")),
            "preview_landmark_overlay": "canonical_anatomical_semantics_with_source_pixels",
            "filtered_qc_landmark_count": int((preview_info or {}).get("filtered_qc_preview", {}).get("draw_counts", {}).get("landmarks", source_landmark_count)),
            "raw_source_landmark_count": int((preview_info or {}).get("raw_source_preview", {}).get("draw_counts", {}).get("landmarks", source_landmark_count)),
            "canonical_source_landmark_count": int((preview_info or {}).get("canonical_source_preview", {}).get("draw_counts", {}).get("landmarks", source_landmark_count)),
            "raw_source_table": "source_face_detections_raw.csv",
            "canonical_table": "face_detections_canonical.csv",
            "bilateral_semantics_audit": "bilateral_semantics_audit.json",
            "source_detection_filename": detection_path.name,
            "source_geometry_policy": "exact_export",
            "canonical_geometry_policy": "source_pixels_preserved",
            "source_bilateral_label_policy": bilateral_audit["resolved_source_convention"],
            "canonical_bilateral_label_policy": "subject_anatomical",
            "bilateral_label_swap_applied": bool(bilateral_audit["bilateral_label_swap_applied"]),
            "geometry_mirror_applied": False,
            "bilateral_audit": bilateral_audit,
            "privacy_rule": "accepted + review; only high-confidence false positives are rejected",
            "analysis_rule": "accepted only",
            "participant_identity_resolved": False,
        }
        atomic_write_json(staging / "filter_manifest.json", manifest_out)
        (staging / "filter_report.html").write_text(_html_report(manifest_out), "utf-8")
        token.raise_if_cancelled()
        staging.rename(final)
        role_id = str(artifact.parameters.get("role_id") or manifest.get("request", {}).get("label") or "")
        artifact_out = ArtifactStore(store).register(
            session_id=session_id,
            kind=ArtifactKind.FACE_MAPPER_FILTER,
            label=label,
            output_root=final,
            source_asset_ids=[value.id for value in sources.values()],
            parent_artifact_ids=[native_trim_artifact_id],
            parameters={
                "native_trim_artifact_id": native_trim_artifact_id,
                "interaction_id": artifact.parameters.get("interaction_id"),
                "role_id": artifact.parameters.get("role_id"),
                "privacy_variant": "privacy_safe",
                "analysis_variant": "analysis_strict",
                "preview_scope": preview_scope,
                "preview_fps": preview_fps,
                "render_source_preview": bool(render_source_preview),
                "raw_source_geometry": "exact_export",
                "source_bilateral_convention": source_bilateral_convention,
                "canonical_bilateral_convention": "subject_anatomical",
                "bilateral_label_swap_applied": bool(bilateral_audit["bilateral_label_swap_applied"]),
                "geometry_mirror_applied": False,
                **manifest_out["thresholds"],
            },
            qc={
                **manifest_out["counts"],
                "rejection_reasons": manifest_out["rejection_reasons"],
                "preview_rendered": bool(preview_info),
                "preview_scope": preview_scope,
                "filtered_qc_landmark_count": manifest_out["filtered_qc_landmark_count"],
                "raw_source_landmark_count": manifest_out["raw_source_landmark_count"],
                "canonical_source_landmark_count": manifest_out["canonical_source_landmark_count"],
                "bilateral_label_swap_applied": bool(bilateral_audit["bilateral_label_swap_applied"]),
                "source_eye_image_left_fraction": bilateral_audit.get("eye", {}).get("image_left_fraction"),
                "source_mouth_image_left_fraction": bilateral_audit.get("mouth", {}).get("image_left_fraction"),
            },
            provenance={
                "recording_id": recording_id,
                "filtering_policy": "auditable_dual_policy",
                "facemapper_filter_engine": "canonical-bilateral-semantics-live-canvas-v4",
                "canonical_face_mapper_schema": "oculostudio-facemapper-bilateral-1.0",
            },
            artifact_id=artifact_id,
        )
        emit(progress, 1.0, "FaceMapper QC/filter complete; Privacy can now use the privacy-safe candidate set")
        return {
            "artifact_id": artifact_out.id,
            "output_dir": str(final),
            "privacy_safe_zip": str(final / "FaceMapper_privacy_safe.zip"),
            "analysis_strict_zip": str(final / "FaceMapper_analysis_strict.zip"),
            "audit_csv": str(final / "face_filter_audit.csv"),
            "report_html": str(final / "filter_report.html"),
            "preview_video": str(final / "facemapper_filter_qc_preview.mp4") if preview_info else None,
            "raw_mapper_preview_video": str(final / "facemapper_raw_mapper_preview.mp4") if preview_info and render_source_preview else None,
            "canonical_mapper_preview_video": str(final / "facemapper_canonical_mapper_preview.mp4") if preview_info and render_source_preview else None,
            "raw_source_table": str(final / "source_face_detections_raw.csv"),
            "canonical_table": str(final / "face_detections_canonical.csv"),
            "bilateral_semantics_audit": str(final / "bilateral_semantics_audit.json"),
            "counts": manifest_out["counts"],
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if final.exists() and artifact_id not in store.project.sessions[session_id].artifacts:
            shutil.rmtree(final, ignore_errors=True)
        raise


def filtered_facemapper_root(artifact, *, variant: Policy) -> Path:
    if artifact.kind != ArtifactKind.FACE_MAPPER_FILTER:
        raise ValidationError("Select a FaceMapper QC/filter artifact")
    manifest_path = find_artifact_file(artifact, names=["filter_manifest.json"])
    root = manifest_path.parent / variant
    if not (root / "face_detections.csv").exists():
        raise ValidationError(f"FaceMapper filter artifact has no {variant} package")
    return root
