from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.mocap import read_mocap
from oculostudio.adapters.pupil import read_cloud_world_timestamps
from oculostudio.domain.models import ArtifactKind, AssetKind
from oculostudio.domain.semantic import SemanticRegion
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.object_primitives import build_object_primitives, intersect_frame_rays
from oculostudio.services.semantic_targets import (
    _artifact_table_path,
    _find_column,
    _nearest_indices,
    _read_table,
    _resolve_cloud_asset,
    _scene_resolution,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


def _gaze_uid(recording_id: str, source_row_index: int, timestamp_ns: int) -> str:
    raw = f"{recording_id}|{source_row_index}|{timestamp_ns}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


def _frame_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = samples.loc[samples["scene_frame_index"].notna()].groupby("scene_frame_index")
    for frame_index, frame in grouped:
        distribution = Counter(frame["object_entity_id"].fillna("no_object").astype(str))
        score: dict[str | None, float] = defaultdict(float)
        category_for: dict[str | None, str | None] = {}
        for row in frame.itertuples(index=False):
            entity = row.object_entity_id if isinstance(row.object_entity_id, str) and row.object_entity_id else None
            score[entity] += max(0.05, float(row.object_confidence or 0.0))
            category_for[entity] = row.object_category if isinstance(row.object_category, str) else None
        primary = max(score, key=score.get) if score else None
        rows.append({
            "scene_frame_index": int(frame_index),
            "scene_timestamp_ns": int(pd.to_numeric(frame["scene_timestamp_ns"], errors="coerce").dropna().iloc[0]),
            "gaze_sample_count": int(len(frame)),
            "valid_ray_count": int(frame["ray_valid"].fillna(False).astype(bool).sum()),
            "primary_object_entity": primary,
            "primary_object_category": category_for.get(primary),
            "object_distribution_json": json.dumps(distribution, sort_keys=True, separators=(",", ":")),
            "mean_object_confidence": float(pd.to_numeric(frame["object_confidence"], errors="coerce").fillna(0).mean()),
            "review_required": bool(frame["review_required"].fillna(False).astype(bool).any()),
        })
    return pd.DataFrame(rows)


def _apply_native_trim(rays: pd.DataFrame, trim_artifact) -> pd.DataFrame:
    trim_gaze_path = find_artifact_file(trim_artifact, names=["gaze.csv"])
    trim_gaze = pd.read_csv(trim_gaze_path, low_memory=False)
    trim_source_col = _find_column(trim_gaze.columns, ["oculo_source_row_index", "source_gaze_row_index", "source_row_index"])
    ray_source_col = _find_column(rays.columns, ["gaze3d_input_row_index", "oculo_source_row_index", "source_row_index"])
    if trim_gaze[trim_source_col].duplicated().any() or rays[ray_source_col].duplicated().any():
        raise ValidationError("Native trim and gaze-ray source-row indices must be unique")
    trim_order = pd.to_numeric(trim_gaze[trim_source_col], errors="raise").astype("int64")
    rays = rays.copy()
    rays[ray_source_col] = pd.to_numeric(rays[ray_source_col], errors="raise").astype("int64")
    rays = rays.set_index(ray_source_col, drop=False)
    missing = trim_order.loc[~trim_order.isin(rays.index)]
    if len(missing):
        raise ValidationError(
            f"Gaze-ray artifact is missing {len(missing)} samples required by the selected Native-reference trim"
        )
    rays = rays.loc[trim_order.to_numpy()].reset_index(drop=True)
    attach_aliases = {
        "native_frame_index": ["oculo_nearest_native_frame_index", "native_frame_index"],
        "clip_time_ns": ["oculo_clip_video_ns", "clip_time_ns"],
        "gaze_x_px": ["gaze x [px]", "gaze_x_px", "gaze x", "x"],
        "gaze_y_px": ["gaze y [px]", "gaze_y_px", "gaze y", "y"],
        "worn": ["worn"],
        "blink_id": ["blink id", "blink_id"],
        "fixation_id": ["fixation id", "fixation_id"],
    }
    for canonical, aliases in attach_aliases.items():
        source = _find_column(trim_gaze.columns, aliases, required=False)
        if source:
            rays[canonical] = trim_gaze[source].to_numpy()
    trim_timestamp_col = _find_column(trim_gaze.columns, ["timestamp [ns]", "timestamp_ns", "timestamp"])
    ray_timestamp_col = _find_column(rays.columns, ["timestamp [ns]", "timestamp_ns", "gaze timestamp [ns]"])
    trim_ts = pd.to_numeric(trim_gaze[trim_timestamp_col], errors="coerce").astype("Int64")
    ray_ts = pd.to_numeric(rays[ray_timestamp_col], errors="coerce").astype("Int64")
    mismatch = (trim_ts.notna() & ray_ts.notna() & (trim_ts != ray_ts)).sum()
    if mismatch:
        raise ValidationError(f"Native trim and gaze-ray timestamps differ for {int(mismatch)} source rows")
    return rays


def _selected_frame_clock_times(
    rays: pd.DataFrame, nearest: np.ndarray, selected_frames: np.ndarray
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Return the best downstream C3D clock already carried by Gaze 3D rays."""
    absolute_col = _find_column(
        rays.columns,
        ["gaze3d_c3d_time_sec", "gaze3d_tracker_time_sec", "c3d_time_sec", "tracker_time_sec"],
        required=False,
    )
    local_col = _find_column(
        rays.columns,
        ["gaze3d_tracker_local_time_sec", "tracker_local_time_sec", "clip_time_ns", "oculo_clip_video_ns"],
        required=False,
    )
    if absolute_col:
        values_sec = pd.to_numeric(rays[absolute_col], errors="coerce").to_numpy(dtype=float)
        mode = "gaze_ray_c3d_time"
        source_col = absolute_col
    elif local_col:
        values_sec = pd.to_numeric(rays[local_col], errors="coerce").to_numpy(dtype=float)
        if local_col in {"clip_time_ns", "oculo_clip_video_ns"}:
            values_sec = values_sec / 1e9
        mode = "pretrimmed_shared_start"
        source_col = local_col
    else:
        return None, None, None
    result = np.full(len(selected_frames), np.nan, dtype=float)
    for position, frame_index in enumerate(selected_frames):
        values = values_sec[(nearest == int(frame_index)) & np.isfinite(values_sec)]
        if len(values):
            result[position] = float(np.median(values))
    if not np.isfinite(result).all():
        missing = int((~np.isfinite(result)).sum())
        raise ValidationError(f"Could not recover gaze-ray C3D time for {missing} selected scene frames")
    return result, mode, source_col


def create_object_aoi_annotations(
    store: ProjectStore,
    *,
    session_id: str,
    gaze_rays_artifact_id: str,
    timeseries_asset_id: str | None = None,
    object_asset_ids: list[str] | None = None,
    object_profile_path: str | None = None,
    object_id_overrides: dict[str, str] | None = None,
    native_trim_artifact_id: str | None = None,
    max_marker_gap_sec: float = 0.1,
    uncertainty_cone_deg: float = 1.5,
    maximum_target_distance_m: float = 10.0,
    scene_frame_tolerance_ms: float = 20.0,
    label: str = "Object AOI",
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    object_asset_ids = list(object_asset_ids or [])
    object_id_overrides = dict(object_id_overrides or {})
    if not object_asset_ids:
        raise ValidationError("Select at least one Object C3D or canonical mocap asset")
    session = store.project.sessions[session_id]
    gaze_artifact = session.artifacts.get(gaze_rays_artifact_id)
    if gaze_artifact is None or gaze_artifact.kind != ArtifactKind.GAZE_RAYS:
        raise ValidationError("Select a Gaze 3D ray artifact")
    gaze_alignment_mode = str((gaze_artifact.parameters or {}).get("alignment_mode") or "timegraph")
    gaze_native_trim_id = (gaze_artifact.parameters or {}).get("native_trim_artifact_id")
    if native_trim_artifact_id and gaze_native_trim_id and native_trim_artifact_id != gaze_native_trim_id:
        raise ValidationError("Selected Native-reference trim differs from the trim used to generate these gaze rays")
    if not native_trim_artifact_id and gaze_native_trim_id:
        native_trim_artifact_id = str(gaze_native_trim_id)
    cloud_asset = _resolve_cloud_asset(session, gaze_artifact, timeseries_asset_id)

    emit(progress, 0.03, "Loading calibrated gaze rays")
    rays = _read_table(_artifact_table_path(gaze_artifact))
    original_ray_count = len(rays)
    trim_artifact = None
    if native_trim_artifact_id:
        trim_artifact = session.artifacts.get(native_trim_artifact_id)
        if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
            raise ValidationError("native_trim_artifact_id must reference a Native-reference trim artifact")
        rays = _apply_native_trim(rays, trim_artifact)
    input_count = len(rays)
    timestamp_col = _find_column(rays.columns, ["timestamp [ns]", "timestamp_ns", "gaze timestamp [ns]"])
    ray_valid_col = _find_column(rays.columns, ["gaze3d_ray_valid", "ray_valid"])
    origin_cols = [
        _find_column(rays.columns, ["gaze3d_origin_x_mm", "origin_x_mm"]),
        _find_column(rays.columns, ["gaze3d_origin_y_mm", "origin_y_mm"]),
        _find_column(rays.columns, ["gaze3d_origin_z_mm", "origin_z_mm"]),
    ]
    direction_cols = [
        _find_column(rays.columns, ["gaze3d_direction_x", "direction_x"]),
        _find_column(rays.columns, ["gaze3d_direction_y", "direction_y"]),
        _find_column(rays.columns, ["gaze3d_direction_z", "direction_z"]),
    ]
    source_row_col = _find_column(rays.columns, ["oculo_source_row_index", "gaze3d_input_row_index", "source_row_index"], required=False)
    timestamp_numeric = pd.to_numeric(rays[timestamp_col], errors="coerce")
    timestamp_valid = timestamp_numeric.notna().to_numpy()
    timestamp_ns = np.zeros(input_count, dtype=np.int64)
    timestamp_ns[timestamp_valid] = timestamp_numeric.loc[timestamp_numeric.notna()].astype("int64").to_numpy()
    scene_timestamps = read_cloud_world_timestamps(cloud_asset.source_path)
    nearest, frame_error = _nearest_indices(scene_timestamps, timestamp_ns)
    frame_valid = timestamp_valid & (frame_error <= int(round(scene_frame_tolerance_ms * 1e6)))
    nearest[~frame_valid] = -1
    selected_frames = np.unique(nearest[frame_valid])
    selected_timestamps = scene_timestamps[selected_frames]

    cache = MaterializationCache(store.cache_dir())
    # Object-ray intersection is performed entirely in the Vicon/world frame
    # and therefore does not require a scene-video resolution. Retain the
    # resolution as optional provenance when a calibration profile or Cloud
    # scene video is available, but do not block object AOI generation when the
    # Timeseries export contains tables only.
    try:
        width, height, resolution_provenance = _scene_resolution(cache, cloud_asset, gaze_artifact)
    except ValidationError:
        width, height = None, None
        resolution_provenance = {"source": "unavailable", "required_for_object_intersection": False}
    selected_ray_clock_sec, object_time_alignment_mode, object_clock_column = (
        _selected_frame_clock_times(rays, nearest, selected_frames)
        if native_trim_artifact_id and len(selected_frames)
        else (None, None, None)
    )
    if object_time_alignment_mode is None:
        object_time_alignment_mode = "timegraph"
    graph = None
    scene_global_ns = None
    if object_time_alignment_mode == "timegraph":
        if not cloud_asset.primary_time_domain_id:
            raise ValidationError(
                "Timeseries asset has no primary time domain and the selected gaze rays do not expose a reusable C3D clock"
            )
        graph = TimeGraph.from_session(session)
        try:
            scene_global_ns = graph.convert(
                cloud_asset.primary_time_domain_id,
                session.global_time_domain_id,
                selected_timestamps,
            ).astype(np.int64)
        except Exception as exc:
            raise ValidationError(
                "Object AOI could not reuse an exact gaze-ray C3D clock and no TimeGraph path connects the selected Timeseries to Global Time. "
                "Regenerate Gaze 3D from the matching C3D-defined Native trim. "
                f"Original error: {exc}"
            ) from exc

    primitives_by_frame: dict[int, list] = defaultdict(list)
    region_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    emit(progress, 0.14, "Building prop geometry from Object C3D")
    for position, asset_id in enumerate(object_asset_ids):
        token.raise_if_cancelled()
        asset = session.assets.get(asset_id)
        if asset is None or asset.kind not in {AssetKind.C3D, AssetKind.MOCAP_NPZ}:
            raise ValidationError(f"Object asset {asset_id} is not C3D or canonical mocap NPZ")
        if object_time_alignment_mode == "gaze_ray_c3d_time":
            assert selected_ray_clock_sec is not None
            query_time_sec = selected_ray_clock_sec.copy()
        elif object_time_alignment_mode == "pretrimmed_shared_start":
            assert selected_ray_clock_sec is not None
            object_mocap = read_mocap(asset.source_path)
            query_time_sec = float(object_mocap.time_sec[0]) + selected_ray_clock_sec
        else:
            if not asset.primary_time_domain_id:
                raise ValidationError(f"Object asset {asset.label} has no time domain")
            assert graph is not None and scene_global_ns is not None
            query_ns = graph.convert(
                session.global_time_domain_id,
                asset.primary_time_domain_id,
                scene_global_ns,
            ).astype(np.int64)
            query_time_sec = query_ns.astype(np.float64) / 1e9
        object_mocap_for_validation = read_mocap(asset.source_path)
        finite_query = np.isfinite(query_time_sec)
        if finite_query.any():
            lower = float(object_mocap_for_validation.time_sec[0]) - float(max_marker_gap_sec)
            upper = float(object_mocap_for_validation.time_sec[-1]) + float(max_marker_gap_sec)
            query_time_range_coverage = float(
                ((query_time_sec[finite_query] >= lower) & (query_time_sec[finite_query] <= upper)).mean()
            )
        else:
            query_time_range_coverage = 0.0
        if object_time_alignment_mode in {"gaze_ray_c3d_time", "pretrimmed_shared_start"} and query_time_range_coverage < 0.90:
            raise ValidationError(
                f"Object asset {asset.label!r} overlaps only {100.0*query_time_range_coverage:.1f}% of the gaze-ray C3D clock. "
                "Select an object C3D from the same interaction/Vicon recording, or provide an explicit TimeGraph."
            )
        build = build_object_primitives(
            object_asset_id=asset_id,
            object_path=asset.source_path,
            scene_frame_indices=selected_frames,
            scene_timestamps_ns=selected_timestamps,
            query_time_sec=query_time_sec,
            profile_path=object_profile_path,
            object_overrides=object_id_overrides,
            max_marker_gap_sec=max_marker_gap_sec,
        )
        for frame_index, primitives in build.primitives_by_frame.items():
            primitives_by_frame[frame_index].extend(primitives)
        if not build.frame_regions.empty:
            region_frames.append(build.frame_regions)
        object_diagnostic = dict(build.diagnostics)
        object_diagnostic["object_time_alignment_mode"] = object_time_alignment_mode
        object_diagnostic["gaze_ray_clock_column"] = object_clock_column
        object_diagnostic["query_time_range_coverage"] = query_time_range_coverage
        diagnostics.append(object_diagnostic)
        emit(progress, 0.14 + 0.25 * (position + 1) / len(object_asset_ids), f"Built object geometry from {asset.label}")

    origins = rays[origin_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    directions = rays[direction_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ray_valid = rays[ray_valid_col].fillna(False).astype(bool).to_numpy()
    hits = [None] * input_count
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, frame_index in enumerate(nearest):
        if frame_index >= 0 and ray_valid[index]:
            by_frame[int(frame_index)].append(index)
    emit(progress, 0.43, "Intersecting gaze rays with prop geometry")
    total_intersections = sum(len(rows) for rows in by_frame.values())
    done = 0
    for frame_index, indices in by_frame.items():
        token.raise_if_cancelled()
        frame_hits = intersect_frame_rays(
            origins[np.asarray(indices)],
            directions[np.asarray(indices)],
            primitives_by_frame.get(frame_index, []),
            uncertainty_cone_deg=uncertainty_cone_deg,
            maximum_distance_mm=maximum_target_distance_m * 1000.0,
        )
        for source_index, hit in zip(indices, frame_hits):
            hits[source_index] = hit
        done += len(indices)
        if done % 2000 < len(indices):
            emit(progress, 0.43 + 0.32 * done / max(1, total_intersections), f"Intersecting Object AOIs ({done}/{total_intersections})")

    recording_id = cloud_asset.recording_id or "unknown_recording"
    native_frame_col = _find_column(rays.columns, ["native_frame_index", "oculo_nearest_native_frame_index"], required=False)
    clip_time_col = _find_column(rays.columns, ["clip_time_ns", "oculo_clip_video_ns"], required=False)
    gaze_x_col = _find_column(rays.columns, ["gaze_x_px", "gaze x [px]", "gaze x", "x"], required=False)
    gaze_y_col = _find_column(rays.columns, ["gaze_y_px", "gaze y [px]", "gaze y", "y"], required=False)
    rows: list[dict[str, Any]] = []
    for ordinal in range(input_count):
        source_row = int(rays[source_row_col].iloc[ordinal]) if source_row_col and pd.notna(rays[source_row_col].iloc[ordinal]) else ordinal
        stamp = int(timestamp_ns[ordinal]) if timestamp_valid[ordinal] else -1
        hit = hits[ordinal]
        if not timestamp_valid[ordinal]:
            status = "invalid_timestamp"; region = SemanticRegion.INVALID_GAZE.value; review = False
        elif not ray_valid[ordinal]:
            status = "invalid_ray"; region = SemanticRegion.INVALID_GAZE.value; review = False
        elif nearest[ordinal] < 0:
            status = "no_scene_frame"; region = SemanticRegion.UNKNOWN.value; review = True
        elif hit is None:
            status = "valid_ray_no_object_hit"; region = "no_object"; review = False
        else:
            status = "object_3d_exact" if hit.exact_hit else "object_3d_uncertainty_cone"
            region = "object"
            review = not hit.exact_hit
        rows.append({
            "recording_id": recording_id,
            "gaze_sample_uid": _gaze_uid(recording_id, source_row, stamp),
            "source_gaze_row_index": source_row,
            "timestamp_ns": stamp if stamp >= 0 else pd.NA,
            "clip_time_ns": int(rays[clip_time_col].iloc[ordinal]) if clip_time_col and pd.notna(rays[clip_time_col].iloc[ordinal]) else pd.NA,
            "native_frame_index": int(rays[native_frame_col].iloc[ordinal]) if native_frame_col and pd.notna(rays[native_frame_col].iloc[ordinal]) else pd.NA,
            "gaze_x_px": float(rays[gaze_x_col].iloc[ordinal]) if gaze_x_col and pd.notna(rays[gaze_x_col].iloc[ordinal]) else np.nan,
            "gaze_y_px": float(rays[gaze_y_col].iloc[ordinal]) if gaze_y_col and pd.notna(rays[gaze_y_col].iloc[ordinal]) else np.nan,
            "scene_frame_index": int(nearest[ordinal]) if nearest[ordinal] >= 0 else pd.NA,
            "scene_timestamp_ns": int(scene_timestamps[nearest[ordinal]]) if nearest[ordinal] >= 0 else pd.NA,
            "scene_frame_match_error_ms": float(frame_error[ordinal]) / 1e6 if frame_valid[ordinal] else np.nan,
            "ray_valid": bool(ray_valid[ordinal]),
            "ray_origin_x_mm": origins[ordinal, 0],
            "ray_origin_y_mm": origins[ordinal, 1],
            "ray_origin_z_mm": origins[ordinal, 2],
            "ray_direction_x": directions[ordinal, 0],
            "ray_direction_y": directions[ordinal, 1],
            "ray_direction_z": directions[ordinal, 2],
            "object_entity_id": hit.object_id if hit else None,
            "object_category": hit.object_category if hit else None,
            "object_region": region,
            "object_distance_mm": hit.distance_mm if hit else np.nan,
            "object_hit_x_mm": float(hit.point_mm[0]) if hit else np.nan,
            "object_hit_y_mm": float(hit.point_mm[1]) if hit else np.nan,
            "object_hit_z_mm": float(hit.point_mm[2]) if hit else np.nan,
            "object_confidence": hit.confidence if hit else 0.0,
            "object_exact_hit": hit.exact_hit if hit else False,
            "object_angular_miss_deg": hit.angular_miss_deg if hit else np.nan,
            "object_geometry_backend": "object_c3d_markers",
            "assignment_status": status,
            "review_required": bool(review),
        })
    samples = pd.DataFrame(rows)
    summary = _frame_summary(samples)
    regions = pd.concat(region_frames, ignore_index=True, sort=False) if region_frames else pd.DataFrame()

    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "object_aoi", gaze_artifact.clip_id or "full") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    samples_csv = output_root / "object_aoi_samples.csv"
    frames_csv = output_root / "object_aoi_frames.csv"
    regions_csv = output_root / "object_regions.csv"
    samples.to_csv(samples_csv, index=False)
    summary.to_csv(frames_csv, index=False)
    regions.to_csv(regions_csv, index=False)
    parquet_files: list[str] = []
    if output_parquet:
        try:
            for frame, name in [
                (samples, "object_aoi_samples.parquet"),
                (summary, "object_aoi_frames.parquet"),
                (regions, "object_regions.parquet"),
            ]:
                path = output_root / name
                frame.to_parquet(path, index=False)
                parquet_files.append(str(path))
        except Exception:
            parquet_files = []
    counts = Counter(samples["object_entity_id"].fillna("no_object").astype(str))
    qc = {
        "source_gaze_ray_rows": int(original_ray_count),
        "input_gaze_rows": int(input_count),
        "output_rows": int(len(samples)),
        "native_trim_applied": bool(native_trim_artifact_id),
        "alignment_mode": gaze_alignment_mode,
        "object_time_alignment_mode": object_time_alignment_mode,
        "gaze_ray_clock_column": object_clock_column,
        "alignment_basis": (
            "Exact C3D query time already stored in calibrated gaze rays; no project Global Time path required"
            if object_time_alignment_mode == "gaze_ray_c3d_time"
            else (
                "Gaze-ray tracker-local clock + first frame of each co-started object C3D"
                if object_time_alignment_mode == "pretrimmed_shared_start"
                else "Verified TimeGraph mapping through project Global Time"
            )
        ),
        "global_timegraph_required": bool(object_time_alignment_mode == "timegraph"),
        "row_count_preserved": bool(input_count == len(samples)),
        "valid_ray_fraction": float(samples["ray_valid"].mean()) if len(samples) else 0.0,
        "object_hit_fraction": float(samples["object_entity_id"].notna().mean()) if len(samples) else 0.0,
        "exact_object_hit_fraction": float(samples["object_exact_hit"].mean()) if len(samples) else 0.0,
        "review_required_fraction": float(samples["review_required"].mean()) if len(samples) else 0.0,
        "object_counts": dict(counts),
        "object_diagnostics": diagnostics,
    }
    manifest = {
        "schema_version": "oculostudio-object-aoi-1.1",
        "artifact_id": artifact_id,
        "gaze_rays_artifact_id": gaze_rays_artifact_id,
        "recording_id": recording_id,
        "parameters": {
            "object_profile_path": object_profile_path,
            "object_id_overrides": object_id_overrides,
            "native_trim_artifact_id": native_trim_artifact_id,
            "alignment_mode": gaze_alignment_mode,
            "object_time_alignment_mode": object_time_alignment_mode,
            "gaze_ray_clock_column": object_clock_column,
            "max_marker_gap_sec": max_marker_gap_sec,
            "uncertainty_cone_deg": uncertainty_cone_deg,
            "maximum_target_distance_m": maximum_target_distance_m,
            "scene_frame_tolerance_ms": scene_frame_tolerance_ms,
        },
        "scene_resolution": {"width": width, "height": height, **resolution_provenance},
        "outputs": {"samples_csv": str(samples_csv), "frames_csv": str(frames_csv), "regions_csv": str(regions_csv), "parquet": parquet_files},
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "object_aoi_manifest.json", manifest)
    atomic_write_json(output_root / "qc_metrics.json", qc)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.OBJECT_AOI,
        label=label,
        output_root=output_root,
        source_asset_ids=list(dict.fromkeys([cloud_asset.id, *object_asset_ids])),
        parent_artifact_ids=[artifact_id for artifact_id in [gaze_rays_artifact_id, native_trim_artifact_id] if artifact_id],
        clip_id=gaze_artifact.clip_id,
        parameters=manifest["parameters"],
        qc=qc,
        provenance={
            "gaze_rays_artifact_id": gaze_rays_artifact_id,
            "native_trim_artifact_id": native_trim_artifact_id,
            "alignment_mode": gaze_alignment_mode,
            "object_time_alignment_mode": object_time_alignment_mode,
            "gaze_ray_clock_column": object_clock_column,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Object AOI annotation complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "samples_csv": str(samples_csv),
        "frames_csv": str(frames_csv),
        "regions_csv": str(regions_csv),
        "manifest": str(output_root / "object_aoi_manifest.json"),
    }
