from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.pupil import (
    find_cloud_candidates,
    materialized_cloud_root,
    read_cloud_world_timestamps,
)
from oculostudio.adapters.video import probe_video
from oculostudio.domain.models import ArtifactKind, ArtifactRecord, AssetKind
from oculostudio.domain.semantic import (
    FACE_REGIONS,
    HEAD_COMPATIBLE_REGIONS,
    SemanticRegion,
)
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ValidationError
from oculostudio.services.body_primitives import (
    BodyHit,
    build_body_primitives,
    intersect_frame_rays,
    load_body_profile,
)
from oculostudio.services.densepose_import import (
    DensePoseEvidence,
    load_densepose_evidence,
    match_densepose_evidence,
)
from oculostudio.services.face_semantics import (
    FaceHit,
    classify_face_point,
    load_face_semantics,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import (
    CancellationToken,
    atomic_write_json,
    emit,
    new_id,
    safe_name,
    utc_now,
)


@dataclass(frozen=True)
class FusionDecision:
    entity_id: str | None
    region: str
    confidence: float
    status: str
    review_required: bool


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _find_column(columns, aliases, *, required: bool = True) -> str | None:
    mapping = {_normalize_column(column): column for column in columns}
    for alias in aliases:
        key = _normalize_column(alias)
        if key in mapping:
            return mapping[key]
    if required:
        raise ValidationError(f"Semantic target input is missing required column: {aliases[0]}")
    return None


def _artifact_table_path(artifact: ArtifactRecord) -> Path:
    preferred: list[Path] = []
    fallback: list[Path] = []
    for file in artifact.files:
        path = Path(file.path)
        if path.suffix.lower() == ".parquet" and "gaze_rays" in path.name.lower():
            preferred.append(path)
        elif path.suffix.lower() == ".csv" and "gaze_rays" in path.name.lower():
            fallback.append(path)
    candidates = preferred + fallback
    if not candidates:
        candidates = [
            Path(file.path) for file in artifact.files
            if Path(file.path).suffix.lower() in {".csv", ".parquet", ".feather"}
        ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise ValidationError("The gaze-ray artifact contains no readable table file")
    return existing[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    raise ValidationError(f"Unsupported semantic target table: {path.suffix}")


def _nearest_indices(reference_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    if len(reference) == 0:
        return np.full(len(query), -1, dtype=int), np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
    position = np.searchsorted(reference, query, side="left")
    right = np.clip(position, 0, len(reference) - 1)
    left = np.clip(position - 1, 0, len(reference) - 1)
    left_error = np.abs(query - reference[left])
    right_error = np.abs(reference[right] - query)
    choose_right = right_error < left_error
    return np.where(choose_right, right, left), np.where(choose_right, right_error, left_error)


def _resolve_cloud_asset(session, gaze_artifact: ArtifactRecord, timeseries_asset_id: str | None):
    if timeseries_asset_id:
        asset = session.assets[timeseries_asset_id]
        if asset.kind != AssetKind.PUPIL_CLOUD:
            raise ValidationError("timeseries_asset_id must reference a Pupil Cloud/Timeseries asset")
        return asset
    candidates = [
        session.assets[asset_id]
        for asset_id in gaze_artifact.source_asset_ids
        if asset_id in session.assets and session.assets[asset_id].kind == AssetKind.PUPIL_CLOUD
    ]
    if len(candidates) != 1:
        raise ValidationError(
            "Could not infer a unique Timeseries source from the gaze-ray artifact. "
            "Select it explicitly in the Gaze Targets module."
        )
    return candidates[0]


def _scene_resolution(cache: MaterializationCache, cloud_asset, gaze_artifact: ArtifactRecord) -> tuple[int, int, dict]:
    # First use the calibrated camera resolution, which is the authoritative
    # coordinate system for gaze pixels.
    profile_path = gaze_artifact.provenance.get("calibration_profile_path")
    if profile_path and Path(profile_path).exists():
        try:
            profile = json.loads(Path(profile_path).read_text("utf-8"))
            resolution = profile.get("camera_intrinsics", {}).get("resolution")
            if isinstance(resolution, list) and len(resolution) >= 2:
                return int(resolution[0]), int(resolution[1]), {"source": "calibration_profile"}
        except Exception:
            pass
    root = cache.materialize(cloud_asset)
    candidates = find_cloud_candidates(root)
    if cloud_asset.recording_id:
        matches = [item for item in candidates if item.recording_id == cloud_asset.recording_id]
        if matches:
            candidates = matches
    if len(candidates) == 1 and candidates[0].scene_video_name:
        video_path = root / candidates[0].scene_video_name
        metadata = probe_video(video_path)
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        if width > 0 and height > 0:
            return width, height, {"source": "cloud_scene_video", "path": str(video_path)}
    raise ValidationError(
        "Could not determine scene-camera resolution from the calibration profile or Cloud scene video"
    )


def _combine_confidences(*values: float | None) -> float:
    finite = [float(np.clip(value, 0.0, 1.0)) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return 0.0
    return float(np.clip(1.0 - np.prod([1.0 - value for value in finite]), 0.0, 1.0))


def _regions_compatible(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    if left == right:
        return True
    if left in FACE_REGIONS and right in HEAD_COMPATIBLE_REGIONS:
        return True
    if right in FACE_REGIONS and left in HEAD_COMPATIBLE_REGIONS:
        return True
    pairs = {
        frozenset({SemanticRegion.LEFT_HAND.value, SemanticRegion.LEFT_FOREARM.value}),
        frozenset({SemanticRegion.RIGHT_HAND.value, SemanticRegion.RIGHT_FOREARM.value}),
        frozenset({SemanticRegion.LEFT_THIGH.value, SemanticRegion.PELVIS.value}),
        frozenset({SemanticRegion.RIGHT_THIGH.value, SemanticRegion.PELVIS.value}),
        frozenset({SemanticRegion.TORSO.value, SemanticRegion.PELVIS.value}),
    }
    return frozenset({left, right}) in pairs


def fuse_target_evidence(
    *,
    ray_valid: bool,
    gaze_in_frame: bool,
    face: FaceHit | None,
    body: BodyHit | None,
    densepose: DensePoseEvidence | None,
) -> FusionDecision:
    if not gaze_in_frame:
        return FusionDecision(None, SemanticRegion.OUT_OF_FRAME.value, 1.0, "out_of_frame", False)
    if not ray_valid and face is None and densepose is None:
        return FusionDecision(None, SemanticRegion.INVALID_GAZE.value, 1.0, "invalid_gaze", False)

    entity: str | None = None
    region = SemanticRegion.UNKNOWN.value
    confidence = 0.0
    status = "no_semantic_evidence"
    review = False

    if body is not None:
        entity = body.entity_id
        region = body.region
        confidence = body.confidence
        status = "body_3d_exact" if body.exact_hit else "body_3d_uncertainty_cone"

    if face is not None:
        if body is None or body.region in HEAD_COMPATIBLE_REGIONS:
            entity = body.entity_id if body is not None else face.entity_id
            region = face.region
            confidence = _combine_confidences(face.confidence, body.confidence if body else None)
            status = "face_refined_3d_head" if body is not None else "face_2d_only"
            if (
                body is not None
                and face.entity_is_explicit
                and face.entity_id != body.entity_id
            ):
                review = True
                confidence *= 0.80
                status += "+identity_conflict"
        else:
            review = True
            if face.confidence > body.confidence + 0.18:
                entity = face.entity_id
                region = face.region
                confidence = face.confidence * 0.82
                status = "face_body_conflict_face_selected"
            else:
                confidence = body.confidence * 0.82
                status = "face_body_conflict_body_selected"

    if densepose is not None:
        if region in {
            SemanticRegion.UNKNOWN.value,
            SemanticRegion.INVALID_GAZE.value,
        }:
            entity = densepose.entity_id
            region = densepose.region
            confidence = densepose.confidence
            status = "densepose_import_only"
        elif _regions_compatible(region, densepose.region):
            if entity is None and densepose.entity_id:
                entity = densepose.entity_id
            confidence = _combine_confidences(confidence, densepose.confidence)
            status = f"{status}+densepose_agreement"
        else:
            review = True
            if body is None and densepose.confidence > confidence + 0.15:
                entity = densepose.entity_id or entity
                region = densepose.region
                confidence = densepose.confidence * 0.80
                status = "densepose_conflict_densepose_selected"
            else:
                confidence *= 0.80
                status = f"{status}+densepose_conflict"
        if entity and densepose.entity_id and entity != densepose.entity_id:
            review = True
            status += "+identity_conflict"

    if region == SemanticRegion.UNKNOWN.value and ray_valid:
        status = "valid_ray_no_person_region_hit"
    return FusionDecision(entity, region, float(np.clip(confidence, 0.0, 1.0)), status, review)


def _frame_summary(
    targets: pd.DataFrame,
    scene_frame_indices: np.ndarray,
    scene_timestamps_ns: np.ndarray,
) -> pd.DataFrame:
    grouped = {int(frame): rows for frame, rows in targets.groupby("scene_frame_index", dropna=False) if pd.notna(frame)}
    rows: list[dict] = []
    for frame_index in scene_frame_indices:
        frame_index = int(frame_index)
        frame_rows = grouped.get(frame_index)
        if frame_rows is None or len(frame_rows) == 0:
            rows.append({
                "scene_frame_index": frame_index,
                "scene_timestamp_ns": int(scene_timestamps_ns[frame_index]),
                "primary_target_entity": None,
                "primary_target_region": "no_gaze_sample",
                "gaze_sample_count": 0,
                "valid_gaze_sample_count": 0,
                "mean_confidence": 0.0,
                "target_distribution_json": "{}",
                "review_required": False,
                "annotation_status": "no_gaze_sample",
            })
            continue
        valid_count = int(frame_rows["gaze3d_ray_valid"].fillna(False).astype(bool).sum())
        distribution = Counter(
            f"{entity or 'none'}|{region}"
            for entity, region in zip(
                frame_rows["target_entity_id"].where(frame_rows["target_entity_id"].notna(), None),
                frame_rows["target_region"],
            )
        )
        score: dict[tuple[str | None, str], float] = defaultdict(float)
        for _, sample in frame_rows.iterrows():
            key = (
                str(sample["target_entity_id"]) if pd.notna(sample["target_entity_id"]) else None,
                str(sample["target_region"]),
            )
            score[key] += max(0.05, float(sample.get("target_confidence", 0.0)))
        primary = max(score, key=score.get) if score else (None, SemanticRegion.UNKNOWN.value)
        status = "annotated"
        if primary[1] in {SemanticRegion.UNKNOWN.value, SemanticRegion.INVALID_GAZE.value, SemanticRegion.OUT_OF_FRAME.value}:
            status = primary[1]
        rows.append({
            "scene_frame_index": frame_index,
            "scene_timestamp_ns": int(scene_timestamps_ns[frame_index]),
            "primary_target_entity": primary[0],
            "primary_target_region": primary[1],
            "gaze_sample_count": int(len(frame_rows)),
            "valid_gaze_sample_count": valid_count,
            "mean_confidence": float(pd.to_numeric(frame_rows["target_confidence"], errors="coerce").fillna(0).mean()),
            "target_distribution_json": json.dumps(distribution, separators=(",", ":"), sort_keys=True),
            "review_required": bool(frame_rows["review_required"].fillna(False).astype(bool).any()),
            "annotation_status": status,
        })
    return pd.DataFrame(rows)


def create_semantic_gaze_targets(
    store: ProjectStore,
    *,
    session_id: str,
    gaze_rays_artifact_id: str,
    timeseries_asset_id: str | None = None,
    facemapper_asset_id: str | None = None,
    body_asset_ids: list[str] | None = None,
    body_profile_path: str | None = None,
    entity_marker_prefixes: dict[str, str] | None = None,
    densepose_asset_id: str | None = None,
    densepose_profile_path: str | None = None,
    scene_frame_tolerance_ms: float = 10.0,
    face_tolerance_ms: float = 8.0,
    face_left_right_policy: Literal["anatomical_auto", "as_exported", "swap_eyes"] = "anatomical_auto",
    max_marker_gap_sec: float = 0.1,
    uncertainty_cone_deg: float = 1.5,
    maximum_target_distance_m: float = 10.0,
    region_frame_policy: Literal["all_scene_frames", "gaze_frames_only"] = "all_scene_frames",
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    body_asset_ids = body_asset_ids or []
    entity_marker_prefixes = entity_marker_prefixes or {}
    session = store.project.sessions[session_id]
    if gaze_rays_artifact_id not in session.artifacts:
        raise ValidationError(f"Unknown gaze-ray artifact: {gaze_rays_artifact_id}")
    gaze_artifact = session.artifacts[gaze_rays_artifact_id]
    if gaze_artifact.kind != ArtifactKind.GAZE_RAYS:
        raise ValidationError("gaze_rays_artifact_id must reference a GAZE_RAYS artifact")
    cloud_asset = _resolve_cloud_asset(session, gaze_artifact, timeseries_asset_id)
    if not cloud_asset.primary_time_domain_id:
        raise ValidationError("Timeseries asset has no primary Pupil clock")

    emit(progress, 0.03, "Loading calibrated gaze rays")
    ray_path = _artifact_table_path(gaze_artifact)
    rays = _read_table(ray_path)
    input_row_count = len(rays)
    timestamp_col = _find_column(rays.columns, ["timestamp [ns]", "timestamp_ns", "gaze timestamp [ns]"])
    x_col = _find_column(rays.columns, ["gaze x [px]", "gaze_x_px", "gaze x", "x"])
    y_col = _find_column(rays.columns, ["gaze y [px]", "gaze_y_px", "gaze y", "y"])
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
    timestamps = pd.to_numeric(rays[timestamp_col], errors="coerce")
    timestamp_valid = timestamps.notna().to_numpy()
    timestamp_ns = np.zeros(len(rays), dtype=np.int64)
    timestamp_ns[timestamp_valid] = timestamps.loc[timestamps.notna()].astype("int64").to_numpy()

    emit(progress, 0.08, "Resolving scene frames")
    scene_timestamps = read_cloud_world_timestamps(cloud_asset.source_path)
    nearest_frame, frame_error = _nearest_indices(scene_timestamps, timestamp_ns)
    frame_valid = timestamp_valid & (frame_error <= int(round(scene_frame_tolerance_ms * 1e6)))
    nearest_frame[~frame_valid] = -1
    cache = MaterializationCache(store.cache_dir())
    width, height, resolution_provenance = _scene_resolution(cache, cloud_asset, gaze_artifact)
    graph = TimeGraph.from_session(session)
    scene_global_ns = graph.convert(
        cloud_asset.primary_time_domain_id,
        session.global_time_domain_id,
        scene_timestamps,
    ).astype(np.int64)

    if gaze_artifact.clip_id and gaze_artifact.clip_id in session.clips:
        clip = session.clips[gaze_artifact.clip_id]
        scene_scope = (scene_global_ns >= clip.start_global_ns) & (scene_global_ns < clip.end_global_ns)
    else:
        scene_scope = np.ones(len(scene_timestamps), dtype=bool)
    if region_frame_policy == "gaze_frames_only":
        selected_scene_frames = np.unique(nearest_frame[frame_valid])
    elif region_frame_policy == "all_scene_frames":
        selected_scene_frames = np.flatnonzero(scene_scope)
    else:
        raise ValidationError("region_frame_policy must be all_scene_frames or gaze_frames_only")
    selected_scene_timestamps = scene_timestamps[selected_scene_frames]
    selected_scene_global = scene_global_ns[selected_scene_frames]

    cancellation.raise_if_cancelled()
    emit(progress, 0.14, "Building Face Mapper semantic AOIs")
    face_result = None
    if facemapper_asset_id:
        face_asset = session.assets[facemapper_asset_id]
        if face_asset.kind != AssetKind.FACE_MAPPER:
            raise ValidationError("facemapper_asset_id must reference a Face Mapper asset")
        face_root = cache.materialize(face_asset)
        face_result = load_face_semantics(
            face_root,
            recording_id=cloud_asset.recording_id or face_asset.recording_id,
            scene_timestamps_ns=scene_timestamps,
            width=width,
            height=height,
            tolerance_ms=face_tolerance_ms,
            left_right_policy=face_left_right_policy,
        )
        if not face_result.frame_regions.empty:
            face_result.frame_regions["source_asset_id"] = face_asset.id

    cancellation.raise_if_cancelled()
    emit(progress, 0.23, "Building C3D body primitives")
    body_profile = load_body_profile(body_profile_path)
    body_primitives: dict[int, list] = defaultdict(list)
    body_region_frames: list[pd.DataFrame] = []
    body_diagnostics: list[dict] = []
    for position, asset_id in enumerate(body_asset_ids):
        cancellation.raise_if_cancelled()
        body_asset = session.assets[asset_id]
        if body_asset.kind not in {AssetKind.C3D, AssetKind.MOCAP_NPZ}:
            raise ValidationError(f"Body asset {body_asset.label} is not C3D or canonical mocap NPZ")
        if not body_asset.primary_time_domain_id:
            raise ValidationError(f"Body asset {body_asset.label} has no time domain")
        query_ns = graph.convert(
            session.global_time_domain_id,
            body_asset.primary_time_domain_id,
            selected_scene_global,
        ).astype(np.int64)
        entity_id = body_asset.participant_id or safe_name(body_asset.label, fallback=f"person_{position+1}")
        build = build_body_primitives(
            body_asset_id=asset_id,
            body_path=body_asset.source_path,
            entity_id=entity_id,
            scene_frame_indices=selected_scene_frames,
            scene_timestamps_ns=selected_scene_timestamps,
            query_time_sec=query_ns.astype(np.float64) / 1e9,
            profile=body_profile,
            marker_prefix=entity_marker_prefixes.get(asset_id),
            max_marker_gap_sec=max_marker_gap_sec,
        )
        for frame_index, items in build.primitives_by_frame.items():
            body_primitives[frame_index].extend(items)
        if not build.frame_regions.empty:
            body_region_frames.append(build.frame_regions)
        body_diagnostics.append(build.diagnostics)
        emit(
            progress,
            0.23 + 0.17 * (position + 1) / max(1, len(body_asset_ids)),
            f"Built body geometry for {entity_id}",
        )

    cancellation.raise_if_cancelled()
    emit(progress, 0.42, "Importing optional DensePose evidence")
    densepose_result = None
    if densepose_asset_id:
        densepose_asset = session.assets[densepose_asset_id]
        if densepose_asset.kind not in {AssetKind.DENSEPOSE_EXPORT, AssetKind.TABLE, AssetKind.ARCHIVE, AssetKind.DIRECTORY}:
            raise ValidationError("DensePose input must be a DensePose export, table, ZIP, or directory")
        densepose_result = load_densepose_evidence(
            densepose_asset,
            cache,
            scene_timestamps_ns=scene_timestamps,
            profile_path=densepose_profile_path,
        )
        if not densepose_result.frame_regions.empty:
            densepose_result.frame_regions["source_asset_id"] = densepose_asset.id

    cancellation.raise_if_cancelled()
    emit(progress, 0.48, "Intersecting world gaze rays with body geometry")
    origins = rays[origin_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    directions = rays[direction_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ray_valid = rays[ray_valid_col].fillna(False).astype(bool).to_numpy()
    x_px = pd.to_numeric(rays[x_col], errors="coerce").to_numpy(dtype=float)
    y_px = pd.to_numeric(rays[y_col], errors="coerce").to_numpy(dtype=float)
    gaze_in_frame = (
        np.isfinite(x_px) & np.isfinite(y_px)
        & (x_px >= 0) & (x_px < width)
        & (y_px >= 0) & (y_px < height)
    )
    body_hits: list[BodyHit | None] = [None] * len(rays)
    indices_by_frame: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(frame_valid & ray_valid):
        indices_by_frame[int(nearest_frame[index])].append(int(index))
    for completed, (frame_index, sample_indices) in enumerate(indices_by_frame.items()):
        cancellation.raise_if_cancelled()
        primitives = body_primitives.get(frame_index, [])
        if primitives:
            sample_array = np.asarray(sample_indices, dtype=int)
            hits = intersect_frame_rays(
                origins[sample_array],
                directions[sample_array],
                primitives,
                uncertainty_cone_deg=uncertainty_cone_deg,
                maximum_distance_mm=maximum_target_distance_m * 1000.0,
            )
            for sample_index, hit in zip(sample_indices, hits):
                body_hits[sample_index] = hit
        if completed % 100 == 0:
            emit(
                progress,
                0.48 + 0.22 * (completed + 1) / max(1, len(indices_by_frame)),
                "Intersecting gaze rays with per-frame body primitives",
            )

    emit(progress, 0.71, "Fusing face, body, and imported DensePose evidence")
    appended: dict[str, list] = defaultdict(list)
    for index in range(len(rays)):
        cancellation.raise_if_cancelled()
        frame_index = int(nearest_frame[index]) if frame_valid[index] else -1
        face_hit = None
        if face_result is not None and frame_index >= 0:
            face_hit = classify_face_point(
                x_px[index],
                y_px[index],
                face_result.instances_by_frame.get(frame_index, []),
            )
        densepose_hit = None
        if densepose_result is not None and frame_index >= 0:
            densepose_hit = match_densepose_evidence(
                frame_index=frame_index,
                timestamp_ns=int(timestamp_ns[index]) if timestamp_valid[index] else None,
                evidence_by_frame=densepose_result.by_frame,
            )
        body_hit = body_hits[index]
        decision = fuse_target_evidence(
            ray_valid=bool(ray_valid[index]),
            gaze_in_frame=bool(gaze_in_frame[index]),
            face=face_hit,
            body=body_hit,
            densepose=densepose_hit,
        )
        appended["scene_frame_index"].append(frame_index if frame_index >= 0 else pd.NA)
        appended["scene_timestamp_ns"].append(int(scene_timestamps[frame_index]) if frame_index >= 0 else pd.NA)
        appended["scene_frame_match_error_ms"].append(float(frame_error[index]) / 1e6 if frame_valid[index] else np.nan)
        appended["face_2d_entity_id"].append(face_hit.entity_id if face_hit else None)
        appended["face_2d_region"].append(face_hit.region if face_hit else None)
        appended["face_2d_confidence"].append(face_hit.confidence if face_hit else np.nan)
        appended["face_2d_track_id"].append(face_hit.track_id if face_hit else pd.NA)
        appended["body_3d_entity_id"].append(body_hit.entity_id if body_hit else None)
        appended["body_3d_region"].append(body_hit.region if body_hit else None)
        appended["body_3d_distance_mm"].append(body_hit.distance_mm if body_hit else np.nan)
        appended["body_3d_hit_x_mm"].append(float(body_hit.point_mm[0]) if body_hit else np.nan)
        appended["body_3d_hit_y_mm"].append(float(body_hit.point_mm[1]) if body_hit else np.nan)
        appended["body_3d_hit_z_mm"].append(float(body_hit.point_mm[2]) if body_hit else np.nan)
        appended["body_3d_confidence"].append(body_hit.confidence if body_hit else np.nan)
        appended["body_3d_exact_hit"].append(body_hit.exact_hit if body_hit else False)
        appended["body_3d_angular_miss_deg"].append(body_hit.angular_miss_deg if body_hit else np.nan)
        appended["densepose_2d_entity_id"].append(densepose_hit.entity_id if densepose_hit else None)
        appended["densepose_2d_region"].append(densepose_hit.region if densepose_hit else None)
        appended["densepose_2d_raw_label"].append(densepose_hit.raw_label if densepose_hit else None)
        appended["densepose_2d_confidence"].append(densepose_hit.confidence if densepose_hit else np.nan)
        appended["target_entity_id"].append(decision.entity_id)
        appended["target_region"].append(decision.region)
        appended["target_confidence"].append(decision.confidence)
        appended["fusion_status"].append(decision.status)
        appended["review_required"].append(decision.review_required)
    targets = rays.copy()
    for column, values in appended.items():
        targets[column] = values

    emit(progress, 0.82, "Creating per-frame semantic summaries")
    frame_summary = _frame_summary(
        targets,
        selected_scene_frames,
        scene_timestamps,
    )
    region_frames: list[pd.DataFrame] = []
    if face_result is not None and not face_result.frame_regions.empty:
        region_frames.append(face_result.frame_regions)
    region_frames.extend(body_region_frames)
    if densepose_result is not None and not densepose_result.frame_regions.empty:
        region_frames.append(densepose_result.frame_regions)
    frame_regions = pd.concat(region_frames, ignore_index=True, sort=False) if region_frames else pd.DataFrame(columns=[
        "scene_frame_index", "scene_timestamp_ns", "entity_id", "region",
        "coordinate_space", "geometry_type", "geometry_json", "source",
        "source_asset_id", "confidence", "priority",
    ])
    if len(frame_regions):
        frame_regions = frame_regions.loc[
            frame_regions["scene_frame_index"].isin(set(int(x) for x in selected_scene_frames))
        ].sort_values(["scene_frame_index", "entity_id", "priority"], ascending=[True, True, False])

    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "semantic_targets", gaze_artifact.clip_id or "full") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    targets_csv = output_root / "gaze_targets.csv"
    summary_csv = output_root / "frame_gaze_summary.csv"
    regions_csv = output_root / "frame_regions.csv"
    targets.to_csv(targets_csv, index=False)
    frame_summary.to_csv(summary_csv, index=False)
    frame_regions.to_csv(regions_csv, index=False)
    parquet_files: list[str] = []
    if output_parquet:
        try:
            for frame_data, name in [
                (targets, "gaze_targets.parquet"),
                (frame_summary, "frame_gaze_summary.parquet"),
                (frame_regions, "frame_regions.parquet"),
            ]:
                path = output_root / name
                frame_data.to_parquet(path, index=False)
                parquet_files.append(str(path))
        except Exception:
            parquet_files = []

    target_counts = Counter(targets["target_region"].astype(str))
    review_fraction = float(targets["review_required"].astype(bool).mean()) if len(targets) else 0.0
    known_mask = ~targets["target_region"].isin([
        SemanticRegion.UNKNOWN.value,
        SemanticRegion.INVALID_GAZE.value,
        SemanticRegion.OUT_OF_FRAME.value,
    ])
    body_hit_count = int(targets["body_3d_region"].notna().sum())
    face_hit_count = int(targets["face_2d_region"].notna().sum())
    densepose_hit_count = int(targets["densepose_2d_region"].notna().sum())
    qc = {
        "input_gaze_row_count": int(input_row_count),
        "output_gaze_target_row_count": int(len(targets)),
        "row_count_preserved": bool(len(targets) == input_row_count),
        "scene_frame_count_in_scope": int(len(selected_scene_frames)),
        "scene_frame_match_fraction": float(frame_valid.mean()) if len(frame_valid) else 0.0,
        "semantic_target_fraction": float(known_mask.mean()) if len(known_mask) else 0.0,
        "review_required_fraction": review_fraction,
        "face_2d_hit_count": face_hit_count,
        "body_3d_hit_count": body_hit_count,
        "densepose_2d_hit_count": densepose_hit_count,
        "target_region_counts": dict(target_counts),
        "body_backend": "c3d_primitives" if body_asset_ids else "disabled",
        "densepose_backend": "import_only" if densepose_asset_id else "not_supplied",
        "densepose_inference_runtime_required": False,
        "windows_native_pipeline": True,
        "face_diagnostics": face_result.diagnostics if face_result else None,
        "body_diagnostics": body_diagnostics,
        "densepose_diagnostics": densepose_result.diagnostics if densepose_result else None,
    }
    manifest = {
        "schema_version": "oculostudio-semantic-targets-manifest-1.0",
        "artifact_id": artifact_id,
        "gaze_rays_artifact_id": gaze_rays_artifact_id,
        "source_asset_ids": [
            cloud_asset.id,
            *([facemapper_asset_id] if facemapper_asset_id else []),
            *body_asset_ids,
            *([densepose_asset_id] if densepose_asset_id else []),
        ],
        "clip_id": gaze_artifact.clip_id,
        "ontology": "oculostudio-semantic-region-1.0",
        "body_profile": body_profile.model_dump(mode="json"),
        "parameters": {
            "scene_frame_tolerance_ms": scene_frame_tolerance_ms,
            "face_tolerance_ms": face_tolerance_ms,
            "face_left_right_policy": face_left_right_policy,
            "max_marker_gap_sec": max_marker_gap_sec,
            "uncertainty_cone_deg": uncertainty_cone_deg,
            "maximum_target_distance_m": maximum_target_distance_m,
            "region_frame_policy": region_frame_policy,
            "output_parquet": output_parquet,
        },
        "scene_resolution": {"width": width, "height": height, **resolution_provenance},
        "densepose_slot": {
            "supported": True,
            "mode": "external_results_import",
            "detectron2_required_on_windows": False,
            "selected_asset_id": densepose_asset_id,
            "profile_path": densepose_profile_path,
        },
        "outputs": {
            "gaze_targets": str(targets_csv),
            "frame_gaze_summary": str(summary_csv),
            "frame_regions": str(regions_csv),
            "parquet": parquet_files,
        },
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "semantic_targets_manifest.json", manifest)
    atomic_write_json(output_root / "qc_metrics.json", qc)
    emit(progress, 0.95, "Registering semantic-target artifact")
    source_asset_ids = [cloud_asset.id]
    if facemapper_asset_id:
        source_asset_ids.append(facemapper_asset_id)
    source_asset_ids.extend(body_asset_ids)
    if densepose_asset_id:
        source_asset_ids.append(densepose_asset_id)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.SEMANTIC_TARGETS,
        label=f"{gaze_artifact.label} — semantic targets",
        output_root=output_root,
        source_asset_ids=list(dict.fromkeys(source_asset_ids)),
        parent_artifact_ids=[gaze_rays_artifact_id],
        clip_id=gaze_artifact.clip_id,
        parameters=manifest["parameters"],
        qc=qc,
        provenance={
            "gaze_rays_artifact_id": gaze_rays_artifact_id,
            "body_profile_path": body_profile_path,
            "densepose_profile_path": densepose_profile_path,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Semantic gaze-target annotation complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "gaze_targets_csv": str(targets_csv),
        "frame_gaze_summary_csv": str(summary_csv),
        "frame_regions_csv": str(regions_csv),
        "parquet_files": parquet_files,
        "manifest": manifest,
    }
