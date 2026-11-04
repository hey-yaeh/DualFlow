from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from oculostudio.adapters.pupil import read_cloud_world_timestamps, read_native_scene_timestamps
from oculostudio.domain.models import ArtifactKind, TimeAnchor, TimeMap
from oculostudio.domain.timegraph import fit_time_map
from oculostudio.errors import ValidationError
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json, new_id


def nearest_indices(sorted_values: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.asarray(sorted_values, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    if len(sorted_values) == 0:
        raise ValidationError("Cannot match against an empty timestamp array")
    pos = np.searchsorted(sorted_values, query, side="left")
    right = np.clip(pos, 0, len(sorted_values) - 1)
    left = np.clip(pos - 1, 0, len(sorted_values) - 1)
    choose_left = np.abs(query - sorted_values[left]) <= np.abs(query - sorted_values[right])
    index = np.where(choose_left, left, right)
    delta = query - sorted_values[index]
    return index.astype(np.int64), delta.astype(np.int64)


def fit_and_store_time_map(
    store: ProjectStore,
    *,
    session_id: str,
    source_domain_id: str,
    target_domain_id: str,
    anchors: list[dict],
    method: str = "robust_affine",
) -> TimeMap:
    model_anchors = [TimeAnchor.model_validate(anchor) for anchor in anchors]
    mapping = fit_time_map(source_domain_id, target_domain_id, model_anchors, method=method)

    def mutate(project):
        session = project.sessions[session_id]
        # Replace an existing direct map for the same ordered pair. Other maps
        # remain available, so alternative hypotheses can still be represented
        # through different intermediate domains.
        for mapping_id, existing in list(session.time_maps.items()):
            if (
                existing.source_domain_id == source_domain_id
                and existing.target_domain_id == target_domain_id
            ):
                del session.time_maps[mapping_id]
        session.time_maps[mapping.id] = mapping

    store.update(mutate)
    return mapping


def build_native_cloud_frame_map(
    store: ProjectStore,
    *,
    session_id: str,
    native_asset_id: str,
    cloud_asset_id: str,
    maximum_error_ms: float = 5.0,
    progress=None,
    cancellation=None,
):
    from oculostudio.util import CancellationToken, emit

    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    native = session.assets[native_asset_id]
    cloud = session.assets[cloud_asset_id]
    emit(progress, 0.05, "Reading Native scene timestamps")
    native_ts = read_native_scene_timestamps(native.source_path)
    cancellation.raise_if_cancelled()
    emit(progress, 0.25, "Reading Cloud world timestamps")
    world_ts = read_cloud_world_timestamps(cloud.source_path)
    cancellation.raise_if_cancelled()
    emit(progress, 0.45, "Matching physical scene frames")
    indices, delta = nearest_indices(world_ts, native_ts)
    maximum_error_ns = int(round(maximum_error_ms * 1_000_000))
    valid = np.abs(delta) <= maximum_error_ns
    rows = pd.DataFrame(
        {
            "native_frame_index": np.arange(len(native_ts), dtype=np.int64),
            "native_timestamp_ns": native_ts,
            "cloud_world_row": indices,
            "cloud_timestamp_ns": world_ts[indices],
            "native_minus_cloud_ns": delta,
            "valid": valid,
        }
    )
    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "frame_maps", artifact_id)
    output = output_root / "native_cloud_frame_map.csv"
    rows.to_csv(output, index=False)
    emit(progress, 0.78, "Computing frame-map quality metrics")
    cancellation.raise_if_cancelled()
    qc = {
        "native_frame_count": int(len(native_ts)),
        "cloud_world_row_count": int(len(world_ts)),
        "matched_fraction": float(valid.mean()),
        "max_abs_error_ms": float(np.max(np.abs(delta)) / 1e6),
        "median_abs_error_ms": float(np.median(np.abs(delta)) / 1e6),
        "cloud_row_minus_native_frame_median": float(np.median(indices - np.arange(len(native_ts)))),
        "row_mapping_contiguous_fraction": float(np.mean(np.diff(indices) == 1)) if len(indices) > 1 else 1.0,
    }
    report_path = output.with_suffix(".json")
    atomic_write_json(report_path, qc)
    emit(progress, 0.9, "Registering frame-map artifact")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.FRAME_MAP,
        label="Native–Cloud frame map",
        output_root=output_root,
        source_asset_ids=[native_asset_id, cloud_asset_id],
        qc=qc,
        parameters={"maximum_error_ms": maximum_error_ms},
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Native–Cloud frame map complete")
    return {"artifact": artifact.model_dump(mode="json"), "qc": qc, "csv": str(output)}
