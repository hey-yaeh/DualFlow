from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache, SourceView
from oculostudio.domain.models import AssetRecord
from oculostudio.domain.semantic import DensePoseImportProfile, SemanticRegion, canonical_region
from oculostudio.errors import ValidationError
from oculostudio.util import read_json


@dataclass(frozen=True)
class DensePoseEvidence:
    source_row_index: int
    scene_frame_index: int | None
    timestamp_ns: int | None
    entity_id: str | None
    region: str
    raw_label: str
    confidence: float
    mapped_to_ontology: bool
    geometry_json: str | None = None


@dataclass
class DensePoseImportResult:
    evidence: list[DensePoseEvidence]
    by_frame: dict[int, list[DensePoseEvidence]]
    frame_regions: pd.DataFrame
    diagnostics: dict[str, Any]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _column(columns, aliases: list[str], *, required: bool = False) -> str | None:
    mapping = {_normalize(column): column for column in columns}
    for alias in aliases:
        key = _normalize(alias)
        if key in mapping:
            return mapping[key]
    if required:
        raise ValidationError(f"DensePose export is missing required column: {aliases[0]}")
    return None


def _read_densepose_table(asset: AssetRecord, cache: MaterializationCache) -> tuple[Path, pd.DataFrame]:
    source = Path(asset.source_path).expanduser().resolve()
    if source.is_file() and source.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
        path = source
    else:
        root = cache.materialize(asset)
        candidates = []
        if root.is_file():
            candidates = [root]
        else:
            for path_candidate in root.rglob("*"):
                if not path_candidate.is_file():
                    continue
                name = path_candidate.name.lower()
                if name in {"densepose.csv", "dense_pose.csv", "gaze_densepose.csv", "densepose_labels.csv"}:
                    candidates.append(path_candidate)
            if not candidates:
                candidates = [
                    path_candidate for path_candidate in root.rglob("*.csv")
                    if "densepose" in path_candidate.name.lower() or "dense_pose" in path_candidate.name.lower()
                ]
        if len(candidates) != 1:
            raise ValidationError(
                f"Expected one DensePose result table, found {len(candidates)}. "
                "Use a standalone table or keep one densepose.csv in the export package."
            )
        path = candidates[0]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".feather":
        frame = pd.read_feather(path)
    else:
        raise ValidationError(f"Unsupported DensePose result format: {suffix}")
    return path, frame


def _nearest_indices(reference_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_ns, dtype=np.int64)
    query = np.asarray(query_ns, dtype=np.int64)
    if len(reference) == 0:
        return np.full(len(query), -1, dtype=int), np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
    pos = np.searchsorted(reference, query, side="left")
    right = np.clip(pos, 0, len(reference) - 1)
    left = np.clip(pos - 1, 0, len(reference) - 1)
    left_error = np.abs(query - reference[left])
    right_error = np.abs(reference[right] - query)
    choose_right = right_error < left_error
    return np.where(choose_right, right, left), np.where(choose_right, right_error, left_error)


def load_densepose_import_profile(path: str | Path | None) -> DensePoseImportProfile:
    if not path:
        return DensePoseImportProfile()
    return DensePoseImportProfile.model_validate(read_json(path))


def load_densepose_evidence(
    asset: AssetRecord,
    cache: MaterializationCache,
    *,
    scene_timestamps_ns: np.ndarray,
    profile_path: str | Path | None = None,
) -> DensePoseImportResult:
    profile = load_densepose_import_profile(profile_path)
    source, frame = _read_densepose_table(asset, cache)
    label_col = _column(
        frame.columns,
        [
            "body part", "body_part", "gazed body part", "gazed_body_part",
            "densepose part", "densepose_part", "part label", "part_label", "part",
        ],
        required=True,
    )
    frame_col = _column(frame.columns, ["scene frame index", "scene_frame_index", "world frame index", "world_frame_index", "frame index", "frame_index"])
    timestamp_col = _column(frame.columns, ["timestamp [ns]", "timestamp_ns", "world timestamp [ns]", "world_timestamp_ns", "gaze timestamp [ns]", "gaze_timestamp_ns"])
    entity_col = _column(frame.columns, ["entity id", "entity_id", "person id", "person_id", "participant id", "participant_id", "instance id", "instance_id"])
    confidence_col = _column(frame.columns, ["confidence", "score", "probability", "overlap fraction", "overlap_fraction"])
    geometry_col = _column(frame.columns, ["mask rle", "mask_rle", "geometry json", "geometry_json", "polygon json", "polygon_json"])
    if not frame_col and not timestamp_col:
        raise ValidationError(
            "DensePose export needs scene_frame_index or a nanosecond timestamp. "
            "The Windows importer does not guess row-to-frame alignment."
        )

    frame_indices = np.full(len(frame), -1, dtype=int)
    timestamp_values = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
    matched_by_timestamp = np.zeros(len(frame), dtype=bool)
    timestamp_errors = np.full(len(frame), np.iinfo(np.int64).max, dtype=np.int64)
    if frame_col:
        parsed = pd.to_numeric(frame[frame_col], errors="coerce")
        valid = parsed.notna()
        frame_indices[valid.to_numpy()] = parsed.loc[valid].astype(int).to_numpy()
    if timestamp_col:
        parsed_ts = pd.to_numeric(frame[timestamp_col], errors="coerce")
        valid_ts = parsed_ts.notna()
        timestamp_values[valid_ts.to_numpy()] = parsed_ts.loc[valid_ts].astype("int64").to_numpy()
        nearest, errors = _nearest_indices(
            scene_timestamps_ns,
            timestamp_values[valid_ts.to_numpy()],
        )
        accepted = errors <= int(round(profile.timestamp_tolerance_ms * 1e6))
        positions = np.flatnonzero(valid_ts.to_numpy())
        frame_indices[positions[accepted]] = nearest[accepted]
        timestamp_errors[positions] = errors
        matched_by_timestamp[positions[accepted]] = True

    evidence: list[DensePoseEvidence] = []
    invalid_rows = 0
    unmapped_labels = 0
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        scene_frame = int(frame_indices[ordinal])
        if scene_frame < 0 or scene_frame >= len(scene_timestamps_ns):
            invalid_rows += 1
            continue
        raw_label = str(row[label_col]).strip()
        region, mapped = canonical_region(raw_label, custom_mapping=profile.label_mapping)
        if not mapped:
            unmapped_labels += 1
            if region.startswith("densepose_part_"):
                if profile.unknown_numeric_policy == "error":
                    raise ValidationError(
                        f"DensePose numeric part {raw_label!r} has no mapping. "
                        "Provide a DensePose import profile JSON."
                    )
                if profile.unknown_numeric_policy == "unknown":
                    region = SemanticRegion.UNKNOWN.value
        confidence = 0.70
        if confidence_col:
            try:
                confidence = float(row[confidence_col])
            except Exception:
                pass
        confidence = float(np.clip(confidence, 0.0, 1.0))
        entity = None
        if entity_col and pd.notna(row[entity_col]):
            entity = str(row[entity_col]).strip() or None
        geometry = None
        if geometry_col and pd.notna(row[geometry_col]):
            geometry = str(row[geometry_col])
        timestamp = None
        if timestamp_values[ordinal] != np.iinfo(np.int64).min:
            timestamp = int(timestamp_values[ordinal])
        item = DensePoseEvidence(
            source_row_index=ordinal,
            scene_frame_index=scene_frame,
            timestamp_ns=timestamp,
            entity_id=entity,
            region=region,
            raw_label=raw_label,
            confidence=confidence,
            mapped_to_ontology=mapped,
            geometry_json=geometry,
        )
        evidence.append(item)

    by_frame: dict[int, list[DensePoseEvidence]] = {}
    region_rows: list[dict] = []
    for item in evidence:
        assert item.scene_frame_index is not None
        by_frame.setdefault(item.scene_frame_index, []).append(item)
        region_rows.append({
            "scene_frame_index": item.scene_frame_index,
            "scene_timestamp_ns": int(scene_timestamps_ns[item.scene_frame_index]),
            "entity_id": item.entity_id or "densepose_person_unknown",
            "region": item.region,
            "coordinate_space": "image_external",
            "geometry_type": "densepose_export" if item.geometry_json is None else "external_mask_or_polygon",
            "geometry_json": item.geometry_json or json.dumps({"raw_label": item.raw_label}, separators=(",", ":")),
            "source": "densepose_import",
            "source_asset_id": asset.id,
            "confidence": item.confidence,
            "priority": 55,
            "densepose_raw_label": item.raw_label,
            "densepose_mapped_to_ontology": item.mapped_to_ontology,
        })
    diagnostics = {
        "source_file": str(source),
        "input_row_count": int(len(frame)),
        "accepted_row_count": int(len(evidence)),
        "invalid_alignment_rows": int(invalid_rows),
        "unmapped_label_count": int(unmapped_labels),
        "matched_by_timestamp_count": int(matched_by_timestamp.sum()),
        "median_timestamp_error_ms": (
            float(np.median(timestamp_errors[timestamp_errors != np.iinfo(np.int64).max])) / 1e6
            if np.any(timestamp_errors != np.iinfo(np.int64).max)
            else None
        ),
        "label_column": label_col,
        "frame_column": frame_col,
        "timestamp_column": timestamp_col,
        "entity_column": entity_col,
        "inference_runtime_required": False,
        "windows_import_only": True,
    }
    return DensePoseImportResult(
        evidence=evidence,
        by_frame=by_frame,
        frame_regions=pd.DataFrame(region_rows),
        diagnostics=diagnostics,
    )


def match_densepose_evidence(
    *,
    frame_index: int,
    timestamp_ns: int | None,
    evidence_by_frame: dict[int, list[DensePoseEvidence]],
) -> DensePoseEvidence | None:
    rows = evidence_by_frame.get(int(frame_index), [])
    if not rows:
        return None
    if timestamp_ns is not None:
        timestamp_rows = [row for row in rows if row.timestamp_ns is not None]
        if timestamp_rows:
            return min(timestamp_rows, key=lambda row: abs(int(row.timestamp_ns) - int(timestamp_ns)))
    return max(rows, key=lambda row: row.confidence)
