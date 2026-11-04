from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from oculostudio.adapters.archive import SourceView
from oculostudio.adapters.mocap import probe_mocap
from oculostudio.adapters.pupil import find_cloud_candidates, find_facemapper_files, find_native_candidates
from oculostudio.adapters.video import probe_video
from oculostudio.domain.models import AssetKind
from oculostudio.errors import AssetError
from oculostudio.util import sha256_file


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather"}


def _looks_like_densepose_table(source: Path) -> tuple[bool, dict[str, Any]]:
    name = source.name.lower()
    name_hit = "densepose" in name or "dense_pose" in name
    columns: list[str] = []
    try:
        if source.suffix.lower() == ".csv":
            with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                columns = next(csv.reader(stream), [])
        elif source.suffix.lower() == ".tsv":
            with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                columns = next(csv.reader(stream, delimiter="\t"), [])
        elif source.suffix.lower() in {".parquet", ".feather"}:
            # Avoid loading table rows; pandas reads only metadata for these formats.
            import pandas as pd
            frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_feather(source)
            columns = [str(column) for column in frame.columns]
    except Exception:
        columns = []
    normalized = {re.sub(r"[^a-z0-9]+", "", str(column).lower()) for column in columns}
    label_hit = bool(normalized.intersection({
        "bodypart", "gazedbodypart", "denseposepart", "partlabel",
    }))
    alignment_hit = bool(normalized.intersection({
        "sceneframeindex", "worldframeindex", "frameindex",
        "timestampns", "worldtimestampns", "gazetimestampns",
    }))
    return bool((name_hit and label_hit) or (label_hit and alignment_hit)), {
        "columns": columns,
        "densepose_label_column_detected": label_hit,
        "frame_or_timestamp_column_detected": alignment_hit,
    }


def _find_densepose_members(view: SourceView) -> list[str]:
    rows = []
    for name in view.names():
        base = PurePosixPath(name).name.lower()
        if base in {"densepose.csv", "dense_pose.csv", "gaze_densepose.csv", "densepose_labels.csv"}:
            rows.append(name)
        elif ("densepose" in base or "dense_pose" in base) and PurePosixPath(base).suffix in TABLE_SUFFIXES:
            rows.append(name)
    return sorted(set(rows))


def probe_asset(path: str | Path, *, checksum: bool = True) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise AssetError(f"Asset does not exist: {source}")
    result: dict[str, Any] = {
        "source_path": str(source),
        "label": source.name,
        "kind": AssetKind.UNKNOWN.value,
        "score": 0.0,
        "size_bytes": source.stat().st_size if source.is_file() else None,
        "sha256": sha256_file(source) if checksum and source.is_file() else None,
        "recording_id": None,
        "metadata": {},
        "warnings": [],
        "time_domains": [],
    }
    suffix = source.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        result.update(kind=AssetKind.VIDEO.value, score=1.0, metadata=probe_video(source))
        result["time_domains"] = [{"label": f"{source.stem} media PTS", "clock_type": "frame_pts"}]
        return result
    if suffix in AUDIO_SUFFIXES:
        result.update(kind=AssetKind.AUDIO.value, score=1.0)
        return result
    if suffix == ".c3d":
        result.update(kind=AssetKind.C3D.value, score=1.0, metadata=probe_mocap(source))
        result["time_domains"] = [{"label": f"{source.stem} C3D time", "clock_type": "relative"}]
        return result
    if suffix == ".npz":
        metadata = probe_mocap(source)
        result.update(kind=AssetKind.MOCAP_NPZ.value, score=1.0, metadata=metadata)
        result["time_domains"] = [{"label": f"{source.stem} mocap time", "clock_type": "relative"}]
        return result
    if suffix in {".oculocal", ".g3dcal"}:
        try:
            data = json.loads(source.read_text("utf-8"))
        except Exception:
            data = {}
        result.update(kind=AssetKind.CALIBRATION.value, score=1.0, metadata={"profile": data})
        return result
    if suffix in TABLE_SUFFIXES:
        is_densepose, densepose_meta = _looks_like_densepose_table(source)
        if is_densepose:
            result.update(
                kind=AssetKind.DENSEPOSE_EXPORT.value,
                score=1.0,
                metadata={
                    **densepose_meta,
                    "inference_runtime_required": False,
                    "windows_import_only": True,
                },
            )
            result["time_domains"] = [{
                "label": f"{source.stem} DensePose result time",
                "clock_type": "absolute",
                "role": "densepose_clock",
            }]
        else:
            result.update(kind=AssetKind.TABLE.value, score=0.9)
        return result

    if source.is_dir() or suffix == ".zip":
        with SourceView(source) as view:
            densepose_members = _find_densepose_members(view)
            native = find_native_candidates(view)
            cloud = find_cloud_candidates(view)
            facemapper = find_facemapper_files(view)
            if densepose_members and not native and not cloud:
                result.update(
                    kind=AssetKind.DENSEPOSE_EXPORT.value,
                    score=1.0,
                    metadata={
                        "files": densepose_members,
                        "file_count": len(view.names()),
                        "inference_runtime_required": False,
                        "windows_import_only": True,
                    },
                )
                result["time_domains"] = [{
                    "label": "DensePose exported result time",
                    "clock_type": "absolute",
                    "role": "densepose_clock",
                }]
                return result
            if native:
                candidate = native[0]
                result.update(
                    kind=AssetKind.PUPIL_NATIVE.value,
                    score=1.0,
                    recording_id=candidate.recording_id,
                    metadata={
                        "candidate_count": len(native),
                        "recordings": [
                            {
                                "recording_id": row.recording_id,
                                "root": row.root,
                                "video": row.video_name,
                                "time": row.time_name,
                                "info": row.info_name,
                            }
                            for row in native
                        ],
                        "file_count": len(view.names()),
                    },
                )
                result["time_domains"] = [
                    {"label": "Pupil recording clock", "clock_type": "absolute", "role": "pupil_clock"},
                    {"label": "Native scene media PTS", "clock_type": "frame_pts", "role": "native_scene_pts"},
                ]
                if len(native) > 1:
                    result["warnings"].append("The source contains multiple Native recordings; select one before processing.")
                return result
            if cloud:
                candidate = cloud[0]
                result.update(
                    kind=AssetKind.PUPIL_CLOUD.value,
                    score=1.0,
                    recording_id=candidate.recording_id,
                    metadata={
                        "candidate_count": len(cloud),
                        "recordings": [
                            {
                                "recording_id": row.recording_id,
                                "root": row.root,
                                "world_timestamps": row.world_timestamps_name,
                                "gaze": row.gaze_name,
                                "scene_video": row.scene_video_name,
                            }
                            for row in cloud
                        ],
                        "file_count": len(view.names()),
                    },
                )
                result["time_domains"] = [
                    {"label": "Pupil recording clock", "clock_type": "absolute", "role": "pupil_clock"},
                    {"label": "Cloud scene media PTS", "clock_type": "frame_pts", "role": "cloud_scene_pts"},
                ]
                return result
            if facemapper["detections"] and facemapper["sections"]:
                result.update(
                    kind=AssetKind.FACE_MAPPER.value,
                    score=1.0,
                    metadata={"files": facemapper, "file_count": len(view.names())},
                )
                result["time_domains"] = [
                    {"label": "Pupil recording clock", "clock_type": "absolute", "role": "pupil_clock"}
                ]
                return result
            result.update(
                kind=(AssetKind.DIRECTORY.value if source.is_dir() else AssetKind.ARCHIVE.value),
                score=0.5,
                metadata={"file_count": len(view.names()), "sample_files": view.names()[:100]},
            )
            return result
    return result
