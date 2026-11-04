from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache, zip_directory
from oculostudio.adapters.pupil import materialized_cloud_root, materialized_native_files
from oculostudio.domain.models import ArtifactKind, Clip
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, find_executable, new_id, safe_name, utc_now


def _normalized_columns(columns) -> dict[str, str]:
    return {re.sub(r"[^a-z0-9]+", "", str(column).lower()): str(column) for column in columns}


def find_column(columns, aliases: list[str], *, required: bool = True) -> str | None:
    normalized = _normalized_columns(columns)
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.lower())
        if key in normalized:
            return normalized[key]
    if required:
        raise ValidationError(f"Missing required column; expected one of {aliases}")
    return None


def detect_timestamp_column(frame: pd.DataFrame) -> str | None:
    return find_column(
        frame.columns,
        ["timestamp [ns]", "timestamp_ns", "world timestamp [ns]", "world_timestamp_ns", "timestamp"],
        required=False,
    )


def detect_interval_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    start = find_column(
        frame.columns,
        ["start timestamp [ns]", "start_timestamp_ns", "section start time [ns]", "section_start_time_ns", "start_ns"],
        required=False,
    )
    end = find_column(
        frame.columns,
        ["end timestamp [ns]", "end_timestamp_ns", "section end time [ns]", "section_end_time_ns", "end_ns"],
        required=False,
    )
    return start, end


def slice_point_table(frame: pd.DataFrame, start_ns: int, end_ns: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp = detect_timestamp_column(frame)
    if timestamp is None:
        raise ValidationError("Point table does not contain a timestamp column")
    numeric = pd.to_numeric(frame[timestamp], errors="coerce")
    mask = numeric.ge(start_ns) & numeric.lt(end_ns)
    output = frame.loc[mask].copy().reset_index(drop=True)
    return output, {
        "mode": "point_half_open",
        "timestamp_column": timestamp,
        "rows_before": int(len(frame)),
        "rows_after": int(len(output)),
        "start_ns": int(start_ns),
        "end_ns_exclusive": int(end_ns),
        "interval_semantics": "[start, end)",
    }


def slice_interval_table(
    frame: pd.DataFrame,
    start_ns: int,
    end_ns: int,
    *,
    policy: str = "clip",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_col, end_col = detect_interval_columns(frame)
    if start_col is None or end_col is None:
        raise ValidationError("Interval table does not contain start/end timestamp columns")
    starts = pd.to_numeric(frame[start_col], errors="coerce")
    ends = pd.to_numeric(frame[end_col], errors="coerce")
    overlap = ends.gt(start_ns) & starts.lt(end_ns)
    inside = starts.ge(start_ns) & ends.le(end_ns)
    if policy == "inside":
        output = frame.loc[inside].copy()
    elif policy == "clip":
        output = frame.loc[overlap].copy()
        output[start_col] = pd.to_numeric(output[start_col], errors="coerce").clip(lower=start_ns)
        output[end_col] = pd.to_numeric(output[end_col], errors="coerce").clip(upper=end_ns)
        duration_col = find_column(output.columns, ["duration [ms]", "duration_ms"], required=False)
        if duration_col:
            output[duration_col] = (output[end_col] - output[start_col]) / 1_000_000.0
    else:
        raise ValidationError(f"Unsupported interval policy: {policy}")
    output = output.reset_index(drop=True)
    return output, {
        "mode": f"interval_{policy}",
        "start_column": start_col,
        "end_column": end_col,
        "rows_before": int(len(frame)),
        "rows_after": int(len(output)),
        "overlap_rows": int(overlap.sum()),
        "inside_rows": int(inside.sum()),
        "partial_rows": int((overlap & ~inside).sum()),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _trim_csv(path: Path, start_ns: int, end_ns: int, *, interval_policy: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    frame = _read_csv(path)
    start_col, end_col = detect_interval_columns(frame)
    timestamp = detect_timestamp_column(frame)
    if start_col and end_col:
        output, report = slice_interval_table(frame, start_ns, end_ns, policy=interval_policy)
        return output, report
    if timestamp:
        output, report = slice_point_table(frame, start_ns, end_ns)
        return output, report
    return None, {"mode": "pass_through", "rows_before": int(len(frame)), "rows_after": int(len(frame))}


def _copy_or_trim_table(
    source: Path,
    target: Path,
    start_ns: int,
    end_ns: int,
    *,
    interval_policy: str,
    recording_id: str | None,
) -> dict[str, Any]:
    frame = _read_csv(source)
    rid_col = find_column(frame.columns, ["recording id", "recording_id"], required=False)
    if rid_col and recording_id:
        selected = frame.loc[frame[rid_col].astype(str) == str(recording_id)].copy()
        other = frame.loc[frame[rid_col].astype(str) != str(recording_id)].copy()
    else:
        selected = frame
        other = frame.iloc[0:0].copy()
    start_col, end_col = detect_interval_columns(selected)
    timestamp = detect_timestamp_column(selected)
    if start_col and end_col:
        trimmed, report = slice_interval_table(selected, start_ns, end_ns, policy=interval_policy)
    elif timestamp:
        trimmed, report = slice_point_table(selected, start_ns, end_ns)
    else:
        trimmed, report = selected, {"mode": "pass_through", "rows_before": len(selected), "rows_after": len(selected)}
    # For a selected recording export, omit unrelated recordings rather than
    # silently mixing them into a derived single-session artifact.
    target.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(target, index=False)
    report["recording_filter_applied"] = bool(rid_col and recording_id)
    report["unrelated_rows_omitted"] = int(len(other))
    return report


def trim_timeseries_asset(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    clip: Clip,
    start_source_ns: int,
    end_source_ns: int,
    interval_policy: str = "clip",
    write_zip: bool = True,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    asset = session.assets[asset_id]
    cache = MaterializationCache(store.cache_dir())
    materialized = cache.materialize(asset)
    recording_root = materialized_cloud_root(materialized, asset.recording_id)
    artifact_id = new_id("artifact")
    artifact_root = store.derived_dir(session_id, "clips", clip.id) / artifact_id
    output_root = artifact_root / f"timeseries_{safe_name(asset.label)}"
    output_root.mkdir(parents=True, exist_ok=False)
    files = [p for p in recording_root.rglob("*") if p.is_file()]
    reports: dict[str, Any] = {}
    emit(progress, 0.02, "Indexing Timeseries files")
    for index, source in enumerate(files):
        cancellation.raise_if_cancelled()
        relative = source.relative_to(recording_root)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".csv":
            try:
                report = _copy_or_trim_table(
                    source,
                    target,
                    start_source_ns,
                    end_source_ns,
                    interval_policy=interval_policy,
                    recording_id=asset.recording_id,
                )
                reports[relative.as_posix()] = report
            except Exception as exc:
                shutil.copy2(source, target)
                reports[relative.as_posix()] = {"mode": "pass_through_after_error", "warning": str(exc)}
        elif source.suffix.lower() in {".json", ".txt", ".yaml", ".yml"}:
            shutil.copy2(source, target)
        elif source.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            # Scene video is handled separately because precise media trimming
            # depends on its own PTS domain. Keep only a reference note here.
            reports[relative.as_posix()] = {"mode": "omitted_media", "source": str(source)}
        elif source.stat().st_size <= 20 * 1024 * 1024:
            shutil.copy2(source, target)
        emit(progress, 0.05 + 0.75 * (index + 1) / max(1, len(files)), f"Trimming {relative.name}")
    manifest = {
        "schema_version": "oculostudio-timeseries-trim-1.0",
        "source_asset_id": asset_id,
        "recording_id": asset.recording_id,
        "clip_id": clip.id,
        "interval_semantics": "[start, end)",
        "source_start_ns": int(start_source_ns),
        "source_end_ns_exclusive": int(end_source_ns),
        "tables": reports,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "oculostudio_trim_manifest.json", manifest)
    zip_path = None
    if write_zip:
        emit(progress, 0.84, "Packaging trimmed Timeseries")
        zip_path = artifact_root / f"{safe_name(clip.label)}_timeseries.zip"
        zip_directory(output_root, zip_path)
    emit(progress, 0.95, "Registering artifact")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.TRIMMED_TIMESERIES,
        label=f"{clip.label} — Timeseries",
        output_root=artifact_root,
        source_asset_ids=[asset_id],
        clip_id=clip.id,
        parameters={
            "source_start_ns": start_source_ns,
            "source_end_ns_exclusive": end_source_ns,
            "interval_policy": interval_policy,
            "write_zip": write_zip,
        },
        qc={
            "tables_processed": len(reports),
            "table_rows_after": int(
                sum(int(row.get("rows_after", 0)) for row in reports.values() if isinstance(row, dict))
            ),
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Timeseries trim complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "folder": str(output_root),
        "zip": str(zip_path) if zip_path else None,
        "manifest": manifest,
    }


def _ffmpeg_trim(
    video_path: Path,
    output_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    ffmpeg_path: str | None = None,
    preserve_audio: bool = True,
    crf: int = 20,
    preset: str = "fast",
) -> list[str]:
    ffmpeg = find_executable("ffmpeg", ffmpeg_path)
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValidationError("Video trim duration must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{start_sec:.9f}",
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:v:0",
    ]
    if preserve_audio:
        command += ["-map", "0:a?"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if preserve_audio:
        command += ["-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(output_path)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode != 0:
        raise ProcessingError(f"FFmpeg video trim failed: {process.stderr.strip()}")
    return command


def trim_native_asset(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    clip: Clip,
    start_source_ns: int,
    end_source_ns: int,
    preserve_audio: bool = True,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    asset = session.assets[asset_id]
    cache = MaterializationCache(store.cache_dir())
    materialized = cache.materialize(asset)
    video_path, scene_time_path, info_path, info = materialized_native_files(materialized, asset.recording_id)
    raw = scene_time_path.read_bytes()
    if len(raw) % 8:
        raise ProcessingError("Native scene .time file is invalid")
    timestamps = np.frombuffer(raw, dtype="<i8").astype(np.int64, copy=True)
    start_index = int(np.searchsorted(timestamps, start_source_ns, side="left"))
    end_index = int(np.searchsorted(timestamps, end_source_ns, side="left"))
    if start_index >= len(timestamps) or end_index <= start_index:
        raise ValidationError("Clip does not intersect the Native scene timeline")
    end_index = min(end_index, len(timestamps))
    cancellation.raise_if_cancelled()
    interval_ns = int(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 33_333_333
    actual_start_ns = int(timestamps[start_index])
    actual_end_ns = int(timestamps[end_index]) if end_index < len(timestamps) else int(timestamps[-1] + interval_ns)
    start_sec = (actual_start_ns - int(timestamps[0])) / 1e9
    end_sec = (actual_end_ns - int(timestamps[0])) / 1e9
    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "clips", clip.id) / artifact_id / f"native_{safe_name(asset.label)}"
    output_root.mkdir(parents=True, exist_ok=False)
    output_video = output_root / f"{safe_name(clip.label)}_scene.mp4"
    emit(progress, 0.08, "Resolving Native scene frames")
    command = _ffmpeg_trim(
        video_path,
        output_video,
        start_sec=start_sec,
        end_sec=end_sec,
        preserve_audio=preserve_audio,
    )
    emit(progress, 0.76, "Writing Native timestamp slice")
    output_time = output_root / scene_time_path.name
    timestamps[start_index:end_index].astype("<i8").tofile(output_time)
    output_info = dict(info)
    output_info["oculostudio_derived_clip"] = {
        "clip_id": clip.id,
        "source_recording_id": asset.recording_id,
        "source_frame_start": start_index,
        "source_frame_end_exclusive": end_index,
        "source_timestamp_start_ns": actual_start_ns,
        "source_timestamp_end_ns_exclusive": actual_end_ns,
        "interval_semantics": "[start, end)",
    }
    atomic_write_json(output_root / "info.json", output_info)
    report = {
        "schema_version": "oculostudio-native-trim-1.0",
        "source_asset_id": asset_id,
        "source_video": str(video_path),
        "clip_id": clip.id,
        "requested_start_ns": int(start_source_ns),
        "requested_end_ns_exclusive": int(end_source_ns),
        "actual_start_ns": actual_start_ns,
        "actual_end_ns_exclusive": actual_end_ns,
        "start_frame_index": start_index,
        "end_frame_index_exclusive": end_index,
        "frame_count": end_index - start_index,
        "ffmpeg_command": command,
        "preserve_audio": preserve_audio,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "oculostudio_trim_report.json", report)
    emit(progress, 0.92, "Registering artifact")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.TRIMMED_NATIVE,
        label=f"{clip.label} — Native scene",
        output_root=output_root,
        source_asset_ids=[asset_id],
        clip_id=clip.id,
        parameters={"preserve_audio": preserve_audio},
        qc={
            "frame_count": end_index - start_index,
            "start_snap_error_ms": (actual_start_ns - start_source_ns) / 1e6,
            "end_snap_error_ms": (actual_end_ns - end_source_ns) / 1e6,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Native trim complete")
    return {"artifact": artifact.model_dump(mode="json"), "folder": str(output_root), "report": report}


def trim_facemapper_asset(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    clip: Clip,
    start_source_ns: int,
    end_source_ns: int,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    asset = session.assets[asset_id]
    cache = MaterializationCache(store.cache_dir())
    root = cache.materialize(asset)
    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "clips", clip.id) / artifact_id / f"facemapper_{safe_name(asset.label)}"
    output_root.mkdir(parents=True, exist_ok=False)
    csv_files = [p for p in root.rglob("*.csv") if ".oculostudio" not in p.parts]
    reports: dict[str, Any] = {}
    for index, source in enumerate(csv_files):
        cancellation.raise_if_cancelled()
        relative = source.relative_to(root)
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            reports[relative.as_posix()] = _copy_or_trim_table(
                source,
                target,
                start_source_ns,
                end_source_ns,
                interval_policy="clip",
                recording_id=asset.recording_id,
            )
        except Exception as exc:
            reports[relative.as_posix()] = {"mode": "error", "error": str(exc)}
        emit(progress, 0.05 + 0.82 * (index + 1) / max(1, len(csv_files)), f"Trimming {relative.name}")
    manifest = {
        "schema_version": "oculostudio-facemapper-trim-1.0",
        "source_asset_id": asset_id,
        "clip_id": clip.id,
        "source_start_ns": start_source_ns,
        "source_end_ns_exclusive": end_source_ns,
        "tables": reports,
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "oculostudio_trim_manifest.json", manifest)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.TRIMMED_FACE_MAPPER,
        label=f"{clip.label} — Face Mapper",
        output_root=output_root,
        source_asset_ids=[asset_id],
        clip_id=clip.id,
        qc={"tables_processed": len(reports)},
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Face Mapper trim complete")
    return {"artifact": artifact.model_dump(mode="json"), "folder": str(output_root), "manifest": manifest}


def run_multimodal_clip(
    store: ProjectStore,
    *,
    session_id: str,
    clip_id: str,
    asset_ids: list[str],
    interval_policy: str = "clip",
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    clip = session.clips[clip_id]
    graph = TimeGraph.from_session(session)
    results: list[dict[str, Any]] = []
    for index, asset_id in enumerate(asset_ids):
        cancellation.raise_if_cancelled()
        asset = session.assets[asset_id]
        if not asset.primary_time_domain_id:
            raise ValidationError(f"Asset has no primary time domain: {asset.label}")
        start_source = graph.convert(session.global_time_domain_id, asset.primary_time_domain_id, clip.start_global_ns)
        end_source = graph.convert(session.global_time_domain_id, asset.primary_time_domain_id, clip.end_global_ns)

        def subprogress(fraction, stage):
            emit(progress, (index + fraction) / max(1, len(asset_ids)), f"{asset.label}: {stage}")

        if asset.kind == "pupil_native":
            result = trim_native_asset(
                store,
                session_id=session_id,
                asset_id=asset_id,
                clip=clip,
                start_source_ns=int(start_source),
                end_source_ns=int(end_source),
                progress=subprogress,
                cancellation=cancellation,
            )
        elif asset.kind == "pupil_cloud":
            result = trim_timeseries_asset(
                store,
                session_id=session_id,
                asset_id=asset_id,
                clip=clip,
                start_source_ns=int(start_source),
                end_source_ns=int(end_source),
                interval_policy=interval_policy,
                progress=subprogress,
                cancellation=cancellation,
            )
        elif asset.kind == "face_mapper":
            result = trim_facemapper_asset(
                store,
                session_id=session_id,
                asset_id=asset_id,
                clip=clip,
                start_source_ns=int(start_source),
                end_source_ns=int(end_source),
                progress=subprogress,
                cancellation=cancellation,
            )
        else:
            result = {"asset_id": asset_id, "skipped": True, "reason": f"Unsupported kind: {asset.kind}"}
        results.append(result)
    emit(progress, 1.0, "Multimodal clip complete")
    return {"clip_id": clip_id, "results": results}
