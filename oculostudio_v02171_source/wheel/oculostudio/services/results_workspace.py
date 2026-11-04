from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from oculostudio.domain.models import ArtifactKind, ArtifactRecord
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.storage import ArtifactStore, ProjectStore
from oculostudio.util import (
    CancellationToken,
    atomic_write_json,
    emit,
    new_id,
    safe_name,
    sha256_file,
    utc_now,
)


_STAGE_BY_KIND: dict[str, str] = {
    ArtifactKind.NATIVE_REFERENCE_TRIM.value: "01_trim",
    ArtifactKind.TRIMMED_NATIVE.value: "01_trim",
    ArtifactKind.TRIMMED_TIMESERIES.value: "01_trim",
    ArtifactKind.TRIMMED_FACE_MAPPER.value: "01_trim",
    ArtifactKind.FACE_MAPPER_FILTER.value: "02_face_mapper_qc",
    ArtifactKind.PRIVACY_VIDEO.value: "02_privacy",
    ArtifactKind.PRIVACY_CLIP.value: "02_privacy",
    ArtifactKind.PARTICIPANT_ASSOCIATION.value: "03_identity",
    ArtifactKind.FACE_AOI.value: "03_face_aoi",
    ArtifactKind.FRAME_MAP.value: "04_sync",
    ArtifactKind.TIME_SYNC.value: "04_sync",
    ArtifactKind.CALIBRATION_PROFILE.value: "05_calibration",
    ArtifactKind.CALIBRATION_ACCEPTANCE.value: "05_calibration",
    ArtifactKind.ETM_ROLE_ASSIGNMENT.value: "06_etm_assignment",
    ArtifactKind.GAZE_RAYS.value: "07_gaze3d",
    ArtifactKind.BODY_AOI.value: "08_body_aoi",
    ArtifactKind.OBJECT_AOI.value: "09_object_aoi",
    ArtifactKind.TARGET_FUSION.value: "10_fusion",
    ArtifactKind.SEMANTIC_TARGETS.value: "10_fusion",
    ArtifactKind.DYADIC_ATTENTION.value: "11_dyadic_attention",
    ArtifactKind.AOI_REVIEW_VIDEO.value: "12_review",
    ArtifactKind.QC_REPORT.value: "99_qc",
    ArtifactKind.RESULTS_PACKAGE.value: "99_export_package",
}

_STAGE_DESCRIPTIONS: dict[str, str] = {
    "01_trim": "Native-reference trimmed TSD and Face Mapper data plus frame/time maps.",
    "02_face_mapper_qc": "Auditable FaceMapper candidate filtering with Privacy-safe and Analysis-strict packages.",
    "02_privacy": "Full or trimmed face-masked videos and frame-level privacy audit data.",
    "03_identity": "C3D-projected participant heads and Face Mapper identity assignment evidence.",
    "03_face_aoi": "Per-gaze and per-frame facial AOI labels, region geometry, and optional overlay video.",
    "04_sync": "Explicit clock/frame mappings and synchronization evidence.",
    "05_calibration": "Camera-to-tracker profiles, transfer validation, production acceptance gates, and quality-control evidence.",
    "06_etm_assignment": "ETM wearer/role assignment and distance/shape evidence.",
    "07_gaze3d": "Calibrated world-space 3D gaze rays and ray-validity reports.",
    "08_body_aoi": "Per-gaze and per-frame 3D body-region labels and geometry evidence.",
    "09_object_aoi": "Per-gaze object-instance labels derived from marker-prop C3D geometry.",
    "10_fusion": "Independent Face/Body/Object evidence fused into complete gaze targets.",
    "11_dyadic_attention": "Mutual gaze, joint attention and gaze-following episodes on a common interaction timeline.",
    "12_review": "Pre-rendered AOI review videos and review frame maps.",
    "99_qc": "Session-level QC reports.",
    "99_export_package": "Previously generated organized result packages.",
    "98_other": "Other immutable derived artifacts.",
}

_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "recording_id": "Stable Pupil recording identifier.",
    "session_id": "OculoStudio session identifier.",
    "interaction_id": "Interaction identifier inferred or supplied for this export package.",
    "role_id": "Role identifier inferred or supplied for this export package.",
    "gaze_sample_uid": "Stable identifier used to join Face AOI, Body AOI, and fusion outputs.",
    "source_gaze_row_index": "Original gaze-table row index before trimming/annotation.",
    "oculo_source_row_index": "Original source row index retained by OculoStudio.",
    "timestamp_ns": "Original timestamp in integer nanoseconds.",
    "timestamp_global_ns": "Timestamp mapped to the session Global Time domain, in integer nanoseconds.",
    "clip_time_ns": "Time from the retained clip start, in integer nanoseconds.",
    "oculo_clip_time_ns": "Time from the retained clip start, in integer nanoseconds.",
    "native_frame_index": "Frame ordinal in the source Native scene video.",
    "output_frame_index": "Frame ordinal in the generated output video.",
    "clip_frame_index": "Frame ordinal in the retained/trimmed video.",
    "gaze_x_px": "Gaze x-coordinate in scene-camera pixels.",
    "gaze_y_px": "Gaze y-coordinate in scene-camera pixels.",
    "face_region": "Assigned facial AOI label for this gaze sample.",
    "matched_entity_id": "Participant or other-person identity assigned to a Face Mapper detection.",
    "target_relation": "Relation of the face/body target to the ego wearer: self, partner, or other_person.",
    "identity_status": "Participant-association status such as matched_participant, untracked_person, or ambiguous_person.",
    "identity_confidence": "Confidence of the participant-to-face association.",
    "include_in_partner_analysis": "True when the face is a confidently matched conversation partner under partner-only analysis.",
    "projected_head_x_px": "C3D head centre projected into scene-camera pixel coordinates.",
    "projected_head_y_px": "C3D head centre projected into scene-camera pixel coordinates.",
    "projected_head_depth_m": "Depth of the projected C3D head in the ego camera frame, in metres.",
    "face_aoi_confidence": "Confidence assigned by the Face AOI procedure.",
    "body_region": "Assigned 3D body-region label for this gaze sample.",
    "body_confidence": "Confidence assigned by the Body AOI procedure.",
    "object_entity_id": "Stable prop instance identifier from Object C3D or an object profile.",
    "object_category": "Object category such as cup, plate, guitar or pillow.",
    "object_confidence": "Confidence assigned by the Object AOI procedure.",
    "mutual_face_gaze": "True when both roles are simultaneously looking at the other participant's face/head.",
    "mutual_eye_contact": "True when both roles are simultaneously looking at the other participant's eye region.",
    "joint_attention": "True when both roles share the same object or third-person target on the common timeline.",
    "initiator_role": "Role whose target onset preceded the other role for a joint-attention episode.",
    "final_target_region": "Final region after fusing independent evidence.",
    "final_entity_id": "Final target entity after fusing independent evidence.",
    "final_confidence": "Confidence of the fused target assignment.",
    "fusion_status": "Rule/status describing how Face, Body and optional Object evidence were combined.",
    "review_required": "True when the sample/frame should be manually reviewed.",
    "ray_valid": "Whether a calibrated world-space gaze ray could be generated.",
    "gaze3d_ray_valid": "Whether a calibrated world-space gaze ray could be generated.",
    "worn": "Eye-tracker worn state retained from the source data.",
    "blink_id": "Source blink identifier, when present.",
    "fixation_id": "Source fixation identifier, when present.",
    "sha256": "SHA-256 digest for integrity verification.",
}


def stage_for_kind(kind: str) -> str:
    return _STAGE_BY_KIND.get(str(kind), "98_other")


def _artifact_root(artifact: ArtifactRecord) -> Path | None:
    paths = [Path(item.path).expanduser().resolve() for item in artifact.files if item.path]
    existing = [path for path in paths if path.exists()]
    paths = existing or paths
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0].parent if paths[0].suffix else paths[0]
    try:
        common = Path(os.path.commonpath([str(path) for path in paths]))
    except ValueError:
        return paths[0].parent
    if common.is_file():
        return common.parent
    return common


def artifact_root_path(artifact: ArtifactRecord) -> str | None:
    root = _artifact_root(artifact)
    return str(root) if root else None


def _extract_code(texts: Iterable[str], prefix: str) -> str | None:
    pattern = re.compile(rf"(?<![A-Za-z0-9])({re.escape(prefix)}\d{{1,6}})(?![A-Za-z0-9])", re.IGNORECASE)
    for text in texts:
        match = pattern.search(str(text or ""))
        if match:
            return match.group(1).upper()
    return None


def infer_sir_codes(store: ProjectStore, session_id: str, artifact_ids: Iterable[str]) -> dict[str, str | None]:
    session = store.snapshot().sessions[session_id]
    texts: list[str] = [session.label]
    for artifact_id in artifact_ids:
        artifact = session.artifacts.get(artifact_id)
        if not artifact:
            continue
        texts.extend([artifact.label, artifact.clip_id or ""])
        texts.extend(str(value) for value in artifact.parameters.values() if isinstance(value, (str, int)))
        texts.extend(str(value) for value in artifact.provenance.values() if isinstance(value, (str, int)))
        for asset_id in artifact.source_asset_ids:
            asset = session.assets.get(asset_id)
            if asset:
                texts.extend([asset.label, asset.source_path, asset.participant_id or ""])
    return {
        "session_code": _extract_code(texts, "S"),
        "interaction_code": _extract_code(texts, "I"),
        "role_code": _extract_code(texts, "R"),
    }


def _normalize_code(value: str | None, prefix: str, fallback: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return fallback
    if re.fullmatch(rf"{prefix}\d{{1,6}}", text):
        return text
    if re.fullmatch(r"\d{1,6}", text):
        return prefix + text.zfill(2)
    raise ValidationError(f"{prefix} code must look like {prefix}001 or a number; received {value!r}")


def _copy_file(
    source: Path,
    destination: Path,
    *,
    token: CancellationToken,
    bytes_done: int,
    bytes_total: int,
    progress,
    stage: str,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    copied = 0
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            while True:
                token.raise_if_cancelled()
                block = src.read(8 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
                copied += len(block)
                if bytes_total:
                    emit(progress, 0.06 + 0.78 * (bytes_done + copied) / bytes_total, stage)
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return copied


def _table_schema(path: Path, package_relative: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    frame: pd.DataFrame | None = None
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path, nrows=256, low_memory=False)
        elif suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t", nrows=256, low_memory=False)
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore

                schema = pq.read_schema(path)
                for field in schema:
                    name = str(field.name)
                    rows.append({
                        "file": package_relative,
                        "column": name,
                        "dtype": str(field.type),
                        "description": _COLUMN_DESCRIPTIONS.get(name, ""),
                    })
                return rows
            except Exception:
                frame = pd.read_parquet(path).head(256)
        else:
            return rows
    except Exception as exc:
        return [{
            "file": package_relative,
            "column": "<schema unavailable>",
            "dtype": "",
            "description": f"Could not inspect schema: {exc}",
        }]
    if frame is not None:
        for name, dtype in frame.dtypes.items():
            key = str(name)
            rows.append({
                "file": package_relative,
                "column": key,
                "dtype": str(dtype),
                "description": _COLUMN_DESCRIPTIONS.get(key, ""),
            })
    return rows


def _readme_markdown(
    *,
    project_name: str,
    session_label: str,
    session_code: str,
    interaction_code: str,
    role_code: str,
    label: str,
    copy_mode: str,
    artifact_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    stage_counts = Counter(row["stage"] for row in artifact_rows)
    stage_lines = "\n".join(
        f"- `{stage}` — {_STAGE_DESCRIPTIONS.get(stage, '')} ({count} artifact(s))"
        for stage, count in sorted(stage_counts.items())
    ) or "- No artifacts"
    warning_lines = "\n".join(f"- {item}" for item in warnings) or "- None"
    return f"""# {label}

Generated by OculoStudio for **{project_name}**.

## Identity

- Studio session: `{session_label}`
- Session code: `{session_code}`
- Interaction code: `{interaction_code}`
- Role code: `{role_code}`
- Package mode: `{copy_mode}`

## Stage folders

{stage_lines}

## Package files

- `artifact_index.csv`: one row per selected artifact file.
- `data_dictionary.csv`: table columns and stored data types where readable.
- `processing_manifest.json`: complete selected-artifact lineage and package settings.
- `checksums.sha256`: integrity digests for copied package files or referenced source files.
- `README.html`: browser-friendly version of this index.

## Warnings

{warning_lines}

## Important use notes

- CSV files are convenient for inspection; Parquet files preserve column types and int64 nanosecond timestamps more reliably.
- MP4 overlays are QC/visualization outputs. Statistical analysis should use the CSV/Parquet tables and manifests.
- Do not delete frame maps or manifests when moving AOI videos; they retain source-frame and timestamp lineage.
"""


def _readme_html(markdown_text: str, file_rows: list[dict[str, Any]]) -> str:
    # A purpose-built HTML index avoids adding a Markdown runtime dependency.
    title = html.escape(markdown_text.splitlines()[0].lstrip("# ") or "OculoStudio results")
    links = []
    for row in file_rows:
        relative = row.get("package_relative_path")
        source = row.get("source_path")
        href = relative or source or ""
        if not href:
            continue
        links.append(
            "<tr>"
            f"<td>{html.escape(row['stage'])}</td>"
            f"<td>{html.escape(row['artifact_label'])}</td>"
            f"<td>{html.escape(row['artifact_kind'])}</td>"
            f"<td><a href='{html.escape(str(href), quote=True)}'>{html.escape(Path(str(href)).name)}</a></td>"
            f"<td>{int(row.get('size_bytes') or 0):,}</td>"
            "</tr>"
        )
    body = "\n".join(f"<p>{html.escape(line)}</p>" for line in markdown_text.splitlines()[1:] if line.strip())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--olive:#484d2c;--lime:#efff36;--cream:#f4f3ec;--ink:#292b20;--line:#d8d7cd}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--lime);font:14px/1.5 Inter,Segoe UI,sans-serif;color:var(--ink)}}
main{{max-width:1200px;margin:30px auto;background:var(--cream);border:12px solid var(--olive);border-radius:30px;padding:30px;box-shadow:0 22px 60px #25291444}}
h1{{margin:0 0 18px;font-size:30px}}p{{margin:6px 0;color:#606355}}table{{width:100%;border-collapse:collapse;margin-top:24px;background:white;border-radius:16px;overflow:hidden}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#e6e5db}}a{{color:#4c6410;font-weight:700;overflow-wrap:anywhere}}
code{{background:#e7e6dc;padding:2px 5px;border-radius:4px}}
</style></head><body><main><h1>{title}</h1>{body}
<h2>Files</h2><table><thead><tr><th>Stage</th><th>Artifact</th><th>Kind</th><th>File</th><th>Bytes</th></tr></thead><tbody>{''.join(links)}</tbody></table>
</main></body></html>"""


def create_results_package(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_ids: list[str],
    session_code: str | None = None,
    interaction_code: str | None = None,
    role_code: str | None = None,
    label: str = "OculoStudio results",
    copy_mode: Literal["copy", "index_only"] = "copy",
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    snapshot = store.snapshot()
    if session_id not in snapshot.sessions:
        raise ValidationError(f"Unknown session: {session_id}")
    session = snapshot.sessions[session_id]
    unique_ids = list(dict.fromkeys(artifact_ids))
    if not unique_ids:
        raise ValidationError("Select at least one derived artifact")
    missing_ids = [artifact_id for artifact_id in unique_ids if artifact_id not in session.artifacts]
    if missing_ids:
        raise ValidationError(f"Unknown artifact IDs: {', '.join(missing_ids)}")
    artifacts = [session.artifacts[artifact_id] for artifact_id in unique_ids]
    if any(artifact.kind == ArtifactKind.RESULTS_PACKAGE for artifact in artifacts):
        raise ValidationError("A results package cannot include another results package")

    inferred = infer_sir_codes(store, session_id, unique_ids)
    s_code = _normalize_code(session_code or inferred["session_code"], "S", safe_name(session.label, "session"))
    i_code = _normalize_code(interaction_code or inferred["interaction_code"], "I", "I_unknown")
    r_code = _normalize_code(role_code or inferred["role_code"], "R", "R_unknown")
    clean_label = safe_name(label, f"{s_code}_{i_code}_{r_code}_results")
    package_artifact_id = new_id("artifact")
    package_root = store.root / "exports" / s_code / i_code / r_code / f"{clean_label}__{package_artifact_id[-6:]}"
    if package_root.exists():
        raise ProcessingError(f"Export package path already exists: {package_root}")
    staging = package_root.with_name(".partial_" + package_root.name)
    staging.mkdir(parents=True, exist_ok=False)

    source_files: list[tuple[ArtifactRecord, Any, Path, Path | None, str]] = []
    warnings: list[str] = []
    total_bytes = 0
    for artifact in artifacts:
        root = _artifact_root(artifact)
        stage = stage_for_kind(str(artifact.kind))
        folder = safe_name(artifact.label, str(artifact.kind)) + "__" + artifact.id[-6:]
        for file in artifact.files:
            source = Path(file.path).expanduser().resolve()
            if not source.is_file():
                raise ValidationError(f"Selected artifact file is missing: {source}")
            if root:
                try:
                    relative = source.relative_to(root)
                except ValueError:
                    relative = Path(source.name)
            else:
                relative = Path(source.name)
            destination = staging / stage / folder / relative if copy_mode == "copy" else None
            source_files.append((artifact, file, source, destination, stage))
            total_bytes += int(source.stat().st_size)

    copied_bytes = 0
    file_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    checksum_lines: list[str] = []
    try:
        emit(progress, 0.02, "Preparing organized result package")
        for index, (artifact, file, source, destination, stage) in enumerate(source_files):
            token.raise_if_cancelled()
            package_relative = ""
            if copy_mode == "copy":
                assert destination is not None
                copied = _copy_file(
                    source,
                    destination,
                    token=token,
                    bytes_done=copied_bytes,
                    bytes_total=max(total_bytes, 1),
                    progress=progress,
                    stage=f"Copying {source.name} ({index + 1}/{len(source_files)})",
                )
                copied_bytes += copied
                digest = sha256_file(destination)
                package_relative = destination.relative_to(staging).as_posix()
                checksum_lines.append(f"{digest}  {package_relative}")
                schema_path = destination
            else:
                digest = file.sha256 or sha256_file(source)
                checksum_lines.append(f"{digest}  SOURCE::{source}")
                schema_path = source
            row = {
                "artifact_id": artifact.id,
                "artifact_label": artifact.label,
                "artifact_kind": str(artifact.kind),
                "stage": stage,
                "clip_id": artifact.clip_id or "",
                "file_role": file.role,
                "source_path": str(source),
                "package_relative_path": package_relative,
                "size_bytes": int(source.stat().st_size),
                "sha256": digest,
            }
            file_rows.append(row)
            dictionary_rows.extend(_table_schema(schema_path, package_relative or str(source)))

        artifact_rows = [
            {
                "artifact_id": artifact.id,
                "artifact_label": artifact.label,
                "artifact_kind": str(artifact.kind),
                "stage": stage_for_kind(str(artifact.kind)),
                "clip_id": artifact.clip_id,
                "created_at": artifact.created_at,
                "source_asset_ids": artifact.source_asset_ids,
                "parent_artifact_ids": artifact.parent_artifact_ids,
                "parameters": artifact.parameters,
                "qc": artifact.qc,
                "provenance": artifact.provenance,
                "artifact_root": artifact_root_path(artifact),
            }
            for artifact in artifacts
        ]
        manifest = {
            "schema_version": "oculostudio-results-package-1.0",
            "package_artifact_id": package_artifact_id,
            "project_id": snapshot.id,
            "project_name": snapshot.name,
            "studio_session_id": session_id,
            "studio_session_label": session.label,
            "session_code": s_code,
            "interaction_code": i_code,
            "role_code": r_code,
            "label": label,
            "copy_mode": copy_mode,
            "created_at": utc_now(),
            "selected_artifact_ids": unique_ids,
            "artifacts": artifact_rows,
            "files": file_rows,
            "time_domains": [item.model_dump(mode="json") for item in session.time_domains.values()],
            "time_maps": [item.model_dump(mode="json") for item in session.time_maps.values()],
            "clips": [item.model_dump(mode="json") for item in session.clips.values()],
            "source_assets": [session.assets[asset_id].model_dump(mode="json") for asset_id in dict.fromkeys(asset_id for artifact in artifacts for asset_id in artifact.source_asset_ids) if asset_id in session.assets],
            "stage_descriptions": _STAGE_DESCRIPTIONS,
            "warnings": warnings,
        }
        emit(progress, 0.87, "Writing package index and provenance")
        with (staging / "artifact_index.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(file_rows[0]) if file_rows else ["artifact_id"])
            writer.writeheader()
            writer.writerows(file_rows)
        with (staging / "data_dictionary.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["file", "column", "dtype", "description"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dictionary_rows)
        atomic_write_json(staging / "processing_manifest.json", manifest)
        readme_md = _readme_markdown(
            project_name=snapshot.name,
            session_label=session.label,
            session_code=s_code,
            interaction_code=i_code,
            role_code=r_code,
            label=label,
            copy_mode=copy_mode,
            artifact_rows=artifact_rows,
            warnings=warnings,
        )
        (staging / "README.md").write_text(readme_md, "utf-8")
        (staging / "README.html").write_text(_readme_html(readme_md, file_rows), "utf-8")
        (staging / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", "utf-8")
        staging.rename(package_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.RESULTS_PACKAGE,
        label=label,
        output_root=package_root,
        parent_artifact_ids=unique_ids,
        source_asset_ids=list(dict.fromkeys(asset_id for item in artifacts for asset_id in item.source_asset_ids)),
        parameters={
            "session_code": s_code,
            "interaction_code": i_code,
            "role_code": r_code,
            "copy_mode": copy_mode,
        },
        qc={
            "selected_artifact_count": len(artifacts),
            "selected_file_count": len(source_files),
            "copied_bytes": copied_bytes,
            "data_dictionary_rows": len(dictionary_rows),
            "warning_count": len(warnings),
        },
        provenance={
            "organized_export": True,
            "source_artifact_ids": unique_ids,
        },
        artifact_id=package_artifact_id,
    )
    emit(progress, 1.0, "Organized results package complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "output_dir": str(package_root),
        "readme_html": str(package_root / "README.html"),
        "manifest": str(package_root / "processing_manifest.json"),
        "artifact_index": str(package_root / "artifact_index.csv"),
        "data_dictionary": str(package_root / "data_dictionary.csv"),
    }
