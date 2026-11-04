from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from oculostudio import __version__
from oculostudio.domain.models import ArtifactRecord, JobRecord, JobStatus, Project
from oculostudio.util import utc_now

SIGNATURE_VERSION = "oculostudio-processing-signature-v1"
# Bump only when a scientific/encoding algorithm changes. v0.20 adds orchestration
# and GUI behavior, so v0.19.x results remain compatible and can be reused.
PROCESSING_ENGINE_VERSION = "oculostudio-core-0.19"

# These fields change labels, UI presentation, local executable discovery, or
# the shape of a QC preview, but do not change the immutable scientific result.
# Keeping them out of the core signature prevents users from creating the same
# artifact again merely by renaming it or changing a preview duration.
_ALWAYS_IGNORED_KEYS = {
    "label",
    "output_label",
    "profile_name",
    "display_label",
    "source_note",
    "force_recompute",
    "ffmpeg_path",
    "ffprobe_path",
    "encoder_preflight",
}

_OPERATION_IGNORED_KEYS: dict[str, set[str]] = {
    # This operation is a temporary cache preview. The service already has a
    # cache key, while the job signature also blocks accidental double-clicks.
    "native_trim_preview": {"label"},
}


def _canonical(value: Any, *, ignored: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical(item, ignored=ignored)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            if str(key) not in ignored and not str(key).startswith("_processing_")
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item, ignored=ignored) for item in value]
    if isinstance(value, set):
        rows = [_canonical(item, ignored=ignored) for item in value]
        return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        # JSON's cross-platform representation is stable enough after rounding
        # numerical UI inputs to a precision well beyond any exposed control.
        return round(value, 12)
    return value


def _collect_string_values(value: Any, out: set[str]) -> None:
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_string_values(item, out)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_string_values(item, out)


def processing_input_context(project: Project, session_id: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    """Resolve IDs and local paths into stable input fingerprints.

    Job parameters normally contain only asset/artifact IDs. Including their
    registered checksums means a refreshed or replaced input cannot silently
    reuse an artifact produced from older bytes.
    """
    if not session_id or session_id not in project.sessions:
        return {}
    session = project.sessions[session_id]
    strings: set[str] = set()
    _collect_string_values(parameters, strings)
    context: dict[str, Any] = {"assets": {}, "artifacts": {}, "clips": {}, "paths": {}}
    for value in sorted(strings):
        asset = session.assets.get(value)
        if asset is not None:
            context["assets"][value] = {
                "sha256": asset.source_sha256,
                "size_bytes": asset.size_bytes,
                "recording_id": asset.recording_id,
                "available": asset.available,
            }
            continue
        artifact = session.artifacts.get(value)
        if artifact is not None:
            context["artifacts"][value] = {
                "kind": artifact.kind,
                "files": [
                    {"role": row.role, "sha256": row.sha256, "size_bytes": row.size_bytes}
                    for row in artifact.files
                ],
                "processing_signature": artifact.provenance.get("processing_signature"),
            }
            continue
        clip = session.clips.get(value)
        if clip is not None:
            context["clips"][value] = {
                "start_global_ns": clip.start_global_ns,
                "end_global_ns": clip.end_global_ns,
            }
            continue
        # External profile/manifests are not necessarily registered as assets.
        # Stat metadata is cheap and detects replacements without hashing huge
        # files on every button click.
        try:
            path = Path(value).expanduser()
            if path.exists() and path.is_file():
                stat = path.stat()
                context["paths"][str(path.resolve())] = {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        except (OSError, ValueError):
            pass
    return {key: value for key, value in context.items() if value}


def processing_signature(
    operation: str,
    session_id: str | None,
    parameters: dict[str, Any],
    *,
    producer_version: str = PROCESSING_ENGINE_VERSION,
    input_context: dict[str, Any] | None = None,
) -> str:
    ignored = set(_ALWAYS_IGNORED_KEYS)
    ignored.update(_OPERATION_IGNORED_KEYS.get(operation, set()))
    payload = {
        "signature_version": SIGNATURE_VERSION,
        "producer_version": producer_version,
        "operation": operation,
        "session_id": session_id,
        "parameters": _canonical(parameters, ignored=ignored),
        "input_context": _canonical(input_context or {}, ignored=set()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_files_available(artifact: ArtifactRecord) -> bool:
    if not artifact.files:
        return False
    return all(Path(row.path).exists() for row in artifact.files if row.path)


def iter_result_artifact_ids(result: Any) -> Iterable[str]:
    if isinstance(result, dict):
        artifact = result.get("artifact")
        if isinstance(artifact, dict) and artifact.get("id"):
            yield str(artifact["id"])
        elif isinstance(artifact, str):
            yield artifact
        artifact_id = result.get("artifact_id")
        if artifact_id:
            yield str(artifact_id)
        artifacts = result.get("artifacts")
        if isinstance(artifacts, list):
            for row in artifacts:
                if isinstance(row, dict) and row.get("id"):
                    yield str(row["id"])
                elif isinstance(row, str):
                    yield row
        for key, value in result.items():
            if key not in {"artifact", "artifact_id", "artifacts"}:
                yield from iter_result_artifact_ids(value)
    elif isinstance(result, list):
        for row in result:
            yield from iter_result_artifact_ids(row)


def result_has_available_artifact(project: Project, session_id: str | None, result: dict[str, Any]) -> bool:
    if not session_id or session_id not in project.sessions:
        return False
    session = project.sessions[session_id]
    for artifact_id in dict.fromkeys(iter_result_artifact_ids(result)):
        artifact = session.artifacts.get(artifact_id)
        if artifact and artifact_files_available(artifact):
            return True
    return False


def find_artifact_by_signature(project: Project, session_id: str | None, signature: str) -> ArtifactRecord | None:
    if not session_id or session_id not in project.sessions:
        return None
    rows = sorted(project.sessions[session_id].artifacts.values(), key=lambda row: row.created_at, reverse=True)
    for artifact in rows:
        if artifact.provenance.get("processing_signature") == signature and artifact_files_available(artifact):
            return artifact
    return None


def find_job_by_signature(project: Project, signature: str) -> JobRecord | None:
    rows = sorted(project.jobs.values(), key=lambda row: row.created_at, reverse=True)
    for job in rows:
        candidate = job.parameters.get("_processing_signature")
        if not candidate:
            # Backward-compatible adoption of v0.19.x jobs. The processing
            # engine version intentionally remains stable for this GUI-only
            # release, so existing completed results can be reused immediately.
            candidate = processing_signature(
                job.operation,
                job.session_id,
                job.parameters,
                input_context=processing_input_context(project, job.session_id, job.parameters),
            )
        if candidate != signature:
            continue
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return job
        if job.status == JobStatus.COMPLETED and result_has_available_artifact(project, job.session_id, job.result):
            return job
    return None


def reused_job_for_artifact(
    *,
    operation: str,
    label: str,
    session_id: str | None,
    parameters: dict[str, Any],
    signature: str,
    artifact: ArtifactRecord,
) -> JobRecord:
    now = utc_now()
    return JobRecord(
        operation=operation,
        label=f"Reuse existing · {label}",
        session_id=session_id,
        status=JobStatus.COMPLETED,
        progress=1.0,
        stage="Reused existing result",
        started_at=now,
        completed_at=now,
        parameters={
            **parameters,
            "_processing_signature": signature,
            "_signature_version": SIGNATURE_VERSION,
            "_reused_existing_artifact": artifact.id,
        },
        result={"artifact": artifact.model_dump(mode="json"), "reused": True},
        log=["Identical immutable result already existed; no processing was started."],
    )
