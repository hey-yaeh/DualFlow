from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ValidationError
from oculostudio.services.calibration_profiles import import_calibration_profile
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import safe_name, sha256_file, utc_now

_INVALID_FOLDER = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def inspect_project_location(*, parent: str, folder: str) -> dict[str, Any]:
    parent_text = str(parent or "").strip()
    folder_text = str(folder or "").strip()
    errors: list[str] = []
    warnings: list[str] = []
    if not parent_text:
        errors.append("Select or type a project parent folder")
    if not folder_text:
        errors.append("Enter a new project folder name")
    if folder_text and (_INVALID_FOLDER.search(folder_text) or folder_text in {".", ".."}):
        errors.append("Project folder contains characters that are not valid on Windows")
    if folder_text and folder_text.rstrip(". ") != folder_text:
        errors.append("Project folder cannot end with a dot or space")

    parent_path = Path(parent_text).expanduser() if parent_text else None
    if parent_path is not None:
        try:
            parent_path = parent_path.resolve()
        except Exception:
            pass
        if not parent_path.exists():
            errors.append(f"Parent folder does not exist: {parent_path}")
        elif not parent_path.is_dir():
            errors.append(f"Parent path is not a folder: {parent_path}")
        elif not os.access(parent_path, os.W_OK):
            errors.append(f"Parent folder is not writable: {parent_path}")

    target = (parent_path / folder_text) if parent_path is not None and folder_text else None
    target_exists = bool(target and target.exists())
    target_empty = True
    if target_exists and target is not None:
        if not target.is_dir():
            errors.append(f"Target exists and is not a folder: {target}")
            target_empty = False
        else:
            ignored = {"Thumbs.db", ".DS_Store", "desktop.ini"}
            items = [item.name for item in target.iterdir() if item.name not in ignored]
            target_empty = not items
            if not target_empty:
                errors.append("Target project folder already contains files")
            elif (target / "project.oculostudio.json").exists():
                errors.append("An OculoStudio project already exists in the target folder")
            else:
                warnings.append("The target folder already exists but is empty; it can be used")

    return {
        "valid": not errors,
        "parent": str(parent_path) if parent_path is not None else "",
        "folder": folder_text,
        "target": str(target) if target is not None else "",
        "target_exists": target_exists,
        "target_empty": target_empty,
        "errors": errors,
        "warnings": warnings,
    }


def _profile_file(artifact) -> Path:
    candidates = [Path(row.path) for row in artifact.files or [] if row.path and Path(row.path).exists()]
    for suffix in (".g3dcal", ".oculocal", ".json"):
        rows = [path for path in candidates if path.suffix.lower() == suffix and "summary" not in path.name.lower()]
        if rows:
            rows.sort(key=lambda path: (len(path.parts), str(path).lower()))
            return rows[0]
    raise ValidationError(f"Calibration profile has no readable profile file: {artifact.label}")


def project_calibration_library(store: ProjectStore) -> list[dict[str, Any]]:
    project = store.snapshot()
    rows: list[dict[str, Any]] = []
    for session_id, session in project.sessions.items():
        for artifact in session.artifacts.values():
            if str(artifact.kind) != ArtifactKind.CALIBRATION_PROFILE.value:
                continue
            try:
                profile_path = _profile_file(artifact)
                source_sha = sha256_file(profile_path)
            except Exception:
                profile_path = None
                source_sha = None
            rows.append({
                "session_id": session_id,
                "session_label": session.label,
                "artifact_id": artifact.id,
                "label": artifact.label,
                "hardware_id": artifact.parameters.get("hardware_display_name")
                    or artifact.qc.get("hardware_display_name")
                    or artifact.parameters.get("hardware_id")
                    or artifact.qc.get("hardware_id"),
                "mount_version": artifact.parameters.get("mount_version")
                    or artifact.qc.get("mount_version"),
                "valid_sessions": artifact.parameters.get("valid_sessions")
                    or artifact.qc.get("valid_sessions") or [],
                "profile_path": str(profile_path) if profile_path else "",
                "profile_sha256": source_sha,
                "created_at": artifact.created_at,
            })
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows


def apply_shared_calibration_profiles(
    store: ProjectStore,
    *,
    source_session_id: str,
    profile_artifact_ids: Iterable[str],
    target_session_ids: Iterable[str],
) -> dict[str, Any]:
    project = store.snapshot()
    if source_session_id not in project.sessions:
        raise ValidationError("Unknown source Session")
    source_session = project.sessions[source_session_id]
    target_ids = []
    for session_id in target_session_ids:
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown target Session: {session_id}")
        if session_id not in target_ids:
            target_ids.append(session_id)
    profile_ids = list(dict.fromkeys(str(value) for value in profile_artifact_ids if str(value)))
    if not profile_ids:
        raise ValidationError("Select at least one calibration profile")
    if not target_ids:
        raise ValidationError("Select at least one target Session")

    all_session_labels = [session.label for session in project.sessions.values()]
    shared_root = store.shared_calibration_dir()
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for artifact_id in profile_ids:
        source_artifact = source_session.artifacts.get(artifact_id)
        if source_artifact is None or str(source_artifact.kind) != ArtifactKind.CALIBRATION_PROFILE.value:
            raise ValidationError(f"Select a calibration profile from {source_session.label}: {artifact_id}")
        source_path = _profile_file(source_artifact)
        source_sha = sha256_file(source_path)
        hardware = str(
            source_artifact.parameters.get("hardware_display_name")
            or source_artifact.qc.get("hardware_display_name")
            or source_artifact.parameters.get("hardware_id")
            or source_artifact.qc.get("hardware_id")
            or "hardware"
        )
        shared_name = f"{safe_name(hardware)}__{safe_name(source_artifact.label)}__{source_sha[:10]}{source_path.suffix.lower()}"
        shared_path = shared_root / shared_name
        if not shared_path.exists() or sha256_file(shared_path) != source_sha:
            shutil.copy2(source_path, shared_path)

        for target_session_id in target_ids:
            target_session = project.sessions[target_session_id]
            if target_session_id == source_session_id:
                skipped.append({"session_id": target_session_id, "profile_artifact_id": artifact_id, "reason": "source_session"})
                continue
            duplicate = None
            for candidate in target_session.artifacts.values():
                if str(candidate.kind) != ArtifactKind.CALIBRATION_PROFILE.value:
                    continue
                if candidate.provenance.get("shared_profile_source_sha256") == source_sha:
                    duplicate = candidate
                    break
            if duplicate is not None:
                skipped.append({
                    "session_id": target_session_id,
                    "profile_artifact_id": duplicate.id,
                    "reason": "already_applied",
                })
                continue

            result = import_calibration_profile(
                store,
                session_id=target_session_id,
                profile_path=str(shared_path),
                label=source_artifact.label,
                hardware_id=source_artifact.parameters.get("hardware_id") or source_artifact.qc.get("hardware_id"),
                mount_version=source_artifact.parameters.get("mount_version") or source_artifact.qc.get("mount_version"),
                valid_sessions=all_session_labels,
                source_note=f"Shared project profile from {source_session.label} / {source_artifact.id}",
            )
            created_id = result["artifact"]["id"]

            def annotate(project_model):
                artifact = project_model.sessions[target_session_id].artifacts[created_id]
                artifact.parent_artifact_ids = list(dict.fromkeys([*artifact.parent_artifact_ids, source_artifact.id]))
                artifact.parameters.update({
                    "calibration_source": "shared_project_profile",
                    "shared_profile_source_session_id": source_session_id,
                    "shared_profile_source_artifact_id": source_artifact.id,
                    "valid_sessions": all_session_labels,
                })
                artifact.qc.update({
                    "calibration_source": "shared_project_profile",
                    "shared_profile": True,
                    "valid_sessions": all_session_labels,
                })
                artifact.provenance.update({
                    "shared_profile_source_session_id": source_session_id,
                    "shared_profile_source_artifact_id": source_artifact.id,
                    "shared_profile_source_sha256": source_sha,
                    "shared_profile_library_path": str(shared_path),
                    "shared_profile_applied_at": utc_now(),
                })
                shared = dict((project_model.settings or {}).get("shared_calibration") or {})
                profiles = list(shared.get("profiles") or [])
                entry = {
                    "source_session_id": source_session_id,
                    "source_artifact_id": source_artifact.id,
                    "profile_sha256": source_sha,
                    "library_path": str(shared_path),
                }
                if entry not in profiles:
                    profiles.append(entry)
                shared.update({"enabled": True, "profiles": profiles, "session_labels": all_session_labels})
                project_model.settings["shared_calibration"] = shared

            store.update(annotate)
            applied.append({
                "session_id": target_session_id,
                "session_label": target_session.label,
                "profile_artifact_id": created_id,
                "source_artifact_id": source_artifact.id,
                "profile_sha256": source_sha,
            })
            project = store.snapshot()

    return {
        "source_session_id": source_session_id,
        "target_session_ids": target_ids,
        "applied": applied,
        "skipped": skipped,
        "shared_calibration_folder": str(shared_root),
    }
