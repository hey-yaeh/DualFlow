from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from oculostudio.domain.models import (
    ArtifactKind,
    CalibrationAssetReference,
    CalibrationProfileReference,
    CalibrationSet,
    CalibrationSetRevision,
)
from oculostudio.errors import ValidationError
from oculostudio.services.assets import add_asset
from oculostudio.services.calibration_profiles import import_calibration_profile
from oculostudio.services.calibration_registry import normalize_calibration_set_id
from oculostudio.services.hardware_registration import canonical_hardware_id, hardware_display_name
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import safe_name, sha256_file, utc_now


def _session_code(label: str) -> str:
    import re

    match = re.search(r"(?<![A-Za-z0-9])(S\d{1,6})(?![A-Za-z0-9])", str(label or ""), re.I)
    return match.group(1).upper() if match else str(label or "").strip()


def _set_by_id_or_code(project, value: str) -> CalibrationSet:
    token = str(value or "").strip()
    if token in project.calibration_sets:
        return project.calibration_sets[token]
    canonical = normalize_calibration_set_id(token)
    for row in project.calibration_sets.values():
        if row.code.casefold() == canonical.casefold():
            return row
    raise ValidationError(f"Unknown Calibration Set: {value}")


def _revision(row: CalibrationSet, revision_id: str | None = None) -> CalibrationSetRevision:
    rid = revision_id or row.active_revision_id
    if not rid or rid not in row.revisions:
        raise ValidationError(f"Calibration Set {row.code} has no active revision")
    return row.revisions[rid]


def _profile_file(artifact) -> Path:
    candidates = [Path(item.path) for item in artifact.files or [] if item.path and Path(item.path).exists()]
    for suffix in (".g3dcal", ".oculocal", ".json"):
        rows = [path for path in candidates if path.suffix.lower() == suffix and "summary" not in path.name.lower()]
        if rows:
            rows.sort(key=lambda path: (len(path.parts), str(path).lower()))
            return rows[0]
    raise ValidationError(f"Calibration profile has no readable profile file: {artifact.label}")


def _revision_hardware_profiles(store: ProjectStore, revision: CalibrationSetRevision | None) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"ETM_A": [], "ETM_B": []}
    if revision is None:
        return result
    project = store.snapshot()
    for ref in revision.profiles:
        session = project.sessions.get(ref.source_session_id)
        artifact = session.artifacts.get(ref.source_artifact_id) if session else None
        value = ref.hardware_id
        if artifact is not None:
            value = (
                artifact.parameters.get("hardware_id")
                or artifact.qc.get("hardware_id")
                or artifact.parameters.get("hardware_display_name")
                or artifact.qc.get("hardware_display_name")
                or value
            )
        hardware = canonical_hardware_id(value)
        if hardware not in result:
            continue
        result[hardware].append({
            "source_session_id": ref.source_session_id,
            "artifact_id": ref.source_artifact_id,
            "label": ref.label or (artifact.label if artifact else ref.source_artifact_id),
            "hardware_id": hardware,
            "hardware_display_name": hardware_display_name(hardware),
            "mount_version": ref.mount_version,
            "available": bool(artifact and any(Path(item.path).exists() for item in artifact.files)),
            "created_at": artifact.created_at if artifact else ref.added_at,
        })
    for rows in result.values():
        rows.sort(key=lambda item: str(item.get("created_at") or ""))
    return result


def calibration_set_summary(store: ProjectStore, row: CalibrationSet) -> dict[str, Any]:
    project = store.snapshot()
    revision = row.revisions.get(row.active_revision_id or "")
    session_rows = []
    for session_id in row.applicable_session_ids:
        session = project.sessions.get(session_id)
        if session is None:
            continue
        binding = dict((session.metadata or {}).get("calibration_binding") or {})
        session_rows.append({
            "session_id": session_id,
            "session_label": session.label,
            "session_code": _session_code(session.label),
            "bound": binding.get("calibration_set_id") == row.id,
            "bound_revision_id": binding.get("revision_id"),
            "override": bool(binding.get("override")),
        })
    hardware_profiles = _revision_hardware_profiles(store, revision)
    hardware_ready = {key: bool(values) for key, values in hardware_profiles.items()}
    board_geometry_by_revision = dict((row.metadata or {}).get("board_geometry_by_revision") or {})
    board_geometry = dict(board_geometry_by_revision.get(revision.id) or {}) if revision else {}
    return {
        "id": row.id,
        "code": row.code,
        "label": row.label,
        "purpose": row.purpose,
        "owner_session_id": row.owner_session_id,
        "active_revision_id": row.active_revision_id,
        "active_revision_number": revision.revision_number if revision else None,
        "active_revision_label": revision.label if revision else None,
        "status": revision.status if revision else "draft",
        "source_count": len(revision.source_assets) if revision else 0,
        "profile_count": len(revision.profiles) if revision else 0,
        "hardware_profiles": hardware_profiles,
        "hardware_ready": hardware_ready,
        "physical_profiles_ready": bool(hardware_ready.get("ETM_A") and hardware_ready.get("ETM_B")),
        "missing_hardware_profiles": [hardware_display_name(key) for key, ready in hardware_ready.items() if not ready],
        "board_geometry": board_geometry,
        "board_geometry_ready": str(board_geometry.get("status") or "").casefold() == "ready",
        "applicable_session_ids": list(row.applicable_session_ids),
        "sessions": session_rows,
        "revision_count": len(row.revisions),
        "revisions": [
            {
                "id": rev.id,
                "revision_number": rev.revision_number,
                "label": rev.label,
                "status": rev.status,
                "source_count": len(rev.source_assets),
                "profile_count": len(rev.profiles),
                "created_at": rev.created_at,
                "active": rev.id == row.active_revision_id,
            }
            for rev in sorted(row.revisions.values(), key=lambda item: item.revision_number)
        ],
        "folder": str(store.calibration_set_dir(row.id)),
        "created_at": row.created_at,
        "modified_at": row.modified_at,
        "metadata": dict(row.metadata or {}),
    }


def list_calibration_sets(store: ProjectStore) -> list[dict[str, Any]]:
    rows = [calibration_set_summary(store, row) for row in store.snapshot().calibration_sets.values()]
    return sorted(rows, key=lambda item: (item["code"].casefold(), item["created_at"]))


def create_calibration_set(
    store: ProjectStore,
    *,
    code: str,
    label: str | None = None,
    applicable_session_ids: Iterable[str] = (),
    purpose: str = "combined",
    owner_session_id: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    canonical = normalize_calibration_set_id(code)
    project = store.snapshot()
    for existing in project.calibration_sets.values():
        if existing.code.casefold() == canonical.casefold():
            raise ValidationError(f"Calibration Set already exists: {canonical}")
    session_ids = list(dict.fromkeys(str(value) for value in applicable_session_ids if str(value)))
    for session_id in session_ids:
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown Session in Calibration Set: {session_id}")
    if owner_session_id and owner_session_id not in project.sessions:
        raise ValidationError("Unknown Calibration Set owner Session")
    owner = owner_session_id or (session_ids[0] if session_ids else project.selected_session_id)
    revision = CalibrationSetRevision(revision_number=1, label="Revision 1", notes=notes)
    row = CalibrationSet(
        code=canonical,
        label=(label or canonical).strip() or canonical,
        purpose=purpose,
        applicable_session_ids=session_ids,
        owner_session_id=owner,
        active_revision_id=revision.id,
        revisions={revision.id: revision},
        metadata={"folder_layout": "project_calibration_set_v1"},
    )

    def mutate(model):
        model.calibration_sets[row.id] = row
        for session_id in session_ids:
            session = model.sessions[session_id]
            session.metadata["calibration_binding"] = {
                "calibration_set_id": row.id,
                "calibration_set_code": row.code,
                "revision_id": revision.id,
                "inherited": True,
                "override": False,
                "assigned_at": utc_now(),
            }

    store.update(mutate)
    store.ensure_calibration_set_directories(row.id)
    return calibration_set_summary(store, store.snapshot().calibration_sets[row.id])


def ensure_calibration_set(
    store: ProjectStore,
    *,
    code: str,
    owner_session_id: str,
    applicable_session_ids: Iterable[str] | None = None,
) -> CalibrationSet:
    project = store.snapshot()
    canonical = normalize_calibration_set_id(code)
    for row in project.calibration_sets.values():
        if row.code.casefold() == canonical.casefold():
            return row
    summary = create_calibration_set(
        store,
        code=canonical,
        label=canonical,
        applicable_session_ids=applicable_session_ids or [owner_session_id],
        owner_session_id=owner_session_id,
    )
    return store.snapshot().calibration_sets[summary["id"]]


def set_calibration_sessions(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    session_ids: Iterable[str],
    revision_id: str | None = None,
) -> dict[str, Any]:
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    revision = _revision(row, revision_id)
    wanted = list(dict.fromkeys(str(value) for value in session_ids if str(value)))
    for session_id in wanted:
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown Session: {session_id}")

    def mutate(model):
        target = model.calibration_sets[row.id]
        old = set(target.applicable_session_ids)
        target.applicable_session_ids = wanted
        target.active_revision_id = revision.id
        target.modified_at = utc_now()
        for session_id, session in model.sessions.items():
            binding = dict((session.metadata or {}).get("calibration_binding") or {})
            if session_id in wanted:
                session.metadata["calibration_binding"] = {
                    "calibration_set_id": target.id,
                    "calibration_set_code": target.code,
                    "revision_id": revision.id,
                    "inherited": True,
                    "override": False,
                    "assigned_at": utc_now(),
                }
            elif session_id in old and binding.get("calibration_set_id") == target.id:
                session.metadata.pop("calibration_binding", None)

    store.update(mutate)
    materialized = materialize_calibration_set(store, calibration_set_id=row.id, session_ids=wanted)
    return {"calibration_set": calibration_set_summary(store, store.snapshot().calibration_sets[row.id]), "materialized": materialized}


def create_calibration_revision(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    label: str | None = None,
    notes: str = "",
    inherit_active: bool = True,
    activate: bool = True,
) -> dict[str, Any]:
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    current = row.revisions.get(row.active_revision_id or "")
    number = max([rev.revision_number for rev in row.revisions.values()] or [0]) + 1
    revision = CalibrationSetRevision(
        revision_number=number,
        label=(label or f"Revision {number}").strip(),
        notes=notes,
        source_assets=list(current.source_assets) if (inherit_active and current) else [],
        profiles=list(current.profiles) if (inherit_active and current) else [],
        status="draft",
    )

    def mutate(model):
        target = model.calibration_sets[row.id]
        if activate and target.active_revision_id in target.revisions:
            previous = target.revisions[target.active_revision_id]
            if previous.status == "ready":
                previous.status = "superseded"
        target.revisions[revision.id] = revision
        if activate:
            target.active_revision_id = revision.id
            for session_id in target.applicable_session_ids:
                if session_id in model.sessions:
                    binding = dict(model.sessions[session_id].metadata.get("calibration_binding") or {})
                    if binding.get("calibration_set_id") == target.id:
                        binding["revision_id"] = revision.id
                        binding["assigned_at"] = utc_now()
                        model.sessions[session_id].metadata["calibration_binding"] = binding
        target.modified_at = utc_now()

    store.update(mutate)
    store.ensure_calibration_set_directories(row.id)
    return calibration_set_summary(store, store.snapshot().calibration_sets[row.id])


def activate_calibration_revision(
    store: ProjectStore, *, calibration_set_id: str, revision_id: str
) -> dict[str, Any]:
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    if revision_id not in row.revisions:
        raise ValidationError("Unknown Calibration Set revision")

    def mutate(model):
        target = model.calibration_sets[row.id]
        target.active_revision_id = revision_id
        target.modified_at = utc_now()
        for session_id in target.applicable_session_ids:
            if session_id in model.sessions:
                binding = dict(model.sessions[session_id].metadata.get("calibration_binding") or {})
                if binding.get("calibration_set_id") == target.id:
                    binding["revision_id"] = revision_id
                    binding["assigned_at"] = utc_now()
                    model.sessions[session_id].metadata["calibration_binding"] = binding

    store.update(mutate)
    materialized = materialize_calibration_set(store, calibration_set_id=row.id)
    return {"calibration_set": calibration_set_summary(store, store.snapshot().calibration_sets[row.id]), "materialized": materialized}


def register_calibration_asset_refs(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    source_session_id: str,
    asset_ids: Iterable[str],
    revision_id: str | None = None,
) -> dict[str, Any]:
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    revision = _revision(row, revision_id)
    if source_session_id not in project.sessions:
        raise ValidationError("Unknown Calibration Set source Session")
    source_session = project.sessions[source_session_id]
    refs: list[CalibrationAssetReference] = []
    for asset_id in dict.fromkeys(str(value) for value in asset_ids if str(value)):
        asset = source_session.assets.get(asset_id)
        if asset is None:
            raise ValidationError(f"Unknown calibration source asset: {asset_id}")
        metadata = dict(asset.metadata or {})
        if str(metadata.get("asset_scope") or "") != "calibration":
            raise ValidationError(f"Asset is not registered as calibration data: {asset.label}")
        refs.append(CalibrationAssetReference(
            source_session_id=source_session_id,
            source_asset_id=asset.id,
            label=asset.label,
            source_path=asset.source_path,
            source_sha256=asset.source_sha256,
            kind=str(asset.kind),
            calibration_role=str(metadata.get("calibration_role") or "calibration_source"),
            participant_id=asset.participant_id,
            hardware_hint=metadata.get("hardware_hint"),
        ))

    def mutate(model):
        target = model.calibration_sets[row.id]
        rev = target.revisions[revision.id]
        existing = {(ref.source_session_id, ref.source_asset_id) for ref in rev.source_assets}
        for ref in refs:
            if (ref.source_session_id, ref.source_asset_id) not in existing:
                rev.source_assets.append(ref)
                existing.add((ref.source_session_id, ref.source_asset_id))
            asset = model.sessions[ref.source_session_id].assets[ref.source_asset_id]
            metadata = dict(asset.metadata or {})
            metadata.update({
                "asset_scope": "calibration",
                "calibration_set_id": target.code,
                "calibration_set_record_id": target.id,
                "calibration_revision_id": revision.id,
                "calibration_revision_number": revision.revision_number,
            })
            asset.metadata = metadata
        if rev.source_assets:
            rev.status = "ready" if rev.profiles else "draft"
        rev.modified_at = utc_now()
        target.owner_session_id = target.owner_session_id or source_session_id
        target.modified_at = utc_now()

    store.update(mutate)
    materialized = materialize_calibration_set(store, calibration_set_id=row.id)
    return {"calibration_set": calibration_set_summary(store, store.snapshot().calibration_sets[row.id]), "materialized": materialized}


def register_calibration_profile_ref(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    source_session_id: str,
    profile_artifact_id: str,
    revision_id: str | None = None,
) -> dict[str, Any]:
    return register_calibration_profile_refs(
        store,
        calibration_set_id=calibration_set_id,
        source_session_id=source_session_id,
        profile_artifact_ids=[profile_artifact_id],
        revision_id=revision_id,
    )


def register_calibration_profile_refs(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    source_session_id: str,
    profile_artifact_ids: Iterable[str],
    revision_id: str | None = None,
) -> dict[str, Any]:
    """Atomically attach one or more measured hardware profiles to a revision.

    Validation is completed for every artifact before the Calibration Set is
    mutated.  This prevents a paired ETM-A/ETM-B build from leaving a revision
    half-registered when the second profile is invalid.
    """
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    revision = _revision(row, revision_id)
    session = project.sessions.get(source_session_id)
    if session is None:
        raise ValidationError("Unknown Calibration Set profile Session")
    ids = list(dict.fromkeys(str(value) for value in profile_artifact_ids if str(value)))
    if not ids:
        raise ValidationError("Select at least one calibration-profile artifact")
    refs: list[CalibrationProfileReference] = []
    hardware_ids: list[str] = []
    for profile_artifact_id in ids:
        artifact = session.artifacts.get(profile_artifact_id)
        if artifact is None or artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
            raise ValidationError(f"Select a calibration-profile artifact: {profile_artifact_id}")
        path = _profile_file(artifact)
        hardware_value = (
            artifact.parameters.get("hardware_id")
            or artifact.qc.get("hardware_id")
            or artifact.parameters.get("hardware_display_name")
            or artifact.qc.get("hardware_display_name")
        )
        hardware = canonical_hardware_id(hardware_value)
        if hardware not in {"ETM_A", "ETM_B"}:
            raise ValidationError(
                f"Calibration profile {artifact.label} is not registered to physical ETM-A or ETM-B hardware"
            )
        hardware_ids.append(hardware)
        refs.append(CalibrationProfileReference(
            source_session_id=source_session_id,
            source_artifact_id=artifact.id,
            label=artifact.label,
            profile_path=str(path),
            profile_sha256=sha256_file(path),
            hardware_id=hardware,
            mount_version=artifact.parameters.get("mount_version") or artifact.qc.get("mount_version"),
        ))
    if len(ids) > 1 and len(set(hardware_ids)) != len(hardware_ids):
        labels = ", ".join(ref.label for ref in refs)
        raise ValidationError(f"The selected profile batch contains duplicate physical hardware identities: {labels}")

    def mutate(model):
        target = model.calibration_sets[row.id]
        rev = target.revisions[revision.id]
        existing = {(item.source_session_id, item.source_artifact_id) for item in rev.profiles}
        for ref in refs:
            if (ref.source_session_id, ref.source_artifact_id) not in existing:
                rev.profiles.append(ref)
                existing.add((ref.source_session_id, ref.source_artifact_id))
            artifact = model.sessions[source_session_id].artifacts[ref.source_artifact_id]
            artifact.parameters.update({
                "calibration_set_id": target.code,
                "calibration_set_record_id": target.id,
                "calibration_revision_id": revision.id,
                "calibration_revision_number": revision.revision_number,
                "hardware_id": ref.hardware_id,
                "hardware_display_name": hardware_display_name(ref.hardware_id),
            })
            artifact.qc.update({
                "calibration_set_id": target.code,
                "calibration_set_record_id": target.id,
                "calibration_revision_id": revision.id,
                "calibration_revision_number": revision.revision_number,
                "hardware_id": ref.hardware_id,
                "hardware_display_name": hardware_display_name(ref.hardware_id),
            })
            artifact.provenance.update({
                "calibration_set_registered_at": utc_now(),
                "calibration_set_profile_sha256": ref.profile_sha256,
            })
        present = set()
        for item in rev.profiles:
            hardware = canonical_hardware_id(item.hardware_id)
            if hardware in {"ETM_A", "ETM_B"}:
                present.add(hardware)
        rev.status = "ready" if {"ETM_A", "ETM_B"}.issubset(present) else "draft"
        rev.modified_at = utc_now()
        target.modified_at = utc_now()

    store.update(mutate)
    materialized = materialize_calibration_set(store, calibration_set_id=row.id)
    return {
        "calibration_set": calibration_set_summary(store, store.snapshot().calibration_sets[row.id]),
        "registered_profile_artifact_ids": ids,
        "hardware_ids": hardware_ids,
        "materialized": materialized,
    }


def _existing_alias(session, *, set_id: str, revision_id: str, source_session_id: str, source_asset_id: str):
    for asset in session.assets.values():
        metadata = dict(asset.metadata or {})
        if metadata.get("calibration_set_record_id") != set_id:
            continue
        if metadata.get("calibration_alias_source_session_id") != source_session_id:
            continue
        if metadata.get("calibration_alias_source_asset_id") != source_asset_id:
            continue
        return asset
    return None


def materialize_calibration_set(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    session_ids: Iterable[str] | None = None,
    revision_id: str | None = None,
) -> dict[str, Any]:
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    revision = _revision(row, revision_id)
    targets = list(dict.fromkeys(session_ids or row.applicable_session_ids))
    created_assets: list[dict[str, Any]] = []
    reused_assets: list[dict[str, Any]] = []
    created_profiles: list[dict[str, Any]] = []
    reused_profiles: list[dict[str, Any]] = []
    warnings: list[str] = []

    for session_id in targets:
        current = store.snapshot()
        if session_id not in current.sessions:
            warnings.append(f"Unknown target Session skipped: {session_id}")
            continue
        target_session = current.sessions[session_id]
        for ref in revision.source_assets:
            source_session = current.sessions.get(ref.source_session_id)
            source_asset = source_session.assets.get(ref.source_asset_id) if source_session else None
            if source_asset is None:
                warnings.append(f"Missing canonical calibration asset: {ref.label}")
                continue
            if session_id == ref.source_session_id:
                reused_assets.append({"session_id": session_id, "asset_id": ref.source_asset_id, "canonical": True})
                continue
            existing = _existing_alias(
                target_session,
                set_id=row.id,
                revision_id=revision.id,
                source_session_id=ref.source_session_id,
                source_asset_id=ref.source_asset_id,
            )
            if existing is not None:
                reused_assets.append({"session_id": session_id, "asset_id": existing.id, "canonical": False})
                continue
            path = Path(source_asset.source_path)
            if not path.exists():
                warnings.append(f"Calibration source is unavailable for {target_session.label}: {source_asset.label}")
                continue
            metadata = dict(source_asset.metadata or {})
            metadata.update({
                "asset_scope": "calibration",
                "calibration_set_id": row.code,
                "calibration_set_record_id": row.id,
                "calibration_revision_id": revision.id,
                "calibration_revision_number": revision.revision_number,
                "calibration_alias_source_session_id": ref.source_session_id,
                "calibration_alias_source_asset_id": ref.source_asset_id,
                "calibration_alias_created_at": utc_now(),
            })
            alias = add_asset(
                store,
                session_id=session_id,
                path=source_asset.source_path,
                label=source_asset.label,
                participant_id=source_asset.participant_id,
                checksum=False,
                source_sha256_override=source_asset.source_sha256,
                metadata_extra=metadata,
            )
            created_assets.append({"session_id": session_id, "asset_id": alias.id, "source_asset_id": ref.source_asset_id})
            target_session = store.snapshot().sessions[session_id]

        for ref in revision.profiles:
            current = store.snapshot()
            target_session = current.sessions[session_id]
            duplicate = None
            for artifact in target_session.artifacts.values():
                if str(artifact.kind) != ArtifactKind.CALIBRATION_PROFILE.value:
                    continue
                if artifact.provenance.get("calibration_set_profile_sha256") == ref.profile_sha256:
                    duplicate = artifact
                    break
                if session_id == ref.source_session_id and artifact.id == ref.source_artifact_id:
                    duplicate = artifact
                    break
            if duplicate is not None:
                reused_profiles.append({"session_id": session_id, "artifact_id": duplicate.id})
                continue
            path = Path(ref.profile_path)
            if not path.exists():
                warnings.append(f"Calibration profile unavailable for {target_session.label}: {ref.label}")
                continue
            result = import_calibration_profile(
                store,
                session_id=session_id,
                profile_path=str(path),
                label=ref.label,
                hardware_id=ref.hardware_id,
                mount_version=ref.mount_version,
                valid_sessions=[_session_code(current.sessions[sid].label) for sid in row.applicable_session_ids if sid in current.sessions],
                source_note=f"Calibration Set {row.code} / revision {revision.revision_number}",
            )
            artifact_id = result["artifact"]["id"]

            def annotate(model):
                artifact = model.sessions[session_id].artifacts[artifact_id]
                artifact.parent_artifact_ids = list(dict.fromkeys([*artifact.parent_artifact_ids, ref.source_artifact_id]))
                artifact.parameters.update({
                    "calibration_set_id": row.code,
                    "calibration_set_record_id": row.id,
                    "calibration_revision_id": revision.id,
                    "calibration_revision_number": revision.revision_number,
                })
                artifact.qc.update({"calibration_set_id": row.code, "calibration_revision_number": revision.revision_number})
                artifact.provenance.update({
                    "calibration_set_profile_sha256": ref.profile_sha256,
                    "calibration_set_source_session_id": ref.source_session_id,
                    "calibration_set_source_artifact_id": ref.source_artifact_id,
                    "calibration_set_materialized_at": utc_now(),
                })

            store.update(annotate)
            created_profiles.append({"session_id": session_id, "artifact_id": artifact_id, "source_artifact_id": ref.source_artifact_id})

    return {
        "calibration_set_id": row.id,
        "calibration_set_code": row.code,
        "revision_id": revision.id,
        "revision_number": revision.revision_number,
        "session_ids": targets,
        "created_assets": created_assets,
        "reused_assets": reused_assets,
        "created_profiles": created_profiles,
        "reused_profiles": reused_profiles,
        "warnings": warnings,
    }


def migrate_legacy_calibration_sets(store: ProjectStore) -> bool:
    project = store.snapshot()
    existing_codes = {row.code.casefold() for row in project.calibration_sets.values()}
    grouped: dict[str, list[tuple[str, str]]] = {}
    for session_id, session in project.sessions.items():
        for asset in session.assets.values():
            metadata = dict(asset.metadata or {})
            if str(metadata.get("asset_scope") or "") != "calibration":
                continue
            code = str(metadata.get("calibration_set_id") or "CALIBRATION")
            grouped.setdefault(normalize_calibration_set_id(code), []).append((session_id, asset.id))
    changed = False
    for code, refs in grouped.items():
        if code.casefold() in existing_codes:
            continue
        owner = refs[0][0]
        session_ids = list(dict.fromkeys(session_id for session_id, _ in refs))
        summary = create_calibration_set(
            store,
            code=code,
            label=code,
            applicable_session_ids=session_ids,
            owner_session_id=owner,
            notes="Migrated from v0.21.66 calibration asset metadata",
        )
        register_calibration_asset_refs(
            store,
            calibration_set_id=summary["id"],
            source_session_id=owner,
            asset_ids=[asset_id for session_id, asset_id in refs if session_id == owner],
        )
        existing_codes.add(code.casefold())
        changed = True
    return changed
