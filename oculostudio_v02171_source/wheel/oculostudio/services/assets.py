from __future__ import annotations

from pathlib import Path
from typing import Any

from oculostudio.adapters.probe import probe_asset
from oculostudio.domain.models import AssetKind, AssetRecord, TimeDomain, TimeMap, TimeMapQuality
from oculostudio.errors import ValidationError
from oculostudio.services.asset_identity import remove_asset_and_refresh_participants, set_initial_asset_identity
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import new_id


def add_asset(
    store: ProjectStore,
    *,
    session_id: str,
    path: str | Path,
    label: str | None = None,
    participant_id: str | None = None,
    checksum: bool = True,
    source_sha256_override: str | None = None,
    before_register=None,
    metadata_extra: dict | None = None,
) -> AssetRecord:
    probe = probe_asset(path, checksum=checksum)
    asset = AssetRecord(
        label=label or probe["label"],
        kind=probe["kind"],
        source_path=probe["source_path"],
        source_sha256=source_sha256_override or probe.get("sha256"),
        size_bytes=probe.get("size_bytes"),
        recording_id=probe.get("recording_id"),
        participant_id=participant_id,
        probe_score=probe.get("score", 0.0),
        metadata={**probe.get("metadata", {}), **(metadata_extra or {})},
        warnings=probe.get("warnings", []),
        available=True,
    )

    def mutate(project):
        if before_register:
            before_register()
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown session: {session_id}")
        session = project.sessions[session_id]
        created_domain_ids: list[str] = []
        for spec in probe.get("time_domains", []):
            role = spec.get("role")
            reusable: TimeDomain | None = None
            if role == "pupil_clock" and asset.recording_id:
                for domain in session.time_domains.values():
                    if domain.metadata.get("role") == role and domain.metadata.get("recording_id") == asset.recording_id:
                        reusable = domain
                        break
            if reusable:
                domain = reusable
            else:
                domain = TimeDomain(
                    id=new_id("time"),
                    label=spec.get("label") or f"{asset.label} time",
                    clock_type=spec.get("clock_type", "relative"),
                    asset_id=asset.id,
                    metadata={
                        "role": role,
                        "recording_id": asset.recording_id,
                    },
                )
                session.time_domains[domain.id] = domain
            created_domain_ids.append(domain.id)
        asset.time_domain_ids = created_domain_ids
        if created_domain_ids:
            asset.primary_time_domain_id = created_domain_ids[0]
        session.assets[asset.id] = asset
        set_initial_asset_identity(session, asset, participant_id, source="import_user")

        # The first temporal asset establishes a provisional zero point. It is
        # deliberately marked provisional because a later clap/drift fit should
        # replace it.
        non_global_maps = [m for m in session.time_maps.values() if m.target_domain_id == session.global_time_domain_id]
        is_calibration_source = str((asset.metadata or {}).get("asset_scope") or "interaction") == "calibration"
        if asset.primary_time_domain_id and not non_global_maps and not is_calibration_source:
            mapping = TimeMap(
                source_domain_id=asset.primary_time_domain_id,
                target_domain_id=session.global_time_domain_id,
                scale=1.0,
                offset_ns=0.0,
                method="provisional_identity",
                quality=TimeMapQuality(warning="Provisional mapping; add synchronization anchors before production export."),
            )
            session.time_maps[mapping.id] = mapping

    store.update(mutate)
    return asset


def remove_asset(store: ProjectStore, *, session_id: str, asset_id: str) -> None:
    def mutate(project):
        session = project.sessions[session_id]
        if asset_id not in session.assets:
            raise ValidationError(f"Unknown asset: {asset_id}")
        asset = remove_asset_and_refresh_participants(session, asset_id)
        for domain_id in asset.time_domain_ids:
            if any(domain_id in other.time_domain_ids for other in session.assets.values()):
                continue
            session.time_domains.pop(domain_id, None)
            for mapping_id, mapping in list(session.time_maps.items()):
                if domain_id in {mapping.source_domain_id, mapping.target_domain_id}:
                    del session.time_maps[mapping_id]

    store.update(mutate)


def refresh_asset(store: ProjectStore, *, session_id: str, asset_id: str, checksum: bool = False) -> AssetRecord:
    session = store.project.sessions[session_id]
    old = session.assets[asset_id]
    probe = probe_asset(old.source_path, checksum=checksum)

    def mutate(project):
        asset = project.sessions[session_id].assets[asset_id]
        asset.available = Path(asset.source_path).exists()
        asset.kind = probe["kind"]
        asset.probe_score = probe["score"]
        preserved = {key: asset.metadata[key] for key in ("import", "identity", "identity_history", "asset_scope", "calibration_set_id", "calibration_set_record_id", "calibration_revision_id", "calibration_revision_number", "calibration_role", "hardware_hint", "calibration_registered_at", "calibration_alias_source_session_id", "calibration_alias_source_asset_id", "calibration_alias_created_at") if key in asset.metadata}
        asset.metadata = {**probe["metadata"], **preserved}
        asset.warnings = probe["warnings"]
        if checksum:
            asset.source_sha256 = probe.get("sha256")
        return asset

    return store.update(mutate)
