from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from oculostudio.adapters.archive import SourceView
from oculostudio.domain.models import AssetKind, AssetRecord
from oculostudio.errors import ValidationError
from oculostudio.services.asset_identity import asset_identity_view, update_asset_identity
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json, utc_now

SESSION_SHARED_SCOPE = "session_shared"
INTERACTION_SCOPE = "interaction"
CALIBRATION_SCOPE = "calibration"
SESSION_SHARED_SCHEMA = "oculostudio-session-shared-1.0"
FACEMAPPER_INDEX_SCHEMA = "oculostudio-facemapper-section-index-1.0"

_CODE_RE = {
    "session": re.compile(r"(?<![A-Za-z0-9])(S\d{1,6})(?![A-Za-z0-9])", re.I),
    "interaction": re.compile(r"(?<![A-Za-z0-9])(I\d{1,6})(?![A-Za-z0-9])", re.I),
    "role": re.compile(r"(?<![A-Za-z0-9])(R\d{1,6})(?![A-Za-z0-9])", re.I),
}
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def asset_scope(asset: AssetRecord) -> str:
    value = str((asset.metadata or {}).get("asset_scope") or INTERACTION_SCOPE).strip().lower()
    return value if value in {INTERACTION_SCOPE, SESSION_SHARED_SCOPE, CALIBRATION_SCOPE} else INTERACTION_SCOPE


def _code(text: Any, key: str) -> str:
    match = _CODE_RE[key].search(str(text or ""))
    return match.group(1).upper() if match else ""


def _asset_identity(asset: AssetRecord, session_label: str) -> dict[str, str]:
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    identity = metadata.get("identity") if isinstance(metadata.get("identity"), dict) else {}
    text = " ".join(
        str(value or "")
        for value in (
            identity.get("session_id"), identity.get("interaction_id"), identity.get("role_id"),
            metadata.get("session_id"), metadata.get("interaction_id"), metadata.get("role_id"),
            asset.label, asset.source_path,
        )
    )
    return {
        "session_id": str(identity.get("session_id") or metadata.get("session_id") or _code(text, "session") or _code(session_label, "session")).upper(),
        "interaction_id": str(identity.get("interaction_id") or metadata.get("interaction_id") or _code(text, "interaction")).upper(),
        "role_id": str(identity.get("role_id") or metadata.get("role_id") or _code(text, "role")).upper(),
        "participant_id": str(asset.participant_id or identity.get("participant_id") or metadata.get("participant_id") or "").upper(),
    }


def _normalise_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _first(row: dict[str, Any], candidates: Iterable[str]) -> str:
    normalized = {_normalise_header(key): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(_normalise_header(candidate))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _int_or_none(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _csv_rows(view: SourceView, member: str) -> list[dict[str, str]]:
    text = view.read_text(member, encoding="utf-8-sig")
    sample = text[:4096]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    return [dict(row) for row in csv.DictReader(io.StringIO(text), delimiter=delimiter)]


def _section_rows_from_sections_csv(view: SourceView, member: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for position, row in enumerate(_csv_rows(view, member)):
        recording_id = _first(row, ("recording id", "recording_id", "recording uuid", "recording_uuid", "recording"))
        section_id = _first(row, ("section id", "section_id", "section uuid", "section_uuid", "id"))
        name = _first(row, ("recording name", "recording_name", "section name", "section_name", "name"))
        start_ns = _int_or_none(_first(row, ("start timestamp ns", "start_timestamp_ns", "start time ns", "start_time_ns", "start timestamp")))
        end_ns = _int_or_none(_first(row, ("end timestamp ns", "end_timestamp_ns", "end time ns", "end_time_ns", "end timestamp")))
        if recording_id and not _UUID_RE.match(recording_id):
            # A number or display name in the broad "recording" column is not a recording UUID.
            recording_id = ""
        output.append({
            "recording_id": recording_id,
            "section_id": section_id,
            "recording_name": name,
            "start_timestamp_ns": start_ns,
            "end_timestamp_ns": end_ns,
            "sections_member": member,
            "sections_row_index": position,
            "source_root": PurePosixPath(member).parent.as_posix(),
        })
    return output


def _recording_metadata_rows(view: SourceView) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for member in view.names():
        base = PurePosixPath(member).name.lower()
        if base == "info.json":
            try:
                info = view.read_json(member)
            except Exception:
                continue
            recording_id = str(info.get("recording_id") or info.get("recording id") or info.get("id") or "").strip()
            if _UUID_RE.match(recording_id):
                output.append({
                    "recording_id": recording_id,
                    "recording_name": str(info.get("recording_name") or info.get("recording name") or info.get("name") or ""),
                    "source_root": PurePosixPath(member).parent.as_posix(),
                    "metadata_member": member,
                })
        elif base.endswith(".csv") and ("recording" in base or base in {"sections.csv", "section.csv"}):
            try:
                rows = _csv_rows(view, member)
            except Exception:
                continue
            for row in rows:
                recording_id = _first(row, ("recording id", "recording_id", "recording uuid", "recording_uuid"))
                if not _UUID_RE.match(recording_id):
                    continue
                output.append({
                    "recording_id": recording_id,
                    "recording_name": _first(row, ("recording name", "recording_name", "name")),
                    "section_id": _first(row, ("section id", "section_id", "section uuid", "section_uuid")),
                    "source_root": PurePosixPath(member).parent.as_posix(),
                    "metadata_member": member,
                })
    return output


def _timestamp_bounds(view: SourceView) -> dict[str, tuple[int | None, int | None, int]]:
    by_recording: dict[str, list[int]] = defaultdict(list)
    for member in view.names():
        base = PurePosixPath(member).name.lower()
        if not base.endswith(".csv") or not any(token in base for token in ("detection", "position", "gaze_on_face", "fixation")):
            continue
        try:
            rows = _csv_rows(view, member)
        except Exception:
            continue
        for row in rows:
            recording_id = _first(row, ("recording id", "recording_id", "recording uuid", "recording_uuid"))
            timestamp = _int_or_none(_first(row, ("timestamp ns", "timestamp [ns]", "timestamp", "world timestamp ns", "world_timestamp_ns")))
            if recording_id and timestamp is not None:
                by_recording[recording_id].append(timestamp)
    return {
        recording_id: (min(values), max(values), len(values))
        for recording_id, values in by_recording.items() if values
    }


def scan_facemapper_sections(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with SourceView(source) as view:
        names = view.names()
        sections_members = [name for name in names if PurePosixPath(name).name.lower() == "sections.csv"]
        if not sections_members:
            raise ValidationError("FaceMapper package has no sections.csv")
        rows: list[dict[str, Any]] = []
        for member in sections_members:
            rows.extend(_section_rows_from_sections_csv(view, member))
        metadata_rows = _recording_metadata_rows(view)
        bounds = _timestamp_bounds(view)

        # Join incomplete section rows to recording metadata by source root and section ID.
        for row in rows:
            candidates = [meta for meta in metadata_rows if meta.get("source_root") == row.get("source_root")]
            if row.get("section_id"):
                exact = [meta for meta in candidates if meta.get("section_id") == row.get("section_id")]
                if exact:
                    candidates = exact
            if len(candidates) == 1:
                row["recording_id"] = row.get("recording_id") or candidates[0].get("recording_id") or ""
                row["recording_name"] = row.get("recording_name") or candidates[0].get("recording_name") or ""
                row["metadata_member"] = candidates[0].get("metadata_member")

        # Some exports contain one sections.csv per recording and no UUID column.
        # In that case the unique info/recording metadata row under the same root is authoritative.
        for meta in metadata_rows:
            if any(row.get("recording_id") == meta.get("recording_id") for row in rows):
                continue
            same_root = [row for row in rows if row.get("source_root") == meta.get("source_root") and not row.get("recording_id")]
            if len(same_root) == 1:
                same_root[0]["recording_id"] = meta.get("recording_id") or ""
                same_root[0]["recording_name"] = same_root[0].get("recording_name") or meta.get("recording_name") or ""
                same_root[0]["section_id"] = same_root[0].get("section_id") or meta.get("section_id") or ""
                same_root[0]["metadata_member"] = meta.get("metadata_member")

        unique: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            rid = str(row.get("recording_id") or "").strip()
            if rid in bounds:
                lower, upper, count = bounds[rid]
                row["start_timestamp_ns"] = row.get("start_timestamp_ns") if row.get("start_timestamp_ns") is not None else lower
                row["end_timestamp_ns"] = row.get("end_timestamp_ns") if row.get("end_timestamp_ns") is not None else upper
                row["timestamp_evidence_rows"] = count
            key = (rid, str(row.get("section_id") or ""), str(row.get("source_root") or ""))
            unique[key] = row
        rows = list(unique.values())
        if not rows:
            raise ValidationError("FaceMapper package contains no readable section rows")
        return {
            "schema_version": FACEMAPPER_INDEX_SCHEMA,
            "source_path": str(source),
            "sections_members": sections_members,
            "section_count": len(rows),
            "sections": rows,
            "scanned_at": utc_now(),
        }


def _active_session_shared_assets(store: ProjectStore, session_id: str, kind: str | None = None, role_id: str | None = None) -> list[AssetRecord]:
    session = store.snapshot().sessions[session_id]
    output = []
    for asset in session.assets.values():
        metadata = asset.metadata or {}
        if asset_scope(asset) != SESSION_SHARED_SCOPE:
            continue
        if kind and str(metadata.get("session_shared_kind") or asset.kind) != kind:
            continue
        if role_id and str(metadata.get("role_id") or "").upper() != role_id.upper():
            continue
        if metadata.get("session_shared_active", True) is False:
            continue
        output.append(asset)
    return sorted(output, key=lambda row: (int((row.metadata or {}).get("session_shared_revision_number") or 0), row.created_at))


def _write_index(store: ProjectStore, session_id: str) -> Path:
    rows = []
    for asset in _active_session_shared_assets(store, session_id, kind="face_mapper"):
        metadata = asset.metadata or {}
        for section in metadata.get("facemapper_sections") or []:
            rows.append({
                **section,
                "source_asset_id": asset.id,
                "source_asset_label": asset.label,
                "source_sha256": asset.source_sha256,
                "role_id": str(metadata.get("role_id") or "").upper(),
                "revision_number": int(metadata.get("session_shared_revision_number") or 1),
                "active": bool(metadata.get("session_shared_active", True)),
            })
    target = store.session_shared_dir(session_id, "indexes") / "facemapper_sections.json"
    atomic_write_json(target, {
        "schema_version": FACEMAPPER_INDEX_SCHEMA,
        "session_id": session_id,
        "generated_at": utc_now(),
        "sections": rows,
    })
    return target


def register_session_shared_facemapper(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    role_id: str,
    conflict_policy: str = "add_revision",
) -> dict[str, Any]:
    role_id = str(role_id or "").strip().upper()
    if not _CODE_RE["role"].fullmatch(role_id):
        raise ValidationError("Session-shared FaceMapper requires a role such as R01 or R02")
    snapshot = store.snapshot()
    if session_id not in snapshot.sessions or asset_id not in snapshot.sessions[session_id].assets:
        raise ValidationError("Unknown Session or FaceMapper asset")
    asset = snapshot.sessions[session_id].assets[asset_id]
    if str(asset.kind) != AssetKind.FACE_MAPPER.value:
        raise ValidationError("Only a FaceMapper package can be registered as Session Shared FaceMapper data")
    scan = scan_facemapper_sections(asset.source_path)

    current = _active_session_shared_assets(store, session_id, kind="face_mapper", role_id=role_id)
    duplicate = next((row for row in current if row.source_sha256 and row.source_sha256 == asset.source_sha256), None)
    if duplicate and duplicate.id != asset_id:
        return {
            "asset": duplicate.model_dump(mode="json"),
            "reused": True,
            "reason": "identical_sha256",
            "section_count": len((duplicate.metadata or {}).get("facemapper_sections") or []),
        }
    if current and conflict_policy == "keep_existing" and all(row.id != asset_id for row in current):
        chosen = current[-1]
        return {"asset": chosen.model_dump(mode="json"), "reused": True, "reason": "kept_existing"}
    revision = max([int((row.metadata or {}).get("session_shared_revision_number") or 0) for row in current] + [0]) + 1

    def mutate(project):
        session = project.sessions[session_id]
        target = session.assets[asset_id]
        if conflict_policy in {"add_revision", "replace_active"}:
            for other in session.assets.values():
                metadata = other.metadata or {}
                if other.id != target.id and str(metadata.get("asset_scope") or INTERACTION_SCOPE) == SESSION_SHARED_SCOPE \
                        and str(metadata.get("session_shared_kind") or "") == "face_mapper" \
                        and str(metadata.get("role_id") or "").upper() == role_id:
                    metadata["session_shared_active"] = False
                    metadata["superseded_by_asset_id"] = target.id
                    other.metadata = metadata
        metadata = dict(target.metadata or {})
        metadata.update({
            "asset_scope": SESSION_SHARED_SCOPE,
            "session_shared_schema": SESSION_SHARED_SCHEMA,
            "session_shared_kind": "face_mapper",
            "session_id": session_id,
            "role_id": role_id,
            "interaction_id": None,
            "session_shared_active": True,
            "session_shared_revision_number": revision,
            "session_shared_registered_at": utc_now(),
            "facemapper_section_count": scan["section_count"],
            "facemapper_sections": scan["sections"],
        })
        target.metadata = metadata
        return target.model_dump(mode="json")

    updated = store.update(mutate)
    index_path = _write_index(store, session_id)
    return {
        "asset": updated,
        "reused": False,
        "section_count": scan["section_count"],
        "index_path": str(index_path),
        "revision_number": revision,
    }


def promote_facemapper_to_session_shared(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    role_id: str | None = None,
    conflict_policy: str = "add_revision",
) -> dict[str, Any]:
    snapshot = store.snapshot()
    asset = snapshot.sessions[session_id].assets.get(asset_id)
    if asset is None:
        raise ValidationError("Unknown FaceMapper asset")
    identity = _asset_identity(asset, snapshot.sessions[session_id].label)
    resolved_role = str(role_id or identity.get("role_id") or "").upper()
    return register_session_shared_facemapper(
        store,
        session_id=session_id,
        asset_id=asset_id,
        role_id=resolved_role,
        conflict_policy=conflict_policy,
    )


def _interaction_recording_ids(session, interaction_id: str, role_id: str) -> list[str]:
    values: set[str] = set()
    for asset in session.assets.values():
        if asset_scope(asset) != INTERACTION_SCOPE:
            continue
        identity = _asset_identity(asset, session.label)
        if identity.get("interaction_id") != interaction_id or identity.get("role_id") != role_id:
            continue
        if str(asset.kind) not in {AssetKind.PUPIL_NATIVE.value, AssetKind.PUPIL_CLOUD.value}:
            continue
        if asset.recording_id:
            values.add(str(asset.recording_id))
    return sorted(values)


def _planned_interactions(session) -> list[str]:
    values = {str(row).upper() for row in (session.metadata.get("planned_interactions") or []) if _CODE_RE["interaction"].fullmatch(str(row).upper())}
    for asset in session.assets.values():
        identity = _asset_identity(asset, session.label)
        if identity.get("interaction_id"):
            values.add(identity["interaction_id"])
    return sorted(values, key=lambda value: (int(value[1:]) if value[1:].isdigit() else 999999, value))


def resolve_session_facemapper(
    store: ProjectStore,
    *,
    session_id: str,
    interaction_id: str,
    role_id: str,
    strict: bool = True,
) -> dict[str, Any] | None:
    snapshot = store.snapshot()
    session = snapshot.sessions[session_id]
    interaction_id = interaction_id.upper()
    role_id = role_id.upper()
    recording_ids = _interaction_recording_ids(session, interaction_id, role_id)
    if len(recording_ids) != 1:
        if strict:
            reason = "missing" if not recording_ids else "ambiguous"
            raise ValidationError(f"{interaction_id}/{role_id} has {reason} Pupil recording identity; cannot resolve Session FaceMapper")
        return None
    recording_id = recording_ids[0]
    matches = []
    for asset in _active_session_shared_assets(store, session_id, kind="face_mapper", role_id=role_id):
        for section in (asset.metadata or {}).get("facemapper_sections") or []:
            if str(section.get("recording_id") or "") == recording_id:
                matches.append((asset, section))
    if len(matches) != 1:
        if strict:
            state = "missing" if not matches else "ambiguous"
            raise ValidationError(f"Session FaceMapper section is {state} for {interaction_id}/{role_id} recording {recording_id}")
        return None
    asset, section = matches[0]
    return {
        "asset_id": asset.id,
        "asset_label": asset.label,
        "source_path": asset.source_path,
        "role_id": role_id,
        "interaction_id": interaction_id,
        "recording_id": recording_id,
        "section": section,
        "match_status": "ready",
    }


def build_session_shared_workspace(store: ProjectStore, session_id: str) -> dict[str, Any]:
    snapshot = store.snapshot()
    if session_id not in snapshot.sessions:
        raise ValidationError("Unknown Session")
    session = snapshot.sessions[session_id]
    interactions = _planned_interactions(session)
    roles = ["R01", "R02"]
    coverage = []
    matched_keys: set[tuple[str, str]] = set()
    for interaction_id in interactions:
        row = {"interaction_id": interaction_id, "roles": {}}
        for role_id in roles:
            recording_ids = _interaction_recording_ids(session, interaction_id, role_id)
            if not recording_ids:
                cell = {"status": "waiting", "message": "No NRD/TSD recording ID yet", "recording_ids": []}
            elif len(recording_ids) > 1:
                cell = {"status": "ambiguous", "message": "Multiple Pupil recording IDs", "recording_ids": recording_ids}
            else:
                rid = recording_ids[0]
                matches = []
                for asset in _active_session_shared_assets(store, session_id, kind="face_mapper", role_id=role_id):
                    for section in (asset.metadata or {}).get("facemapper_sections") or []:
                        if str(section.get("recording_id") or "") == rid:
                            matches.append((asset, section))
                if len(matches) == 1:
                    asset, section = matches[0]
                    matched_keys.add((asset.id, str(section.get("section_id") or section.get("recording_id") or "")))
                    cell = {
                        "status": "ready",
                        "message": "Exact recording-ID match",
                        "recording_ids": [rid],
                        "asset_id": asset.id,
                        "asset_label": asset.label,
                        "section_id": section.get("section_id"),
                    }
                elif len(matches) > 1:
                    cell = {"status": "ambiguous", "message": "Multiple FaceMapper sections match this recording ID", "recording_ids": [rid], "match_count": len(matches)}
                else:
                    cell = {"status": "missing", "message": "No FaceMapper section for this recording ID", "recording_ids": [rid]}
            row["roles"][role_id] = cell
        coverage.append(row)

    sources = []
    all_sections = []
    for role_id in roles:
        for asset in _active_session_shared_assets(store, session_id, kind="face_mapper", role_id=role_id):
            metadata = asset.metadata or {}
            sections = metadata.get("facemapper_sections") or []
            sources.append({
                "asset_id": asset.id,
                "label": asset.label,
                "source_path": asset.source_path,
                "sha256": asset.source_sha256,
                "role_id": role_id,
                "revision_number": int(metadata.get("session_shared_revision_number") or 1),
                "section_count": len(sections),
                "available": asset.available,
            })
            for section in sections:
                key = (asset.id, str(section.get("section_id") or section.get("recording_id") or ""))
                all_sections.append({
                    **section,
                    "asset_id": asset.id,
                    "role_id": role_id,
                    "matched": key in matched_keys,
                })
    unassigned = [row for row in all_sections if not row["matched"]]
    counts = defaultdict(int)
    for row in coverage:
        for cell in row["roles"].values():
            counts[cell["status"]] += 1
    index_path = store.session_shared_dir(session_id, "indexes") / "facemapper_sections.json"
    return {
        "schema_version": SESSION_SHARED_SCHEMA,
        "session_id": session_id,
        "session_code": _code(session.label, "session") or session.label,
        "paths": {
            "root": str(store.session_shared_dir(session_id)),
            "face_mapper": str(store.session_shared_dir(session_id, "face_mapper")),
            "index": str(index_path),
        },
        "sources": sources,
        "coverage": coverage,
        "unassigned_sections": unassigned,
        "counts": dict(counts),
        "generated_at": utc_now(),
    }


def mark_interaction_source_revision(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    interaction_id: str | None,
    role_id: str | None,
    conflict_policy: str = "add_revision",
) -> dict[str, Any]:
    snapshot = store.snapshot()
    session = snapshot.sessions[session_id]
    asset = session.assets[asset_id]
    interaction_id = str(interaction_id or "").upper()
    role_id = str(role_id or "").upper()
    candidates = []
    for other in session.assets.values():
        if other.id == asset_id or asset_scope(other) != INTERACTION_SCOPE:
            continue
        identity = _asset_identity(other, session.label)
        if identity.get("interaction_id") == interaction_id and identity.get("role_id") == role_id and str(other.kind) == str(asset.kind):
            candidates.append(other)
    duplicate = next((row for row in candidates if row.source_sha256 and row.source_sha256 == asset.source_sha256), None)
    if duplicate:
        return {"asset": duplicate.model_dump(mode="json"), "reused": True, "reason": "identical_sha256"}
    revision = max([int((row.metadata or {}).get("interaction_revision_number") or 1) for row in candidates] + [0]) + 1

    def mutate(project):
        target = project.sessions[session_id].assets[asset_id]
        if conflict_policy in {"add_revision", "replace_active"}:
            for other in project.sessions[session_id].assets.values():
                if other.id == target.id:
                    continue
                identity = _asset_identity(other, project.sessions[session_id].label)
                if asset_scope(other) == INTERACTION_SCOPE and identity.get("interaction_id") == interaction_id and identity.get("role_id") == role_id and str(other.kind) == str(target.kind):
                    metadata = dict(other.metadata or {})
                    metadata["interaction_active"] = False
                    metadata["superseded_by_asset_id"] = target.id
                    other.metadata = metadata
        metadata = dict(target.metadata or {})
        metadata.update({
            "asset_scope": INTERACTION_SCOPE,
            "session_id": session_id,
            "interaction_id": interaction_id or None,
            "role_id": role_id or None,
            "interaction_active": True,
            "interaction_revision_number": revision,
            "interaction_registered_at": utc_now(),
        })
        target.metadata = metadata
        return target.model_dump(mode="json")
    updated = store.update(mutate)
    if interaction_id or role_id:
        update_asset_identity(
            store,
            session_id=session_id,
            asset_id=asset_id,
            interaction_id=interaction_id or None,
            role_id=role_id or None,
            source="interaction_append",
        )
    return {"asset": updated, "reused": False, "revision_number": revision}
