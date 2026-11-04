from __future__ import annotations

import re
from typing import Any

from oculostudio.domain.models import AssetRecord
from oculostudio.errors import ValidationError
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import utc_now

_CODE_PATTERNS = {
    "session_id": re.compile(r"(?<![A-Za-z0-9])(S\d{1,6})(?![A-Za-z0-9])", re.I),
    "interaction_id": re.compile(r"(?<![A-Za-z0-9])(I\d{1,6})(?![A-Za-z0-9])", re.I),
    "role_id": re.compile(r"(?<![A-Za-z0-9])(R\d{1,6})(?![A-Za-z0-9])", re.I),
}


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_code(value: str | None, key: str) -> str | None:
    text = _clean_optional(value)
    if not text:
        return None
    match = _CODE_PATTERNS[key].fullmatch(text.upper())
    if not match:
        expected = {"session_id": "S001", "interaction_id": "I01", "role_id": "R01"}[key]
        raise ValidationError(f"Invalid {key.replace('_', ' ')}: {text!r}. Use a code such as {expected}.")
    return match.group(1).upper()


def infer_asset_codes(asset: AssetRecord, session_label: str) -> dict[str, str | None]:
    identity = asset.metadata.get("identity") if isinstance(asset.metadata, dict) else None
    identity = identity if isinstance(identity, dict) else {}
    text = "\n".join(
        str(value)
        for value in (
            identity.get("session_id"),
            identity.get("interaction_id"),
            identity.get("role_id"),
            session_label,
            asset.label,
            asset.source_path,
        )
        if value
    )
    result: dict[str, str | None] = {}
    for key, pattern in _CODE_PATTERNS.items():
        explicit = _clean_optional(identity.get(key))
        match = pattern.fullmatch(explicit.upper()) if explicit else None
        if not match:
            match = pattern.search(text)
        result[key] = match.group(1).upper() if match else None
    return result


def asset_identity_view(asset: AssetRecord, session_label: str) -> dict[str, Any]:
    codes = infer_asset_codes(asset, session_label)
    identity = asset.metadata.get("identity") if isinstance(asset.metadata, dict) else None
    identity = identity if isinstance(identity, dict) else {}
    inferred_any = any(codes.values())
    source = str(identity.get("source") or ("import_user" if asset.participant_id else ("filename_inference" if inferred_any else "unassigned")))
    return {
        "participant_id": asset.participant_id,
        "session_id": codes["session_id"],
        "interaction_id": codes["interaction_id"],
        "role_id": codes["role_id"],
        "source": source,
        "updated_at": identity.get("updated_at"),
        "history_count": len(asset.metadata.get("identity_history") or []) if isinstance(asset.metadata, dict) else 0,
    }


def initial_identity_metadata(
    *,
    label: str,
    source_path: str,
    session_label: str,
    participant_id: str | None,
    source: str = "import_user",
) -> dict[str, Any]:
    pseudo = AssetRecord(label=label, source_path=source_path, participant_id=_clean_optional(participant_id))
    codes = infer_asset_codes(pseudo, session_label)
    return {
        "session_id": codes["session_id"],
        "interaction_id": codes["interaction_id"],
        "role_id": codes["role_id"],
        "participant_id": _clean_optional(participant_id),
        "source": source if _clean_optional(participant_id) else ("filename_inference" if any(codes.values()) else "unassigned"),
        "updated_at": utc_now(),
    }


def _refresh_session_participant_ids(session) -> None:
    values = sorted({
        str(asset.participant_id).strip()
        for asset in session.assets.values()
        if asset.participant_id and str(asset.participant_id).strip()
    })
    session.participant_ids = values


def update_asset_identity(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    participant_id: str | None,
    interaction_id: str | None = None,
    role_id: str | None = None,
    source: str = "user_edit",
) -> AssetRecord:
    participant = _clean_optional(participant_id)
    interaction = _normalise_code(interaction_id, "interaction_id") if interaction_id is not None else None
    role = _normalise_code(role_id, "role_id") if role_id is not None else None

    def mutate(project):
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown session: {session_id}")
        session = project.sessions[session_id]
        if asset_id not in session.assets:
            raise ValidationError(f"Unknown asset: {asset_id}")
        asset = session.assets[asset_id]
        before = asset_identity_view(asset, session.label)
        identity = dict(asset.metadata.get("identity") or {})
        identity.update({
            "session_id": before.get("session_id") or session.label,
            "interaction_id": interaction if interaction_id is not None else before.get("interaction_id"),
            "role_id": role if role_id is not None else before.get("role_id"),
            "participant_id": participant,
            "source": source,
            "updated_at": utc_now(),
        })
        history = list(asset.metadata.get("identity_history") or [])
        history.append({"changed_at": utc_now(), "source": source, "before": before, "after": dict(identity)})
        asset.participant_id = participant
        asset.metadata = {**asset.metadata, "identity": identity, "identity_history": history[-100:]}
        _refresh_session_participant_ids(session)
        return asset

    return store.update(mutate)


def set_initial_asset_identity(session, asset: AssetRecord, participant_id: str | None, source: str = "import_user") -> None:
    participant = _clean_optional(participant_id)
    asset.participant_id = participant
    identity = initial_identity_metadata(
        label=asset.label,
        source_path=asset.source_path,
        session_label=session.label,
        participant_id=participant,
        source=source,
    )
    asset.metadata = {**asset.metadata, "identity": identity}
    _refresh_session_participant_ids(session)


def remove_asset_and_refresh_participants(session, asset_id: str) -> AssetRecord:
    asset = session.assets.pop(asset_id)
    _refresh_session_participant_ids(session)
    return asset
