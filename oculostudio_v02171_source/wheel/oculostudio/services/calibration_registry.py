from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from oculostudio.errors import ValidationError
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import utc_now

CALIBRATION_EXTENSIONS = {".zip", ".c3d", ".npz", ".json", ".g3dcal", ".oculocal"}
_SET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PARTICIPANT_RE = re.compile(r"(?<![A-Za-z0-9])(P\d{1,6})(?![A-Za-z0-9])", re.I)


def normalize_calibration_set_id(value: str | None) -> str:
    text = str(value or "CALIBRATION").strip().replace(" ", "_")
    if not _SET_RE.fullmatch(text):
        raise ValidationError("Calibration set ID must use letters, numbers, underscore, dash or dot")
    return text


def infer_calibration_role(path: str | Path) -> str:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if suffix in {".g3dcal", ".oculocal"}:
        return "completed_profile"
    if suffix == ".json" and ("board" in name or "apriltag" in name):
        return "board_definition"
    if "markers_world" in name or "world_marker" in name or re.search(r"(?:^|[_-])world(?:[_-]|\.)", name):
        return "world_reference"
    if "calibration_board" in name or ("board" in name and suffix in {".c3d", ".npz"}):
        return "calibration_board"
    if re.search(r"eye[_-]?tracker|(?:^|[_-])etm[_-]?[ab12]", name) and suffix in {".c3d", ".npz"}:
        return "tracker_candidate"
    if suffix == ".zip" and ("calibration" in name or _PARTICIPANT_RE.search(name)):
        return "pupil_calibration"
    return "calibration_source"


def infer_hardware_hint(path: str | Path) -> str | None:
    name = Path(path).name
    patterns = [
        (r"eye[_-]?tracker[_-]?a(?:[^a-z0-9]|$)", "ETM-A"),
        (r"eye[_-]?tracker[_-]?b(?:[^a-z0-9]|$)", "ETM-B"),
        (r"(?:^|[_-])etm[_-]?a(?:[^a-z0-9]|$)", "ETM-A"),
        (r"(?:^|[_-])etm[_-]?b(?:[^a-z0-9]|$)", "ETM-B"),
    ]
    for pattern, value in patterns:
        if re.search(pattern, name, re.I):
            return value
    return None


def infer_participant(path: str | Path) -> str | None:
    match = _PARTICIPANT_RE.search(Path(path).name)
    return match.group(1).upper() if match else None


def calibration_metadata(path: str | Path, calibration_set_id: str) -> dict:
    return {
        "asset_scope": "calibration",
        "calibration_set_id": normalize_calibration_set_id(calibration_set_id),
        "calibration_role": infer_calibration_role(path),
        "hardware_hint": infer_hardware_hint(path),
        "calibration_registered_at": utc_now(),
    }


def expand_calibration_paths(paths: Iterable[str | Path], *, recursive: bool = False, maximum_files: int = 200) -> list[Path]:
    rows: list[Path] = []
    for raw in paths:
        path = Path(str(raw)).expanduser().resolve()
        if not path.exists():
            raise ValidationError(f"Calibration source does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in CALIBRATION_EXTENSIONS:
                raise ValidationError(f"Unsupported calibration file: {path.name}")
            rows.append(path)
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        for child in sorted(iterator):
            if child.is_file() and child.suffix.lower() in CALIBRATION_EXTENSIONS:
                rows.append(child.resolve())
                if len(rows) > maximum_files:
                    raise ValidationError(f"Calibration import is limited to {maximum_files} files")
    unique: list[Path] = []
    seen = set()
    for row in rows:
        key = str(row).casefold()
        if key not in seen:
            unique.append(row)
            seen.add(key)
    if not unique:
        raise ValidationError("No supported calibration files were found")
    return unique


def set_asset_calibration_scope(
    store: ProjectStore,
    *,
    session_id: str,
    asset_id: str,
    calibration_set_id: str,
    calibration_role: str | None = None,
) -> dict:
    set_id = normalize_calibration_set_id(calibration_set_id)

    def mutate(project):
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown session: {session_id}")
        session = project.sessions[session_id]
        if asset_id not in session.assets:
            raise ValidationError(f"Unknown asset: {asset_id}")
        asset = session.assets[asset_id]
        metadata = dict(asset.metadata or {})
        metadata.update({
            "asset_scope": "calibration",
            "calibration_set_id": set_id,
            "calibration_role": calibration_role or infer_calibration_role(asset.source_path),
            "hardware_hint": metadata.get("hardware_hint") or infer_hardware_hint(asset.source_path),
            "calibration_registered_at": utc_now(),
        })
        # Calibration sources are project-level resources. They must not inherit
        # an interaction wearer role from filename inference or a previous edit.
        identity = dict(metadata.get("identity") or {})
        identity.update({"interaction_id": None, "role_id": None, "source": "calibration_registry", "updated_at": utc_now()})
        metadata["identity"] = identity
        asset.metadata = metadata
        return asset.model_dump(mode="json")

    asset = store.update(mutate)
    from oculostudio.services.calibration_sets import ensure_calibration_set, register_calibration_asset_refs
    row = ensure_calibration_set(store, code=set_id, owner_session_id=session_id, applicable_session_ids=[session_id])
    register_calibration_asset_refs(
        store, calibration_set_id=row.id, source_session_id=session_id, asset_ids=[asset["id"]]
    )
    return asset


def scan_calibration_sources(
    store: ProjectStore,
    *,
    session_id: str,
    paths: Iterable[str | Path],
    calibration_set_id: str,
    recursive: bool = False,
) -> dict:
    """Return the exact files a calibration import will register.

    The scan is intentionally separate from import so the desktop can show a
    complete, auditable list before any mutation occurs.  This also avoids the
    previous ambiguity where a folder path looked selected but only one file
    had actually been queued.
    """
    set_id = normalize_calibration_set_id(calibration_set_id)
    files = expand_calibration_paths(paths, recursive=recursive)
    snapshot = store.snapshot()
    if session_id not in snapshot.sessions:
        raise ValidationError(f"Unknown calibration registry session: {session_id}")
    existing_by_path: dict[str, object] = {}
    # Calibration Sets are project-level. A source already registered in any
    # Session is considered present, even though target Sessions may hold
    # traceable aliases to the same immutable path.
    for candidate_session in snapshot.sessions.values():
        for asset in candidate_session.assets.values():
            metadata = dict(asset.metadata or {})
            if str(metadata.get("asset_scope") or "interaction") != "calibration":
                continue
            if str(metadata.get("calibration_set_id") or "") != set_id:
                continue
            if metadata.get("calibration_alias_source_asset_id"):
                continue
            try:
                key = str(Path(asset.source_path).expanduser().resolve()).casefold()
            except Exception:
                key = str(asset.source_path).casefold()
            existing_by_path[key] = asset

    rows = []
    role_counts: dict[str, int] = {}
    for path in files:
        metadata = calibration_metadata(path, set_id)
        role = str(metadata.get("calibration_role") or "calibration_source")
        role_counts[role] = role_counts.get(role, 0) + 1
        key = str(path.resolve()).casefold()
        existing = existing_by_path.get(key)
        rows.append({
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "calibration_role": role,
            "hardware_hint": metadata.get("hardware_hint"),
            "participant_id": infer_participant(path) if role == "pupil_calibration" else None,
            "already_registered": existing is not None,
            "existing_asset_id": getattr(existing, "id", None),
        })
    return {
        "calibration_set_id": set_id,
        "file_count": len(rows),
        "new_count": sum(1 for row in rows if not row["already_registered"]),
        "existing_count": sum(1 for row in rows if row["already_registered"]),
        "role_counts": role_counts,
        "files": rows,
    }


def import_calibration_batch(
    store: ProjectStore,
    *,
    session_id: str,
    paths: Iterable[str | Path],
    calibration_set_id: str,
    checksum: bool = True,
    recursive: bool = False,
    progress=None,
    cancellation=None,
) -> dict:
    """Register every calibration source as one visible batch job.

    Individual files remain independently immutable assets, but discovery,
    progress and completion are represented by one job.  A failure in one file
    is reported without silently preventing the remaining valid files from
    being registered.
    """
    from oculostudio.services.asset_import import import_local_asset

    set_id = normalize_calibration_set_id(calibration_set_id)
    from oculostudio.services.calibration_sets import ensure_calibration_set
    calibration_set = ensure_calibration_set(
        store, code=set_id, owner_session_id=session_id, applicable_session_ids=[session_id]
    )
    owner_session_id = calibration_set.owner_session_id or session_id
    files = expand_calibration_paths(paths, recursive=recursive)
    total = len(files)
    imported = []
    reused = []
    errors = []

    for index, path in enumerate(files):
        if cancellation is not None:
            cancellation.raise_if_cancelled()

        def file_progress(fraction: float, stage: str, *, _index=index, _path=path):
            if progress is not None:
                overall = (_index + max(0.0, min(1.0, float(fraction)))) / max(1, total)
                progress(overall, f"{_index + 1}/{total} · {_path.name} · {stage}")

        metadata = calibration_metadata(path, set_id)
        participant = infer_participant(path) if metadata.get("calibration_role") == "pupil_calibration" else None
        try:
            result = import_local_asset(
                store,
                session_id=owner_session_id,
                path=str(path),
                label=None,
                participant_id=participant,
                checksum=checksum,
                metadata_extra=metadata,
                progress=file_progress,
                cancellation=cancellation,
            )
            asset = result.get("asset") or {}
            row = {
                "asset_id": asset.get("id"),
                "label": asset.get("label") or path.name,
                "path": str(path),
                "calibration_role": metadata.get("calibration_role"),
            }
            (reused if result.get("reused") else imported).append(row)
        except Exception as exc:
            if cancellation is not None and getattr(cancellation, "cancelled", False):
                raise
            errors.append({"path": str(path), "name": path.name, "error": str(exc)})
        if progress is not None:
            progress((index + 1) / max(1, total), f"Processed {index + 1}/{total} calibration sources")

    if not imported and not reused:
        detail = "; ".join(f"{row['name']}: {row['error']}" for row in errors[:5])
        raise ValidationError("No calibration source could be registered" + (f": {detail}" if detail else ""))

    asset_ids = [row.get("asset_id") for row in [*imported, *reused] if row.get("asset_id")]
    from oculostudio.services.calibration_sets import register_calibration_asset_refs
    registration = register_calibration_asset_refs(
        store, calibration_set_id=calibration_set.id, source_session_id=owner_session_id, asset_ids=asset_ids
    )
    return {
        "calibration_set_id": set_id,
        "calibration_set_record_id": calibration_set.id,
        "owner_session_id": owner_session_id,
        "registration": registration,
        "file_count": total,
        "imported_count": len(imported),
        "reused_count": len(reused),
        "failed_count": len(errors),
        "imported": imported,
        "reused": reused,
        "errors": errors,
        "partial": bool(errors),
    }
