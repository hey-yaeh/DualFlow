from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from oculostudio.errors import ValidationError
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json, safe_name, utc_now


def _contains_artifact_id(value: Any, artifact_id: str) -> bool:
    if isinstance(value, str):
        return value == artifact_id
    if isinstance(value, dict):
        return any(_contains_artifact_id(item, artifact_id) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_artifact_id(item, artifact_id) for item in value)
    return False


def _collect_known_artifact_ids(value: Any, known_ids: set[str], out: set[str]) -> None:
    if isinstance(value, str):
        if value in known_ids:
            out.add(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_known_artifact_ids(item, known_ids, out)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_known_artifact_ids(item, known_ids, out)


def _project_artifacts(project) -> dict[str, tuple[str, Any]]:
    rows: dict[str, tuple[str, Any]] = {}
    for session_id, session in project.sessions.items():
        for artifact_id, artifact in session.artifacts.items():
            rows[artifact_id] = (session_id, artifact)
    return rows


def _dependency_maps(project) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    rows = _project_artifacts(project)
    known_ids = set(rows)
    dependencies: dict[str, set[str]] = {artifact_id: set() for artifact_id in rows}
    dependents: dict[str, set[str]] = {artifact_id: set() for artifact_id in rows}
    for artifact_id, (_session_id, artifact) in rows.items():
        refs = set(artifact.parent_artifact_ids or []) & known_ids
        # Some older artifacts recorded lineage only in parameters/QC/provenance.
        for value in (artifact.parameters, artifact.qc, artifact.provenance):
            _collect_known_artifact_ids(value, known_ids, refs)
        refs.discard(artifact_id)
        dependencies[artifact_id] = refs
        for ref in refs:
            dependents.setdefault(ref, set()).add(artifact_id)
    return dependencies, dependents


def _artifact_dependents(project, artifact_id: str) -> list[tuple[str, Any]]:
    rows = _project_artifacts(project)
    _dependencies, dependents = _dependency_maps(project)
    result: list[tuple[str, Any]] = []
    for dependent_id in sorted(dependents.get(artifact_id, set())):
        if dependent_id in rows:
            result.append(rows[dependent_id])
    return result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _controlled_roots(project_root: Path) -> tuple[Path, ...]:
    return tuple((project_root / name).resolve() for name in ("derived", "reports", "exports"))


def _is_controlled_path(path: Path, project_root: Path) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == root or _path_is_within(resolved, root) for root in _controlled_roots(project_root))


def _path_size(path: Path) -> tuple[int, int]:
    try:
        if path.is_file():
            return int(path.stat().st_size), 1
        if not path.exists():
            return 0, 0
        total = 0
        count = 0
        for row in path.rglob("*"):
            try:
                if row.is_file():
                    total += int(row.stat().st_size)
                    count += 1
            except OSError:
                continue
        return total, count
    except OSError:
        return 0, 0


def _artifact_existing_paths(artifact) -> list[Path]:
    rows: list[Path] = []
    seen: set[str] = set()
    for file_row in artifact.files or []:
        if not file_row.path:
            continue
        path = Path(file_row.path).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        rows.append(resolved)
    return rows


def _artifact_common_root(artifact) -> Path | None:
    paths = _artifact_existing_paths(artifact)
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    try:
        return Path(os.path.commonpath([str(path) for path in paths])).resolve()
    except (ValueError, OSError):
        return paths[0]


def _all_active_file_paths(project, *, excluding_ids: set[str] | None = None) -> dict[str, str]:
    excluding_ids = excluding_ids or set()
    rows: dict[str, str] = {}
    for _session_id, session in project.sessions.items():
        for artifact in session.artifacts.values():
            if artifact.id in excluding_ids:
                continue
            for file_row in artifact.files or []:
                if not file_row.path:
                    continue
                try:
                    resolved = Path(file_row.path).expanduser().resolve()
                except OSError:
                    continue
                rows[os.path.normcase(str(resolved))] = artifact.id
    return rows


def _directory_contains_other_artifact_files(
    directory: Path,
    project,
    *,
    excluding_ids: set[str],
) -> bool:
    for path_text in _all_active_file_paths(project, excluding_ids=excluding_ids):
        path = Path(path_text)
        if path == directory or _path_is_within(path, directory):
            return True
    return False


def _prune_empty_parents(path: Path, stop_roots: Iterable[Path]) -> None:
    stop = {root.resolve() for root in stop_roots}
    current = path.resolve()
    while current not in stop:
        try:
            if not current.exists() or not current.is_dir() or any(current.iterdir()):
                return
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _move_artifact_files(
    store: ProjectStore,
    *,
    artifact,
    trash_root: Path,
    excluding_ids: set[str],
) -> list[dict[str, str]]:
    project_root = store.root.resolve()
    controlled_roots = _controlled_roots(project_root)
    existing_files = _artifact_existing_paths(artifact)
    moved: list[dict[str, str]] = []
    if not existing_files:
        return moved

    common = _artifact_common_root(artifact)
    can_move_whole = bool(
        common
        and common.is_dir()
        and _is_controlled_path(common, project_root)
        and common.resolve() not in set(controlled_roots)
        and not _directory_contains_other_artifact_files(common, store.project, excluding_ids=excluding_ids)
    )
    if can_move_whole and common is not None:
        target = trash_root / "files" / common.relative_to(project_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(common), str(target))
        moved.append({"source": str(common), "target": str(target), "mode": "directory"})
        _prune_empty_parents(common.parent, controlled_roots)
        return moved

    for index, path in enumerate(existing_files):
        if not _is_controlled_path(path, project_root):
            moved.append(
                {
                    "source": str(path),
                    "target": str(trash_root),
                    "mode": "external_left_untouched",
                }
            )
            continue
        try:
            relative = path.relative_to(project_root)
        except ValueError:
            relative = Path(f"external_{index:03d}_{path.name}")
        destination = trash_root / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}_{index:03d}{destination.suffix}")
        shutil.move(str(path), str(destination))
        moved.append({"source": str(path), "target": str(destination), "mode": "file"})
        _prune_empty_parents(path.parent, controlled_roots)
    return moved


def move_artifact_to_trash(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_id: str,
) -> dict[str, Any]:
    """Move one derived artifact out of the active project.

    Raw assets are never touched. The operation fails closed when any active
    downstream artifact still references the selected artifact. Project-controlled
    files under derived/, reports/ or exports/ are moved into .trash; external
    files are left untouched and only the active registry entry is removed.
    """

    project = store.snapshot()
    session = project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    artifact = session.artifacts.get(artifact_id)
    if artifact is None:
        raise ValidationError(f"Unknown artifact: {artifact_id}")

    dependents = _artifact_dependents(project, artifact_id)
    if dependents:
        labels = [f"{row.label} ({row.kind})" for _sid, row in dependents]
        preview = "; ".join(labels[:4])
        suffix = "…" if len(labels) > 4 else ""
        raise ValidationError(
            "This artifact is still used by downstream result(s): "
            f"{preview}{suffix}. Remove those dependent results first, or select the complete dependency chain."
        )

    timestamp = utc_now().replace(":", "-").replace("+", "_")
    trash_root = store.root / ".trash" / "artifacts" / f"{timestamp}_{safe_name(artifact.id)}"
    trash_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(trash_root / "artifact_record.json", artifact.model_dump(mode="json"))

    moved = _move_artifact_files(
        store,
        artifact=artifact,
        trash_root=trash_root,
        excluding_ids={artifact_id},
    )
    atomic_write_json(
        trash_root / "trash_manifest.json",
        {
            "schema_version": "oculostudio-artifact-trash-1.1",
            "artifact_id": artifact.id,
            "artifact_label": artifact.label,
            "artifact_kind": artifact.kind,
            "session_id": session_id,
            "moved": moved,
            "removed_at": utc_now(),
        },
    )

    def mutate(active_project):
        current_session = active_project.sessions[session_id]
        current_session.artifacts.pop(artifact_id, None)
        for job_id, job in list(active_project.jobs.items()):
            if job.session_id != session_id:
                continue
            if _contains_artifact_id(job.result, artifact_id):
                del active_project.jobs[job_id]

    store.update(mutate)
    return {
        "artifact_id": artifact_id,
        "label": artifact.label,
        "kind": artifact.kind,
        "trash_path": str(trash_root),
        "moved": moved,
    }


def _transitive_dependents(artifact_id: str, dependents: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    stack = list(dependents.get(artifact_id, set()))
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(dependents.get(current, set()))
    return result


def move_artifacts_to_trash(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_ids: list[str],
) -> list[dict[str, Any]]:
    selected = {str(value) for value in artifact_ids if str(value)}
    if not selected:
        return []
    project = store.snapshot()
    session = project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    unknown = selected - set(session.artifacts)
    if unknown:
        raise ValidationError(f"Unknown artifact(s): {', '.join(sorted(unknown))}")

    _dependencies, dependents = _dependency_maps(project)
    blocked: dict[str, set[str]] = {}
    for artifact_id in selected:
        external = _transitive_dependents(artifact_id, dependents) - selected
        if external:
            blocked[artifact_id] = external
    if blocked:
        rows = _project_artifacts(project)
        messages: list[str] = []
        for artifact_id, dependent_ids in sorted(blocked.items()):
            artifact = session.artifacts[artifact_id]
            labels = [rows[row_id][1].label for row_id in sorted(dependent_ids) if row_id in rows]
            messages.append(f"{artifact.label}: {', '.join(labels[:4])}{'…' if len(labels) > 4 else ''}")
        raise ValidationError(
            "The selected cleanup would leave active downstream artifacts without their inputs. "
            "Select the complete dependency chain or remove the downstream artifacts first. "
            + " | ".join(messages)
        )

    # Remove downstream leaves first so the single-artifact fail-closed check
    # remains active throughout the batch operation.
    remaining = set(selected)
    order: list[str] = []
    while remaining:
        leaves = sorted(
            artifact_id
            for artifact_id in remaining
            if not (dependents.get(artifact_id, set()) & remaining)
        )
        if not leaves:
            raise ValidationError("Artifact dependency cycle detected; cleanup was not started")
        order.extend(leaves)
        remaining.difference_update(leaves)

    removed: list[dict[str, Any]] = []
    for artifact_id in order:
        removed.append(move_artifact_to_trash(store, session_id=session_id, artifact_id=artifact_id))
    return removed


def _registered_project_paths(project) -> set[str]:
    result: set[str] = set()
    for _session_id, session in project.sessions.items():
        for artifact in session.artifacts.values():
            for file_row in artifact.files or []:
                if not file_row.path:
                    continue
                try:
                    result.add(os.path.normcase(str(Path(file_row.path).expanduser().resolve())))
                except OSError:
                    continue
    return result


def _candidate_orphan_paths(store: ProjectStore, session_id: str) -> list[Path]:
    project_root = store.root.resolve()
    session_key = safe_name(session_id)
    paths = store.session_paths(session_id)
    roots = [
        paths["derived"],
        paths["reports"],
        paths["exports"],
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        if root == paths["derived"]:
            # session/derived/<category>/<artifact-root-or-file> (or legacy derived/<session>/...)
            for category in sorted(root.iterdir()):
                if not category.is_dir():
                    candidates.append(category)
                    continue
                children = list(category.iterdir())
                if not children:
                    candidates.append(category)
                else:
                    candidates.extend(children)
        else:
            candidates.extend(root.iterdir())
    return [row.resolve() for row in candidates if row.exists()]


def scan_orphan_project_outputs(store: ProjectStore, *, session_id: str) -> list[dict[str, Any]]:
    project = store.snapshot()
    if session_id not in project.sessions:
        raise ValidationError(f"Unknown session: {session_id}")
    registered = _registered_project_paths(project)
    rows: list[dict[str, Any]] = []
    for path in _candidate_orphan_paths(store, session_id):
        path_key = os.path.normcase(str(path))
        referenced = False
        for registered_text in registered:
            registered_path = Path(registered_text)
            if registered_path == path or _path_is_within(registered_path, path) or _path_is_within(path, registered_path):
                referenced = True
                break
        if referenced:
            continue
        size_bytes, file_count = _path_size(path)
        try:
            relative = str(path.relative_to(store.root.resolve()))
        except ValueError:
            relative = str(path)
        rows.append(
            {
                "path": str(path),
                "relative_path": relative,
                "name": path.name,
                "entry_type": "folder" if path.is_dir() else "file",
                "size_bytes": size_bytes,
                "file_count": file_count,
                "safe_to_remove": True,
                "reason": "No active artifact record references this project-controlled output.",
            }
        )
    return sorted(rows, key=lambda row: (str(row["relative_path"]).lower()))


def move_orphan_outputs_to_trash(
    store: ProjectStore,
    *,
    session_id: str,
    paths: list[str],
) -> list[dict[str, Any]]:
    requested = {os.path.normcase(str(Path(value).expanduser().resolve())) for value in paths if value}
    if not requested:
        return []
    current = {os.path.normcase(row["path"]): row for row in scan_orphan_project_outputs(store, session_id=session_id)}
    invalid = requested - set(current)
    if invalid:
        raise ValidationError(
            "One or more selected paths are no longer unreferenced project outputs. Refresh the cleanup scan before deleting."
        )
    timestamp = utc_now().replace(":", "-").replace("+", "_")
    trash_root = store.root / ".trash" / "orphans" / timestamp
    trash_root.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, Any]] = []
    project_root = store.root.resolve()
    for key in sorted(requested):
        path = Path(key)
        if not _is_controlled_path(path, project_root):
            raise ValidationError(f"Refusing to move a path outside project-controlled outputs: {path}")
        try:
            relative = path.relative_to(project_root)
        except ValueError as exc:
            raise ValidationError(f"Invalid orphan output path: {path}") from exc
        destination = trash_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        size_bytes, file_count = _path_size(path)
        shutil.move(str(path), str(destination))
        moved.append(
            {
                "source": str(path),
                "target": str(destination),
                "relative_path": str(relative),
                "size_bytes": size_bytes,
                "file_count": file_count,
            }
        )
        _prune_empty_parents(path.parent, _controlled_roots(project_root))
    atomic_write_json(
        trash_root / "trash_manifest.json",
        {
            "schema_version": "oculostudio-orphan-trash-1.0",
            "session_id": session_id,
            "moved": moved,
            "removed_at": utc_now(),
        },
    )
    return moved


def _trash_summary(store: ProjectStore) -> dict[str, Any]:
    root = store.root / ".trash"
    artifact_entries: list[dict[str, Any]] = []
    orphan_entries: list[dict[str, Any]] = []
    for category, target in (("artifacts", artifact_entries), ("orphans", orphan_entries)):
        category_root = root / category
        if not category_root.exists():
            continue
        for entry in sorted(category_root.iterdir(), reverse=True):
            size_bytes, file_count = _path_size(entry)
            manifest_path = entry / "trash_manifest.json"
            manifest: dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text("utf-8"))
                except Exception:
                    manifest = {}
            target.append(
                {
                    "path": str(entry),
                    "name": entry.name,
                    "size_bytes": size_bytes,
                    "file_count": file_count,
                    "manifest": manifest,
                }
            )
    total_bytes = sum(row["size_bytes"] for row in artifact_entries + orphan_entries)
    return {
        "path": str(root),
        "artifact_entries": artifact_entries,
        "orphan_entries": orphan_entries,
        "entry_count": len(artifact_entries) + len(orphan_entries),
        "size_bytes": total_bytes,
    }


def build_artifact_cleanup_plan(
    store: ProjectStore,
    *,
    session_id: str,
    scan_orphans: bool = False,
) -> dict[str, Any]:
    project = store.snapshot()
    session = project.sessions.get(session_id)
    if session is None:
        raise ValidationError(f"Unknown session: {session_id}")
    rows = _project_artifacts(project)
    dependencies, dependents = _dependency_maps(project)

    duplicates: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for artifact in session.artifacts.values():
        duplicates[(str(artifact.kind), str(artifact.label).strip().casefold())].append(artifact)
    superseded_by: dict[str, str] = {}
    for group in duplicates.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda row: (str(row.created_at), row.id))
        newest = ordered[-1]
        for older in ordered[:-1]:
            superseded_by[older.id] = newest.id

    artifact_rows: list[dict[str, Any]] = []
    for artifact in sorted(session.artifacts.values(), key=lambda row: (str(row.created_at), row.id), reverse=True):
        existing = _artifact_existing_paths(artifact)
        registered_count = len(artifact.files or [])
        missing_count = max(0, registered_count - len(existing))
        existing_bytes = sum(_path_size(path)[0] for path in existing if path.is_file())
        # Avoid double counting directory roots: registered file sizes are authoritative.
        registered_bytes = sum(int(row.size_bytes or 0) for row in artifact.files or [])
        dependent_ids = sorted(dependents.get(artifact.id, set()))
        dependency_ids = sorted(dependencies.get(artifact.id, set()))
        transitive = sorted(_transitive_dependents(artifact.id, dependents))
        replacement_id = superseded_by.get(artifact.id)
        replacement = rows.get(replacement_id, (None, None))[1] if replacement_id else None
        reasons: list[str] = []
        suggested = False
        if replacement is not None:
            suggested = True
            reasons.append(f"A newer artifact with the same kind and label exists: {replacement.label}")
        if registered_count and missing_count == registered_count:
            suggested = True
            reasons.append("All registered output files are missing from disk.")
        elif missing_count:
            reasons.append(f"{missing_count} of {registered_count} registered files are missing.")
        if not registered_count:
            reasons.append("Artifact record has no registered output files.")
        common_root = _artifact_common_root(artifact)
        external_count = sum(1 for path in existing if not _is_controlled_path(path, store.root.resolve()))
        dependent_rows = []
        for dependent_id in dependent_ids:
            if dependent_id not in rows:
                continue
            dep_session_id, dep = rows[dependent_id]
            dependent_rows.append(
                {
                    "id": dependent_id,
                    "label": dep.label,
                    "kind": dep.kind,
                    "session_id": dep_session_id,
                }
            )
        artifact_rows.append(
            {
                "id": artifact.id,
                "label": artifact.label,
                "kind": artifact.kind,
                "created_at": artifact.created_at,
                "producer_version": artifact.provenance.get("producer_version") if isinstance(artifact.provenance, dict) else None,
                "registered_file_count": registered_count,
                "missing_file_count": missing_count,
                "registered_size_bytes": registered_bytes,
                "existing_size_bytes": existing_bytes,
                "root_path": str(common_root) if common_root else "",
                "external_file_count": external_count,
                "dependency_ids": dependency_ids,
                "dependent_ids": dependent_ids,
                "dependents": dependent_rows,
                "transitive_dependent_ids": transitive,
                "safe_to_remove": not dependent_ids,
                "suggested": suggested,
                "suggested_reasons": reasons,
                "superseded_by_id": replacement_id,
                "superseded_by_label": replacement.label if replacement is not None else None,
            }
        )

    orphans = scan_orphan_project_outputs(store, session_id=session_id) if scan_orphans else []
    trash = _trash_summary(store)
    return {
        "session_id": session_id,
        "generated_at": utc_now(),
        "scan_orphans": bool(scan_orphans),
        "artifacts": artifact_rows,
        "orphans": orphans,
        "trash": trash,
        "summary": {
            "artifact_count": len(artifact_rows),
            "safe_artifact_count": sum(1 for row in artifact_rows if row["safe_to_remove"]),
            "blocked_artifact_count": sum(1 for row in artifact_rows if not row["safe_to_remove"]),
            "suggested_artifact_count": sum(1 for row in artifact_rows if row["suggested"]),
            "artifact_size_bytes": sum(int(row["registered_size_bytes"] or 0) for row in artifact_rows),
            "orphan_count": len(orphans),
            "orphan_size_bytes": sum(int(row["size_bytes"] or 0) for row in orphans),
        },
    }


def purge_project_trash(store: ProjectStore, *, confirmation: str) -> dict[str, Any]:
    project = store.snapshot()
    expected = project.name.strip()
    if confirmation.strip() != expected:
        raise ValidationError(f"Type the exact project name to permanently empty Project Trash: {expected}")
    if any(str(job.status) in {"queued", "running"} for job in project.jobs.values()):
        raise ValidationError("Project Trash cannot be purged while processing jobs are active")
    root = store.root / ".trash"
    size_bytes, file_count = _path_size(root)
    entry_count = 0
    if root.exists():
        entry_count = sum(1 for _ in root.iterdir())
        shutil.rmtree(root)
    return {
        "path": str(root),
        "size_bytes": size_bytes,
        "file_count": file_count,
        "entry_count": entry_count,
        "purged_at": utc_now(),
    }
