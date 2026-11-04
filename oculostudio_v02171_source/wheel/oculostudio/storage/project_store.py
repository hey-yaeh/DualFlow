from __future__ import annotations

import copy
import re
import threading
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from oculostudio.domain.models import Project, Session
from oculostudio.errors import ValidationError
from oculostudio.util import atomic_write_json, safe_name, utc_now

T = TypeVar("T")

_SESSION_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(S\d{1,6})(?![A-Za-z0-9])", re.I)
_INTERACTION_CODE_RE = re.compile(r"^I\d{1,6}$", re.I)


def _session_code(label: str) -> str:
    match = _SESSION_CODE_RE.search(str(label or ""))
    return match.group(1).upper() if match else (str(label or "Session").strip().replace(" ", "_") or "SESSION")


def _normalise_session_labels(labels: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        text = str(raw or "").strip()
        if not text:
            continue
        match = _SESSION_CODE_RE.search(text)
        if not match:
            raise ValidationError(f"Session label must contain an ID such as S004: {text}")
        code = match.group(1).upper()
        key = code.casefold()
        if key in seen:
            continue
        seen.add(key)
        # Preserve legacy human-readable labels such as "Session S001" while
        # deriving the folder name from the embedded canonical S-code.
        result.append(text)
    if not result:
        raise ValidationError("Create at least one session")
    if len(result) > 200:
        raise ValidationError("A project can create at most 200 sessions in one operation")
    return result


def _normalise_interactions(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip().upper()
        if not text:
            continue
        if not _INTERACTION_CODE_RE.fullmatch(text):
            raise ValidationError(f"Interaction ID must look like I01: {text}")
        if text not in seen:
            seen.add(text)
            result.append(text)
    return sorted(result, key=lambda value: (int(value[1:]) if value[1:].isdigit() else 999999, value))


class ProjectStore:
    """Thread-safe, atomic JSON project persistence.

    Raw assets are referenced rather than copied unless the browser upload path
    is used. New projects use a human-readable folder layout:

        project/
          shared/calibration/
          sessions/S004/{imports,derived,reports,annotations,cache}/
          exports/

    Existing projects keep their legacy paths unchanged.
    """

    def __init__(self, project: Project):
        self._project = project
        self._lock = threading.RLock()
        self._ensure_directories()

    @classmethod
    def create(
        cls,
        root: str | Path,
        name: str,
        *,
        session_label: str = "S001",
        session_labels: Iterable[str] | None = None,
        settings: dict | None = None,
        planned_interactions: Iterable[str] | None = None,
    ) -> "ProjectStore":
        root = Path(root).expanduser().resolve()
        manifest = root / "project.oculostudio.json"
        if manifest.exists():
            raise ValidationError(f"An OculoStudio project already exists at {manifest}")
        if root.exists():
            ignored = {"Thumbs.db", ".DS_Store", "desktop.ini"}
            existing = [item.name for item in root.iterdir() if item.name not in ignored]
            if existing:
                preview = ", ".join(sorted(existing)[:5])
                more = "…" if len(existing) > 5 else ""
                raise ValidationError(
                    f"The new project directory must be empty: {root}. Existing items: {preview}{more}"
                )
        labels = _normalise_session_labels(session_labels or [session_label])
        interaction_plan = _normalise_interactions(planned_interactions or [])
        root.mkdir(parents=True, exist_ok=True)

        project_settings = copy.deepcopy(settings or {})
        project_settings.setdefault("folder_layout_version", 2)
        project_settings.setdefault("session_folder_strategy", "session_code")
        shared = dict(project_settings.get("shared_calibration") or {})
        shared.setdefault("enabled", True)
        shared.setdefault("scope", "project_hardware_profiles")
        shared["session_labels"] = labels
        project_settings["shared_calibration"] = shared

        sessions: dict[str, Session] = {}
        for label in labels:
            session = Session(
                label=label,
                metadata={
                    "session_code": _session_code(label),
                    "folder_slug": safe_name(_session_code(label)),
                    "planned_interactions": list(interaction_plan),
                    "created_from_project_wizard": True,
                },
            )
            sessions[session.id] = session
        selected_session_id = next(iter(sessions))
        project = Project(
            name=name.strip() or root.name,
            root=str(root),
            sessions=sessions,
            selected_session_id=selected_session_id,
            settings=project_settings,
        )
        store = cls(project)
        store.save()
        initial = dict(project_settings.get("initial_calibration_set") or {})
        if initial.get("enabled", True) and initial.get("code"):
            from oculostudio.services.calibration_sets import create_calibration_set
            create_calibration_set(
                store,
                code=str(initial.get("code")),
                label=str(initial.get("label") or initial.get("code")),
                applicable_session_ids=list(sessions.keys()) if initial.get("all_sessions", True) else list(initial.get("session_ids") or []),
                owner_session_id=selected_session_id,
                purpose=str(initial.get("purpose") or "combined"),
                notes=str(initial.get("notes") or ""),
            )
        return store

    @classmethod
    def open(cls, path: str | Path) -> "ProjectStore":
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path / "project.oculostudio.json"
        if not path.exists():
            raise ValidationError(f"Project manifest does not exist: {path}")
        project = Project.model_validate_json(path.read_text("utf-8"))
        project.root = str(path.parent)
        store = cls(project)
        # v0.21.67 promotes the earlier per-Session calibration_set_id tags into
        # project-level Calibration Sets. The migration is idempotent and never
        # copies or rewrites raw source files.
        try:
            from oculostudio.services.calibration_sets import migrate_legacy_calibration_sets
            migrate_legacy_calibration_sets(store)
        except Exception:
            # Opening an older project must remain possible even when one legacy
            # calibration source is unavailable. The manager can adopt it later.
            pass
        return store

    @property
    def project(self) -> Project:
        with self._lock:
            return self._project

    @property
    def root(self) -> Path:
        return Path(self._project.root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "project.oculostudio.json"

    @property
    def folder_layout_version(self) -> int:
        try:
            return int((self._project.settings or {}).get("folder_layout_version") or 1)
        except Exception:
            return 1

    def _ensure_directories(self) -> None:
        if self.folder_layout_version >= 2:
            for rel in ("cache", "logs", "exports", "shared/calibration", "calibration_sets", "sessions"):
                (self.root / rel).mkdir(parents=True, exist_ok=True)
            for session_id in self._project.sessions:
                self._ensure_session_directories(session_id)
            for calibration_set_id in self._project.calibration_sets:
                self.ensure_calibration_set_directories(calibration_set_id)
        else:
            for rel in ("derived", "reports", "cache", "logs", "exports", "annotations", "imports"):
                (self.root / rel).mkdir(parents=True, exist_ok=True)

    def _ensure_session_directories(self, session_id: str) -> None:
        if session_id not in self._project.sessions:
            return
        root = self.session_dir(session_id)
        root.mkdir(parents=True, exist_ok=True)
        for rel in ("imports", "derived", "reports", "annotations", "cache"):
            (root / rel).mkdir(parents=True, exist_ok=True)

    def _write_session_indexes(self) -> None:
        if self.folder_layout_version < 2:
            return
        for session_id, session in self._project.sessions.items():
            root = self.session_dir(session_id)
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                root / "session.oculostudio.json",
                {
                    "schema_version": "oculostudio-session-folder-1.0",
                    "session_id": session_id,
                    "session_code": _session_code(session.label),
                    "label": session.label,
                    "project_id": self._project.id,
                    "project_name": self._project.name,
                    "planned_interactions": list(session.metadata.get("planned_interactions") or []),
                    "paths": {key: str(value) for key, value in self.session_paths(session_id).items()},
                },
            )

    def calibration_sets_dir(self) -> Path:
        path = self.root / "calibration_sets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def calibration_set_dir(self, calibration_set_id: str) -> Path:
        row = self._project.calibration_sets.get(calibration_set_id)
        if row is None:
            for candidate in self._project.calibration_sets.values():
                if candidate.code.casefold() == str(calibration_set_id).casefold():
                    row = candidate
                    break
        if row is None:
            raise ValidationError(f"Unknown Calibration Set: {calibration_set_id}")
        path = self.calibration_sets_dir() / safe_name(row.code)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def calibration_revision_dir(self, calibration_set_id: str, revision_id: str) -> Path:
        row = self._project.calibration_sets.get(calibration_set_id)
        if row is None:
            for candidate in self._project.calibration_sets.values():
                if candidate.code.casefold() == str(calibration_set_id).casefold():
                    row = candidate
                    break
        if row is None or revision_id not in row.revisions:
            raise ValidationError("Unknown Calibration Set revision")
        revision = row.revisions[revision_id]
        path = self.calibration_set_dir(row.id) / "revisions" / f"rev_{revision.revision_number:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_calibration_set_directories(self, calibration_set_id: str) -> None:
        row = self._project.calibration_sets.get(calibration_set_id)
        if row is None:
            return
        root = self.calibration_set_dir(row.id)
        for rel in ("sources", "derived", "reports"):
            (root / rel).mkdir(parents=True, exist_ok=True)
        for revision_id in row.revisions:
            revision_root = self.calibration_revision_dir(row.id, revision_id)
            for rel in ("sources", "derived", "reports"):
                (revision_root / rel).mkdir(parents=True, exist_ok=True)

    def _write_calibration_set_indexes(self) -> None:
        if self.folder_layout_version < 2:
            return
        for calibration_set_id, row in self._project.calibration_sets.items():
            root = self.calibration_set_dir(calibration_set_id)
            atomic_write_json(
                root / "calibration_set.json",
                {
                    "schema_version": "oculostudio-calibration-set-1.0",
                    "project_id": self._project.id,
                    "calibration_set": row.model_dump(mode="json"),
                    "folder": str(root),
                },
            )
            for revision_id, revision in row.revisions.items():
                revision_root = self.calibration_revision_dir(calibration_set_id, revision_id)
                atomic_write_json(
                    revision_root / "revision.json",
                    {
                        "schema_version": "oculostudio-calibration-set-revision-1.0",
                        "project_id": self._project.id,
                        "calibration_set_id": calibration_set_id,
                        "calibration_set_code": row.code,
                        "revision": revision.model_dump(mode="json"),
                    },
                )

    def snapshot(self) -> Project:
        with self._lock:
            return self._project.model_copy(deep=True)

    def save(self) -> None:
        with self._lock:
            self._ensure_directories()
            self._project.revision += 1
            self._project.modified_at = utc_now()
            backup = self.manifest_path.with_suffix(".json.bak")
            if self.manifest_path.exists():
                backup.write_bytes(self.manifest_path.read_bytes())
            atomic_write_json(self.manifest_path, self._project.model_dump(mode="json"))
            self._write_session_indexes()
            self._write_calibration_set_indexes()

    def update(self, mutator: Callable[[Project], T], *, save: bool = True) -> T:
        with self._lock:
            result = mutator(self._project)
            if save:
                self.save()
            return result

    def selected_session(self) -> Session:
        with self._lock:
            sid = self._project.selected_session_id
            if not sid or sid not in self._project.sessions:
                raise ValidationError("No session is selected")
            return self._project.sessions[sid]

    def session_code(self, session_id: str) -> str:
        if session_id not in self._project.sessions:
            raise ValidationError(f"Unknown session: {session_id}")
        return _session_code(self._project.sessions[session_id].label)

    def session_dir(self, session_id: str) -> Path:
        if session_id not in self._project.sessions:
            raise ValidationError(f"Unknown session: {session_id}")
        session = self._project.sessions[session_id]
        if self.folder_layout_version >= 2:
            slug = str(session.metadata.get("folder_slug") or safe_name(_session_code(session.label)))
            return self.root / "sessions" / slug
        return self.root

    def session_paths(self, session_id: str) -> dict[str, Path]:
        base = self.session_dir(session_id)
        if self.folder_layout_version >= 2:
            return {
                "root": base,
                "imports": base / "imports",
                "derived": base / "derived",
                "reports": base / "reports",
                "annotations": base / "annotations",
                "cache": base / "cache",
                "exports": self.root / "exports" / self.session_code(session_id),
            }
        key = safe_name(session_id)
        return {
            "root": self.root,
            "imports": self.root / "imports",
            "derived": self.root / "derived" / key,
            "reports": self.root / "reports" / key,
            "annotations": self.root / "annotations" / key,
            "cache": self.root / "cache",
            "exports": self.root / "exports",
        }

    def shared_calibration_dir(self) -> Path:
        path = self.root / "shared" / "calibration"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def import_dir(self, session_id: str) -> Path:
        path = self.session_paths(session_id)["imports"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def annotations_dir(self, session_id: str, *parts: str) -> Path:
        path = self.session_paths(session_id)["annotations"]
        for part in parts:
            path = path / safe_name(part)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_sessions(self, labels: Iterable[str], *, select_first: bool = False) -> list[Session]:
        normalized = _normalise_session_labels(labels)
        created: list[Session] = []

        def mutate(project: Project) -> None:
            existing_codes = {_session_code(session.label).casefold() for session in project.sessions.values()}
            for label in normalized:
                code = _session_code(label)
                if code.casefold() in existing_codes:
                    continue
                session = Session(
                    label=code,
                    metadata={
                        "session_code": code,
                        "folder_slug": safe_name(code),
                        "planned_interactions": [],
                        "created_from_session_manager": True,
                    },
                )
                project.sessions[session.id] = session
                created.append(session)
                existing_codes.add(code.casefold())
            shared = dict((project.settings or {}).get("shared_calibration") or {})
            shared["session_labels"] = [_session_code(row.label) for row in project.sessions.values()]
            project.settings["shared_calibration"] = shared
            if select_first and created:
                project.selected_session_id = created[0].id

        self.update(mutate)
        return created

    def add_session(self, label: str) -> Session:
        rows = self.add_sessions([label], select_first=True)
        if rows:
            return rows[0]
        code = _session_code(label)
        for session in self._project.sessions.values():
            if _session_code(session.label) == code:
                self.select_session(session.id)
                return session
        raise ValidationError(f"Could not create session {label}")

    def plan_interactions(self, session_id: str, interaction_ids: Iterable[str], *, replace: bool = False) -> list[str]:
        normalized = _normalise_interactions(interaction_ids)

        def mutate(project: Project) -> list[str]:
            if session_id not in project.sessions:
                raise ValidationError(f"Unknown session: {session_id}")
            session = project.sessions[session_id]
            current = [] if replace else list(session.metadata.get("planned_interactions") or [])
            result = _normalise_interactions([*current, *normalized])
            session.metadata["planned_interactions"] = result
            return result

        return self.update(mutate)

    def select_session(self, session_id: str) -> None:
        def mutate(project: Project) -> None:
            if session_id not in project.sessions:
                raise ValidationError(f"Unknown session: {session_id}")
            project.selected_session_id = session_id

        self.update(mutate)

    def derived_dir(self, session_id: str, category: str, item_id: str | None = None) -> Path:
        path = self.session_paths(session_id)["derived"] / safe_name(category)
        if item_id:
            path = path / safe_name(item_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def report_dir(self, session_id: str) -> Path:
        path = self.session_paths(session_id)["reports"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_dir(self, asset_id: str | None = None) -> Path:
        # Cache stays project-level because many immutable plans bridge two roles
        # and sometimes two sessions. Session folders hold the durable outputs.
        path = self.root / "cache"
        if asset_id:
            path = path / "assets" / safe_name(asset_id)
        path.mkdir(parents=True, exist_ok=True)
        return path
