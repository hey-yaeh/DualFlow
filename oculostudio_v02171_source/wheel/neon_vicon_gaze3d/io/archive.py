from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import tempfile
import zipfile


class ArchiveError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(zip_path: str | Path, destination: str | Path) -> Path:
    zip_path = Path(zip_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if base not in target.parents and target != base:
                raise ArchiveError(f"Unsafe ZIP member path: {member.filename}")
        archive.extractall(destination)
    return destination


class MaterializedSource:
    """Materialise a ZIP or retain a directory for the life of this object."""

    def __init__(self, source: str | Path, parent: str | Path | None = None):
        self.source = Path(source)
        if not self.source.exists():
            raise ArchiveError(f"Input does not exist: {self.source}")
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        if self.source.is_dir():
            self.root = self.source
        elif self.source.suffix.lower() == ".zip":
            self._temp = tempfile.TemporaryDirectory(
                prefix="neon_vicon_gaze3d_", dir=str(parent) if parent else None
            )
            self.root = safe_extract_zip(self.source, self._temp.name)
        else:
            raise ArchiveError("Expected a directory or ZIP archive")

    def close(self) -> None:
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self) -> "MaterializedSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def find_unique(root: Path, filename: str, *, required: bool = True) -> Path | None:
    matches = list(root.rglob(filename))
    if not matches:
        if required:
            raise ArchiveError(f"Could not find {filename} below {root}")
        return None
    if len(matches) > 1:
        # Prefer the shallowest recording folder, then deterministic ordering.
        matches.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    return matches[0]


def copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
