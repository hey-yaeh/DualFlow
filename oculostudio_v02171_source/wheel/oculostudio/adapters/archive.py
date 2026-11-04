from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from oculostudio.errors import AssetError
from oculostudio.util import atomic_write_json, sha256_file


class SourceView:
    """Read a directory, plain file, or ZIP without trusting archive paths."""

    def __init__(self, source: str | Path):
        self.source = Path(source).expanduser().resolve()
        if not self.source.exists():
            raise AssetError(f"Asset does not exist: {self.source}")
        self._zip: zipfile.ZipFile | None = None
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.source)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> "SourceView":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def is_zip(self) -> bool:
        return self._zip is not None

    @property
    def is_directory(self) -> bool:
        return self.source.is_dir()

    def names(self) -> list[str]:
        if self._zip is not None:
            return [n.replace("\\", "/") for n in self._zip.namelist() if not n.endswith("/")]
        if self.source.is_dir():
            return [p.relative_to(self.source).as_posix() for p in self.source.rglob("*") if p.is_file()]
        return [self.source.name]

    def find(self, *, basename: str | None = None, suffix: str | None = None, contains: str | None = None) -> list[str]:
        rows = []
        for name in self.names():
            base = PurePosixPath(name).name
            if basename is not None and base.lower() != basename.lower():
                continue
            if suffix is not None and not base.lower().endswith(suffix.lower()):
                continue
            if contains is not None and contains.lower() not in name.lower():
                continue
            rows.append(name)
        return sorted(rows)

    def read_bytes(self, name: str | None = None) -> bytes:
        if self._zip is not None:
            if name is None:
                raise AssetError("A ZIP member name is required")
            return self._zip.read(name)
        if self.source.is_dir():
            if name is None:
                raise AssetError("A relative file name is required")
            path = (self.source / name).resolve()
            if not str(path).startswith(str(self.source) + os.sep):
                raise AssetError("Path escapes the source directory")
            return path.read_bytes()
        if name not in (None, self.source.name):
            raise AssetError(f"Plain-file source has no member {name}")
        return self.source.read_bytes()

    def read_text(self, name: str | None = None, encoding: str = "utf-8") -> str:
        return self.read_bytes(name).decode(encoding, errors="replace")

    def read_json(self, name: str | None = None) -> dict:
        value = json.loads(self.read_text(name))
        if not isinstance(value, dict):
            raise AssetError(f"JSON root is not an object: {name or self.source}")
        return value


def _validate_zip_member(name: str, destination: Path) -> Path:
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise AssetError(f"Unsafe ZIP member: {name}")
    target = (destination / Path(*member.parts)).resolve()
    if not str(target).startswith(str(destination.resolve()) + os.sep) and target != destination.resolve():
        raise AssetError(f"Unsafe ZIP member: {name}")
    return target


def safe_extract_zip(
    zip_path: str | Path,
    destination: str | Path,
    *,
    maximum_uncompressed_bytes: int = 100 * 1024**3,
) -> Path:
    zip_path = Path(zip_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise AssetError(f"ZIP appears corrupt; first bad member: {bad}")
        total = sum(info.file_size for info in archive.infolist())
        if total > maximum_uncompressed_bytes:
            raise AssetError(f"ZIP expands to {total / 1024**3:.1f} GiB, above the safety limit")
        for info in archive.infolist():
            target = _validate_zip_member(info.filename, destination)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    return destination


class MaterializationCache:
    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def materialize(self, asset) -> Path:
        source = Path(asset.source_path).expanduser().resolve()
        if source.is_dir():
            return source
        if source.suffix.lower() != ".zip":
            return source
        target = self.cache_root / asset.id
        marker = target / ".oculostudio_materialized.json"
        expected_sha = asset.source_sha256 or sha256_file(source)
        if marker.exists():
            try:
                payload = json.loads(marker.read_text("utf-8"))
                if payload.get("source_sha256") == expected_sha:
                    return target
            except Exception:
                pass
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        safe_extract_zip(source, target)
        atomic_write_json(marker, {"source_path": str(source), "source_sha256": expected_sha})
        return target


def zip_directory(source: str | Path, output_zip: str | Path) -> Path:
    source = Path(source)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temp = output_zip.with_suffix(output_zip.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source).as_posix())
    os.replace(temp, output_zip)
    return output_zip
