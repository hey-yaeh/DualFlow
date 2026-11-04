from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PACKAGE_MANIFEST.json"
EXCLUDED_DIRS = {
    ".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "__pycache__", "build", "OculoStudio_Demo",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT:
        return True
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def main() -> int:
    rows = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and not excluded(p)):
        rows.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "schema_version": "oculostudio-package-manifest-1.0",
        "release": "OculoStudio v0.21.71 Independent Board Geometry Resolver",
        "file_count": len(rows),
        "files": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    print(f"Wrote {OUTPUT} with {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
