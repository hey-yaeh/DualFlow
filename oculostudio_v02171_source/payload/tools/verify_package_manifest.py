from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"Missing package manifest: {MANIFEST}", file=sys.stderr)
        return 2
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in payload.get("files", []):
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != row["size_bytes"]:
            failures.append(
                f"size: {row['path']} expected {row['size_bytes']} got {actual_size}"
            )
        actual_hash = sha256(path)
        if actual_hash != row["sha256"]:
            failures.append(f"sha256: {row['path']}")
    if failures:
        print("PACKAGE MANIFEST FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PACKAGE MANIFEST PASS: {len(payload.get('files', []))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
