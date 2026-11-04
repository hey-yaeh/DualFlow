from __future__ import annotations

import sys
from collections.abc import Sequence

SUPPORTED = {(3, 10), (3, 11), (3, 12), (3, 13)}


def is_supported(version: Sequence[int]) -> bool:
    return tuple(version[:2]) in SUPPORTED


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--print" in args:
        print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return 0 if is_supported(sys.version_info) else 1


if __name__ == "__main__":
    raise SystemExit(main())
