from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import numpy as np

class NPZContractError(RuntimeError):
    pass


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    try:
        with np.load(p, allow_pickle=False) as z:
            return {k: np.array(z[k]) for k in z.files}
    except ValueError as exc:
        raise NPZContractError(
            f"{p.name} contains object/pickle arrays. MotionStudio interchange must use numeric or Unicode arrays: {exc}"
        ) from exc


def first(data: dict[str, np.ndarray], names: Iterable[str], default: Any = None):
    lower = {k.lower(): k for k in data}
    for n in names:
        if n in data:
            return data[n]
        k = lower.get(n.lower())
        if k is not None:
            return data[k]
    return default


def scalar_text(value: Any, default: str = "") -> str:
    if value is None: return default
    arr = np.asarray(value)
    if arr.size == 0: return default
    x = arr.reshape(-1)[0]
    if isinstance(x, bytes): return x.decode("utf-8", errors="replace")
    return str(x)


def string_list(value: Any) -> list[str]:
    if value is None: return []
    arr = np.asarray(value)
    result=[]
    for x in arr.reshape(-1):
        if isinstance(x, bytes): result.append(x.decode("utf-8", errors="replace"))
        else: result.append(str(x))
    return result
