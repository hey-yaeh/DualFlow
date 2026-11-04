from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

EPS = 1e-9


def apply_transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", m[:3, :3], p) + m[:3, 3]


def apply_transform_directions(matrix: np.ndarray, directions: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    d = np.einsum("ij,...j->...i", m[:3, :3], np.asarray(directions, dtype=np.float64))
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.maximum(n, EPS)


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def ray_aabb_intersection(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[bool, float, float]:
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    inv = np.empty_like(d)
    mask = np.abs(d) > EPS
    inv[mask] = 1.0 / d[mask]
    inv[~mask] = np.inf
    t1 = (lo - o) * inv
    t2 = (hi - o) * inv
    tmin = float(np.max(np.minimum(t1, t2)))
    tmax = float(np.min(np.maximum(t1, t2)))
    return bool(tmax >= max(tmin, 0.0)), tmin, tmax


@dataclass(frozen=True)
class RayHit:
    distance: float
    triangle_index: int
    point: np.ndarray
    barycentric: np.ndarray


def intersect_ray_triangles(
    origin: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    triangle_indices: np.ndarray | None = None,
    max_distance: float = np.inf,
) -> RayHit | None:
    """Nearest positive Möller–Trumbore hit; distances use vertex units."""
    v = np.asarray(vertices, dtype=np.float64)
    f_all = np.asarray(faces, dtype=np.int64)
    if triangle_indices is None:
        tri_ids = np.arange(len(f_all), dtype=np.int64)
    else:
        tri_ids = np.asarray(triangle_indices, dtype=np.int64)
    if tri_ids.size == 0:
        return None
    f = f_all[tri_ids]
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    d = np.asarray(direction, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    pvec = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    valid = np.abs(det) > EPS
    if not np.any(valid):
        return None
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    tvec = o - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    valid &= (u >= -EPS) & (u <= 1.0 + EPS)
    qvec = np.cross(tvec, e1)
    vv = np.einsum("ij,j->i", qvec, d) * inv_det
    valid &= (vv >= -EPS) & ((u + vv) <= 1.0 + EPS)
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    valid &= (t > EPS) & (t <= max_distance)
    if not np.any(valid):
        return None
    candidates = np.where(valid)[0]
    local = candidates[int(np.argmin(t[candidates]))]
    dist = float(t[local])
    point = o + dist * d
    bary = np.array([1.0 - u[local] - vv[local], u[local], vv[local]], dtype=np.float64)
    return RayHit(dist, int(tri_ids[local]), point, bary)


def closest_ray_vertex_angle(origin: np.ndarray, direction: np.ndarray, points: np.ndarray) -> tuple[float, float, int] | None:
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        return None
    rel = p - np.asarray(origin, dtype=np.float64)
    along = rel @ np.asarray(direction, dtype=np.float64)
    valid = along > 0
    if not np.any(valid):
        return None
    rel = rel[valid]
    along_v = along[valid]
    norms = np.linalg.norm(rel, axis=1)
    cos = np.clip(along_v / np.maximum(norms, EPS), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    local = int(np.argmin(angles))
    original = int(np.where(valid)[0][local])
    perpendicular = float(np.sqrt(max(norms[local] ** 2 - along_v[local] ** 2, 0.0)))
    return float(angles[local]), perpendicular, original


def nearest_timestamp_index(times_ns: np.ndarray, value_ns: int, max_error_ns: int | None = None) -> tuple[int, int] | None:
    times = np.asarray(times_ns, dtype=np.int64)
    if times.size == 0:
        return None
    pos = int(np.searchsorted(times, int(value_ns)))
    candidates = []
    if pos < len(times): candidates.append(pos)
    if pos > 0: candidates.append(pos - 1)
    idx = min(candidates, key=lambda i: (abs(int(times[i]) - int(value_ns)), i))
    error = abs(int(times[idx]) - int(value_ns))
    if max_error_ns is not None and error > max_error_ns:
        return None
    return idx, error


def topology_sha256(faces: np.ndarray) -> str:
    import hashlib
    arr = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()
