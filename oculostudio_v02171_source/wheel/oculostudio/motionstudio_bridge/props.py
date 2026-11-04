from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import numpy as np

from .contracts import SceneBundle, SceneBundleError
from .npzio import load_npz, first, string_list, scalar_text


@dataclass
class PropTrack:
    object_id: str
    category: str
    times_ns: np.ndarray
    transforms_world: np.ndarray
    collision_vertices: np.ndarray
    collision_faces: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    rigid_fit_rmse_mm: np.ndarray | None
    visible_marker_count: np.ndarray | None
    collision_face_part_ids: np.ndarray | None = None
    collision_part_names: list[str] | None = None
    key_prefix: str | None = None


def _times_ns(d: dict[str, np.ndarray], n: int) -> np.ndarray:
    x = first(d, ("global_time_ns", "source_timestamps_ns", "timestamps_ns"))
    if x is not None:
        out = np.asarray(x, dtype=np.int64).reshape(-1)
    else:
        x = first(d, ("source_timestamps_s", "timestamps_s", "output_timestamps_s"))
        if x is not None:
            out = np.rint(np.asarray(x, dtype=np.float64) * 1e9).astype(np.int64).reshape(-1)
        else:
            fps = float(np.asarray(first(d, ("source_fps", "fps"), np.array([0.0]))).reshape(-1)[0])
            if fps <= 0:
                raise SceneBundleError("Props package has no timestamps or valid FPS")
            out = np.rint(np.arange(n, dtype=np.float64) / fps * 1e9).astype(np.int64)
    if len(out) != n:
        raise SceneBundleError(f"Props timebase has {len(out)} rows but pose track has {n} frames")
    return out


def _load_manifest(bundle: SceneBundle, d: dict[str, np.ndarray]) -> dict[str, Any]:
    p = bundle.linked_files.get("props_manifest")
    if p:
        try:
            value = json.loads(p.read_text("utf-8"))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    embedded = first(d, ("props_manifest_json",))
    if embedded is not None:
        try:
            value = json.loads(scalar_text(embedded, "{}"))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


def _manifest_props(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("props") or manifest.get("objects") or []
    if isinstance(raw, dict):
        return [dict(v, prop_id=k) if isinstance(v, dict) else {"prop_id": k} for k, v in raw.items()]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _json_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        obj = json.loads(scalar_text(value, "[]"))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return string_list(value)


def _scalar(d: dict[str, np.ndarray], key: str, default: str = "") -> str:
    return scalar_text(d.get(key), default)


def _indexed_prefixes(d: dict[str, np.ndarray]) -> list[str]:
    result = set()
    pattern = re.compile(r"^(prop_\d{4})_prop_instance_id$")
    for key in d:
        match = pattern.match(key)
        if match:
            result.add(match.group(1))
    return sorted(result)


def _vector(d: dict[str, np.ndarray], key: str, n: int, dtype, default) -> np.ndarray:
    if key not in d:
        return np.full(n, default, dtype=dtype)
    out = np.asarray(d[key], dtype=dtype).reshape(-1)
    if len(out) != n:
        raise SceneBundleError(f"{key} has {len(out)} values; expected {n}")
    return out


def _load_indexed_contract(d: dict[str, np.ndarray], manifest: dict[str, Any]) -> list[PropTrack]:
    prefixes = _indexed_prefixes(d)
    if not prefixes:
        return []
    manifest_items = _manifest_props(manifest)
    by_prefix = {str(x.get("key_prefix")): x for x in manifest_items if x.get("key_prefix")}
    by_instance = {str(x.get("prop_instance_id")): x for x in manifest_items if x.get("prop_instance_id")}
    shared_times = None
    tracks: list[PropTrack] = []

    for prefix in prefixes:
        object_id = _scalar(d, f"{prefix}_prop_instance_id") or _scalar(d, f"{prefix}_prop_id") or prefix
        item = by_prefix.get(prefix) or by_instance.get(object_id) or {}
        category = str(item.get("category") or _scalar(d, f"{prefix}_category", "object"))
        transform_key = str((item.get("transform_keys") or {}).get("geometry_local_to_world") or f"{prefix}_geometry_transforms_world")
        if transform_key not in d:
            transform_key = f"{prefix}_transforms_world"
        if transform_key not in d:
            raise SceneBundleError(f"Prop {object_id} has no geometry/world transform track")
        transforms = np.asarray(d[transform_key], dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise SceneBundleError(f"Prop {object_id} transforms must be [T,4,4]")
        n = transforms.shape[0]
        if shared_times is None:
            shared_times = _times_ns(d, n)
        elif len(shared_times) != n:
            raise SceneBundleError("Indexed prop tracks do not share a common frame count")

        vertices_key = f"{prefix}_collision_vertices"
        faces_key = f"{prefix}_collision_faces"
        if vertices_key not in d or faces_key not in d:
            raise SceneBundleError(f"Prop {object_id} lacks collision vertices/faces")
        vertices = np.asarray(d[vertices_key], dtype=np.float64)
        faces = np.asarray(d[faces_key], dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise SceneBundleError(f"Prop {object_id} collision vertices must be [V,3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise SceneBundleError(f"Prop {object_id} collision faces must be [F,3]")

        part_ids = d.get(f"{prefix}_collision_face_part_ids")
        if part_ids is not None:
            part_ids = np.asarray(part_ids, dtype=np.int64).reshape(-1)
            if len(part_ids) != len(faces):
                raise SceneBundleError(f"Prop {object_id} collision face-part labels do not match triangle count")
        part_names = item.get("collision_part_names") or item.get("proxy_part_names") or []
        part_names = [str(x) for x in part_names] if isinstance(part_names, list) else []

        tracks.append(
            PropTrack(
                object_id=object_id,
                category=category,
                times_ns=np.asarray(shared_times, dtype=np.int64),
                transforms_world=transforms,
                collision_vertices=vertices,
                collision_faces=faces,
                valid_mask=_vector(d, f"{prefix}_valid_mask", n, bool, True),
                confidence=_vector(d, f"{prefix}_confidence", n, np.float32, 1.0),
                rigid_fit_rmse_mm=_vector(d, f"{prefix}_rigid_fit_rmse_mm", n, np.float32, np.nan),
                visible_marker_count=_vector(d, f"{prefix}_visible_marker_count", n, np.int32, 0),
                collision_face_part_ids=part_ids,
                collision_part_names=part_names,
                key_prefix=prefix,
            )
        )
    return tracks


def _slice_object(arr: np.ndarray, index: int, count: int):
    a = np.asarray(arr)
    if count == 1:
        return a
    if a.ndim > 0 and a.shape[0] == count:
        return a[index]
    return a


def _load_aggregate_contract(d: dict[str, np.ndarray], manifest: dict[str, Any]) -> list[PropTrack]:
    ids = _json_string_list(first(d, ("prop_instance_ids_json", "prop_instance_ids", "prop_instance_id", "object_ids", "object_id", "prop_ids", "prop_id")))
    items = _manifest_props(manifest)
    if not ids:
        ids = [str(x.get("prop_instance_id") or x.get("object_id") or x.get("prop_id") or f"prop_{i}") for i, x in enumerate(items)]
    transforms = first(d, ("geometry_transforms_world", "transforms_world", "object_transforms_world", "poses_world"))
    verts = first(d, ("collision_vertices", "collision_vertices_local", "vertices_local"))
    faces = first(d, ("collision_faces", "faces", "triangles"))
    if transforms is None or verts is None or faces is None:
        raise SceneBundleError("Props package lacks indexed prop_#### fields or aggregate geometry fields")
    transforms = np.asarray(transforms, dtype=np.float64)
    count = len(ids) if ids else (transforms.shape[0] if transforms.ndim == 4 and transforms.shape[-2:] == (4, 4) else 1)
    if not ids:
        ids = [f"prop_{i+1}" for i in range(count)]
    categories = string_list(first(d, ("categories", "category", "object_categories")))
    if not categories and items:
        categories = [str(x.get("category") or "object") for x in items]
    valid = first(d, ("valid_mask", "pose_valid"))
    conf = first(d, ("confidence", "pose_confidence"))
    rmse = first(d, ("rigid_fit_rmse_mm", "pose_rms_mm"))
    visible = first(d, ("visible_marker_count",))
    tracks = []
    for i, obj_id in enumerate(ids):
        ta = np.asarray(transforms)
        t = ta[i] if ta.ndim == 4 and ta.shape[0] == count else ta
        if t.ndim != 3 or t.shape[1:] != (4, 4):
            raise SceneBundleError(f"Prop {obj_id} transforms must be [T,4,4]")
        n = t.shape[0]
        times = _times_ns(d, n)
        va = np.asarray(verts)
        fa = np.asarray(faces)
        v = np.asarray(va[i] if va.ndim == 3 and va.shape[0] == count else va, dtype=np.float64)
        f = np.asarray(fa[i] if fa.ndim == 3 and fa.shape[0] == count else fa, dtype=np.int64)

        def per_object(value, dtype, default=None):
            if value is None:
                return None if default is None else np.full(n, default, dtype=dtype)
            a = np.asarray(value)
            if a.ndim >= 2 and a.shape[0] == count:
                a = a[i]
            return np.asarray(a, dtype=dtype).reshape(-1)

        tracks.append(
            PropTrack(
                object_id=obj_id,
                category=categories[i] if i < len(categories) else "object",
                times_ns=times,
                transforms_world=t,
                collision_vertices=v,
                collision_faces=f,
                valid_mask=per_object(valid, bool, True),
                confidence=per_object(conf, np.float32, 1.0),
                rigid_fit_rmse_mm=per_object(rmse, np.float32),
                visible_marker_count=per_object(visible, np.int32),
            )
        )
    return tracks


def load_props(bundle: SceneBundle) -> list[PropTrack]:
    path = bundle.linked_files.get("props_motion")
    if path is None:
        return []
    data = load_npz(path)
    manifest = _load_manifest(bundle, data)

    npz_fp = scalar_text(first(data, ("package_fingerprint_sha256",)), "")
    manifest_fp = str(manifest.get("package_fingerprint_sha256") or "")
    if npz_fp and manifest_fp and npz_fp != manifest_fp:
        raise SceneBundleError("Props NPZ and manifest fingerprints disagree")
    if bundle.props_fingerprint and (npz_fp or manifest_fp) and bundle.props_fingerprint != (npz_fp or manifest_fp):
        raise SceneBundleError("Scene catalog Props fingerprint does not match loaded Props package")

    indexed = _load_indexed_contract(data, manifest)
    if indexed:
        return indexed
    return _load_aggregate_contract(data, manifest)
