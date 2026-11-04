from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import re

import numpy as np

SUPPORTED_SCENE_SCHEMAS = {
    "motionstudio.oculostudio_scene.v3",
    "motionstudio.oculostudio_scene.v2",
    "motionstudio.oculostudio_scene.v1",
}


class SceneBundleError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _walk(obj: Any, prefix: tuple[str, ...] = ()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = prefix + (str(k),)
            yield p, v
            yield from _walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = prefix + (str(i),)
            yield p, v
            yield from _walk(v, p)


def _find_value(data: Any, names: Iterable[str], default=None):
    wanted = {n.lower() for n in names}
    for path, value in _walk(data):
        if path and path[-1].lower() in wanted and value not in (None, ""):
            return value
    return default


def _find_bool(data: Any, fragments: Iterable[str]) -> bool | None:
    fs = tuple(f.lower() for f in fragments)
    for path, value in _walk(data):
        key = ".".join(path).lower()
        if all(f in key for f in fs) and isinstance(value, bool):
            return value
    return None


def _resolve_path(manifest_path: Path, raw: str) -> Path | None:
    if not raw or not isinstance(raw, str):
        return None
    if "://" in raw or raw.startswith("urn:"):
        return None
    p = Path(raw.replace("\\", "/"))
    candidates = [p] if p.is_absolute() else [manifest_path.parent / p]
    if not p.is_absolute():
        # Older MotionStudio fragments sometimes wrote interaction-root-relative
        # paths while the manifest lived under body_motion/Rxx.
        for parent in list(manifest_path.parents)[:8]:
            candidates.append(parent / p)
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _classify_file(path: Path | str) -> str | None:
    p = Path(path)
    n = p.name.lower()
    if n.endswith("_oculostudio_scene.json") or "oculostudio_scene" in n:
        return "scene_manifest"
    if "body_motion" in n and p.suffix.lower() == ".npz":
        return "body_motion"
    if "bodyparts" in n and p.suffix.lower() == ".npz":
        return "bodyparts_npz"
    if "bodyparts" in n and p.suffix.lower() == ".json":
        return "bodyparts_json"
    if "fitting_report" in n and p.suffix.lower() == ".json":
        return "fitting_report"
    if "marker_review" in n and p.suffix.lower() == ".csv":
        return "marker_review_csv"
    if "marker_qc" in n and p.suffix.lower() == ".npz":
        return "marker_qc_npz"
    if "marker_qc" in n and p.suffix.lower() == ".json":
        return "marker_qc_json"
    if "marker_qc_summary" in n and p.suffix.lower() == ".csv":
        return "marker_qc_summary"
    if "props" in n and p.suffix.lower() == ".npz":
        return "props_motion"
    if "props" in n and p.suffix.lower() == ".json":
        return "props_manifest"
    if "vertices" in n and p.suffix.lower() in {".npz", ".npy"}:
        return "vertices_cache"
    if p.suffix.lower() == ".pkl":
        return "pickle_compat"
    return None


def _discover_linked_files(manifest_path: Path, data: dict[str, Any]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for _, value in _walk(data):
        if not isinstance(value, str):
            continue
        if not re.search(r"\.(npz|npy|json|csv|parquet|pkl)$", value, re.I):
            continue
        p = _resolve_path(manifest_path, value)
        if p is None:
            continue
        kind = _classify_file(p)
        if kind and kind != "scene_manifest":
            found.setdefault(kind, p)
    # Actor-local compatibility fallback. Shared props are found through the
    # explicit ../../props paths, not by an unbounded filename search.
    for p in manifest_path.parent.iterdir():
        if not p.is_file():
            continue
        kind = _classify_file(p)
        if kind and kind != "scene_manifest":
            found.setdefault(kind, p.resolve())
    return found


def _collect_declared_checksums(data: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    # MotionStudio v2/v3 stores a direct path -> SHA mapping.
    direct = data.get("checksums_sha256")
    if isinstance(direct, dict):
        for raw_path, value in direct.items():
            if isinstance(raw_path, str) and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                result[raw_path] = value.lower()
    # Also support structured file entries used by synthetic/newer contracts.
    for _, value in _walk(data):
        if not isinstance(value, dict):
            continue
        sha = None
        raw_path = None
        for k, v in value.items():
            kl = str(k).lower()
            if "sha256" in kl and isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v):
                sha = v.lower()
            if (kl in {"path", "file", "filename", "relative_path"} or kl.endswith("_path")) and isinstance(v, str):
                raw_path = v
        if sha and raw_path:
            result.setdefault(raw_path, sha)
    return result


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _read_npz_scalar(path: Path | None, key: str) -> str | None:
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if key not in z.files:
                return None
            arr = np.asarray(z[key])
            if arr.size == 0:
                return None
            value = arr.reshape(-1)[0]
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)
    except Exception:
        return None


def _npz_has_embedded_bodyparts_contract(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            return {
                "faces",
                "face_part_ids",
            }.issubset(keys) and bool({"part_names_json", "manifest_json", "part_names"} & keys)
    except Exception:
        return False


def _matrix4_from_rotation(value: Any) -> list[list[float]] | None:
    try:
        a = [[float(x) for x in row] for row in value]
    except Exception:
        return None
    if len(a) != 3 or any(len(row) != 3 for row in a):
        return None
    return [
        [a[0][0], a[0][1], a[0][2], 0.0],
        [a[1][0], a[1][1], a[1][2], 0.0],
        [a[2][0], a[2][1], a[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _extract_matrix(data: Any) -> tuple[list[list[float]] | None, str | None]:
    entries = list(_walk(data))
    matrix_keys = (
        "motionstudio_world_from_source_vicon_world",
        "result_world_from_source_world",
        "world_from_source_matrix",
        "source_to_result_transform",
        "coordinate_transform",
    )
    for key in matrix_keys:
        for path, value in entries:
            if path and path[-1].lower() == key:
                try:
                    m = [[float(x) for x in row] for row in value]
                    if len(m) == 4 and all(len(r) == 4 for r in m):
                        return m, ".".join(path)
                except Exception:
                    pass
    # This key is emitted by the real MotionStudio Props package and maps the
    # original Vicon z-up axes into motionstudio_world. OculoStudio converts ray
    # positions to metres before applying this matrix, so the matrix is rotation
    # only; do not apply the source mm scale a second time.
    rotation_keys = (
        "input_to_result_world_3x3",
        "motionstudio_world_from_source_vicon_world_rotation",
        "result_world_from_source_world_rotation",
        "source_to_result_rotation",
        "source_to_result_basis",
    )
    for key in rotation_keys:
        for path, value in entries:
            if path and path[-1].lower() == key:
                matrix = _matrix4_from_rotation(value)
                if matrix is not None:
                    return matrix, ".".join(path)
    return None, None


def _authoritative_props_contract(linked: dict[str, Path]) -> tuple[str | None, Any, list[str], dict[str, Any]]:
    notes: list[str] = []
    manifest = _read_json(linked.get("props_manifest"))
    manifest_fp = manifest.get("package_fingerprint_sha256") if isinstance(manifest, dict) else None
    manifest_revision = manifest.get("package_revision") if isinstance(manifest, dict) else None
    npz_fp = _read_npz_scalar(linked.get("props_motion"), "package_fingerprint_sha256")
    if manifest_fp and npz_fp and str(manifest_fp) != str(npz_fp):
        raise SceneBundleError(
            "MotionStudio Props manifest and NPZ disagree on package_fingerprint_sha256"
        )
    fp = str(manifest_fp or npz_fp) if (manifest_fp or npz_fp) else None
    if manifest_fp and npz_fp:
        notes.append("Shared Props manifest and NPZ package fingerprints agree")
    return fp, manifest_revision, notes, manifest


@dataclass
class SceneBundle:
    manifest_path: Path
    schema_version: str
    scene_id: str
    session_id: str
    interaction_id: str
    role_id: str
    participant_id: str
    actor_instance_id: str
    linked_files: dict[str, Path] = field(default_factory=dict)
    declared_checksums: dict[str, str] = field(default_factory=dict)
    checksum_results: dict[str, bool] = field(default_factory=dict)
    checksum_failures: list[str] = field(default_factory=list)
    stale_shared_props_checksums: list[str] = field(default_factory=list)
    complete_for_mesh_aoi: bool = False
    props_fingerprint: str | None = None
    props_revision: str | int | None = None
    props_package_sha256: str | None = None
    props_manifest_sha256: str | None = None
    coordinate_transform: list[list[float]] | None = None
    coordinate_transform_source: str | None = None
    source_frame_mapping_exact: bool | None = None
    compatibility_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["manifest_path"] = str(self.manifest_path)
        d["linked_files"] = {k: str(v) for k, v in self.linked_files.items()}
        d.pop("raw", None)
        return d


def inspect_scene_bundle(path: str | Path, *, verify_checksums: bool = True) -> SceneBundle:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise SceneBundleError(f"Scene manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text("utf-8"))
    except Exception as exc:
        raise SceneBundleError(f"Cannot parse scene manifest JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SceneBundleError("Scene manifest must contain a JSON object")

    schema = str(
        _find_value(
            data,
            ("format_version", "schema_version", "contract_version", "schema"),
            "",
        )
    )
    if not schema:
        raise SceneBundleError("Scene manifest has no format_version/schema_version")
    if "motionstudio" not in schema.lower() or "oculostudio_scene" not in schema.lower():
        raise SceneBundleError(f"Not a MotionStudio OculoStudio scene manifest: {schema}")
    forward_warning = None
    if schema not in SUPPORTED_SCENE_SCHEMAS:
        forward_warning = f"Unverified scene schema {schema}; supported: {sorted(SUPPORTED_SCENE_SCHEMAS)}"

    session_id = str(_find_value(data, ("session_id", "session"), ""))
    interaction_id = str(_find_value(data, ("interaction_id", "interaction"), ""))
    role_id = str(_find_value(data, ("role_id", "role"), ""))
    participant_id = str(_find_value(data, ("participant_id", "participant"), ""))
    actor_instance_id = str(_find_value(data, ("actor_instance_id", "actor_id", "entity_id"), ""))
    scene_id = str(_find_value(data, ("scene_id", "merge_key"), "")) or "_".join(
        x for x in (session_id, interaction_id) if x
    )

    linked = _discover_linked_files(manifest_path, data)
    declared = _collect_declared_checksums(data)
    checksum_results: dict[str, bool] = {}
    checksum_failures: list[str] = []
    stale_props: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    if forward_warning:
        warnings.append(forward_warning)

    required = ("body_motion", "bodyparts_npz")
    for key in required:
        if key not in linked:
            checksum_failures.append(f"Missing required linked file: {key}")
    if "bodyparts_json" not in linked:
        if _npz_has_embedded_bodyparts_contract(linked.get("bodyparts_npz")):
            notes.append("External bodyparts JSON is absent; using taxonomy/manifest embedded in bodyparts NPZ")
        else:
            warnings.append("Missing bodyparts JSON and no embedded taxonomy was found")

    props_fp = None
    props_rev = None
    props_manifest_data: dict[str, Any] = {}
    if linked.get("props_motion") or linked.get("props_manifest"):
        props_fp, props_rev, prop_notes, props_manifest_data = _authoritative_props_contract(linked)
        notes.extend(prop_notes)

    critical_kinds = {"body_motion", "bodyparts_npz"}
    optional_kinds = {
        "bodyparts_json",
        "fitting_report",
        "marker_review_csv",
        "marker_qc_npz",
        "marker_qc_json",
        "marker_qc_summary",
        "pickle_compat",
    }
    props_kinds = {"props_motion", "props_manifest"}

    for raw_path, expected in declared.items():
        resolved = _resolve_path(manifest_path, raw_path)
        kind = _classify_file(raw_path)
        if resolved is None:
            checksum_results[raw_path] = False
            if kind in critical_kinds:
                checksum_failures.append(f"Declared required file missing: {raw_path}")
            elif kind == "bodyparts_json" and _npz_has_embedded_bodyparts_contract(linked.get("bodyparts_npz")):
                warnings.append(f"Declared bodyparts JSON missing; embedded NPZ contract used: {raw_path}")
            else:
                warnings.append(f"Optional declared file not present in this import: {raw_path}")
            continue
        if not verify_checksums:
            continue
        actual = sha256_file(resolved)
        ok = actual == expected
        checksum_results[raw_path] = ok
        if ok:
            continue
        if kind in props_kinds:
            # v0.7.19 selective props merge can update the shared interaction
            # package after an earlier actor fragment was written. The current
            # Props manifest/NPZ fingerprint is authoritative; retain the stale
            # actor checksum as an auditable warning instead of rejecting valid
            # R01/R02 scene fragments.
            stale_props.append(raw_path)
            warnings.append(
                f"Actor scene fragment has a stale shared-Props checksum: {raw_path}; "
                "the current Props package fingerprint will be used"
            )
        elif kind in critical_kinds:
            checksum_failures.append(f"SHA-256 mismatch for required file: {raw_path}")
        elif kind in optional_kinds:
            warnings.append(f"SHA-256 mismatch for optional QC/compatibility file: {raw_path}")
        else:
            checksum_failures.append(f"SHA-256 mismatch: {raw_path}")

    complete = _find_bool(data, ("complete", "oculostudio", "mesh", "aoi"))
    if complete is None:
        complete = all(k in linked for k in required)
    source_exact = _find_bool(data, ("source", "mapping", "exact"))

    transform, transform_source = _extract_matrix(data)
    if transform is None and props_manifest_data:
        transform, transform_source = _extract_matrix(props_manifest_data)
        if transform is not None:
            transform_source = f"props_manifest:{transform_source}"
            notes.append("Source Vicon → MotionStudio world rotation recovered from authoritative Props manifest")
    if transform is None and linked.get("props_motion"):
        raw = _read_npz_scalar(linked.get("props_motion"), "coordinate_system_json")
        if raw:
            try:
                transform, transform_source = _extract_matrix(json.loads(raw))
                if transform is not None:
                    transform_source = f"props_npz.coordinate_system_json:{transform_source}"
            except Exception:
                pass

    if not session_id or not interaction_id or not role_id or not actor_instance_id:
        checksum_failures.append("Incomplete actor identity fields (session/interaction/role/actor_instance)")
    if source_exact is False:
        checksum_failures.append("MotionStudio source-frame mapping is not exact")
    if transform is None:
        warnings.append(
            "No explicit source-Vicon → MotionStudio-world transform was found; "
            "mesh AOI requires an explicit user-confirmed world assumption"
        )
    if checksum_failures:
        complete = False
    if schema not in SUPPORTED_SCENE_SCHEMAS:
        complete = False

    props_motion = linked.get("props_motion")
    props_manifest = linked.get("props_manifest")
    return SceneBundle(
        manifest_path=manifest_path,
        schema_version=schema,
        scene_id=scene_id,
        session_id=session_id,
        interaction_id=interaction_id,
        role_id=role_id,
        participant_id=participant_id,
        actor_instance_id=actor_instance_id,
        linked_files=linked,
        declared_checksums=declared,
        checksum_results=checksum_results,
        checksum_failures=checksum_failures,
        stale_shared_props_checksums=stale_props,
        complete_for_mesh_aoi=bool(complete),
        props_fingerprint=props_fp,
        props_revision=props_rev,
        props_package_sha256=sha256_file(props_motion) if props_motion else None,
        props_manifest_sha256=sha256_file(props_manifest) if props_manifest else None,
        coordinate_transform=transform,
        coordinate_transform_source=transform_source,
        source_frame_mapping_exact=source_exact,
        compatibility_notes=notes,
        warnings=warnings + checksum_failures,
        raw=data,
    )


def validate_scene_pair(bundles: Iterable[SceneBundle]) -> dict[str, Any]:
    items = list(bundles)
    if not items:
        raise SceneBundleError("No MotionStudio scene bundles were supplied")
    scene_ids = {x.scene_id for x in items}
    if len(scene_ids) != 1:
        raise SceneBundleError(f"Bundles belong to different scenes: {sorted(scene_ids)}")
    actor_ids = [x.actor_instance_id for x in items]
    if len(set(actor_ids)) != len(actor_ids):
        raise SceneBundleError("Duplicate actor_instance_id in scene bundles")
    role_ids = [x.role_id for x in items]
    if len(set(role_ids)) != len(role_ids):
        raise SceneBundleError("Duplicate role_id in scene bundles")

    prop_fps = {x.props_fingerprint for x in items if x.props_fingerprint}
    if len(prop_fps) > 1:
        raise SceneBundleError("Actor manifests resolve to different authoritative shared-Props fingerprints")
    prop_shas = {x.props_package_sha256 for x in items if x.props_package_sha256}
    if len(prop_shas) > 1:
        raise SceneBundleError("Actor manifests resolve to different current shared-Props NPZ files")

    return {
        "scene_id": items[0].scene_id,
        "session_id": items[0].session_id,
        "interaction_id": items[0].interaction_id,
        "actors": [x.to_jsonable() for x in items],
        "shared_props_fingerprint": next(iter(prop_fps), None),
        "shared_props_sha256": next(iter(prop_shas), None),
        "ready": all(x.complete_for_mesh_aoi for x in items),
        "warnings": [w for x in items for w in x.warnings],
        "compatibility_notes": [n for x in items for n in x.compatibility_notes],
    }
