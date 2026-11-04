from __future__ import annotations

"""Project-level Calibration Set physical-profile builder.

The original Calibration page estimated a hardware profile from a role detail
view.  That coupled a project-level hardware task to the participant currently
wearing the device in one Interaction.  For shared Calibration Sets this is
wrong: a calibration actor (for example P03) may have worn ETM-A during the
AprilTag recording while a different participant (for example P02) wears
ETM-A in the Interaction being processed.

This module resolves the raw Calibration Set sources independently of the
current Interaction, matches each Pupil calibration recording to the physical
ETM tracker, builds the measured camera-to-tracker profile, and commits the
resulting ETM-A / ETM-B profiles to one Calibration Set revision.
"""

import json
from pathlib import Path
from typing import Any

from neon_vicon_gaze3d.io.c3d_reader import read_c3d as legacy_read_c3d
from neon_vicon_gaze3d.processing.tracker_identity import (
    TrackerIdentityError,
    detect_matching_tracker,
)

from oculostudio.domain.models import ArtifactKind, CalibrationSet
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.services.hardware_registration import (
    canonical_hardware_id,
    hardware_display_name,
    infer_hardware_identity,
)
from oculostudio.services.legacy_gaze3d_calibration import (
    create_legacy_gaze3d_calibration_profile,
)
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, utc_now

_CONFIDENCE_RANK = {"LOW": 0, "SINGLE_CANDIDATE": 1, "MEDIUM": 2, "HIGH": 3}
_REQUIRED_HARDWARE = ("ETM_A", "ETM_B")
_TAG_IDS = ("0", "1", "2", "3")
_CORNER_ORDER = ("TL", "TR", "BR", "BL")


def _set_by_id_or_code(project, value: str) -> CalibrationSet:
    token = str(value or "").strip()
    if token in project.calibration_sets:
        return project.calibration_sets[token]
    compact = token.casefold().replace("-", "_")
    for row in project.calibration_sets.values():
        if row.code.casefold().replace("-", "_") == compact:
            return row
    raise ValidationError(f"Unknown Calibration Set: {value}")


def _revision(row: CalibrationSet, revision_id: str | None = None):
    rid = revision_id or row.active_revision_id
    if not rid or rid not in row.revisions:
        raise ValidationError(f"Calibration Set {row.code} has no active revision")
    return row.revisions[rid]


def _asset_rows(store: ProjectStore, row: CalibrationSet, revision_id: str | None = None):
    project = store.snapshot()
    revision = _revision(row, revision_id)
    rows = []
    for ref in revision.source_assets:
        session = project.sessions.get(ref.source_session_id)
        asset = session.assets.get(ref.source_asset_id) if session else None
        if asset is None:
            rows.append({"ref": ref, "asset": None, "session_id": ref.source_session_id})
            continue
        rows.append({"ref": ref, "asset": asset, "session_id": ref.source_session_id})
    return rows


def _hardware_from_artifact(artifact) -> str | None:
    value = (
        artifact.parameters.get("hardware_id")
        or artifact.qc.get("hardware_id")
        or artifact.parameters.get("hardware_display_name")
        or artifact.qc.get("hardware_display_name")
    )
    return canonical_hardware_id(value)


def _existing_profiles(store: ProjectStore, row: CalibrationSet, revision_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
    project = store.snapshot()
    revision = _revision(row, revision_id)
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in _REQUIRED_HARDWARE}
    for ref in revision.profiles:
        session = project.sessions.get(ref.source_session_id)
        artifact = session.artifacts.get(ref.source_artifact_id) if session else None
        hardware = canonical_hardware_id(ref.hardware_id)
        if hardware is None and artifact is not None:
            hardware = _hardware_from_artifact(artifact)
        if hardware not in result:
            continue
        result[hardware].append({
            "source_session_id": ref.source_session_id,
            "artifact_id": ref.source_artifact_id,
            "label": ref.label or (artifact.label if artifact else ref.source_artifact_id),
            "hardware_id": hardware,
            "hardware_display_name": hardware_display_name(hardware),
            "mount_version": ref.mount_version,
            "available": bool(artifact and any(Path(item.path).exists() for item in artifact.files)),
            "created_at": artifact.created_at if artifact else ref.added_at,
        })
    for values in result.values():
        values.sort(key=lambda item: str(item.get("created_at") or ""))
    return result


def _recoverable_profiles(store: ProjectStore, row: CalibrationSet, revision_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return profile artifacts created for this Set but not committed to the revision.

    Older role-detail jobs could successfully write a measured profile and then
    fail while registering it to the shared Calibration Set.  Such artifacts
    retain the Set identity in their parameters.  They are safe to recover only
    when the Set identity and physical hardware are explicit; arbitrary legacy
    profiles are never adopted automatically.
    """
    project = store.snapshot()
    revision = _revision(row, revision_id)
    committed = {(ref.source_session_id, ref.source_artifact_id) for ref in revision.profiles}
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in _REQUIRED_HARDWARE}
    for session_id, session in project.sessions.items():
        for artifact in session.artifacts.values():
            if str(artifact.kind) != ArtifactKind.CALIBRATION_PROFILE.value:
                continue
            if (session_id, artifact.id) in committed:
                continue
            if str(artifact.parameters.get("calibration_set_profile_build_status") or "") == "failed_uncommitted":
                continue
            set_code = str(artifact.parameters.get("calibration_set_id") or artifact.qc.get("calibration_set_id") or "")
            set_record = str(artifact.parameters.get("calibration_set_record_id") or artifact.qc.get("calibration_set_record_id") or "")
            if set_code != row.code and set_record != row.id:
                continue
            artifact_revision = str(artifact.parameters.get("calibration_revision_id") or artifact.qc.get("calibration_revision_id") or "")
            if artifact_revision and artifact_revision != revision.id:
                continue
            hardware = _hardware_from_artifact(artifact)
            if hardware not in result:
                continue
            try:
                available = any(Path(item.path).exists() for item in artifact.files)
            except Exception:
                available = False
            result[hardware].append({
                "source_session_id": session_id,
                "artifact_id": artifact.id,
                "label": artifact.label,
                "hardware_id": hardware,
                "hardware_display_name": hardware_display_name(hardware),
                "available": available,
                "created_at": artifact.created_at,
                "build_status": artifact.parameters.get("calibration_set_profile_build_status"),
            })
    for rows in result.values():
        rows.sort(key=lambda item: str(item.get("created_at") or ""))
    return result


def _source_public(row: dict[str, Any]) -> dict[str, Any]:
    ref, asset = row["ref"], row["asset"]
    return {
        "asset_id": ref.source_asset_id,
        "source_session_id": ref.source_session_id,
        "label": ref.label or (asset.label if asset else ref.source_asset_id),
        "path": ref.source_path or (asset.source_path if asset else ""),
        "kind": ref.kind or (str(asset.kind) if asset else "unknown"),
        "calibration_role": ref.calibration_role,
        "participant_id": ref.participant_id or (asset.participant_id if asset else None),
        "hardware_hint": ref.hardware_hint or ((asset.metadata or {}).get("hardware_hint") if asset else None),
        "available": bool(asset and Path(asset.source_path).exists()),
    }


def _normalise_board_definition(payload: dict[str, Any]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Return one strict 4-tag / 16-corner mapping.

    Calibration Set sources may use either the current explicit field names or
    the older ``tag_corners_mm`` wrapper used by pilot tools.  Only label-based
    mappings are accepted here; numeric board coordinates remain useful QC
    evidence but cannot identify the corresponding Vicon trajectories.
    """

    tags_raw = payload.get("board_corner_labels") or payload.get("tag_corner_labels")
    outer_raw = payload.get("outer_corner_labels") or payload.get("outer_corners")
    if not isinstance(tags_raw, dict) or not isinstance(outer_raw, dict):
        raise ValidationError(
            "Board definition must contain board_corner_labels and outer_corner_labels"
        )
    tags: dict[str, dict[str, str]] = {}
    for marker_id in _TAG_IDS:
        row = tags_raw.get(marker_id)
        if row is None:
            row = tags_raw.get(int(marker_id))
        if not isinstance(row, dict):
            raise ValidationError(f"Board definition is missing AprilTag {marker_id}")
        tags[marker_id] = {}
        for corner in _CORNER_ORDER:
            value = str(row.get(corner) or "").strip()
            if not value:
                raise ValidationError(
                    f"Board definition is missing AprilTag {marker_id} {corner}"
                )
            tags[marker_id][corner] = value
    outer: dict[str, str] = {}
    for corner in _CORNER_ORDER:
        value = str(outer_raw.get(corner) or "").strip()
        if not value:
            raise ValidationError(f"Board definition is missing outer corner {corner}")
        outer[corner] = value
    tag_labels = [tags[tag][corner] for tag in _TAG_IDS for corner in _CORNER_ORDER]
    if len(set(tag_labels)) != 16:
        raise ValidationError(
            "Board definition must map the four AprilTags to 16 distinct C3D marker labels"
        )
    if any(value not in set(tag_labels) for value in outer.values()):
        raise ValidationError(
            "Every outer-corner label must also be one of the 16 AprilTag corner labels"
        )
    return tags, outer


def _board_signature(tags: dict[str, dict[str, str]], outer: dict[str, str]) -> tuple[str, ...]:
    return tuple(tags[tag][corner] for tag in _TAG_IDS for corner in _CORNER_ORDER) + tuple(
        outer[corner] for corner in _CORNER_ORDER
    )


def _validate_board_definition_against_c3d(
    *,
    board_path: Path,
    tags: dict[str, dict[str, str]],
    outer: dict[str, str],
) -> dict[str, Any]:
    board = legacy_read_c3d(board_path)
    labels = set(str(label) for label in board.labels)
    required = [tags[tag][corner] for tag in _TAG_IDS for corner in _CORNER_ORDER]
    required.extend(outer[corner] for corner in _CORNER_ORDER)
    missing = sorted({label for label in required if label not in labels})
    if missing:
        raise ValidationError(
            "Board definition refers to marker labels that are absent from the selected board C3D: "
            + ", ".join(missing)
        )
    if int(board.point_count) != 16:
        raise ValidationError(
            f"Automatic four-AprilTag board mode requires 16 markers; {board_path.name} has {board.point_count}"
        )
    return {
        "board_filename": board_path.name,
        "point_count": int(board.point_count),
        "frame_count": int(board.frame_count),
        "frame_rate_hz": float(board.frame_rate_hz),
        "marker_labels": list(board.labels),
        "tag_corner_count": 16,
        "outer_corner_count": 4,
    }


def _board_definition_file(
    store: ProjectStore,
    row: CalibrationSet,
    revision_id: str,
) -> Path:
    root = store.calibration_revision_dir(row.id, revision_id) / "derived" / "board_geometry"
    root.mkdir(parents=True, exist_ok=True)
    return root / "board_definition.json"


def _read_board_definition_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read saved board definition {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"Saved board definition is not a JSON object: {path}")
    return payload


def _board_definition_from_source(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    if len(items) > 1:
        labels = ", ".join(_source_public(item)["label"] for item in items)
        raise ValidationError(
            "More than one board-definition source is registered. Keep one active definition: " + labels
        )
    path = Path(items[0]["asset"].source_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise ValidationError(f"Could not read board-definition source {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"Board-definition source is not a JSON object: {path.name}")
    payload = dict(payload)
    payload.setdefault("mapping_source", f"registered_source:{path.name}")
    payload.setdefault("source_path", str(path))
    return payload


def _persist_board_definition(
    store: ProjectStore,
    *,
    row: CalibrationSet,
    revision_id: str,
    payload: dict[str, Any],
) -> Path:
    path = _board_definition_file(store, row, revision_id)
    atomic_write_json(path, payload)

    def mutate(project):
        target = project.calibration_sets[row.id]
        by_revision = dict(target.metadata.get("board_geometry_by_revision") or {})
        by_revision[revision_id] = {
            "status": "ready",
            "definition_path": str(path),
            "mapping_source": payload.get("mapping_source"),
            "mapping_variant": payload.get("mapping_variant"),
            "resolved_at": payload.get("resolved_at") or utc_now(),
            "tag_corner_count": 16,
            "outer_corner_count": 4,
        }
        target.metadata["board_geometry_by_revision"] = by_revision
        target.modified_at = utc_now()

    store.update(mutate)
    return path


def _board_geometry_status(
    store: ProjectStore,
    *,
    row: CalibrationSet,
    revision_id: str,
    board_items: list[dict[str, Any]],
    definition_items: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False,
        "resolvable": False,
        "mapping_source": None,
        "mapping_variant": None,
        "definition_path": None,
        "tag_corner_count": 0,
        "outer_corner_count": 0,
        "problems": [],
        "warnings": [],
    }
    if len(board_items) != 1 or board_items[0].get("asset") is None:
        result["problems"].append("Exactly one available calibration-board C3D is required")
        return result
    board_path = Path(board_items[0]["asset"].source_path).expanduser().resolve()
    try:
        board = legacy_read_c3d(board_path)
        result.update({
            "board_filename": board_path.name,
            "board_point_count": int(board.point_count),
            "board_frame_count": int(board.frame_count),
            "board_frame_rate_hz": float(board.frame_rate_hz),
        })
        if int(board.point_count) != 16:
            result["problems"].append(
                f"The selected board C3D has {board.point_count} markers; automatic 4×4 inference requires 16"
            )
    except Exception as exc:
        result["problems"].append(f"Could not inspect the calibration-board C3D: {exc}")
        return result

    stored_path = _board_definition_file(store, row, revision_id)
    payload = _read_board_definition_file(stored_path)
    if payload is None and definition_items:
        payload = _board_definition_from_source(definition_items)
    if payload is not None:
        try:
            tags, outer = _normalise_board_definition(payload)
            metrics = _validate_board_definition_against_c3d(
                board_path=board_path, tags=tags, outer=outer
            )
            result.update({
                "ready": True,
                "resolvable": True,
                "mapping_source": payload.get("mapping_source") or "saved_definition",
                "mapping_variant": payload.get("mapping_variant"),
                "definition_path": str(stored_path if stored_path.exists() else payload.get("source_path") or ""),
                "tag_corner_count": 16,
                "outer_corner_count": 4,
                "metrics": metrics,
            })
            return result
        except Exception as exc:
            result["warnings"].append(f"A board definition is present but not usable: {exc}")
    result["resolvable"] = not result["problems"]
    return result


def calibration_set_profile_build_status(
    store: ProjectStore,
    *,
    calibration_set_id: str,
    revision_id: str | None = None,
) -> dict[str, Any]:
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    revision = _revision(row, revision_id)
    source_rows = _asset_rows(store, row, revision.id)
    by_role: dict[str, list[dict[str, Any]]] = {}
    for item in source_rows:
        by_role.setdefault(str(item["ref"].calibration_role or "calibration_source"), []).append(item)
    pupil = by_role.get("pupil_calibration", [])
    board = by_role.get("calibration_board", [])
    trackers = by_role.get("tracker_candidate", [])
    world = by_role.get("world_reference", [])
    definitions = by_role.get("board_definition", [])
    board_geometry = _board_geometry_status(
        store,
        row=row,
        revision_id=revision.id,
        board_items=board,
        definition_items=definitions,
    )
    profiles = _existing_profiles(store, row, revision.id)
    recoverable = _recoverable_profiles(store, row, revision.id)
    hardware_slots = {
        key: {
            "hardware_id": key,
            "hardware_display_name": hardware_display_name(key),
            "ready": bool(profiles.get(key)),
            "profiles": profiles.get(key) or [],
        }
        for key in _REQUIRED_HARDWARE
    }
    problems: list[str] = []
    if not pupil:
        problems.append("No Pupil Native AprilTag calibration recording is registered")
    if len(board) != 1:
        problems.append(
            "Exactly one calibration-board C3D is required" if not board else
            f"{len(board)} calibration-board C3Ds are registered; choose one explicitly"
        )
    if len(trackers) < 2:
        problems.append("Both ETM-A and ETM-B calibration tracker C3Ds are required")
    problems.extend(str(item) for item in board_geometry.get("problems") or [])
    unavailable = [_source_public(item)["label"] for item in source_rows if not _source_public(item)["available"]]
    if unavailable:
        problems.append("Unavailable calibration source paths: " + ", ".join(unavailable))
    missing_hardware = [key for key in _REQUIRED_HARDWARE if not hardware_slots[key]["ready"]]
    return {
        "calibration_set_id": row.id,
        "calibration_set_code": row.code,
        "calibration_set_label": row.label,
        "revision_id": revision.id,
        "revision_number": revision.revision_number,
        "revision_status": revision.status,
        "owner_session_id": row.owner_session_id,
        "applicable_session_ids": list(row.applicable_session_ids),
        "sources": [_source_public(item) for item in source_rows],
        "pupil_recordings": [_source_public(item) for item in pupil],
        "board_candidates": [_source_public(item) for item in board],
        "tracker_candidates": [_source_public(item) for item in trackers],
        "world_candidates": [_source_public(item) for item in world],
        "board_definitions": [_source_public(item) for item in definitions],
        "board_geometry": board_geometry,
        "hardware_slots": hardware_slots,
        "recoverable_profiles": recoverable,
        "recoverable_hardware": [key for key in missing_hardware if len(recoverable.get(key) or []) == 1 and (recoverable.get(key) or [])[0].get("available")],
        "missing_hardware": missing_hardware,
        "source_complete": not problems,
        "ready": not missing_hardware,
        "can_build": bool(not problems and missing_hardware),
        "problems": problems,
        "contract": {
            "profile_scope": "physical_hardware",
            "current_interaction_participant_required": False,
            "calibration_actor_may_differ_from_interaction_wearer": True,
            "identity_method": "AprilTag camera / world-head evidence selects the physical ETM tracker",
            "board_geometry_method": "Head-marker evidence selects hardware; scene AprilTags + planar 16-marker board resolve orientation",
        },
    }


def _confidence_ok(actual: str, required: str) -> bool:
    return _CONFIDENCE_RANK.get(str(actual).upper(), -1) >= _CONFIDENCE_RANK.get(str(required).upper(), 2)


def _preflight_pairing(
    *,
    native_asset,
    participant_id: str,
    tracker_items: list[dict[str, Any]],
    board_item: dict[str, Any],
    world_item: dict[str, Any] | None,
    sample_every_n_frames: int,
    minimum_confidence: str,
    progress=None,
) -> dict[str, Any]:
    logs: list[str] = []
    tracker_paths = [Path(item["asset"].source_path).resolve() for item in tracker_items]
    try:
        identity = detect_matching_tracker(
            pupil_native_source=Path(native_asset.source_path).resolve(),
            tracker_candidate_paths=tracker_paths,
            world_c3d_path=Path(world_item["asset"].source_path).resolve() if world_item else None,
            canonical_participant_id=participant_id,
            board_c3d_path=Path(board_item["asset"].source_path).resolve(),
            board_corner_labels={},
            outer_corner_labels={},
            sample_every_n_frames=max(1, int(sample_every_n_frames)),
            log=logs.append,
            progress=progress,
        )
    except TrackerIdentityError as exc:
        raise ProcessingError(
            f"Could not match calibration actor {participant_id or native_asset.label} to ETM-A/ETM-B: {exc}"
        ) from exc
    if not _confidence_ok(identity.confidence, minimum_confidence):
        raise ProcessingError(
            f"Calibration actor {participant_id or native_asset.label} matched {identity.selected_tracker_filename} "
            f"with {identity.confidence} confidence, below the required {minimum_confidence}."
        )
    selected_path = Path(identity.selected_path).resolve()
    selected_item = next(
        (item for item, path in zip(tracker_items, tracker_paths) if path == selected_path),
        None,
    )
    if selected_item is None:
        raise ProcessingError("The selected calibration tracker could not be mapped back to a Calibration Set asset")
    asset = selected_item["asset"]
    inferred = infer_hardware_identity(asset.label or Path(asset.source_path).name, identity.selected_tracker_labels)
    hardware = canonical_hardware_id((asset.metadata or {}).get("hardware_hint"))
    if inferred is not None:
        if hardware and hardware != inferred.hardware_id:
            raise ProcessingError(
                f"Tracker source {asset.label} has hardware hint {hardware_display_name(hardware)} but marker/filename "
                f"identity implies {inferred.display_name}. Correct the Calibration Set metadata before building profiles."
            )
        hardware = inferred.hardware_id
    if hardware not in _REQUIRED_HARDWARE:
        raise ProcessingError(
            f"The selected tracker {asset.label} could not be resolved to ETM-A or ETM-B. "
            "Rename it with eye_tracker_A / eye_tracker_B or edit its Calibration metadata."
        )
    return {
        "participant_id": participant_id,
        "native_asset_id": native_asset.id,
        "native_label": native_asset.label,
        "selected_tracker_asset_id": asset.id,
        "selected_tracker_label": asset.label,
        "hardware_id": hardware,
        "hardware_display_name": hardware_display_name(hardware),
        "confidence": identity.confidence,
        "identity": identity.to_dict(),
        "logs": logs,
    }


def _resolve_shared_board_definition(
    store: ProjectStore,
    *,
    row: CalibrationSet,
    revision_id: str,
    board_item: dict[str, Any],
    definition_items: list[dict[str, Any]],
    tracker_items: list[dict[str, Any]],
    preflight: list[dict[str, Any]],
    sample_every_n_frames: int,
    progress=None,
) -> dict[str, Any]:
    """Resolve one board mapping independently of physical-ETM identity.

    Labelled head markers are the strongest evidence for deciding which tracker
    belongs to the calibration actor.  They do *not* determine which side of a
    geometrically inferred 4×4 board is image top/left.  The old profile builder
    treated those tasks as one branch and therefore returned an empty mapping
    whenever head evidence was available.  This resolver deliberately combines
    the two evidence streams:

    * head/world evidence already stored in ``preflight`` selects ETM-A/B; and
    * the selected tracker plus scene AprilTags resolves the board orientation.
    """

    board_asset = board_item.get("asset")
    if board_asset is None:
        raise ValidationError("The calibration-board C3D is unavailable")
    board_path = Path(board_asset.source_path).expanduser().resolve()

    # A previously accepted revision definition is authoritative and avoids an
    # expensive repeat of scene-video AprilTag detection.
    stored_path = _board_definition_file(store, row, revision_id)
    stored = _read_board_definition_file(stored_path)
    if stored is not None:
        tags, outer = _normalise_board_definition(stored)
        metrics = _validate_board_definition_against_c3d(
            board_path=board_path, tags=tags, outer=outer
        )
        return {
            **stored,
            "board_corner_labels": tags,
            "outer_corner_labels": outer,
            "definition_path": str(stored_path),
            "metrics": metrics,
            "reused": True,
        }

    # An explicitly imported definition takes precedence over inference, but is
    # copied into the immutable revision-derived evidence location.
    explicit = _board_definition_from_source(definition_items)
    if explicit is not None:
        tags, outer = _normalise_board_definition(explicit)
        metrics = _validate_board_definition_against_c3d(
            board_path=board_path, tags=tags, outer=outer
        )
        payload = {
            "schema_version": "oculostudio-apriltag-board-definition-1.0",
            **explicit,
            "board_corner_labels": tags,
            "outer_corner_labels": outer,
            "resolved_at": utc_now(),
            "calibration_set_id": row.id,
            "calibration_set_code": row.code,
            "revision_id": revision_id,
            "metrics": metrics,
            "proposals": [],
        }
        path = _persist_board_definition(
            store, row=row, revision_id=revision_id, payload=payload
        )
        payload["definition_path"] = str(path)
        payload["reused"] = False
        return payload

    tracker_by_id = {
        item["asset"].id: item
        for item in tracker_items
        if item.get("asset") is not None
    }
    source_rows = _asset_rows(store, row, revision_id)
    source_by_id = {
        item["asset"].id: item
        for item in source_rows
        if item.get("asset") is not None
    }
    proposals: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = max(1, len(preflight))
    for index, plan in enumerate(preflight):
        native_item = source_by_id.get(plan["native_asset_id"])
        tracker_item = tracker_by_id.get(plan["selected_tracker_asset_id"])
        if native_item is None or tracker_item is None:
            failures.append({
                "participant_id": plan.get("participant_id"),
                "error": "The preflight source assets could not be recovered",
            })
            continue
        native_asset = native_item["asset"]
        tracker_asset = tracker_item["asset"]
        logs: list[str] = []
        try:
            evidence = detect_matching_tracker(
                pupil_native_source=Path(native_asset.source_path).expanduser().resolve(),
                tracker_candidate_paths=[Path(tracker_asset.source_path).expanduser().resolve()],
                world_c3d_path=None,
                canonical_participant_id=str(plan.get("participant_id") or ""),
                board_c3d_path=board_path,
                board_corner_labels={},
                outer_corner_labels={},
                sample_every_n_frames=max(1, int(sample_every_n_frames)),
                max_board_mapping_frames=32,
                log=logs.append,
                progress=lambda value, text, _index=index: emit(
                    progress,
                    0.34 + 0.11 * (_index + value) / total,
                    "Resolving board orientation · " + text,
                ),
            )
            tags, outer = _normalise_board_definition({
                "board_corner_labels": evidence.recommended_board_corner_labels,
                "outer_corner_labels": evidence.recommended_outer_corner_labels,
            })
            metrics = _validate_board_definition_against_c3d(
                board_path=board_path, tags=tags, outer=outer
            )
            candidate = evidence.candidates[0] if evidence.candidates else None
            proposals.append({
                "participant_id": plan.get("participant_id"),
                "native_asset_id": native_asset.id,
                "native_label": native_asset.label,
                "tracker_asset_id": tracker_asset.id,
                "tracker_label": tracker_asset.label,
                "hardware_id": plan.get("hardware_id"),
                "mapping_source": evidence.board_mapping_source,
                "mapping_variant": evidence.board_mapping_source,
                "board_corner_labels": tags,
                "outer_corner_labels": outer,
                "signature": list(_board_signature(tags, outer)),
                "camera_distance_mm": dict(candidate.camera_distance_mm) if candidate else {},
                "camera_proximity_score_mm": candidate.camera_proximity_score_mm if candidate else None,
                "direct_pnp_rms_px": dict(candidate.camera_direct_pnp_rms_px) if candidate else {},
                "metrics": metrics,
                "logs": logs,
            })
        except Exception as exc:
            failures.append({
                "participant_id": plan.get("participant_id"),
                "native_asset_id": native_asset.id,
                "tracker_asset_id": tracker_asset.id,
                "hardware_id": plan.get("hardware_id"),
                "error": str(exc),
                "logs": logs,
            })

    if not proposals:
        details = "; ".join(
            f"{item.get('participant_id') or 'actor'}: {item.get('error')}"
            for item in failures
        )
        raise ProcessingError(
            "The 16-marker board was found, but no calibration recording produced a usable AprilTag corner orientation. "
            + details
        )

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for proposal in proposals:
        grouped.setdefault(tuple(proposal["signature"]), []).append(proposal)
    if len(grouped) > 1:
        rendered = []
        for rows in grouped.values():
            rendered.append(
                f"{rows[0].get('mapping_variant')} from "
                + ", ".join(str(item.get("participant_id")) for item in rows)
            )
        raise ProcessingError(
            "Calibration actors disagreed on the AprilTag board orientation ("
            + "; ".join(rendered)
            + "). Review the board geometry instead of publishing inconsistent profiles."
        )

    accepted_rows = next(iter(grouped.values()))
    accepted = accepted_rows[0]
    tags = accepted["board_corner_labels"]
    outer = accepted["outer_corner_labels"]
    metrics = _validate_board_definition_against_c3d(
        board_path=board_path, tags=tags, outer=outer
    )
    payload = {
        "schema_version": "oculostudio-apriltag-board-definition-1.0",
        "calibration_set_id": row.id,
        "calibration_set_code": row.code,
        "revision_id": revision_id,
        "resolved_at": utc_now(),
        "mapping_source": "combined_head_identity_plus_apriltag_board_geometry",
        "mapping_variant": accepted.get("mapping_variant"),
        "apriltag_family": "36h11",
        "apriltag_ids": [0, 1, 2, 3],
        "board_corner_labels": tags,
        "outer_corner_labels": outer,
        "metrics": metrics,
        "proposals": proposals,
        "failures": failures,
        "warnings": (
            [
                "Only one calibration actor produced board-orientation evidence; the mapping was geometrically validated but not cross-confirmed."
            ]
            if len(accepted_rows) == 1
            else []
        ),
    }
    path = _persist_board_definition(
        store, row=row, revision_id=revision_id, payload=payload
    )
    payload["definition_path"] = str(path)
    payload["reused"] = False
    emit(progress, 0.45, "Shared AprilTag board geometry resolved")
    return payload


def _annotate_candidate_artifact(
    store: ProjectStore,
    *,
    session_id: str,
    artifact_id: str,
    build_id: str,
    source_participant_id: str,
    expected_hardware: str,
    status: str,
    calibration_set_id: str | None = None,
    calibration_set_record_id: str | None = None,
    calibration_revision_id: str | None = None,
) -> None:
    def mutate(project):
        artifact = project.sessions[session_id].artifacts.get(artifact_id)
        if artifact is None:
            return
        artifact.parameters.update({
            "calibration_set_profile_build_id": build_id,
            "calibration_set_profile_build_status": status,
            "calibration_actor_id": source_participant_id,
            "expected_hardware_id": expected_hardware,
            "calibration_set_id": calibration_set_id,
            "calibration_set_record_id": calibration_set_record_id,
            "calibration_revision_id": calibration_revision_id,
        })
        artifact.qc.update({
            "calibration_set_id": calibration_set_id,
            "calibration_set_record_id": calibration_set_record_id,
            "calibration_revision_id": calibration_revision_id,
            "hardware_id": expected_hardware,
        })
        artifact.provenance.update({
            "calibration_set_profile_build_id": build_id,
            "calibration_set_profile_build_status": status,
        })
    store.update(mutate)


def build_calibration_set_profiles(
    store: ProjectStore,
    *,
    session_id: str,
    calibration_set_id: str,
    revision_id: str | None = None,
    sample_every_n_frames: int = 4,
    max_detection_frames: int = 240,
    max_pose_gap_sec: float = 0.1,
    minimum_tracker_identity_confidence: str = "MEDIUM",
    mount_version: str = "mount_v1",
    force_rebuild: bool = False,
    output_g3dcal: bool = True,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    project = store.snapshot()
    row = _set_by_id_or_code(project, calibration_set_id)
    revision = _revision(row, revision_id)
    owner_session_id = row.owner_session_id or session_id
    if owner_session_id not in project.sessions:
        raise ValidationError("Calibration Set owner Session is unavailable")
    status = calibration_set_profile_build_status(
        store, calibration_set_id=row.id, revision_id=revision.id
    )
    recovered: list[dict[str, Any]] = []
    if not force_rebuild:
        from oculostudio.services.calibration_sets import register_calibration_profile_ref
        for hardware in list(status.get("missing_hardware") or []):
            candidates = [item for item in (status.get("recoverable_profiles", {}).get(hardware) or []) if item.get("available")]
            if len(candidates) != 1:
                continue
            candidate = candidates[0]
            register_calibration_profile_ref(
                store,
                calibration_set_id=row.id,
                source_session_id=candidate["source_session_id"],
                profile_artifact_id=candidate["artifact_id"],
                revision_id=revision.id,
            )
            recovered.append(candidate)
        if recovered:
            status = calibration_set_profile_build_status(
                store, calibration_set_id=row.id, revision_id=revision.id
            )
    if status["problems"] and status["missing_hardware"]:
        raise ValidationError("Calibration Set source preflight failed: " + "; ".join(status["problems"]))
    existing = status["hardware_slots"]
    targets = list(_REQUIRED_HARDWARE) if force_rebuild else list(status["missing_hardware"])
    if not targets:
        return {"status": "ready", "message": "ETM-A and ETM-B profiles already exist", "recovered_profiles": recovered, "build_status": status}

    source_rows = _asset_rows(store, row, revision.id)
    pupil_items = [item for item in source_rows if item["ref"].calibration_role == "pupil_calibration"]
    board_items = [item for item in source_rows if item["ref"].calibration_role == "calibration_board"]
    tracker_items = [item for item in source_rows if item["ref"].calibration_role == "tracker_candidate"]
    world_items = [item for item in source_rows if item["ref"].calibration_role == "world_reference"]
    definition_items = [item for item in source_rows if item["ref"].calibration_role == "board_definition"]
    board_item = board_items[0]
    world_item = world_items[0] if world_items else None
    valid_sessions = [
        project.sessions[sid].label for sid in row.applicable_session_ids if sid in project.sessions
    ]
    build_id = new_id("calbuild")
    report_root = store.calibration_set_dir(row.id) / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{build_id}.json"
    report: dict[str, Any] = {
        "schema_version": "oculostudio-calibration-set-profile-build-1.0",
        "build_id": build_id,
        "calibration_set_id": row.id,
        "calibration_set_code": row.code,
        "revision_id": revision.id,
        "revision_number": revision.revision_number,
        "created_at": utc_now(),
        "targets": targets,
        "preflight": [],
        "board_geometry": None,
        "created_profiles": [],
        "recovered_profiles": recovered,
        "status": "running",
    }
    atomic_write_json(report_path, report)

    emit(progress, 0.02, "Checking Calibration Set source completeness")
    preflight: list[dict[str, Any]] = []
    total = max(1, len(pupil_items))
    for index, item in enumerate(pupil_items):
        cancellation.raise_if_cancelled()
        asset = item["asset"]
        asset_kind = getattr(asset.kind, "value", asset.kind) if asset is not None else None
        if asset is None or str(asset_kind) != "pupil_native":
            raise ValidationError(
                f"Calibration actor recording {item['ref'].label} is not a Pupil Native recording. "
                "Use the NRD ZIP that contains the scene video and calibration.bin."
            )
        participant = item["ref"].participant_id or asset.participant_id or asset.label
        start = 0.05 + 0.28 * index / total
        span = 0.28 / total
        plan = _preflight_pairing(
            native_asset=asset,
            participant_id=str(participant),
            tracker_items=tracker_items,
            board_item=board_item,
            world_item=world_item,
            sample_every_n_frames=max(2, int(sample_every_n_frames)),
            minimum_confidence=minimum_tracker_identity_confidence,
            progress=lambda value, text, _start=start, _span=span: emit(progress, _start + _span * value, text),
        )
        preflight.append(plan)
        report["preflight"] = preflight
        atomic_write_json(report_path, report)

    by_hardware: dict[str, list[dict[str, Any]]] = {key: [] for key in _REQUIRED_HARDWARE}
    for plan in preflight:
        by_hardware.setdefault(plan["hardware_id"], []).append(plan)
    for hardware in targets:
        plans = by_hardware.get(hardware) or []
        if not plans:
            actor_rows = ", ".join(plan["participant_id"] for plan in preflight) or "none"
            raise ProcessingError(
                f"No calibration recording matched {hardware_display_name(hardware)}. "
                f"The available actors ({actor_rows}) matched other hardware. Verify that both physical ETMs were recorded."
            )
        if len(plans) > 1:
            details = ", ".join(f"{plan['participant_id']} ({plan['confidence']})" for plan in plans)
            raise ProcessingError(
                f"More than one calibration recording matched {hardware_display_name(hardware)}: {details}. "
                "Resolve the source identities before building profiles."
            )

    emit(progress, 0.34, "Resolving the shared 4×4 AprilTag board geometry")
    board_definition = _resolve_shared_board_definition(
        store,
        row=row,
        revision_id=revision.id,
        board_item=board_item,
        definition_items=definition_items,
        tracker_items=tracker_items,
        preflight=preflight,
        sample_every_n_frames=max(1, int(sample_every_n_frames)),
        progress=progress,
    )
    report["board_geometry"] = board_definition
    atomic_write_json(report_path, report)

    created: list[dict[str, Any]] = []
    artifact_ids: list[str] = []
    try:
        for target_index, hardware in enumerate(targets):
            cancellation.raise_if_cancelled()
            plan = by_hardware[hardware][0]
            native_id = plan["native_asset_id"]
            source_asset = store.snapshot().sessions[owner_session_id].assets[native_id]
            participant = str(plan["participant_id"])
            selected_tracker_id = plan["selected_tracker_asset_id"]
            emit(
                progress,
                0.46 + 0.47 * target_index / max(1, len(targets)),
                f"Estimating {hardware_display_name(hardware)} from {source_asset.label}",
            )
            result = create_legacy_gaze3d_calibration_profile(
                store,
                session_id=owner_session_id,
                native_asset_id=native_id,
                tracker_asset_ids=[selected_tracker_id],
                board_asset_id=board_item["asset"].id,
                world_asset_id=world_item["asset"].id if world_item else None,
                participant_id=participant,
                profile_name=f"{hardware_display_name(hardware)}_{mount_version}_{row.code}",
                sample_every_n_frames=int(sample_every_n_frames),
                max_detection_frames=int(max_detection_frames),
                max_pose_gap_sec=float(max_pose_gap_sec),
                hardware_id=hardware,
                mount_version=mount_version,
                valid_sessions=valid_sessions,
                auto_register_hardware=True,
                auto_profile_name=True,
                allow_hardware_id_override=False,
                output_g3dcal=bool(output_g3dcal),
                include_raw_identity_metadata=False,
                minimum_tracker_identity_confidence="SINGLE_CANDIDATE",
                board_corner_labels=board_definition["board_corner_labels"],
                outer_corner_labels=board_definition["outer_corner_labels"],
                board_mapping_source=str(
                    board_definition.get("mapping_source")
                    or board_definition.get("mapping_variant")
                    or "calibration_set_resolved_board_geometry"
                ),
                calibration_set_id=None,
                calibration_revision_id=None,
                progress=lambda value, text, _index=target_index: emit(
                    progress,
                    0.46 + 0.47 * (_index + value) / max(1, len(targets)),
                    text,
                ),
                cancellation=cancellation,
            )
            artifact_id = result["artifact"]["id"]
            produced_hardware = canonical_hardware_id(
                result.get("profile", {}).get("hardware_id")
                or result.get("hardware_registration", {}).get("hardware_id")
            )
            if produced_hardware != hardware:
                raise ProcessingError(
                    f"{source_asset.label} was preflighted as {hardware_display_name(hardware)} but the estimator "
                    f"produced {hardware_display_name(produced_hardware)}. No profile was committed to the Calibration Set."
                )
            _annotate_candidate_artifact(
                store,
                session_id=owner_session_id,
                artifact_id=artifact_id,
                build_id=build_id,
                source_participant_id=participant,
                expected_hardware=hardware,
                status="candidate",
                calibration_set_id=row.code,
                calibration_set_record_id=row.id,
                calibration_revision_id=revision.id,
            )
            artifact_ids.append(artifact_id)
            created.append({
                "hardware_id": hardware,
                "hardware_display_name": hardware_display_name(hardware),
                "participant_id": participant,
                "native_asset_id": native_id,
                "tracker_asset_id": selected_tracker_id,
                "artifact_id": artifact_id,
                "profile_path": result.get("profile_path"),
                "g3dcal_path": result.get("g3dcal_path"),
                "board_definition_path": board_definition.get("definition_path"),
                "board_mapping_variant": board_definition.get("mapping_variant"),
            })
            report["created_profiles"] = created
            atomic_write_json(report_path, report)

        from oculostudio.services.calibration_sets import register_calibration_profile_refs

        registration = register_calibration_profile_refs(
            store,
            calibration_set_id=row.id,
            source_session_id=owner_session_id,
            profile_artifact_ids=artifact_ids,
            revision_id=revision.id,
        )
        for item in created:
            _annotate_candidate_artifact(
                store,
                session_id=owner_session_id,
                artifact_id=item["artifact_id"],
                build_id=build_id,
                source_participant_id=item["participant_id"],
                expected_hardware=item["hardware_id"],
                status="committed",
                calibration_set_id=row.code,
                calibration_set_record_id=row.id,
                calibration_revision_id=revision.id,
            )
        report.update({
            "status": "completed",
            "completed_at": utc_now(),
            "registration": registration,
        })
        atomic_write_json(report_path, report)
        final_status = calibration_set_profile_build_status(
            store, calibration_set_id=row.id, revision_id=revision.id
        )
        emit(progress, 1.0, "Calibration Set physical profiles ready")
        return {
            "status": "completed",
            "build_id": build_id,
            "report_path": str(report_path),
            "created_profiles": created,
            "recovered_profiles": recovered,
            "registration": registration,
            "build_status": final_status,
        }
    except BaseException as exc:
        report.update({
            "status": "failed",
            "failed_at": utc_now(),
            "error": str(exc),
            "created_profiles": created,
            "note": "Candidate profile artifacts may remain in the owner Session for audit, but are not committed to the Calibration Set revision.",
        })
        atomic_write_json(report_path, report)
        for item in created:
            _annotate_candidate_artifact(
                store,
                session_id=owner_session_id,
                artifact_id=item["artifact_id"],
                build_id=build_id,
                source_participant_id=item["participant_id"],
                expected_hardware=item["hardware_id"],
                status="failed_uncommitted",
                calibration_set_id=row.code,
                calibration_set_record_id=row.id,
                calibration_revision_id=revision.id,
            )
        raise
