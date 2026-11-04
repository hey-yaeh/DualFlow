from __future__ import annotations

import copy
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from oculostudio.adapters.mocap import read_mocap
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.services.calibration import load_calibration_profile
from oculostudio.services.hardware_registration import resolve_measured_registration
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, sha256_file, utc_now


def _label_parts(label: str) -> tuple[str | None, str]:
    text = str(label).strip()
    if ":" in text:
        prefix, suffix = text.rsplit(":", 1)
        return prefix.strip() or None, suffix.strip().upper().replace(" ", "")
    return None, text.upper().replace(" ", "")


def _profile_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    hardware = profile.get("hardware") or {}
    transfer = profile.get("transfer_validation") or {}
    qc = profile.get("qc") or {}
    metadata = copy.deepcopy(profile.get("metadata") or {})
    metadata.update(
        {
            "calibration_source": profile.get("calibration_source") or metadata.get("calibration_source") or "oculostudio_profile",
            "hardware_id": profile.get("hardware_id") or hardware.get("hardware_id"),
            "hardware_display_name": profile.get("hardware_display_name") or hardware.get("hardware_display_name"),
            "mount_version": profile.get("mount_version") or hardware.get("mount_version"),
            "valid_sessions": profile.get("valid_sessions") or hardware.get("valid_sessions") or [],
            "legacy_hardware_aliases": profile.get("legacy_hardware_aliases") or metadata.get("legacy_hardware_aliases") or [],
            "hardware_registration": profile.get("hardware_registration") or metadata.get("hardware_registration") or {},
            "source_profile_id": profile.get("source_profile_id"),
            "transfer_validation": transfer,
        }
    )
    return {
        "profile_id": profile.get("profile_id") or new_id("cal"),
        "display_name": profile.get("profile_name") or profile.get("display_name") or "Calibration profile",
        "canonical_participant_id": profile.get("participant_id") or profile.get("canonical_participant_id"),
        "created_utc": profile.get("created_at") or profile.get("created_utc") or utc_now(),
        "module_serial_number": metadata.get("module_serial_number"),
        "frame_name": metadata.get("frame_name") or hardware.get("frame_name"),
        "source_recording_id": metadata.get("source_recording_id"),
        "tracker_marker_labels": list((profile.get("tracker") or {}).get("marker_labels") or profile.get("tracker_marker_labels") or []),
        "format": "NeonVicon-Gaze3D-CalibrationProfile",
        "format_version": 3,
        "metadata": metadata,
        "metrics": qc,
        "warnings": list(profile.get("warnings") or []),
    }


def write_g3dcal_profile(
    profile: dict[str, Any],
    output_path: str | Path,
    *,
    qc_files: list[str | Path] | None = None,
) -> Path:
    """Write a portable .g3dcal ZIP using the profile contract already read by OculoStudio."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = profile.get("tracker") or {}
    intrinsics = profile.get("camera_intrinsics") or {}
    labels = list(tracker.get("marker_labels") or profile.get("tracker_marker_labels") or [])
    template = np.asarray(tracker.get("marker_template_mm"), dtype=float)
    transform = np.asarray(profile.get("T_tracker_from_camera"), dtype=float)
    matrix_value = intrinsics.get("matrix")
    if matrix_value is None:
        matrix_value = intrinsics.get("camera_matrix")
    distortion_value = intrinsics.get("distortion")
    if distortion_value is None:
        distortion_value = intrinsics.get("distortion_coefficients")
    camera_matrix = np.asarray(matrix_value, dtype=float)
    distortion = np.asarray([] if distortion_value is None else distortion_value, dtype=float)
    if template.ndim != 2 or template.shape[1] != 3 or len(template) < 3:
        raise ValidationError("Calibration profile needs at least three tracker marker-template points")
    if len(labels) != len(template):
        raise ValidationError("Calibration tracker labels and template lengths differ")
    transform = transform.reshape(4, 4)
    camera_matrix = camera_matrix.reshape(3, 3)
    metadata = _profile_metadata(profile)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        tracker_template_mm=template,
        T_tracker_from_camera=transform,
        scene_camera_matrix=camera_matrix,
        scene_distortion_coefficients=distortion.reshape(-1),
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("profile.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        archive.writestr("arrays.npz", buffer.getvalue())
        archive.writestr(
            "qc/calibration_qc.json",
            json.dumps(profile.get("qc") or {}, indent=2, ensure_ascii=False),
        )
        for candidate in qc_files or []:
            path = Path(candidate)
            if path.exists() and path.is_file() and path.name.lower() != "calibration_qc.json":
                archive.write(path, f"qc/{safe_name(path.name)}")
    return output_path


def _normalise_profile(
    profile: dict[str, Any],
    *,
    label: str | None,
    hardware_id: str | None,
    mount_version: str | None,
    valid_sessions: list[str] | None,
    calibration_source: str,
) -> dict[str, Any]:
    result = copy.deepcopy(profile)
    result["profile_id"] = result.get("profile_id") or new_id("cal")
    result["profile_name"] = label or result.get("profile_name") or result.get("display_name") or result["profile_id"]
    if hardware_id:
        result["hardware_id"] = hardware_id.strip()
    if mount_version:
        result["mount_version"] = mount_version.strip()
    if valid_sessions is not None:
        result["valid_sessions"] = [str(value).strip() for value in valid_sessions if str(value).strip()]
    result["calibration_source"] = calibration_source
    result["created_at"] = result.get("created_at") or utc_now()
    result.setdefault("warnings", [])
    result.setdefault("qc", {})
    return result


def _extract_g3dcal_qc(profile_path: Path, output_root: Path) -> list[Path]:
    extracted: list[Path] = []
    if profile_path.suffix.lower() != ".g3dcal":
        return extracted
    try:
        with zipfile.ZipFile(profile_path) as archive:
            for name in archive.namelist():
                if not name.startswith("qc/") or name.endswith("/"):
                    continue
                target = output_root / "qc" / safe_name(Path(name).name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
                extracted.append(target)
    except Exception:
        return []
    return extracted


def import_calibration_profile(
    store: ProjectStore,
    *,
    session_id: str,
    profile_path: str,
    label: str | None = None,
    hardware_id: str | None = None,
    mount_version: str | None = None,
    valid_sessions: list[str] | None = None,
    source_note: str | None = None,
    calibration_set_id: str | None = None,
    calibration_revision_id: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    path = Path(profile_path).expanduser().resolve()
    if not path.exists() or path.suffix.lower() not in {".g3dcal", ".oculocal", ".json"}:
        raise ValidationError("Select an existing .g3dcal, .oculocal, or compatible JSON profile")
    emit(progress, 0.1, "Reading calibration profile")
    profile = _normalise_profile(
        load_calibration_profile(path),
        label=label,
        hardware_id=hardware_id,
        mount_version=mount_version,
        valid_sessions=valid_sessions,
        calibration_source="imported_external_profile",
    )
    profile.setdefault("metadata", {})["import_source_path"] = str(path)
    profile["metadata"]["import_source_sha256"] = sha256_file(path)
    if source_note:
        profile["metadata"]["source_note"] = source_note
    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "calibration_profiles", "imported") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    cancellation.raise_if_cancelled()
    original = output_root / f"original{path.suffix.lower()}"
    shutil.copy2(path, original)
    _extract_g3dcal_qc(path, output_root)
    normalized_path = output_root / f"{safe_name(profile['profile_name'])}.oculocal"
    atomic_write_json(normalized_path, profile)
    portable_path = output_root / f"{safe_name(profile['profile_name'])}.g3dcal"
    write_g3dcal_profile(profile, portable_path, qc_files=list((output_root / "qc").glob("*")))
    summary = {
        "format": profile.get("format"),
        "profile_id": profile.get("profile_id"),
        "profile_name": profile.get("profile_name"),
        "participant_id": profile.get("participant_id"),
        "hardware_id": profile.get("hardware_id"),
        "mount_version": profile.get("mount_version"),
        "valid_sessions": profile.get("valid_sessions") or [],
        "calibration_source": profile.get("calibration_source"),
        "camera_origin_tracker_mm": profile.get("camera_origin_tracker_mm"),
        "tracker_marker_count": len((profile.get("tracker") or {}).get("marker_labels") or []),
        "warnings": profile.get("warnings") or [],
    }
    atomic_write_json(output_root / "profile_summary.json", summary)
    qc = {
        **(profile.get("qc") or {}),
        "profile_imported": True,
        "hardware_id": profile.get("hardware_id"),
        "mount_version": profile.get("mount_version"),
        "valid_sessions": profile.get("valid_sessions") or [],
        "calibration_source": profile.get("calibration_source"),
    }
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=profile["profile_name"],
        output_root=output_root,
        qc=qc,
        parameters={
            "hardware_id": profile.get("hardware_id"),
            "mount_version": profile.get("mount_version"),
            "valid_sessions": profile.get("valid_sessions") or [],
            "calibration_source": "imported_external_profile",
            "calibration_set_id": calibration_set_id,
            "calibration_revision_id": calibration_revision_id,
        },
        provenance={"external_profile_sha256": sha256_file(path)},
        artifact_id=artifact_id,
    )
    calibration_set_registration = None
    if calibration_set_id:
        from oculostudio.services.calibration_sets import register_calibration_profile_ref

        calibration_set_registration = register_calibration_profile_ref(
            store,
            calibration_set_id=calibration_set_id,
            source_session_id=session_id,
            profile_artifact_id=artifact.id,
            revision_id=calibration_revision_id,
        )
    emit(progress, 1.0, "Calibration profile imported")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "profile_path": str(portable_path),
        "normalized_profile_path": str(normalized_path),
        "profile": profile,
        "calibration_set_registration": calibration_set_registration,
    }


def _profile_artifact_path(store: ProjectStore, session_id: str, artifact_id: str) -> Path:
    session = store.project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
        raise ValidationError("Select a calibration-profile artifact")
    candidates = [Path(row.path) for row in artifact.files if Path(row.path).exists()]
    for suffix in (".g3dcal", ".oculocal", ".json"):
        matches = [path for path in candidates if path.suffix.lower() == suffix]
        if matches:
            matches.sort(key=lambda value: ("summary" in value.name.lower(), len(value.parts), str(value).lower()))
            return matches[0]
    raise ValidationError("The selected calibration artifact contains no readable profile")


def register_existing_measured_profile_hardware(
    store: ProjectStore,
    *,
    session_id: str,
    profile_artifact_id: str,
    mount_version: str | None = None,
    valid_sessions: list[str] | None = None,
    label: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Register an older measured profile without rerunning calibration.

    This is a migration path for v0.16.1 artifacts that already contain a
    geometry-selected tracker but were saved with Hardware ID unset.  A new
    immutable profile artifact is created and linked to the measured source.
    """
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    source_artifact = session.artifacts.get(profile_artifact_id)
    if source_artifact is None or source_artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
        raise ValidationError("Select a measured calibration-profile artifact")
    existing_hardware = source_artifact.parameters.get("hardware_id") or source_artifact.qc.get("hardware_id")
    if existing_hardware:
        raise ValidationError("The selected calibration profile is already registered to hardware")
    source_type = source_artifact.parameters.get("calibration_source") or source_artifact.qc.get("calibration_source") or ""
    if "measured" not in str(source_type) and "apriltag" not in str(source_type):
        raise ValidationError("Only measured AprilTag calibration results can be auto-registered")

    emit(progress, 0.1, "Reading measured calibration result")
    profile_path = _profile_artifact_path(store, session_id, profile_artifact_id)
    profile = load_calibration_profile(profile_path)
    marker_labels = list((profile.get("tracker") or {}).get("marker_labels") or [])
    selected_asset_id = source_artifact.qc.get("selected_tracker_asset_id") or source_artifact.parameters.get("selected_tracker_asset_id")
    selected_label = source_artifact.qc.get("selected_tracker_filename")
    if not selected_label and selected_asset_id and selected_asset_id in session.assets:
        selected_label = session.assets[selected_asset_id].label
    if not selected_label:
        selected_label = (profile.get("hardware_registration") or {}).get("selected_tracker_label")
    if not selected_label:
        # Marker-prefix alias inference remains available when the old artifact
        # did not preserve the selected candidate filename.
        selected_label = marker_labels[0].split(":", 1)[0] if marker_labels else "tracker"

    registration = resolve_measured_registration(
        session=session,
        selected_tracker_label=str(selected_label),
        marker_labels=marker_labels,
        calibration_wearer_id=str(profile.get("participant_id") or source_artifact.parameters.get("participant_id") or ""),
        requested_hardware_id=None,
        mount_version=mount_version or profile.get("mount_version") or source_artifact.parameters.get("mount_version") or "mount_v1",
        valid_sessions=valid_sessions or profile.get("valid_sessions") or source_artifact.parameters.get("valid_sessions") or [session.label],
        requested_profile_name=label or source_artifact.label,
        auto_register_hardware=True,
        auto_profile_name=True,
        allow_hardware_id_override=False,
    )
    cancellation.raise_if_cancelled()
    profile.update(
        {
            "profile_id": new_id("cal"),
            "profile_name": registration["profile_name"],
            "hardware_id": registration["hardware_id"],
            "hardware_display_name": registration["hardware_display_name"],
            "legacy_hardware_aliases": registration["legacy_aliases"],
            "hardware_registration": registration,
            "mount_version": registration["mount_version"],
            "valid_sessions": registration["valid_sessions"],
            "calibration_source": source_type or profile.get("calibration_source") or "session_measured_apriltag",
            "source_profile_id": profile.get("profile_id"),
            "created_at": utc_now(),
        }
    )
    profile.setdefault("metadata", {})["hardware_registration"] = registration
    profile["metadata"]["registered_from_artifact_id"] = profile_artifact_id

    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "calibration_profiles", "registered") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    normalized = output_root / f"{safe_name(profile['profile_name'])}.oculocal"
    portable = output_root / f"{safe_name(profile['profile_name'])}.g3dcal"
    atomic_write_json(normalized, profile)
    qc_files: list[Path] = []
    for row in source_artifact.files:
        candidate = Path(row.path)
        if not candidate.exists() or not candidate.is_file():
            continue
        lower = candidate.name.lower()
        if lower.endswith((".g3dcal", ".oculocal")):
            continue
        if lower.endswith((".png", ".html", ".csv", ".json", ".txt")):
            target = output_root / "source_qc" / safe_name(candidate.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            qc_files.append(target)
    write_g3dcal_profile(profile, portable, qc_files=qc_files)
    summary = {
        "profile_id": profile.get("profile_id"),
        "profile_name": profile.get("profile_name"),
        "hardware_id": registration["hardware_id"],
        "hardware_display_name": registration["hardware_display_name"],
        "mount_version": registration["mount_version"],
        "valid_sessions": registration["valid_sessions"],
        "hardware_registration": registration,
        "registered_from_artifact_id": profile_artifact_id,
    }
    atomic_write_json(output_root / "profile_summary.json", summary)
    qc = {
        **(profile.get("qc") or {}),
        "hardware_id": registration["hardware_id"],
        "hardware_display_name": registration["hardware_display_name"],
        "mount_version": registration["mount_version"],
        "valid_sessions": registration["valid_sessions"],
        "hardware_registration_status": "registered_from_existing_measured_profile",
        "hardware_registration_revision": registration["registration_revision"],
    }
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=profile["profile_name"],
        output_root=output_root,
        source_asset_ids=list(source_artifact.source_asset_ids),
        parent_artifact_ids=[profile_artifact_id],
        qc=qc,
        parameters={
            "hardware_id": registration["hardware_id"],
            "hardware_display_name": registration["hardware_display_name"],
            "mount_version": registration["mount_version"],
            "valid_sessions": registration["valid_sessions"],
            "calibration_source": profile.get("calibration_source"),
            "hardware_registration_status": "registered_from_existing_measured_profile",
            "hardware_registration_revision": registration["registration_revision"],
            "supersedes_profile_artifact_id": registration["supersedes_profile_artifact_id"],
            "legacy_hardware_aliases": registration["legacy_aliases"],
        },
        provenance={
            "registered_from_artifact_id": profile_artifact_id,
            "source_profile_sha256": sha256_file(profile_path),
            "hardware_registration": registration,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Measured profile registered to hardware")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "profile_path": str(portable),
        "normalized_profile_path": str(normalized),
        "hardware_registration": registration,
        "profile": profile,
    }


def resolve_profile_tracker_points(
    profile_labels: list[str], mocap
) -> tuple[np.ndarray, np.ndarray, list[str], str | None]:
    # Exact labels are preferred.  If a session used a different prefix, match
    # marker suffixes within one candidate rigid-body group.
    try:
        points, valid = mocap.select(profile_labels)
        prefixes = {_label_parts(label)[0] for label in profile_labels if _label_parts(label)[0]}
        return points, valid, profile_labels, next(iter(prefixes)) if len(prefixes) == 1 else None
    except ValidationError:
        pass
    profile_suffixes = [_label_parts(label)[1] for label in profile_labels]
    groups: dict[str | None, dict[str, str]] = {}
    for label in mocap.labels:
        prefix, suffix = _label_parts(label)
        groups.setdefault(prefix, {})[suffix] = label
    candidates: list[tuple[str | None, list[str]]] = []
    for prefix, rows in groups.items():
        if all(suffix in rows for suffix in profile_suffixes):
            candidates.append((prefix, [rows[suffix] for suffix in profile_suffixes]))
    if len(candidates) != 1:
        names = [prefix or "<unprefixed>" for prefix, _ in candidates]
        raise ValidationError(
            f"Profile marker suffixes do not identify one tracker group in the selected C3D; candidates={names}"
        )
    prefix, labels = candidates[0]
    points, valid = mocap.select(labels)
    return points, valid, labels, prefix


# Backward-compatible private alias used by older tests and downstream code.
_ordered_tracker_points = resolve_profile_tracker_points


def _pairwise_medians(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    pairs: list[str] = []
    for i in range(points.shape[1]):
        for j in range(i + 1, points.shape[1]):
            mask = valid[:, i] & valid[:, j]
            if int(mask.sum()) < 3:
                values.append(float("nan"))
            else:
                values.append(float(np.median(np.linalg.norm(points[mask, i] - points[mask, j], axis=1))))
            pairs.append(f"{i}|{j}")
    return np.asarray(values, dtype=float), pairs


def validate_calibration_profile_transfer(
    store: ProjectStore,
    *,
    session_id: str,
    tracker_asset_id: str,
    profile_artifact_id: str | None = None,
    profile_path: str | None = None,
    interaction_id: str | None = None,
    role_id: str | None = None,
    label: str | None = None,
    maximum_shape_rms_mm: float = 5.0,
    high_confidence_rms_mm: float = 2.0,
    minimum_marker_valid_fraction: float = 0.5,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    if profile_artifact_id:
        source_profile_path = _profile_artifact_path(store, session_id, profile_artifact_id)
    elif profile_path:
        source_profile_path = Path(profile_path).expanduser().resolve()
        if not source_profile_path.exists():
            raise ValidationError(f"Calibration profile does not exist: {source_profile_path}")
    else:
        raise ValidationError("Select a calibration profile artifact or external profile path")
    session = store.project.sessions[session_id]
    tracker_asset = session.assets.get(tracker_asset_id)
    if tracker_asset is None:
        raise ValidationError("Selected tracker asset is unavailable")
    emit(progress, 0.1, "Reading source hardware calibration profile")
    profile = load_calibration_profile(source_profile_path)
    profile_labels = list((profile.get("tracker") or {}).get("marker_labels") or [])
    template = np.asarray((profile.get("tracker") or {}).get("marker_template_mm"), dtype=float)
    if len(profile_labels) < 3 or template.shape != (len(profile_labels), 3):
        raise ValidationError("Calibration profile has no usable tracker marker template")
    cancellation.raise_if_cancelled()
    emit(progress, 0.35, "Comparing current ETM rigid-body geometry")
    mocap = read_mocap(tracker_asset.source_path)
    points, valid, matched_labels, matched_prefix = resolve_profile_tracker_points(profile_labels, mocap)
    current, pair_names = _pairwise_medians(points, valid)
    reference, _ = _pairwise_medians(
        np.repeat(template[None, :, :], 3, axis=0),
        np.ones((3, len(template)), dtype=bool),
    )
    finite = np.isfinite(current) & np.isfinite(reference)
    if int(finite.sum()) < 3:
        raise ProcessingError("Too few valid tracker marker pairs for transfer validation")
    delta = current[finite] - reference[finite]
    rms = float(np.sqrt(np.mean(delta * delta)))
    max_abs = float(np.max(np.abs(delta)))
    marker_valid_fraction = float(valid.mean())
    passed = rms <= maximum_shape_rms_mm and marker_valid_fraction >= minimum_marker_valid_fraction
    confidence = "HIGH" if passed and rms <= high_confidence_rms_mm else ("MEDIUM" if passed else "LOW")
    validation = {
        "status": "passed" if passed else "failed",
        "confidence": confidence,
        "scope": "tracker_cluster_shape_only",
        "pairwise_distance_rms_mm": rms,
        "pairwise_distance_max_abs_mm": max_abs,
        "valid_pair_count": int(finite.sum()),
        "marker_count": len(profile_labels),
        "marker_valid_fraction": marker_valid_fraction,
        "matched_marker_labels": matched_labels,
        "matched_tracker_prefix": matched_prefix,
        "maximum_shape_rms_mm": maximum_shape_rms_mm,
        "high_confidence_rms_mm": high_confidence_rms_mm,
        "minimum_marker_valid_fraction": minimum_marker_valid_fraction,
        "interaction_id": interaction_id,
        "role_id": role_id,
        "warning": (
            "Tracker-shape agreement verifies the rigid marker cluster, but cannot prove that the camera did not move relative to that cluster. Review head projection or a known target before production use."
        ),
    }
    if not passed:
        raise ProcessingError(
            f"Calibration transfer failed: tracker-shape RMS {rms:.3f} mm, valid fraction {marker_valid_fraction:.3f}"
        )
    emit(progress, 0.7, "Creating session-specific transferred profile")
    transferred = copy.deepcopy(profile)
    source_profile_id = transferred.get("profile_id")
    transferred["source_profile_id"] = source_profile_id
    transferred["profile_id"] = new_id("cal")
    transferred["profile_name"] = label or f"{transferred.get('profile_name') or source_profile_path.stem} · transferred"
    transferred["calibration_source"] = "transferred_hardware_profile"
    transferred["transfer_validation"] = validation
    transferred.setdefault("tracker", {})["source_marker_labels"] = list(profile_labels)
    transferred["tracker"]["marker_labels"] = list(matched_labels)
    transferred["tracker"]["matched_tracker_prefix"] = matched_prefix
    transferred["participant_id"] = tracker_asset.participant_id or transferred.get("participant_id")
    transferred["valid_sessions"] = sorted(set((transferred.get("valid_sessions") or []) + [session.label]))
    transferred.setdefault("warnings", []).append(validation["warning"])
    transferred.setdefault("qc", {})["transfer_validation"] = validation
    transferred["created_at"] = utc_now()
    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "calibration_profiles", "transferred") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    normalized_path = output_root / f"{safe_name(transferred['profile_name'])}.oculocal"
    atomic_write_json(normalized_path, transferred)
    portable_path = output_root / f"{safe_name(transferred['profile_name'])}.g3dcal"
    write_g3dcal_profile(transferred, portable_path)
    atomic_write_json(output_root / "transfer_validation.json", validation)
    qc = {
        "calibration_source": "transferred_hardware_profile",
        "transfer_status": validation["status"],
        "transfer_confidence": confidence,
        "tracker_shape_rms_mm": rms,
        "tracker_shape_max_abs_mm": max_abs,
        "marker_valid_fraction": marker_valid_fraction,
        "review_required": True,
        "allowed_target_granularity": (
            "body_and_object_pilot" if confidence == "HIGH" else "coarse_body_and_large_object_pilot"
        ),
    }
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=transferred["profile_name"],
        output_root=output_root,
        source_asset_ids=[tracker_asset_id],
        parent_artifact_ids=[profile_artifact_id] if profile_artifact_id else [],
        qc=qc,
        parameters={
            "interaction_id": interaction_id,
            "role_id": role_id,
            "profile_source": str(source_profile_path),
            "maximum_shape_rms_mm": maximum_shape_rms_mm,
            "high_confidence_rms_mm": high_confidence_rms_mm,
            "minimum_marker_valid_fraction": minimum_marker_valid_fraction,
        },
        provenance={"source_profile_sha256": sha256_file(source_profile_path)},
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Transferred calibration profile validated")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "profile_path": str(portable_path),
        "normalized_profile_path": str(normalized_path),
        "profile": transferred,
        "validation": validation,
    }
