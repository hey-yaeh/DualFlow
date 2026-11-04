from __future__ import annotations

"""Integration bridge for Neon–Vicon Gaze3D Studio v0.3.2.

This module deliberately reuses the audited Workflow-B implementation rather
than reimplementing its special 16-marker AprilTag-board assumptions in a
second, subtly different form.  The public OculoStudio artifact remains the
normal calibration_profile contract, so downstream Gaze 3D code does not need
any legacy-specific branches.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from neon_vicon_gaze3d.io.c3d_reader import read_c3d as legacy_read_c3d
from neon_vicon_gaze3d.processing.calibration import (
    CalibrationRequest,
    run_workflow_b_calibration,
)
from neon_vicon_gaze3d.processing.tracker_identity import (
    TrackerIdentityError,
    detect_matching_tracker,
    discover_head_subjects,
)

from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.services.calibration import load_calibration_profile
from oculostudio.services.calibration_profiles import write_g3dcal_profile
from oculostudio.services.hardware_registration import resolve_measured_registration
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import (
    CancellationToken,
    atomic_write_json,
    emit,
    new_id,
    safe_name,
    sha256_file,
    utc_now,
)

_CONFIDENCE_RANK = {
    "LOW": 0,
    "SINGLE_CANDIDATE": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def _map_progress(progress, start: float, span: float):
    def callback(value: float, text: str) -> None:
        emit(progress, start + span * float(np.clip(value, 0.0, 1.0)), text)

    return callback


def _asset_path(store: ProjectStore, session_id: str, asset_id: str) -> Path:
    session = store.project.sessions[session_id]
    asset = session.assets.get(asset_id)
    if asset is None:
        raise ValidationError(f"Asset is unavailable: {asset_id}")
    path = Path(asset.source_path).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"Asset file does not exist: {path}")
    return path


def _normalise_explicit_board_mapping(
    board_labels: list[str],
    board_corner_labels: dict[str, dict[str, str]] | None,
    outer_corner_labels: dict[str, str] | None,
) -> tuple[dict[str, dict[str, str]], dict[str, str]] | None:
    if not board_corner_labels and not outer_corner_labels:
        return None
    if not isinstance(board_corner_labels, dict) or not isinstance(outer_corner_labels, dict):
        raise ValidationError(
            "Explicit board mapping requires both board_corner_labels and outer_corner_labels"
        )
    tags: dict[str, dict[str, str]] = {}
    ordered = ("TL", "TR", "BR", "BL")
    for marker_id in ("0", "1", "2", "3"):
        row = board_corner_labels.get(marker_id)
        if row is None:
            row = board_corner_labels.get(int(marker_id))
        if not isinstance(row, dict):
            raise ValidationError(f"Explicit board mapping is missing AprilTag {marker_id}")
        tags[marker_id] = {}
        for corner in ordered:
            value = str(row.get(corner) or "").strip()
            if not value:
                raise ValidationError(
                    f"Explicit board mapping is missing AprilTag {marker_id} {corner}"
                )
            tags[marker_id][corner] = value
    outer: dict[str, str] = {}
    for corner in ordered:
        value = str(outer_corner_labels.get(corner) or "").strip()
        if not value:
            raise ValidationError(f"Explicit board mapping is missing outer corner {corner}")
        outer[corner] = value
    required = [tags[tag][corner] for tag in ("0", "1", "2", "3") for corner in ordered]
    required.extend(outer[corner] for corner in ordered)
    missing = sorted({label for label in required if label not in set(board_labels)})
    if missing:
        raise ValidationError(
            "Explicit board mapping refers to labels absent from the board C3D: "
            + ", ".join(missing)
        )
    if len(set(required[:16])) != 16:
        raise ValidationError("Explicit board mapping must contain 16 distinct AprilTag corner labels")
    return tags, outer


def _normalised_profile(
    raw_path: Path,
    *,
    profile_name: str,
    participant_id: str,
    hardware_id: str | None,
    mount_version: str | None,
    valid_sessions: list[str] | None,
    identity: dict[str, Any],
    native_asset_id: str,
    board_asset_id: str,
    selected_tracker_asset_id: str,
    world_asset_id: str | None,
) -> dict[str, Any]:
    profile = load_calibration_profile(raw_path)
    metadata = dict(profile.get("metadata") or {})
    metadata.update(
        {
            "legacy_engine": {
                "name": "NeonVicon Gaze3D Studio",
                "version": "0.3.2",
                "integration": "OculoStudio legacy Workflow-B bridge",
            },
            "workflow": "legacy_auto_16_marker_workflow_b",
            "board_mode": "auto_infer_planar_4x4_marker_grid",
            "board_mapping_source": identity.get("board_mapping_source"),
            "board_corner_labels": identity.get("recommended_board_corner_labels") or {},
            "outer_corner_labels": identity.get("recommended_outer_corner_labels") or {},
            "tracker_identity_detection": identity,
            "native_asset_id": native_asset_id,
            "board_asset_id": board_asset_id,
            "selected_tracker_asset_id": selected_tracker_asset_id,
            "world_asset_id": world_asset_id,
            "hardware_id": hardware_id,
            "mount_version": mount_version,
            "valid_sessions": [str(value).strip() for value in (valid_sessions or []) if str(value).strip()],
            "calibration_source": "session_measured_apriltag_auto_16_marker",
        }
    )
    profile.update(
        {
            "profile_name": profile_name,
            "participant_id": participant_id,
            "hardware_id": hardware_id,
            "mount_version": mount_version,
            "valid_sessions": [str(value).strip() for value in (valid_sessions or []) if str(value).strip()],
            "calibration_source": "session_measured_apriltag_auto_16_marker",
            "metadata": metadata,
            "created_at": profile.get("created_at") or profile.get("created_utc") or utc_now(),
        }
    )
    profile.setdefault("warnings", [])
    profile["warnings"] = list(dict.fromkeys([*(profile.get("warnings") or []), *(identity.get("warnings") or [])]))
    return profile


def create_legacy_gaze3d_calibration_profile(
    store: ProjectStore,
    *,
    session_id: str,
    native_asset_id: str,
    tracker_asset_ids: list[str],
    board_asset_id: str,
    world_asset_id: str | None = None,
    participant_id: str = "P01",
    profile_name: str = "Neon to Tracker Calibration",
    sample_every_n_frames: int = 2,
    max_detection_frames: int = 400,
    max_pose_gap_sec: float = 0.1,
    hardware_id: str | None = None,
    mount_version: str | None = None,
    valid_sessions: list[str] | None = None,
    auto_register_hardware: bool = True,
    auto_profile_name: bool = True,
    allow_hardware_id_override: bool = False,
    output_g3dcal: bool = True,
    include_raw_identity_metadata: bool = False,
    minimum_tracker_identity_confidence: str = "MEDIUM",
    board_corner_labels: dict[str, dict[str, str]] | None = None,
    outer_corner_labels: dict[str, str] | None = None,
    board_mapping_source: str | None = None,
    calibration_set_id: str | None = None,
    calibration_revision_id: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Create an OculoStudio profile using the legacy automatic 16-marker board workflow.

    The legacy engine performs all of the behaviour that made the original tool
    useful for this dataset: geometry-only 4x4 board inference, eight
    rotation/reflection hypotheses, automatic tracker candidate selection,
    calibration-only video/C3D offset search, robust multi-frame SE(3)
    aggregation, and the original visual QC report.
    """

    cancellation = cancellation or CancellationToken.create()
    if len(tracker_asset_ids) < 1:
        raise ValidationError("Select at least one eye-tracker C3D candidate")
    confidence_key = str(minimum_tracker_identity_confidence or "MEDIUM").strip().upper()
    if confidence_key not in _CONFIDENCE_RANK:
        raise ValidationError("minimum_tracker_identity_confidence must be LOW, SINGLE_CANDIDATE, MEDIUM, or HIGH")

    session = store.project.sessions[session_id]
    native_path = _asset_path(store, session_id, native_asset_id)
    board_path = _asset_path(store, session_id, board_asset_id)
    tracker_paths = [_asset_path(store, session_id, asset_id) for asset_id in tracker_asset_ids]
    world_path = _asset_path(store, session_id, world_asset_id) if world_asset_id else None

    cancellation.raise_if_cancelled()
    emit(progress, 0.01, "Inspecting legacy 16-marker calibration inputs")
    try:
        board_probe = legacy_read_c3d(board_path)
    except Exception as exc:
        raise ProcessingError(f"The legacy engine could not read the board C3D: {exc}") from exc
    if board_probe.point_count != 16:
        raise ValidationError(
            f"Legacy automatic board mode requires exactly 16 C3D markers; the selected board has {board_probe.point_count}. "
            "Use Explicit Board Definition JSON for another board geometry."
        )
    explicit_mapping = _normalise_explicit_board_mapping(
        list(board_probe.labels), board_corner_labels, outer_corner_labels
    )

    logs: list[str] = []
    try:
        identity_result = detect_matching_tracker(
            pupil_native_source=native_path,
            tracker_candidate_paths=tracker_paths,
            world_c3d_path=world_path,
            canonical_participant_id=participant_id,
            board_c3d_path=board_path,
            board_corner_labels=(explicit_mapping[0] if explicit_mapping else {}),
            outer_corner_labels=(explicit_mapping[1] if explicit_mapping else {}),
            sample_every_n_frames=max(1, int(sample_every_n_frames)),
            log=logs.append,
            progress=_map_progress(progress, 0.03, 0.31),
        )
    except TrackerIdentityError as exc:
        raise ProcessingError(f"Automatic tracker/board inference failed: {exc}") from exc
    cancellation.raise_if_cancelled()

    actual_rank = _CONFIDENCE_RANK.get(identity_result.confidence.upper(), 0)
    required_rank = _CONFIDENCE_RANK[confidence_key]
    if actual_rank < required_rank:
        raise ProcessingError(
            f"Automatic tracker identity confidence is {identity_result.confidence}, below the required {confidence_key}. "
            "Review the identity evidence or use the explicit/manual calibration path."
        )
    if explicit_mapping is not None:
        identity_result.recommended_board_corner_labels = {
            str(key): dict(value) for key, value in explicit_mapping[0].items()
        }
        identity_result.recommended_outer_corner_labels = dict(explicit_mapping[1])
        identity_result.board_mapping_source = (
            str(board_mapping_source or "explicit_calibration_set_board_definition")
        )
        identity_result.reasons.append(
            "AprilTag board orientation was supplied by the active Calibration Set revision and validated against the 16-marker board C3D."
        )

    # Safety fallback for legacy/manual callers: a labelled-head route can
    # identify the correct tracker without returning an AprilTag orientation.
    # Resolve the missing orientation against that already-selected tracker
    # instead of discarding the strong head evidence or failing at 45%.
    if not identity_result.recommended_board_corner_labels or not identity_result.recommended_outer_corner_labels:
        if identity_result.selected_tracker_path:
            try:
                mapping_identity = detect_matching_tracker(
                    pupil_native_source=native_path,
                    tracker_candidate_paths=[Path(identity_result.selected_tracker_path)],
                    world_c3d_path=None,
                    canonical_participant_id=participant_id,
                    board_c3d_path=board_path,
                    board_corner_labels={},
                    outer_corner_labels={},
                    sample_every_n_frames=max(1, int(sample_every_n_frames)),
                    log=logs.append,
                    progress=_map_progress(progress, 0.28, 0.08),
                )
                identity_result.recommended_board_corner_labels = {
                    str(key): dict(value)
                    for key, value in mapping_identity.recommended_board_corner_labels.items()
                }
                identity_result.recommended_outer_corner_labels = dict(
                    mapping_identity.recommended_outer_corner_labels
                )
                identity_result.board_mapping_source = (
                    "head_identity_plus_" + str(mapping_identity.board_mapping_source)
                )
                identity_result.reasons.append(
                    "Head-marker evidence selected the physical tracker; a second single-tracker AprilTag pass resolved board orientation."
                )
            except Exception as exc:
                raise ProcessingError(
                    "The physical tracker was identified, but the 16-marker AprilTag board orientation could not be resolved: "
                    + str(exc)
                ) from exc
        if not identity_result.recommended_board_corner_labels or not identity_result.recommended_outer_corner_labels:
            raise ProcessingError(
                "The physical tracker was identified, but no usable 4×4 AprilTag board mapping was produced. "
                "Review the board-geometry evidence in the active Calibration Set revision."
            )

    selected_tracker_path = identity_result.selected_path.resolve()
    selected_tracker_asset_id = next(
        (
            asset_id
            for asset_id, candidate_path in zip(tracker_asset_ids, tracker_paths)
            if candidate_path.resolve() == selected_tracker_path
        ),
        None,
    )
    if selected_tracker_asset_id is None:
        raise ProcessingError("The automatically selected tracker could not be mapped back to an OculoStudio asset")

    selected_tracker_asset = session.assets[selected_tracker_asset_id]
    registration = resolve_measured_registration(
        session=session,
        selected_tracker_label=selected_tracker_asset.label or selected_tracker_path.name,
        marker_labels=list(identity_result.selected_tracker_labels),
        calibration_wearer_id=participant_id,
        requested_hardware_id=hardware_id,
        mount_version=mount_version,
        valid_sessions=valid_sessions or [session.label],
        requested_profile_name=profile_name,
        auto_register_hardware=auto_register_hardware,
        auto_profile_name=auto_profile_name,
        allow_hardware_id_override=allow_hardware_id_override,
    )
    hardware_id = registration["hardware_id"]
    mount_version = registration["mount_version"]
    valid_sessions = registration["valid_sessions"]
    profile_name = registration["profile_name"]

    head_labels: list[str] | None = None
    if world_path and identity_result.matched_c3d_subject:
        try:
            subjects = discover_head_subjects(legacy_read_c3d(world_path))
            head_labels = subjects.get(identity_result.matched_c3d_subject)
        except Exception:
            head_labels = None

    profile_id = new_id("cal")
    output_root = store.derived_dir(session_id, "calibrations", profile_id)
    identity_json = identity_result.to_dict()
    atomic_write_json(output_root / "tracker_identity_detection.json", identity_json)
    (output_root / "legacy_engine_log.txt").write_text("\n".join(logs), encoding="utf-8")

    request = CalibrationRequest(
        pupil_calibration_source=native_path,
        tracker_c3d_path=selected_tracker_path,
        board_c3d_path=board_path,
        tracker_marker_labels=list(identity_result.selected_tracker_labels),
        board_corner_labels={str(key): dict(value) for key, value in identity_result.recommended_board_corner_labels.items()},
        outer_corner_labels=dict(identity_result.recommended_outer_corner_labels),
        canonical_participant_id=participant_id,
        display_name=profile_name,
        output_directory=output_root,
        head_c3d_path=world_path if head_labels else None,
        head_marker_labels=head_labels,
        sample_every_n_frames=max(1, int(sample_every_n_frames)),
        max_detection_frames=max(20, int(max_detection_frames)),
        max_pose_gap_sec=float(max_pose_gap_sec),
        tracker_identity_audit=identity_json,
        include_raw_identity_metadata=bool(include_raw_identity_metadata),
    )
    try:
        result = run_workflow_b_calibration(
            request,
            log=logs.append,
            progress=_map_progress(progress, 0.36, 0.57),
        )
    except Exception as exc:
        raise ProcessingError(f"Legacy Workflow-B calibration failed: {exc}") from exc
    cancellation.raise_if_cancelled()
    (output_root / "legacy_engine_log.txt").write_text("\n".join(logs), encoding="utf-8")

    raw_profile_path = output_root / "legacy_engine_raw.g3dcal"
    if result.profile_path.resolve() != raw_profile_path.resolve():
        if raw_profile_path.exists():
            raw_profile_path.unlink()
        shutil.move(str(result.profile_path), raw_profile_path)

    profile = _normalised_profile(
        raw_profile_path,
        profile_name=profile_name,
        participant_id=participant_id,
        hardware_id=hardware_id,
        mount_version=mount_version,
        valid_sessions=valid_sessions,
        identity=identity_json,
        native_asset_id=native_asset_id,
        board_asset_id=board_asset_id,
        selected_tracker_asset_id=selected_tracker_asset_id,
        world_asset_id=world_asset_id,
    )
    profile["hardware_display_name"] = registration["hardware_display_name"]
    profile["legacy_hardware_aliases"] = registration["legacy_aliases"]
    profile["hardware_registration"] = registration
    profile.setdefault("metadata", {})["hardware_registration"] = registration
    normalized_path = output_root / f"{safe_name(profile_name)}.oculocal"
    atomic_write_json(normalized_path, profile)
    portable_path: Path | None = None
    if output_g3dcal:
        portable_path = output_root / f"{safe_name(profile_name)}.g3dcal"
        portable_qc = [*result.qc_files, result.metrics_path, result.report_path, output_root / "tracker_identity_detection.json"]
        write_g3dcal_profile(profile, portable_path, qc_files=portable_qc)

    qc = dict(profile.get("qc") or {})
    qc.update(
        {
            "legacy_gaze3d_engine_integrated": True,
            "legacy_engine_version": "0.3.2",
            "board_mode": "auto_infer_planar_4x4_marker_grid",
            "board_mapping_source": identity_result.board_mapping_source,
            "tracker_identity_confidence": identity_result.confidence,
            "selected_tracker_asset_id": selected_tracker_asset_id,
            "selected_tracker_filename": identity_result.selected_tracker_filename,
            "calibration_source": "session_measured_apriltag_auto_16_marker",
            "hardware_id": hardware_id,
            "hardware_display_name": registration["hardware_display_name"],
            "mount_version": mount_version,
            "hardware_registration_status": registration["registration_status"],
            "hardware_registration_revision": registration["registration_revision"],
        }
    )
    atomic_write_json(output_root / "profile_summary.json", {
        "profile_id": profile.get("profile_id"),
        "profile_name": profile_name,
        "participant_id": participant_id,
        "hardware_id": hardware_id,
        "hardware_display_name": registration["hardware_display_name"],
        "mount_version": mount_version,
        "valid_sessions": profile.get("valid_sessions") or [],
        "hardware_registration": registration,
        "selected_tracker_asset_id": selected_tracker_asset_id,
        "tracker_identity_confidence": identity_result.confidence,
        "board_mapping_source": identity_result.board_mapping_source,
        "camera_origin_tracker_mm": profile.get("camera_origin_tracker_mm"),
        "warnings": profile.get("warnings") or [],
    })

    source_ids = [native_asset_id, selected_tracker_asset_id, board_asset_id]
    if world_asset_id:
        source_ids.append(world_asset_id)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.CALIBRATION_PROFILE,
        label=profile_name,
        output_root=output_root,
        source_asset_ids=source_ids,
        qc=qc,
        parameters={
            "board_mode": "auto_legacy_16_marker",
            "participant_id": participant_id,
            "sample_every_n_frames": int(sample_every_n_frames),
            "max_detection_frames": int(max_detection_frames),
            "max_pose_gap_sec": float(max_pose_gap_sec),
            "hardware_id": hardware_id,
            "hardware_display_name": registration["hardware_display_name"],
            "mount_version": mount_version,
            "valid_sessions": profile.get("valid_sessions") or [],
            "auto_register_hardware": bool(auto_register_hardware),
            "auto_profile_name": bool(auto_profile_name),
            "allow_hardware_id_override": bool(allow_hardware_id_override),
            "hardware_registration_status": registration["registration_status"],
            "hardware_registration_revision": registration["registration_revision"],
            "supersedes_profile_artifact_id": registration["supersedes_profile_artifact_id"],
            "legacy_hardware_aliases": registration["legacy_aliases"],
            "output_g3dcal": bool(output_g3dcal),
            "include_raw_identity_metadata": bool(include_raw_identity_metadata),
            "minimum_tracker_identity_confidence": confidence_key,
            "calibration_set_id": calibration_set_id,
            "calibration_revision_id": calibration_revision_id,
        },
        provenance={
            "legacy_engine": "NeonVicon Gaze3D Studio v0.3.2",
            "native_sha256": session.assets[native_asset_id].source_sha256,
            "board_sha256": session.assets[board_asset_id].source_sha256,
            "tracker_sha256": session.assets[selected_tracker_asset_id].source_sha256,
            "world_sha256": session.assets[world_asset_id].source_sha256 if world_asset_id else None,
            "board_mapping_source": identity_result.board_mapping_source,
            "hardware_registration": registration,
        },
        artifact_id=profile_id.replace("cal_", "artifact_", 1),
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
    emit(progress, 1.0, "Legacy Workflow-B calibration profile complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "profile_path": str(portable_path or normalized_path),
        "g3dcal_path": str(portable_path) if portable_path else None,
        "normalized_profile_path": str(normalized_path),
        "legacy_raw_profile_path": str(raw_profile_path),
        "calibration_report_path": str(result.report_path),
        "tracker_identity": identity_json,
        "hardware_registration": registration,
        "profile": profile,
        "calibration_set_registration": calibration_set_registration,
    }
