from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.mocap import read_mocap
from oculostudio.adapters.pupil import materialized_cloud_root
from oculostudio.domain.geometry import PoseSeries, make_transform, normalize
from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.timegraph import TimeGraph, apply_time_map
from oculostudio.errors import ProcessingError, TimeGraphError, ValidationError
from oculostudio.services.calibration import CameraIntrinsics, load_calibration_profile
from oculostudio.services.calibration_profiles import resolve_profile_tracker_points
from oculostudio.services.artifact_io import existing_artifact_paths, find_artifact_file
from oculostudio.services.etm_role_assignment import resolve_assignment_for_role
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _find_column(columns, aliases, *, required=True):
    mapping = {_normalize_column(column): column for column in columns}
    for alias in aliases:
        key = _normalize_column(alias)
        if key in mapping:
            return mapping[key]
    if required:
        raise ValidationError(f"gaze.csv is missing required column: {aliases[0]}")
    return None


def _find_gaze_csv(root: Path) -> Path:
    candidates = list(root.rglob("gaze.csv"))
    if len(candidates) != 1:
        raise ValidationError(f"Expected one gaze.csv, found {len(candidates)}")
    return candidates[0]


def _native_trim_gaze_frame(trim_artifact) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the exact TSD gaze rows exported by a Native-reference trim.

    Point timestamps remain absolute for provenance, while
    ``oculo_clip_video_ns`` provides the authoritative local interaction clock.
    ``oculo_source_row_index`` preserves the original full-recording row ID.
    """
    if trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
        raise ValidationError("native_trim_artifact_id must reference a Native-reference trim artifact")
    paths = existing_artifact_paths(trim_artifact)
    gaze_candidates = [
        path for path in paths
        if path.name.lower() == "gaze.csv" and any(part.lower() == "tsd" for part in path.parts)
    ]
    if len(gaze_candidates) == 1:
        frame = pd.read_csv(gaze_candidates[0], low_memory=False)
    elif len(gaze_candidates) > 1:
        raise ValidationError("Native-reference trim contains more than one TSD gaze.csv")
    else:
        archive_path = find_artifact_file(trim_artifact, names=["TSD_trimmed.zip"], required=False)
        if archive_path is None:
            raise ValidationError("Native-reference trim contains no TSD gaze.csv or TSD_trimmed.zip")
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if Path(name).name.lower() == "gaze.csv"]
            if len(members) != 1:
                raise ValidationError(f"Expected one gaze.csv in TSD_trimmed.zip, found {len(members)}")
            with archive.open(members[0]) as handle:
                frame = pd.read_csv(handle, low_memory=False)

    source_col = _find_column(
        frame.columns,
        ["oculo_source_row_index", "source_gaze_row_index", "source_row_index"],
    )
    clip_col = _find_column(frame.columns, ["oculo_clip_video_ns", "clip_time_ns"])
    if frame[source_col].duplicated().any():
        raise ValidationError("Native-reference trim gaze source-row indices are not unique")
    source_rows = pd.to_numeric(frame[source_col], errors="raise").astype("int64")
    clip_ns = pd.to_numeric(frame[clip_col], errors="coerce")
    if clip_ns.isna().any():
        raise ValidationError("Native-reference trim contains invalid local clip timestamps")
    if (clip_ns < 0).any():
        raise ValidationError("Native-reference trim contains negative local gaze timestamps")
    frame = frame.copy()
    frame["__oculostudio_source_row_index"] = source_rows.to_numpy(dtype=np.int64)
    frame["__oculostudio_local_clip_ns"] = clip_ns.to_numpy(dtype=np.float64)

    manifest_path = find_artifact_file(trim_artifact, names=["trim_manifest.json"], required=False)
    manifest: dict[str, Any] = {}
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except Exception as exc:
            raise ValidationError(f"Could not read Native-reference trim manifest: {exc}") from exc
    source_row_count = None
    for row in (manifest.get("report") or {}).get("tables") or []:
        if str(row.get("group", "")).lower() != "tsd":
            continue
        if Path(str(row.get("member", ""))).name.lower() != "gaze.csv":
            continue
        value = row.get("input_rows") or row.get("selected_recording_rows")
        if value is not None:
            source_row_count = int(value)
            break
    metadata = {
        "manifest": manifest,
        "source_row_count": int(source_row_count) if source_row_count is not None else int(len(frame)),
        "selected_row_count": int(len(frame)),
        "local_start_ns": 0,
        "local_end_ns": int(round(float(clip_ns.max()))) if len(clip_ns) else 0,
        "source_column": source_col,
        "clip_column": clip_col,
    }
    return frame, metadata


def _trim_belongs_to_timeseries(trim_artifact, timeseries_asset_id: str) -> bool:
    requested = trim_artifact.parameters.get("timeseries_asset_id")
    if requested:
        return str(requested) == str(timeseries_asset_id)
    manifest_path = find_artifact_file(trim_artifact, names=["trim_manifest.json"], required=False)
    if manifest_path is not None:
        try:
            request = (json.loads(manifest_path.read_text("utf-8")) or {}).get("request") or {}
            if request.get("timeseries_asset_id"):
                return str(request["timeseries_asset_id"]) == str(timeseries_asset_id)
        except Exception:
            pass
    return timeseries_asset_id in (trim_artifact.source_asset_ids or [])


def _time_domain_label(session, domain_id: str | None) -> str:
    domain = session.time_domains.get(domain_id) if domain_id else None
    if domain is None:
        return domain_id or "unregistered clock"
    return f"{domain.label} ({domain.clock_type})"


def resolve_gaze3d_time_path(session, timeseries_asset, tracker_asset):
    """Resolve the Pupil/TSD → Vicon/C3D clock path with a user-facing error.

    Gaze pixels are timestamped in the Pupil recording clock while the tracker
    pose is sampled in the C3D clock.  A scientifically valid world-space ray
    therefore requires an explicit TimeGraph path.  Never substitute an
    identity map or align first samples silently.
    """
    source_domain_id = timeseries_asset.primary_time_domain_id
    target_domain_id = tracker_asset.primary_time_domain_id
    if not source_domain_id or not target_domain_id:
        raise ValidationError("Timeseries and tracker assets need primary time domains")
    graph = TimeGraph.from_session(session)
    try:
        mapping = graph.resolve(source_domain_id, target_domain_id)
    except TimeGraphError as exc:
        source_clock = _time_domain_label(session, source_domain_id)
        target_clock = _time_domain_label(session, target_domain_id)
        raise ValidationError(
            "Synchronization required before Gaze 3D: "
            f"Timeseries '{timeseries_asset.label}' uses {source_clock}, but tracker "
            f"'{tracker_asset.label}' uses {target_clock}, and no TimeGraph path connects them. "
            "Open Synchronize and fit a verified Pupil↔Vicon mapping from at least one shared event "
            "(two or more anchors are recommended to estimate clock drift). "
            "OculoStudio will not invent an identity or first-frame alignment."
        ) from exc
    if mapping.quality.anchor_count <= 0 or "provisional" in str(mapping.method).lower():
        source_clock = _time_domain_label(session, source_domain_id)
        target_clock = _time_domain_label(session, target_domain_id)
        raise ValidationError(
            "Verified synchronization required before Gaze 3D: "
            f"a provisional zero-offset path currently connects '{timeseries_asset.label}' ({source_clock}) "
            f"to '{tracker_asset.label}' ({target_clock}), but it contains no shared-event anchors. "
            "Open Synchronize and replace it with at least one verified Pupil↔Vicon anchor; "
            "two or more anchors are recommended to estimate drift."
        )
    return mapping


def unproject_gaze_pixels(
    x_px: np.ndarray,
    y_px: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    points = np.stack([x_px, y_px], axis=-1).reshape(-1, 1, 2)
    valid = np.isfinite(points).all(axis=(1, 2))
    directions = np.full((len(points), 3), np.nan, dtype=float)
    if valid.any():
        undistorted = cv2.undistortPoints(
            points[valid].astype(np.float64),
            intrinsics.matrix.astype(np.float64),
            intrinsics.distortion.astype(np.float64),
        ).reshape(-1, 2)
        vectors = np.column_stack([undistorted, np.ones(len(undistorted))])
        directions[valid] = normalize(vectors)
    return directions


def _resolve_profile_path(store: ProjectStore, session_id: str, profile_path: str | None, profile_artifact_id: str | None) -> Path:
    if profile_path:
        path = Path(profile_path).expanduser().resolve()
        if not path.exists():
            raise ValidationError(f"Calibration profile does not exist: {path}")
        if path.suffix.lower() not in {".oculocal", ".g3dcal", ".json"}:
            raise ValidationError("Calibration profile must be .oculocal, .g3dcal, or compatible JSON")
        return path
    if not profile_artifact_id:
        raise ValidationError("A calibration profile path or artifact ID is required")
    artifact = store.project.sessions[session_id].artifacts[profile_artifact_id]
    candidates = [Path(file.path) for file in artifact.files if Path(file.path).suffix.lower() in {".oculocal", ".g3dcal", ".json"}]
    existing = [path for path in candidates if path.exists()]
    if existing:
        existing.sort(key=lambda path: (0 if path.suffix.lower() in {".oculocal", ".g3dcal"} else 1, len(path.parts), str(path).lower()))
        return existing[0]
    raise ValidationError("The selected artifact contains no supported calibration profile")


def _load_acceptance_artifact(artifact) -> dict[str, Any]:
    if artifact.kind != ArtifactKind.CALIBRATION_ACCEPTANCE:
        raise ValidationError("Selected artifact is not a calibration acceptance")
    path = find_artifact_file(artifact, names=["calibration_acceptance.json"])
    return json.loads(path.read_text("utf-8"))


def generate_gaze_rays(
    store: ProjectStore,
    *,
    session_id: str,
    timeseries_asset_id: str,
    tracker_asset_id: str | None = None,
    profile_path: str | None = None,
    profile_artifact_id: str | None = None,
    etm_assignment_artifact_id: str | None = None,
    wearer_role_id: str | None = None,
    calibration_acceptance_artifact_id: str | None = None,
    require_calibration_acceptance: bool = False,
    clip_id: str | None = None,
    native_trim_artifact_id: str | None = None,
    scope: Literal["full", "clip", "native_trim"] = "full",
    alignment_mode: Literal["timegraph", "pretrimmed_shared_start"] = "timegraph",
    shared_start_offset_sec: float = 0.0,
    max_pose_gap_sec: float = 0.1,
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    cancellation = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    acceptance = None
    if scope == "native_trim" and not etm_assignment_artifact_id:
        raise ValidationError(
            "Calibration stage incomplete: Native-reference Gaze 3D requires a resolved ETM-role assignment created in Calibration"
        )
    if calibration_acceptance_artifact_id:
        acceptance_artifact = session.artifacts.get(calibration_acceptance_artifact_id)
        if acceptance_artifact is None:
            raise ValidationError("Selected calibration-acceptance artifact is unavailable")
        acceptance = _load_acceptance_artifact(acceptance_artifact)
        accepted_profile_artifact = acceptance.get("transferred_profile_artifact_id")
        if not profile_artifact_id and not profile_path and accepted_profile_artifact:
            profile_artifact_id = accepted_profile_artifact
    assignment_row = None
    assignment_artifact = None
    handoff_requires_time_sync = False
    handoff_time_sync: dict[str, Any] | None = None
    handoff_time_map_id: str | None = None
    handoff_time_sync_artifact_id: str | None = None
    if etm_assignment_artifact_id:
        assignment_artifact = session.artifacts.get(etm_assignment_artifact_id)
        if assignment_artifact is None:
            raise ValidationError("Selected ETM-role assignment artifact is unavailable")
        if not wearer_role_id:
            raise ValidationError("Select the wearer role when using an ETM-role assignment")
        assignment_row = resolve_assignment_for_role(assignment_artifact, wearer_role_id)
        assigned_tracker_id = assignment_row["etm_asset_id"]
        if tracker_asset_id and tracker_asset_id != assigned_tracker_id:
            raise ValidationError(
                f"Selected tracker conflicts with ETM assignment: role {wearer_role_id} uses {assignment_row['etm_asset_label']}"
            )
        tracker_asset_id = assigned_tracker_id
        recommended_profile = assignment_row.get("recommended_calibration_artifact_id")
        if scope == "native_trim" and not recommended_profile:
            raise ValidationError(
                f"Calibration stage incomplete: role {wearer_role_id} is resolved to an ETM, but no immutable profile binding handoff exists"
            )
        if scope == "native_trim" and profile_artifact_id and recommended_profile and profile_artifact_id != recommended_profile:
            raise ValidationError(
                "The manually selected profile conflicts with the immutable Calibration handoff"
            )
        if not profile_artifact_id and not profile_path and recommended_profile:
            profile_artifact_id = recommended_profile
        assignment_parameters = assignment_artifact.parameters or {}
        assignment_qc = assignment_artifact.qc or {}
        is_calibration_handoff = bool(
            assignment_parameters.get("calibration_phase") == "handoff"
            or assignment_parameters.get("binding_source_assignment_artifact_id")
            or assignment_qc.get("binding_source_assignment_artifact_id")
            or assignment_qc.get("handoff_complete") is not None
        )
        handoff_requires_time_sync = bool(
            assignment_parameters.get("require_time_sync")
            or assignment_qc.get("time_sync_required")
        )
        if scope == "native_trim" and is_calibration_handoff and not handoff_requires_time_sync:
            raise ValidationError(
                "The selected legacy Calibration handoff has no role-specific Pupil IMU↔ETM TimeGraph; "
                "open Calibration, synchronize both roles, and create a new handoff"
            )
        handoff_time_sync = assignment_row.get("time_sync_check") or (
            (assignment_artifact.qc or {}).get("time_sync_by_role") or {}
        ).get(str(wearer_role_id).upper())
        handoff_time_map_id = str(
            assignment_row.get("recommended_time_map_id")
            or (handoff_time_sync or {}).get("time_map_id")
            or ""
        ) or None
        handoff_time_sync_artifact_id = str(
            assignment_row.get("recommended_time_sync_artifact_id")
            or (handoff_time_sync or {}).get("artifact_id")
            or ""
        ) or None
        if scope == "native_trim" and handoff_requires_time_sync:
            if not handoff_time_sync or not bool(handoff_time_sync.get("production_ready")):
                raise ValidationError(
                    f"Calibration stage incomplete: role {wearer_role_id} has no production-ready Pupil IMU↔ETM synchronization"
                )
            if not handoff_time_map_id or handoff_time_map_id not in session.time_maps:
                raise ValidationError(
                    f"Calibration handoff for {wearer_role_id} refers to an unavailable TimeGraph mapping"
                )
            if alignment_mode != "timegraph":
                raise ValidationError(
                    "This Calibration handoff requires the verified IMU↔ETM TimeGraph; shared-start alignment is not allowed"
                )
        if scope == "native_trim" and not profile_artifact_id and not profile_path:
            raise ValidationError(
                f"Calibration stage incomplete: role {wearer_role_id} has no compatible hardware profile bound by the Calibration handoff"
            )
    if not tracker_asset_id:
        raise ValidationError("Select interaction tracker motion, or use an ETM-role assignment artifact")
    timeseries_asset = session.assets[timeseries_asset_id]
    tracker_asset = session.assets[tracker_asset_id]
    if alignment_mode not in {"timegraph", "pretrimmed_shared_start"}:
        raise ValidationError("alignment_mode must be 'timegraph' or 'pretrimmed_shared_start'")
    trim_artifact = None
    trim_metadata: dict[str, Any] | None = None
    if scope == "native_trim":
        if not native_trim_artifact_id:
            raise ValidationError("scope='native_trim' requires native_trim_artifact_id")
        trim_artifact = session.artifacts.get(native_trim_artifact_id)
        if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
            raise ValidationError("Select a Native-reference trim artifact")
        if not _trim_belongs_to_timeseries(trim_artifact, timeseries_asset_id):
            raise ValidationError("The selected Native-reference trim was created from a different Timeseries asset")
    elif native_trim_artifact_id:
        raise ValidationError("native_trim_artifact_id is only valid when scope='native_trim'")
    if alignment_mode == "pretrimmed_shared_start" and scope != "native_trim":
        raise ValidationError("Pre-trimmed shared-start alignment requires scope='native_trim'")
    time_mapping = None
    if alignment_mode == "timegraph":
        if not timeseries_asset.primary_time_domain_id or not tracker_asset.primary_time_domain_id:
            raise ValidationError("Timeseries and tracker assets need primary time domains")
        if handoff_requires_time_sync and handoff_time_map_id:
            time_mapping = session.time_maps.get(handoff_time_map_id)
            if time_mapping is None:
                raise ValidationError("Calibration handoff TimeGraph mapping is unavailable")
            if (
                time_mapping.source_domain_id != timeseries_asset.primary_time_domain_id
                or time_mapping.target_domain_id != tracker_asset.primary_time_domain_id
            ):
                raise ValidationError(
                    "Calibration handoff TimeGraph does not connect the selected Pupil Timeseries clock to the assigned ETM clock"
                )
            if len(time_mapping.anchors) < 2:
                raise ValidationError(
                    "Calibration handoff TimeGraph has fewer than two distributed anchors; clock drift is not verified"
                )
        else:
            time_mapping = resolve_gaze3d_time_path(session, timeseries_asset, tracker_asset)
    profile_file = _resolve_profile_path(store, session_id, profile_path, profile_artifact_id)
    profile = load_calibration_profile(profile_file)
    if acceptance:
        accepted_profile_artifact = acceptance.get("transferred_profile_artifact_id")
        if profile_artifact_id and accepted_profile_artifact and accepted_profile_artifact != profile_artifact_id:
            raise ValidationError("Calibration acceptance belongs to a different transferred profile")
        if acceptance.get("acceptance_status") not in {"accepted", "accepted_limited"}:
            raise ValidationError(
                f"Calibration acceptance is {acceptance.get('acceptance_status')}; production Gaze 3D is blocked"
            )
    if (require_calibration_acceptance or scope == "native_trim") and profile.get("calibration_source") == "transferred_hardware_profile" and not acceptance:
        raise ValidationError(
            "Calibration stage incomplete: a transferred hardware profile requires an accepted interaction-specific calibration gate before Native-reference Gaze 3D"
        )
    # A calibration profile is normally created from a calibration C3D and then
    # applied to a different interaction C3D for the same physical tracker.
    # Therefore source asset IDs are provenance only; compatibility is enforced
    # below through the calibrated marker labels/template rather than file identity.
    profile_tracker_asset = profile.get("tracker", {}).get("source_asset_id")
    tracker_asset_differs_from_calibration_source = bool(
        profile_tracker_asset and profile_tracker_asset != tracker_asset_id
    )
    emit(progress, 0.03, "Reading Timeseries gaze data")
    local_clip_ns: np.ndarray | None = None
    if scope == "native_trim":
        assert trim_artifact is not None
        frame, trim_metadata = _native_trim_gaze_frame(trim_artifact)
        input_row_count = int(trim_metadata["source_row_count"])
        local_clip_ns = frame["__oculostudio_local_clip_ns"].to_numpy(dtype=np.float64)
    else:
        cache = MaterializationCache(store.cache_dir())
        root = materialized_cloud_root(cache.materialize(timeseries_asset), timeseries_asset.recording_id)
        gaze_path = _find_gaze_csv(root)
        frame = pd.read_csv(gaze_path, low_memory=False)
        input_row_count = len(frame)
        frame["__oculostudio_source_row_index"] = np.arange(input_row_count, dtype=np.int64)
    timestamp_col = _find_column(frame.columns, ["timestamp [ns]", "timestamp_ns"])
    x_col = _find_column(frame.columns, ["gaze x [px]", "gaze_x_px", "gaze x", "x"])
    y_col = _find_column(frame.columns, ["gaze y [px]", "gaze_y_px", "gaze y", "y"])
    worn_col = _find_column(frame.columns, ["worn", "is worn", "is_worn"], required=False)
    valid_col = _find_column(frame.columns, ["gaze valid", "gaze_valid", "valid"], required=False)
    timestamps = pd.to_numeric(frame[timestamp_col], errors="coerce").to_numpy(dtype=float)
    selected_mask = np.isfinite(timestamps)
    if scope == "clip":
        if not clip_id:
            raise ValidationError("scope='clip' requires a clip_id")
        clip = session.clips[clip_id]
        graph = TimeGraph.from_session(session)
        start_ts = int(graph.convert(session.global_time_domain_id, timeseries_asset.primary_time_domain_id, clip.start_global_ns))
        end_ts = int(graph.convert(session.global_time_domain_id, timeseries_asset.primary_time_domain_id, clip.end_global_ns))
        selected_mask &= (timestamps >= start_ts) & (timestamps < end_ts)
        frame = frame.loc[selected_mask].copy().reset_index(drop=True)
        timestamps = pd.to_numeric(frame[timestamp_col], errors="coerce").to_numpy(dtype=float)
    elif scope == "native_trim":
        if local_clip_ns is None or len(local_clip_ns) != len(frame):
            raise ValidationError("Native-reference trim local clock is unavailable")
    elif scope != "full":
        raise ValidationError("scope must be 'full', 'clip' or 'native_trim'")
    selected_input_row_count = len(frame)
    finite_timestamp = np.isfinite(timestamps)
    cancellation.raise_if_cancelled()
    emit(progress, 0.14, "Reading tracker motion capture")
    tracker_data = read_mocap(tracker_asset.source_path)
    labels = [str(label) for label in profile["tracker"]["marker_labels"]]
    template = np.asarray(profile["tracker"]["marker_template_mm"], dtype=float)
    positions, marker_valid, matched_tracker_labels, matched_tracker_prefix = resolve_profile_tracker_points(
        labels, tracker_data
    )
    pose_series = PoseSeries.from_markers(
        tracker_data.time_sec,
        template,
        positions,
        marker_valid,
        min_markers=3,
        max_fit_rmse_mm=12.0,
    )
    query_ns = np.full(len(timestamps), 0, dtype=np.int64)
    query_sec = np.full(len(timestamps), np.nan, dtype=np.float64)
    if alignment_mode == "pretrimmed_shared_start":
        if local_clip_ns is None:
            raise ValidationError("Pre-trimmed shared-start alignment requires Native-reference trim local timestamps")
        finite_local_time = np.isfinite(local_clip_ns)
        timing_valid = finite_timestamp & finite_local_time
        query_sec[timing_valid] = (
            float(tracker_data.time_sec[0])
            + local_clip_ns[timing_valid] / 1e9
            + float(shared_start_offset_sec)
        )
        query_ns[timing_valid] = np.rint(query_sec[timing_valid] * 1e9).astype(np.int64)
    else:
        assert time_mapping is not None
        timing_valid = finite_timestamp
        query_ns[timing_valid] = apply_time_map(
            time_mapping, timestamps[timing_valid].astype(np.int64)
        )
        query_sec[timing_valid] = query_ns[timing_valid].astype(np.float64) / 1e9
    emit(progress, 0.28, "Interpolating tracker pose")
    interpolated = pose_series.interpolate(query_sec, max_gap_sec=max_pose_gap_sec)
    intrinsics = CameraIntrinsics.from_dict(profile["camera_intrinsics"])
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    emit(progress, 0.43, "Unprojecting gaze pixels")
    directions_camera = unproject_gaze_pixels(x, y, intrinsics)
    T_tracker_from_camera = np.asarray(profile["T_tracker_from_camera"], dtype=float).reshape(4, 4)
    R_tc = T_tracker_from_camera[:3, :3]
    p_tc = T_tracker_from_camera[:3, 3]
    origins = np.full((len(frame), 3), np.nan)
    directions = np.full((len(frame), 3), np.nan)
    valid = timing_valid & interpolated.valid & np.isfinite(directions_camera).all(axis=1)
    if worn_col:
        worn = pd.to_numeric(frame[worn_col], errors="coerce").fillna(0).to_numpy() > 0
        valid &= worn
    if valid_col:
        source_valid = frame[valid_col]
        if source_valid.dtype == bool:
            valid &= source_valid.to_numpy()
        else:
            valid &= pd.to_numeric(source_valid, errors="coerce").fillna(0).to_numpy() > 0
    valid_indices = np.flatnonzero(valid)
    emit(progress, 0.55, "Transforming gaze rays into the Vicon world frame")
    for chunk_start in range(0, len(valid_indices), 50_000):
        cancellation.raise_if_cancelled()
        indices = valid_indices[chunk_start : chunk_start + 50_000]
        R_wt = interpolated.rotations[indices]
        p_wt = interpolated.translations[indices]
        origins[indices] = np.einsum("nij,j->ni", R_wt, p_tc) + p_wt
        direction_tracker = directions_camera[indices] @ R_tc.T
        directions[indices] = normalize(np.einsum("nij,nj->ni", R_wt, direction_tracker))
        emit(progress, 0.55 + 0.28 * min(1.0, (chunk_start + len(indices)) / max(1, len(valid_indices))), "Writing calibrated ray arrays")
    output = frame.copy()
    source_row_index = output.pop("__oculostudio_source_row_index").to_numpy(dtype=np.int64)
    if "__oculostudio_local_clip_ns" in output:
        output.pop("__oculostudio_local_clip_ns")
    output.insert(0, "gaze3d_input_row_index", source_row_index)
    output["gaze3d_c3d_time_sec"] = query_sec
    output["gaze3d_tracker_local_time_sec"] = query_sec - float(tracker_data.time_sec[0])
    output["gaze3d_c3d_frame_float"] = (query_sec - tracker_data.time_sec[0]) * tracker_data.frame_rate + tracker_data.first_frame
    in_tracker_time_range = timing_valid & (query_sec >= tracker_data.time_sec[0]) & (query_sec <= tracker_data.time_sec[-1])
    output["gaze3d_in_c3d_time_range"] = in_tracker_time_range
    output["gaze3d_alignment_mode"] = alignment_mode
    output["gaze3d_native_trim_artifact_id"] = native_trim_artifact_id
    output["gaze3d_origin_x_mm"] = origins[:, 0]
    output["gaze3d_origin_y_mm"] = origins[:, 1]
    output["gaze3d_origin_z_mm"] = origins[:, 2]
    output["gaze3d_direction_x"] = directions[:, 0]
    output["gaze3d_direction_y"] = directions[:, 1]
    output["gaze3d_direction_z"] = directions[:, 2]
    output["gaze3d_ray_valid"] = valid
    output["gaze3d_tracker_pose_interpolated"] = interpolated.interpolated
    output["gaze3d_tracker_pose_gap_sec"] = interpolated.interpolation_gap_sec
    output["gaze3d_tracker_fit_rmse_mm"] = interpolated.fit_rmse_mm
    output["gaze3d_calibration_profile_id"] = profile.get("profile_id")
    output["gaze3d_calibration_profile_format_version"] = profile.get("format_version")
    output["gaze3d_calibration_source"] = profile.get("calibration_source") or "unspecified"
    output["gaze3d_calibration_hardware_id"] = profile.get("hardware_id")
    output["gaze3d_calibration_mount_version"] = profile.get("mount_version")
    output["gaze3d_calibration_transfer_status"] = (profile.get("transfer_validation") or {}).get("status")
    output["gaze3d_calibration_transfer_confidence"] = (profile.get("transfer_validation") or {}).get("confidence")
    output["gaze3d_calibration_acceptance_status"] = acceptance.get("acceptance_status") if acceptance else None
    output["gaze3d_calibration_allowed_target_granularity"] = acceptance.get("allowed_target_granularity") if acceptance else None
    artifact_id = new_id("artifact")
    scope_key = native_trim_artifact_id or clip_id or "full"
    output_root = store.derived_dir(session_id, "gaze3d", scope_key) / artifact_id / safe_name(timeseries_asset.label)
    output_root.mkdir(parents=True, exist_ok=False)
    output_csv = output_root / "gaze_rays_world.csv"
    output.to_csv(output_csv, index=False)
    output_parquet_path = None
    if output_parquet:
        try:
            output_parquet_path = output_root / "gaze_rays_world.parquet"
            output.to_parquet(output_parquet_path, index=False)
        except Exception as exc:
            output_parquet_path = None
    selected_valid_fraction = float(valid.mean()) if len(valid) else 0.0
    full_source_context_fraction = (
        float(valid.sum()) / float(input_row_count) if input_row_count else 0.0
    )
    tracker_time_overlap_fraction = float(in_tracker_time_range.mean()) if len(in_tracker_time_range) else 0.0
    tracker_duration_sec = (
        float(tracker_data.time_sec[-1] - tracker_data.time_sec[0])
        if len(tracker_data.time_sec) > 1 else 0.0
    )
    selected_local_duration_sec = None
    if local_clip_ns is not None and len(local_clip_ns):
        selected_local_duration_sec = float(np.nanmax(local_clip_ns) - np.nanmin(local_clip_ns)) / 1e9
    qc = {
        "source_input_row_count": int(input_row_count),
        "selected_input_row_count": int(selected_input_row_count),
        "output_row_count": int(len(output)),
        "scope": scope,
        "alignment_mode": alignment_mode,
        "alignment_basis": (
            "Native-reference trim local clock + first C3D frame"
            if alignment_mode == "pretrimmed_shared_start"
            else "Verified TimeGraph mapping"
        ),
        "native_trim_artifact_id": native_trim_artifact_id,
        "shared_start_offset_sec": float(shared_start_offset_sec),
        "row_count_preserved": bool(len(output) == selected_input_row_count),
        "full_recording_row_count_preserved": bool(scope == "full" and len(output) == input_row_count),
        "valid_ray_count": int(valid.sum()),
        "valid_ray_fraction": selected_valid_fraction,
        "trim_valid_ray_count": int(valid.sum()) if scope == "native_trim" else None,
        "trim_valid_ray_fraction": selected_valid_fraction if scope == "native_trim" else None,
        "valid_ray_fraction_relative_to_full_source": full_source_context_fraction,
        "tracker_time_overlap_fraction": tracker_time_overlap_fraction,
        "tracker_duration_sec": tracker_duration_sec,
        "selected_local_duration_sec": selected_local_duration_sec,
        "tracker_pose_valid_fraction": float(interpolated.valid.mean()) if len(interpolated.valid) else 0.0,
        "tracker_pose_interpolated_count": int(interpolated.interpolated.sum()),
        "median_tracker_fit_rmse_mm": float(np.nanmedian(interpolated.fit_rmse_mm)) if np.isfinite(interpolated.fit_rmse_mm).any() else None,
        "calibration_applied": True,
        "calibration_profile_id": profile.get("profile_id"),
        "calibration_source": profile.get("calibration_source") or "unspecified",
        "calibration_hardware_id": profile.get("hardware_id"),
        "calibration_mount_version": profile.get("mount_version"),
        "calibration_valid_sessions": profile.get("valid_sessions") or [],
        "calibration_transfer_validation": profile.get("transfer_validation") or {},
        "calibration_acceptance": acceptance or {},
        "calibration_acceptance_artifact_id": calibration_acceptance_artifact_id,
        "calibration_acceptance_required": bool(require_calibration_acceptance),
        "matched_tracker_marker_labels": matched_tracker_labels,
        "matched_tracker_prefix": matched_tracker_prefix,
        "calibration_warnings": profile.get("warnings") or [],
        "camera_origin_tracker_mm": p_tc.tolist(),
        "tracker_asset_differs_from_calibration_source": tracker_asset_differs_from_calibration_source,
        "calibration_source_tracker_asset_id": profile_tracker_asset,
        "interaction_tracker_asset_id": tracker_asset_id,
        "etm_assignment_artifact_id": etm_assignment_artifact_id,
        "wearer_role_id": wearer_role_id,
        "assignment_confidence": assignment_row.get("confidence") if assignment_row else None,
        "calibration_handoff_requires_time_sync": bool(handoff_requires_time_sync),
        "calibration_time_sync_artifact_id": handoff_time_sync_artifact_id,
        "calibration_time_map_id": handoff_time_map_id,
        "calibration_time_sync_production_ready": bool((handoff_time_sync or {}).get("production_ready")) if handoff_requires_time_sync else None,
        "calibration_time_sync_distributed_drift_validated": bool((handoff_time_sync or {}).get("distributed_drift_validated")) if handoff_requires_time_sync else None,
        "calibration_time_sync_ego_video_used": bool((handoff_time_sync or {}).get("ego_video_used", False)) if handoff_requires_time_sync else False,
    }
    manifest = {
        "schema_version": "oculostudio-gaze3d-manifest-1.1",
        "source_asset_ids": [timeseries_asset_id, tracker_asset_id],
        "etm_assignment_artifact_id": etm_assignment_artifact_id,
        "wearer_role_id": wearer_role_id,
        "profile_path": str(profile_file),
        "profile_id": profile.get("profile_id"),
        "calibration_source": profile.get("calibration_source") or "unspecified",
        "hardware_id": profile.get("hardware_id"),
        "mount_version": profile.get("mount_version"),
        "valid_sessions": profile.get("valid_sessions") or [],
        "transfer_validation": profile.get("transfer_validation") or {},
        "calibration_acceptance": acceptance or {},
        "calibration_acceptance_artifact_id": calibration_acceptance_artifact_id,
        "require_calibration_acceptance": bool(require_calibration_acceptance),
        "clip_id": clip_id,
        "native_trim_artifact_id": native_trim_artifact_id,
        "scope": scope,
        "alignment": {
            "mode": alignment_mode,
            "basis": qc["alignment_basis"],
            "shared_start_offset_sec": float(shared_start_offset_sec),
            "timegraph_method": time_mapping.method if time_mapping is not None else None,
            "time_map_id": time_mapping.id if time_mapping is not None else None,
            "calibration_time_sync_artifact_id": handoff_time_sync_artifact_id,
            "distributed_drift_validated": bool((handoff_time_sync or {}).get("distributed_drift_validated")) if handoff_requires_time_sync else None,
            "ego_video_used_for_sync": False,
            "source_local_zero": "oculo_clip_video_ns=0" if scope == "native_trim" else None,
            "tracker_local_zero": float(tracker_data.time_sec[0]),
        },
        "calibrated_transform_used": {
            "T_tracker_from_camera": T_tracker_from_camera.tolist(),
            "camera_origin_tracker_mm": p_tc.tolist(),
        },
        "qc": qc,
        "tracker_compatibility": {
            "validated_by": "calibrated marker labels and marker template",
            "calibration_source_tracker_asset_id": profile_tracker_asset,
            "interaction_tracker_asset_id": tracker_asset_id,
            "different_source_file_expected": tracker_asset_differs_from_calibration_source,
        },
        "created_at": utc_now(),
    }
    atomic_write_json(output_root / "processing_manifest.json", manifest)
    atomic_write_json(output_root / "qc_metrics.json", qc)
    emit(progress, 0.95, "Registering gaze-ray artifact")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.GAZE_RAYS,
        label=(
            f"{trim_artifact.label} — 3D gaze rays"
            if trim_artifact is not None
            else f"{timeseries_asset.label} — 3D gaze rays"
        ),
        output_root=output_root,
        source_asset_ids=[timeseries_asset_id, tracker_asset_id],
        parent_artifact_ids=[
            artifact_id for artifact_id in
            [profile_artifact_id, etm_assignment_artifact_id, handoff_time_sync_artifact_id, native_trim_artifact_id]
            if artifact_id
        ],
        clip_id=clip_id,
        parameters={
            "scope": scope,
            "native_trim_artifact_id": native_trim_artifact_id,
            "alignment_mode": alignment_mode,
            "shared_start_offset_sec": float(shared_start_offset_sec),
            "max_pose_gap_sec": max_pose_gap_sec,
            "output_parquet": output_parquet,
            "etm_assignment_artifact_id": etm_assignment_artifact_id,
            "wearer_role_id": wearer_role_id,
            "calibration_time_sync_artifact_id": handoff_time_sync_artifact_id,
            "calibration_time_map_id": handoff_time_map_id,
            "calibration_handoff_requires_time_sync": bool(handoff_requires_time_sync),
        },
        qc=qc,
        provenance={
            "calibration_profile_path": str(profile_file),
            "alignment_basis": qc["alignment_basis"],
            "native_trim_artifact_id": native_trim_artifact_id,
            "calibration_time_sync_artifact_id": handoff_time_sync_artifact_id,
            "calibration_time_map_id": handoff_time_map_id,
            "ego_video_used_for_sync": False,
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "3D gaze-ray export complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "csv": str(output_csv),
        "parquet": str(output_parquet_path) if output_parquet_path else None,
        "manifest": manifest,
    }
