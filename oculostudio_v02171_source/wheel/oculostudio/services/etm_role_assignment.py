from __future__ import annotations

import itertools
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from oculostudio.adapters.mocap import MocapData, read_mocap
from oculostudio.domain.models import ArtifactKind, AssetKind, ArtifactRecord
from oculostudio.errors import ProcessingError, ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, safe_name, utc_now

# Preferred Vicon head markers.  When at least three are present, only these
# four are used so ARIEL or accessory markers do not pull the head centre.
_PRIMARY_HEAD_SUFFIXES = ("LFHD", "LBHD", "RFHD", "RBHD")
_FALLBACK_HEAD_SUFFIXES = (
    "ARIEL", "LHEAD", "RHEAD", "FHEAD", "BHEAD", "HEADL", "HEADR", "HEADF", "HEADB",
)
_ROLE_PATTERN = re.compile(r"(?:^|[_\-\s])R(\d{1,3})(?:$|[_\-\s])", re.IGNORECASE)
_ETM_PATTERN = re.compile(r"(?:^|[_\-\s])ETM[_\-\s]*([A-Z0-9]+)(?:$|[_\-\s])", re.IGNORECASE)


@dataclass(frozen=True)
class CentreSeries:
    asset_id: str
    label: str
    identifier: str
    subject_prefix: str | None
    marker_labels: tuple[str, ...]
    marker_suffixes: tuple[str, ...]
    frame_numbers: np.ndarray
    time_sec: np.ndarray
    centres_mm: np.ndarray
    valid: np.ndarray
    frame_rate: float
    first_frame: int
    marker_valid_fraction: float
    fingerprint_labels: tuple[str, ...]
    fingerprint_vector_mm: np.ndarray
    fingerprint_pair_names: tuple[str, ...]


@dataclass(frozen=True)
class ProfileFingerprint:
    artifact_id: str
    label: str
    profile_path: str
    profile_id: str | None
    participant_id: str | None
    tracker_prefix: str | None
    marker_labels: tuple[str, ...]
    marker_suffixes: tuple[str, ...]
    fingerprint_vector_mm: np.ndarray
    fingerprint_pair_names: tuple[str, ...]


def _label_parts(label: str) -> tuple[str | None, str]:
    text = str(label).strip()
    if ":" in text:
        prefix, suffix = text.rsplit(":", 1)
        return prefix.strip() or None, suffix.strip().upper().replace(" ", "")
    return None, text.upper().replace(" ", "")


def _normalise_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper().replace(" ", "") or None


def _role_from_asset(label: str, participant_id: str | None, fallback_index: int) -> str:
    for candidate in (participant_id, label):
        if candidate:
            match = _ROLE_PATTERN.search(str(candidate))
            if match:
                return f"R{int(match.group(1)):02d}"
    if participant_id and str(participant_id).strip():
        return str(participant_id).strip()
    return f"R{fallback_index + 1:02d}"


def _etm_name_from_asset(label: str, fallback_index: int) -> str:
    match = _ETM_PATTERN.search(str(label))
    return f"ETM_{match.group(1).upper()}" if match else f"ETM_{chr(ord('A') + fallback_index)}"


def _candidate_prefix_groups(labels: list[str], suffixes: set[str] | None = None) -> dict[str | None, list[int]]:
    groups: dict[str | None, list[int]] = {}
    for index, label in enumerate(labels):
        prefix, suffix = _label_parts(label)
        if suffixes is not None and suffix not in suffixes:
            continue
        groups.setdefault(prefix, []).append(index)
    return groups


def _choose_body_head_markers(data: MocapData, requested_prefix: str | None) -> tuple[str | None, list[int]]:
    requested = _normalise_prefix(requested_prefix)
    primary_groups = _candidate_prefix_groups(data.labels, set(_PRIMARY_HEAD_SUFFIXES))
    fallback_groups = _candidate_prefix_groups(data.labels, set(_FALLBACK_HEAD_SUFFIXES))

    if requested is not None:
        primary = [i for p, rows in primary_groups.items() if _normalise_prefix(p) == requested for i in rows]
        fallback = [i for p, rows in fallback_groups.items() if _normalise_prefix(p) == requested for i in rows]
        rows = primary if len(primary) >= 3 else primary + fallback
        if len(rows) < 3:
            raise ValidationError(
                f"Body marker prefix {requested_prefix!r} does not provide at least three recognised head markers"
            )
        prefix = next((p for p in set(primary_groups) | set(fallback_groups) if _normalise_prefix(p) == requested), requested_prefix)
        return prefix, sorted(set(rows))

    viable: list[tuple[int, int, str | None, list[int]]] = []
    all_prefixes = set(primary_groups) | set(fallback_groups)
    for prefix in all_prefixes:
        primary = primary_groups.get(prefix, [])
        fallback = fallback_groups.get(prefix, [])
        rows = primary if len(primary) >= 3 else primary + fallback
        if len(rows) >= 3:
            # Prefer more primary markers, then more total markers.
            viable.append((len(primary), len(set(rows)), prefix, sorted(set(rows))))
    if not viable:
        raise ValidationError(
            "No subject has at least three recognised head markers. Expected LFHD/LBHD/RFHD/RBHD or configured aliases."
        )
    viable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = viable[0]
    tied = [row for row in viable if row[:2] == best[:2]]
    if len(tied) > 1:
        names = [row[2] or "<unprefixed>" for row in tied]
        raise ValidationError(
            f"Body C3D contains multiple equally plausible head-marker subjects {names}; provide a body marker prefix."
        )
    return best[2], best[3]


def _choose_etm_markers(data: MocapData, requested_prefix: str | None) -> tuple[str | None, list[int]]:
    requested = _normalise_prefix(requested_prefix)
    groups = _candidate_prefix_groups(data.labels)
    if requested is not None:
        matches = [(prefix, rows) for prefix, rows in groups.items() if _normalise_prefix(prefix) == requested]
        if len(matches) != 1 or len(matches[0][1]) < 3:
            raise ValidationError(
                f"ETM marker prefix {requested_prefix!r} does not uniquely identify at least three markers"
            )
        return matches[0][0], matches[0][1]

    viable = [(len(rows), prefix, rows) for prefix, rows in groups.items() if len(rows) >= 3]
    if not viable:
        raise ValidationError("ETM C3D contains fewer than three markers")
    viable.sort(key=lambda item: item[0], reverse=True)
    best = viable[0]
    tied = [row for row in viable if row[0] == best[0]]
    # A normal ETM file has one six-marker prefix.  Fail rather than averaging
    # unrelated rigid bodies if a world C3D was selected accidentally.
    if len(tied) > 1:
        names = [row[1] or "<unprefixed>" for row in tied]
        raise ValidationError(
            f"ETM C3D contains multiple marker groups with {best[0]} markers: {names}; select the dedicated ETM file or provide a prefix."
        )
    return best[1], best[2]


def _centres(points: np.ndarray, valid: np.ndarray, minimum_markers: int) -> tuple[np.ndarray, np.ndarray]:
    counts = valid.sum(axis=1)
    finite = counts >= minimum_markers
    weighted = np.where(valid[:, :, None], points, 0.0).sum(axis=1)
    centre = np.full((len(points), 3), np.nan, dtype=float)
    centre[finite] = weighted[finite] / counts[finite, None]
    return centre, finite


def _pairwise_fingerprint(points: np.ndarray, valid: np.ndarray, labels: list[str]) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    suffixes = tuple(_label_parts(label)[1] for label in labels)
    values: list[float] = []
    names: list[str] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            mask = valid[:, i] & valid[:, j]
            if int(mask.sum()) < 3:
                values.append(float("nan"))
            else:
                distance = np.linalg.norm(points[mask, i] - points[mask, j], axis=1)
                values.append(float(np.median(distance)))
            names.append(f"{suffixes[i]}|{suffixes[j]}")
    return np.asarray(values, dtype=float), tuple(names), suffixes


def _centre_series(
    asset_id: str,
    label: str,
    identifier: str,
    path: str,
    *,
    mode: str,
    marker_prefix: str | None,
) -> CentreSeries:
    data = read_mocap(path)
    if mode == "body":
        prefix, indices = _choose_body_head_markers(data, marker_prefix)
        minimum = min(3, len(indices))
    elif mode == "etm":
        prefix, indices = _choose_etm_markers(data, marker_prefix)
        minimum = 3
    else:
        raise ValueError(mode)
    points = data.points_mm[:, indices]
    valid = data.valid[:, indices]
    centres, centre_valid = _centres(points, valid, minimum)
    labels = [data.labels[index] for index in indices]
    fingerprint, pair_names, suffixes = _pairwise_fingerprint(points, valid, labels)
    frame_numbers = data.first_frame + np.arange(data.frame_count, dtype=np.int64)
    return CentreSeries(
        asset_id=asset_id,
        label=label,
        identifier=identifier,
        subject_prefix=prefix,
        marker_labels=tuple(labels),
        marker_suffixes=suffixes,
        frame_numbers=frame_numbers,
        time_sec=np.asarray(data.time_sec, dtype=float),
        centres_mm=centres,
        valid=centre_valid,
        frame_rate=float(data.frame_rate),
        first_frame=int(data.first_frame),
        marker_valid_fraction=float(valid.mean()) if valid.size else 0.0,
        fingerprint_labels=suffixes,
        fingerprint_vector_mm=fingerprint,
        fingerprint_pair_names=pair_names,
    )


def _distance_stats(body: CentreSeries, etm: CentreSeries, frame_rate_tolerance: float = 1e-6) -> dict[str, Any]:
    if not math.isclose(body.frame_rate, etm.frame_rate, rel_tol=frame_rate_tolerance, abs_tol=frame_rate_tolerance):
        raise ValidationError(
            f"Frame-rate mismatch between {body.label} ({body.frame_rate:g} Hz) and {etm.label} ({etm.frame_rate:g} Hz)"
        )
    common, body_index, etm_index = np.intersect1d(
        body.frame_numbers, etm.frame_numbers, assume_unique=True, return_indices=True
    )
    valid = body.valid[body_index] & etm.valid[etm_index]
    if int(valid.sum()) < 10:
        raise ValidationError(
            f"Fewer than ten shared valid head/ETM frames for {body.label} and {etm.label}"
        )
    distance = np.linalg.norm(
        body.centres_mm[body_index[valid]] - etm.centres_mm[etm_index[valid]], axis=1
    )
    return {
        "n": int(len(distance)),
        "median_mm": float(np.median(distance)),
        "mean_mm": float(np.mean(distance)),
        "std_mm": float(np.std(distance)),
        "p05_mm": float(np.percentile(distance, 5)),
        "p95_mm": float(np.percentile(distance, 95)),
        "min_mm": float(np.min(distance)),
        "max_mm": float(np.max(distance)),
        "common_frame_start": int(common[0]) if len(common) else None,
        "common_frame_end": int(common[-1]) if len(common) else None,
        "common_frame_count": int(len(common)),
    }


def _invalid_gap_metrics(valid: np.ndarray, frame_rate: float) -> dict[str, Any]:
    invalid = ~np.asarray(valid, dtype=bool)
    runs: list[int] = []
    start: int | None = None
    for index, value in enumerate(invalid):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append(index - start)
            start = None
    if start is not None:
        runs.append(len(invalid) - start)
    return {
        "valid_frames": int((~invalid).sum()),
        "total_frames": int(len(invalid)),
        "valid_fraction": float((~invalid).mean()) if len(invalid) else 0.0,
        "missing_frames": int(invalid.sum()),
        "gap_count": int(len(runs)),
        "maximum_gap_frames": int(max(runs, default=0)),
        "maximum_gap_ms": float(1000.0 * max(runs, default=0) / frame_rate) if frame_rate else None,
        "single_frame_gap_fraction": float(sum(length == 1 for length in runs) / len(runs)) if runs else 0.0,
    }


def _profile_file(artifact: ArtifactRecord) -> Path | None:
    candidates = [Path(row.path) for row in artifact.files if Path(row.path).exists()]
    for suffix in (".oculocal", ".g3dcal", ".json"):
        matches = [path for path in candidates if path.suffix.lower() == suffix]
        if matches:
            matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
            return matches[0]
    return None


def _load_profile_fingerprint(artifact: ArtifactRecord) -> ProfileFingerprint | None:
    path = _profile_file(artifact)
    if path is None:
        return None
    try:
        if path.suffix.lower() == ".g3dcal":
            with zipfile.ZipFile(path) as archive:
                metadata = json.loads(archive.read("profile.json"))
                with archive.open("arrays.npz") as handle:
                    # ZipExtFile is not seekable enough for every numpy version.
                    import io
                    arrays = np.load(io.BytesIO(handle.read()))
                    template = np.asarray(arrays["tracker_template_mm"], dtype=float)
            labels = list(metadata.get("tracker_marker_labels") or [])
            participant = metadata.get("canonical_participant_id")
            profile_id = metadata.get("profile_id")
        else:
            metadata = json.loads(path.read_text("utf-8"))
            tracker = metadata.get("tracker") or {}
            template = np.asarray(tracker.get("marker_template_mm"), dtype=float)
            labels = list(tracker.get("marker_labels") or metadata.get("tracker_marker_labels") or [])
            participant = metadata.get("participant_id") or metadata.get("canonical_participant_id")
            profile_id = metadata.get("profile_id")
        if template.ndim != 2 or template.shape[1] != 3 or len(template) < 3:
            return None
        if len(labels) != len(template):
            labels = [f"M_{index+1}" for index in range(len(template))]
        dummy = np.repeat(template[None, :, :], 3, axis=0)
        valid = np.ones(dummy.shape[:2], dtype=bool)
        fingerprint, names, suffixes = _pairwise_fingerprint(dummy, valid, labels)
        prefixes = {_label_parts(label)[0] for label in labels if _label_parts(label)[0]}
        return ProfileFingerprint(
            artifact_id=artifact.id,
            label=artifact.label,
            profile_path=str(path),
            profile_id=profile_id,
            participant_id=participant,
            tracker_prefix=next(iter(prefixes)) if len(prefixes) == 1 else None,
            marker_labels=tuple(labels),
            marker_suffixes=suffixes,
            fingerprint_vector_mm=fingerprint,
            fingerprint_pair_names=names,
        )
    except Exception:
        return None


def _shape_difference(etm: CentreSeries, profile: ProfileFingerprint) -> dict[str, Any]:
    a = etm.fingerprint_vector_mm
    b = profile.fingerprint_vector_mm
    sorted_a = np.sort(a[np.isfinite(a)])
    sorted_b = np.sort(b[np.isfinite(b)])
    if len(sorted_a) != len(sorted_b) or not len(sorted_a):
        sorted_rms = float("inf")
        sorted_max = float("inf")
    else:
        delta = sorted_a - sorted_b
        sorted_rms = float(np.sqrt(np.mean(delta * delta)))
        sorted_max = float(np.max(np.abs(delta)))

    label_rms = None
    label_max = None
    etm_map = {name: value for name, value in zip(etm.fingerprint_pair_names, a) if np.isfinite(value)}
    profile_map = {name: value for name, value in zip(profile.fingerprint_pair_names, b) if np.isfinite(value)}
    common = sorted(set(etm_map) & set(profile_map))
    if common and len(common) == len(etm_map) == len(profile_map):
        delta = np.asarray([etm_map[name] - profile_map[name] for name in common], dtype=float)
        label_rms = float(np.sqrt(np.mean(delta * delta)))
        label_max = float(np.max(np.abs(delta)))
    score = label_rms if label_rms is not None else sorted_rms
    return {
        "profile_artifact_id": profile.artifact_id,
        "profile_label": profile.label,
        "profile_path": profile.profile_path,
        "profile_id": profile.profile_id,
        "participant_id": profile.participant_id,
        "tracker_prefix": profile.tracker_prefix,
        "pairwise_distance_rms_mm": float(score),
        "label_aligned_rms_mm": label_rms,
        "label_aligned_max_abs_mm": label_max,
        "sorted_pairwise_rms_mm": sorted_rms,
        "sorted_pairwise_max_abs_mm": sorted_max,
        "marker_count": len(profile.marker_labels),
    }


def _assignment_confidence(
    assigned_distance: float,
    alternate_distance: float | None,
    *,
    maximum_head_distance_mm: float,
    minimum_role_margin_mm: float,
    minimum_role_ratio: float,
    global_margin_mm: float | None,
    minimum_global_margin_mm: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if assigned_distance > maximum_head_distance_mm:
        reasons.append(
            f"Assigned ETM-to-head median distance {assigned_distance:.1f} mm exceeds {maximum_head_distance_mm:.1f} mm"
        )
        return "LOW", reasons
    if alternate_distance is not None:
        margin = alternate_distance - assigned_distance
        ratio = alternate_distance / max(assigned_distance, 1e-9)
        if margin < minimum_role_margin_mm or ratio < minimum_role_ratio:
            reasons.append(
                f"Alternative role is insufficiently separated (margin {margin:.1f} mm, ratio {ratio:.2f})"
            )
            return "AMBIGUOUS", reasons
    if global_margin_mm is not None and global_margin_mm < minimum_global_margin_mm:
        reasons.append(
            f"Global one-to-one assignment margin {global_margin_mm:.1f} mm is below {minimum_global_margin_mm:.1f} mm"
        )
        return "AMBIGUOUS", reasons
    if assigned_distance <= 200.0:
        reasons.append("ETM cluster stays close to the assigned head across the shared valid frames")
        return "HIGH", reasons
    reasons.append("Assignment is geometrically separated but ETM-to-head distance is larger than the preferred range")
    return "MEDIUM", reasons


def load_etm_assignment(artifact: ArtifactRecord) -> dict[str, Any]:
    if artifact.kind != ArtifactKind.ETM_ROLE_ASSIGNMENT:
        raise ValidationError("Artifact is not an ETM role assignment")
    path = find_artifact_file(artifact, names=("etm_role_assignment.json",))
    return json.loads(path.read_text("utf-8"))


def resolve_assignment_for_role(artifact: ArtifactRecord, role_id: str) -> dict[str, Any]:
    data = load_etm_assignment(artifact)
    if data.get("status") != "resolved":
        raise ValidationError(
            "ETM-role assignment requires review and cannot drive Gaze 3D automatically"
        )
    assignments = data.get("assignments") or {}
    if role_id not in assignments:
        raise ValidationError(
            f"Role {role_id!r} is not present in ETM assignment {artifact.label}; available roles: {sorted(assignments)}"
        )
    row = dict(assignments[role_id])
    row["role_id"] = role_id
    return row


def create_etm_role_assignment(
    store: ProjectStore,
    *,
    session_id: str,
    body_asset_ids: list[str],
    etm_asset_ids: list[str],
    role_labels: dict[str, str] | None = None,
    body_marker_prefixes: dict[str, str] | None = None,
    etm_marker_prefixes: dict[str, str] | None = None,
    calibration_profile_artifact_ids: list[str] | None = None,
    maximum_head_distance_mm: float = 350.0,
    minimum_role_margin_mm: float = 100.0,
    minimum_role_ratio: float = 1.5,
    minimum_global_margin_mm: float = 100.0,
    maximum_profile_shape_rms_mm: float = 8.0,
    minimum_profile_shape_margin_mm: float = 4.0,
    label: str = "ETM role assignment",
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    role_labels = role_labels or {}
    body_marker_prefixes = body_marker_prefixes or {}
    etm_marker_prefixes = etm_marker_prefixes or {}
    calibration_profile_artifact_ids = calibration_profile_artifact_ids or []
    session = store.project.sessions[session_id]

    if len(body_asset_ids) < 2 or len(etm_asset_ids) < 2:
        raise ValidationError("Select at least two role/body C3Ds and two ETM C3Ds")
    if len(body_asset_ids) != len(etm_asset_ids):
        raise ValidationError("The number of role/body assets must equal the number of ETM candidates")
    if len(set(body_asset_ids)) != len(body_asset_ids) or len(set(etm_asset_ids)) != len(etm_asset_ids):
        raise ValidationError("Body and ETM asset selections must not contain duplicates")
    if len(body_asset_ids) > 8:
        raise ValidationError("This resolver supports at most eight simultaneous roles")

    emit(progress, 0.03, "Reading role/body C3Ds and head markers")
    bodies: list[CentreSeries] = []
    for index, asset_id in enumerate(body_asset_ids):
        token.raise_if_cancelled()
        asset = session.assets.get(asset_id)
        if asset is None or asset.kind not in {AssetKind.C3D, AssetKind.MOCAP_NPZ}:
            raise ValidationError(f"Body asset {asset_id} is not a C3D / mocap NPZ")
        role = role_labels.get(asset_id) or _role_from_asset(asset.label, asset.participant_id, index)
        bodies.append(
            _centre_series(
                asset.id,
                asset.label,
                role,
                asset.source_path,
                mode="body",
                marker_prefix=body_marker_prefixes.get(asset.id),
            )
        )
        emit(progress, 0.03 + 0.18 * (index + 1) / len(body_asset_ids), f"Read {role} head trajectory")

    role_ids = [item.identifier for item in bodies]
    if len(set(role_ids)) != len(role_ids):
        raise ValidationError(f"Role labels are not unique: {role_ids}")

    emit(progress, 0.23, "Reading ETM rigid-body candidates")
    etms: list[CentreSeries] = []
    for index, asset_id in enumerate(etm_asset_ids):
        token.raise_if_cancelled()
        asset = session.assets.get(asset_id)
        if asset is None or asset.kind not in {AssetKind.C3D, AssetKind.MOCAP_NPZ}:
            raise ValidationError(f"ETM asset {asset_id} is not a C3D / mocap NPZ")
        etm_name = _etm_name_from_asset(asset.label, index)
        etms.append(
            _centre_series(
                asset.id,
                asset.label,
                etm_name,
                asset.source_path,
                mode="etm",
                marker_prefix=etm_marker_prefixes.get(asset.id),
            )
        )
        emit(progress, 0.23 + 0.18 * (index + 1) / len(etm_asset_ids), f"Read {etm_name} marker cluster")

    rates = [item.frame_rate for item in [*bodies, *etms]]
    if max(rates) - min(rates) > 1e-6:
        raise ValidationError(f"All clip C3Ds must share one Vicon frame rate; found {rates}")

    emit(progress, 0.43, "Computing ETM-to-head distance matrix")
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    costs = np.zeros((len(etms), len(bodies)), dtype=float)
    for i, etm in enumerate(etms):
        stats[etm.identifier] = {}
        for j, body in enumerate(bodies):
            token.raise_if_cancelled()
            row = _distance_stats(body, etm)
            stats[etm.identifier][body.identifier] = row
            costs[i, j] = row["median_mm"]

    row_ind, col_ind = linear_sum_assignment(costs)
    if len(row_ind) != len(etms):
        raise ProcessingError("Could not obtain a complete one-to-one ETM/role assignment")
    best_total = float(costs[row_ind, col_ind].sum())
    ranked_totals: list[tuple[float, tuple[int, ...]]] = []
    for permutation in itertools.permutations(range(len(bodies))):
        total = float(sum(costs[i, permutation[i]] for i in range(len(etms))))
        ranked_totals.append((total, permutation))
    ranked_totals.sort(key=lambda item: item[0])
    second_total = ranked_totals[1][0] if len(ranked_totals) > 1 else None
    global_margin = second_total - best_total if second_total is not None else None

    profiles: list[ProfileFingerprint] = []
    for artifact_id in calibration_profile_artifact_ids:
        artifact = session.artifacts.get(artifact_id)
        if artifact is None or artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
            raise ValidationError(f"Calibration artifact {artifact_id} is not available")
        fingerprint = _load_profile_fingerprint(artifact)
        if fingerprint is not None:
            profiles.append(fingerprint)

    shape_checks: dict[str, list[dict[str, Any]]] = {}
    profile_recommendations: dict[str, dict[str, Any] | None] = {}
    for etm in etms:
        checks = sorted((_shape_difference(etm, profile) for profile in profiles), key=lambda row: row["pairwise_distance_rms_mm"])
        shape_checks[etm.identifier] = checks
        recommendation = None
        if checks:
            best = checks[0]
            next_score = checks[1]["pairwise_distance_rms_mm"] if len(checks) > 1 else None
            margin = next_score - best["pairwise_distance_rms_mm"] if next_score is not None else None
            accepted = best["pairwise_distance_rms_mm"] <= maximum_profile_shape_rms_mm
            separated = margin is None or margin >= minimum_profile_shape_margin_mm
            recommendation = {
                **best,
                "accepted": bool(accepted and separated),
                "shape_margin_mm": margin,
                "reason": (
                    "Unique low-RMS tracker-shape match"
                    if accepted and separated
                    else "No unique calibration-profile shape match"
                ),
            }
        profile_recommendations[etm.identifier] = recommendation

    emit(progress, 0.70, "Applying one-to-one identity and ambiguity gates")
    assignments: dict[str, dict[str, Any]] = {}
    assignment_by_etm: dict[str, dict[str, Any]] = {}
    confidence_levels: list[str] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        etm = etms[i]
        body = bodies[j]
        assigned_distance = float(costs[i, j])
        alternate = min((float(costs[i, k]) for k in range(len(bodies)) if k != j), default=None)
        confidence, reasons = _assignment_confidence(
            assigned_distance,
            alternate,
            maximum_head_distance_mm=maximum_head_distance_mm,
            minimum_role_margin_mm=minimum_role_margin_mm,
            minimum_role_ratio=minimum_role_ratio,
            global_margin_mm=global_margin,
            minimum_global_margin_mm=minimum_global_margin_mm,
        )
        confidence_levels.append(confidence)
        recommendation = profile_recommendations.get(etm.identifier)
        row = {
            "role_id": body.identifier,
            "body_asset_id": body.asset_id,
            "body_asset_label": body.label,
            "body_subject_prefix": body.subject_prefix,
            "body_head_marker_labels": list(body.marker_labels),
            "etm_asset_id": etm.asset_id,
            "etm_asset_label": etm.label,
            "etm_id": etm.identifier,
            "tracker_prefix": etm.subject_prefix,
            "tracker_marker_labels": list(etm.marker_labels),
            "median_head_distance_mm": assigned_distance,
            "alternative_head_distance_mm": alternate,
            "distance_margin_mm": (alternate - assigned_distance) if alternate is not None else None,
            "distance_ratio": (alternate / assigned_distance) if alternate is not None and assigned_distance > 0 else None,
            "confidence": confidence,
            "reasons": reasons,
            "recommended_calibration_artifact_id": (
                recommendation["profile_artifact_id"] if recommendation and recommendation.get("accepted") else None
            ),
            "recommended_calibration_profile_path": (
                recommendation["profile_path"] if recommendation and recommendation.get("accepted") else None
            ),
            "calibration_shape_check": recommendation,
        }
        assignments[body.identifier] = row
        assignment_by_etm[etm.identifier] = row

    blocked = [row for row in assignments.values() if row["confidence"] in {"LOW", "AMBIGUOUS"}]
    status = "resolved" if not blocked else "review_required"

    frame_rate = rates[0]
    body_rows = [
        {
            "role_id": item.identifier,
            "asset_id": item.asset_id,
            "asset_label": item.label,
            "subject_prefix": item.subject_prefix,
            "head_marker_labels": list(item.marker_labels),
            "frame_count": int(len(item.frame_numbers)),
            "frame_rate_hz": item.frame_rate,
            "first_frame": item.first_frame,
            "head_valid_fraction": float(item.valid.mean()),
            "marker_valid_fraction": item.marker_valid_fraction,
        }
        for item in bodies
    ]
    etm_rows = []
    for item in etms:
        gap = _invalid_gap_metrics(item.valid, item.frame_rate)
        etm_rows.append(
            {
                "etm_id": item.identifier,
                "asset_id": item.asset_id,
                "asset_label": item.label,
                "tracker_prefix": item.subject_prefix,
                "marker_labels": list(item.marker_labels),
                "frame_count": int(len(item.frame_numbers)),
                "frame_rate_hz": item.frame_rate,
                "first_frame": item.first_frame,
                "marker_valid_fraction": item.marker_valid_fraction,
                "pose_proxy_coverage": gap,
            }
        )

    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "etm_assignment", "clip") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    matrix_rows = []
    for etm in etms:
        for body in bodies:
            matrix_rows.append({
                "etm_id": etm.identifier,
                "etm_asset_id": etm.asset_id,
                "role_id": body.identifier,
                "body_asset_id": body.asset_id,
                **stats[etm.identifier][body.identifier],
            })
    distance_csv = output_root / "etm_head_distance_matrix.csv"
    pd.DataFrame(matrix_rows).to_csv(distance_csv, index=False)
    shape_csv = output_root / "tracker_shape_comparison.csv"
    shape_rows = [
        {"etm_id": etm_id, **row}
        for etm_id, rows in shape_checks.items()
        for row in rows
    ]
    pd.DataFrame(shape_rows).to_csv(shape_csv, index=False)

    result = {
        "schema_version": "oculostudio-etm-role-assignment-1.0",
        "artifact_id": artifact_id,
        "status": status,
        "method": "global_one_to_one_median_head_proximity",
        "frame_rate_hz": frame_rate,
        "role_assets": body_rows,
        "etm_assets": etm_rows,
        "distance_statistics_mm": stats,
        "assignment_cost": {
            "best_total_median_distance_mm": best_total,
            "second_best_total_median_distance_mm": second_total,
            "global_margin_mm": global_margin,
        },
        "assignments": assignments,
        "assignment_by_etm": assignment_by_etm,
        "tracker_shape_comparisons": shape_checks,
        "thresholds": {
            "maximum_head_distance_mm": maximum_head_distance_mm,
            "minimum_role_margin_mm": minimum_role_margin_mm,
            "minimum_role_ratio": minimum_role_ratio,
            "minimum_global_margin_mm": minimum_global_margin_mm,
            "maximum_profile_shape_rms_mm": maximum_profile_shape_rms_mm,
            "minimum_profile_shape_margin_mm": minimum_profile_shape_margin_mm,
        },
        "warnings": (
            ["One or more assignments are ambiguous or too far from the head; do not use them for Gaze 3D until reviewed."]
            if blocked else []
        ),
        "created_at": utc_now(),
    }
    result_path = output_root / "etm_role_assignment.json"
    atomic_write_json(result_path, result)
    report_path = output_root / "etm_role_assignment_report.html"
    report_path.write_text(_assignment_html(result, matrix_rows), "utf-8")

    qc = {
        "status": status,
        "role_count": len(bodies),
        "etm_count": len(etms),
        "resolved_assignment_count": int(sum(row["confidence"] not in {"LOW", "AMBIGUOUS"} for row in assignments.values())),
        "high_confidence_fraction": float(sum(row["confidence"] == "HIGH" for row in assignments.values()) / len(assignments)),
        "best_total_median_distance_mm": best_total,
        "global_assignment_margin_mm": global_margin,
        "maximum_assigned_head_distance_mm": float(max(row["median_head_distance_mm"] for row in assignments.values())),
        "profile_recommendation_count": int(sum(bool(row["recommended_calibration_artifact_id"]) for row in assignments.values())),
        "wearer_assignment_complete": status == "resolved",
        "handoff_complete": bool(assignments) and all(bool(row.get("recommended_calibration_artifact_id")) for row in assignments.values()),
        "assignments_summary": {
            role_id: {
                "etm_id": row["etm_id"],
                "tracker_prefix": row["tracker_prefix"],
                "etm_asset_id": row["etm_asset_id"],
                "body_asset_id": row["body_asset_id"],
                "body_subject_prefix": row["body_subject_prefix"],
                "median_head_distance_mm": row["median_head_distance_mm"],
                "confidence": row["confidence"],
                "recommended_calibration_artifact_id": row["recommended_calibration_artifact_id"],
                "recommended_calibration_participant_id": (row.get("calibration_shape_check") or {}).get("participant_id"),
                "calibration_shape_rms_mm": (row.get("calibration_shape_check") or {}).get("pairwise_distance_rms_mm"),
            }
            for role_id, row in assignments.items()
        },
    }
    source_asset_ids = [*body_asset_ids, *etm_asset_ids]
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
        label=label,
        output_root=output_root,
        source_asset_ids=source_asset_ids,
        parent_artifact_ids=calibration_profile_artifact_ids,
        parameters={
            "calibration_phase": "wearer_assignment",
            "role_labels": role_labels,
            "body_marker_prefixes": body_marker_prefixes,
            "etm_marker_prefixes": etm_marker_prefixes,
            **result["thresholds"],
        },
        qc=qc,
        provenance={"method": result["method"]},
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "ETM wearer assignment complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "assignment_json": str(result_path),
        "distance_csv": str(distance_csv),
        "shape_csv": str(shape_csv),
        "report": str(report_path),
        "result": result,
    }



def _time_sync_summary_for_handoff(
    session,
    *,
    source_assignments: dict[str, dict[str, Any]],
    source_assignment_artifact_id: str,
    artifact_ids: list[str],
    require_distributed_drift: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate accepted Pupil↔ETM mappings against the immutable wearer assignment.

    Time synchronization is estimated from Pupil IMU and the assigned ETM C3D;
    ego video is deliberately not an input.  Precision-audited artifacts carry the
    role and time-map IDs explicitly.  Older automatic-sync artifacts can still
    be interpreted by matching their ETM source asset to the resolved role, but
    they do not bypass the distributed-drift gate.
    """
    unique_ids = list(dict.fromkeys(str(value) for value in artifact_ids if str(value)))
    by_role: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    target_to_role = {
        str(row.get("etm_asset_id") or ""): str(role_id)
        for role_id, row in source_assignments.items()
        if row.get("etm_asset_id")
    }
    for artifact_id in unique_ids:
        artifact = session.artifacts.get(artifact_id)
        if artifact is None or artifact.kind != ArtifactKind.TIME_SYNC:
            raise ValidationError(f"Time-synchronization artifact {artifact_id!r} is unavailable")
        parameters = artifact.parameters or {}
        qc = artifact.qc or {}
        source_asset_id = str(parameters.get("source_asset_id") or "")
        target_asset_id = str(parameters.get("target_asset_id") or "")
        if not source_asset_id or not target_asset_id:
            source_rows = [str(value) for value in artifact.source_asset_ids if str(value)]
            assigned_targets = set(target_to_role)
            target_matches = [value for value in source_rows if value in assigned_targets]
            if not target_asset_id and len(target_matches) == 1:
                target_asset_id = target_matches[0]
            if not source_asset_id:
                source_candidates = [value for value in source_rows if value != target_asset_id]
                if len(source_candidates) == 1:
                    source_asset_id = source_candidates[0]
        role_id = str(parameters.get("wearer_role_id") or qc.get("wearer_role_id") or "").upper()
        if not role_id and target_asset_id:
            role_id = target_to_role.get(target_asset_id, "").upper()
        if role_id not in source_assignments:
            raise ValidationError(
                f"Time synchronization {artifact.label!r} cannot be matched to a resolved wearer role"
            )
        expected_target = str(source_assignments[role_id].get("etm_asset_id") or "")
        if not target_asset_id or target_asset_id != expected_target:
            raise ValidationError(
                f"Time synchronization for {role_id} targets the wrong ETM C3D; rerun it from Calibration"
            )
        declared_assignment = str(parameters.get("assignment_artifact_id") or "")
        if declared_assignment and declared_assignment != source_assignment_artifact_id:
            raise ValidationError(
                f"Time synchronization for {role_id} belongs to a different wearer assignment; rerun it from Calibration"
            )
        time_map_id = str(parameters.get("time_map_id") or qc.get("time_map_id") or "")
        if not time_map_id and source_asset_id in session.assets and target_asset_id in session.assets:
            source_domain = session.assets[source_asset_id].primary_time_domain_id
            target_domain = session.assets[target_asset_id].primary_time_domain_id
            compatible = [
                row for row in session.time_maps.values()
                if row.source_domain_id == source_domain
                and row.target_domain_id == target_domain
                and str(row.method).startswith("auto_imu")
            ]
            compatible.sort(key=lambda row: str(row.created_at))
            if compatible:
                time_map_id = compatible[-1].id
        mapping = session.time_maps.get(time_map_id) if time_map_id else None
        confidence = str(qc.get("confidence") or "").upper()
        accepted_by_user = bool(parameters.get("accepted_by_user", qc.get("accepted_by_user", False)))
        matched_windows = int(qc.get("matched_window_count") or 0)
        anchor_count = int(qc.get("anchor_count") or (len(mapping.anchors) if mapping else 0))
        sync_verified = bool(
            qc.get("sync_verified")
            if "sync_verified" in qc
            else confidence == "HIGH" or (confidence == "REVIEW" and accepted_by_user)
        )
        distributed = bool(
            qc.get("distributed_drift_validated")
            if "distributed_drift_validated" in qc
            else matched_windows >= 2 and anchor_count >= 2
        )
        ego_video_used = bool(parameters.get("ego_video_used", qc.get("ego_video_used", False)))
        precision_validated = bool(
            qc.get("precision_validated")
            and int(qc.get("holdout_window_count") or 0) >= 3
            and qc.get("drift_ci95_ppm") is not None
            and qc.get("residual_validation_mode") == "leave_one_window_out"
        )
        production_ready = bool(
            mapping is not None
            and sync_verified
            and not ego_video_used
            and precision_validated
            and (distributed or not require_distributed_drift)
        )
        if role_id in by_role:
            raise ValidationError(f"Select only one time-synchronization artifact for {role_id}")
        by_role[role_id] = {
            "artifact_id": artifact.id,
            "artifact_label": artifact.label,
            "source_asset_id": source_asset_id,
            "target_asset_id": target_asset_id,
            "time_map_id": time_map_id or None,
            "confidence": confidence or None,
            "accepted_by_user": accepted_by_user,
            "sync_verified": sync_verified,
            "distributed_drift_validated": distributed,
            "precision_validated": precision_validated,
            "production_ready": production_ready,
            "ego_video_used": ego_video_used,
            "estimated_offset_sec": qc.get("estimated_offset_sec"),
            "estimated_drift_ppm": qc.get("estimated_drift_ppm"),
            "matched_window_count": matched_windows,
            "window_count": int(qc.get("window_count") or 0),
            "p95_residual_ms": qc.get("p95_residual_ms"),
            "assignment_artifact_id": declared_assignment or None,
        }
        if not production_ready:
            reason = (
                "no saved TimeGraph mapping" if mapping is None
                else "ego-video-derived synchronization is not accepted" if ego_video_used
                else "independent precision audit is missing" if not precision_validated
                else "distributed drift is not verified"
            )
            warnings.append(f"{role_id} automatic synchronization is not production-ready: {reason}")
        if by_role[role_id]["ego_video_used"]:
            warnings.append(f"{role_id} synchronization provenance unexpectedly reports ego-video input")
    return by_role, warnings


def bind_calibration_profiles_to_assignment(
    store: ProjectStore,
    *,
    session_id: str,
    assignment_artifact_id: str,
    calibration_profile_artifact_ids: list[str],
    time_sync_artifact_ids: list[str] | None = None,
    require_time_sync: bool = False,
    require_distributed_drift: bool = True,
    maximum_profile_shape_rms_mm: float = 8.0,
    minimum_profile_shape_margin_mm: float = 4.0,
    label: str = "Calibration handoff",
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Bind physical ETM profiles to an already resolved wearer assignment.

    This deliberately does *not* recompute who wore which ETM. The wearer
    assignment is immutable input; this job only performs one-to-one tracker
    shape matching between the resolved interaction ETMs and selected hardware
    profiles, then emits a new handoff artifact for Gaze 3D.
    """
    token = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    source_assignment = session.artifacts.get(assignment_artifact_id)
    if source_assignment is None or source_assignment.kind != ArtifactKind.ETM_ROLE_ASSIGNMENT:
        raise ValidationError("Select a resolved ETM wearer assignment")
    source_result = load_etm_assignment(source_assignment)
    if source_result.get("status") != "resolved":
        raise ValidationError("The selected wearer assignment requires review before profile binding")
    source_assignments = source_result.get("assignments") or {}
    if not source_assignments:
        raise ValidationError("The selected wearer assignment contains no role mappings")

    sync_ids = list(dict.fromkeys(str(value) for value in (time_sync_artifact_ids or []) if str(value)))
    sync_by_role, sync_warnings = _time_sync_summary_for_handoff(
        session,
        source_assignments=source_assignments,
        source_assignment_artifact_id=assignment_artifact_id,
        artifact_ids=sync_ids,
        require_distributed_drift=require_distributed_drift,
    )
    if require_time_sync:
        missing_sync = [role_id for role_id in source_assignments if role_id not in sync_by_role]
        blocked_sync = [
            role_id for role_id in source_assignments
            if role_id in sync_by_role and not sync_by_role[role_id].get("production_ready")
        ]
        if missing_sync:
            raise ValidationError(
                "Calibration handoff needs an accepted Pupil IMU↔ETM time map for: " + ", ".join(sorted(missing_sync))
            )
        if blocked_sync:
            details = []
            for role_id in sorted(blocked_sync):
                row = sync_by_role.get(role_id) or {}
                if row.get("ego_video_used"):
                    reason = "ego-video-derived synchronization is not accepted"
                elif not row.get("time_map_id"):
                    reason = "no saved TimeGraph mapping"
                elif not row.get("sync_verified"):
                    reason = "mapping was not formally accepted"
                else:
                    reason = "distributed drift validation did not pass"
                details.append(f"{role_id} ({reason})")
            raise ValidationError(
                "Calibration handoff is blocked: " + ", ".join(details)
            )

    profile_ids = list(dict.fromkeys(str(value) for value in calibration_profile_artifact_ids if str(value)))
    etm_rows_by_id: dict[str, dict[str, Any]] = {}
    for role_id, row in source_assignments.items():
        etm_id = str(row.get("etm_id") or "").strip()
        asset_id = str(row.get("etm_asset_id") or "").strip()
        if not etm_id or not asset_id:
            raise ValidationError(f"Wearer assignment row {role_id!r} is missing ETM provenance")
        etm_rows_by_id.setdefault(etm_id, row)
    if len(profile_ids) < len(etm_rows_by_id):
        raise ValidationError(
            f"Select at least {len(etm_rows_by_id)} hardware profiles for {len(etm_rows_by_id)} resolved ETMs"
        )

    emit(progress, 0.05, "Loading resolved interaction ETM marker clusters")
    etms: list[CentreSeries] = []
    for index, (etm_id, row) in enumerate(sorted(etm_rows_by_id.items())):
        token.raise_if_cancelled()
        asset = session.assets.get(str(row["etm_asset_id"]))
        if asset is None or asset.kind not in {AssetKind.C3D, AssetKind.MOCAP_NPZ}:
            raise ValidationError(f"Resolved ETM asset for {etm_id} is unavailable")
        etms.append(
            _centre_series(
                asset.id,
                asset.label,
                etm_id,
                asset.source_path,
                mode="etm",
                marker_prefix=row.get("tracker_prefix"),
            )
        )
        emit(progress, 0.05 + 0.20 * (index + 1) / len(etm_rows_by_id), f"Loaded {etm_id} tracker shape")

    emit(progress, 0.28, "Loading hardware calibration profiles")
    profiles: list[ProfileFingerprint] = []
    profile_artifacts: list[ArtifactRecord] = []
    for artifact_id in profile_ids:
        artifact = session.artifacts.get(artifact_id)
        if artifact is None or artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
            raise ValidationError(f"Calibration profile {artifact_id} is unavailable")
        fingerprint = _load_profile_fingerprint(artifact)
        if fingerprint is None:
            raise ValidationError(f"Calibration profile {artifact.label!r} has no usable tracker template")
        profiles.append(fingerprint)
        profile_artifacts.append(artifact)

    emit(progress, 0.42, "Comparing ETM marker-cluster shapes")
    checks_by_etm: dict[str, list[dict[str, Any]]] = {}
    cost = np.full((len(etms), len(profiles)), np.inf, dtype=float)
    for i, etm in enumerate(etms):
        rows = []
        for j, profile in enumerate(profiles):
            row = _shape_difference(etm, profile)
            rows.append(row)
            cost[i, j] = float(row["pairwise_distance_rms_mm"])
        checks_by_etm[etm.identifier] = sorted(rows, key=lambda row: row["pairwise_distance_rms_mm"])
    if not np.isfinite(cost).any() or any(not np.isfinite(cost[index]).any() for index in range(len(etms))):
        raise ValidationError("At least one resolved ETM has no comparable selected tracker profile")

    def hardware_token(value: Any) -> str | None:
        match = re.search(r"(?:^|[^A-Z0-9])ETM[ _\-]*([A-Z0-9]+)", str(value or "").upper())
        return f"ETM_{match.group(1)}" if match else None

    etm_tokens = [hardware_token(item.identifier) or hardware_token(item.label) for item in etms]
    profile_tokens = [
        hardware_token(artifact.parameters.get("hardware_id"))
        or hardware_token(artifact.qc.get("hardware_id"))
        or hardware_token(artifact.label)
        for artifact in profile_artifacts
    ]
    # Stable physical ETM IDs take priority when available, while the marker
    # template still has to pass the geometry gate. If IDs are absent, the
    # assignment falls back to one-to-one shape matching.
    assignment_cost = cost.copy()
    for i, etm_token in enumerate(etm_tokens):
        for j, profile_token in enumerate(profile_tokens):
            if etm_token and profile_token:
                # Explicit physical ETM IDs are authoritative across sessions.
                # Never silently bind ETM-A to a profile registered as ETM-B.
                assignment_cost[i, j] = cost[i, j] - 1000.0 if etm_token == profile_token else np.inf
    if any(not np.isfinite(assignment_cost[index]).any() for index in range(len(etms))):
        missing = [etms[index].identifier for index in range(len(etms)) if not np.isfinite(assignment_cost[index]).any()]
        raise ValidationError(
            "No selected hardware profile matches the resolved physical ETM ID(s): " + ", ".join(missing)
        )

    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    if len(row_ind) != len(etms) or not np.isfinite(cost[row_ind, col_ind]).all():
        raise ValidationError("Could not obtain a complete one-to-one ETM/profile binding")
    best_total = float(cost[row_ind, col_ind].sum())
    best_assignment_cost = float(assignment_cost[row_ind, col_ind].sum())
    # Estimate a separation margin in the same identity-aware assignment space.
    alternative_totals: list[float] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        alt = assignment_cost.copy()
        alt[i, j] = np.inf
        try:
            ar, ac = linear_sum_assignment(alt)
        except ValueError:
            continue
        if len(ar) == len(etms) and np.isfinite(alt[ar, ac]).all():
            alternative_totals.append(float(alt[ar, ac].sum()))
    global_margin = min((value - best_assignment_cost for value in alternative_totals), default=None)

    binding_by_etm: dict[str, dict[str, Any]] = {}
    all_accepted = True
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        etm = etms[i]
        selected = _shape_difference(etm, profiles[j])
        alternatives = [float(cost[i, k]) for k in range(len(profiles)) if k != j and np.isfinite(cost[i, k])]
        row_margin = (min(alternatives) - float(selected["pairwise_distance_rms_mm"])) if alternatives else None
        exact_hardware_match = bool(etm_tokens[i] and profile_tokens[j] and etm_tokens[i] == profile_tokens[j])
        accepted = (
            float(selected["pairwise_distance_rms_mm"]) <= maximum_profile_shape_rms_mm
            and (exact_hardware_match or row_margin is None or row_margin >= minimum_profile_shape_margin_mm)
        )
        all_accepted = all_accepted and accepted
        binding_by_etm[etm.identifier] = {
            **selected,
            "hardware_id": profile_artifacts[j].parameters.get("hardware_id") or profile_artifacts[j].qc.get("hardware_id"),
            "exact_hardware_id_match": exact_hardware_match,
            "accepted": bool(accepted),
            "shape_margin_mm": row_margin,
            "reason": (
                "Physical hardware ID matched and tracker shape passed"
                if accepted and exact_hardware_match
                else "Unique one-to-one low-RMS tracker-shape match"
                if accepted
                else "Profile binding is not uniquely within the configured shape gates"
            ),
        }

    emit(progress, 0.72, "Creating immutable Calibration handoff")
    import copy
    assignments = copy.deepcopy(source_assignments)
    for role_id, row in assignments.items():
        binding = binding_by_etm.get(str(row.get("etm_id") or ""))
        sync = sync_by_role.get(role_id)
        row["recommended_calibration_artifact_id"] = (
            binding["profile_artifact_id"] if binding and binding.get("accepted") else None
        )
        row["recommended_calibration_profile_path"] = (
            binding["profile_path"] if binding and binding.get("accepted") else None
        )
        row["calibration_shape_check"] = binding
        row["recommended_time_sync_artifact_id"] = sync.get("artifact_id") if sync else None
        row["recommended_time_map_id"] = sync.get("time_map_id") if sync else None
        row["time_sync_check"] = sync

    artifact_id = new_id("artifact")
    output_root = store.derived_dir(session_id, "calibration_handoff", "interaction") / artifact_id
    output_root.mkdir(parents=True, exist_ok=False)
    all_time_sync_ready = bool(
        not require_time_sync
        or all(sync_by_role.get(role_id, {}).get("production_ready") for role_id in source_assignments)
    )
    any_time_sync_uses_video = any(bool(row.get("ego_video_used")) for row in sync_by_role.values())
    handoff_complete = bool(all_accepted and all_time_sync_ready)
    status = "resolved" if handoff_complete else "review_required"
    result = copy.deepcopy(source_result)
    result.update({
        "schema_version": "oculostudio-calibration-handoff-1.1",
        "artifact_id": artifact_id,
        "status": status,
        "method": "resolved_wearer_then_imu_time_sync_then_profile_binding",
        "source_assignment_artifact_id": assignment_artifact_id,
        "assignments": assignments,
        "assignment_by_etm": {str(row.get("etm_id")): row for row in assignments.values()},
        "tracker_shape_comparisons": checks_by_etm,
        "profile_binding": {
            "best_total_shape_rms_mm": best_total,
            "global_binding_margin_mm": global_margin,
            "maximum_profile_shape_rms_mm": maximum_profile_shape_rms_mm,
            "minimum_profile_shape_margin_mm": minimum_profile_shape_margin_mm,
            "all_roles_bound": bool(all_accepted),
        },
        "time_synchronization": {
            "required": bool(require_time_sync),
            "require_distributed_drift": bool(require_distributed_drift),
            "all_roles_verified": bool(all_time_sync_ready),
            "ego_video_used": bool(any_time_sync_uses_video),
            "by_role": sync_by_role,
        },
        "warnings": (
            ([] if all_accepted else [
                "One or more physical ETMs could not be bound uniquely to a selected hardware profile; Gaze 3D remains blocked."
            ])
            + sync_warnings
        ),
        "created_at": utc_now(),
    })
    result_path = output_root / "etm_role_assignment.json"
    atomic_write_json(result_path, result)
    comparison_path = output_root / "tracker_shape_comparison.csv"
    pd.DataFrame([
        {"etm_id": etm_id, **row}
        for etm_id, rows in checks_by_etm.items()
        for row in rows
    ]).to_csv(comparison_path, index=False)
    matrix_rows = []
    for etm_id, by_role in (source_result.get("distance_statistics_mm") or {}).items():
        for role_id, row in by_role.items():
            matrix_rows.append({"etm_id": etm_id, "role_id": role_id, **row})
    report_path = output_root / "calibration_handoff_report.html"
    report_path.write_text(_assignment_html(result, matrix_rows), "utf-8")

    qc = {
        "status": status,
        "wearer_assignment_status": source_result.get("status"),
        "binding_source_assignment_artifact_id": assignment_artifact_id,
        "profile_binding_count": int(sum(bool(row.get("recommended_calibration_artifact_id")) for row in assignments.values())),
        "role_count": len(assignments),
        "handoff_complete": handoff_complete,
        "time_sync_required": bool(require_time_sync),
        "time_sync_complete": bool(all_time_sync_ready),
        "time_sync_role_count": int(sum(bool(row.get("production_ready")) for row in sync_by_role.values())),
        "time_sync_by_role": sync_by_role,
        "time_sync_ego_video_used": bool(any_time_sync_uses_video),
        "best_total_shape_rms_mm": best_total,
        "global_binding_margin_mm": global_margin,
        "assignments_summary": {
            role_id: {
                "etm_id": row.get("etm_id"),
                "tracker_prefix": row.get("tracker_prefix"),
                "etm_asset_id": row.get("etm_asset_id"),
                "body_asset_id": row.get("body_asset_id"),
                "body_subject_prefix": row.get("body_subject_prefix"),
                "median_head_distance_mm": row.get("median_head_distance_mm"),
                "confidence": row.get("confidence"),
                "recommended_calibration_artifact_id": row.get("recommended_calibration_artifact_id"),
                "recommended_calibration_participant_id": (row.get("calibration_shape_check") or {}).get("participant_id"),
                "calibration_shape_rms_mm": (row.get("calibration_shape_check") or {}).get("pairwise_distance_rms_mm"),
                "recommended_time_sync_artifact_id": row.get("recommended_time_sync_artifact_id"),
                "recommended_time_map_id": row.get("recommended_time_map_id"),
                "time_sync_production_ready": (row.get("time_sync_check") or {}).get("production_ready"),
                "time_sync_confidence": (row.get("time_sync_check") or {}).get("confidence"),
                "time_sync_drift_ppm": (row.get("time_sync_check") or {}).get("estimated_drift_ppm"),
                "time_sync_p95_residual_ms": (row.get("time_sync_check") or {}).get("p95_residual_ms"),
            }
            for role_id, row in assignments.items()
        },
    }
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.ETM_ROLE_ASSIGNMENT,
        label=label,
        output_root=output_root,
        source_asset_ids=list(source_assignment.source_asset_ids),
        parent_artifact_ids=[assignment_artifact_id, *profile_ids, *sync_ids],
        parameters={
            "calibration_phase": "handoff",
            "binding_source_assignment_artifact_id": assignment_artifact_id,
            "calibration_profile_artifact_ids": profile_ids,
            "time_sync_artifact_ids": sync_ids,
            "require_time_sync": bool(require_time_sync),
            "require_distributed_drift": bool(require_distributed_drift),
            "maximum_profile_shape_rms_mm": maximum_profile_shape_rms_mm,
            "minimum_profile_shape_margin_mm": minimum_profile_shape_margin_mm,
        },
        qc=qc,
        provenance={"method": result["method"]},
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Calibration handoff complete" if handoff_complete else "Calibration handoff requires review")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "handoff_json": str(result_path),
        "shape_csv": str(comparison_path),
        "report": str(report_path),
        "result": result,
    }

def _assignment_html(result: dict[str, Any], matrix_rows: list[dict[str, Any]]) -> str:
    import html

    assignment_rows = "".join(
        f"<tr><td>{html.escape(role)}</td><td>{html.escape(str(row['etm_id']))}</td>"
        f"<td>{html.escape(str(row.get('tracker_prefix') or '—'))}</td>"
        f"<td>{row['median_head_distance_mm']:.2f} mm</td>"
        f"<td>{html.escape(row['confidence'])}</td>"
        f"<td>{html.escape(str(row.get('recommended_calibration_artifact_id') or '—'))}</td></tr>"
        for role, row in sorted(result["assignments"].items())
    )
    matrix = "".join(
        f"<tr><td>{html.escape(str(row['etm_id']))}</td><td>{html.escape(str(row['role_id']))}</td>"
        f"<td>{row['median_mm']:.2f}</td><td>{row['std_mm']:.3f}</td><td>{row['n']}</td></tr>"
        for row in matrix_rows
    )
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in result.get("warnings", [])) or "<li>None</li>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>ETM role assignment</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:30px auto;color:#282b1e}}h1{{color:#42472a}}table{{border-collapse:collapse;width:100%;margin:14px 0 28px}}th,td{{border:1px solid #d6d8c8;padding:9px;text-align:left}}th{{background:#eef2d5}}.status{{display:inline-block;padding:5px 10px;border-radius:999px;background:#effd3f;font-weight:700}}code,pre{{background:#f0f0e8;padding:3px 5px}}</style></head><body>
<h1>OculoStudio ETM wearer assignment</h1><p class='status'>{html.escape(result['status'])}</p>
<p>Method: global one-to-one assignment from robust median ETM-cluster-to-head-centre distance. Upload order is not used.</p>
<h2>Assignments</h2><table><thead><tr><th>Role</th><th>ETM</th><th>Tracker prefix</th><th>Median head distance</th><th>Confidence</th><th>Recommended profile artifact</th></tr></thead><tbody>{assignment_rows}</tbody></table>
<h2>Distance matrix</h2><table><thead><tr><th>ETM</th><th>Role</th><th>Median mm</th><th>Std mm</th><th>Valid frames</th></tr></thead><tbody>{matrix}</tbody></table>
<h2>Review gates</h2><ul>{warnings}</ul>
<pre>{html.escape(json.dumps(result.get('assignment_cost') or result.get('profile_binding') or {}, indent=2))}</pre>
</body></html>"""
