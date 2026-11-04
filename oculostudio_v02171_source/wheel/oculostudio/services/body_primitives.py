from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oculostudio.adapters.mocap import MocapData, read_mocap
from oculostudio.domain.semantic import BodyPrimitiveProfile, PrimitiveDefinition
from oculostudio.errors import ValidationError


@dataclass(frozen=True)
class BodyPrimitive:
    scene_frame_index: int
    scene_timestamp_ns: int
    entity_id: str
    source_asset_id: str
    region: str
    shape: str
    radius_mm: float
    priority: int
    confidence: float
    p0_mm: np.ndarray
    p1_mm: np.ndarray

    def geometry_json(self) -> str:
        if self.shape == "sphere":
            data = {"center_mm": self.p0_mm.tolist(), "radius_mm": self.radius_mm}
        else:
            data = {
                "start_mm": self.p0_mm.tolist(),
                "end_mm": self.p1_mm.tolist(),
                "radius_mm": self.radius_mm,
            }
        return json.dumps(data, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BodyHit:
    entity_id: str
    region: str
    distance_mm: float
    point_mm: np.ndarray
    confidence: float
    exact_hit: bool
    angular_miss_deg: float
    primitive_confidence: float
    source_asset_id: str
    priority: int = 0


_COARSE_BODY_REGION = {
    "head": "head",
    "neck": "head",
    "torso": "torso",
    "upper_torso": "torso",
    "lower_torso": "torso",
    "pelvis": "torso",
    "left_upper_arm": "left_arm",
    "left_forearm": "left_arm",
    "left_arm": "left_arm",
    "right_upper_arm": "right_arm",
    "right_forearm": "right_arm",
    "right_arm": "right_arm",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "left_thigh": "left_leg",
    "left_lower_leg": "left_leg",
    "left_shin": "left_leg",
    "left_foot": "left_leg",
    "left_leg": "left_leg",
    "right_thigh": "right_leg",
    "right_lower_leg": "right_leg",
    "right_shin": "right_leg",
    "right_foot": "right_leg",
    "right_leg": "right_leg",
}


def coarse_body_region(region: str | None) -> str:
    """Map detailed C3D primitive labels to stable social-body AOIs."""
    value = str(region or "unknown").strip().lower()
    return _COARSE_BODY_REGION.get(value, value or "unknown")


@dataclass(frozen=True)
class LandmarkResolution:
    name: str
    marker_indices: tuple[int, ...]
    marker_labels: tuple[str, ...]
    minimum_markers: int


@dataclass
class BodyPrimitiveBuildResult:
    primitives_by_frame: dict[int, list[BodyPrimitive]]
    frame_regions: pd.DataFrame
    diagnostics: dict


BODY_GEOMETRY_INTEGRITY_SCHEMA = "oculostudio-body-geometry-integrity-1.0"
BODY_GEOMETRY_GATE_POLICIES = {"required", "warn", "off", "legacy"}
DEFAULT_MINIMUM_TORSO_COVERAGE = 0.80
DEFAULT_MINIMUM_BODY_FRAME_COVERAGE = 0.90

_VICON_FULLBODY_10FINGERS_MARKERS = {
    "LFWT", "LMWT", "LBWT", "RFWT", "RMWT", "RBWT",
    "LFSH", "LTSH", "LBSH", "RFSH", "RTSH", "RBSH",
    "LIWR", "LOWR", "RIWR", "ROWR",
    "LOHAND", "LIHAND", "ROHAND", "RIHAND",
    "LKNU", "RKNU", "LT10", "RT10",
}
_CONVENTIONAL_VICON_MARKERS = {
    "LASI", "RASI", "LPSI", "RPSI", "LSHO", "RSHO",
    "LWRA", "LWRB", "RWRA", "RWRB", "LFIN", "RFIN",
}


def _safe_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coverage(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def assess_body_geometry_integrity(
    diagnostics: Iterable[dict[str, Any]],
    *,
    policy: str = "required",
    minimum_torso_coverage: float = DEFAULT_MINIMUM_TORSO_COVERAGE,
    minimum_body_frame_coverage: float = DEFAULT_MINIMUM_BODY_FRAME_COVERAGE,
    expected_relations: Iterable[str] = ("partner",),
    legacy: bool = False,
) -> dict[str, Any]:
    """Assess whether C3D body geometry is scientifically review-ready.

    The gate is deliberately evaluated before gaze-ray intersection.  A Body AOI
    result with no torso can otherwise complete successfully while redirecting
    chest-directed gaze to head/neck uncertainty candidates.  ``review_ready``
    is false whenever a required self/partner entity fails, even when the caller
    selects warn/off for diagnostic processing.
    """
    policy = str(policy or "required").strip().lower()
    if policy not in BODY_GEOMETRY_GATE_POLICIES:
        raise ValidationError(
            "body geometry acceptance policy must be required, warn or off"
        )
    minimum_torso_coverage = float(minimum_torso_coverage)
    minimum_body_frame_coverage = float(minimum_body_frame_coverage)
    if not 0.0 <= minimum_torso_coverage <= 1.0:
        raise ValidationError("minimum torso coverage must be between 0 and 1")
    if not 0.0 <= minimum_body_frame_coverage <= 1.0:
        raise ValidationError("minimum body-frame coverage must be between 0 and 1")

    expected = tuple(dict.fromkeys(
        str(value or "").strip().lower() for value in expected_relations
        if str(value or "").strip()
    ))
    rows = [dict(row or {}) for row in diagnostics]
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []

    def add_issue(
        target: list[dict[str, Any]],
        *,
        code: str,
        message: str,
        entity_id: str,
        relation: str,
        source_asset_id: str | None,
    ) -> None:
        target.append({
            "code": code,
            "message": message,
            "entity_id": entity_id,
            "entity_relation": relation,
            "source_asset_id": source_asset_id,
        })

    observed_required_relations: set[str] = set()
    for raw in rows:
        relation = str(raw.get("entity_relation") or "other").strip().lower() or "other"
        entity_id = str(raw.get("entity_id") or "unknown_entity")
        source_asset_id = raw.get("source_asset_id")
        required = relation in expected
        if required:
            observed_required_relations.add(relation)

        expected_frames = _safe_nonnegative_int(raw.get("scene_frame_count"))
        any_frames = _safe_nonnegative_int(raw.get("frames_with_any_primitive"))
        torso_frames = _safe_nonnegative_int(raw.get("torso_primitive_frame_count"))
        pelvis_frames = _safe_nonnegative_int(raw.get("pelvis_primitive_frame_count"))
        torso_derived = _safe_nonnegative_int(raw.get("torso_derived_frame_count"))
        torso_direct = _safe_nonnegative_int(raw.get("torso_direct_frame_count"))
        body_coverage = _coverage(any_frames, expected_frames)
        torso_coverage = _coverage(torso_frames, expected_frames)
        torso_given_body = _coverage(torso_frames, any_frames)
        pelvis_coverage = _coverage(pelvis_frames, expected_frames)
        torso_fallback_fraction = _coverage(torso_derived, torso_frames)
        entity_failures: list[str] = []
        entity_warnings: list[str] = []

        if required:
            if expected_frames <= 0:
                code = "no_expected_scene_frames"
                entity_failures.append(code)
                add_issue(
                    failures, code=code, relation=relation, entity_id=entity_id,
                    source_asset_id=source_asset_id,
                    message=f"{relation} {entity_id} has no aligned scene frames to evaluate.",
                )
            elif any_frames <= 0:
                code = "no_body_geometry"
                entity_failures.append(code)
                add_issue(
                    failures, code=code, relation=relation, entity_id=entity_id,
                    source_asset_id=source_asset_id,
                    message=f"{relation} {entity_id} produced no body primitives in any aligned frame.",
                )
            elif body_coverage < minimum_body_frame_coverage:
                code = "body_timeline_coverage_below_threshold"
                entity_failures.append(code)
                add_issue(
                    failures, code=code, relation=relation, entity_id=entity_id,
                    source_asset_id=source_asset_id,
                    message=(
                        f"{relation} {entity_id} body timeline coverage is {body_coverage:.1%}; "
                        f"required minimum is {minimum_body_frame_coverage:.1%}."
                    ),
                )

            if torso_frames <= 0:
                code = "torso_missing"
                entity_failures.append(code)
                add_issue(
                    failures, code=code, relation=relation, entity_id=entity_id,
                    source_asset_id=source_asset_id,
                    message=f"{relation} {entity_id} produced no torso primitive.",
                )
            elif torso_coverage < minimum_torso_coverage:
                code = "torso_coverage_below_threshold"
                entity_failures.append(code)
                add_issue(
                    failures, code=code, relation=relation, entity_id=entity_id,
                    source_asset_id=source_asset_id,
                    message=(
                        f"{relation} {entity_id} torso coverage is {torso_coverage:.1%}; "
                        f"required minimum is {minimum_torso_coverage:.1%}."
                    ),
                )
        elif any_frames > 0 and torso_frames <= 0:
            code = "optional_entity_torso_missing"
            entity_warnings.append(code)
            add_issue(
                warnings, code=code, relation=relation, entity_id=entity_id,
                source_asset_id=source_asset_id,
                message=f"Optional entity {entity_id} has limbs/head evidence but no torso primitive.",
            )

        if torso_frames > 0 and pelvis_frames <= 0:
            code = "pelvis_surface_missing"
            entity_warnings.append(code)
            add_issue(
                warnings, code=code, relation=relation, entity_id=entity_id,
                source_asset_id=source_asset_id,
                message=f"{relation} {entity_id} has a torso capsule but no pelvis surface primitive.",
            )
        elif torso_frames > 0 and pelvis_coverage + 0.05 < torso_coverage:
            code = "pelvis_coverage_lower_than_torso"
            entity_warnings.append(code)
            add_issue(
                warnings, code=code, relation=relation, entity_id=entity_id,
                source_asset_id=source_asset_id,
                message=(
                    f"{relation} {entity_id} pelvis coverage ({pelvis_coverage:.1%}) is lower "
                    f"than torso coverage ({torso_coverage:.1%})."
                ),
            )
        if torso_frames > 0 and torso_fallback_fraction >= 0.95:
            code = "torso_almost_entirely_derived"
            entity_warnings.append(code)
            add_issue(
                warnings, code=code, relation=relation, entity_id=entity_id,
                source_asset_id=source_asset_id,
                message=(
                    f"{relation} {entity_id} torso is derived in {torso_fallback_fraction:.1%} "
                    "of torso frames; verify shoulder/hip marker mapping during QC."
                ),
            )

        status = "failed" if entity_failures else "warning" if entity_warnings else "pass"
        entities.append({
            "entity_id": entity_id,
            "entity_relation": relation,
            "source_asset_id": source_asset_id,
            "source_asset_label": raw.get("source_asset_label"),
            "source_file_name": raw.get("source_file_name"),
            "required": required,
            "status": status,
            "expected_frame_count": expected_frames,
            "frames_with_any_primitive": any_frames,
            "body_frame_coverage_fraction": body_coverage,
            "torso_primitive_frame_count": torso_frames,
            "torso_coverage_fraction": torso_coverage,
            "torso_coverage_given_body_fraction": torso_given_body,
            "torso_direct_frame_count": torso_direct,
            "torso_derived_frame_count": torso_derived,
            "torso_fallback_fraction": torso_fallback_fraction,
            "pelvis_primitive_frame_count": pelvis_frames,
            "pelvis_coverage_fraction": pelvis_coverage,
            "issue_codes": entity_failures + entity_warnings,
            "unresolved_landmarks": raw.get("unresolved_landmarks") or {},
            "landmark_matches": raw.get("landmark_matches") or {},
            "landmark_valid_frame_counts": raw.get("landmark_valid_frame_counts") or {},
            "landmark_fallback_counts": raw.get("landmark_fallback_counts") or {},
            "geometry_provenance_counts": raw.get("geometry_provenance_counts") or {},
            "marker_protocol": raw.get("marker_protocol") or {},
            "marker_prefix_requested": raw.get("marker_prefix_requested"),
            "marker_prefix_used": raw.get("marker_prefix_used"),
            "available_marker_prefixes": raw.get("available_marker_prefixes") or [],
        })

    for missing_relation in expected:
        if missing_relation in observed_required_relations:
            continue
        code = "required_relation_geometry_missing"
        entity_id = f"missing_{missing_relation}"
        add_issue(
            failures, code=code, relation=missing_relation, entity_id=entity_id,
            source_asset_id=None,
            message=f"No geometry diagnostics were produced for required relation {missing_relation}.",
        )
        entities.append({
            "entity_id": entity_id,
            "entity_relation": missing_relation,
            "source_asset_id": None,
            "source_asset_label": None,
            "source_file_name": None,
            "required": True,
            "status": "failed",
            "expected_frame_count": 0,
            "frames_with_any_primitive": 0,
            "body_frame_coverage_fraction": 0.0,
            "torso_primitive_frame_count": 0,
            "torso_coverage_fraction": 0.0,
            "torso_coverage_given_body_fraction": 0.0,
            "torso_direct_frame_count": 0,
            "torso_derived_frame_count": 0,
            "torso_fallback_fraction": 0.0,
            "pelvis_primitive_frame_count": 0,
            "pelvis_coverage_fraction": 0.0,
            "issue_codes": [code],
            "unresolved_landmarks": {},
            "landmark_matches": {},
            "landmark_valid_frame_counts": {},
            "landmark_fallback_counts": {},
            "geometry_provenance_counts": {},
            "marker_protocol": {},
            "marker_prefix_requested": None,
            "marker_prefix_used": None,
            "available_marker_prefixes": [],
        })

    gate_passed = not failures
    if gate_passed:
        enforcement_outcome = "passed"
    elif policy == "required":
        enforcement_outcome = "blocked"
    elif policy == "warn":
        enforcement_outcome = "continued_with_warning"
    elif policy == "off":
        enforcement_outcome = "not_enforced"
    else:
        enforcement_outcome = "legacy_not_review_ready"
    status = "failed" if failures else "warning" if warnings else "pass"
    return {
        "schema_version": BODY_GEOMETRY_INTEGRITY_SCHEMA,
        "policy": policy,
        "legacy": bool(legacy),
        "status": status,
        "enforcement_outcome": enforcement_outcome,
        "gate_passed": gate_passed,
        "review_ready": gate_passed,
        "fusion_ready": gate_passed,
        "minimum_torso_coverage": minimum_torso_coverage,
        "minimum_body_frame_coverage": minimum_body_frame_coverage,
        "expected_relations": list(expected),
        "required_entity_count": int(sum(bool(row.get("required")) for row in entities)),
        "failed_entity_count": int(sum(row.get("status") == "failed" for row in entities)),
        "warning_entity_count": int(sum(row.get("status") == "warning" for row in entities)),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "entities": entities,
        "recommended_actions": [
            "Verify wearer/partner role-to-C3D assignment.",
            "Verify C3D subject marker prefix and shoulder/hip marker aliases.",
            "Check that the Body C3D covers the retained video interval.",
            "Rerun Body AOI before starting human review or target fusion.",
        ] if failures else [],
    }


def _require_gaze_sample_clock_integrity(
    report: dict[str, Any], qc: dict[str, Any]
) -> dict[str, Any]:
    """Block review/fusion for Body AOI computed on the legacy video clock.

    v0.21.51 and earlier could retain every gaze row while reusing one body
    snapshot for all samples assigned to the same world-video frame.  The
    geometry itself could pass the torso gate, but the artifact is not the
    sample-clock result required by v0.21.52.  Keep the geometry gate result
    auditable while forcing regeneration before manual review or fusion.
    """
    output = dict(report or {})
    clock = str(
        qc.get("body_geometry_clock")
        or (qc.get("timebase_summary") or {}).get("body_geometry_clock")
        or output.get("body_geometry_clock")
        or ""
    ).strip()
    output["body_geometry_clock"] = clock or "legacy_video_frame_clock"
    output["sample_clock_verified"] = clock == "gaze_sample_timestamp"
    if output["sample_clock_verified"]:
        return output
    failures = [dict(row) for row in (output.get("failures") or [])]
    if not any(row.get("code") == "legacy_video_frame_body_geometry_clock" for row in failures):
        failures.append({
            "code": "legacy_video_frame_body_geometry_clock",
            "message": (
                "This Body AOI reused video-frame body snapshots instead of "
                "interpolating self/partner C3D at every gaze timestamp. "
                "Regenerate Body AOI with v0.21.52 or later before human review or fusion."
            ),
            "entity_id": "all",
            "entity_relation": "all",
            "source_asset_id": None,
        })
    output.update({
        "status": "failed",
        "review_ready": False,
        "fusion_ready": False,
        "failure_count": len(failures),
        "failures": failures,
        "enforcement_outcome": "sample_clock_regeneration_required",
    })
    actions = list(output.get("recommended_actions") or [])
    action = "Regenerate Body AOI using the gaze-sample geometry clock."
    if action not in actions:
        actions.append(action)
    output["recommended_actions"] = actions
    return output


def body_geometry_integrity_from_qc(qc: dict[str, Any] | None) -> dict[str, Any]:
    """Return current geometry and temporal-integrity status from artifact QC."""
    qc = dict(qc or {})
    current = qc.get("body_geometry_integrity")
    if isinstance(current, dict) and current.get("schema_version"):
        return _require_gaze_sample_clock_integrity(current, qc)
    diagnostics = qc.get("body_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        return {
            "schema_version": BODY_GEOMETRY_INTEGRITY_SCHEMA,
            "policy": "legacy",
            "legacy": True,
            "status": "failed",
            "enforcement_outcome": "legacy_not_review_ready",
            "gate_passed": False,
            "review_ready": False,
            "fusion_ready": False,
            "failure_count": 1,
            "warning_count": 0,
            "failures": [{
                "code": "legacy_geometry_diagnostics_missing",
                "message": "This Body AOI predates verifiable torso geometry diagnostics.",
                "entity_id": "unknown",
                "entity_relation": "unknown",
                "source_asset_id": None,
            }],
            "warnings": [],
            "entities": [],
            "recommended_actions": ["Regenerate Body AOI with the current torso integrity gate."],
        }
    expected = ["partner"]
    if bool(qc.get("include_self_body")):
        expected.append("self")
    return _require_gaze_sample_clock_integrity(
        assess_body_geometry_integrity(
            diagnostics,
            policy="legacy",
            minimum_torso_coverage=DEFAULT_MINIMUM_TORSO_COVERAGE,
            minimum_body_frame_coverage=DEFAULT_MINIMUM_BODY_FRAME_COVERAGE,
            expected_relations=expected,
            legacy=True,
        ),
        qc,
    )


def load_body_profile(path: str | Path | None = None) -> BodyPrimitiveProfile:
    if path:
        source = Path(path).expanduser().resolve()
        try:
            payload = json.loads(source.read_text("utf-8"))
        except Exception as exc:
            raise ValidationError(f"Could not read body primitive profile: {source}: {exc}") from exc
    else:
        resource = files("oculostudio").joinpath("resources/default_body_primitives.json")
        payload = json.loads(resource.read_text("utf-8"))
    return BodyPrimitiveProfile.model_validate(payload)


def _marker_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _subject_and_tail(label: str) -> tuple[str, str]:
    raw = str(label).strip()
    if ":" in raw:
        subject, tail = raw.rsplit(":", 1)
        return _marker_key(subject), _marker_key(tail)
    return "", _marker_key(raw)


def subject_prefixes(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({
        subject for subject, _tail in (_subject_and_tail(label) for label in labels)
        if subject
    }))


def infer_subject_prefix(labels: Iterable[str], preferred: str | None = None) -> str | None:
    populated = list(subject_prefixes(labels))
    if preferred:
        key = _marker_key(preferred)
        matches = [
            row for row in populated
            if row == key or row.endswith(key) or key.endswith(row)
        ]
        if len(matches) == 1:
            return matches[0]
    return populated[0] if len(populated) == 1 else None


def _candidate_matches_label(
    *,
    label: str,
    subject: str,
    tail: str,
    candidate_key: str,
    prefix: str | None,
) -> bool:
    """Match exact marker tails first, then safe exporter-prefix suffixes.

    Vicon normally exports ``Subject:MARKER`` and is handled by the exact-tail
    path.  Some BVH/C3D bridges instead emit ``Subject_MARKER`` or
    ``Subject-MARKER``.  Those labels previously left all torso anchors
    unresolved even though the anatomical suffix was present.
    """
    if prefix and subject and not (subject == prefix or subject.endswith(prefix) or prefix.endswith(subject)):
        return False
    if tail == candidate_key:
        return True
    raw = str(label).strip().upper()
    tokens = [_marker_key(token) for token in re.split(r"[:/\\|._\-\s]+", raw) if token]
    if tokens and tokens[-1] == candidate_key:
        return True
    full_key = _marker_key(raw)
    if len(candidate_key) >= 3 and full_key.endswith(candidate_key):
        leading = full_key[: -len(candidate_key)]
        if not prefix or not leading or leading == prefix or leading.endswith(prefix) or prefix.endswith(leading):
            return True
    return False


def detect_marker_protocol(labels: Iterable[str]) -> dict[str, Any]:
    normalized: set[str] = set()
    for label in labels:
        _subject, tail = _subject_and_tail(label)
        normalized.add(tail)
        raw_key = _marker_key(label)
        for candidate in _VICON_FULLBODY_10FINGERS_MARKERS | _CONVENTIONAL_VICON_MARKERS:
            if raw_key.endswith(candidate):
                normalized.add(candidate)
    production = sorted(normalized & _VICON_FULLBODY_10FINGERS_MARKERS)
    conventional = sorted(normalized & _CONVENTIONAL_VICON_MARKERS)
    production_torso_core = {"LFWT", "RFWT", "LFSH", "RFSH"}
    if production_torso_core.issubset(normalized) and len(production) >= 8:
        # These labels are present in the raw Vicon C3D used by OCUMO.  They
        # are not evidence that the file has passed through MotionStudio.
        name = "vicon_fullbody_10fingers"
        confidence = min(1.0, len(production) / 16.0)
    elif len(conventional) >= 6:
        name = "vicon_conventional"
        confidence = min(1.0, len(conventional) / 10.0)
    else:
        name = "custom_or_partial"
        confidence = min(0.75, max(len(production), len(conventional)) / 10.0)
    return {
        "name": name,
        "legacy_name": (
            "motionstudio_production10fingers"
            if name == "vicon_fullbody_10fingers" else None
        ),
        "confidence": float(confidence),
        "matched_production10fingers_markers": production,
        "matched_conventional_markers": conventional,
    }


def resolve_landmarks(
    labels: list[str],
    profile: BodyPrimitiveProfile,
    *,
    marker_prefix: str | None = None,
) -> dict[str, LandmarkResolution]:
    prefix = _marker_key(marker_prefix) if marker_prefix else None
    parsed = [(_subject_and_tail(label), label) for label in labels]

    result: dict[str, LandmarkResolution] = {}
    for name, definition in profile.landmarks.items():
        indices: list[int] = []
        matched_labels: list[str] = []
        candidates: list[str] = []
        for marker in definition.markers:
            candidates.append(marker)
            candidates.extend(profile.marker_aliases.get(marker, []))
        candidate_keys = tuple(dict.fromkeys(_marker_key(candidate) for candidate in candidates))
        for index, ((subject, tail), label) in enumerate(parsed):
            if any(
                _candidate_matches_label(
                    label=label, subject=subject, tail=tail,
                    candidate_key=key, prefix=prefix,
                )
                for key in candidate_keys
            ):
                indices.append(index)
                matched_labels.append(label)
        result[name] = LandmarkResolution(
            name=name,
            marker_indices=tuple(indices),
            marker_labels=tuple(matched_labels),
            minimum_markers=definition.minimum_markers,
        )
    return result


def _interpolate_marker(
    data: MocapData,
    marker_index: int,
    query_sec: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_valid = data.valid[:, marker_index] & np.isfinite(data.points_mm[:, marker_index]).all(axis=1)
    source_indices = np.flatnonzero(source_valid)
    output = np.full((len(query_sec), 3), np.nan, dtype=float)
    valid = np.zeros(len(query_sec), dtype=bool)
    if len(source_indices) == 0:
        return output, valid
    source_time = data.time_sec[source_indices]
    source_points = data.points_mm[source_indices, marker_index]
    if len(source_indices) == 1:
        exact = np.isclose(query_sec, source_time[0], atol=1e-9, rtol=0)
        output[exact] = source_points[0]
        valid[exact] = True
        return output, valid

    positions = np.searchsorted(source_time, query_sec, side="left")
    right = np.clip(positions, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    exact_right = np.isclose(query_sec, source_time[right], atol=1e-9, rtol=0)
    left[exact_right] = right[exact_right]
    in_range = (query_sec >= source_time[0]) & (query_sec <= source_time[-1])
    gap = source_time[right] - source_time[left]
    permitted = in_range & (exact_right | (gap <= max_gap_sec))
    alpha = np.zeros(len(query_sec), dtype=float)
    nonzero = permitted & (right != left)
    alpha[nonzero] = (query_sec[nonzero] - source_time[left[nonzero]]) / gap[nonzero]
    output[permitted] = (
        source_points[left[permitted]] * (1.0 - alpha[permitted, None])
        + source_points[right[permitted]] * alpha[permitted, None]
    )
    valid[permitted] = True
    return output, valid


def interpolate_landmarks(
    data: MocapData,
    resolutions: dict[str, LandmarkResolution],
    query_sec: np.ndarray,
    *,
    max_gap_sec: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    marker_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    positions: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    confidence: dict[str, np.ndarray] = {}
    for name, resolution in resolutions.items():
        arrays: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for marker_index in resolution.marker_indices:
            if marker_index not in marker_cache:
                marker_cache[marker_index] = _interpolate_marker(
                    data, marker_index, query_sec, max_gap_sec=max_gap_sec
                )
            marker_position, marker_valid = marker_cache[marker_index]
            arrays.append(marker_position)
            masks.append(marker_valid)
        if not arrays:
            positions[name] = np.full((len(query_sec), 3), np.nan)
            valid[name] = np.zeros(len(query_sec), dtype=bool)
            confidence[name] = np.zeros(len(query_sec), dtype=float)
            continue
        stack = np.stack(arrays, axis=1)
        mask = np.stack(masks, axis=1)
        count = mask.sum(axis=1)
        weighted = np.where(mask[..., None], stack, 0.0).sum(axis=1)
        landmark = np.divide(
            weighted,
            count[:, None],
            out=np.full((len(query_sec), 3), np.nan),
            where=count[:, None] > 0,
        )
        landmark_valid = count >= resolution.minimum_markers
        positions[name] = landmark
        valid[name] = landmark_valid
        confidence[name] = np.clip(count / max(1, len(resolution.marker_indices)), 0.0, 1.0)
    return positions, valid, confidence




def _synthesise_body_landmark_fallbacks(
    positions: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    resolutions: dict[str, LandmarkResolution],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Fill missing torso anchors from already-observed articulated landmarks.

    Many Vicon exports retain shoulders and hips but omit one of the central
    thorax marker combinations required by the default profile.  In that case
    the limbs and head are visible while the torso capsule disappears.  These
    fallbacks never invent a full pose from constants: every derived point is
    computed from markers that are valid in the same queried frame.
    """
    if not positions:
        return {}, {}
    sample_count = len(next(iter(positions.values())))
    source: dict[str, np.ndarray] = {}
    for name in positions:
        values = np.full(sample_count, "unavailable", dtype=object)
        values[np.asarray(valid[name], dtype=bool)] = "marker_average"
        source[name] = values
    counts: dict[str, int] = {}
    required = {
        "head_center", "neck_top", "chest_center", "pelvis_center",
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    }
    if not required.issubset(positions):
        return source, counts

    def finite(name: str) -> np.ndarray:
        return np.isfinite(positions[name]).all(axis=1)

    def assign(
        name: str,
        candidate: np.ndarray,
        mask: np.ndarray,
        candidate_confidence: np.ndarray | float,
        label: str,
    ) -> None:
        mask = np.asarray(mask, dtype=bool) & ~np.asarray(valid[name], dtype=bool)
        mask &= np.isfinite(candidate).all(axis=1)
        if not mask.any():
            return
        conf = np.asarray(candidate_confidence, dtype=float)
        if conf.ndim == 0:
            conf = np.full(sample_count, float(conf), dtype=float)
        positions[name][mask] = candidate[mask]
        valid[name][mask] = True
        confidence[name][mask] = np.clip(conf[mask], 0.0, 1.0)
        source[name][mask] = label
        counts[label] = counts.get(label, 0) + int(mask.sum())

    def midpoint(left: str, right: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.asarray(valid[left], dtype=bool) & np.asarray(valid[right], dtype=bool)
        point = 0.5 * (positions[left] + positions[right])
        conf = np.sqrt(np.clip(confidence[left], 0.0, 1.0) * np.clip(confidence[right], 0.0, 1.0))
        return point, mask, conf

    # Pelvis: the midpoint of bilateral hip anchors is the most reliable
    # fallback.  A single central pelvis marker is accepted only at lower
    # confidence when the profile requested two central markers.
    hip_mid, hip_mask, hip_conf = midpoint("left_hip", "right_hip")
    assign("pelvis_center", hip_mid, hip_mask, hip_conf * 0.90, "bilateral_hips")
    pelvis_partial = finite("pelvis_center") & ~valid["pelvis_center"]
    if resolutions.get("pelvis_center") and resolutions["pelvis_center"].marker_indices:
        assign(
            "pelvis_center",
            positions["pelvis_center"].copy(),
            pelvis_partial,
            np.maximum(0.42, np.minimum(0.58, confidence["pelvis_center"] * 0.58)),
            "single_pelvis_marker",
        )

    shoulder_mid, shoulder_mask, shoulder_conf = midpoint("left_shoulder", "right_shoulder")
    assign("neck_top", shoulder_mid, shoulder_mask, shoulder_conf * 0.86, "bilateral_shoulders")

    # If the neck marker set is absent, place a conservative upper-torso anchor
    # along the observed head-to-pelvis axis.  This is preferable to omitting
    # the torso entirely and is explicitly recorded as derived evidence.
    head_pelvis = valid["head_center"] & valid["pelvis_center"]
    neck_from_axis = positions["pelvis_center"] + 0.82 * (
        positions["head_center"] - positions["pelvis_center"]
    )
    neck_axis_conf = np.minimum(confidence["head_center"], confidence["pelvis_center"]) * 0.66
    assign("neck_top", neck_from_axis, head_pelvis, neck_axis_conf, "head_pelvis_axis")

    # Chest: bilateral shoulders provide the preferred fallback.  A single
    # central thorax marker is useful but down-weighted; finally the observed
    # neck/head-to-pelvis axis supplies a conservative last resort.
    assign("chest_center", shoulder_mid, shoulder_mask, shoulder_conf * 0.88, "bilateral_shoulders")
    chest_partial = finite("chest_center") & ~valid["chest_center"]
    if resolutions.get("chest_center") and resolutions["chest_center"].marker_indices:
        assign(
            "chest_center",
            positions["chest_center"].copy(),
            chest_partial,
            np.maximum(0.45, np.minimum(0.62, confidence["chest_center"] * 0.62)),
            "single_chest_marker",
        )
    neck_pelvis = valid["neck_top"] & valid["pelvis_center"]
    chest_from_neck = positions["pelvis_center"] + 0.72 * (
        positions["neck_top"] - positions["pelvis_center"]
    )
    chest_neck_conf = np.minimum(confidence["neck_top"], confidence["pelvis_center"]) * 0.78
    assign("chest_center", chest_from_neck, neck_pelvis, chest_neck_conf, "neck_pelvis_axis")
    chest_from_head = positions["pelvis_center"] + 0.64 * (
        positions["head_center"] - positions["pelvis_center"]
    )
    chest_head_conf = np.minimum(confidence["head_center"], confidence["pelvis_center"]) * 0.62
    assign("chest_center", chest_from_head, head_pelvis, chest_head_conf, "head_pelvis_axis")

    return source, counts


def build_body_primitives(
    *,
    body_asset_id: str,
    body_path: str | Path,
    entity_id: str,
    scene_frame_indices: np.ndarray,
    scene_timestamps_ns: np.ndarray,
    query_time_sec: np.ndarray,
    profile: BodyPrimitiveProfile,
    marker_prefix: str | None = None,
    max_marker_gap_sec: float = 0.1,
    mocap_data: MocapData | None = None,
    diagnostics_only: bool = False,
    include_frame_regions: bool = True,
) -> BodyPrimitiveBuildResult:
    data = mocap_data if mocap_data is not None else read_mocap(body_path)
    scene_frame_indices = np.asarray(scene_frame_indices)
    scene_timestamps_ns = np.asarray(scene_timestamps_ns)
    query_time_sec = np.asarray(query_time_sec, dtype=float)
    if not (
        len(scene_frame_indices) == len(scene_timestamps_ns) == len(query_time_sec)
    ):
        raise ValidationError(
            "Body primitive query indices, timestamps and query times must have equal length"
        )
    available_prefixes = subject_prefixes(data.labels)
    if marker_prefix:
        selected_prefix = infer_subject_prefix(data.labels, marker_prefix)
        if available_prefixes and selected_prefix is None:
            raise ValidationError(
                f"Marker prefix {marker_prefix!r} does not uniquely match any subject in "
                f"{Path(body_path).name}. Available prefixes: {list(available_prefixes)}"
            )
    elif len(available_prefixes) == 1:
        selected_prefix = available_prefixes[0]
    elif len(available_prefixes) > 1:
        raise ValidationError(
            f"Body asset {Path(body_path).name} contains multiple subject prefixes "
            f"{list(available_prefixes)}. Register the file once per participant and "
            "provide entity_marker_prefixes so OculoStudio never averages markers "
            "across people."
        )
    else:
        selected_prefix = None
    marker_protocol = detect_marker_protocol(data.labels)
    resolutions = resolve_landmarks(
        data.labels,
        profile,
        marker_prefix=selected_prefix,
    )
    landmark_pos, landmark_valid, landmark_conf = interpolate_landmarks(
        data,
        resolutions,
        query_time_sec,
        max_gap_sec=max_marker_gap_sec,
    )
    landmark_source, fallback_counts = _synthesise_body_landmark_fallbacks(
        landmark_pos, landmark_valid, landmark_conf, resolutions
    )
    by_frame: dict[int, list[BodyPrimitive]] = {}
    rows: list[dict] = []
    frames_with_any_mask = np.zeros(len(scene_frame_indices), dtype=bool)
    primitive_region_counts: Counter[str] = Counter()
    derived_region_counts: Counter[str] = Counter()
    direct_region_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    primitive_row_count = 0
    unresolved = {
        name: {
            "requested": profile.landmarks[name].markers,
            "matched": list(res.marker_labels),
            "minimum_markers": res.minimum_markers,
        }
        for name, res in resolutions.items()
        if len(res.marker_indices) < res.minimum_markers
    }
    for primitive in profile.primitives:
        if not primitive.enabled:
            continue
        if primitive.shape == "sphere":
            assert primitive.center is not None
            if primitive.center not in landmark_pos:
                continue
            mask = landmark_valid[primitive.center]
            p0 = landmark_pos[primitive.center]
            p1 = p0
            conf = landmark_conf[primitive.center]
            primitive_source = landmark_source[primitive.center]
        else:
            assert primitive.start is not None and primitive.end is not None
            if primitive.start not in landmark_pos or primitive.end not in landmark_pos:
                continue
            mask = landmark_valid[primitive.start] & landmark_valid[primitive.end]
            p0 = landmark_pos[primitive.start]
            p1 = landmark_pos[primitive.end]
            conf = np.sqrt(landmark_conf[primitive.start] * landmark_conf[primitive.end])
            primitive_source = np.asarray([
                left if left == right else f"{left}+{right}"
                for left, right in zip(landmark_source[primitive.start], landmark_source[primitive.end])
            ], dtype=object)
            mask &= np.linalg.norm(p1 - p0, axis=1) > 1e-6
        valid_indices = np.flatnonzero(mask)
        if len(valid_indices):
            frames_with_any_mask[valid_indices] = True
            region_name = str(primitive.region)
            primitive_region_counts[region_name] += int(len(valid_indices))
            sources = np.asarray(primitive_source, dtype=object)[valid_indices].astype(str)
            provenance_counts.update(sources.tolist())
            derived_count = int(np.count_nonzero(sources != "marker_average"))
            direct_count = int(len(valid_indices) - derived_count)
            if derived_count:
                derived_region_counts[region_name] += derived_count
            if direct_count:
                direct_region_counts[region_name] += direct_count
            primitive_row_count += int(len(valid_indices))
        if diagnostics_only:
            continue
        for local_index in valid_indices:
            frame_index = int(scene_frame_indices[local_index])
            item = BodyPrimitive(
                scene_frame_index=frame_index,
                scene_timestamp_ns=int(scene_timestamps_ns[local_index]),
                entity_id=entity_id,
                source_asset_id=body_asset_id,
                region=str(primitive.region),
                shape=primitive.shape,
                radius_mm=float(primitive.radius_mm),
                priority=int(primitive.priority),
                confidence=float(conf[local_index]),
                p0_mm=np.asarray(p0[local_index], dtype=float),
                p1_mm=np.asarray(p1[local_index], dtype=float),
            )
            by_frame.setdefault(frame_index, []).append(item)
            if include_frame_regions:
                rows.append({
                    "scene_frame_index": frame_index,
                    "scene_timestamp_ns": item.scene_timestamp_ns,
                    "entity_id": entity_id,
                    "region": item.region,
                    "coordinate_space": "world_mm",
                    "geometry_type": item.shape,
                    "geometry_json": item.geometry_json(),
                    "source": "c3d_primitives",
                    "source_asset_id": body_asset_id,
                    "geometry_provenance": str(primitive_source[local_index]),
                    "geometry_is_derived": str(primitive_source[local_index]) != "marker_average",
                    "confidence": item.confidence,
                    "priority": item.priority,
                })
    source_time = np.asarray(data.time_sec, dtype=float)
    finite_query = np.isfinite(query_time_sec)
    in_source_range = finite_query.copy()
    if len(source_time):
        in_source_range &= (
            (query_time_sec >= float(source_time[0]))
            & (query_time_sec <= float(source_time[-1]))
        )
        insertion = np.searchsorted(source_time, query_time_sec[finite_query], side="left")
        insertion = np.clip(insertion, 0, max(0, len(source_time) - 1))
        previous = np.clip(insertion - 1, 0, max(0, len(source_time) - 1))
        query_values = query_time_sec[finite_query]
        nearest_error = np.minimum(
            np.abs(source_time[insertion] - query_values),
            np.abs(source_time[previous] - query_values),
        )
        nearest_error_ms = nearest_error * 1000.0
    else:
        nearest_error_ms = np.asarray([], dtype=float)
    diagnostics = {
        "entity_id": entity_id,
        "source_asset_id": body_asset_id,
        "profile_name": profile.name,
        "marker_prefix_requested": marker_prefix,
        "marker_prefix_used": selected_prefix,
        "available_marker_prefixes": list(available_prefixes),
        "marker_count": len(data.labels),
        "marker_protocol": marker_protocol,
        "scene_frame_count": int(len(scene_frame_indices)),
        "frames_with_any_primitive": int(frames_with_any_mask.sum()),
        "primitive_row_count": int(primitive_row_count),
        "diagnostics_only": bool(diagnostics_only),
        "frame_regions_emitted": bool(include_frame_regions and not diagnostics_only),
        "source_frame_rate_hz": float(data.frame_rate),
        "source_frame_count": int(len(data.time_sec)),
        "query_time_valid_count": int(finite_query.sum()),
        "query_time_in_source_range_count": int(in_source_range.sum()),
        "query_time_in_source_range_fraction": _coverage(
            int(in_source_range.sum()), int(finite_query.sum())
        ),
        "median_query_to_source_sample_error_ms": (
            float(np.median(nearest_error_ms)) if len(nearest_error_ms) else None
        ),
        "p95_query_to_source_sample_error_ms": (
            float(np.quantile(nearest_error_ms, 0.95)) if len(nearest_error_ms) else None
        ),
        "maximum_query_to_source_sample_error_ms": (
            float(np.max(nearest_error_ms)) if len(nearest_error_ms) else None
        ),
        "primitive_region_counts": dict(sorted(primitive_region_counts.items())),
        "derived_region_counts": dict(sorted(derived_region_counts.items())),
        "direct_region_counts": dict(sorted(direct_region_counts.items())),
        "geometry_provenance_counts": dict(sorted(provenance_counts.items())),
        "unresolved_landmarks": unresolved,
        "landmark_matches": {
            name: list(res.marker_labels) for name, res in resolutions.items()
        },
        "landmark_valid_frame_counts": {
            name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in landmark_valid.items()
        },
        "landmark_fallback_counts": dict(sorted(fallback_counts.items())),
        "torso_primitive_frame_count": int(primitive_region_counts.get("torso", 0)),
        "torso_derived_frame_count": int(derived_region_counts.get("torso", 0)),
        "torso_direct_frame_count": int(direct_region_counts.get("torso", 0)),
        "pelvis_primitive_frame_count": int(primitive_region_counts.get("pelvis", 0)),
        "pelvis_derived_frame_count": int(derived_region_counts.get("pelvis", 0)),
        "pelvis_direct_frame_count": int(direct_region_counts.get("pelvis", 0)),
    }
    return BodyPrimitiveBuildResult(
        primitives_by_frame=by_frame,
        frame_regions=pd.DataFrame(rows),
        diagnostics=diagnostics,
    )


def _closest_ray_segment(
    origins: np.ndarray,
    directions: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ray t, segment u, and closest distance for normalized rays."""
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    segment = end - start
    c = float(np.dot(segment, segment))
    if c <= 1e-12:
        offset = origins - start
        t = np.maximum(0.0, -np.einsum("ij,ij->i", directions, offset))
        closest_ray = origins + t[:, None] * directions
        distance = np.linalg.norm(closest_ray - start, axis=1)
        return t, np.zeros(len(origins)), distance
    offset = origins - start
    b = directions @ segment
    d = np.einsum("ij,ij->i", directions, offset)
    e = offset @ segment
    denominator = c - b * b
    numerator = b * e - c * d
    t = np.empty_like(denominator, dtype=float)
    safe = np.abs(denominator) > 1e-12
    np.divide(numerator, denominator, out=t, where=safe)
    t[~safe] = -d[~safe]
    u = (e + b * t) / c
    u = np.clip(u, 0.0, 1.0)
    t = b * u - d
    negative = t < 0.0
    if negative.any():
        t[negative] = 0.0
        u[negative] = np.clip(e[negative] / c, 0.0, 1.0)
    closest_ray = origins + t[:, None] * directions
    closest_segment = start + u[:, None] * segment
    distance = np.linalg.norm(closest_ray - closest_segment, axis=1)
    return t, u, distance


def _primitive_candidates(
    origins: np.ndarray,
    directions: np.ndarray,
    primitive: BodyPrimitive,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if primitive.shape == "sphere":
        center = primitive.p0_mm
        offset = origins - center
        t_closest = np.maximum(0.0, -np.einsum("ij,ij->i", directions, offset))
        closest = origins + t_closest[:, None] * directions
        distance = np.linalg.norm(closest - center, axis=1)
    else:
        t_closest, _, distance = _closest_ray_segment(
            origins, directions, primitive.p0_mm, primitive.p1_mm
        )
    inside = distance <= primitive.radius_mm
    entry = np.maximum(
        0.0,
        t_closest - np.sqrt(np.maximum(0.0, primitive.radius_mm**2 - distance**2)),
    )
    surface_miss = np.maximum(0.0, distance - primitive.radius_mm)
    angular_miss = np.degrees(np.arctan2(surface_miss, np.maximum(t_closest, 1e-9)))
    point = origins + entry[:, None] * directions
    return inside, entry, angular_miss, point


def _paired_primitive_candidates(
    origins: np.ndarray,
    directions: np.ndarray,
    primitives: list[BodyPrimitive],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one independently moving primitive against each matching ray.

    ``intersect_frame_rays_ranked`` is optimal when many gaze rays share one
    video-frame body snapshot.  Gaze-sample-clock Body AOI instead has a
    different interpolated primitive for every gaze timestamp.  This paired
    vectorisation avoids one Python intersection call per gaze sample while
    preserving the same sphere/capsule mathematics.
    """
    count = len(primitives)
    if count == 0:
        empty = np.asarray([], dtype=float)
        return empty.astype(bool), empty, empty, np.empty((0, 3), dtype=float)
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    p0 = np.stack([np.asarray(item.p0_mm, dtype=float) for item in primitives])
    p1 = np.stack([np.asarray(item.p1_mm, dtype=float) for item in primitives])
    radii = np.asarray([float(item.radius_mm) for item in primitives], dtype=float)
    sphere = np.asarray([item.shape == "sphere" for item in primitives], dtype=bool)
    t_closest = np.zeros(count, dtype=float)
    distance = np.full(count, np.inf, dtype=float)

    if sphere.any():
        indices = np.flatnonzero(sphere)
        offset = origins[indices] - p0[indices]
        t = np.maximum(0.0, -np.einsum("ij,ij->i", directions[indices], offset))
        closest = origins[indices] + t[:, None] * directions[indices]
        t_closest[indices] = t
        distance[indices] = np.linalg.norm(closest - p0[indices], axis=1)

    capsule = ~sphere
    if capsule.any():
        indices = np.flatnonzero(capsule)
        start = p0[indices]
        end = p1[indices]
        ray_origin = origins[indices]
        ray_direction = directions[indices]
        segment = end - start
        c = np.einsum("ij,ij->i", segment, segment)
        offset = ray_origin - start
        b = np.einsum("ij,ij->i", ray_direction, segment)
        d = np.einsum("ij,ij->i", ray_direction, offset)
        e = np.einsum("ij,ij->i", offset, segment)
        denominator = c - b * b
        t = np.empty(len(indices), dtype=float)
        safe = (c > 1e-12) & (np.abs(denominator) > 1e-12)
        np.divide(b * e - c * d, denominator, out=t, where=safe)
        t[~safe] = -d[~safe]
        u = np.zeros(len(indices), dtype=float)
        valid_segment = c > 1e-12
        np.divide(e + b * t, c, out=u, where=valid_segment)
        u = np.clip(u, 0.0, 1.0)
        t = b * u - d
        negative = t < 0.0
        if negative.any():
            t[negative] = 0.0
            replacement = np.zeros(int(negative.sum()), dtype=float)
            np.divide(
                e[negative], c[negative], out=replacement,
                where=c[negative] > 1e-12,
            )
            u[negative] = np.clip(replacement, 0.0, 1.0)
        degenerate = ~valid_segment
        if degenerate.any():
            t[degenerate] = np.maximum(0.0, -d[degenerate])
            u[degenerate] = 0.0
        closest_ray = ray_origin + t[:, None] * ray_direction
        closest_segment = start + u[:, None] * segment
        t_closest[indices] = t
        distance[indices] = np.linalg.norm(closest_ray - closest_segment, axis=1)

    inside = distance <= radii
    entry = np.maximum(
        0.0,
        t_closest - np.sqrt(np.maximum(0.0, radii**2 - distance**2)),
    )
    surface_miss = np.maximum(0.0, distance - radii)
    angular_miss = np.degrees(
        np.arctan2(surface_miss, np.maximum(t_closest, 1e-9))
    )
    point = origins + entry[:, None] * directions
    return inside, entry, angular_miss, point


def _rank_body_hit_candidates(
    candidates: list[list[BodyHit]], *, exact_tie_tolerance_mm: float
) -> list[list[BodyHit]]:
    ranked: list[list[BodyHit]] = []
    tie = max(0.0, float(exact_tie_tolerance_mm))
    for rows in candidates:
        exact_rows = sorted(
            (row for row in rows if row.exact_hit),
            key=lambda row: (row.distance_mm, -row.confidence, -row.priority),
        )
        if exact_rows:
            minimum = exact_rows[0].distance_mm
            tied = [row for row in exact_rows if row.distance_mm <= minimum + tie]
            first = sorted(
                tied,
                key=lambda row: (-row.priority, row.distance_mm, -row.confidence),
            )[0]
            exact_rows.remove(first)
            exact_rows = [
                first,
                *sorted(
                    exact_rows,
                    key=lambda row: (
                        row.distance_mm,
                        -row.priority,
                        -row.confidence,
                    ),
                ),
            ]
        cone_rows = sorted(
            (row for row in rows if not row.exact_hit),
            key=lambda row: (
                row.angular_miss_deg,
                row.distance_mm,
                -row.priority,
                -row.confidence,
            ),
        )
        ranked.append([*exact_rows, *cone_rows])
    return ranked


def intersect_sample_rays_ranked(
    origins_mm: np.ndarray,
    directions: np.ndarray,
    primitives_by_sample: list[list[BodyPrimitive]],
    *,
    uncertainty_cone_deg: float = 1.5,
    maximum_distance_mm: float = 10_000.0,
    exact_tie_tolerance_mm: float = 8.0,
    primitive_batch_size: int = 200_000,
) -> list[list[BodyHit]]:
    """Intersect gaze rays with body geometry at each ray's exact timestamp.

    Each entry in ``primitives_by_sample`` belongs only to the ray at the same
    position.  The function processes independently moving geometry in large
    vectorised batches and returns the same ranked ``BodyHit`` structure as the
    legacy video-frame-clock intersection.
    """
    origins = np.asarray(origins_mm, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if len(origins) != len(directions) or len(origins) != len(primitives_by_sample):
        raise ValidationError(
            "Gaze origins, directions and sample-clock body geometry must have equal length"
        )
    norms = np.linalg.norm(directions, axis=1)
    valid_ray = (
        np.isfinite(origins).all(axis=1)
        & np.isfinite(directions).all(axis=1)
        & (norms > 1e-12)
    )
    directions = np.divide(
        directions,
        norms[:, None],
        out=np.full_like(directions, np.nan),
        where=norms[:, None] > 1e-12,
    )
    candidates: list[list[BodyHit]] = [[] for _ in range(len(origins))]
    flat_ray_indices: list[int] = []
    flat_primitives: list[BodyPrimitive] = []
    for ray_index, primitives in enumerate(primitives_by_sample):
        if not valid_ray[ray_index] or not primitives:
            continue
        flat_ray_indices.extend([ray_index] * len(primitives))
        flat_primitives.extend(primitives)

    batch_size = max(1, int(primitive_batch_size))
    for start in range(0, len(flat_primitives), batch_size):
        stop = min(len(flat_primitives), start + batch_size)
        batch_primitives = flat_primitives[start:stop]
        ray_indices = np.asarray(flat_ray_indices[start:stop], dtype=np.int64)
        exact, distance, angular, points = _paired_primitive_candidates(
            origins[ray_indices], directions[ray_indices], batch_primitives
        )
        possible = np.isfinite(distance) & (distance <= maximum_distance_mm)
        possible &= exact | (angular <= uncertainty_cone_deg)
        for local_index in np.flatnonzero(possible):
            primitive = batch_primitives[int(local_index)]
            ray_index = int(ray_indices[local_index])
            exact_hit = bool(exact[local_index])
            angular_conf = 1.0 if exact_hit else max(
                0.0,
                1.0
                - float(angular[local_index])
                / max(uncertainty_cone_deg, 1e-9),
            )
            confidence = float(primitive.confidence) * (
                0.94 if exact_hit else 0.62 * angular_conf
            )
            candidates[ray_index].append(
                BodyHit(
                    entity_id=primitive.entity_id,
                    region=primitive.region,
                    distance_mm=float(distance[local_index]),
                    point_mm=np.asarray(points[local_index], dtype=float),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    exact_hit=exact_hit,
                    angular_miss_deg=float(angular[local_index]),
                    primitive_confidence=float(primitive.confidence),
                    source_asset_id=primitive.source_asset_id,
                    priority=int(primitive.priority),
                )
            )
    return _rank_body_hit_candidates(
        candidates, exact_tie_tolerance_mm=exact_tie_tolerance_mm
    )


def intersect_frame_rays_ranked(
    origins_mm: np.ndarray,
    directions: np.ndarray,
    primitives: list[BodyPrimitive],
    *,
    uncertainty_cone_deg: float = 1.5,
    maximum_distance_mm: float = 10_000.0,
    exact_tie_tolerance_mm: float = 8.0,
) -> list[list[BodyHit]]:
    """Return ordered body intersections for every ray.

    Exact surface intersections are always considered before uncertainty-cone
    candidates.  The visible hit is ordered by physical distance.  Primitive
    priority is used only when exact surfaces overlap within a small numerical
    tolerance, so a farther hand/arm primitive cannot silently jump ahead of a
    physically nearer surface.
    """
    origins = np.asarray(origins_mm, dtype=float)
    directions = np.asarray(directions, dtype=float)
    norms = np.linalg.norm(directions, axis=1)
    valid_ray = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1) & (norms > 1e-12)
    directions = np.divide(
        directions,
        norms[:, None],
        out=np.full_like(directions, np.nan),
        where=norms[:, None] > 1e-12,
    )
    candidates: list[list[BodyHit]] = [[] for _ in range(len(origins))]
    for primitive in primitives:
        exact, distance, angular, points = _primitive_candidates(origins, directions, primitive)
        possible = valid_ray & np.isfinite(distance) & (distance <= maximum_distance_mm)
        possible &= exact | (angular <= uncertainty_cone_deg)
        for index in np.flatnonzero(possible):
            exact_hit = bool(exact[index])
            angular_conf = 1.0 if exact_hit else max(
                0.0, 1.0 - float(angular[index]) / max(uncertainty_cone_deg, 1e-9)
            )
            confidence = float(primitive.confidence) * (0.94 if exact_hit else 0.62 * angular_conf)
            candidates[index].append(
                BodyHit(
                    entity_id=primitive.entity_id,
                    region=primitive.region,
                    distance_mm=float(distance[index]),
                    point_mm=np.asarray(points[index], dtype=float),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    exact_hit=exact_hit,
                    angular_miss_deg=float(angular[index]),
                    primitive_confidence=float(primitive.confidence),
                    source_asset_id=primitive.source_asset_id,
                    priority=int(primitive.priority),
                )
            )

    return _rank_body_hit_candidates(
        candidates, exact_tie_tolerance_mm=exact_tie_tolerance_mm
    )


def intersect_frame_rays(
    origins_mm: np.ndarray,
    directions: np.ndarray,
    primitives: list[BodyPrimitive],
    *,
    uncertainty_cone_deg: float = 1.5,
    maximum_distance_mm: float = 10_000.0,
) -> list[BodyHit | None]:
    """Backward-compatible single-hit wrapper."""
    ranked = intersect_frame_rays_ranked(
        origins_mm,
        directions,
        primitives,
        uncertainty_cone_deg=uncertainty_cone_deg,
        maximum_distance_mm=maximum_distance_mm,
    )
    return [rows[0] if rows else None for rows in ranked]
