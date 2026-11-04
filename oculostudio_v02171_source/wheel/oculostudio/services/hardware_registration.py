from __future__ import annotations

"""Measured-calibration hardware identity and automatic profile registration.

The calibration solver selects a tracker from geometry.  This module turns that
selection into a stable hardware identity without conflating the calibration
wearer (P01/P02) with the physical eye-tracker mount (ETM-A/ETM-B).
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Any

from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ValidationError


_AB_PATTERN = re.compile(
    r"(?:^|[_\-\s])(?:etm|eye[_\-\s]?tracker|tracker)[_\-\s]?([ab])(?:[_\-\s.]|$)",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"^eye[_\-\s]?tracker[_\-\s]?([12])$", re.IGNORECASE)
_SESSION_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)$")
_GENERIC_PROFILE_NAMES = {
    "",
    "calibration",
    "neon to tracker calibration",
    "neon-to-tracker calibration",
}


@dataclass(frozen=True)
class HardwareIdentity:
    hardware_id: str
    display_name: str
    source: str
    selected_tracker_label: str
    marker_prefix: str | None
    legacy_aliases: tuple[str, ...]


def _marker_prefix(labels: Iterable[str]) -> str | None:
    prefixes = {
        str(label).split(":", 1)[0].strip()
        for label in labels
        if ":" in str(label) and str(label).split(":", 1)[0].strip()
    }
    return next(iter(prefixes)) if len(prefixes) == 1 else None


def canonical_hardware_id(value: str | None) -> str | None:
    """Normalise UI/user aliases to ETM_A or ETM_B when possible."""
    text = str(value or "").strip()
    if not text:
        return None
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    aliases = {
        "A": "ETM_A",
        "ETMA": "ETM_A",
        "TRACKERA": "ETM_A",
        "EYETRACKERA": "ETM_A",
        "EYETRACKER1": "ETM_A",
        "1": "ETM_A",
        "B": "ETM_B",
        "ETMB": "ETM_B",
        "TRACKERB": "ETM_B",
        "EYETRACKERB": "ETM_B",
        "EYETRACKER2": "ETM_B",
        "2": "ETM_B",
    }
    return aliases.get(compact, text)


def hardware_display_name(hardware_id: str | None) -> str:
    canonical = canonical_hardware_id(hardware_id)
    if canonical == "ETM_A":
        return "ETM-A"
    if canonical == "ETM_B":
        return "ETM-B"
    return str(hardware_id or "Unassigned hardware")


def infer_hardware_identity(selected_tracker_label: str, marker_labels: Iterable[str]) -> HardwareIdentity | None:
    """Infer ETM-A/B from the selected calibration candidate and raw marker labels.

    Filename/candidate identity is preferred because it directly preserves the
    user's A/B hardware convention.  The historical marker prefixes are kept as
    aliases: eye_tracker_1 -> ETM-A and eye_tracker_2 -> ETM-B.
    """
    label = Path(str(selected_tracker_label)).name
    match = _AB_PATTERN.search(label)
    from_label = f"ETM_{match.group(1).upper()}" if match else None

    prefix = _marker_prefix(marker_labels)
    from_prefix = None
    if prefix:
        number = _NUMBER_PATTERN.match(prefix)
        if number:
            from_prefix = "ETM_A" if number.group(1) == "1" else "ETM_B"
        else:
            match_prefix = _AB_PATTERN.search(prefix)
            if match_prefix:
                from_prefix = f"ETM_{match_prefix.group(1).upper()}"

    if from_label and from_prefix and from_label != from_prefix:
        raise ValidationError(
            "The selected tracker filename and marker prefix imply different hardware identities: "
            f"{label} -> {from_label}, {prefix} -> {from_prefix}. Fix the tracker labels before registration."
        )
    hardware_id = from_label or from_prefix
    if hardware_id is None:
        return None
    aliases = []
    if prefix:
        aliases.append(prefix)
    stem = Path(label).stem
    if stem and stem not in aliases:
        aliases.append(stem)
    return HardwareIdentity(
        hardware_id=hardware_id,
        display_name=hardware_display_name(hardware_id),
        source="selected_tracker_filename" if from_label else "marker_prefix_alias",
        selected_tracker_label=label,
        marker_prefix=prefix,
        legacy_aliases=tuple(aliases),
    )


def compact_session_scope(valid_sessions: Iterable[str]) -> str:
    values = [str(value).strip() for value in valid_sessions if str(value).strip()]
    if not values:
        return "sessions"
    parsed = [_SESSION_PATTERN.match(value) for value in values]
    if all(parsed):
        prefixes = {match.group(1) for match in parsed if match}
        widths = {len(match.group(2)) for match in parsed if match}
        numbers = sorted({int(match.group(2)) for match in parsed if match})
        if len(prefixes) == 1 and len(widths) == 1 and numbers == list(range(numbers[0], numbers[-1] + 1)):
            prefix = next(iter(prefixes))
            width = next(iter(widths))
            if len(numbers) == 1:
                return f"{prefix}{numbers[0]:0{width}d}"
            return f"{prefix}{numbers[0]:0{width}d}-{prefix}{numbers[-1]:0{width}d}"
    return "-".join(values)


def is_generic_profile_name(value: str | None) -> bool:
    text = str(value or "").strip()
    if text.lower() in _GENERIC_PROFILE_NAMES:
        return True
    return bool(re.fullmatch(r"P\d+_calibration(?:_[A-Za-z0-9-]+)*_auto", text, flags=re.IGNORECASE))


def registration_revision(session: Any, hardware_id: str, mount_version: str | None) -> tuple[int, str | None]:
    """Return an immutable revision number; never overwrite an older profile."""
    canonical = canonical_hardware_id(hardware_id)
    mount = str(mount_version or "").strip()
    existing = []
    for artifact in session.artifacts.values():
        if artifact.kind != ArtifactKind.CALIBRATION_PROFILE:
            continue
        candidate = canonical_hardware_id(
            artifact.parameters.get("hardware_id") or artifact.qc.get("hardware_id")
        )
        candidate_mount = str(
            artifact.parameters.get("mount_version") or artifact.qc.get("mount_version") or ""
        ).strip()
        if candidate == canonical and candidate_mount == mount:
            existing.append(artifact)
    latest = existing[-1].id if existing else None
    return len(existing) + 1, latest


def resolve_measured_registration(
    *,
    session: Any,
    selected_tracker_label: str,
    marker_labels: Iterable[str],
    calibration_wearer_id: str,
    requested_hardware_id: str | None,
    mount_version: str | None,
    valid_sessions: Iterable[str],
    requested_profile_name: str | None,
    auto_register_hardware: bool,
    auto_profile_name: bool,
    allow_hardware_id_override: bool,
) -> dict[str, Any]:
    inferred = infer_hardware_identity(selected_tracker_label, marker_labels)
    requested = canonical_hardware_id(requested_hardware_id)

    if auto_register_hardware:
        if inferred is None and requested is None:
            raise ValidationError(
                "Automatic hardware registration could not infer ETM-A/ETM-B from the selected tracker. "
                "Rename the tracker candidate with A/B or enter a manual Hardware ID override."
            )
        if inferred and requested and inferred.hardware_id != requested and not allow_hardware_id_override:
            raise ValidationError(
                f"Manual Hardware ID {requested_hardware_id!r} conflicts with the automatically selected "
                f"{inferred.display_name} ({inferred.selected_tracker_label}). Clear the manual value or enable override."
            )
        final_hardware = requested if (requested and allow_hardware_id_override) else (
            inferred.hardware_id if inferred else requested
        )
        registration_status = "manual_override" if (requested and allow_hardware_id_override) else "auto_registered"
    else:
        final_hardware = requested
        registration_status = "manual_registration" if requested else "unassigned"

    if not final_hardware:
        raise ValidationError("A Hardware ID is required when automatic registration is disabled")

    final_hardware = canonical_hardware_id(final_hardware) or str(final_hardware)
    display = hardware_display_name(final_hardware)
    sessions = [str(value).strip() for value in valid_sessions if str(value).strip()]
    mount = str(mount_version or "mount_v1").strip() or "mount_v1"
    revision, supersedes = registration_revision(session, final_hardware, mount)
    base_name = f"{display}_{mount}_{compact_session_scope(sessions)}"
    if auto_profile_name or is_generic_profile_name(requested_profile_name):
        final_name = base_name if revision == 1 else f"{base_name}_rev{revision}"
    else:
        final_name = str(requested_profile_name).strip()
        if revision > 1 and not re.search(r"(?:^|[_\-])rev\d+$", final_name, flags=re.IGNORECASE):
            final_name = f"{final_name}_rev{revision}"

    return {
        "hardware_id": final_hardware,
        "hardware_display_name": display,
        "mount_version": mount,
        "valid_sessions": sessions,
        "profile_name": final_name,
        "registration_status": registration_status,
        "registration_revision": revision,
        "supersedes_profile_artifact_id": supersedes,
        "calibration_wearer_id": calibration_wearer_id,
        "selected_tracker_label": selected_tracker_label,
        "identity_source": inferred.source if inferred else "manual",
        "legacy_marker_prefix": inferred.marker_prefix if inferred else _marker_prefix(marker_labels),
        "legacy_aliases": list(inferred.legacy_aliases) if inferred else [],
        "inferred_hardware_id": inferred.hardware_id if inferred else None,
        "manual_hardware_id": requested,
        "manual_override_allowed": bool(allow_hardware_id_override),
    }
