"""Canonical FaceMapper bilateral semantics without geometric mirroring.

Pupil Labs Face Mapper/RetinaFace exports the five landmark coordinates in the
scene-image coordinate system.  In the recordings used by OculoStudio the
bilateral source names (``eye left``, ``eye right``, ``mouth left`` and
``mouth right``) follow image side.  OculoStudio's analysis convention is the
observed person's own anatomical left/right.

Canonicalisation therefore *reassigns semantic names only*.  Pixel coordinates
are never reflected, rescaled or otherwise moved.  The exact source table is
preserved beside every filtered artifact for audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from oculostudio.errors import ValidationError

SourceBilateralConvention = Literal[
    "pupil_image_side",
    "subject_anatomical",
    "auto_audit",
]

_CANONICAL_VERSION = "oculostudio-facemapper-bilateral-1.0"


@dataclass(frozen=True)
class BilateralColumns:
    left_eye_x: str
    left_eye_y: str
    right_eye_x: str
    right_eye_y: str
    mouth_left_x: str
    mouth_left_y: str
    mouth_right_x: str
    mouth_right_y: str

    @classmethod
    def from_mapping(cls, columns: dict[str, str | None]) -> "BilateralColumns":
        values = {name: columns.get(name) for name in cls.__dataclass_fields__}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValidationError(
                "Face Mapper table is missing bilateral landmark columns: "
                + ", ".join(missing)
            )
        return cls(**{name: str(value) for name, value in values.items()})


def _finite_pair(frame: pd.DataFrame, left: str, right: str) -> tuple[np.ndarray, np.ndarray]:
    l = pd.to_numeric(frame[left], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(frame[right], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(l) & np.isfinite(r)
    return l[valid], r[valid]


def audit_bilateral_order(frame: pd.DataFrame, columns: BilateralColumns) -> dict[str, Any]:
    eye_left, eye_right = _finite_pair(frame, columns.left_eye_x, columns.right_eye_x)
    mouth_left, mouth_right = _finite_pair(frame, columns.mouth_left_x, columns.mouth_right_x)

    def stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        if len(left) == 0:
            return {
                "evidence_count": 0,
                "image_left_fraction": None,
                "median_source_left_minus_right_x_px": None,
            }
        delta = left - right
        return {
            "evidence_count": int(len(delta)),
            "image_left_fraction": float(np.mean(delta < 0.0)),
            "median_source_left_minus_right_x_px": float(np.median(delta)),
        }

    eye = stats(eye_left, eye_right)
    mouth = stats(mouth_left, mouth_right)
    fractions = [
        value
        for value in (eye["image_left_fraction"], mouth["image_left_fraction"])
        if value is not None
    ]
    inferred_image_side = bool(fractions and float(np.median(fractions)) >= 0.75)
    confidence = float(np.clip(abs(float(np.median(fractions)) - 0.5) * 2.0, 0.0, 1.0)) if fractions else 0.0
    return {
        "schema_version": _CANONICAL_VERSION,
        "eye": eye,
        "mouth": mouth,
        "inferred_source_convention": "image_side" if inferred_image_side else "subject_anatomical_or_ambiguous",
        "inference_confidence": confidence,
        "geometry_mirror_recommended": False,
        "note": "The audit examines source x-order only; canonicalisation never moves pixel coordinates.",
    }


def _source_copy_name(column: str) -> str:
    return f"oculo source {column}"


def canonicalize_bilateral_semantics(
    frame: pd.DataFrame,
    *,
    columns: BilateralColumns,
    source_convention: SourceBilateralConvention = "pupil_image_side",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a canonical anatomical table while preserving exact source values.

    The standard FaceMapper bilateral columns in the returned table are
    canonical anatomical semantics.  Their numeric values are copied from the
    corresponding source-side columns; no x/y coordinate is transformed.
    Exact source values are retained in ``oculo source ...`` columns and in the
    parent artifact's immutable raw-source CSV.
    """
    if source_convention not in {"pupil_image_side", "subject_anatomical", "auto_audit"}:
        raise ValidationError(
            "source_bilateral_convention must be pupil_image_side, subject_anatomical or auto_audit"
        )
    audit = audit_bilateral_order(frame, columns)
    if source_convention == "pupil_image_side":
        apply_swap = True
        resolved_source = "image_side"
        decision_source = "explicit_pupil_facemapper_convention"
    elif source_convention == "subject_anatomical":
        apply_swap = False
        resolved_source = "subject_anatomical"
        decision_source = "explicit_subject_anatomical_override"
    else:
        apply_swap = audit["inferred_source_convention"] == "image_side"
        resolved_source = "image_side" if apply_swap else "subject_anatomical_or_ambiguous"
        decision_source = "x_order_audit"

    out = frame.copy()
    source_columns = [
        columns.left_eye_x,
        columns.left_eye_y,
        columns.right_eye_x,
        columns.right_eye_y,
        columns.mouth_left_x,
        columns.mouth_left_y,
        columns.mouth_right_x,
        columns.mouth_right_y,
    ]
    for column in source_columns:
        out[_source_copy_name(column)] = frame[column]

    if apply_swap:
        # Semantic reassignment only.  The copied values are not modified.
        out[columns.left_eye_x] = frame[columns.right_eye_x].to_numpy(copy=True)
        out[columns.left_eye_y] = frame[columns.right_eye_y].to_numpy(copy=True)
        out[columns.right_eye_x] = frame[columns.left_eye_x].to_numpy(copy=True)
        out[columns.right_eye_y] = frame[columns.left_eye_y].to_numpy(copy=True)
        out[columns.mouth_left_x] = frame[columns.mouth_right_x].to_numpy(copy=True)
        out[columns.mouth_left_y] = frame[columns.mouth_right_y].to_numpy(copy=True)
        out[columns.mouth_right_x] = frame[columns.mouth_left_x].to_numpy(copy=True)
        out[columns.mouth_right_y] = frame[columns.mouth_left_y].to_numpy(copy=True)

    out["oculo source bilateral convention"] = resolved_source
    out["oculo canonical bilateral convention"] = "subject_anatomical"
    out["oculo bilateral label swap applied"] = bool(apply_swap)
    out["oculo geometry mirror applied"] = False
    out["oculo bilateral canonicalization version"] = _CANONICAL_VERSION

    audit = {
        **audit,
        "requested_source_convention": source_convention,
        "resolved_source_convention": resolved_source,
        "canonical_output_convention": "subject_anatomical",
        "bilateral_label_swap_applied": bool(apply_swap),
        "geometry_mirror_applied": False,
        "decision_source": decision_source,
        "canonical_columns": {
            "subject_left_eye": [columns.left_eye_x, columns.left_eye_y],
            "subject_right_eye": [columns.right_eye_x, columns.right_eye_y],
            "subject_left_mouth": [columns.mouth_left_x, columns.mouth_left_y],
            "subject_right_mouth": [columns.mouth_right_x, columns.mouth_right_y],
        },
        "source_copy_columns": {_source_copy_name(column): column for column in source_columns},
    }
    return out, audit


def canonical_landmarks(
    landmarks: dict[str, tuple[float, float]] | None,
    *,
    bilateral_label_swap_applied: bool,
) -> dict[str, tuple[float, float]]:
    """Map a source landmark dictionary to canonical anatomical semantics.

    Coordinates are returned unchanged; only dictionary keys are reassigned.
    """
    src = dict(landmarks or {})
    if not bilateral_label_swap_applied:
        return src
    out = dict(src)
    out["left_eye"] = src.get("right_eye")
    out["right_eye"] = src.get("left_eye")
    out["mouth_left"] = src.get("mouth_right")
    out["mouth_right"] = src.get("mouth_left")
    return {key: value for key, value in out.items() if value is not None}
