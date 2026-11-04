from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True)


class SemanticRegion(str, Enum):
    INVALID_GAZE = "invalid_gaze"
    OUT_OF_FRAME = "out_of_frame"
    UNKNOWN = "unknown"
    ENVIRONMENT = "environment"

    FACE = "face"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    MOUTH = "mouth"
    OTHER_FACE = "other_face"
    HEAD = "head"
    NECK = "neck"

    TORSO = "torso"
    PELVIS = "pelvis"
    LEFT_UPPER_ARM = "left_upper_arm"
    LEFT_FOREARM = "left_forearm"
    LEFT_HAND = "left_hand"
    RIGHT_UPPER_ARM = "right_upper_arm"
    RIGHT_FOREARM = "right_forearm"
    RIGHT_HAND = "right_hand"
    LEFT_THIGH = "left_thigh"
    LEFT_LOWER_LEG = "left_lower_leg"
    LEFT_FOOT = "left_foot"
    RIGHT_THIGH = "right_thigh"
    RIGHT_LOWER_LEG = "right_lower_leg"
    RIGHT_FOOT = "right_foot"


FACE_REGIONS = {
    SemanticRegion.LEFT_EYE.value,
    SemanticRegion.RIGHT_EYE.value,
    SemanticRegion.NOSE.value,
    SemanticRegion.MOUTH.value,
    SemanticRegion.OTHER_FACE.value,
    SemanticRegion.FACE.value,
}

HEAD_COMPATIBLE_REGIONS = FACE_REGIONS | {
    SemanticRegion.HEAD.value,
    SemanticRegion.NECK.value,
}

BODY_REGIONS = {
    value.value for value in SemanticRegion
    if value.value not in {
        SemanticRegion.INVALID_GAZE.value,
        SemanticRegion.OUT_OF_FRAME.value,
        SemanticRegion.UNKNOWN.value,
        SemanticRegion.ENVIRONMENT.value,
        *FACE_REGIONS,
    }
}


class LandmarkDefinition(SemanticModel):
    """A landmark estimated by averaging any available marker aliases."""

    markers: list[str]
    minimum_markers: int = 1

    @field_validator("markers")
    @classmethod
    def non_empty_markers(cls, value: list[str]) -> list[str]:
        clean = [str(item).strip() for item in value if str(item).strip()]
        if not clean:
            raise ValueError("A landmark requires at least one marker alias")
        return clean

    @model_validator(mode="after")
    def minimum_is_possible(self) -> "LandmarkDefinition":
        if self.minimum_markers < 1 or self.minimum_markers > len(self.markers):
            raise ValueError("minimum_markers must be between 1 and marker count")
        return self


class PrimitiveDefinition(SemanticModel):
    region: SemanticRegion
    shape: Literal["sphere", "capsule"]
    center: str | None = None
    start: str | None = None
    end: str | None = None
    radius_mm: float
    priority: int = 0
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("radius_mm")
    @classmethod
    def positive_radius(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Primitive radius must be positive")
        return float(value)

    @model_validator(mode="after")
    def shape_fields(self) -> "PrimitiveDefinition":
        if self.shape == "sphere" and not self.center:
            raise ValueError("Sphere primitive requires center landmark")
        if self.shape == "capsule" and (not self.start or not self.end):
            raise ValueError("Capsule primitive requires start and end landmarks")
        return self


class BodyPrimitiveProfile(SemanticModel):
    schema_version: str = "oculostudio-body-primitives-1.0"
    name: str = "OculoStudio standard body primitives"
    units: Literal["millimetres"] = "millimetres"
    landmarks: dict[str, LandmarkDefinition]
    primitives: list[PrimitiveDefinition]
    marker_aliases: dict[str, list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DensePoseImportProfile(SemanticModel):
    schema_version: str = "oculostudio-densepose-import-1.0"
    label_mapping: dict[str, str] = Field(default_factory=dict)
    timestamp_tolerance_ms: float = 8.0
    unknown_numeric_policy: Literal["preserve", "unknown", "error"] = "preserve"

    @field_validator("timestamp_tolerance_ms")
    @classmethod
    def positive_tolerance(cls, value: float) -> float:
        if value < 0:
            raise ValueError("timestamp_tolerance_ms cannot be negative")
        return float(value)


DEFAULT_DENSEPOSE_TEXT_LABELS: dict[str, str] = {
    "face": SemanticRegion.FACE.value,
    "head": SemanticRegion.HEAD.value,
    "neck": SemanticRegion.NECK.value,
    "torso": SemanticRegion.TORSO.value,
    "body": SemanticRegion.TORSO.value,
    "pelvis": SemanticRegion.PELVIS.value,
    "hips": SemanticRegion.PELVIS.value,
    "left upper arm": SemanticRegion.LEFT_UPPER_ARM.value,
    "left upperarm": SemanticRegion.LEFT_UPPER_ARM.value,
    "left arm upper": SemanticRegion.LEFT_UPPER_ARM.value,
    "left lower arm": SemanticRegion.LEFT_FOREARM.value,
    "left forearm": SemanticRegion.LEFT_FOREARM.value,
    "left hand": SemanticRegion.LEFT_HAND.value,
    "right upper arm": SemanticRegion.RIGHT_UPPER_ARM.value,
    "right upperarm": SemanticRegion.RIGHT_UPPER_ARM.value,
    "right arm upper": SemanticRegion.RIGHT_UPPER_ARM.value,
    "right lower arm": SemanticRegion.RIGHT_FOREARM.value,
    "right forearm": SemanticRegion.RIGHT_FOREARM.value,
    "right hand": SemanticRegion.RIGHT_HAND.value,
    "left upper leg": SemanticRegion.LEFT_THIGH.value,
    "left thigh": SemanticRegion.LEFT_THIGH.value,
    "left lower leg": SemanticRegion.LEFT_LOWER_LEG.value,
    "left shin": SemanticRegion.LEFT_LOWER_LEG.value,
    "left foot": SemanticRegion.LEFT_FOOT.value,
    "right upper leg": SemanticRegion.RIGHT_THIGH.value,
    "right thigh": SemanticRegion.RIGHT_THIGH.value,
    "right lower leg": SemanticRegion.RIGHT_LOWER_LEG.value,
    "right shin": SemanticRegion.RIGHT_LOWER_LEG.value,
    "right foot": SemanticRegion.RIGHT_FOOT.value,
    "background": SemanticRegion.ENVIRONMENT.value,
    "environment": SemanticRegion.ENVIRONMENT.value,
    "unknown": SemanticRegion.UNKNOWN.value,
}


def canonical_region(value: Any, *, custom_mapping: dict[str, str] | None = None) -> tuple[str, bool]:
    """Return (label, mapped).

    Text labels are normalized into the shared ontology. Numeric DensePose IDs
    are deliberately not assigned anatomical meaning without an explicit
    mapping file because export conventions differ across pipelines.
    """

    if value is None:
        return SemanticRegion.UNKNOWN.value, False
    raw = str(value).strip()
    if not raw:
        return SemanticRegion.UNKNOWN.value, False
    normalized = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    mapping = {
        **DEFAULT_DENSEPOSE_TEXT_LABELS,
        **{
            " ".join(str(k).lower().replace("_", " ").replace("-", " ").split()): str(v)
            for k, v in (custom_mapping or {}).items()
        },
    }
    if normalized in mapping:
        return mapping[normalized], True
    try:
        number = int(float(raw))
    except (TypeError, ValueError):
        return normalized.replace(" ", "_"), False
    key = str(number)
    if key in (custom_mapping or {}):
        return str((custom_mapping or {})[key]), True
    return f"densepose_part_{number}", False
