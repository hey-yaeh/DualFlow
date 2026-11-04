from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oculostudio import ARTIFACT_SCHEMA_VERSION, PROJECT_SCHEMA_VERSION, __version__
from oculostudio.util import new_id, utc_now


class OSModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True)


class AssetKind(str, Enum):
    PUPIL_NATIVE = "pupil_native"
    PUPIL_CLOUD = "pupil_cloud"
    FACE_MAPPER = "face_mapper"
    DENSEPOSE_EXPORT = "densepose_export"
    VIDEO = "video"
    AUDIO = "audio"
    C3D = "c3d"
    MOCAP_NPZ = "mocap_npz"
    CALIBRATION = "calibration"
    TABLE = "table"
    ARCHIVE = "archive"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


class ArtifactKind(str, Enum):
    NATIVE_REFERENCE_TRIM = "native_reference_trim"
    ALIGNED_CLIP_SET = "aligned_clip_set"
    ALIGNED_CLIP_PACKAGE = "aligned_clip_package"
    ALIGNED_FACIAL_VIDEO = "aligned_facial_video"
    MOTIONSTUDIO_SCENE = "motionstudio_scene"
    FRAME_MAP = "frame_map"
    TIME_SYNC = "time_sync"
    TRIMMED_NATIVE = "trimmed_native"
    TRIMMED_TIMESERIES = "trimmed_timeseries"
    TRIMMED_FACE_MAPPER = "trimmed_face_mapper"
    FACE_MAPPER_FILTER = "face_mapper_filter"
    PRIVACY_MASK_PREVIEW = "privacy_mask_preview"
    PRIVACY_VIDEO = "privacy_video"
    PRIVACY_CLIP = "privacy_clip"
    PARTICIPANT_ASSOCIATION = "participant_association"
    FACE_AOI = "face_aoi"
    BODY_AOI = "body_aoi"
    OBJECT_AOI = "object_aoi"
    TARGET_FUSION = "target_fusion"
    AOI_REVIEW_VIDEO = "aoi_review_video"
    DYADIC_ATTENTION = "dyadic_attention"
    CALIBRATION_PROFILE = "calibration_profile"
    CALIBRATION_ACCEPTANCE = "calibration_acceptance"
    ETM_ROLE_ASSIGNMENT = "etm_role_assignment"
    GAZE_RAYS = "gaze_rays"
    SEMANTIC_TARGETS = "semantic_targets"
    QC_REPORT = "qc_report"
    RESULTS_PACKAGE = "results_package"
    THUMBNAIL = "thumbnail"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TimeDomain(OSModel):
    id: str = Field(default_factory=lambda: new_id("time"))
    label: str
    unit: Literal["nanoseconds"] = "nanoseconds"
    clock_type: Literal["absolute", "relative", "frame_pts", "global"] = "relative"
    asset_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimeAnchor(OSModel):
    id: str = Field(default_factory=lambda: new_id("anchor"))
    source_ns: int
    target_ns: int
    label: str = ""
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def positive_weight(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Anchor weight must be positive")
        return float(value)


class TimeMapQuality(OSModel):
    anchor_count: int = 0
    rms_residual_ns: float = 0.0
    max_abs_residual_ns: float = 0.0
    median_abs_residual_ns: float = 0.0
    robust_iterations: int = 0
    warning: str | None = None


class TimeMap(OSModel):
    id: str = Field(default_factory=lambda: new_id("map"))
    source_domain_id: str
    target_domain_id: str
    scale: float = 1.0
    offset_ns: float = 0.0
    # New maps retain integer origins so epoch-scale nanoseconds are never
    # multiplied directly by float64. None keeps old v0.3.x manifests valid.
    source_origin_ns: int | None = None
    target_origin_ns: int | None = None
    anchors: list[TimeAnchor] = Field(default_factory=list)
    quality: TimeMapQuality = Field(default_factory=TimeMapQuality)
    valid_source_start_ns: int | None = None
    valid_source_end_ns: int | None = None
    method: str = "manual_affine"
    created_at: str = Field(default_factory=utc_now)

    @field_validator("scale")
    @classmethod
    def valid_scale(cls, value: float) -> float:
        if value <= 0 or not value < float("inf"):
            raise ValueError("Time-map scale must be finite and positive")
        return float(value)


class AssetRecord(OSModel):
    id: str = Field(default_factory=lambda: new_id("asset"))
    label: str
    kind: AssetKind = AssetKind.UNKNOWN
    source_path: str
    source_sha256: str | None = None
    size_bytes: int | None = None
    recording_id: str | None = None
    participant_id: str | None = None
    primary_time_domain_id: str | None = None
    time_domain_ids: list[str] = Field(default_factory=list)
    probe_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    available: bool = True
    created_at: str = Field(default_factory=utc_now)


class Clip(OSModel):
    id: str = Field(default_factory=lambda: new_id("clip"))
    label: str
    start_global_ns: int
    end_global_ns: int
    notes: str = ""
    source_display_domain_id: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_interval(self) -> "Clip":
        if self.end_global_ns <= self.start_global_ns:
            raise ValueError("Clip end must be greater than clip start")
        return self

    @property
    def duration_ns(self) -> int:
        return self.end_global_ns - self.start_global_ns


class ArtifactFile(OSModel):
    role: str
    path: str
    size_bytes: int | None = None
    sha256: str | None = None


class ArtifactRecord(OSModel):
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("artifact"))
    kind: ArtifactKind
    label: str
    session_id: str
    source_asset_ids: list[str] = Field(default_factory=list)
    parent_artifact_ids: list[str] = Field(default_factory=list)
    clip_id: str | None = None
    files: list[ArtifactFile] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    qc: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class JobRecord(OSModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    operation: str
    label: str
    session_id: str | None = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    stage: str = "Queued"
    created_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    log: list[str] = Field(default_factory=list)

    @field_validator("progress")
    @classmethod
    def bounded_progress(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class Session(OSModel):
    id: str = Field(default_factory=lambda: new_id("session"))
    label: str
    participant_ids: list[str] = Field(default_factory=list)
    assets: dict[str, AssetRecord] = Field(default_factory=dict)
    time_domains: dict[str, TimeDomain] = Field(default_factory=dict)
    time_maps: dict[str, TimeMap] = Field(default_factory=dict)
    clips: dict[str, Clip] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    global_time_domain_id: str = "global"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def ensure_global_domain(self) -> "Session":
        if self.global_time_domain_id not in self.time_domains:
            self.time_domains[self.global_time_domain_id] = TimeDomain(
                id=self.global_time_domain_id,
                label="Global Session Time",
                clock_type="global",
            )
        return self


class CalibrationAssetReference(OSModel):
    """Canonical reference to a calibration source registered in one Session.

    The underlying bytes are never duplicated merely because the Calibration Set
    applies to several Sessions. Target Sessions receive traceable aliases that
    point to the same immutable source path.
    """

    source_session_id: str
    source_asset_id: str
    label: str = ""
    source_path: str = ""
    source_sha256: str | None = None
    kind: str = "unknown"
    calibration_role: str = "calibration_source"
    participant_id: str | None = None
    hardware_hint: str | None = None
    added_at: str = Field(default_factory=utc_now)


class CalibrationProfileReference(OSModel):
    source_session_id: str
    source_artifact_id: str
    label: str = ""
    profile_path: str = ""
    profile_sha256: str | None = None
    hardware_id: str | None = None
    mount_version: str | None = None
    added_at: str = Field(default_factory=utc_now)


class CalibrationSetRevision(OSModel):
    id: str = Field(default_factory=lambda: new_id("calrev"))
    revision_number: int = 1
    label: str = "Revision 1"
    status: Literal["draft", "ready", "superseded", "archived"] = "draft"
    source_assets: list[CalibrationAssetReference] = Field(default_factory=list)
    profiles: list[CalibrationProfileReference] = Field(default_factory=list)
    notes: str = ""
    created_at: str = Field(default_factory=utc_now)
    modified_at: str = Field(default_factory=utc_now)


class CalibrationSet(OSModel):
    """Project-level calibration collection shared by several Sessions."""

    id: str = Field(default_factory=lambda: new_id("calset"))
    code: str
    label: str
    purpose: Literal["hardware", "known_target_gaze", "combined"] = "combined"
    applicable_session_ids: list[str] = Field(default_factory=list)
    owner_session_id: str | None = None
    active_revision_id: str | None = None
    revisions: dict[str, CalibrationSetRevision] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    modified_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def valid_active_revision(self) -> "CalibrationSet":
        if self.active_revision_id and self.active_revision_id not in self.revisions:
            self.active_revision_id = None
        if not self.active_revision_id and self.revisions:
            self.active_revision_id = next(iter(self.revisions))
        return self



class Project(OSModel):
    schema_version: str = PROJECT_SCHEMA_VERSION
    app_version: str = __version__
    id: str = Field(default_factory=lambda: new_id("project"))
    name: str
    root: str
    sessions: dict[str, Session] = Field(default_factory=dict)
    calibration_sets: dict[str, CalibrationSet] = Field(default_factory=dict)
    selected_session_id: str | None = None
    jobs: dict[str, JobRecord] = Field(default_factory=dict)
    revision: int = 0
    created_at: str = Field(default_factory=utc_now)
    modified_at: str = Field(default_factory=utc_now)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_selected_session(self) -> "Project":
        if self.selected_session_id and self.selected_session_id not in self.sessions:
            self.selected_session_id = None
        if not self.selected_session_id and self.sessions:
            self.selected_session_id = next(iter(self.sessions))
        return self

    @property
    def manifest_path(self) -> Path:
        return Path(self.root) / "project.oculostudio.json"
