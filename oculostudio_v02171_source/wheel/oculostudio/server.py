from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import re
import mimetypes
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from oculostudio import __version__
from oculostudio.adapters.archive import MaterializationCache
from oculostudio.adapters.pupil import find_cloud_candidates, materialized_native_files
from oculostudio.adapters.video import make_thumbnail
from oculostudio.adapters.native_reference_clock import native_options
from oculostudio.services.native_reference_trim import preview_native_reference_trim, export_native_reference_trim
from oculostudio.domain.models import ArtifactKind, Clip
from oculostudio.domain.timecode import parse_timecode_ns
from oculostudio.domain.video_input_time import parse_native_trim_times
from oculostudio.errors import OculoStudioError, ValidationError
from oculostudio.jobs import JobManager
from oculostudio.services import (
    add_asset,
    build_native_cloud_frame_map,
    build_session_qc,
    create_calibration_profile,
    create_legacy_gaze3d_calibration_profile,
    import_calibration_profile,
    register_existing_measured_profile_hardware,
    validate_calibration_profile_transfer,
    run_transferred_calibration_acceptance,
    create_etm_role_assignment,
    bind_calibration_profiles_to_assignment,
    create_privacy_video,
    create_filtered_facemapper,
    facemapper_live_manifest_path,
    facemapper_live_video_path,
    create_trimmed_privacy_video,
    create_synchronized_privacy_video,
    create_privacy_mask_preview,
    render_pretrim_overlay_preview,
    pretrim_overlay_video_path,
    create_participant_association,
    create_face_aoi_annotations,
    create_body_aoi_annotations,
    reprocess_body_aoi_review_reduction,
    create_object_aoi_annotations,
    create_dyadic_attention_annotations,
    fuse_face_and_body_aoi,
    review_options,
    review_manifest,
    review_frame,
    review_frame_window,
    save_face_review_decision,
    apply_face_aoi_manual_review,
    save_body_review_decision,
    apply_body_aoi_manual_review,
    render_aoi_review_video,
    review_unblurred_video_path,
    create_semantic_gaze_targets,
    export_qc_report,
    create_results_package,
    build_interaction_workspace,
    build_project_workspace,
    set_interaction_context,
    inspect_project_location,
    project_calibration_library,
    apply_shared_calibration_profiles,
    list_calibration_sets,
    create_calibration_set,
    set_calibration_sessions,
    create_calibration_revision,
    activate_calibration_revision,
    register_calibration_profile_ref,
    register_calibration_profile_refs,
    materialize_calibration_set,
    calibration_set_profile_build_status,
    build_calibration_set_profiles,
    fit_and_store_time_map,
    automatic_pupil_vicon_sync,
    accept_automatic_time_sync,
    get_automatic_time_sync_preview,
    automatic_shared_audio_video_sync,
    accept_audio_video_sync,
    get_audio_video_sync_preview,
    audio_video_sync_video_path,
    create_c3d_defined_aligned_clips,
    export_aligned_clip_package,
    export_aligned_facial_video,
    generate_gaze_rays,
    refresh_asset,
    remove_asset,
    move_artifact_to_trash,
    move_artifacts_to_trash,
    build_artifact_cleanup_plan,
    move_orphan_outputs_to_trash,
    purge_project_trash,
    run_multimodal_clip,
)
from oculostudio.storage import ProjectStore
from oculostudio.util import find_executable, safe_name
from oculostudio.media_tools import media_tools_status, privacy_encoder_preflight, save_tool_settings
from oculostudio.media_runtime import complete_preflight, probe_preflight, install_runtime_job
from oculostudio.file_dialogs import isolated_dialog
from oculostudio.services.asset_import import UploadManager, import_local_asset, CHUNK_BYTES
from oculostudio.services.asset_identity import asset_identity_view, update_asset_identity
from oculostudio.services.calibration_registry import (
    calibration_metadata, expand_calibration_paths, infer_participant,
    normalize_calibration_set_id, set_asset_calibration_scope,
    scan_calibration_sources, import_calibration_batch,
)
from oculostudio.services.privacy import PRIVACY_FRAME_COVERAGE_ENGINE
from oculostudio.services.privacy_clip import PRIVACY_CLIP_ENGINE
from oculostudio.services.privacy_final import SYNCHRONIZED_PRIVACY_ENGINE
from oculostudio.services.participant_association import PARTICIPANT_ASSOCIATION_ENGINE
from oculostudio.services.face_aoi import FACE_AOI_ENGINE
from oculostudio.services.preview3d import (
    get_etm_assignment_preview,
    get_gaze_rays_preview,
    get_dyadic_gaze_rays_preview,
    get_calibration_motion_preview,
)
from oculostudio.services.etm_role_assignment import resolve_assignment_for_role
from oculostudio.services.gaze3d import resolve_gaze3d_time_path
from starlette.concurrency import run_in_threadpool
from urllib.parse import urlsplit


class ProjectCreateRequest(BaseModel):
    root: str
    name: str
    session_label: str = "S001"
    session_labels: list[str] | None = None
    selected_session_label: str | None = None
    planned_interactions: list[str] = Field(default_factory=list, max_length=200)
    settings: dict[str, Any] = Field(default_factory=dict)


class ProjectLocationCheckRequest(BaseModel):
    parent: str
    folder: str


class ProjectOpenRequest(BaseModel):
    path: str


class SessionRequest(BaseModel):
    label: str


class SessionBatchRequest(BaseModel):
    labels: list[str] = Field(min_length=1, max_length=200)
    select_first: bool = False


class InteractionPlanRequest(BaseModel):
    interaction_ids: list[str] = Field(default_factory=list, max_length=200)
    replace: bool = False


class SharedCalibrationApplyRequest(BaseModel):
    source_session_id: str
    profile_artifact_ids: list[str] = Field(min_length=1, max_length=20)
    target_session_ids: list[str] = Field(min_length=1, max_length=200)


class CalibrationSetCreateRequest(BaseModel):
    code: str
    label: str | None = None
    applicable_session_ids: list[str] = Field(default_factory=list, max_length=200)
    purpose: Literal["hardware", "known_target_gaze", "combined"] = "combined"
    owner_session_id: str | None = None
    notes: str = ""


class CalibrationSetSessionsRequest(BaseModel):
    session_ids: list[str] = Field(default_factory=list, max_length=200)
    revision_id: str | None = None


class CalibrationSetRevisionRequest(BaseModel):
    label: str | None = None
    notes: str = ""
    inherit_active: bool = True
    activate: bool = True


class CalibrationSetActivateRequest(BaseModel):
    revision_id: str


class CalibrationSetProfileRequest(BaseModel):
    source_session_id: str
    profile_artifact_id: str
    revision_id: str | None = None


class CalibrationSetMaterializeRequest(BaseModel):
    session_ids: list[str] | None = None
    revision_id: str | None = None


class CalibrationSetProfileBuildRequest(BaseModel):
    session_id: str
    revision_id: str | None = None
    sample_every_n_frames: int = Field(default=4, ge=1, le=60)
    max_detection_frames: int = Field(default=240, ge=20, le=2000)
    max_pose_gap_sec: float = Field(default=0.1, gt=0.0, le=2.0)
    minimum_tracker_identity_confidence: Literal["LOW", "SINGLE_CANDIDATE", "MEDIUM", "HIGH"] = "MEDIUM"
    mount_version: str = Field(default="mount_v1", max_length=120)
    force_rebuild: bool = False
    output_g3dcal: bool = True


class InteractionContextRequest(BaseModel):
    interaction_id: str
    role_id: str = "R01"
    workflow: Literal["face", "scene", "dyadic"] = "face"


class AssetAddRequest(BaseModel):
    session_id: str
    path: str
    label: str | None = None
    participant_id: str | None = None
    checksum: bool = True


class AssetIdentityUpdateRequest(BaseModel):
    participant_id: str | None = None
    interaction_id: str | None = None
    role_id: str | None = None


class AssetIdentityBatchUpdateRequest(AssetIdentityUpdateRequest):
    asset_ids: list[str] = Field(min_length=1, max_length=200)


class TimeMapRequest(BaseModel):
    session_id: str
    source_domain_id: str
    target_domain_id: str
    anchors: list[dict[str, Any]]
    method: str = "robust_affine"


class AutomaticTimeMapRequest(BaseModel):
    session_id: str
    timeseries_asset_id: str
    tracker_asset_id: str | None = None
    etm_assignment_artifact_id: str | None = None
    wearer_role_id: str | None = None
    maximum_offset_sec: float = Field(default=120.0, gt=0, le=3600)
    sample_rate_hz: float = Field(default=120.0, ge=20, le=240)
    save_if_high: bool = True


class AcceptAutomaticTimeMapRequest(BaseModel):
    session_id: str
    candidate_id: str


class AudioVideoSyncJobRequest(BaseModel):
    session_id: str
    source_video_asset_id: str
    reference_video_asset_id: str
    source_video_member: str | None = None
    reference_video_member: str | None = None
    wearer_role_id: str | None = None
    maximum_offset_sec: float = Field(default=300.0, gt=0, le=7200)
    decode_rate_hz: int = Field(default=8000, ge=4000, le=48000)
    feature_rate_hz: float = Field(default=100.0, ge=25.0, le=200.0)
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class AcceptAudioVideoSyncRequest(BaseModel):
    session_id: str
    candidate_id: str
    visual_offset_adjustment_ms: float = Field(default=0.0, ge=-2000.0, le=2000.0)
    visual_review_confirmed: bool = False
    ffprobe_path: str | None = None


class ClipRequest(BaseModel):
    session_id: str
    label: str
    start_global_ns: int | None = None
    end_global_ns: int | None = None
    start_timecode: str | None = None
    end_timecode: str | None = None
    frame_rate: float = 30.0
    notes: str = ""
    source_display_domain_id: str | None = None


class FrameMapRequest(BaseModel):
    session_id: str
    native_asset_id: str
    cloud_asset_id: str
    maximum_error_ms: float = 5.0


class ClipJobRequest(BaseModel):
    session_id: str
    clip_id: str
    asset_ids: list[str]
    interval_policy: Literal["clip", "inside"] = "clip"


class NativeTimeInputRequest(BaseModel):
    start_timecode: str = Field(min_length=1, max_length=64)
    duration_timecode: str = Field(min_length=1, max_length=64)
    timecode_fps: int = Field(default=30, strict=True)


class NativeTrimPreviewRequest(BaseModel):
    session_id: str
    native_asset_id: str
    timeseries_asset_id: str
    facemapper_asset_id: str
    start_timecode: str
    duration_timecode: str
    timecode_fps: int = Field(default=30, strict=True)
    label: str = "Native data clip"
    native_video_member: str | None = None
    interval_policy: Literal["clip", "inside"] = "clip"
    face_tolerance_ms: float = Field(default=5.0, gt=0, le=15)
    ffprobe_path: str | None = None


class NativeTrimExportRequest(BaseModel):
    session_id: str
    preview_id: str


class C3DDefinedAlignedClipsJobRequest(BaseModel):
    session_id: str
    calibration_handoff_artifact_id: str
    role_inputs: dict[str, dict[str, Any]]
    interval_policy: Literal["clip", "inside"] = "clip"
    face_tolerance_ms: float = Field(default=5.0, gt=0, le=15)
    boundary_tolerance_ms: float = Field(default=20.0, ge=0, le=100)
    label_prefix: str = Field(default="C3D-defined aligned clip", min_length=1, max_length=160)
    ffprobe_path: str | None = None


class AlignedClipExportJobRequest(BaseModel):
    session_id: str
    native_trim_artifact_id: str
    label: str = Field(default="Aligned clip package", min_length=1, max_length=180)
    include_ego_video: bool = True
    include_tsd: bool = True
    include_facemapper: bool = True
    output_fps: int = Field(default=30, strict=True)
    preserve_audio: bool = True
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class AlignedFacialVideoExportJobRequest(BaseModel):
    session_id: str
    native_trim_artifact_id: str
    time_sync_artifact_id: str
    label: str = Field(default="Aligned facial video", min_length=1, max_length=180)
    output_fps: int = Field(default=30, strict=True)
    preserve_audio: bool = True
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = Field(default="fast", min_length=1, max_length=32)
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class NativeTrimOverlayPreviewRequest(BaseModel):
    session_id: str
    preview_id: str
    scope: Literal["boundaries", "representative", "distributed", "full"] = "representative"
    segment_seconds: float = Field(default=8.0, gt=0)
    sample_count: int = Field(default=7, ge=3, le=12)
    output_fps: int = Field(default=30, strict=True)
    include_face_boxes: bool = True
    include_landmarks: bool = True
    include_aoi: bool = True
    include_gaze: bool = True
    left_right_policy: Literal["anatomical_auto", "as_exported", "swap_eyes"] = "anatomical_auto"
    face_tolerance_ms: float = Field(default=8.0, gt=0, le=30)
    gaze_frame_tolerance_ms: float = Field(default=20.0, gt=0, le=100)
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class FaceMapperFilterJobRequest(BaseModel):
    session_id: str
    native_trim_artifact_id: str
    label: str = "Filtered Face Mapper"
    large_box_area_ratio: float = Field(default=0.075, ge=0.01, le=0.5)
    maximum_area_ratio: float = Field(default=0.45, ge=0.05, le=0.95)
    minimum_landmark_span_x_ratio: float = Field(default=0.16, ge=0.01, le=1.0)
    minimum_landmark_span_y_ratio: float = Field(default=0.22, ge=0.01, le=1.0)
    minimum_interocular_ratio: float = Field(default=0.10, ge=0.01, le=1.0)
    minimum_landmark_in_box_fraction: float = Field(default=0.80, ge=0.0, le=1.0)
    maximum_landmark_center_offset: float = Field(default=0.36, ge=0.0, le=2.0)
    maximum_scale_jump_ratio: float = Field(default=3.5, ge=1.0, le=20.0)
    minimum_track_frames: int = Field(default=3, ge=1, le=100)
    render_preview: bool = True
    render_source_preview: bool = True
    source_bilateral_convention: Literal["pupil_image_side", "subject_anatomical", "auto_audit"] = "pupil_image_side"
    preview_sample_count: int = Field(default=9, ge=3, le=15)
    preview_segment_seconds: float = Field(default=6.0, ge=1.0, le=60.0)
    preview_fps: int = Field(default=15, ge=5, le=60)
    preview_scope: Literal["sampled", "full"] = "sampled"
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class PrivacyJobRequest(BaseModel):
    session_id: str
    native_asset_id: str
    facemapper_asset_id: str | None = None
    facemapper_filter_artifact_id: str | None = None
    native_trim_artifact_id: str | None = None
    clip_id: str | None = None
    mode: Literal["gaussian", "pixelate", "solid"] = "gaussian"
    strength: int = 30
    padding_percent: float = 0.0
    shape: Literal["rectangle", "ellipse"] = "rectangle"
    gap_fill_frames: int = 3
    edge_fill_frames: int = 1
    repair_maximum_center_shift_per_frame: float = 0.85
    repair_maximum_scale_ratio: float = 1.75
    repair_minimum_track_observations: int = 3
    timestamp_tolerance_ms: float = 5.0
    detector_mode: Literal["facemapper", "haar", "hybrid"] = "facemapper"
    candidate_policy: Literal["accepted_review", "accepted_only", "accepted_repair"] = "accepted_review"
    exact_filter_match: bool = False
    coverage_policy: Literal["strict", "allow"] = "strict"
    preserve_audio: bool = True
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class SynchronizedPrivacyJobRequest(BaseModel):
    session_id: str
    native_asset_id: str
    native_trim_artifact_id: str
    facemapper_asset_id: str | None = None
    facemapper_filter_artifact_id: str | None = None
    privacy_mask_preview_artifact_id: str | None = None
    output_fps: int = Field(default=30, strict=True)
    label: str = "Synchronized Privacy video"
    mode: Literal["gaussian", "pixelate", "solid"] = "gaussian"
    strength: int = 30
    padding_percent: float = 0.0
    shape: Literal["rectangle", "ellipse"] = "rectangle"
    gap_fill_frames: int = 3
    edge_fill_frames: int = 1
    repair_maximum_center_shift_per_frame: float = 0.85
    repair_maximum_scale_ratio: float = 1.75
    repair_minimum_track_observations: int = 3
    timestamp_tolerance_ms: float = 5.0
    detector_mode: Literal["facemapper", "haar", "hybrid"] = "facemapper"
    candidate_policy: Literal["accepted_review", "accepted_only", "accepted_repair"] = "accepted_repair"
    exact_filter_match: bool = False
    coverage_policy: Literal["strict", "allow"] = "strict"
    preserve_audio: bool = True
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class PrivacyMaskPreviewJobRequest(BaseModel):
    session_id: str
    native_asset_id: str
    native_trim_artifact_id: str
    facemapper_filter_artifact_id: str
    label: str = "Accepted + repair mask preview"
    output_fps: int = Field(default=30, ge=5, le=60)
    preview_scope: Literal["sampled", "full"] = "full"
    preview_sample_count: int = Field(default=9, ge=3, le=15)
    preview_segment_seconds: float = Field(default=6.0, ge=1.0, le=60.0)
    padding_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    gap_fill_frames: int = Field(default=3, ge=0, le=30)
    edge_fill_frames: int = Field(default=1, ge=0, le=10)
    repair_maximum_center_shift_per_frame: float = Field(default=0.85, gt=0.0, le=5.0)
    repair_maximum_scale_ratio: float = Field(default=1.75, ge=1.0, le=10.0)
    repair_minimum_track_observations: int = Field(default=3, ge=2, le=100)
    timestamp_tolerance_ms: float = Field(default=5.0, gt=0.0, le=1000.0)
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class PrivacyClipJobRequest(BaseModel):
    session_id: str
    privacy_artifact_id: str
    native_asset_id: str | None = None
    native_trim_artifact_id: str | None = None
    start_timecode: str | None = None
    duration_timecode: str | None = None
    timecode_fps: int = Field(default=30, strict=True)
    output_fps: int = Field(default=30, strict=True)
    native_video_member: str | None = None
    label: str = "Trimmed privacy video"
    preserve_audio: bool = True
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class ParticipantAssociationJobRequest(BaseModel):
    session_id: str
    native_trim_artifact_id: str
    etm_assignment_artifact_id: str
    wearer_role_id: str
    calibration_profile_artifact_id: str | None = None
    calibration_profile_path: str | None = None
    facemapper_filter_artifact_id: str | None = None
    target_policy: Literal["partner_only", "tracked_participants", "all_detected"] = "partner_only"
    alignment_mode: Literal["timegraph", "pretrimmed_shared_start"] = "pretrimmed_shared_start"
    shared_start_offset_sec: float = Field(default=0.0, ge=-5.0, le=5.0)
    face_tolerance_ms: float = Field(default=8.0, gt=0, le=30)
    max_pose_gap_sec: float = Field(default=0.05, ge=0, le=1.0)
    max_head_marker_gap_sec: float = Field(default=0.05, ge=0, le=1.0)
    maximum_normalized_center_error: float = Field(default=1.25, gt=0, le=5.0)
    minimum_projected_iou: float = Field(default=0.015, ge=0, le=1.0)
    minimum_assignment_margin: float = Field(default=0.12, ge=0, le=5.0)
    minimum_identity_confidence: float = Field(default=0.42, ge=0, le=1.0)
    output_parquet: bool = False
    label: str = "Participant association"


class CalibrationAcceptanceJobRequest(BaseModel):
    session_id: str
    transferred_profile_artifact_id: str
    native_trim_artifact_id: str
    etm_assignment_artifact_id: str
    wearer_role_id: str
    participant_association_artifact_id: str | None = None
    known_target_observations_path: str | None = None
    label: str = "Transferred calibration acceptance"
    maximum_sync_rms_ms: float = Field(default=25.0, gt=0, le=500)
    minimum_projection_valid_fraction: float = Field(default=0.60, ge=0, le=1)
    minimum_partner_match_frame_fraction: float = Field(default=0.55, ge=0, le=1)
    maximum_median_normalized_center_error: float = Field(default=0.75, gt=0, le=5)
    minimum_median_projected_iou: float = Field(default=0.03, ge=0, le=1)
    maximum_systematic_offset_normalized: float = Field(default=0.40, ge=0, le=5)
    maximum_review_fraction: float = Field(default=0.40, ge=0, le=1)
    known_target_maximum_timestamp_error_ms: float = Field(default=25.0, gt=0, le=500)
    known_target_maximum_median_angular_error_deg: float = Field(default=3.0, gt=0, le=45)
    known_target_minimum_pass_fraction: float = Field(default=0.67, ge=0, le=1)
    known_target_minimum_observation_count: int = Field(default=3, ge=1, le=10000)
    known_target_maximum_point_to_ray_distance_mm: float = Field(default=80.0, gt=0, le=5000)
    output_parquet: bool = False


class FaceAOIJobRequest(BaseModel):
    session_id: str
    native_trim_artifact_id: str
    participant_association_artifact_id: str | None = None
    facemapper_filter_artifact_id: str | None = None
    target_person_policy: Literal["partner_only", "tracked_participants", "all_detected"] = "all_detected"
    privacy_clip_artifact_id: str | None = None
    label: str = "Face AOI"
    face_tolerance_ms: float = Field(default=8.0, gt=0, le=30)
    face_left_right_policy: Literal["canonical_filter", "anatomical_auto", "as_exported", "swap_eyes"] = "canonical_filter"
    face_landmark_geometry_policy: Literal["as_exported", "auto_3d", "mirror_x"] = "as_exported"
    fine_feature_aoi_scale: float = Field(default=1.55, ge=0.75, le=2.0)
    eye_feature_aoi_scale: float = Field(default=1.35, ge=0.75, le=1.75)
    nose_region_mode: Literal["nasal_envelope", "bridge_polygon", "ellipse"] = "nasal_envelope"
    mouth_center_policy: Literal["adaptive_lip_envelope", "anatomical_strict", "speech_sensitive_mouth_jaw", "landmark_midpoint", "lowered_landmark_midpoint", "nose_base_to_face_bottom"] = "adaptive_lip_envelope"
    face_edge_tolerance_policy: Literal["exact", "edge_inclusive", "forgiving"] = "edge_inclusive"
    auto_review_policy: Literal["off", "conservative", "adaptive", "balanced"] = "adaptive"
    face_geometry_time_policy: Literal["track_interpolated", "nearest_frame"] = "track_interpolated"
    face_interpolation_max_gap_ms: float = Field(default=75.0, ge=10.0, le=500.0)
    face_interpolation_max_motion_ratio: float = Field(default=1.25, ge=0.25, le=5.0)
    output_parquet: bool = False
    render_overlay: bool = True
    preserve_audio: bool = True
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class BodyAOIJobRequest(BaseModel):
    session_id: str
    gaze_rays_artifact_id: str
    timeseries_asset_id: str | None = None
    body_asset_ids: list[str] = Field(default_factory=list)
    body_profile_path: str | None = None
    entity_marker_prefixes: dict[str, str] = Field(default_factory=dict)
    etm_assignment_artifact_id: str | None = None
    native_trim_artifact_id: str | None = None
    wearer_role_id: str | None = None
    wearer_body_asset_id: str | None = None
    partner_body_asset_id: str | None = None
    include_self_body: bool = True
    exclude_self_head_neck: bool = True
    self_near_origin_exclusion_m: float = Field(default=0.20, ge=0, le=2)
    first_hit_occlusion: bool = True
    output_granularity: Literal["coarse_social_regions", "detailed_regions"] = "coarse_social_regions"
    auto_review_policy: Literal[
        "off", "conservative", "adaptive", "evidence_guided", "semantic_guided", "balanced"
    ] = "semantic_guided"
    geometry_acceptance_policy: Literal["required", "warn", "off"] = "required"
    minimum_torso_coverage: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_body_frame_coverage: float = Field(default=0.90, ge=0.0, le=1.0)
    max_marker_gap_sec: float = 0.1
    uncertainty_cone_deg: float = 1.5
    maximum_target_distance_m: float = 10.0
    scene_frame_tolerance_ms: float = 20.0
    sample_clock_chunk_size: int = Field(default=4096, ge=128, le=50000)
    label: str = "Body AOI"
    output_parquet: bool = False


class BodyAOIReviewReductionJobRequest(BaseModel):
    session_id: str
    body_aoi_artifact_id: str
    auto_review_policy: Literal[
        "off", "conservative", "adaptive", "evidence_guided", "semantic_guided", "balanced"
    ] = "semantic_guided"
    label: str | None = None
    output_parquet: bool = False


class ObjectAOIJobRequest(BaseModel):
    session_id: str
    gaze_rays_artifact_id: str
    timeseries_asset_id: str | None = None
    object_asset_ids: list[str] = Field(default_factory=list)
    object_profile_path: str | None = None
    object_id_overrides: dict[str, str] = Field(default_factory=dict)
    native_trim_artifact_id: str | None = None
    max_marker_gap_sec: float = 0.1
    uncertainty_cone_deg: float = 1.5
    maximum_target_distance_m: float = 10.0
    scene_frame_tolerance_ms: float = 20.0
    label: str = "Object AOI"
    output_parquet: bool = False


class TargetFusionJobRequest(BaseModel):
    session_id: str
    face_aoi_artifact_id: str
    body_aoi_artifact_id: str
    object_aoi_artifact_id: str | None = None
    label: str = "Full gaze targets"
    output_parquet: bool = False


class DyadicAttentionJobRequest(BaseModel):
    session_id: str
    role_a_target_artifact_id: str
    role_b_target_artifact_id: str
    role_a_id: str = "R01"
    role_b_id: str = "R02"
    alignment_mode: Literal["pretrimmed_shared_start", "timegraph"] = "pretrimmed_shared_start"
    role_a_time_domain_id: str | None = None
    role_b_time_domain_id: str | None = None
    canonical_fps: int = Field(default=30, ge=1, le=120)
    cross_wearer_tolerance_ms: float = Field(default=25.0, gt=0, le=500)
    minimum_episode_ms: float = Field(default=100.0, ge=0, le=10000)
    short_gap_ms: float = Field(default=50.0, ge=0, le=5000)
    label: str = "Dyadic attention"
    output_parquet: bool = False


class AOIReviewVideoJobRequest(BaseModel):
    session_id: str
    mode: Literal["face", "body", "fusion"]
    artifact_id: str
    video_artifact_id: str | None = None
    label: str = "AOI review video"
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    draw_face_regions: bool = True
    draw_trail: bool = True
    draw_confidence: bool = True
    preserve_audio: bool = True
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class FaceAOIReviewDecisionRequest(BaseModel):
    video_artifact_id: str | None = None
    output_frame_index: int = Field(ge=0)
    gaze_sample_uid: str | None = None
    apply_to: Literal["sample", "frame", "fixation", "review_episode"] = "sample"
    action: Literal["accept_auto", "override", "exclude", "defer", "clear"]
    override_relation: Literal["partner", "self", "other_person"] | None = None
    override_entity_id: str | None = Field(default=None, max_length=128)
    override_region: Literal["left_eye", "right_eye", "nose", "mouth", "other_face", "face", "no_face"] | None = None
    note: str = Field(default="", max_length=1000)
    reviewer: str = Field(default="local reviewer", max_length=120)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class FaceAOIReviewApplyJobRequest(BaseModel):
    session_id: str
    artifact_id: str
    label: str | None = Field(default=None, max_length=240)
    require_complete: bool = True


class BodyAOIReviewDecisionRequest(BaseModel):
    video_artifact_id: str | None = None
    output_frame_index: int = Field(ge=0)
    gaze_sample_uid: str | None = None
    apply_to: Literal["sample", "frame", "fixation"] = "sample"
    action: Literal["accept_auto", "override", "exclude", "defer", "clear"]
    override_relation: Literal["partner", "self", "other"] | None = None
    override_entity_id: str | None = Field(default=None, max_length=128)
    override_region: Literal[
        "head", "neck", "torso", "pelvis",
        "left_arm", "left_upper_arm", "left_forearm", "left_hand",
        "right_arm", "right_upper_arm", "right_forearm", "right_hand",
        "left_leg", "left_thigh", "left_lower_leg", "left_foot",
        "right_leg", "right_thigh", "right_lower_leg", "right_foot",
        "no_body",
    ] | None = None
    note: str = Field(default="", max_length=1000)
    reviewer: str = Field(default="local reviewer", max_length=120)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class BodyAOIReviewApplyJobRequest(BaseModel):
    session_id: str
    artifact_id: str
    label: str | None = Field(default=None, max_length=240)
    require_complete: bool = True


class MediaToolsSettingsRequest(BaseModel):
    ffmpeg: str | None = None
    ffprobe: str | None = None


class MediaToolsInstallRequest(BaseModel):
    session_id: str


class MediaToolsCheckRequest(BaseModel):
    mode: Literal["encoder", "probe", "all"] = "encoder"
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    preserve_audio: bool = True


class CalibrationJobRequest(BaseModel):
    session_id: str
    native_asset_id: str
    tracker_asset_ids: list[str]
    board_asset_id: str
    board_mode: Literal["explicit_definition", "auto_legacy_16_marker"] = "explicit_definition"
    board_definition_path: str | None = None
    world_asset_id: str | None = None
    intrinsics_path: str | None = None
    participant_id: str = "P01"
    profile_name: str = "Calibration"
    tracker_marker_labels: dict[str, list[str]] = Field(default_factory=dict)
    sample_every_n_frames: int = 5
    max_detection_frames: int = 400
    max_pose_gap_sec: float = 0.1
    minimum_identity_margin_mm: float = 10.0
    minimum_tracker_identity_confidence: Literal["LOW", "SINGLE_CANDIDATE", "MEDIUM", "HIGH"] = "MEDIUM"
    include_raw_identity_metadata: bool = False
    hardware_id: str | None = None
    mount_version: str | None = None
    valid_sessions: list[str] = Field(default_factory=list)
    auto_register_hardware: bool = True
    auto_profile_name: bool = True
    allow_hardware_id_override: bool = False
    output_g3dcal: bool = True
    calibration_set_id: str | None = None
    calibration_revision_id: str | None = None


class CalibrationProfileImportRequest(BaseModel):
    session_id: str
    profile_path: str
    label: str | None = None
    hardware_id: str | None = None
    mount_version: str | None = None
    valid_sessions: list[str] = Field(default_factory=list)
    source_note: str | None = None
    calibration_set_id: str | None = None
    calibration_revision_id: str | None = None




class CalibrationMeasuredRegistrationRequest(BaseModel):
    session_id: str
    profile_artifact_id: str
    mount_version: str | None = None
    valid_sessions: list[str] = Field(default_factory=list)
    label: str | None = None


class CalibrationTransferRequest(BaseModel):
    session_id: str
    tracker_asset_id: str
    profile_artifact_id: str | None = None
    profile_path: str | None = None
    interaction_id: str | None = None
    role_id: str | None = None
    label: str | None = None
    maximum_shape_rms_mm: float = Field(default=5.0, gt=0)
    high_confidence_rms_mm: float = Field(default=2.0, gt=0)
    minimum_marker_valid_fraction: float = Field(default=0.5, ge=0.0, le=1.0)


class ETMRoleAssignmentJobRequest(BaseModel):
    session_id: str
    body_asset_ids: list[str] = Field(min_length=2, max_length=8)
    etm_asset_ids: list[str] = Field(min_length=2, max_length=8)
    role_labels: dict[str, str] = Field(default_factory=dict)
    body_marker_prefixes: dict[str, str] = Field(default_factory=dict)
    etm_marker_prefixes: dict[str, str] = Field(default_factory=dict)
    calibration_profile_artifact_ids: list[str] = Field(default_factory=list)
    maximum_head_distance_mm: float = Field(default=350.0, gt=0)
    minimum_role_margin_mm: float = Field(default=100.0, ge=0)
    minimum_role_ratio: float = Field(default=1.5, ge=1.0)
    minimum_global_margin_mm: float = Field(default=100.0, ge=0)
    maximum_profile_shape_rms_mm: float = Field(default=8.0, gt=0)
    minimum_profile_shape_margin_mm: float = Field(default=4.0, ge=0)
    label: str = "ETM role assignment"


class CalibrationProfileBindingJobRequest(BaseModel):
    session_id: str
    assignment_artifact_id: str
    calibration_profile_artifact_ids: list[str] = Field(min_length=2, max_length=16)
    time_sync_artifact_ids: list[str] = Field(default_factory=list, max_length=16)
    require_time_sync: bool = False
    require_distributed_drift: bool = True
    maximum_profile_shape_rms_mm: float = Field(default=8.0, gt=0)
    minimum_profile_shape_margin_mm: float = Field(default=4.0, ge=0)
    label: str = "Calibration handoff"


class Gaze3DJobRequest(BaseModel):
    session_id: str
    timeseries_asset_id: str
    tracker_asset_id: str | None = None
    profile_path: str | None = None
    profile_artifact_id: str | None = None
    etm_assignment_artifact_id: str | None = None
    wearer_role_id: str | None = None
    calibration_acceptance_artifact_id: str | None = None
    require_calibration_acceptance: bool = False
    clip_id: str | None = None
    native_trim_artifact_id: str | None = None
    scope: Literal["full", "clip", "native_trim"] = "full"
    alignment_mode: Literal["timegraph", "pretrimmed_shared_start"] = "timegraph"
    shared_start_offset_sec: float = Field(default=0.0, ge=-5.0, le=5.0)
    max_pose_gap_sec: float = 0.1
    output_parquet: bool = False


class SemanticTargetsJobRequest(BaseModel):
    session_id: str
    gaze_rays_artifact_id: str
    timeseries_asset_id: str | None = None
    facemapper_asset_id: str | None = None
    body_asset_ids: list[str] = Field(default_factory=list)
    body_profile_path: str | None = None
    entity_marker_prefixes: dict[str, str] = Field(default_factory=dict)
    densepose_asset_id: str | None = None
    densepose_profile_path: str | None = None
    scene_frame_tolerance_ms: float = 10.0
    face_tolerance_ms: float = 8.0
    face_left_right_policy: Literal["anatomical_auto", "as_exported", "swap_eyes"] = "anatomical_auto"
    max_marker_gap_sec: float = 0.1
    uncertainty_cone_deg: float = 1.5
    maximum_target_distance_m: float = 10.0
    region_frame_policy: Literal["all_scene_frames", "gaze_frames_only"] = "all_scene_frames"
    output_parquet: bool = False


class QCJobRequest(BaseModel):
    session_id: str


class ResultsPackageJobRequest(BaseModel):
    session_id: str
    artifact_ids: list[str] = Field(min_length=1)
    session_code: str | None = None
    interaction_code: str | None = None
    role_code: str | None = None
    label: str = "OculoStudio results"
    copy_mode: Literal["copy", "index_only"] = "copy"


class ArtifactCleanupRequest(BaseModel):
    session_id: str
    artifact_ids: list[str] = Field(default_factory=list, max_length=500)
    orphan_paths: list[str] = Field(default_factory=list, max_length=500)


class TrashPurgeRequest(BaseModel):
    confirmation: str


class DialogRequest(BaseModel):
    kind: Literal["file", "directory", "save"] = "file"
    title: str = "Select a path"
    filetypes: list[list[str]] = Field(default_factory=list)
    multiple: bool = False


class ImportPathsRequest(BaseModel):
    project_id: str
    session_id: str
    paths: list[str] = Field(min_length=1, max_length=200)
    label: str | None = None
    participant_id: str | None = None
    checksum: bool = True


class UploadStartRequest(BaseModel):
    project_id: str
    session_id: str
    filename: str
    size: int = Field(gt=0)


class UploadFinishRequest(BaseModel):
    project_id: str
    label: str | None = None
    participant_id: str | None = None


class CalibrationUploadStartRequest(BaseModel):
    project_id: str
    session_id: str
    calibration_set_id: str
    filename: str
    size: int = Field(gt=0)


class CalibrationUploadFinishRequest(BaseModel):
    project_id: str
    calibration_set_id: str
    label: str | None = None
    participant_id: str | None = None


class CalibrationAssetImportRequest(BaseModel):
    project_id: str
    session_id: str
    paths: list[str] = Field(min_length=1, max_length=200)
    calibration_set_id: str = "CALIBRATION"
    checksum: bool = True
    recursive: bool = False


class CalibrationAssetAdoptRequest(BaseModel):
    calibration_set_id: str = "CALIBRATION"
    calibration_role: str | None = None


class OpenPathRequest(BaseModel):
    path: str


class StudioRuntime:
    def __init__(self):
        self._store: ProjectStore | None = None
        self._lock = threading.RLock()
        self.jobs = JobManager(self.require_store)
        self.uploads = UploadManager()

    def require_store(self) -> ProjectStore:
        with self._lock:
            if self._store is None:
                raise ValidationError("No OculoStudio project is open")
            return self._store

    def open(self, path: str | Path) -> ProjectStore:
        self.ensure_import_idle()
        store = ProjectStore.open(path)
        with self._lock:
            self._store = store
        return store

    def create(
        self,
        root: str | Path,
        name: str,
        session_label: str,
        *,
        session_labels: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        planned_interactions: list[str] | None = None,
    ) -> ProjectStore:
        self.ensure_import_idle()
        store = ProjectStore.create(
            root,
            name,
            session_label=session_label,
            session_labels=session_labels,
            settings=settings,
            planned_interactions=planned_interactions,
        )
        with self._lock:
            self._store = store
        return store

    def ensure_import_idle(self):
        with self._lock:
            busy = self.uploads.has_pending()
            if self._store:
                busy = busy or any(j.operation in {'asset_import','calibration_batch_import','calibration_asset_import'} and j.status in {'queued','running'}
                                   for j in self._store.snapshot().jobs.values())
            if busy:
                raise ValidationError('An import is active. Finish or cancel it before switching projects.')

    def close(self) -> None:
        self.ensure_import_idle()
        with self._lock:
            self._store = None

    def state(self) -> dict[str, Any]:
        with self._lock:
            if self._store is None:
                return {"project": None}
            project = self._store.snapshot()
        return {"project": project.model_dump(mode="json")}


runtime = StudioRuntime()
app = FastAPI(title="OculoStudio CleanCore", version=__version__)


@app.middleware("http")
async def disable_frontend_asset_cache(request: Request, call_next):
    """Keep the desktop WebView from mixing frontend files across upgrades."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.middleware("http")
async def protect_import_mutations(request: Request, call_next):
    if request.method not in {'GET','HEAD','OPTIONS'} and request.url.path.startswith(('/api/imports','/api/calibration-assets','/api/dialog','/api/media-tools')):
        origin = request.headers.get('origin')
        cross_site = request.headers.get('sec-fetch-site') == 'cross-site'
        if cross_site or (origin and urlsplit(origin).netloc != request.headers.get('host')):
            return JSONResponse(status_code=403, content={'message':'Cross-origin import requests are not allowed'})
    return await call_next(request)


def import_store(project_id):
    store = runtime.require_store()
    if store.project.id != project_id:
        raise ValidationError('Project changed during import; return to the intended project and retry')
    return store


@app.post('/api/imports/paths')
def import_paths(request: ImportPathsRequest):
    store = import_store(request.project_id)
    if request.session_id not in store.snapshot().sessions:
        raise ValidationError('Unknown import session')
    jobs = []
    for path in request.paths:
        job = runtime.jobs.submit(operation='asset_import', label='Import ' + str(path).replace('\\','/').split('/')[-1],
            session_id=request.session_id,
            parameters={'path':path, 'storage_mode':'local_reference', 'participant_id':request.participant_id},
            function=import_local_asset,
            kwargs={'store':store, 'session_id':request.session_id, 'path':path,
                    'label':request.label if len(request.paths)==1 else None,
                    'participant_id':request.participant_id, 'checksum':request.checksum})
        jobs.append(job.model_dump(mode='json'))
    return {'jobs':jobs}


@app.post('/api/calibration-assets/scan-paths')
def scan_calibration_paths(request: CalibrationAssetImportRequest):
    store = import_store(request.project_id)
    return scan_calibration_sources(
        store,
        session_id=request.session_id,
        paths=request.paths,
        calibration_set_id=request.calibration_set_id,
        recursive=request.recursive,
    )


@app.post('/api/calibration-assets/import-paths')
def import_calibration_paths(request: CalibrationAssetImportRequest):
    store = import_store(request.project_id)
    scan = scan_calibration_sources(
        store,
        session_id=request.session_id,
        paths=request.paths,
        calibration_set_id=request.calibration_set_id,
        recursive=request.recursive,
    )
    set_id = scan['calibration_set_id']
    explicit_paths = [row['path'] for row in scan['files']]
    job = runtime.jobs.submit(
        operation='calibration_batch_import',
        label=f"Import calibration set · {set_id} · {len(explicit_paths)} files",
        session_id=request.session_id,
        parameters={
            'paths': explicit_paths,
            'storage_mode': 'local_reference',
            'asset_scope': 'calibration',
            'calibration_set_id': set_id,
            'file_count': len(explicit_paths),
        },
        function=import_calibration_batch,
        kwargs={
            'store': store,
            'session_id': request.session_id,
            'paths': explicit_paths,
            'calibration_set_id': set_id,
            'checksum': request.checksum,
            'recursive': False,
        },
    )
    return {
        'job': job.model_dump(mode='json'),
        'jobs': [job.model_dump(mode='json')],
        'calibration_set_id': set_id,
        'file_count': len(explicit_paths),
        'new_count': scan['new_count'],
        'existing_count': scan['existing_count'],
        'files': scan['files'],
    }


@app.post('/api/calibration-assets/{session_id}/{asset_id}/adopt')
def adopt_calibration_asset(session_id: str, asset_id: str, request: CalibrationAssetAdoptRequest):
    asset = set_asset_calibration_scope(
        runtime.require_store(), session_id=session_id, asset_id=asset_id,
        calibration_set_id=request.calibration_set_id, calibration_role=request.calibration_role,
    )
    return {'asset': asset}


@app.post('/api/calibration-assets/uploads')
def start_calibration_upload(request: CalibrationUploadStartRequest):
    store = import_store(request.project_id)
    rows = list_calibration_sets(store)
    target = next((row for row in rows if row.get('id') == request.calibration_set_id or str(row.get('code','')).casefold() == str(request.calibration_set_id).casefold()), None)
    if target is None:
        raise ValidationError(f"Unknown Calibration Set: {request.calibration_set_id}")
    owner_session_id = str(target.get('owner_session_id') or request.session_id)
    root = store.calibration_set_dir(str(target['id'])) / 'sources'
    return runtime.uploads.start(
        store,
        session_id=owner_session_id,
        filename=request.filename,
        size=request.size,
        root=root,
        scope='calibration',
        calibration_set_id=str(target.get('code') or request.calibration_set_id),
    )


@app.post('/api/calibration-assets/uploads/{upload_id}/finish')
def finish_calibration_upload(upload_id: str, request: CalibrationUploadFinishRequest):
    store = import_store(request.project_id)
    item = runtime.uploads.seal(store, upload_id)
    job = runtime.jobs.submit(
        operation='calibration_asset_import',
        label='Import calibration source · ' + item.name,
        session_id=item.session_id,
        parameters={
            'upload_id': upload_id,
            'filename': item.name,
            'storage_mode': 'project_copy',
            'asset_scope': 'calibration',
            'calibration_set_id': request.calibration_set_id,
            'participant_id': request.participant_id,
        },
        function=runtime.uploads.finish_calibration_job,
        kwargs={
            'item': item,
            'calibration_set_id': request.calibration_set_id,
            'label': request.label,
            'participant_id': request.participant_id,
        },
    )
    return {'job': job.model_dump(mode='json')}


@app.post('/api/imports/uploads')
def start_upload(request: UploadStartRequest):
    store = import_store(request.project_id)
    return runtime.uploads.start(store,session_id=request.session_id,filename=request.filename,size=request.size)


@app.put('/api/imports/uploads/{upload_id}')
async def put_upload(upload_id: str, request: Request, offset: int = Query(ge=0)):
    store = runtime.require_store()
    item = runtime.uploads.get(store,upload_id)
    # A File is uploaded in bounded pieces, never converted to base64/JSON.
    length = request.headers.get('content-length')
    if length and int(length)>CHUNK_BYTES:
        raise ValidationError('Upload chunk exceeds 8 MiB')
    body = bytearray()
    async for data in request.stream():
        if len(body)+len(data)>CHUNK_BYTES:
            raise ValidationError('Upload chunk exceeds 8 MiB')
        body.extend(data)
    return await run_in_threadpool(runtime.uploads.append, store, upload_id, offset, body)


@app.post('/api/imports/uploads/{upload_id}/finish')
def finish_upload(upload_id: str, request: UploadFinishRequest):
    store = import_store(request.project_id)
    item = runtime.uploads.seal(store,upload_id)
    job = runtime.jobs.submit(operation='asset_import', label='Inspect ' + item.name,
        session_id=item.session_id,
        parameters={'upload_id':upload_id,'filename':item.name,'storage_mode':'project_copy','participant_id':request.participant_id},
        function=runtime.uploads.finish_job,
        kwargs={'item':item,'label':request.label,'participant_id':request.participant_id})
    return {'job':job.model_dump(mode='json')}


@app.delete('/api/imports/uploads/{upload_id}')
def abandon_upload(upload_id: str):
    runtime.uploads.abandon(runtime.require_store(),upload_id)
    return {'cancelled':True}


@app.get('/api/imports/jobs/{job_id}')
def import_job_state(job_id: str):
    job = runtime.require_store().snapshot().jobs.get(job_id)
    if job is None:
        raise HTTPException(404,'Import job not found')
    return {'job':job.model_dump(mode='json')}



@app.exception_handler(OculoStudioError)
async def handle_oculo_error(_request: Request, exc: OculoStudioError):
    return JSONResponse(status_code=400, content={"error": exc.__class__.__name__, "message": str(exc)})


@app.exception_handler(KeyError)
async def handle_key_error(_request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"error": "NotFound", "message": f"Unknown ID: {exc}"})


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "OculoStudio", "version": __version__}


@app.get("/api/runtime")
def runtime_info():
    media = media_tools_status()
    optional = {}
    for module in ("ezc3d", "av", "webview"):
        try:
            __import__(module)
            optional[module] = True
        except Exception:
            optional[module] = False
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": media["tools"]["ffmpeg"].get("path"),
        "ffprobe": media["tools"]["ffprobe"].get("path"),
        "media_tools": media,
        "optional": optional,
        "builtin_c3d_reader": True,
        "long_c3d_recovery": True,
        "clean_core": True,
        "legacy_engines_loaded": False,
    }


@app.get("/api/media-tools")
def get_media_tools():
    return media_tools_status()


@app.put("/api/media-tools/settings")
def configure_media_tools(request: MediaToolsSettingsRequest):
    save_tool_settings(request.model_dump(exclude_unset=True))
    return media_tools_status()


@app.post("/api/media-tools/check")
def check_media_tools(request: MediaToolsCheckRequest):
    try:
        if request.mode == "probe":
            preflight = probe_preflight(request.ffprobe_path, ffmpeg_path=request.ffmpeg_path)
        elif request.mode == "all":
            preflight = complete_preflight(request.ffmpeg_path, request.ffprobe_path,
                                          preserve_audio=request.preserve_audio)
        else:
            preflight = privacy_encoder_preflight(request.ffmpeg_path,
                ffprobe_path=request.ffprobe_path, preserve_audio=request.preserve_audio)
        return {"ready": True, "mode": request.mode, "preflight": preflight}
    except OculoStudioError as exc:
        return {"ready": False, "error": str(exc)}


@app.post("/api/media-tools/install")
def install_media_tools(request: MediaToolsInstallRequest):
    # Explicit action, protected by the same-origin middleware. No automatic HTTP download on startup.
    return _submit("media_tools_install", "Install/check FFmpeg + ffprobe", request.session_id,
                   {"download_if_missing": True}, install_runtime_job, request.model_dump())


@app.get("/api/state")
def get_state():
    state = runtime.state()
    if state["project"]:
        project = state["project"]
        sid = project.get("selected_session_id")
        if sid:
            try:
                state["qc"] = build_session_qc(runtime.require_store(), sid)
            except Exception as exc:
                state["qc"] = {"metrics": [], "warnings": [str(exc)]}
            try:
                state["interaction_workspace"] = build_interaction_workspace(runtime.require_store(), sid)
            except Exception as exc:
                state["interaction_workspace"] = {
                    "session_id": sid, "interactions": [], "unassigned": [],
                    "counts": {}, "warning": str(exc),
                }
            try:
                state["project_workspace"] = build_project_workspace(runtime.require_store())
            except Exception as exc:
                state["project_workspace"] = {"sessions": [], "counts": {}, "warning": str(exc)}
    return state


@app.post("/api/project/location-check")
def project_location_check(request: ProjectLocationCheckRequest):
    return inspect_project_location(parent=request.parent, folder=request.folder)


@app.post("/api/project/create")
def create_project(request: ProjectCreateRequest):
    labels = request.session_labels or [request.session_label]
    store = runtime.create(
        request.root,
        request.name,
        request.session_label,
        session_labels=labels,
        settings=request.settings,
        planned_interactions=request.planned_interactions,
    )
    if request.selected_session_label:
        wanted = request.selected_session_label.strip().upper()
        for session_id, session in store.snapshot().sessions.items():
            if session.label.strip().upper() == wanted:
                store.select_session(session_id)
                break
    return {"project": store.snapshot().model_dump(mode="json"), "project_workspace": build_project_workspace(store)}


@app.post("/api/project/open")
def open_project(request: ProjectOpenRequest):
    store = runtime.open(request.path)
    return {"project": store.snapshot().model_dump(mode="json")}


@app.post("/api/project/close")
def close_project():
    runtime.close()
    return {"ok": True}


@app.post("/api/session/add")
def add_session(request: SessionRequest):
    session = runtime.require_store().add_session(request.label)
    return {"session": session.model_dump(mode="json"), "project_workspace": build_project_workspace(runtime.require_store())}


@app.post("/api/sessions/batch")
def add_sessions_batch(request: SessionBatchRequest):
    store = runtime.require_store()
    rows = store.add_sessions(request.labels, select_first=request.select_first)
    return {"sessions": [row.model_dump(mode="json") for row in rows], "project_workspace": build_project_workspace(store)}


@app.post("/api/session/{session_id}/interactions/plan")
def plan_session_interactions(session_id: str, request: InteractionPlanRequest):
    store = runtime.require_store()
    rows = store.plan_interactions(session_id, request.interaction_ids, replace=request.replace)
    return {"interaction_ids": rows, "project_workspace": build_project_workspace(store)}


@app.get("/api/project/calibration-library")
def get_project_calibration_library():
    store = runtime.require_store()
    return {"profiles": project_calibration_library(store), "project_workspace": build_project_workspace(store)}


@app.post("/api/project/calibration-library/apply")
def apply_project_calibration_library(request: SharedCalibrationApplyRequest):
    store = runtime.require_store()
    result = apply_shared_calibration_profiles(
        store,
        source_session_id=request.source_session_id,
        profile_artifact_ids=request.profile_artifact_ids,
        target_session_ids=request.target_session_ids,
    )
    result["project_workspace"] = build_project_workspace(store)
    return result


@app.get("/api/project/calibration-sets")
def get_project_calibration_sets():
    store = runtime.require_store()
    return {"calibration_sets": list_calibration_sets(store), "project_workspace": build_project_workspace(store)}


@app.get("/api/project/calibration-sets/{calibration_set_id}/profile-build-status")
def get_calibration_set_profile_build_status(calibration_set_id: str, revision_id: str | None = None):
    store = runtime.require_store()
    return {
        "build_status": calibration_set_profile_build_status(
            store, calibration_set_id=calibration_set_id, revision_id=revision_id
        ),
        "project_workspace": build_project_workspace(store),
    }


@app.post("/api/jobs/calibration-set-profiles/{calibration_set_id}")
def job_calibration_set_profiles(calibration_set_id: str, request: CalibrationSetProfileBuildRequest):
    resolved = request.model_dump()
    session_id = resolved.pop("session_id")
    return _submit(
        "calibration_set_profile_build",
        f"Build physical ETM profiles · {calibration_set_id}",
        session_id,
        {"calibration_set_id": calibration_set_id, **request.model_dump()},
        build_calibration_set_profiles,
        {"session_id": session_id, "calibration_set_id": calibration_set_id, **resolved},
    )


@app.post("/api/project/calibration-sets")
def create_project_calibration_set(request: CalibrationSetCreateRequest):
    store = runtime.require_store()
    result = create_calibration_set(store, **request.model_dump())
    return {"calibration_set": result, "project_workspace": build_project_workspace(store)}


@app.post("/api/project/calibration-sets/{calibration_set_id}/sessions")
def bind_project_calibration_set_sessions(calibration_set_id: str, request: CalibrationSetSessionsRequest):
    store = runtime.require_store()
    result = set_calibration_sessions(
        store, calibration_set_id=calibration_set_id,
        session_ids=request.session_ids, revision_id=request.revision_id,
    )
    result["project_workspace"] = build_project_workspace(store)
    return result


@app.post("/api/project/calibration-sets/{calibration_set_id}/revisions")
def create_project_calibration_set_revision(calibration_set_id: str, request: CalibrationSetRevisionRequest):
    store = runtime.require_store()
    result = create_calibration_revision(store, calibration_set_id=calibration_set_id, **request.model_dump())
    return {"calibration_set": result, "project_workspace": build_project_workspace(store)}


@app.post("/api/project/calibration-sets/{calibration_set_id}/revisions/activate")
def activate_project_calibration_set_revision(calibration_set_id: str, request: CalibrationSetActivateRequest):
    store = runtime.require_store()
    result = activate_calibration_revision(
        store, calibration_set_id=calibration_set_id, revision_id=request.revision_id
    )
    result["project_workspace"] = build_project_workspace(store)
    return result


@app.post("/api/project/calibration-sets/{calibration_set_id}/profiles")
def attach_project_calibration_set_profile(calibration_set_id: str, request: CalibrationSetProfileRequest):
    store = runtime.require_store()
    result = register_calibration_profile_ref(
        store, calibration_set_id=calibration_set_id,
        source_session_id=request.source_session_id,
        profile_artifact_id=request.profile_artifact_id,
        revision_id=request.revision_id,
    )
    result["project_workspace"] = build_project_workspace(store)
    return result


@app.post("/api/project/calibration-sets/{calibration_set_id}/materialize")
def materialize_project_calibration_set(calibration_set_id: str, request: CalibrationSetMaterializeRequest):
    store = runtime.require_store()
    result = materialize_calibration_set(
        store, calibration_set_id=calibration_set_id,
        session_ids=request.session_ids, revision_id=request.revision_id,
    )
    result["project_workspace"] = build_project_workspace(store)
    return result


@app.post("/api/session/{session_id}/select")
def select_session(session_id: str):
    runtime.require_store().select_session(session_id)
    return {"ok": True}


@app.get("/api/interactions/{session_id}")
def interaction_workspace(session_id: str):
    return build_interaction_workspace(runtime.require_store(), session_id)


@app.post("/api/interactions/{session_id}/context")
def interaction_context(session_id: str, request: InteractionContextRequest):
    return set_interaction_context(
        runtime.require_store(),
        session_id,
        interaction_id=request.interaction_id,
        role_id=request.role_id,
        workflow=request.workflow,
    )


@app.post("/api/assets/add")
def api_add_asset(request: AssetAddRequest):
    asset = add_asset(
        runtime.require_store(),
        session_id=request.session_id,
        path=request.path,
        label=request.label,
        participant_id=request.participant_id,
        checksum=request.checksum,
    )
    return {"asset": asset.model_dump(mode="json")}


@app.patch("/api/assets/{session_id}/{asset_id}/identity")
def api_update_asset_identity(session_id: str, asset_id: str, request: AssetIdentityUpdateRequest):
    asset = update_asset_identity(
        runtime.require_store(),
        session_id=session_id,
        asset_id=asset_id,
        participant_id=request.participant_id,
        interaction_id=request.interaction_id,
        role_id=request.role_id,
        source="user_edit",
    )
    session = runtime.require_store().snapshot().sessions[session_id]
    return {"asset": asset.model_dump(mode="json"), "identity": asset_identity_view(asset, session.label)}


@app.post("/api/assets/{session_id}/identity/batch")
def api_update_asset_identity_batch(session_id: str, request: AssetIdentityBatchUpdateRequest):
    store = runtime.require_store()
    assets = []
    for asset_id in request.asset_ids:
        asset = update_asset_identity(
            store,
            session_id=session_id,
            asset_id=asset_id,
            participant_id=request.participant_id,
            interaction_id=request.interaction_id,
            role_id=request.role_id,
            source="user_batch_edit",
        )
        assets.append(asset.model_dump(mode="json"))
    session = store.snapshot().sessions[session_id]
    return {"assets": assets, "participant_ids": session.participant_ids}


@app.delete("/api/assets/{session_id}/{asset_id}")
def api_remove_asset(session_id: str, asset_id: str):
    remove_asset(runtime.require_store(), session_id=session_id, asset_id=asset_id)
    return {"ok": True}


@app.delete("/api/artifacts/{session_id}/{artifact_id}")
def api_remove_artifact(session_id: str, artifact_id: str):
    result = move_artifact_to_trash(
        runtime.require_store(), session_id=session_id, artifact_id=artifact_id
    )
    return {"ok": True, "removed": result}




@app.get("/api/artifacts/{session_id}/cleanup-plan")
def api_artifact_cleanup_plan(session_id: str, scan_orphans: bool = False):
    return build_artifact_cleanup_plan(
        runtime.require_store(), session_id=session_id, scan_orphans=scan_orphans
    )


@app.post("/api/artifacts/cleanup")
def api_artifact_cleanup(request: ArtifactCleanupRequest):
    store = runtime.require_store()
    removed = move_artifacts_to_trash(
        store, session_id=request.session_id, artifact_ids=request.artifact_ids
    )
    orphaned = move_orphan_outputs_to_trash(
        store, session_id=request.session_id, paths=request.orphan_paths
    )
    return {"ok": True, "removed_artifacts": removed, "removed_orphans": orphaned}


@app.post("/api/trash/purge")
def api_purge_project_trash(request: TrashPurgeRequest):
    return {"ok": True, "purged": purge_project_trash(runtime.require_store(), confirmation=request.confirmation)}

@app.post("/api/assets/{session_id}/{asset_id}/refresh")
def api_refresh_asset(session_id: str, asset_id: str, checksum: bool = False):
    asset = refresh_asset(runtime.require_store(), session_id=session_id, asset_id=asset_id, checksum=checksum)
    return {"asset": asset.model_dump(mode="json")}


@app.post("/api/time-maps/fit")
def api_fit_time_map(request: TimeMapRequest):
    mapping = fit_and_store_time_map(
        runtime.require_store(),
        session_id=request.session_id,
        source_domain_id=request.source_domain_id,
        target_domain_id=request.target_domain_id,
        anchors=request.anchors,
        method=request.method,
    )
    return {"time_map": mapping.model_dump(mode="json")}


@app.post("/api/time-maps/auto-fit")
async def api_auto_fit_time_map(request: AutomaticTimeMapRequest):
    store = runtime.require_store()
    return await run_in_threadpool(
        automatic_pupil_vicon_sync,
        store,
        session_id=request.session_id,
        timeseries_asset_id=request.timeseries_asset_id,
        tracker_asset_id=request.tracker_asset_id,
        etm_assignment_artifact_id=request.etm_assignment_artifact_id,
        wearer_role_id=request.wearer_role_id,
        maximum_offset_sec=request.maximum_offset_sec,
        sample_rate_hz=request.sample_rate_hz,
        save_if_high=request.save_if_high,
    )


@app.post("/api/time-maps/auto-accept")
async def api_accept_auto_time_map(request: AcceptAutomaticTimeMapRequest):
    return await run_in_threadpool(
        accept_automatic_time_sync,
        runtime.require_store(),
        session_id=request.session_id,
        candidate_id=request.candidate_id,
        accepted_by_user=True,
    )


@app.get("/api/time-maps/auto-preview/{session_id}")
async def api_automatic_time_sync_preview(
    session_id: str,
    candidate_id: str | None = Query(default=None),
    artifact_id: str | None = Query(default=None),
    maximum_samples: int = Query(default=1800, ge=100, le=10000),
):
    data = await run_in_threadpool(
        get_automatic_time_sync_preview,
        runtime.require_store(),
        session_id=session_id,
        candidate_id=candidate_id,
        artifact_id=artifact_id,
        maximum_samples=maximum_samples,
    )
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.post("/api/time-maps/audio-video/accept")
async def api_accept_audio_video_time_map(request: AcceptAudioVideoSyncRequest):
    return await run_in_threadpool(
        accept_audio_video_sync,
        runtime.require_store(),
        session_id=request.session_id,
        candidate_id=request.candidate_id,
        visual_offset_adjustment_ms=request.visual_offset_adjustment_ms,
        visual_review_confirmed=request.visual_review_confirmed,
        ffprobe_path=request.ffprobe_path,
    )


@app.get("/api/time-maps/audio-video/preview/{session_id}")
async def api_audio_video_time_sync_preview(
    session_id: str,
    candidate_id: str | None = Query(default=None),
    artifact_id: str | None = Query(default=None),
    maximum_samples: int = Query(default=2200, ge=100, le=10000),
):
    data = await run_in_threadpool(
        get_audio_video_sync_preview,
        runtime.require_store(),
        session_id=session_id,
        candidate_id=candidate_id,
        artifact_id=artifact_id,
        maximum_samples=maximum_samples,
    )
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.delete("/api/time-maps/{session_id}/{mapping_id}")
def api_delete_time_map(session_id: str, mapping_id: str):
    store = runtime.require_store()

    def mutate(project):
        project.sessions[session_id].time_maps.pop(mapping_id, None)

    store.update(mutate)
    return {"ok": True}


@app.post("/api/clips/add")
def api_add_clip(request: ClipRequest):
    start = request.start_global_ns
    end = request.end_global_ns
    if start is None:
        if request.start_timecode is None:
            raise ValidationError("Clip start is required")
        start = parse_timecode_ns(request.start_timecode, frame_rate=request.frame_rate)
    if end is None:
        if request.end_timecode is None:
            raise ValidationError("Clip end is required")
        end = parse_timecode_ns(request.end_timecode, frame_rate=request.frame_rate)
    clip = Clip(
        label=request.label,
        start_global_ns=start,
        end_global_ns=end,
        notes=request.notes,
        source_display_domain_id=request.source_display_domain_id,
    )
    store = runtime.require_store()

    def mutate(project):
        project.sessions[request.session_id].clips[clip.id] = clip

    store.update(mutate)
    return {"clip": clip.model_dump(mode="json")}


@app.delete("/api/clips/{session_id}/{clip_id}")
def api_delete_clip(session_id: str, clip_id: str):
    store = runtime.require_store()

    def mutate(project):
        project.sessions[session_id].clips.pop(clip_id, None)

    store.update(mutate)
    return {"ok": True}


def _submit(operation: str, label: str, session_id: str, parameters: dict, function, kwargs):
    force_recompute = bool(parameters.get("force_recompute", False))
    job, deduplication = runtime.jobs.submit_idempotent(
        operation=operation,
        label=label,
        session_id=session_id,
        parameters=parameters,
        function=function,
        kwargs={"store": runtime.require_store(), **kwargs},
        force_recompute=force_recompute,
    )
    return {"job": job.model_dump(mode="json"), "deduplication": deduplication}


@app.post("/api/jobs/frame-map")
def job_frame_map(request: FrameMapRequest):
    return _submit(
        "frame_map",
        "Build Native–Cloud frame map",
        request.session_id,
        request.model_dump(),
        build_native_cloud_frame_map,
        request.model_dump(),
    )


@app.post("/api/jobs/audio-video-sync")
def job_audio_video_sync(request: AudioVideoSyncJobRequest):
    resolved = request.model_dump()
    return _submit(
        "shared_audio_video_sync",
        "Match facial and ego video by shared audio",
        request.session_id,
        {**resolved, "engine": "shared_audio_video_affine_v1_distributed_holdout"},
        automatic_shared_audio_video_sync,
        resolved,
    )


@app.post("/api/jobs/clip")
def job_clip(request: ClipJobRequest):
    return _submit(
        "multimodal_clip",
        "Export multimodal clip",
        request.session_id,
        request.model_dump(),
        run_multimodal_clip,
        request.model_dump(),
    )


@app.get("/api/native-reference/options/{session_id}/{asset_id}")
def native_reference_options(session_id: str, asset_id: str):
    asset = runtime.require_store().snapshot().sessions[session_id].assets[asset_id]
    if asset.kind != "pupil_native":
        raise ValidationError("Select a Native Recording asset")
    return {"options": native_options(asset.source_path)}


@app.post("/api/native-reference/parse-time")
def native_time_input(request: NativeTimeInputRequest):
    # Lightweight and source-independent: safe to call while typing, before
    # creating a project or extracting a large Native ZIP.
    return parse_native_trim_times(**request.model_dump())


@app.post("/api/jobs/native-trim-preview")
def native_trim_preview_job(request: NativeTrimPreviewRequest):
    # Validate before queueing, so bad FF values are visible immediately and
    # do not start an expensive Native indexing job.
    parse_native_trim_times(request.start_timecode, request.duration_timecode,
                            timecode_fps=request.timecode_fps)
    return _submit("native_trim_preview", "Check NRD start + duration", request.session_id,
                   request.model_dump(), preview_native_reference_trim, request.model_dump())




@app.post("/api/jobs/aligned-clip-export")
def aligned_clip_export_job(request: AlignedClipExportJobRequest):
    resolved = request.model_dump()
    return _submit(
        "aligned_clip_export",
        "Export aligned ego + TSD + FaceMapper",
        request.session_id,
        {**resolved, "engine": "native_pts_aligned_ego_tsd_fm_v1"},
        export_aligned_clip_package,
        resolved,
    )


@app.post("/api/jobs/aligned-facial-video-export")
def aligned_facial_video_export_job(request: AlignedFacialVideoExportJobRequest):
    resolved = request.model_dump()
    return _submit(
        "aligned_facial_video_export",
        "Export aligned facial-expression video",
        request.session_id,
        {**resolved, "engine": "accepted_audio_timegraph_reference_clock_resample_v1"},
        export_aligned_facial_video,
        resolved,
    )


@app.post("/api/jobs/native-trim-overlay-preview")
def native_trim_overlay_preview_job(request: NativeTrimOverlayPreviewRequest):
    return _submit(
        "native_trim_overlay_preview",
        "Render pre-trim FM + AOI overlay",
        request.session_id,
        request.model_dump(),
        render_pretrim_overlay_preview,
        request.model_dump(),
    )

@app.post("/api/jobs/aligned-clip-qc-preview")
def aligned_clip_qc_preview_job(request: NativeTrimOverlayPreviewRequest):
    return _submit(
        "aligned_clip_qc_preview",
        "Render aligned clip QC preview",
        request.session_id,
        request.model_dump(),
        render_pretrim_overlay_preview,
        request.model_dump(),
    )


@app.post("/api/jobs/native-trim")
def native_trim_export_job(request: NativeTrimExportRequest):
    return _submit("native_reference_trim", "Trim TSD + Face Mapper", request.session_id,
                   request.model_dump(), export_native_reference_trim, request.model_dump())


@app.post("/api/jobs/c3d-defined-aligned-clips")
def c3d_defined_aligned_clips_job(request: C3DDefinedAlignedClipsJobRequest):
    resolved = request.model_dump()
    return _submit(
        "c3d_defined_aligned_clips",
        "Create C3D-defined R01 + R02 clips",
        request.session_id,
        resolved,
        create_c3d_defined_aligned_clips,
        resolved,
    )



@app.post("/api/jobs/facemapper-filter")
def job_facemapper_filter(request: FaceMapperFilterJobRequest):
    resolved = request.model_dump()
    signature = {
        **resolved,
        "facemapper_filter_engine": "canonical-bilateral-semantics-live-canvas-v4",
    }
    return _submit(
        "facemapper_filter",
        "QC and filter FaceMapper candidates",
        request.session_id,
        signature,
        create_filtered_facemapper,
        resolved,
    )


@app.post("/api/jobs/privacy-mask-preview")
def job_privacy_mask_preview(request: PrivacyMaskPreviewJobRequest):
    encoder = privacy_encoder_preflight(
        request.ffmpeg_path,
        preserve_audio=False,
        ffprobe_path=request.ffprobe_path,
    )
    resolved = request.model_dump()
    resolved["ffmpeg_path"] = encoder["ffmpeg"]["path"]
    return _submit(
        "privacy_mask_preview",
        "Render accepted-only repair mask preview",
        request.session_id,
        {
            **resolved,
            "privacy_mask_preview_engine": "accepted-only-motion-repair-preview-v1",
            "encoder_preflight": encoder,
        },
        create_privacy_mask_preview,
        resolved,
    )

@app.post("/api/jobs/privacy-final")
def job_synchronized_privacy(request: SynchronizedPrivacyJobRequest):
    # The normal Privacy workflow publishes the final trim-matched, constant-FPS
    # artifact directly. The encoder is checked before the job is queued.
    encoder = privacy_encoder_preflight(
        request.ffmpeg_path,
        preserve_audio=request.preserve_audio,
        ffprobe_path=request.ffprobe_path,
    )
    resolved = request.model_dump()
    resolved["ffmpeg_path"] = encoder["ffmpeg"]["path"]
    signature_parameters = {
        **resolved,
        "synchronized_privacy_engine": SYNCHRONIZED_PRIVACY_ENGINE,
        "privacy_frame_coverage_engine": PRIVACY_FRAME_COVERAGE_ENGINE,
    }
    return _submit(
        "privacy_final",
        "Create synchronized Privacy video",
        request.session_id,
        {**signature_parameters, "encoder_preflight": encoder},
        create_synchronized_privacy_video,
        resolved,
    )


@app.post("/api/jobs/privacy")
def job_privacy(request: PrivacyJobRequest):
    # Fail as an actionable HTTP error before queueing expensive work.
    encoder = privacy_encoder_preflight(request.ffmpeg_path, preserve_audio=request.preserve_audio,
                                         ffprobe_path=request.ffprobe_path)
    resolved = request.model_dump()
    resolved["ffmpeg_path"] = encoder["ffmpeg"]["path"]
    signature_parameters = {
        **resolved,
        "privacy_frame_coverage_engine": PRIVACY_FRAME_COVERAGE_ENGINE,
    }
    return _submit(
        "privacy",
        "Create privacy video",
        request.session_id,
        {**signature_parameters, "encoder_preflight": encoder},
        create_privacy_video,
        resolved,
    )


@app.post("/api/jobs/privacy-clip")
def job_privacy_clip(request: PrivacyClipJobRequest):
    runtime.require_store()
    # Validate time input before queueing. If an existing trim artifact is
    # selected, its resolved media window is authoritative.
    if not request.native_trim_artifact_id:
        parse_native_trim_times(request.start_timecode or "", request.duration_timecode or "",
                                timecode_fps=request.timecode_fps)
    parameters = {
        **request.model_dump(),
        "privacy_clip_engine": PRIVACY_CLIP_ENGINE,
    }
    return _submit(
        "privacy_clip",
        "Trim blurred Privacy video",
        request.session_id,
        parameters,
        create_trimmed_privacy_video,
        request.model_dump(),
    )


@app.post("/api/jobs/participant-association")
def job_participant_association(request: ParticipantAssociationJobRequest):
    parameters = {**request.model_dump(), "participant_identity_engine": PARTICIPANT_ASSOCIATION_ENGINE}
    return _submit(
        "participant_association",
        "Project participant heads and assign faces",
        request.session_id,
        parameters,
        create_participant_association,
        request.model_dump(),
    )


@app.post("/api/jobs/face-aoi")
def job_face_aoi(request: FaceAOIJobRequest):
    parameters = {**request.model_dump(), "face_aoi_engine": FACE_AOI_ENGINE}
    return _submit(
        "face_aoi",
        "Annotate facial AOIs",
        request.session_id,
        parameters,
        create_face_aoi_annotations,
        request.model_dump(),
    )


@app.post("/api/jobs/body-aoi")
def job_body_aoi(request: BodyAOIJobRequest):
    _require_interaction_assets(
        request.session_id,
        [
            *(request.body_asset_ids or []),
            *([request.wearer_body_asset_id] if request.wearer_body_asset_id else []),
            *([request.partner_body_asset_id] if request.partner_body_asset_id else []),
            *([request.timeseries_asset_id] if request.timeseries_asset_id else []),
        ],
        purpose="Body AOI",
    )
    return _submit(
        "body_aoi",
        "Annotate body AOIs",
        request.session_id,
        request.model_dump(),
        create_body_aoi_annotations,
        request.model_dump(),
    )


@app.post("/api/jobs/body-aoi-review-reduction")
def job_body_aoi_review_reduction(request: BodyAOIReviewReductionJobRequest):
    return _submit(
        "body_aoi_review_reduction",
        "Reduce Body AOI review queue",
        request.session_id,
        request.model_dump(),
        reprocess_body_aoi_review_reduction,
        request.model_dump(),
    )


@app.post("/api/jobs/object-aoi")
def job_object_aoi(request: ObjectAOIJobRequest):
    _require_interaction_assets(
        request.session_id,
        [*(request.object_asset_ids or []), *([request.timeseries_asset_id] if request.timeseries_asset_id else [])],
        purpose="Object AOI",
    )
    return _submit(
        "object_aoi",
        "Annotate object AOIs",
        request.session_id,
        request.model_dump(),
        create_object_aoi_annotations,
        request.model_dump(),
    )


@app.post("/api/jobs/target-fusion")
def job_target_fusion(request: TargetFusionJobRequest):
    return _submit(
        "target_fusion",
        "Fuse Face, Body and Object AOIs",
        request.session_id,
        request.model_dump(),
        fuse_face_and_body_aoi,
        request.model_dump(),
    )


@app.post("/api/jobs/dyadic-attention")
def job_dyadic_attention(request: DyadicAttentionJobRequest):
    return _submit(
        "dyadic_attention",
        "Annotate mutual gaze and joint attention",
        request.session_id,
        request.model_dump(),
        create_dyadic_attention_annotations,
        request.model_dump(),
    )


@app.post("/api/jobs/face-aoi-manual-review")
def job_face_aoi_manual_review(request: FaceAOIReviewApplyJobRequest):
    return _submit(
        "face_aoi_manual_review",
        "Create reviewed Face AOI",
        request.session_id,
        request.model_dump(),
        apply_face_aoi_manual_review,
        request.model_dump(),
    )


@app.post("/api/jobs/body-aoi-manual-review")
def job_body_aoi_manual_review(request: BodyAOIReviewApplyJobRequest):
    return _submit(
        "body_aoi_manual_review",
        "Create reviewed Body AOI",
        request.session_id,
        request.model_dump(),
        apply_body_aoi_manual_review,
        request.model_dump(),
    )


@app.post("/api/jobs/aoi-review-video")
def job_aoi_review_video(request: AOIReviewVideoJobRequest):
    runtime.require_store()
    encoder = privacy_encoder_preflight(
        request.ffmpeg_path,
        preserve_audio=request.preserve_audio,
        ffprobe_path=request.ffprobe_path,
    )
    resolved = request.model_dump()
    resolved["ffmpeg_path"] = encoder["ffmpeg"]["path"]
    return _submit(
        "aoi_review_video",
        f"Render {request.mode} AOI review video",
        request.session_id,
        {**resolved, "encoder_preflight": encoder},
        render_aoi_review_video,
        resolved,
    )


def _require_interaction_assets(session_id: str, asset_ids: list[str], *, purpose: str) -> None:
    """Reject project-level calibration sources in interaction processing jobs."""
    ids = [str(value) for value in asset_ids if str(value)]
    if not ids:
        return
    snapshot = runtime.require_store().snapshot()
    if session_id not in snapshot.sessions:
        raise ValidationError(f"Unknown session: {session_id}")
    session = snapshot.sessions[session_id]
    missing = [asset_id for asset_id in ids if asset_id not in session.assets]
    if missing:
        raise ValidationError(f"{purpose} assets are unavailable: {missing}")
    wrong = [
        session.assets[asset_id].label
        for asset_id in ids
        if str((session.assets[asset_id].metadata or {}).get("asset_scope") or "interaction") == "calibration"
    ]
    if wrong:
        raise ValidationError(
            f"{purpose} only accepts interaction assets; Calibration Session Registry sources are isolated: "
            + ", ".join(wrong)
        )


@app.post("/api/jobs/etm-role-assignment")
def job_etm_role_assignment(request: ETMRoleAssignmentJobRequest):
    _require_interaction_assets(
        request.session_id,
        [*request.body_asset_ids, *request.etm_asset_ids],
        purpose="ETM wearer assignment",
    )
    return _submit(
        "etm_role_assignment",
        "Resolve ETM wearer / role",
        request.session_id,
        request.model_dump(),
        create_etm_role_assignment,
        request.model_dump(),
    )


@app.post("/api/jobs/calibration-profile-binding")
def job_calibration_profile_binding(request: CalibrationProfileBindingJobRequest):
    return _submit(
        "calibration_profile_binding",
        "Bind hardware profiles to resolved ETMs",
        request.session_id,
        request.model_dump(),
        bind_calibration_profiles_to_assignment,
        request.model_dump(),
    )


def _require_calibration_sources(session_id: str, asset_ids: list[str]) -> None:
    snapshot = runtime.require_store().snapshot()
    if session_id not in snapshot.sessions:
        raise ValidationError(f"Unknown session: {session_id}")
    session = snapshot.sessions[session_id]
    missing = [asset_id for asset_id in asset_ids if asset_id not in session.assets]
    if missing:
        raise ValidationError(f"Calibration assets are unavailable: {missing}")
    wrong = [session.assets[asset_id].label for asset_id in asset_ids
             if str((session.assets[asset_id].metadata or {}).get('asset_scope') or 'interaction') != 'calibration']
    if wrong:
        raise ValidationError('Raw calibration estimation only accepts Calibration Session Registry sources: ' + ', '.join(wrong))
    set_ids = {str((session.assets[asset_id].metadata or {}).get('calibration_set_id') or '') for asset_id in asset_ids}
    if len(set_ids) > 1:
        raise ValidationError('Selected calibration sources belong to different calibration sets')


@app.post("/api/jobs/calibration")
def job_calibration(request: CalibrationJobRequest):
    source_ids = [request.native_asset_id, request.board_asset_id, *request.tracker_asset_ids]
    if request.world_asset_id:
        source_ids.append(request.world_asset_id)
    _require_calibration_sources(request.session_id, source_ids)
    payload = request.model_dump()
    mode = payload.pop("board_mode")
    if mode == "auto_legacy_16_marker":
        payload.pop("board_definition_path", None)
        payload.pop("intrinsics_path", None)
        payload.pop("tracker_marker_labels", None)
        payload.pop("minimum_identity_margin_mm", None)
        return _submit(
            "calibration_legacy_workflow_b",
            "Create calibration profile · legacy auto 16-marker board",
            request.session_id,
            request.model_dump(),
            create_legacy_gaze3d_calibration_profile,
            payload,
        )
    if not request.board_definition_path:
        raise HTTPException(status_code=422, detail="Explicit board mode requires Board Definition JSON")
    payload.pop("world_asset_id", None)
    payload.pop("max_detection_frames", None)
    payload.pop("minimum_tracker_identity_confidence", None)
    payload.pop("include_raw_identity_metadata", None)
    return _submit(
        "calibration",
        "Create calibration profile · explicit board definition",
        request.session_id,
        request.model_dump(),
        create_calibration_profile,
        payload,
    )


@app.post("/api/jobs/calibration-profile-import")
def job_calibration_profile_import(request: CalibrationProfileImportRequest):
    return _submit(
        "calibration_profile_import",
        "Import hardware calibration profile",
        request.session_id,
        request.model_dump(),
        import_calibration_profile,
        request.model_dump(),
    )




@app.post("/api/jobs/calibration-profile-register-measured")
def job_calibration_profile_register_measured(request: CalibrationMeasuredRegistrationRequest):
    return _submit(
        "calibration_profile_register_measured",
        "Register measured calibration to ETM hardware",
        request.session_id,
        request.model_dump(),
        register_existing_measured_profile_hardware,
        request.model_dump(),
    )


@app.post("/api/jobs/calibration-transfer")
def job_calibration_transfer(request: CalibrationTransferRequest):
    return _submit(
        "calibration_transfer",
        "Validate transferred hardware calibration",
        request.session_id,
        request.model_dump(),
        validate_calibration_profile_transfer,
        request.model_dump(),
    )


@app.post("/api/jobs/calibration-acceptance")
def job_calibration_acceptance(request: CalibrationAcceptanceJobRequest):
    return _submit(
        "calibration_acceptance",
        "Run transferred-calibration acceptance gate",
        request.session_id,
        request.model_dump(),
        run_transferred_calibration_acceptance,
        request.model_dump(),
    )


def _preflight_gaze3d_time_path(request: Gaze3DJobRequest) -> dict[str, Any]:
    store = runtime.require_store()
    snapshot = store.snapshot()
    if request.session_id not in snapshot.sessions:
        raise ValidationError("Unknown session")
    session = snapshot.sessions[request.session_id]
    timeseries_asset = session.assets.get(request.timeseries_asset_id)
    if timeseries_asset is None:
        raise ValidationError("Selected Timeseries asset is unavailable")
    tracker_asset_id = request.tracker_asset_id
    if request.etm_assignment_artifact_id:
        assignment_artifact = session.artifacts.get(request.etm_assignment_artifact_id)
        if assignment_artifact is None:
            raise ValidationError("Selected ETM-role assignment artifact is unavailable")
        if not request.wearer_role_id:
            raise ValidationError("Select the wearer role when using an ETM-role assignment")
        assignment_row = resolve_assignment_for_role(assignment_artifact, request.wearer_role_id)
        tracker_asset_id = assignment_row.get("etm_asset_id")
    if not tracker_asset_id:
        raise ValidationError("Select interaction tracker motion, or use an ETM-role assignment artifact")
    tracker_asset = session.assets.get(tracker_asset_id)
    if tracker_asset is None:
        raise ValidationError("The tracker selected by the ETM assignment is unavailable")
    if request.scope == "native_trim":
        if not request.native_trim_artifact_id:
            raise ValidationError("Select a Native-reference trim")
        trim_artifact = session.artifacts.get(request.native_trim_artifact_id)
        if trim_artifact is None or trim_artifact.kind != ArtifactKind.NATIVE_REFERENCE_TRIM:
            raise ValidationError("Selected Native-reference trim is unavailable")
        trim_timeseries = trim_artifact.parameters.get("timeseries_asset_id")
        if trim_timeseries and str(trim_timeseries) != str(timeseries_asset.id):
            raise ValidationError("The selected Native-reference trim belongs to another Timeseries asset")
        if not trim_timeseries and timeseries_asset.id not in (trim_artifact.source_asset_ids or []):
            raise ValidationError("The selected Native-reference trim does not reference this Timeseries asset")
    elif request.native_trim_artifact_id:
        raise ValidationError("native_trim_artifact_id requires scope='native_trim'")
    if request.alignment_mode == "pretrimmed_shared_start":
        if request.scope != "native_trim":
            raise ValidationError("Pre-trimmed shared-start alignment requires a Native-reference trim")
        return {
            "connected": True,
            "timeseries_asset_id": timeseries_asset.id,
            "timeseries_label": timeseries_asset.label,
            "tracker_asset_id": tracker_asset.id,
            "tracker_label": tracker_asset.label,
            "source_domain_id": timeseries_asset.primary_time_domain_id,
            "target_domain_id": tracker_asset.primary_time_domain_id,
            "method": "pretrimmed_shared_start",
            "anchor_count": 0,
            "rms_residual_ms": 0.0,
            "native_trim_artifact_id": request.native_trim_artifact_id,
            "shared_start_offset_sec": request.shared_start_offset_sec,
            "warning": "Local zero is defined by the Native-reference trim and the first C3D frame; no global TimeGraph map is required.",
        }
    mapping = resolve_gaze3d_time_path(session, timeseries_asset, tracker_asset)
    return {
        "connected": True,
        "timeseries_asset_id": timeseries_asset.id,
        "timeseries_label": timeseries_asset.label,
        "tracker_asset_id": tracker_asset.id,
        "tracker_label": tracker_asset.label,
        "source_domain_id": timeseries_asset.primary_time_domain_id,
        "target_domain_id": tracker_asset.primary_time_domain_id,
        "method": mapping.method,
        "anchor_count": int(mapping.quality.anchor_count),
        "rms_residual_ms": float(mapping.quality.rms_residual_ns) / 1e6,
        "warning": mapping.quality.warning,
    }


@app.post("/api/gaze3d/time-preflight")
def gaze3d_time_preflight(request: Gaze3DJobRequest):
    return _preflight_gaze3d_time_path(request)


@app.post("/api/jobs/gaze3d")
def job_gaze3d(request: Gaze3DJobRequest):
    _require_interaction_assets(
        request.session_id,
        [request.timeseries_asset_id, *([request.tracker_asset_id] if request.tracker_asset_id else [])],
        purpose="Gaze 3D",
    )
    # Fail before a background job is queued.  This avoids a cryptic 14% failure
    # and preserves the user's form while directing them to Synchronize.
    _preflight_gaze3d_time_path(request)
    return _submit(
        "gaze3d",
        "Generate 3D gaze rays",
        request.session_id,
        request.model_dump(),
        generate_gaze_rays,
        request.model_dump(),
    )


@app.post("/api/jobs/semantic-targets")
def job_semantic_targets(request: SemanticTargetsJobRequest):
    _require_interaction_assets(
        request.session_id,
        [
            *(request.body_asset_ids or []),
            *([request.timeseries_asset_id] if request.timeseries_asset_id else []),
            *([request.facemapper_asset_id] if request.facemapper_asset_id else []),
            *([request.densepose_asset_id] if request.densepose_asset_id else []),
        ],
        purpose="Semantic targets",
    )
    return _submit(
        "semantic_targets",
        "Annotate semantic gaze targets",
        request.session_id,
        request.model_dump(),
        create_semantic_gaze_targets,
        request.model_dump(),
    )


@app.post("/api/jobs/qc")
def job_qc(request: QCJobRequest):
    def run_qc(*, store, session_id, progress, cancellation):
        progress(0.2, "Collecting QC metrics")
        cancellation.raise_if_cancelled()
        result = export_qc_report(store, session_id)
        progress(1.0, "QC report complete")
        return result

    return _submit("qc", "Export QC report", request.session_id, request.model_dump(), run_qc, request.model_dump())


@app.post("/api/jobs/results-package")
def job_results_package(request: ResultsPackageJobRequest):
    return _submit(
        "results_package",
        "Build organized results package",
        request.session_id,
        request.model_dump(),
        create_results_package,
        request.model_dump(),
    )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    store = runtime.require_store()
    job = store.snapshot().jobs.get(job_id)
    result = runtime.jobs.cancel(job_id)
    # A queued upload may never run; remove its partial staging file now.
    if result and job and job.parameters.get('upload_id'):
        try:
            runtime.uploads.abandon(store,job.parameters['upload_id'])
        except ValidationError:
            pass  # Running inspector cleans its own staging in finally.
    return {"cancelled": result}


@app.delete("/api/jobs/finished")
def clear_jobs():
    return {"removed": runtime.jobs.clear_finished()}


@app.get("/api/qc/{session_id}")
def api_qc(session_id: str):
    return build_session_qc(runtime.require_store(), session_id)


def _range_video_response(path: Path, request: Request):
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video file no longer exists")
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    header = request.headers.get("range")
    common = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    if not header:
        return FileResponse(path, media_type=media_type, headers=common)
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise HTTPException(status_code=416, detail="Invalid byte range")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        suffix = int(end_text)
        start = max(0, size - suffix)
        end = size - 1
    if start < 0 or start >= size or end < start:
        raise HTTPException(status_code=416, detail="Requested range is outside the file")
    end = min(end, size - 1)
    length = end - start + 1

    def iterator():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    headers = {**common, "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)}
    return StreamingResponse(iterator(), status_code=206, media_type=media_type, headers=headers)


@app.get("/api/media/audio-video-sync/{session_id}/{candidate_id}/{side}")
def audio_video_sync_media(
    session_id: str, candidate_id: str, side: Literal["source", "reference"], request: Request
):
    path = audio_video_sync_video_path(
        runtime.require_store(), session_id=session_id, candidate_id=candidate_id, side=side
    )
    return _range_video_response(path, request)


@app.get("/api/aoi-review/{session_id}/options")
def aoi_review_options(session_id: str):
    return review_options(runtime.require_store(), session_id=session_id)


@app.get("/api/aoi-review/{session_id}/unblurred-video/{video_artifact_id}")
def aoi_review_unblurred_video(
    session_id: str, video_artifact_id: str, request: Request
):
    path = review_unblurred_video_path(
        runtime.require_store(), session_id, video_artifact_id
    )
    return _range_video_response(path, request)


@app.get("/api/aoi-review/{session_id}/{mode}/{artifact_id}/manifest")
def aoi_review_manifest(
    session_id: str,
    mode: Literal["face", "body", "fusion"],
    artifact_id: str,
    video_artifact_id: str | None = None,
    confidence_threshold: float = Query(0.65, ge=0.0, le=1.0),
):
    return review_manifest(
        runtime.require_store(), session_id=session_id, mode=mode,
        artifact_id=artifact_id, video_artifact_id=video_artifact_id,
        confidence_threshold=confidence_threshold,
    )


@app.get("/api/aoi-review/{session_id}/{mode}/{artifact_id}/frame-window")
def aoi_review_frame_window(
    session_id: str,
    mode: Literal["face", "body", "fusion"],
    artifact_id: str,
    start_frame_index: int = Query(0, ge=0),
    frame_count: int = Query(24, ge=1, le=120),
    trail_frames: int = Query(6, ge=0, le=180),
    compact_playback: bool = Query(False),
    video_artifact_id: str | None = None,
    confidence_threshold: float = Query(0.65, ge=0.0, le=1.0),
):
    return review_frame_window(
        runtime.require_store(), session_id=session_id, mode=mode,
        artifact_id=artifact_id, video_artifact_id=video_artifact_id,
        start_frame_index=start_frame_index, frame_count=frame_count,
        trail_frames=trail_frames, confidence_threshold=confidence_threshold,
        compact_playback=compact_playback,
    )


@app.post("/api/aoi-review/{session_id}/face/{artifact_id}/decision")
def aoi_review_face_decision(
    session_id: str,
    artifact_id: str,
    request: FaceAOIReviewDecisionRequest,
):
    return save_face_review_decision(
        runtime.require_store(),
        session_id=session_id,
        artifact_id=artifact_id,
        **request.model_dump(),
    )


@app.post("/api/aoi-review/{session_id}/body/{artifact_id}/decision")
def aoi_review_body_decision(
    session_id: str,
    artifact_id: str,
    request: BodyAOIReviewDecisionRequest,
):
    return save_body_review_decision(
        runtime.require_store(),
        session_id=session_id,
        artifact_id=artifact_id,
        **request.model_dump(),
    )


@app.get("/api/aoi-review/{session_id}/{mode}/{artifact_id}/frame")
def aoi_review_frame(
    session_id: str,
    mode: Literal["face", "body", "fusion"],
    artifact_id: str,
    output_frame_index: int = Query(0, ge=0),
    sample_index: int = Query(0, ge=0),
    trail_frames: int = Query(6, ge=0, le=180),
    video_artifact_id: str | None = None,
    confidence_threshold: float = Query(0.65, ge=0.0, le=1.0),
):
    return review_frame(
        runtime.require_store(), session_id=session_id, mode=mode,
        artifact_id=artifact_id, video_artifact_id=video_artifact_id,
        output_frame_index=output_frame_index, sample_index=sample_index,
        trail_frames=trail_frames, confidence_threshold=confidence_threshold,
    )




@app.get("/api/media/native-trim-overlay/{session_id}/{preview_id}/{overlay_id}")
def native_trim_overlay_media(session_id: str, preview_id: str, overlay_id: str, request: Request):
    path = pretrim_overlay_video_path(runtime.require_store(), session_id, preview_id, overlay_id)
    return _range_video_response(path, request)


@app.get("/api/facemapper-live/{session_id}/{artifact_id}/manifest")
def facemapper_live_manifest(session_id: str, artifact_id: str):
    path = facemapper_live_manifest_path(runtime.require_store(), session_id, artifact_id)
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-cache"})


@app.get("/api/facemapper-live/{session_id}/{artifact_id}/video")
def facemapper_live_video(session_id: str, artifact_id: str, request: Request):
    path = facemapper_live_video_path(runtime.require_store(), session_id, artifact_id)
    return _range_video_response(path, request)


@app.get("/api/media/artifact/{session_id}/{artifact_id}")
def artifact_media(session_id: str, artifact_id: str, request: Request, variant: str | None = None):
    session = runtime.require_store().project.sessions[session_id]
    artifact = session.artifacts.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    videos = [Path(row.path) for row in artifact.files if Path(row.path).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"} and Path(row.path).exists()]
    if variant:
        selected = [path for path in videos if variant.lower() in path.name.lower()]
        if selected:
            videos = selected
    if not videos:
        raise HTTPException(status_code=404, detail="Artifact contains no previewable video")
    videos.sort(key=lambda path: (len(path.parts), str(path).lower()))
    return _range_video_response(videos[0], request)


@app.get("/api/thumbnail/{session_id}/{asset_id}")
def thumbnail(session_id: str, asset_id: str, at_sec: float = Query(0.0, ge=0.0)):
    store = runtime.require_store()
    session = store.project.sessions[session_id]
    asset = session.assets[asset_id]
    output = store.cache_dir(asset_id) / f"thumbnail_{int(at_sec * 1000)}.jpg"
    if not output.exists():
        cache = MaterializationCache(store.cache_dir())
        if asset.kind == "video":
            video_path = Path(asset.source_path)
        elif asset.kind == "pupil_native":
            video_path, _, _, _ = materialized_native_files(cache.materialize(asset), asset.recording_id)
        elif asset.kind == "pupil_cloud":
            root = cache.materialize(asset)
            candidates = find_cloud_candidates(root)
            if len(candidates) != 1 or not candidates[0].scene_video_name:
                raise ValidationError("Cloud asset has no uniquely identifiable scene video")
            video_path = root / candidates[0].scene_video_name
        else:
            raise ValidationError("The selected asset has no previewable video")
        make_thumbnail(video_path, output, at_sec=at_sec)
    return FileResponse(output, media_type="image/jpeg")


@app.get("/api/preview/etm-assignment/{session_id}/{artifact_id}")
async def etm_assignment_preview(session_id: str, artifact_id: str):
    data = await run_in_threadpool(
        get_etm_assignment_preview, runtime.require_store(), session_id, artifact_id
    )
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.get("/api/preview/gaze-rays/{session_id}/{artifact_id}")
async def gaze_rays_preview(session_id: str, artifact_id: str):
    data = await run_in_threadpool(
        get_gaze_rays_preview, runtime.require_store(), session_id, artifact_id
    )
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.get("/api/preview/gaze-rays-dyadic/{session_id}/{artifact_a_id}/{artifact_b_id}")
async def dyadic_gaze_rays_preview(
    session_id: str,
    artifact_a_id: str,
    artifact_b_id: str,
    maximum_samples: int = Query(default=1200, ge=120, le=2400),
    pairing_tolerance_ms: float | None = Query(default=None, gt=0, le=500),
):
    data = await run_in_threadpool(
        get_dyadic_gaze_rays_preview,
        runtime.require_store(),
        session_id,
        artifact_a_id,
        artifact_b_id,
        maximum_samples=maximum_samples,
        pairing_tolerance_ms=pairing_tolerance_ms,
    )
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.get("/api/preview/calibration/{session_id}")
async def calibration_motion_preview(
    session_id: str,
    board_asset_id: str | None = Query(default=None),
    tracker_asset_id: list[str] = Query(default=[]),
    world_asset_id: str | None = Query(default=None),
    profile_artifact_id: str | None = Query(default=None),
    maximum_samples: int = Query(default=600, ge=60, le=1200),
):
    data = await run_in_threadpool(
        get_calibration_motion_preview,
        runtime.require_store(),
        session_id,
        board_asset_id=board_asset_id,
        tracker_asset_ids=tracker_asset_id,
        world_asset_id=world_asset_id,
        profile_artifact_id=profile_artifact_id,
        maximum_samples=maximum_samples,
    )
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@app.get("/api/artifact-file/{session_id}/{artifact_id}/{file_index}")
def artifact_file(session_id: str, artifact_id: str, file_index: int):
    artifact = runtime.require_store().project.sessions[session_id].artifacts[artifact_id]
    if file_index < 0 or file_index >= len(artifact.files):
        raise HTTPException(status_code=404, detail="Artifact file index is out of range")
    path = Path(artifact.files[file_index].path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file no longer exists")
    return FileResponse(path)


@app.post("/api/dialog")
def dialog(request: DialogRequest):
    try:
        return isolated_dialog(request.kind, request.title, request.filetypes, request.multiple)
    except Exception as exc:
        raise ValidationError(f"File picker unavailable: {exc}. Use Choose files (browser) or paste a path.") from exc


@app.post("/api/open-path")
def open_path(request: OpenPathRequest):
    path = Path(request.path).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return {"ok": True}


@app.post("/api/reveal-path")
def reveal_path(request: OpenPathRequest):
    path = Path(request.path).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    if path.is_dir():
        target = path
        select_file = False
    else:
        target = path.parent
        select_file = True
    if os.name == "nt":
        if select_file:
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(target))  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-R", str(path)] if select_file else ["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return {"ok": True, "revealed": str(path)}


# Register the optional MotionStudio Scene Bridge on the same live runtime.
# v0.12.0 shipped the bridge files and UI but did not install its API routes,
# leaving the MotionStudio button unable to complete real imports.
from oculostudio.motionstudio_bridge.api import install_motionstudio_routes
install_motionstudio_routes(app, runtime=runtime)

STATIC_ROOT = Path(__file__).with_name("static")


@app.get("/", include_in_schema=False)
def studio_index():
    """Serve the application shell explicitly.

    Static assets live under /static.  Mounting the static directory at / while
    the HTML requested /static/... made the browser look for
    static/static/styles.css and static/static/app.js, leaving an unstyled,
    non-functional page.
    """
    return FileResponse(
        STATIC_ROOT / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
