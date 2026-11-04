from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.semantic import FACE_REGIONS, HEAD_COMPATIBLE_REGIONS, SemanticRegion
from oculostudio.errors import ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.services.body_primitives import body_geometry_integrity_from_qc
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, utc_now


FACE_VALID = set(FACE_REGIONS) | {SemanticRegion.FACE.value}
BODY_EXPLICIT_NO_HIT = {"no_body", "no_body_part", "no_body_hit"}
BODY_NO_RESULT = BODY_EXPLICIT_NO_HIT | {SemanticRegion.UNKNOWN.value, SemanticRegion.INVALID_GAZE.value, ""}
FACE_NO_RESULT = {"no_face", "no_face_detection", SemanticRegion.UNKNOWN.value, SemanticRegion.INVALID_GAZE.value, SemanticRegion.OUT_OF_FRAME.value, ""}
OBJECT_NO_RESULT = {"no_object", SemanticRegion.UNKNOWN.value, SemanticRegion.INVALID_GAZE.value, ""}


def _samples_path(artifact, filename: str) -> Path:
    return find_artifact_file(artifact, names=[filename])


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _number(value, default=0.0) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])


def _combine_confidence(a: float, b: float) -> float:
    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b, 0.0, 1.0))
    return float(np.clip(1.0 - (1.0-a)*(1.0-b), 0.0, 1.0))


def _face_body_decision(row: pd.Series) -> dict[str, Any]:
    face_region = _clean(row.get("face_region"))
    face_social_region = _clean(row.get("face_social_region")) or face_region
    face_entity = _clean(row.get("face_entity_id"))
    face_relation = _clean(row.get("target_relation"))
    body_region = _clean(row.get("body_region"))
    body_social_region = _clean(row.get("body_social_region")) or body_region
    body_raw_region = _clean(row.get("body_raw_region")) or body_region
    body_coarse_region = _clean(row.get("body_coarse_region")) or body_region
    body_entity = _clean(row.get("body_entity_id"))
    body_relation = _clean(row.get("body_relation"))
    face_conf = _number(row.get("face_aoi_confidence"))
    body_conf = _number(row.get("body_confidence"))
    body_exact = bool(row.get("body_exact_hit", False))
    self_occlusion = bool(row.get("self_occlusion", False))
    occlusion_status = _clean(row.get("occlusion_status")) or "none"
    face_valid = face_region in FACE_VALID
    body_valid = body_region not in BODY_NO_RESULT and body_region is not None
    review = bool(row.get("face_review_required", False) or row.get("body_review_required", False))
    distance = _number(row.get("body_distance_mm"), default=np.nan)

    projected_face_region = face_social_region if face_valid else None
    projected_face_entity = face_entity if face_valid else None

    # Visibility is determined by the first physical body surface.  A valid
    # 2D partner-face projection remains valuable evidence, but it cannot
    # silently override a nearer self hand/arm/body surface.
    if body_valid and body_relation == "self" and (body_exact or self_occlusion):
        final_region = body_social_region
        final_entity = body_entity
        domain = "body"
        confidence = body_conf
        source = "body_3d" + ("+face_2d_projected" if face_valid else "")
        status = "self_body_visible_first"
        if face_valid:
            status = "self_occludes_partner_face" if self_occlusion else "self_body_first_face_projected_behind"
            review = True
            confidence *= 0.95
    elif face_valid and body_valid:
        entity_conflict = bool(face_entity and body_entity and face_entity != body_entity)
        head_compatible = body_coarse_region == "head" or body_raw_region in HEAD_COMPATIBLE_REGIONS
        if body_relation == "partner" and head_compatible:
            final_region = face_social_region
            final_entity = body_entity or face_entity
            confidence = _combine_confidence(face_conf, body_conf)
            status = "face_refines_partner_head"
            domain = "face"
        elif body_exact:
            final_region, final_entity, domain = body_social_region, body_entity, "body"
            confidence = body_conf * 0.90
            status = "face_body_visibility_conflict_body_first"
            review = True
        elif face_conf > body_conf + 0.18:
            final_region, final_entity, domain = face_social_region, face_entity, "face"
            confidence = face_conf * 0.82
            status = "face_selected_over_uncertain_body"
            review = True
        else:
            final_region, final_entity, domain = body_social_region, body_entity, "body"
            confidence = body_conf * 0.82
            status = "uncertain_face_body_conflict_body_selected"
            review = True
        if entity_conflict:
            review = True
            status += "+identity_conflict"
            confidence *= 0.8
        source = "face_2d+body_3d"
    elif face_valid:
        final_region, final_entity, domain = face_social_region, face_entity, "face"
        confidence = face_conf
        status = "face_2d_only"
        source = "face_2d"
    elif body_valid:
        final_region, final_entity, domain = body_social_region, body_entity, "body"
        confidence = body_conf
        status = "body_3d_only"
        source = "body_3d"
    else:
        face_invalid = face_region in {SemanticRegion.INVALID_GAZE.value, SemanticRegion.OUT_OF_FRAME.value}
        body_invalid = body_region == SemanticRegion.INVALID_GAZE.value
        if face_invalid or body_invalid:
            final_region = SemanticRegion.INVALID_GAZE.value if body_invalid else face_region
            status = "invalid_gaze"
            confidence = 1.0
        elif face_region == "no_face" and body_region in BODY_EXPLICIT_NO_HIT:
            # A valid gaze sample outside all accepted green face boxes and with
            # no 3D body-part intersection is the residual scene/environment.
            # This is distinct from no_face_detection or an invalid body ray,
            # which remain unknown rather than being over-interpreted.
            final_region = SemanticRegion.ENVIRONMENT.value
            status = "strict_no_face_and_no_body_hit_environment"
            confidence = float(np.clip(max(face_conf, 0.95), 0.0, 1.0))
        elif face_region == "no_face_detection" and body_region in BODY_NO_RESULT:
            final_region = SemanticRegion.UNKNOWN.value
            status = "no_face_detection_and_no_body_hit"
            confidence = 0.0
        else:
            final_region = SemanticRegion.UNKNOWN.value
            status = "no_semantic_target"
            confidence = 0.0
        final_entity = None
        domain = "environment" if final_region == SemanticRegion.ENVIRONMENT.value else "unknown"
        source = "face_2d_no_hit+body_3d_no_hit" if domain == "environment" else "none"
    return {
        "entity": final_entity,
        "domain": domain,
        "region": final_region,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "status": status,
        "source": source,
        "review": bool(review),
        "distance_mm": distance,
        "visible_relation": body_relation if domain == "body" else face_relation,
        "occlusion_status": occlusion_status,
        "projected_face_entity_id": projected_face_entity,
        "projected_face_region": projected_face_region,
    }


def _fuse_row(row: pd.Series) -> dict[str, Any]:
    base = _face_body_decision(row)
    object_entity = _clean(row.get("object_entity_id"))
    object_region = _clean(row.get("object_region"))
    object_category = _clean(row.get("object_category"))
    object_conf = _number(row.get("object_confidence"))
    object_distance = _number(row.get("object_distance_mm"), default=np.nan)
    object_exact = bool(row.get("object_exact_hit", False))
    object_review = bool(row.get("object_review_required", False))
    object_valid = bool(object_entity and object_region not in OBJECT_NO_RESULT)

    if object_valid:
        if base["domain"] in {"unknown", "environment"}:
            base.update(
                entity=object_entity,
                domain="object",
                region=object_category or object_region or "object",
                confidence=object_conf,
                status="object_3d_only",
                source="object_3d",
                review=object_review,
                distance_mm=object_distance,
            )
        else:
            base_distance = base.get("distance_mm", np.nan)
            choose_object = False
            rationale = ""
            if np.isfinite(object_distance) and np.isfinite(base_distance):
                choose_object = object_distance + 10.0 < base_distance
                rationale = "nearest_3d_surface"
            elif object_exact and object_conf >= base["confidence"] - 0.05:
                choose_object = True
                rationale = "exact_object_hit"
            elif object_conf > base["confidence"] + 0.18:
                choose_object = True
                rationale = "higher_confidence"
            if choose_object:
                prior_domain = base["domain"]
                base.update(
                    entity=object_entity,
                    domain="object",
                    region=object_category or object_region or "object",
                    confidence=float(np.clip(object_conf * (1.0 if object_exact else 0.85), 0.0, 1.0)),
                    status=f"object_selected_over_{prior_domain}+{rationale}",
                    source=f"{base['source']}+object_3d",
                    review=bool(base["review"] or object_review or prior_domain == "face"),
                    distance_mm=object_distance,
                    visible_relation="object",
                    occlusion_status="object_visible_first",
                )
            else:
                base["status"] += "+object_conflict_nonselected"
                base["source"] += "+object_3d"
                base["review"] = True
                base["confidence"] *= 0.9
    return {
        "final_entity_id": base["entity"],
        "final_target_domain": base["domain"],
        "final_target_region": base["region"],
        "final_visible_relation": base.get("visible_relation"),
        "final_confidence": float(np.clip(base["confidence"], 0.0, 1.0)),
        "fusion_status": base["status"],
        "evidence_sources": base["source"],
        "final_occlusion_status": base.get("occlusion_status", "none"),
        "projected_face_entity_id": base.get("projected_face_entity_id"),
        "projected_face_region": base.get("projected_face_region"),
        "review_required": bool(base["review"]),
    }


def _frame_summary(frame: pd.DataFrame) -> pd.DataFrame:
    key = "native_frame_index" if "native_frame_index" in frame.columns else "scene_frame_index"
    rows = []
    for frame_index, group in frame.loc[frame[key].notna()].groupby(key):
        distribution = Counter(group["final_target_region"].fillna("unknown").astype(str))
        score: dict[tuple[str | None, str, str], float] = defaultdict(float)
        for row in group.itertuples(index=False):
            entity = getattr(row, "final_entity_id", None)
            domain = getattr(row, "final_target_domain", "unknown")
            region = getattr(row, "final_target_region", SemanticRegion.UNKNOWN.value)
            score[(entity if isinstance(entity, str) and entity else None, str(domain), str(region))] += max(0.05, float(getattr(row, "final_confidence", 0.0)))
        primary = max(score, key=score.get) if score else (None, "unknown", SemanticRegion.UNKNOWN.value)
        rows.append({
            key: int(frame_index),
            "gaze_sample_count": int(len(group)),
            "primary_entity_id": primary[0],
            "primary_target_domain": primary[1],
            "primary_target_region": primary[2],
            "target_distribution_json": json.dumps(distribution, sort_keys=True, separators=(",", ":")),
            "mean_confidence": float(pd.to_numeric(group["final_confidence"], errors="coerce").fillna(0).mean()),
            "review_required": bool(group["review_required"].fillna(False).astype(bool).any()),
        })
    return pd.DataFrame(rows)


def fuse_face_and_body_aoi(
    store: ProjectStore,
    *,
    session_id: str,
    face_aoi_artifact_id: str,
    body_aoi_artifact_id: str,
    object_aoi_artifact_id: str | None = None,
    label: str = "Full gaze targets",
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    session = store.project.sessions[session_id]
    face_artifact = session.artifacts.get(face_aoi_artifact_id)
    body_artifact = session.artifacts.get(body_aoi_artifact_id)
    object_artifact = session.artifacts.get(object_aoi_artifact_id) if object_aoi_artifact_id else None
    if face_artifact is None or face_artifact.kind != ArtifactKind.FACE_AOI:
        raise ValidationError("Select a Face AOI artifact")
    if body_artifact is None or body_artifact.kind != ArtifactKind.BODY_AOI:
        raise ValidationError("Select a Body AOI artifact")
    body_geometry_integrity = body_geometry_integrity_from_qc(body_artifact.qc or {})
    if not bool(body_geometry_integrity.get("fusion_ready")):
        failures = body_geometry_integrity.get("failures") or []
        reason = "; ".join(str(row.get("message") or row.get("code")) for row in failures[:4])
        raise ValidationError(
            "Selected Body AOI is not fusion-ready because torso/body geometry failed integrity checks. "
            + (reason or "Regenerate Body AOI with the current geometry gate.")
        )
    if object_aoi_artifact_id and (object_artifact is None or object_artifact.kind != ArtifactKind.OBJECT_AOI):
        raise ValidationError("object_aoi_artifact_id must reference an Object AOI artifact")
    emit(progress, 0.05, "Loading Face, Body and optional Object AOI files")
    face = pd.read_csv(_samples_path(face_artifact, "face_aoi_samples.csv"), low_memory=False)
    body = pd.read_csv(_samples_path(body_artifact, "body_aoi_samples.csv"), low_memory=False)
    object_frame = pd.read_csv(_samples_path(object_artifact, "object_aoi_samples.csv"), low_memory=False) if object_artifact else None
    for name, frame in [("Face", face), ("Body", body), *(([("Object", object_frame)] if object_frame is not None else []))]:
        if "gaze_sample_uid" not in frame.columns or frame["gaze_sample_uid"].duplicated().any():
            raise ValidationError(f"{name} AOI file requires unique gaze_sample_uid values")
    face_recordings = set(face["recording_id"].dropna().astype(str)) if "recording_id" in face else set()
    body_recordings = set(body["recording_id"].dropna().astype(str)) if "recording_id" in body else set()
    if face_recordings and body_recordings and face_recordings != body_recordings:
        raise ValidationError(f"Face and Body AOI recording IDs differ: {face_recordings} vs {body_recordings}")

    face_keep = [
        "gaze_sample_uid", "recording_id", "source_gaze_row_index", "timestamp_ns", "clip_time_ns", "native_frame_index",
        "gaze_x_px", "gaze_y_px", "worn", "blink_id", "fixation_id", "fm_gaze_on_face",
        "face_geometry_time_policy", "face_geometry_time_mode", "face_geometry_observation_clock", "face_geometry_classification_clock",
        "face_geometry_previous_frame_index", "face_geometry_next_frame_index", "face_geometry_interpolation_alpha",
        "face_geometry_interpolation_gap_ms", "face_geometry_interpolation_confidence", "face_geometry_interpolation_reason",
        "face_entity_id", "face_track_id", "face_region", "face_social_region", "face_aoi_confidence", "identity_status", "identity_confidence",
        "target_relation", "candidate_entity_id", "candidate_role", "candidate_relation",
        "include_in_primary_analysis", "assignment_status", "review_required",
    ]
    body_keep = [
        "gaze_sample_uid", "source_gaze_row_index", "timestamp_ns", "scene_frame_index", "scene_timestamp_ns",
        "scene_frame_available", "review_preview_available", "body_geometry_time_mode", "body_geometry_query_timestamp_ns",
        "ray_valid", "ray_origin_x_mm", "ray_origin_y_mm", "ray_origin_z_mm", "ray_direction_x", "ray_direction_y", "ray_direction_z",
        "wearer_role_id", "partner_role_id", "body_entity_id", "body_relation", "body_raw_region", "body_coarse_region",
        "body_region", "body_social_region", "body_distance_mm", "body_hit_x_mm", "body_hit_y_mm", "body_hit_z_mm",
        "body_confidence", "body_exact_hit", "body_angular_miss_deg", "body_geometry_backend", "candidate_hit_count",
        "first_hit_entity_id", "first_hit_relation", "first_hit_raw_region", "first_hit_coarse_region", "first_hit_region",
        "first_hit_social_region", "first_hit_distance_mm", "first_hit_x_mm", "first_hit_y_mm", "first_hit_z_mm",
        "first_hit_confidence", "first_hit_exact", "first_hit_angular_miss_deg", "first_hit_source_asset_id",
        "second_hit_entity_id", "second_hit_relation", "second_hit_raw_region", "second_hit_coarse_region", "second_hit_region",
        "second_hit_social_region", "second_hit_distance_mm", "second_hit_x_mm", "second_hit_y_mm", "second_hit_z_mm",
        "second_hit_confidence", "second_hit_exact", "second_hit_angular_miss_deg", "second_hit_source_asset_id",
        "nearest_partner_hit_entity_id", "nearest_partner_hit_relation", "nearest_partner_hit_raw_region",
        "nearest_partner_hit_coarse_region", "nearest_partner_hit_region", "nearest_partner_hit_social_region",
        "nearest_partner_hit_distance_mm", "nearest_partner_hit_x_mm", "nearest_partner_hit_y_mm", "nearest_partner_hit_z_mm",
        "nearest_partner_hit_confidence", "nearest_partner_hit_exact", "nearest_partner_hit_angular_miss_deg",
        "nearest_partner_hit_source_asset_id", "self_occlusion", "occlusion_status", "assignment_status", "review_required",
    ]
    face2 = face[[c for c in face_keep if c in face.columns]].copy().rename(columns={"assignment_status": "face_assignment_status", "review_required": "face_review_required"})
    body2 = body[[c for c in body_keep if c in body.columns]].copy().rename(columns={"source_gaze_row_index": "body_source_gaze_row_index", "timestamp_ns": "body_timestamp_ns", "assignment_status": "body_assignment_status", "review_required": "body_review_required"})
    merged = face2.merge(body2, on="gaze_sample_uid", how="outer", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValidationError(
            f"Face/Body gaze universes differ (missing Face={int((merged['_merge']=='right_only').sum())}, missing Body={int((merged['_merge']=='left_only').sum())})"
        )
    merged = merged.drop(columns=["_merge"])
    timestamp_diff = pd.to_numeric(merged["timestamp_ns"], errors="coerce") - pd.to_numeric(merged["body_timestamp_ns"], errors="coerce")
    if timestamp_diff.abs().fillna(0).max() > 0:
        raise ValidationError("Face and Body AOI timestamps differ for the same gaze_sample_uid")
    if object_frame is not None:
        object_keep = [
            "gaze_sample_uid", "source_gaze_row_index", "timestamp_ns", "object_entity_id", "object_category", "object_region",
            "object_distance_mm", "object_hit_x_mm", "object_hit_y_mm", "object_hit_z_mm", "object_confidence", "object_exact_hit",
            "object_angular_miss_deg", "object_geometry_backend", "assignment_status", "review_required",
        ]
        object2 = object_frame[[c for c in object_keep if c in object_frame.columns]].copy().rename(columns={
            "source_gaze_row_index": "object_source_gaze_row_index",
            "timestamp_ns": "object_timestamp_ns",
            "assignment_status": "object_assignment_status",
            "review_required": "object_review_required",
        })
        merged = merged.merge(object2, on="gaze_sample_uid", how="left", validate="one_to_one")
        object_ts = pd.to_numeric(merged.get("object_timestamp_ns"), errors="coerce")
        face_ts = pd.to_numeric(merged["timestamp_ns"], errors="coerce")
        mismatch = object_ts.notna() & face_ts.notna() & (object_ts != face_ts)
        if mismatch.any():
            raise ValidationError(f"Object AOI timestamps differ for {int(mismatch.sum())} gaze samples")
    emit(progress, 0.38, "Fusing Face, Body and Object evidence")
    decision_frame = pd.DataFrame([_fuse_row(row) for _, row in merged.iterrows()])
    output = pd.concat([merged, decision_frame], axis=1)
    output["timestamp_ns"] = output["timestamp_ns"].fillna(output["body_timestamp_ns"])
    summary = _frame_summary(output)
    token.raise_if_cancelled()
    artifact_id = new_id("artifact")
    root = store.derived_dir(session_id, "target_fusion") / artifact_id
    root.mkdir(parents=True, exist_ok=False)
    samples_csv = root / "gaze_targets_full.csv"
    frames_csv = root / "gaze_targets_frames.csv"
    output.to_csv(samples_csv, index=False)
    summary.to_csv(frames_csv, index=False)
    parquet_files: list[str] = []
    if output_parquet:
        try:
            for frame, name in [(output, "gaze_targets_full.parquet"), (summary, "gaze_targets_frames.parquet")]:
                path = root / name
                frame.to_parquet(path, index=False)
                parquet_files.append(str(path))
        except Exception:
            parquet_files = []
    counts = Counter(output["final_target_region"].astype(str))
    domains = Counter(output["final_target_domain"].astype(str))
    qc = {
        "face_rows": int(len(face)),
        "body_rows": int(len(body)),
        "object_rows": int(len(object_frame)) if object_frame is not None else 0,
        "output_rows": int(len(output)),
        "row_count_preserved": bool(len(face) == len(body) == len(output)),
        "known_target_fraction": float((~output["final_target_region"].isin([SemanticRegion.UNKNOWN.value, SemanticRegion.INVALID_GAZE.value, SemanticRegion.OUT_OF_FRAME.value])).mean()) if len(output) else 0.0,
        "object_target_fraction": float((output["final_target_domain"] == "object").mean()) if len(output) else 0.0,
        "environment_target_fraction": float((output["final_target_domain"] == "environment").mean()) if len(output) else 0.0,
        "environment_target_count": int((output["final_target_domain"] == "environment").sum()) if len(output) else 0,
        "self_visible_target_fraction": float((output["final_visible_relation"] == "self").mean()) if len(output) else 0.0,
        "partner_visible_target_fraction": float((output["final_visible_relation"] == "partner").mean()) if len(output) else 0.0,
        "self_occludes_partner_fraction": float((output["final_occlusion_status"] == "self_occludes_partner").mean()) if len(output) else 0.0,
        "review_required_fraction": float(output["review_required"].mean()) if len(output) else 0.0,
        "target_region_counts": dict(counts),
        "target_domain_counts": dict(domains),
        "visibility_rule": "strict_first_hit_with_self_partner_occlusion",
        "environment_rule": "strict_no_face_and_no_body_hit",
        "face_geometry_time_policy": (face_artifact.qc or {}).get("face_geometry_time_policy"),
        "face_track_interpolated_sample_fraction": (face_artifact.qc or {}).get("track_interpolated_sample_fraction"),
        "body_geometry_clock": (body_artifact.qc or {}).get("body_geometry_clock"),
        "body_video_clock_usage": (body_artifact.qc or {}).get("video_clock_usage"),
    }
    manifest = {
        "schema_version": "oculostudio-target-fusion-2.2",
        "artifact_id": artifact_id,
        "face_aoi_artifact_id": face_aoi_artifact_id,
        "body_aoi_artifact_id": body_aoi_artifact_id,
        "object_aoi_artifact_id": object_aoi_artifact_id,
        "join_key": "gaze_sample_uid",
        "outputs": {"samples_csv": str(samples_csv), "frames_csv": str(frames_csv), "parquet": parquet_files},
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "target_fusion_manifest.json", manifest)
    atomic_write_json(root / "qc_metrics.json", qc)
    parents = [face_aoi_artifact_id, body_aoi_artifact_id] + ([object_aoi_artifact_id] if object_aoi_artifact_id else [])
    sources = [*face_artifact.source_asset_ids, *body_artifact.source_asset_ids, *((object_artifact.source_asset_ids if object_artifact else []))]
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.TARGET_FUSION,
        label=label,
        output_root=root,
        source_asset_ids=list(dict.fromkeys(sources)),
        parent_artifact_ids=parents,
        parameters={
            "join_key": "gaze_sample_uid",
            "output_parquet": output_parquet,
            "object_aoi_artifact_id": object_aoi_artifact_id,
            "visibility_rule": "strict_first_hit_with_self_partner_occlusion",
            "environment_rule": "strict_no_face_and_no_body_hit",
            "body_geometry_clock": (body_artifact.qc or {}).get("body_geometry_clock"),
            "face_geometry_time_policy": (face_artifact.qc or {}).get("face_geometry_time_policy"),
        },
        qc=qc,
        provenance={
            "face_aoi_artifact_id": face_aoi_artifact_id,
            "body_aoi_artifact_id": body_aoi_artifact_id,
            "object_aoi_artifact_id": object_aoi_artifact_id,
            "face_identity_basis": "2d_facemapper_plus_3d_partner_head_projection",
            "body_geometry_source": "c3d_articulated_primitives_v6_gaze_sample_clock",
            "body_geometry_clock": "gaze_sample_timestamp",
            "face_geometry_observation_clock": "native_video_frames",
            "face_geometry_classification_clock": "gaze_sample_timestamps",
            "environment_definition": "valid gaze outside accepted green face boxes and with no body-part hit",
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Face, Body and Object target fusion complete")
    return {"artifact": artifact.model_dump(mode="json"), "samples_csv": str(samples_csv), "frames_csv": str(frames_csv), "manifest": str(root / "target_fusion_manifest.json")}
