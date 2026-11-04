from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.semantic import FACE_REGIONS, SemanticRegion
from oculostudio.domain.timegraph import TimeGraph
from oculostudio.errors import ValidationError
from oculostudio.services.artifact_io import find_artifact_file
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, utc_now


SUPPORTED_KINDS = {
    ArtifactKind.TARGET_FUSION,
    ArtifactKind.FACE_AOI,
    ArtifactKind.BODY_AOI,
    ArtifactKind.OBJECT_AOI,
    ArtifactKind.SEMANTIC_TARGETS,
}
FACE_LIKE = set(FACE_REGIONS) | {SemanticRegion.HEAD.value, SemanticRegion.NECK.value}
EYE_REGIONS = {SemanticRegion.LEFT_EYE.value, SemanticRegion.RIGHT_EYE.value}
UNKNOWN_REGIONS = {
    "", "unknown", "invalid_gaze", "out_of_frame", "no_face", "no_face_detection", "no_body", "no_object",
}


def _read_samples(artifact) -> pd.DataFrame:
    names = {
        ArtifactKind.TARGET_FUSION: ["gaze_targets_full.csv"],
        ArtifactKind.FACE_AOI: ["face_aoi_samples.csv"],
        ArtifactKind.BODY_AOI: ["body_aoi_samples.csv"],
        ArtifactKind.OBJECT_AOI: ["object_aoi_samples.csv"],
        ArtifactKind.SEMANTIC_TARGETS: ["gaze_targets.csv", "gaze_targets_full.csv"],
    }
    path = find_artifact_file(artifact, names=names.get(artifact.kind, []))
    return pd.read_csv(path, low_memory=False)


def _col(frame: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise ValidationError(f"Missing required column; expected one of {names}")
    return None


def _normalise_artifact(
    frame: pd.DataFrame,
    artifact_kind: str,
    role_id: str,
    *,
    alignment_mode: Literal["timegraph", "pretrimmed_shared_start"] = "timegraph",
) -> pd.DataFrame:
    if alignment_mode == "pretrimmed_shared_start":
        timestamp_col = _col(
            frame,
            ["clip_time_ns", "oculo_clip_video_ns", "oculo_clip_time_ns"],
            required=False,
        )
        local_seconds_col = _col(
            frame,
            ["gaze3d_tracker_local_time_sec", "tracker_local_time_sec", "local_time_sec"],
            required=False,
        )
        if timestamp_col:
            timestamp = pd.to_numeric(frame[timestamp_col], errors="coerce").astype("Int64")
        elif local_seconds_col:
            seconds = pd.to_numeric(frame[local_seconds_col], errors="coerce")
            timestamp = pd.Series(
                pd.array(np.rint(seconds.to_numpy(dtype=float) * 1_000_000_000), dtype="Int64"),
                index=frame.index,
            )
        else:
            raise ValidationError(
                f"{role_id} target artifact has no trim-local timestamp column. "
                "Regenerate Face/Body/Fused targets from a Native-reference trim, "
                "or use verified TimeGraph alignment."
            )
    else:
        timestamp_col = _col(
            frame,
            ["timestamp_global_ns", "global_timestamp_ns", "timestamp_ns", "timestamp [ns]"],
        )
        timestamp = pd.to_numeric(frame[timestamp_col], errors="coerce").astype("Int64")

    uid_col = _col(frame, ["gaze_sample_uid"], required=False)
    review_col = _col(frame, ["review_required"], required=False)
    native_col = _col(frame, ["native_frame_index", "scene_frame_index"], required=False)
    if artifact_kind in {ArtifactKind.TARGET_FUSION, ArtifactKind.SEMANTIC_TARGETS}:
        entity_col = _col(frame, ["final_entity_id", "target_entity_id"], required=False)
        domain_col = _col(frame, ["final_target_domain", "target_domain"], required=False)
        region_col = _col(frame, ["final_target_region", "target_region"])
        confidence_col = _col(frame, ["final_confidence", "target_confidence"], required=False)
    elif artifact_kind == ArtifactKind.FACE_AOI:
        entity_col = _col(frame, ["face_entity_id", "matched_entity_id"], required=False)
        domain_col = None
        region_col = _col(frame, ["face_region"])
        confidence_col = _col(frame, ["face_aoi_confidence", "identity_confidence"], required=False)
    elif artifact_kind == ArtifactKind.BODY_AOI:
        entity_col = _col(frame, ["body_entity_id"], required=False)
        domain_col = None
        region_col = _col(frame, ["body_region"])
        confidence_col = _col(frame, ["body_confidence"], required=False)
    elif artifact_kind == ArtifactKind.OBJECT_AOI:
        entity_col = _col(frame, ["object_entity_id"])
        domain_col = None
        region_col = _col(frame, ["object_region"], required=False)
        confidence_col = _col(frame, ["object_confidence"], required=False)
    else:
        raise ValidationError(f"Unsupported target artifact kind: {artifact_kind}")
    entity = frame[entity_col].astype("string") if entity_col else pd.Series(pd.NA, index=frame.index, dtype="string")
    region = frame[region_col].astype("string")
    if domain_col:
        domain = frame[domain_col].astype("string")
    else:
        default = "face" if artifact_kind == ArtifactKind.FACE_AOI else ("body" if artifact_kind == ArtifactKind.BODY_AOI else "object")
        domain = pd.Series(default, index=frame.index, dtype="string")
    confidence = pd.to_numeric(frame[confidence_col], errors="coerce").fillna(0.0) if confidence_col else pd.Series(0.0, index=frame.index)
    review = frame[review_col].fillna(False).astype(bool) if review_col else pd.Series(False, index=frame.index)
    uid = frame[uid_col].astype("string") if uid_col else pd.Series([f"{role_id}:{i}" for i in range(len(frame))], dtype="string")
    native = pd.to_numeric(frame[native_col], errors="coerce").astype("Int64") if native_col else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    output = pd.DataFrame({
        "role_id": role_id,
        "gaze_sample_uid": uid,
        "source_timestamp_ns": timestamp,
        "native_frame_index": native,
        "target_entity_id": entity,
        "target_domain": domain,
        "target_region": region,
        "target_confidence": confidence.clip(0.0, 1.0),
        "review_required": review,
    })
    output = output.loc[output["source_timestamp_ns"].notna()].copy()
    output["source_timestamp_ns"] = output["source_timestamp_ns"].astype("int64")
    output["timeline_basis"] = alignment_mode
    return output.sort_values("source_timestamp_ns", kind="stable").reset_index(drop=True)


def _convert_global(frame: pd.DataFrame, graph: TimeGraph, source_domain_id: str, global_domain_id: str) -> pd.DataFrame:
    output = frame.copy()
    output["global_timestamp_ns"] = graph.convert(
        source_domain_id,
        global_domain_id,
        output["source_timestamp_ns"].to_numpy(dtype=np.int64),
    ).astype(np.int64)
    return output.sort_values("global_timestamp_ns", kind="stable").reset_index(drop=True)


def _primary_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "target_entity_id": None,
            "target_domain": "unknown",
            "target_region": "unknown",
            "confidence": 0.0,
            "review_required": True,
            "distribution": {},
        }
    valid = frame.loc[~frame["target_region"].fillna("unknown").astype(str).isin(UNKNOWN_REGIONS)].copy()
    if valid.empty:
        return {
            "sample_count": int(len(frame)),
            "target_entity_id": None,
            "target_domain": "unknown",
            "target_region": "unknown",
            "confidence": float(frame["target_confidence"].mean()) if len(frame) else 0.0,
            "review_required": bool(frame["review_required"].any()),
            "distribution": Counter(frame["target_region"].fillna("unknown").astype(str)),
        }
    scores: dict[tuple[str | None, str, str], float] = defaultdict(float)
    counts: Counter = Counter()
    for row in valid.itertuples(index=False):
        entity = str(row.target_entity_id) if pd.notna(row.target_entity_id) and str(row.target_entity_id) else None
        domain = str(row.target_domain) if pd.notna(row.target_domain) else "unknown"
        region = str(row.target_region)
        scores[(entity, domain, region)] += max(0.05, float(row.target_confidence))
        counts[f"{entity or 'none'}|{domain}|{region}"] += 1
    primary = max(scores, key=scores.get)
    chosen = valid.loc[
        (valid["target_entity_id"].fillna("").astype(str) == (primary[0] or ""))
        & (valid["target_domain"].astype(str) == primary[1])
        & (valid["target_region"].astype(str) == primary[2])
    ]
    return {
        "sample_count": int(len(frame)),
        "target_entity_id": primary[0],
        "target_domain": primary[1],
        "target_region": primary[2],
        "confidence": float(chosen["target_confidence"].mean()) if len(chosen) else 0.0,
        "review_required": bool(frame["review_required"].any()),
        "distribution": counts,
    }


def _looks_at_partner(target: dict[str, Any], partner_role: str) -> tuple[bool, bool]:
    entity = str(target.get("target_entity_id") or "")
    region = str(target.get("target_region") or "")
    normalized = region
    for prefix in ("partner_", "self_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    broad = entity == partner_role and normalized in FACE_LIKE
    strict = entity == partner_role and normalized in EYE_REGIONS
    return broad, strict


def _frame_state(a: dict[str, Any], b: dict[str, Any], role_a: str, role_b: str) -> dict[str, Any]:
    a_broad, a_eye = _looks_at_partner(a, role_b)
    b_broad, b_eye = _looks_at_partner(b, role_a)
    mutual_face = a_broad and b_broad
    mutual_eye = a_eye and b_eye
    a_entity = str(a.get("target_entity_id") or "")
    b_entity = str(b.get("target_entity_id") or "")
    a_domain = str(a.get("target_domain") or "unknown")
    b_domain = str(b.get("target_domain") or "unknown")
    same_entity = bool(a_entity and b_entity and a_entity == b_entity)
    joint_object = same_entity and a_domain == "object" and b_domain == "object"
    shared_social = same_entity and not joint_object and a_entity not in {role_a, role_b}
    if mutual_eye:
        state = "mutual_eye_contact"
    elif mutual_face:
        state = "mutual_face_gaze"
    elif joint_object:
        state = "joint_attention_object"
    elif shared_social:
        state = "shared_social_attention"
    elif a_broad or b_broad:
        state = "unilateral_partner_gaze"
    elif a.get("target_region") == "unknown" or b.get("target_region") == "unknown":
        state = "unknown"
    else:
        state = "divergent_attention"
    confidence = min(float(a.get("confidence") or 0.0), float(b.get("confidence") or 0.0))
    review = bool(a.get("review_required") or b.get("review_required") or confidence < 0.5)
    return {
        "role_a_looks_at_role_b_face": a_broad,
        "role_b_looks_at_role_a_face": b_broad,
        "mutual_face_gaze": mutual_face,
        "mutual_eye_contact": mutual_eye,
        "joint_attention": joint_object or shared_social,
        "joint_attention_type": "object" if joint_object else ("shared_social" if shared_social else "none"),
        "shared_target_entity_id": a_entity if same_entity else None,
        "dyadic_state": state,
        "dyadic_confidence": confidence,
        "review_required": review,
    }


def _bridge_boolean(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    result = values.astype(bool).copy()
    if max_gap_frames <= 0 or len(result) < 3:
        return result
    true_indices = np.flatnonzero(result)
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        gap = right - left - 1
        if 0 < gap <= max_gap_frames:
            result[left + 1:right] = True
    return result


def _episodes(frames: pd.DataFrame, mask: np.ndarray, *, episode_type: str, min_frames: int, step_ns: int) -> pd.DataFrame:
    mask = mask.astype(bool)
    rows: list[dict[str, Any]] = []
    start = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index
            if end - start >= min_frames:
                group = frames.iloc[start:end]
                rows.append({
                    "episode_id": f"{episode_type}_{len(rows)+1:05d}",
                    "episode_type": episode_type,
                    "start_frame_index": int(group["interaction_frame_index"].iloc[0]),
                    "end_frame_index_exclusive": int(group["interaction_frame_index"].iloc[-1]) + 1,
                    "start_global_ns": int(group["global_start_ns"].iloc[0]),
                    "end_global_ns": int(group["global_end_ns"].iloc[-1]),
                    "duration_ms": float((group["global_end_ns"].iloc[-1] - group["global_start_ns"].iloc[0]) / 1e6),
                    "frame_count": int(len(group)),
                    "mean_confidence": float(group["dyadic_confidence"].mean()),
                    "review_required": bool(group["review_required"].any()),
                    "strict_eye_fraction": float(group["mutual_eye_contact"].mean()) if "mutual_eye_contact" in group else 0.0,
                    "shared_target_entity_id": (
                        group["shared_target_entity_id"].dropna().astype(str).mode().iloc[0]
                        if group["shared_target_entity_id"].notna().any() else None
                    ),
                })
            start = None
    return pd.DataFrame(rows)


def _joint_episodes(frames: pd.DataFrame, *, min_frames: int, max_gap_frames: int, role_a_samples: pd.DataFrame,
                    role_b_samples: pd.DataFrame, role_a: str, role_b: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = [value for value in frames["shared_target_entity_id"].dropna().astype(str).unique() if value]
    for target in targets:
        raw = frames["joint_attention"].astype(bool).to_numpy() & (frames["shared_target_entity_id"].fillna("").astype(str).to_numpy() == target)
        mask = _bridge_boolean(raw, max_gap_frames)
        episodes = _episodes(frames, mask, episode_type="joint_attention", min_frames=min_frames, step_ns=0)
        for episode in episodes.to_dict("records"):
            start_ns = int(episode["start_global_ns"])
            end_ns = int(episode["end_global_ns"])
            lookback = 2_000_000_000
            def onset(samples: pd.DataFrame) -> int | None:
                selected = samples.loc[
                    (samples["global_timestamp_ns"] >= start_ns - lookback)
                    & (samples["global_timestamp_ns"] < end_ns)
                    & (samples["target_entity_id"].fillna("").astype(str) == target)
                ]
                return int(selected["global_timestamp_ns"].iloc[0]) if len(selected) else None
            onset_a, onset_b = onset(role_a_samples), onset(role_b_samples)
            if onset_a is None or onset_b is None or onset_a == onset_b:
                initiator = "simultaneous_or_unknown"
                latency_ms = None if onset_a is None or onset_b is None else 0.0
            elif onset_a < onset_b:
                initiator = role_a
                latency_ms = (onset_b - onset_a) / 1e6
            else:
                initiator = role_b
                latency_ms = (onset_a - onset_b) / 1e6
            episode.update({
                "shared_target_entity_id": target,
                "initiator_role": initiator,
                "role_a_target_onset_ns": onset_a,
                "role_b_target_onset_ns": onset_b,
                "follow_latency_ms": latency_ms,
            })
            rows.append(episode)
    return pd.DataFrame(rows)


def _mutual_nearest_pairs(a: pd.DataFrame, b: pd.DataFrame, tolerance_ns: int, role_a: str, role_b: str) -> pd.DataFrame:
    a_times = a["global_timestamp_ns"].to_numpy(dtype=np.int64)
    b_times = b["global_timestamp_ns"].to_numpy(dtype=np.int64)
    if not len(a_times) or not len(b_times):
        return pd.DataFrame()
    positions = np.searchsorted(b_times, a_times)
    right = np.clip(positions, 0, len(b_times) - 1)
    left = np.clip(positions - 1, 0, len(b_times) - 1)
    choose_right = np.abs(b_times[right] - a_times) < np.abs(b_times[left] - a_times)
    nearest_b = np.where(choose_right, right, left)
    reverse_positions = np.searchsorted(a_times, b_times)
    reverse_right = np.clip(reverse_positions, 0, len(a_times) - 1)
    reverse_left = np.clip(reverse_positions - 1, 0, len(a_times) - 1)
    reverse_choose_right = np.abs(a_times[reverse_right] - b_times) < np.abs(a_times[reverse_left] - b_times)
    nearest_a = np.where(reverse_choose_right, reverse_right, reverse_left)
    rows: list[dict[str, Any]] = []
    for ai, bi in enumerate(nearest_b):
        delta = int(b_times[bi] - a_times[ai])
        if abs(delta) > tolerance_ns or int(nearest_a[bi]) != ai:
            continue
        row_a = a.iloc[ai]
        row_b = b.iloc[bi]
        state = _frame_state(
            {
                "target_entity_id": row_a.target_entity_id,
                "target_domain": row_a.target_domain,
                "target_region": row_a.target_region,
                "confidence": row_a.target_confidence,
                "review_required": row_a.review_required,
            },
            {
                "target_entity_id": row_b.target_entity_id,
                "target_domain": row_b.target_domain,
                "target_region": row_b.target_region,
                "confidence": row_b.target_confidence,
                "review_required": row_b.review_required,
            },
            role_a,
            role_b,
        )
        rows.append({
            "role_a_gaze_sample_uid": row_a.gaze_sample_uid,
            "role_b_gaze_sample_uid": row_b.gaze_sample_uid,
            "role_a_global_timestamp_ns": int(row_a.global_timestamp_ns),
            "role_b_global_timestamp_ns": int(row_b.global_timestamp_ns),
            "time_difference_ms": delta / 1e6,
            "role_a_target_entity_id": row_a.target_entity_id,
            "role_a_target_domain": row_a.target_domain,
            "role_a_target_region": row_a.target_region,
            "role_a_confidence": float(row_a.target_confidence),
            "role_b_target_entity_id": row_b.target_entity_id,
            "role_b_target_domain": row_b.target_domain,
            "role_b_target_region": row_b.target_region,
            "role_b_confidence": float(row_b.target_confidence),
            **state,
        })
    return pd.DataFrame(rows)


def create_dyadic_attention_annotations(
    store: ProjectStore,
    *,
    session_id: str,
    role_a_target_artifact_id: str,
    role_b_target_artifact_id: str,
    role_a_id: str = "R01",
    role_b_id: str = "R02",
    role_a_time_domain_id: str | None = None,
    role_b_time_domain_id: str | None = None,
    alignment_mode: Literal["timegraph", "pretrimmed_shared_start"] = "timegraph",
    canonical_fps: int = 30,
    cross_wearer_tolerance_ms: float = 25.0,
    minimum_episode_ms: float = 100.0,
    short_gap_ms: float = 50.0,
    label: str = "Dyadic attention",
    output_parquet: bool = False,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    if role_a_id == role_b_id:
        raise ValidationError("Role A and Role B IDs must differ")
    if not (1 <= canonical_fps <= 120):
        raise ValidationError("canonical_fps must be between 1 and 120")
    if cross_wearer_tolerance_ms <= 0:
        raise ValidationError("cross_wearer_tolerance_ms must be positive")
    session = store.project.sessions[session_id]
    artifact_a = session.artifacts.get(role_a_target_artifact_id)
    artifact_b = session.artifacts.get(role_b_target_artifact_id)
    if artifact_a is None or artifact_a.kind not in SUPPORTED_KINDS:
        raise ValidationError("Role A target artifact must be Face, Body, Object, Fusion or Semantic Targets")
    if artifact_b is None or artifact_b.kind not in SUPPORTED_KINDS:
        raise ValidationError("Role B target artifact must be Face, Body, Object, Fusion or Semantic Targets")
    if alignment_mode not in {"timegraph", "pretrimmed_shared_start"}:
        raise ValidationError(f"Unsupported dyadic alignment mode: {alignment_mode}")

    graph: TimeGraph | None = None
    if alignment_mode == "timegraph":
        if role_a_time_domain_id not in session.time_domains or role_b_time_domain_id not in session.time_domains:
            raise ValidationError("Select valid source time domains for both roles")
        graph = TimeGraph.from_session(session)
        # Resolve early and fail closed if either clock is not connected to Global Time.
        graph.resolve(role_a_time_domain_id, session.global_time_domain_id)
        graph.resolve(role_b_time_domain_id, session.global_time_domain_id)

    emit(progress, 0.05, "Loading both target streams")
    a = _normalise_artifact(
        _read_samples(artifact_a), artifact_a.kind, role_a_id, alignment_mode=alignment_mode
    )
    b = _normalise_artifact(
        _read_samples(artifact_b), artifact_b.kind, role_b_id, alignment_mode=alignment_mode
    )
    if alignment_mode == "timegraph":
        assert graph is not None
        a = _convert_global(a, graph, str(role_a_time_domain_id), session.global_time_domain_id)
        b = _convert_global(b, graph, str(role_b_time_domain_id), session.global_time_domain_id)
    else:
        # Both streams were derived from jointly trimmed recordings. Their local
        # timestamps already share zero; keep the historical column name for
        # downstream compatibility while recording the actual timeline basis.
        a["global_timestamp_ns"] = a["source_timestamp_ns"].astype("int64")
        b["global_timestamp_ns"] = b["source_timestamp_ns"].astype("int64")
    if a.empty or b.empty:
        raise ValidationError("Both roles require non-empty target streams")
    overlap_start = max(int(a.global_timestamp_ns.min()), int(b.global_timestamp_ns.min()))
    overlap_end = min(int(a.global_timestamp_ns.max()), int(b.global_timestamp_ns.max()))
    if overlap_end <= overlap_start:
        basis = "shared Native-trim local time" if alignment_mode == "pretrimmed_shared_start" else "Global Time"
        raise ValidationError(f"The two target streams have no overlap in {basis}")
    a = a.loc[(a.global_timestamp_ns >= overlap_start) & (a.global_timestamp_ns <= overlap_end)].reset_index(drop=True)
    b = b.loc[(b.global_timestamp_ns >= overlap_start) & (b.global_timestamp_ns <= overlap_end)].reset_index(drop=True)

    emit(progress, 0.2, "Pairing cross-wearer gaze samples")
    pairs = _mutual_nearest_pairs(a, b, int(round(cross_wearer_tolerance_ms * 1e6)), role_a_id, role_b_id)
    token.raise_if_cancelled()

    step_ns = int(round(1_000_000_000 / canonical_fps))
    frame_count = max(1, int(math.ceil((overlap_end - overlap_start) / step_ns)))
    a_index = np.floor((a.global_timestamp_ns.to_numpy(dtype=np.int64) - overlap_start) / step_ns).astype(np.int64)
    b_index = np.floor((b.global_timestamp_ns.to_numpy(dtype=np.int64) - overlap_start) / step_ns).astype(np.int64)
    a_groups = {int(index): group for index, group in a.assign(__frame=a_index).groupby("__frame")}
    b_groups = {int(index): group for index, group in b.assign(__frame=b_index).groupby("__frame")}
    emit(progress, 0.38, "Building common interaction frames")
    frame_rows: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        if frame_index % 2000 == 0:
            token.raise_if_cancelled()
            emit(progress, 0.38 + 0.28 * frame_index / max(1, frame_count), f"Summarising dyadic frame {frame_index+1}/{frame_count}")
        frame_start = overlap_start + frame_index * step_ns
        frame_end = min(overlap_end + 1, frame_start + step_ns)
        primary_a = _primary_for_frame(a_groups.get(frame_index, pd.DataFrame(columns=a.columns)))
        primary_b = _primary_for_frame(b_groups.get(frame_index, pd.DataFrame(columns=b.columns)))
        state = _frame_state(primary_a, primary_b, role_a_id, role_b_id)
        frame_rows.append({
            "interaction_frame_index": frame_index,
            "global_start_ns": frame_start,
            "global_end_ns": frame_end,
            "role_a_id": role_a_id,
            "role_b_id": role_b_id,
            "role_a_sample_count": primary_a["sample_count"],
            "role_a_target_entity_id": primary_a["target_entity_id"],
            "role_a_target_domain": primary_a["target_domain"],
            "role_a_target_region": primary_a["target_region"],
            "role_a_confidence": primary_a["confidence"],
            "role_a_review_required": primary_a["review_required"],
            "role_a_distribution_json": json.dumps(primary_a["distribution"], sort_keys=True, separators=(",", ":")),
            "role_b_sample_count": primary_b["sample_count"],
            "role_b_target_entity_id": primary_b["target_entity_id"],
            "role_b_target_domain": primary_b["target_domain"],
            "role_b_target_region": primary_b["target_region"],
            "role_b_confidence": primary_b["confidence"],
            "role_b_review_required": primary_b["review_required"],
            "role_b_distribution_json": json.dumps(primary_b["distribution"], sort_keys=True, separators=(",", ":")),
            **state,
        })
    frames = pd.DataFrame(frame_rows)
    gap_frames = max(0, int(round(short_gap_ms / 1000.0 * canonical_fps)))
    min_frames = max(1, int(math.ceil(minimum_episode_ms / 1000.0 * canonical_fps)))
    mutual_mask = _bridge_boolean(frames["mutual_face_gaze"].to_numpy(dtype=bool), gap_frames)
    mutual_episodes = _episodes(frames, mutual_mask, episode_type="mutual_face_gaze", min_frames=min_frames, step_ns=step_ns)
    joint_episodes = _joint_episodes(
        frames,
        min_frames=min_frames,
        max_gap_frames=gap_frames,
        role_a_samples=a,
        role_b_samples=b,
        role_a=role_a_id,
        role_b=role_b_id,
    )

    artifact_id = new_id("artifact")
    root = store.derived_dir(session_id, "dyadic_attention") / artifact_id
    root.mkdir(parents=True, exist_ok=False)
    role_a_csv = root / f"{role_a_id}_targets_global.csv"
    role_b_csv = root / f"{role_b_id}_targets_global.csv"
    pairs_csv = root / "dyadic_attention_samples.csv"
    frames_csv = root / "dyadic_attention_frames.csv"
    mutual_csv = root / "mutual_gaze_episodes.csv"
    joint_csv = root / "joint_attention_episodes.csv"
    a.to_csv(role_a_csv, index=False)
    b.to_csv(role_b_csv, index=False)
    pairs.to_csv(pairs_csv, index=False)
    frames.to_csv(frames_csv, index=False)
    mutual_episodes.to_csv(mutual_csv, index=False)
    joint_episodes.to_csv(joint_csv, index=False)
    parquet_files: list[str] = []
    if output_parquet:
        try:
            for frame, name in [
                (pairs, "dyadic_attention_samples.parquet"),
                (frames, "dyadic_attention_frames.parquet"),
                (mutual_episodes, "mutual_gaze_episodes.parquet"),
                (joint_episodes, "joint_attention_episodes.parquet"),
            ]:
                path = root / name
                frame.to_parquet(path, index=False)
                parquet_files.append(str(path))
        except Exception:
            parquet_files = []
    qc = {
        "alignment_mode": alignment_mode,
        "timeline_basis": "shared_native_trim_local_time" if alignment_mode == "pretrimmed_shared_start" else "global_time",
        "role_a_rows": int(len(a)),
        "role_b_rows": int(len(b)),
        "mutual_nearest_pair_count": int(len(pairs)),
        "interaction_frame_count": int(len(frames)),
        "overlap_duration_seconds": float((overlap_end - overlap_start) / 1e9),
        "mutual_face_gaze_fraction": float(frames["mutual_face_gaze"].mean()) if len(frames) else 0.0,
        "mutual_eye_contact_fraction": float(frames["mutual_eye_contact"].mean()) if len(frames) else 0.0,
        "joint_attention_fraction": float(frames["joint_attention"].mean()) if len(frames) else 0.0,
        "mutual_gaze_episode_count": int(len(mutual_episodes)),
        "joint_attention_episode_count": int(len(joint_episodes)),
        "review_required_fraction": float(frames["review_required"].mean()) if len(frames) else 0.0,
        "dyadic_state_counts": dict(Counter(frames["dyadic_state"].astype(str))),
    }
    manifest = {
        "schema_version": "oculostudio-dyadic-attention-1.1",
        "artifact_id": artifact_id,
        "timeline_basis": "shared_native_trim_local_time" if alignment_mode == "pretrimmed_shared_start" else "global_time",
        "role_a": {"role_id": role_a_id, "artifact_id": role_a_target_artifact_id, "time_domain_id": role_a_time_domain_id},
        "role_b": {"role_id": role_b_id, "artifact_id": role_b_target_artifact_id, "time_domain_id": role_b_time_domain_id},
        "parameters": {
            "alignment_mode": alignment_mode,
            "canonical_fps": canonical_fps,
            "cross_wearer_tolerance_ms": cross_wearer_tolerance_ms,
            "minimum_episode_ms": minimum_episode_ms,
            "short_gap_ms": short_gap_ms,
        },
        "timeline_overlap_ns": [overlap_start, overlap_end],
        "global_overlap_ns": [overlap_start, overlap_end] if alignment_mode == "timegraph" else None,
        "outputs": {
            "role_a_global_csv": str(role_a_csv),
            "role_b_global_csv": str(role_b_csv),
            "sample_pairs_csv": str(pairs_csv),
            "frames_csv": str(frames_csv),
            "mutual_gaze_episodes_csv": str(mutual_csv),
            "joint_attention_episodes_csv": str(joint_csv),
            "parquet": parquet_files,
        },
        "qc": qc,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "dyadic_attention_manifest.json", manifest)
    atomic_write_json(root / "qc_metrics.json", qc)
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.DYADIC_ATTENTION,
        label=label,
        output_root=root,
        source_asset_ids=list(dict.fromkeys([*artifact_a.source_asset_ids, *artifact_b.source_asset_ids])),
        parent_artifact_ids=[role_a_target_artifact_id, role_b_target_artifact_id],
        parameters=manifest["parameters"],
        qc=qc,
        provenance={
            "role_a_target_artifact_id": role_a_target_artifact_id,
            "role_b_target_artifact_id": role_b_target_artifact_id,
            "role_a_time_domain_id": role_a_time_domain_id,
            "role_b_time_domain_id": role_b_time_domain_id,
            "alignment_mode": alignment_mode,
            "timeline_basis": "shared_native_trim_local_time" if alignment_mode == "pretrimmed_shared_start" else "global_time",
        },
        artifact_id=artifact_id,
    )
    emit(progress, 1.0, "Dyadic attention annotation complete")
    return {
        "artifact": artifact.model_dump(mode="json"),
        "frames_csv": str(frames_csv),
        "mutual_gaze_episodes_csv": str(mutual_csv),
        "joint_attention_episodes_csv": str(joint_csv),
        "manifest": str(root / "dyadic_attention_manifest.json"),
    }
