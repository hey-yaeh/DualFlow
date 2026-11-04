"""Targeted validation for OculoStudio v0.21.71 fixation-aware Body AOI review reduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

import oculostudio
from oculostudio.domain.models import ArtifactKind
from oculostudio.errors import ValidationError
from oculostudio.services.body_aoi import (
    _apply_body_auto_review_reduction,
    _frame_summary,
    reprocess_body_aoi_review_reduction,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate(
    uid: str,
    *,
    fixation: str | None,
    timestamp_ns: int,
    frame_index: int,
    relation: str = "partner",
    entity: str = "R02",
    region: str = "torso",
    angle: float = 0.45,
    confidence: float = 0.62,
    primitive_confidence: float = 0.82,
    candidate_count: int = 1,
    second_angle: float | None = None,
    self_occlusion: bool = False,
    exact: bool = False,
    review_required: bool = True,
) -> dict[str, object]:
    return {
        "gaze_sample_uid": uid,
        "source_gaze_row_index": int(uid.lstrip("g") or 0),
        "timestamp_ns": timestamp_ns,
        "clip_time_ns": timestamp_ns,
        "scene_frame_index": frame_index,
        "native_frame_index": frame_index,
        "scene_timestamp_ns": timestamp_ns,
        "review_preview_available": True,
        "scene_frame_available": True,
        "gaze_x_px": 400.0,
        "gaze_y_px": 260.0,
        "fixation_id": fixation,
        "ray_valid": True,
        "body_entity_id": entity,
        "body_relation": relation,
        "body_raw_region": region,
        "body_coarse_region": region,
        "body_region": region,
        "body_social_region": f"{relation}_{region}",
        "body_confidence": confidence,
        "body_exact_hit": exact,
        "body_distance_mm": 1450.0,
        "body_angular_miss_deg": 0.0 if exact else angle,
        "first_hit_primitive_confidence": primitive_confidence,
        "candidate_hit_count": candidate_count,
        "second_hit_angular_miss_deg": second_angle,
        "assignment_status": (
            f"body_3d_exact_{relation}" if exact
            else f"body_3d_uncertainty_cone_{relation}"
        ),
        "self_occlusion": self_occlusion,
        "review_required": review_required,
        "review_required_before_auto_resolution": review_required,
        "body_geometry_time_mode": "gaze_sample_timestamp",
    }


def _direct_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    # A homogeneous all-candidate fixation: the new policy should clear it.
    for offset in range(8):
        rows.append(_candidate(
            f"g{index}", fixation="stable", timestamp_ns=offset * 5_000_000,
            frame_index=offset // 6, angle=0.45,
        ))
        index += 1
    # A fixation with a target conflict must remain pending.
    for offset, region in enumerate(["torso", "torso", "head", "head"]):
        rows.append(_candidate(
            f"g{index}", fixation="target-conflict",
            timestamp_ns=100_000_000 + offset * 5_000_000,
            frame_index=10, region=region, angle=0.35,
        ))
        index += 1
    # A close competing surface must remain pending.
    for offset in range(4):
        rows.append(_candidate(
            f"g{index}", fixation="competing",
            timestamp_ns=200_000_000 + offset * 5_000_000,
            frame_index=20, angle=0.40, candidate_count=2, second_angle=0.50,
        ))
        index += 1
    # Self occlusion must remain pending.
    for offset in range(3):
        rows.append(_candidate(
            f"g{index}", fixation="self-occluded",
            timestamp_ns=300_000_000 + offset * 5_000_000,
            frame_index=30, angle=0.30, self_occlusion=True,
        ))
        index += 1
    # No fixation ID: a very short, unanimous temporal run may be cleared.
    for offset in range(4):
        rows.append(_candidate(
            f"g{index}", fixation=None,
            timestamp_ns=400_000_000 + offset * 5_000_000,
            frame_index=40, angle=0.30, confidence=0.35,
            primitive_confidence=0.86,
        ))
        index += 1
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    source = _direct_fixture()
    target_columns = [
        "body_relation", "body_entity_id", "body_raw_region",
        "body_coarse_region", "body_region", "body_social_region",
        "body_exact_hit", "body_angular_miss_deg", "body_distance_mm",
    ]
    before_targets = source[target_columns].copy(deep=True)
    reduced, report, table = _apply_body_auto_review_reduction(
        source, "evidence_guided"
    )
    checks["automatic_targets_never_change"] = before_targets.equals(
        reduced[target_columns]
    )
    stable = reduced[reduced["fixation_id"] == "stable"]
    conflict = reduced[reduced["fixation_id"] == "target-conflict"]
    competing = reduced[reduced["fixation_id"] == "competing"]
    occluded = reduced[reduced["fixation_id"] == "self-occluded"]
    temporal = reduced[reduced["fixation_id"].isna()]
    checks["homogeneous_candidate_fixation_is_resolved"] = bool(
        len(stable) == 8
        and not stable["review_required"].any()
        and set(stable["auto_review_resolution_source"])
        == {"same_fixation_candidate_consensus"}
    )
    checks["target_conflict_stays_pending"] = bool(conflict["review_required"].all())
    checks["close_competing_surface_stays_pending"] = bool(competing["review_required"].all())
    checks["self_occlusion_stays_pending"] = bool(occluded["review_required"].all())
    checks["short_unanimous_run_without_fixation_can_resolve"] = bool(
        len(temporal) == 4
        and not temporal["review_required"].any()
        and set(temporal["auto_review_resolution_source"])
        == {"short_temporal_candidate_run"}
    )
    checks["sample_and_review_unit_metrics_are_reported"] = bool(
        report["review_samples_before"] == len(source)
        and report["review_samples_remaining"] == len(conflict) + len(competing) + len(occluded)
        and report["review_units_before"] > report["review_units_remaining"]
        and "review_units" in set(table["scope"])
    )
    checks["audit_evidence_is_written"] = bool(
        stable["auto_review_evidence_sample_count"].eq(8).all()
        and stable["auto_review_evidence_target_share"].notna().all()
        and stable["auto_review_evidence_median_angle_deg"].notna().all()
    )

    # End-to-end review-only reprocessing: source bytes and automatic labels
    # must stay immutable while a child artifact receives the new queue.
    with tempfile.TemporaryDirectory(prefix="oculo_body_review_reduction_") as temp:
        store = ProjectStore.create(
            Path(temp) / "project", "Body review reduction validation",
            session_label="S001",
        )
        session_id = store.project.selected_session_id
        artifact_store = ArtifactStore(store)
        source_root = store.derived_dir(session_id, "body_aoi", "source")
        source_path = source_root / "body_aoi_samples.csv"
        source.to_csv(source_path, index=False)
        _frame_summary(source).to_csv(source_root / "body_aoi_frames.csv", index=False)
        pd.DataFrame([
            {
                "query_index": 0, "scene_frame_index": 0,
                "entity_id": "R02", "relation_to_wearer": "partner",
                "region": "torso", "primitive_type": "capsule",
                "x1_mm": 0.0, "y1_mm": 900.0, "z1_mm": 1500.0,
                "x2_mm": 0.0, "y2_mm": 1450.0, "z2_mm": 1500.0,
                "radius_mm": 185.0,
            }
        ]).to_csv(source_root / "body_regions.csv", index=False)
        (source_root / "body_aoi_timebase.json").write_text(
            json.dumps({"body_geometry_clock": "gaze_sample_timestamp"}),
            encoding="utf-8",
        )
        source_artifact = artifact_store.register(
            session_id=session_id,
            kind=ArtifactKind.BODY_AOI,
            label="S001 I01 Body AOI",
            output_root=source_root,
            parameters={"auto_review_policy": "adaptive"},
            qc={
                "body_geometry_clock": "gaze_sample_timestamp",
                "body_geometry_review_ready": True,
                "body_geometry_fusion_ready": True,
            },
            provenance={"validation_fixture": True},
        )
        source_hash_before = _sha256(source_path)
        result = reprocess_body_aoi_review_reduction(
            store,
            session_id=session_id,
            body_aoi_artifact_id=source_artifact.id,
            auto_review_policy="evidence_guided",
            label="S001 I01 Body AOI · reduced review",
        )
        refreshed = store.snapshot().sessions[session_id]
        child = refreshed.artifacts[result["artifact"]["id"]]
        child_samples_path = next(
            Path(row.path) for row in child.files
            if Path(row.path).name == "body_aoi_samples.csv"
        )
        child_samples = pd.read_csv(child_samples_path, low_memory=False)
        checks["review_only_reprocessing_preserves_source_bytes"] = (
            source_hash_before == _sha256(source_path)
        )
        checks["review_only_reprocessing_creates_child_artifact"] = bool(
            child.id != source_artifact.id
            and child.parent_artifact_ids == [source_artifact.id]
            and child.qc.get("review_reduction_reprocessed") is True
            and child.provenance.get("automatic_body_targets_unchanged") is True
        )
        checks["review_only_reprocessing_preserves_targets"] = bool(
            source[target_columns].reset_index(drop=True).equals(
                child_samples[target_columns].reset_index(drop=True)
            )
        )
        checks["review_only_reprocessing_reduces_units"] = bool(
            result["review_units_before"] > result["review_units_remaining"]
            and child.qc.get("review_units_remaining")
            == result["review_units_remaining"]
        )

        draft_path = (
            store.root / "annotations" / session_id / "body_aoi"
            / source_artifact.id / "manual_review.json"
        )
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(json.dumps({
            "schema_version": "oculostudio-body-aoi-manual-review-1.0",
            "session_id": session_id,
            "body_aoi_artifact_id": source_artifact.id,
            "decisions": {"g0": {"action": "accept_auto"}},
        }), encoding="utf-8")
        try:
            reprocess_body_aoi_review_reduction(
                store,
                session_id=session_id,
                body_aoi_artifact_id=source_artifact.id,
                auto_review_policy="evidence_guided",
            )
        except ValidationError as exc:
            checks["saved_manual_decisions_are_never_silently_discarded"] = (
                "saved manual-review decisions" in str(exc)
            )
        else:
            checks["saved_manual_decisions_are_never_silently_discarded"] = False

    checks["installed_version"] = oculostudio.__version__ == "0.21.71"
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "release": "OculoStudio v0.21.71",
        "feature": "Evidence-guided fixation-level Body AOI review reduction",
        "checks": checks,
        "direct_report": report,
        "passed": len(checks) - len(failed),
        "failed": failed,
        "ok": not failed,
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
