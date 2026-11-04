"""Targeted v0.21.71 temporal-consensus automatic review reduction validation."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

import oculostudio
from oculostudio.server import FaceAOIJobRequest
from oculostudio.services.face_aoi import (
    FACE_AOI_ENGINE,
    _apply_auto_review_reduction,
    create_face_aoi_annotations,
)


def base_row(**updates):
    row = {
        "timestamp_ns": 0,
        "native_frame_index": 0,
        "fixation_id": "f0",
        "face_track_id": 7,
        "raw_face_region": "mouth",
        "face_region": "mouth",
        "face_entity_id": "R02",
        "target_relation": "partner",
        "identity_status": "matched_participant",
        "identity_confidence": 0.9,
        "face_aoi_confidence": 0.9,
        "include_in_primary_analysis": True,
        "review_required": False,
        "fm_on_face_match_count": 1,
        "assignment_status": "participant_face_aoi_hit",
        "candidate_entity_id": "R02",
        "candidate_relation": "partner",
        "identity_geometry_strong": True,
        "identity_projected_center_inside_face": True,
        "identity_face_center_inside_projection": True,
        "identity_temporal_track_consistency": True,
        "identity_normalized_center_error": 0.3,
        "identity_projected_box_iou": 0.2,
        "identity_duplicate_iou": 0.8,
    }
    row.update(updates)
    return row


def main() -> int:
    checks: dict[str, bool] = {}

    # Any exact gaze centre inside an identity-resolved green FaceMapper box
    # is accepted directly, even when the upstream association retained a
    # low-confidence review flag. Edge-band hits are not covered by this rule.
    direct = base_row(
        review_required=True,
        green_face_exact_hit=True,
        identity_confidence=0.22,
        face_aoi_confidence=0.48,
        assignment_status="participant_face_aoi_hit",
    )
    direct_out, direct_report, _ = _apply_auto_review_reduction(pd.DataFrame([direct]), "adaptive")
    checks["direct_green_box_hit_auto_accepted"] = bool(
        not direct_out.loc[0, "review_required"]
        and direct_out.loc[0, "auto_review_resolution_source"] == "direct_green_box_matched_identity"
        and direct_out.loc[0, "face_entity_id"] == "R02"
        and direct_out.loc[0, "face_region"] == "mouth"
        and direct_report["review_samples_remaining"] == 0
    )
    edge = dict(direct)
    edge["green_face_exact_hit"] = False
    edge["green_face_edge_tolerance_hit"] = True
    edge["identity_status"] = "low_confidence_participant_candidate"
    edge["target_relation"] = "ambiguous_partner"
    edge["face_entity_id"] = None
    edge_out, _, _ = _apply_auto_review_reduction(pd.DataFrame([edge]), "adaptive")
    checks["ambiguous_edge_hit_remains_reviewable"] = bool(edge_out.loc[0, "review_required"])

    # Upstream temporal identity evidence can be auto-accepted, but a
    # contradictory candidate must remain for human review.
    identity = base_row(
        review_required=True,
        identity_status="temporal_bridge_matched_participant",
        identity_confidence=0.65,
        face_aoi_confidence=0.75,
        identity_geometry_strong=False,
        identity_projected_center_inside_face=False,
        identity_face_center_inside_projection=False,
        identity_temporal_track_consistency=True,
    )
    resolved, identity_report, _ = _apply_auto_review_reduction(pd.DataFrame([identity]), "conservative")
    checks["strong_temporal_identity_auto_resolved"] = bool(
        not resolved.loc[0, "review_required"]
        and resolved.loc[0, "face_entity_id"] == "R02"
        and resolved.loc[0, "face_social_region"] == "partner_mouth"
        and resolved.loc[0, "auto_review_resolution_source"] == "identity_temporal_bridge_matched_participant"
    )
    conflict = dict(identity)
    conflict["candidate_entity_id"] = "R03"
    conflicted, _, _ = _apply_auto_review_reduction(pd.DataFrame([conflict]), "conservative")
    checks["conflicting_identity_not_auto_resolved"] = bool(conflicted.loc[0, "review_required"])

    # Same-track bracketing preserves a real fine region while repairing only
    # the missing identity decision.
    track_rows = [
        base_row(timestamp_ns=0, native_frame_index=10, fixation_id="f1", face_region="left_eye", raw_face_region="left_eye"),
        base_row(
            timestamp_ns=10_000_000,
            native_frame_index=11,
            fixation_id="f1",
            raw_face_region="mouth",
            face_region="ambiguous_face_identity",
            face_entity_id=None,
            target_relation="ambiguous_partner",
            identity_status="low_confidence_participant_candidate",
            identity_confidence=0.2,
            face_aoi_confidence=0.2,
            include_in_primary_analysis=False,
            review_required=True,
            assignment_status="low_confidence_participant_candidate",
        ),
        base_row(timestamp_ns=20_000_000, native_frame_index=12, fixation_id="f1", face_region="mouth", raw_face_region="mouth"),
    ]
    track, track_report, _ = _apply_auto_review_reduction(pd.DataFrame(track_rows), "conservative")
    checks["same_track_bridge_preserves_fine_region"] = bool(
        not track.loc[1, "review_required"]
        and track.loc[1, "face_entity_id"] == "R02"
        and track.loc[1, "face_region"] == "mouth"
        and track.loc[1, "auto_review_resolution_source"] == "same_track_temporal_bridge"
    )

    # Missing direct green-box geometry is never fabricated from fixation
    # context. Temporal consensus can resolve identity for a direct geometry
    # hit, but it cannot convert an outside/unknown sample into a face hit.
    fixation_rows = [
        base_row(timestamp_ns=0, native_frame_index=20, fixation_id="f2", face_track_id=None, face_region="left_eye", raw_face_region="left_eye"),
        base_row(timestamp_ns=10_000_000, native_frame_index=21, fixation_id="f2", face_track_id=None, face_region="left_eye", raw_face_region="left_eye"),
        base_row(
            timestamp_ns=20_000_000,
            native_frame_index=22,
            fixation_id="f2",
            face_track_id=None,
            raw_face_region=None,
            face_region="face",
            face_entity_id=None,
            target_relation=None,
            identity_status="missing_identity_row",
            identity_confidence=0.0,
            face_aoi_confidence=0.55,
            include_in_primary_analysis=False,
            review_required=True,
            assignment_status="on_face_without_fine_aoi",
        ),
        base_row(timestamp_ns=30_000_000, native_frame_index=23, fixation_id="f2", face_track_id=None, face_region="left_eye", raw_face_region="left_eye"),
        base_row(timestamp_ns=40_000_000, native_frame_index=24, fixation_id="f2", face_track_id=None, face_region="left_eye", raw_face_region="left_eye"),
    ]
    fixation, fixation_report, table = _apply_auto_review_reduction(pd.DataFrame(fixation_rows), "conservative")
    checks["fixation_consensus_does_not_fabricate_missing_face_geometry"] = bool(
        fixation.loc[2, "review_required"]
        and pd.isna(fixation.loc[2, "face_entity_id"])
        and fixation.loc[2, "face_region"] == "face"
        and not bool(fixation.loc[2, "auto_review_resolved"])
        and str(fixation.loc[2, "auto_review_resolution_source"]) == ""
    )

    # Invalid gaze and policy-off behavior must never be promoted.
    invalid = base_row(
        review_required=True,
        face_region="invalid_gaze",
        raw_face_region=None,
        face_entity_id=None,
        target_relation=None,
        identity_status=None,
        include_in_primary_analysis=False,
        assignment_status="not_worn",
    )
    invalid_out, _, _ = _apply_auto_review_reduction(pd.DataFrame([invalid]), "balanced")
    checks["invalid_gaze_never_promoted"] = bool(invalid_out.loc[0, "review_required"])
    off_out, off_report, _ = _apply_auto_review_reduction(pd.DataFrame(track_rows), "off")
    checks["off_policy_preserves_all_review_flags"] = bool(
        off_out.loc[1, "review_required"] and off_report["auto_resolved_samples"] == 0
    )

    # Audit fields, report arithmetic and API/UI wiring.
    checks["audit_columns_present"] = all(
        column in fixation.columns
        for column in (
            "review_required_before_auto_resolution",
            "auto_review_resolved",
            "auto_review_resolution_source",
            "auto_review_confidence",
            "auto_review_policy",
        )
    )
    checks["report_arithmetic_consistent"] = bool(
        fixation_report["review_samples_before"] == 1
        and fixation_report["auto_resolved_samples"] == 0
        and fixation_report["review_samples_remaining"] == 1
        and fixation_report["sample_review_reduction_fraction"] == 0.0
        and fixation_report["remaining_review_reason_counts"] == {"on_face_without_fine_aoi": 1}
        and not table.empty
    )
    request = FaceAOIJobRequest(
        session_id="S001",
        native_trim_artifact_id="trim",
        auto_review_policy="balanced",
    )
    checks["api_accepts_policy"] = request.auto_review_policy == "balanced"
    try:
        FaceAOIJobRequest(
            session_id="S001",
            native_trim_artifact_id="trim",
            auto_review_policy="unsafe",  # type: ignore[arg-type]
        )
        checks["api_rejects_unknown_policy"] = False
    except PydanticValidationError:
        checks["api_rejects_unknown_policy"] = True

    app_js = (Path(oculostudio.__file__).parent / "static" / "app.js").read_text("utf-8")
    checks["ui_exposes_four_policies"] = all(
        token in app_js
        for token in (
            'id="face-aoi-auto-review"',
            'value="adaptive" selected',
            'value="conservative"',
            'value="balanced"',
            'value="off"',
            "auto_review_policy:$('#face-aoi-auto-review')",
            "Adaptive evidence-fusion auto labelling",
        )
    )
    service_source = inspect.getsource(create_face_aoi_annotations)
    checks["artifact_exports_auto_review_audit"] = all(
        token in service_source
        for token in (
            "face_aoi_auto_review_summary.csv",
            "face_aoi_auto_review_summary.json",
            "auto_review_sample_reduction_fraction",
            "review_samples_remaining",
        )
    )
    checks["version_and_engine_updated"] = bool(
        oculostudio.__version__ == "0.21.71"
        and FACE_AOI_ENGINE == "track-aware-gaze-timestamp-face-geometry-v14"
    )

    report = {
        "application_version": oculostudio.__version__,
        "feature": "Direct exact-green-box acceptance plus temporal identity consensus",
        "checks": checks,
        "all_passed": all(checks.values()),
        "examples": {
            "identity_report": identity_report,
            "track_report": track_report,
            "fixation_report": fixation_report,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
