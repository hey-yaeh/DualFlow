"""Create role-specific Native-reference trims from a verified Vicon C3D window.

The normal interaction workflow should not ask a user to type NRD start/end
values when the interaction body C3Ds already preserve the authoritative Vicon
frame range.  This module derives one common dyadic Vicon interval, inverts each
role's accepted Pupil↔Vicon affine TimeGraph, resolves the exact NRD scene-video
frame interval through the companion ``.time`` clock, and then reuses the
existing immutable Native-reference trim exporter.

No ego-video pixel is used to estimate synchronization and no source MP4 is
rewritten.  The resulting Native-reference trim is a virtual video interval plus
trimmed TSD/FaceMapper tables and an exact frame map.  Downstream Privacy can
optionally encode a physical clip later.
"""
from __future__ import annotations

import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from oculostudio import __version__
from oculostudio.adapters.mocap import read_mocap
from oculostudio.adapters.native_reference_clock import build_clock, source_stamp
from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.timegraph import apply_time_map, invert_time_map
from oculostudio.errors import ValidationError
from oculostudio.services.native_reference_trim import (
    SCHEMA as NATIVE_TRIM_SCHEMA,
    TablePass,
    Window,
    _assets,
    _public_preview,
    export_native_reference_trim,
    fmt_ns,
    resolve_sources,
)
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import CancellationToken, atomic_write_json, emit, new_id, sha256_file, utc_now


ENGINE = "c3d_defined_pupil_clip_v1"
PAIR_SCHEMA = "oculostudio-c3d-defined-clip-set-1.0"


def _role(value: str) -> str:
    result = str(value or "").strip().upper()
    if result not in {"R01", "R02"}:
        raise ValidationError(f"Unsupported wearer role: {value!r}")
    return result


def _handoff_rows(session, handoff_artifact_id: str) -> tuple[Any, dict[str, dict[str, Any]]]:
    artifact = session.artifacts.get(handoff_artifact_id)
    if artifact is None or artifact.kind != ArtifactKind.ETM_ROLE_ASSIGNMENT:
        raise ValidationError("Select the synchronized Calibration handoff")
    if str((artifact.parameters or {}).get("calibration_phase") or "") != "handoff":
        raise ValidationError("The selected artifact is a wearer assignment, not the synchronized Calibration handoff")
    if not bool((artifact.qc or {}).get("handoff_complete")):
        raise ValidationError("Calibration handoff is incomplete")
    if not bool((artifact.qc or {}).get("time_sync_complete")):
        raise ValidationError("Calibration handoff does not contain both verified role-specific TimeGraph mappings")
    rows = dict((artifact.qc or {}).get("assignments_summary") or {})
    missing = [role for role in ("R01", "R02") if role not in rows]
    if missing:
        raise ValidationError(f"Calibration handoff is missing role rows: {missing}")
    return artifact, rows


def _body_interval(asset) -> dict[str, Any]:
    data = read_mocap(asset.source_path)
    if data.frame_count < 2 or not math.isfinite(float(data.frame_rate)) or float(data.frame_rate) <= 0:
        raise ValidationError(f"Body motion {asset.label!r} has no valid C3D time range")
    start_ns = int(round(float(data.time_sec[0]) * 1e9))
    # The retained interval is half-open.  Include the presentation interval of
    # the final C3D sample instead of ending exactly on the last sample centre.
    end_ns = int(round((float(data.time_sec[-1]) + 1.0 / float(data.frame_rate)) * 1e9))
    if end_ns <= start_ns:
        raise ValidationError(f"Body motion {asset.label!r} has a non-positive time range")
    return {
        "asset_id": asset.id,
        "asset_label": asset.label,
        "source_path": asset.source_path,
        "start_vicon_ns": start_ns,
        "end_vicon_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "first_frame": int(data.first_frame),
        "last_frame_inclusive": int(data.first_frame + data.frame_count - 1),
        "frame_count": int(data.frame_count),
        "frame_rate_hz": float(data.frame_rate),
        "reader_backend": (data.metadata or {}).get("reader_backend"),
        "frame_count_source": (data.metadata or {}).get("frame_count_source"),
    }


def _common_body_window(session, rows: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_role: dict[str, dict[str, Any]] = {}
    for role in ("R01", "R02"):
        body_id = str(rows[role].get("body_asset_id") or "")
        asset = session.assets.get(body_id)
        if asset is None or asset.kind not in {"c3d", "mocap_npz"}:
            raise ValidationError(f"Calibration handoff {role} body motion is unavailable")
        per_role[role] = _body_interval(asset)
    start_ns = max(row["start_vicon_ns"] for row in per_role.values())
    end_ns = min(row["end_vicon_ns"] for row in per_role.values())
    if end_ns <= start_ns:
        raise ValidationError("R01 and R02 body C3Ds have no common Vicon interval")
    individual_duration = min(row["duration_ns"] for row in per_role.values())
    overlap_fraction = float((end_ns - start_ns) / max(1, individual_duration))
    if overlap_fraction < 0.90:
        raise ValidationError(
            f"R01/R02 body C3D common overlap is only {100.0*overlap_fraction:.1f}%; verify that both files describe the same interaction"
        )
    return {
        "source": "common_body_c3d_overlap",
        "start_vicon_ns": int(start_ns),
        "end_vicon_ns": int(end_ns),
        "duration_ns": int(end_ns - start_ns),
        "duration_sec": float((end_ns - start_ns) / 1e9),
        "overlap_fraction_of_shorter_body_recording": overlap_fraction,
    }, per_role


def _sync_for_role(session, handoff, row: dict[str, Any], role: str, timeseries_asset_id: str) -> tuple[Any, Any, Any, Any]:
    sync_id = str(row.get("recommended_time_sync_artifact_id") or "")
    map_id = str(row.get("recommended_time_map_id") or "")
    sync = session.artifacts.get(sync_id)
    mapping = session.time_maps.get(map_id)
    timeseries = session.assets.get(timeseries_asset_id)
    if sync is None or sync.kind != ArtifactKind.TIME_SYNC:
        raise ValidationError(f"Calibration handoff has no accepted {role} synchronization artifact")
    if mapping is None:
        raise ValidationError(f"Calibration handoff {role} TimeGraph mapping is unavailable")
    if timeseries is None or timeseries.kind != "pupil_cloud":
        raise ValidationError(f"Select the {role} Pupil Timeseries asset")
    expected_source = str((sync.parameters or {}).get("source_asset_id") or "")
    if expected_source and expected_source != timeseries.id:
        raise ValidationError(
            f"{role} Timeseries does not match the source used by the accepted synchronization ({timeseries.label!r})"
        )
    etm_id = str(row.get("etm_asset_id") or "")
    etm = session.assets.get(etm_id)
    if etm is None:
        raise ValidationError(f"Calibration handoff {role} ETM tracker is unavailable")
    if mapping.source_domain_id != timeseries.primary_time_domain_id:
        raise ValidationError(f"{role} TimeGraph source domain does not match the selected Pupil Timeseries")
    if mapping.target_domain_id != etm.primary_time_domain_id:
        raise ValidationError(f"{role} TimeGraph target domain does not match the assigned ETM C3D")
    qc = sync.qc or {}
    if not bool(qc.get("production_ready")) or not bool(qc.get("precision_validated")):
        raise ValidationError(f"{role} automatic synchronization is not production-ready under the precision audit")
    if bool(qc.get("ego_video_used")):
        raise ValidationError(f"{role} synchronization unexpectedly reports ego-video input")
    return timeseries, sync, mapping, etm


def _validate_etm_clock_coverage(etm, common_window: dict[str, Any]) -> dict[str, Any]:
    interval = _body_interval(etm)
    tolerance_ns = int(round(1e9 / max(float(interval["frame_rate_hz"]), 1.0)))
    start_gap = max(0, int(interval["start_vicon_ns"]) - int(common_window["start_vicon_ns"]))
    end_gap = max(0, int(common_window["end_vicon_ns"]) - int(interval["end_vicon_ns"]))
    if start_gap > tolerance_ns or end_gap > tolerance_ns:
        raise ValidationError(
            f"Assigned ETM C3D {etm.label!r} does not cover the common body-C3D Vicon interval "
            f"(start gap {start_gap/1e6:.3f} ms, end gap {end_gap/1e6:.3f} ms)"
        )
    return {**interval, "start_coverage_gap_ns": start_gap, "end_coverage_gap_ns": end_gap}


def _clock_media_window(clock, pupil_start_ns: int, pupil_end_ns: int, *, boundary_tolerance_ms: float) -> dict[str, Any]:
    tolerance_ns = int(round(float(boundary_tolerance_ms) * 1e6))
    capture_start = int(clock.capture_ns[0])
    capture_end = int(clock.end_utc_ns)
    requested_start = int(pupil_start_ns)
    requested_end = int(pupil_end_ns)
    start_clamp_ns = 0
    end_clamp_ns = 0
    if requested_start < capture_start:
        start_clamp_ns = capture_start - requested_start
        if start_clamp_ns > tolerance_ns:
            raise ValidationError(
                f"C3D-defined Pupil start precedes the NRD scene-video clock by {start_clamp_ns/1e6:.3f} ms"
            )
        pupil_start_ns = capture_start
    if requested_end > capture_end:
        end_clamp_ns = requested_end - capture_end
        if end_clamp_ns > tolerance_ns:
            raise ValidationError(
                f"C3D-defined Pupil end exceeds the NRD scene-video clock by {end_clamp_ns/1e6:.3f} ms"
            )
        pupil_end_ns = capture_end
    if pupil_end_ns <= pupil_start_ns:
        raise ValidationError("Mapped Pupil interval is empty after NRD coverage validation")
    start_media_ns = int(clock.media_at(int(pupil_start_ns)))
    end_media_ns = int(clock.media_at(int(pupil_end_ns)))
    if end_media_ns <= start_media_ns:
        raise ValidationError("Mapped NRD media interval is empty")
    start_frame, start_error = clock.nearest_capture(int(pupil_start_ns))
    end_frame, end_error = clock.nearest_capture(int(pupil_end_ns))
    return {
        "requested_pupil_start_ns": requested_start,
        "requested_pupil_end_ns": requested_end,
        "pupil_start_ns": int(pupil_start_ns),
        "pupil_end_ns": int(pupil_end_ns),
        "start_media_ns": start_media_ns,
        "end_media_ns": end_media_ns,
        "duration_media_ns": end_media_ns - start_media_ns,
        "native_first_frame_nearest": int(start_frame),
        "native_end_frame_nearest": int(end_frame),
        "start_boundary_error_ns": int(start_error),
        "end_boundary_error_ns": int(end_error),
        "start_coverage_clamp_ns": int(start_clamp_ns),
        "end_coverage_clamp_ns": int(end_clamp_ns),
        "native_clock_start_ns": capture_start,
        "native_clock_end_ns": capture_end,
    }


def _exact_time_interpretation(start_ns: int, duration_ns: int) -> dict[str, Any]:
    return {
        "schema_version": "oculostudio-video-time-input-1.0",
        "start": {"nanoseconds": str(int(start_ns)), "normalized_time": fmt_ns(int(start_ns))},
        "duration": {"nanoseconds": str(int(duration_ns)), "normalized_time": fmt_ns(int(duration_ns))},
        "end": {"nanoseconds": str(int(start_ns + duration_ns)), "normalized_time": fmt_ns(int(start_ns + duration_ns))},
        "timecode_fps": 30,
        "uses_frame_timecode": False,
        "source": "exact_inverse_timegraph",
    }


def _create_exact_plan(
    store: ProjectStore,
    *,
    session_id: str,
    role: str,
    native_asset_id: str,
    timeseries_asset_id: str,
    facemapper_asset_id: str,
    native_video_member: str | None,
    label: str,
    interval_policy: str,
    face_tolerance_ms: float,
    ffprobe_path: str | None,
    pair_id: str,
    handoff,
    row: dict[str, Any],
    body_window: dict[str, Any],
    body_intervals: dict[str, dict[str, Any]],
    sync,
    mapping,
    pupil_window: dict[str, Any],
    boundary_tolerance_ms: float,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    records = _assets(store, session_id, native_asset_id, timeseries_asset_id, facemapper_asset_id)
    sources = {key: asset.source_path for key, asset in records.items()}
    stamps = {key: source_stamp(path) for key, path in sources.items()}
    emit(progress, 0.02, f"{role}: matching NRD, TSD and FaceMapper by recording id")
    selection = resolve_sources(sources["native"], sources["tsd"], sources["face_mapper"], native_video_member)
    clock = build_clock(
        sources["native"],
        store.cache_dir(),
        video_member=selection["native"]["video_member"],
        ffprobe_path=ffprobe_path,
        cancellation=token,
        progress=progress,
    )
    mapped = _clock_media_window(
        clock,
        int(pupil_window["start_pupil_ns"]),
        int(pupil_window["end_pupil_ns"]),
        boundary_tolerance_ms=boundary_tolerance_ms,
    )
    window = Window.create(
        clock,
        int(mapped["start_media_ns"]),
        int(mapped["duration_media_ns"]),
        float(face_tolerance_ms),
        interval_policy,
    )
    report = TablePass(window, selection["recording_id"], None, token, progress).run(sources, selection)
    if any(source_stamp(sources[key]) != stamps[key] for key in stamps):
        raise ValidationError(f"{role} input changed during automatic trim planning")
    preview_id = new_id("nativeplan")
    root = store.cache_dir() / "native_trim_plans"
    root.mkdir(exist_ok=True)
    clock_summary = clock.metadata.copy()
    interpretation = _exact_time_interpretation(mapped["start_media_ns"], mapped["duration_media_ns"])
    request = {
        "label": label,
        "start_timecode": fmt_ns(mapped["start_media_ns"]),
        "duration_timecode": fmt_ns(mapped["duration_media_ns"]),
        "timecode_fps": 30,
        "timecode_interpretation": interpretation,
        "interval_policy": interval_policy,
        "face_tolerance_ms": float(face_tolerance_ms),
        "native_asset_id": native_asset_id,
        "timeseries_asset_id": timeseries_asset_id,
        "facemapper_asset_id": facemapper_asset_id,
        "native_video_member": selection["native"]["video_member"],
        "ffprobe_path": ffprobe_path,
        "window_source": "common_body_c3d_overlap",
        "automatic_c3d_defined": True,
        "wearer_role_id": role,
        "paired_trim_set_id": pair_id,
        "calibration_handoff_artifact_id": handoff.id,
        "body_asset_id": str(row.get("body_asset_id") or ""),
        "assigned_etm_asset_id": str(row.get("etm_asset_id") or ""),
        "time_sync_artifact_id": sync.id,
        "time_map_id": mapping.id,
        "vicon_window": {key: str(value) if key.endswith("_ns") else value for key, value in body_window.items()},
        "pupil_window": {key: str(value) if key.endswith("_ns") else value for key, value in pupil_window.items()},
        "native_window": {key: str(value) if key.endswith("_ns") else value for key, value in mapped.items()},
        "body_intervals": {
            key: {field: str(value) if field.endswith("_ns") else value for field, value in value.items()}
            for key, value in body_intervals.items()
        },
        "boundary_tolerance_ms": float(boundary_tolerance_ms),
        "parent_artifact_ids": [handoff.id, sync.id],
        "additional_source_asset_ids": [str(row.get("body_asset_id") or ""), str(row.get("etm_asset_id") or "")],
        "provenance_extra": {
            "trim_window_source": "common R01/R02 body C3D overlap",
            "time_conversion": "inverse accepted affine Pupil↔assigned-ETM TimeGraph",
            "frame_conversion": "NRD companion .time + actual MP4 PTS",
            "ego_video_used_for_synchronization": False,
            "source_mp4_reencoded": False,
        },
    }
    plan = {
        "schema_version": NATIVE_TRIM_SCHEMA,
        "preview_id": preview_id,
        "session_id": session_id,
        "project_id": store.project.id,
        "created_at": utc_now(),
        "request": request,
        "recording_id": selection["recording_id"],
        "sources": sources,
        "source_stamps": stamps,
        "asset_ids": {key: asset.id for key, asset in records.items()},
        "selection": selection,
        "clock_index": str(clock.index_path),
        "clock_summary": clock_summary,
        "clock_index_sha256": sha256_file(clock.index_path),
        "window": {
            "start_media_ns": str(window.start),
            "end_media_ns": str(window.end),
            "utc_start_ns": str(window.utc_start),
            "utc_end_ns": str(window.utc_end),
        },
        "report": report,
    }
    atomic_write_json(root / f"{preview_id}.json", plan)
    public = _public_preview(plan)
    public.update(
        {
            "wearer_role_id": role,
            "window_source": request["window_source"],
            "paired_trim_set_id": pair_id,
            "vicon_start": fmt_ns(body_window["start_vicon_ns"]),
            "vicon_end": fmt_ns(body_window["end_vicon_ns"]),
            "pupil_start_ns": str(pupil_window["start_pupil_ns"]),
            "pupil_end_ns": str(pupil_window["end_pupil_ns"]),
            "start_boundary_error_ms": mapped["start_boundary_error_ns"] / 1e6,
            "end_boundary_error_ms": mapped["end_boundary_error_ns"] / 1e6,
            "native_first_frame_nearest": mapped["native_first_frame_nearest"],
            "native_end_frame_nearest": mapped["native_end_frame_nearest"],
        }
    )
    return {"plan": plan, "preview": public, "mapped": mapped}


def _rollback_artifacts(store: ProjectStore, session_id: str, artifact_ids: list[str]) -> None:
    if not artifact_ids:
        return
    paths: list[Path] = []
    session = store.project.sessions[session_id]
    for artifact_id in artifact_ids:
        artifact = session.artifacts.get(artifact_id)
        if artifact:
            for row in artifact.files:
                path = Path(row.path)
                paths.append(path if path.is_dir() else path.parent)
    def mutate(project):
        target = project.sessions[session_id]
        for artifact_id in artifact_ids:
            target.artifacts.pop(artifact_id, None)
    store.update(mutate)
    for path in sorted(set(paths), key=lambda value: len(value.parts), reverse=True):
        try:
            root = path
            # Registered trim files all share one artifact directory.  Do not
            # climb above the explicit directory returned by file inventory.
            shutil.rmtree(root, ignore_errors=True)
        except Exception:
            pass


def _pair_report_html(payload: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(role)}</td><td>{html.escape(row['recording_id'])}</td>"
        f"<td>{html.escape(row['native_start'])}</td><td>{html.escape(row['native_end'])}</td>"
        f"<td>{row['start_boundary_error_ms']:.3f} ms</td><td>{row['end_boundary_error_ms']:.3f} ms</td></tr>"
        for role, row in sorted(payload["roles"].items())
    )
    return f"""<!doctype html><html><meta charset='utf-8'><title>C3D-defined aligned clips</title>
<style>body{{font:15px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#2b2e20}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7d8c8;padding:9px;text-align:left}}th{{background:#eef2d5}}code{{background:#f1f1e9;padding:2px 5px}}</style>
<h1>OculoStudio v{__version__} — C3D-defined aligned clips</h1>
<p>The common body-C3D interval was mapped independently to each Pupil clock through the accepted affine TimeGraph, then to exact NRD frame PTS through the companion <code>.time</code> index. No start/end value was typed and no source MP4 was re-encoded.</p>
<p><b>Vicon interval:</b> {fmt_ns(int(payload['common_vicon_window']['start_vicon_ns']))} → {fmt_ns(int(payload['common_vicon_window']['end_vicon_ns']))} · {payload['common_vicon_window']['duration_sec']:.6f} s</p>
<table><thead><tr><th>Role</th><th>Pupil recording</th><th>NRD media start</th><th>NRD media end</th><th>Start frame error</th><th>End frame error</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Interpretation</h2><ul><li>Each role keeps the same physical Vicon interval, even though the two Pupil absolute timestamps differ.</li><li>The trim artifacts are virtual NRD intervals plus TSD/FaceMapper subsets and exact frame maps.</li><li>Use the resulting role-specific trim artifact in Gaze 3D, Privacy, Face AOI and Body AOI.</li></ul>
</html>"""


def create_c3d_defined_aligned_clips(
    store: ProjectStore,
    *,
    session_id: str,
    calibration_handoff_artifact_id: str,
    role_inputs: dict[str, dict[str, Any]],
    interval_policy: str = "clip",
    face_tolerance_ms: float = 5.0,
    boundary_tolerance_ms: float = 20.0,
    label_prefix: str = "C3D-defined aligned clip",
    ffprobe_path: str | None = None,
    progress=None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    token = cancellation or CancellationToken.create()
    normalized_role_inputs = {
        _role(key): dict(value or {})
        for key, value in dict(role_inputs or {}).items()
        if str(key or "").strip()
    }
    if interval_policy not in {"clip", "inside"}:
        raise ValidationError("Event policy must be clip or inside")
    if not (0 < float(face_tolerance_ms) <= 15):
        raise ValidationError("Face-frame tolerance must be in (0, 15] ms")
    if not (0 <= float(boundary_tolerance_ms) <= 100):
        raise ValidationError("Boundary coverage tolerance must be between 0 and 100 ms")
    session = store.project.sessions.get(session_id)
    if session is None:
        raise ValidationError("Unknown session")
    handoff, rows = _handoff_rows(session, calibration_handoff_artifact_id)
    common_window, body_intervals = _common_body_window(session, rows)
    pair_id = new_id("clipset")
    previews: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    created_artifact_ids: list[str] = []
    try:
        for index, role in enumerate(("R01", "R02")):
            token.raise_if_cancelled()
            supplied = dict(normalized_role_inputs.get(role) or {})
            native_id = str(supplied.get("native_asset_id") or "")
            timeseries_id = str(supplied.get("timeseries_asset_id") or "")
            facemapper_id = str(supplied.get("facemapper_asset_id") or "")
            if not all((native_id, timeseries_id, facemapper_id)):
                raise ValidationError(f"Select {role} NRD, TSD and FaceMapper inputs")
            timeseries, sync, mapping, etm = _sync_for_role(session, handoff, rows[role], role, timeseries_id)
            etm_interval = _validate_etm_clock_coverage(etm, common_window)
            inverse = invert_time_map(mapping)
            start_pupil_ns = int(apply_time_map(inverse, int(common_window["start_vicon_ns"])))
            end_pupil_ns = int(apply_time_map(inverse, int(common_window["end_vicon_ns"])))
            if end_pupil_ns <= start_pupil_ns:
                raise ValidationError(f"{role} inverse TimeGraph produced a non-positive Pupil interval")
            pupil_window = {
                "start_pupil_ns": start_pupil_ns,
                "end_pupil_ns": end_pupil_ns,
                "duration_pupil_ns": end_pupil_ns - start_pupil_ns,
                "time_map_scale": float(mapping.scale),
                "time_map_method": mapping.method,
                "estimated_drift_ppm": float((mapping.scale - 1.0) * 1e6),
                "source_timeseries_asset_id": timeseries.id,
                "target_etm_asset_id": str(rows[role].get("etm_asset_id") or ""),
                "target_etm_interval": {
                    key: str(value) if key.endswith("_ns") else value
                    for key, value in etm_interval.items()
                },
            }
            role_progress = (lambda f, s, base=index: emit(progress, 0.02 + 0.38 * base + 0.36 * float(f), s))
            planned = _create_exact_plan(
                store,
                session_id=session_id,
                role=role,
                native_asset_id=native_id,
                timeseries_asset_id=timeseries_id,
                facemapper_asset_id=facemapper_id,
                native_video_member=supplied.get("native_video_member") or None,
                label=f"{label_prefix} · {role}",
                interval_policy=interval_policy,
                face_tolerance_ms=face_tolerance_ms,
                ffprobe_path=ffprobe_path,
                pair_id=pair_id,
                handoff=handoff,
                row=rows[role],
                body_window=common_window,
                body_intervals=body_intervals,
                sync=sync,
                mapping=mapping,
                pupil_window=pupil_window,
                boundary_tolerance_ms=boundary_tolerance_ms,
                progress=role_progress,
                cancellation=token,
            )
            previews[role] = planned
        emit(progress, 0.78, "Both role windows verified; publishing immutable trim artifacts")
        # Publish only after both dry-runs succeeded.  A second-export failure is
        # rolled back so the pair never appears half-complete.
        for index, role in enumerate(("R01", "R02")):
            token.raise_if_cancelled()
            result = export_native_reference_trim(
                store,
                session_id=session_id,
                preview_id=previews[role]["plan"]["preview_id"],
                progress=lambda f, s, base=index: emit(progress, 0.78 + 0.08 * base + 0.07 * float(f), f"{role}: {s}"),
                cancellation=token,
            )
            outputs[role] = result
            created_artifact_ids.append(str(result["artifact_id"]))

        pair_artifact_id = new_id("artifact")
        pair_root = store.derived_dir(session_id, "aligned_clip_set", pair_artifact_id)
        role_rows: dict[str, Any] = {}
        for role in ("R01", "R02"):
            preview = previews[role]["preview"]
            result = outputs[role]
            role_rows[role] = {
                **preview,
                "trim_artifact_id": result["artifact_id"],
                "output_dir": result["output_dir"],
                "time_sync_artifact_id": previews[role]["plan"]["request"]["time_sync_artifact_id"],
                "time_map_id": previews[role]["plan"]["request"]["time_map_id"],
            }
        pair_payload = {
            "schema_version": PAIR_SCHEMA,
            "producer_version": __version__,
            "engine": ENGINE,
            "paired_trim_set_id": pair_id,
            "calibration_handoff_artifact_id": handoff.id,
            "window_source": "common_body_c3d_overlap",
            "common_vicon_window": common_window,
            "body_intervals": body_intervals,
            "roles": role_rows,
            "source_mp4_reencoded": False,
            "ego_video_used_for_synchronization": False,
            "created_at": utc_now(),
        }
        manifest_path = pair_root / "c3d_defined_clip_set.json"
        report_path = pair_root / "c3d_defined_clip_set_report.html"
        atomic_write_json(manifest_path, pair_payload)
        report_path.write_text(_pair_report_html(pair_payload), "utf-8")
        source_ids = []
        for role in ("R01", "R02"):
            source_ids.extend([
                str(rows[role].get("body_asset_id") or ""),
                str(rows[role].get("etm_asset_id") or ""),
                str(normalized_role_inputs[role].get("native_asset_id") or ""),
                str(normalized_role_inputs[role].get("timeseries_asset_id") or ""),
                str(normalized_role_inputs[role].get("facemapper_asset_id") or ""),
            ])
        pair_artifact = ArtifactStore(store).register(
            session_id=session_id,
            kind=ArtifactKind.ALIGNED_CLIP_SET,
            label=f"{label_prefix} · R01 + R02",
            output_root=pair_root,
            source_asset_ids=list(dict.fromkeys(value for value in source_ids if value)),
            parent_artifact_ids=[handoff.id, *created_artifact_ids],
            parameters={
                "engine": ENGINE,
                "paired_trim_set_id": pair_id,
                "calibration_handoff_artifact_id": handoff.id,
                "window_source": "common_body_c3d_overlap",
                "role_trim_artifact_ids": {role: outputs[role]["artifact_id"] for role in ("R01", "R02")},
                "interval_policy": interval_policy,
                "face_tolerance_ms": float(face_tolerance_ms),
                "boundary_tolerance_ms": float(boundary_tolerance_ms),
            },
            qc={
                "role_count": 2,
                "all_roles_created": True,
                "common_duration_sec": common_window["duration_sec"],
                "maximum_start_boundary_error_ms": max(role_rows[role]["start_boundary_error_ms"] for role in role_rows),
                "maximum_end_boundary_error_ms": max(role_rows[role]["end_boundary_error_ms"] for role in role_rows),
                "source_video_reencoded": False,
            },
            provenance={
                "method": ENGINE,
                "time_mapping": "C3D Vicon window → inverse verified affine TimeGraph → Pupil clock → NRD .time/PTS",
                "manual_start_end_entered": False,
            },
            artifact_id=pair_artifact_id,
        )
        result_payload = {
            "artifact_id": pair_artifact.id,
            "paired_trim_set_id": pair_id,
            "common_vicon_window": common_window,
            "roles": role_rows,
            "manifest": str(manifest_path),
            "report_html": str(report_path),
        }
        store.update(lambda project: project.sessions[session_id].metadata.__setitem__("c3d_defined_trim_last", result_payload))
        emit(progress, 1.0, "C3D-defined R01/R02 Native-reference clips are ready")
        return result_payload
    except BaseException:
        _rollback_artifacts(store, session_id, created_artifact_ids)
        raise
