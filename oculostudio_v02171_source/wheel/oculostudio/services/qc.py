from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from oculostudio.domain.models import ArtifactKind
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import atomic_write_json, new_id, utc_now


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def build_session_qc(store: ProjectStore, session_id: str) -> dict[str, Any]:
    session = store.project.sessions[session_id]
    interaction_assets = [asset for asset in session.assets.values()
                          if str((asset.metadata or {}).get("asset_scope") or "interaction") != "calibration"]
    calibration_sources = [asset for asset in session.assets.values()
                           if str((asset.metadata or {}).get("asset_scope") or "interaction") == "calibration"]
    kinds = Counter(str(asset.kind) for asset in interaction_assets)
    available = sum(asset.available and Path(asset.source_path).exists() for asset in interaction_assets)
    time_maps = list(session.time_maps.values())
    direct_global = [mapping for mapping in time_maps if mapping.target_domain_id == session.global_time_domain_id]
    calibration_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.CALIBRATION_PROFILE]
    etm_assignment_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.ETM_ROLE_ASSIGNMENT]
    participant_association_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.PARTICIPANT_ASSOCIATION]
    gaze_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.GAZE_RAYS]
    semantic_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.SEMANTIC_TARGETS]
    fusion_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.TARGET_FUSION]
    face_aoi_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.FACE_AOI]
    body_aoi_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.BODY_AOI]
    object_aoi_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.OBJECT_AOI]
    dyadic_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.DYADIC_ATTENTION]
    privacy_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.PRIVACY_VIDEO]
    privacy_clip_artifacts = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.PRIVACY_CLIP]
    frame_maps = [artifact for artifact in session.artifacts.values() if artifact.kind == ArtifactKind.FRAME_MAP]

    def last(items):
        return sorted(items, key=lambda item: item.created_at)[-1] if items else None

    last_frame_map = last(frame_maps)
    last_privacy = last(privacy_artifacts)
    last_calibration = last(calibration_artifacts)
    last_etm_assignment = last(etm_assignment_artifacts)
    last_participant_association = last(participant_association_artifacts)
    last_gaze = last(gaze_artifacts)
    last_semantic = last(semantic_artifacts)
    last_fusion = last(fusion_artifacts)
    last_face_aoi = last(face_aoi_artifacts)
    last_body_aoi = last(body_aoi_artifacts)
    last_object_aoi = last(object_aoi_artifacts)
    last_dyadic = last(dyadic_artifacts)
    last_privacy_clip = last(privacy_clip_artifacts)
    metrics = [
        {
            "id": "recording_match",
            "label": "Recording Match",
            "value": available / len(interaction_assets) if interaction_assets else None,
            "display": f"{available}/{len(interaction_assets)}",
            "status": "good" if available == len(interaction_assets) and interaction_assets else "warning",
            "detail": "Registered source paths available",
        },
        {
            "id": "native_cloud_sync",
            "label": "Native–Cloud Sync",
            "value": (last_frame_map.qc.get("matched_fraction") if last_frame_map else None),
            "display": _percent(last_frame_map.qc.get("matched_fraction")) if last_frame_map else "Not run",
            "status": "good" if last_frame_map and last_frame_map.qc.get("matched_fraction", 0) >= 0.95 else "neutral",
            "detail": (
                f"Median {last_frame_map.qc.get('median_abs_error_ms', 0):.3f} ms"
                if last_frame_map
                else "Generate a Native–Cloud frame map"
            ),
        },
        {
            "id": "privacy",
            "label": "Face Blur Coverage",
            "value": (last_privacy.qc.get("mask_frame_fraction") if last_privacy else None),
            "display": _percent(last_privacy.qc.get("mask_frame_fraction")) if last_privacy else "Not run",
            "status": "warning" if last_privacy else "neutral",
            "detail": (
                "Trimmed privacy clip available; detection coverage is not a privacy guarantee."
                if last_privacy_clip
                else "Detection coverage is not a privacy guarantee; visually review output."
            ),
        },
        {
            "id": "gaze3d",
            "label": (
                "Trim 3D Ray Validity"
                if last_gaze and last_gaze.qc.get("scope") == "native_trim"
                else "3D Ray Validity"
            ),
            "value": (
                last_gaze.qc.get("trim_valid_ray_fraction")
                if last_gaze and last_gaze.qc.get("scope") == "native_trim"
                else (last_gaze.qc.get("valid_ray_fraction") if last_gaze else None)
            ),
            "display": (
                _percent(last_gaze.qc.get("trim_valid_ray_fraction"))
                if last_gaze and last_gaze.qc.get("scope") == "native_trim"
                else (_percent(last_gaze.qc.get("valid_ray_fraction")) if last_gaze else "Not run")
            ),
            "status": "good" if last_gaze and (
                last_gaze.qc.get("trim_valid_ray_fraction")
                if last_gaze.qc.get("scope") == "native_trim"
                else last_gaze.qc.get("valid_ray_fraction", 0)
            ) >= 0.8 else "neutral",
            "detail": (
                (
                    f"{last_gaze.qc.get('trim_valid_ray_count', 0)}/{last_gaze.qc.get('selected_input_row_count', 0)} retained rows · "
                    f"{last_gaze.qc.get('alignment_mode', 'timegraph').replace('_', ' ')} · role {last_gaze.qc.get('wearer_role_id') or 'manual'}"
                    if last_gaze.qc.get("scope") == "native_trim"
                    else f"{last_gaze.qc.get('valid_ray_count', 0)} valid rays · role {last_gaze.qc.get('wearer_role_id') or 'manual'}"
                )
                if last_gaze
                else (
                    f"ETM roles {last_etm_assignment.qc.get('status', 'review')} · generate calibrated rays next"
                    if last_etm_assignment else "Resolve ETM wearer, then generate calibrated gaze rays"
                )
            ),
        },
        {
            "id": "participant_association",
            "label": "Partner Face Identity",
            "value": (last_participant_association.qc.get("matched_participant_fraction") if last_participant_association else None),
            "display": (_percent(last_participant_association.qc.get("matched_participant_fraction")) if last_participant_association else "Not run"),
            "status": "good" if last_participant_association and last_participant_association.qc.get("matched_participant_fraction", 0) >= 0.4 else "neutral",
            "detail": (
                f"{last_participant_association.qc.get('partner_face_count', 0)} partner faces · "
                f"{last_participant_association.qc.get('other_person_count', 0)} other-person faces · "
                f"{100*last_participant_association.qc.get('review_required_fraction', 0):.1f}% review"
                if last_participant_association
                else "Project C3D heads into the ego camera before partner-only Face AOI"
            ),
        },
        {
            "id": "semantic_targets",
            "label": "Semantic Targets",
            "value": (
                last_fusion.qc.get("known_target_fraction")
                if last_fusion
                else (last_semantic.qc.get("semantic_target_fraction") if last_semantic else None)
            ),
            "display": (
                _percent(last_fusion.qc.get("known_target_fraction"))
                if last_fusion
                else (_percent(last_semantic.qc.get("semantic_target_fraction")) if last_semantic else "Not run")
            ),
            "status": (
                "good"
                if (
                    last_fusion and last_fusion.qc.get("known_target_fraction", 0) >= 0.7
                ) or (
                    not last_fusion and last_semantic and last_semantic.qc.get("semantic_target_fraction", 0) >= 0.7
                )
                else "neutral"
            ),
            "detail": (
                f"Face {100*last_face_aoi.qc.get('face_aoi_hit_fraction', 0):.1f}% · "
                f"Body {100*last_body_aoi.qc.get('body_hit_fraction', 0):.1f}% · "
                f"Object {100*(last_object_aoi.qc.get('object_hit_fraction', 0) if last_object_aoi else 0):.1f}% · "
                f"{100*last_fusion.qc.get('review_required_fraction', 0):.1f}% review"
                if last_fusion and last_face_aoi and last_body_aoi
                else (
                    f"{last_semantic.qc.get('body_3d_hit_count', 0)} body hits · "
                    f"{last_semantic.qc.get('face_2d_hit_count', 0)} face hits · "
                    f"{100*last_semantic.qc.get('review_required_fraction', 0):.1f}% review"
                    if last_semantic
                    else "Generate independent Face AOI and Body AOI evidence, then fuse by gaze_sample_uid"
                )
            ),
        },
        {
            "id": "dyadic_attention",
            "label": "Dyadic Attention",
            "value": (last_dyadic.qc.get("mutual_face_gaze_fraction") if last_dyadic else None),
            "display": (_percent(last_dyadic.qc.get("mutual_face_gaze_fraction")) if last_dyadic else "Not run"),
            "status": "good" if last_dyadic else "neutral",
            "detail": (
                f"Mutual eye {100*last_dyadic.qc.get('mutual_eye_contact_fraction', 0):.1f}% · "
                f"Joint attention {100*last_dyadic.qc.get('joint_attention_fraction', 0):.1f}%"
                if last_dyadic else "Fuse two synchronized role target streams"
            ),
        },
        {
            "id": "calibration",
            "label": "Calibration Status",
            "value": None,
            "display": (
                f"{last_calibration.qc.get('direct_reprojection_rms_px_median', 0):.2f}px"
                if last_calibration
                else "Not run"
            ),
            "status": "good" if last_calibration else "neutral",
            "detail": (
                f"{last_calibration.qc.get('accepted_count', 0)} robustly accepted frames"
                if last_calibration
                else "Create an .oculocal profile"
            ),
        },
        {
            "id": "timegraph",
            "label": "TimeGraph",
            "value": None,
            "display": f"{len(time_maps)} maps",
            "status": "good" if direct_global else "warning",
            "detail": (
                f"{len(direct_global)} source clock(s) connected directly to Global Time"
                if direct_global
                else "No source clock is connected to Global Time"
            ),
        },
    ]
    warnings: list[str] = []
    for asset in session.assets.values():
        if not Path(asset.source_path).exists():
            warnings.append(f"Missing source asset: {asset.label}")
        warnings.extend(f"{asset.label}: {warning}" for warning in asset.warnings)
    if last_etm_assignment and last_etm_assignment.qc.get("status") != "resolved":
        warnings.append("The latest ETM-role assignment requires review; do not use it for Gaze 3D automatically.")
    if last_participant_association and last_participant_association.qc.get("review_required_fraction", 0) > 0.1:
        warnings.append("More than 10% of participant face assignments require review; inspect projected heads and other-person detections.")
    if last_gaze and last_gaze.qc.get("scope") == "native_trim":
        overlap = float(last_gaze.qc.get("tracker_time_overlap_fraction", 0) or 0)
        if overlap < 0.8:
            warnings.append(
                f"Only {100*overlap:.1f}% of Native-trim gaze rows fall inside the selected tracker C3D time range; "
                "verify that both sources were cut to the same physical interval."
            )
    for mapping in time_maps:
        if mapping.quality.warning:
            warnings.append(f"{mapping.source_domain_id} → {mapping.target_domain_id}: {mapping.quality.warning}")
        if mapping.quality.max_abs_residual_ns > 20_000_000:
            warnings.append(
                f"Large sync residual ({mapping.quality.max_abs_residual_ns / 1e6:.1f} ms) for {mapping.id}"
            )
    return {
        "schema_version": "oculostudio-session-qc-1.0",
        "project_id": store.project.id,
        "project_name": store.project.name,
        "session_id": session_id,
        "session_label": session.label,
        "generated_at": utc_now(),
        "asset_count": len(session.assets),
        "asset_kind_counts": dict(kinds),
        "time_domain_count": len(session.time_domains),
        "time_map_count": len(session.time_maps),
        "clip_count": len(session.clips),
        "artifact_count": len(session.artifacts),
        "metrics": metrics,
        "warnings": warnings,
        "assets": [asset.model_dump(mode="json") for asset in session.assets.values()],
        "time_maps": [mapping.model_dump(mode="json") for mapping in session.time_maps.values()],
        "clips": [clip.model_dump(mode="json") for clip in session.clips.values()],
        "artifacts": [artifact.model_dump(mode="json") for artifact in session.artifacts.values()],
    }


def _html_report(report: dict[str, Any]) -> str:
    cards = []
    for metric in report["metrics"]:
        cards.append(
            f"""
            <article class="card {html.escape(metric['status'])}">
              <div class="eyebrow">{html.escape(metric['label'])}</div>
              <div class="value">{html.escape(str(metric['display']))}</div>
              <p>{html.escape(str(metric['detail']))}</p>
            </article>
            """
        )
    warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in report["warnings"]) or "<li>No project-level warnings.</li>"
    assets = "".join(
        f"<tr><td>{html.escape(asset['label'])}</td><td>{html.escape(str(asset['kind']))}</td><td>{html.escape(asset['source_path'])}</td><td>{'Available' if Path(asset['source_path']).exists() else 'Missing'}</td></tr>"
        for asset in report["assets"]
    )
    artifact_rows: list[str] = []
    for artifact in report["artifacts"]:
        files = artifact.get("files") or []
        file_paths = [Path(item["path"]) for item in files if item.get("path")]
        root = ""
        if file_paths:
            try:
                import os
                root_path = Path(os.path.commonpath([str(path.parent) for path in file_paths]))
            except Exception:
                root_path = file_paths[0].parent
            root = str(root_path)
        file_items = []
        for item in files:
            path = Path(item.get("path") or "")
            if not path:
                continue
            try:
                href = path.resolve().as_uri() if path.exists() else ""
            except Exception:
                href = ""
            name = html.escape(path.name)
            shown = f"<a href='{html.escape(href, quote=True)}'>{name}</a>" if href else name
            file_items.append(f"<li>{shown}<br><code>{html.escape(str(path))}</code></li>")
        details = f"<details><summary>{len(file_items)} file(s)</summary><ul>{''.join(file_items)}</ul></details>" if file_items else "No files"
        artifact_rows.append(
            f"<tr><td>{html.escape(artifact['label'])}<br><code>{html.escape(artifact['id'])}</code></td>"
            f"<td>{html.escape(str(artifact['kind']))}</td>"
            f"<td><code>{html.escape(root)}</code></td>"
            f"<td>{details}</td>"
            f"<td>{html.escape(artifact['created_at'])}</td></tr>"
        )
    artifacts = "".join(artifact_rows) or "<tr><td colspan='5'>No derived artifacts yet.</td></tr>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>OculoStudio QC — {html.escape(report['session_label'])}</title>
<style>
:root{{--ink:#262719;--olive:#454825;--lime:#eefc39;--cream:#f4f3ec;--line:#d9d8cd;}}
*{{box-sizing:border-box}}body{{margin:0;background:#eef13d;font:15px/1.5 Inter,Segoe UI,sans-serif;color:var(--ink)}}
main{{max-width:1180px;margin:34px auto;background:var(--cream);border:14px solid var(--olive);border-radius:34px;padding:30px;box-shadow:0 24px 70px #31341455}}
h1{{margin:0;font-size:34px}}.sub{{color:#6c6e5f}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}}
.card{{border:1px solid var(--line);border-radius:20px;padding:18px;background:#fff}}.card.good{{box-shadow:inset 0 4px var(--lime)}}.card.warning{{box-shadow:inset 0 4px #e3a71a}}
.eyebrow{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:#77796b}}.value{{font-size:28px;font-weight:700;margin:8px 0}}p{{margin:0;color:#66685a}}
h2{{margin-top:30px}}table{{width:100%;border-collapse:collapse;background:#fff;border-radius:16px;overflow:hidden}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#e8e7dd}}code{{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}}details summary{{cursor:pointer;font-weight:700}}details ul{{padding-left:18px}}a{{color:#526710}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{margin:0;border-radius:0}}}}
</style></head><body><main>
<div class="eyebrow">OculoStudio CleanCore • QC & provenance report</div>
<h1>{html.escape(report['session_label'])}</h1><div class="sub">{html.escape(report['project_name'])} • generated {html.escape(report['generated_at'])}</div>
<section class="cards">{''.join(cards)}</section>
<h2>Warnings and review gates</h2><ul>{warnings}</ul>
<h2>Source assets</h2><table><thead><tr><th>Label</th><th>Kind</th><th>Path</th><th>Status</th></tr></thead><tbody>{assets}</tbody></table>
<h2>Derived artifacts and processed files</h2><table><thead><tr><th>Artifact</th><th>Kind</th><th>Folder</th><th>Files</th><th>Created</th></tr></thead><tbody>{artifacts}</tbody></table>
<p style="margin-top:30px"><strong>Privacy notice:</strong> successful face-mask processing does not prove that every identifiable face was detected. Perform visual review before sharing video.</p>
</main></body></html>"""


def export_qc_report(store: ProjectStore, session_id: str) -> dict[str, Any]:
    report = build_session_qc(store, session_id)
    artifact_id = new_id("artifact")
    output = store.report_dir(session_id) / artifact_id
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "oculostudio_qc_report.json"
    html_path = output / "oculostudio_qc_report.html"
    atomic_write_json(json_path, report)
    html_path.write_text(_html_report(report), "utf-8")
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=ArtifactKind.QC_REPORT,
        label="Session QC report",
        output_root=output,
        source_asset_ids=list(store.project.sessions[session_id].assets),
        qc={"warning_count": len(report["warnings"]), "metric_count": len(report["metrics"])},
        artifact_id=artifact_id,
    )
    return {
        "json": str(json_path),
        "html": str(html_path),
        "report": report,
        "artifact": artifact.model_dump(mode="json"),
    }
