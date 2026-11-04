from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

from oculostudio.domain.models import ArtifactRecord, AssetRecord
from oculostudio.errors import ValidationError
from oculostudio.storage import ProjectStore

_CODE_PATTERNS = {
    "session": re.compile(r"(?<![A-Za-z0-9])(S\d{1,6})(?![A-Za-z0-9])", re.I),
    "interaction": re.compile(r"(?<![A-Za-z0-9])(I\d{1,6})(?![A-Za-z0-9])", re.I),
    "role": re.compile(r"(?<![A-Za-z0-9])(R\d{1,6})(?![A-Za-z0-9])", re.I),
}

_ASSET_STAGE = {
    "pupil_native": "native",
    "pupil_cloud": "timeseries",
    "face_mapper": "facemapper",
    "c3d": "motion_capture",
    "mocap_npz": "motion_capture",
    "calibration": "calibration_source",
}

_ARTIFACT_STAGE = {
    "native_reference_trim": "trim",
    "aligned_facial_video": "aligned_facial_video",
    "face_mapper_filter": "face_filter",
    "privacy_video": "privacy_full",
    "privacy_clip": "privacy_clip",
    "participant_association": "participant_association",
    "face_aoi": "face_aoi",
    "frame_map": "frame_map",
    "calibration_profile": "calibration",
    "calibration_acceptance": "calibration_acceptance",
    "etm_role_assignment": "etm_assignment",
    "gaze_rays": "gaze3d",
    "motionstudio_scene": "motionstudio_scene",
    "body_aoi": "body_aoi",
    "object_aoi": "object_aoi",
    "target_fusion": "fusion",
    "semantic_targets": "fusion",
    "dyadic_attention": "dyadic",
    "aoi_review_video": "review_video",
    "qc_report": "qc",
    "results_package": "results_package",
}

_ROLE_STAGES = {
    "native",
    "timeseries",
    "facemapper",
    "trim",
    "aligned_facial_video",
    "face_filter",
    "privacy_full",
    "privacy_clip",
    "participant_association",
    "calibration_acceptance",
    "face_aoi",
    "gaze3d",
    "body_aoi",
    "object_aoi",
    "fusion",
    "review_video",
}

_SHARED_STAGES = {
    "motion_capture",
    "frame_map",
    "calibration",
    "calibration_source",
    "etm_assignment",
    "motionstudio_scene",
    "dyadic",
    "qc",
    "results_package",
}


def _flatten_strings(value: Any, *, limit: int = 2000) -> list[str]:
    """Collect a bounded set of scalar strings from nested metadata.

    Interaction indexing is deliberately read-only and heuristic. It never mutates
    scientific records, and it favours explicit metadata over filename parsing.
    """

    output: list[str] = []
    stack: list[Any] = [value]
    while stack and len(output) < limit:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, str):
            output.append(current)
        elif isinstance(current, (int, float, bool)):
            continue
        elif isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)
        elif hasattr(current, "model_dump"):
            stack.append(current.model_dump(mode="json"))
        else:
            output.append(str(current))
    return output


def _first_explicit(mapping: dict[str, Any], names: Iterable[str]) -> str:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip().upper()
    return ""


def _codes_from_record(record: AssetRecord | ArtifactRecord, session_label: str) -> dict[str, str]:
    raw = record.model_dump(mode="json")
    metadata: dict[str, Any] = {}
    for key in ("metadata", "parameters", "provenance", "qc"):
        value = raw.get(key)
        if isinstance(value, dict):
            metadata.update(value)

    explicit = {
        "session": _first_explicit(metadata, ("session_id", "session", "session_code")),
        "interaction": _first_explicit(
            metadata,
            ("interaction_id", "interaction", "interaction_code", "scene_id", "merge_key"),
        ),
        "role": _first_explicit(metadata, ("role_id", "role", "role_code", "viewer_role", "wearer_role_id")),
    }

    strings = [session_label, str(raw.get("label") or ""), str(raw.get("source_path") or "")]
    strings.extend(str(item.get("path") or "") for item in raw.get("files") or [])
    strings.extend(_flatten_strings(metadata))
    joined = "\n".join(strings)

    result: dict[str, str] = {}
    for key, pattern in _CODE_PATTERNS.items():
        value = explicit[key]
        match = pattern.search(value) if value else None
        if not match:
            match = pattern.search(joined)
        result[key] = match.group(1).upper() if match else ""
    return result


def _record_summary(record: AssetRecord | ArtifactRecord, stage: str, codes: dict[str, str]) -> dict[str, Any]:
    raw = record.model_dump(mode="json")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    participant_id = str(
        raw.get("participant_id")
        or metadata.get("participant_id")
        or parameters.get("participant_id")
        or provenance.get("participant_id")
        or ""
    ).strip().upper()
    return {
        "id": record.id,
        "label": record.label,
        "kind": str(record.kind),
        "stage": stage,
        "session_id": codes["session"],
        "interaction_id": codes["interaction"],
        "role_id": codes["role"],
        "participant_id": participant_id,
        "created_at": raw.get("created_at"),
        "available": raw.get("available", True),
    }


def _stage_status(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    matches = [row for row in rows if row["stage"] == stage]
    return {
        "state": "complete" if matches else "missing",
        "count": len(matches),
        "record_ids": [row["id"] for row in matches],
        "labels": [row["label"] for row in matches],
    }


def _role_summary(rows: list[dict[str, Any]], role_id: str) -> dict[str, Any]:
    role_rows = [row for row in rows if row.get("role_id") == role_id]
    # Records without a role can still be interaction-level inputs. Do not silently
    # assign them to a person; the guided UI shows them under Shared Scene instead.
    stages = {stage: _stage_status(role_rows, stage) for stage in sorted(_ROLE_STAGES)}
    # The clean workflow keeps Privacy separate from target generation.
    # Calibration profiles are reusable project-level resources; Object AOI is optional.
    face_required = ("native", "timeseries", "facemapper", "trim", "face_filter", "privacy_clip")
    scene_required = ("gaze3d", "participant_association", "face_aoi", "body_aoi", "fusion")
    participant_ids = sorted({row.get("participant_id", "") for row in role_rows if row.get("participant_id")})
    return {
        "role_id": role_id,
        "participant_id": participant_ids[0] if len(participant_ids) == 1 else "",
        "participant_ids": participant_ids,
        "records": role_rows,
        "stages": stages,
        "face_progress": sum(stages[name]["state"] == "complete" for name in face_required),
        "face_total": len(face_required),
        "scene_progress": sum(stages[name]["state"] == "complete" for name in scene_required),
        "scene_total": len(scene_required),
    }


def _interaction_status(rows: list[dict[str, Any]], time_maps: list[dict[str, Any]], interaction_id: str) -> dict[str, Any]:
    roles = sorted({row["role_id"] for row in rows if row.get("role_id")})
    for default in ("R01", "R02"):
        if default not in roles:
            roles.append(default)
    role_summaries = {role: _role_summary(rows, role) for role in roles}
    shared_rows = [row for row in rows if not row.get("role_id") or row["stage"] in _SHARED_STAGES]
    shared_stages = {stage: _stage_status(shared_rows, stage) for stage in sorted(_SHARED_STAGES)}

    interaction_token = interaction_id.lower()
    relevant_maps = []
    for mapping in time_maps:
        text = " ".join(_flatten_strings(mapping)).lower()
        if interaction_token in text or not _CODE_PATTERNS["interaction"].search(text):
            relevant_maps.append(mapping)
    formal_maps = [m for m in relevant_maps if "provisional" not in str(m.get("method", "")).lower()]
    shared_stages["synchronization"] = {
        "state": "complete" if formal_maps else "missing",
        "count": len(formal_maps),
        "record_ids": [str(m.get("id", "")) for m in formal_maps],
        "labels": [str(m.get("method", "time map")) for m in formal_maps],
    }

    dyadic = shared_stages["dyadic"]["state"] == "complete"
    face_complete = all(
        role_summaries[role]["stages"]["face_aoi"]["state"] == "complete"
        for role in ("R01", "R02")
        if role in role_summaries
    )
    full_complete = all(
        role_summaries[role]["stages"]["fusion"]["state"] == "complete"
        for role in ("R01", "R02")
        if role in role_summaries
    )
    return {
        "roles": role_summaries,
        "shared": {"records": shared_rows, "stages": shared_stages},
        "summary": {
            "face_ready": face_complete,
            "scene_ready": full_complete,
            "dyadic_ready": dyadic,
            "record_count": len(rows),
        },
    }


def build_interaction_workspace(store: ProjectStore, session_id: str) -> dict[str, Any]:
    if session_id not in store.project.sessions:
        raise ValidationError(f"Unknown session: {session_id}")
    session = store.project.sessions[session_id]
    rows_by_interaction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unassigned: list[dict[str, Any]] = []

    for asset in session.assets.values():
        # Calibration-session sources are project-level resources, not interaction inputs.
        # They must never create Guided-workspace interactions or leak into shared scene
        # candidates merely because their filenames contain session-like tokens.
        metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
        if str(metadata.get("asset_scope") or "interaction").strip().lower() == "calibration":
            continue
        codes = _codes_from_record(asset, session.label)
        stage = _ASSET_STAGE.get(str(asset.kind), "asset")
        summary = _record_summary(asset, stage, codes)
        if codes["interaction"]:
            rows_by_interaction[codes["interaction"]].append(summary)
        else:
            unassigned.append(summary)

    for artifact in session.artifacts.values():
        codes = _codes_from_record(artifact, session.label)
        stage = _ARTIFACT_STAGE.get(str(artifact.kind), "artifact")
        summary = _record_summary(artifact, stage, codes)
        if codes["interaction"]:
            rows_by_interaction[codes["interaction"]].append(summary)
        else:
            unassigned.append(summary)

    # Explicit interaction planning lets a clean Session show I01/I02/... before
    # the first source file is imported. Imported identities later populate the
    # same cards; no scientific record is fabricated by this plan.
    for planned in session.metadata.get("planned_interactions") or []:
        token = str(planned or "").strip().upper()
        if _CODE_PATTERNS["interaction"].fullmatch(token):
            rows_by_interaction.setdefault(token, [])

    # Session/role-level sources such as one Face Mapper export covering many
    # interactions can be inherited without changing the original asset record.
    # Raw calibration-session sources are deliberately excluded; only completed,
    # reusable calibration-profile artifacts may appear as project-level evidence.
    reusable = [
        row for row in unassigned
        if row["stage"] in {"facemapper", "calibration"}
    ]

    time_maps = [mapping.model_dump(mode="json") for mapping in session.time_maps.values()]
    interactions = []
    for interaction_id in sorted(rows_by_interaction, key=lambda value: (int(value[1:]) if value[1:].isdigit() else 999999, value)):
        rows = list(rows_by_interaction[interaction_id])
        for inherited in reusable:
            copy = dict(inherited)
            copy["inherited"] = True
            rows.append(copy)
        status = _interaction_status(rows, time_maps, interaction_id)
        interactions.append(
            {
                "interaction_id": interaction_id,
                "scene_id": f"{_session_code(session.label)}_{interaction_id}",
                **status,
            }
        )

    selected = str(session.metadata.get("selected_interaction_id") or "")
    if selected and selected not in {row["interaction_id"] for row in interactions}:
        selected = ""
    if not selected and interactions:
        selected = interactions[0]["interaction_id"]
    selected_role = str(session.metadata.get("selected_role_id") or "R01").upper()
    if not _CODE_PATTERNS["role"].fullmatch(selected_role):
        selected_role = "R01"

    return {
        "session_id": session_id,
        "session_code": _session_code(session.label),
        "selected_interaction_id": selected,
        "selected_role_id": selected_role,
        "selected_workflow": str(session.metadata.get("selected_workflow") or "face"),
        "interactions": interactions,
        "unassigned": unassigned,
        "counts": {
            "interactions": len(interactions),
            "unassigned": len(unassigned),
            "assets": len(session.assets),
            "artifacts": len(session.artifacts),
        },
    }


def _session_code(label: str) -> str:
    match = _CODE_PATTERNS["session"].search(label or "")
    return match.group(1).upper() if match else (label.strip().replace(" ", "_") or "SESSION")


def set_interaction_context(
    store: ProjectStore,
    session_id: str,
    *,
    interaction_id: str,
    role_id: str = "R01",
    workflow: str = "face",
) -> dict[str, Any]:
    interaction_id = interaction_id.upper().strip()
    role_id = role_id.upper().strip()
    if interaction_id and not _CODE_PATTERNS["interaction"].fullmatch(interaction_id):
        raise ValidationError("Interaction ID must look like I01")
    if role_id and not _CODE_PATTERNS["role"].fullmatch(role_id):
        raise ValidationError("Role ID must look like R01")
    if workflow not in {"face", "scene", "dyadic"}:
        raise ValidationError("Workflow must be face, scene or dyadic")

    def mutate(project):
        if session_id not in project.sessions:
            raise ValidationError(f"Unknown session: {session_id}")
        metadata = project.sessions[session_id].metadata
        metadata["selected_interaction_id"] = interaction_id
        metadata["selected_role_id"] = role_id or "R01"
        metadata["selected_workflow"] = workflow

    store.update(mutate)
    return build_interaction_workspace(store, session_id)


def build_project_workspace(store: ProjectStore) -> dict[str, Any]:
    """Return a project-level Session/folder/calibration overview.

    This is a navigation summary only. Scientific computations remain scoped to
    the selected Session and Interaction.
    """
    project = store.snapshot()
    sessions = []
    calibration_profiles = []
    total_interactions = 0
    total_assets = 0
    total_artifacts = 0
    for session_id, session in project.sessions.items():
        workspace = build_interaction_workspace(store, session_id)
        paths = store.session_paths(session_id)
        profiles = [
            artifact for artifact in session.artifacts.values()
            if str(artifact.kind) == "calibration_profile"
        ]
        for artifact in profiles:
            calibration_profiles.append({
                "session_id": session_id,
                "session_label": session.label,
                "artifact_id": artifact.id,
                "label": artifact.label,
                "hardware_id": artifact.parameters.get("hardware_display_name")
                    or artifact.qc.get("hardware_display_name")
                    or artifact.parameters.get("hardware_id")
                    or artifact.qc.get("hardware_id"),
                "mount_version": artifact.parameters.get("mount_version")
                    or artifact.qc.get("mount_version"),
                "valid_sessions": artifact.parameters.get("valid_sessions")
                    or artifact.qc.get("valid_sessions") or [],
                "created_at": artifact.created_at,
            })
        interaction_count = int(workspace.get("counts", {}).get("interactions", 0))
        total_interactions += interaction_count
        total_assets += len(session.assets)
        total_artifacts += len(session.artifacts)
        sessions.append({
            "session_id": session_id,
            "label": session.label,
            "session_code": _session_code(session.label),
            "selected": session_id == project.selected_session_id,
            "interaction_count": interaction_count,
            "planned_interactions": list(session.metadata.get("planned_interactions") or []),
            "asset_count": len(session.assets),
            "artifact_count": len(session.artifacts),
            "calibration_profile_count": len(profiles),
            "calibration_binding": dict((session.metadata or {}).get("calibration_binding") or {}),
            "folder": str(paths["root"]),
            "paths": {key: str(value) for key, value in paths.items()},
        })
    sessions.sort(key=lambda row: (int(row["session_code"][1:]) if row["session_code"][1:].isdigit() else 999999, row["session_code"]))
    from oculostudio.services.calibration_sets import list_calibration_sets
    calibration_sets = list_calibration_sets(store)
    return {
        "project_id": project.id,
        "project_name": project.name,
        "project_root": project.root,
        "folder_layout_version": int((project.settings or {}).get("folder_layout_version") or 1),
        "shared_calibration": dict((project.settings or {}).get("shared_calibration") or {}),
        "shared_calibration_folder": str(store.shared_calibration_dir()),
        "calibration_sets_folder": str(store.calibration_sets_dir()),
        "selected_session_id": project.selected_session_id,
        "sessions": sessions,
        "calibration_sets": calibration_sets,
        "calibration_profiles": sorted(calibration_profiles, key=lambda row: str(row.get("created_at") or ""), reverse=True),
        "counts": {
            "sessions": len(sessions),
            "interactions": total_interactions,
            "assets": total_assets,
            "artifacts": total_artifacts,
            "calibration_profiles": len(calibration_profiles),
            "calibration_sets": len(calibration_sets),
        },
    }
