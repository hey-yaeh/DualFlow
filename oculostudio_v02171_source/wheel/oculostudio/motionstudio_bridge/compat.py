from __future__ import annotations
from pathlib import Path
from typing import Any
import json, os, uuid


def _artifact_classes():
    candidates=[
        ("oculostudio.domain.models","Artifact","ArtifactKind"),
        ("oculostudio.domain","Artifact","ArtifactKind"),
    ]
    for mod,a,k in candidates:
        try:
            m=__import__(mod,fromlist=[a,k]);return getattr(m,a),getattr(m,k)
        except Exception:continue
    return None,None


def _kind(enum_cls, value:str):
    if enum_cls is None:return value
    try:return enum_cls(value)
    except Exception:
        for name in dir(enum_cls):
            if name.startswith("_"):continue
            x=getattr(enum_cls,name)
            if getattr(x,"value",None)==value:return x
        return value


def register_result(app: Any, *, session_id:str, kind_value:str, label:str, output_dir:str|Path,
                    primary_path:str|Path, parameters:dict[str,Any], qc:dict[str,Any], files:list[dict[str,Any]]|None=None)->dict[str,Any]|None:
    controller=getattr(getattr(app,"state",None),"controller",None)
    if controller is None:return None
    try:store,project=controller.require_project()
    except Exception:return None
    try:session=project.get_session(session_id)
    except Exception:return None
    Artifact,ArtifactKind=_artifact_classes()
    if Artifact is None:return None
    out=Path(output_dir); primary=Path(primary_path)
    artifact_id=out.name if out.name.startswith("artifact") else f"artifact_{uuid.uuid4().hex[:16]}"
    file_entries=files or []
    if not file_entries:
        for p in out.rglob("*"):
            if p.is_file():file_entries.append({"path":str(p),"role":p.suffix.lower().lstrip(".") or "file","size_bytes":p.stat().st_size})
    payload={
        "id":artifact_id,"session_id":session_id,"kind":_kind(ArtifactKind,kind_value),"label":label,
        "output_path":str(primary),"source_asset_ids":[],"parent_artifact_ids":[],"clip_id":None,
        "parameters":parameters,"metrics":qc,"qc":qc,"files":file_entries,"status":"created",
        "producer_version":"0.16.1",
    }
    fields=getattr(Artifact,"model_fields",None) or getattr(Artifact,"__fields__",{})
    if fields:payload={k:v for k,v in payload.items() if k in fields}
    artifact=Artifact(**payload)
    session.artifacts.append(artifact);store.save(project)
    return artifact.model_dump(mode="json") if hasattr(artifact,"model_dump") else payload
