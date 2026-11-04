from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from oculostudio.domain.models import ArtifactKind
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.services.idempotency import find_artifact_by_signature, processing_input_context, processing_signature

from .contracts import inspect_scene_bundle
from .service import import_scene_bundles, annotate_body_mesh_aoi, annotate_object_mesh_aoi
from .picker import native_pick


class PickRequest(BaseModel):
    kind: str = "scene"
    multiple: bool = False


class InspectRequest(BaseModel):
    manifest_path: str
    verify_checksums: bool = True


class ImportRequest(BaseModel):
    manifest_paths: list[str]
    project_root: str
    session_id: str
    verify_checksums: bool = True
    allow_incomplete: bool = False


class BodyAOIRequest(BaseModel):
    scene_catalog_path: str
    gaze_rays_path: str
    project_root: str
    session_id: str
    output_label: str
    viewer_role: str = "R01"
    body_model_root: str | None = None
    ray_length_unit: str = "m"
    max_time_error_ms: float = 12.0
    uncertainty_deg: float = 1.5
    max_distance_m: float = 10.0
    assume_same_world: bool = False
    also_parquet: bool = True
    motion_time_scale: float = 1.0
    motion_source_origin_ns: int = 0
    motion_target_origin_ns: int = 0


class ObjectAOIRequest(BaseModel):
    scene_catalog_path: str
    gaze_rays_path: str
    project_root: str
    session_id: str
    output_label: str
    ray_length_unit: str = "m"
    max_time_error_ms: float = 12.0
    uncertainty_deg: float = 1.5
    max_distance_m: float = 10.0
    assume_same_world: bool = False
    also_parquet: bool = True
    motion_time_scale: float = 1.0
    motion_source_origin_ns: int = 0
    motion_target_origin_ns: int = 0


def _register(runtime: Any, *, session_id: str, kind: ArtifactKind, label: str,
              output_dir: str, parameters: dict[str, Any], qc: dict[str, Any],
              operation: str, signature: str):
    if runtime is None:
        return None
    store = runtime.require_store()
    artifact = ArtifactStore(store).register(
        session_id=session_id,
        kind=kind,
        label=label,
        output_root=output_dir,
        parameters=parameters,
        qc=qc,
        provenance={
            "integration": "MotionStudio Scene Bridge",
            "processing_signature": signature,
            "processing_operation": operation,
            "signature_version": "oculostudio-processing-signature-v1",
        },
    )
    return artifact.model_dump(mode="json")


def _artifact_root(artifact) -> Path:
    paths=[Path(row.path) for row in artifact.files if row.path]
    if not paths:
        return Path('.')
    parents=[path if path.is_dir() else path.parent for path in paths]
    common=Path(__import__('os').path.commonpath([str(path) for path in parents]))
    return common


def _reuse(runtime: Any, *, session_id: str, operation: str, parameters: dict[str, Any]):
    if runtime is None:
        return None, processing_signature(operation, session_id, parameters)
    store=runtime.require_store()
    snapshot=store.snapshot()
    signature=processing_signature(
        operation,
        session_id,
        parameters,
        input_context=processing_input_context(snapshot,session_id,parameters),
    )
    artifact=find_artifact_by_signature(store.snapshot(),session_id,signature)
    if artifact is None:
        return None,signature
    root=_artifact_root(artifact)
    result={
        "artifact":artifact.model_dump(mode="json"),
        "output_dir":str(root),
        "deduplication":{
            "reused":True,
            "reason":"completed_artifact",
            "processing_signature":signature,
            "artifact_id":artifact.id,
            "message":"An identical immutable MotionStudio result already exists and was reused.",
        },
    }
    for file in artifact.files:
        path=Path(file.path)
        name=path.name.lower()
        if name=='motionstudio_scene_catalog.json':
            result['catalog']=str(path)
            try:
                payload=json.loads(path.read_text('utf-8'))
                result.update(scene_id=payload.get('scene_id'),actors=payload.get('actors',[]),
                              shared_props=payload.get('shared_props'),warnings=payload.get('warnings',[]))
            except Exception:
                pass
        elif name=='body_aoi_manifest.json':
            result['manifest']=str(path)
            try: result['qc']=json.loads(path.read_text('utf-8')).get('qc',{})
            except Exception: pass
        elif name=='object_aoi_manifest.json':
            result['manifest']=str(path)
            try: result['qc']=json.loads(path.read_text('utf-8')).get('qc',{})
            except Exception: pass
    return result,signature


def install_motionstudio_routes(app, runtime=None):
    @app.post("/api/motionstudio/pick")
    def motionstudio_pick(req: PickRequest):
        try:
            return {"paths": native_pick(req.kind, req.multiple)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/motionstudio/inspect")
    def motionstudio_inspect(req: InspectRequest):
        try:
            return inspect_scene_bundle(req.manifest_path, verify_checksums=req.verify_checksums).to_jsonable()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/motionstudio/import")
    def motionstudio_import(req: ImportRequest):
        try:
            parameters=req.model_dump()
            parameters["manifest_paths"]=sorted(parameters["manifest_paths"])
            reused,signature=_reuse(runtime,session_id=req.session_id,operation="motionstudio_import",parameters=parameters)
            if reused is not None:
                return reused
            result = import_scene_bundles(
                req.manifest_paths,
                project_root=req.project_root,
                session_id=req.session_id,
                verify_checksums=req.verify_checksums,
                allow_incomplete=req.allow_incomplete,
            )
            result["artifact"] = _register(
                runtime,
                session_id=req.session_id,
                kind=ArtifactKind.MOTIONSTUDIO_SCENE,
                label=f"{result['scene_id']} · MotionStudio scene",
                output_dir=result["output_dir"],
                parameters=parameters,
                qc={
                    "actor_count": len(result["actors"]),
                    "warning_count": len(result["warnings"]),
                    "complete_for_mesh_aoi": True,
                },
                operation="motionstudio_import",
                signature=signature,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/motionstudio/body-aoi")
    def motionstudio_body_aoi(req: BodyAOIRequest):
        try:
            parameters=req.model_dump()
            reused,signature=_reuse(runtime,session_id=req.session_id,operation="motionstudio_body_aoi",parameters=parameters)
            if reused is not None:
                return reused
            result = annotate_body_mesh_aoi(**parameters)
            result["artifact"] = _register(
                runtime,
                session_id=req.session_id,
                kind=ArtifactKind.BODY_AOI,
                label=req.output_label,
                output_dir=result["output_dir"],
                parameters=parameters,
                qc=result["qc"],
                operation="motionstudio_body_aoi",
                signature=signature,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/motionstudio/object-aoi")
    def motionstudio_object_aoi(req: ObjectAOIRequest):
        try:
            parameters=req.model_dump()
            reused,signature=_reuse(runtime,session_id=req.session_id,operation="motionstudio_object_aoi",parameters=parameters)
            if reused is not None:
                return reused
            result = annotate_object_mesh_aoi(**parameters)
            result["artifact"] = _register(
                runtime,
                session_id=req.session_id,
                kind=ArtifactKind.OBJECT_AOI,
                label=req.output_label,
                output_dir=result["output_dir"],
                parameters=parameters,
                qc=result["qc"],
                operation="motionstudio_object_aoi",
                signature=signature,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
