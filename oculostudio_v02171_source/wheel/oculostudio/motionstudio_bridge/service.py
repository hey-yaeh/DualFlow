from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import math
import os
import time
import uuid

import numpy as np
import pandas as pd

from .contracts import SceneBundle, SceneBundleError, inspect_scene_bundle, validate_scene_pair, sha256_file
from .body import load_body_motion, load_bodyparts, BodyMeshProvider
from .props import load_props
from .geometry import (
    apply_transform_points, apply_transform_directions, intersect_ray_triangles,
    nearest_timestamp_index, ray_aabb_intersection, closest_ray_vertex_angle,
)

Progress = Callable[[float, str], None] | None
Cancel = Callable[[], bool] | None


def _emit(cb: Progress, value: float, message: str):
    if cb: cb(float(value), message)


def _cancel(cb: Cancel):
    if cb and cb(): raise InterruptedError("MotionStudio bridge task cancelled")


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _safe_name(value: str) -> str:
    s="".join(c if c.isalnum() or c in "-_." else "_" for c in str(value)).strip("._")
    return s or "result"


def _atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),"utf-8")
    os.replace(tmp,path)


def _read_table(path: str | Path) -> pd.DataFrame:
    p=Path(path)
    if p.suffix.lower()==".parquet": return pd.read_parquet(p)
    if p.suffix.lower()==".csv": return pd.read_csv(p)
    raise SceneBundleError(f"Unsupported gaze-ray table: {p.suffix}")


def _write_table(df: pd.DataFrame, path: Path, also_parquet: bool = True) -> dict[str,str]:
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)
    result={"csv":str(path)}
    if also_parquet:
        try:
            pq=path.with_suffix(".parquet"); df.to_parquet(pq,index=False); result["parquet"]=str(pq)
        except Exception:
            pass
    return result


def _col(df: pd.DataFrame, names: Iterable[str], required: bool=True) -> str | None:
    norm={str(c).strip().lower():str(c) for c in df.columns}
    for n in names:
        if n.lower() in norm:return norm[n.lower()]
    if required: raise SceneBundleError(f"Required column not found; expected one of {list(names)}")
    return None


def _ray_columns(df: pd.DataFrame) -> dict[str,str|None]:
    return {
        "uid":_col(df,("gaze_sample_uid","source_row_id","sample_id","row_id"),False),
        "time":_col(df,("timestamp_global_ns","global_time_ns","timestamp_ns")),
        "source_time":_col(df,("timestamp_source_ns","source_timestamp_ns"),False),
        "valid":_col(df,("ray_valid","valid"),False),
        "ox":_col(df,("gaze_origin_x","ray_origin_x","origin_x")),
        "oy":_col(df,("gaze_origin_y","ray_origin_y","origin_y")),
        "oz":_col(df,("gaze_origin_z","ray_origin_z","origin_z")),
        "dx":_col(df,("gaze_direction_x","ray_direction_x","direction_x")),
        "dy":_col(df,("gaze_direction_y","ray_direction_y","direction_y")),
        "dz":_col(df,("gaze_direction_z","ray_direction_z","direction_z")),
        "source_row":_col(df,("source_gaze_row_index","source_row_index","source_row_id"),False),
        "clip_time":_col(df,("clip_time_ns","oculo_clip_video_ns"),False),
        "native_frame":_col(df,("native_frame_index",),False),
        "clip_frame":_col(df,("clip_frame_index","scene_frame_index"),False),
    }


def _unit_scale(unit: str) -> float:
    u=unit.lower().strip()
    if u in {"m","metre","meter","metres","meters"}: return 1.0
    if u in {"mm","millimetre","millimeter","millimetres","millimeters"}: return 0.001
    raise SceneBundleError(f"Unsupported ray length unit: {unit}")


def _load_catalog(path: str | Path) -> dict[str,Any]:
    p=Path(path).expanduser().resolve()
    obj=json.loads(p.read_text("utf-8"))
    if obj.get("schema_version")!="oculostudio.motionstudio_scene_catalog.v1":
        raise SceneBundleError("Not an OculoStudio MotionStudio scene catalog")
    return obj


def import_scene_bundles(
    manifest_paths: Iterable[str | Path], *, project_root: str | Path, session_id: str,
    verify_checksums: bool=True, allow_incomplete: bool=False, progress: Progress=None, cancelled: Cancel=None,
) -> dict[str,Any]:
    paths=[Path(x).expanduser().resolve() for x in manifest_paths]
    if not paths: raise SceneBundleError("Select at least one MotionStudio *_oculostudio_scene.json")
    bundles=[]
    for i,p in enumerate(paths):
        _cancel(cancelled); _emit(progress,0.05+0.35*(i/max(len(paths),1)),f"Inspecting {p.name}")
        bundles.append(inspect_scene_bundle(p,verify_checksums=verify_checksums))
    pair=validate_scene_pair(bundles)
    if not allow_incomplete and not pair["ready"]:
        raise SceneBundleError("MotionStudio scene is not complete_for_oculostudio_mesh_aoi; inspect warnings before import")
    scene_id=_safe_name(pair["scene_id"]); artifact_id=_new_id("artifact")
    out=Path(project_root).expanduser().resolve()/"derived"/session_id/"motionstudio_scene"/scene_id/artifact_id
    out.mkdir(parents=True,exist_ok=False)
    actors=[]
    shared_props: dict[str,Any]|None=None
    seen_props=set()
    for b in bundles:
        actor={"role_id":b.role_id,"participant_id":b.participant_id,"actor_instance_id":b.actor_instance_id,"manifest_path":str(b.manifest_path),"manifest_sha256":sha256_file(b.manifest_path),"bundle":b.to_jsonable()}
        actors.append(actor)
        pp=b.linked_files.get("props_motion")
        if pp:
            key=b.props_fingerprint or sha256_file(pp)
            if key not in seen_props:
                seen_props.add(key)
                shared_props={"fingerprint":key,"revision":b.props_revision,"motion_path":str(pp),"manifest_path":str(b.linked_files.get("props_manifest","")),"source_actor_manifest":str(b.manifest_path)}
    catalog={
        "schema_version":"oculostudio.motionstudio_scene_catalog.v1",
        "created_at":_now_iso(),"scene_id":pair["scene_id"],"session_id":pair["session_id"] or session_id,
        "interaction_id":pair["interaction_id"],"actors":actors,"shared_props":shared_props,
        "complete_for_mesh_aoi":pair["ready"],"warnings":pair["warnings"],
        "source_manifest_paths":[str(p) for p in paths],
    }
    catalog_path=out/"motionstudio_scene_catalog.json"; _atomic_json(catalog_path,catalog)
    _atomic_json(out/"import_report.json",{"catalog":str(catalog_path),"actors":len(actors),"shared_props":shared_props,"warnings":pair["warnings"]})
    _emit(progress,1.0,"MotionStudio scene imported")
    return {"catalog":str(catalog_path),"output_dir":str(out),"scene_id":pair["scene_id"],"actors":actors,"shared_props":shared_props,"warnings":pair["warnings"]}


def _bundles_from_catalog(catalog: dict[str,Any])->list[SceneBundle]:
    return [inspect_scene_bundle(a["manifest_path"],verify_checksums=False) for a in catalog["actors"]]


def _matrix_for_bundle(bundle: SceneBundle, assume_same_world: bool)->np.ndarray:
    if bundle.coordinate_transform is not None:return np.asarray(bundle.coordinate_transform,dtype=np.float64)
    if assume_same_world:return np.eye(4,dtype=np.float64)
    raise SceneBundleError(f"{bundle.role_id} manifest lacks explicit source-Vicon→MotionStudio-world transform; enable an explicit identity assumption only after QC")


def _part_triangle_groups(ids:np.ndarray)->dict[int,np.ndarray]:
    return {int(x):np.where(ids==x)[0] for x in np.unique(ids)}


def annotate_body_mesh_aoi(
    *, scene_catalog_path: str | Path, gaze_rays_path: str | Path, project_root: str | Path,
    session_id: str, output_label: str, viewer_role: str, body_model_root: str | Path | None=None,
    ray_length_unit: str="m", max_time_error_ms: float=12.0, uncertainty_deg: float=1.5,
    max_distance_m: float=10.0, assume_same_world: bool=False, also_parquet: bool=True,
    motion_time_scale: float=1.0, motion_source_origin_ns: int=0, motion_target_origin_ns: int=0,
    progress: Progress=None, cancelled: Cancel=None,
) -> dict[str,Any]:
    catalog=_load_catalog(scene_catalog_path); bundles=_bundles_from_catalog(catalog)
    rays=_read_table(gaze_rays_path); c=_ray_columns(rays); scale=_unit_scale(ray_length_unit)
    providers=[]
    for i,b in enumerate(bundles):
        _emit(progress,0.03+0.12*i/max(len(bundles),1),f"Loading fitted body {b.role_id}")
        motion=load_body_motion(b); parts=load_bodyparts(b); provider=BodyMeshProvider(motion,parts,model_root=body_model_root)
        providers.append((b,motion,parts,provider,_part_triangle_groups(parts.face_part_ids),_matrix_for_bundle(b,assume_same_world)))
    out_root=Path(project_root).expanduser().resolve()/"derived"/session_id/"body_aoi_mesh"/_new_id("artifact")
    out_root.mkdir(parents=True,exist_ok=False)
    rows=[]; total=len(rays); max_err=int(max_time_error_ms*1e6); max_dist=max_distance_m
    for n in range(total):
        _cancel(cancelled)
        row=rays.iloc[n]; uid=str(row[c["uid"]]) if c["uid"] else f"row_{n}"; t=int(row[c["time"]])
        valid=True if c["valid"] is None else bool(row[c["valid"]])
        base={"gaze_sample_uid":uid,"source_gaze_row_index":int(row[c["source_row"]]) if c["source_row"] and pd.notna(row[c["source_row"]]) else n,"timestamp_ns":t,"clip_time_ns":int(row[c["clip_time"]]) if c["clip_time"] and pd.notna(row[c["clip_time"]]) else None,"native_frame_index":int(row[c["native_frame"]]) if c["native_frame"] and pd.notna(row[c["native_frame"]]) else None,"clip_frame_index":int(row[c["clip_frame"]]) if c["clip_frame"] and pd.notna(row[c["clip_frame"]]) else None,"ray_valid":valid}
        best=None; uncertainty_best=None
        if valid:
            origin=np.array([row[c["ox"]],row[c["oy"]],row[c["oz"]]],dtype=float)*scale
            direction=np.array([row[c["dx"]],row[c["dy"]],row[c["dz"]]],dtype=float); direction/=max(np.linalg.norm(direction),1e-12)
            for b,motion,parts,provider,groups,matrix in providers:
                motion_global_times = motion_target_origin_ns + np.rint((motion.source_times_ns.astype(np.int64) - int(motion_source_origin_ns)).astype(np.float64) * float(motion_time_scale)).astype(np.int64)
                match=nearest_timestamp_index(motion_global_times,t,max_err)
                if match is None:continue
                fi,time_error=match; verts=provider.vertices(fi)
                o=apply_transform_points(matrix,origin); d=apply_transform_directions(matrix,direction)
                # broad phase by semantic part, then exact triangles
                for pid,tri_ids in groups.items():
                    face_subset=parts.faces[tri_ids]; vids=np.unique(face_subset.reshape(-1)); pv=verts[vids]; lo=pv.min(0);hi=pv.max(0)
                    ok,_,_=ray_aabb_intersection(o,d,lo,hi)
                    if ok:
                        hit=intersect_ray_triangles(o,d,verts,parts.faces,tri_ids,max_distance=max_dist)
                        if hit and (best is None or hit.distance<best[0]):
                            pname=parts.part_names[pid] if 0<=pid<len(parts.part_names) else f"part_{pid}"
                            pc=float(parts.face_part_confidence[hit.triangle_index]) if parts.face_part_confidence is not None and hit.triangle_index<len(parts.face_part_confidence) else 1.0
                            best=(hit.distance,b,pname,hit,pc,time_error,fi)
                    if best is None and uncertainty_deg>0:
                        cent=verts[parts.faces[tri_ids]].mean(axis=1); ca=closest_ray_vertex_angle(o,d,cent)
                        if ca and ca[0]<=uncertainty_deg:
                            angle,perp,local=ca; along=float(np.dot(cent[local]-o,d))
                            cand=(angle,along,b,parts.part_names[pid] if pid<len(parts.part_names) else f"part_{pid}",cent[local],time_error,fi)
                            if uncertainty_best is None or (angle,along)<(uncertainty_best[0],uncertainty_best[1]):uncertainty_best=cand
        if best:
            dist,b,pname,hit,pc,time_error,fi=best; conf=max(0.,min(1.,pc*(1-time_error/max(max_err,1)))); exact=True; point=hit.point; angle=0.; status="exact_mesh_hit"
        elif uncertainty_best:
            angle,dist,b,pname,point,time_error,fi=uncertainty_best; conf=max(0.,min(.75,(1-angle/max(uncertainty_deg,1e-9))*.75*(1-time_error/max(max_err,1)))); exact=False; status="uncertainty_mesh_candidate"
        else:
            b=None;pname="no_body";dist=np.nan;point=np.array([np.nan]*3);conf=0.;exact=False;angle=np.nan;time_error=None;fi=None;status="invalid_ray" if not valid else "no_body_hit"
        base.update({"body_entity_id":b.actor_instance_id if b else None,"body_role_id":b.role_id if b else None,"target_relation":"self" if b and b.role_id==viewer_role else ("partner" if b else None),"body_region":pname,"intersection_x_m":float(point[0]),"intersection_y_m":float(point[1]),"intersection_z_m":float(point[2]),"intersection_distance_m":float(dist) if np.isfinite(dist) else np.nan,"body_distance_mm":float(dist*1000.0) if np.isfinite(dist) else np.nan,"body_3d_distance_mm":float(dist*1000.0) if np.isfinite(dist) else np.nan,"body_hit_x_mm":float(point[0]*1000.0),"body_hit_y_mm":float(point[1]*1000.0),"body_hit_z_mm":float(point[2]*1000.0),"exact_hit":exact,"body_exact_hit":exact,"angular_miss_deg":float(angle) if np.isfinite(angle) else np.nan,"body_angular_miss_deg":float(angle) if np.isfinite(angle) else np.nan,"body_confidence":conf,"geometry_backend":"motionstudio_smpl_mesh","source_motion_frame":fi,"source_time_error_ms":None if time_error is None else time_error/1e6,"assignment_status":status,"review_required":bool(conf<0.65 or status not in {"exact_mesh_hit"})})
        rows.append(base)
        if n%max(1,total//100)==0:_emit(progress,0.18+0.76*(n/max(total,1)),f"Mesh Body AOI {n+1}/{total}")
    df=pd.DataFrame(rows); files=_write_table(df,out_root/"body_aoi_samples.csv",also_parquet)
    frame_cols=[x for x in ("clip_frame_index","native_frame_index") if x in df.columns]
    if frame_cols:
        group=frame_cols[0]; summary=df.groupby(group,dropna=False).agg(gaze_sample_count=("gaze_sample_uid","count"),primary_body_region=("body_region",lambda s:s.value_counts().index[0] if len(s) else ""),mean_body_confidence=("body_confidence","mean"),review_required=("review_required","max")).reset_index(); files.update({f"frames_{k}":v for k,v in _write_table(summary,out_root/"body_aoi_frames.csv",also_parquet).items()})
    qc={"input_rows":total,"output_rows":len(df),"exact_hit_fraction":float(df.exact_hit.mean()) if len(df) else 0.,"assigned_fraction":float((df.body_region!="no_body").mean()) if len(df) else 0.,"review_fraction":float(df.review_required.mean()) if len(df) else 0.}
    manifest={"schema_version":"oculostudio.body_aoi.motionstudio_mesh.v1","created_at":_now_iso(),"scene_catalog":str(Path(scene_catalog_path).resolve()),"gaze_rays":str(Path(gaze_rays_path).resolve()),"viewer_role":viewer_role,"body_model_root":str(body_model_root) if body_model_root else None,"parameters":{"ray_length_unit":ray_length_unit,"max_time_error_ms":max_time_error_ms,"uncertainty_deg":uncertainty_deg,"max_distance_m":max_distance_m,"assume_same_world":assume_same_world,"motion_time_scale":motion_time_scale,"motion_source_origin_ns":motion_source_origin_ns,"motion_target_origin_ns":motion_target_origin_ns},"files":files,"qc":qc}
    _atomic_json(out_root/"body_aoi_manifest.json",manifest); _emit(progress,1.,"MotionStudio mesh Body AOI complete")
    return {"output_dir":str(out_root),"samples":files,"manifest":str(out_root/"body_aoi_manifest.json"),"qc":qc}


def annotate_object_mesh_aoi(
    *, scene_catalog_path:str|Path, gaze_rays_path:str|Path, project_root:str|Path, session_id:str,
    output_label:str, ray_length_unit:str="m", max_time_error_ms:float=12., uncertainty_deg:float=1.5,
    max_distance_m:float=10., assume_same_world:bool=False, also_parquet:bool=True,
    motion_time_scale:float=1.0, motion_source_origin_ns:int=0, motion_target_origin_ns:int=0,
    progress:Progress=None,cancelled:Cancel=None,
)->dict[str,Any]:
    catalog=_load_catalog(scene_catalog_path); bundles=_bundles_from_catalog(catalog)
    # Props are shared: choose first bundle that exposes them and reject conflicting fingerprints at import time.
    props=[]; source_bundle=None
    for b in bundles:
        loaded=load_props(b)
        if loaded:props=loaded;source_bundle=b;break
    if not props:raise SceneBundleError("Scene catalog has no MotionStudio shared collision-mesh props")
    matrix=_matrix_for_bundle(source_bundle,assume_same_world)
    rays=_read_table(gaze_rays_path);c=_ray_columns(rays);scale=_unit_scale(ray_length_unit);max_err=int(max_time_error_ms*1e6)
    out_root=Path(project_root).expanduser().resolve()/"derived"/session_id/"object_aoi_mesh"/_new_id("artifact");out_root.mkdir(parents=True,exist_ok=False)
    rows=[];total=len(rays)
    for n in range(total):
        _cancel(cancelled);row=rays.iloc[n];uid=str(row[c["uid"]]) if c["uid"] else f"row_{n}";t=int(row[c["time"]]);valid=True if c["valid"] is None else bool(row[c["valid"]])
        base={"gaze_sample_uid":uid,"source_gaze_row_index":int(row[c["source_row"]]) if c["source_row"] and pd.notna(row[c["source_row"]]) else n,"timestamp_ns":t,"clip_time_ns":int(row[c["clip_time"]]) if c["clip_time"] and pd.notna(row[c["clip_time"]]) else None,"native_frame_index":int(row[c["native_frame"]]) if c["native_frame"] and pd.notna(row[c["native_frame"]]) else None,"clip_frame_index":int(row[c["clip_frame"]]) if c["clip_frame"] and pd.notna(row[c["clip_frame"]]) else None,"ray_valid":valid}
        best=None;unc=None
        if valid:
            origin=np.array([row[c["ox"]],row[c["oy"]],row[c["oz"]]],float)*scale;direction=np.array([row[c["dx"]],row[c["dy"]],row[c["dz"]]],float);direction/=max(np.linalg.norm(direction),1e-12);o=apply_transform_points(matrix,origin);d=apply_transform_directions(matrix,direction)
            for prop in props:
                prop_global_times = motion_target_origin_ns + np.rint((prop.times_ns.astype(np.int64) - int(motion_source_origin_ns)).astype(np.float64) * float(motion_time_scale)).astype(np.int64)
                match=nearest_timestamp_index(prop_global_times,t,max_err)
                if match is None:continue
                fi,terr=match
                if fi>=len(prop.valid_mask) or not prop.valid_mask[fi]:continue
                verts=apply_transform_points(prop.transforms_world[fi],prop.collision_vertices);lo=verts.min(0);hi=verts.max(0);ok,_,_=ray_aabb_intersection(o,d,lo,hi)
                if ok:
                    hit=intersect_ray_triangles(o,d,verts,prop.collision_faces,max_distance=max_distance_m)
                    if hit and (best is None or hit.distance<best[0]):
                        tri=int(hit.triangle_index)
                        part_name=None
                        if prop.collision_face_part_ids is not None and tri < len(prop.collision_face_part_ids):
                            pid=int(prop.collision_face_part_ids[tri])
                            if prop.collision_part_names and 0 <= pid < len(prop.collision_part_names):
                                part_name=prop.collision_part_names[pid]
                            else:
                                part_name=f"part_{pid}"
                        best=(hit.distance,prop,hit,terr,fi,part_name)
                if best is None and uncertainty_deg>0:
                    cent=verts[prop.collision_faces].mean(1);ca=closest_ray_vertex_angle(o,d,cent)
                    if ca and ca[0]<=uncertainty_deg:
                        angle,perp,local=ca;along=float(np.dot(cent[local]-o,d))
                        part_name=None
                        if prop.collision_face_part_ids is not None and local < len(prop.collision_face_part_ids):
                            pid=int(prop.collision_face_part_ids[local])
                            if prop.collision_part_names and 0 <= pid < len(prop.collision_part_names):
                                part_name=prop.collision_part_names[pid]
                            else:
                                part_name=f"part_{pid}"
                        cand=(angle,along,prop,cent[local],terr,fi,part_name)
                        if unc is None or (angle,along)<(unc[0],unc[1]):unc=cand
        if best:
            dist,prop,hit,terr,fi,object_part=best;point=hit.point;angle=0.;exact=True;status="exact_collision_mesh_hit";base_conf=float(prop.confidence[fi]) if fi<len(prop.confidence) else 1.;conf=max(0.,min(1.,base_conf*(1-terr/max(max_err,1))))
        elif unc:
            angle,dist,prop,point,terr,fi,object_part=unc;exact=False;status="uncertainty_collision_mesh_candidate";base_conf=float(prop.confidence[fi]) if fi<len(prop.confidence) else 1.;conf=max(0.,min(.75,base_conf*.75*(1-angle/max(uncertainty_deg,1e-9))*(1-terr/max(max_err,1))))
        else:
            prop=None;object_part=None;dist=np.nan;point=np.array([np.nan]*3);angle=np.nan;exact=False;status="invalid_ray" if not valid else "no_object_hit";conf=0.;terr=None;fi=None
        base.update({"object_entity_id":prop.object_id if prop else None,"object_category":prop.category if prop else None,"object_part":object_part,"object_distance_m":float(dist) if np.isfinite(dist) else np.nan,"object_distance_mm":float(dist*1000.0) if np.isfinite(dist) else np.nan,"object_hit_x_m":float(point[0]),"object_hit_y_m":float(point[1]),"object_hit_z_m":float(point[2]),"object_hit_x_mm":float(point[0]*1000.0),"object_hit_y_mm":float(point[1]*1000.0),"object_hit_z_mm":float(point[2]*1000.0),"object_exact_hit":exact,"object_angular_miss_deg":float(angle) if np.isfinite(angle) else np.nan,"object_confidence":conf,"object_pose_rms_mm":float(prop.rigid_fit_rmse_mm[fi]) if prop and prop.rigid_fit_rmse_mm is not None and fi<len(prop.rigid_fit_rmse_mm) else np.nan,"visible_marker_count":int(prop.visible_marker_count[fi]) if prop and prop.visible_marker_count is not None and fi<len(prop.visible_marker_count) else None,"geometry_backend":"motionstudio_collision_mesh","source_motion_frame":fi,"source_time_error_ms":None if terr is None else terr/1e6,"assignment_status":status,"review_required":bool(conf<0.65 or status!="exact_collision_mesh_hit")})
        rows.append(base)
        if n%max(1,total//100)==0:_emit(progress,.12+.82*(n/max(total,1)),f"Collision-mesh Object AOI {n+1}/{total}")
    df=pd.DataFrame(rows);files=_write_table(df,out_root/"object_aoi_samples.csv",also_parquet)
    group="clip_frame_index" if "clip_frame_index" in df else None
    if group:
        summary=df.groupby(group,dropna=False).agg(gaze_sample_count=("gaze_sample_uid","count"),primary_object=("object_entity_id",lambda s:s.dropna().value_counts().index[0] if len(s.dropna()) else None),mean_object_confidence=("object_confidence","mean"),review_required=("review_required","max")).reset_index();files.update({f"frames_{k}":v for k,v in _write_table(summary,out_root/"object_aoi_frames.csv",also_parquet).items()})
    qc={"input_rows":total,"output_rows":len(df),"exact_hit_fraction":float(df.object_exact_hit.mean()) if len(df) else 0.,"assigned_fraction":float(df.object_entity_id.notna().mean()) if len(df) else 0.,"review_fraction":float(df.review_required.mean()) if len(df) else 0.,"object_count":len(props)}
    manifest={"schema_version":"oculostudio.object_aoi.motionstudio_collision_mesh.v1","created_at":_now_iso(),"scene_catalog":str(Path(scene_catalog_path).resolve()),"gaze_rays":str(Path(gaze_rays_path).resolve()),"parameters":{"ray_length_unit":ray_length_unit,"max_time_error_ms":max_time_error_ms,"uncertainty_deg":uncertainty_deg,"max_distance_m":max_distance_m,"assume_same_world":assume_same_world,"motion_time_scale":motion_time_scale,"motion_source_origin_ns":motion_source_origin_ns,"motion_target_origin_ns":motion_target_origin_ns},"files":files,"qc":qc}
    _atomic_json(out_root/"object_aoi_manifest.json",manifest);_emit(progress,1.,"MotionStudio collision-mesh Object AOI complete")
    return {"output_dir":str(out_root),"samples":files,"manifest":str(out_root/"object_aoi_manifest.json"),"qc":qc}
