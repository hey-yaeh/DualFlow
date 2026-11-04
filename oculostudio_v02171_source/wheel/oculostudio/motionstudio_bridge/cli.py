from __future__ import annotations
import argparse,json
from .contracts import inspect_scene_bundle
from .service import import_scene_bundles,annotate_body_mesh_aoi,annotate_object_mesh_aoi

def main(argv=None):
    p=argparse.ArgumentParser(prog="oculostudio-motionstudio")
    s=p.add_subparsers(dest="cmd",required=True)
    q=s.add_parser("inspect");q.add_argument("manifest")
    q=s.add_parser("import");q.add_argument("manifests",nargs="+");q.add_argument("--project-root",required=True);q.add_argument("--session",required=True);q.add_argument("--allow-incomplete",action="store_true")
    for name in ("body-aoi","object-aoi"):
        q=s.add_parser(name);q.add_argument("--catalog",required=True);q.add_argument("--rays",required=True);q.add_argument("--project-root",required=True);q.add_argument("--session",required=True);q.add_argument("--label",required=True);q.add_argument("--model-root");q.add_argument("--viewer-role",default="R01");q.add_argument("--assume-same-world",action="store_true")
    a=p.parse_args(argv)
    if a.cmd=="inspect":r=inspect_scene_bundle(a.manifest).to_jsonable()
    elif a.cmd=="import":r=import_scene_bundles(a.manifests,project_root=a.project_root,session_id=a.session,allow_incomplete=a.allow_incomplete)
    elif a.cmd=="body-aoi":r=annotate_body_mesh_aoi(scene_catalog_path=a.catalog,gaze_rays_path=a.rays,project_root=a.project_root,session_id=a.session,output_label=a.label,viewer_role=a.viewer_role,body_model_root=a.model_root,assume_same_world=a.assume_same_world)
    else:r=annotate_object_mesh_aoi(scene_catalog_path=a.catalog,gaze_rays_path=a.rays,project_root=a.project_root,session_id=a.session,output_label=a.label,assume_same_world=a.assume_same_world)
    print(json.dumps(r,indent=2,default=str));return 0
if __name__=="__main__":raise SystemExit(main())
