from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import numpy as np

from .contracts import SceneBundle, SceneBundleError
from .geometry import topology_sha256
from .npzio import load_npz, first, scalar_text, string_list


@dataclass
class BodyParts:
    faces: np.ndarray
    face_part_ids: np.ndarray
    part_names: list[str]
    face_part_confidence: np.ndarray | None
    topology_hash: str


@dataclass
class BodyMotion:
    model_type: str
    gender: str
    betas: np.ndarray
    global_orient: np.ndarray | None
    body_pose: np.ndarray | None
    left_hand_pose: np.ndarray | None
    right_hand_pose: np.ndarray | None
    transl: np.ndarray | None
    faces: np.ndarray
    source_times_ns: np.ndarray
    source_frame_indices: np.ndarray
    vertices_world: np.ndarray | None = None
    topology_hash: str = ""
    raw: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def load_bodyparts(bundle: SceneBundle) -> BodyParts:
    p = bundle.linked_files.get("bodyparts_npz")
    if p is None: raise SceneBundleError("Scene bundle has no bodyparts NPZ")
    d = load_npz(p)
    faces = np.asarray(first(d, ("faces", "triangles")), dtype=np.int64)
    ids = np.asarray(first(d, ("face_part_ids", "triangle_part_ids")), dtype=np.int64).reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3: raise SceneBundleError("bodyparts faces must be [F,3]")
    if ids.shape[0] != faces.shape[0]: raise SceneBundleError("face_part_ids length does not match faces")
    names = string_list(first(d, ("part_names", "body_part_names")))
    if not names:
        embedded = first(d, ("part_names_json",))
        if embedded is not None:
            try:
                parsed = json.loads(scalar_text(embedded, "[]"))
                if isinstance(parsed, list):
                    names = [str(x) for x in parsed]
                elif isinstance(parsed, dict):
                    names = [str(parsed.get(str(i), parsed.get(i, f"part_{i}"))) for i in range(int(ids.max()) + 1)]
            except Exception:
                names = []
    if not names:
        embedded_manifest = first(d, ("manifest_json",))
        if embedded_manifest is not None:
            try:
                obj = json.loads(scalar_text(embedded_manifest, "{}"))
                raw_parts = obj.get("parts") or []
                if isinstance(raw_parts, list):
                    by_id = {int(x["id"]): str(x["name"]) for x in raw_parts if isinstance(x, dict) and "id" in x and "name" in x}
                    names = [by_id.get(i, f"part_{i}") for i in range(int(ids.max()) + 1)]
            except Exception:
                names = []
    if not names:
        jp = bundle.linked_files.get("bodyparts_json")
        if jp:
            obj = json.loads(jp.read_text("utf-8"))
            names = obj.get("part_names") or obj.get("body_part_names") or []
            if not names and isinstance(obj.get("parts"), list):
                by_id = {int(x["id"]): str(x["name"]) for x in obj["parts"] if isinstance(x, dict) and "id" in x and "name" in x}
                names = [by_id.get(i, f"part_{i}") for i in range(int(ids.max()) + 1)]
            if isinstance(names, dict):
                names = [str(names.get(str(i), names.get(i, f"part_{i}"))) for i in range(int(ids.max()) + 1)]
    if not names:
        names = [f"part_{i}" for i in range(int(ids.max()) + 1)]
    conf = first(d, ("face_part_confidence", "triangle_part_confidence"))
    conf = None if conf is None else np.asarray(conf, dtype=np.float32).reshape(-1)
    declared = scalar_text(first(d, ("topology_sha256", "topology_hash")), "")
    actual = topology_sha256(faces)
    # MotionStudio may hash the original integer dtype. Preserve its declared
    # cross-file fingerprint and verify topology structurally in the provider.
    return BodyParts(faces, ids, names, conf, declared or actual)


def _times_ns(data: dict[str, np.ndarray], frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    times = first(data, ("global_time_ns", "source_timestamps_ns", "timestamps_ns"))
    if times is None:
        seconds = first(data, ("source_timestamps_s", "output_timestamps_s", "timestamps_s"))
        if seconds is not None:
            times = np.rint(np.asarray(seconds, dtype=np.float64) * 1e9).astype(np.int64)
    if times is None:
        fps = float(np.asarray(first(data, ("source_fps", "fps"), np.array([0.0]))).reshape(-1)[0])
        if fps <= 0: raise SceneBundleError("Body motion has no source timestamps or valid FPS")
        times = np.rint(np.arange(frame_count, dtype=np.float64) / fps * 1e9).astype(np.int64)
    frames = first(data, ("source_frame_indices", "output_frame_indices", "frame_indices"))
    if frames is None: frames=np.arange(frame_count,dtype=np.int64)
    return np.asarray(times,dtype=np.int64).reshape(-1), np.asarray(frames,dtype=np.int64).reshape(-1)


def load_body_motion(bundle: SceneBundle) -> BodyMotion:
    p=bundle.linked_files.get("body_motion")
    if p is None: raise SceneBundleError("Scene bundle has no body motion NPZ")
    d=load_npz(p)
    vertices=first(d,("vertices_world","verts_world","body_vertices_world"))
    vertices=None if vertices is None else np.asarray(vertices,dtype=np.float32)
    transl=first(d,("transl","translation","root_translation"))
    go=first(d,("global_orient","root_orient"))
    bp=first(d,("body_pose",))
    lhp=first(d,("left_hand_pose","left_hand"))
    rhp=first(d,("right_hand_pose","right_hand"))
    frame_count = int(vertices.shape[0] if vertices is not None else np.asarray(transl if transl is not None else go if go is not None else bp).shape[0])
    times,frames=_times_ns(d,frame_count)
    if len(times)!=frame_count: raise SceneBundleError("Body source timestamp count does not match motion frame count")
    faces=np.asarray(first(d,("faces","triangles")),dtype=np.int64)
    return BodyMotion(
        model_type=scalar_text(first(d,("model_type",)),"smplh").lower().replace("-",""),
        gender=scalar_text(first(d,("gender",)),"neutral").lower(),
        betas=np.asarray(first(d,("betas","shape"),np.zeros(10)),dtype=np.float32),
        global_orient=None if go is None else np.asarray(go,dtype=np.float32),
        body_pose=None if bp is None else np.asarray(bp,dtype=np.float32),
        left_hand_pose=None if lhp is None else np.asarray(lhp,dtype=np.float32),
        right_hand_pose=None if rhp is None else np.asarray(rhp,dtype=np.float32),
        transl=None if transl is None else np.asarray(transl,dtype=np.float32),
        faces=faces,
        source_times_ns=times,
        source_frame_indices=frames,
        vertices_world=vertices,
        topology_hash=scalar_text(first(d,("topology_sha256","topology_hash","body_part_topology_sha256")), topology_sha256(faces)),
        raw=d,
    )


class BodyMeshProvider:
    def __init__(self, motion: BodyMotion, parts: BodyParts, model_root: str | Path | None = None, device: str = "cpu", chunk_size: int = 128):
        self.motion=motion; self.parts=parts; self.model_root=Path(model_root).expanduser() if model_root else None
        self.device=device; self.chunk_size=int(chunk_size); self._cache: dict[int,np.ndarray]={}; self._model=None
        if motion.faces.shape != parts.faces.shape or not np.array_equal(np.asarray(motion.faces,dtype=np.int64), np.asarray(parts.faces,dtype=np.int64)):
            raise SceneBundleError("Body motion faces do not match Body Parts faces")
        if motion.topology_hash and parts.topology_hash and motion.topology_hash != parts.topology_hash:
            # If the structural arrays are identical, allow differing legacy
            # dtype hashes but retain a visible warning in the output manifest.
            self.topology_hash_warning = True
        else:
            self.topology_hash_warning = False
        if motion.vertices_world is None and self.model_root is None:
            raise SceneBundleError("Body motion contains parameters only; configure Body Model Root or export vertices_world cache")

    @property
    def frame_count(self): return len(self.motion.source_times_ns)

    def _create_model(self):
        try:
            import torch
            import smplx
        except Exception as exc:
            raise SceneBundleError("SMPL parameter reconstruction requires the optional 'smplx' and 'torch' packages") from exc
        model_type=self.motion.model_type
        if model_type not in {"smpl","smplh","smplx"}: model_type="smplh"
        num_betas = int(np.asarray(self.motion.betas).shape[-1]) if np.asarray(self.motion.betas).ndim else 10
        try:
            self._model = smplx.create(
                str(self.model_root),
                model_type=model_type,
                gender=self.motion.gender,
                use_pca=False,
                flat_hand_mean=True,
                num_betas=num_betas,
                batch_size=self.chunk_size,
            ).to(self.device)
        except Exception as exc:
            raise SceneBundleError(
                f"Could not create {model_type.upper()} {self.motion.gender} model with num_betas={num_betas} "
                f"from Body Model Root {self.model_root}: {exc}"
            ) from exc
        model_faces = np.asarray(getattr(self._model, "faces", []), dtype=np.int64)
        if model_faces.shape != self.motion.faces.shape or not np.array_equal(model_faces, np.asarray(self.motion.faces, dtype=np.int64)):
            raise SceneBundleError(
                "The selected SMPL model topology does not match MotionStudio body_motion.faces; "
                "check model type, gender, release and Body Model Root"
            )
        return self._model

    def vertices(self,index:int)->np.ndarray:
        i=int(index)
        if self.motion.vertices_world is not None: return np.asarray(self.motion.vertices_world[i],dtype=np.float64)
        if i in self._cache: return self._cache[i]
        try: import torch
        except Exception as exc: raise SceneBundleError("PyTorch is required for parametric body reconstruction") from exc
        model=self._model or self._create_model()
        start=(i//self.chunk_size)*self.chunk_size; end=min(start+self.chunk_size,self.frame_count); n=end-start
        def tensor(x,shape_last=None):
            if x is None:return None
            a=np.asarray(x[start:end],dtype=np.float32)
            return torch.from_numpy(a).to(self.device)
        kwargs={"global_orient":tensor(self.motion.global_orient),"body_pose":tensor(self.motion.body_pose),"transl":tensor(self.motion.transl),"return_verts":True}
        if self.motion.left_hand_pose is not None: kwargs["left_hand_pose"]=tensor(self.motion.left_hand_pose)
        if self.motion.right_hand_pose is not None: kwargs["right_hand_pose"]=tensor(self.motion.right_hand_pose)
        betas=np.asarray(self.motion.betas,dtype=np.float32)
        if betas.ndim==1: betas=np.broadcast_to(betas[None,:],(n,betas.shape[0])).copy()
        elif betas.shape[0]!=n:
            if betas.shape[0]>=end: betas=betas[start:end]
            else: betas=np.broadcast_to(betas.reshape(1,-1),(n,betas.size)).copy()
        kwargs["betas"]=torch.from_numpy(betas).to(self.device)
        with torch.no_grad(): verts=model(**kwargs).vertices.detach().cpu().numpy()
        for j,v in enumerate(verts,start=start): self._cache[j]=np.asarray(v,dtype=np.float64)
        return self._cache[i]
