"""MotionStudio Scene Bundle bridge for OculoStudio.

This package reads MotionStudio's explicit Scene Bundle contract, validates its
linked files, and provides mesh/collision-mesh AOI backends. It deliberately
keeps the legacy C3D primitive backends available elsewhere in OculoStudio.
"""
from .contracts import SceneBundle, SceneBundleError, inspect_scene_bundle
from .service import import_scene_bundles, annotate_body_mesh_aoi, annotate_object_mesh_aoi

__all__ = [
    "SceneBundle", "SceneBundleError", "inspect_scene_bundle",
    "import_scene_bundles", "annotate_body_mesh_aoi", "annotate_object_mesh_aoi",
]
