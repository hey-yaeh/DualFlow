"""Targeted validation for OculoStudio v0.21.71 Body AOI resolution fallback."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import oculostudio
from oculostudio.errors import ValidationError
from oculostudio.services import body_aoi


def _artifact_with_manifest(root: Path, payload: dict):
    path = root / "trim_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return SimpleNamespace(id="artifact_trim_test", files=[SimpleNamespace(path=str(path))])


def main() -> int:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="oculo_body_resolution_") as temp:
        root = Path(temp)
        trim = _artifact_with_manifest(
            root,
            {"native_reference": {"width": 1600, "height": 1200, "decoded_frames": 30}},
        )
        recovered = body_aoi._native_trim_scene_resolution(trim)
        checks["native_trim_manifest_resolution"] = bool(
            recovered
            and recovered[0] == 1600
            and recovered[1] == 1200
            and recovered[2].get("source") == "native_reference_trim_manifest"
            and recovered[2].get("required_for_body_intersection") is False
        )

        original = body_aoi._scene_resolution
        try:
            body_aoi._scene_resolution = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValidationError("no calibration resolution and no Cloud scene video")
            )
            width, height, provenance = body_aoi._body_scene_resolution(None, None, None, trim)
            checks["trim_resolution_prevents_legacy_failure"] = (
                width == 1600
                and height == 1200
                and provenance.get("source") == "native_reference_trim_manifest"
            )

            width, height, provenance = body_aoi._body_scene_resolution(None, None, None, None)
            checks["missing_resolution_is_nonblocking"] = (
                width is None
                and height is None
                and provenance.get("source") == "unavailable_nonblocking"
                and provenance.get("required_for_body_intersection") is False
                and "no calibration resolution" in provenance.get("reason", "")
            )
        finally:
            body_aoi._scene_resolution = original

        body_aoi._scene_resolution = lambda *_args, **_kwargs: (
            1920,
            1080,
            {"source": "calibration_profile"},
        )
        try:
            width, height, provenance = body_aoi._body_scene_resolution(None, None, None, None)
            checks["legacy_resolution_still_used_when_available"] = (
                width == 1920
                and height == 1080
                and provenance.get("source") == "calibration_profile"
                and provenance.get("required_for_body_intersection") is False
            )
        finally:
            body_aoi._scene_resolution = original

    source = Path(body_aoi.__file__).read_text(encoding="utf-8")
    checks["body_job_uses_nonblocking_resolver"] = (
        "_body_scene_resolution(" in source
        and "Scene-video resolution unavailable; Body AOI will still use" in source
        and '"scene_resolution_required_for_body_intersection": False' in source
    )
    checks["installed_version"] = oculostudio.__version__ == "0.21.71"

    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({"release": "OculoStudio v0.21.71", "checks": checks, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
