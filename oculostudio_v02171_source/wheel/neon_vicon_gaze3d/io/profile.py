from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json
import uuid
import zipfile
from copy import deepcopy

import numpy as np


def _privacy_safe_identity_audit(identity: dict | None) -> dict | None:
    """Remove wearer/actor names and local paths from stored identity evidence."""

    if not identity:
        return identity
    if identity.get("privacy_redacted"):
        return deepcopy(identity)
    selected_index = identity.get("selected_candidate_index")
    candidates: list[dict] = []
    selected_hash = None
    for item in identity.get("candidates", []):
        candidate = {
            "candidate_index": item.get("candidate_index"),
            "sha256": item.get("sha256"),
            "marker_labels": list(item.get("marker_labels", [])),
            "frame_count": item.get("frame_count"),
            "frame_rate_hz": item.get("frame_rate_hz"),
            "valid_centroid_frames": item.get("valid_centroid_frames"),
            "camera_distance_mm": deepcopy(item.get("camera_distance_mm", {})),
            "camera_proximity_score_mm": item.get("camera_proximity_score_mm"),
            "camera_frames_under_250mm_fraction": item.get(
                "camera_frames_under_250mm_fraction"
            ),
            "camera_board_mapping_variant": item.get("camera_board_mapping_variant"),
            "camera_direct_pnp_rms_px": deepcopy(
                item.get("camera_direct_pnp_rms_px", {})
            ),
        }
        candidates.append(candidate)
        if item.get("candidate_index") == selected_index:
            selected_hash = item.get("sha256")
    return {
        "privacy_redacted": True,
        "identity_evidence_type": identity.get("identity_evidence_type", ""),
        "method": identity.get("method", ""),
        "confidence": identity.get("confidence", ""),
        "selected_candidate_index": selected_index,
        "selected_tracker_sha256": selected_hash,
        "selected_tracker_labels": list(identity.get("selected_tracker_labels", [])),
        "board_mapping_source": identity.get("board_mapping_source", ""),
        "recommended_board_corner_labels": deepcopy(
            identity.get("recommended_board_corner_labels", {})
        ),
        "recommended_outer_corner_labels": deepcopy(
            identity.get("recommended_outer_corner_labels", {})
        ),
        "head_qc_available": bool(identity.get("head_qc_available", False)),
        "candidates": candidates,
        "warnings": [
            "Raw wearer identity, C3D subject names, filenames, and local paths were redacted."
        ],
    }


class ProfileError(RuntimeError):
    pass


@dataclass
class CalibrationProfile:
    profile_id: str
    display_name: str
    canonical_participant_id: str
    created_utc: str
    module_serial_number: str
    frame_name: str
    source_recording_id: str
    tracker_marker_labels: list[str]
    tracker_template_mm: np.ndarray
    T_tracker_from_camera: np.ndarray
    scene_camera_matrix: np.ndarray
    scene_distortion_coefficients: np.ndarray
    format_version: int = 2
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        display_name: str,
        canonical_participant_id: str,
        module_serial_number: str,
        frame_name: str,
        source_recording_id: str,
        tracker_marker_labels: list[str],
        tracker_template_mm: np.ndarray,
        T_tracker_from_camera: np.ndarray,
        scene_camera_matrix: np.ndarray,
        scene_distortion_coefficients: np.ndarray,
        metadata: dict | None = None,
        metrics: dict | None = None,
        warnings: list[str] | None = None,
    ) -> "CalibrationProfile":
        return cls(
            profile_id=str(uuid.uuid4()),
            display_name=display_name,
            canonical_participant_id=canonical_participant_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
            module_serial_number=str(module_serial_number),
            frame_name=str(frame_name),
            source_recording_id=str(source_recording_id),
            tracker_marker_labels=list(tracker_marker_labels),
            tracker_template_mm=np.asarray(tracker_template_mm, dtype=float),
            T_tracker_from_camera=np.asarray(T_tracker_from_camera, dtype=float),
            scene_camera_matrix=np.asarray(scene_camera_matrix, dtype=float),
            scene_distortion_coefficients=np.asarray(
                scene_distortion_coefficients, dtype=float
            ).reshape(-1),
            format_version=2,
            metadata=metadata or {},
            metrics=metrics or {},
            warnings=warnings or [],
        )

    @property
    def R_tracker_from_camera(self) -> np.ndarray:
        return self.T_tracker_from_camera[:3, :3]

    @property
    def camera_origin_tracker_mm(self) -> np.ndarray:
        return self.T_tracker_from_camera[:3, 3]

    @property
    def calibration_estimator_id(self) -> str:
        return str(self.metadata.get("calibration_estimator_id", "legacy_unknown"))

    @property
    def is_robust_v2_profile(self) -> bool:
        return (
            int(self.format_version) >= 2
            and self.calibration_estimator_id
            == "robust_multiframe_se3_geometric_median_v2"
        )

    def summary(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "canonical_participant_id": self.canonical_participant_id,
            "created_utc": self.created_utc,
            "module_serial_number": self.module_serial_number,
            "frame_name": self.frame_name,
            "source_recording_id": self.source_recording_id,
            "tracker_marker_labels": self.tracker_marker_labels,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "format": "NeonVicon-Gaze3D-CalibrationProfile",
            "format_version": int(self.format_version),
        }

    def audit_summary(
        self,
        *,
        include_raw_identity_metadata: bool = False,
        canonical_participant_id_override: str = "",
    ) -> dict:
        """Return profile provenance suitable for a processing manifest.

        By default this deliberately omits profile display names, profile-stored
        participant names, raw wearer/subject evidence, and local source paths.
        The calibrated numeric transform and all QC metrics remain available.
        """

        if include_raw_identity_metadata:
            return self.summary()
        metadata = deepcopy(self.metadata)
        metadata["tracker_identity_detection"] = _privacy_safe_identity_audit(
            metadata.get("tracker_identity_detection")
        )
        if "source_files" in metadata:
            source_files = metadata.get("source_files") or {}
            metadata["source_files"] = {
                key: (None if value is None else "<redacted; see source_sha256>")
                for key, value in source_files.items()
            }
        metadata.setdefault("privacy", {})["raw_identity_metadata_included"] = False
        return {
            "profile_id": self.profile_id,
            "display_name": "<redacted>",
            "canonical_participant_id": canonical_participant_id_override,
            "created_utc": self.created_utc,
            "module_serial_number": self.module_serial_number,
            "frame_name": self.frame_name,
            "source_recording_id": self.source_recording_id,
            "tracker_marker_labels": list(self.tracker_marker_labels),
            "metadata": metadata,
            "metrics": deepcopy(self.metrics),
            "warnings": list(self.warnings),
            "format": "NeonVicon-Gaze3D-CalibrationProfile",
            "format_version": int(self.format_version),
        }

    def save(self, path: str | Path, qc_files: list[Path] | None = None) -> Path:
        path = Path(path)
        if path.suffix.lower() != ".g3dcal":
            path = path.with_suffix(".g3dcal")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = BytesIO()
        np.savez_compressed(
            arrays,
            tracker_template_mm=self.tracker_template_mm,
            T_tracker_from_camera=self.T_tracker_from_camera,
            scene_camera_matrix=self.scene_camera_matrix,
            scene_distortion_coefficients=self.scene_distortion_coefficients,
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "profile.json",
                json.dumps(self.summary(), ensure_ascii=False, indent=2),
            )
            archive.writestr("arrays.npz", arrays.getvalue())
            for qc_path in qc_files or []:
                if qc_path.is_file():
                    archive.write(qc_path, f"qc/{qc_path.name}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        path = Path(path)
        if not path.is_file():
            raise ProfileError(f"Calibration profile does not exist: {path}")
        with zipfile.ZipFile(path) as archive:
            try:
                summary = json.loads(archive.read("profile.json").decode("utf-8"))
                arrays_bytes = BytesIO(archive.read("arrays.npz"))
            except KeyError as exc:
                raise ProfileError("Invalid .g3dcal profile") from exc
        arrays = np.load(arrays_bytes)
        return cls(
            profile_id=summary["profile_id"],
            display_name=summary.get("display_name", path.stem),
            canonical_participant_id=summary.get("canonical_participant_id", ""),
            created_utc=summary.get("created_utc", ""),
            module_serial_number=str(summary.get("module_serial_number", "")),
            frame_name=str(summary.get("frame_name", "")),
            source_recording_id=str(summary.get("source_recording_id", "")),
            tracker_marker_labels=list(summary["tracker_marker_labels"]),
            tracker_template_mm=np.asarray(arrays["tracker_template_mm"], dtype=float),
            T_tracker_from_camera=np.asarray(
                arrays["T_tracker_from_camera"], dtype=float
            ),
            scene_camera_matrix=np.asarray(arrays["scene_camera_matrix"], dtype=float),
            scene_distortion_coefficients=np.asarray(
                arrays["scene_distortion_coefficients"], dtype=float
            ),
            format_version=int(summary.get("format_version", 1)),
            metadata=summary.get("metadata", {}),
            metrics=summary.get("metrics", {}),
            warnings=list(summary.get("warnings", [])),
        )

    def extract_qc(self, profile_path: str | Path, destination: str | Path) -> list[Path]:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        with zipfile.ZipFile(profile_path) as archive:
            for name in archive.namelist():
                if name.startswith("qc/") and not name.endswith("/"):
                    target = destination / Path(name).name
                    target.write_bytes(archive.read(name))
                    extracted.append(target)
        return extracted
