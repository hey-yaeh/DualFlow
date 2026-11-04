from __future__ import annotations

from pathlib import Path
from typing import Any

from oculostudio import __version__
from oculostudio.domain.models import ArtifactFile, ArtifactKind, ArtifactRecord
from oculostudio.storage.project_store import ProjectStore
from oculostudio.util import file_inventory, new_id, sha256_file, utc_now


class ArtifactStore:
    def __init__(self, project_store: ProjectStore):
        self.project_store = project_store

    def register(
        self,
        *,
        session_id: str,
        kind: ArtifactKind,
        label: str,
        output_root: str | Path,
        source_asset_ids: list[str] | None = None,
        parent_artifact_ids: list[str] | None = None,
        clip_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        qc: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRecord:
        output_root = Path(output_root)
        if not output_root.exists():
            raise FileNotFoundError(f"Artifact output does not exist: {output_root}")
        artifact_id = artifact_id or new_id("artifact")
        files: list[ArtifactFile] = []
        if output_root.is_file():
            files.append(
                ArtifactFile(
                    role="primary",
                    path=str(output_root),
                    size_bytes=output_root.stat().st_size,
                    sha256=sha256_file(output_root),
                )
            )
        elif output_root.exists():
            for row in file_inventory(output_root):
                files.append(
                    ArtifactFile(
                        role="primary" if len(files) == 0 else "support",
                        path=str(output_root / row["path"]),
                        size_bytes=row["size_bytes"],
                        sha256=row["sha256"],
                    )
                )
        artifact = ArtifactRecord(
            id=artifact_id,
            kind=kind,
            label=label,
            session_id=session_id,
            source_asset_ids=source_asset_ids or [],
            parent_artifact_ids=parent_artifact_ids or [],
            clip_id=clip_id,
            files=files,
            parameters=parameters or {},
            qc=qc or {},
            provenance={
                "producer": "OculoStudio CleanCore",
                "producer_version": __version__,
                "created_at": utc_now(),
                **(provenance or {}),
            },
        )

        def mutate(project):
            session = project.sessions[session_id]
            if artifact.id in session.artifacts:
                raise ValueError(f"Artifact ID already exists: {artifact.id}")
            session.artifacts[artifact.id] = artifact

        self.project_store.update(mutate)
        return artifact
