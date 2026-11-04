from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from oculostudio.domain.models import ArtifactRecord
from oculostudio.errors import ValidationError


def existing_artifact_paths(artifact: ArtifactRecord) -> list[Path]:
    return [Path(row.path) for row in artifact.files if Path(row.path).exists()]


def find_artifact_file(
    artifact: ArtifactRecord,
    *,
    names: Iterable[str] = (),
    suffixes: Iterable[str] = (),
    contains: Iterable[str] = (),
    required: bool = True,
) -> Path | None:
    name_set = {str(x).lower() for x in names}
    suffix_set = {str(x).lower() for x in suffixes}
    contains_set = tuple(str(x).lower() for x in contains)
    candidates: list[Path] = []
    for path in existing_artifact_paths(artifact):
        low_name = path.name.lower()
        if name_set and low_name not in name_set:
            continue
        if suffix_set and path.suffix.lower() not in suffix_set:
            continue
        if contains_set and not all(token in low_name for token in contains_set):
            continue
        candidates.append(path)
    if not candidates:
        if required:
            criteria = f"names={sorted(name_set)}, suffixes={sorted(suffix_set)}, contains={contains_set}"
            raise ValidationError(f"Artifact {artifact.id} contains no matching file ({criteria})")
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0]


def artifact_root(artifact: ArtifactRecord) -> Path:
    paths = existing_artifact_paths(artifact)
    if not paths:
        raise ValidationError(f"Artifact {artifact.id} has no available files")
    common = Path(os.path.commonpath([str(path) for path in paths]))
    return common if common.is_dir() else common.parent


def find_video_file(artifact: ArtifactRecord, *, preferred_tokens: Iterable[str] = ()) -> Path:
    videos = [p for p in existing_artifact_paths(artifact) if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
    if not videos:
        raise ValidationError(f"Artifact {artifact.id} contains no video file")
    tokens = tuple(str(x).lower() for x in preferred_tokens)
    if tokens:
        preferred = [p for p in videos if all(token in p.name.lower() for token in tokens)]
        if preferred:
            videos = preferred
    videos.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return videos[0]
