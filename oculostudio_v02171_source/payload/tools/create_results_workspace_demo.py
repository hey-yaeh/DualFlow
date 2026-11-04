from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from oculostudio.demo import create_demo_project
from oculostudio.domain.models import ArtifactKind
from oculostudio.services.results_workspace import create_results_package
from oculostudio.storage import ArtifactStore, ProjectStore


def _register_demo_artifact(
    store: ProjectStore,
    session_id: str,
    kind: ArtifactKind,
    label: str,
    *,
    parent_ids: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
):
    root = store.derived_dir(session_id, kind.value, label)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps({"label": label, "kind": kind.value}, indent=2), "utf-8"
    )
    (root / "samples.csv").write_text("sample_id,value\n1,demo\n", "utf-8")
    return ArtifactStore(store).register(
        session_id=session_id,
        kind=kind,
        label=label,
        output_root=root,
        parent_artifact_ids=parent_ids or [],
        parameters=parameters or {},
    )


def build(output: Path, *, overwrite: bool = False) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() and overwrite:
        shutil.rmtree(output)
    result = create_demo_project(output, overwrite=False, run_processing=False)
    store = ProjectStore.open(output)
    session_id = result["session_id"]
    chains: dict[str, list[Any]] = {}
    for role in ("R02", "R01"):
        trim = _register_demo_artifact(
            store, session_id, ArtifactKind.NATIVE_REFERENCE_TRIM, f"S001_I01_{role}_trim"
        )
        filtered = _register_demo_artifact(
            store,
            session_id,
            ArtifactKind.FACE_MAPPER_FILTER,
            f"S001_I01_{role}_trim_facemapper_filter",
            parent_ids=[trim.id],
        )
        privacy = _register_demo_artifact(
            store,
            session_id,
            ArtifactKind.PRIVACY_CLIP,
            f"S001_I01_{role}_trim_privacy_30fps",
            parent_ids=[filtered.id],
        )
        association = _register_demo_artifact(
            store,
            session_id,
            ArtifactKind.PARTICIPANT_ASSOCIATION,
            f"S001_I01_{role}_participant_association",
            parent_ids=[filtered.id],
            parameters={
                "ego_wearer_role": role,
                "partner_role": "R01" if role == "R02" else "R02",
            },
        )
        face = _register_demo_artifact(
            store,
            session_id,
            ArtifactKind.FACE_AOI,
            f"S001_I01_{role}_face_aoi",
            parent_ids=[privacy.id, association.id],
        )
        chains[role] = [trim, filtered, privacy, association, face]

    dyadic = _register_demo_artifact(
        store,
        session_id,
        ArtifactKind.DYADIC_ATTENTION,
        "S001_I01_R01_R02_dyadic_attention",
        parent_ids=[chains["R01"][-1].id, chains["R02"][-1].id],
        parameters={"role_a": "R01", "role_b": "R02"},
    )
    review = _register_demo_artifact(
        store,
        session_id,
        ArtifactKind.AOI_REVIEW_VIDEO,
        "S001_I01_R01_face_review",
        parent_ids=[chains["R01"][-1].id],
    )
    selected = [
        artifact.id
        for artifact in store.project.sessions[session_id].artifacts.values()
        if artifact.kind not in {ArtifactKind.RESULTS_PACKAGE}
    ]
    package = create_results_package(
        store,
        session_id=session_id,
        artifact_ids=selected,
        session_code="S001",
        interaction_code="I01",
        role_code="R01",
        label="S001_I01_R01_results",
        copy_mode="index_only",
    )
    return {
        **result,
        "role_chains": {role: [row.id for row in rows] for role, rows in chains.items()},
        "dyadic_artifact_id": dyadic.id,
        "latest_review_artifact_id": review.id,
        "results_package": package,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.output), overwrite=args.overwrite), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
