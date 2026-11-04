from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from oculostudio.domain.models import ArtifactKind, TimeDomain
from oculostudio.services.dyadic_attention import create_dyadic_attention_annotations
from oculostudio.services.synchronization import fit_and_store_time_map
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.storage.project_store import ProjectStore


def _target_artifact(store: ProjectStore, sid: str, root: Path, role: str, base_ns: int):
    root.mkdir(parents=True, exist_ok=True)
    local = np.arange(0, 4_000_000_000, 20_000_000, dtype=np.int64)
    times = local + base_ns
    entities: list[str | None] = []
    domains: list[str] = []
    regions: list[str] = []
    confidences: list[float] = []
    for t in local:
        if t < 900_000_000:
            entities.append("R02" if role == "R01" else "R01")
            domains.append("face")
            regions.append("left_eye" if (t // 100_000_000) % 2 == 0 else "right_eye")
            confidences.append(0.93)
        elif 1_300_000_000 <= t < 2_700_000_000:
            entities.append("cup_01")
            domains.append("object")
            regions.append("cup")
            confidences.append(0.89)
        elif 3_000_000_000 <= t < 3_500_000_000:
            entities.append("R03")
            domains.append("face")
            regions.append("other_face")
            confidences.append(0.80)
        else:
            entities.append(None)
            domains.append("unknown")
            regions.append("unknown")
            confidences.append(0.20)
    pd.DataFrame({
        "gaze_sample_uid": [f"{role}_{i:05d}" for i in range(len(times))],
        "timestamp_ns": times,
        "native_frame_index": np.floor(local / (1e9 / 30)).astype(int),
        "final_entity_id": entities,
        "final_target_domain": domains,
        "final_target_region": regions,
        "final_confidence": confidences,
        "review_required": [False if c >= .5 else True for c in confidences],
    }).to_csv(root / "gaze_targets_full.csv", index=False)
    return ArtifactStore(store).register(
        session_id=sid,
        kind=ArtifactKind.TARGET_FUSION,
        label=f"{role} Face + Body + Object targets",
        output_root=root,
        qc={"known_target_fraction": float(np.mean(np.asarray(domains) != "unknown")), "object_target_fraction": float(np.mean(np.asarray(domains) == "object"))},
        provenance={"synthetic": True, "role_id": role},
    )


def build(output: Path, *, overwrite: bool = False) -> dict:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    store = ProjectStore.create(output, "OculoStudio Dyadic Attention Demo", session_label="Session S001")
    sid = store.project.selected_session_id
    session = store.project.sessions[sid]
    domain_a = TimeDomain(label="R01 Pupil clock", clock_type="absolute", metadata={"role_id": "R01"})
    domain_b = TimeDomain(label="R02 Pupil clock", clock_type="absolute", metadata={"role_id": "R02"})
    store.update(lambda p: p.sessions[sid].time_domains.update({domain_a.id: domain_a, domain_b.id: domain_b}))
    base_a = 1_800_000_000_000_000_000
    base_b = 1_700_000_000_000_000_000
    for domain, base in [(domain_a, base_a), (domain_b, base_b)]:
        fit_and_store_time_map(
            store,
            session_id=sid,
            source_domain_id=domain.id,
            target_domain_id=session.global_time_domain_id,
            anchors=[
                {"source_ns": base, "target_ns": 0, "label": "interaction start", "weight": 1},
                {"source_ns": base + 4_000_000_000, "target_ns": 4_000_000_000, "label": "interaction end", "weight": 1},
            ],
        )
    role_a = _target_artifact(store, sid, output / "derived_demo" / "R01", "R01", base_a)
    # Deliberate 10 ms difference between the two eye trackers.
    role_b = _target_artifact(store, sid, output / "derived_demo" / "R02", "R02", base_b + 10_000_000)
    result = create_dyadic_attention_annotations(
        store,
        session_id=sid,
        role_a_target_artifact_id=role_a.id,
        role_b_target_artifact_id=role_b.id,
        role_a_id="R01",
        role_b_id="R02",
        role_a_time_domain_id=domain_a.id,
        role_b_time_domain_id=domain_b.id,
        canonical_fps=30,
        cross_wearer_tolerance_ms=25,
        minimum_episode_ms=100,
        short_gap_ms=50,
        label="S001_I01 dyadic attention",
    )
    payload = {
        "project": str(store.manifest_path),
        "session_id": sid,
        "role_a_target_artifact_id": role_a.id,
        "role_b_target_artifact_id": role_b.id,
        "role_a_time_domain_id": domain_a.id,
        "role_b_time_domain_id": domain_b.id,
        "dyadic_attention": result,
    }
    (output / "DYADIC_ATTENTION_DEMO.json").write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.output).resolve(), overwrite=args.overwrite), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
