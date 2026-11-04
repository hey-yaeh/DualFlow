# OculoStudio v0.21.68 — Project Calibration Sets

This release promotes calibration recordings and derived profiles to a true Project-level, revisioned **Calibration Set** that can be assigned to multiple Sessions. It replaces the previous workflow in which calibration material was owned by one Session and profiles were copied outward.

## Data model

```text
Project
├── Calibration Sets
│   └── CAL_S004_S007
│       ├── revision rev_001
│       ├── calibration source references
│       ├── ETM-A / ETM-B profile references
│       └── applies to S004, S005, S006, S007
└── Sessions
    ├── S004 → I01, I02, ...
    ├── S005 → I01, I02, ...
    ├── S006 → I01, I02, ...
    └── S007 → I01, I02, ...
```

Calibration recordings are registered once. Assigned Sessions inherit audited references to the same immutable source paths and active revision; raw bytes are not duplicated. Interaction recordings and synchronization remain independent.

## Project creation

The three-step wizard now creates an optional initial Calibration Set alongside the selected Sessions. The user specifies:

- Calibration Set ID and display name;
- purpose: combined, hardware geometry, or known-target gaze calibration;
- whether the Set applies to all Sessions created by the wizard.

New projects use folder layout v3:

```text
calibration_sets/<SET_ID>/
├── calibration_set.json
├── revisions/rev_001/
├── sources/
├── derived/
└── reports/
```

## Data page

The Data page separates:

1. **Project Calibration Sets** — shared calibration recordings, profiles, revisions and Session assignment;
2. **Current Interaction data** — NRD/TSD/FM, body/ETM C3D, facial videos, props and other recordings for the selected Interaction.

Calibration source files no longer appear as if they belonged to the current Interaction and do not create provisional Global-Time maps.

## Calibration Set Manager

The manager can:

- create additional Sets;
- assign or unassign multiple Sessions;
- create and activate immutable revisions;
- attach an existing profile;
- refresh inherited source/profile references;
- open the Set folder;
- inspect which revision each Session inherits.

Creating a new revision inherits the preceding revision's source and profile references, preserving reproducibility. Existing results retain the revision they used.

## Profile integration

Newly estimated or imported calibration profiles can be attached directly to the active Set revision. OculoStudio materializes role-appropriate profile artifacts into assigned Sessions with explicit lineage; it does not duplicate the original calibration recording.

## Scientific boundary

A shared Calibration Set may provide hardware geometry and/or participant known-target calibration. It does **not** share Interaction-specific evidence. Every Interaction still independently computes wearer assignment, Pupil–ETM offset/drift, TimeGraph, C3D-defined Trim, Privacy, Gaze 3D, AOIs and Dyadic Attention.

## Compatibility

Legacy projects continue to open. Their previous shared-calibration metadata can be migrated into a Project Calibration Set without moving source recordings. Existing artifacts, annotations and exports remain immutable.
