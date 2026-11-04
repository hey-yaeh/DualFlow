# OculoStudio v0.21.70 — Scope-Separated Data Management

v0.21.70 makes **Data** the single place for source-file management and reduces **Calibration** to calibration operations only.

## Data workspaces

### Calibration Data

A self-contained Project-level workspace with its own drop zone, file/folder pickers, append flow, Calibration Set/revision/Session management, source inventory and inspector. Browser files are copied into the selected Set's canonical source folder; desktop path selection can reference originals without copying.

### Interaction Data

A separate Session/Interaction-local workspace with its own drop zone, file/folder pickers, identity tools, inventory and inspector. It accepts NRD, TSD/TSDEV, FaceMapper, body/ETM C3D, facial videos, props and other Interaction media.

The Interaction table now shows only the active I-code plus explicitly unassigned files; assets from other Interactions in the same Session are hidden.

## Unambiguous drag-and-drop

Drops are routed by the visible target block. Drops outside both blocks are rejected rather than assigned by guesswork. The previous unscoped global drop zone has been removed.

## Calibration operations only

Calibration now performs wearer resolution, physical-profile building/binding, Pupil-clock synchronization and paired-handoff publication. Missing input messages link back to **Data → Calibration Data**. Routine source importing and Set management no longer live on the Calibration page.

Existing projects, Calibration Sets, Sessions, assets and immutable artifacts remain compatible and do not need to be regenerated.
