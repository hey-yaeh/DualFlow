# OculoStudio v0.21.68 — Direct Calibration Import Workspace

v0.21.68 separates Project-level Calibration Set ingestion from current-Interaction ingestion.

- Data now presents two explicit import scopes.
- **Import Calibration Set files** opens a dedicated modal without navigating away from Data.
- The modal shows Set, active revision, assigned Sessions and canonical destination.
- Folder or multi-file selection is scanned before any project mutation.
- Participant, ETM-A/ETM-B, board and world-reference roles are proposed automatically and remain reviewable.
- Duplicate state, file size and intended destination are visible before a single audited batch import.
- The former inline Calibration-page scanner remains available only as an advanced recovery path.
- Existing projects and immutable artifacts remain compatible.
