# OculoStudio v0.21.66 — Project–Session Folder Workspace

This release redesigns project storage around the explicit hierarchy Project → Session → Interaction → Role.

## Highlights

- Three-step project wizard with a real parent-directory check, a dedicated project folder, and a separate display name.
- Batch Session creation from ranges such as `S004-S007`.
- Optional Interaction planning before any data import.
- Human-readable `sessions/S###/{imports,derived,reports,annotations,cache}` folders.
- Project Session Manager for later Session creation, Interaction planning, switching and folder access.
- Project-level shared Calibration library for traceable and idempotent ETM-A/ETM-B profile reuse across Sessions.
- Interaction-specific wearer assignment, clock offset/drift, TimeGraph and Trim remain independent.
- Existing projects and legacy labels such as `Session S001` remain compatible and are not moved.
