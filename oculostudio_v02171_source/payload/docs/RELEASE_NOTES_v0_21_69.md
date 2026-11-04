# OculoStudio v0.21.69 — Calibration Set Physical Profile Builder

## Fixed architecture

Project-level Calibration Set source import and physical profile estimation are now one coherent workflow. Earlier role-detail estimation incorrectly coupled hardware profile generation to the current Interaction wearer and required a Project Global TimeGraph that Calibration Set sources intentionally do not create.

The new builder:

- inventories the active Calibration Set revision;
- treats calibration actors independently of current Interaction wearers;
- matches each AprilTag recording to physical ETM-A or ETM-B from geometry;
- uses the measured legacy calibration engine's internal calibration-only offset search;
- estimates both camera-to-tracker transforms;
- commits the pair transactionally;
- propagates traceable profile references to assigned Sessions;
- safely recovers explicitly tagged, valid but previously unbound profile artifacts.

## UI

- Data Calibration Set cards show ETM-A/ETM-B readiness and a direct **Build profiles** action.
- Calibration Phase 2 is now Project-level and Interaction-independent.
- R01/R02 detail views are evidence/recovery views only.
- The builder shows source inventory, hardware slots, blocking reasons, progress and a persistent primary action.

## Safety

Ambiguous actor→hardware mappings, duplicate matches, missing hardware, unsupported board geometry, unavailable files and low-confidence identities fail closed. A paired build never commits only one newly estimated hardware profile.
