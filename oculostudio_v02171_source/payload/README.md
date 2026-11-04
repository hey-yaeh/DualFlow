# OculoStudio v0.21.71 — Independent AprilTag Board Geometry Resolver

## v0.21.71

- Fixes Calibration Set profile jobs that failed at 45% with `The legacy engine did not produce a usable 4x4 board mapping` even though all six Calibration sources were present.
- Separates two independent evidence tasks: labelled head-marker geometry resolves the physical ETM-A/ETM-B identity, while scene AprilTags plus the planar 16-marker board resolve the 4×4 Tag-corner orientation.
- Stores the accepted 16 Tag-corner + 4 outer-corner mapping as a revision-scoped Calibration Set resource and reuses it for both ETM profiles and later rebuilds.
- Requires all calibration actors to agree on one board orientation; disagreement or incomplete marker geometry fails closed before profile estimation.
- Adds a Calibration source preflight that reports board marker count, C3D frames/rate, mapping status and specific blocking reasons before expensive profile estimation.
- Passes explicit board geometry into the unchanged measured camera-to-tracker estimator, preserving AprilTag detection, PnP, calibration-only clock search, robust SE(3) aggregation and QC.
- Shows Board mapped / pending alongside ETM-A and ETM-B readiness in Data and in the Project-level Profile Builder.
- Validated end-to-end on the uploaded `CAL_S004_S007` sources: P01/James→ETM-B and P03/Jessica→ETM-A, with both full production profiles published transactionally.

# OculoStudio v0.21.71 — Independent Board Geometry Resolver

All source-file management now lives in **Data**:

- **Calibration Data** — Project-level Calibration Sets, revisions, shared sources and Session assignments.
- **Interaction Data** — current Session/Interaction NRD, TSD, FaceMapper, C3D, videos and props.

Each scope has an independent drop zone, picker, inventory and inspector. Calibration performs calibration calculations only.

See `docs/RELEASE_NOTES_v0_21_70.md`.
