# OculoStudio v0.21.55 — Calibration-Integrated Pupil IMU ↔ Vicon Synchronization

## Why this release exists

A shared trimmed start does not prove that the Pupil and Vicon clocks remain aligned throughout a long interaction. v0.21.55 makes physical-clock synchronization a required, auditable part of Calibration instead of a hidden Gaze 3D fallback.

## Four-stage Calibration workflow

1. **Resolve the interaction wearer.** The immutable wearer assignment remains authoritative; in the current OCUMO configuration R01 maps to ETM-B and R02 maps to ETM-A.
2. **Prepare hardware profiles.** ETM-A and ETM-B each need one measured/imported/validated profile.
3. **Synchronize clocks.** For each role, OculoStudio compares Pupil `imu.csv` angular motion with the rigid-body angular motion of the ETM C3D assigned in stage 1.
4. **Create the Gaze 3D handoff.** The wearer assignment, two hardware profiles, and two accepted affine TimeGraphs are bound into a new immutable handoff.

## No ego video is used for synchronization

The estimator reads:

- Pupil Timeseries `imu.csv` timestamps and gyroscope channels;
- the assigned ETM wearer C3D/NPZ marker trajectories and Vicon timestamps.

It does not decode or analyse ego-video pixels. Ego video remains available later for Face AOI, overlays, and human review.

## Offset and drift estimation

The engine constructs orientation-independent angular-motion signatures, searches for the global clock offset, then repeats matching in distributed time windows. The accepted anchors fit an affine TimeGraph:

```text
t_vicon = a + b · t_pupil
```

The UI and artifact report:

- offset in seconds;
- drift in parts per million;
- global/aligned correlation;
- peak separation;
- matched and total windows;
- median and P95 residual;
- usable overlap;
- exact source/target assets, wearer role, and assignment revision.

## Acceptance policy

- **HIGH:** may be saved automatically when the user leaves automatic saving enabled.
- **REVIEW:** never promoted silently; the user must explicitly accept the candidate.
- **FAILED:** does not create a TimeMap or change Calibration state.

A handoff is production-ready only when both roles have a saved TimeMap, formal acceptance, at least two distributed anchors/windows, and no ego-video-derived provenance.

## Fail-closed protections

- A Pupil Timeseries explicitly labelled for another role is rejected.
- A synchronization result from another wearer-assignment revision cannot be reused.
- A time map targeting the wrong ETM is rejected.
- Missing R01 or R02 synchronization blocks handoff creation.
- Video-derived or drift-unverified synchronization cannot satisfy the new Calibration contract.
- Legacy profile-only handoffs cannot enter Native-trim Gaze 3D.
- Once a synchronized handoff is selected, Native-trim Gaze 3D cannot fall back to `pretrimmed_shared_start`.

## Upgrade guidance

Existing projects and artifacts are preserved. Older Calibration handoffs remain visible for provenance, but a new synchronized handoff must be created before generating new Native-trim Gaze 3D rays. The recommended sequence is:

1. Open **Calibration**.
2. Confirm the interaction wearer assignment.
3. Confirm ETM-A and ETM-B hardware profiles.
4. In **Phase 3**, select each role's Pupil Timeseries and run `R01 + R02 synchronization`.
5. Review any `REVIEW` candidate; do not accept implausible offset/drift evidence.
6. In **Phase 4**, create the synchronized Gaze 3D handoff.
7. Regenerate Gaze 3D and downstream Body AOI if the time mapping changes.

## Compatibility

No raw data, previous artifacts, manual review decisions, or project files are overwritten. New TimeGraph and handoff artifacts are immutable children with full lineage.
