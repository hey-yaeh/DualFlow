# OculoStudio v0.21.64 — Role-Locked Partner Projection

## Problem fixed

When the active wearer was **R02**, the Gaze Targets panel could retain or auto-select the **ETM-B** calibration profile that belongs to R01. The participant-association job could then produce no valid in-view partner-head projection. The Face AOI review player consequently showed FaceMapper geometry without the light-blue partner identity gate. A second lineage problem allowed an association generated from another Native-reference trim to remain selectable, leading to `Participant association and Face AOI must use the same Native-reference trim`.

## Role-locked calibration handoff

Participant association now treats the paired Calibration handoff as authoritative:

- R01 uses the profile bound to **ETM-B**;
- R02 uses the profile bound to **ETM-A**;
- changing Active role cannot reuse the other wearer's profile;
- an explicit role-mismatched profile override is rejected by the backend;
- a manual override remains available only inside a collapsed emergency-recovery section.

The Gaze Targets panel displays a read-only card such as `R02 → ETM-A → <bound profile>` before the projection job is started.

## Exact trim lineage

The Participant Association and Face AOI selectors are now filtered by the exact Native-reference Trim artifact, not merely by label text. Only compatible items are listed:

- FaceMapper filter descended from the selected trim;
- participant association generated for the same wearer role and trim;
- Privacy clip generated from the same trim.

The UI validates the lineage before queueing, and the backend reports both conflicting trim IDs if a stale or manually supplied association is used.

## Fail-closed blue-box gate

For `Partner only` processing, a participant-association job must produce at least one valid, in-view partner-head projection. If it produces zero, OculoStudio now fails without publishing an artifact and reports the role, ETM and likely causes. A completed-but-unusable association can no longer silently proceed to Face AOI.

The light-blue projected box remains an **identity gate only**. The accepted green FaceMapper box remains the authoritative face/no-face boundary and the source of eye, nose and mouth geometry. No blue box is expected on frames where the partner is genuinely outside the ego camera view or the head pose is invalid.

## Existing projects

Existing Calibration, TimeGraph, Trim, Privacy, Gaze 3D, Body AOI, Object AOI and previously reviewed artifacts remain immutable and valid. An old R02 participant-association or Face AOI produced without a valid partner projection is not rewritten.

After installing this release, regenerate only:

1. R02 **Participant association** for the current R02 Native-reference trim;
2. R02 **Face AOI** using that new association.

Rerun R01 only if its review also lacks a valid partner projection or was generated from a mismatched trim.
