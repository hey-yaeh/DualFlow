# OculoStudio v0.21.61 — Shared-Audio Facial ↔ Ego Video Synchronization

## Purpose

This release adds an optional, auditable method for synchronizing an external facial-expression video with the wearer’s ego / Pupil scene video when both devices recorded the same physical interaction.

The method is intended for simultaneous co-recordings. It does not align two independently reenacted recordings merely because they contain similar words.

## New workflow

Open **Calibration → Paired summary** or the relevant **R01 / R02 details** view, then use:

> **Shared-audio facial ↔ ego video sync · optional**

Select:

- wearer / interaction role;
- facial-expression video (source clock);
- ego video or Pupil Native recording (reference clock);
- maximum start-offset search range.

Click **Auto-align videos from shared audio**. The job creates a review candidate; it does not modify the TimeGraph automatically.

## Synchronization model

The accepted mapping is one physical affine clock relationship:

```text
t_ego = scale × t_facial + offset
```

The engine:

1. extracts both audio tracks without modifying source media;
2. creates microphone-robust energy/onset signatures;
3. performs a coarse offset search;
4. matches distributed local windows across the recording;
5. rejects low-agreement, ambiguous, inactive, or affine-inconsistent windows;
6. tolerates short one-sided instructions only when the remaining common audio uniquely supports the same lag;
7. refines local lags below one feature sample;
8. robustly estimates offset and clock drift;
9. evaluates accepted anchors with leave-one-window-out prediction.

Arbitrary dynamic time warping is never used. The system publishes a single affine map, not a continuously warped timeline.

## Mandatory review before acceptance

Every non-failed candidate opens a synchronized dual-video reviewer containing:

- estimated offset and drift;
- accepted-window count and temporal coverage;
- leave-one-window-out P95 residual;
- aligned acoustic-pattern plot;
- facial and ego video players;
- **Review start**, **Review middle**, and **Review end** controls;
- selectable audio monitor;
- an optional stable visual A/V correction;
- a mandatory visual-review confirmation.

The backend refuses to publish the map until visual review is explicitly confirmed. A stable device-internal A/V correction is stored in the accepted evidence; it is not silently hidden inside the estimate.

## Pupil Native integration

When the ego reference is a Pupil Native recording, acceptance publishes:

1. facial-video media PTS → Native ego-video media PTS; and
2. the exact Native ego-video media PTS → Pupil capture-clock bridge.

This joins the external facial-expression video to the same TimeGraph used by TSD, Gaze 3D and Vicon processing.

## Safety and provenance

- R01/R02 and interaction identities are checked before decoding.
- A candidate never changes the TimeGraph.
- FAILED or unrelated audio never publishes a map.
- Source-video and Native ZIP stamps are rechecked before acceptance.
- Acceptance creates an immutable Time Sync artifact.
- Publishing is transactional: a registration or bridge failure restores the previous direct maps and removes partial output.
- Source videos and archives are never modified.
- Accepted and rejected windows, acoustic metrics, holdout residuals, correction, and review confirmation remain auditable.

## Interpreting different instructions

A short instruction recorded by only one device can be rejected as an unmatched local window. The mapping is accepted only when enough common acoustic evidence remains distributed across the interaction. If common sound is sparse, repetitive, confined to one short region, or inconsistent with one affine clock map, the candidate is marked REVIEW or FAILED rather than being forced into alignment.

## Limits

Audio synchronization establishes the relation between the two audio/video media clocks. The final visual alignment can also include stable device-internal audio-to-video latency and small sound-propagation differences. The start/middle/end split-video review is therefore mandatory.

Recordings with pauses, restarts, large dropped sections, or genuine discontinuities may require a future piecewise-affine or manual-anchor workflow. v0.21.61 deliberately fails closed instead of applying unconstrained time warping.

## Compatibility

The release preserves all existing:

- wearer assignments and R01↔ETM mappings;
- Pupil IMU ↔ Vicon TimeGraphs;
- Calibration handoffs;
- C3D-defined aligned clips;
- aligned ego/TSD/FaceMapper packages;
- Privacy outputs;
- Gaze 3D rays;
- Face/Body AOIs, dyadic outputs and review decisions.

No existing processing needs to be rerun unless this new facial-video TimeGraph is required.
