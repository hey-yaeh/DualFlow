# OculoStudio v0.21.65 — Context-Locked AOI Review

## Problem fixed

When the active context was R02 and the current R02 trim had not yet produced Face AOI or Body AOI, the Human Review sidebar and AOI Review Player could fall back to an older R02 trim or an R01 artifact. This exposed review buttons and pending counts before the current workflow had generated its own evidence.

## New context contract

Every Gaze Targets surface now resolves evidence through the same exact tuple:

- session;
- interaction;
- active wearer role;
- exact Native-reference trim artifact.

This contract now governs the pipeline cards, summary analytics, Human Review cards, AOI Review Player, video choices and Face + Body fusion selectors. Empty current-context results remain empty; they never fall back to another role or an obsolete trim.

## Review entry behaviour

- Before the current role/trim produces Face AOI, the sidebar shows `Generate <role> Face AOI first`.
- Before it produces Body AOI, the sidebar shows `Generate <role> Body AOI first`.
- Review buttons appear only for a current-context artifact.
- Pending counts are read only from that artifact.
- An old artifact remains available in Results & QC, but it is not offered as the current review queue.
- Failed jobs do not create review entries.

## Compatibility

No scientific evidence is rewritten. Existing Calibration, Trim, Privacy, Gaze 3D, Participant Association, Face AOI, Body AOI, Object AOI and review artifacts remain immutable and accessible through Results. No earlier processing stage needs to be rerun.
