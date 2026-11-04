# OculoStudio v0.21.51 — Raw Vicon Marker Profile and Torso Recovery

## Root cause fixed

The raw Vicon full-body 10-finger marker set used in OCUMO does not use the conventional pelvis labels expected by v0.21.50. It uses six waist markers (`LFWT`, `LMWT`, `LBWT`, `RFWT`, `RMWT`, `RBWT`) and three shoulder-surface markers per side (`LFSH`, `LTSH`, `LBSH`, `RFSH`, `RTSH`, `RBSH`). Consequently v0.21.50 could build head and distal limb geometry for every frame while producing zero pelvis and zero torso primitives.

## Changes

- Native production10fingers pelvis/hip, shoulder, elbow, wrist, hand, knee and foot mapping.
- Direct torso capsule from central thorax markers to the six-marker waist centre.
- Marker protocol detection and per-landmark matched-marker / valid-frame diagnostics.
- Support for `Subject_MARKER` and `Subject-MARKER` exporter labels in addition to `Subject:MARKER`.
- Marker protocol displayed in Gaze Targets and Body Human Review geometry cards.
- v0.21.50 fail-closed body geometry integrity gate remains enabled.

## Required action

Install v0.21.51 and rerun **Generate self + partner Body AOI**. The failed v0.21.50 job did not perform gaze-ray intersection and therefore did not damage existing annotations.
