# OculoStudio v0.21.46 — Body AOI Torso Surfaces

## Problem

The black **3D Body evidence** inset contained torso and pelvis primitives, but every capsule was rendered as a thin centre-line. The torso therefore looked absent or was visually confused with a limb.

## Changes

- Torso capsules are rendered with their projected primitive radius in Split, X–Z and Y–Z views.
- Pelvis spheres are rendered as connected translucent torso surfaces.
- Both self and partner bodies receive the surface treatment while retaining dashed-self and solid-partner evidence semantics.
- Selected torso hits retain the white evidence boundary and gaze-hit marker.
- Stable and frame-local world bounds account for primitive radius, preventing torso clipping.
- Pre-rendered QC MP4 insets use the same torso/pelvis surface rendering.

## Scientific compatibility

This is a visualisation-only change. Ray intersections, first-hit ordering, self occlusion, social relation labels, region labels, CSV and Parquet outputs are unchanged. Existing Body AOI artifacts do not need to be regenerated.
