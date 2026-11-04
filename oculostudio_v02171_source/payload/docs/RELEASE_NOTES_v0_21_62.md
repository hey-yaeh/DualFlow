# OculoStudio v0.21.62 — Direct Gaze-Ray C3D Clock for Body and Object AOI

## Problem fixed

A valid Body AOI run could stop at approximately 3% with:

```text
No time-map path connects 'time_…' to 'global'
```

This happened after the new verified Calibration/Gaze 3D workflow. Gaze 3D correctly used the accepted Pupil→ETM affine TimeGraph and stored the exact ETM/Vicon query time for every gaze sample in `gaze3d_c3d_time_sec`. Body AOI nevertheless looked at the ray artifact's historical `alignment_mode=timegraph` flag and tried to route the original Pupil clock through the project `global` domain before returning to each body C3D. That route is redundant for C3D-defined interaction clips and may not exist.

## New clock routing

For a matching Native-reference trim and Calibration handoff, Body AOI now uses:

```text
Gaze sample
  → gaze3d_c3d_time_sec already stored in the calibrated ray
  → self/partner Body C3D interpolation at the same Vicon time
```

The project Global Time domain is not required for this path.

The ray-generation clock and the downstream geometry clock are now audited separately:

- `alignment_mode`: how Gaze 3D generated the calibrated world-space ray;
- `body_time_alignment_mode`: how Body AOI queried body geometry;
- `global_timegraph_required`: whether a project Global path was actually used;
- `gaze_ray_c3d_clock_column`: the exact ray column reused downstream.

## Safety gates

The direct C3D path is enabled only when:

- a matching Native-reference trim is selected;
- the gaze-ray artifact exposes an exact C3D query time;
- the selected self/partner body assets are the role-resolved body C3Ds recorded by the Calibration handoff or C3D-defined trim;
- at least 90% of finite gaze query times overlap each selected C3D's time range.

A mismatched body/object recording still fails closed with an explicit interaction/timebase error. Legacy or raw workflows without an embedded C3D clock retain the existing verified TimeGraph fallback.

## Object AOI

Object AOI receives the same downstream-clock correction. Marker-prop C3Ds from the same Vicon interaction can reuse the exact gaze-ray C3D time, while mismatched prop recordings are rejected by the time-range overlap gate.

## UI and QC

The Body AOI panel now states that exact gaze-ray C3D time is preferred and that a separate Timeseries→Global path is not required for C3D-defined clips. Results report whether the job used `DIRECT C3D CLOCK` or the project TimeGraph.

## Compatibility

Existing projects, Calibration handoffs, TimeGraphs, aligned clips, Gaze 3D rays, Face AOIs and reviewed artifacts remain valid. Regenerate only the failed Body AOI (and Object AOI if affected); Gaze 3D does not need to be regenerated when its CSV already contains `gaze3d_c3d_time_sec`.
