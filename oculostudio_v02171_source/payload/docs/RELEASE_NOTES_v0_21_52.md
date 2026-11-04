# OculoStudio v0.21.52 — Gaze-Sample Clock AOI

## Body AOI

- Uses every gaze sample's exact timestamp as the 3D computation clock.
- Interpolates each selected self/partner C3D independently at that timestamp.
- Performs vectorised paired ray–primitive intersection in bounded chunks.
- Builds a separate lower-rate body geometry table only for video overlay and review navigation.
- Removes `no_scene_frame` as a 3D Body AOI failure state; preview availability is exported separately.
- Evaluates torso/body coverage over valid gaze-sample timestamps rather than video frames.
- Writes `body_aoi_timebase.json` with gaze, video and body-source rates and timing errors.
- Blocks legacy video-frame-clock Body AOI artifacts from human review and target fusion.

## Face AOI

- Keeps FaceMapper detections as observed native-video-frame evidence.
- Adds optional same-track interpolation to each gaze timestamp.
- Requires short gaps, plausible motion and stable identity for partner/tracked-person modes.
- Falls back to the nearest observed frame when any safety condition fails.
- Exports per-sample interpolation mode, previous/next frame, alpha, gap, confidence and fallback reason.

## Provenance

- Renames the raw marker profile to `vicon_fullbody_10fingers`.
- Retains `motionstudio_production10fingers` only as a legacy diagnostic alias.
- Clarifies that the OCUMO marker labels are read directly from raw Vicon C3D.
