# OculoStudio v0.21.63 — Aligned Facial-Expression Video Export

## Purpose

This release completes the facial-video branch of the verified interaction workflow. After a facial-expression video has been synchronized to its role-matched ego/Pupil recording through **Shared-audio facial ↔ ego video sync**, OculoStudio can now export a formal facial video on the same physical interval as the automatic C3D-defined clip.

## New Trim action

Open **Expert tools → Trim**, select the active role, then use:

> **4 · Export aligned facial video**

The action is enabled only when OculoStudio can resolve all of the following for the current role:

1. an automatic C3D-defined Native-reference Trim;
2. a Shared-audio Time Sync artifact that was accepted after start/middle/end visual review;
3. the same original Native ego recording as the reference used by that Trim;
4. unchanged facial and ego source media; and
5. complete facial-video coverage of the requested physical interval.

## Clock and drift handling

The accepted mapping is an affine TimeGraph:

```text
ego/reference time = scale × facial-source time + offset
```

The export does not merely cut the facial video at two approximate timecodes. It samples each output frame on the ego/reference clock, applies the inverse accepted mapping to find the corresponding original facial frame PTS, and records the complete lineage in `aligned_facial_frame_map.csv`. This applies both the initial offset and the estimated device clock drift across the entire interaction.

When audio preservation is enabled, the matching source facial-audio segment is retimed by the same inverse affine scale before being muxed into the clean output MP4. No arbitrary Dynamic Time Warping is used.

## Output artifact

Each role receives an independent immutable `aligned_facial_video` artifact containing:

```text
<role>_aligned_facial_video.mp4
aligned_facial_frame_map.csv
aligned_facial_manifest.json
aligned_facial_sync_report.html
audio_video_sync_accepted.json
source_trim_manifest.json
native_source_frame_map.csv
README.txt
```

The frame map records, for every output frame:

- output frame index and PTS;
- interaction/reference time;
- Native ego frame index and PTS;
- queried facial-source time;
- selected original facial frame index and PTS;
- source-frame selection error;
- duplicate-source-frame and boundary indicators.

## Lineage and safety

The published artifact has two immutable parents:

- the accepted role-specific Shared-audio Time Sync artifact; and
- the role-specific automatic Native-reference Trim.

The exporter never re-estimates synchronization or Trim boundaries. It fails closed for role/interaction mismatch, wrong ego reference, missing accepted visual review, changed source files, invalid scale, missing TimeGraph mapping, or incomplete source coverage. Original facial video, Native recording, TSD, FaceMapper, Calibration, Gaze 3D, AOI, and prior artifacts are not modified.

## Guided workflow

The interaction workspace now shows an optional facial-video path for each role:

```text
Shared-audio FV ↔ ego TimeGraph
→ Export drift-corrected aligned facial video
```

This branch is optional; interactions without facial-expression video continue through the existing ego/TSD/FM, Privacy, Gaze 3D, AOI, and Dyadic workflow unchanged.

## Compatibility

Existing projects and all previous artifacts remain valid. After installing v0.21.63, no existing Calibration, Trim, Privacy, Gaze 3D, Face AOI, Body AOI, Object AOI, or shared-audio synchronization needs to be rerun. A previously accepted and visually reviewed shared-audio Time Sync can be used immediately for facial export.
