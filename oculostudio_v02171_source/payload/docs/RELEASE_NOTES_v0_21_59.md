# OculoStudio v0.21.59 — Aligned Ego Video + TSD + FaceMapper Export

## Purpose

The automatic C3D-defined Trim already produced an exact role-specific Native interval plus verified TSD and FaceMapper subsets. It did not publish a clean, reusable aligned ego-video. v0.21.59 adds that formal export without changing the accepted alignment.

## New Trim export

After Calibration creates the verified R01/R02 clips, open **Trim**, select the active role, and use **Export aligned ego + TSD + FM**. Every run creates a new immutable `aligned_clip_package` artifact.

The export can include any combination of:

- **Clean aligned ego video** — no QC slate, boxes, landmarks, AOIs or gaze overlay;
- **Aligned TSD ZIP** — byte-identical copy of the verified `TSD_trimmed.zip`;
- **Aligned FaceMapper ZIP** — byte-identical copy of the verified `FaceMapper_trimmed.zip`.

Source audio can be preserved in the MP4. Supported output rates are 24, 25, 30, 50 and 60 FPS.

## Timing contract

The immutable Native-reference trim remains the sole boundary authority.

1. The export reads `start_media_ns` and `end_media_ns` from the parent trim manifest.
2. It rebuilds the original NRD `.time` / MP4 PTS clock for the exact scene-video member recorded by the trim.
3. It selects only Native frames already present in the verified parent `native_frame_map.csv`.
4. It writes `aligned_ego_frame_map.csv`, mapping each CFR output frame to the original Native frame index, presentation interval, capture UTC and selection error.
5. It never asks for or infers a new start/end time.

The CFR video can duplicate or omit source frames when the requested output FPS differs from the Native variable/actual frame cadence. Downstream analysis must use the explicit frame map rather than assuming output-frame ordinal equals Native-frame ordinal.

## Package contents

A combined export contains:

```text
<role-specific aligned package>/
  <label>_ego.mp4
  aligned_ego_frame_map.csv
  <label>_TSD_aligned.zip
  <label>_FaceMapper_aligned.zip
  native_source_frame_map.csv
  source_trim_manifest.json
  aligned_clip_manifest.json
  aligned_clip_report.html
  README.txt
```

The manifest records the parent artifact, role, recording, exact interval, selected outputs, video settings, source checksums and FFmpeg command.

## Safety and compatibility

- Original NRD, TSD and FaceMapper sources are never modified.
- The parent Native-reference trim is never overwritten.
- TSD/FaceMapper copies are checksum-verified after copying.
- A changed Native source or mismatched recording ID fails closed.
- R01 and R02 packages have independent parent trims, artifacts and output paths.
- Video-only and data-only exports are supported.
- Existing projects and all previous artifacts remain valid.
- Temporary Aligned Clip QC Preview remains available and is intentionally separate from the formal package.

## Validation highlights

- Both synthetic R01 and R02 exports created independent immutable artifacts.
- Every exported video frame remained inside the parent verified Native frame map.
- Output MP4 frame counts matched `aligned_ego_frame_map.csv`.
- TSD and FaceMapper SHA-256 hashes matched their parent trim ZIPs exactly.
- A data-only package completed without video encoding.
- The API route, artifact kind, central Trim button, right-panel controls and formal in-page video player were validated.
- Python compilation, JavaScript syntax, wheel RECORD integrity, isolated wheel installation and extracted-installer verification are included in the release audit.

## Installation

Close OculoStudio, extract the complete installer ZIP, and run:

```text
INSTALL_OCULOSTUDIO.bat
```

No re-import, re-calibration or re-trim is required for existing verified C3D-defined clips.
