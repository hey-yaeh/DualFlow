# OculoStudio v0.21.58 — Aligned Clip QC Preview

## Problem fixed

The automatic C3D-defined Trim page displayed every overlay option—scope, segment length, FPS, face boxes, landmarks, preliminary AOIs, gaze points and left/right convention—but exposed no action that could actually render the preview. The automatic clips were valid and TIMEGRAPH VERIFIED; only the UI-to-preview-engine connection was missing.

## New review workflow

After Calibration creates the paired R01/R02 clips:

1. Open **Trim** and select the role with **Use R01 / Use R02**.
2. In **2 · Aligned clip QC preview**, choose the review scope and overlays.
3. Click **Generate QC preview**. The job queue shows **Render aligned clip QC preview**.
4. When complete, use **Open preview** or the central **Open aligned clip preview** action.
5. Regenerate after changing any preview setting.

The central verified-clip card also exposes **Review aligned clip video**, so the default representative review can be started without searching the property panel.

## Scientific and data-safety behavior

- The preview uses the automatic trim artifact's exact cached `preview_id` and verified NRD `.time` / MP4 PTS interval.
- It never asks for a second start/end value and never recomputes the aligned interval from nominal FPS.
- It reads retained TSD gaze samples and FaceMapper geometry directly from the already matched sources.
- It creates only a temporary cache video. No new scientific artifact is registered, the aligned clip is not replaced, and the source NRD video is not re-encoded.
- Manual recovery trim and its original pre-trim preview remain available when Calibration cannot create an automatic clip.

## R01/R02 isolation

Preview results are keyed by automatic `preview_id`, not by one global 'last preview' slot. Switching to R02 does not show R01's player. Returning to R01 or reopening the project restores R01's matching preview. The same applies symmetrically to R02.

## Validation

Targeted release validation covered:

- visible and enabled Generate/Open actions;
- real FFmpeg/H.264 overlay generation for both roles;
- FaceMapper and gaze mapping;
- distinct R01/R02 preview and overlay IDs;
- role switching without cross-role reuse;
- restoration in a fresh browser document;
- no new immutable artifact and no source-video re-encode;
- JavaScript syntax, Python compilation and wheel integrity.

The automated browser validation ran against a live FastAPI backend through the project's offline relay harness. The Windows WebView shell still requires confirmation on the user's Windows installation.

## Installation

Close OculoStudio, extract the complete installer ZIP and run `INSTALL_OCULOSTUDIO.bat`. Existing projects and derived artifacts are preserved.
