# OculoStudio v0.21.60 — Paired Calibration Views + Explicit Mode Selector

## Why this release was needed

The application had two controls whose labels described the current state rather than the destination:

- the top-right Guided/Expert mode button; and
- `Use R01 / Use R02` on Calibration and other Expert pages.

Because Calibration is now one paired R01 + R02 workflow, `Use R01 / Use R02` incorrectly suggested that only one role was calibrated or that clicking the control could change the wearer assignment.

## New top-level mode selector

The top bar now contains a dropdown with two explicit choices:

- **Guided workflow**
- **Expert tools**

The selected value is the mode that will be displayed. The choice remains persisted locally, but the control no longer requires users to infer the click target from the current label.

## Paired Calibration

Calibration now defaults to **Paired summary**. It presents both role-specific evidence records under one authoritative paired workflow:

- R01 → ETM-B
- R02 → ETM-A

The page provides three display views:

- **Paired summary**
- **R01 details**
- **R02 details**

The two detail views are read-only workflow views. Switching them changes only the evidence/profile content displayed. It cannot:

- change wearer assignment;
- exchange ETM-A and ETM-B;
- rewrite role-specific synchronization;
- replace a calibration profile;
- regenerate the paired handoff; or
- change the role used on another page.

Actions that operate on both roles remain in Paired summary, including wearer resolution, dual clock synchronization, handoff creation, and paired C3D-defined clip generation.

## Role-specific Expert pages

Trim, Privacy, Gaze 3D, Targets, and other wearer-specific pages retain role selection because their outputs are role-specific. Their control is now labelled **Active role**, with an explicit R01/R02 dropdown. The selection affects only the role currently reviewed, processed, or exported.

## Compatibility

This release preserves:

- source recordings and project manifests;
- wearer assignments and hardware profiles;
- role-specific TimeGraphs and calibration handoffs;
- C3D-defined aligned clips;
- aligned ego/TSD/FaceMapper packages;
- Privacy videos;
- gaze-ray, Face AOI, Body AOI, and fused-target artifacts;
- manual-review drafts and immutable reviewed artifacts.

No scientific processing stage needs to be rerun solely because of this update.

## Validation scope

The release was validated with a live FastAPI project state and a real Chromium rendering path. Validation covered mode switching, paired-summary default state, display-only role details, explicit Active role selection on non-Calibration pages, JavaScript syntax, Python compilation, wheel integrity, package manifests, and install-from-final-wheel checks.

The final Windows WebView2 shell and batch installer still require confirmation on the user's Windows machine after installation.
