# OculoStudio v0.21.47 — Human Review Entry Restore

## Problem fixed

Opening a completed Body AOI result left the AOI Review Player in Body mode. The Face AOI manual-review editor was rendered only in Face mode and had no persistent launcher, so the human-review workflow appeared to have disappeared.

## Changes

- A persistent **Human review** card is shown above the AOI Review Player in Face, Body and Fused-target modes.
- The card selects the best matching Face AOI, switches only the player to Face mode, opens the editable review panel, and seeks to the first pending frame.
- The Expert Gaze Targets property panel, Guided Review page, interaction workflow and completed Face AOI jobs now expose the same action.
- Pending counts are taken from audited sample counts. A review fraction is used only as a yes/no signal when an exact count is unavailable.
- Body AOI tables and artifacts are never modified by this navigation action.

## Compatibility

Existing projects and existing Face/Body AOI artifacts are compatible. No reprocessing is required. Editable manual corrections remain a Face AOI function; Body AOI retains visual QC and evidence navigation.
