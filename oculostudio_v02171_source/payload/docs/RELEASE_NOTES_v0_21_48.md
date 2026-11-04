# OculoStudio v0.21.48 — Body AOI Human Review

## Fixed

Body AOI samples marked `review_required` previously appeared as red review frames, but no Body-specific editable review entry existed. The only editable review panel was for Face AOI.

## Added

- Persistent **Human review · Body AOI** entry above the AOI Review Player.
- **Review flagged Body AOI** entry in the Gaze Targets property panel.
- **Open Body Human Review** action on completed Body AOI jobs.
- Automatic jump to the first pending Body review frame and sample.
- Review actions: accept automatic, semantic override, mark `no_body`, defer and clear.
- Editable relation (`partner`, `self`, `other`), body region and entity ID.
- Sample, frame and fixation scopes.
- Live effective reviewed-target preview and automatic-vs-reviewed comparison.
- Immutable reviewed Body AOI output with parent lineage, decision CSV, automatic-value audit columns, updated frames and QC metrics.

## Compatibility

Existing Body AOI artifacts can be reviewed directly. Body AOI computation, first-hit/occlusion evidence, torso visualization, CSV/Parquet inputs and original automatic artifacts are not overwritten.
