# OculoStudio v0.21.49 — Torso Fallback and Auditable Body Review Reduction

## Fixed: torso absent while limbs remain visible

The old standard Body primitive profile required multiple central thorax and pelvis markers. Some valid Vicon exports retain shoulders, hips, head and limbs but do not satisfy those central combinations, so no torso capsule was generated.

v0.21.49 adds frame-local, evidence-based fallbacks:

1. `pelvis_center` from the midpoint of valid bilateral hip anchors;
2. `neck_top` from bilateral shoulders, or conservatively from the observed head-to-pelvis axis;
3. `chest_center` from bilateral shoulders, a partial chest marker, or the neck/head-to-pelvis axis.

Every derived landmark receives reduced confidence and provenance. Diagnostics include fallback counts and torso/pelvis primitive frame coverage.

For older artifacts, AOI Review derives a display-only torso from limb anchors. This makes the black 3D evidence view readable immediately, but the old sample labels were computed without that torso. Rerun Body AOI to obtain real torso intersections.

## Reduced Body AOI review load

Previously every non-exact uncertainty-cone candidate was marked `review_required`. v0.21.49 adds an auditable policy that never changes the predicted relation, entity or body region:

- clear angular miss with sufficient confidence and margin to the next surface;
- same-frame consensus from multiple matching exact hits;
- same-fixation consensus from matching nearby exact hits.

The following always remain pending:

- self-occlusion;
- missing scene frame;
- unresolved or other-person entity;
- multiple competing surfaces for temporal consensus;
- low-confidence or large-angle near-body candidates.

Policies: `conservative`, `adaptive` (default), `balanced`, `off`.

## Outputs

New sample audit fields:

- `review_required_before_auto_resolution`
- `review_reason_before_auto_resolution`
- `auto_review_resolved`
- `auto_review_resolution_source`
- `auto_review_confidence`
- `auto_review_policy`

New reports:

- `body_aoi_auto_review_summary.csv`
- `body_aoi_auto_review_summary.json`

The Gaze Targets and AOI Review interfaces report before/after queue size, auto-resolution sources, remaining reasons and torso primitive coverage.
