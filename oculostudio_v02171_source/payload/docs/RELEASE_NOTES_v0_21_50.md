# OculoStudio v0.21.50 — Body Geometry Integrity Gate and Fixation Review

## Why this release exists

Earlier Body AOI artifacts could complete even when the required torso primitive was absent. In that failure mode, chest-directed gaze could become a head/neck near-body candidate, producing an inflated uncertain-hit queue.

## Changes

1. **Pre-intersection geometry acceptance**
   - Checks required self/partner streams before ray intersection.
   - Default thresholds: body timeline ≥ 90%; torso coverage ≥ 80%.
   - `required` blocks the job; `warn` and `off` continue only for diagnosis and never unlock review/fusion.

2. **Actionable diagnostics**
   - Saves a preflight JSON when blocked.
   - Reports relation, entity, source C3D, torso/pelvis counts, coverage, fallback fraction, unresolved landmarks and recommended actions.

3. **Downstream safety**
   - Body human review and target fusion reject geometry-invalid or unverifiable Body AOI artifacts.
   - The UI marks old invalid results as `Geometry blocked` and leaves them viewable for diagnosis.

4. **Fixation-level review units**
   - Repeated flagged samples in one fixation appear as one pending review group.
   - All sample/frame evidence remains available and decisions retain sample/frame/fixation scopes.

5. **Visible per-person coverage**
   - Gaze Targets and Body Review show body, torso, pelvis and fallback coverage for each required entity.

Existing projects and artifacts are preserved. To obtain corrected labels, regenerate Body AOI with the required geometry gate.
