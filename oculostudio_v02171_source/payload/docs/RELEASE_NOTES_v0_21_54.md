# OculoStudio v0.21.54 — Real-Time AOI Review and Semantic Queue Reduction

## Playback

- Replaces per-frame full-inspector loading with compact rolling playback packets.
- Keeps the decoded video, 2D AOI overlay, 3D body evidence and gaze ray frame-locked.
- Updates only a lightweight live frame readout while playing; reloads the complete scientific review frame after pause.
- Restores an explicit, always-visible `▶ Play` / `Ⅱ Pause` control and live/buffering status.
- Separates full paused-frame and compact playback caches so compact packets cannot be replaced by undefined/full entries.

## Body review reduction

- Adds `semantic_guided` as the recommended policy for coarse social-body labels.
- Treats two raw surfaces as non-ambiguous only when relation, entity and coarse body region are identical and exactly two candidates are present.
- Adds safe adjacent exact-hit anchoring across exporter-split fixation IDs.
- Groups adjacent target-stable fixations into review episodes for one-decision review.
- Never changes the automatic target; self occlusion, different people/entities, different coarse regions, missing preview and unresolved competing surfaces remain pending.

## Compatibility

- Preserves v0.21.52 gaze-sample-clock Body AOI, v0.21.52 Face gaze-time interpolation, torso integrity gates, Face/Body human review and immutable artifact lineage.
- Existing v0.21.52+ Body AOI intersections can be re-evaluated without rerunning C3D or ray intersection.
