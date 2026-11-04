# OculoStudio v0.21.53 — Evidence-Guided Body Review Reduction

## Why

A high-rate gaze stream can produce thousands of flagged samples while representing far fewer human decisions. v0.21.52 grouped samples for review, but a fixation containing only consistent near-body candidates still remained pending unless it had enough exact-hit anchors.

## What changed

- New `evidence_guided` policy, now the recommended default.
- Candidate-only fixation consensus with strict target, angular, confidence, continuity and competing-surface checks.
- Short unanimous temporal-run fallback when fixation IDs are absent.
- Review-unit counts and reduction fractions in QC.
- Sample-level evidence audit columns.
- Review-only reprocessing endpoint and UI for existing v0.21.52+ Body AOI artifacts.
- Source artifact and automatic semantic/geometric targets remain immutable.
- Review-only reprocessing fails closed when the source already has saved manual decisions, preventing silent loss of human work.

## Safety boundaries

The policy does not clear self occlusion, target conflicts, unresolved/other entities, missing preview evidence, close competing surfaces, low-confidence primitives or discontinuous groups.
