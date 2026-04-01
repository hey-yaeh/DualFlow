+++
title = "FAQ"
slug = "faq"
+++

# FAQ

## Is the full dataset publicly downloadable from this website?

No. The website is the public documentation and request hub. The core multimodal recordings and participant-linked annotations are distributed through a gated research-use workflow.

## What modalities are included in the release?

The current release is designed around identity-obscured egocentric video, full Pupil Cloud eye-tracking exports, mocap-localized 3D gaze vectors, FLAME and ARKit-compatible facial parameters, FBX and SMPL-H body motion, consent-aware audio, aligned transcripts, and structured metadata.

## Are raw identity-revealing videos released?

No. Egocentric videos are distributed only in identity-obscured form.

## Is all audio released in original form?

Not necessarily. Audio release follows participant consent. Some files may be distributed in transformed form rather than original voice form.

## Does the current release include a finalized benchmark suite?

No. The paper and website frame DualFlow as a dataset and curation release. The site highlights representative research directions and recommended evaluation practice rather than one fixed benchmark package.

## Do you provide manually annotated facial-region or body-part masks?

No. The current release does not assume manually produced region masks. Attention-related work should build on the native eye-tracking exports, egocentric context, facial parameters, and synchronized body motion.

## What evaluation split is recommended?

A good default is **session-disjoint** or **dyad-disjoint** evaluation. This helps reduce temporal and identity leakage when the same performers recur across different sessions and roles.

## How do I request access?

Email **hey3@tcd.ie** and **jovanea@tcd.ie** with your affiliation, intended non-commercial research or teaching use, and confirmation that you agree to the research-use conditions.

## Will public code be released?

The website is structured so that loaders, schemas, parsers, documentation, and low-risk supplementary materials can be hosted publicly. Core gated files are handled separately from public helper code.
