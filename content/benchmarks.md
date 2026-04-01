+++
title = "Representative research directions"
slug = "benchmarks"
+++

# Representative research directions

The current paper presents DualFlow as a **dataset and curation paper**, not as a finalized benchmark suite. The release supports multiple downstream tasks with different label spaces and evaluation protocols, so this page highlights the most natural research directions rather than a single fixed leaderboard.

<div class="callout">
  <strong>Important note.</strong> The current release does <em>not</em> assume manually produced facial-region or body-part masks. Attention-related work should build on native gaze traces, egocentric context, eye-state exports, facial parameters, and synchronized body motion.
</div>

## 1. Egocentric social attention and partner-directed gaze

The synchronized egocentric video, native eye tracking, and mocap-localized 3D gaze vectors support work on:

- gaze allocation during conversation;
- fixation structure and blink dynamics;
- shared attention and eye-contact events;
- joint analysis of eye and body motion in a common coordinate frame.

Typical inputs include egocentric video, gaze history, full eye-state exports, facial parameters, body motion, and optionally transcript or audio context.

## 2. Turn-taking, backchannels, and conversational timing

Because audio, transcripts, eye tracking, facial parameters, and body motion are aligned on a common timeline, DualFlow can support research on:

- turn holds and turn shifts;
- backchannel timing;
- pre-turn cues in gaze, face, and body behavior;
- speech-linked nonverbal behavior.

For evaluation, we recommend **session-level** or **dyad-level** splits to reduce temporal and identity leakage.

## 3. Role understanding and social interpretation

Role-centered questionnaires and cross-cultural interpretation annotations support research on:

- multimodal role understanding;
- perceived social intention;
- emotion change descriptions over time;
- multimodal language grounding for social interaction.

These annotations are better viewed as a flexible resource for future modeling than as one fixed label set.

## 4. Animation, expressive generation, and embodied agents

DualFlow is also useful beyond recognition. The release of **FBX / SMPL-H body motion**, **FLAME / ARKit-compatible facial parameters**, **participant anthropometrics**, and **3D gaze vectors** makes the dataset suitable for:

- animation retargeting;
- expressive motion and face generation;
- digital-human pipelines;
- embodied-agent behavior modeling.

## Recommended evaluation protocol

When performers recur across sessions with different enacted roles, identity leakage can otherwise dominate reported performance. A strong default protocol is therefore:

- **session-disjoint** or **dyad-disjoint** splits;
- clearly documented modality availability;
- ablations over eye tracking, facial parameters, audio, and motion;
- explicit reporting of any role-level or participant-level metadata used during training.

## Suggested reporting practice

Even before a formal benchmark package is frozen, we recommend that future work report:

| Study type | Minimum reporting details |
| --- | --- |
| Attention modeling | modality set, gaze representation, split protocol, temporal window, evaluation target |
| Conversational timing | boundary definition, label scheme, onset tolerance if used, class balance |
| Role or intention analysis | annotation subset, annotator source, text usage, aggregation strategy |
| Animation or generation | released representation used, retargeting setup, identity leakage safeguards |
