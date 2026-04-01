+++
title = "Collection and curation"
slug = "curation"
+++

# Collection and curation

A major contribution of DualFlow is the pipeline that turns raw multimodal recordings into a shareable research resource. The release is not only about capture hardware; it is also about **synchronization**, **audio cleaning**, **privacy processing**, **format conversion**, and **quality control**.

## Recording workflow

A typical recording follows this sequence:

1. **Scenario briefing.** Performers are briefed on the situation, assumed relationship, and high-level dramatic objective.
2. **Device fitting and preparation.** Motion-capture suits, markers, head-mounted devices, and eye-tracker calibration are prepared.
3. **Synchronized take start.** A clapper board is used as a shared synchronization event.
4. **Improvised performance.** Participants perform the scenario while keeping the chosen role personalities consistent.
5. **Post-take annotation.** Participants evaluate their own role and their partner's role.

This workflow balances control and improvisation, which is central to the dataset design.

## Capture stack

| Component | Device or software | Key details | Released form |
| --- | --- | --- | --- |
| Egocentric video + native eye tracking + audio | Pupil Labs Neon | 1600 x 1200 RGB scene video at 30 Hz; gaze and eye-state related streams exported through Pupil Cloud; IMU-aligned signals available | Identity-obscured ego video, full eye-tracking export, 3D gaze vectors |
| Full-body motion | Vicon Shogun + 27 optical cameras | 120 Hz capture with whole-body motion and fingers | FBX and SMPL-H |
| Facial performance | Rokoko Headcam | Head-mounted facial capture at 60 fps | FLAME and ARKit-compatible parameters |
| Audio cleaning | RX Advanced | Dialogue Isolate and De-Bleed used during curation | Speaker-cleaned, consent-aware audio |
| Synchronization + privacy processing | Clapper board, Adobe Premiere Pro, Mocha Pro | Timeline refinement and tracked facial mosaicing | Synchronized and identity-obscured release assets |

## Synchronization

All modalities are aligned to a common timeline. Because egocentric video, eye tracking, and audio are recorded on the same wearable device, those streams are naturally coupled at the device level. Cross-device alignment is then refined using the clapper-board event and timeline inspection during curation.

The released package is intended to support both **frame-level** and **segment-level** multimodal analysis.

## Audio cleaning

Audio is recorded directly from the integrated microphones of the Neon devices and then cleaned during curation. The current workflow emphasizes:

- dialogue isolation between the two speakers;
- de-bleeding and cross-talk reduction;
- alignment between audio, transcripts, and the synchronized interaction timeline.

This makes the release more usable for conversational modeling than raw headset audio alone.

## Privacy processing

The egocentric videos are released only in **identity-obscured form**. Facial mosaics are tracked and applied during curation to reduce direct identity exposure while preserving the interactional visual context needed for research.

For audio, release follows participant consent:

- some recordings can be distributed in original form;
- some recordings require transformed voice release.

## Quality control

Before inclusion in the release, DualFlow applies consistency checks across modalities and annotations.

### Examples of checks

- temporal continuity of eye-tracking streams;
- alignment between eye-related streams and scene video;
- temporal consistency of exported motion representations;
- alignment between headcam takes and the synchronized interaction timeline;
- timeline and speaker-link quality for cleaned audio and transcripts;
- coherence checks for role-centered and higher-level annotations.

## Release philosophy

DualFlow is intended to support **analysis-ready** and **animation-ready** workflows. Instead of publishing only task-specific features, the release keeps reusable representations such as full Pupil Cloud exports, FBX, SMPL-H, FLAME, and ARKit-compatible parameters whenever possible.
