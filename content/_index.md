

![DualFlow overview](modalities_w_s.png)


**DualFlow** is a multimodal dataset of **dyadic social interaction** designed for research on social attention, conversational timing, multimodal language grounding, expressive animation, and embodied agents. The release combines **identity-obscured egocentric video**, **native wearable eye tracking**, **facial-performance parameters**, **consent-aware audio**, **aligned transcripts**, and **high-fidelity full-body motion capture including fingers**.

<video controls width="100%">
  <source src="videos/web_trim_2.mp4" type="video/mp4">
</video>

- To access the data follow the instruction in [Dataset]({{< relref "dataset.md" >}}).
- Together with the data we share an open source tool to align modalities called [*aligner*](https://github.com/filippe24/aligner).

<div class="stats-grid">
  <div class="stat-card"><span class="value">14</span><span class="label">participants</span></div>
  <div class="stat-card"><span class="value">90</span><span class="label">dyadic interactions</span></div>
  <div class="stat-card"><span class="value">30</span><span class="label">recording sessions</span></div>
  <div class="stat-card"><span class="value">8</span><span class="label">recurring dyads</span></div>
  <div class="stat-card"><span class="value">180</span><span class="label">role instances</span></div>
  <div class="stat-card"><span class="value">19</span><span class="label">locations</span></div>
  <div class="stat-card"><span class="value">22</span><span class="label">relationship settings</span></div>
  <div class="stat-card"><span class="value">1:40-10:00</span><span class="label">interaction duration</span></div>
</div>

<div class="callout">
  <strong>Release focus.</strong> This website is organized as a dataset paper companion. It emphasizes collection, curation, metadata design, access, and ethics, and presents representative research directions rather than a finalized benchmark suite.
</div>

## Why DualFlow

- **Social-first design.** The primary unit is a dyadic interaction episode rather than an isolated action clip or single-user task.
- **Native eye tracking.** Egocentric scene video and eye-related streams are recorded from the same wearable device, enabling synchronized study of gaze behavior, eye states, blink dynamics, and pupil-linked signals.
- **Face + body release.** The dataset provides both animation-friendly body motion and facial-performance parameters, making it useful for analysis, generation, digital humans, and embodied-agent work.
- **Conversation-ready curation.** Audio cleaning, aligned transcripts, privacy-preserving video transformation, and timeline synchronization are part of the release pipeline rather than afterthoughts.
- **Structured metadata.** Participant anthropometrics, role-centered personality ratings, and social-interpretation annotations broaden the dataset beyond raw sensor streams alone.

![Eye-tracking stream teaser](minieye.gif)

## What is released

| Modality group | Released representation | Notes |
| --- | --- | --- |
| Egocentric video | Identity-obscured scene video | Facial mosaicing is applied before release. |
| Eye tracking | Full Pupil Cloud export + mocap-localized 3D gaze vectors | Supports gaze, fixations, saccades, 3D eye states, blinks, and IMU-aligned analysis. |
| Facial performance | FLAME and ARKit-compatible parameters | Derived from synchronized head-mounted facial capture. |
| Body motion | FBX and SMPL-H | 120 Hz motion capture including fingers. |
| Audio and text | Consent-aware audio + aligned transcripts | Audio may be original or transformed depending on consent. |
| Metadata and annotations | Anthropometrics, role-centered questionnaires, social-interpretation annotations | Organized across participants, sessions, interactions, and role instances. |

![ego-a](gifs/ego_video_a.gif) ![ego-b](gifs/ego_video_b.gif)

## Diversity and scenarios

DualFlow spans **19 locations**, **22 relationship settings**, and **five familiarity categories**: strangers, acquaintances, friends, close friends, and family. The participant pool includes **professional actors, acting students, and non-professional performers**, and the released manifests cover English, Chinese, Spanish, and Portuguese language backgrounds.

## Use this site

- [Dataset]({{< relref "dataset.md" >}}) for the download link, release scope, modalities, and metadata.
- [Collection and curation]({{< relref "curation.md" >}}) for the recording workflow, hardware stack, synchronization, and quality control.
- [Representative research directions]({{< relref "benchmarks.md" >}}) for the task families supported by the current release.
- [Data organization]({{< relref "organization.md" >}}) for manifests, identifiers, and recommended directory layout.
- [Access and license]({{< relref "access.md" >}}) for the gated release workflow and research-use conditions.
- [Ethics and privacy]({{< relref "ethics.md" >}}) for consent boundaries, identity protection, and residual-risk notes.
- [Citation]({{< relref "citation.md" >}}) and [FAQ]({{< relref "faq.md" >}}) for paper status, temporary citation text, and common access questions.




The website is intended to host documentation, review materials, supplementary assets, and request instructions for the DualFlow release. Public materials such as schemas and helper code can be hosted openly, while the core multimodal package is distributed through a research-use agreement.
