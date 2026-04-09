+++
title = "Dataset"
slug = "dataset"
+++

# Dataset

Dataset can be currently access [here](https://drive.google.com/drive/folders/10VXA4nWM4lNd5JxDo_33nAFeB1iKor7C?usp=drive_link).

DualFlow is a multimodal dataset of dyadic social interaction built around synchronized face-to-face episodes rather than isolated gestures or single-user tasks. The dataset is designed to support analysis of **where people look**, **how they move**, **how facial behavior evolves**, **what they say**, and **how these channels interact over time**.

## Overview

| Item | Value |
| --- | --- |
| Participants | 14 |
| Age range | 21-51 |
| Mean age | 28.9 |
| Gender | 8 male, 5 female, 1 non-binary |
| Native languages | 10 English, 2 Chinese, 1 Spanish, 1 Portuguese |
| Acting background | 5 professional, 2 acting students, 7 non-professional |
| Interactions | 90 |
| Recording sessions | 30 |
| Recurring dyads | 8 |
| Role instances | 180 |
| Locations | 19 |
| Relationship settings | 22 |
| Familiarity categories | strangers, acquaintances, friends, close friends, family |
| Interaction duration | 1:40 to 10:00 |

## Released modalities

| Modality group | Capture or curation source | Released representation |
| --- | --- | --- |
| Egocentric video | Pupil Labs Neon scene camera | Identity-obscured egocentric video |
| Eye tracking | Pupil Labs Neon + Pupil Cloud export | Full Pupil Cloud timeseries export and mocap-localized 3D gaze vectors |
| Body motion | Vicon Shogun optical motion capture | FBX, SMPL-H, synchronized motion sequences |
| Facial performance | Rokoko Headcam | FLAME and ARKit-compatible facial-parameter streams |
| Audio | Integrated Neon microphones + curation pipeline | Consent-aware cleaned audio |
| Text | Transcript alignment pipeline | Aligned transcripts |

## Metadata and annotations

DualFlow includes more than sensor streams. The release is designed so that stable participant identity, session-level context, interaction structure, and role-specific interpretation can all be studied separately.

### Participant metadata

Participant-level metadata include demographic attributes and **manual anthropometric measurements** such as total height, torso height, neck height, head height, shoulder width, upper-arm length, forearm length, hand length, thigh length, lower-leg length, and inseam. These measurements support shape-aware analysis and generation pipelines.

### Role-centered annotations

Each interaction contains two enacted roles. Role-centered questionnaires are linked to these **role instances** rather than to stable participant identity, which is important because the same performer may enact different characters across sessions.

The current release is designed to support:

- role-centered personality ratings based on IPIP-style items and BFI-10 style summaries;
- participant self-assessment and partner-role assessment after recording;
- analysis of repeated performers across multiple role configurations.

### Social-interpretation annotations

DualFlow also contains higher-level **social-interpretation annotations** collected from annotators with different cultural backgrounds. These annotations are intended to support research on perceived social intention, role understanding, multimodal description, and language interfaces for social behavior analysis.

## Scenario structure

Each interaction is built around a scenario prompt specifying:

- the social setting;
- the assumed relationship between the two roles;
- a high-level interaction goal.

The prompts provide enough structure to ensure broad coverage while still allowing improvised verbal and nonverbal behavior. This keeps the release socially rich without making every take fully unconstrained.

## Notes on identifiers and canonical names

The release uses stable identifiers for **participants**, **sessions**, **interactions**, and **role instances**. When metadata appear at multiple levels, the **interaction-level manifest is treated as canonical** for location and relationship naming. This keeps the website and future manifests aligned with the actual interaction units used for analysis and evaluation.
