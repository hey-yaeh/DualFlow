+++
title = "Data organization"
slug = "organization"
+++

# Data organization

DualFlow is organized around **participants**, **sessions**, **interactions**, and **role instances**. This hierarchy makes it possible to analyze repeated performers across multiple sessions while keeping role-specific interpretation separate from stable participant identity.

## Core manifest tables

The current release is designed around manifest tables like the following:

| Table | Purpose |
| --- | --- |
| `participants` | Stable participant identifiers, demographics, language background, acting experience |
| `consent` | Modality-specific release permissions |
| `participant_anthropometrics` | Manual body measurements for shape-aware analysis |
| `sessions` | Session-level metadata such as dyad membership, language, and interaction counts |
| `interactions` | Interaction identifiers, participant pairing, canonical location and relationship naming, and scenario-level context |
| `interaction_roles` | Role-instance identifiers and role assignment metadata |
| `role_personality_item_responses` / `role_personality_scores` | Role-centered questionnaire responses and derived summaries |
| `social_interpretation_annotations` | Higher-level interpretation annotations and free-text descriptions |

## Canonical identifiers

A typical naming scheme follows the existing file-prefix conventions:

| Prefix | Meaning |
| --- | --- |
| `S` | session |
| `I` | interaction |
| `R` | role instance |
| `P` | participant |
| `EV` | ego video |
| `MEV` | masked ego video |
| `M` | motion asset |

Examples:

- session: `S001`
- interaction: `S001_I01`
- role instance: `S001_I01_R01`
- participant: `P01`

## Canonical naming rule

Location and relationship names should be treated as **interaction-level metadata**. When a higher-level summary and an interaction-level manifest disagree, the **interaction-level manifest is the canonical source**. This keeps naming aligned with the actual analysis unit used in the release.

## Recommended directory structure

```text
DualFlow/
  manifests/
    participants.csv
    consent.csv
    participant_anthropometrics.csv
    sessions.csv
    interactions.csv
    interaction_roles.csv
    role_personality_item_responses.csv
    role_personality_scores.csv
    social_interpretation_annotations.csv
  sessions/
    S001/
      sync/
      calibration/
      interactions/
        S001_I01/
          interaction.json
          roles.json
          ego/
          eye_tracking/
          motion/
          face/
          audio/
          transcripts/
          derived/
```

## Why the hierarchy matters

This layout helps separate:

- **stable participant information** from session-specific and role-specific context;
- **capture metadata** from downstream released representations;
- **interaction-level analysis units** from reusable participant-level descriptors.

It also supports participant-disjoint, session-disjoint, or dyad-disjoint splits without forcing those decisions into the physical folder layout.

## Practical guidance

### Participant level

Use participant-level tables only for metadata that remain stable across the dataset, such as demographics, language background, or anthropometrics.

### Session level

Use session-level tables for capture context, participant pairing, language, and synchronization notes.

### Interaction level

Use interaction-level tables for location, relationship framing, scenario prompt, familiarity category, start or end boundaries, and any benchmark split assignment.

### Role level

Attach role personality and interpretation data to **role instances**, not directly to participant IDs. The same person may enact very different roles across sessions, so role-centered analysis should remain role-centered in the manifests as well.
