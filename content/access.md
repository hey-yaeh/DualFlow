+++
title = "Access and license"
slug = "access"
+++

# Access and license

DualFlow adopts a **tiered, consent-aware release model**. Public website materials can be shared openly, while the core multimodal recordings and participant-linked annotations are distributed through a **research-use agreement**.

## What is public

The following materials can be hosted openly on the website or in public repositories:

- dataset documentation and schema descriptions;
- modality and manifest descriptions;
- helper code for loading or parsing the release;
- citation information;
- access instructions;
- selected low-risk examples and supplementary assets.

## What is gated

Access to the core release is controlled because the dataset contains sensitive multimodal recordings of social interaction. The gated package may include:

- identity-obscured egocentric video;
- full Pupil Cloud eye-tracking exports;
- mocap-localized 3D gaze vectors;
- FBX and SMPL-H body motion;
- FLAME and ARKit-compatible facial parameters;
- consent-aware audio;
- aligned transcripts;
- participant-linked and role-linked annotations.

## Who can request access

The current access pathway is intended for **non-commercial academic research and teaching**. If your use case falls outside that scope, please contact the maintainers before assuming access is available.

## How to request access

Please email **hey3@tcd.ie** and **jovanea@tcd.ie** with:

1. your name, affiliation, and institutional webpage if available;
2. a short description of the intended research or teaching use;
3. confirmation that the request is for non-commercial use;
4. confirmation that you agree to the DualFlow research-use conditions.

If the request is approved, the maintainers can provide the current access instructions and any required agreement documents.

<div class="callout">
  <strong>Access note.</strong> The public website is not the download endpoint for the core recordings. It is the documentation and request hub for the gated release.
</div>

## Research-use conditions

Approved access is expected to prohibit the following without prior written permission:

- redistribution of any gated files;
- re-identification of participants;
- biometric identification or speaker identification;
- attempts to reverse facial obfuscation or voice transformation;
- sharing data with unauthorized third parties.

Approved users are also expected to:

- store the data securely;
- limit access to authorized project personnel;
- keep participant-linked material confidential;
- cite DualFlow in resulting publications.

## Consent-aware release boundaries

Release follows the participant consent protocol:

- egocentric videos are distributed only in identity-obscured form;
- audio may be distributed in original or transformed form depending on consent;
- body motion, facial parameters, eye-tracking signals, and transcripts are released only within the scope permitted by the signed consent forms.

## Third-party dependencies

DualFlow may provide derived exports or parameter streams that are compatible with third-party models or formats such as **SMPL-H**, **FLAME**, and **ARKit-compatible facial parameters**. The release does **not** redistribute third-party model assets themselves, and users remain responsible for complying with any upstream license terms.

## Research-use agreement summary

A plain-language summary of the expected conditions is available on the [research-use agreement page]({{< relref "agreement.md" >}}). For projects that require a more formal institutional workflow, the maintainers can provide the applicable access paperwork during review.
