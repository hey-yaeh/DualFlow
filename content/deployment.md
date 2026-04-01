+++
title = "Deployment guide"
menu = ""
weight = 6
+++

This website is configured for deployment on GitHub Pages using GitHub Actions.

## Recommended workflow

1. Keep the Hugo source in the `main` branch.
2. Let GitHub Actions build the site.
3. Let GitHub Pages publish the generated `public/` artifact.

## What to host here

This repository is best used for:

- dataset overview
- benchmark definitions
- data organization
- access and licensing
- documentation
- citation
- low-risk examples

Sensitive or access-controlled multimodal files should be hosted separately and linked from the website.
