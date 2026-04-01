# DualFlow dataset website

This repository contains the Hugo source for the DualFlow dataset website, prepared for deployment on GitHub Pages with GitHub Actions.

## Main pages

- Home
- Dataset
- Collection and curation
- Representative research directions
- Data organization
- Access and license
- Ethics and privacy
- Citation
- FAQ

## Stack

- Hugo
- Bearblog theme
- GitHub Pages
- GitHub Actions

## Local preview

```bash
hugo server
```

## Deploy to GitHub Pages

1. Keep the Hugo source in the repository root.
2. In GitHub, open **Settings -> Pages**.
3. Set **Source** to **GitHub Actions**.
4. Push to the `main` branch.
5. Wait for the workflow in **Actions** to finish.
6. GitHub Pages will publish the generated `public/` artifact.

## Notes

- Large or access-controlled dataset files should not be stored in this repository.
- The website is intended to host documentation, supplementary material, example assets, and access instructions.
- Public helper code, schemas, and parsers can be linked from this site once they are ready.
