---
description: Update the root README.md based on the latest documentation and source code changes.
---

# Update README Workflow

1. Analyze recent significant changes in the `src/` directory and conceptual documentation in `docs/`.
   - Check `docs/getting_started.md` for installation/usage changes.
   - Check `docs/architecture_overview.md` and `docs/models/architecture_design.md` for model or architecture changes.
   - Check `docs/development/project_structure.md` for directory layout changes.

2. Compare the current root `README.md` with these "source" documentation files.
   - Section Mapping:
     - `## Features` -> `docs/index.md` or `docs/architecture_overview.md`.
     - `## Configuration` -> `docs/deployment/environment_configuration.md`.
     - `## Installation` -> `docs/getting_started.md`.
     - `## Usage` -> `docs/getting_started.md`.
     - `## Repository Structure` -> `docs/development/project_structure.md`.
     - `## Model Architecture` -> `docs/models/architecture_design.md`.
     - `## Resources` -> `docs/index.md` or `docs/reference/dataset_citation.md`.

3. Update the root `README.md` to reflect the latest state, ensuring it remains a concise but comprehensive overview.
   - Follow the formatting and standards defined in `.agent/workflows/GUIDE.md`.
   - Preserve the `bibtex` citation and project links.

4. Verify that all links in the updated `README.md` are functional and point to the correct locations.
