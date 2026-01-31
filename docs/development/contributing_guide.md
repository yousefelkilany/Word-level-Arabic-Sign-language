---
title: Contributing Guide
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Developer Onboarding", "Contribution Standards"]
---

# Contributing Guide

#development #contributing #standards

We welcome contributions! This guide will help you set up your development environment and understand our coding standards.

## Development Setup

We use **uv** for fast package management.

### Prerequisites
- Python 3.12+
- `uv` (Install via `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yousefelkilany/word-level-arabic-sign-language.git
    cd word-level-arabic-sign-language
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```
    This will create a virtual environment in `.venv`.

3.  **Activate environment**:
    - Windows: `.venv\Scripts\activate`
    - Linux/Mac: `source .venv/bin/activate`

## Development Workflow

1.  **Create a Branch**: `git checkout -b feature/my-new-feature`
2.  **Make Changes**: Edit code in `src/`.
3.  **Run Tests**: (Add instructions if tests exist, currently manual verification).
4.  **Format Code**: We use `ruff` (implied by modern standards, though check if installed).
5.  **Submit PR**: Push to GitHub and open a Pull Request.

## Coding Standards

- **Type Hints**: All function signatures must have type hints.
- **Docstrings**: Use Google-style docstrings.
- **Path Handling**: Use `pathlib` or `os.path.join`. Avoid hardcoded separators.

## Related Documentation

- [[../development/project_structure|Project Structure]]
- [[../development/makefile_commands|Makefile Commands]]
