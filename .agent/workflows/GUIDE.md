# README Update Guide for LLMs

This guide provides instructions for AI agents to maintain the root `README.md` of the project.

## 🎯 Objectives

- Keep the `README.md` in sync with high-level documentation in `docs/`.
- Maintain a professional, clear, and inviting presentation for developers and researchers.
- Ensure all technical details (commands, paths, features) are accurate.

## 📝 Section-Specific Rules

### 1. Header & Intro

- Keep the title: `# Word-Level Arabic Sign Language Recognition`.
- The intro should mention **KArSL-502**, **MediaPipe**, and **Attention-based BiLSTM**.

### 2. Features

- Highlight the core strengths: Real-time, Web Interface, Deep Learning Model, Optimized Pipeline.

### 3. Installation & Usage

- **ALWAYS** include both Docker (Recommended) and Local (uv) options.
- Ensure commands match `Makefile` targets where possible.
- Update `uv sync` or `pip install` commands if dependency management changes.

### 4. Repository Structure

- Keep it high-level. Only mention top-level directories and their main purpose.
- Synchronize with `docs/development/project_structure.md`.

### 5. Model Architecture

- This section is highly technical. Ensure it reflects changes in `src/modelling/model.py` and `docs/models/architecture_design.md`.
- Use subheadings for Spatial Group Embedding, Residual Temporal Processing, Multi-Head Self-Attention, etc.

### 6. Resources & Citation

- Ensure the Kaggle and Google Drive links are correct.
- **DO NOT** modify the BibTeX citation unless specifically requested.

## 🛠️ Technical Standards

- Use **GitHub Flavored Markdown**.
- Use code blocks with appropriate language tags (e.g., `bash`, `python`).
- Use tables for configuration variables.
- Ensure all relative links from root work correctly.

> [!TIP]
> Before updating the `README.md`, ensure that all conceptual and source documentation in `docs/` is up to date by running the `/sync-docs` workflow. Refer to `docs/SOURCE_DOCS_GUIDE.md` for standards on source-linked documentation.

## ⚡ Sync Trigger

Perform this update whenever:

- A new core feature is added.
- The installation/setup process changes.
- The repository structure is significantly refactored.
- The model architecture is modified.
