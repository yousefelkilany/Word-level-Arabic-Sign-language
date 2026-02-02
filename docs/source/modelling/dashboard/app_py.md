---
title: app.py
date: 2026-01-28
lastmod: 2026-02-01
src_hash: bce8d36dbc3ceaf5924f3d8c2ec149040d60e10ccd1b24e7d0b73b30ca2e3663
aliases: ["Dashboard Entry Point", "Streamlit App Main"]
---

# app.py

#source #dashboard #streamlit #analytics

**File Path**: `src/modelling/dashboard/app.py`

**Purpose**: Entry point for the Streamlit-based KArSL Analytics Dashboard.

## Overview

The dashboard provides a visual interface to explore model performance, inspect specific data samples, and experiment with data augmentations. It uses a sidebar for global configuration and a tabbed interface for focused analysis.

## Dashboard Structure

### Sidebar Configuration
- **Checkpoint Selector**: Lists all `.pth` files found in `TRAIN_CHECKPOINTS_DIR`.
- **Split Selector**: Choice between `train`, `val`, and `test` splits.
- **Run Evaluation**: Triggers the inference pipeline for the selected model and split.

### Analysis Tabs

1. **Global Metrics**: (Conditional) Displays confusion matrices and per-class performance plots.
2. **Error Analysis**: (Conditional) Detailed view of prediction probabilities and specific misclassification instances.
3. **Sample Inspector**: Explores the raw keypoints and labels of the dataset.
4. **Augmentation Lab**: Interactive playground for testing `DataAugmentor` transformations.

---

## Workflow

1. **State Management**: Uses `st.session_state` to persist evaluation results and random visualization keys.
2. **Caching**: Leverages `st.cache_data` and `st.cache_resource` (via `loader.py`) for efficient model/data loading across reruns.
3. **Dynamic Tabs**: The "Global Metrics" and "Error Analysis" tabs only appear after an evaluation has been successfully run.

## Key Functions

### `main()`

Orchestrates the sidebar logic, state initialization, and tab rendering.

## Related Documentation

- [[loader_py|loader.py]] - Backend logic for data and model caching.
- [[views_py|views.py]] - UI components for each dashboard tab.
- [[../train_py|train.py]] - Source of the training checkpoints.
