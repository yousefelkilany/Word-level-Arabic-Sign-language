---
title: loader.py
date: 2026-01-28
lastmod: 2026-02-01
src_hash: 389b9c31f608384bdd8e467c280b213f39eddf0a0bd972efb7914b2fa49cfce0
aliases: ["Dashboard Data Loader", "Streamlit Caching Wrappers"]
---

# loader.py

#source #dashboard #data-loading #caching

**File Path**: `src/modelling/dashboard/loader.py`

**Purpose**: Data access layer for the dashboard. Handles efficient loading of models and datasets using Streamlit caching.

## Overview

Provides cached wrappers around core functions to ensure the dashboard remains responsive while handling large models and datasets.

## Functions

### `get_checkpoints_metadata(checkpoint_path)`
**Decorator**: `@st.cache_data`
Extracts both class count and architecture from the checkpoint file.
**Returns**: `(num_signs, ModelSize)`

### `load_cached_checkpoints(checkpoints_dir)`
**Decorator**: `@st.cache_data`
Recursively scans the directory for `.pth` files to populate the selection sidebar.

### `load_cached_model(checkpoint_path, _metadata)`
**Decorator**: `@st.cache_resource`
Loads the PyTorch model into memory on the configured device and sets it to evaluation mode.

### `get_cached_dataloaders(num_signs)`
**Decorator**: `@st.cache_resource`
Initializes and caches lazy dataloaders for all three data splits (`train`, `val`, `test`).

### `run_inference(_model, _dataloader, ...)`
**Decorator**: `@st.cache_data`
Executes a full inference pass on the provided dataloader. Displays a progress bar in the UI.
**Returns**: `(y_true, y_pred, y_probs)` as NumPy arrays.

## Related Documentation

- [[../../data/dataloader_py|dataloader.py]] - Underlying data provider logic.
- [[../../core/utils_py|utils.py]] - `extract_metadata_from_checkpoint` implementation.
- [[../model_py|model.py]] - `load_model` implementation.
- [[app_py|app.py]] - Consumer of these loader functions.
