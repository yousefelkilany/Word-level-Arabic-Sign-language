---
title: loader.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Dashboard Data Loader", "Streamlit Caching Wrappers"]
---

# source/modelling/dashboard/loader.py

#source-code #dashboard #data-loading #caching

**File Path**: `src/modelling/dashboard/loader.py`

**Purpose**: Data access layer for the dashboard. Handles efficient loading of models and datasets using Streamlit caching.

## Functions

### `get_checkpoints_num_signs(checkpoint_path)`
**Decorator**: `@st.cache_data`
Extracts class count from checkpoint file.

### `load_cached_checkpoints(checkpoints_dir)`
**Decorator**: `@st.cache_data`
Scans directory for `.pth` files.

### `load_cached_model(checkpoint_path, num_signs)`
**Decorator**: `@st.cache_resource`
Loads the PyTorch model and sets it to eval mode.

**Calls**: [[../model_py#load_model|model.load_model()]]

### `get_cached_dataloaders(num_signs)`
**Decorator**: `@st.cache_resource`
Creates lazy dataloaders for Train/Val/Test splits.

### `run_inference(_model, _dataloader, device, ...)`
**Decorator**: `@st.cache_data`
Runs full inference pass on a split.
- **Returns**: `(y_true, y_pred, y_probs)` as numpy arrays.

## Related Documentation

- [[../../data/dataloader_py|dataloader.py]]
- [[app_py|app.py]] - Consumer

---

**File Location**: `src/modelling/dashboard/loader.py`
