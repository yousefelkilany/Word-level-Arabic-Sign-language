---
title: source/modelling/dashboard/app.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Dashboard Entry Point", "Streamlit App Main"]
---

# source/modelling/dashboard/app.py

#source-code #dashboard #streamlit #analytics

**File Path**: `src/modelling/dashboard/app.py`

**Purpose**: Entry point for the Streamlit Analytics Dashboard. Manage layout, state, and view navigation.

## Overview

Initializes the Streamlit application, handles sidebar inputs (checkpoints, splits), manages session state for caching inference results, and renders different tabs based on user selection.

## Functions

### `main()`

#function #entry-point

**Purpose**: Main execution function for the dashboard.

**Logic**:
1. **Sidebar**:
   - Loads cached checkpoints via [[loader_py#load_cached_checkpoints|load_cached_checkpoints]].
   - Selects Data Split using `SplitType`.
   - Button "Run Evaluation" triggers inference.
2. **State Management**:
   - Initializes `results`, `inspector_rnd_key`, etc.
3. **Routing**:
   - if `results` exist: Shows Global Metrics, Error Analysis.
   - Always shows: Sample Inspector, Augmentation Lab.
4. **Rendering**:
   - Calls view functions from [[views_py|views.py]].

**Calls**:
- [[loader_py#load_cached_checkpoints|loader.load_cached_checkpoints()]]
- [[loader_py#load_cached_model|loader.load_cached_model()]]
- [[loader_py#run_inference|loader.run_inference()]]
- [[views_py#render_metrics_view|views.render_metrics_view()]]
- [[views_py#render_error_view|views.render_error_view()]]
- [[views_py#render_inspector_view|views.render_inspector_view()]]

## Related Documentation

- [[../../../models/training_process|Training Process]]
- [[views_py|views.py]] - UI Components
- [[loader_py|loader.py]] - Data Access

---

**File Location**: `src/modelling/dashboard/app.py`
