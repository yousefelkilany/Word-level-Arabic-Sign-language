---
title: shared_elements.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Shared Dashboard Components", "Streamlit UI Widgets"]
---

# shared_elements.py

#source #data #streamlit

**File Path**: `src/data/shared_elements.py`

**Purpose**: UI components shared across Streamlit dashboard pages.

## Overview

Provides standardized widgets for selecting samples and configuring visualization options (body parts, drawing styles) to ensure consistency across different analysis views.

## Functions

### `get_visual_controls(total_samples, rnd_key)`
**Returns**: `(idx, draw_lines, draw_points, separate_view, active_slices)`

**UI Elements Created**:
- **Number Input**: "Sample Index".
- **Checkboxes**: "Lines", "Points", "Only one" (View Mode).
- **Conditionals**:
  - If "Only one": Selectbox for Body Part (Pose/Face/RH/LH).
  - If "Separated": Multi-checkboxes for enabling parts.

**Logic**:
- Uses `rnd_key` to ensure unique widget IDs when reused.
- Maps user selection to `active_slices` using `KP2SLICE`.

## Related Documentation

**Depends On**:
- [[../core/mediapipe_utils_py|mediapipe_utils.py]] - `KP2SLICE`.

**Used By**:
- [[../modelling/dashboard/visualization_py|Visualization Dashboard]]
