---
title: views.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Dashboard Tabs Rendering", "UI View Components"]
---

# source/modelling/dashboard/views.py

#source-code #dashboard #visualization #ui

**File Path**: `src/modelling/dashboard/views.py`

**Purpose**: high-level UI component rendering functions for different dashboard tabs.

## Functions

### `render_metrics_view(y_true, y_pred, num_signs)`
Displays classification metrics.
- **Components**:
  - Overall Accuracy (`st.metric`).
  - Classification Report (Precision, Recall, F1) as styled dataframe.
  - Confusion Matrix (Plotly heatmap).

### `render_error_view(y_true, y_pred, y_probs)`
Analyzes misclassifications/
- **Components**:
  - "Top Confused Pairs" horizontal bar chart.

### `render_inspector_view(rnd_key, dataloader, model)`
Interactive single-sample inspector.
- **Features**:
  - Random sample selection.
  - 3D Skeleton Animation (Plotly).
  - Real-time model prediction (if model loaded).
  - Confidence Bar visualization.

**Calls**: 
- [[visualization_py#plot_3d_animation|plot_3d_animation()]]
- [[../../data/shared_elements_py#get_visual_controls|shared_elements.get_visual_controls()]]

### `render_augmentation_view(rnd_key, dataloader)`
Playground for testing data augmentations.
- **Controls**: Flip, Affine, Rotate, Scale, Shift.
- **Visuals**: Side-by-side comparison of "Original" vs "Augmented" 3D skeletons.

**Calls**:
- [[../../data/data_preparation_py#DataAugmentor|DataAugmentor]]

---

**File Location**: `src/modelling/dashboard/views.py`
