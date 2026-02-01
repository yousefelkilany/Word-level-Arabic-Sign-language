---
title: visualization.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["3D Skeleton Plotting", "Interactive Plotly Visualizer"]
---

# source/modelling/dashboard/visualization.py

#source-code #dashboard #plotly #3d

**File Path**: `src/modelling/dashboard/visualization.py`

**Purpose**: Low-level 3D plotting logic using Plotly Graph Objects.

## Functions

### `plot_3d_animation(sequence, active_slices_map, ...)`

Generates the interactive 3D Figure.
- **Animation**: Creates `go.Frame` for each timestep.
- **Interactivity**: Play/Pause buttons, Time Slider.
- **Structure**: Handles separate traces for Pose, Left Hand, Right Hand, Face.

### `_generate_frame_traces(...)`
Internal helper. Creates the 3D Scatter (points) and 3D Lines (edges) for a single frame.
- **Logic**: Iterates over body parts and draws connections (e.g., `HAND_KPS_CONNECTIONS`).

### `calculate_layout_ranges(...)`
Computes the global bounding box `(min, max)` for X, Y, Z across the *entire* sequence. 
- **Why**: Prevents the camera/axis from resizing strictly during animation, keeping the view stable.

### `get_face_camera_view(points_nx3)`
Calculates optimal camera angle for face landmarks using SVD (Singular Value Decomposition) to normalize orientation.

## Related Documentation

- [[views_py|views.py]] - Main consumer
- [[../../core/draw_kps_py|draw_kps.py]] - Connection definitions

---

**File Location**: `src/modelling/dashboard/visualization.py`
