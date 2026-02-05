---
title: generate_mediapipe_face_symmetry_map.py
date: 2026-02-05
lastmod: 2026-02-03
src_hash: 6fd36775876621263d7d8d81cc8858085ddb97e49059cc5b68426e61c715b8e4
aliases: ["Face Symmetry Mapping", "Landmark Mirroring"]
---

# generate_mediapipe_face_symmetry_map.py

#source #scripts #mediapipe #mathematics

**File Path**: `scripts/generate_mediapipe_face_symmetry_map.py`

**Purpose**: Generates a bidirectional mapping of MediaPipe Face Mesh landmarks to their mirror equivalents (left side to right side).

## Overview

This script uses a reference image of a front-facing face to detect 478 landmarks. It then applies the **Hungarian Algorithm** (via `scipy.optimize.linear_sum_assignment`) to find the optimal mirror pairs by minimizing the Euclidean distance between points and their horizontally flipped versions.

## Functions

### `init_face_model()`
Initializes the MediaPipe Face Landmarker task.

### `get_face_mesh_symmetry_indices(...)`
Detects landmarks and calculates the symmetry map using `cdist` and `linear_sum_assignment`.

### `gen_symmetry_map(...)`
Wraps the detection with sanity checks:
- Verifies all 478 points are mapped uniquely.
- Checks specific known pairs (e.g., eye corners).
- Ensures the mapping is bidirectional ($A \to B \implies B \to A$).

## Execution

When run directly, it:
1. Loads two reference images (`frontal-face-1.jpg`, `frontal-face-2.jpg`).
2. Confirms the symmetry map is identical for both.
3. Saves the mapping as a `.npy` file to `FACE_SYMMETRY_MAP_PATH`.

## Related Documentation

**Used By**:
- [[generate_mediapipe_simplified_face_kps_py|generate_mediapipe_simplified_face_kps.py]] - To ensure simplified contours remain symmetric.
