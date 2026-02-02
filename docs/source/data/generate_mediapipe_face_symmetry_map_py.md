---
title: generate_mediapipe_face_symmetry_map.py
date: 2026-01-28
lastmod: 2026-01-29
src_hash: 41953e673940d060822d3805c58701e0c0814a924505763020b6db14b0248816
aliases: ["Face Symmetry Mapping", "Horizontal Flip Mapping Generator"]
---

# source/data/generate_mediapipe_face_symmetry_map.py

#source-code #data #mediapipe #augmentation #face-mesh

**File Path**: `src/data/generate_mediapipe_face_symmetry_map.py`

**Purpose**: Generates a symmetry mapping array for MediaPipe Face Mesh landmarks to enable correct horizontal flipping.

## Overview

MediaPipe's 478 face landmarks are not perfectly symmetrical by index (e.g., left eye point index is not simply right eye point index + offset). This script calculates the corresponding "mirror" index for every landmark by analyzing a frontal face image.

## Key Functions

### `gen_symmetry_map(face_model, image_path)`

Calculates the symmetry indices.
1. **Detect**: extract face landmarks from a reference image.
2. **Center**: Centralize points around the mean.
3. **Flip**: Create a horizontally flipped copy of the points.
4. **Match**: Use linear sum assignment (Hungarian Algorithm) to find the closest matching point in the original set for each point in the flipped set.
5. **Validate**: Perform sanity checks on known pairs (e.g., eye corners) to ensure mapping correctness.

**Returns**: NumPy array where `arr[i]` is the symmetric index of point `i`.

### `get_face_mesh_symmetry_indices(face_model, image_path)`

Helper coroutine that performs the actual geometric matching logic.

## Usage

Generated file `face_symmetry_map.npy` is used during data augmentation to properly flip face landmarks.

## Related Documentation

- [[data_preparation_py|data_preparation.py]] - Uses the symmetry map for `hflip`.
- [[../core/mediapipe_utils_py|mediapipe_utils.py]] - MediaPipe constants.

---

**File Location**: `src/data/generate_mediapipe_face_symmetry_map.py`
