---
title: draw_kps.py
date: 2026-01-28
lastmod: 2026-02-05
src_hash: edf51f81ce412fb05f3fecdfa3aee5606f123b84ff0c0ec38556e053a584c6c2
aliases: ["Landmark Visualization", "Keypoint Drawing Utilities"]
---

# draw_kps.py

#source #core #visualization #opencv

**File Path**: `src/core/draw_kps.py`

**Purpose**: Utilities to visualize extracted landmarks on images using MediaPipe's drawing styles.

## Overview

Converts raw numpy keypoint arrays back into MediaPipe's `NormalizedLandmark` format to leverage their built-in drawing utilities (`draw_landmarks`). Supports drawing individual body parts or the full skeleton.

## Functions

### `get_lms_list(...)`
Converts a numpy array segment into a list of `landmark_pb2.NormalizedLandmark` or a numpy array of coordinates.
- **Input**: `(N, 3)` or `(N, 4)` array, `kps_idx`, `lms_num`, `return_as_lm=True`.
- **Output**: List of MediaPipe Landmark objects or numpy array of coordinates.

### `get_pose_lms_list(...)`, `get_face_lms_list(...)`, `get_hand_lms_list(...)`
Specialized body-part getters that wrap `get_lms_list`.

### `draw_kps_on_image(...)`
Generic drawer.
- **Args**: Image, landmarks list, connections list, styling.
- **Action**: Overlays the landmarks and connections onto the image.

### Usage Functions
- `draw_pose_kps_on_image`: Draws the 6 upper-body pose points.
- `draw_face_kps_on_image`: Draws the FaceMesh tesselation.
- `draw_hand_kps_on_image`: Draws hand skeleton (21 points).

### `draw_all_kps_on_image(rgb_image, frame_kps, ...)`
**Purpose**: Comprehensive visualizer.
1. Slices the flat `frame_kps` using `KP2SLICE`.
2. Draws Pose -> Face -> Right Hand -> Left Hand sequentially.
3. Can optionally return separate images for each part.

## Execution (`__main__`)

Debug script to visualize a specific sample.
- **Config**: Hardcoded `signer`, `split`, `sign`.
- **Logic**: Loads the `.npz` file, reads the corresponding video frames, and saves an annotated side-by-side comparison.

## Key Connections

| Body Part | Connection Set         | Style                                     |
| :-------- | :--------------------- | :---------------------------------------- |
| **Pose**  | `POSE_KPS_CONNECTIONS` | `get_default_pose_landmarks_style`        |
| **Face**  | `FACE_KPS_CONNECTIONS` | `get_default_face_mesh_tesselation_style` |
| **Hands** | `HAND_KPS_CONNECTIONS` | `get_default_hand_landmarks_style`        |

## Related Documentation

**Depends On**:
- [[mediapipe_utils_py|mediapipe_utils.py]] - Connection definitions.

**Used By**:
- [[../api/live_processing_py|live_processing.py]] - Debug visualization frames.
- [[../modelling/dashboard/visualization_py|Database Visualization]] - Streamlit component.
