# draw_kps.py

#source #core #visualization #opencv

**File Path**: `src/core/draw_kps.py`

**Purpose**: Utilities to visualize extracted landmarks on images using MediaPipe's drawing styles.

## Overview

Converts raw numpy keypoint arrays back into MediaPipe's `NormalizedLandmark` format to leverage their built-in drawing utilities (`draw_landmarks`). Supports drawing individual body parts or the full skeleton.

## Functions

### `get_lms_list(...)`
Converts a numpy array segment into a list of `landmark_pb2.NormalizedLandmark`.
- **Input**: `(N, 3)` or `(N, 4)` array.
- **Output**: List of Protobuf objects with `.x`, `.y`, `.z`.

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
- **Config**: Hardcoded `signer`, `split`, `word`.
- **Logic**: Loads the `.npz` file, reads the corresponding video frames, and saves an annotated side-by-side comparison.

## Key Connections

| Body Part | Connection Set         | Style                                     |
| :-------- | :--------------------- | :---------------------------------------- |
| **Pose**  | `POSE_KPS_CONNECTIONS` | `get_default_pose_landmarks_style`        |
| **Face**  | `FACE_KPS_CONNECTIONS` | `get_default_face_mesh_tesselation_style` |
| **Hands** | `HAND_KPS_CONNECTIONS` | `get_default_hand_landmarks_style`        |

## Related Documentation

**Depends On**:
- [[../../source/core/mediapipe_utils_py|mediapipe_utils.py]] - Connection definitions.

**Used By**:
- [[../../source/api/live_processing_py|live_processing.py]] - Debug visualization frames.
- [[../../source/modelling/dashboard/visualization_py|Database Visualization]] - Streamlit component.
