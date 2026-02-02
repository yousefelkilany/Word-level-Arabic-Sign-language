---
title: mediapipe_utils.py
date: 2026-01-28
lastmod: 2026-01-31
src_hash: f06d99903c66aa101232357da87a6a1b28e3827e596cd2de338b3fe4d560ea5c
aliases: ["MediaPipe Wrappers", "Keypoint Extraction Core"]
---

# mediapipe_utils.py

#source #core #mediapipe #vision

**File Path**: `src/core/mediapipe_utils.py`

**Purpose**: Wrappers for MediaPipe solutions to extract Pose, Face, and Hand landmarks.

## Overview

Handles the initialization and concurrent execution of MediaPipe's `PoseLandmarker`, `FaceLandmarker`, and `HandLandmarker`. It normalizes the extraction process across different input types (Image vs Video).

## Setup Logic

```mermaid
graph TD
    A[Init] --> B{Inference Mode?}
    B -->|True| C[RunningMode.VIDEO]
    B -->|False| D[RunningMode.IMAGE]
    C --> E[Load Task Files]
    D --> E
    E --> F[Create Landmarkers]
```

### Key Features
- **Concurrent Extraction**: Uses `ThreadPoolExecutor` to run landmarkers in parallel.
- **Normalization**: Indices mapping for specific body parts.
- **Async Creation**: Supports asynchronous initialization for non-blocking startup.

## Constants

### Keypoint Counts
- `POSE_NUM`: 6 (Shoulders, Elbows, Wrists)
- `FACE_NUM`: 468 (Full FaceMesh) + Iris
- `HAND_NUM`: 21 (Standard Hand Model)

### Slicing
`KP2SLICE` dictionary maps body parts to indices in the flattened feature vector:
- `pose`: `[0:POSE_NUM]`
- `face`: `[POSE_NUM:POSE_NUM+FACE_NUM]`
- `rh`: `[...:...+HAND_NUM]`
- `lh`: `[...:...+HAND_NUM]`

## Classes

### `LandmarkerProcessor`

**Purpose**: Singleton-like manager for MediaPipe instances.

#### `__init__`
Initializes logger and definition references.

#### `create(landmarkers, inference_mode)`
**Parameters**:
- `landmarkers`: List of strings (e.g., `["pose", "face"]`).
- `inference_mode`: `True` for Video (stateful), `False` for Image (stateless).
**Returns**: Initialized instance.

#### `extract_frame_keypoints(frame_rgb, timestamp_ms=-1, adjusted=False)`
**Core Logic**:
1. Defines nested functions (`get_pose`, `get_face`, `get_hands`) to capture local state.
2. Submits tasks to `ThreadPoolExecutor(max_workers=3)`.
3. Waits for all results.
4. Aggregates results into a single `(N, 4)` numpy array `[x, y, z, visibility]`.
5. **Adjusted Mode**: If `True`, normalizes points relative to a reference (e.g., Nose) and scales by a body-part metric (e.g., Shoulder width).

## Related Documentation

**Depends On**:
- [[constants_py|constants.py]] - `LANDMARKERS_DIR`.

**Used By**:
- [[../data/prepare_npz_kps_py|prepare_npz_kps.py]] - Data ingestion.
- [[../api/live_processing_py|live_processing.py]] - Real-time inference.
