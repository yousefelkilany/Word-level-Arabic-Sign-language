---
title: live_processing.py
date: 2026-01-28
lastmod: 2026-02-05
src_hash: 8eee887f6a864ca0a9aaef85fe99ece1772d984d59f862ed8d7da4d1075ab036
aliases: ["Real-time Processing", "Stream Ingestion", "Live Sign Detection"]
---

# live_processing.py

#source #api #real-time #async #inference

**File Path**: `src/api/live_processing.py`

**Purpose**: Orchestrates real-time video frame ingestion, motion detection, keypoint extraction, and sign language inference.

## Overview

This module implements a dual-handler architecture (Producer/Consumer) using `asyncio.Queue` to process video streams from WebSockets. It handles:
1.  **Frame Ingestion**: Decoding binary bytes from WebSockets into OpenCV images.
2.  **Motion Detection**: Skipping static frames to save compute.
3.  **Feature Extraction**: Concurrent MediaPipe landmarking.
4.  **Inference Orchestration**: Sliding window based classification using ONNX models.

## Handlers

### `producer_handler(client_id, websocket, buffer)`
**Purpose**: Receives raw bytes from the client and decodes them.
- **Protocol**: Receives `bytes`. First byte is `draw_mode` (toggle), remaining are JPEG-encoded frame.
- **Decoding**: Uses `cv2.imdecode` via `asyncio.to_thread`.
- **Queueing**: Drops oldest frames if the buffer is full to maintain low latency.

### `consumer_handler(client_id, websocket, buffer)`
**Purpose**: The main processing engine for each client connection.
- **Initialization**: Creates `LandmarkerProcessor` asynchronously.
- **Workflow**:
    1.  Polls frames from the buffer.
    2.  Performs motion detection via `MotionDetector`.
    3.  If motion detected:
        - Extracts keypoints via `get_frame_kps`.
        - Updates `client_buffer` (size: `MAX_SIGN_FRAMES`).
    4.  If buffer meets `MIN_SIGN_FRAMES`:
        - Runs `onnx_inference`.
        - Applies Softmax and checks `CONFIDENCE_THRESHOLD`.
        - Uses a `sign_history` deque and `HISTORY_THRESHOLD` to stabilize predictions.
    5.  Sends JSON response (landmarks, detected sign, confidence).

## Core Functions

### `get_frame_kps(mp_processor, frame, timestamp_ms=-1)`
**Purpose**: Wrapper for thread-pool based keypoint extraction.
- **Returns**: `tuple(adjusted_kps, raw_kps)`.

## Constants & Configuration

| Constant               | Value | Description                                  |
| :--------------------- | :---- | :------------------------------------------- |
| `NUM_IDLE_FRAMES`      | 15    | Frames before switching back to idle state.  |
| `MIN_SIGN_FRAMES`      | 15    | Minimum frames required for inference.       |
| `MAX_SIGN_FRAMES`      | 50    | Maximum temporal window (matches `SEQ_LEN`). |
| `CONFIDENCE_THRESHOLD` | 0.7   | Minimum probability for valid detection.     |
| `HISTORY_THRESHOLD`    | 2     | Required repetitions in history for output.  |

## Related Documentation

**Depends On**:
- [[../core/mediapipe_utils_py|mediapipe_utils.py]] - `LandmarkerProcessor`.
- [[cv2_utils_py|cv2_utils.py]] - `MotionDetector`.
- [[../modelling/model_py|model.py]] - `onnx_inference`.

**Used By**:
- [[websocket_py|websocket.py]] - Spawns the handlers.
