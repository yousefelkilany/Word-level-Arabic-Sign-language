---
title: source/api/live-processing.py
date: 2026-01-28
lastmod: 2026-01-28
---

# source/api/live-processing.py

#source-code #api #frame-processing #async

**File Path**: `src/api/live_processing.py`

**Purpose**: Frame buffer management and asynchronous keypoint extraction for real-time processing.

## Overview

Provides infrastructure for buffering incoming video frames and extracting keypoints asynchronously using thread pools.

## Classes

### `FrameBuffer`

#class #buffer #circular-buffer

**Purpose**: Circular buffer for managing video frames during inference.

**Attributes**:
- `_frames` (dict[int, np.ndarray]): Frame storage indexed by frame number
- `_max_size` (int): Maximum buffer capacity
- `_latest_idx` (int): Index of most recent frame
- `logger`: Logger instance

**Methods**:

#### `__init__(max_size)`
Initializes buffer with maximum size.

#### `add_frame(frame)`
Adds frame to buffer, removes oldest if full.

**Called By**: [[source/api/websocket_py#ws_live_signs|ws_live_signs()]]

#### `get_frame(idx) -> Optional[np.ndarray]`
Retrieves frame by index.

**Called By**: [[source/api/websocket_py#ws_live_signs|ws_live_signs()]]

#### `latest_idx` (property)
Returns index of most recent frame.

#### `oldest_idx` (property)
Returns index of oldest frame in buffer.

#### `clear()`
Clears all frames and resets index.

**Called By**: [[source/api/websocket_py#ws_live_signs|ws_live_signs()]]

## Functions

### `producer_handler(websocket, buffer: FrameBuffer)`

#function #async #producer

**Purpose**: Asynchronously receives and decodes frames from WebSocket.

**Parameters**:
- `websocket`: WebSocket connection
- `buffer` (FrameBuffer): Frame buffer to populate

**Implementation**:
- Receives binary frame data via WebSocket
- Decodes JPEG frames using cv2.imdecode
- Adds decoded frames to buffer
- Runs until WebSocket closes or error occurs

**Called By**: [[source/api/websocket_py#ws_live_signs|ws_live_signs()]]

**Calls**:
- `websocket.receive_bytes()` - Receives frame data
- `cv2.imdecode()` - Decodes JPEG to numpy array
- [[#FrameBuffer.add_frame|buffer.add_frame()]] - Stores frame

### `get_frame_kps(mp_processor, frame, timestamp_ms=-1)`

#function #async #keypoint-extraction

**Purpose**: Asynchronously extracts keypoints from frame using MediaPipe.

**Parameters**:
- `mp_processor` ([[source/core/mediapipe_utils_py#LandmarkerProcessor|LandmarkerProcessor]]): MediaPipe processor
- `frame` (np.ndarray): Video frame
- `timestamp_ms` (int): Frame timestamp

**Returns**: Extracted keypoints array

**Implementation**:
Uses thread pool executor to run MediaPipe processing off main thread.

**Called By**: [[source/api/websocket_py#ws_live_signs|ws_live_signs()]]

**Calls**:
- [[../../source/core/mediapipe_utils_py#LandmarkerProcessor.extract_frame_keypoints|mp_processor.extract_frame_keypoints()]]

## Thread Pool

```python
keypoints_detection_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
```

**Configuration**: [[source/core/constants_py#MAX_WORKERS|MAX_WORKERS]] = 4

**Used By**: [[#get_frame_kps|get_frame_kps()]]

## Related Documentation

- [[../../source/api/websocket_py|websocket.py]] - Main consumer
- [[../../source/core/mediapipe_utils_py|mediapipe_utils.py]] - Keypoint extraction
- [[../../api/live_processing_pipeline|Live Processing Pipeline]]

---

**File Location**: `src/api/live_processing.py`
