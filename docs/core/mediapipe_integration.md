---
title: MediaPipe Integration
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Feature Extraction Engine", "Landmark Detection"]
---

# MediaPipe Integration

#core #mediapipe #computer-vision

The core feature extraction engine of the project relies on **Google MediaPipe**, a cross-platform framework for building multimodal applied machine learning pipelines. We use MediaPipe to extract high-fidelity landmarks from the user's face, hands, and pose, which serve as the input features for our recognition model.

## Landmark Components

We utilize three distinct MediaPipe solutions, integrated into a unified `LandmarkerProcessor` class:

### 1. Pose Landmarks
- **Model**: `PoseLandmarker`
- **Output**: 33 3D landmarks (x, y, z, visibility).
- **Usage**: Critical for tracking arm movements and overall body posture. We focus specifically on the upper body (shoulders, elbows, wrists).

### 2. Hand Landmarks
- **Model**: `HandLandmark`
- **Output**: 21 3D landmarks per hand.
- **Usage**: The most critical component for sign language. Captures detailed finger configurations and palm orientation.
- **Refinement**: We distinguish between Left and Right hands and handle cases where hands cross or occlude each other.

### 3. Face Landmarks
- **Model**: `FaceLandmarker`
- **Output**: 478 3D landmarks (Face Mesh).
- **Usage**: Captures facial expressions and mouth movements (mouthing), which are grammatical markers in Arabic Sign Language. We select a specific subset of landmarks (loops around eyes, lips, and face contour) to reduce dimensionality.

## Integration Strategy

### efficient Asynchronous Execution
Since running three separate deep learning models per frame is computationally expensive, we execute them in parallel using a `ThreadPoolExecutor`.

```python
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(get_pose)
    executor.submit(get_face)
    executor.submit(get_hands)
```

This ensures we maximize CPU utilization and minimize latency.

### Feature Normalization
Raw landmarks from MediaPipe are in screen coordinates (pixels) or normalized [0, 1] coordinates. To make the model robust to camera distance and position:

1.  **Centering**: We subtract a reference point (e.g., the nose tip) from all other points.
2.  **Scaling**: We divide by a reference distance (e.g., shoulder width) to normalize for the user's size and distance from the camera.

## Configuration

MediaPipe models are loaded from the `assets/` directory. The specific model complexity (Lite, Full, Heavy) can be configured, though we default to the Full models for accuracy.

## Related Documentation

- [[../source/core/mediapipe_utils_py|mediapipe_utils.py Source Code]]
- [[../source/core/constants_py|constants.py Source Code]]
