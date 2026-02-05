---
title: mediapipe_contours.py
date: 2026-02-05
lastmod: 2026-02-05
src_hash: 54ab0af8e94d57661cc12657628c434e155ce13a2ef379ce3a57db22e698e03c
aliases: ["Face Mapping", "MediaPipe Connections"]
---

# mediapipe_contours.py

#source #data #mediapipe #visualization

**File Path**: `src/data/mediapipe_contours.py`

**Purpose**: Defines connection maps (contours) for MediaPipe Face Mesh and Pose landmarker points.

## Overview

This module provides structured tuples and dictionaries that map individual landmarker indices into semantic "contours" (e.g., the oval of the face, the inner lips). These are used primarily for visualization and region-of-interest extraction.

## Helper Functions

### `join_upper_lower_connections(upper, lower)`
Combines upper and lower boundary connections into a single continuous loop.

### `get_contour_from_path(path)`
Converts a sequence of points (path) into a list of adjacent pairs (edges).

### `get_path_from_contour(contour)`
Converts a list of edges back into a unique sequence of point indices.

## Face Mesh Contours

The module defines several semantic regions for the `468` face landmarks:

| Constant                                  | Description                             |
| :---------------------------------------- | :-------------------------------------- |
| `FACEMESH_OUTER_LIPS`                     | Outer boundary of the lips.             |
| `FACEMESH_INNER_LIPS`                     | Inner boundary of the mouth.            |
| `FACEMESH_LEFT_EYE` / `RIGHT_EYE`         | Boundaries of the eyes.                 |
| `FACEMESH_LEFT_EYEBROW` / `RIGHT_EYEBROW` | Eyebrow arcs.                           |
| `FACEMESH_FACE_OVAL`                      | The silhouette/oval of the entire face. |

### `FACEMESH_CONTOURS`
A dictionary aggregating all face regions for easy iteration.

### `FACEMESH_CONTOUR_PATHS`
The same regions represented as ordered paths (lists of indices).

## Pose Contours

### `POSEMESH_OPEN`
Defines an "open" connection path across the upper body:
`Right Wrist` -> `Elbow` -> `Shoulder` -> `Left Shoulder` -> `Elbow` -> `Wrist`.

## Related Documentation

- [[../core/draw_kps_py|draw_kps.py]] - Uses these contours for rendering visualizations.
- [[../core/mediapipe_utils_py|mediapipe_utils.py]] - Defines the landmarker processors that produce these indices.
