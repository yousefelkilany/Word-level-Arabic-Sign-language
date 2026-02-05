---
title: generate_mediapipe_simplified_face_kps.py
date: 2026-02-05
lastmod: 2026-02-05
src_hash: 978bc1c816e851e22a0760c09bb8903909103599c0e5a7d47eb1a2606d202bf8
aliases: ["Face Contour Simplification", "Visvalingam-Whyatt Utility"]
---

# generate_mediapipe_simplified_face_kps.py

#source #scripts #mediapipe #optimization

**File Path**: `scripts/generate_mediapipe_simplified_face_kps.py`

**Purpose**: Reduces the number of landmarks in facial contours (lips, eyes, face oval) to optimize frontend rendering and data processing.

## Overview

The script uses the **Visvalingam-Whyatt algorithm** to simplify complex MediaPipe contours. It ensures that the simplified "shape" retains at least 90% of the area of the original high-resolution contour. It also enforces symmetry using the pre-generated symmetry map.

## Key Logic

### Visvalingam-Whyatt Implementation
The `run_visvalingam` function iteratively removes points that form triangles with the smallest area (least significance to the overall shape) until a target point count or area threshold is reached.

### `get_simplified_contours(...)`
- Simplifies the `face_oval`, `inner_lips`, and `outer_lips`.
- Simplifies left-side features (`left_eye`, `left_eyebrow`) and then mirrors them to the right side to ensure perfect visual symmetry, regardless of the reference image's slight asymmetries.

### `validate_simplified_contours(...)`
Uses the `shapely` library to calculate the intersection area between the original and simplified polygons.

## Execution

Saves a JSON file (`SIMPLIFIED_FACE_CONNECTIONS_PATH`) containing:
- `face_contours`: The simplified landmark indices.
- `face_paths`: The ordered paths for drawing.

## Related Documentation

**Depends On**:
- [[generate_mediapipe_face_symmetry_map_py|generate_mediapipe_face_symmetry_map.py]]
- [[../source/data/mediapipe_contours_py|mediapipe_contours.py]]
