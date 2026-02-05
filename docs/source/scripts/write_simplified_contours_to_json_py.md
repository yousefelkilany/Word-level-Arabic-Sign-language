---
title: write-simplified-contours-to-json.py
date: 2026-02-05
lastmod: 2026-02-05
src_hash: b06cd36540d2e0e6735d2560053ffaba22c29ab0cf5071abd1cf888bf983ee97
aliases: ["JSON Export Utility", "Frontend Connection Map"]
---

# write-simplified-contours-to-json.py

#source #scripts #data #frontend

**File Path**: `scripts/write-simplified-contours-to-json.py`

**Purpose**: Exports a comprehensive map of landmark indices and connections to a JSON file for consumption by the frontend visualization engine.

## Overview

The script aggregates simplified contour data (face, hands, pose) and saves it to the static assets directory. This allows the browser-based UI to draw optimized skeletons without needing heavy compute or the full MediaPipe dictionary.

## Exported Data Structure

The output file `simplified_kps_connections.json` includes:
- **`pose_kps`**: Indices for pose landmarks.
- **`face_kps`**: Indices for the reduced face mesh.
- **`hand_kps`**: Indices for hands.
- **`pose_connections`**: Lines to draw for the pose.
- **`face_contours` / `face_paths`**: Simplified facial features.
- **`hand_connections`**: Hand skeleton edges.
- **`mp_idx_to_kps_idx`**: A mapping from raw MediaPipe indices to the reduced indices used in the app's internal processing.

## Related Documentation

**Used By**:
- [[../source/frontend/live_signs_js|live-signs.js]] - Uses this JSON to draw landmarks on the video canvas.
