---
title: prepare_npz_kps.py
date: 2026-01-28
lastmod: 2026-01-29
aliases: ["Keypoint Extraction ETL", "Video to Skeleton Script"]
---

# prepare_npz_kps.py

#source #data #script #mediapipe

**File Path**: `src/data/prepare_npz_kps.py`

**Purpose**: The primary ETL (Extract, Transform, Load) script that converts raw videos into skeleton keypoints using MediaPipe.

## Overview

Uses Python's `ProcessPoolExecutor` to parallelize processing across CPU cores. Each worker initializes its own MediaPipe instances (since they are not pickle-able).

## Key Functions

### `init_worker(landmarkers)`

**Role**: Process initializer.
**Action**: Creates a global `_worker_processor` (instance of `LandmarkerProcessor`) inside the worker process memory space.

### `process_video(video_dir, adjusted)`

**Role**: Single video processor.
**Logic**:

1. Iterates frames sorted by name.
2. Converts BGR -> RGB.
3. Calls `_worker_processor.extract_frame_keypoints`.
4. Returns `(Seq_Len, 184, 4)` array.

### `process_sign_wrapper(info)`

**Role**: Unit of work for the executor.
**Scope**: Processes ALL videos for a specific **Sign ID**.

- Iterates (Signer × Split).
- Saves results to: `karsl-kps/{signer}-{split}/{sign_id}.npz`.
- **Format**: Dictionary `{video_filename: keypoint_array}`.

## Parallelization Strategy

The script parallelizes at the **Sign** level.

- If you have 500 signs, it creates 500 tasks.
- A 16-core CPU will process 16 signs simultaneously.

## Usage

```bash
python src/data/prepare_npz_kps.py \
    --splits train test \
    --signers 01 02 03 \
    --selected_signs_from 1 --selected_signs_to 10 \
    --adjusted
```

> [!WARNING]
> This script is computationally intensive. Ensure `MEDIAPIPE_DISABLE_GPU=1` if running many parallel workers to avoid GPU context limits, or allow GPU usage if `num_workers` is low.

## Related Documentation

**Depends On**:

- [[../core/mediapipe_utils_py|mediapipe_utils.py]] - Extraction logic
- [[../core/constants_py|constants.py]] - Paths

**Produces**:

- Raw `.npz` files consumed by [[lazy_dataset_py|LazyDataset]] and [[mmap_dataset_preprocessing_py|Preprocessing]].
