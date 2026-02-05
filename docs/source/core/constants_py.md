---
title: constants.py
date: 2026-01-28
lastmod: 2026-02-05
src_hash: 40468146b1c7439db4a931b9798a95e3a9d347abc315d61fc28fcef10139a88d
aliases: ["System-wide Constants", "Path Configuration"]
---

# constants.py

#source #core #configuration #constants

**File Path**: `src/core/constants.py`

**Purpose**: System-wide constants, configuration values, and path definitions.

## Overview

Central configuration module that defines:

- Device configuration (CPU/GPU)
- Comprehensive directory structure (local vs Kaggle)
- Dataset parameters and split types
- Advanced model architecture logic (`ModelSize`, `HeadSize`)

## Environment & Device

- `DEVICE`: Execution device for PyTorch/ONNX. Automatically selects `"cuda"` if available and `USE_CPU` is "0".
- `LOCAL_DEV`: (int) Toggles between Kaggle (0) and Local (1) environments.

## Directory Structure

### Base Directories

- `PROJECT_ROOT_DIR`: Root folder containing the `src/` directory.
- `STATIC_ASSETS_DIR`: `{PROJECT_ROOT_DIR}/static`.
- `MODELS_DIR`: `{PROJECT_ROOT_DIR}/models`.
- `LOGS_DIR`: `{PROJECT_ROOT_DIR}/logs`.
- `LANDMARKERS_DIR`: Folder containing MediaPipe `.task` files.
- `PROJECT_DATA_DIR`: `{PROJECT_ROOT_DIR}/data`.

### Dataset Root Paths

- `DATA_INPUT_DIR`: Input path (e.g., `/kaggle/input` or `./data`).
- `DATA_OUTPUT_DIR`: Output path (e.g., `/kaggle/working` or `./data`).
- `KARSL_DATA_DIR`: `{DATA_INPUT_DIR}/karsl-502`.
- `TRAIN_CHECKPOINTS_DIR`: `{DATA_OUTPUT_DIR}/checkpoints`.

### Specialized Data Paths (KAGGLE/LOCAL)

- `NPZ_KPS_DIR`: Extracted keypoints in NPZ format.
- `MMAP_PREPROCESSED_DIR`: Local/Input directory for preprocessed memory-mapped files.
- `MMAP_OUTPUT_PREPROCESSED_DIR`: Output directory for generated memory-mapped files.

## Data Files & Assets

- `LABELS_PATH`: Path to the `KARSL-502_Labels.xlsx` file.
- `LABELS_JSON_PATH`: Path to the `KARSL-502_Labels.json` file.
- `FACE_SYMMETRY_MAP_PATH`: Path to the `.npy` symmetry map used for face mesh processing.
- `SIMPLIFIED_FACE_CONNECTIONS_PATH`: Path to the `simplified_face_connections.json` file.

---

## Model Architecture Definitions

### `HeadSize` (StrEnum)

Defines supported attention head sizes and their corresponding feature dimensions.

- `tiny` ('t'): 16 dimensions
- `small` ('s'): 32 dimensions
- `medium` ('m'): 64 dimensions
- `large` ('l'): 128 dimensions

### `ModelSize` (Class)

Manages the complexity of the transformer model (heads and layers) based on the chosen `HeadSize`.

| HeadSize | Num Heads | Num Layers |
| :------- | :-------- | :--------- |
| Tiny     | 4         | 2          |
| Small    | 4         | 4          |
| Medium   | 4         | 6          |
| Large    | 6         | 8          |

---

## Configuration & Performance

### Feature Configuration

- `SEQ_LEN`: 50 (Temporal window size).
- `FEAT_NUM`: 184 (Total keypoints: POSE + FACE + 2*HANDS).
- `FEAT_DIM`: 4 (x, y, z, visibility).

### Timing & Performance

- `MAX_WORKERS`: 4 (Thread pool size for extraction).
- `MS_30FPS`: ~33.33ms (Float interval for 30 FPS).
- `MS_30FPS_INT`: 33ms (Integer interval for 30 FPS).

---

## Enumerations & Types

### `SplitType`

- `train`, `val`, `test`

### `DatasetType`

- `lazy`, `mmap`

### `KarslDatasetType`

- Union of `LazyKArSLDataset` and `MmapKArSLDataset`.

## Related Documentation

**Used By**:

- [[../modelling/model_py|model.py]] - Model architecture constants.
- [[../data/dataloader_py|dataloader.py]] - Dataset paths and split types.
- [[../modelling/train_py|train.py]] - Checkpoint paths and device config.
