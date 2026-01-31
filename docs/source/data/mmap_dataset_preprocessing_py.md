---
title: mmap_dataset_preprocessing.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Mmap Dataset Creation", "Keypoint Binary Compilation"]
---

# mmap_dataset_preprocessing.py

#source #data #script #preprocessing

**File Path**: `src/data/mmap_dataset_preprocessing.py`

**Purpose**: Compiles thousands of individual `.npz` keypoint files into a monolithic memory-mapped binary file for efficient training.

## Process Overview

1. **Scan**: Iterates over all (Signer, Word) pairs.
2. **Load**: Reads every `.npz` file into RAM (accumulating a large list).
3. **Concatenate**: Merges into a single `(Total_Frames, 184, 4)` float32 array.
4. **Save**:
   - `X.mmap`: The raw binary data.
   - `y.npz`: Corresponding labels per sample.
   - `X_shape.npy`: Dimensions metadata.
   - `X_map_samples_lens.npy`: Lookup table for sample lengths.

## Functions

### `load_raw_kps(...)`
Traverses the `NPZ_KPS_DIR` and aggregates data.
- **Handling Missing Data**: Prints error but continues if a file is missing.

### `mmap_process_and_save_split(...)`
Orchestrates the conversion for a specific split (train/test).
- **Memory Management**: Uses `gc.collect()` and `del` to free RAM after processing each split to avoid OOM kills.

## CLI Usage

```bash
python src/data/mmap_dataset_preprocessing.py \
    --splits train test \
    --signers 01 02 03 \
    --selected_signs_from 1 --selected_signs_to 502
```

## Output Structure

```
data/
└── word-level-arabic-sign-language-preprcsd-keypoints/
    ├── train_X.mmap      (Several GBs)
    ├── train_y.npz
    ├── train_X_shape.npy
    └── train_X_map_samples_lens.npy
```

## Related Documentation

**Depends On**:
- [[../../source/core/constants_py|constants.py]] - Directory paths

**Used By**:
- Used offline before training.
- Generates data for [[../../source/data/mmap_dataset_py|mmap_dataset.py]]
