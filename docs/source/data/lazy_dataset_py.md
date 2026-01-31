---
title: lazy_dataset.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Lazy Loading Dataset", "Individual NPZ File Loader"]
---

# lazy_dataset.py

#source #data #pytorch #dataloader

**File Path**: `src/data/lazy_dataset.py`

**Purpose**: A PyTorch Dataset implementation that loads individual `.npz` files on demand (Lazy Loading).

## Overview

Unlike the Memory-Mapped dataset (which loads a massive monolithic file), this class keeps the data as thousands of individual small files. This is efficient for memory but incurs higher I/O overhead (many `open()` calls).

## Class `LazyKArSLDataset`

**Inherits**: `torch.utils.data.Dataset`

### `__init__`
**Parameters**: `split`, `signers`, `signs`, `transforms`.
**Logic**:
1. Initializes `TSNSampler` and `DataAugmentor`.
2. Iterates over all requested `signs` and `signers`.
3. Checks for existence of `.npz` files in `NPZ_KPS_DIR`.
4. Builds a list of metadata tuples: `self.samples = [(signer, vid_id, label), ...]`.

### `_load_file(path)`
**Decorator**: `@lru_cache(maxsize=1024)`
**Purpose**: Caches recently accessed file contents to reduce disk I/O for frequently accessed samples (though in efficient training, re-access is rare per epoch).

### `__getitem__(index)`
**Logic**:
1. Retrieves metadata `(signer, vid, label)`.
2. Constructs file path.
3. Loads raw keypoints via `_load_file`.
4. **Sampling**: Applies `TSNSampler` to get fixed-length `SEQ_LEN`.
5. **Transform**: Applies spatial augmentation.
6. **Return**: `(FloatTensor, LongTensor)`.

## Comparison

| Feature          | Lazy Dataset            | MMap Dataset       |
| :--------------- | :---------------------- | :----------------- |
| **Startup Time** | Slow (File Scanning)    | Fast (Offset Calc) |
| **Memory Usage** | Low                     | Low (Virtual Mem)  |
| **IO Pattern**   | Random Small Reads      | Random Seek/Read   |
| **Flexibility**  | High (Add/Remove files) | Low (Rebuild MMap) |

## Related Documentation

**Depends On**:
- [[../../source/data/data_preparation_py|data_preparation.py]] - `TSNSampler`, `DataAugmentor`
- [[../../source/core/constants_py|constants.py]] - `NPZ_KPS_DIR`

**Used By**:
- [[../../source/data/dataloader_py|dataloader.py]]
