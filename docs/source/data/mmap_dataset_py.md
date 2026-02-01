---
title: mmap_dataset.py
date: 2026-01-28
lastmod: 2026-01-29
aliases: ["Memory-mapped Dataset", "High-performance PyTorch Dataset"]
---

# mmap_dataset.py

#source #data #pytorch #performance

**File Path**: `src/data/mmap_dataset.py`

**Purpose**: High-performance PyTorch Dataset backed by `numpy.memmap`, allowing training on datasets larger than RAM.

## Overview

Instead of loading thousands of small files (Lazy) or the whole dataset into RAM, this class maps a single giant binary file (`train_X.mmap`) into virtual memory. The OS handles paging pages in/out of RAM as needed.

## Class `MmapKArSLDataset`

**Inherits**: `torch.utils.data.Dataset`

### `__init__`

**Logic**:

1. Loads metadata:
   - `X_shape.npy`: Total dimensions of the giant array.
   - `y.npz`: Labels array.
   - `X_map_samples_lens.npy`: Length of each sample within the giant array.
2. **Memmap**: Creates a read-only view (`mode="r"`) of the data.

   ```python
   self.X = np.memmap(data_path, dtype="float32", mode="r", shape=X_shape)
   ```

3. **Offset Calculation**: Pre-calculates the start index (`X_offsets`) for every sample to allow O(1) random access.

### `__getitem__(index)`

**Logic**:

1. usage `index` to find `start_offset` and `length`.
2. Slices the memmap (Zero-copy operation): `raw = self.X[start:start+len]`.
3. Applies `TSNSampler` to get fixed size.
4. Applies `DataAugmentor`.

## Performance Note
>
> [!TIP]
> This is the recommended dataset for training on high-performance clusters or machines with fast SSDs (NVMe). It significantly increases GPU utilization by removing CPU/IO bottlenecks.

## Related Documentation

**Depends On**:

- [[mmap_dataset_preprocessing_py|mmap_dataset_preprocessing.py]] - Creates the mmap files
- [[../core/constants_py|constants.py]] - `MMAP_PREPROCESSED_DIR`

**Used By**:

- [[dataloader_py|dataloader.py]]
