---
title: Memory-Mapped Datasets
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Memory Mapping Guide", "Optimized Data Loading"]
---

# Memory-Mapped Datasets

#data #optimization #performance

Efficiently training on large datasets requires fast data loading. We use **NumPy Memory Mapping (`np.memmap`)** to handle our dataset, which allows us to access small segments of a large file on disk without reading the entire file into memory.

## Implementation Strategy

### 1. Consolidation
Instead of opening thousands of individual `.npz` files during training (which causes high I/O overhead), we consolidate all preprocessed samples into a single large binary file per split (train/val/test).

- **Data File**: `{split}_X.mmap` (Contains concatenated feature vectors)
- **Label File**: `{split}_y.npz` (Contains labels)
- **Index File**: `{split}_X_map_samples_lens.npy` (Maps sample indices to their length and location)

### 2. Random Access
The `MmapKArSLDataset` class uses the index file to locate the specific byte range for a requested sample and reads only that segment.

```python
chunk_idx = self.X_offsets[index]
sample = self.X[chunk_idx : chunk_idx + length]
```

### 3. Impact
- **RAM Usage**: drastically reduced. We only hold the current batch in memory.
- **IOPS**: Reduced OS file handle overhead.
- **Speed**: significantly faster epoch times compared to loading individual files.

## Related Documentation

- [[../source/data/mmap_dataset_py|mmap_dataset.py Source Code]]
- [[../source/data/mmap_dataset_preprocessing_py|mmap_dataset_preprocessing.py Source Code]]
