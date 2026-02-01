---
title: dataloader.py
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["PyTorch DataLoader Manager", "Splits and Transforms Config"]
---

# dataloader.py

#source #data #pytorch

Manages the creation of PyTorch DataLoaders for training, validation, and testing.

## Functions

### `prepare_dataloader(...)`
Creates a single `DataLoader` for a specific split.
- **Args**: `dataset_type`, `split`, `signers`, `signs`, `batch_size`, `train_transforms`.
- **Logic**:
    - Selects `LazyKArSLDataset` or `MmapKArSLDataset`.
    - For `train` split, it further performs an 80/20 random split for training/validation if not explicitly handling separate files (though typically `train` and `test` are distinct).
    - Returns `(train_dl, val_dl)` or `test_dl`.

### `prepare_dataloaders(...)`
Wrapper function to generate all three dataloaders (Train, Val, Test) in one go.

## Related Code

- [[mmap_dataset_py|mmap_dataset.py]]
- [[data_preparation_py|data_preparation.py]]
