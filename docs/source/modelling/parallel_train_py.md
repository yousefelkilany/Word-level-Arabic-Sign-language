---
title: parallel_train.py
date: 2026-01-28
lastmod: 2026-02-05
src_hash: e6e05cc73da977e632f01815fe4026cdad5a804d22ed4856e4433b2faa5fbdbf
aliases: ["Distributed Training Launcher", "Multi-GPU Training Script"]
---

# parallel_train.py

#source #modelling #distributed #gpu

**File Path**: `src/modelling/parallel_train.py`

**Purpose**: High-performance multi-GPU training launcher using PyTorch Distributed Data Parallel (DDP).

## Overview

This script orchestrates distributed training across all available GPUs on a single node. It manages process spawning, environment setup, and data division.

## Logic Workflow

1. **Initialization**: Detects number of available GPUs (`world_size`).
2. **Process Spawning**: Uses `torch.multiprocessing.spawn` to launch `training_wrapper` across $N$ processes (one per GPU).
3. **Environment Setup**:
   - Initializes the process group using the `nccl` backend.
   - Sets the local `rank` for each process.
   - Configures `MASTER_ADDR` and `MASTER_PORT`.
4. **Data Distribution**:
   - Uses `DistributedSampler` to ensure each GPU sees a unique subset of the data.
   - Wraps the dataset in `DataLoader` with `pin_memory=True`.
5. **Model Parallelism**:
   - Wraps the model in `DistributedDataParallel (DDP)`.
   - Converts Batch Norm layers to `SyncBatchNorm` for consistent statistics across GPUs.
6. **Dynamic Scaling**:
   - Scales the learning rate by the square root of the `world_size`: `lr = 1e-3 * sqrt(world_size)`.
7. **Training**: Invokes the shared `train()` function from `train.py`.

## Key Functions

### `setup(rank, world_size)`

Initializes the distributed environment and the backend communications.

### `cleanup()`

Destroys the process group after training completes.

### `run_training(rank, world_size, ...)`

The core training logic executed on each GPU. Handles device setting, model wrapping, and result visualization (on `rank 0`).

## Usage

```bash
# Automatically detects and uses all available GPUs
python src/modelling/parallel_train.py --selected_signs_to 502 --num_epochs 20
```

## Related Documentation

- [[train_py|train.py]] - Shared training loop.
- [[../data/dataloader_py|dataloader.py]] - Distributed sampling logic.
