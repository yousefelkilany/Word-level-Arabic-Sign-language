---
title: Training Process
date: 2026-01-28
lastmod: 2026-01-28
---

# Training Process

#model #training #pytorch

The training pipeline is robust, supporting distributed training, mixed precision, and dynamic learning rate scheduling.

## Training Concepts

### 1. Loop Structure

Standard PyTorch training loop:

1.  **Forward Pass**: Compute model predictions.
2.  **Loss Calculation**: CrossEntropyLoss.
3.  **Backward Pass**: Backpropagate gradients.
4.  **Optimization Step**: Update weights (Adam optimizer).

### 2. Mixed Precision (AMP)

We use `torch.cuda.amp` (Automatic Mixed Precision) to speed up training and reduce memory usage.

- **Scaler**: `GradScaler` is used to prevent gradient underflow/overflow.
- **BFloat16**: Used for matrix multiplications on supported hardware.

### 3. Distributed Data Parallel (DDP)

The `train.py` script is designed to run on multiple GPUs.

- **DistributedSampler**: Ensures each GPU sees a unique subset of the data.
- **Sync**: Gradients are synchronized across GPUs during the backward pass.
- **Metric Reduction**: Validation metrics are aggregated from all ranks.

### 4. Hyperparameters

Key hyperparameters (defaults):

- **Hidden Size**: 384
- **Layers**: 4
- **Learning Rate**: 1e-3
- **Scheduler**: ReduceLROnPlateau (factor 0.2, patience 3)
- **Dropout**: 0.3 - 0.5

## Checkpointing

The model is saved only when validation loss improves.

- **Format**: `.pth` file containing model state, optimizer state, and scheduler state.
- **Naming**: `checkpoint_{timestamp}-signs_{num_signs}/{epoch}.pth`

## Related Documentation

- [[../source/modelling/train_py|train.py Source Code]]
- [[../source/modelling/parallel_train_py|parallel_train.py Source Code]]
