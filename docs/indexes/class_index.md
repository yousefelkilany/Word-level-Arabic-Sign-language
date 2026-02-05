---
title: Class Index
date: 2026-01-28
lastmod: 2026-02-05
aliases: ["Classes Reference", "OOP Index"]
---

# Class Index

#index #classes

A comprehensive lookup of project classes.

## Core
- **`LandmarkerProcessor`**: [[../source/core/mediapipe_utils_py#Classes|mediapipe_utils.py]] - Manages MediaPipe instances and threading.
- **`SplitType`**: [[../source/core/constants_py#Enumerations|constants.py]] - Enum (`train`, `val`, `test`).
- **`DatasetType`**: [[../source/core/constants_py#Enumerations|constants.py]] - Enum (`lazy`, `mmap`).

## Data
- **`MmapKArSLDataset`**: [[../source/data/mmap_dataset_py#Class MmapKArSLDataset|mmap_dataset.py]] - High-performance memory-mapped dataset.
- **`LazyKArSLDataset`**: [[../source/data/lazy_dataset_py#Class LazyKArSLDataset|lazy_dataset.py]] - Low-memory file-based dataset.
- **`TSNSampler`**: [[../source/data/data_preparation_py#Classes|data_preparation.py]] - Temporal sampling logic.
- **`DataAugmentor`**: [[../source/data/data_preparation_py#Classes|data_preparation.py]] - Spatial augmentation logic.

## Modelling
- **`STTransformer`**: [[../source/modelling/model_py#Class STTransformer|model.py]] - The main Spatial-Temporal Transformer architecture.
- **`GroupTokenEmbedding`**: [[../source/modelling/model_py#Class GroupTokenEmbedding|model.py]] - Body-part specific projection and tokenization layer.
- **`AttentionPooling`**: [[../source/modelling/model_py#Class AttentionPooling|model.py]] - Weighted temporal aggregation layer.
- **`STTransformerBlock`**: [[../source/modelling/model_py#Class STTransformerBlock|model.py]] - Dual-stream spatial-temporal attention block.
- **`PositionalEncoding`**: [[../source/modelling/model_py#Class PositionalEncoding|model.py]] - Sinusoidal temporal positional encoding layer.
