# Class Index

#index #classes

A comprehensive lookup of project classes.

## Core
- **`LandmarkerProcessor`**: [[source/core/mediapipe_utils_py#Classes|mediapipe_utils.py]] - Manages MediaPipe instances and threading.
- **`SplitType`**: [[source/core/constants_py#Enumerations|constants.py]] - Enum (`train`, `val`, `test`).
- **`DatasetType`**: [[source/core/constants_py#Enumerations|constants.py]] - Enum (`lazy`, `mmap`).

## Data
- **`MmapKArSLDataset`**: [[source/data/mmap_dataset_py#Class MmapKArSLDataset|mmap_dataset.py]] - High-performance memory-mapped dataset.
- **`LazyKArSLDataset`**: [[source/data/lazy_dataset_py#Class LazyKArSLDataset|lazy_dataset.py]] - Low-memory file-based dataset.
- **`TSNSampler`**: [[source/data/data_preparation_py#Classes|data_preparation.py]] - Temporal sampling logic.
- **`DataAugmentor`**: [[source/data/data_preparation_py#Classes|data_preparation.py]] - Spatial augmentation logic.

## Modelling
- **`AttentionBiLSTM`**: [[source/modelling/model_py#Classes|model.py]] - The main neural network architecture.
- **`SpatialGroupEmbedding`**: [[source/modelling/model_py#Classes|model.py]] - Body-part specific projection layer.
- **`AttentionPooling`**: [[source/modelling/model_py#Classes|model.py]] - Weighted temporal aggregation layer.
- **`ResidualBiLSTMBlock`**: [[source/modelling/model_py#Classes|model.py]] - Building block for the sequence encoder.
