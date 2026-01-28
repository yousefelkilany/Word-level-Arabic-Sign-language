# Function Index

#index #functions

A comprehensive lookup of all documented functions.

## API
- **`run`**: [[../source/api/run_py#Execution|run.py]] - Entry point; starts Uvicorn server.
- **`socket.onmessage`**: [[../source/frontend/live_signs_js#Functions|live_signs.js]] - Frontend handler for inference results.
- **`setupWebcam`**: [[../source/frontend/live_signs_js#Functions|live_signs.js]] - Initializes camera stream.

## Core
- **`init_words`**: [[../source/core/utils_py#Functions|utils.py]] - Loads Arabic/English label lists.
- **`get_default_logger`**: [[../source/core/utils_py#Functions|utils.py]] - Configures application logging.
- **`extract_frame_keypoints`**: [[../source/core/mediapipe_utils_py#Classes|mediapipe_utils.py]] - Main extraction logic for a single frame.
- **`draw_all_kps_on_image`**: [[../source/core/draw_kps_py#Functions|draw_kps.py]] - Visualizes full skeleton on a frame.

## Data Preparation
- **`TSNSampler`**: [[../source/data/data_preparation_py#Classes|data_preparation.py]] - Temporal sampling functor.
- **`DataAugmentor`**: [[../source/data/data_preparation_py#Classes|data_preparation.py]] - Spatial augmentation functor.
- **`load_raw_kps`**: [[../source/data/mmap_dataset_preprocessing_py#Functions|mmap_dataset_preprocessing.py]] - Aggregates raw NPZ files.
- **`mmap_process_and_save_split`**: [[../source/data/mmap_dataset_preprocessing_py#Functions|mmap_dataset_preprocessing.py]] - Writes binary mmap files.
- **`process_video`**: [[../source/data/prepare_npz_kps_py#Key Functions|prepare_npz_kps.py]] - Extracts keypoints from a video folder.

## Modelling
- **`get_model_instance`**: [[../source/modelling/model_py#Functions|model.py]] - Factory for creating model objects.
- **`load_onnx_model`**: [[../source/modelling/model_py#Functions|model.py]] - Initializes ONNX Runtime session.
- **`onnx_inference`**: [[../source/modelling/model_py#Functions|model.py]] - Runs inference helper.
- **`train`**: [[../source/modelling/train_py#Functions|train.py]] - Main training loop (Epoch/Batch iteration).
- **`export_model`**: [[../source/modelling/export_py#Functions|export.py]] - Converts PyTorch checkpoint to ONNX.
