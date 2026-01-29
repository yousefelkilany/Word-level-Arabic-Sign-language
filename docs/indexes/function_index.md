---
title: Function Index
date: 2026-01-28
lastmod: 2026-01-28
---

# Function Index

#index #functions

A comprehensive lookup of all documented functions.

## API

- **`run`**: [[../source/api/run_py#Execution|run.py]] - Entry point; starts Uvicorn server.
- **`socket.onmessage`**: [[../source/frontend/live_signs_js#Functions|live_signs.js]] - Frontend handler for inference results.
- **`setupWebcam`**: [[../source/frontend/live_signs_js#Functions|live_signs.js]] - Initializes camera stream.

## Core

- **`init_signs`**: [[../source/core/utils_py#Functions|utils.py]] - Loads Arabic/English label lists.
- **`get_default_logger`**: [[../source/core/utils_py#Functions|utils.py]] - Configures application logging.
- **`extract_frame_keypoints`**: [[../source/core/mediapipe_utils_py#Classes|mediapipe_utils.py]] - Main extraction logic for a single frame.
- **`draw_all_kps_on_image`**: [[../source/core/draw_kps_py#Functions|draw_kps.py]] - Visualizes full skeleton on a frame.

## Data Preparation

- **`TSNSampler`**: [[../source/data/data_preparation_py#Classes|data_preparation.py]] - Temporal sampling functor.
- **`DataAugmentor`**: [[../source/data/data_preparation_py#Classes|data_preparation.py]] - Spatial augmentation functor.
- **`load_raw_kps`**: [[../source/data/mmap_dataset_preprocessing_py#Functions|mmap_dataset_preprocessing.py]] - Aggregates raw NPZ files.
- **`mmap_process_and_save_split`**: [[../source/data/mmap_dataset_preprocessing_py#Functions|mmap_dataset_preprocessing.py]] - Writes binary mmap files.
- **`process_video`**: [[../source/data/prepare_npz_kps_py#Key Functions|prepare_npz_kps.py]] - Extracts keypoints from a video folder.
- **`gen_symmetry_map`**: [[../source/data/generate_mediapipe_face_symmetry_map_py#Key Functions|generate_mediapipe_face_symmetry_map.py]] - Calculates face mesh mirror indices.

## Modelling

- **`get_model_instance`**: [[../source/modelling/model_py#Functions|model.py]] - Factory for creating model objects.
- **`load_onnx_model`**: [[../source/modelling/model_py#Functions|model.py]] - Initializes ONNX Runtime session.
- **`onnx_inference`**: [[../source/modelling/model_py#Functions|model.py]] - Runs inference helper.
- **`train`**: [[source/modelling/train_py#Functions|train.py]] - Main training loop (Epoch/Batch iteration).
- **`export_model`**: [[source/modelling/export_py#Functions|export.py]] - Converts PyTorch checkpoint to ONNX.

## Dashboard
- **`main`**: [[source/modelling/dashboard/app_py#Functions|app.py]] - Dashboard entry point.
- **`run_inference`**: [[source/modelling/dashboard/loader_py#Functions|loader.py]] - Runs cached inference.
- **`render_metrics_view`**: [[source/modelling/dashboard/views_py#Functions|views.py]] - Renders accuracy/confusion matrix.
- **`render_inspector_view`**: [[source/modelling/dashboard/views_py#Functions|views.py]] - Renders single-sample 3D visualization.
- **`plot_3d_animation`**: [[source/modelling/dashboard/visualization_py#Functions|visualization.py]] - Low-level Plotly 3D plotting.
