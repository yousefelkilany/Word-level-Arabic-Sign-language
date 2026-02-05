---
title: Arabic Sign Language Recognition - Documentation
date: 2026-01-28
lastmod: 2026-02-05
aliases: ["Documentation Home", "Obsidian Vault Entry"]
---

# Arabic Sign Language Recognition - Documentation

Welcome to the comprehensive documentation for the **Word-Level Arabic Sign Language Recognition** project. This documentation is organized as an Obsidian vault with interconnected pages covering both high-level concepts and detailed source code documentation.

#documentation #karsl #sign-language #deep-learning

## 🚀 Quick Start

- [[getting_started|Getting Started]] - Installation, configuration, and first-time setup
- [[architecture_overview|Architecture Overview]] - System architecture and component interactions
- [[reference/troubleshooting|Troubleshooting]] - Common issues and solutions

## 📚 Conceptual Documentation

### API & Backend
- [[api/fastapi_application|FastAPI Application]] - Application structure, middleware, and routing
- [[api/websocket_communication|WebSocket Communication]] - Real-time communication patterns
- [[api/live_processing_pipeline|Live Processing Pipeline]] - Frame processing workflow

### Core Components
- [[core/mediapipe_integration|MediaPipe Integration]] - Landmark detection and keypoint extraction
- [[core/keypoint_visualization|Keypoint Visualization]] - Drawing and visualization strategies

### Data Processing
- [[data/dataset_overview|Dataset Overview]] - KArSL-502 dataset description and structure
- [[data/data_preparation_pipeline|Data Preparation Pipeline]] - Preprocessing workflow
- [[data/memory_mapped_datasets|Memory-Mapped Datasets]] - Efficient dataset implementation

### Models
- [[models/architecture_design|Architecture Design]] - ST-Transformer model architecture
- [[models/training_process|Training Process]] - Training, hyperparameters, and best practices
- [[models/onnx_export_process|ONNX Export Process]] - Model export and conversion

### Frontend
- [[frontend/web_interface_design|Web Interface Design]] - HTML/CSS/JS architecture
- [[frontend/websocket_client_implementation|WebSocket Client]] - Client-side implementation

### Deployment & Development
- [[deployment/docker_setup|Docker Setup]] - Container configuration and usage
- [[deployment/environment_configuration|Environment Configuration]] - Environment variables
- [[development/project_structure|Project Structure]] - Repository organization
- [[development/contributing_guide|Contributing Guide]] - Contribution guidelines
- [[development/makefile_commands|Makefile Commands]] - Build automation

## 🔍 Source Code Documentation

Complete function-level documentation mirroring the repository structure:

### API Source Code (`src/api/`)
- [[source/api/main_py|main.py]] - FastAPI application setup and routes
- [[source/api/websocket_py|websocket.py]] - WebSocket handler for real-time detection
- [[source/api/live_processing_py|live_processing.py]] - Frame buffer and processing
- [[source/api/cv2_utils_py|cv2_utils.py]] - OpenCV utilities
- [[source/api/run_py|run.py]] - Application entry point

### Core Source Code (`src/core/`)
- [[source/core/constants_py|constants.py]] - System constants and configuration
- [[source/core/mediapipe_utils_py|mediapipe_utils.py]] - MediaPipe processing
- [[source/core/utils_py|utils.py]] - General utilities
- [[source/core/draw_kps_py|draw_kps.py]] - Keypoint visualization

### Data Source Code (`src/data/`)
- [[source/data/data_preparation_py|data_preparation.py]] - Dataset preparation
- [[source/data/dataloader_py|dataloader.py]] - PyTorch DataLoader
- [[source/data/lazy_dataset_py|lazy_dataset.py]] - Lazy loading dataset
- [[source/data/mmap_dataset_py|mmap_dataset.py]] - Memory-mapped dataset
- [[source/data/mmap_dataset_preprocessing_py|mmap_dataset_preprocessing.py]] - Preprocessing
- [[source/data/prepare_npz_kps_py|prepare_npz_kps.py]] - NPZ keypoint preparation
- [[source/data/shared_elements_py|shared_elements.py]] - Shared utilities
- [[source/data/write_signs_to_json_py|write-signs-to-json.py]] - JSON export
- [[source/data/generate_mediapipe_face_symmetry_map_py|generate_mediapipe_face_symmetry_map.py]] - Face symmetry mapping

### Modelling Source Code (`src/modelling/`)
- [[source/modelling/model_py|model.py]] - Neural network architecture
- [[source/modelling/train_py|train.py]] - Training script
- [[source/modelling/parallel_train_py|parallel_train.py]] - Parallel training
- [[source/modelling/export_py|export.py]] - Model export to ONNX
- [[source/modelling/onnx_benchmark_py|onnx_benchmark.py]] - ONNX benchmarking
- [[source/modelling/visualize_model_performance_py|visualize_model_performance.py]] - Performance visualization

#### Dashboard (`src/modelling/dashboard/`)
- [[source/modelling/dashboard/app_py|app.py]] - Dashboard application
- [[source/modelling/dashboard/loader_py|loader.py]] - Data loader
- [[source/modelling/dashboard/views_py|views.py]] - Dashboard views
- [[source/modelling/dashboard/visualization_py|visualization.py]] - Visualization utilities

### Frontend Source Code (`static/`)
- [[source/frontend/live_signs_js|live-signs.js]] - WebSocket client and camera handling
- [[source/frontend/index_html|index.html]] - Main HTML structure
- [[source/frontend/styles_css|styles.css]] - Styling and layout

### Configuration Files
- [[source/config/dockerfile|Dockerfile]] - Container image configuration
- [[source/config/docker_compose_yml|docker-compose.yml]] - Multi-container orchestration
- [[source/config/makefile|Makefile]] - Build automation
- [[source/config/pyproject_toml|pyproject.toml]] - Python project configuration

## 📖 Reference

- [[function_index|Function Index]] - Alphabetical index of all functions
- [[class_index|Class Index]] - Alphabetical index of all classes
- [[reference/api_endpoints|API Endpoints]] - Complete API reference
- [[reference/configuration_options|Configuration Options]] - All configuration variables
- [[reference/dataset_citation|Dataset Citation]] - KArSL dataset citation

## 🏗️ Project Overview

This project implements a real-time Arabic Sign Language (ArSL) recognition system using:
- **Dataset**: KArSL-502 (502 Arabic sign words)
- **Keypoint Extraction**: MediaPipe (pose, face, hands)
- **Model**: Spatial-Temporal Transformer (ST-Transformer)
- **Inference**: ONNX Runtime for optimized CPU execution
- **Frontend**: HTML5/JavaScript with WebSocket communication
- **Backend**: FastAPI with async WebSocket support

## 📝 Documentation Conventions

- **Wiki Links**: Use `[[page_name]]` to navigate between pages
- **Tags**: Use `#tag` for categorization
- **Code References**: Functions and classes link to their detailed documentation
- **Bidirectional Links**: Each function shows where it's called from and what it calls
