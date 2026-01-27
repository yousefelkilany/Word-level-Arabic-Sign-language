# Arabic Sign Language Recognition - Documentation

Welcome to the comprehensive documentation for the **Word-Level Arabic Sign Language Recognition** project. This documentation is organized as an Obsidian vault with interconnected pages covering both high-level concepts and detailed source code documentation.

#documentation #karsl #sign-language #deep-learning

## 🚀 Quick Start

- [[getting-started|Getting Started]] - Installation, configuration, and first-time setup
- [[architecture-overview|Architecture Overview]] - System architecture and component interactions
- [[reference/troubleshooting|Troubleshooting]] - Common issues and solutions

## 📚 Conceptual Documentation

### API & Backend
- [[api/fastapi-application|FastAPI Application]] - Application structure, middleware, and routing
- [[api/websocket-communication|WebSocket Communication]] - Real-time communication patterns
- [[api/live-processing-pipeline|Live Processing Pipeline]] - Frame processing workflow

### Core Components
- [[core/mediapipe-integration|MediaPipe Integration]] - Landmark detection and keypoint extraction
- [[core/keypoint-visualization|Keypoint Visualization]] - Drawing and visualization strategies

### Data Processing
- [[data/dataset-overview|Dataset Overview]] - KArSL-502 dataset description and structure
- [[data/data-preparation-pipeline|Data Preparation Pipeline]] - Preprocessing workflow
- [[data/memory-mapped-datasets|Memory-Mapped Datasets]] - Efficient dataset implementation

### Models
- [[models/architecture-design|Architecture Design]] - AttentionBiLSTM model architecture
- [[models/training-process|Training Process]] - Training, hyperparameters, and best practices
- [[models/onnx-export-process|ONNX Export Process]] - Model export and conversion

### Frontend
- [[frontend/web-interface-design|Web Interface Design]] - HTML/CSS/JS architecture
- [[frontend/websocket-client-implementation|WebSocket Client]] - Client-side implementation

### Deployment & Development
- [[deployment/docker-setup|Docker Setup]] - Container configuration and usage
- [[deployment/environment-configuration|Environment Configuration]] - Environment variables
- [[development/project-structure|Project Structure]] - Repository organization
- [[development/contributing-guide|Contributing Guide]] - Contribution guidelines
- [[development/makefile-commands|Makefile Commands]] - Build automation

## 🔍 Source Code Documentation

Complete function-level documentation mirroring the repository structure:

### API Source Code (`src/api/`)
- [[source/api/main-py|main.py]] - FastAPI application setup and routes
- [[source/api/websocket-py|websocket.py]] - WebSocket handler for real-time detection
- [[source/api/live-processing-py|live_processing.py]] - Frame buffer and processing
- [[source/api/cv2-utils-py|cv2_utils.py]] - OpenCV utilities
- [[source/api/run-py|run.py]] - Application entry point

### Core Source Code (`src/core/`)
- [[source/core/constants-py|constants.py]] - System constants and configuration
- [[source/core/mediapipe-utils-py|mediapipe_utils.py]] - MediaPipe processing
- [[source/core/utils-py|utils.py]] - General utilities
- [[source/core/draw-kps-py|draw_kps.py]] - Keypoint visualization

### Data Source Code (`src/data/`)
- [[source/data/data-preparation-py|data_preparation.py]] - Dataset preparation
- [[source/data/dataloader-py|dataloader.py]] - PyTorch DataLoader
- [[source/data/lazy-dataset-py|lazy_dataset.py]] - Lazy loading dataset
- [[source/data/mmap-dataset-py|mmap_dataset.py]] - Memory-mapped dataset
- [[source/data/mmap-dataset-preprocessing-py|mmap_dataset_preprocessing.py]] - Preprocessing
- [[source/data/prepare-npz-kps-py|prepare_npz_kps.py]] - NPZ keypoint preparation
- [[source/data/shared-elements-py|shared_elements.py]] - Shared utilities
- [[source/data/write-signs-to-json-py|write-signs-to-json.py]] - JSON export
- [[source/data/generate-mediapipe-face-symmetry-map-py|generate_mediapipe_face_symmetry_map.py]] - Face symmetry mapping

### Modelling Source Code (`src/modelling/`)
- [[source/modelling/model-py|model.py]] - Neural network architecture
- [[source/modelling/train-py|train.py]] - Training script
- [[source/modelling/parallel-train-py|parallel_train.py]] - Parallel training
- [[source/modelling/export-py|export.py]] - Model export to ONNX
- [[source/modelling/onnx-benchmark-py|onnx_benchmark.py]] - ONNX benchmarking
- [[source/modelling/visualize-model-performance-py|visualize_model_performance.py]] - Performance visualization

#### Dashboard (`src/modelling/dashboard/`)
- [[source/modelling/dashboard/app-py|app.py]] - Dashboard application
- [[source/modelling/dashboard/loader-py|loader.py]] - Data loader
- [[source/modelling/dashboard/views-py|views.py]] - Dashboard views
- [[source/modelling/dashboard/visualization-py|visualization.py]] - Visualization utilities

### Frontend Source Code (`static/`)
- [[source/frontend/live-signs-js|live-signs.js]] - WebSocket client and camera handling
- [[source/frontend/index-html|index.html]] - Main HTML structure
- [[source/frontend/styles-css|styles.css]] - Styling and layout

### Configuration Files
- [[source/config/dockerfile|Dockerfile]] - Container image configuration
- [[source/config/docker-compose-yml|docker-compose.yml]] - Multi-container orchestration
- [[source/config/makefile|Makefile]] - Build automation
- [[source/config/pyproject-toml|pyproject.toml]] - Python project configuration

## 📖 Reference

- [[function-index|Function Index]] - Alphabetical index of all functions
- [[class-index|Class Index]] - Alphabetical index of all classes
- [[reference/api-endpoints|API Endpoints]] - Complete API reference
- [[reference/configuration-options|Configuration Options]] - All configuration variables
- [[reference/dataset-citation|Dataset Citation]] - KArSL dataset citation

## 🏗️ Project Overview

This project implements a real-time Arabic Sign Language (ArSL) recognition system using:
- **Dataset**: KArSL-502 (502 Arabic sign words)
- **Keypoint Extraction**: MediaPipe (pose, face, hands)
- **Model**: Attention-based Bidirectional LSTM
- **Inference**: ONNX Runtime for optimized CPU execution
- **Frontend**: HTML5/JavaScript with WebSocket communication
- **Backend**: FastAPI with async WebSocket support

## 📝 Documentation Conventions

- **Wiki Links**: Use `[[page-name]]` to navigate between pages
- **Tags**: Use `#tag` for categorization
- **Code References**: Functions and classes link to their detailed documentation
- **Bidirectional Links**: Each function shows where it's called from and what it calls

---

*Last Updated: 2026-01-27*
