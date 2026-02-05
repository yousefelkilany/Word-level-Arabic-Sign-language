# Word-Level Arabic Sign Language Recognition

This project implements a real-time Arabic Sign Language (ArSL) recognition system using the **KArSL-502** dataset. It utilizes **MediaPipe** for pose and hand landmark extraction and a **Spatial-Temporal Transformer (ST-Transformer)** PyTorch model for sequence classification.

## Table of Contents

- [Word-Level Arabic Sign Language Recognition](#word-level-arabic-sign-language-recognition)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Configuration](#configuration)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
    - [Option 1: Running with Docker](#option-1-running-with-docker)
    - [Option 2: Running Locally](#option-2-running-locally)
    - [Model Export](#model-export)
    - [Benchmarking](#benchmarking)
    - [Accessing the Web Interface](#accessing-the-web-interface)
    - [Local Setup (Data/Labels)](#local-setup-datalabels)
  - [Repository Structure](#repository-structure)
  - [Model Architecture](#model-architecture)
    - [1. Group Token Embedding](#1-group-token-embedding)
    - [2. Positional Encoding](#2-positional-encoding)
    - [3. Spatial-Temporal Blocks](#3-spatial-temporal-blocks)
    - [4. Self-Attention Pooling](#4-self-attention-pooling)
    - [5. Classification Head](#5-classification-head)
  - [Resources](#resources)
    - [Project Derived Data](#project-derived-data)
    - [Official Source Materials](#official-source-materials)
    - [Citation](#citation)

## Features

- **Sign Recognition**: Token-based inference via WebSocket using an ONNX model.
- **Web Interface**: Browser-based frontend for webcam interaction.
- **Model Architecture**: Spatial-Temporal Transformer (ST-Transformer) for temporal sequence classification.
- **Preprocessing Pipeline**: Keypoint extraction via MediaPipe for CPU execution.

## Configuration

Create a `.env` file in the root directory based on `.env.example`. This is required for path resolution.

```bash
cp .env.example .env
```

| Variable                   | Description                                                                        | Default                 |
| -------------------------- | ---------------------------------------------------------------------------------- | ----------------------- |
| `ONNX_CHECKPOINT_FILENAME` | Filename of the ONNX model in `models/`                                            | `checkpoint...onnx`     |
| `DOMAIN_NAME`              | Allowed origin for CORS (Frontend URL)                                             | `http://localhost:8000` |
| `LOCAL_DEV`                | Set to `1` to use local `data/` and `models/` directories instead of Kaggle paths. | `1`                     |
| `USE_CPU`                  | Force PyTorch and ONNX to use CPU even if GPU is available.                        | `1`                     |

## Installation

### Prerequisites

- **Docker** (Primary setup)
- *OR* **Python 3.12+** and [uv](https://github.com/astral-sh/uv)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yousefelkilany/word-level-arabic-sign-language.git
    cd word-level-arabic-sign-language
    ```

2. **Setup Configuration:**
    Ensure you have created the `.env` file as described in the Configuration section.

3. **Download LFS Files:**
    Retrieve MediaPipe landmarkers and the ONNX model files:

    ```bash
    make download_lfs_files
    ```

## Usage

### Option 1: Running with Docker

Build and start the services using Docker Compose.

```bash
# Using Makefile
make docker_run

# OR Direct command
docker compose up --build --force-recreate
```

The API will be available at `http://localhost:8000`.

*Note: The project mounts the current directory to `/app` in the container; code changes are reflected immediately.*

### Option 2: Running Locally

1. **Install Dependencies:**

    ```bash
    uv sync
    ```

2. **Run the Backend:**

    The application can be run directly with `uv` or via the `Makefile`.

    ```bash
    # Using Makefile (Recommended)
    make run

    # OR Direct execution
    cd src && uv run -m api.run
    ```

    **Key Makefile commands:**
    - `make run`: Runs the API server locally.
    - `make docker_run`: Builds and runs with Docker Compose.
    - `make train`: Runs the training script (`modelling.train`).
    - `make parallel_train`: Runs multi-GPU training (`modelling.parallel_train`).
    - `make export_onnx`: Exports a PyTorch checkpoint to ONNX (`modelling.export`).
    - `make onnx_benchmark`: Benchmarks the ONNX model performance.
    - `make prepare_npz_kps`: Prepares NPZ keypoints from raw data.
    - `make preprocess_mmap_data`: Preprocesses data for Memory Mapping.

### Model Export

To export a trained PyTorch model to ONNX format:

```bash
make export_onnx checkpoint_path=path/to/checkpoint.pth onnx_model_path=path/to/output.onnx
```

### Benchmarking

To benchmark the inference speed of an ONNX model:

```bash
make onnx_benchmark checkpoint_path=path/to/model.onnx
```

### Accessing the Web Interface

Navigate to [http://localhost:8000/live-signs](http://localhost:8000/live-signs) in a web browser. The system detects the user's pose and hands to predict Arabic signs.

### Local Setup (Data/Labels)

For manual training or evaluation:

- **Labels**: Download `KARSL-502_Labels.xlsx` from the [Google Drive link](https://drive.google.com/drive/folders/1LI6L7MSXOIwSgbVL0zmjnw7wryZ6aYl-) and place it in the `data/` directory.
- **Data**: Place raw videos or preprocessed keypoints in the `data/` directory.

## Repository Structure

The core logic resides in the `src/` directory:

- `src/api/`: FastAPI application, WebSocket handlers, and server entry point.
  - `main.py`: FastAPI app definition.
  - `run.py`: Server runner with reload enabled.
  - `websocket.py`: WebSocket logic for real-time inference.
- `src/modelling/`: PyTorch model architecture and training/export scripts.
  - `model.py`: **STTransformer** definition and ONNX loading.
  - `train.py`: Training loop and validation.
  - `export.py`: ONNX export utility.
- `src/data/`: Data processing, loading, and preprocessing scripts.
  - `prepare_npz_kps.py`: Keypoint extraction and preparation.
  - `dataloader.py`: PyTorch DataLoaders.
- `src/core/`: Shared constants and utility functions.
  - `constants.py`: Path configurations and global constants.
- `static/`: Frontend interface files (`index.html`, JS, CSS).
- `landmarkers/`: Pre-trained MediaPipe landmark models (`.task` files).
- `models/`: Production-ready ONNX models.
- `data/`: Dataset storage (videos, labels, keypoints).

## Model Architecture

The recognition system is based on the **Spatial-Temporal Transformer (ST-Transformer)**, a dual-stream attention neural network architecture for temporal sequence classification. It models both spatial relationships between anatomical regions and temporal dynamics of signs.

### 1. Group Token Embedding

Keypoints are grouped into anatomical regions (Pose, Face, Hands) and projected into a latent space:

- **Anatomical Tokens**: Pose, Face, and Hands are treated as independent tokens for attention mechanisms.
- **Normalization**: Input features are stabilized using Batch Normalization.
- **Learnable Embeddings**: Part-specific embeddings are added to distinguish anatomical regions.

### 2. Positional Encoding

- **Temporal Context**: Sinusoidal positional encodings are added to the temporal dimension to preserve frame order information.
- **Scale Invariance**: The model handles variable-length sequences through temporal attention.

### 3. Spatial-Temporal Blocks

The core processing consists of multiple ST-Transformer blocks:

- **Spatial Attention**: Multi-head self-attention operating across body parts in a single frame.
- **Temporal Attention**: Multi-head self-attention operating across the sequence for each body part.
- **Residual Processing**: Each attention stream uses residual connections and Layer Normalization.

### 4. Self-Attention Pooling

- **Weighted Aggregation**: A trainable attention mechanism identifies the most significant frames in a sequence for sign recognition.
- **Feature Fusion**: Produces a unified context vector for the entire sign clip.

### 5. Classification Head

- **Projections**: The context vector is passed through a classification head with dropout regularization.
- **Logits**: Produces probability distributions for the **502 sign classes**.

## Resources

### Project Derived Data

- **[Preprocessed NPZ Dataset](https://www.kaggle.com/datasets/youssefelkilany/word-level-arabic-sign-language-extrcted-keypoints/data)**: Keypoints extracted for training using `prepare_npz_kps` script.
- **[Preprocessed Mmap Dataset](https://www.kaggle.com/datasets/youssefelkilany/word-level-arabic-sign-language-preprcsd-keypoints/data)**: Keypoints serialized as Mmap files for efficient loading using `mmap_dataset_preprocessing` script.

### Official Source Materials

- **[KArSL Dataset](https://hamzah-luqman.github.io/KArSL/)**: Official database source for the Arabic Sign Language dataset.
- **[KArSL Dataset on Google Drive](https://drive.google.com/drive/folders/1LI6L7MSXOIwSgbVL0zmjnw7wryZ6aYl-)**: Raw dataset and metadata files.
- **[KArSL Dataset on Kaggle](https://www.kaggle.com/datasets/yousefdotpy/karsl-502)**: Image dataset hosted on Kaggle.

### Citation

```bibtex
@article{sidig2021karsl,
    title={KArSL: Arabic Sign Language Database},
    author={Sidig, Ala Addin I and Luqman, Hamzah and Mahmoud, Sabri and Mohandes, Mohamed},
    journal={ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)},
    volume={20},
    number={1},
    pages={1--19},
    year={2021},
    publisher={ACM New York, NY, USA}
}
```
