# Word-Level Arabic Sign Language Recognition

This project implements a real-time Arabic Sign Language (ArSL) recognition system using the **KArSL-502** dataset. It utilizes **MediaPipe** for pose and hand landmark extraction and a custom **Attention-based BiLSTM** PyTorch model for sequence classification.

## Features

* **Real-time Recognition**: Inference via WebSocket connectivity using a lightweight **ONNX** model.
* **Web Interface**: Minimalist HTML5/JS frontend for live webcam interaction.
* **Deep Learning Model**: Bidirectional LSTM with Multi-Head Self-Attention.
* **Efficient Pipeline**: Preprocessing and keypoint extraction pipeline optimized for CPU execution.

## Configuration

Create a `.env` file in the root directory based on `.env.example`. This is **required** for correct path resolution.

```bash
cp .env.example .env
```

| Variable                   | Description                                                                                       | Default                 |
| -------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------- |
| `ONNX_CHECKPOINT_FILENAME` | Filename of the ONNX model in `models/`                                                           | `checkpoint...onnx`     |
| `DOMAIN_NAME`              | Allowed origin for CORS (Frontend URL)                                                            | `http://localhost:8000` |
| `LOCAL_DEV`                | **Important**: Set to `1` to use local `data/` and `models/` directories instead of Kaggle paths. | `1`                     |
| `USE_CPU`                  | Force PyTorch and ONNX to use CPU even if GPU is available.                                       | `1`                     |

## Installation

### Prerequisites

* **Docker** (Recommended for easy setup)
* *OR* **Python 3.12+** and [uv](https://github.com/astral-sh/uv)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yousefelkilany/word-level-arabic-sign-language.git
    cd word-level-arabic-sign-language
    ```

2. **Setup Configuration:**
    Ensure you have created the `.env` file as described in the Configuration section.

## Usage

### Option 1: Running with Docker (Recommended)

Build and start the services using Docker Compose. This handles all dependencies and environment setup.

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

*Note: The project mounts the current directory to `/app` in the container, so code changes are reflected immediately.*

### Option 2: Running Locally

1. **Install Dependencies:**

    ```bash
    uv sync
    ```

2. **Run the Backend:**

    You can run the application directly with Python or use the provided Makefile.

    ```bash
    # Direct Python execution (Ensure LOCAL_DEV=1 is in .env)
    python api.py

    # OR with Make (Linux/Windows with Make)
    make local_setup && python api.py
    ```

    **Makefile commands:**
    * `make local_setup`: Sets `LOCAL_DEV=1` for the command duration.
    * `make train`: Runs the training script (`train.py`).
    * `make cpu_train`: Runs training on CPU (`USE_CPU=1`).
    * `make export_model`: Exports a PyTorch checkpoint to ONNX (`export.py`).
    * `make onnx_benchmark`: Benchmarks the ONNX model performance (`onnx_benchmark.py`).

### Model Export

To export a trained PyTorch model to ONNX format:

```bash
make export_model checkpoint_path=path/to/checkpoint.pth onnx_path=path/to/output_onnx/
```

### Benchmarking

To benchmark the inference speed of an ONNX model:

```bash
make onnx_benchmark checkpoint_path=path/to/model_onnx.onnx
```

### Accessing the Web Interface

Navigate to [http://localhost:8000/live-signs](http://localhost:8000/live-signs) in a web browser. Ensure the browser has permission to access the webcam. The system will detect the user's pose and hands to predict Arabic signs in real-time.

### Local Setup (Data/Labels)

For training or evaluation manually:

* **Labels**: Download `KARSL-502_Labels.xlsx` from the Google Drive link and place it in the `data/` directory.
* **Data**: Place raw videos or preprocessed keypoints in the `data/` directory.

## Repository Structure

* `api.py`: FastAPI application and WebSocket handlers.
* `run.py`: Entry point for starting the FastAPI server with reload enabled.
* `model.py`: PyTorch model architecture definition (**AttentionBiLSTM**).
* `train.py`: Training loop, validation, and checkpoint saving.
* `export.py`: Utility to export PyTorch models to ONNX with consistency checking.
* `onnx_benchmark.py`: Benchmarks ONNX model inference latency.
* `dataset_preprocessing.py`: Processes raw dataset videos into structured formats.
* `prepare_kps.py`: Extracts keypoints from the dataset using MediaPipe.
* `draw_kps.py`: Visualization utility for MediaPipe keypoints.
* `utils.py`, `mediapipe_utils.py`, `cv2_utils.py`: Core utilities for processing and landmarks.
* `static/`: Frontend interface files (`live-signs.html`, JS, CSS).
* `*.task`: Pre-trained MediaPipe landmark models for hands, pose, and face.
* `Dockerfile` & `docker-compose.yml`: Containerization configuration.
* `models/`: Production-ready ONNX models.
* `checkpoints/`: Training output and model weights.

## Model Architecture

The core model is an **AttentionBiLSTM** consisting of the following components:

1. **Input Projection**: Linearly projects keypoints to hidden dimensions.
2. **BiLSTM Layers**: 4 layers of Residual Bidirectional LSTMs.
3. **Self-Attention**: Multi-Head Attention mechanism to capture temporal dependencies.
4. **Classification Head**: Fully Connected layer to output probabilities for 502 classes.

## Resources

* **[Official Website](https://hamzah-luqman.github.io/KArSL/)**: The primary source for information about the dataset.
* **[Kaggle Dataset](https://www.kaggle.com/datasets/yousefdotpy/karsl-502)**: Used for training models and extracting keypoints on Kaggle GPUs.
* **[Google Drive](https://drive.google.com/drive/folders/1LI6L7MSXOIwSgbVL0zmjnw7wryZ6aYl-)**: Contains the raw dataset and the required `KARSL-502_Labels.xlsx` file.

### Dataset

This project relies on the **KArSL (Video dataset for Word-Level Arabic sign language)** dataset.

### Citation

KArSL: Arabic Sign Language Database

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
