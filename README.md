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
    * `make train`: Runs the training script.
    * `make cpu_train`: Runs training on CPU (`USE_CPU=1`).

### Accessing the Web Interface

Navigate to [http://localhost:8000/live-signs](http://localhost:8000/live-signs) in a web browser. Ensure the browser has permission to access the webcam. The system will detect the user's pose and hands to predict Arabic signs in real-time.

### Local Setup (Data/Labels)

For training or evaluation manually:

* **Labels**: Download `KARSL-502_Labels.xlsx` from the Google Drive link and place it in the `data/` directory.
* **Data**: Place raw videos or preprocessed keypoints in the `data/` directory.

## Repository Structure

* `api.py`: FastAPI application and WebSocket handlers.
* `model.py`: PyTorch model architecture definition.
* `train.py`: Training loop and validation procedures.
* `utils.py`, `mediapipe_utils.py`: Helper functions for data processing and keypoint extraction.
* `live-signs.html/js`: Frontend interface files.
* `onnx_benchmark.py`: Utility for benchmarking ONNX model performance.
* `Dockerfile` & `docker-compose.yml`: Container configuration.
* `models/`: Directory for ONNX models.

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

## License

[MIT License](LICENSE)
