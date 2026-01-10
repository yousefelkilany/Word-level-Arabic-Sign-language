# Word-Level Arabic Sign Language Recognition

This project implements a real-time Arabic Sign Language (ArSL) recognition system using the **KArSL-502** dataset. It utilizes **MediaPipe** for pose and hand landmark extraction and a custom **Attention-based BiLSTM** PyTorch model for sequence classification.

## Features

* **Real-time Recognition**: Inference via WebSocket connectivity using a lightweight **ONNX** model.
* **Web Interface**: Minimalist HTML5/JS frontend for live webcam interaction.
* **Deep Learning Model**: Bidirectional LSTM with Multi-Head Self-Attention.
* **Efficient Pipeline**: Preprocessing and keypoint extraction pipeline optimized for CPU execution.

## Installation

### Prerequisites

* Python 3.12 or higher
* [uv](https://github.com/astral-sh/uv) package manager

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yousefelkilany/word-level-arabic-sign-language.git
    cd word-level-arabic-sign-language
    ```

2. **Install Dependencies:**
    This project uses `uv` for dependency management.

    ```bash
    # Sync dependencies from pyproject.toml
    uv sync
    ```

## Usage

### Running the Backend

Start the FastAPI server using `uvicorn` (or the provided `Makefile` if on Windows/Linux with Make installed).

```bash
python api.py
# OR if you have make installed
make local_setup && python api.py
```

The server defaults to `http://0.0.0.0:8000`.

### Accessing the Web Interface

Navigate to [http://localhost:8000/live-signs](http://localhost:8000/live-signs) in a web browser. Ensure the browser has permission to access the webcam. The system will detect the user's pose and hands to predict Arabic signs in real-time.

### Local Setup

* **Labels**: Download `KARSL-502_Labels.xlsx` from the Google Drive link and place it in the `data/` directory.
* **Data**: Place raw videos or preprocessed keypoints in the `data/` directory.

## Repository Structure

* `api.py`: FastAPI application and WebSocket handlers.
* `model.py`: PyTorch model architecture definition.
* `train.py`: Training loop and validation procedures.
* `utils.py`, `mediapipe_utils.py`: Helper functions for data processing and keypoint extraction.
* `live-signs.html/js`: Frontend interface files.
* `onnx_benchmark.py`: Utility for benchmarking ONNX model performance.

## Model Architecture

The core model is an **AttentionBiLSTM** consisting of the following components:

1. **Input Projection**: Linearly projects keypoints to hidden dimensions.
2. **BiLSTM Layers**: 4 layers of Residual Bidirectional LSTMs.
3. **Self-Attention**: Multi-Head Attention mechanism to capture temporal dependencies.
4. **Classification Head**: Fully Connected layer to output probabilities for 502 classes.

### Resources

* **[Official Website](https://hamzah-luqman.github.io/KArSL/)**: The primary source for information about the dataset.
* **[Kaggle Dataset](https://www.kaggle.com/datasets/yousefdotpy/karsl-502)**: Used for training models and extracting keypoints on Kaggle GPUs.
* **[Google Drive](https://drive.google.com/drive/folders/1LI6L7MSXOIwSgbVL0zmjnw7wryZ6aYl-)**: Contains the raw dataset and the required `KARSL-502_Labels.xlsx` file.

## Dataset

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
