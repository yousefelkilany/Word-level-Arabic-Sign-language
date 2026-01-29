---
title: Environment Configuration
date: 2026-01-28
lastmod: 2026-01-28
---

# Environment Configuration

#deployment #configuration #env

The application behavior is controlled by environment variables, managed via a `.env` file in the project root.

## Variable Reference

| Variable                   | Description                                                         | Default / Example                    |
| :------------------------- | :------------------------------------------------------------------ | :----------------------------------- |
| `ONNX_CHECKPOINT_FILENAME` | Filename of the .onnx model to load from `models/`.                 | `last-checkpoint-signs_502.pth.onnx` |
| `DOMAIN_NAME`              | Allowed origin for CORS and WebSocket connections.                  | `http://localhost:8000`              |
| `LOCAL_DEV`                | Flag to indicate local development environment. Affects data paths. | `1`                                  |
| `USE_CPU`                  | Force PyTorch/ONNX to use CPU instead of CUDA.                      | `1`                                  |
| `OMP_NUM_THREADS`          | Number of threads for OpenMP (Linear Algebra).                      | `1` (Container Default)              |

## Setup

1.  Copy the example file:

    ```bash
    cp .env.example .env
    ```

2.  Edit `.env` to match your local setup.

## Constants (`src/core/constants.py`)
Some configurations are hardcoded constants but can be influenced by environment flags:

- `MODELS_DIR`: Directory for model checkpoints.
- `DATA_DIR`: Root data directory.

## Related Documentation

- [[../source/core/constants_py|constants.py Source Code]]
- [[../deployment/docker_setup|Docker Setup]]
