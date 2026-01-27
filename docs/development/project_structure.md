# Project Structure

#development #structure #codebase

The project follows a standard Python project layout with separated source, documentation, and configuration directories.

## Top-Level Directory

```text
.
├── docker/                 # Docker related files (if any extra config)
├── docs/                   # Documentation (Obsidian Vault)
├── logs/                   # Application logs
├── src/                    # Source code
├── static/                 # Frontend assets (HTML/JS/CSS)
├── checkpoints/            # Model checkpoints
├── data/                   # Dataset directory
├── Dockerfile              # Main Docker build file
├── docker-compose.yml      # Service orchestration
├── Makefile                # Command shortcuts
├── pyproject.toml          # Python dependencies (uv/pip)
└── .env                    # Environment variables
```

## Source Code (`src/`)

The `src/` directory is the python package root.

- **`api/`**: FastAPI application and routing.
    - `main.py`: Entry point.
    - `websocket.py`: Connection handling.
- **`core/`**: Shared utilities and constants.
    - `mediapipe_utils.py`: Landmark extraction logic.
    - `draw_kps.py`: Visualization.
- **`data/`**: Data loading and processing.
    - `mmap_dataset.py`: Memory-mapped dataset class.
- **`modelling/`**: Neural network components.
    - `model.py`: Architecture definition.
    - `train.py`: Training loop.
    - `export.py`: ONNX export script.

## Frontend (`static/`)

Contains the vanilla JS frontend.
- `index.html`: Main entry UI.
- `live-signs.js`: Logic.
- `styles.css`: Styling.

## Related Documentation

- [[development/contributing_guide|Contributing Guide]]
