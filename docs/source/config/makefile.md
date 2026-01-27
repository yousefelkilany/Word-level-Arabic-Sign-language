# Makefile

#source #config #makefile

**File Path**: `Makefile`

**Purpose**: Task runner for common development commands (Training, Preprocessing, Visualization).

## Overview

Defines shortcuts that execute python modules via `uv run`. Supports argument passing from the command line.

## Key Commands

### Training & Modelling
- **`make train`**: Runs `modelling.train`.
- **`make parallel_train`**: Runs DDP training.
- **`make export_model`**: Runs `modelling.export`.
- **`make onnx_benchmark`**: Profiling script.

### Data Processing
- **`make prepare_npz_kps`**: Frame -> NPZ extraction.
- **`make preprocess_mmap_data`**: NPZ -> mmap conversion.
- **`make generate_face_map`**: Utility script.

### Dashboard
- **`make visualization_dashboard`**: Launches Streamlit app.

## Variable Handling

The Makefile creates argument strings (e.g., `ARGS_DATA`) by checking environment variables or command-line overrides.

**Example**:
```makefile
ARGS_DATA += $(if $(splits),--splits $(splits))
```
**Usage**:
```bash
make prepare_npz_kps splits="train" signers="01"
```

## Helper Targets
- `cpu_%`: Prepends `USE_CPU=1` to any command.
  - usage `make cpu_train`
- `local_%`: Prepends `LOCAL_DEV=1`.
  - usage `make local_train`

## Related Documentation

**Invokes**:
- [[../../source/modelling/train_py|train.py]]
- [[../../source/data/prepare_npz_kps_py|prepare_npz_kps.py]]
