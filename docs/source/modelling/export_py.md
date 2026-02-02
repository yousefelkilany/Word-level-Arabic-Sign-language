---
title: export.py
date: 2026-01-28
lastmod: 2026-01-29
src_hash: e77cf69257c66d05af5c35fc4b6f51d85cfee05119416b78c52cd8ca1812a094
aliases: ["ONNX Export Script", "Model Conversion Pipeline"]
---

# export.py

#source #modelling #onnx #deployment

**File Path**: `src/modelling/export.py`

**Purpose**: Exports a trained PyTorch checkpoint to the ONNX format for optimized inference.

## Overview

Loads a saved PyTorch model state, verifies it against a dummy input, and traces the computation graph to produce an ONNX file. It also verifies the exported model matches the PyTorch outputs within a strict tolerance.

## process

```mermaid
graph LR
    A[Checkpoint.pth] --> B[Load PyTorch Model]
    B --> C{Verify Output?}
    C -->|Yes| D[torch.onnx.export]
    D --> E[Model.onnx]
    E --> F[onnxruntime Check]
```

## Functions

### `export_model(checkpoint_path, output_path)`

```python
def export_model(checkpoint_path, output_path=None):
```

**Steps**:

1. **Load**: Recreates model structure using `num_signs` from filename.
2. **Dummy Input**: Creates a random tensor `(1, 50, 736)`.
3. **Export**:
   - **Dynamic Axes**: Allows batch size to vary (e.g., `batch_size: "batch"`).
   - **Opset**: Version 13.
   - **Names**: Input=`input`, Output=`output`.
4. **Verification**:
   - runs PyTorch inference.
   - runs ONNX Runtime inference.
   - Asserts `np.allclose(torch_out, onnx_out, atol=1e-5)`.

**Returns**: Path to the generated `.onnx` file.

## Usage

```bash
# via Makefile (RECOMMENDED)
make export_model checkpoint_path="data/checkpoints/best.pth"

# Direct
python -m modelling.export --checkpoint_path "..."
```

## Related Documentation

**Depends On**:

- [[model_py|model.py]] - `load_model`

**Produces**:

- Artifacts used by [[../api/main_py|main.py]] (FastAPI).
