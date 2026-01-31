---
title: ONNX Export Process
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Model Conversion", "ONNX Deployment Guide"]
---

# ONNX Export Process

#model #onnx #deployment

To verify efficient inference on CPU-based edge devices or web backends, we export the trained PyTorch model to **ONNX (Open Neural Network Exchange)** format.

## Why ONNX?
- **Interoperability**: Can be run in many environments (Python, C++, JS).
- **Optimization**: ONNX Runtime performs graph optimizations (fusion, constant folding) specific to the hardware.
- **Speed**: Significant inference speedup on CPU compared to PyTorch.

## Export Workflow

The script `src/modelling/export.py` handles the conversion:

1.  **Load PyTorch Model**: Loads the `.pth` checkpoint.
2.  **Validation**: Runs a dummy inference in PyTorch to get baseline outputs.
3.  **Export**: Uses `torch.onnx.export` to trace the graph.
    - **Dynamic Axes**: We configure the batch size to be dynamic (`{0: 'batch_size'}`) so the exported model can handle any batch size.
4.  **Verification**:
    - **Checker**: Runs `onnx.checker.check_model` to validate the schema.
    - **Numerical check**: Runs the exported ONNX model using `onnxruntime` and compares the output with the PyTorch baseline using `torch.testing.assert_close`.

## Usage
```bash
python -m modelling.export --checkpoint_path checkpoints/best_model.pth
```

## Related Documentation

- [[../source/modelling/export_py|export.py Source Code]]
- [[../source/api/main_py|main.py (ONNX Loading)]]
