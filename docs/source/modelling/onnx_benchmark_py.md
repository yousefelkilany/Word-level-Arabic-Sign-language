---
title: onnx_benchmark.py
date: 2026-01-28
lastmod: 2026-02-01
src_hash: 52460f7f6f19919c66842b5a66cb30ca88240ec9a405d53c6c6f250168f5ccf4
aliases: ["Performance Profiling", "Inference Speed Benchmark"]
---

# onnx_benchmark.py

#source #modelling #profiling #onnx

**File Path**: `src/modelling/onnx_benchmark.py`

**Purpose**: Evaluates the performance of exported ONNX models on the test dataset.

## Overview

This script loads an ONNX model and runs inference on the test split of the KArSL-502 dataset to calculate classification metrics. It uses the `onnxruntime` for execution.

## Metrics Calculated

- **Accuracy**: Overall classification accuracy.
- **Weighted F1 Score**: F1 score accounting for class imbalance.

## Workflow

1. **CLI Parsing**: Accepts `--onnx_model_path`, `--num_signs`, and `--model_metadata`.
2. **Metadata Discovery**: Automatically attempts to extract `num_signs` and `model_size` from the filename if not provided.
3. **Data Loading**: Prepares a lazy dataloader for the test split (`SplitType.test`).
4. **Inference Loop**:
   - Converts PyTorch tensors to NumPy for ONNX.
   - Runs `onnx_inference` via `onnxruntime`.
   - Aggregates predictions and true labels.
5. **Evaluation**: Uses `sklearn.metrics` to compute the final scores.

## Usage

```bash
# Example command
python src/modelling/onnx_benchmark.py --onnx_model_path models/checkpoint_signs_502_s_4_4.pth.onnx
```

## Related Documentation

- [[model_py|model.py]] - `load_onnx_model`, `onnx_inference` logic.
- [[../core/utils_py|utils.py]] - `extract_metadata_from_checkpoint`.
- [[../data/dataloader_py|dataloader.py]] - `prepare_dataloader`.
