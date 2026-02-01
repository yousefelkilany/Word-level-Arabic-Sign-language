---
title: onnx_benchmark.py
date: 2026-01-28
lastmod: 2026-01-29
aliases: ["Performance Profiling", "Inference Speed Benchmark"]
---

# onnx_benchmark.py

#source #modelling #profiling

Benchmarks the inference speed of the exported ONNX model.

## Metrics
- **Latency**: Average time per inference.
- **Throughput**: Inferences per second.
- **CPU vs GPU**: Compares execution providers if available.

## Usage
```bash
make onnx_benchmark
```
