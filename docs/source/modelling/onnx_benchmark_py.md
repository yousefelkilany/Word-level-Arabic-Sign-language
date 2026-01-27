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
