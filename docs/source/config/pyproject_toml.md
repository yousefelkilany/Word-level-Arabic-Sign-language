---
title: pyproject.toml
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Project Metadata", "Dependency Management"]
---

# pyproject.toml

#source #config #python

**File Path**: `pyproject.toml`

**Purpose**: Defines project metadata, dependencies, and build system configuration (PEP 518).

## Project Metadata
- **Name**: `arabic-sign-language-recognition`
- **Python**: `>=3.12`
- **Dependencies**:
  - Web: `fastapi`, `uvicorn`
  - ML: `numpy`, `opencv-python-headless`, `mediapipe`
  - ONNX: `onnx`, `onnxruntime`, `onnxscript`

## Optional Dependencies (`extras`)
- **`torch-cpu`**: Installs CPU-only PyTorch.
- **`torch-cuda`**: Installs CUDA-enabled PyTorch.

## Development Group (`dev`)
- Data Analysis: `pandas`, `scikit-learn`
- Visualization: `seaborn`, `plotly`, `streamlit`
- Utils: `tqdm`, `openpyxl`

## Tool Configuration (`[tool.uv]`)
Configures `uv` (the package manager) to handle the complex PyTorch CPU/CUDA split using mutually exclusive index URLs.

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "torch-cpu" },
    { index = "pytorch-cu126", marker = "sys_platform != 'darwin'", extra = "torch-cuda" },
]
```

## Related Documentation

**Consumed By**:
- [[../../source/config/dockerfile|Dockerfile]] - Install step
