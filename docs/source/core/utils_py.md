---
title: utils.py
date: 2026-01-28
lastmod: 2026-02-05
src_hash: e8dbb5054e835c18d479b1b600ae0852c89c79603419ee7f454f12eea144f1b2
aliases: ["General Utilities", "Logger Configuration"]
---

# utils.py

#source #core #utilities

**File Path**: `src/core/utils.py`

**Purpose**: General-purpose utility functions for logging, file I/O, and string parsing.

## Overview

Provides shared helper functions used across the application for:

- Loading dataset labels (Arabic/English)
- Parsing metadata from checkpoint filenames (Class count + Model Architecture)
- Configuring the application logger
- Verifying file integrity (Git LFS check)

## Global Variables

### `AR_WORDS` & `EN_WORDS`

**Type**: `list[str]`
**Source**: Loaded from `KARSL-502_Labels.json`
**Purpose**: Global lists of sign labels in Arabic and English.
**Initialized By**: [[#init_signs|init_signs()]]

---

## Functions

### `init_signs()`

```python
def init_signs() -> None:
```

**Purpose**: Loads and parses the JSON labels file into the global `AR_WORDS` and `EN_WORDS` lists.

**Action**:
- Reads `KARSL-502_Labels.json`.
- Populates `AR_WORDS` and `EN_WORDS`.

**Usage**:
```python
init_signs() # Usually called at module level
```

### `extract_metadata_from_checkpoint(checkpoint_path)`

```python
def extract_metadata_from_checkpoint(checkpoint_path) -> Optional[tuple[int, ModelSize]]:
```

**Purpose**: Extracts both the number of classes and the model architecture (ModelSize) from a checkpoint filename.

**Regex**: `r".*?signs_(\d+)_(\w_\d_\d).*?"`

**Example**:
- **Input**: `checkpoint_signs_502_s_4_2.pth`
- **Output**: `(502, ModelSize(small, 4, 2))`

**Raises**: `ValueError` if the pattern is not found.

### `get_default_logger()`

```python
def get_default_logger() -> logging.Logger:
```

**Purpose**: Singleton-like accessor for the application logger.

**Configuration**:
- **Name**: `wl-ar-sl`
- **Output**: File (`logs/server_producer.log`)
- **Format**: `%(asctime)s - %(levelname)s - %(message)s`
- **Level**: `logging.DEBUG`

### `is_git_lfs_pointer(filepath)`

```python
def is_git_lfs_pointer(filepath) -> bool:
```

**Purpose**: Checks if a file is actually downloaded or just a Git LFS pointer.

**Logic**: Reads the first 50 bytes and checks for the LFS prefix `version https://git-lfs.github.com`.

---

## Related Documentation

**Depends On**:
- [[constants_py|constants.py]] - `LABELS_JSON_PATH`, `LOGS_DIR`, `ModelSize`

**Used By**:
- [[../modelling/model_py|model.py]] - Loading checkpoints and verifying models.
- [[../modelling/onnx_benchmark_py|onnx_benchmark.py]] - Parsing model paths.
- [[../modelling/dashboard/loader_py|loader.py]] - Metadata extraction for dashboard.
