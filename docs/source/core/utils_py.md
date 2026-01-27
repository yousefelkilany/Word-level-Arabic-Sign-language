# utils.py

#source #core #utilities

**File Path**: `src/core/utils.py`

**Purpose**: General-purpose utility functions for logging, file I/O, and string parsing.

## Overview

Provides shared helper functions used across the application for:
- Loading dataset labels (Arabic/English)
- Parsing metadata from checkpoint filenames
- Configuring the application logger

## Global Variables

### `AR_WORDS` & `EN_WORDS`
**Type**: `list[str]`
**Source**: Loaded from `KARSL-502_Labels.json`
**Purpose**: Global lists of sign labels in Arabic and English.
**Initialized By**: [[#init_words|init_words()]]

## Functions

### `init_words()`

```python
def init_words() -> tuple[list[str], list[str]]:
```

**Purpose**: Loads and parses the JSON labels file.
**Returns**:
- `Tuple`: `([Arabic Labels], [English Labels])`
**Usage**:
```python
AR_WORDS, EN_WORDS = init_words()
```

### `extract_num_words_from_checkpoint(checkpoint_path)`

```python
def extract_num_words_from_checkpoint(checkpoint_path) -> Optional[int]:
```

**Purpose**: Extracts the number of classes from a checkpoint filename using Regex.
**Regex**: `r".*?words_(\d+).*?"`
**Example**:
- Input: `checkpoint_words_502.pth`
- Output: `502`
**Raises**: `ValueError` if pattern not found.

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

## Related Documentation

**Depends On**:
- [[source/core/constants_py|constants.py]] - `LABELS_JSON_PATH`, `LOGS_DIR`

**Used By**:
- [[source/api/websocket_py|websocket.py]] - Logging connection events
- [[source/modelling/model_py|model.py]] - Loading checkpoints
