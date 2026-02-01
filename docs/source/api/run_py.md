---
title: run.py
date: 2026-01-28
lastmod: 2026-01-14
aliases: ["Application Server Entry", "Uvicorn Runner"]
---

# run.py

#source #api #entrypoint

**File Path**: `src/api/run.py`

**Purpose**: Application entry point for development and production execution.

## Overview

This script initializes and runs the ASGI application using `uvicorn`. It serves as the primary command-line interface for starting the server.

### Execution Flow
1. Checks for `__main__` entry.
2. Invokes `uvicorn.run()` with configured parameters.
3. Reloads on code changes (development mode).

## Configuration

```python
uvicorn.run(
    "api.main:app",
    host="0.0.0.0",
    port=8000,
    # workers=2, # FIXME: enable in production
    reload=True,  # FIXME: disable in production
)
```

### Parameters

| Parameter | Value            | Description                                           |
| :-------- | :--------------- | :---------------------------------------------------- |
| `app`     | `"api.main:app"` | Import string for the FastAPI application instance    |
| `host`    | `"0.0.0.0"`      | Binds to all network interfaces (required for Docker) |
| `port`    | `8000`           | Listening port                                        |
| `reload`  | `True`           | Enables hot-reloading for development                 |

> [!WARNING]
> The current configuration is optimized for **development**. For production deployment:
> - Set `reload=False`
> - Enable `workers` (e.g., `workers=4`)
> - Consider using Gunicorn as a process manager.

## Usage

### Command Line
```bash
python src/api/run.py
```

### via Makefile
```bash
make run  # (Assuming standard makefile command)
```

## Related Documentation

**Calls**:
- [[main_py|api.main:app]] - The application instance being run

**Conceptual**:
- [[../../deployment/docker_setup|Docker Setup]] - Uses this script as entrypoint
