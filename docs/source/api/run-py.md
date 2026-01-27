# source/api/run.py

#source-code #api #entry-point #uvicorn

**File Path**: `src/api/run.py`

**Purpose**: Application entry point for running the FastAPI server with Uvicorn.

## Overview

Simple entry point script that starts the Uvicorn ASGI server with development configuration.

## Implementation

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        # workers=2, # FIXME: enable in production
        reload=True,  # FIXME: disable in production
    )
```

## Configuration

| Parameter | Value            | Purpose                                |
| --------- | ---------------- | -------------------------------------- |
| `app`     | `"api.main:app"` | Import path to FastAPI app             |
| `host`    | `"0.0.0.0"`      | Listen on all interfaces               |
| `port`    | `8000`           | HTTP port                              |
| `reload`  | `True`           | Auto-reload on code changes (dev only) |
| `workers` | Commented out    | Multi-worker mode (production)         |

## Usage

```bash
# Direct execution
python src/api/run.py

# With uv
uv run src/api/run.py
```

## Production Considerations

For production deployment:
1. Set `reload=False`
2. Enable `workers=2` or more
3. Use process manager (systemd, supervisor)
4. Consider using Gunicorn with Uvicorn workers

## Related Documentation

- [[source/api/main-py|main.py]] - FastAPI application
- [[deployment/docker-setup|Docker Setup]] - Container deployment
- [[getting-started|Getting Started]]

---

**File Location**: `../../../src/api/run.py`
