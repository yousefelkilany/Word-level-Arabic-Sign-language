# Dockerfile

#source #config #docker

**File Path**: `Dockerfile`

**Purpose**: Multi-stage build configuration for creating a lightweight production image.

## Build Stages

### Stage 1: `builder`

- **Base Image**: `python:3.12-slim-bookworm`
- **Tools**: Installs `uv` for ultra-fast dependency management.
- **Action**:
  1. Sets up virtual environment at `/opt/venv`.
  2. Syncs dependencies from `uv.lock` and `pyproject.toml`.
  3. Uses `--no-install-project` to keep layer small (code is mounted or copied later).

### Stage 2: `runtime`

- **Base Image**: `python:3.12-slim-bookworm`
- **System Deps**:
  - `libgl1`, `libglib2.0-0`: Required for OpenCV (cv2).
- **Setup**:
  1. Copies `/opt/venv` from builder.
  2. Sets environment variables (`PATH`, `PYTHONPATH`, OpenMP threads).
  3. Creates non-root user `appuser` (UID 1000).
- **Entrypoint**: `python src/api/run.py`

## Environment Variables

| Variable              | Value      | Purpose                              |
| :-------------------- | :--------- | :----------------------------------- |
| `UV_COMPILE_BYTECODE` | `1`        | Faster startup                       |
| `OMP_NUM_THREADS`     | `1`        | Prevents CPU contention in container |
| `PYTHONPATH`          | `/app/src` | Allows importing modules from src    |

## Usage

```bash
# Build
docker build -t arsl-app .

# Run
docker run -p 8000:8000 arsl-app
```

## Related Documentation

**Driven By**:

- [[../../source/config/pyproject_toml|pyproject.toml]] - Dependency definitions
- [[../../source/config/docker_compose_yml|docker-compose.yml]] - Orchestration
