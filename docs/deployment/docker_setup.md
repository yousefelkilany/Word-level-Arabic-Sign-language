---
title: Docker Setup
date: 2026-01-28
lastmod: 2026-01-28
---

# Docker Setup

#deployment #docker #containers

The project is fully containerized to ensure consistent behavior across development and production environments. We use **Docker** for the application runtime and **Docker Compose** for orchestration.

## Dockerfile Overview

The `Dockerfile` uses a multi-stage build to minimize image size.

1.  **Builder Stage**:
    - Base: `python:3.12-slim-bookworm`
    - Tooling: Installs `uv` for fast package management.
    - Action: Compiles dependencies into a virtual environment in `/opt/venv`.

2.  **Runtime Stage**:
    - Base: `python:3.12-slim-bookworm`
    - System Deps: Installs runtime libraries like `libgl1` (for OpenCV).
    - Setup: Copies the pre-built venv, creates a non-root user `appuser`, and sets up the entrypoint.

## Docker Compose

The `docker-compose.yml` orchestrates the services:

- **live-app**: The main FastAPI application.
    - Ports: Maps internal `8000` to host `8000`.
    - Volumes: Mounts `src`, `data`, `static` as read-only for hot-reloading (in dev).
    - Environment: Loads from `.env`.
- **docs**: A local documentation server (Perlite) for viewing this Obsidian vault.
- **nginx**: Reverse proxy for the documentation.

## Commands

### Build and Run
```bash
docker compose up --build
```
This will start:
- Web App at `http://localhost:8000`
- Documentation at `http://localhost:8080` (if enabled)

### Stop
```bash
docker compose down
```

## Related Documentation

- [[../source/config/dockerfile|Dockerfile Source]]
- [[../source/config/docker_compose_yml|docker-compose.yml Source]]
- [[../deployment/environment_configuration|Environment Configuration]]
