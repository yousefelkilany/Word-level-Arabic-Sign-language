---
title: docker-compose.yml
date: 2026-01-28
lastmod: 2026-01-28
aliases: ["Docker Orchestration", "Service Configuration"]
---

# docker-compose.yml

#source #config #docker

**File Path**: `docker-compose.yml`

**Purpose**: Orchestrates the application services and documentation server.

## Services

### 1. `live-app`
The main FastAPI Application.
- **Build**: Context `.` (Uses [[dockerfile|Dockerfile]]).
- **Ports**: Maps `8000:8000`.
- **Volumes**: Mounts source directories (`src`, `data`, `static`) as Read-Only (`:ro`) for hot-reloading without container pollution.
- **Resources**:
  - Limit: 2.0 CPUs, 2GB RAM.
  - Reserve: 0.5 CPU, 512MB RAM.

### 2. `docs`
A `perlite` container serving this markdown documentation as a searchable website.
- **Image**: `sec77/perlite:latest`.
- **Environment**: Configures documentation parsing (WikiLinks enabled, HTML safe mode on).
- **Volumes**: Mounts `./docs` to `/var/www/perlite/arsl-docs`.

### 3. `nginx`
Reverse proxy (optional, dependent on setup).
- **Ports**: Maps `8080:80`.
- **Depends On**: `docs`.

## Usage

```bash
# Start all services
docker compose up -d

# Rebuild app
docker compose up -d --build live-app
```

## Related Documentation

**Depends On**:
- [[dockerfile|Dockerfile]]
- [[../../deployment/environment_configuration|.env]]
