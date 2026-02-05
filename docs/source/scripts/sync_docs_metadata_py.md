---
title: sync_docs_metadata.py
date: 2026-02-05
lastmod: 2026-02-05
src_hash: 64c3a98068a9fd1a24b2406839036e937c405580087a4f9dbbca2f12870e9327
aliases: ["Documentation Sync Tool", "Metadata Automator"]
---

# sync_docs_metadata.py

#source #scripts #maintenance #automation

**File Path**: `scripts/sync_docs_metadata.py`

**Purpose**: Automates the maintenance of source-linked documentation by updating timestamps, verifying content hashes, and detecting orphaned files.

## Overview

This script is the core of the documentation CI/CD workflow. It ensures that every source file has a corresponding documentation file and that the documentation accurately reflects the latest version of the code.

## Key Features

### 1. Multi-Root Synchronization
Maps multiple source directories to their documentation counterparts:
- `src/` $\to$ `docs/source/`
- `scripts/` $\to$ `docs/source/scripts/`
- `static/` $\to$ `docs/source/frontend/`

### 2. Metadata Updates
Automatically updates the YAML frontmatter of `.md` files:
- `lastmod`: The last modification date of the source file.
- `src_hash`: A SHA256 hash of the source file. If the hash changes, it flags the file for content review.

### 3. Orphan Detection
Identifies "orphaned" documentation files—markdown files that exist in the docs folders but no longer have a corresponding source file in the project. It handles:
- Codebase moves/refactors.
- Deleted utility scripts.
- Cross-directory mappings (e.g., `config/`, `static/`, or `scripts/`).
- Shared documentation folders.

## Workflow Integration

This script is intended to be run via the `sync-docs` workflow:
```bash
uv run scripts/sync_docs_metadata.py
```

## Related Documentation

- [[../SOURCE_DOCS_GUIDE|SOURCE_DOCS_GUIDE]] - Instructions for maintaining docs.
- [[../../.agent/workflows/sync-docs|sync-docs workflow]] - The automation process.
