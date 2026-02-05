---
title: sync_quartz_configs.py
date: 2026-02-05
lastmod: 2026-02-01
src_hash: 7b500fd5604467eef6569ba062c9301e5abbfd5b825a140c2b665e2ac51d8694
aliases: ["Quartz Config Sync", "Internal Fork Sync"]
---

# sync_quartz_configs.py

#source #scripts #maintenance #quartz

**File Path**: `scripts/sync_quartz_configs.py`

**Purpose**: Synchronizes Quartz configuration files from a separate fork repository (`quartz-fork`) into the local `quartz-config` directory.

## Overview

This utility ensures that the documentation site's layout and configuration remain in sync with any upstream changes or customizations made in the dedicated Quartz fork. It is typically run as part of a pre-commit or maintenance routine.

## Execution Logic

1. **Source Discovery**: Locates the `quartz-fork` directory (assumed to be a sibling of the project root).
2. **File Copying**: Copies `quartz.config.ts` and `quartz.layout.ts`.
3. **Git Integration**: Automatically runs `git add` on the copied files to ensure they are staged for the next commit.

## Configuration

- `SYNC_FILES`: List of files to track.
- `SOURCE_DIR`: Path to the external fork.
- `DEST_DIR`: Path to the local configuration storage (`quartz-config/`).

## Usage

```bash
uv run scripts/sync_quartz_configs.py
```
