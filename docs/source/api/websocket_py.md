---
title: websocket.py
date: 2026-01-28
lastmod: 2026-02-05
src_hash: 96bf9b4bf269a6795d7cf00705b9f12e27fc9d2262cacf41e03a72d902e7bdf4
aliases: ["WebSocket Router", "Handler Orchestration"]
---

# websocket.py

#source #api #websocket #orchestration

**File Path**: `src/api/websocket.py`

**Purpose**: WebSocket router and task orchestrator for live sign detection sessions.

## Overview

This module defines the `/live-signs` WebSocket endpoint. It uses a producer-consumer pattern to handle real-time video streams. For each connection, it:
1.  Creates an `asyncio.Queue` to buffer decoded frames.
2.  Spawns a `producer_handler` task to receive and decode frames.
3.  Spawns a `consumer_handler` task to process frames and send predictions.
4.  Monitors both tasks and ensures clean resource teardown on disconnect.

## Routes

### `ws_live_signs(websocket)`
**Route**: `WS /live-signs`
**Description**: Main entry point for live sign detection.
- **Workflow**:
    - Accepts connection.
    - Initializes a shared `asyncio.Queue(maxsize=MAX_SIGN_FRAMES)`.
    - Runs `producer_handler` and `consumer_handler` concurrently using `asyncio.create_task`.
    - Waits for either task to complete (usually via disconnect or error).
    - **Cleanup**: Cancels pending tasks, closes MediaPipe models, and triggers garbage collection.

## Relationships

- **Depends On**:
    - [[live_processing_py|live_processing.py]] - For the actual handler logic and `MAX_SIGN_FRAMES`.
    - [[../core/utils_py|utils.py]] - For logging.
- **Used By**:
    - [[main_py|main.py]] - Included in the main FastAPI application router.

## Implementation Details

The core processing logic formerly in this module has been moved to [[live_processing_py|live_processing.py]] to improve modularity and testability. This file now focuses strictly on WebSocket lifecycle management.
