import asyncio
import gc

import fastapi

from api.live_processing import MAX_SIGN_FRAMES, consumer_handler, producer_handler
from core.utils import get_default_logger

logger = get_default_logger()
websocket_router = fastapi.APIRouter()


@websocket_router.websocket("/live-signs")
async def ws_live_signs(websocket: fastapi.WebSocket):
    await websocket.accept()
    client_id = websocket.client
    if not client_id:
        logger.error("Recieved client connection but couldn't get client id")
        return

    logger.info(f"Connected client: {client_id}")

    frame_buffer = asyncio.Queue(MAX_SIGN_FRAMES)
    producer_task = asyncio.create_task(
        producer_handler(client_id, websocket, frame_buffer)
    )
    consumer_task = asyncio.create_task(
        consumer_handler(client_id, websocket, frame_buffer)
    )

    try:
        done, pending = await asyncio.wait(
            [producer_task, consumer_task], return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            if task.exception():
                logger.error(f"Task failed: {task = }")

    finally:
        logger.info(f"Cleaning up resources for {client_id}")

        for task in [producer_task, consumer_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    ...
                except Exception as e:
                    logger.error(f"Error during task cleanup: {e}")
        gc.collect()
