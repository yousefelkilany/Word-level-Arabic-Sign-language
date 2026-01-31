import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from core.constants import MAX_WORKERS
from core.mediapipe_utils import LandmarkerProcessor
from core.utils import get_default_logger


async def producer_handler(websocket, buffer: asyncio.Queue):
    logger = get_default_logger()
    logger.info(f"Producer started at {time.time()}")
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = await asyncio.to_thread(
                cv2.imdecode,
                np.frombuffer(data, np.uint8),  # type: ignore
                cv2.IMREAD_COLOR,  # type: ignore
            )
            if frame is not None:
                if buffer.full():
                    try:
                        buffer.get_nowait()
                    except asyncio.QueueEmpty:
                        ...

                await buffer.put(frame)
            else:
                logger.error(f"new bad frame recieved at {time.time()}")

    except Exception as e:
        logger.error(f"Producer stopped: {e}")

    finally:
        await buffer.put(None)
        logger.info(f"Producer stopped normally at {time.time()}")


keypoints_detection_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


async def get_frame_kps(
    mp_processor: LandmarkerProcessor, frame: np.ndarray, timestamp_ms=-1
):
    return await asyncio.get_event_loop().run_in_executor(
        keypoints_detection_executor,
        mp_processor.extract_frame_keypoints,
        frame,
        timestamp_ms,
        True,
    )
