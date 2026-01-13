import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from mediapipe_utils import LandmarkerProcessor
from utils import MAX_WORKERS, get_default_logger


class FrameBuffer:
    def __init__(self, max_size):
        self._frames: dict[int, np.ndarray] = {}
        self._max_size = max_size
        self._latest_idx = -1
        self.logger = get_default_logger()

    def add_frame(self, frame):
        self._latest_idx += 1
        self._frames[self._latest_idx] = frame
        if len(self._frames) > self._max_size:
            oldest = self.oldest_idx
            if oldest != -1 and oldest in self._frames:
                del self._frames[oldest]

    def get_frame(self, idx) -> np.ndarray | None:
        return self._frames.get(idx)

    @property
    def latest_idx(self):
        return self._latest_idx

    @property
    def oldest_idx(self):
        if not self._frames:
            return -1
        return min(self._frames.keys())

    def clear(self):
        self._frames.clear()
        self._latest_idx = -1


async def producer_handler(websocket, buffer: FrameBuffer):
    buffer.logger.info(f"Producer: Started at {time.time()}")
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = await asyncio.to_thread(
                cv2.imdecode,
                np.frombuffer(data, np.uint8),  # type: ignore
                cv2.IMREAD_COLOR,  # type: ignore
            )
            if frame is not None:
                buffer.add_frame(frame)
            else:
                buffer.logger.error(f"new bad frame recieved at {time.time()}")

    except Exception as e:
        buffer.logger.error(f"Producer stopped: {e}")


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
