import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from mediapipe_utils import init_mediapipe_worker
from prepare_kps import extract_frame_keypoints
from utils import MAX_WORKERS


class FrameBuffer:
    def __init__(self, max_size):
        self._frames: dict[int, np.ndarray] = {}
        self._max_size = max_size
        self._latest_idx = -1

    def add_frame(self, frame):
        self._latest_idx += 1
        self._frames[self._latest_idx] = frame
        if len(self._frames) > self._max_size:
            del self._frames[self.oldest_idx]

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


async def producer_handler(websocket, buffer):
    try:
        while True:
            data = await websocket.receive_bytes()
            # Optimization: Decode frame in a separate thread
            frame = await asyncio.to_thread(
                cv2.imdecode,
                np.frombuffer(data, np.uint8),  # type: ignore
                cv2.IMREAD_COLOR,  # type: ignore
            )
            if frame is not None:
                buffer.add_frame(frame)
    except Exception as e:
        # WebSocketDisconnect is often raised as a normal exception here depending on implementation
        # We'll just log and let the consumer handle the shutdown
        print(f"Producer stopped: {e}")


keypoints_detection_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS, initializer=init_mediapipe_worker, initargs=(True,)
)


async def get_frame_kps(frame):
    return await asyncio.get_event_loop().run_in_executor(
        keypoints_detection_executor, extract_frame_keypoints, frame, True
    )
