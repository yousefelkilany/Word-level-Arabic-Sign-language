import asyncio
import gc
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import fastapi
import numpy as np
import torch
from fastapi.datastructures import Address
from torch import nn

from api.cv2_utils import MotionDetector
from core.constants import MAX_WORKERS, SEQ_LEN
from core.mediapipe_utils import LandmarkerProcessor, reduced_mp_kps_idx_view
from core.utils import AR_WORDS, EN_WORDS, get_default_logger
from modelling.model import onnx_inference

logger = get_default_logger()

NUM_IDLE_FRAMES = 45
HISTORY_LEN = 5
HISTORY_THRESHOLD = 3
MIN_SIGN_FRAMES = 15
MAX_SIGN_FRAMES = SEQ_LEN
CONFIDENCE_THRESHOLD = 0.8
EXT_FRAME = ".jpg"


def get_default_state():
    return {
        "is_idle": False,
        "idle_frames_num": 0,
        "sign_history": deque(maxlen=5),
        "last_sent_sign": None,
    }


async def consumer_handler(
    client_id: Address, websocket: fastapi.WebSocket, buffer: asyncio.Queue
):
    client_info = f"{client_id.host}:{client_id.port}"
    client_buffer = []
    client_state = get_default_state()

    async def send_over_ws(response):
        try:
            await websocket.send_json(response)
        except Exception as e:
            logger.error(f"[Client {client_info}] Error sending response: {e}")

    motion_detector = MotionDetector()
    mp_processor = await LandmarkerProcessor.create_async(None, False)

    start_time = int(time.time())
    last_processed_ts = start_time
    prev_gray = np.array([])
    try:
        while True:
            frame = await buffer.get()
            if frame is None:
                break

            (draw_mode, frame) = frame
            if frame.size == 0:
                continue

            if prev_gray.size == 0:
                prev_gray = frame

            has_motion, gray = await asyncio.to_thread(
                motion_detector.detect, prev_gray, frame
            )
            prev_gray = gray

            if not has_motion:
                client_state["idle_frames_num"] += 1
                if (
                    client_state["idle_frames_num"] >= NUM_IDLE_FRAMES
                    and not client_state["is_idle"]
                ):
                    client_state["is_idle"] = True
                    client_buffer.clear()
                    client_state["sign_history"].clear()
                    client_state["last_sent_sign"] = None
                    await websocket.send_json({"status": "idle"})
                continue

            client_state["is_idle"] = False
            client_state["idle_frames_num"] = 0

            now_ms = int((time.time() - start_time) * 1000)
            if now_ms <= last_processed_ts:
                now_ms = last_processed_ts + 1
            last_processed_ts = now_ms

            mp_kps: np.ndarray
            adjusted_kps: np.ndarray
            try:
                adjusted_kps, mp_kps = await get_frame_kps(mp_processor, frame)
            except Exception as e:
                logger.error(f"[Client {client_info}] Error extracting keypoints: {e}")
                continue

            if draw_mode:
                await send_over_ws(
                    {"landmarks": mp_kps[reduced_mp_kps_idx_view].tolist()}
                )

            client_buffer.append(adjusted_kps)

            if len(client_buffer) < MIN_SIGN_FRAMES:
                continue

            if len(client_buffer) > MAX_SIGN_FRAMES:
                client_buffer = client_buffer[-MAX_SIGN_FRAMES:]

            try:
                input_kps = np.array(client_buffer, dtype=np.float32)
                input_kps = input_kps.reshape(1, input_kps.shape[0], -1)
                raw_outputs = await asyncio.to_thread(
                    onnx_inference, websocket.app.state.onnx_model, [input_kps]
                )
                if raw_outputs is not None:
                    raw_outputs = raw_outputs.flatten()
                    probs = nn.functional.softmax(torch.Tensor(raw_outputs), dim=0)
                    pred_idx = int(torch.argmax(probs).item())
                    confidence = probs[pred_idx].item()
                    if confidence > CONFIDENCE_THRESHOLD:
                        client_state["sign_history"].append(pred_idx)

                        most_common_sign, sign_count = Counter(
                            client_state["sign_history"]
                        ).most_common(1)[0]
                        if (
                            sign_count >= HISTORY_THRESHOLD
                            and most_common_sign != client_state["last_sent_sign"]
                        ):
                            client_state["last_sent_sign"] = most_common_sign
                            await send_over_ws(
                                {
                                    "detected_sign": {
                                        "sign_ar": AR_WORDS[pred_idx],
                                        "sign_en": EN_WORDS[pred_idx],
                                    },
                                    "confidence": confidence,
                                }
                            )

            except Exception as e:
                logger.error(f"[Client {client_info}] Error detecting sign: {e}")

    except fastapi.WebSocketDisconnect:
        logger.info(f"[Client {client_info}] Consumer disconnected")

    except asyncio.CancelledError:
        logger.info(f"[Client {client_info}] Consumer cancelled")
        raise

    except Exception as e:
        logger.error(f"[Client {client_info}] Consumer crashed with error: {e}")
        raise

    finally:
        logger.info(f"[Client {client_info}] Cleaning up consumer resources...")
        logger.info(f"[Client {client_info}] Consumer loop finished")
        client_buffer = None
        client_state = None
        mp_processor.close()
        gc.collect()


async def producer_handler(
    client_id: Address, websocket: fastapi.WebSocket, buffer: asyncio.Queue
):
    client_info = f"{client_id.host}:{client_id.port}"
    logger.info(f"[Client {client_info}] Producer started at {time.time()}")
    try:
        frame = None
        while True:
            data = await websocket.receive_bytes()
            draw_mode = data[0] == 1
            frame = await asyncio.to_thread(
                cv2.imdecode,
                np.frombuffer(data[1:], np.uint8),  # type: ignore
                cv2.IMREAD_COLOR_RGB,  # type: ignore
            )
            if frame is not None:
                if buffer.full():
                    try:
                        buffer.get_nowait()
                    except asyncio.QueueEmpty:
                        ...

                await buffer.put((draw_mode, frame))
            else:
                logger.error(
                    f"[Client {client_info}] new bad frame recieved at {time.time()}"
                )

    except fastapi.WebSocketDisconnect:
        logger.info(f"[Client {client_info}] producer disconnected")

    except asyncio.CancelledError:
        logger.info(f"[Client {client_info}] Producer cancelled")
        raise

    except Exception as e:
        logger.error(f"[Client {client_info}] Producer crashed with error: {e}")
        raise

    finally:
        logger.info(f"[Client {client_info}] Cleaning up producer resources...")
        logger.info(f"[Client {client_info}] Producer loop finished")
        if frame is not None:
            del frame
        gc.collect()


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
        True,
    )
