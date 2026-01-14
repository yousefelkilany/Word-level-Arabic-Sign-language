import asyncio
import gc
import time
from collections import Counter, deque

import fastapi
import numpy as np
import torch
from torch import nn

from api.cv2_utils import MotionDetector
from api.live_processing import FrameBuffer, get_frame_kps, producer_handler
from core.constants import SEQ_LEN
from core.mediapipe_utils import LandmarkerProcessor
from core.utils import AR_WORDS, EN_WORDS, get_default_logger
from modelling.model import onnx_inference

logger = get_default_logger()

NUM_IDLE_FRAMES = 15
HISTORY_LEN = 5
HISTORY_THRESHOLD = 4
MIN_SIGN_FRAMES = 15
MAX_SIGN_FRAMES = SEQ_LEN
CONFIDENCE_THRESHOLD = 0.4
EXT_FRAME = ".jpg"

websocket_router = fastapi.APIRouter()


def get_default_state():
    return {
        "is_idle": False,
        "idle_frames_num": 0,
        "sign_history": deque(maxlen=5),
        "last_sent_sign": None,
    }


@websocket_router.websocket("/live-signs")
async def ws_live_signs(websocket: fastapi.WebSocket):
    await websocket.accept()
    client_id = websocket.client
    logger.info(f"Connected client: {client_id}")

    client_buffer = []
    client_state = get_default_state()

    frame_buffer = FrameBuffer(MAX_SIGN_FRAMES)
    producer_task = asyncio.create_task(producer_handler(websocket, frame_buffer))

    motion_detector = MotionDetector()
    mp_processor: LandmarkerProcessor = await LandmarkerProcessor.create(True)

    start_time = int(time.time())
    last_processed_ts = start_time
    prev_gray = np.array([])
    current_proc_idx = 0

    try:
        while True:
            if producer_task.done():
                break

            if frame_buffer.latest_idx < current_proc_idx:
                await asyncio.sleep(0.001)
                continue

            oldest_avail = frame_buffer.oldest_idx
            if current_proc_idx < oldest_avail:
                current_proc_idx = oldest_avail

            frame = frame_buffer.get_frame(current_proc_idx)
            current_proc_idx += 1

            if frame is None or frame.size == 0:
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

            try:
                now_ms = int((time.time() - start_time) * 1000)
                if now_ms <= last_processed_ts:
                    now_ms = last_processed_ts + 1
                last_processed_ts = now_ms

                kps = await get_frame_kps(mp_processor, frame, now_ms)
                client_buffer.append(kps)
            except Exception as e:
                logger.error(f"Error extracting keypoints: {e}")
                continue

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

                    if len(client_state["sign_history"]) == HISTORY_LEN:
                        most_common_sign, sign_count = Counter(
                            client_state["sign_history"]
                        ).most_common(1)[0]
                        if (
                            sign_count >= HISTORY_THRESHOLD
                            and most_common_sign != client_state["last_sent_sign"]
                        ):
                            client_state["last_sent_sign"] = most_common_sign
                            await websocket.send_json(
                                {
                                    "detected_sign": {
                                        "sign_ar": AR_WORDS[pred_idx],
                                        "sign_en": EN_WORDS[pred_idx],
                                    },
                                    "confidence": confidence,
                                }
                            )

            except Exception as e:
                logger.error(f"Error detecting sign: {e}")
                continue

    except fastapi.WebSocketDisconnect:
        logger.info(f"Disconnected client (consumer): {client_id}")

    except Exception as e:
        logger.error(f"Error (consumer): {e}")

    finally:
        logger.info(f"Cleaning up resources for {client_id}")

        client_buffer = None
        client_state = None

        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            ...

        frame_buffer.clear()
        mp_processor.close()

        gc.collect()
