import asyncio
import os
from collections import defaultdict

import cv2
import fastapi
import uvicorn
from dotenv import dotenv_values
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from live_frame_processing import (
    FrameBuffer,
    get_frame_kps,
    process_motion,
    producer_handler,
)
from model import load_onnx_model, onnx_inference
from utils import AR_WORDS, MODELS_DIR, SEQ_LEN

env = dotenv_values()
ONNX_CHECKPOINT_FILENAME = (
    env.get("ONNX_CHECKPOINT_FILENAME") or "ONNX_CHECKPOINT_FILENAME"
)
onnx_checkpoint_path = os.path.join(MODELS_DIR, ONNX_CHECKPOINT_FILENAME)
model = load_onnx_model(onnx_checkpoint_path)


NUM_IDLE_FRAMES = 10
MIN_SIGN_FRAMES = 15
MAX_SIGN_FRAMES = SEQ_LEN
EXT_FRAME = ".jpg"

detection_buffers = defaultdict(list)
default_state = {"is_idle": True, "idle_frames_num": 0}
detection_state = defaultdict(lambda: default_state.copy())


origins = [env.get("DOMAIN_NAME") or "DOMAIN_NAME"]
app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_assets_dir = "./static"
app.mount("/static", StaticFiles(directory=static_assets_dir, html=True), name="static")


@app.websocket("/live-signs")
async def ws_live_signs(websocket: fastapi.WebSocket):
    client_id = websocket.client
    print(f"Connected client: {client_id}")
    await websocket.accept()

    buffer = FrameBuffer(MAX_SIGN_FRAMES)

    producer_task = asyncio.create_task(producer_handler(websocket, buffer))

    try:
        gray, prev_gray = None, None
        # Retrieve client-specific state
        client_buffer = detection_buffers[client_id]
        client_state = detection_state[client_id]

        current_proc_idx = 0

        while True:
            if producer_task.done():
                # If producer finished (connection closed), we stop too
                break

            # Wait for frames if we are ahead
            if buffer.latest_idx < current_proc_idx:
                await asyncio.sleep(0.001)
                continue

            # If we fell behind, jump to the oldest available frame
            oldest_avail = buffer.oldest_idx
            if current_proc_idx < oldest_avail:
                # print(f"Lagged behind! Jumping from {current_proc_idx} to {oldest_avail}")
                current_proc_idx = oldest_avail

            frame = buffer.get_frame(current_proc_idx)
            current_proc_idx += 1

            if frame is None:
                continue

            prev_gray = gray
            gray, motion_frame, _, motion_thresh, has_motion = await asyncio.to_thread(
                process_motion, frame, prev_gray
            )

            if motion_frame is not None:
                ok, img_buffer = await asyncio.to_thread(
                    cv2.imencode,
                    EXT_FRAME,  # type: ignore
                    motion_frame,  # type: ignore
                )
                if not ok:
                    continue

                # Only send if socket is open. Producer handles receive errors, we handle send errors.
                try:
                    await websocket.send_bytes(img_buffer.tobytes())
                except Exception:
                    break

            if not has_motion:
                client_state["idle_frames_num"] += 1
                if (
                    client_state["idle_frames_num"] >= NUM_IDLE_FRAMES
                    and not client_state["is_idle"]
                ):
                    client_state["is_idle"] = True
                    client_buffer.clear()
                continue

            client_state["is_idle"] = False
            client_state["idle_frames_num"] = 0

            try:
                kps = await get_frame_kps(frame)
                client_buffer.append(kps)
            except Exception as e:
                print(f"Error extracting keypoints: {e}")
                continue

            if len(client_buffer) < MIN_SIGN_FRAMES:
                continue

            if len(client_buffer) > MAX_SIGN_FRAMES:
                del client_buffer[:-MAX_SIGN_FRAMES]

            try:
                # Run inference in a separate thread, this will BLOCK the consumer loop until inference is done
                sid = await asyncio.to_thread(
                    onnx_inference, model, list(client_buffer)
                )

                # TODO: check when a sign is repeatedly detected, skip it
                sign_repeated = False
                if sign_repeated:
                    continue

                if sid is not None:
                    await websocket.send_json({"detected_word": AR_WORDS[sid]})

            except Exception as e:
                print(f"Error detecting sign: {e}")
                continue

    except fastapi.WebSocketDisconnect:
        print(f"Disconnected client (consumer): {client_id}")
    except Exception as e:
        print(f"Error (consumer): {e}")

    finally:
        producer_task.cancel()
        if client_id in detection_buffers:
            del detection_buffers[client_id]
        if client_id in detection_state:
            del detection_state[client_id]


@app.get("/")
@app.get("/live-signs")
async def live_signs_ui():
    return FileResponse(f"{static_assets_dir}/live-signs.html")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
