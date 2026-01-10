import asyncio
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import fastapi
import numpy as np
import uvicorn
from dotenv import dotenv_values
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cv2_utils import detect_motion
from mediapipe_utils import init_mediapipe_worker
from model import load_onnx_model, onnx_inference
from prepare_kps import extract_frame_keypoints
from utils import AR_WORDS, MODELS_DIR, SEQ_LEN


async def get_frame_kps(frame):
    # do some preprocessing on frame if needed
    kps = await asyncio.get_event_loop().run_in_executor(
        keypoints_detection_executor, extract_frame_keypoints, frame, True
    )

    # do some preprocessing on kps if needed
    return kps


def run_inference(kps_buffer):
    # do some preprocessing on kps_buffer if needed
    return onnx_inference(model, kps_buffer)


env = dotenv_values()
ONNX_CHECKPOINT_FILENAME = (
    env.get("ONNX_CHECKPOINT_FILENAME") or "ONNX_CHECKPOINT_FILENAME"
)
onnx_checkpoint_path = os.path.join(MODELS_DIR, ONNX_CHECKPOINT_FILENAME)
model = load_onnx_model(onnx_checkpoint_path)

NUM_IDLE_FRAMES = 10
MIN_SIGN_FRAMES = 15
MAX_SIGN_FARMES = SEQ_LEN

detection_buffers = defaultdict(list)
default_state = {"is_idle": True, "idle_frames_num": 0}
detection_state = defaultdict(lambda: default_state.copy())

keypoints_detection_executor = ProcessPoolExecutor(
    max_workers=os.cpu_count(), initializer=init_mediapipe_worker(True)
)

origins = ["localhost"]  # TODO: update with domain name
app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=".", html=True), name="static")


@app.websocket("/live-signs")
async def ws_live_signs(websocket: fastapi.WebSocket):
    client_id = websocket.client
    print(f"Connected client: {client_id}")
    await websocket.accept()

    try:
        prev_gray = None
        # h, w = None, None
        # gaussian_x, gaussian_y = None, None
        client_buffer = detection_buffers[client_id]
        client_state = detection_state[client_id]

        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # if not prev_gray:  # first frame - initialization
            #     prev_gray = frame
            #     h, w = frame.shape
            #     gaussian_x, gaussian_y = get_gaussian_kernels(frame)
            #     continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_detected = detect_motion(prev_gray, gray, 0.1)
            has_changes = (prev_gray is not None) and motion_detected[2]
            prev_gray = gray

            if motion_detected[1] is not None:
                gray_3ch = np.tile(motion_detected[1][:, :, None], (1, 1, 3))
                frame = frame + gray_3ch
                await websocket.send_bytes(cv2.imencode(".jpg", frame)[1].tobytes())

            if not has_changes:
                client_state["idle_frames_num"] += 1
                if (
                    client_state["idle_frames_num"] >= NUM_IDLE_FRAMES
                    and not client_state["is_idle"]
                ):
                    client_state["is_idle"] = True
                    del detection_buffers[client_id]
                    detection_buffers[client_id] = []
                continue

            client_state["is_idle"] = False
            client_state["idle_frames_num"] = 0
            client_buffer.append(get_frame_kps(frame))
            if len(client_buffer) < MIN_SIGN_FRAMES:
                continue

            client_buffer = client_buffer[-MAX_SIGN_FARMES:]
            try:
                sid = run_inference(client_buffer)

                # TODO: check when a sign is repeatedly detected, skip it
                sign_repeated = False
                if sign_repeated:
                    continue
            except Exception as e:
                print(f"Error detecting sign: {e}")
                continue

            # await websocket.send_json({"new_detection": True, "word": word})
            await websocket.send_json({"detected_word": AR_WORDS[sid]})

    except fastapi.WebSocketDisconnect:
        del detection_buffers[client_id]
        del detection_state[client_id]
        print(f"Disconnected client: {client_id}")

    except Exception as e:
        print(f"Error: {e}")


@app.get("/live-signs")
async def live_signs_ui():
    return FileResponse("live-signs.html")


@app.get("/")
async def root():
    return {"instructions": "go to /live-signs to start detection"}


@app.websocket("/live-chat")
async def ws_live_chat(websocket: fastapi.WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
        except Exception as e:
            print(f"Error: {e}")
            break


@app.get("/live-chat")
async def live_chat_ui():
    return FileResponse("live-chat.html")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
