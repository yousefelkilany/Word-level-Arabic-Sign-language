from collections import defaultdict
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor

import fastapi

import numpy as np
import cv2

from model import load_onnx_model, onnx_inference
from prepare_kps import extract_frame_keypoints
from utils import AR_WORDS, detect_motion, init_mediapipe_worker


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


onnx_checkpoint_path = os.path.join(os.getcwd(), "model.onnx")
model = load_onnx_model(onnx_checkpoint_path)

NUM_IDLE_FRAMES = 10
MIN_SIGN_FRAMES = 15
MAX_SIGN_FARMES = 60

detection_buffers = defaultdict(list)
default_state = {"is_idle": True, "idle_frames_num": 0}
detection_state = defaultdict(lambda: default_state.copy())

keypoints_detection_executor = ProcessPoolExecutor(
    max_workers=os.cpu_count(), initializer=init_mediapipe_worker(True)
)

origins = ["localhost"]  # TODO: update with domain name
app = fastapi.FastAPI()
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# add route for websocket and server-side detection
@app.websocket("/live-signs")
async def websocket_endpoint(websocket: fastapi.WebSocket):
    client_id = websocket.client
    print(f"Connected client: {client_id}")
    await websocket.accept()

    try:
        prev_frame = None
        # h, w = None, None
        # gaussian_x, gaussian_y = None, None
        client_buffer = detection_buffers[client_id]
        client_state = detection_state[client_id]

        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if not frame:
                continue

            # if not prev_frame:  # first frame - initialization
            #     prev_frame = frame
            #     h, w = frame.shape
            #     gaussian_x, gaussian_y = get_gaussian_kernels(frame)
            #     continue

            has_changes = prev_frame and detect_motion(prev_frame, frame, 0.1)
            prev_frame = frame

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


@app.get("/")
async def root():
    return {"instructions": "go to /live-signs to start detection"}
