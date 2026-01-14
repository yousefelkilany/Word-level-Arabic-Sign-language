import os
from contextlib import asynccontextmanager

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.websocket import websocket_router
from core.constants import MODELS_DIR
from modelling.model import load_onnx_model


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("Loading model...")
    ONNX_CHECKPOINT_FILENAME = (
        os.environ.get("ONNX_CHECKPOINT_FILENAME") or "ONNX_CHECKPOINT_FILENAME"
    )
    onnx_checkpoint_path = os.path.join(MODELS_DIR, ONNX_CHECKPOINT_FILENAME)
    app.state.onnx_model = load_onnx_model(onnx_checkpoint_path)

    yield

    print("Shutting down...")
    del app.state.onnx_model


origins = [os.environ.get("DOMAIN_NAME") or "DOMAIN_NAME"]
app = fastapi.FastAPI(lifespan=lifespan)
static_assets_dir = "./static"
app.mount("/static", StaticFiles(directory=static_assets_dir, html=True), name="static")
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(websocket_router)


@app.get("/")
@app.get("/live-signs")
async def live_signs_ui():
    return FileResponse(os.path.join(static_assets_dir, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(
        path=os.path.join(static_assets_dir, "mediapipe-logo.ico"),
        headers={"Content-Disposition": "attachment; filename=favicon.ico"},
    )


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    return JSONResponse({})


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
