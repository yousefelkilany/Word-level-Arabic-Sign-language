# WebSocket Client Implementation

#frontend #websocket #javascript

The client-side logic in `live-signs.js` manages the camera feed, WebSocket communication, and UI updates.

## Core Logic

### 1. Camera Setup (`setupWebcam`)
- Uses `navigator.mediaDevices.getUserMedia` to access the webcam.
- Targets a resolution of 640x480 (ideal).
- Renders the stream to a `<video>` element.

### 2. Main Loop (`loop`)
A `requestAnimationFrame` loop controls the frame rate at which images are sent to the server (capped at `CONFIG.fps` = 30).
- Checks if the socket is open and not currently "sending" (to avoid backpressure).
- Draws the video frame to a canvas.

### 3. Frame Transmission (`processFrame`)
- Converts the canvas content to a **JPEG Blob**.
- Sends the binary blob over the WebSocket.

### 4. Message Handling (`socket.onmessage`)
Receives JSON responses from the server.
- **Events**:
  - `status`: Updates connection indicator (Live/Offline).
  - `prediction`: Updates the UI with the detected sign and confidence.

## Stability & Filtering
To prevent flickering predictions, the client implements a stability check (though primarily handled server-side now).
- **History Buffer**: Tracks the last `N` predictions.
- **Threshold**: Only updates the main display if a sign is consistent for sequential frames.

## Text-to-Speech (TTS)
Integrates the browser's `SpeechSynthesis` API to vocalize recognized sentences.
- Supports multiple Arabic/English dialects (configurable in settings).

## Related Documentation

- [[../source/frontend/live_signs_js|live-signs.js Source Code]]
- [[../api/websocket_communication|WebSocket Communication Protocol]]
