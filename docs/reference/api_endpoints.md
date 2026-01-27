# API Endpoints

#api #reference #http #websocket

The application exposes a minimal set of endpoints, primarily handling static assets and a single WebSocket channel.

## HTTP Endpoints

### Root
- **Method**: `GET`
- **URL**: `/`
- **Description**: Returns the main application interface (`index.html`).
- **Response**: `200 OK` (HTML)

### Static Files
- **Method**: `GET`
- **URL**: `/static/{file_path}`
- **Description**: Serves frontend assets (JS, CSS, Images).
- **Directory**: `static/`
- **Response**: `200 OK` or `404 Not Found`

### Health Check (Implicit)
- **Method**: `GET`
- **URL**: `/`
- **Process**: If the root loads, the server is running.

## WebSocket Endpoints

### Live Signs
- **URL**: `/live-signs`
- **Protocol**: `ws://` or `wss://`
- **Description**: Bidirectional channel for real-time inference.
- **Workflow**:
    1.  Client connects.
    2.  Client sends binary frames.
    3.  Server responds with JSON predictions.

## Related Documentation

- [[../api/websocket_communication|WebSocket Protocol]]
- [[../source/api/main_py|main.py Source]]
