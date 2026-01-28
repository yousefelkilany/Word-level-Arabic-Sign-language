# Web Interface Design

#frontend #ui #ux #html #css

The frontend provides a modern, responsive user interface for real-time sign language recognition, built with **HTML5, CSS3, and Vanilla JavaScript**.

## Design Philosophy

- **Clean & Minimalist**: Focuses on the video feed and the predicted result.
- **Responsive**: Adapts to different screen sizes (desktop/mobile).
- **Feedback-Driven**: Provides visual cues for socket connection, confidence levels, and motion stability.

## Layout Structure

The application is structured into three main layers:

### 1. Viewport Layer
The background element that hosts the video and recognition overlay.
- `<video id="webcam">`: Displays the raw camera feed.
- `<canvas id="process-canvas">`: (Hidden) used for capturing frames to send to the server.
- `.hud-layer`: Transparent overlay containing the prediction card.

### 2. HUD (Heads-Up Display)
Contains floating elements that provide real-time information.
- **Prediction Card**: Displays the current sign (Arabic/English) and a confidence bar.
- **Sentence Bar**: Accumulates recognized signs into a sentence.
- **Controls**: Buttons for Text-to-Speech (TTS) and clearing the sentence.

### 3. Modals & Overlays
- **Settings**: Theme selection and TTS language.
- **History**: Side panel showing a log of recent predictions.
- **Archives**: Modal for viewing past sessions.

## Styling System

We use **CSS Variables** for easy theming (Light/Dark mode).

```css
:root {
    --primary-color: #2563eb;
    --bg-color: #f8fafc;
    --text-color: #1e293b;
    /* ... */
}

[data-theme="dark"] {
    --bg-color: #0f172a;
    --text-color: #f1f5f9;
}
```

## Related Documentation

- [[../source/frontend/index_html|index.html Source Code]]
- [[../source/frontend/styles_css|styles.css Source Code]]
- [[../frontend/websocket_client_implementation|WebSocket Client]]
