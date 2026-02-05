---
title: index.html
date: 2026-01-28
lastmod: 2026-02-05
src_hash: 457bc0e60d303240264b57fa75f6d555140d57bf5f7cbb187efa3e40aac3083f
aliases: ["Main Application Layout", "SPA Structure"]
---

# index.html

#source #frontend #html #ui #skeleton

**File Path**: `static/index.html`

**Purpose**: Single Page Application (SPA) structure for the Sign Language Recognition interface.

## Structure Overview

The layout is built using semantic HTML5 and CSS Grid/Flexbox.

```html
<body>
  <main class="app-container">
    <header>...</header>     <!-- Status, Settings, Archive -->
    
    <aside>...</aside>      <!-- History Sidebar -->
    
    <div class="viewport">  <!-- Main Content Area -->
       <video>              <!-- Raw Camera Stream (Hidden) -->
       <canvas id="overlay-canvas"> <!-- Drawing layer for skeletons -->
       <canvas id="process-canvas"> <!-- Buffer for frame extraction (Hidden) -->
       <div class="hud">    <!-- Overlay (Prediction Card) -->
       <div class="sentence-bar"> <!-- Bottom Output -->
    </div>
    
    <div class="modals">...</div> <!-- Settings/Archive Overlays -->
  </main>
</body>
```

## Key Elements

### Video Processing & Visualization
- **`<video id="webcam">`**: Source stream used for processing.
- **`<canvas id="overlay-canvas">`**: Visible layer where skeletal landmarks and connections are rendered.
- **`<canvas id="process-canvas">`**: Hidden buffer for scaling and JPEG compression.

### Heads-Up Display (HUD)
- **`#prediction-text-ar`**: Primary Arabic display for the detected sign.
- **`#confidence-bar`**: Dynamic progress-fill displaying inference confidence.
- **`#status-pill`**: Connection state indicator (Live/Offline/Error).
- **`#history-toggle`**: Icon button to flip open the session log.

### Settings & Controls
- **Visualization Chips**: Buttons for toggling Face/Pose/Hands visibility.
- **Draw Style Toggles**: Checkboxes for Skeleton (Lines) and Keypoints (Dots).
- **Voice Selection**: Language dropdown for Text-to-Speech engine.

## External Resources
- **CSS**: `/static/styles.css`
- **JS**: `/static/live-signs.js` (Module type)
- **Fonts**: Google Fonts (Inter)

## Related Documentation

**Styled By**:
- [[../frontend/styles_css|styles.css]]

**Scripted By**:
- [[../frontend/live_signs_js|live_signs.js]]
