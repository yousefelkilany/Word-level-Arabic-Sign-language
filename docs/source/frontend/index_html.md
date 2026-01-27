# index.html

#source #frontend #html #ui

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
       <video>              <!-- Hidden Source -->
       <canvas>             <!-- Visible Feed -->
       <div class="hud">    <!-- Overlay (Prediction Card) -->
       <div class="sentence-bar"> <!-- Bottom Output -->
    </div>
    
    <div class="modals">...</div> <!-- Settings/Archive Overlays -->
  </main>
</body>
```

## Key Elements

### Video Processing
- **`<video id="webcam">`**: Raw stream source (hidden/muted).
- **`<canvas id="process-canvas">`**: Used for capturing frames to send to server.

### Heads-Up Display (HUD)
- **`#prediction-text-ar`**: Primary Arabic display.
- **`#confidence-bar`**: Visual confidence indicator.
- **`#status-pill`**: Connection state (Live/Offline/Error).

### Overlays
- **`#settings-overlay`**: Modal for Theme/Language.
- **`#archive-modal`**: Modal for viewing `localStorage` history.
- **`#history-sidebar`**: Slide-out remote drawer.

## External Resources
- **CSS**: `/static/styles.css`
- **JS**: `/static/live-signs.js` (Module type)
- **Fonts**: Google Fonts (Inter)

## Related Documentation

**Styled By**:
- [[source/frontend/styles_css|styles.css]]

**Scripted By**:
- [[source/frontend/live_signs_js|live_signs.js]]
