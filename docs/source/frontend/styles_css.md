---
title: styles.css
date: 2026-01-28
lastmod: 2026-01-28
---

# styles.css

#source #frontend #css

Contains all styles for the application, implementing a modern, dark-mode supported design.

## Theming
Uses CSS variables (`:root` vs `[data-theme="dark"]`) for color schemes.

- `--bg-color`: Background.
- `--text-color`: Primary text.
- `--accent-color`: Blue accent (`#3b82f6`).
- `--glass-bg`: Translucent background for HUD elements.

## Layouts
- **Flexbox**: Used extensively for alignment (`.app-container`, `.top-bar`, `.hud-layer`).
- **Responsive**: `max-width`, `clamp()`, and relative units ensure mobile compatibility.

## Animations
- `fadeIn`: Keyframe animation for new history items.
- Transitions on buttons and theme toggles.

## Related Code

- [[../../frontend/web_interface_design|Web Interface Design]]
