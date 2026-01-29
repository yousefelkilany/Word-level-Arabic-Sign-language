---
title: Troubleshooting
date: 2026-01-28
lastmod: 2026-01-28
---

# Troubleshooting

#reference #debug #troubleshooting

Common issues and solutions when running the application.

## Connection Issues

### "Connecting..." stuck forever
- **Cause**: The WebSocket connection failed to establish.
- **Solution**:
    1.  Check if the server is running (`docker compose ps`).
    2.  Check the browser console (`F12`) for "Connection Refused".
    3.  Verify `DOMAIN_NAME` in `.env` matches the URL you are accessing.

### "Camera access denied"
- **Cause**: Browser permission blocked.
- **Solution**:
    1.  Click the lock icon in the address bar.
    2.  Allow "Camera" permissions.
    3.  Reload the page.

## Inference Issues

### High Latency / Lag
- **Cause**: CPU bottleneck or slow network.
- **Solution**:
    - Ensure `use_gpu` is set appropriately in `constants.py` or `.env` (currently CPU focused).
    - Lower the `CONFIG.fps` in `live-signs.js`.

### Poor Accuracy
- **Cause**: Bad lighting or occlusion.
- **Solution**:
    - Ensure your hands and face are clearly visible.
    - Check the "Skeleton" visualization (if enabled via debug flag) to see what the model "sees".

## Build Failures

### "uv not found"
- **Cause**: Docker build step failed to grab `uv`.
- **Solution**: Ensure internet access for the container builder or verify the `COPY --from` instruction in Dockerfile.

## Related Documentation

- [[../deployment/docker_setup|Docker Setup]]
- [[../api/websocket_communication|WebSocket Protocol]]
