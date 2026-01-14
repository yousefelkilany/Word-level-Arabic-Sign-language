import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        # workers=2, # FIXME: enable in production
        reload=True,  # FIXME: disable in production
        reload_excludes=["/app"],
        reload_includes=["/app/src"],
    )
