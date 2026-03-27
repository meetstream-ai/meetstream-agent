"""CLI entry: run the MeetStream bridge (same as `uv run uvicorn app.server:app`)."""

import os


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("UVICORN_RELOAD", "1") == "1",
    )


if __name__ == "__main__":
    main()
