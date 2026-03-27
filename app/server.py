"""
FastAPI application: wires HTTP/WebSocket routes and static assets.

Architecture: https://docs.meetstream.ai/guides/get-started/bridge-server-architecture

Layout
------

``app/meetstream/`` — MeetStream protocol (audio frames, speaker filter, outbound JSON, WS loops).

``app/realtime/`` — OpenAI Realtime session, MCP preconnect, event serialization, model→MeetStream pump.

``app/routes/`` — Thin FastAPI routers (pages + websockets).

``app/agent.py`` — Agent definition, tools, MCP wiring (LLM-facing only).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import pages, websockets


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="MeetStream Bridge", lifespan=lifespan)
    application.include_router(pages.router)
    application.include_router(websockets.router)
    if pages.STATIC_DIR.is_dir():
        application.mount(
            "/static",
            StaticFiles(directory=str(pages.STATIC_DIR)),
            name="static",
        )
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=True)
