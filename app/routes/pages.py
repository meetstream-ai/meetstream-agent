"""HTTP routes: health check and optional static demo page."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {
        "status": "ok",
        "service": "meetstream-bridge",
        "docs": "https://docs.meetstream.ai/guides/get-started/bridge-server-architecture",
    }
