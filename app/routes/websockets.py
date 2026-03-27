"""
WebSocket endpoints: MeetStream bridge channels + optional browser debug UI.

MeetStream receive loops: ``app.meetstream.ws_handlers``. Orchestration: ``app.realtime.pipeline``.
"""

from __future__ import annotations

import json
import logging
import struct

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.meetstream.outbound import safe_send_json
from app.meetstream.ws_handlers import audio_ingest_loop, control_channel_loop
from app.realtime.pipeline import bridge

logger = logging.getLogger("bridge.routes.ws")

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def debug_ui_socket(websocket: WebSocket, session_id: str) -> None:
    """Optional local UI: mirror events and send mic audio for testing."""
    await websocket.accept()

    q = websocket.scope.get("query_string", b"").decode() if websocket.scope else ""
    bot_id = None
    if q:
        try:
            for kv in q.split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k == "bot_id":
                        bot_id = v
                        break
        except Exception:
            pass
    if not bot_id:
        bot_id = session_id

    await bridge.ensure_session(bot_id)
    await bridge.attach_ui(session_id, websocket, bot_id)

    await safe_send_json(websocket, {"type": "ack", "message": f"UI bound {session_id} → {bot_id}"})

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "audio":
                int16 = data.get("data") or []
                if int16:
                    pcm = struct.pack(f"{len(int16)}h", *int16)
                    await bridge.ensure_session(bot_id)
                    try:
                        await bridge.sessions[bot_id].send_audio(pcm)
                    except Exception:
                        await bridge.close_session(bot_id)
                        await bridge.ensure_session(bot_id)
                        await bridge.sessions[bot_id].send_audio(pcm)

            elif data.get("type") == "usermsg":
                msg = data.get("message")
                if msg:
                    await bridge.ingest_user_text(bot_id, msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("UI socket error: %s", e)
    finally:
        await bridge.detach_ui(session_id)


@router.websocket("/bridge")
async def meetstream_control(websocket: WebSocket) -> None:
    await control_channel_loop(websocket, bridge)


@router.websocket("/bridge/audio")
async def meetstream_audio(websocket: WebSocket) -> None:
    await audio_ingest_loop(websocket, bridge)


# MeetStream Create Bot API often uses websocket_url templates like
# wss://host/{bot_id}/bridge — same handshake, path includes bot id for routing.
@router.websocket("/{bot_id}/bridge")
async def meetstream_control_with_bot_id(websocket: WebSocket, bot_id: str) -> None:
    _ = bot_id  # identity still comes from JSON ``ready``; path must exist for routing
    await control_channel_loop(websocket, bridge)


@router.websocket("/{bot_id}/bridge/audio")
async def meetstream_audio_with_bot_id(websocket: WebSocket, bot_id: str) -> None:
    _ = bot_id
    await audio_ingest_loop(websocket, bridge)
