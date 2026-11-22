"""
WebSocket endpoints: MeetStream bridge channels + optional browser debug UI.

MeetStream receive loops: ``app.meetstream.ws_handlers``. Orchestration: ``app.realtime.pipeline``.

The bridge sockets carry a per-session token in the path
(``app.meetstream.session_token``); the debug UI is off unless
``DEBUG_UI_ENABLED=1``. Both refuse before the WebSocket is accepted.
"""

from __future__ import annotations

import json
import logging
import os
import struct

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.meetstream import session_token
from app.meetstream.outbound import safe_send_json
from app.meetstream.ws_handlers import CLOSE_POLICY_VIOLATION, audio_ingest_loop, control_channel_loop
from app.realtime.pipeline import bridge

logger = logging.getLogger("bridge.routes.ws")

router = APIRouter()


def debug_ui_enabled() -> bool:
    """The browser debug UI talks to any bot's session with no login: local use only."""
    return os.getenv("DEBUG_UI_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")


@router.websocket("/ws/{session_id}")
async def debug_ui_socket(websocket: WebSocket, session_id: str) -> None:
    """Optional local UI: mirror events and send mic audio for testing."""
    if not debug_ui_enabled():
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return
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


# Old URLs without a token: registered first so they never reach the token
# routes below ("/bridge/audio" would otherwise match "/bridge/{token}").
@router.websocket("/bridge")
@router.websocket("/bridge/audio")
@router.websocket("/{bot_id}/bridge")
@router.websocket("/{bot_id}/bridge/audio")
async def meetstream_without_token(websocket: WebSocket, bot_id: str | None = None) -> None:
    logger.warning(
        "Refused a MeetStream socket on %s: the bridge needs a session token in the URL "
        "(wss://<host>/bridge/<token>). Make one with: uv run python -m app.meetstream.session_token",
        websocket.url.path,
    )
    await websocket.close(code=CLOSE_POLICY_VIOLATION)


def _authorized(token: str, path_bot_id: str | None):
    """
    None when the socket must be refused; otherwise the check for the bot
    named in ``ready``: the token's bot, and the bot in the path if any.
    """
    secret = session_token.bridge_secret()
    if secret is None:
        logger.error(
            "Refused a MeetStream socket: BRIDGE_SECRET is not set (or shorter than %d characters).",
            session_token.MIN_SECRET_LENGTH,
        )
        return None
    if not session_token.verify(token, secret):
        logger.warning("Refused a MeetStream socket: the session token is invalid or expired.")
        return None

    def allow_bot(bot_id: str) -> bool:
        if path_bot_id is not None and bot_id != path_bot_id:
            return False
        if not session_token.bindings.bind(token, bot_id):
            logger.warning("Refused bot %s: its session token belongs to another bot.", bot_id)
            return False
        return True

    return allow_bot


@router.websocket("/bridge/{token}")
async def meetstream_control(websocket: WebSocket, token: str) -> None:
    allow_bot = _authorized(token, None)
    if allow_bot is None:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return
    await control_channel_loop(websocket, bridge, allow_bot)


@router.websocket("/bridge/audio/{token}")
async def meetstream_audio(websocket: WebSocket, token: str) -> None:
    allow_bot = _authorized(token, None)
    if allow_bot is None:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return
    await audio_ingest_loop(websocket, bridge, allow_bot)


# MeetStream Create Bot API often uses websocket_url templates like
# wss://host/{bot_id}/bridge/<token> — same handshake; the bot in the path
# must also be the bot in ``ready``.
@router.websocket("/{bot_id}/bridge/{token}")
async def meetstream_control_with_bot_id(websocket: WebSocket, bot_id: str, token: str) -> None:
    allow_bot = _authorized(token, bot_id)
    if allow_bot is None:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return
    await control_channel_loop(websocket, bridge, allow_bot)


@router.websocket("/{bot_id}/bridge/audio/{token}")
async def meetstream_audio_with_bot_id(websocket: WebSocket, bot_id: str, token: str) -> None:
    allow_bot = _authorized(token, bot_id)
    if allow_bot is None:
        await websocket.close(code=CLOSE_POLICY_VIOLATION)
        return
    await audio_ingest_loop(websocket, bridge, allow_bot)
