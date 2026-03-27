"""JSON commands sent to MeetStream on the control WebSocket."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger("bridge.meetstream.outbound")


async def safe_send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps(payload))
    except Exception as e:
        logger.warning("websocket send failed: %s", e)


async def ack_control_channel(ws: WebSocket, bot_id: str) -> None:
    await safe_send_json(
        ws,
        {
            "command": "ack",
            "bot_id": bot_id,
            "message": f"Control channel bound to {bot_id}",
        },
    )


async def ack_audio_channel(ws: WebSocket, bot_id: str) -> None:
    await safe_send_json(
        ws,
        {
            "type": "ack",
            "message": f"Audio channel bound to {bot_id}",
        },
    )


async def send_pcm_to_meeting(
    ws: WebSocket,
    bot_id: str,
    pcm_b64: str,
    *,
    sample_rate: int,
) -> None:
    await safe_send_json(
        ws,
        {
            "command": "sendaudio",
            "bot_id": bot_id,
            "audiochunk": pcm_b64,
            "sample_rate": sample_rate,
            "encoding": "pcm16",
            "channels": 1,
            "endianness": "little",
        },
    )


async def send_chat_to_meeting(ws: WebSocket, bot_id: str, text: str) -> None:
    await safe_send_json(
        ws,
        {
            "command": "sendmsg",
            "bot_id": bot_id,
            "message": text,
            "msg": text,
        },
    )


async def send_interrupt_clear_queue(ws: WebSocket, bot_id: str) -> None:
    await safe_send_json(
        ws,
        {
            "command": "interrupt",
            "bot_id": bot_id,
            "action": "clear_audio_queue",
        },
    )
