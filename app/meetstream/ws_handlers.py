"""
WebSocket receive loops for MeetStream `/bridge/audio` and `/bridge` (control).

These are thin: parse frames, apply speaker filter, forward PCM/text into ``RealtimeMeetingBridge``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable

from fastapi import WebSocket, WebSocketDisconnect

from app.meetstream.audio import decode_audio_frame, should_ignore_speaker
from app.meetstream.outbound import ack_audio_channel, ack_control_channel

if TYPE_CHECKING:
    from app.realtime.pipeline import RealtimeMeetingBridge

logger = logging.getLogger("bridge.meetstream.ws")


#: Policy violation: the socket's session token does not allow this bot.
CLOSE_POLICY_VIOLATION = 1008


async def control_channel_loop(
    ws: WebSocket, pipeline: RealtimeMeetingBridge, allow_bot: Callable[[str], bool] | None = None
) -> None:
    """
    JSON control socket: ``ready`` handshake, then ``usermsg`` / ``interrupt``.

    ``allow_bot`` decides whether the ``bot_id`` in ``ready`` may use this
    socket (see app.meetstream.session_token).
    """
    await ws.accept()
    bot_id: str | None = None
    try:
        init = json.loads(await ws.receive_text())
        if init.get("type") != "ready" or not init.get("bot_id"):
            await ws.close(code=1003)
            return
        if allow_bot is not None and not allow_bot(str(init["bot_id"])):
            await ws.close(code=CLOSE_POLICY_VIOLATION)
            return
        if not await pipeline.bind_control(init["bot_id"], ws):
            await ws.close(code=CLOSE_POLICY_VIOLATION)
            return
        bot_id = init["bot_id"]

        await ack_control_channel(ws, bot_id)

        while True:
            data = json.loads(await ws.receive_text())
            cmd = data.get("command")
            if cmd == "usermsg":
                msg = data.get("message", "")
                if msg:
                    await pipeline.ingest_user_text(bot_id, msg)
            elif cmd == "interrupt":
                await pipeline.interrupt_model(bot_id)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("[%s] Control endpoint error: %s", bot_id, e)
    finally:
        if bot_id:
            await pipeline.unbind_control(bot_id, ws)


async def audio_ingest_loop(
    ws: WebSocket, pipeline: RealtimeMeetingBridge, allow_bot: Callable[[str], bool] | None = None
) -> None:
    """Binary or legacy JSON PCM from MeetStream → model. ``allow_bot`` as for the control socket."""
    await ws.accept()
    bot_id: str | None = None
    try:
        while True:
            raw = await ws.receive()

            if "text" in raw and raw["text"]:
                data = json.loads(raw["text"])

                if data.get("type") == "ready":
                    if allow_bot is not None and data.get("bot_id") and not allow_bot(str(data["bot_id"])):
                        await ws.close(code=CLOSE_POLICY_VIOLATION)
                        return
                    if not data.get("bot_id"):
                        await ws.close(code=1003)
                        return
                    if not await pipeline.bind_audio_channel(data["bot_id"], ws):
                        await ws.close(code=CLOSE_POLICY_VIOLATION)
                        return
                    bot_id = data["bot_id"]
                    await ack_audio_channel(ws, bot_id)
                    continue

                if data.get("type") == "PCMChunk" and bot_id:
                    speaker = data.get("speakerName", "")
                    if should_ignore_speaker(speaker):
                        continue
                    b64 = data.get("audioData")
                    if b64:
                        await pipeline.ingest_meeting_audio_b64(bot_id, b64)
                continue

            if "bytes" in raw and raw["bytes"] and bot_id:
                result = decode_audio_frame(raw["bytes"])
                if result is None:
                    continue
                _sid, speaker_name, pcm_bytes = result
                if should_ignore_speaker(speaker_name):
                    continue
                if pcm_bytes:
                    await pipeline.ingest_meeting_audio_pcm(bot_id, pcm_bytes)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("[%s] Audio endpoint error: %s", bot_id, e)
    finally:
        if bot_id:
            await pipeline.unbind_audio_channel(bot_id, ws)
