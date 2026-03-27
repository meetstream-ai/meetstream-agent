"""
OpenAI Realtime session per bot_id: ingest meeting audio/text, pump model output to MeetStream.

MeetStream WebSocket registry (control/audio/UI) lives here so the event pump can send outbound
commands. Channel bind/unbind is MeetStream-specific but kept next to the pump for clarity.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from agents.realtime import RealtimeRunner, RealtimeSession, RealtimeSessionEvent

from app.meetstream.audio import resample_pcm16
from app.meetstream.config import AudioConfig, load_audio_config
from app.meetstream.outbound import (
    safe_send_json,
    send_chat_to_meeting,
    send_interrupt_clear_queue,
    send_pcm_to_meeting,
)
from app.realtime.events import serialize_session_event, tool_end_message_or_raw
from app.realtime.mcp import preconnect_mcp_servers

try:
    from app.agent import get_starting_agent
except Exception:
    from agent import get_starting_agent  # noqa: I001

logger = logging.getLogger("bridge.realtime.pipeline")


class RealtimeMeetingBridge:
    """
    One Realtime session per ``bot_id``, tied to MeetStream control/audio sockets and optional UI.

    Lifecycle: created on first ``ensure_session``; closed when both MeetStream channels disconnect
    (see ``maybe_cleanup``).
    """

    def __init__(self, audio_config: Optional[AudioConfig] = None):
        self._audio = audio_config or load_audio_config()

        self.sessions: Dict[str, RealtimeSession] = {}
        self.session_contexts: Dict[str, Any] = {}
        self._text_buf: Dict[str, str] = {}

        self.control_ws: Dict[str, WebSocket] = {}
        self.audio_ws: Dict[str, WebSocket] = {}

        self.ui_ws: Dict[str, WebSocket] = {}
        self.bot_to_ui: Dict[str, str] = {}

        self._locks: Dict[str, asyncio.Lock] = {}

        # Monotonic per bot_id so an old pump's cleanup cannot tear down a newer session.
        self._pump_generation: Dict[str, int] = {}
        self._ingest_dead_logged: Set[str] = set()
        # Model-rate PCM queued for MeetStream (batch tiny Realtime deltas into fewer sendaudio calls).
        self._meeting_out_buf: Dict[str, bytearray] = {}
        # While > 0, drop outbound TTS — Realtime may still emit audio during MCP; forwarding it garbles post-tool speech.
        self._meeting_tool_depth: Dict[str, int] = {}

    def _lock_for(self, bot_id: str) -> asyncio.Lock:
        if bot_id not in self._locks:
            self._locks[bot_id] = asyncio.Lock()
        return self._locks[bot_id]

    async def ensure_session(self, bot_id: str) -> None:
        async with self._lock_for(bot_id):
            if bot_id in self.sessions:
                return
            agent = get_starting_agent()
            try:
                await preconnect_mcp_servers(agent)
            except Exception as e:
                logger.error("Preconnect MCP failed: %s", e)
            runner = RealtimeRunner(agent)
            ctx = await runner.run()
            session = await ctx.__aenter__()
            self.session_contexts[bot_id] = ctx
            self.sessions[bot_id] = session
            gen = self._pump_generation.get(bot_id, 0) + 1
            self._pump_generation[bot_id] = gen
            self._ingest_dead_logged.discard(bot_id)
            asyncio.create_task(self._pump_model_events(bot_id, gen))
            logger.info("[%s] Realtime session started", bot_id)

    def _meeting_pcm_chunk_bytes(self) -> int:
        samples = int(self._audio.model_hz * self._audio.meeting_out_chunk_ms / 1000)
        samples = max(1, samples)
        return samples * 2

    async def _buffer_and_send_meeting_audio(self, bot_id: str, ws: WebSocket, pcm_model: bytes) -> None:
        if not pcm_model:
            return
        buf = self._meeting_out_buf.setdefault(bot_id, bytearray())
        buf.extend(pcm_model)
        chunk = self._meeting_pcm_chunk_bytes()
        pace = self._audio.sendaudio_pace_seconds
        while len(buf) >= chunk:
            block = bytes(buf[:chunk])
            del buf[:chunk]
            raw_out = resample_pcm16(block, self._audio.model_hz, self._audio.outgoing_hz)
            await send_pcm_to_meeting(
                ws,
                bot_id,
                base64.b64encode(raw_out).decode("utf-8"),
                sample_rate=self._audio.outgoing_hz,
            )
            if pace > 0:
                await asyncio.sleep(pace)

    async def _flush_meeting_audio_tail(self, bot_id: str, ws: WebSocket) -> None:
        buf = self._meeting_out_buf.get(bot_id)
        if not buf or len(buf) == 0:
            return
        # Drop incomplete sample (should not happen with int16 streams).
        n = len(buf) - (len(buf) % 2)
        if n == 0:
            buf.clear()
            return
        block = bytes(buf[:n])
        buf.clear()
        raw_out = resample_pcm16(block, self._audio.model_hz, self._audio.outgoing_hz)
        await send_pcm_to_meeting(
            ws,
            bot_id,
            base64.b64encode(raw_out).decode("utf-8"),
            sample_rate=self._audio.outgoing_hz,
        )
        if self._audio.sendaudio_pace_seconds > 0:
            await asyncio.sleep(self._audio.sendaudio_pace_seconds)

    async def close_session(self, bot_id: str) -> None:
        async with self._lock_for(bot_id):
            self._text_buf.pop(bot_id, None)
            self._meeting_out_buf.pop(bot_id, None)
            self._meeting_tool_depth.pop(bot_id, None)
            if bot_id in self.session_contexts:
                try:
                    await self.session_contexts[bot_id].__aexit__(None, None, None)
                except Exception as e:
                    logger.warning("session teardown error for %s: %s", bot_id, e)
                self.session_contexts.pop(bot_id, None)
            self.sessions.pop(bot_id, None)
            logger.info("[%s] Session closed", bot_id)

    async def maybe_cleanup(self, bot_id: str) -> None:
        if bot_id not in self.audio_ws and bot_id not in self.control_ws:
            await self.close_session(bot_id)

    async def bind_control(self, bot_id: str, ws: WebSocket) -> None:
        self.control_ws[bot_id] = ws
        await self.ensure_session(bot_id)
        logger.info("[control connected] bot=%s", bot_id)

    async def unbind_control(self, bot_id: str) -> None:
        self.control_ws.pop(bot_id, None)
        logger.info("[control disconnected] bot=%s", bot_id)
        await self.maybe_cleanup(bot_id)

    async def bind_audio_channel(self, bot_id: str, ws: WebSocket) -> None:
        self.audio_ws[bot_id] = ws
        await self.ensure_session(bot_id)

    async def unbind_audio_channel(self, bot_id: str) -> None:
        self.audio_ws.pop(bot_id, None)
        await self.maybe_cleanup(bot_id)

    async def attach_ui(self, session_id: str, ws: WebSocket, bot_id: Optional[str] = None) -> None:
        self.ui_ws[session_id] = ws
        if bot_id:
            self.bot_to_ui[bot_id] = session_id
        logger.info("[ui connected] session=%s bot=%s", session_id, bot_id)

    async def detach_ui(self, session_id: str) -> None:
        self.ui_ws.pop(session_id, None)
        for b, s in list(self.bot_to_ui.items()):
            if s == session_id:
                self.bot_to_ui.pop(b, None)
        logger.info("[ui disconnected] session=%s", session_id)

    def _is_dead_session_error(self, err: Exception) -> bool:
        msg = str(err).lower()
        return "not connected" in msg or "1000" in msg or "connection closed" in msg

    async def _on_ingest_send_failure(self, bot_id: str, err: Exception) -> None:
        if bot_id not in self._ingest_dead_logged:
            logger.warning("[%s] send to Realtime failed (will reset session): %s", bot_id, err)
            self._ingest_dead_logged.add(bot_id)
        else:
            logger.debug("[%s] send to Realtime failed: %s", bot_id, err)
        if self._is_dead_session_error(err):
            await self.close_session(bot_id)

    async def ingest_meeting_audio_b64(self, bot_id: str, b64: str) -> None:
        if not b64:
            return
        try:
            pcm_in = base64.b64decode(b64)
        except Exception as e:
            logger.warning("bad base64 for %s: %s", bot_id, e)
            return
        pcm_model = resample_pcm16(pcm_in, self._audio.incoming_hz, self._audio.model_hz)
        await self.ensure_session(bot_id)
        try:
            await self.sessions[bot_id].send_audio(pcm_model)
        except Exception as e:
            await self._on_ingest_send_failure(bot_id, e)

    async def ingest_meeting_audio_pcm(self, bot_id: str, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        pcm_model = resample_pcm16(pcm_bytes, self._audio.incoming_hz, self._audio.model_hz)
        await self.ensure_session(bot_id)
        try:
            await self.sessions[bot_id].send_audio(pcm_model)
        except Exception as e:
            await self._on_ingest_send_failure(bot_id, e)

    async def ingest_user_text(self, bot_id: str, text: str) -> None:
        await self.ensure_session(bot_id)
        try:
            if hasattr(self.sessions[bot_id], "send_text"):
                await self.sessions[bot_id].send_text(text)
            else:
                logger.error("RealtimeSession.send_text missing.")
        except Exception as e:
            logger.error("send_text error for %s: %s", bot_id, e)

    async def interrupt_model(self, bot_id: str) -> None:
        if bot_id not in self.sessions:
            return
        try:
            if hasattr(self.sessions[bot_id], "interrupt"):
                await self.sessions[bot_id].interrupt()
            else:
                logger.warning("interrupt() not available on session.")
        except Exception as e:
            logger.error("interrupt error for %s: %s", bot_id, e)

    async def _pump_model_events(self, bot_id: str, gen: int) -> None:
        try:
            session = self.sessions[bot_id]

            async for event in session:
                if event.type == "raw_model_event":
                    t = getattr(event.data, "type", None)

                    if t == "response.output_text.delta":
                        delta = getattr(event.data, "delta", "") or ""
                        if delta:
                            self._text_buf[bot_id] = self._text_buf.get(bot_id, "") + delta
                        continue

                    if t in ("response.completed", "response.finished"):
                        full = (self._text_buf.get(bot_id, "") or "").strip()
                        self._text_buf[bot_id] = ""
                        if full:
                            ws = self.control_ws.get(bot_id)
                            if ws and ws.client_state == WebSocketState.CONNECTED:
                                await send_chat_to_meeting(ws, bot_id, full)
                        continue

                    if t in ("response.error", "response.canceled"):
                        self._text_buf[bot_id] = ""
                        continue

                    continue

                payload = await serialize_session_event(event)

                ws = self.control_ws.get(bot_id)
                if ws and ws.client_state == WebSocketState.CONNECTED:
                    et = payload.get("type")
                    in_tool = self._meeting_tool_depth.get(bot_id, 0) > 0

                    if et == "audio":
                        audio_b64 = payload.get("audio")
                        if audio_b64 and not in_tool:
                            raw_model = base64.b64decode(audio_b64)
                            await self._buffer_and_send_meeting_audio(bot_id, ws, raw_model)

                    if et == "audio_end":
                        if in_tool:
                            self._meeting_out_buf.pop(bot_id, None)
                        elif ws and ws.client_state == WebSocketState.CONNECTED:
                            await self._flush_meeting_audio_tail(bot_id, ws)
                        else:
                            self._meeting_out_buf.pop(bot_id, None)

                    if et == "audio_interrupted":
                        self._meeting_out_buf.pop(bot_id, None)
                        await send_interrupt_clear_queue(ws, bot_id)

                    # Tool calls interrupt the speech stream; staged PCM must not bleed into
                    # the next utterance (common cause of stutter right after tool_end).
                    if et == "tool_start":
                        self._meeting_tool_depth[bot_id] = self._meeting_tool_depth.get(bot_id, 0) + 1
                        self._meeting_out_buf.pop(bot_id, None)
                        await send_interrupt_clear_queue(ws, bot_id)

                    if et == "tool_end":
                        self._meeting_out_buf.pop(bot_id, None)
                        d = self._meeting_tool_depth.get(bot_id, 0)
                        if d <= 1:
                            self._meeting_tool_depth.pop(bot_id, None)
                        else:
                            self._meeting_tool_depth[bot_id] = d - 1
                        tool_name = payload.get("tool")
                        raw_output = payload.get("output")
                        msg = tool_end_message_or_raw(tool_name, raw_output)
                        if msg:
                            try:
                                await send_chat_to_meeting(ws, bot_id, msg)
                            except Exception as e:
                                logger.warning("failed to forward tool output for %s: %s", bot_id, e)

                ui_session_id = self.bot_to_ui.get(bot_id)
                if ui_session_id and ui_session_id in self.ui_ws:
                    try:
                        await self.ui_ws[ui_session_id].send_text(json.dumps(payload))
                    except Exception:
                        pass

        except Exception as e:
            logger.error("pump events error for %s: %s", bot_id, e)
        finally:
            # When the pump exits (error or Realtime stream ended), drop the session so
            # ``ensure_session`` can open a new one. Generation prevents a stale pump
            # from closing a newer session.
            if self._pump_generation.get(bot_id) == gen:
                await self.close_session(bot_id)


bridge = RealtimeMeetingBridge()
