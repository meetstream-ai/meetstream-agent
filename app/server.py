import asyncio
import base64
import json
import logging
import os
import struct
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing_extensions import assert_never
from starlette.websockets import WebSocketState

# === Your realtime agent stack ===
# Uses the exact imports you showed in server.py
from agents.realtime import RealtimeRunner, RealtimeSession, RealtimeSessionEvent
try:
    from .agent import get_starting_agent  # when used as a package
except Exception:
    from agent import get_starting_agent     # when run directly

import os, numpy as np
try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bridge")

IGNORED_SPEAKERS = {
    "Nav's test Meeting Agent",
    "Meetstream Agent",
    # add more display names here if needed
}


# configure I/O rates (defaults keep working if envs not set)
INCOMING_AUDIO_RATE = int(os.getenv("MEETSTREAM_IN_RATE", "48000"))   # ECS mic -> server
OUTGOING_AUDIO_RATE = int(os.getenv("MEETSTREAM_OUT_RATE", "48000"))  # server -> ECS speaker

def _resample_pcm16(pcm_bytes: bytes, src_hz: int, dst_hz: int) -> bytes:
    if src_hz == dst_hz:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    if _HAS_SCIPY:
        # exact rational resample with polyphase
        from math import gcd
        g = gcd(src_hz, dst_hz)
        up, down = dst_hz // g, src_hz // g
        y = resample_poly(x, up=up, down=down)
    else:
        # simple linear interp fallback (good enough for voice)
        n_out = int(len(x) * (dst_hz / src_hz))
        t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        t_dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(t_dst, t_src, x)

    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y.tobytes()

async def _preconnect_mcp(agent):
    """Connect all MCP servers on the agent before starting the session."""
    mcp_servers = getattr(agent, "mcp_servers", None) or []
    for srv in mcp_servers:
        # Most SDK wrappers expose connect()/disconnect() and an is_connected flag
        try:
            if hasattr(srv, "connect"):
                # optional: check flag to avoid double-connects
                is_connected = getattr(srv, "is_connected", False)
                if not is_connected:
                    await srv.connect()
        except Exception as e:
            logger.error(f"MCP connect failed for {getattr(srv,'name','<unnamed>')}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Manager that pairs:
#   - a RealtimeSession (OpenAI) per bot_id
#   - an optional Meetstream control WS per bot_id
#   - optional browser UI peers per session_id (unchanged from your demo)
# ──────────────────────────────────────────────────────────────────────────────
class BridgeManager:
    def __init__(self):
        # OpenAI realtime sessions keyed by bot_id
        self.sessions: Dict[str, RealtimeSession] = {}
        self.session_contexts: Dict[str, Any] = {}
        self._text_buf: Dict[str, str] = {} 
        self.last_sent: Dict[str, str] = {}

        # Meetstream control sockets keyed by bot_id
        self.ms_control_ws: Dict[str, WebSocket] = {}

        # Browser UI (unchanged) keyed by session_id
        self.ui_ws: Dict[str, WebSocket] = {}

        # Map bot_id -> ui session_id (optional, populated when a UI joins with ?bot_id=)
        self.bot_to_ui: Dict[str, str] = {}

        # Guard
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, bot_id: str) -> asyncio.Lock:
        if bot_id not in self._locks:
            self._locks[bot_id] = asyncio.Lock()
        return self._locks[bot_id]
    
    
    async def ensure_session(self, bot_id: str):
        """Create an OpenAI Realtime session for this bot_id if needed."""
        async with self._lock_for(bot_id):
            if bot_id in self.sessions:
                return
            agent = get_starting_agent()
            try:
                await _preconnect_mcp(agent)
            except Exception as e:
                logger.error(f"Preconnect MCP failed: {e}")
            runner = RealtimeRunner(agent)
            ctx = await runner.run()
            session = await ctx.__aenter__()
            self.session_contexts[bot_id] = ctx
            self.sessions[bot_id] = session
            asyncio.create_task(self._pump_openai_events(bot_id))

    async def close_session(self, bot_id: str):
        async with self._lock_for(bot_id):
            if bot_id in self.session_contexts:
                try:
                    await self.session_contexts[bot_id].__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"__aexit__ error for {bot_id}: {e}")
                self.session_contexts.pop(bot_id, None)
            self.sessions.pop(bot_id, None)

    async def attach_ms_control(self, bot_id: str, ws: WebSocket):
        self.ms_control_ws[bot_id] = ws
        await self.ensure_session(bot_id)
        logger.info(f"[control connected] bot={bot_id}")

    async def detach_ms_control(self, bot_id: str):
        self.ms_control_ws.pop(bot_id, None)
        logger.info(f"[control disconnected] bot={bot_id}")

    async def attach_ui(self, session_id: str, ws: WebSocket, bot_id: Optional[str] = None):
        self.ui_ws[session_id] = ws
        if bot_id:
            self.bot_to_ui[bot_id] = session_id
        logger.info(f"[ui connected] session={session_id} bot={bot_id}")

    async def detach_ui(self, session_id: str):
        self.ui_ws.pop(session_id, None)
        # also drop any reverse mapping
        for b, s in list(self.bot_to_ui.items()):
            if s == session_id:
                self.bot_to_ui.pop(b, None)
        logger.info(f"[ui disconnected] session={session_id}")

    # ── Inputs from Meetstream audio ───────────────────────────────────────────
    async def ingest_ms_audio_b64(self, bot_id: str, b64: str):
        if not b64:
            return
        try:
            pcm_in = base64.b64decode(b64)
        except Exception as e:
            logger.warning(f"bad base64 for {bot_id}: {e}")
            return
        pcm_24k = _resample_pcm16(pcm_in, INCOMING_AUDIO_RATE, 24000)
        await self.ensure_session(bot_id)
        try:
            await self.sessions[bot_id].send_audio(pcm_24k)
        except Exception as e:
            logger.error(f"send_audio error for {bot_id}: {e}")

    # ── Inputs from Meetstream text (control) ─────────────────────────────────
    async def ingest_ms_text(self, bot_id: str, text: str):
        await self.ensure_session(bot_id)
        # If your API differs, replace with the correct call to push user text.
        try:
            if hasattr(self.sessions[bot_id], "send_text"):
                await self.sessions[bot_id].send_text(text)
            else:
                # fallback: synthesize a "user message" via whatever your session supports
                logger.error("RealtimeSession.send_text missing; wire your text API here.")
        except Exception as e:
            logger.error(f"send_text error for {bot_id}: {e}")

    async def interrupt(self, bot_id: str):
        if bot_id not in self.sessions:
            return
        try:
            if hasattr(self.sessions[bot_id], "interrupt"):
                await self.sessions[bot_id].interrupt()
            else:
                # Optionally send a guardrail or audio stop signal if your API exposes it.
                logger.warning("interrupt() missing; implement your stop call here.")
        except Exception as e:
            logger.error(f"interrupt error for {bot_id}: {e}")

    # ── Outbound: pump OpenAI events to Meetstream control (and UI) ───────────
    async def _pump_openai_events(self, bot_id: str):
        try:
            session = self.sessions[bot_id]

            async for event in session:
                # --- 1) Use raw model events for clean turn-based text ---
                if event.type == "raw_model_event":
                    t = getattr(event.data, "type", None)

                    # Streamed text delta from the model
                    if t == "response.output_text.delta":
                        delta = getattr(event.data, "delta", "") or ""
                        if delta:
                            self._text_buf[bot_id] = self._text_buf.get(bot_id, "") + delta
                        continue

                    # Turn finished -> emit one final message
                    if t in ("response.completed", "response.finished"):
                        full = (self._text_buf.get(bot_id, "") or "").strip()
                        self._text_buf[bot_id] = ""  # reset buffer
                        if full:
                            ws = self.ms_control_ws.get(bot_id)
                            if ws and ws.client_state == WebSocketState.CONNECTED:
                                await _safe_send(ws, {
                                    "command": "sendmsg",
                                    "message": full,
                                    "bot_id": bot_id
                                })
                        continue

                    # (Optional) If the model errors/cancels, flush buffer
                    if t in ("response.error", "response.canceled"):
                        self._text_buf[bot_id] = ""
                        continue

                    # Ignore other raw events
                    continue

                # --- 2) Non-raw events (audio, interruptions, etc.) ---
                payload = await self._serialize_event(event)

                # Send audio to Meetstream (upsampled to OUTGOING_AUDIO_RATE, e.g., 48k)
                ws = self.ms_control_ws.get(bot_id)
                if ws and ws.client_state == WebSocketState.CONNECTED:
                    if payload.get("type") == "audio":
                        audio_b64 = payload.get("audio")
                        if audio_b64:
                            raw_24k = base64.b64decode(audio_b64)  # model outputs 24k PCM16
                            raw_out  = _resample_pcm16(raw_24k, 24000, OUTGOING_AUDIO_RATE)
                            audio_out_b64 = base64.b64encode(raw_out).decode("utf-8")
                            await _safe_send(ws, {
                                "command": "sendaudio",
                                "audiochunk": audio_out_b64,
                                "bot_id": bot_id,
                                "sample_rate": OUTGOING_AUDIO_RATE,
                                "encoding": "pcm16",
                                "channels": 1,
                                "endianness": "little"
                            })

                    if payload.get("type") == "audio_interrupted":
                        await _safe_send(ws, {
                            "command": "sendaudio",
                            "audiochunk": "",
                            "bot_id": bot_id
                        })

                    # Forward tool outputs (e.g., Playwright search results, Canva designs) to Meetstream control
                    if payload.get("type") == "tool_end":
                        tool_name = payload.get("tool")
                        raw_output = payload.get("output")
                        pretty_message = None
                        # Try to parse JSON-like outputs (Canva returns JSON with job/result/generated_designs)
                        try:
                            parsed = None
                            if isinstance(raw_output, str):
                                # raw_output is often a JSON string; attempt to load
                                parsed = json.loads(raw_output)
                            elif isinstance(raw_output, dict):
                                parsed = raw_output
                            if isinstance(parsed, dict):
                                job = parsed.get("job")
                                if isinstance(job, dict):
                                    result = job.get("result") or {}
                                    designs = result.get("generated_designs") or []
                                    links = []
                                    for d in designs:
                                        if not isinstance(d, dict):
                                            continue
                                        url = d.get("url")
                                        thumb = (d.get("thumbnail") or {}).get("url") if isinstance(d.get("thumbnail"), dict) else None
                                        if url:
                                            links.append((url, thumb))
                                    if links:
                                        lines = ["Canva designs generated:" if (tool_name and "canva" in tool_name.lower()) else "Designs generated:"]
                                        for i, (u, t) in enumerate(links, start=1):
                                            line = f"{i}. {u}"
                                            if t:
                                                line += f" (thumb: {t})"
                                            lines.append(line)
                                        pretty_message = "\n".join(lines)
                        except Exception:
                            # Ignore parse errors and fall back to raw output
                            pretty_message = None
                        try:
                            await _safe_send(ws, {
                                "command": "sendmsg",
                                "message": pretty_message if pretty_message else (str(raw_output) if raw_output is not None else ""),
                                "bot_id": bot_id
                            })
                        except Exception as e:
                            logger.warning(f"failed to forward tool output for {bot_id}: {e}")

                # --- 3) (Optional) Mirror to browser UI for debugging ---
                ui_session_id = self.bot_to_ui.get(bot_id)
                if ui_session_id and ui_session_id in self.ui_ws:
                    try:
                        await self.ui_ws[ui_session_id].send_text(json.dumps(payload))
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"pump events error for {bot_id}: {e}")



    async def _serialize_event(self, event: RealtimeSessionEvent) -> Dict[str, Any]:
        base_event: Dict[str, Any] = {"type": event.type}
        if event.type == "agent_start":
            base_event["agent"] = event.agent.name
        elif event.type == "agent_end":
            base_event["agent"] = event.agent.name
        elif event.type == "handoff":
            base_event["from"] = event.from_agent.name
            base_event["to"] = event.to_agent.name
        elif event.type == "tool_start":
            base_event["tool"] = event.tool.name
        elif event.type == "tool_end":
            base_event["tool"] = event.tool.name
            base_event["output"] = str(event.output)
        elif event.type == "audio":
            base_event["audio"] = base64.b64encode(event.audio.data).decode("utf-8")
        elif event.type == "audio_interrupted":
            pass
        elif event.type == "audio_end":
            pass
        elif event.type == "history_updated":
            base_event["history"] = [item.model_dump(mode="json") for item in event.history]
        elif event.type == "history_added":
            pass
        elif event.type == "guardrail_tripped":
            base_event["guardrail_results"] = [{"name": r.guardrail.name} for r in event.guardrail_results]
        elif event.type == "raw_model_event":
            base_event["raw_model_event"] = {"type": event.data.type}
        elif event.type == "error":
            base_event["error"] = getattr(event, "error", "Unknown error")
        elif event.type == "input_audio_timeout_triggered":
            pass
        else:
            assert_never(event)
        return base_event


def _extract_assistant_text(history: Optional[list]) -> Optional[str]:
    """Pulls a last assistant message’s text for convenience."""
    if not history:
        return None
    text = []
    for item in history:
        if item.get("type") == "message" and item.get("role") == "assistant":
            # flatten assistant content chunks
            for part in item.get("content", []):
                if part.get("type") in ("text", "input_text") and part.get("text"):
                    text.append(part["text"])
                elif part.get("type") in ("input_audio", "audio") and part.get("transcript"):
                    text.append(part["transcript"])
    return " ".join(t for t in text if t).strip() or None


async def _safe_send(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload))
    except Exception as e:
        logger.warning(f"send failed: {e}")


manager = BridgeManager()

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app + routes
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

# ----- 1) Optional UI socket (unchanged protocol) --------------------------------
@app.websocket("/ws/{session_id}")
async def ui_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()  # handshake first

    # bind bot_id (from ?bot_id=..., else fallback to session_id)
    q = websocket.scope.get("query_string", b"").decode() if websocket.scope else ""
    bot_id = None
    if q:
        try:
            for kv in q.split("&"):
                k, v = kv.split("=")
                if k == "bot_id":
                    bot_id = v
                    break
        except Exception:
            pass
    if not bot_id:
        bot_id = session_id

    # 👉 ensure the Realtime session is fully created/connected *before* UI sends anything
    await manager.ensure_session(bot_id)

    # now link the UI
    await manager.attach_ui(session_id, websocket, bot_id)

    # (optional) small ack to the UI
    await _safe_send(websocket, {"type": "ack", "message": f"UI bound {session_id} → {bot_id}"})

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "audio":
                int16 = data.get("data") or []
                if int16:
                    pcm = struct.pack(f"{len(int16)}h", *int16)
                    # session already ensured above, but keeping this is fine
                    await manager.ensure_session(bot_id)
                    try:
                        await manager.sessions[bot_id].send_audio(pcm)
                    except Exception as e:
                        # recover if session dropped
                        await manager.close_session(bot_id)
                        await manager.ensure_session(bot_id)
                        await manager.sessions[bot_id].send_audio(pcm)

            elif data.get("type") == "usermsg":
                msg = data.get("message")
                if msg:
                    await manager.ingest_ms_text(bot_id, msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"UI socket error: {e}")
    finally:
        await manager.detach_ui(session_id)



# ----- 2) Meetstream control channel --------------------------------------------
@app.websocket("/bridge")
async def meetstream_control_bind(websocket: WebSocket):
    await websocket.accept()
    bot_id = None
    try:
        # 1) handshake: { "type": "ready", "bot_id": "..." }
        init = json.loads(await websocket.receive_text())
        print("Bridge init", init)
        if init.get("type") != "ready" or not init.get("bot_id"):
            await websocket.close(code=1003)
            return
        bot_id = init["bot_id"]

        await manager.attach_ms_control(bot_id, websocket)

        # optional ack
        await _safe_send(websocket, {
            "command": "sendmsg",
            "message": f"Control channel bound to {bot_id}",
            "bot_id": bot_id
        })

        # 2) main loop
        while True:
            data = json.loads(await websocket.receive_text())
            cmd = data.get("command")
            if cmd == "usermsg":
                msg = data.get("message", "")
                if msg:
                    await manager.ingest_ms_text(bot_id, msg)
            elif cmd == "interrupt":
                await manager.interrupt(bot_id)
            # (extend with other commands as needed)

    except WebSocketDisconnect:
        pass
    finally:
        if bot_id:
            await manager.detach_ms_control(bot_id)


# ----- 3) Meetstream audio ingest channel ---------------------------------------
@app.websocket("/bridge/audio")
async def meetstream_audio_bind(websocket: WebSocket):
    await websocket.accept()
    bot_id = None
    try:
        # 1) handshake: { "type": "ready", "bot_id": "..." }
        init = json.loads(await websocket.receive_text())
        print("Audio init", init)
        if init.get("type") != "ready" or not init.get("bot_id"):
            await websocket.close(code=1003)
            return
        bot_id = init["bot_id"]

        await manager.ensure_session(bot_id)

        # optional ack
        await _safe_send(websocket, {
            "type": "ack",
            "message": f"Audio channel bound to {bot_id}"
        })

        # 2) chunk loop
        while True:
            # print("waiting for audio chunk")
            data = json.loads(await websocket.receive_text())
            # print("audio chunk received", data)
            if data.get("type") != "PCMChunk":
                continue
            speaker = data.get("speakerName", "")
            if speaker in IGNORED_SPEAKERS:
                # print("audio chunk received", data)
                # silently ignore agent/self audio
                continue
            print("got audio chunk", data.get("speakerName"))
            b64 = data.get("audioData")
            if b64:
                await manager.ingest_ms_audio_b64(bot_id, b64)

    except WebSocketDisconnect:
        pass
    finally:
        # no control ws to detach here
        ...


# ----- Static UI (optional) ------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

# ----- Entrypoint ----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
