# MeetStream Bridge

Open-source **FastAPI** bridge between [MeetStream](https://meetstream.ai) and **OpenAI Realtime** via the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) (`openai-agents[voice]`). It implements the two WebSocket endpoints and session lifecycle described in the official guide: [Bridge Server Architecture & Session Management](https://docs.meetstream.ai/guides/get-started/bridge-server-architecture).

Use this repo as a starting point: run the server on a public URL (or tunnel), point MeetStream at your `/bridge` and `/bridge/audio` URLs, and supply your own **OpenAI** and **MeetStream** credentials where each service expects them.

## Start here

### 1. Run locally

| Step | Action |
|------|--------|
| 1 | `uv sync` |
| 2 | Copy `.env.example` → `.env` and set `OPENAI_API_KEY` and `BRIDGE_SECRET` (see below) |
| 3 | Start the server |

```bash
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Check [http://localhost:8000/health](http://localhost:8000/health).

`BRIDGE_SECRET` signs the session tokens that MeetStream must present (step 2). Make one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep it secret. The bridge refuses every MeetStream socket while `BRIDGE_SECRET` is unset or shorter than 32 characters.

### 2. Expose to MeetStream (tunnel)

The MeetStream API expects **public** `wss://` URLs, not `localhost`. With the bridge still on port `8000`, start a tunnel in another terminal—for example [ngrok](https://ngrok.com):

```bash
ngrok http 8000
```

A tunnel makes the bridge reachable by anyone who finds the URL, so each bot gets its own **session token** in its socket URLs. Make one token per bot, just before you create the bot:

```bash
uv run python -m app.meetstream.session_token          # valid 12 hours; --ttl-hours to change
```

Use the HTTPS hostname ngrok prints, swap the scheme to `wss://`, and append the paths with the token:

| MeetStream field | Example URL |
|------------------|-------------|
| `socket_connection_url` → control | `wss://abc123.ngrok-free.app/bridge/<token>` |
| `live_audio_required` → audio | `wss://abc123.ngrok-free.app/bridge/audio/<token>` |

Same host and token for both; only the path changes. The first `ready` message binds the token to its `bot_id`: the other channel and any reconnect must be for the same bot, and the token cannot be used for another bot. A token is checked when a socket opens: open sockets stay up past its expiry, but a reconnect after expiry is refused, so pick a lifetime longer than your longest meeting.

Treat the socket URLs as secrets: until the token expires, whoever has the URL can connect as that bot whenever the bot itself is not connected. While a bot is connected, a second control or audio socket for it is refused (1008), so nobody can take over a live session; if a socket dies without closing, uvicorn's WebSocket ping drops it within about 40 seconds and a reconnect is accepted again. The server redacts tokens from its own logs (`/bridge/<token>`), but the ngrok inspector (`http://localhost:4040`) and anything else that records full URLs will show them. Keep `--ttl-hours` close to the meeting length.

**Upgrading from the token-less URLs.** Bots created with `wss://<host>/bridge` and `wss://<host>/bridge/audio` are refused after this change (the server log says why). Set `BRIDGE_SECRET`, restart the server, then create new bots with a token in both URLs as above; there is no setting that turns the token off.

To make tokens in your own backend (any language): `<expires_unix>.<nonce>.<sig>`, where `sig` is the hex HMAC-SHA256 of `"<expires_unix>.<nonce>"` with `BRIDGE_SECRET` as the key, and `nonce` is random hex. See `app/meetstream/session_token.py`.

### 3. Entry point in code

| File | Role |
|------|------|
| `app/server.py` | Builds the FastAPI app, static files, `.env` load |
| `app/routes/websockets.py` | Route map: `/bridge/<token>`, `/bridge/audio/<token>`, optional debug UI |

You rarely edit `server.py` unless you add middleware or new HTTP routes.

**4. What to open first (by goal)**  

| Goal | Open first |
|------|------------|
| Change how the AI talks, tools, or MCP | `app/agent.py` |
| Change MeetStream audio decoding / speaker filtering | `app/meetstream/audio.py`, `app/meetstream/config.py` |
| Change what we send back to the meeting (audio/chat/interrupt JSON) | `app/meetstream/outbound.py` |
| Wire meeting audio ↔ model ↔ outbound commands | `app/realtime/pipeline.py` (`RealtimeMeetingBridge`) |
| See raw WebSocket receive loops | `app/meetstream/ws_handlers.py` |

**5. Reading order for different levels**  
- **New to the repo:** this README → **Repository layout** (below) → skim `app/realtime/pipeline.py` (the “glue” class).  
- **Comfortable with FastAPI:** `app/routes/` → `app/meetstream/` → `app/realtime/pipeline.py`.  
- **MCP setup (Canva, JSON config):** [docs/mcp.md](docs/mcp.md) → then [README.agents.md](README.agents.md) for advanced agent/provider topics.

The layout is intentional: **MeetStream wire format** lives under `app/meetstream/`, **OpenAI Realtime + pump** under `app/realtime/`, **persona and tools** in `app/agent.py`. That matches the official docs split ([audio streaming](https://docs.meetstream.ai/guides/get-started/real-time-audio-streaming) vs [control commands](https://docs.meetstream.ai/guides/get-started/meeting-control-and-command-patterns)) plus a single orchestrator in `pipeline.py`.

## What this server does

| Endpoint | Direction | Purpose |
|----------|-----------|---------|
| `GET /health` | — | Liveness check |
| `GET /` | — | Demo UI (`app/static/`) when `DEBUG_UI_ENABLED=1`; otherwise a JSON status |
| `WebSocket /bridge/<token>` (or `/{bot_id}/bridge/<token>`) | MeetStream ↔ you | JSON control: handshake, `usermsg`, `interrupt`; outbound `sendaudio`, `sendmsg`, etc. |
| `WebSocket /bridge/audio/<token>` (or `/{bot_id}/bridge/audio/<token>`) | MeetStream → you | Binary frames (current protocol) or legacy JSON `PCMChunk` |
| `WebSocket /ws/{session_id}` | browser → you | Debug UI socket; only when `DEBUG_UI_ENABLED=1` |

The old URLs without a token (`/bridge`, `/bridge/audio`) are refused, and the server log says what to change.

### Debug UI (local only)

`app/static/` has a browser page that sends your microphone to the agent. It can attach to the session of any `bot_id` and has no login, so it is off by default. Turn it on only on your own machine, never behind a tunnel:

```bash
DEBUG_UI_ENABLED=1 uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
```

The bridge keeps **one OpenAI Realtime session per `bot_id`**. The session is created when the first channel connects and is torn down when **both** WebSockets have disconnected, matching the docs.

### Repository layout

| Path | Role |
|------|------|
| `app/server.py` | FastAPI app factory, static mount, `.env` load |
| `app/routes/` | HTTP (`/`, `/health`) and WebSocket route declarations |
| `app/meetstream/` | MeetStream wire format: binary audio decode, resampling, speaker filter, outbound JSON commands, `/bridge` receive loops |
| `app/realtime/` | OpenAI Realtime session lifecycle, MCP preconnect, event serialization, model → MeetStream pump (`RealtimeMeetingBridge`) |
| `app/agent.py` | `RealtimeAgent`, tools, MCP server list (LLM-facing only) |
| `app/meetstream/session_token.py` | Per-session tokens for the bridge sockets (make, check, bind to a bot) |
| `app/static/` | Optional browser UI for local debugging (`DEBUG_UI_ENABLED=1`) |
| `mcp.config.json` | MCP servers (default: Canva); expand with more `mcpServers` entries |
| `docs/mcp.md` | Reference for `mcp.config.json` transports and examples |

## Prerequisites

- Python **3.12+**
- [uv](https://github.com/astral-sh/uv) (recommended)
- **OpenAI API key** (realtime / voice stack)
- **MeetStream API key** (when you create bots/sessions via the MeetStream API—not stored in this app unless you add that yourself)

## Quickstart

```bash
git clone <this-repo>
cd meetstream-agent

cp .env.example .env
# Edit .env: set OPENAI_API_KEY and BRIDGE_SECRET

uv sync
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Or:

```bash
uv run python main.py
```

Load `.env` from the project root automatically (via `python-dotenv` in the app lifespan).

## MeetStream session configuration

When you create a MeetStream session (HTTP API), point both WebSockets at your deployed bridge (use `wss://` in production):

```json
{
  "meeting_url": "https://meet.google.com/your-meeting-id",
  "bot_name": "My Assistant",
  "live_audio_required": {
    "websocket_url": "wss://your-host.example.com/bridge/audio/<token>"
  },
  "socket_connection_url": {
    "websocket_url": "wss://your-host.example.com/bridge/<token>"
  }
}
```

`<token>` is a new session token for this bot (`uv run python -m app.meetstream.session_token`). The same `bot_id` is sent on both channels in the `ready` handshake so this server can pair them, and the token is bound to that `bot_id`.

If your MeetStream API uses URL templates with **`{bot_id}`** in the path, use the same host and paths **`wss://…/{bot_id}/bridge/<token>`** and **`wss://…/{bot_id}/bridge/audio/<token>`**. The bot in the path must be the bot in `ready`.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for Realtime / Agents |
| `BRIDGE_SECRET` | Yes | Signs the per-session tokens in the bridge socket URLs (at least 32 characters). Without it, every MeetStream socket is refused |
| `DEBUG_UI_ENABLED` | No | `1` turns on the browser debug UI (`/` and `/ws/{session_id}`). Default off; local use only |
| `PORT` | No | HTTP port (default `8000`) |
| `MEETSTREAM_BOT_NAME` | No | Display name of your bot in the meeting; audio from this speaker is ignored (echo control) |
| `MEETSTREAM_IGNORE_SPEAKERS` | No | Comma-separated display names to ignore |
| `AGENT_SPEAKER_KEYWORDS` | No | Comma-separated substrings (default `bot,agent,assistant,ai`) to filter self/agent audio |
| `MEETSTREAM_IN_RATE` / `MEETSTREAM_OUT_RATE` | No | PCM rates for MeetStream I/O (default `48000`) |
| `MODEL_AUDIO_RATE` | No | Rate your model expects for input/output PCM in the bridge (default `24000` for OpenAI Realtime) |

MeetStream **API keys** belong in whatever client or backend calls MeetStream to create sessions—not in this bridge unless you add that integration.

See `.env.example` for a template.

## MCP servers (Canva by default + optional Docker)

### Canva (`mcp.config.json`)

The repo ships **`mcp.config.json`** with a **Canva** entry (`npx mcp-remote` → Canva’s MCP). Install **Node.js** so **`npx`** is on your `PATH`.

- **Disable Canva:** remove the `canva` block or rename the config file (see [docs/mcp.md](docs/mcp.md)).
- **More MCPs in JSON:** add entries under `mcpServers` — [docs/mcp.md](docs/mcp.md).
- **Different file:** `MCP_CONFIG=/path/to/file.json` in `.env`.

### Optional Docker / streamable HTTP MCP (env toggle)

Use this when your MCP runs in **Docker** (or elsewhere) and is reachable at an **HTTPS** URL (often via **ngrok**), with a **Bearer** token.

1. In **`.env`** (never commit secrets):

   ```env
   DOCKER_MCP_ENABLED=1
   DOCKER_MCP_URL=https://YOUR-NGROK-HOST/mcp
   DOCKER_MCP_BEARER_TOKEN=YOUR_TOKEN
   ```

2. Set **`DOCKER_MCP_ENABLED=0`** (or unset) to turn this MCP off. Canva from `mcp.config.json` can stay on at the same time.

3. Details and timeouts: [docs/mcp.md](docs/mcp.md).

**Agent tuning, Framer/n8n via env only, multi-provider notes:** [README.agents.md](README.agents.md)

## Protocol notes (aligned with MeetStream docs)

- **Audio in:** Binary frames per [Live Audio Capture & Frame Decoding](https://docs.meetstream.ai/guides/get-started/real-time-audio-streaming), or legacy JSON `PCMChunk` with base64 audio.
- **Control in:** After `{"type":"ready","bot_id":"..."}`, MeetStream may send `usermsg` and `interrupt`.
- **Control out:** The bridge sends `command: "ack"` after binding, `sendaudio` for PCM back to the meeting, `sendmsg` for chat (includes both `message` and `msg`), and `interrupt` with `action: "clear_audio_queue"` when the model stops audio or **starts a tool** (so queued TTS does not overlap the next utterance after the tool). Outbound model audio is **batched** (default ~240 ms at the model sample rate, then resampled to 48 kHz) so MeetStream gets fewer, larger chunks than raw Realtime deltas—tune with **`MEETSTREAM_OUT_AUDIO_CHUNK_MS`** (40–2000) if playback glitches or latency feels off. Optional **`MEETSTREAM_SENDAUDIO_PACE_MS`** adds a short pause between consecutive `sendaudio` frames if the client struggles with burst traffic.

## Development

```bash
uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

Optional: install **SciPy** for higher-quality resampling (`scipy.signal.resample_poly`); otherwise NumPy linear interpolation is used.

## License

Use and modify per your open-source license (add a `LICENSE` file if this repo should ship under a specific license).
