# agent.py — Realtime agent with local tools + Playwright MCP (stdio) + Framer MCP (SSE)
from __future__ import annotations

import os
import json
import logging
from typing import Optional, List
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Quiet noisy logs (optional)
logging.getLogger("agents.mcp").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("openai.agents").setLevel(logging.ERROR)

# Agents SDK
from agents import function_tool
from agents.realtime import RealtimeAgent

# MCP server classes (compat imports across SDK versions)
# MCP classes (compat across SDK versions)
try:
    from agents.mcp import (
        MCPServerStdio, MCPServerSse,
        MCPServerStdioParams, MCPServerSseParams,
    )
except Exception:
    from agents.mcp.server import (  # type: ignore
        MCPServerStdio, MCPServerSse,
        MCPServerStdioParams, MCPServerSseParams,
    )
# Optional HTTP client for weather
try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


# ───────────────────────────────── Local tools ─────────────────────────────────

@function_tool(
    name_override="current_time",
    description_override="Return the current time in ISO 8601. Optional timezone as an IANA string, e.g., 'America/Toronto'.",
)
def current_time(timezone_name: Optional[str] = None) -> str:
    try:
        if timezone_name:
            return datetime.now(ZoneInfo(timezone_name)).isoformat()
        return datetime.now(timezone.utc).isoformat()
    except Exception as e:
        return f"current_time error: {e}"


@function_tool(
    name_override="weather_now",
    description_override="Get current weather for a city using Open-Meteo (no API key).",
)
async def weather_now(city: str) -> str:
    if not httpx:
        return "weather_now unavailable: httpx not installed."
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Geocode
            g = (await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
            )).json()
            if not g.get("results"):
                return f"Couldn't find '{city}'."
            r0 = g["results"][0]
            lat, lon = r0["latitude"], r0["longitude"]
            loc = r0.get("name") or city

            # Current weather
            w = (await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": True},
            )).json()
            cw = w.get("current_weather") or {}
            t = cw.get("temperature")
            wind = cw.get("windspeed")
            code = cw.get("weathercode")
            ts = cw.get("time")
            return f"Weather in {loc}: {t}°C, wind {wind} km/h, code {code}, at {ts}."
    except Exception as e:
        return f"weather_now failed: {e}"


# ───────────────────────────── MCP config loader ──────────────────────────────

def _expand_env(val: str) -> str:
    return os.path.expandvars(val) if isinstance(val, str) else val

def build_mcp_servers_from_config(path: str) -> List[object]:
    """Load MCP servers from a JSON config file (supports sse, stdio, streamable_http)."""
    servers: List[object] = []
    if not path or not os.path.exists(path):
        return servers
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    spec_map = cfg.get("mcpServers") or {}
    for name, spec in spec_map.items():
        t = (spec.get("type") or "").lower()
        try:
            if t == "stdio":
                params = {
                    "command": _expand_env(spec.get("command", "")),
                    "args": [_expand_env(a) for a in (spec.get("args") or [])],
                    "env": {k: _expand_env(v) for k, v in (spec.get("env") or {}).items()},
                }
                servers.append(MCPServerStdio(
                    name=name,
                    params=params,
                    cache_tools_list=True,
                    client_session_timeout_seconds=int(spec.get("timeout", 60)),
                ))
            elif t == "sse":
                servers.append(MCPServerSse(
                    name=name,
                    url=_expand_env(spec.get("url", "")),
                    headers=spec.get("headers"),
                    cache_tools_list=True,
                ))
            elif t in ("stream", "streamable_http", "http"):
                servers.append(MCPServerStreamableHttp(
                    name=name,
                    url=_expand_env(spec.get("url", "")),
                    headers=spec.get("headers"),
                    cache_tools_list=True,
                ))
            else:
                logging.warning(f"Unknown MCP server type for '{name}': {t}")
        except Exception as e:
            logging.error(f"Failed to init MCP server '{name}': {e}")
    return servers


# ───────────────────────────── Default MCP wiring ─────────────────────────────

def build_mcp_servers_default() -> List[object]:
    servers: List[object] = []

    # A) Playwright MCP (stdio) — no API key; anti-captcha flags
    pw_params = MCPServerStdioParams(
        command="npx",
        args=[
            "-y", "@playwright/mcp@latest",
            "--browser", "chrome",
            "--viewport-size=1366,820",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/126.0.0.0 Safari/537.36",
            "--user-data-dir", os.path.expanduser("~/.cache/ms-playwright/mcp-chrome-profile"),
            "--save-session",
            "--blocked-origins", "https://www.google.com;https://duckduckgo.com;https://www.bing.com;https://consent.google.com",
            "--allowed-origins", "https://meetstream.ai;https://www.linkedin.com;https://x.com;https://github.com;https://docs.meetstream.ai",
        ],
        env={"PATH": os.getenv("PATH", "")},
    )
    servers.append(MCPServerStdio(
        name="playwright",
        params=pw_params,
        cache_tools_list=True,
        client_session_timeout_seconds=120,
    ))

    # B) Framer MCP (SSE) — env or fallback URL
    framer_url = "https://mcp.unframer.co/sse?id=f6b7d9348abed570aafe96724fe7e42183ae4275a28991e4d3153cb992691ecc&secret=1H4c6ydjR516HUg3z6NCKmnK6RegD1Eq"
    if framer_url:
        fr_params = MCPServerSseParams(
            url=framer_url,
            headers=None,
            request_timeout_seconds=30.0,
            sse_timeout_seconds=30.0,
        )
        servers.append(MCPServerSse(
            name="framer",
            params=fr_params,
            cache_tools_list=True,
            client_session_timeout_seconds=90,
        ))

    return servers


def build_mcp_servers() -> List[object]:
    """Prefer external JSON config; otherwise use defaults (Playwright + Framer)."""
    cfg_path = os.getenv("MCP_CONFIG", "mcp.config.json")
    servers = build_mcp_servers_from_config(cfg_path)
    if servers:
        logging.info(f"Loaded MCP servers from {cfg_path}: {[getattr(s, 'name', '<unnamed>') for s in servers]}")
        return servers
    servers = build_mcp_servers_default()
    logging.info(f"Using default MCP servers: {[getattr(s, 'name', '<unnamed>') for s in servers]}")
    return servers


# ─────────────────────────────── MCP preconnect ───────────────────────────────

class _MCPRegistry:
    def __init__(self) -> None:
        self.servers = build_mcp_servers()
        self._connected = False

    async def connect_all(self) -> None:
        if self._connected:
            return
        names = [getattr(s, "name", "<unnamed>") for s in self.servers]
        logging.info(f"MCP servers to connect: {names}")
        for s in self.servers:
            try:
                if hasattr(s, "connect"):
                    await s.connect()
                    logging.info(f"Connected MCP: {getattr(s, 'name', '<unnamed>')}")
            except Exception as e:
                logging.error(f"Failed to connect MCP server {getattr(s,'name','<unnamed>')}: {e}")
        self._connected = True


MCP_REGISTRY = _MCPRegistry()

async def mcp_connect_all():
    await MCP_REGISTRY.connect_all()

_connected_once = False
async def mcp_connect_once_if_needed():
    global _connected_once
    if not _connected_once:
        await MCP_REGISTRY.connect_all()
        _connected_once = True


# ─────────────────────────────── Realtime Agent ───────────────────────────────

AGENT_INSTRUCTIONS = """
You are a realtime meeting assistant.

Prefer tools when available:
- Use `current_time` for time questions.
- Use `weather_now` for current weather.
- Use Playwright MCP tools for browsing (e.g., `browser_navigate`, `browser_wait_for_selector`,
  `browser_take_screenshot`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_install`).
- Use Framer MCP tools for UI/component/design actions if present.

Keep spoken responses concise and avoid repeating prior text verbatim.
"""

assistant_agent = RealtimeAgent(
    name="Meetstream Realtime Agent",
    handoff_description="Single agent with local tools and Playwright/Framer MCPs.",
    instructions=AGENT_INSTRUCTIONS,
    tools=[current_time, weather_now],
    mcp_servers=MCP_REGISTRY.servers,
)


def get_starting_agent() -> RealtimeAgent:
    """
    IMPORTANT: Ensure MCP servers are connected before the session starts.
    In your server (before RealtimeRunner.run()), call:
        from agent import mcp_connect_once_if_needed
        await mcp_connect_once_if_needed()
    """
    return assistant_agent
