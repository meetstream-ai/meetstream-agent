# agent.py — Realtime agent with local tools + Playwright MCP (stdio) + Framer MCP (SSE)
from __future__ import annotations

import os
import json
import logging
import subprocess
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
    # PW_BROWSER = os.getenv("PW_BROWSER", "chromium")
    # PW_HEADLESS = os.getenv("PW_HEADLESS", "1").lower()
    # PW_PROXY = os.getenv("PW_PROXY")
    # PW_UA = os.getenv("PW_UA")
    # PW_ALLOWED_ORIGINS = os.getenv("PW_ALLOWED_ORIGINS")
    # # New env reads
    # PW_SAVE_SESSION = os.getenv("PW_SAVE_SESSION", "1").lower()
    # PW_USER_DATA_DIR = os.getenv("PW_USER_DATA_DIR")
    # PW_BLOCKED_ORIGINS = os.getenv("PW_BLOCKED_ORIGINS")

    # pw_args = [
    #     "-y", "@playwright/mcp@latest",
    #     "--browser", PW_BROWSER,
    #     "--viewport-size=1366,820",
    #     # user-data-dir, save-session, blocked-origins, allowed-origins handled below
    # ]

    # # profile persistence / cookies
    # if PW_USER_DATA_DIR:
    #     pw_args.extend(["--user-data-dir", PW_USER_DATA_DIR])
    # else:
    #     # default persistent profile unless disabled
    #     default_profile = os.path.expanduser("~/.cache/ms-playwright/mcp-chrome-profile")
    #     pw_args.extend(["--user-data-dir", default_profile])
    # if PW_SAVE_SESSION in ("1", "true", "yes"):
    #     pw_args.append("--save-session")

    # # blocked / allowed origins (all optional now)
    # pw_args.append("--isolated")
    # if PW_BLOCKED_ORIGINS:
    #     pw_args.extend(["--blocked-origins", PW_BLOCKED_ORIGINS])
    # if PW_ALLOWED_ORIGINS:
    #     pw_args.extend(["--allowed-origins", PW_ALLOWED_ORIGINS])
    # if PW_HEADLESS in ("1", "true", "yes"):
    #     pw_args.append("--headless")
    # if PW_PROXY:
    #     pw_args.extend(["--proxy", PW_PROXY])
    # if PW_UA:
    #     pw_args.extend(["--user-agent", PW_UA])
    # else:
    #     # Default UA matches the selected browser reasonably well
    #     if PW_BROWSER.lower() in ("chromium", "chrome"):
    #         pw_args.extend(["--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"])
    #     elif PW_BROWSER.lower() == "firefox":
    #         pw_args.extend(["--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"])
    #     else:
    #         pw_args.extend(["--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"])

    # # Ensure the requested Playwright browser is installed
    # try:
    #     # Check if the Playwright browser is installed; if not, install it
    #     check_cmd = ["npx", "playwright", "launch-server", f"--browser={PW_BROWSER}", "--help"]
    #     subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # except Exception:
    #     install_cmd = ["npx", "-y", "playwright@latest", "install", "--with-deps", PW_BROWSER]
    #     try:
    #         subprocess.run(install_cmd, check=True)
    #         logging.info(f"Installed Playwright browser: {PW_BROWSER}")
    #     except Exception as e:
    #         logging.error(f"Failed to auto-install Playwright browser {PW_BROWSER}: {e}")

    # pw_params = MCPServerStdioParams(
    #     command="npx",
    #     args=pw_args,
    #     env={"PATH": os.getenv("PATH", "")},
    # )
    # servers.append(MCPServerStdio(
    #     name="playwright",
    #     params=pw_params,
    #     cache_tools_list=True,
    #     client_session_timeout_seconds=120,
    # ))

    # B) Framer MCP (SSE) — env or fallback URL
    framer_url = os.getenv("FRAMER_MCP_SSE_URL")
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


    # C) n8n MCP (remote via stdio) — no auth by default
    # Prefer using the MCP Remote transport over stdio using `npx mcp-remote <url>`.
    n8n_remote_url = os.getenv("N8N_MCP_SSE_URL") or \
                     os.getenv("N8N_MCP_REMOTE_URL")
    n8n_auth = os.getenv("N8N_MCP_AUTH") or os.getenv("AUTH_TOKEN")
    if n8n_remote_url:
        n8n_args = ["-y", "mcp-remote", n8n_remote_url]
        # If an auth token is present, pass an Authorization header
        if n8n_auth:
            n8n_args += ["--header", f"Authorization: Bearer {n8n_auth}"]
        n8n_stdio = MCPServerStdioParams(
            command="npx",
            args=n8n_args,
            env={
                "PATH": os.getenv("PATH", ""),
            },
        )
        servers.append(MCPServerStdio(
            name="n8n",
            params=n8n_stdio,
            cache_tools_list=True,
            client_session_timeout_seconds=120,
        ))

    # D) Canva MCP (remote via stdio)
    canva_url = "https://mcp.canva.com/mcp"
    canva_args = ["-y", "mcp-remote@latest", canva_url]

    canva_stdio = MCPServerStdioParams(
        command="npx",
        args=canva_args,
        env={"PATH": os.getenv("PATH", "")},
    )
    servers.append(MCPServerStdio(
        name="canva",
        params=canva_stdio,
        cache_tools_list=True,
        client_session_timeout_seconds=120,
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
        connected: List[object] = []
        for s in self.servers:
            try:
                if hasattr(s, "connect"):
                    is_connected = getattr(s, "is_connected", False)
                    if not is_connected:
                        await s.connect()
                connected.append(s)
                logging.info(f"Connected MCP: {getattr(s, 'name', '<unnamed>')}")
            except Exception as e:
                logging.error(f"Failed to connect MCP server {getattr(s,'name','<unnamed>')}: {e}")
        # Only keep the successfully connected servers
        self.servers = connected
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
- Use Canva MCP tools for UI/component/design actions if present.
- Use n8n MCP tools for workflows if present.

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
