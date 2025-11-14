"""MCP server setup for non-realtime agents."""
from __future__ import annotations

import os
import json
import logging
import subprocess
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def _expand_env(val: str) -> str:
    """Expand environment variables in a string."""
    return os.path.expandvars(val) if isinstance(val, str) else val


def build_mcp_servers_from_config(path: str) -> List[Dict[str, Any]]:
    """Load MCP servers from a JSON config file (supports sse, stdio, streamable_http)."""
    servers: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return servers
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        spec_map = cfg.get("mcpServers") or {}
        for name, spec in spec_map.items():
            t = (spec.get("type") or "").lower()
            try:
                server_config = {
                    "name": name,
                    "type": t,
                }
                if t == "stdio":
                    server_config.update({
                        "command": _expand_env(spec.get("command", "")),
                        "args": [_expand_env(a) for a in (spec.get("args") or [])],
                        "env": {k: _expand_env(v) for k, v in (spec.get("env") or {}).items()},
                        "timeout": int(spec.get("timeout", 60)),
                    })
                elif t == "sse":
                    server_config.update({
                        "url": _expand_env(spec.get("url", "")),
                        "headers": spec.get("headers"),
                        "timeout": int(spec.get("timeout", 60)),
                    })
                elif t in ("stream", "streamable_http", "http"):
                    server_config.update({
                        "url": _expand_env(spec.get("url", "")),
                        "headers": spec.get("headers"),
                        "timeout": int(spec.get("timeout", 60)),
                    })
                else:
                    logger.warning(f"Unknown MCP server type for '{name}': {t}")
                    continue
                servers.append(server_config)
            except Exception as e:
                logger.error(f"Failed to init MCP server '{name}': {e}")
    except Exception as e:
        logger.error(f"Failed to load MCP config from {path}: {e}")
    return servers


def build_mcp_servers_default() -> List[Dict[str, Any]]:
    """Build default MCP servers configuration."""
    servers: List[Dict[str, Any]] = []

    # Framer MCP (SSE)
    framer_url = os.getenv("FRAMER_MCP_SSE_URL")
    if framer_url:
        servers.append({
            "name": "framer",
            "type": "sse",
            "url": framer_url,
            "headers": None,
            "timeout": 90,
        })

    # n8n MCP (remote via stdio)
    n8n_remote_url = os.getenv("N8N_MCP_SSE_URL") or os.getenv("N8N_MCP_REMOTE_URL")
    n8n_auth = os.getenv("N8N_MCP_AUTH") or os.getenv("AUTH_TOKEN")
    if n8n_remote_url:
        n8n_args = ["-y", "mcp-remote", n8n_remote_url]
        if n8n_auth:
            n8n_args += ["--header", f"Authorization: Bearer {n8n_auth}"]
        servers.append({
            "name": "n8n",
            "type": "stdio",
            "command": "npx",
            "args": n8n_args,
            "env": {"PATH": os.getenv("PATH", "")},
            "timeout": 120,
        })

    # Canva MCP (remote via stdio)
    canva_url = "https://mcp.canva.com/mcp"
    canva_args = ["-y", "mcp-remote@latest", canva_url]
    servers.append({
        "name": "canva",
        "type": "stdio",
        "command": "npx",
        "args": canva_args,
        "env": {"PATH": os.getenv("PATH", "")},
        "timeout": 120,
    })

    return servers


def get_mcp_servers_config() -> List[Dict[str, Any]]:
    """
    Get MCP servers configuration.
    Prefers external JSON config; otherwise uses defaults.
    
    Returns:
        List of MCP server configuration dictionaries.
    """
    cfg_path = os.getenv("MCP_CONFIG", "mcp.config.json")
    servers = build_mcp_servers_from_config(cfg_path)
    if servers:
        logger.info(f"Loaded MCP servers from {cfg_path}: {[s['name'] for s in servers]}")
        return servers
    servers = build_mcp_servers_default()
    logger.info(f"Using default MCP servers: {[s['name'] for s in servers]}")
    return servers

