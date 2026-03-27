"""Connect MCP servers attached to the agent before starting a Realtime session."""

from __future__ import annotations

import logging

logger = logging.getLogger("bridge.realtime.mcp")


async def preconnect_mcp_servers(agent) -> None:
    mcp_servers = getattr(agent, "mcp_servers", None) or []
    for srv in mcp_servers:
        try:
            if hasattr(srv, "connect"):
                is_connected = getattr(srv, "is_connected", False)
                if not is_connected:
                    await srv.connect()
        except Exception as e:
            logger.error(
                "MCP connect failed for %s: %s",
                getattr(srv, "name", "<unnamed>"),
                e,
            )
