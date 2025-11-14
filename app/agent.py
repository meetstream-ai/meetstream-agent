# agent.py — Legacy file for backward compatibility
# DEPRECATED: Use app.agents.realtime_agent or app.agents.non_realtime_agent instead
# This file is kept for backward compatibility with existing server.py imports
from __future__ import annotations

import logging
import warnings

# Quiet noisy logs (optional)
logging.getLogger("agents.mcp").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("openai.agents").setLevel(logging.ERROR)

# Agents SDK
from agents.realtime import RealtimeAgent

# Import from new structure
try:
    from .agent_modules import get_realtime_agent
except Exception:
    from agent_modules import get_realtime_agent

# Backward compatibility: get_starting_agent() now returns realtime agent without MCPs
_deprecation_warned = False

def get_starting_agent() -> RealtimeAgent:
    """
    Get the starting realtime agent (without MCPs).
    
    DEPRECATED: Use get_realtime_agent() from agent_modules.realtime_agent instead.
    
    Note: MCPs have been removed from realtime agents.
    For MCP support, use non-realtime agents (Langchain-based).
    """
    global _deprecation_warned
    if not _deprecation_warned:
        warnings.warn(
            "agent.py is deprecated. Use agent_modules.realtime_agent or agent_modules.non_realtime_agent instead.",
            DeprecationWarning,
            stacklevel=2
        )
        _deprecation_warned = True
    return get_realtime_agent()

# Legacy exports for backward compatibility
__all__ = ["get_starting_agent", "RealtimeAgent"]
