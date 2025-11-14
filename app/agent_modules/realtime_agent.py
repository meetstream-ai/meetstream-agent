"""Realtime agent using OpenAI Realtime API - no MCPs."""
from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

from agents.realtime import RealtimeAgent
from .tools import get_realtime_tools

logger = logging.getLogger(__name__)


@dataclass
class RealtimeAgentConfig:
    """Configuration for realtime agent."""
    name: str = "Meetstream Realtime Agent"
    instructions: str = """You are a realtime meeting assistant.

Prefer tools when available:
- Use `current_time` for time questions.
- Use `weather_now` for current weather.

Keep spoken responses concise and avoid repeating prior text verbatim."""
    handoff_description: str = "Realtime agent with local tools (no MCPs)."


def get_realtime_agent(config: Optional[RealtimeAgentConfig] = None) -> RealtimeAgent:
    """
    Create and return a realtime agent without MCPs.
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        
    Returns:
        RealtimeAgent instance with local tools only (no MCPs).
    """
    if config is None:
        config = RealtimeAgentConfig()
    
    tools = get_realtime_tools()
    
    agent = RealtimeAgent(
        name=config.name,
        handoff_description=config.handoff_description,
        instructions=config.instructions,
        tools=tools,
        # Explicitly no MCP servers
        mcp_servers=[],
    )
    
    logger.info(f"Created realtime agent: {config.name} (no MCPs)")
    return agent

