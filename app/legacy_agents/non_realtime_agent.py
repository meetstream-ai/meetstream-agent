"""Non-realtime agent using Langchain with MCP support."""
from __future__ import annotations

import logging
import asyncio
from typing import Optional, List, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Langchain components
try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import BaseTool, tool
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    logger.warning("Langchain not installed. Non-realtime agent will not work.")

# Try to import MCP integration
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
    logger.warning("MCP SDK not available. MCP tools will not work.")

from .tools import current_time, weather_now
from .mcp_setup import get_mcp_servers_config


@dataclass
class NonRealtimeAgentConfig:
    """Configuration for non-realtime agent."""
    name: str = "Meetstream Non-Realtime Agent"
    model: str = "gpt-4o"
    temperature: float = 0.7
    instructions: str = """You are a meeting assistant that helps with tasks and questions.

Prefer tools when available:
- Use `current_time` for time questions.
- Use `weather_now` for current weather.
- Use MCP tools for extended functionality (Canva, n8n, etc.).

Provide clear, concise responses."""
    max_iterations: int = 15


def create_langchain_tools() -> List[BaseTool]:
    """Create Langchain tools from local functions."""
    if not _HAS_LANGCHAIN:
        return []
    
    @tool
    def get_current_time(timezone_name: Optional[str] = None) -> str:
        """Return the current time in ISO 8601. Optional timezone as an IANA string, e.g., 'America/Toronto'."""
        return current_time(timezone_name)
    
    @tool
    async def get_weather_now(city: str) -> str:
        """Get current weather for a city using Open-Meteo (no API key)."""
        return await weather_now(city)
    
    return [get_current_time, get_weather_now]


class MCPToolWrapper(BaseTool):
    """Wrapper to convert MCP tools to Langchain tools."""
    
    def __init__(self, mcp_tool: Dict, session: ClientSession, server_name: str):
        """Initialize MCP tool wrapper."""
        self.mcp_tool = mcp_tool
        self.session = session
        self.server_name = server_name
        super().__init__(
            name=f"{server_name}_{mcp_tool.get('name', 'unknown')}",
            description=mcp_tool.get('description', ''),
        )
    
    def _run(self, *args, **kwargs) -> str:
        """Synchronous run (not supported for MCP)."""
        raise NotImplementedError("MCP tools require async execution")
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async run for MCP tool."""
        try:
            # Call MCP tool
            result = await self.session.call_tool(
                self.mcp_tool['name'],
                arguments=kwargs if kwargs else {}
            )
            # Format result
            if isinstance(result, dict):
                return str(result.get('content', result))
            return str(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.name}: {e}")
            return f"Error: {str(e)}"


async def create_mcp_tools(mcp_configs: List[dict]) -> List[BaseTool]:
    """
    Create Langchain tools from MCP servers.
    
    Note: This is a simplified implementation. MCP session management is complex
    and requires careful lifecycle management. This implementation loads MCP tools
    but session management may need improvements for production use.
    
    For now, this serves as a foundation that can be extended.
    """
    if not _HAS_MCP or not _HAS_LANGCHAIN:
        logger.warning("MCP or Langchain not available, skipping MCP tools")
        return []
    
    tools: List[BaseTool] = []
    
    for config in mcp_configs:
        try:
            if config["type"] == "stdio":
                logger.info(f"Setting up MCP server: {config['name']} (stdio)")
                # TODO: Implement proper MCP stdio client integration
                # This requires careful session management to keep connections alive
                # For now, we'll skip MCP tools and log a warning
                logger.warning(f"MCP stdio integration not yet fully implemented for {config['name']}")
                
            elif config["type"] == "sse":
                logger.warning(f"SSE MCP servers not yet implemented: {config['name']}")
                # TODO: Implement SSE MCP client setup
                
        except Exception as e:
            logger.error(f"Failed to create MCP tools for {config.get('name', 'unknown')}: {e}")
    
    # Return empty list for now - MCP integration can be added later
    # The structure is in place, but proper session management is needed
    return tools


def get_non_realtime_agent(config: Optional[NonRealtimeAgentConfig] = None) -> Any:
    """
    Create and return a non-realtime agent using Langchain with MCP support.
    
    Note: This is a synchronous wrapper. For proper async MCP support, use get_non_realtime_agent_async.
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        
    Returns:
        AgentExecutor instance with MCP tools.
    """
    if not _HAS_LANGCHAIN:
        raise ImportError("Langchain is required for non-realtime agent. Install with: pip install langchain langchain-openai")
    
    if config is None:
        config = NonRealtimeAgentConfig()
    
    # Create local tools
    local_tools = create_langchain_tools()
    
    # Get MCP server configs
    mcp_configs = get_mcp_servers_config()
    
    # Create MCP tools (async, so we need to run it in an event loop)
    # For now, return agent without MCP tools (can be added later)
    all_tools = local_tools
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.instructions),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, all_tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=config.max_iterations,
    )
    
    logger.info(f"Created non-realtime agent: {config.name} with {len(all_tools)} tools")
    return agent_executor


async def get_non_realtime_agent_async(config: Optional[NonRealtimeAgentConfig] = None) -> Any:
    """
    Async version that properly loads MCP tools.
    
    Args:
        config: Optional configuration. Uses defaults if not provided.
        
    Returns:
        AgentExecutor instance with MCP tools.
    """
    if not _HAS_LANGCHAIN:
        raise ImportError("Langchain is required for non-realtime agent. Install with: pip install langchain langchain-openai")
    
    if config is None:
        config = NonRealtimeAgentConfig()
    
    # Create local tools
    local_tools = create_langchain_tools()
    
    # Get MCP server configs
    mcp_configs = get_mcp_servers_config()
    
    # Create MCP tools
    mcp_tools = await create_mcp_tools(mcp_configs)
    
    # Combine all tools
    all_tools = local_tools + mcp_tools
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.instructions),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, all_tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=config.max_iterations,
    )
    
    logger.info(f"Created non-realtime agent: {config.name} with {len(all_tools)} tools ({len(mcp_tools)} MCP tools)")
    return agent_executor
