"""Non-realtime agent using Langchain with MCP support."""
from __future__ import annotations

import logging
import asyncio
from typing import Optional, List, Any, Dict
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Langchain components
_HAS_LANGCHAIN = False
_AgentExecutor = None
_create_openai_tools_agent = None
BaseTool = None
tool = None
ChatOpenAI = None
ChatPromptTemplate = None
MessagesPlaceholder = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import BaseTool, tool
    
    # Try to import AgentExecutor and create_openai_tools_agent
    # These might be in different locations depending on Langchain version
    try:
        # Standard import for Langchain 0.3+
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        _AgentExecutor = AgentExecutor
        _create_openai_tools_agent = create_openai_tools_agent
    except ImportError:
        # Try importing from agent_executor module directly
        try:
            from langchain.agents.agent_executor import AgentExecutor as _AgentExecutor_impl
            from langchain.agents.create_openai_tools_agent import create_openai_tools_agent as _create_impl
            _AgentExecutor = _AgentExecutor_impl
            _create_openai_tools_agent = _create_impl
        except ImportError:
            # Try alternative structure (some versions use different paths)
            try:
                from langchain.agents.executor import AgentExecutor as _AgentExecutor_impl
                from langchain.agents import create_openai_tools_agent as _create_impl
                _AgentExecutor = _AgentExecutor_impl
                _create_openai_tools_agent = _create_impl
            except ImportError:
                raise ImportError(
                    "Could not import AgentExecutor from langchain.agents. "
                    "Please ensure you have 'langchain>=0.3.0' installed. "
                    "Try: uv sync to update dependencies."
                )
    
    if _AgentExecutor is not None and _create_openai_tools_agent is not None:
        _HAS_LANGCHAIN = True
    else:
        raise ImportError("AgentExecutor or create_openai_tools_agent could not be imported")
        
except ImportError as e:
    _HAS_LANGCHAIN = False
    logger.warning(
        f"Langchain not installed or incompatible version. Non-realtime agent will not work. "
        f"Error: {e}. Please ensure 'langchain>=0.3.0' is installed. Run: uv sync"
    )

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
    model: str = os.getenv("NON_REALTIME_MODEL", "gpt-4o")
    temperature: float = 0.7
    instructions: str = """You are a meeting assistant that helps with tasks and questions.

Prefer tools when available:
- Use `current_time` for time questions.
- Use `weather_now` for current weather.
- Use MCP tools for extended functionality (Canva, n8n, etc.).

Provide clear, concise responses."""
    max_iterations: int = 15


def create_langchain_tools() -> List[Any]:
    """Create Langchain tools from local functions."""
    if not _HAS_LANGCHAIN or tool is None:
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


# Define MCPToolWrapper only if BaseTool is available
if _HAS_LANGCHAIN and BaseTool is not None:
    class MCPToolWrapper(BaseTool):
        """Wrapper to convert MCP tools to Langchain tools."""
        
        def __init__(self, mcp_tool: Dict, session: Any, server_name: str):
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
else:
    # Dummy class if Langchain is not available
    class MCPToolWrapper:
        pass


# Global storage for MCP sessions to keep them alive
_mcp_sessions: Dict[str, Dict[str, Any]] = {}
_mcp_tool_registry: Dict[str, List[str]] = {}


async def create_mcp_tools(mcp_configs: List[dict]) -> List[Any]:
    """
    Create Langchain tools from MCP servers using the MCP SDK.
    
    This implementation connects to MCP servers and wraps their tools as Langchain tools.
    Sessions are kept alive globally to ensure tools remain available.
    """
    if not _HAS_MCP or not _HAS_LANGCHAIN:
        logger.warning("MCP or Langchain not available, skipping MCP tools")
        return []
    
    if BaseTool is None or tool is None:
        logger.warning("BaseTool or tool decorator not available")
        return []
    
    tools: List[BaseTool] = []
    
    for config in mcp_configs:
        try:
            if config["type"] == "stdio":
                server_name = config['name']
                logger.info(f"Connecting to MCP server: {server_name} (stdio)")
                
                try:
                    # Create stdio server parameters
                    server_params = StdioServerParameters(
                        command=config["command"],
                        args=config.get("args", []),
                        env=config.get("env", {}),
                    )
                    
                    # Create stdio client and session (keep them alive)
                    # Use context manager but extract the session before exiting
                    stdio_ctx = stdio_client(server_params)
                    read, write = await stdio_ctx.__aenter__()
                    session_ctx = ClientSession(read, write)
                    session = await session_ctx.__aenter__()
                    
                    # Initialize the session
                    await session.initialize()
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    logger.info(f"MCP server '{server_name}' has {len(tools_result.tools)} tools available")
                    
                    # Store session and clients to keep them alive (don't exit context managers)
                    _mcp_sessions[server_name] = {
                        "session": session,
                        "session_ctx": session_ctx,
                        "stdio_ctx": stdio_ctx,
                        "read": read,
                        "write": write
                    }
                    _mcp_tool_registry[server_name] = []
                    
                    # Create Langchain tool wrappers for each MCP tool
                    for mcp_tool in tools_result.tools:
                        tool_name = mcp_tool.name
                        tool_desc = mcp_tool.description or f"Tool from {server_name}"
                        
                        # Capture session, server, and tool details in closure
                        def make_tool_func(sess, svr_name, t_name, desc):
                            @tool(name=f"{svr_name}_{t_name}", description=desc)
                            async def mcp_tool_wrapper(**kwargs):
                                try:
                                    result = await sess.call_tool(t_name, arguments=kwargs)
                                    # Extract content from result
                                    if isinstance(result, dict):
                                        if "content" in result:
                                            content = result["content"]
                                            if isinstance(content, list):
                                                return "\n".join(
                                                    str(c.get("text", c))
                                                    for c in content
                                                    if isinstance(c, dict)
                                                )
                                            return str(content)
                                        return str(result)
                                    return str(result)
                                except Exception as e:
                                    logger.error(f"Error calling MCP tool {t_name}: {e}")
                                    return f"Error: {str(e)}"
                            return mcp_tool_wrapper
                        
                        mcp_wrapper = make_tool_func(session, server_name, tool_name, tool_desc)
                        tools.append(mcp_wrapper)
                        _mcp_tool_registry[server_name].append(tool_name)
                        logger.info(f"Added MCP tool: {server_name}_{tool_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to connect to MCP server '{server_name}': {e}", exc_info=True)
                    continue
                
            elif config["type"] == "sse":
                logger.warning(f"SSE MCP servers not yet implemented: {config['name']}")
                # TODO: Implement SSE MCP client setup
                
        except Exception as e:
            logger.error(f"Failed to create MCP tools for {config.get('name', 'unknown')}: {e}", exc_info=True)
    
    if tools:
        logger.info(f"✅ Successfully loaded {len(tools)} MCP tools from {len([c for c in mcp_configs if c.get('type') == 'stdio'])} servers")
    else:
        logger.warning("⚠️ No MCP tools loaded. Check your MCP configuration.")
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
    if _create_openai_tools_agent is None:
        raise ImportError("create_openai_tools_agent is not available. Langchain agents module not properly loaded.")
    agent = _create_openai_tools_agent(llm, all_tools, prompt)
    
    # Create executor
    if _AgentExecutor is None:
        raise ImportError("AgentExecutor is not available. Langchain agents module not properly loaded.")
    agent_executor = _AgentExecutor(
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
    if _create_openai_tools_agent is None:
        raise ImportError("create_openai_tools_agent is not available. Langchain agents module not properly loaded.")
    agent = _create_openai_tools_agent(llm, all_tools, prompt)
    
    # Create executor
    if _AgentExecutor is None:
        raise ImportError("AgentExecutor is not available. Langchain agents module not properly loaded.")
    agent_executor = _AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=config.max_iterations,
    )
    
    logger.info(f"Created non-realtime agent: {config.name} with {len(all_tools)} tools ({len(mcp_tools)} MCP tools)")
    return agent_executor


def get_mcp_connection_status() -> Dict[str, Any]:
    """Return MCP connection status for debugging/inspection."""
    servers = []
    for name, session_info in _mcp_sessions.items():
        servers.append({
            "name": name,
            "tool_count": len(_mcp_tool_registry.get(name, [])),
            "tools": list(_mcp_tool_registry.get(name, [])),
            "connected": True,
        })
    return {
        "connected_servers": len(_mcp_sessions),
        "servers": servers,
    }
