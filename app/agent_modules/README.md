# Agents Module

This module contains the restructured agent implementations for realtime and non-realtime processing.

## Structure

- **realtime_agent.py**: Realtime agent using OpenAI Realtime API (no MCPs)
- **non_realtime_agent.py**: Non-realtime agent using Langchain (with MCP support)
- **tools.py**: Shared local tools for both agent types
- **mcp_setup.py**: MCP server configuration for non-realtime agents

## Usage

### Realtime Agent

```python
from agent_modules import get_realtime_agent

# Get realtime agent (no MCPs)
agent = get_realtime_agent()
```

### Non-Realtime Agent

```python
from agent_modules import get_non_realtime_agent_async, NonRealtimeAgentConfig

# Get non-realtime agent with MCPs
config = NonRealtimeAgentConfig(
    model="gpt-4o",
    temperature=0.7,
)
agent = await get_non_realtime_agent_async(config)

# Use the agent
result = await agent.ainvoke({"input": "What's the weather in Toronto?"})
```

## MCP Integration

MCPs are only available for non-realtime agents (Langchain-based). This makes it easier to add and manage MCPs since Langchain has better tool integration support.

### MCP Configuration

MCP servers are configured via:
1. Environment variable `MCP_CONFIG` pointing to a JSON config file
2. Default configuration in `mcp_setup.py`

### Adding MCPs

To add MCPs to non-realtime agents:

1. Configure MCP servers in `mcp.config.json` or via environment variables
2. The non-realtime agent will automatically load and use MCP tools

Example `mcp.config.json`:
```json
{
  "mcpServers": {
    "canva": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "mcp-remote@latest", "https://mcp.canva.com/mcp"]
    },
    "framer": {
      "type": "sse",
      "url": "https://your-framer-mcp-url.com"
    }
  }
}
```

## Migration Notes

- **Realtime agents**: MCPs have been removed. Use non-realtime agents for MCP functionality.
- **Non-realtime agents**: Use Langchain-based agents for MCP support.
- **Legacy agent.py**: Still works for backward compatibility but is deprecated.

