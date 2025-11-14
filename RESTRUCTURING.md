# Codebase Restructuring Summary

## Overview

The codebase has been restructured to separate realtime and non-realtime agents, making it easier to add MCPs to non-realtime agents while keeping realtime agents clean and focused.

## Key Changes

### 1. New Directory Structure

```
app/
├── agent_modules/
│   ├── __init__.py              # Module exports
│   ├── realtime_agent.py        # Realtime agent (no MCPs)
│   ├── non_realtime_agent.py    # Langchain-based non-realtime agent (with MCP support)
│   ├── tools.py                 # Shared local tools
│   ├── mcp_setup.py             # MCP server configuration
│   └── README.md                # Agent module documentation
├── agent.py                     # Legacy file (deprecated, kept for backward compatibility)
└── server.py                    # Updated to use new agent structure
```

Note: The directory is named `agent_modules` (not `agents`) to avoid shadowing the installed `agents` package.

### 2. Realtime Agents

- **Location**: `app/agent_modules/realtime_agent.py`
- **MCPs**: Removed (explicitly no MCP servers)
- **Tools**: Local tools only (`current_time`, `weather_now`)
- **Usage**: 
  ```python
  from agent_modules import get_realtime_agent
  agent = get_realtime_agent()
  ```

### 3. Non-Realtime Agents

- **Location**: `app/agent_modules/non_realtime_agent.py`
- **Framework**: Langchain-based
- **MCPs**: Structure in place for MCP integration (can be extended)
- **Tools**: Local tools + MCP tools (when implemented)
- **Usage**:
  ```python
  from agent_modules import get_non_realtime_agent_async
  agent = await get_non_realtime_agent_async()
  ```

### 4. Shared Tools

- **Location**: `app/agent_modules/tools.py`
- **Purpose**: Common tools used by both agent types
- **Tools**: `current_time`, `weather_now`

### 5. MCP Configuration

- **Location**: `app/agent_modules/mcp_setup.py`
- **Purpose**: MCP server configuration for non-realtime agents
- **Config Sources**:
  - Environment variable `MCP_CONFIG` (JSON file path)
  - Default configuration in code

### 6. Server Updates

- **Location**: `app/server.py`
- **Changes**:
  - Updated to use new agent structure
  - Removed MCP preconnect logic (realtime agents don't use MCPs)
  - Maintains backward compatibility with legacy `agent.py`

### 7. Legacy Support

- **Location**: `app/agent.py`
- **Status**: Deprecated but kept for backward compatibility
- **Function**: Wrapper that uses new realtime agent structure
- **Note**: Shows deprecation warning on first use

## Benefits

1. **Separation of Concerns**: Realtime and non-realtime agents are clearly separated
2. **MCP Integration**: Easier to add MCPs to non-realtime agents (Langchain-based)
3. **Maintainability**: Cleaner structure makes it easier to maintain and extend
4. **Backward Compatibility**: Legacy code still works
5. **Flexibility**: Easy to add new agent types or modify existing ones

## Migration Guide

### For Realtime Agents

**Before**:
```python
from agent import get_starting_agent
agent = get_starting_agent()  # Had MCPs
```

**After**:
```python
from agent_modules import get_realtime_agent
agent = get_realtime_agent()  # No MCPs
```

### For Non-Realtime Agents

**New**:
```python
from agent_modules import get_non_realtime_agent_async, NonRealtimeAgentConfig

config = NonRealtimeAgentConfig(
    model="gpt-4o",
    temperature=0.7,
)
agent = await get_non_realtime_agent_async(config)
result = await agent.ainvoke({"input": "What's the weather in Toronto?"})
```

## Next Steps

1. **MCP Integration**: Complete MCP integration for non-realtime agents
   - Implement proper session management
   - Add support for SSE MCP servers
   - Test with various MCP servers (Canva, n8n, etc.)

2. **Non-Realtime Endpoints**: Add API endpoints for non-realtime agents
   - Create HTTP endpoints for non-realtime agent queries
   - Add request/response handling
   - Integrate with Meetstream API if needed

3. **Testing**: Add tests for new agent structure
   - Unit tests for agents
   - Integration tests for server
   - MCP integration tests

4. **Documentation**: Expand documentation
   - API documentation
   - Usage examples
   - MCP configuration guide

## Dependencies

New dependencies added:
- `langchain>=0.3.0`
- `langchain-openai>=0.2.0`
- `langchain-core>=0.3.0`
- `mcp>=1.0.0`
- `httpx>=0.27.0`
- `numpy>=1.24.0`

## Notes

- Realtime agents are now MCP-free, making them simpler and faster
- Non-realtime agents use Langchain, which has better tool integration support
- MCP integration structure is in place but needs completion for production use
- The structure is designed to be easily extensible

