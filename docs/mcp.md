# MCP servers (`mcp.config.json`)

The bridge loads MCP servers from **`mcp.config.json`** in the repository root (or from the path in **`MCP_CONFIG`**). This is the default way to register **Canva** and any other servers.

For agent personality, local tools, and model-provider notes, see [README.agents.md](../README.agents.md).

---

## Default: Canva (shipped sample)

This repo includes **`mcp.config.json`** with a **`canva`** entry that runs Canva’s MCP over **`npx mcp-remote`**.

**Requirements**

- [Node.js](https://nodejs.org/) installed so **`npx`** is on your `PATH` (same environment as `uv run uvicorn`).

**Disable Canva without removing the file**

- Delete the `canva` block under `mcpServers`, or
- Rename `mcp.config.json` so the loader falls back to env-based defaults in `app/agent.py` (Framer/n8n only if their env vars are set).

**Override the config path**

```bash
export MCP_CONFIG=/path/to/other-mcp.config.json
```

Restart the bridge after any change.

---

## How loading works

1. Load **`mcp.config.json`** (or **`MCP_CONFIG`**). If **`mcpServers`** is non-empty → use those servers (e.g. **Canva**).
2. If the file is missing or **`mcpServers`** is empty → **`build_mcp_servers_default()`** (Framer/n8n when env vars are set).
3. **Merge:** if **`DOCKER_MCP_ENABLED=1`** and **`DOCKER_MCP_URL`** + **`DOCKER_MCP_BEARER_TOKEN`** are set, a **streamable HTTP** MCP named **`docker`** is **appended** (e.g. local Docker MCP exposed via ngrok). Turn off with **`DOCKER_MCP_ENABLED=0`** or unset.
4. On each Realtime session, **`app/realtime/mcp.py`** connects all configured servers.
5. **`_build_agent_instructions()`** mentions Canva/n8n/Framer/**docker** when a matching server **name** is present.

---

## Optional: Docker MCP (streamable HTTP + Bearer token)

For an MCP served over **HTTP streamable** (for example a **Docker** container reached through **ngrok**):

1. Copy **`.env.example`** → **`.env`** and set:

   ```env
   DOCKER_MCP_ENABLED=1
   DOCKER_MCP_URL=https://your-subdomain.ngrok-free.app/mcp
   DOCKER_MCP_BEARER_TOKEN=your-secret-token
   ```

   Put **only the secret** in `DOCKER_MCP_BEARER_TOKEN` — do **not** prefix with the word `Bearer`; the bridge adds that.

2. Use the **https** URL ngrok gives you; path is usually **`/mcp`** (match whatever your server exposes).

3. The bridge sends **`Authorization: Bearer <token>`** on every request to that MCP.

4. Restart the bridge. Logs should list **`docker`** next to **`canva`** (if Canva is still in `mcp.config.json`).

5. **Disable:** `DOCKER_MCP_ENABLED=0` or remove these variables.

Optional: **`DOCKER_MCP_SESSION_TIMEOUT`** (seconds, default `120`) for the MCP client session.

**Security:** Keep the Bearer token only in **`.env`** (gitignored). Do not commit real URLs or tokens.

---

## Expand the config

Add more entries under **`mcpServers`**. Each key is the **server name** exposed to the agent.

### `stdio` (local process, e.g. `npx`, `python`)

```json
"my_tool": {
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "some-mcp-package@latest"],
  "env": { "PATH": "$PATH" },
  "timeout": 90
}
```

- **`timeout`**: session timeout in seconds (optional; default `60` in the loader).

### `sse` (HTTP + Server-Sent Events)

```json
"design_sse": {
  "type": "sse",
  "url": "https://example.com/mcp/sse",
  "headers": { "Authorization": "Bearer $MY_TOKEN" }
}
```

Environment variables in strings are expanded with `os.path.expandvars` where supported.

### `streamable_http` (also `stream` or `http`)

```json
"internal_api": {
  "type": "streamable_http",
  "url": "https://api.example.com/mcp",
  "headers": { "Authorization": "Bearer $SEARCH_TOKEN" }
}
```

Implementation: `build_mcp_servers_from_config()` in **`app/agent.py`**.

---

## Verify

- Server logs: MCP names and connect success or errors.
- If the model calls a missing tool, ensure the **server name** and actual tools match, and you did not remove that server from JSON while the prompt still implied it.

---

## See also

- [README.agents.md](../README.agents.md) — agent tuning, Framer/n8n env-only defaults, provider swaps.
- [OpenAI Agents SDK — MCP](https://openai.github.io/openai-agents-python/mcp/) — transports and tool lists.
