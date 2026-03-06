# Local RAG for Codebase — MCP Server

Local RAG with ChromaDB (Docker) + `all-MiniLM-L6-v2` embeddings (CPU) integrated into Claude Code CLI via MCP.

(link README.md PT-BR: https://github.com/JocsaPB/my-won-rag/blob/dee053edc0dc4a10e3aa5c05fb83a5819a174944/README.md)

## Structure

```
my-custom-rag-python/
├── rag-setup.run       # self-contained installer (copy to any project)
├── chroma_monitor.sh   # real-time database monitor
├── README.md
└── bin/                # source files and support scripts
    ├── build_run.sh    # rebuilds rag-setup.run
    ├── docker-compose.yml
    ├── indexer_full.py
    ├── init.sh
    ├── mcp_server.py
    └── requirements.txt
```

## Quick start

Copy `rag-setup.run` to the project you want to index and run:

```bash
./rag-setup.run [path/to/project] [flags]
```

| Flag | What it does |
|---|---|
| *(no flags)* | installs everything + indexes current directory |
| `/path/to/project` | installs everything + indexes the specified path |
| `--skip-index` | installs infrastructure without indexing |
| `--only-index` | indexes only (infrastructure already installed) |
| `--reinstall` | forces complete reinstallation |

The installer is idempotent: it detects what is already installed and skips it.

## What the setup does

1. Creates venv at `~/.rag_venv` and installs Python dependencies
2. Starts ChromaDB via Docker (`restart: always` — starts with the OS)
3. Installs `mcp-rag-server` in `~/.local/bin/` with venv shebang
4. Indexes the project with progress bar

## Claude Code configuration

Add to `~/.claude.json` inside `"mcpServers"`:

```json
"rag-codebase": {
  "command": "~/.local/bin/mcp-rag-server",
  "args": [],
  "env": {
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": "8000",
    "MCP_MODEL_DIR": "<PROJECT_ROOT>/model"
  }
}
```

Restart Claude Code CLI to load the server.

### Model download + fallback (Hugging Face -> local `model/`)

On startup, `mcp-rag-server` now:
1. Tries to download/update `BAAI/bge-m3` from Hugging Face API.
2. Saves/updates the files in `model/` (or path from `MCP_MODEL_DIR`).
3. If Hugging Face is unavailable, it loads the model from local `model/` as fallback.

## Available MCP tools

| Tool | Description |
|---|---|
| `semantic_search_code` | semantic search in the indexed codebase |
| `update_file_index` | re-indexes a file after editing |
| `delete_file_index` | removes a file from the index |
| `index_specific_folder` | indexes a subfolder on demand |

## Database monitor

```bash
./chroma_monitor.sh           # interactive menu
./chroma_monitor.sh chunks    # chunk count (snapshot)
./chroma_monitor.sh watch     # real-time chunks
./chroma_monitor.sh disk      # disk size
./chroma_monitor.sh logs      # HTTP logs from container
./chroma_monitor.sh mcp-logs  # real-time MCP usage logs (actor + tool + status)
./chroma_monitor.sh mcp-summary # aggregated MCP usage (top tools/actors, 24h)
./chroma_monitor.sh full      # full dashboard
```

### MCP usage telemetry

`mcp-rag-server` now writes structured usage logs to:

```bash
~/.rag_db/mcp_usage.log
```

Each entry includes:
- timestamp
- actor (who accessed)
- MCP tool used
- event (`tool_call_start` / `tool_call_end`)
- status and details (when available)
- client process metadata

This helps validate whether MCP is effectively used by tools.

## Restart `mcp-rag-server`

After changing `bin/mcp_server.py`, restart the server so changes take effect.

1. Stop current process(es):

```bash
pgrep -f "$HOME/.rag_venv/bin/python3 $HOME/.local/bin/mcp-rag-server" | xargs -r kill
```

2. Start it again in background:

```bash
nohup "$HOME/.local/bin/mcp-rag-server" >/tmp/mcp-rag-server.out 2>/tmp/mcp-rag-server.err < /dev/null &
```

3. Verify:

```bash
ps -ef | rg "mcp-rag-server|bin/mcp_server.py"
```

If you use Claude Code CLI as the MCP host, restarting the CLI also reloads the server process.

## Rebuild the installer

If you edit files in `bin/`, regenerate `rag-setup.run`:

```bash
./bin/build_run.sh
```
