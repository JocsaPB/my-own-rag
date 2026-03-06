# MCP binary checksum (SHA-256, payload without shebang): `ea30fb3af9e79b7f98fb12e445a5da0d6d4761b6f50215db775b6a3c3765af95` | Verify: `tail -n +2 ~/.local/bin/mcp-rag-server | sha256sum`

# Local RAG for Codebase — MCP Server

Local RAG with ChromaDB (Docker) + `jinaai/jina-embeddings-v3` embeddings (CPU) integrated into Claude Code CLI via MCP.

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
| `--chage-model` or `-cg` | resets the environment and runs a fresh setup to switch embedding model |

The installer is idempotent and now always asks about MCP refresh in interactive mode.
- If `mcp-rag-server` already exists, setup checks whether it is up to date.
- On every run, setup asks if you want to reinstall/update `mcp-rag-server`.
- Setup preserves the current MCP version configured in your client and does not auto-increment it during `.run` execution.
- If HF token is present but invalid, setup/download flow allows entering a new token or continuing without token.

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
    "MCP_MODEL_DIR": "~/.cache/my-custom-rag-python/models",
    "MCP_EMBEDDING_MODEL": "jina",
    "MCP_JINA_QUANTIZATION": "default"
  }
}
```

Restart Claude Code CLI to load the server.

### Model download + fallback (Hugging Face -> local cache)

On startup, `mcp-rag-server` now:
1. Tries to download/update `jinaai/jina-embeddings-v3` from Hugging Face API.
2. Saves/updates the files in `~/.cache/my-custom-rag-python/models` (or path from `MCP_MODEL_DIR`).
3. If it fails, it falls back to `BAAI/bge-m3` and tries providers by priority.

### Choosing model and quantization

You can choose embedding model and Jina quantization through environment variables:

```bash
MCP_EMBEDDING_MODEL=jina|bge
MCP_JINA_QUANTIZATION=default|dynamic-int8
```

Recommendation:
- `jina` (`jinaai/jina-embeddings-v3`): better performance for code-only projects.
- `bge` (`BAAI/bge-m3`): better for mixed content projects (code + documentation).

Jina quantization options (CPU):
- `default`: no quantization, best quality, slower indexing.
- `dynamic-int8`: faster indexing and lower RAM usage, with small quality tradeoff.

Notes:
- `MCP_JINA_QUANTIZATION` is only applied when `MCP_EMBEDDING_MODEL=jina`.
- Default behavior is `MCP_EMBEDDING_MODEL=jina` and `MCP_JINA_QUANTIZATION=default`.

For standalone indexing (`bin/indexer_full.py`), you can also pass flags:

```bash
python3 bin/indexer_full.py . --embedding-model jina --jina-quantization dynamic-int8
python3 bin/indexer_full.py . --embedding-model bge
```

If no flags/env are provided and the command is run in a terminal, the script prompts the user to choose.

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
