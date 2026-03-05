# Local RAG for Codebase — MCP Server

Local RAG with ChromaDB (Docker) + `all-MiniLM-L6-v2` embeddings (CPU) integrated into Claude Code CLI via MCP.

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
  "command": "/home/jocsa/.local/bin/mcp-rag-server",
  "args": [],
  "env": {
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": "8000"
  }
}
```

Restart Claude Code CLI to load the server.

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
./chroma_monitor.sh full      # full dashboard
```

## Rebuild the installer

If you edit files in `bin/`, regenerate `rag-setup.run`:

```bash
./bin/build_run.sh
```
