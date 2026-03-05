# RAG Local para Codebase — MCP Server

RAG local com ChromaDB (Docker) + embeddings `all-MiniLM-L6-v2` (CPU) integrado ao Claude Code CLI via MCP.

## Estrutura

```
my-custom-rag-python/
├── rag-setup.run       # instalador auto-suficiente (copie para qualquer projeto)
├── chroma_monitor.sh   # monitor do banco em tempo real
├── README.md
└── bin/                # fontes e scripts de suporte
    ├── build_run.sh    # reconstrói o rag-setup.run
    ├── docker-compose.yml
    ├── indexer_full.py
    ├── init.sh
    ├── mcp_server.py
    └── requirements.txt
```

## Uso rápido

Copie o `rag-setup.run` para o projeto que quer indexar e execute:

```bash
./rag-setup.run [caminho/do/projeto] [flags]
```

| Flag | O que faz |
|---|---|
| *(sem flags)* | instala tudo + indexa o diretório atual |
| `/caminho/projeto` | instala tudo + indexa o caminho informado |
| `--skip-index` | instala a infra sem indexar |
| `--only-index` | só indexa (infra já instalada) |
| `--reinstall` | força reinstalação completa |

O instalador é idempotente: detecta o que já está instalado e pula.

## O que o setup faz

1. Cria venv em `~/.rag_venv` e instala as dependências Python
2. Levanta o ChromaDB via Docker (`restart: always` — sobe com o SO)
3. Instala o `mcp-rag-server` em `~/.local/bin/` com shebang do venv
4. Indexa o projeto com barra de progresso

## Configuração do Claude Code

Adicione em `~/.claude.json` dentro de `"mcpServers"`:

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

Reinicie o Claude Code CLI para carregar o servidor.

## Ferramentas MCP disponíveis

| Ferramenta | Descrição |
|---|---|
| `semantic_search_code` | busca semântica na codebase indexada |
| `update_file_index` | reindexa um arquivo após edição |
| `delete_file_index` | remove um arquivo do índice |
| `index_specific_folder` | indexa uma subpasta sob demanda |

## Monitor do banco

```bash
./chroma_monitor.sh           # menu interativo
./chroma_monitor.sh chunks    # contagem de chunks (snapshot)
./chroma_monitor.sh watch     # chunks em tempo real
./chroma_monitor.sh disk      # tamanho no disco
./chroma_monitor.sh logs      # logs HTTP do container
./chroma_monitor.sh full      # painel completo
```

## Reconstruir o instalador

Se editar arquivos em `bin/`, regenere o `rag-setup.run`:

```bash
./bin/build_run.sh
```
