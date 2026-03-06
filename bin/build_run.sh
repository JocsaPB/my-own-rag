#!/usr/bin/env bash
# build_run.sh — Gera o arquivo rag-setup.run auto-suficiente.
# Execute este script uma vez para (re)construir o .run.
# O .run gerado pode ser copiado para qualquer projeto e executado lá.

set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${BIN_DIR}")"
OUTPUT="${ROOT_DIR}/rag-setup.run"

COMPOSE_FILE="${BIN_DIR}/docker-compose.yml"
REQUIREMENTS_FILE="${BIN_DIR}/requirements.txt"
INDEXER_FILE="${BIN_DIR}/indexer_full.py"
MCP_FILE="${BIN_DIR}/mcp_server.py"

for f in "$COMPOSE_FILE" "$REQUIREMENTS_FILE" "$INDEXER_FILE" "$MCP_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERRO] Arquivo não encontrado: $f"
        exit 1
    fi
done

echo "[+] Gerando base64 dos arquivos..."

B64_COMPOSE=$(base64 -w0 "$COMPOSE_FILE")
B64_REQUIREMENTS=$(base64 -w0 "$REQUIREMENTS_FILE")
B64_INDEXER=$(base64 -w0 "$INDEXER_FILE")
B64_MCP=$(base64 -w0 "$MCP_FILE")

echo "[+] Escrevendo $OUTPUT ..."

cat > "$OUTPUT" << OUTER_EOF
#!/usr/bin/env bash
# =============================================================================
# rag-setup.run — Instalador auto-suficiente do RAG local com ChromaDB + MCP
# =============================================================================
# Gerado automaticamente por build_run.sh
# Versão: $(date '+%Y-%m-%d %H:%M')
#
# Uso:
#   chmod +x rag-setup.run
#   ./rag-setup.run                # instala tudo e indexa o diretório atual
#   ./rag-setup.run --skip-index   # instala sem indexar
#   ./rag-setup.run --only-index   # apenas indexa (infra já instalada)
#   ./rag-setup.run --reinstall    # força reinstalação completa
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
SKIP_INDEX=false
ONLY_INDEX=false
REINSTALL=false
CUSTOM_PROJECT_DIR=""

for arg in "\$@"; do
    case "\$arg" in
        --skip-index)  SKIP_INDEX=true ;;
        --only-index)  ONLY_INDEX=true ;;
        --reinstall)   REINSTALL=true ;;
        --help|-h)
            echo "Uso: \$0 [caminho/do/projeto] [--skip-index] [--only-index] [--reinstall]"
            exit 0 ;;
        -*)
            echo "Opção desconhecida: \$arg. Use --help para ver as opções."
            exit 1 ;;
        *)
            # Argumento posicional = caminho do projeto
            CUSTOM_PROJECT_DIR="\$arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

log_info()    { echo -e "\${GREEN}[+]\${NC} \$*"; }
log_warn()    { echo -e "\${YELLOW}[!]\${NC} \$*"; }
log_error()   { echo -e "\${RED}[ERRO]\${NC} \$*" >&2; }
log_section() { echo -e "\n\${BLUE}\${BOLD}==> \$*\${NC}"; }

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
USER_HOME="\${HOME}"
VENV_DIR="\${USER_HOME}/.rag_venv"
VENV_PYTHON="\${VENV_DIR}/bin/python3"
VENV_PIP="\${VENV_DIR}/bin/pip"
DOCKER_COMPOSE_DIR="\${USER_HOME}/docker-chromadb"
RAG_DB_DIR="\${USER_HOME}/.rag_db"
BIN_DIR="\${USER_HOME}/.local/bin"
MCP_SERVER_DEST="\${BIN_DIR}/mcp-rag-server"
EXTRACT_DIR="\$(mktemp -d /tmp/rag-setup.XXXXXX)"
# Usa o argumento posicional se fornecido, senão usa o diretório atual
if [[ -n "\$CUSTOM_PROJECT_DIR" ]]; then
    PROJECT_DIR="\$(cd "\$CUSTOM_PROJECT_DIR" && pwd)"
else
    PROJECT_DIR="\$(pwd)"
fi

trap 'rm -rf "\$EXTRACT_DIR"' EXIT

echo ""
echo -e "\${BOLD}\${BLUE}================================================================\${NC}"
echo -e "\${BOLD}\${BLUE}  RAG Local Setup — ChromaDB + MCP Server\${NC}"
echo -e "\${BOLD}\${BLUE}================================================================\${NC}"
echo -e "  Projeto a indexar: \${BOLD}\${PROJECT_DIR}\${NC}"
echo ""

# ---------------------------------------------------------------------------
# Extrai arquivos embutidos
# ---------------------------------------------------------------------------
log_section "Extraindo arquivos embutidos"

echo "${B64_COMPOSE}"      | base64 -d > "\${EXTRACT_DIR}/docker-compose.yml"
echo "${B64_REQUIREMENTS}" | base64 -d > "\${EXTRACT_DIR}/requirements.txt"
echo "${B64_INDEXER}"      | base64 -d > "\${EXTRACT_DIR}/indexer_full.py"
echo "${B64_MCP}"          | base64 -d > "\${EXTRACT_DIR}/mcp_server.py"

log_info "Arquivos extraídos para: \${EXTRACT_DIR}"

# ---------------------------------------------------------------------------
# Verifica pré-requisitos
# ---------------------------------------------------------------------------
log_section "Verificando pré-requisitos"

if ! command -v python3 &>/dev/null; then
    log_error "Python 3 não encontrado. Instale: sudo apt install python3 python3-venv"
    exit 1
fi
PY_VER=\$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=\$(echo "\$PY_VER" | cut -d. -f1)
PY_MINOR=\$(echo "\$PY_VER" | cut -d. -f2)
if [[ "\$PY_MAJOR" -lt 3 ]] || [[ "\$PY_MAJOR" -eq 3 && "\$PY_MINOR" -lt 10 ]]; then
    log_error "Python 3.10+ necessário. Atual: \$PY_VER"
    exit 1
fi
if ! python3 -m venv --help &>/dev/null; then
    log_error "python3-venv não encontrado. Instale: sudo apt install python3-venv"
    exit 1
fi
log_info "Python \$PY_VER OK."

if ! command -v docker &>/dev/null; then
    log_error "Docker não encontrado: https://docs.docker.com/engine/install/"
    exit 1
fi
if ! docker info &>/dev/null 2>&1; then
    log_error "Docker daemon não está rodando. Inicie: sudo systemctl start docker"
    exit 1
fi
if docker compose version &>/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    log_error "Docker Compose não encontrado: sudo apt install docker-compose-plugin"
    exit 1
fi
log_info "Docker + Compose OK."

if ! command -v curl &>/dev/null; then
    log_warn "curl não encontrado — healthcheck do ChromaDB será pulado."
    HAS_CURL=false
else
    HAS_CURL=true
fi

# ---------------------------------------------------------------------------
# Modo --only-index
# ---------------------------------------------------------------------------
if [[ "\$ONLY_INDEX" == "true" ]]; then
    log_section "Modo: apenas indexação"
    if [[ ! -f "\${VENV_PYTHON}" ]]; then
        log_error "Venv não encontrado em \${VENV_DIR}. Execute sem --only-index primeiro para instalar."
        exit 1
    fi
    if [[ ! -d "\${PROJECT_DIR}" ]]; then
        log_error "Caminho não encontrado: \${PROJECT_DIR}"
        exit 1
    fi
    log_info "Indexando: \${PROJECT_DIR}"
    "\${VENV_PYTHON}" "\${EXTRACT_DIR}/indexer_full.py" "\${PROJECT_DIR}"
    log_info "Indexação concluída."
    exit 0
fi

# ---------------------------------------------------------------------------
# Cria/atualiza venv e instala dependências
# ---------------------------------------------------------------------------
log_section "Configurando ambiente virtual Python (~/.rag_venv)"

DEPS_OK=false
if [[ "\$REINSTALL" == "false" ]] && [[ -f "\${VENV_PYTHON}" ]]; then
    if "\${VENV_PYTHON}" -c "import chromadb, sentence_transformers, langchain_text_splitters, tqdm, mcp" 2>/dev/null; then
        log_info "Dependências já instaladas no venv. Pulando. (use --reinstall para forçar)"
        DEPS_OK=true
    fi
fi

if [[ "\$DEPS_OK" == "false" ]]; then
    if [[ ! -f "\${VENV_PYTHON}" ]]; then
        log_info "Criando venv em \${VENV_DIR}..."
        python3 -m venv "\${VENV_DIR}"
        log_info "Venv criado."
    fi
    log_info "Atualizando pip..."
    "\${VENV_PIP}" install --upgrade pip

    echo ""
    echo -e "\${YELLOW}  Instalando pacotes Python (isso pode levar alguns minutos na 1ª vez):\${NC}"
    echo -e "\${DIM}  sentence-transformers baixará o modelo all-MiniLM-L6-v2 (~90MB)\${NC}"
    echo ""

    # Instala com output visível (sem --quiet) para o usuário acompanhar
    "\${VENV_PIP}" install \
        --progress-bar on \
        -r "\${EXTRACT_DIR}/requirements.txt"

    echo ""
    log_info "Todas as dependências instaladas no venv."
fi

# ---------------------------------------------------------------------------
# ChromaDB via Docker
# ---------------------------------------------------------------------------
log_section "Configurando ChromaDB (Docker)"

mkdir -p "\${RAG_DB_DIR}" "\${DOCKER_COMPOSE_DIR}"

if [[ "\$REINSTALL" == "true" ]] || [[ ! -f "\${DOCKER_COMPOSE_DIR}/docker-compose.yml" ]]; then
    cp "\${EXTRACT_DIR}/docker-compose.yml" "\${DOCKER_COMPOSE_DIR}/docker-compose.yml"
    log_info "docker-compose.yml instalado em: \${DOCKER_COMPOSE_DIR}"
else
    log_info "docker-compose.yml já existe. Mantendo."
fi

if docker ps --format '{{.Names}}' | grep -q '^chromadb-rag$'; then
    log_info "Container chromadb-rag já está rodando."
else
    log_info "Iniciando ChromaDB (restart:always)..."
    (cd "\${DOCKER_COMPOSE_DIR}" && \$DOCKER_COMPOSE_CMD up -d)

    log_info "Aguardando ChromaDB inicializar..."
    WAITED=0
    while true; do
        if [[ "\$HAS_CURL" == "true" ]] && curl -sf "http://localhost:8000/api/v1/heartbeat" &>/dev/null; then
            log_info "ChromaDB respondendo em http://localhost:8000"
            break
        fi
        if [[ \$WAITED -ge 30 ]]; then
            log_warn "ChromaDB não respondeu em 30s. Verifique: docker logs chromadb-rag"
            break
        fi
        sleep 2; WAITED=\$((WAITED+2))
    done
fi

# ---------------------------------------------------------------------------
# Instala mcp-rag-server com shebang do venv
# ---------------------------------------------------------------------------
log_section "Instalando mcp-rag-server globalmente"

mkdir -p "\${BIN_DIR}"

NEEDS_INSTALL=true
if [[ "\$REINSTALL" == "false" ]] && [[ -f "\${MCP_SERVER_DEST}" ]]; then
    log_info "mcp-rag-server já instalado. Mantendo. (use --reinstall para atualizar)"
    NEEDS_INSTALL=false
fi

if [[ "\$NEEDS_INSTALL" == "true" ]]; then
    cp "\${EXTRACT_DIR}/mcp_server.py" "\${MCP_SERVER_DEST}"
    # Shebang aponta para o venv — garante que tem todas as dependências
    sed -i "1s|.*|#!\${VENV_PYTHON}|" "\${MCP_SERVER_DEST}"
    chmod +x "\${MCP_SERVER_DEST}"
    log_info "mcp-rag-server instalado: \${MCP_SERVER_DEST}"
    log_info "Shebang: \${VENV_PYTHON}"
fi

for RC in "\${USER_HOME}/.bashrc" "\${USER_HOME}/.zshrc"; do
    if [[ -f "\$RC" ]] && ! grep -qF '.local/bin' "\$RC"; then
        echo "" >> "\$RC"
        echo '# RAG setup — adicionado ao PATH' >> "\$RC"
        echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> "\$RC"
        log_info "Adicionado ~/.local/bin ao PATH em \$RC"
    fi
done

# ---------------------------------------------------------------------------
# Verifica claude.json
# ---------------------------------------------------------------------------
log_section "Verificando configuração do Claude Code"

CLAUDE_JSON="\${USER_HOME}/.claude.json"

if [[ -f "\${CLAUDE_JSON}" ]] && python3 -c "
import json, sys
data = json.load(open('\${CLAUDE_JSON}'))
sys.exit(0 if 'rag-codebase' in data.get('mcpServers', {}) else 1)
" 2>/dev/null; then
    log_info "Servidor 'rag-codebase' já registrado no claude.json."
else
    log_warn "Adicione em 'mcpServers' no arquivo \${CLAUDE_JSON}:"
    echo ""
    cat << 'MCP_CONFIG_EOF'
    "rag-codebase": {
      "command": "~/.local/bin/mcp-rag-server",
      "args": [],
      "env": {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000"
      }
    }
MCP_CONFIG_EOF
    echo ""
fi

# ---------------------------------------------------------------------------
# Indexa o projeto atual
# ---------------------------------------------------------------------------
if [[ "\$SKIP_INDEX" == "false" ]]; then
    if [[ ! -d "\${PROJECT_DIR}" ]]; then
        log_warn "Caminho não encontrado: \${PROJECT_DIR}. Pulando indexação."
    else
        log_section "Indexando o projeto: \${PROJECT_DIR}"
        "\${VENV_PYTHON}" "\${EXTRACT_DIR}/indexer_full.py" "\${PROJECT_DIR}"
    fi
else
    log_section "Indexação pulada (--skip-index)"
    log_info "Para indexar: ./rag-setup.run /caminho/do/projeto --only-index"
fi

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
echo ""
echo -e "\${BOLD}\${GREEN}================================================================\${NC}"
echo -e "\${BOLD}\${GREEN}  Setup concluído!\${NC}"
echo -e "\${BOLD}\${GREEN}================================================================\${NC}"
echo ""
echo -e "  \${GREEN}Venv Python\${NC} : \${VENV_DIR}"
echo -e "  \${GREEN}ChromaDB\${NC}    : http://localhost:8000 (Docker, auto-start)"
echo -e "  \${GREEN}Dados\${NC}       : \${RAG_DB_DIR}"
echo -e "  \${GREEN}MCP Server\${NC}  : \${MCP_SERVER_DEST}"
echo -e "  \${GREEN}Projeto\${NC}     : \${PROJECT_DIR}"
echo ""
echo -e "  \${BOLD}Próximos passos:\${NC}"
echo -e "  1. Reinicie o Claude Code CLI para carregar o MCP"
echo -e "  2. Use semantic_search_code para buscar no código"
echo -e "  3. Para reindexar: ./rag-setup.run --only-index"
echo ""
OUTER_EOF

chmod +x "$OUTPUT"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "[+] Arquivo gerado: $OUTPUT ($SIZE)"
echo "[+] Pronto! Copie para qualquer projeto e execute:"
echo "      cp ${OUTPUT} ~/seu-projeto/ && cd ~/seu-projeto && ./rag-setup.run"
