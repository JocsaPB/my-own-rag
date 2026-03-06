#!/usr/bin/env bash
# =============================================================================
# init.sh — Setup automatizado do RAG local com ChromaDB + MCP Server
# =============================================================================
# O que este script faz:
#   1. Verifica pré-requisitos (Docker, Python 3.10+)
#   2. Cria venv em ~/.rag_venv e instala dependências Python lá
#   3. Cria o diretório de dados do ChromaDB (~/.rag_db)
#   4. Copia o docker-compose.yml e levanta o ChromaDB via Docker
#   5. Instala o mcp_server.py globalmente no PATH do usuário
#   6. Exibe as instruções para configurar o claude.json
#
# Uso:
#   chmod +x init.sh && ./init.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Cores para output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[+]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
log_error()   { echo -e "${RED}[ERRO]${NC} $*" >&2; }
log_section() { echo -e "\n${BLUE}${BOLD}==> $*${NC}"; }

# ---------------------------------------------------------------------------
# Variáveis de configuração
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="${HOME}"
VENV_DIR="${USER_HOME}/.rag_venv"
VENV_PYTHON="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"
DOCKER_COMPOSE_DIR="${USER_HOME}/docker-chromadb"
RAG_DB_DIR="${USER_HOME}/.rag_db"
BIN_DIR="${USER_HOME}/.local/bin"
MCP_SERVER_DEST="${BIN_DIR}/mcp-rag-server"

# ---------------------------------------------------------------------------
# Passo 0: Verificação de pré-requisitos
# ---------------------------------------------------------------------------
log_section "Verificando pré-requisitos"

# Python 3.10+
if ! command -v python3 &>/dev/null; then
    log_error "Python 3 não encontrado. Instale com: sudo apt install python3 python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    log_error "Python 3.10+ necessário. Versão atual: $PYTHON_VERSION"
    exit 1
fi
log_info "Python $PYTHON_VERSION encontrado."

# Módulo venv disponível
if ! python3 -m venv --help &>/dev/null; then
    log_error "Módulo venv não encontrado. Instale com: sudo apt install python3-venv"
    exit 1
fi
log_info "python3-venv disponível."

# Docker
if ! command -v docker &>/dev/null; then
    log_error "Docker não encontrado. Instale em: https://docs.docker.com/engine/install/"
    exit 1
fi

if docker compose version &>/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    log_error "Docker Compose não encontrado. Instale: sudo apt install docker-compose-plugin"
    exit 1
fi
log_info "Docker + Compose encontrados (cmd: '$DOCKER_COMPOSE_CMD')."

if ! docker info &>/dev/null; then
    log_error "O daemon do Docker não está rodando."
    log_error "Inicie com: sudo systemctl start docker"
    exit 1
fi
log_info "Docker daemon ativo."

# ---------------------------------------------------------------------------
# Passo 1: Criar venv e instalar dependências Python
# ---------------------------------------------------------------------------
log_section "Configurando ambiente virtual Python (~/.rag_venv)"

REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    log_error "requirements.txt não encontrado em: $REQUIREMENTS_FILE"
    exit 1
fi

if [[ ! -f "${VENV_PYTHON}" ]]; then
    log_info "Criando venv em: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    log_info "Venv criado com sucesso."
else
    log_info "Venv já existe em ${VENV_DIR}."
fi

log_info "Instalando/atualizando dependências no venv..."
"${VENV_PIP}" install --quiet --upgrade pip
"${VENV_PIP}" install --quiet -r "$REQUIREMENTS_FILE"
log_info "Dependências instaladas no venv."

# ---------------------------------------------------------------------------
# Passo 2: Criar diretório de dados do ChromaDB
# ---------------------------------------------------------------------------
log_section "Configurando diretório de dados do ChromaDB"

mkdir -p "$RAG_DB_DIR"
log_info "Diretório de dados: $RAG_DB_DIR"

# ---------------------------------------------------------------------------
# Passo 3: Configurar e iniciar o ChromaDB via Docker Compose
# ---------------------------------------------------------------------------
log_section "Configurando ChromaDB Docker"

mkdir -p "$DOCKER_COMPOSE_DIR"
cp "${SCRIPT_DIR}/docker-compose.yml" "${DOCKER_COMPOSE_DIR}/docker-compose.yml"
log_info "docker-compose.yml copiado para: $DOCKER_COMPOSE_DIR"

log_info "Iniciando ChromaDB (restart:always — sobe automaticamente com o SO)..."
cd "$DOCKER_COMPOSE_DIR"
$DOCKER_COMPOSE_CMD up -d

log_info "Aguardando ChromaDB inicializar..."
MAX_WAIT=30; WAITED=0
while ! curl -sf "http://localhost:8000/api/v1/heartbeat" &>/dev/null; do
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        log_warn "ChromaDB não respondeu em ${MAX_WAIT}s. Verifique: docker logs chromadb-rag"
        break
    fi
    sleep 2; WAITED=$((WAITED + 2))
done

if curl -sf "http://localhost:8000/api/v1/heartbeat" &>/dev/null; then
    log_info "ChromaDB respondendo em http://localhost:8000"
fi

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Passo 4: Instalar o mcp_server.py globalmente no PATH
# ---------------------------------------------------------------------------
log_section "Instalando mcp-rag-server globalmente"

MCP_SERVER_SRC="${SCRIPT_DIR}/mcp_server.py"
if [[ ! -f "$MCP_SERVER_SRC" ]]; then
    log_error "mcp_server.py não encontrado em: $MCP_SERVER_SRC"
    exit 1
fi

mkdir -p "$BIN_DIR"
cp "$MCP_SERVER_SRC" "$MCP_SERVER_DEST"

# Shebang aponta para o python do venv (tem todas as dependências)
sed -i "1s|.*|#!${VENV_PYTHON}|" "$MCP_SERVER_DEST"
chmod +x "$MCP_SERVER_DEST"

log_info "mcp-rag-server instalado em: $MCP_SERVER_DEST"
log_info "Shebang configurado para: ${VENV_PYTHON}"

# Garante ~/.local/bin no PATH permanente
for RC in "${USER_HOME}/.bashrc" "${USER_HOME}/.zshrc"; do
    if [[ -f "$RC" ]] && ! grep -qF '.local/bin' "$RC"; then
        echo "" >> "$RC"
        echo "# RAG — adicionado pelo init.sh" >> "$RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC"
        log_info "Adicionado ~/.local/bin ao PATH em: $RC"
    fi
done

# ---------------------------------------------------------------------------
# Passo 5: Indexar o projeto atual (opcional)
# ---------------------------------------------------------------------------
log_section "Indexação inicial (opcional)"

read -rp "Deseja indexar o projeto em '$SCRIPT_DIR' agora? [s/N] " answer
if [[ "${answer,,}" == "s" || "${answer,,}" == "sim" || "${answer,,}" == "y" || "${answer,,}" == "yes" ]]; then
    log_info "Iniciando indexação com o Python do venv..."
    "${VENV_PYTHON}" "${SCRIPT_DIR}/indexer_full.py" "$SCRIPT_DIR"
else
    log_info "Pulando. Para indexar depois:"
    log_info "  ${VENV_PYTHON} ${SCRIPT_DIR}/indexer_full.py /caminho/do/projeto"
fi

# ---------------------------------------------------------------------------
# Passo 6: Instruções para configurar o claude.json
# ---------------------------------------------------------------------------
log_section "Configuração do Claude Code (claude.json)"

CLAUDE_JSON="${USER_HOME}/.claude.json"
echo ""
echo -e "${BOLD}Adicione em 'mcpServers' no arquivo ${YELLOW}${CLAUDE_JSON}${NC}${BOLD}:${NC}"
echo ""
cat << 'EOF'
    "rag-codebase": {
      "command": "~/.local/bin/mcp-rag-server",
      "args": [],
      "env": {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000"
      }
    }
EOF
echo ""

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
log_section "Setup concluído!"
echo ""
echo -e "  ${GREEN}Venv Python${NC}   : ${VENV_DIR}"
echo -e "  ${GREEN}ChromaDB${NC}      : http://localhost:8000 (Docker, auto-start)"
echo -e "  ${GREEN}Dados do banco${NC}: ${RAG_DB_DIR}"
echo -e "  ${GREEN}MCP Server${NC}    : ${MCP_SERVER_DEST}"
echo ""
echo -e "  ${BOLD}Comandos úteis:${NC}"
echo -e "    Indexar projeto : ${VENV_PYTHON} ${SCRIPT_DIR}/indexer_full.py /caminho"
echo -e "    Monitor banco   : ${SCRIPT_DIR}/chroma_monitor.sh"
echo -e "    Logs do Docker  : docker logs chromadb-rag -f"
echo -e "    Testar ChromaDB : curl http://localhost:8000/api/v1/heartbeat"
echo ""
