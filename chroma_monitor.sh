#!/usr/bin/env bash
# =============================================================================
# chroma_monitor.sh — Monitor do banco de dados ChromaDB
# =============================================================================
# Uso:
#   ./chroma_monitor.sh          # menu interativo
#   ./chroma_monitor.sh chunks   # contagem de chunks (roda uma vez)
#   ./chroma_monitor.sh watch    # monitora chunks em tempo real (loop)
#   ./chroma_monitor.sh disk     # tamanho do banco no disco (watch)
#   ./chroma_monitor.sh logs     # logs em tempo real do container Docker
#   ./chroma_monitor.sh full     # painel completo: chunks + disco + detalhes
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Cores e formatação
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'
BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CHROMA_HOST="localhost"
CHROMA_PORT="8000"
RAG_DB_DIR="${HOME}/.rag_db"
VENV_DIR="${HOME}/.rag_venv"
VENV_PYTHON="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"
CONTAINER_NAME="chromadb-rag"
REFRESH_INTERVAL=5

# ---------------------------------------------------------------------------
# Funções: venv e dependências
# ---------------------------------------------------------------------------

log_info()  { echo -e "${GREEN}[+]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
log_error() { echo -e "${RED}[ERRO]${NC} $*" >&2; }

ensure_venv() {
    # Garante que o venv existe e tem o chromadb instalado
    if [[ ! -f "${VENV_PYTHON}" ]]; then
        if ! command -v python3 &>/dev/null; then
            log_error "python3 não encontrado. Instale: sudo apt install python3 python3-venv"
            exit 1
        fi
        if ! python3 -m venv --help &>/dev/null; then
            log_error "python3-venv não encontrado. Instale: sudo apt install python3-venv"
            exit 1
        fi
        log_info "Criando ambiente virtual em ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
        log_info "Venv criado."
    fi

    # Instala chromadb no venv se ainda não estiver lá
    if ! "${VENV_PYTHON}" -c "import chromadb" 2>/dev/null; then
        log_info "Instalando chromadb no venv (necessário apenas na primeira vez)..."
        "${VENV_PIP}" install --quiet --upgrade pip
        "${VENV_PIP}" install --quiet chromadb
        log_info "chromadb instalado."
    fi
}

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

is_chroma_running() {
    "${VENV_PYTHON}" -c "
import chromadb, sys
try:
    chromadb.HttpClient(host='${CHROMA_HOST}', port=${CHROMA_PORT}).heartbeat()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null
}

print_header() {
    clear
    echo -e "${BOLD}${BLUE}"
    echo "  ╔══════════════════════════════════════════════════════╗"
    echo "  ║          ChromaDB Monitor — RAG Local                ║"
    echo "  ╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  ${DIM}Banco: ${CHROMA_HOST}:${CHROMA_PORT}  |  Dados: ${RAG_DB_DIR}${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Modo 1: chunks — contagem detalhada (roda uma vez)
# ---------------------------------------------------------------------------
cmd_chunks() {
    ensure_venv

    echo ""
    echo -e "${BOLD}${CYAN}  Consultando ChromaDB em ${CHROMA_HOST}:${CHROMA_PORT}...${NC}"
    echo ""

    "${VENV_PYTHON}" - << 'PYEOF'
import sys
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

BOLD   = "\033[1m"
GREEN  = "\033[0;32m"
CYAN   = "\033[0;36m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
DIM    = "\033[2m"
NC     = "\033[0m"

try:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()
except Exception as e:
    print(f"{RED}[ERRO] ChromaDB offline ou inacessível: {e}{NC}")
    print(f"       Verifique: docker ps | grep chromadb")
    sys.exit(1)

collections = client.list_collections()

if not collections:
    print(f"{YELLOW}  Nenhuma coleção encontrada.{NC}")
    print(f"  Execute o indexer para popular o banco.")
    sys.exit(0)

total_global = 0

print(f"  {'Coleção':<25} {'Chunks':>10}  {'Detalhes'}")
print(f"  {'-'*60}")

for col in collections:
    count = col.count()
    total_global += count

    sample = col.get(limit=1, include=["metadatas"])
    meta_keys = []
    if sample["metadatas"]:
        meta_keys = list(sample["metadatas"][0].keys())

    all_meta = col.get(include=["metadatas"])
    unique_files = set()
    for m in all_meta["metadatas"]:
        if m and "file_path" in m:
            unique_files.add(m["file_path"])

    print(f"  {GREEN}{BOLD}{col.name:<25}{NC}  {CYAN}{count:>7} chunks{NC}  "
          f"{DIM}({len(unique_files)} arquivos únicos){NC}")
    if meta_keys:
        print(f"  {'':25}  {'':10}  {DIM}campos: {', '.join(meta_keys)}{NC}")
    print()

print(f"  {'-'*60}")
print(f"  {BOLD}Total global: {CYAN}{total_global} chunks{NC}{BOLD} em {len(collections)} coleção(ões){NC}")
print()
PYEOF
}

# ---------------------------------------------------------------------------
# Modo 2: watch — atualiza contagem a cada N segundos
# ---------------------------------------------------------------------------
cmd_watch() {
    ensure_venv

    echo -e "${DIM}  Pressione Ctrl+C para sair.${NC}"
    echo ""

    LAST_COUNT=-1

    while true; do
        RESULT=$("${VENV_PYTHON}" - 2>/dev/null << 'PYEOF'
import chromadb, sys

try:
    client = chromadb.HttpClient(host="localhost", port=8000)
    client.heartbeat()
    cols = client.list_collections()
    total = sum(c.count() for c in cols)
    details = " | ".join(f"{c.name}: {c.count()}" for c in cols)
    print(f"{total}||{details}||online")
except:
    print("0||||offline")
PYEOF
)

        TOTAL=$(echo "$RESULT" | cut -d'|' -f1)
        DETAILS=$(echo "$RESULT" | cut -d'|' -f3)
        STATUS=$(echo "$RESULT" | cut -d'|' -f5)
        TIMESTAMP=$(date '+%H:%M:%S')

        if [[ "$STATUS" == "offline" ]]; then
            printf "\r  ${RED}[%s]${NC} ChromaDB offline...%20s" "$TIMESTAMP" ""
        else
            DIFF=""
            if [[ "$LAST_COUNT" -ge 0 ]] && [[ "$TOTAL" -ne "$LAST_COUNT" ]]; then
                DELTA=$((TOTAL - LAST_COUNT))
                if [[ "$DELTA" -gt 0 ]]; then
                    DIFF=" ${GREEN}(+${DELTA})${NC}"
                else
                    DIFF=" ${RED}(${DELTA})${NC}"
                fi
            fi

            printf "\r  ${BOLD}[%s]${NC}  Chunks: ${CYAN}${BOLD}%s${NC}%b  ${DIM}%s${NC}%30s" \
                "$TIMESTAMP" "$TOTAL" "$DIFF" "$DETAILS" ""

            LAST_COUNT="$TOTAL"
        fi

        sleep "$REFRESH_INTERVAL"
    done
}

# ---------------------------------------------------------------------------
# Modo 3: disk — monitora tamanho do banco no disco
# ---------------------------------------------------------------------------
cmd_disk() {
    if [[ ! -d "$RAG_DB_DIR" ]]; then
        echo -e "${YELLOW}[!]${NC} Diretório não encontrado: ${RAG_DB_DIR}"
        echo -e "    O banco ainda não foi inicializado."
        exit 1
    fi

    echo -e "${DIM}  Monitorando tamanho de ${RAG_DB_DIR} (Ctrl+C para sair)${NC}"
    echo ""

    if command -v watch &>/dev/null; then
        watch -n "$REFRESH_INTERVAL" -d \
            "du -sh ${RAG_DB_DIR} && echo '' && find ${RAG_DB_DIR} -type f | wc -l | xargs -I{} echo 'Arquivos no banco: {}'"
    else
        while true; do
            SIZE=$(du -sh "$RAG_DB_DIR" 2>/dev/null | cut -f1)
            FILES=$(find "$RAG_DB_DIR" -type f 2>/dev/null | wc -l)
            printf "\r  ${BOLD}[%s]${NC}  Tamanho: ${CYAN}${BOLD}%s${NC}  |  Arquivos internos: ${DIM}%s${NC}%20s" \
                "$(date '+%H:%M:%S')" "$SIZE" "$FILES" ""
            sleep "$REFRESH_INTERVAL"
        done
    fi
}

# ---------------------------------------------------------------------------
# Modo 4: logs — logs do container Docker em tempo real
# ---------------------------------------------------------------------------
cmd_logs() {
    if ! command -v docker &>/dev/null; then
        log_error "Docker não encontrado."
        exit 1
    fi

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}[!]${NC} Container '${CONTAINER_NAME}' não está rodando."
        echo ""
        echo -e "  Containers ativos:"
        docker ps --format "  - {{.Names}} ({{.Status}})" 2>/dev/null || echo "  Nenhum."
        exit 1
    fi

    echo -e "${DIM}  Exibindo logs de '${CONTAINER_NAME}' (Ctrl+C para sair)${NC}"
    echo -e "${DIM}  Dica: rode o indexer em outro terminal para ver as requisições entrando.${NC}"
    echo ""
    docker logs -f --tail=30 "$CONTAINER_NAME"
}

# ---------------------------------------------------------------------------
# Modo 5: full — painel completo
# ---------------------------------------------------------------------------
cmd_full() {
    ensure_venv

    echo -e "${DIM}  Painel atualiza a cada ${REFRESH_INTERVAL}s — Ctrl+C para sair${NC}"
    echo ""
    sleep 1

    while true; do
        print_header

        # Status do container
        echo -e "  ${BOLD}Container Docker${NC}"
        if command -v docker &>/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
            UPTIME=$(docker inspect --format='{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null | cut -c1-19 | tr 'T' ' ')
            echo -e "  ${GREEN}●${NC} ${CONTAINER_NAME} rodando  ${DIM}(iniciado: ${UPTIME}Z)${NC}"
        else
            echo -e "  ${RED}●${NC} ${CONTAINER_NAME} offline"
        fi
        echo ""

        # Tamanho em disco
        echo -e "  ${BOLD}Banco em disco${NC}"
        if [[ -d "$RAG_DB_DIR" ]]; then
            DISK_SIZE=$(du -sh "$RAG_DB_DIR" 2>/dev/null | cut -f1)
            DISK_FILES=$(find "$RAG_DB_DIR" -type f 2>/dev/null | wc -l)
            echo -e "  ${CYAN}${DISK_SIZE}${NC}  ${DIM}(${DISK_FILES} arquivos internos em ${RAG_DB_DIR})${NC}"
        else
            echo -e "  ${YELLOW}Diretório não encontrado: ${RAG_DB_DIR}${NC}"
        fi
        echo ""

        # Coleções e chunks
        echo -e "  ${BOLD}Coleções e Chunks${NC}"
        "${VENV_PYTHON}" - 2>/dev/null << 'PYEOF'
import chromadb, sys

GREEN  = "\033[0;32m"; CYAN   = "\033[0;36m"
YELLOW = "\033[1;33m"; RED    = "\033[0;31m"
BOLD   = "\033[1m";    DIM    = "\033[2m"; NC = "\033[0m"

try:
    client = chromadb.HttpClient(host="localhost", port=8000)
    client.heartbeat()
except Exception as e:
    print(f"  {RED}ChromaDB inacessível: {e}{NC}")
    sys.exit(0)

collections = client.list_collections()

if not collections:
    print(f"  {YELLOW}Banco vazio — nenhuma coleção encontrada.{NC}")
    print(f"  Execute: indexer_full.py")
    sys.exit(0)

total_chunks = 0
total_files  = 0

for col in collections:
    count = col.count()
    total_chunks += count

    all_meta = col.get(include=["metadatas"])
    unique_files = {m["file_path"] for m in all_meta["metadatas"] if m and "file_path" in m}
    total_files += len(unique_files)

    max_count = max(c.count() for c in collections) or 1
    bar_len = min(30, int(count / max_count * 30))
    bar = "█" * bar_len + "░" * (30 - bar_len)

    print(f"  {GREEN}{BOLD}{col.name}{NC}")
    print(f"  {CYAN}{bar}{NC}  {BOLD}{count:,}{NC} chunks  |  {len(unique_files):,} arquivos")
    print()

print(f"  {'─'*55}")
print(f"  {BOLD}Total: {CYAN}{total_chunks:,} chunks{NC}{BOLD} em {total_files:,} arquivo(s){NC}")
PYEOF

        echo ""
        echo -e "  ${DIM}Última atualização: $(date '+%H:%M:%S') — próxima em ${REFRESH_INTERVAL}s${NC}"

        sleep "$REFRESH_INTERVAL"
    done
}

# ---------------------------------------------------------------------------
# Menu interativo
# ---------------------------------------------------------------------------
cmd_menu() {
    ensure_venv
    print_header

    if is_chroma_running; then
        echo -e "  Status: ${GREEN}${BOLD}● ChromaDB online${NC}"
    else
        echo -e "  Status: ${RED}${BOLD}● ChromaDB offline${NC}"
        echo -e "  ${DIM}Inicie com: docker compose -f ~/docker-chromadb/docker-compose.yml up -d${NC}"
    fi
    echo ""
    echo -e "  ${BOLD}Escolha uma opção:${NC}"
    echo ""
    echo -e "  ${CYAN}[1]${NC} Contar chunks agora           ${DIM}(snapshot único)${NC}"
    echo -e "  ${CYAN}[2]${NC} Monitorar chunks em tempo real ${DIM}(atualiza a cada ${REFRESH_INTERVAL}s)${NC}"
    echo -e "  ${CYAN}[3]${NC} Monitorar tamanho no disco     ${DIM}(watch no ~/.rag_db)${NC}"
    echo -e "  ${CYAN}[4]${NC} Ver logs do Docker             ${DIM}(requisições HTTP ao ChromaDB)${NC}"
    echo -e "  ${CYAN}[5]${NC} Painel completo                ${DIM}(tudo junto, atualiza automático)${NC}"
    echo ""
    echo -e "  ${CYAN}[0]${NC} Sair"
    echo ""
    read -rp "  Opção: " choice

    case "$choice" in
        1) echo ""; cmd_chunks ;;
        2) echo ""; cmd_watch ;;
        3) cmd_disk ;;
        4) cmd_logs ;;
        5) cmd_full ;;
        0) exit 0 ;;
        *) echo -e "\n  ${YELLOW}Opção inválida.${NC}"; sleep 1; cmd_menu ;;
    esac
}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
case "${1:-menu}" in
    chunks)  cmd_chunks ;;
    watch)   cmd_watch  ;;
    disk)    cmd_disk   ;;
    logs)    cmd_logs   ;;
    full)    cmd_full   ;;
    menu)    cmd_menu   ;;
    *)
        echo -e "${RED}[ERRO]${NC} Modo desconhecido: '$1'"
        echo "Uso: $0 [chunks|watch|disk|logs|full]"
        exit 1
        ;;
esac
