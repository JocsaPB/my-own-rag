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
#   ./chroma_monitor.sh mcp-logs # logs de uso das ferramentas MCP
#   ./chroma_monitor.sh mcp-summary # resumo agregado de uso MCP (24h + geral)
#   ./chroma_monitor.sh full     # painel completo: chunks + disco + detalhes
#   ./chroma_monitor.sh reset    # zera o ChromaDB (remove todas as coleções)
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
MCP_USAGE_LOG="${MCP_USAGE_LOG:-${HOME}/.rag_db/mcp_usage.log}"
REFRESH_INTERVAL=5
UI_LANG="${CHROMA_MONITOR_LANG:-${RAG_SETUP_LANG:-}}"
YES_NO_HINT="[s/N]"

# ---------------------------------------------------------------------------
# Idioma e i18n
# ---------------------------------------------------------------------------

set_lang_defaults() {
    if [[ "$UI_LANG" == "en-us" ]]; then
        YES_NO_HINT="[y/N]"
    else
        YES_NO_HINT="[s/N]"
    fi
}

select_ui_language() {
    if [[ -n "$UI_LANG" ]]; then
        UI_LANG="$(echo "$UI_LANG" | tr '[:upper:]' '[:lower:]')"
        case "$UI_LANG" in
            pt-br|pt|en-us|en) ;;
            *) UI_LANG="pt-br" ;;
        esac
        [[ "$UI_LANG" == "pt" ]] && UI_LANG="pt-br"
        [[ "$UI_LANG" == "en" ]] && UI_LANG="en-us"
        set_lang_defaults
        return
    fi

    if [[ ! -t 0 ]]; then
        UI_LANG="pt-br"
        set_lang_defaults
        return
    fi

    echo ""
    echo -e "${GREEN}Idioma / Language: [1] PT-BR [2] EN-US (padrão/default: 1)${NC}"
    read -r -p "> " LANG_CHOICE
    case "$LANG_CHOICE" in
        2|en|EN|en-us|EN-US|english|English) UI_LANG="en-us" ;;
        *) UI_LANG="pt-br" ;;
    esac
    set_lang_defaults
}

t() {
    local key="$1"
    if [[ "$UI_LANG" == "en-us" ]]; then
        case "$key" in
            err_prefix) echo "ERROR" ;;
            py_missing) echo "python3 not found. Install: sudo apt install python3 python3-venv" ;;
            py_venv_missing) echo "python3-venv not found. Install: sudo apt install python3-venv" ;;
            creating_venv) echo "Creating virtual environment at" ;;
            venv_created) echo "Venv created." ;;
            installing_chromadb) echo "Installing chromadb in venv (first time only)..." ;;
            chromadb_installed) echo "chromadb installed." ;;
            header_title) echo "ChromaDB Monitor — Local RAG" ;;
            db_label) echo "Database" ;;
            data_label) echo "Data" ;;
            querying_chroma) echo "Querying ChromaDB at" ;;
            press_ctrl_c) echo "Press Ctrl+C to exit." ;;
            dir_not_found) echo "Directory not found:" ;;
            db_not_init) echo "Database not initialized yet." ;;
            monitoring_size) echo "Monitoring size of" ;;
            files_in_db) echo "Files in database" ;;
            size_label) echo "Size" ;;
            internal_files_short) echo "Internal files" ;;
            docker_not_found) echo "Docker not found." ;;
            container_not_running) echo "Container is not running:" ;;
            active_containers) echo "Active containers:" ;;
            none) echo "None." ;;
            showing_logs) echo "Showing logs from" ;;
            logs_tip) echo "Tip: run indexer in another terminal to see incoming requests." ;;
            mcp_log_missing) echo "MCP log file not found:" ;;
            mcp_log_hint1) echo "File is created automatically when MCP server receives calls." ;;
            mcp_log_hint2) echo "Restart/update mcp-rag-server and use an MCP tool." ;;
            mcp_log_hint_short) echo "File is created automatically when MCP server receives calls." ;;
            showing_mcp_usage) echo "Showing MCP usage in real time (Ctrl+C to exit)" ;;
            source) echo "Source" ;;
            panel_updates) echo "Panel updates every" ;;
            container_docker) echo "Docker container" ;;
            running) echo "running" ;;
            started) echo "started" ;;
            offline) echo "offline" ;;
            start_with) echo "Start with" ;;
            disk_db) echo "Database on disk" ;;
            internal_files) echo "internal files in" ;;
            collections_chunks) echo "Collections and chunks" ;;
            last_update) echo "Last update" ;;
            next_in) echo "next in" ;;
            reset_attention) echo "[WARNING] This action will delete all ChromaDB collections and chunks." ;;
            reset_irreversible) echo "This operation is irreversible." ;;
            confirm_continue) echo "Continue?" ;;
            confirm_type_reset) echo "Type RESET to confirm:" ;;
            reset_cancelled) echo "Reset canceled." ;;
            status_online) echo "Status: ● ChromaDB online" ;;
            status_offline) echo "Status: ● ChromaDB offline" ;;
            choose_option) echo "Choose an option:" ;;
            menu_1) echo "[1] Count chunks now           (single snapshot)" ;;
            menu_2) echo "[2] Watch chunks in real time (updates every ${REFRESH_INTERVAL}s)" ;;
            menu_3) echo "[3] Watch disk size            (watch ~/.rag_db)" ;;
            menu_4) echo "[4] Show Docker logs           (HTTP requests to ChromaDB)" ;;
            menu_5) echo "[5] Show MCP usage logs        (actor + tool)" ;;
            menu_6) echo "[6] MCP usage summary          (top tools/actors + 24h)" ;;
            menu_7) echo "[7] Full dashboard             (all in one, auto refresh)" ;;
            menu_8) echo "[8] Reset ChromaDB             (deletes all collections)" ;;
            menu_0) echo "[0] Exit" ;;
            option_prompt) echo "Option:" ;;
            invalid_option) echo "Invalid option." ;;
            unknown_mode) echo "Unknown mode" ;;
            usage) echo "Usage: $0 [chunks|watch|disk|logs|mcp-logs|mcp-summary|full|reset|menu]" ;;
            *) echo "$key" ;;
        esac
    else
        case "$key" in
            err_prefix) echo "ERRO" ;;
            py_missing) echo "python3 não encontrado. Instale: sudo apt install python3 python3-venv" ;;
            py_venv_missing) echo "python3-venv não encontrado. Instale: sudo apt install python3-venv" ;;
            creating_venv) echo "Criando ambiente virtual em" ;;
            venv_created) echo "Venv criado." ;;
            installing_chromadb) echo "Instalando chromadb no venv (necessário apenas na primeira vez)..." ;;
            chromadb_installed) echo "chromadb instalado." ;;
            header_title) echo "Monitor ChromaDB — RAG Local" ;;
            db_label) echo "Banco" ;;
            data_label) echo "Dados" ;;
            querying_chroma) echo "Consultando ChromaDB em" ;;
            press_ctrl_c) echo "Pressione Ctrl+C para sair." ;;
            dir_not_found) echo "Diretório não encontrado:" ;;
            db_not_init) echo "O banco ainda não foi inicializado." ;;
            monitoring_size) echo "Monitorando tamanho de" ;;
            files_in_db) echo "Arquivos no banco" ;;
            size_label) echo "Tamanho" ;;
            internal_files_short) echo "Arquivos internos" ;;
            docker_not_found) echo "Docker não encontrado." ;;
            container_not_running) echo "Container não está rodando:" ;;
            active_containers) echo "Containers ativos:" ;;
            none) echo "Nenhum." ;;
            showing_logs) echo "Exibindo logs de" ;;
            logs_tip) echo "Dica: rode o indexer em outro terminal para ver as requisições entrando." ;;
            mcp_log_missing) echo "Arquivo de log MCP não encontrado:" ;;
            mcp_log_hint1) echo "O arquivo será criado automaticamente quando o MCP Server receber chamadas." ;;
            mcp_log_hint2) echo "Reinicie/atualize o mcp-rag-server e use alguma ferramenta MCP." ;;
            mcp_log_hint_short) echo "O arquivo será criado automaticamente quando o MCP Server receber chamadas." ;;
            showing_mcp_usage) echo "Exibindo uso MCP em tempo real (Ctrl+C para sair)" ;;
            source) echo "Fonte" ;;
            panel_updates) echo "Painel atualiza a cada" ;;
            container_docker) echo "Container Docker" ;;
            running) echo "rodando" ;;
            started) echo "iniciado" ;;
            offline) echo "offline" ;;
            start_with) echo "Inicie com" ;;
            disk_db) echo "Banco em disco" ;;
            internal_files) echo "arquivos internos em" ;;
            collections_chunks) echo "Coleções e Chunks" ;;
            last_update) echo "Última atualização" ;;
            next_in) echo "próxima em" ;;
            reset_attention) echo "[ATENÇÃO] Esta ação vai apagar todas as coleções e chunks do ChromaDB." ;;
            reset_irreversible) echo "Essa operação é irreversível." ;;
            confirm_continue) echo "Deseja continuar?" ;;
            confirm_type_reset) echo "Digite ZERAR para confirmar:" ;;
            reset_cancelled) echo "Reset cancelado." ;;
            status_online) echo "Status: ● ChromaDB online" ;;
            status_offline) echo "Status: ● ChromaDB offline" ;;
            choose_option) echo "Escolha uma opção:" ;;
            menu_1) echo "[1] Contar chunks agora           (snapshot único)" ;;
            menu_2) echo "[2] Monitorar chunks em tempo real (atualiza a cada ${REFRESH_INTERVAL}s)" ;;
            menu_3) echo "[3] Monitorar tamanho no disco     (watch no ~/.rag_db)" ;;
            menu_4) echo "[4] Ver logs do Docker             (requisições HTTP ao ChromaDB)" ;;
            menu_5) echo "[5] Ver logs de uso MCP            (quem acessa + ferramenta usada)" ;;
            menu_6) echo "[6] Resumo de uso MCP              (top ferramentas/atores e 24h)" ;;
            menu_7) echo "[7] Painel completo                (tudo junto, atualiza automático)" ;;
            menu_8) echo "[8] Zerar ChromaDB                 (apaga todas as coleções)" ;;
            menu_0) echo "[0] Sair" ;;
            option_prompt) echo "Opção:" ;;
            invalid_option) echo "Opção inválida." ;;
            unknown_mode) echo "Modo desconhecido" ;;
            usage) echo "Uso: $0 [chunks|watch|disk|logs|mcp-logs|mcp-summary|full|reset|menu]" ;;
            *) echo "$key" ;;
        esac
    fi
}

log_info()  { echo -e "${GREEN}[+]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
log_error() { echo -e "${RED}[$(t err_prefix)]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Funções: venv e dependências
# ---------------------------------------------------------------------------

ensure_venv() {
    # Garante que o venv existe e tem o chromadb instalado
    if [[ ! -f "${VENV_PYTHON}" ]]; then
        if ! command -v python3 &>/dev/null; then
            log_error "$(t py_missing)"
            exit 1
        fi
        if ! python3 -m venv --help &>/dev/null; then
            log_error "$(t py_venv_missing)"
            exit 1
        fi
        log_info "$(t creating_venv) ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
        log_info "$(t venv_created)"
    fi

    # Instala chromadb no venv se ainda não estiver lá
    if ! "${VENV_PYTHON}" -c "import chromadb" 2>/dev/null; then
        log_info "$(t installing_chromadb)"
        "${VENV_PIP}" install --quiet --upgrade pip
        "${VENV_PIP}" install --quiet chromadb
        log_info "$(t chromadb_installed)"
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
    printf "  ║          %-44s║\n" "$(t header_title)"
    echo "  ╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  ${DIM}$(t db_label): ${CHROMA_HOST}:${CHROMA_PORT}  |  $(t data_label): ${RAG_DB_DIR}${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Modo 1: chunks — contagem detalhada (roda uma vez)
# ---------------------------------------------------------------------------
cmd_chunks() {
    ensure_venv

    echo ""
    echo -e "${BOLD}${CYAN}  $(t querying_chroma) ${CHROMA_HOST}:${CHROMA_PORT}...${NC}"
    echo ""

    MONITOR_LANG="${UI_LANG}" "${VENV_PYTHON}" - << 'PYEOF'
import os
import sys
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
LANG = os.getenv("MONITOR_LANG", "pt-br").lower()
IS_EN = LANG == "en-us"

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
    if IS_EN:
        print(f"{RED}[ERROR] ChromaDB offline or unreachable: {e}{NC}")
        print("       Check: docker ps | grep chromadb")
    else:
        print(f"{RED}[ERRO] ChromaDB offline ou inacessível: {e}{NC}")
        print("       Verifique: docker ps | grep chromadb")
    sys.exit(1)

collections = client.list_collections()

if not collections:
    if IS_EN:
        print(f"{YELLOW}  No collections found.{NC}")
        print("  Run indexer to populate database.")
    else:
        print(f"{YELLOW}  Nenhuma coleção encontrada.{NC}")
        print("  Execute o indexer para popular o banco.")
    sys.exit(0)

total_global = 0

print(f"  {('Collection' if IS_EN else 'Coleção'):<25} {'Chunks':>10}  {('Details' if IS_EN else 'Detalhes')}")
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

    file_label = "unique files" if IS_EN else "arquivos únicos"
    print(f"  {GREEN}{BOLD}{col.name:<25}{NC}  {CYAN}{count:>7} chunks{NC}  "
          f"{DIM}({len(unique_files)} {file_label}){NC}")
    if meta_keys:
        fields_label = "fields" if IS_EN else "campos"
        print(f"  {'':25}  {'':10}  {DIM}{fields_label}: {', '.join(meta_keys)}{NC}")
    print()

print(f"  {'-'*60}")
if IS_EN:
    print(f"  {BOLD}Global total: {CYAN}{total_global} chunks{NC}{BOLD} in {len(collections)} collection(s){NC}")
else:
    print(f"  {BOLD}Total global: {CYAN}{total_global} chunks{NC}{BOLD} em {len(collections)} coleção(ões){NC}")
print()
PYEOF
}

# ---------------------------------------------------------------------------
# Modo 2: watch — atualiza contagem a cada N segundos
# ---------------------------------------------------------------------------
cmd_watch() {
    ensure_venv

    echo -e "${DIM}  $(t press_ctrl_c)${NC}"
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
        echo -e "${YELLOW}[!]${NC} $(t dir_not_found) ${RAG_DB_DIR}"
        echo -e "    $(t db_not_init)"
        exit 1
    fi

    echo -e "${DIM}  $(t monitoring_size) ${RAG_DB_DIR} (Ctrl+C)${NC}"
    echo ""

    if command -v watch &>/dev/null; then
        watch -n "$REFRESH_INTERVAL" -d \
            "du -sh ${RAG_DB_DIR} && echo '' && find ${RAG_DB_DIR} -type f | wc -l | xargs -I{} echo '$(t files_in_db): {}'"
    else
        while true; do
            SIZE=$(du -sh "$RAG_DB_DIR" 2>/dev/null | cut -f1)
            FILES=$(find "$RAG_DB_DIR" -type f 2>/dev/null | wc -l)
            printf "\r  ${BOLD}[%s]${NC}  %s: ${CYAN}${BOLD}%s${NC}  |  %s: ${DIM}%s${NC}%20s" \
                "$(date '+%H:%M:%S')" "$(t size_label)" "$SIZE" "$(t internal_files_short)" "$FILES" ""
            sleep "$REFRESH_INTERVAL"
        done
    fi
}

# ---------------------------------------------------------------------------
# Modo 4: logs — logs do container Docker em tempo real
# ---------------------------------------------------------------------------
cmd_logs() {
    if ! command -v docker &>/dev/null; then
        log_error "$(t docker_not_found)"
        exit 1
    fi

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}[!]${NC} $(t container_not_running) '${CONTAINER_NAME}'."
        echo ""
        echo -e "  $(t active_containers)"
        docker ps --format "  - {{.Names}} ({{.Status}})" 2>/dev/null || echo "  $(t none)"
        exit 1
    fi

    echo -e "${DIM}  $(t showing_logs) '${CONTAINER_NAME}' (Ctrl+C)${NC}"
    echo -e "${DIM}  $(t logs_tip)${NC}"
    echo ""
    docker logs -f --tail=30 "$CONTAINER_NAME"
}

# ---------------------------------------------------------------------------
# Modo 5: mcp-logs — logs de uso do MCP (quem acessa + ferramenta)
# ---------------------------------------------------------------------------
cmd_mcp_logs() {
    ensure_venv

    if [[ ! -f "$MCP_USAGE_LOG" ]]; then
        echo -e "${YELLOW}[!]${NC} $(t mcp_log_missing) ${MCP_USAGE_LOG}"
        echo -e "    $(t mcp_log_hint1)"
        echo -e "    $(t mcp_log_hint2)"
        exit 1
    fi

    echo -e "${DIM}  $(t showing_mcp_usage)${NC}"
    echo -e "${DIM}  $(t source): ${MCP_USAGE_LOG}${NC}"
    echo ""

    # Formata JSONL em linhas legíveis:
    # [timestamp] actor=<ator> tool=<ferramenta> event=<start/end> status=<status> client=<processo>
    tail -n 50 -f "$MCP_USAGE_LOG" | while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        "${VENV_PYTHON}" - "$line" <<'PYEOF'
import json
import sys

raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception:
    print(raw)
    sys.exit(0)

ts = data.get("timestamp", "-")
actor = data.get("actor", "-")
tool = data.get("tool", "-")
event = data.get("event", "-")
client = data.get("client_process", "-")
details = data.get("details", {}) or {}
status = details.get("status", "-")

print(f"[{ts}] actor={actor} tool={tool} event={event} status={status} client={client}")
PYEOF
    done
}

# ---------------------------------------------------------------------------
# Modo 6: mcp-summary — visão agregada de uso do MCP
# ---------------------------------------------------------------------------
cmd_mcp_summary() {
    ensure_venv

    if [[ ! -f "$MCP_USAGE_LOG" ]]; then
        echo -e "${YELLOW}[!]${NC} $(t mcp_log_missing) ${MCP_USAGE_LOG}"
        echo -e "    $(t mcp_log_hint_short)"
        exit 1
    fi

    MONITOR_LANG="${UI_LANG}" "${VENV_PYTHON}" - "$MCP_USAGE_LOG" <<'PYEOF'
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

LANG = os.getenv("MONITOR_LANG", "pt-br").lower()
IS_EN = LANG == "en-us"

path = sys.argv[1]
now = datetime.now(timezone.utc)
window_24h = now - timedelta(hours=24)

total_events = 0
total_calls_end = 0
ok_calls = 0
error_calls = 0

tools_all = Counter()
actors_all = Counter()
tools_24h = Counter()
actors_24h = Counter()
status_24h = Counter()

def parse_ts(raw: str):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue

        total_events += 1
        tool = item.get("tool", "-")
        actor = item.get("actor", "-")
        event = item.get("event", "-")
        details = item.get("details", {}) or {}
        status = details.get("status", "-")
        ts = parse_ts(item.get("timestamp", ""))

        if event == "tool_call_end":
            total_calls_end += 1
            if status == "ok":
                ok_calls += 1
            elif status == "error":
                error_calls += 1
            tools_all[tool] += 1
            actors_all[actor] += 1

            if ts and ts >= window_24h:
                tools_24h[tool] += 1
                actors_24h[actor] += 1
                status_24h[status] += 1

print("MCP usage summary" if IS_EN else "Resumo de uso MCP")
print("-" * 72)
print(f"{'Log file' if IS_EN else 'Arquivo de log'}: {path}")
print(f"{'Total events (start/end)' if IS_EN else 'Eventos totais (start/end)'}: {total_events}")
print(f"{'Completed calls (tool_call_end)' if IS_EN else 'Chamadas finalizadas (tool_call_end)'}: {total_calls_end}")
print(f"{'Completed status' if IS_EN else 'Status finalizadas'}: ok={ok_calls} | error={error_calls}")
print()

print("Top tools (overall)" if IS_EN else "Top ferramentas (geral)")
if tools_all:
    for name, count in tools_all.most_common(10):
        print(f"  - {name}: {count}")
else:
    print("  - no data" if IS_EN else "  - sem dados")
print()

print("Top actors (overall)" if IS_EN else "Top atores (geral)")
if actors_all:
    for name, count in actors_all.most_common(10):
        print(f"  - {name}: {count}")
else:
    print("  - no data" if IS_EN else "  - sem dados")
print()

print("Last 24h" if IS_EN else "Ultimas 24h")
print(f"  - {'completed total' if IS_EN else 'total finalizadas'}: {sum(tools_24h.values())}")
print(f"  - status: {dict(status_24h) if status_24h else ({'no_data': 0} if IS_EN else {'sem_dados': 0})}")

print("  - top tools (24h)" if IS_EN else "  - top ferramentas (24h)")
if tools_24h:
    for name, count in tools_24h.most_common(10):
        print(f"    * {name}: {count}")
else:
    print("    * no data" if IS_EN else "    * sem dados")

print("  - top actors (24h)" if IS_EN else "  - top atores (24h)")
if actors_24h:
    for name, count in actors_24h.most_common(10):
        print(f"    * {name}: {count}")
else:
    print("    * no data" if IS_EN else "    * sem dados")
PYEOF
}

# ---------------------------------------------------------------------------
# Modo 7: full — painel completo
# ---------------------------------------------------------------------------
cmd_full() {
    ensure_venv

    echo -e "${DIM}  $(t panel_updates) ${REFRESH_INTERVAL}s — Ctrl+C${NC}"
    echo ""
    sleep 1

    while true; do
        print_header

        # Status do container
        echo -e "  ${BOLD}$(t container_docker)${NC}"
        if command -v docker &>/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
            UPTIME=$(docker inspect --format='{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null | cut -c1-19 | tr 'T' ' ')
            echo -e "  ${GREEN}●${NC} ${CONTAINER_NAME} $(t running)  ${DIM}($(t started): ${UPTIME}Z)${NC}"
        else
            echo -e "  ${RED}●${NC} ${CONTAINER_NAME} $(t offline)"
        fi
        echo ""

        # Tamanho em disco
        echo -e "  ${BOLD}$(t disk_db)${NC}"
        if [[ -d "$RAG_DB_DIR" ]]; then
            DISK_SIZE=$(du -sh "$RAG_DB_DIR" 2>/dev/null | cut -f1)
            DISK_FILES=$(find "$RAG_DB_DIR" -type f 2>/dev/null | wc -l)
            echo -e "  ${CYAN}${DISK_SIZE}${NC}  ${DIM}(${DISK_FILES} $(t internal_files) ${RAG_DB_DIR})${NC}"
        else
            echo -e "${YELLOW}$(t dir_not_found) ${RAG_DB_DIR}${NC}"
        fi
        echo ""

        # Coleções e chunks
        echo -e "  ${BOLD}$(t collections_chunks)${NC}"
        MONITOR_LANG="${UI_LANG}" "${VENV_PYTHON}" - 2>/dev/null << 'PYEOF'
import chromadb, os, sys

GREEN  = "\033[0;32m"; CYAN   = "\033[0;36m"
YELLOW = "\033[1;33m"; RED    = "\033[0;31m"
BOLD   = "\033[1m";    DIM    = "\033[2m"; NC = "\033[0m"
LANG = os.getenv("MONITOR_LANG", "pt-br").lower()
IS_EN = LANG == "en-us"

try:
    client = chromadb.HttpClient(host="localhost", port=8000)
    client.heartbeat()
except Exception as e:
    msg = "ChromaDB unreachable" if IS_EN else "ChromaDB inacessível"
    print(f"  {RED}{msg}: {e}{NC}")
    sys.exit(0)

collections = client.list_collections()

if not collections:
    if IS_EN:
        print(f"  {YELLOW}Database empty — no collections found.{NC}")
        print("  Run: indexer_full.py")
    else:
        print(f"  {YELLOW}Banco vazio — nenhuma coleção encontrada.{NC}")
        print("  Execute: indexer_full.py")
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
    print(f"  {CYAN}{bar}{NC}  {BOLD}{count:,}{NC} chunks  |  {len(unique_files):,} {'files' if IS_EN else 'arquivos'}")
    print()

print(f"  {'─'*55}")
if IS_EN:
    print(f"  {BOLD}Total: {CYAN}{total_chunks:,} chunks{NC}{BOLD} in {total_files:,} file(s){NC}")
else:
    print(f"  {BOLD}Total: {CYAN}{total_chunks:,} chunks{NC}{BOLD} em {total_files:,} arquivo(s){NC}")
PYEOF

        echo ""
        echo -e "  ${DIM}$(t last_update): $(date '+%H:%M:%S') — $(t next_in) ${REFRESH_INTERVAL}s${NC}"

        sleep "$REFRESH_INTERVAL"
    done
}

# ---------------------------------------------------------------------------
# Modo 8: reset — remove todas as coleções do ChromaDB
# ---------------------------------------------------------------------------
cmd_reset() {
    ensure_venv

    echo ""
    echo -e "${RED}${BOLD}$(t reset_attention)${NC}"
    echo -e "${YELLOW}$(t reset_irreversible)${NC}"
    echo ""

    read -rp "$(t confirm_continue) ${YES_NO_HINT}: " confirm_reset
    confirm_reset="$(echo "${confirm_reset:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ "$confirm_reset" != "s" && "$confirm_reset" != "sim" && "$confirm_reset" != "y" && "$confirm_reset" != "yes" ]]; then
        log_info "$(t reset_cancelled)"
        exit 0
    fi

    local reset_keyword="ZERAR"
    if [[ "$UI_LANG" == "en-us" ]]; then
        reset_keyword="RESET"
    fi

    read -rp "$(t confirm_type_reset) " final_confirmation
    if [[ "${final_confirmation:-}" != "${reset_keyword}" ]]; then
        log_info "$(t reset_cancelled)"
        exit 0
    fi

    MONITOR_LANG="${UI_LANG}" "${VENV_PYTHON}" - << 'PYEOF'
import os
import sys
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
LANG = os.getenv("MONITOR_LANG", "pt-br").lower()
IS_EN = LANG == "en-us"

try:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()
except Exception as e:
    if IS_EN:
        print(f"[ERROR] Could not connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}: {e}")
    else:
        print(f"[ERRO] Não foi possível conectar ao ChromaDB em {CHROMA_HOST}:{CHROMA_PORT}: {e}")
    sys.exit(1)

collections = client.list_collections()
removed = 0
for col in collections:
    client.delete_collection(name=col.name)
    removed += 1

if IS_EN:
    print(f"[+] Reset completed. Collections removed: {removed}")
else:
    print(f"[+] Reset concluído. Coleções removidas: {removed}")
PYEOF
}

# ---------------------------------------------------------------------------
# Menu interativo
# ---------------------------------------------------------------------------
cmd_menu() {
    ensure_venv
    print_header

    if is_chroma_running; then
        echo -e "  ${GREEN}${BOLD}$(t status_online)${NC}"
    else
        echo -e "  ${RED}${BOLD}$(t status_offline)${NC}"
        echo -e "  ${DIM}$(t start_with): docker compose -f ~/docker-chromadb/docker-compose.yml up -d${NC}"
    fi
    echo ""
    echo -e "  ${BOLD}$(t choose_option)${NC}"
    echo ""
    echo -e "  ${CYAN}$(t menu_1)${NC}"
    echo -e "  ${CYAN}$(t menu_2)${NC}"
    echo -e "  ${CYAN}$(t menu_3)${NC}"
    echo -e "  ${CYAN}$(t menu_4)${NC}"
    echo -e "  ${CYAN}$(t menu_5)${NC}"
    echo -e "  ${CYAN}$(t menu_6)${NC}"
    echo -e "  ${CYAN}$(t menu_7)${NC}"
    echo -e "  ${CYAN}$(t menu_8)${NC}"
    echo ""
    echo -e "  ${CYAN}$(t menu_0)${NC}"
    echo ""
    read -rp "  $(t option_prompt) " choice

    case "$choice" in
        1) echo ""; cmd_chunks ;;
        2) echo ""; cmd_watch ;;
        3) cmd_disk ;;
        4) cmd_logs ;;
        5) cmd_mcp_logs ;;
        6) cmd_mcp_summary ;;
        7) cmd_full ;;
        8) cmd_reset ;;
        0) exit 0 ;;
        *) echo -e "\n  ${YELLOW}$(t invalid_option)${NC}"; sleep 1; cmd_menu ;;
    esac
}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
select_ui_language
case "${1:-menu}" in
    chunks)  cmd_chunks ;;
    watch)   cmd_watch  ;;
    disk)    cmd_disk   ;;
    logs)    cmd_logs   ;;
    mcp-logs) cmd_mcp_logs ;;
    mcp-summary) cmd_mcp_summary ;;
    full)    cmd_full   ;;
    reset)   cmd_reset  ;;
    menu)    cmd_menu   ;;
    *)
        echo -e "${RED}[$(t err_prefix)]${NC} $(t unknown_mode): '$1'"
        echo "$(t usage)"
        exit 1
        ;;
esac
