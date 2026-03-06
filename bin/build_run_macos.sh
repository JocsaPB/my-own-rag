#!/usr/bin/env bash
# build_run_macos.sh — Gera o instalador macOS auto-suficiente (rag-setup-macos.run)

set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${BIN_DIR}")"
OUTPUT="${ROOT_DIR}/rag-setup-macos.run"

COMPOSE_FILE="${BIN_DIR}/docker-compose.yml"
REQUIREMENTS_FILE="${BIN_DIR}/requirements.txt"
INDEXER_FILE="${BIN_DIR}/indexer_full.py"
MCP_FILE="${BIN_DIR}/mcp_server.py"
MODEL_DL_HF_FILE="${BIN_DIR}/download_model_from_hugginface.py"
MODEL_DL_MS_FILE="${BIN_DIR}/download_model_from_modelscope.py"

for f in "$COMPOSE_FILE" "$REQUIREMENTS_FILE" "$INDEXER_FILE" "$MCP_FILE" "$MODEL_DL_HF_FILE" "$MODEL_DL_MS_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERRO] Arquivo não encontrado: $f"
        exit 1
    fi
done

sha256_file() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    else
        shasum -a 256 "$file" | awk '{print $1}'
    fi
}

b64_encode() {
    local input="$1"
    python3 - "$input" <<'PYEOF'
import base64
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
print(base64.b64encode(path.read_bytes()).decode("ascii"), end="")
PYEOF
}

MCP_SHA256="$(sha256_file "$MCP_FILE")"

echo "[+] Gerando base64 dos arquivos (macOS installer)..."

B64_COMPOSE="$(b64_encode "$COMPOSE_FILE")"
B64_REQUIREMENTS="$(b64_encode "$REQUIREMENTS_FILE")"
B64_INDEXER="$(b64_encode "$INDEXER_FILE")"
B64_MCP="$(b64_encode "$MCP_FILE")"
B64_MODEL_DL_HF="$(b64_encode "$MODEL_DL_HF_FILE")"
B64_MODEL_DL_MS="$(b64_encode "$MODEL_DL_MS_FILE")"

echo "[+] Escrevendo $OUTPUT ..."

cat > "$OUTPUT" <<OUTER_EOF
#!/usr/bin/env bash
# =============================================================================
# rag-setup-macos.run — Instalador macOS auto-suficiente (ChromaDB + MCP + RAG)
# =============================================================================
# Gerado automaticamente por build_run_macos.sh
# Versão: $(date '+%Y-%m-%d %H:%M')
# MCP checksum (payload sem shebang): ${MCP_SHA256}
#
# Uso:
#   chmod +x rag-setup-macos.run
#   ./rag-setup-macos.run [path/to/project] [--skip-index] [--only-index]
#   ./rag-setup-macos.run --reinstall
#   ./rag-setup-macos.run --change-model|-cm
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
SKIP_INDEX=false
ONLY_INDEX=false
REINSTALL=false
CHANGE_MODEL=false
CUSTOM_PROJECT_DIR=""

for arg in "\$@"; do
    case "\$arg" in
        --skip-index) SKIP_INDEX=true ;;
        --only-index) ONLY_INDEX=true ;;
        --reinstall) REINSTALL=true ;;
        --change-model|-cm|--chage-model|-cg) CHANGE_MODEL=true ;;
        --help|-h)
            echo "Usage: \$0 [path/to/project] [--skip-index] [--only-index] [--reinstall] [--change-model|-cm]"
            exit 0 ;;
        -*)
            echo "Unknown option: \$arg"
            exit 1 ;;
        *)
            CUSTOM_PROJECT_DIR="\$arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

UI_LANG="\${RAG_SETUP_LANG:-}"
YES_NO_HINT="[s/N]"

set_lang_defaults() {
    if [[ "\$UI_LANG" == "en-us" ]]; then
        YES_NO_HINT="[y/N]"
    else
        YES_NO_HINT="[s/N]"
    fi
}

select_ui_language() {
    if [[ -n "\$UI_LANG" ]]; then
        UI_LANG="\$(echo "\$UI_LANG" | tr '[:upper:]' '[:lower:]')"
        case "\$UI_LANG" in
            pt-br|pt|en-us|en) ;;
            *) UI_LANG="pt-br" ;;
        esac
        [[ "\$UI_LANG" == "pt" ]] && UI_LANG="pt-br"
        [[ "\$UI_LANG" == "en" ]] && UI_LANG="en-us"
        set_lang_defaults
        return
    fi

    if [[ ! -t 0 ]]; then
        UI_LANG="pt-br"
        set_lang_defaults
        return
    fi

    echo ""
    echo -e "\${GREEN}Idioma / Language: [1] PT-BR [2] EN-US (padrão/default: 1)\${NC}"
    read -r -p "> " LANG_CHOICE
    case "\$LANG_CHOICE" in
        2|en|EN|en-us|EN-US|english|English) UI_LANG="en-us" ;;
        *) UI_LANG="pt-br" ;;
    esac
    set_lang_defaults
}

t() {
    local key="\$1"
    if [[ "\$UI_LANG" == "en-us" ]]; then
        case "\$key" in
            err_prefix) echo "ERROR" ;;
            title) echo "RAG Local Setup (macOS) — ChromaDB + MCP" ;;
            project) echo "Project" ;;
            platform_only) echo "This installer is only for macOS (Darwin)." ;;
            extracting) echo "Extracting embedded files" ;;
            extracted_to) echo "Extracted to" ;;
            checking_prereq) echo "Checking prerequisites" ;;
            python_missing) echo "Python 3 not found. Install with Homebrew: brew install python" ;;
            python_min) echo "Python 3.10+ required. Current" ;;
            py_venv_missing) echo "Python venv module not found." ;;
            docker_missing) echo "Docker not found. Install Docker Desktop: brew install --cask docker" ;;
            docker_daemon) echo "Docker is not running. Open Docker Desktop and wait until it is ready." ;;
            compose_missing) echo "Docker Compose not available (docker compose)." ;;
            prereq_ok) echo "Prerequisites OK." ;;
            no_curl) echo "curl not found. Chroma healthcheck will be skipped." ;;
            reset_start) echo "Resetting previous environment" ;;
            reset_done) echo "Reset completed." ;;
            op_cancelled) echo "Operation canceled by user." ;;
            change_model_msg) echo "Model change requested. Chroma reset + full reindex required." ;;
            change_model_confirm) echo "Confirm Chroma reset and full reindex?" ;;
            change_model_noninteractive) echo "Non-interactive mode: proceeding with reset for --change-model." ;;
            only_index_mode) echo "Mode: only index" ;;
            venv_not_found) echo "Venv not found at" ;;
            path_not_found) echo "Path not found" ;;
            indexing) echo "Indexing" ;;
            indexing_done) echo "Indexing completed." ;;
            index_failed_code) echo "Indexer failed. Exit code" ;;
            index_oom_title) echo "Indexing stopped due to out-of-memory (OOM killer, exit 137)." ;;
            section_venv) echo "Setting up Python venv (~/.rag_venv)" ;;
            deps_ok) echo "Dependencies already installed. Skipping." ;;
            deps_reinstall) echo "Installing/updating dependencies..." ;;
            creating_venv) echo "Creating venv at" ;;
            venv_created) echo "Venv created." ;;
            upgrading_pip) echo "Upgrading pip..." ;;
            deps_installed) echo "Dependencies installed." ;;
            section_chroma) echo "Setting up ChromaDB (Docker Desktop)" ;;
            compose_installed) echo "docker-compose.yml installed at" ;;
            compose_keep) echo "docker-compose.yml already exists. Keeping current file." ;;
            chroma_running) echo "chromadb-rag container already running." ;;
            chroma_start) echo "Starting ChromaDB..." ;;
            chroma_wait) echo "Waiting ChromaDB to initialize..." ;;
            chroma_ready) echo "ChromaDB is responding at http://localhost:8000" ;;
            chroma_timeout) echo "ChromaDB did not respond in time. Check Docker Desktop and container logs." ;;
            section_install_mcp) echo "Installing mcp-rag-server globally" ;;
            mcp_keep) echo "mcp-rag-server already up to date. Keeping." ;;
            mcp_outdated) echo "mcp-rag-server is outdated." ;;
            mcp_prompt_update) echo "Update mcp-rag-server now? \$YES_NO_HINT " ;;
            mcp_prompt_reinstall) echo "mcp-rag-server is current. Reinstall anyway? \$YES_NO_HINT " ;;
            mcp_skip) echo "Keeping current mcp-rag-server." ;;
            mcp_installed) echo "mcp-rag-server installed" ;;
            mod_hf_installed) echo "Download module installed" ;;
            mod_ms_installed) echo "Optional provider module installed" ;;
            shebang_set) echo "Shebang" ;;
            path_added) echo "Added ~/.local/bin to PATH in" ;;
            section_mcp_cfg) echo "Optional MCP configuration (Claude/Cursor)" ;;
            no_cfg_files) echo "No config files detected for automatic MCP update." ;;
            cfg_all_current) echo "MCP 'rag-codebase' already up to date in detected files. Skipping." ;;
            cfg_detected) echo "Config files pending update" ;;
            ask_apply_cfg) echo "Apply MCP 'rag-codebase' update in these files? \$YES_NO_HINT " ;;
            noninteractive_cfg_skip) echo "Non-interactive mode: skipping automatic MCP config update." ;;
            cannot_update_cfg) echo "Could not update" ;;
            already_updated) echo "MCP 'rag-codebase' already up to date. Skipping." ;;
            updated_cfg) echo "MCP 'rag-codebase' updated to version" ;;
            replaced_cfg) echo "Old RAG key replaced by 'rag-codebase' (version" ;;
            added_cfg) echo "MCP 'rag-codebase' added (version" ;;
            setup_done) echo "Setup completed!" ;;
            next) echo "Next" ;;
            next_1) echo "Restart Claude Code CLI" ;;
            next_2) echo "Use semantic_search_code" ;;
            next_3) echo "Reindex: ./rag-setup-macos.run --only-index" ;;
            hf_detected) echo "HF token detected in environment." ;;
            hf_noninteractive) echo "Non-interactive mode: continuing without HF token prompt." ;;
            hf_title) echo "Hugging Face token (optional)" ;;
            hf_desc) echo "Speeds up model download/rate limits." ;;
            hf_prompt_now) echo "Provide HF token now? \$YES_NO_HINT " ;;
            hf_prompt_paste) echo "Paste HF_TOKEN (hidden): " ;;
            hf_set) echo "HF token set for this run." ;;
            hf_empty) echo "Empty token. Continuing without auth." ;;
            hf_skip) echo "Continuing without HF token." ;;
            invalid_option) echo "Invalid option. Use allowed answers." ;;
            *) echo "\$key" ;;
        esac
    else
        case "\$key" in
            err_prefix) echo "ERRO" ;;
            title) echo "RAG Local Setup (macOS) — ChromaDB + MCP" ;;
            project) echo "Projeto" ;;
            platform_only) echo "Este instalador é apenas para macOS (Darwin)." ;;
            extracting) echo "Extraindo arquivos embutidos" ;;
            extracted_to) echo "Extraído em" ;;
            checking_prereq) echo "Verificando pré-requisitos" ;;
            python_missing) echo "Python 3 não encontrado. Instale via Homebrew: brew install python" ;;
            python_min) echo "Python 3.10+ necessário. Atual" ;;
            py_venv_missing) echo "Módulo venv do Python não encontrado." ;;
            docker_missing) echo "Docker não encontrado. Instale Docker Desktop: brew install --cask docker" ;;
            docker_daemon) echo "Docker não está rodando. Abra o Docker Desktop e aguarde ficar pronto." ;;
            compose_missing) echo "Docker Compose indisponível (docker compose)." ;;
            prereq_ok) echo "Pré-requisitos OK." ;;
            no_curl) echo "curl não encontrado. Healthcheck do Chroma será pulado." ;;
            reset_start) echo "Zerando ambiente anterior" ;;
            reset_done) echo "Reset concluído." ;;
            op_cancelled) echo "Operação cancelada pelo usuário." ;;
            change_model_msg) echo "Troca de modelo solicitada. Reset do Chroma + reindexação total." ;;
            change_model_confirm) echo "Confirmar reset do Chroma e reindexação total?" ;;
            change_model_noninteractive) echo "Modo não interativo: seguindo com reset por --change-model." ;;
            only_index_mode) echo "Modo: apenas indexação" ;;
            venv_not_found) echo "Venv não encontrado em" ;;
            path_not_found) echo "Caminho não encontrado" ;;
            indexing) echo "Indexando" ;;
            indexing_done) echo "Indexação concluída." ;;
            index_failed_code) echo "Indexador falhou. Código de saída" ;;
            index_oom_title) echo "Indexação interrompida por falta de memória (OOM killer, exit 137)." ;;
            section_venv) echo "Configurando venv Python (~/.rag_venv)" ;;
            deps_ok) echo "Dependências já instaladas. Pulando." ;;
            deps_reinstall) echo "Instalando/atualizando dependências..." ;;
            creating_venv) echo "Criando venv em" ;;
            venv_created) echo "Venv criado." ;;
            upgrading_pip) echo "Atualizando pip..." ;;
            deps_installed) echo "Dependências instaladas." ;;
            section_chroma) echo "Configurando ChromaDB (Docker Desktop)" ;;
            compose_installed) echo "docker-compose.yml instalado em" ;;
            compose_keep) echo "docker-compose.yml já existe. Mantendo arquivo atual." ;;
            chroma_running) echo "Container chromadb-rag já está rodando." ;;
            chroma_start) echo "Iniciando ChromaDB..." ;;
            chroma_wait) echo "Aguardando ChromaDB inicializar..." ;;
            chroma_ready) echo "ChromaDB respondendo em http://localhost:8000" ;;
            chroma_timeout) echo "ChromaDB não respondeu no tempo esperado. Verifique Docker Desktop e logs." ;;
            section_install_mcp) echo "Instalando mcp-rag-server globalmente" ;;
            mcp_keep) echo "mcp-rag-server já está atualizado. Mantendo." ;;
            mcp_outdated) echo "mcp-rag-server está desatualizado." ;;
            mcp_prompt_update) echo "Atualizar mcp-rag-server agora? \$YES_NO_HINT " ;;
            mcp_prompt_reinstall) echo "mcp-rag-server já atualizado. Reinstalar mesmo assim? \$YES_NO_HINT " ;;
            mcp_skip) echo "Mantendo mcp-rag-server atual." ;;
            mcp_installed) echo "mcp-rag-server instalado" ;;
            mod_hf_installed) echo "Módulo de download instalado" ;;
            mod_ms_installed) echo "Módulo de provider opcional instalado" ;;
            shebang_set) echo "Shebang" ;;
            path_added) echo "Adicionado ~/.local/bin ao PATH em" ;;
            section_mcp_cfg) echo "Configuração opcional de MCP (Claude/Cursor)" ;;
            no_cfg_files) echo "Nenhum arquivo de config detectado para atualização automática do MCP." ;;
            cfg_all_current) echo "MCP 'rag-codebase' já está atualizado nos arquivos detectados. Pulando." ;;
            cfg_detected) echo "Arquivos de config pendentes de atualização" ;;
            ask_apply_cfg) echo "Aplicar atualização do MCP 'rag-codebase' nesses arquivos? \$YES_NO_HINT " ;;
            noninteractive_cfg_skip) echo "Modo não interativo: pulando atualização automática de config MCP." ;;
            cannot_update_cfg) echo "Não foi possível atualizar" ;;
            already_updated) echo "MCP 'rag-codebase' já atualizado. Ignorando." ;;
            updated_cfg) echo "MCP 'rag-codebase' atualizado para versão" ;;
            replaced_cfg) echo "Chave RAG antiga substituída por 'rag-codebase' (versão" ;;
            added_cfg) echo "MCP 'rag-codebase' adicionado (versão" ;;
            setup_done) echo "Setup concluído!" ;;
            next) echo "Próximos" ;;
            next_1) echo "Reinicie o Claude Code CLI" ;;
            next_2) echo "Use semantic_search_code" ;;
            next_3) echo "Reindexar: ./rag-setup-macos.run --only-index" ;;
            hf_detected) echo "Token HF detectado no ambiente." ;;
            hf_noninteractive) echo "Modo não interativo: seguindo sem prompt de token HF." ;;
            hf_title) echo "Token Hugging Face (opcional)" ;;
            hf_desc) echo "Acelera download de modelos e limites de taxa." ;;
            hf_prompt_now) echo "Informar HF token agora? \$YES_NO_HINT " ;;
            hf_prompt_paste) echo "Cole o HF_TOKEN (oculto): " ;;
            hf_set) echo "HF token definido para este run." ;;
            hf_empty) echo "Token vazio. Seguindo sem autenticação." ;;
            hf_skip) echo "Seguindo sem HF token." ;;
            invalid_option) echo "Opção inválida. Use respostas permitidas." ;;
            *) echo "\$key" ;;
        esac
    fi
}

log_info()  { echo -e "\${GREEN}[+]\${NC} \$*"; }
log_warn()  { echo -e "\${YELLOW}[!]\${NC} \$*"; }
log_error() { echo -e "\${RED}[\$(t err_prefix)]\${NC} \$*" >&2; }

is_yes_answer() {
    local ans="\$(echo "\${1:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ "\$UI_LANG" == "en-us" ]]; then
        [[ "\$ans" == "y" || "\$ans" == "yes" ]]
    else
        [[ "\$ans" == "s" || "\$ans" == "sim" || "\$ans" == "y" || "\$ans" == "yes" ]]
    fi
}

is_no_answer() {
    local ans="\$(echo "\${1:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ -z "\$ans" ]]; then
        return 0
    fi
    if [[ "\$UI_LANG" == "en-us" ]]; then
        [[ "\$ans" == "n" || "\$ans" == "no" ]]
    else
        [[ "\$ans" == "n" || "\$ans" == "nao" || "\$ans" == "não" || "\$ans" == "no" ]]
    fi
}

ask_yes_no_loop() {
    local prompt="\$1"
    local answer=""
    while true; do
        read -r -p "\$prompt" answer
        if is_yes_answer "\$answer"; then
            return 0
        fi
        if is_no_answer "\$answer"; then
            return 1
        fi
        log_warn "\$(t invalid_option)"
    done
}

decode_to_file() {
    local payload="\$1"
    local destination="\$2"
    printf '%s' "\$payload" | python3 - "\$destination" <<'PYEOF'
import base64
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
out.write_bytes(base64.b64decode(sys.stdin.read().encode("ascii")))
PYEOF
}

replace_shebang() {
    local file="\$1"
    local new_shebang="\$2"
    local tmp_file
    tmp_file="\$(mktemp "\${TMPDIR:-/tmp}/rag-shebang.XXXXXX")"
    {
        printf '%s\n' "#!\${new_shebang}"
        tail -n +2 "\$file"
    } > "\$tmp_file"
    mv "\$tmp_file" "\$file"
}

detect_current_mcp_version() {
    python3 - "\${HOME}" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

home = Path(sys.argv[1]).expanduser()
paths = [
    home / ".claude.json",
    home / ".cursor" / "mcp.json",
    home / "Library" / "Application Support" / "Cursor" / "User" / "mcp.json",
    home / ".config" / "Cursor" / "User" / "mcp.json",
]
version_re = re.compile(r"^(\d+)\.(\d+)$")
best = None

for p in paths:
    if not p.exists():
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(data, dict):
        continue
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        continue
    cfg = servers.get("rag-codebase")
    if not isinstance(cfg, dict):
        continue
    version = cfg.get("version")
    if not isinstance(version, str):
        continue
    m = version_re.match(version.strip())
    if not m:
        continue
    parsed = (int(m.group(1)), int(m.group(2)))
    if best is None or parsed > best:
        best = parsed

if best is None:
    print("1.0")
else:
    print(f"{best[0]}.{best[1]}")
PYEOF
}

select_ui_language

USER_HOME="\${HOME}"
VENV_DIR="\${USER_HOME}/.rag_venv"
VENV_PYTHON="\${VENV_DIR}/bin/python3"
VENV_PIP="\${VENV_DIR}/bin/pip"
DOCKER_COMPOSE_DIR="\${USER_HOME}/docker-chromadb"
RAG_DB_DIR="\${USER_HOME}/.rag_db"
BIN_DIR="\${USER_HOME}/.local/bin"
MCP_SERVER_DEST="\${BIN_DIR}/mcp-rag-server"
MODEL_DL_HF_DEST="\${BIN_DIR}/download_model_from_hugginface.py"
MODEL_DL_MS_DEST="\${BIN_DIR}/download_model_from_modelscope.py"
MODEL_CACHE_DIR="\${USER_HOME}/.cache/my-custom-rag-python/models"
EXTRACT_DIR="\$(mktemp -d "\${TMPDIR:-/tmp}/rag-setup-macos.XXXXXX")"

if [[ -n "\$CUSTOM_PROJECT_DIR" ]]; then
    PROJECT_DIR="\$(cd "\$CUSTOM_PROJECT_DIR" && pwd)"
else
    PROJECT_DIR="\$(pwd)"
fi

trap 'rm -rf "\$EXTRACT_DIR"' EXIT

echo ""
echo -e "\${BOLD}\${BLUE}================================================================\${NC}"
echo -e "\${BOLD}\${BLUE}  \$(t title)\${NC}"
echo -e "\${BOLD}\${BLUE}================================================================\${NC}"
echo -e "  \$(t project): \${BOLD}\${PROJECT_DIR}\${NC}"
echo ""

if [[ "\$(uname -s)" != "Darwin" ]]; then
    log_error "\$(t platform_only)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------
log_info "\$(t extracting)"

B64_COMPOSE="${B64_COMPOSE}"
B64_REQUIREMENTS="${B64_REQUIREMENTS}"
B64_INDEXER="${B64_INDEXER}"
B64_MCP="${B64_MCP}"
B64_MODEL_DL_HF="${B64_MODEL_DL_HF}"
B64_MODEL_DL_MS="${B64_MODEL_DL_MS}"

decode_to_file "\$B64_COMPOSE" "\${EXTRACT_DIR}/docker-compose.yml"
decode_to_file "\$B64_REQUIREMENTS" "\${EXTRACT_DIR}/requirements.txt"
decode_to_file "\$B64_INDEXER" "\${EXTRACT_DIR}/indexer_full.py"
decode_to_file "\$B64_MCP" "\${EXTRACT_DIR}/mcp_server.py"
decode_to_file "\$B64_MODEL_DL_HF" "\${EXTRACT_DIR}/download_model_from_hugginface.py"
decode_to_file "\$B64_MODEL_DL_MS" "\${EXTRACT_DIR}/download_model_from_modelscope.py"

log_info "\$(t extracted_to): \${EXTRACT_DIR}"

# ---------------------------------------------------------------------------
# Pré-requisitos
# ---------------------------------------------------------------------------
log_info "\$(t checking_prereq)"

if ! command -v python3 >/dev/null 2>&1; then
    log_error "\$(t python_missing)"
    exit 1
fi

PY_VER="\$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
PY_MAJOR="\$(echo "\$PY_VER" | cut -d. -f1)"
PY_MINOR="\$(echo "\$PY_VER" | cut -d. -f2)"
if [[ "\$PY_MAJOR" -lt 3 ]] || [[ "\$PY_MAJOR" -eq 3 && "\$PY_MINOR" -lt 10 ]]; then
    log_error "\$(t python_min): \$PY_VER"
    exit 1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
    log_error "\$(t py_venv_missing)"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    log_error "\$(t docker_missing)"
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    log_error "\$(t docker_daemon)"
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    log_error "\$(t compose_missing)"
    exit 1
fi
DOCKER_COMPOSE_CMD="docker compose"

if ! command -v curl >/dev/null 2>&1; then
    log_warn "\$(t no_curl)"
    HAS_CURL=false
else
    HAS_CURL=true
fi

log_info "\$(t prereq_ok)"

reset_rag_environment() {
    log_info "\$(t reset_start)"
    rm -rf "\${VENV_DIR}" "\${RAG_DB_DIR}" "\${DOCKER_COMPOSE_DIR}" "\${MODEL_CACHE_DIR}"
    rm -f "\${MCP_SERVER_DEST}" "\${MODEL_DL_HF_DEST}" "\${MODEL_DL_MS_DEST}"
    log_info "\$(t reset_done)"
}

if [[ "\$CHANGE_MODEL" == "true" ]]; then
    if [[ "\$ONLY_INDEX" == "true" ]]; then
        ONLY_INDEX=false
    fi
    log_info "\$(t change_model_msg)"
    if [[ -t 0 ]]; then
        if ! ask_yes_no_loop "\$(t change_model_confirm) \$YES_NO_HINT "; then
            log_info "\$(t op_cancelled)"
            exit 0
        fi
    else
        log_warn "\$(t change_model_noninteractive)"
    fi
    export MCP_FORCE_MODEL_RECONFIG=1
    REINSTALL=true
    reset_rag_environment
fi

prompt_optional_hf_token() {
    if [[ -n "\${HF_TOKEN_PROMPTED:-}" ]]; then
        return
    fi
    HF_TOKEN_PROMPTED=1

    if [[ -n "\${HF_TOKEN:-}" || -n "\${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        log_info "\$(t hf_detected)"
        return
    fi

    if [[ ! -t 0 ]]; then
        log_info "\$(t hf_noninteractive)"
        return
    fi

    echo ""
    echo -e "\${BOLD}\$(t hf_title)\${NC}"
    echo -e "\${DIM}\$(t hf_desc)\${NC}"
    if ask_yes_no_loop "\$(t hf_prompt_now)"; then
        read -r -s -p "\$(t hf_prompt_paste)" INPUT_HF_TOKEN
        echo ""
        if [[ -n "\${INPUT_HF_TOKEN}" ]]; then
            export HF_TOKEN="\${INPUT_HF_TOKEN}"
            log_info "\$(t hf_set)"
        else
            log_warn "\$(t hf_empty)"
        fi
    else
        log_info "\$(t hf_skip)"
    fi
}

run_indexer_with_diagnostics() {
    local target_project_dir="\$1"
    local indexer_status=0

    local tokenizers_parallelism="\${TOKENIZERS_PARALLELISM:-false}"
    local force_model_reconfig="\${MCP_FORCE_MODEL_RECONFIG:-0}"

    set +e
    TOKENIZERS_PARALLELISM="\${tokenizers_parallelism}" \
    MCP_FORCE_MODEL_RECONFIG="\${force_model_reconfig}" \
    "\${VENV_PYTHON}" "\${EXTRACT_DIR}/indexer_full.py" "\${target_project_dir}"
    indexer_status=\$?
    set -e

    if [[ "\$indexer_status" -eq 137 ]]; then
        log_error "\$(t index_oom_title)"
    elif [[ "\$indexer_status" -ne 0 ]]; then
        log_error "\$(t index_failed_code): \${indexer_status}"
    fi

    return "\$indexer_status"
}

# ---------------------------------------------------------------------------
# --only-index
# ---------------------------------------------------------------------------
if [[ "\$ONLY_INDEX" == "true" ]]; then
    log_info "\$(t only_index_mode)"
    if [[ ! -f "\${VENV_PYTHON}" ]]; then
        log_error "\$(t venv_not_found) \${VENV_DIR}"
        exit 1
    fi
    if [[ ! -d "\${PROJECT_DIR}" ]]; then
        log_error "\$(t path_not_found): \${PROJECT_DIR}"
        exit 1
    fi
    prompt_optional_hf_token
    log_info "\$(t indexing): \${PROJECT_DIR}"
    run_indexer_with_diagnostics "\${PROJECT_DIR}"
    log_info "\$(t indexing_done)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Venv + dependências
# ---------------------------------------------------------------------------
log_info "\$(t section_venv)"

DEPS_OK=false
if [[ "\$REINSTALL" == "false" ]] && [[ -f "\${VENV_PYTHON}" ]]; then
    if "\${VENV_PYTHON}" -c "import chromadb, sentence_transformers, langchain_text_splitters, tqdm, mcp, transformers, sys; sys.exit(0 if int(transformers.__version__.split('.')[0]) < 5 else 1)" 2>/dev/null; then
        log_info "\$(t deps_ok)"
        DEPS_OK=true
    fi
fi

if [[ "\$DEPS_OK" == "false" ]]; then
    if [[ ! -f "\${VENV_PYTHON}" ]]; then
        log_info "\$(t creating_venv) \${VENV_DIR}"
        python3 -m venv "\${VENV_DIR}"
        log_info "\$(t venv_created)"
    fi

    log_info "\$(t upgrading_pip)"
    "\${VENV_PIP}" install --upgrade pip

    log_info "\$(t deps_reinstall)"
    "\${VENV_PIP}" install --progress-bar on -r "\${EXTRACT_DIR}/requirements.txt"

    log_info "\$(t deps_installed)"
fi

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
log_info "\$(t section_chroma)"

mkdir -p "\${RAG_DB_DIR}" "\${DOCKER_COMPOSE_DIR}"

if [[ "\$REINSTALL" == "true" ]] || [[ ! -f "\${DOCKER_COMPOSE_DIR}/docker-compose.yml" ]]; then
    cp "\${EXTRACT_DIR}/docker-compose.yml" "\${DOCKER_COMPOSE_DIR}/docker-compose.yml"
    log_info "\$(t compose_installed) \${DOCKER_COMPOSE_DIR}"
else
    log_info "\$(t compose_keep)"
fi

if docker ps --format '{{.Names}}' | grep -q '^chromadb-rag$'; then
    log_info "\$(t chroma_running)"
else
    log_info "\$(t chroma_start)"
    (cd "\${DOCKER_COMPOSE_DIR}" && \$DOCKER_COMPOSE_CMD up -d)

    log_info "\$(t chroma_wait)"
    WAITED=0
    while true; do
        if [[ "\$HAS_CURL" == "true" ]] && curl -sf "http://localhost:8000/api/v1/heartbeat" >/dev/null 2>&1; then
            log_info "\$(t chroma_ready)"
            break
        fi
        if [[ \$WAITED -ge 40 ]]; then
            log_warn "\$(t chroma_timeout)"
            break
        fi
        sleep 2
        WAITED=\$((WAITED+2))
    done
fi

# ---------------------------------------------------------------------------
# Instala mcp-rag-server
# ---------------------------------------------------------------------------
log_info "\$(t section_install_mcp)"

mkdir -p "\${BIN_DIR}"

NEEDS_INSTALL=true
MCP_WAS_OUTDATED=false
if [[ -f "\${MCP_SERVER_DEST}" ]]; then
    if cmp -s <(tail -n +2 "\${MCP_SERVER_DEST}") <(tail -n +2 "\${EXTRACT_DIR}/mcp_server.py"); then
        log_info "\$(t mcp_keep)"
    else
        MCP_WAS_OUTDATED=true
        log_warn "\$(t mcp_outdated)"
    fi

    if [[ "\$REINSTALL" == "true" ]]; then
        NEEDS_INSTALL=true
    elif [[ -t 0 ]]; then
        if [[ "\$MCP_WAS_OUTDATED" == "true" ]]; then
            if ask_yes_no_loop "\$(t mcp_prompt_update)"; then
                NEEDS_INSTALL=true
            else
                NEEDS_INSTALL=false
                log_info "\$(t mcp_skip)"
            fi
        else
            if ask_yes_no_loop "\$(t mcp_prompt_reinstall)"; then
                NEEDS_INSTALL=true
            else
                NEEDS_INSTALL=false
                log_info "\$(t mcp_skip)"
            fi
        fi
    else
        NEEDS_INSTALL=false
    fi
fi

if [[ "\$NEEDS_INSTALL" == "true" ]]; then
    cp "\${EXTRACT_DIR}/mcp_server.py" "\${MCP_SERVER_DEST}"
    cp "\${EXTRACT_DIR}/download_model_from_hugginface.py" "\${MODEL_DL_HF_DEST}"
    cp "\${EXTRACT_DIR}/download_model_from_modelscope.py" "\${MODEL_DL_MS_DEST}"

    replace_shebang "\${MCP_SERVER_DEST}" "\${VENV_PYTHON}"
    chmod +x "\${MCP_SERVER_DEST}"

    log_info "\$(t mcp_installed): \${MCP_SERVER_DEST}"
    log_info "\$(t mod_hf_installed): \${MODEL_DL_HF_DEST}"
    log_info "\$(t mod_ms_installed): \${MODEL_DL_MS_DEST}"
    log_info "\$(t shebang_set): \${VENV_PYTHON}"
fi

for RC in "\${USER_HOME}/.zshrc" "\${USER_HOME}/.zprofile" "\${USER_HOME}/.bash_profile"; do
    if [[ -f "\$RC" ]] && ! grep -qF '.local/bin' "\$RC"; then
        echo "" >> "\$RC"
        echo '# RAG setup — add local bin to PATH' >> "\$RC"
        echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> "\$RC"
        log_info "\$(t path_added) \$RC"
    fi
done

# ---------------------------------------------------------------------------
# Configuração MCP opcional
# ---------------------------------------------------------------------------
log_info "\$(t section_mcp_cfg)"

CLAUDE_JSON="\${USER_HOME}/.claude.json"
CURSOR_MCP_JSON_1="\${USER_HOME}/.cursor/mcp.json"
CURSOR_MCP_JSON_2="\${USER_HOME}/Library/Application Support/Cursor/User/mcp.json"
MCP_VERSION="\$(detect_current_mcp_version)"

TARGET_CONFIGS=()
TARGET_LABELS=()

_append_target() {
    local cfg="\$1"
    local label="\$2"
    for existing in "\${TARGET_CONFIGS[@]:-}"; do
        if [[ "\$existing" == "\$cfg" ]]; then
            return 0
        fi
    done
    TARGET_CONFIGS+=("\$cfg")
    TARGET_LABELS+=("\$label")
}

if [[ -f "\${CLAUDE_JSON}" ]]; then
    _append_target "\${CLAUDE_JSON}" "Claude Code"
fi
if [[ -f "\${CURSOR_MCP_JSON_1}" ]]; then
    _append_target "\${CURSOR_MCP_JSON_1}" "Cursor"
fi
if [[ -f "\${CURSOR_MCP_JSON_2}" ]]; then
    _append_target "\${CURSOR_MCP_JSON_2}" "Cursor"
fi

if [[ "\${#TARGET_CONFIGS[@]}" -eq 0 ]]; then
    log_info "\$(t no_cfg_files)"
else
    PENDING_CONFIGS=()
    PENDING_LABELS=()

    for i in "\${!TARGET_CONFIGS[@]}"; do
        CFG_PATH="\${TARGET_CONFIGS[\$i]}"
        CFG_LABEL="\${TARGET_LABELS[\$i]}"
        CHECK_RESULT=\$(
            python3 - "\${CFG_PATH}" "\${MCP_SERVER_DEST}" "\${MCP_VERSION}" <<'PYEOF'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1]).expanduser()
mcp_server_command = sys.argv[2]
mcp_version = sys.argv[3]

try:
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception:
    print("needs_update")
    sys.exit(0)

if not isinstance(data, dict):
    print("needs_update")
    sys.exit(0)

mcp_servers = data.get("mcpServers")
if mcp_servers is None:
    mcp_servers = {}
if not isinstance(mcp_servers, dict):
    print("needs_update")
    sys.exit(0)

desired = {
    "command": mcp_server_command,
    "args": [],
    "env": {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "TOKENIZERS_PARALLELISM": "false",
    },
    "version": mcp_version,
}

def is_rag_server_entry(value):
    if not isinstance(value, dict):
        return False
    cmd = value.get("command")
    return isinstance(cmd, str) and "mcp-rag-server" in cmd

if "rag-codebase" in mcp_servers and mcp_servers["rag-codebase"] == desired:
    print("already_up_to_date")
    sys.exit(0)

for key, value in mcp_servers.items():
    if key != "rag-codebase" and is_rag_server_entry(value):
        print("needs_update")
        sys.exit(0)

print("needs_update")
PYEOF
        )

        if [[ "\${CHECK_RESULT}" != "already_up_to_date" ]]; then
            PENDING_CONFIGS+=("\${CFG_PATH}")
            PENDING_LABELS+=("\${CFG_LABEL}")
        fi
    done

    if [[ "\${#PENDING_CONFIGS[@]}" -eq 0 ]]; then
        log_info "\$(t cfg_all_current)"
    else
        log_info "\$(t cfg_detected)"
        for i in "\${!PENDING_CONFIGS[@]}"; do
            echo -e "  - \${PENDING_LABELS[\$i]}: \${PENDING_CONFIGS[\$i]}"
        done

        APPLY_MCP_CONFIG=false
        if [[ -t 0 ]]; then
            if ask_yes_no_loop "\$(t ask_apply_cfg)"; then
                APPLY_MCP_CONFIG=true
            fi
        else
            log_info "\$(t noninteractive_cfg_skip)"
        fi

        if [[ "\$APPLY_MCP_CONFIG" == "true" ]]; then
            for i in "\${!PENDING_CONFIGS[@]}"; do
                CFG_PATH="\${PENDING_CONFIGS[\$i]}"
                CFG_LABEL="\${PENDING_LABELS[\$i]}"

                if ! RESULT=\$(
                    python3 - "\${CFG_PATH}" "\${MCP_SERVER_DEST}" "\${MCP_VERSION}" <<'PYEOF'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1]).expanduser()
mcp_server_command = sys.argv[2]
mcp_version = sys.argv[3]

try:
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"error:json_invalido:{exc}")
    sys.exit(2)

if not isinstance(data, dict):
    print("error:estrutura_invalida")
    sys.exit(2)

mcp_servers = data.get("mcpServers")
if mcp_servers is None:
    mcp_servers = {}
if not isinstance(mcp_servers, dict):
    print("error:mcpServers_invalido")
    sys.exit(2)

desired = {
    "command": mcp_server_command,
    "args": [],
    "env": {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "TOKENIZERS_PARALLELISM": "false",
    },
    "version": mcp_version,
}

def is_rag_server_entry(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    cmd = value.get("command")
    return isinstance(cmd, str) and "mcp-rag-server" in cmd

if "rag-codebase" in mcp_servers:
    if mcp_servers["rag-codebase"] == desired:
        print("ok:already_exists")
        sys.exit(0)
    mcp_servers["rag-codebase"] = desired
    data["mcpServers"] = mcp_servers
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ok:updated_rag_codebase")
    sys.exit(0)

old_rag_key = None
for key, value in mcp_servers.items():
    if is_rag_server_entry(value):
        old_rag_key = key
        break

if old_rag_key is not None:
    del mcp_servers[old_rag_key]
    mcp_servers["rag-codebase"] = desired
    data["mcpServers"] = mcp_servers
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ok:replaced_old:{old_rag_key}")
    sys.exit(0)

mcp_servers["rag-codebase"] = desired
data["mcpServers"] = mcp_servers
cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("ok:added")
PYEOF
                ); then
                    log_warn "\$(t cannot_update_cfg) \${CFG_LABEL} (\${CFG_PATH}): \${RESULT}"
                    continue
                fi

                case "\${RESULT}" in
                    ok:already_exists)
                        log_info "\${CFG_LABEL}: \$(t already_updated)"
                        ;;
                    ok:updated_rag_codebase)
                        log_info "\${CFG_LABEL}: \$(t updated_cfg) \${MCP_VERSION}."
                        ;;
                    ok:replaced_old:*)
                        log_info "\${CFG_LABEL}: \$(t replaced_cfg) \${MCP_VERSION})."
                        ;;
                    ok:added)
                        log_info "\${CFG_LABEL}: \$(t added_cfg) \${MCP_VERSION})."
                        ;;
                    *)
                        log_warn "\${CFG_LABEL}: \$(t cannot_update_cfg) -> \${RESULT}"
                        ;;
                esac
            done
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Indexação
# ---------------------------------------------------------------------------
if [[ "\$SKIP_INDEX" == "false" ]]; then
    if [[ ! -d "\${PROJECT_DIR}" ]]; then
        log_warn "\$(t path_not_found): \${PROJECT_DIR}"
    else
        prompt_optional_hf_token
        log_info "\$(t indexing): \${PROJECT_DIR}"
        run_indexer_with_diagnostics "\${PROJECT_DIR}"
        log_info "\$(t indexing_done)"
    fi
fi

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
echo ""
echo -e "\${BOLD}\${GREEN}================================================================\${NC}"
echo -e "\${BOLD}\${GREEN}  \$(t setup_done)\${NC}"
echo -e "\${BOLD}\${GREEN}================================================================\${NC}"
echo ""
echo -e "  \${GREEN}Venv Python\${NC} : \${VENV_DIR}"
echo -e "  \${GREEN}ChromaDB\${NC}    : http://localhost:8000 (Docker Desktop)"
echo -e "  \${GREEN}Dados\${NC}       : \${RAG_DB_DIR}"
echo -e "  \${GREEN}MCP Server\${NC}  : \${MCP_SERVER_DEST}"
echo -e "  \${GREEN}Projeto\${NC}     : \${PROJECT_DIR}"
echo ""
echo -e "  \${BOLD}\$(t next):\${NC}"
echo -e "  1. \$(t next_1)"
echo -e "  2. \$(t next_2)"
echo -e "  3. \$(t next_3)"
echo ""
OUTER_EOF

chmod +x "$OUTPUT"
SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "[+] Arquivo gerado: $OUTPUT ($SIZE)"
echo "[+] Pronto! Para usar no macOS:" 
echo "    cp ${OUTPUT} ~/seu-projeto/ && cd ~/seu-projeto && ./rag-setup-macos.run"
