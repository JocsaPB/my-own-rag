#!/usr/bin/env bash
# build_run.sh — Gera o arquivo rag-setup.run auto-suficiente.
# Execute este script uma vez para (re)construir o .run.
# O .run gerado pode ser copiado para qualquer projeto e executado lá.

set -euo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${BIN_DIR}")"
OUTPUT="${ROOT_DIR}/rag-setup.run"
README_FILE="${ROOT_DIR}/README.md"

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

MCP_SHA256=$(sha256sum "$MCP_FILE" | awk '{print $1}')
if [[ -f "$README_FILE" ]]; then
    TMP_README="$(mktemp)"
    {
        printf '# MCP binary checksum (SHA-256, payload without shebang): `%s` | Verify: `tail -n +2 ~/.local/bin/mcp-rag-server | sha256sum`\n' "$MCP_SHA256"
        tail -n +2 "$README_FILE"
    } > "$TMP_README"
    mv "$TMP_README" "$README_FILE"
fi

echo "[+] Gerando base64 dos arquivos..."

B64_COMPOSE=$(base64 -w0 "$COMPOSE_FILE")
B64_REQUIREMENTS=$(base64 -w0 "$REQUIREMENTS_FILE")
B64_INDEXER=$(base64 -w0 "$INDEXER_FILE")
B64_MCP=$(base64 -w0 "$MCP_FILE")
B64_MODEL_DL_HF=$(base64 -w0 "$MODEL_DL_HF_FILE")
B64_MODEL_DL_MS=$(base64 -w0 "$MODEL_DL_MS_FILE")

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
#   ./rag-setup.run --change-model # zera ChromaDB e reconfigura modelo/perfil
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
        --skip-index)  SKIP_INDEX=true ;;
        --only-index)  ONLY_INDEX=true ;;
        --reinstall)   REINSTALL=true ;;
        --change-model|-cm|--chage-model|-cg) CHANGE_MODEL=true ;;
        --help|-h)
            echo "Uso/Usage: \$0 [caminho/do/projeto|path/to/project] [--skip-index] [--only-index] [--reinstall] [--change-model|-cm]"
            exit 0 ;;
        -*)
            echo "Opção desconhecida / Unknown option: \$arg. Use --help."
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
            usage) echo "Usage: \$0 [path/to/project] [--skip-index] [--only-index] [--reinstall] [--change-model|-cm]" ;;
            unknown_option) echo "Unknown option: \$2. Use --help to see available options." ;;
            header_title) echo "RAG Local Setup - ChromaDB + MCP Server" ;;
            header_project) echo "Project to index" ;;
            section_extract) echo "Extracting embedded files" ;;
            extracted_to) echo "Files extracted to" ;;
            section_prereq) echo "Checking prerequisites" ;;
            py_missing) echo "Python 3 not found. Install: sudo apt install python3 python3-venv" ;;
            py_min) echo "Python 3.10+ required. Current:" ;;
            py_venv_missing) echo "python3-venv not found. Install: sudo apt install python3-venv" ;;
            py_ok) echo "Python" ;;
            docker_missing) echo "Docker not found: https://docs.docker.com/engine/install/" ;;
            docker_daemon) echo "Docker daemon is not running. Start it with: sudo systemctl start docker" ;;
            compose_missing) echo "Docker Compose not found: sudo apt install docker-compose-plugin" ;;
            docker_ok) echo "Docker + Compose OK." ;;
            curl_missing) echo "curl not found - ChromaDB healthcheck will be skipped." ;;
            hf_token_detected) echo "HuggingFace token detected in environment. Existing HF_TOKEN will be used." ;;
            non_interactive_no_hf) echo "No interactive terminal; continuing without HF_TOKEN." ;;
            hf_token_title) echo "HuggingFace token (optional)" ;;
            hf_token_desc) echo "Speeds up downloads and increases rate limits on HuggingFace Hub." ;;
            hf_prompt_now) echo "Do you want to provide an HF_TOKEN now? \$YES_NO_HINT " ;;
            hf_prompt_paste) echo "Paste your HF_TOKEN (hidden input): " ;;
            hf_set) echo "HF_TOKEN set for this execution." ;;
            hf_empty) echo "Empty token; continuing without Hugging Face authentication." ;;
            hf_continue_no) echo "Continuing without HF_TOKEN." ;;
            change_model_requested) echo "Model change requested (--change-model). ChromaDB reset + model reconfiguration required." ;;
            reset_start) echo "Resetting previous RAG environment" ;;
            reset_done) echo "Environment reset complete." ;;
            only_index_mode) echo "Mode: index only" ;;
            venv_not_found) echo "Venv not found at" ;;
            run_without_only_index) echo "Run without --only-index first to install." ;;
            path_not_found) echo "Path not found:" ;;
            indexing) echo "Indexing:" ;;
            indexing_done) echo "Indexing completed." ;;
            index_failed_code) echo "Indexing failed. Indexer exit code:" ;;
            index_oom_title) echo "Indexing stopped due to out-of-memory (OOM killer)." ;;
            index_oom_reason) echo "The kernel force-killed the Python process (exit 137 / SIGKILL)." ;;
            index_oom_hw_recommend) echo "For Jina in this workload, prefer RAM > 32 GB (practical baseline: 48 GB with dynamic-int8, 64 GB with default) and swap >= 16 GB." ;;
            index_oom_next_1) echo "Immediate fallback: MCP_EMBEDDING_MODEL=bge ./rag-setup.run --only-index" ;;
            index_oom_next_2) echo "If you need Jina, close heavy apps and increase swap before retrying." ;;
            section_venv) echo "Setting up Python virtual environment (~/.rag_venv)" ;;
            deps_ok) echo "Dependencies already installed in venv (transformers<5). Skipping. (use --reinstall to force)" ;;
            deps_incompatible) echo "Existing dependencies are incompatible or incomplete (transformers<5 is required for jinaai/jina-embeddings-v3). Reinstalling." ;;
            creating_venv) echo "Creating venv at" ;;
            venv_created) echo "Venv created." ;;
            upgrading_pip) echo "Upgrading pip..." ;;
            installing_packages) echo "Installing Python packages (this may take a few minutes on first run):" ;;
            model_download_note) echo "sentence-transformers will download the jinaai/jina-embeddings-v3 model (large)" ;;
            deps_installed) echo "All dependencies installed in the venv." ;;
            section_chroma) echo "Setting up ChromaDB (Docker)" ;;
            compose_installed) echo "docker-compose.yml installed at:" ;;
            compose_keep) echo "docker-compose.yml already exists. Keeping current file." ;;
            chroma_running) echo "Container chromadb-rag is already running." ;;
            chroma_start) echo "Starting ChromaDB (restart:always)..." ;;
            chroma_wait) echo "Waiting for ChromaDB to initialize..." ;;
            chroma_ready) echo "ChromaDB responding at http://localhost:8000" ;;
            chroma_timeout) echo "ChromaDB did not respond in 30s. Check: docker logs chromadb-rag" ;;
            section_install_mcp) echo "Installing mcp-rag-server globally" ;;
            mcp_keep) echo "mcp-rag-server already installed and up to date. Keeping it." ;;
            mcp_outdated) echo "mcp-rag-server already installed but outdated." ;;
            mcp_prompt_update) echo "Do you want to update mcp-rag-server now? \$YES_NO_HINT " ;;
            mcp_prompt_reinstall) echo "mcp-rag-server is already up to date. Reinstall/refresh now? \$YES_NO_HINT " ;;
            mcp_skip_update) echo "Keeping existing mcp-rag-server version." ;;
            mcp_version_detected) echo "Current MCP version detected:" ;;
            mcp_version_non_interactive) echo "No interactive terminal; using automatic hotfix version:" ;;
            mcp_version_prompt_current) echo "mcp-rag-server was updated. Choose MCP version strategy (current:" ;;
            mcp_version_option_hotfix) echo "Hotfix (next minor):" ;;
            mcp_version_option_major) echo "Major change: 2.0" ;;
            mcp_version_option_keep) echo "Keep current version" ;;
            mcp_version_prompt_select) echo "Select [1/2/3] (default 1): " ;;
            mcp_version_selected) echo "Selected MCP version:" ;;
            mcp_installed) echo "mcp-rag-server installed:" ;;
            mod_dl_installed) echo "Download module installed:" ;;
            mod_provider_installed) echo "Optional provider module installed:" ;;
            shebang) echo "Shebang:" ;;
            path_added) echo "Added ~/.local/bin to PATH in" ;;
            section_mcp_cfg) echo "Optional MCP setup (Claude/Cursor)" ;;
            no_default_cfg) echo "No default config file detected for automatic setup (Claude/Cursor)." ;;
            detected_cfg_files) echo "Detected files available for configuration:" ;;
            ask_apply_cfg) echo "Do you want to add/update MCP 'rag-codebase' in these files? \$YES_NO_HINT " ;;
            non_interactive_cfg_skip) echo "No interactive terminal; automatic MCP setup will not be applied." ;;
            mcp_cfg_all_current) echo "MCP 'rag-codebase' already up to date in detected config files. Skipping." ;;
            cannot_update_cfg) echo "Could not update" ;;
            already_updated) echo "MCP 'rag-codebase' already exists and is up to date. Skipping." ;;
            updated_version) echo "Existing MCP 'rag-codebase' was updated to version" ;;
            replaced_old) echo "Old RAG server key replaced by 'rag-codebase' (version" ;;
            added_cfg) echo "MCP 'rag-codebase' added (version" ;;
            unexpected_return) echo "Unexpected return while configuring MCP:" ;;
            user_skipped_cfg) echo "Automatic MCP setup skipped by user." ;;
            path_not_found_skip) echo "Path not found; skipping indexing:" ;;
            section_index_project) echo "Indexing project:" ;;
            section_skip_index) echo "Indexing skipped (--skip-index)" ;;
            how_to_index) echo "To index: ./rag-setup.run /path/to/project --only-index" ;;
            setup_done) echo "Setup complete!" ;;
            summary_next) echo "Next:" ;;
            next_1) echo "Restart Claude Code CLI" ;;
            next_2) echo "Use semantic_search_code" ;;
            next_3) echo "Reindex: ./rag-setup.run --only-index" ;;
            err_prefix) echo "ERROR" ;;
            invalid_option) echo "Invalid option. Type one of the allowed answers shown in the prompt. Press Ctrl+C to exit." ;;
            *) echo "\$key" ;;
        esac
    else
        case "\$key" in
            usage) echo "Uso: \$0 [caminho/do/projeto] [--skip-index] [--only-index] [--reinstall] [--change-model|-cm]" ;;
            unknown_option) echo "Opção desconhecida: \$2. Use --help para ver as opções." ;;
            header_title) echo "RAG Local Setup — ChromaDB + MCP Server" ;;
            header_project) echo "Projeto a indexar" ;;
            section_extract) echo "Extraindo arquivos embutidos" ;;
            extracted_to) echo "Arquivos extraídos para" ;;
            section_prereq) echo "Verificando pré-requisitos" ;;
            py_missing) echo "Python 3 não encontrado. Instale: sudo apt install python3 python3-venv" ;;
            py_min) echo "Python 3.10+ necessário. Atual:" ;;
            py_venv_missing) echo "python3-venv não encontrado. Instale: sudo apt install python3-venv" ;;
            py_ok) echo "Python" ;;
            docker_missing) echo "Docker não encontrado: https://docs.docker.com/engine/install/" ;;
            docker_daemon) echo "Docker daemon não está rodando. Inicie: sudo systemctl start docker" ;;
            compose_missing) echo "Docker Compose não encontrado: sudo apt install docker-compose-plugin" ;;
            docker_ok) echo "Docker + Compose OK." ;;
            curl_missing) echo "curl não encontrado — healthcheck do ChromaDB será pulado." ;;
            hf_token_detected) echo "Token do HuggingFace detectado no ambiente. O HF_TOKEN existente será usado." ;;
            non_interactive_no_hf) echo "Sem terminal interativo; seguindo sem HF_TOKEN." ;;
            hf_token_title) echo "Token HuggingFace (opcional)" ;;
            hf_token_desc) echo "Acelera download e aumenta limite de requisições no HuggingFace Hub." ;;
            hf_prompt_now) echo "Deseja informar um HF_TOKEN agora? \$YES_NO_HINT " ;;
            hf_prompt_paste) echo "Cole seu HF_TOKEN (entrada oculta): " ;;
            hf_set) echo "HF_TOKEN definido para esta execução." ;;
            hf_empty) echo "Token vazio; seguindo sem autenticação no Hugging Face." ;;
            hf_continue_no) echo "Seguindo sem HF_TOKEN." ;;
            change_model_requested) echo "Troca de modelo solicitada (--change-model). É necessário zerar o ChromaDB e reindexar." ;;
            reset_start) echo "Zerando ambiente RAG anterior" ;;
            reset_done) echo "Reset do ambiente concluído." ;;
            only_index_mode) echo "Modo: apenas indexação" ;;
            venv_not_found) echo "Venv não encontrado em" ;;
            run_without_only_index) echo "Execute sem --only-index primeiro para instalar." ;;
            path_not_found) echo "Caminho não encontrado:" ;;
            indexing) echo "Indexando:" ;;
            indexing_done) echo "Indexação concluída." ;;
            index_failed_code) echo "Indexação falhou. Código de saída do indexador:" ;;
            index_oom_title) echo "Indexação interrompida por falta de memória (OOM killer)." ;;
            index_oom_reason) echo "O kernel encerrou o processo Python à força (exit 137 / SIGKILL)." ;;
            index_oom_hw_recommend) echo "Para Jina nesta carga, prefira RAM > 32 GB (referência prática: 48 GB com dynamic-int8, 64 GB com default) e swap >= 16 GB." ;;
            index_oom_next_1) echo "Fallback imediato: MCP_EMBEDDING_MODEL=bge ./rag-setup.run --only-index" ;;
            index_oom_next_2) echo "Se precisar usar Jina, feche apps pesados e aumente swap antes de tentar novamente." ;;
            section_venv) echo "Configurando ambiente virtual Python (~/.rag_venv)" ;;
            deps_ok) echo "Dependências já instaladas no venv (transformers<5). Pulando. (use --reinstall para forçar)" ;;
            deps_incompatible) echo "Dependências existentes incompatíveis ou incompletas (é necessário transformers<5 para jinaai/jina-embeddings-v3). Reinstalando." ;;
            creating_venv) echo "Criando venv em" ;;
            venv_created) echo "Venv criado." ;;
            upgrading_pip) echo "Atualizando pip..." ;;
            installing_packages) echo "Instalando pacotes Python (isso pode levar alguns minutos na 1ª vez):" ;;
            model_download_note) echo "sentence-transformers baixará o modelo jinaai/jina-embeddings-v3 (grande)" ;;
            deps_installed) echo "Todas as dependências instaladas no venv." ;;
            section_chroma) echo "Configurando ChromaDB (Docker)" ;;
            compose_installed) echo "docker-compose.yml instalado em:" ;;
            compose_keep) echo "docker-compose.yml já existe. Mantendo." ;;
            chroma_running) echo "Container chromadb-rag já está rodando." ;;
            chroma_start) echo "Iniciando ChromaDB (restart:always)..." ;;
            chroma_wait) echo "Aguardando ChromaDB inicializar..." ;;
            chroma_ready) echo "ChromaDB respondendo em http://localhost:8000" ;;
            chroma_timeout) echo "ChromaDB não respondeu em 30s. Verifique: docker logs chromadb-rag" ;;
            section_install_mcp) echo "Instalando mcp-rag-server globalmente" ;;
            mcp_keep) echo "mcp-rag-server já instalado e atualizado. Mantendo." ;;
            mcp_outdated) echo "mcp-rag-server já instalado, mas desatualizado." ;;
            mcp_prompt_update) echo "Deseja atualizar o mcp-rag-server agora? \$YES_NO_HINT " ;;
            mcp_prompt_reinstall) echo "mcp-rag-server já está atualizado. Deseja reinstalar/atualizar agora? \$YES_NO_HINT " ;;
            mcp_skip_update) echo "Mantendo a versão atual do mcp-rag-server." ;;
            mcp_version_detected) echo "Versão MCP atual detectada:" ;;
            mcp_version_non_interactive) echo "Sem terminal interativo; usando versão hotfix automática:" ;;
            mcp_version_prompt_current) echo "mcp-rag-server foi atualizado. Escolha a estratégia de versão MCP (atual:" ;;
            mcp_version_option_hotfix) echo "Hotfix (próximo minor):" ;;
            mcp_version_option_major) echo "Mudança major: 2.0" ;;
            mcp_version_option_keep) echo "Manter versão atual" ;;
            mcp_version_prompt_select) echo "Selecione [1/2/3] (padrão 1): " ;;
            mcp_version_selected) echo "Versão MCP selecionada:" ;;
            mcp_installed) echo "mcp-rag-server instalado:" ;;
            mod_dl_installed) echo "Módulo de download instalado:" ;;
            mod_provider_installed) echo "Módulo de provider opcional instalado:" ;;
            shebang) echo "Shebang:" ;;
            path_added) echo "Adicionado ~/.local/bin ao PATH em" ;;
            section_mcp_cfg) echo "Configuração opcional de MCP (Claude/Cursor)" ;;
            no_default_cfg) echo "Nenhum arquivo default detectado para configuração automática (Claude/Cursor)." ;;
            detected_cfg_files) echo "Arquivos detectados para possível configuração:" ;;
            ask_apply_cfg) echo "Deseja adicionar/atualizar o MCP 'rag-codebase' nesses arquivos? \$YES_NO_HINT " ;;
            non_interactive_cfg_skip) echo "Sem terminal interativo; configuração automática de MCP não será aplicada." ;;
            mcp_cfg_all_current) echo "MCP 'rag-codebase' já está atualizado nos arquivos detectados. Pulando." ;;
            cannot_update_cfg) echo "Não foi possível atualizar" ;;
            already_updated) echo "MCP 'rag-codebase' já existe e está atualizado. Ignorando." ;;
            updated_version) echo "MCP 'rag-codebase' existente foi atualizado para a versão" ;;
            replaced_old) echo "Servidor RAG antigo substituído por 'rag-codebase' (versão" ;;
            added_cfg) echo "MCP 'rag-codebase' adicionado (versão" ;;
            unexpected_return) echo "Retorno inesperado ao configurar MCP:" ;;
            user_skipped_cfg) echo "Configuração automática de MCP ignorada pelo usuário." ;;
            path_not_found_skip) echo "Caminho não encontrado. Pulando indexação:" ;;
            section_index_project) echo "Indexando o projeto:" ;;
            section_skip_index) echo "Indexação pulada (--skip-index)" ;;
            how_to_index) echo "Para indexar: ./rag-setup.run /caminho/do/projeto --only-index" ;;
            setup_done) echo "Setup concluído!" ;;
            summary_next) echo "Próximos:" ;;
            next_1) echo "Reinicie o Claude Code CLI" ;;
            next_2) echo "Use semantic_search_code" ;;
            next_3) echo "Reindexar: ./rag-setup.run --only-index" ;;
            err_prefix) echo "ERRO" ;;
            invalid_option) echo "Opção inválida. Digite uma das respostas permitidas exibidas no prompt. Para sair, pressione Ctrl+C." ;;
            *) echo "\$key" ;;
        esac
    fi
}

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

detect_current_mcp_version() {
    python3 - "\${USER_HOME}" <<'PYEOF'
import json
import re
import sys
from pathlib import Path

user_home = Path(sys.argv[1]).expanduser()
paths = [
    user_home / ".claude.json",
    user_home / ".cursor" / "mcp.json",
    user_home / ".config" / "Cursor" / "User" / "mcp.json",
]
version_re = re.compile(r"^(\\d+)\\.(\\d+)$")
best = None

for path in paths:
    if not path.exists():
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(data, dict):
        continue
    mcp_servers = data.get("mcpServers")
    if not isinstance(mcp_servers, dict):
        continue
    rag_cfg = mcp_servers.get("rag-codebase")
    if not isinstance(rag_cfg, dict):
        continue
    version = rag_cfg.get("version")
    if not isinstance(version, str):
        continue
    match = version_re.match(version.strip())
    if not match:
        continue
    parsed = (int(match.group(1)), int(match.group(2)))
    if best is None or parsed > best:
        best = parsed

if best is None:
    print("1.0")
else:
    print(f"{best[0]}.{best[1]}")
PYEOF
}

log_info()    { echo -e "\${GREEN}[+]\${NC} \$*"; }
log_warn()    { echo -e "\${YELLOW}[!]\${NC} \$*"; }
log_error()   { echo -e "\${RED}[\$(t err_prefix)]\${NC} \$*" >&2; }
log_section() { echo -e "\n\${BLUE}\${BOLD}==> \$*\${NC}"; }

select_ui_language

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
MODEL_DL_HF_DEST="\${BIN_DIR}/download_model_from_hugginface.py"
MODEL_DL_MS_DEST="\${BIN_DIR}/download_model_from_modelscope.py"
MODEL_CACHE_DIR="\${USER_HOME}/.cache/my-custom-rag-python/models"
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
echo -e "\${BOLD}\${BLUE}  \$(t header_title)\${NC}"
echo -e "\${BOLD}\${BLUE}================================================================\${NC}"
echo -e "  \$(t header_project): \${BOLD}\${PROJECT_DIR}\${NC}"
echo ""

# ---------------------------------------------------------------------------
# Extrai arquivos embutidos
# ---------------------------------------------------------------------------
log_section "\$(t section_extract)"

echo "${B64_COMPOSE}"      | base64 -d > "\${EXTRACT_DIR}/docker-compose.yml"
echo "${B64_REQUIREMENTS}" | base64 -d > "\${EXTRACT_DIR}/requirements.txt"
echo "${B64_INDEXER}"      | base64 -d > "\${EXTRACT_DIR}/indexer_full.py"
echo "${B64_MCP}"          | base64 -d > "\${EXTRACT_DIR}/mcp_server.py"
echo "${B64_MODEL_DL_HF}"  | base64 -d > "\${EXTRACT_DIR}/download_model_from_hugginface.py"
echo "${B64_MODEL_DL_MS}"  | base64 -d > "\${EXTRACT_DIR}/download_model_from_modelscope.py"

log_info "\$(t extracted_to): \${EXTRACT_DIR}"

# ---------------------------------------------------------------------------
# Verifica pré-requisitos
# ---------------------------------------------------------------------------
log_section "\$(t section_prereq)"

if ! command -v python3 &>/dev/null; then
    log_error "\$(t py_missing)"
    exit 1
fi
PY_VER=\$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=\$(echo "\$PY_VER" | cut -d. -f1)
PY_MINOR=\$(echo "\$PY_VER" | cut -d. -f2)
if [[ "\$PY_MAJOR" -lt 3 ]] || [[ "\$PY_MAJOR" -eq 3 && "\$PY_MINOR" -lt 10 ]]; then
    log_error "\$(t py_min) \$PY_VER"
    exit 1
fi
if ! python3 -m venv --help &>/dev/null; then
    log_error "\$(t py_venv_missing)"
    exit 1
fi
log_info "\$(t py_ok) \$PY_VER OK."

if ! command -v docker &>/dev/null; then
    log_error "\$(t docker_missing)"
    exit 1
fi
if ! docker info &>/dev/null 2>&1; then
    log_error "\$(t docker_daemon)"
    exit 1
fi
if docker compose version &>/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    log_error "\$(t compose_missing)"
    exit 1
fi
log_info "\$(t docker_ok)"

if ! command -v curl &>/dev/null; then
    log_warn "\$(t curl_missing)"
    HAS_CURL=false
else
    HAS_CURL=true
fi

reset_rag_environment() {
    log_section "\$(t reset_start)"
    rm -rf "\${VENV_DIR}" "\${RAG_DB_DIR}" "\${DOCKER_COMPOSE_DIR}" "\${MODEL_CACHE_DIR}"
    rm -f "\${MCP_SERVER_DEST}" "\${MODEL_DL_HF_DEST}" "\${MODEL_DL_MS_DEST}"
    log_info "\$(t reset_done)"
}

if [[ "\$CHANGE_MODEL" == "true" ]]; then
    if [[ "\$ONLY_INDEX" == "true" ]]; then
        log_warn "Ignoring --only-index because --change-model requires full setup."
        ONLY_INDEX=false
    fi
    log_info "\$(t change_model_requested)"
    log_warn "ATENÇÃO/WARNING: trocar o modelo exige zerar o ChromaDB e reindexar todos os projetos."
    if [[ -t 0 ]]; then
        if ! ask_yes_no_loop "Confirmar reset do ChromaDB e nova indexação? \$YES_NO_HINT "; then
            log_info "Operação cancelada pelo usuário."
            exit 0
        fi
    else
        log_warn "Sem terminal interativo: seguindo com reset total por --change-model."
    fi
    export MCP_FORCE_MODEL_RECONFIG=1
    REINSTALL=true
    reset_rag_environment
fi

prompt_optional_hf_token() {
    # Evita perguntar duas vezes no mesmo run.
    if [[ -n "\${HF_TOKEN_PROMPTED:-}" ]]; then
        return
    fi
    HF_TOKEN_PROMPTED=1

    # Se já veio no ambiente, reaproveita sem perguntar.
    if [[ -n "\${HF_TOKEN:-}" || -n "\${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        log_info "\$(t hf_token_detected)"
        return
    fi

    # Só pergunta em terminal interativo.
    if [[ ! -t 0 ]]; then
        log_info "\$(t non_interactive_no_hf)"
        return
    fi

    echo ""
    echo -e "\${BOLD}\$(t hf_token_title)\${NC}"
    echo -e "\${DIM}\$(t hf_token_desc)\${NC}"
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
        log_info "\$(t hf_continue_no)"
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
        log_error "\$(t index_oom_reason)"
        log_error "\$(t index_oom_hw_recommend)"
        log_error "\$(t index_oom_next_1)"
        log_error "\$(t index_oom_next_2)"
    elif [[ "\$indexer_status" -ne 0 ]]; then
        log_error "\$(t index_failed_code) \${indexer_status}"
    fi

    return "\$indexer_status"
}

# ---------------------------------------------------------------------------
# Modo --only-index
# ---------------------------------------------------------------------------
if [[ "\$ONLY_INDEX" == "true" ]]; then
    log_section "\$(t only_index_mode)"
    if [[ ! -f "\${VENV_PYTHON}" ]]; then
        log_error "\$(t venv_not_found) \${VENV_DIR}. \$(t run_without_only_index)"
        exit 1
    fi
    if [[ ! -d "\${PROJECT_DIR}" ]]; then
        log_error "\$(t path_not_found) \${PROJECT_DIR}"
        exit 1
    fi
    prompt_optional_hf_token
    log_info "\$(t indexing) \${PROJECT_DIR}"
    run_indexer_with_diagnostics "\${PROJECT_DIR}"
    log_info "\$(t indexing_done)"
    exit 0
fi

# ---------------------------------------------------------------------------
# Cria/atualiza venv e instala dependências
# ---------------------------------------------------------------------------
log_section "\$(t section_venv)"

DEPS_OK=false
if [[ "\$REINSTALL" == "false" ]] && [[ -f "\${VENV_PYTHON}" ]]; then
    if "\${VENV_PYTHON}" -c "import chromadb, sentence_transformers, langchain_text_splitters, tqdm, mcp, transformers, sys; sys.exit(0 if int(transformers.__version__.split('.')[0]) < 5 else 1)" 2>/dev/null; then
        log_info "\$(t deps_ok)"
        DEPS_OK=true
    else
        log_warn "\$(t deps_incompatible)"
    fi
fi

if [[ "\$DEPS_OK" == "false" ]]; then
    if [[ ! -f "\${VENV_PYTHON}" ]]; then
        log_info "\$(t creating_venv) \${VENV_DIR}..."
        python3 -m venv "\${VENV_DIR}"
        log_info "\$(t venv_created)"
    fi
    log_info "\$(t upgrading_pip)"
    "\${VENV_PIP}" install --upgrade pip

    echo ""
    echo -e "\${YELLOW}  \$(t installing_packages)\${NC}"
    echo -e "\${DIM}  \$(t model_download_note)\${NC}"
    echo ""

    # Instala com output visível (sem --quiet) para o usuário acompanhar
    "\${VENV_PIP}" install \
        --progress-bar on \
        -r "\${EXTRACT_DIR}/requirements.txt"

    echo ""
    log_info "\$(t deps_installed)"
fi

# ---------------------------------------------------------------------------
# ChromaDB via Docker
# ---------------------------------------------------------------------------
log_section "\$(t section_chroma)"

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
        if [[ "\$HAS_CURL" == "true" ]] && curl -sf "http://localhost:8000/api/v1/heartbeat" &>/dev/null; then
            log_info "\$(t chroma_ready)"
            break
        fi
        if [[ \$WAITED -ge 30 ]]; then
            log_warn "\$(t chroma_timeout)"
            break
        fi
        sleep 2; WAITED=\$((WAITED+2))
    done
fi

# ---------------------------------------------------------------------------
# Instala mcp-rag-server com shebang do venv
# ---------------------------------------------------------------------------
log_section "\$(t section_install_mcp)"

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
        if [[ "\${MCP_WAS_OUTDATED}" == "true" ]]; then
            if ask_yes_no_loop "\$(t mcp_prompt_update)"; then
                NEEDS_INSTALL=true
            else
                NEEDS_INSTALL=false
                log_info "\$(t mcp_skip_update)"
            fi
        else
            if ask_yes_no_loop "\$(t mcp_prompt_reinstall)"; then
                NEEDS_INSTALL=true
            else
                NEEDS_INSTALL=false
                log_info "\$(t mcp_skip_update)"
            fi
        fi
    else
        if [[ "\${MCP_WAS_OUTDATED}" == "true" ]]; then
            NEEDS_INSTALL=false
            log_info "\$(t mcp_skip_update)"
        else
            NEEDS_INSTALL=false
        fi
    fi
fi

if [[ "\$NEEDS_INSTALL" == "true" ]]; then
    cp "\${EXTRACT_DIR}/mcp_server.py" "\${MCP_SERVER_DEST}"
    cp "\${EXTRACT_DIR}/download_model_from_hugginface.py" "\${MODEL_DL_HF_DEST}"
    cp "\${EXTRACT_DIR}/download_model_from_modelscope.py" "\${MODEL_DL_MS_DEST}"
    # Shebang aponta para o venv — garante que tem todas as dependências
    sed -i "1s|.*|#!\${VENV_PYTHON}|" "\${MCP_SERVER_DEST}"
    chmod +x "\${MCP_SERVER_DEST}"
    log_info "\$(t mcp_installed) \${MCP_SERVER_DEST}"
    log_info "\$(t mod_dl_installed) \${MODEL_DL_HF_DEST}"
    log_info "\$(t mod_provider_installed) \${MODEL_DL_MS_DEST}"
    log_info "\$(t shebang) \${VENV_PYTHON}"
fi

for RC in "\${USER_HOME}/.bashrc" "\${USER_HOME}/.zshrc"; do
    if [[ -f "\$RC" ]] && ! grep -qF '.local/bin' "\$RC"; then
        echo "" >> "\$RC"
        echo '# RAG setup — adicionado ao PATH' >> "\$RC"
        echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> "\$RC"
        log_info "\$(t path_added) \$RC"
    fi
done

# ---------------------------------------------------------------------------
# Configuração opcional do MCP no Claude/Cursor
# ---------------------------------------------------------------------------
log_section "\$(t section_mcp_cfg)"

CLAUDE_JSON="\${USER_HOME}/.claude.json"
CURSOR_MCP_JSON_1="\${USER_HOME}/.cursor/mcp.json"
CURSOR_MCP_JSON_2="\${USER_HOME}/.config/Cursor/User/mcp.json"
MCP_VERSION="\$(detect_current_mcp_version)"
log_info "\$(t mcp_version_detected) \${MCP_VERSION}"

CLAUDE_APP_INSTALLED=false
CURSOR_APP_INSTALLED=false
if command -v claude &>/dev/null || [[ -d "\${USER_HOME}/.claude" ]] || [[ -d "\${USER_HOME}/.config/Claude" ]]; then
    CLAUDE_APP_INSTALLED=true
fi
if command -v cursor &>/dev/null || [[ -d "\${USER_HOME}/.cursor" ]] || [[ -d "\${USER_HOME}/.config/Cursor" ]]; then
    CURSOR_APP_INSTALLED=true
fi

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

if [[ "\$CLAUDE_APP_INSTALLED" == "true" ]] && [[ -f "\${CLAUDE_JSON}" ]]; then
    _append_target "\${CLAUDE_JSON}" "Claude Code"
fi
if [[ "\$CURSOR_APP_INSTALLED" == "true" ]] && [[ -f "\${CURSOR_MCP_JSON_1}" ]]; then
    _append_target "\${CURSOR_MCP_JSON_1}" "Cursor"
fi
if [[ "\$CURSOR_APP_INSTALLED" == "true" ]] && [[ -f "\${CURSOR_MCP_JSON_2}" ]]; then
    _append_target "\${CURSOR_MCP_JSON_2}" "Cursor"
fi

if [[ "\${#TARGET_CONFIGS[@]}" -eq 0 ]]; then
    log_info "\$(t no_default_cfg)"
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

def is_rag_server_entry(value: object) -> bool:
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
        log_info "\$(t mcp_cfg_all_current)"
    else
    echo ""
    log_info "\$(t detected_cfg_files)"
    for i in "\${!PENDING_CONFIGS[@]}"; do
        echo -e "  - \${PENDING_LABELS[\$i]}: \${PENDING_CONFIGS[\$i]}"
    done

    APPLY_MCP_CONFIG=false
    if [[ -t 0 ]]; then
        echo ""
        if ask_yes_no_loop "\$(t ask_apply_cfg)"; then
            APPLY_MCP_CONFIG=true
        else
            APPLY_MCP_CONFIG=false
        fi
    else
        log_info "\$(t non_interactive_cfg_skip)"
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
                    log_info "\${CFG_LABEL}: \$(t updated_version) \${MCP_VERSION}."
                    ;;
                ok:replaced_old:*)
                    OLD_KEY="\${RESULT#ok:replaced_old:}"
                    log_info "\${CFG_LABEL}: \$(t replaced_old) \${MCP_VERSION})."
                    ;;
                ok:added)
                    log_info "\${CFG_LABEL}: \$(t added_cfg) \${MCP_VERSION})."
                    ;;
                *)
                    log_warn "\${CFG_LABEL}: \$(t unexpected_return) \${RESULT}"
                    ;;
            esac
        done
    else
        log_info "\$(t user_skipped_cfg)"
    fi
    fi
fi

# ---------------------------------------------------------------------------
# Indexa o projeto atual
# ---------------------------------------------------------------------------
if [[ "\$SKIP_INDEX" == "false" ]]; then
    if [[ ! -d "\${PROJECT_DIR}" ]]; then
        log_warn "\$(t path_not_found_skip) \${PROJECT_DIR}"
    else
        log_section "\$(t section_index_project) \${PROJECT_DIR}"
        prompt_optional_hf_token
        run_indexer_with_diagnostics "\${PROJECT_DIR}"
    fi
else
    log_section "\$(t section_skip_index)"
    log_info "\$(t how_to_index)"
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
echo -e "  \${GREEN}ChromaDB\${NC}    : http://localhost:8000 (Docker, auto-start)"
echo -e "  \${GREEN}Dados\${NC}       : \${RAG_DB_DIR}"
echo -e "  \${GREEN}MCP Server\${NC}  : \${MCP_SERVER_DEST}"
echo -e "  \${GREEN}Projeto\${NC}     : \${PROJECT_DIR}"
echo ""
echo -e "  \${BOLD}\$(t summary_next)\${NC}"
echo -e "  1. \$(t next_1)"
echo -e "  2. \$(t next_2)"
echo -e "  3. \$(t next_3)"
echo ""
OUTER_EOF

chmod +x "$OUTPUT"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "[+] Arquivo gerado: $OUTPUT ($SIZE)"
echo "[+] Pronto! Copie para qualquer projeto e execute:"
echo "      cp ${OUTPUT} ~/seu-projeto/ && cd ~/seu-projeto && ./rag-setup.run"
