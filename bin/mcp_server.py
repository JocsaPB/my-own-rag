#!/usr/bin/env python3
from __future__ import annotations
"""
mcp_server.py — Servidor MCP para RAG local de codebase.

Expõe ferramentas de busca semântica e indexação via stdio para o Claude Code CLI.
Conecta-se ao ChromaDB rodando em Docker (localhost:8000).

Transporte: stdio (Claude Code CLI conecta diretamente ao processo).
"""

import sys
import os
import hashlib
import json
import logging
import getpass
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Evita mensagens advisory do transformers em stderr durante a carga do modelo.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


class _TorchDtypeWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "`torch_dtype` is deprecated! Use `dtype` instead!" not in record.getMessage()


for _logger_name in ("transformers.configuration_utils", "transformers.modeling_utils"):
    logging.getLogger(_logger_name).addFilter(_TorchDtypeWarningFilter())

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP
from download_model_from_hugginface import download_model_with_fallback

# ---------------------------------------------------------------------------
# Configuração de logging (stderr para não poluir o protocolo stdio)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[MCP-RAG] %(asctime)s %(levelname)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log estruturado de uso MCP (JSONL)
# ---------------------------------------------------------------------------

MCP_USAGE_LOG_PATH = Path(
    os.environ.get("MCP_USAGE_LOG", str(Path.home() / ".rag_db" / "mcp_usage.log"))
).expanduser()


def _safe_preview(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _get_parent_cmdline() -> str:
    ppid = os.getppid()
    cmdline_path = Path(f"/proc/{ppid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
        if not raw:
            return "unknown"
        parts = [p.decode("utf-8", errors="ignore") for p in raw.split(b"\x00") if p]
        return " ".join(parts) if parts else "unknown"
    except Exception:
        return "unknown"


def _infer_actor() -> dict[str, str]:
    actor = os.environ.get("MCP_CLIENT_NAME") or os.environ.get("CLAUDE_USER") or getpass.getuser()
    source = (
        "MCP_CLIENT_NAME" if os.environ.get("MCP_CLIENT_NAME")
        else "CLAUDE_USER" if os.environ.get("CLAUDE_USER")
        else "system_user"
    )
    return {
        "actor": actor,
        "actor_source": source,
        "client_process": _get_parent_cmdline(),
    }


def _log_tool_usage(event: str, tool_name: str, details: dict[str, object] | None = None) -> None:
    try:
        MCP_USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "tool": tool_name,
            "pid": os.getpid(),
            **_infer_actor(),
        }
        if details:
            payload["details"] = details

        with MCP_USAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        log.warning(f"Falha ao registrar uso MCP em {MCP_USAGE_LOG_PATH}: {e}")

# ---------------------------------------------------------------------------
# Configurações — mantidas em sincronia com indexer_full.py
# ---------------------------------------------------------------------------

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_NAME = "codebase"
JINA_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL_CHOICE = "jina"
DEFAULT_JINA_QUANTIZATION = "default"
_embedding_model_choice = os.environ.get("MCP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL_CHOICE).strip().lower()
if _embedding_model_choice not in {"jina", "bge"}:
    log.warning(
        "MCP_EMBEDDING_MODEL invalido '%s'. Usando '%s'.",
        _embedding_model_choice,
        DEFAULT_EMBEDDING_MODEL_CHOICE,
    )
    _embedding_model_choice = DEFAULT_EMBEDDING_MODEL_CHOICE

EMBEDDING_MODEL = JINA_EMBEDDING_MODEL if _embedding_model_choice == "jina" else BGE_EMBEDDING_MODEL
FALLBACK_EMBEDDING_MODEL = BGE_EMBEDDING_MODEL if _embedding_model_choice == "jina" else BGE_EMBEDDING_MODEL
_raw_jina_quantization = os.environ.get("MCP_JINA_QUANTIZATION", DEFAULT_JINA_QUANTIZATION)
JINA_QUANTIZATION = _raw_jina_quantization.strip().lower().replace("_", "-")
if JINA_QUANTIZATION not in {"default", "dynamic-int8"}:
    log.warning(
        "MCP_JINA_QUANTIZATION invalido '%s'. Usando '%s'.",
        JINA_QUANTIZATION,
        DEFAULT_JINA_QUANTIZATION,
    )
    JINA_QUANTIZATION = DEFAULT_JINA_QUANTIZATION
_env_model_dir = os.environ.get("MCP_MODEL_DIR")
MODEL_DIR = (
    Path(_env_model_dir).expanduser()
    if _env_model_dir
    else Path.home() / ".cache" / "my-custom-rag-python" / "models"
)

# Parâmetros do splitter (idênticos ao indexer_full.py)
# Jina Embeddings V3 suporta janelas longas — chunks maiores preservam funções inteiras
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 800

# Pastas e extensões ignoradas durante indexação de pasta/arquivo
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "out", ".next", ".nuxt", ".cache", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", "bin", "obj",
    ".idea", ".vscode", "vendor", "tmp", "temp", "logs", ".rag_db",
}

IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp4", ".mp3", ".wav", ".ogg", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".rar", ".7z", ".jar", ".war",
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".lock", ".sum", ".sqlite", ".db", ".sqlite3",
    ".ttf", ".woff", ".woff2", ".eot",
    ".pdf", ".docx", ".xlsx", ".pptx",
}

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB
TOP_K_RESULTS = 7  # Quantidade padrão de resultados na busca semântica


# ---------------------------------------------------------------------------
# Inicialização dos componentes (lazy — acontece uma vez ao iniciar o servidor)
# ---------------------------------------------------------------------------

_chroma_client: chromadb.HttpClient | None = None
_collection: chromadb.Collection | None = None
_model: SentenceTransformer | None = None
_splitter: RecursiveCharacterTextSplitter | None = None
_loaded_model_id: str | None = None


def get_chroma_collection() -> chromadb.Collection:
    """Retorna a coleção ChromaDB, conectando se necessário."""
    global _chroma_client, _collection
    if _collection is None:
        try:
            _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            _chroma_client.heartbeat()
            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"Conectado ao ChromaDB em {CHROMA_HOST}:{CHROMA_PORT}")
        except Exception as e:
            raise RuntimeError(
                f"Não foi possível conectar ao ChromaDB em {CHROMA_HOST}:{CHROMA_PORT}. "
                f"Verifique se o Docker está rodando. Erro: {e}"
            )
    return _collection


def get_model() -> SentenceTransformer:
    """Retorna o modelo de embeddings, carregando se necessário."""
    global _model, _loaded_model_id
    if _model is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL} (CPU)...")
        log.info(f"Diretório local de modelo: {MODEL_DIR}")

        selection = download_model_with_fallback(
            preferred_model_id=EMBEDDING_MODEL,
            fallback_model_id=FALLBACK_EMBEDDING_MODEL,
            local_dir=MODEL_DIR,
        )
        selected_model_dir = selection.local_dir
        _loaded_model_id = selection.model_id
        log.info(
            "Modelo pronto: %s (provider=%s, path=%s)",
            selection.model_id,
            selection.provider,
            selected_model_dir,
        )

        def _clear_hf_dynamic_modules_cache() -> None:
            cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
            if cache_dir.exists():
                log.warning("Limpando cache dinâmico do Hugging Face em %s", cache_dir)
                shutil.rmtree(cache_dir, ignore_errors=True)

        def _load_from_local_dir(model_id: str, local_model_dir: Path) -> SentenceTransformer:
            trust_remote_code = model_id.startswith("jinaai/")
            tokenizer_kwargs = {"fix_mistral_regex": True}

            def _instantiate_model() -> SentenceTransformer:
                return SentenceTransformer(
                    str(local_model_dir),
                    device="cpu",
                    trust_remote_code=trust_remote_code,
                    tokenizer_kwargs=tokenizer_kwargs,
                )

            def _load_with_mistral_regex_patch() -> SentenceTransformer:
                # O código remoto da Jina instancia um tokenizer interno sem repassar tokenizer_kwargs.
                if not trust_remote_code:
                    return _instantiate_model()

                from transformers import AutoModel, AutoTokenizer
                from transformers.modeling_utils import PreTrainedModel

                original_from_pretrained = AutoTokenizer.from_pretrained
                original_model_from_pretrained = AutoModel.from_pretrained
                original_pretrained_model_from_pretrained = PreTrainedModel.from_pretrained
                original_pretrained_model_from_config = PreTrainedModel._from_config
                model_refs = {str(local_model_dir), str(local_model_dir.resolve())}

                def _patched_from_pretrained(*args, **kwargs):
                    model_ref = args[0] if args else kwargs.get("pretrained_model_name_or_path")
                    if model_ref is not None and str(model_ref) in model_refs:
                        kwargs.setdefault("fix_mistral_regex", True)
                    return original_from_pretrained(*args, **kwargs)

                def _patched_model_from_pretrained(*args, **kwargs):
                    model_ref = args[0] if args else kwargs.get("pretrained_model_name_or_path")
                    if model_ref is not None and str(model_ref) in model_refs and "torch_dtype" in kwargs:
                        kwargs = dict(kwargs)
                        if "dtype" not in kwargs:
                            kwargs["dtype"] = kwargs["torch_dtype"]
                        kwargs.pop("torch_dtype", None)
                    return original_model_from_pretrained(*args, **kwargs)

                original_pretrained_model_from_pretrained_fn = original_pretrained_model_from_pretrained.__func__

                @classmethod
                def _patched_pretrained_model_from_pretrained(cls, *args, **kwargs):
                    if "torch_dtype" in kwargs:
                        kwargs = dict(kwargs)
                        if "dtype" not in kwargs:
                            kwargs["dtype"] = kwargs["torch_dtype"]
                        kwargs.pop("torch_dtype", None)
                    return original_pretrained_model_from_pretrained_fn(cls, *args, **kwargs)

                original_pretrained_model_from_config_fn = original_pretrained_model_from_config.__func__

                @classmethod
                def _patched_pretrained_model_from_config(cls, *args, **kwargs):
                    if "torch_dtype" in kwargs:
                        kwargs = dict(kwargs)
                        if "dtype" not in kwargs:
                            kwargs["dtype"] = kwargs["torch_dtype"]
                        kwargs.pop("torch_dtype", None)
                    return original_pretrained_model_from_config_fn(cls, *args, **kwargs)

                AutoTokenizer.from_pretrained = _patched_from_pretrained
                AutoModel.from_pretrained = _patched_model_from_pretrained
                PreTrainedModel.from_pretrained = _patched_pretrained_model_from_pretrained
                PreTrainedModel._from_config = _patched_pretrained_model_from_config
                try:
                    return _instantiate_model()
                finally:
                    AutoTokenizer.from_pretrained = original_from_pretrained
                    AutoModel.from_pretrained = original_model_from_pretrained
                    PreTrainedModel.from_pretrained = original_pretrained_model_from_pretrained
                    PreTrainedModel._from_config = original_pretrained_model_from_config

            try:
                return _load_with_mistral_regex_patch()
            except FileNotFoundError as e:
                if trust_remote_code and "transformers_modules" in str(e):
                    log.warning("Cache dinâmico inconsistente detectado: %s", e)
                    _clear_hf_dynamic_modules_cache()
                    return _load_with_mistral_regex_patch()
                raise

        def _apply_jina_quantization_if_needed(model: SentenceTransformer, model_id: str) -> SentenceTransformer:
            if model_id != JINA_EMBEDDING_MODEL or JINA_QUANTIZATION == "default":
                return model
            try:
                import torch

                for module in model.modules():
                    if hasattr(module, "auto_model"):
                        module.auto_model = torch.quantization.quantize_dynamic(
                            module.auto_model,
                            {torch.nn.Linear},
                            dtype=torch.qint8,
                        )
                        log.info("Quantizacao Jina aplicada: dynamic-int8 (CPU).")
                        return model
                log.warning(
                    "Nao foi possivel localizar auto_model para aplicar dynamic-int8; usando modelo padrao."
                )
                return model
            except Exception as quant_error:
                log.warning("Falha ao aplicar dynamic-int8 (%s); usando modelo padrao.", quant_error)
                return model

        try:
            _model = _load_from_local_dir(selection.model_id, selected_model_dir)
            _model = _apply_jina_quantization_if_needed(_model, selection.model_id)
            log.info("Modelo carregado a partir da pasta local sincronizada.")
        except Exception as local_error:
            if selection.model_id == FALLBACK_EMBEDDING_MODEL:
                raise RuntimeError(
                    f"Falha ao carregar modelo fallback '{FALLBACK_EMBEDDING_MODEL}' em '{selected_model_dir}'. "
                    f"Erro: {local_error}"
                ) from local_error

            log.warning(
                "Falha ao carregar '%s' (%s). Tentando fallback '%s'.",
                selection.model_id,
                local_error,
                FALLBACK_EMBEDDING_MODEL,
            )
            fallback_selection = download_model_with_fallback(
                preferred_model_id=FALLBACK_EMBEDDING_MODEL,
                fallback_model_id=FALLBACK_EMBEDDING_MODEL,
                local_dir=MODEL_DIR,
            )
            selected_model_dir = fallback_selection.local_dir
            _loaded_model_id = fallback_selection.model_id
            _model = _load_from_local_dir(fallback_selection.model_id, selected_model_dir)
            _model = _apply_jina_quantization_if_needed(_model, fallback_selection.model_id)
            log.info(
                "Modelo fallback carregado com sucesso: %s (provider=%s, path=%s)",
                fallback_selection.model_id,
                fallback_selection.provider,
                selected_model_dir,
            )
    return _model


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Retorna o text splitter compartilhado."""
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    return _splitter


# ---------------------------------------------------------------------------
# Funções internas reutilizáveis
# ---------------------------------------------------------------------------

def _make_chunk_id(file_path: str, chunk_index: int) -> str:
    raw = f"{file_path}::chunk::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _delete_file_chunks(collection: chromadb.Collection, file_path: str) -> int:
    """Remove todos os chunks de um arquivo. Retorna quantos foram deletados."""
    results = collection.get(where={"file_path": file_path})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _read_file_safe(filepath: Path) -> str | None:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return filepath.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _index_single_file(
    filepath: Path,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    splitter: RecursiveCharacterTextSplitter,
) -> int:
    """
    Indexa um único arquivo: divide em chunks, gera embeddings e faz upsert.
    Retorna o número de chunks inseridos.
    """
    content = _read_file_safe(filepath)
    if not content or not content.strip():
        return 0

    abs_path = str(filepath.resolve())
    chunks = splitter.split_text(content)
    if not chunks:
        return 0

    ids = [_make_chunk_id(abs_path, i) for i in range(len(chunks))]
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    metadatas = [
        {
            "file_path": abs_path,
            "chunk_index": i,
            "file_name": filepath.name,
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def _scan_folder(folder_path: Path) -> list[Path]:
    """Varre uma pasta retornando arquivos indexáveis."""
    files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            fp = Path(dirpath) / filename
            if fp.suffix.lower() in IGNORED_EXTENSIONS:
                continue
            try:
                if fp.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue
            files.append(fp)
    return sorted(files)


# ---------------------------------------------------------------------------
# Servidor MCP via FastMCP
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="rag-codebase",
    instructions=(
        "Servidor RAG para busca semântica em código-fonte local. "
        "Use semantic_search_code para encontrar trechos relevantes de código. "
        "Use update_file_index após editar um arquivo para manter o índice atualizado. "
        "Use index_specific_folder para indexar novos diretórios sob demanda."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: semantic_search_code
# ---------------------------------------------------------------------------

@mcp.tool()
def semantic_search_code(query: str, top_k: int = TOP_K_RESULTS) -> str:
    """
    Busca semântica na codebase indexada.

    Converte a query em embedding e retorna os chunks mais similares
    do banco de dados vetorial, com caminho do arquivo e trecho do código.

    Args:
        query: Descrição em linguagem natural do que você está procurando.
               Ex: "função que valida CPF", "conexão com banco de dados".
        top_k: Número de resultados a retornar (padrão: 7, máximo recomendado: 15).

    Returns:
        Lista formatada dos chunks mais relevantes com file_path e snippet.
    """
    raw_query = (query or "").strip()
    _log_tool_usage(
        event="tool_call_start",
        tool_name="semantic_search_code",
        details={"top_k": top_k, "query_preview": _safe_preview(raw_query), "query_len": len(raw_query)},
    )

    if not raw_query:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "error", "reason": "empty_query"},
        )
        return "Erro: a query não pode ser vazia."

    top_k = max(1, min(top_k, 20))  # Clamp entre 1 e 20

    try:
        collection = get_chroma_collection()
        model = get_model()
    except RuntimeError as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "error", "reason": "connection_error", "error": str(e)},
        )
        return f"Erro de conexão: {e}"

    try:
        query_embedding = model.encode([raw_query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "error", "reason": "query_failed", "error": str(e)},
        )
        return f"Erro ao executar busca no ChromaDB: {e}"

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "ok", "result_count": 0},
        )
        return "Nenhum resultado encontrado. O índice pode estar vazio — rode o indexer_full.py primeiro."

    output_parts = [f"# Resultados para: '{query}'\n"]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        file_path = meta.get("file_path", "desconhecido")
        similarity = round((1 - dist) * 100, 1)  # Converte distância cosseno em %

        # Trunca snippets muito longos para não estourar o contexto
        snippet = doc.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "\n... [truncado]"

        output_parts.append(
            f"## [{i}] {file_path}\n"
            f"**Similaridade:** {similarity}%\n\n"
            f"```\n{snippet}\n```\n"
        )

    _log_tool_usage(
        event="tool_call_end",
        tool_name="semantic_search_code",
        details={"status": "ok", "result_count": len(documents), "top_k": top_k},
    )
    return "\n".join(output_parts)


# ---------------------------------------------------------------------------
# Tool 2: update_file_index
# ---------------------------------------------------------------------------

@mcp.tool()
def update_file_index(file_path: str) -> str:
    """
    Atualiza o índice RAG para um arquivo específico.

    Remove os chunks antigos do arquivo e reindexa com o conteúdo atual do disco.
    Use após editar ou criar um arquivo para manter o RAG sincronizado.

    Args:
        file_path: Caminho absoluto ou relativo ao arquivo a reindexar.
                   Ex: "/home/<usuario>/meu-projeto/src/auth.py"

    Returns:
        Mensagem de confirmação com o número de chunks gerados.
    """
    filepath = Path(file_path).resolve()
    _log_tool_usage(
        event="tool_call_start",
        tool_name="update_file_index",
        details={"file_path": str(filepath)},
    )

    if not filepath.exists():
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "file_not_found", "file_path": str(filepath)},
        )
        return f"Erro: arquivo não encontrado: {filepath}"

    if not filepath.is_file():
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "not_a_file", "file_path": str(filepath)},
        )
        return f"Erro: o caminho não aponta para um arquivo: {filepath}"

    if filepath.stat().st_size > MAX_FILE_SIZE_BYTES:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "file_too_large", "file_path": str(filepath)},
        )
        return f"Erro: arquivo muito grande (>{MAX_FILE_SIZE_BYTES // 1024}KB): {filepath}"

    try:
        collection = get_chroma_collection()
        model = get_model()
        splitter = get_splitter()
    except RuntimeError as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "connection_error", "error": str(e)},
        )
        return f"Erro de conexão: {e}"

    try:
        # Remove versão antiga
        deleted = _delete_file_chunks(collection, str(filepath))

        # Reindexar
        n_chunks = _index_single_file(filepath, collection, model, splitter)

        if n_chunks == 0:
            _log_tool_usage(
                event="tool_call_end",
                tool_name="update_file_index",
                details={"status": "ok", "file_path": str(filepath), "deleted": deleted, "inserted": 0},
            )
            return f"Arquivo processado, mas nenhum chunk gerado (arquivo vazio ou binário): {filepath}"

        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "ok", "file_path": str(filepath), "deleted": deleted, "inserted": n_chunks},
        )
        return (
            f"Arquivo reindexado com sucesso.\n"
            f"  Arquivo  : {filepath}\n"
            f"  Chunks antigos removidos: {deleted}\n"
            f"  Novos chunks inseridos  : {n_chunks}"
        )
    except Exception as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "reindex_failed", "file_path": str(filepath), "error": str(e)},
        )
        return f"Erro ao reindexar {filepath}: {e}"


# ---------------------------------------------------------------------------
# Tool 3: delete_file_index
# ---------------------------------------------------------------------------

@mcp.tool()
def delete_file_index(file_path: str) -> str:
    """
    Remove completamente um arquivo do índice RAG.

    Útil quando um arquivo foi deletado do projeto ou não deve mais
    aparecer nos resultados de busca.

    Args:
        file_path: Caminho absoluto ou relativo ao arquivo a remover do índice.

    Returns:
        Mensagem confirmando quantos chunks foram removidos.
    """
    filepath = Path(file_path).resolve()
    abs_path = str(filepath)
    _log_tool_usage(
        event="tool_call_start",
        tool_name="delete_file_index",
        details={"file_path": abs_path},
    )

    try:
        collection = get_chroma_collection()
    except RuntimeError as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="delete_file_index",
            details={"status": "error", "reason": "connection_error", "error": str(e)},
        )
        return f"Erro de conexão: {e}"

    try:
        deleted = _delete_file_chunks(collection, abs_path)

        if deleted == 0:
            _log_tool_usage(
                event="tool_call_end",
                tool_name="delete_file_index",
                details={"status": "ok", "file_path": abs_path, "deleted": 0},
            )
            return f"Nenhum chunk encontrado para o arquivo: {abs_path}\n(O arquivo pode não estar indexado.)"

        _log_tool_usage(
            event="tool_call_end",
            tool_name="delete_file_index",
            details={"status": "ok", "file_path": abs_path, "deleted": deleted},
        )
        return f"Removido do índice com sucesso.\n  Arquivo : {abs_path}\n  Chunks deletados: {deleted}"
    except Exception as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="delete_file_index",
            details={"status": "error", "reason": "delete_failed", "file_path": abs_path, "error": str(e)},
        )
        return f"Erro ao deletar chunks de {abs_path}: {e}"


# ---------------------------------------------------------------------------
# Tool 4: index_specific_folder
# ---------------------------------------------------------------------------

@mcp.tool()
def index_specific_folder(folder_path: str) -> str:
    """
    Indexa (ou reindexar) todos os arquivos de um subdiretório específico.

    Varre a pasta recursivamente, ignora arquivos binários e pastas de build,
    e faz upsert de todos os chunks no ChromaDB.

    Args:
        folder_path: Caminho absoluto ou relativo à pasta a indexar.
                     Ex: "/home/<usuario>/meu-projeto/src/auth/"

    Returns:
        Relatório com o número de arquivos e chunks processados.
    """
    folder = Path(folder_path).resolve()
    _log_tool_usage(
        event="tool_call_start",
        tool_name="index_specific_folder",
        details={"folder_path": str(folder)},
    )

    if not folder.exists():
        _log_tool_usage(
            event="tool_call_end",
            tool_name="index_specific_folder",
            details={"status": "error", "reason": "folder_not_found", "folder_path": str(folder)},
        )
        return f"Erro: pasta não encontrada: {folder}"

    if not folder.is_dir():
        _log_tool_usage(
            event="tool_call_end",
            tool_name="index_specific_folder",
            details={"status": "error", "reason": "not_a_folder", "folder_path": str(folder)},
        )
        return f"Erro: o caminho não é um diretório: {folder}"

    try:
        collection = get_chroma_collection()
        model = get_model()
        splitter = get_splitter()
    except RuntimeError as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="index_specific_folder",
            details={"status": "error", "reason": "connection_error", "error": str(e)},
        )
        return f"Erro de conexão: {e}"

    files = _scan_folder(folder)

    if not files:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="index_specific_folder",
            details={"status": "ok", "folder_path": str(folder), "files_processed": 0, "chunks": 0},
        )
        return f"Nenhum arquivo indexável encontrado em: {folder}"

    total_chunks = 0
    processed = 0
    errors = []

    for filepath in files:
        try:
            # Remove versão antiga do arquivo antes de reinserir
            _delete_file_chunks(collection, str(filepath.resolve()))
            n = _index_single_file(filepath, collection, model, splitter)
            total_chunks += n
            processed += 1
        except Exception as e:
            errors.append(f"{filepath.name}: {e}")

    report = (
        f"Indexação da pasta concluída.\n"
        f"  Pasta    : {folder}\n"
        f"  Arquivos processados: {processed}/{len(files)}\n"
        f"  Total de chunks     : {total_chunks}\n"
    )

    if errors:
        report += f"  Erros ({len(errors)}):\n"
        for err in errors[:10]:  # Mostra no máximo 10 erros
            report += f"    - {err}\n"
        if len(errors) > 10:
            report += f"    ... e mais {len(errors) - 10} erros.\n"

    _log_tool_usage(
        event="tool_call_end",
        tool_name="index_specific_folder",
        details={
            "status": "ok",
            "folder_path": str(folder),
            "files_processed": processed,
            "files_total": len(files),
            "chunks": total_chunks,
            "errors": len(errors),
        },
    )
    return report


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Iniciando servidor MCP RAG (stdio)...")
    log.info(f"ChromaDB: {CHROMA_HOST}:{CHROMA_PORT} | Coleção: {COLLECTION_NAME}")
    log.info(
        "Modelo preferido: %s (choice=%s, CPU)",
        EMBEDDING_MODEL,
        _embedding_model_choice,
    )
    log.info(f"Modelo fallback: {FALLBACK_EMBEDDING_MODEL}")
    log.info(f"Quantizacao Jina: {JINA_QUANTIZATION}")
    log.info(f"Pasta de modelo local: {MODEL_DIR}")
    log.info(f"Uso MCP será registrado em: {MCP_USAGE_LOG_PATH}")

    # Pré-aquece os componentes para evitar latência na primeira chamada
    try:
        get_model()
        get_chroma_collection()
        log.info("Componentes inicializados. Servidor pronto.")
    except Exception as e:
        log.error(f"Falha na inicialização: {e}")
        log.error("O servidor continuará, mas as ferramentas retornarão erro até o ChromaDB estar disponível.")

    mcp.run(transport="stdio")
