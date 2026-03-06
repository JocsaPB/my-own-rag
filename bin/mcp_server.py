#!/usr/bin/env python3
from __future__ import annotations
"""
mcp_server.py — Servidor MCP para RAG local de codebase.

Expõe ferramentas de busca semântica e indexação via stdio para o Claude Code CLI.
Conecta-se ao ChromaDB rodando em Docker (localhost:8000).

Novidade: modo híbrido ensemble com duas coleções separadas + RRF + reranking leve.
"""

import sys
import os
import hashlib
import json
import logging
import getpass
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
from sentence_transformers import CrossEncoder, SentenceTransformer
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
        log.warning("Falha ao registrar uso MCP em %s: %s", MCP_USAGE_LOG_PATH, e)


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------


INDEXER_CONFIG_PATH = Path(
    os.environ.get("MCP_INDEXER_CONFIG_FILE", str(Path.home() / ".rag_db" / "indexer_tuning.json"))
).expanduser()


def _load_indexer_tuning_config() -> dict[str, object]:
    try:
        if not INDEXER_CONFIG_PATH.exists():
            return {}
        payload = json.loads(INDEXER_CONFIG_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


INDEXER_TUNING_CONFIG = _load_indexer_tuning_config()


def _config_str(env_name: str, config_key: str, default: str) -> str:
    env_raw = os.environ.get(env_name)
    if env_raw is not None and env_raw.strip():
        return env_raw
    cfg_raw = INDEXER_TUNING_CONFIG.get(config_key)
    if isinstance(cfg_raw, str) and cfg_raw.strip():
        return cfg_raw
    return default


def _config_int(env_name: str, config_key: str, default: int, *, min_value: int = 1) -> int:
    env_raw = os.environ.get(env_name)
    if env_raw is not None and env_raw.strip():
        try:
            return max(min_value, int(env_raw))
        except ValueError:
            pass

    cfg_raw = INDEXER_TUNING_CONFIG.get(config_key)
    if isinstance(cfg_raw, int):
        return max(min_value, cfg_raw)
    if isinstance(cfg_raw, str):
        try:
            return max(min_value, int(cfg_raw))
        except ValueError:
            pass

    return max(min_value, default)


CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

# Coleções separadas por especialização de embedding
COLLECTION_CODE_JINA = "code_vectors_jina"
COLLECTION_DOC_BGE = "doc_vectors_bge"

JINA_V3_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
JINA_V2_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_EMBEDDING_MODEL_CHOICE = "jina"
DEFAULT_JINA_QUANTIZATION = "dynamic-int8"
DEFAULT_SEARCH_MODE = "single"  # single | ensemble

_embedding_model_choice = _config_str(
    "MCP_EMBEDDING_MODEL",
    "embedding_model",
    DEFAULT_EMBEDDING_MODEL_CHOICE,
).strip().lower()
if _embedding_model_choice not in {"jina", "bge", "hybrid"}:
    log.warning(
        "MCP_EMBEDDING_MODEL invalido '%s'. Usando '%s'.",
        _embedding_model_choice,
        DEFAULT_EMBEDDING_MODEL_CHOICE,
    )
    _embedding_model_choice = DEFAULT_EMBEDDING_MODEL_CHOICE

_raw_jina_quantization = _config_str(
    "MCP_JINA_QUANTIZATION",
    "jina_quantization",
    DEFAULT_JINA_QUANTIZATION,
)
JINA_QUANTIZATION = _raw_jina_quantization.strip().lower().replace("_", "-")
if JINA_QUANTIZATION not in {"default", "dynamic-int8"}:
    log.warning(
        "MCP_JINA_QUANTIZATION invalido '%s'. Usando '%s'.",
        JINA_QUANTIZATION,
        DEFAULT_JINA_QUANTIZATION,
    )
    JINA_QUANTIZATION = DEFAULT_JINA_QUANTIZATION

SEARCH_MODE_DEFAULT = os.environ.get("MCP_SEARCH_MODE", DEFAULT_SEARCH_MODE).strip().lower()
if SEARCH_MODE_DEFAULT not in {"single", "ensemble"}:
    SEARCH_MODE_DEFAULT = DEFAULT_SEARCH_MODE

if _embedding_model_choice == "hybrid" and "MCP_SEARCH_MODE" not in os.environ:
    # No modo híbrido, o comportamento esperado costuma ser ensemble por padrão.
    SEARCH_MODE_DEFAULT = "ensemble"

RERANK_MODEL_ID = os.environ.get("MCP_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.environ.get("MCP_RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RERANK_CANDIDATE_MULTIPLIER = int(os.environ.get("MCP_RERANK_CANDIDATE_MULTIPLIER", "3"))
RERANK_MAX_CANDIDATES = int(os.environ.get("MCP_RERANK_MAX_CANDIDATES", "40"))
RERANKER_MAX_LENGTH = int(os.environ.get("MCP_RERANK_MAX_LENGTH", "512"))
RERANKER_QUANTIZATION = os.environ.get("MCP_RERANK_QUANTIZATION", "dynamic-int8").strip().lower()
if RERANKER_QUANTIZATION not in {"default", "dynamic-int8"}:
    RERANKER_QUANTIZATION = "dynamic-int8"

RRF_K = int(os.environ.get("MCP_RRF_K", "60"))
EMBEDDING_BATCH_SIZE = _config_int("MCP_EMBEDDING_BATCH_SIZE", "embedding_batch_size", 4, min_value=1)

_env_model_dir = os.environ.get("MCP_MODEL_DIR")
MODEL_DIR = (
    Path(_env_model_dir).expanduser()
    if _env_model_dir
    else Path.home() / ".cache" / "my-custom-rag-python" / "models"
)

# Parâmetros do splitter (alinhados com indexer_full.py, perfil low-memory)
CHUNK_SIZE = _config_int("MCP_CHUNK_SIZE", "chunk_size", 3000, min_value=256)
CHUNK_OVERLAP = min(CHUNK_SIZE - 1, _config_int("MCP_CHUNK_OVERLAP", "chunk_overlap", 400, min_value=0))

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB
TOP_K_RESULTS = 7
MAX_QUERY_RESULTS = 30

# Filtros de varredura
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

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".kts", ".scala", ".sql",
    ".sh", ".bash", ".zsh", ".ps1", ".yaml", ".yml", ".toml", ".ini", ".conf",
    ".json", ".xml", ".html", ".css", ".scss", ".sass", ".vue", ".svelte", ".dart",
    ".lua", ".r", ".m", ".mm",
}

DOC_EXTENSIONS = {
    ".md", ".mdx", ".rst", ".txt", ".adoc", ".org", ".tex", ".csv",
}


@dataclass(frozen=True)
class BranchSpec:
    key: str
    model_choice: str
    model_id: str
    collection_name: str
    content_domain: str
    label: str


JINA_CODE_BRANCH_MODEL_CHOICE = "jina_v2" if _embedding_model_choice == "hybrid" else "jina"
JINA_CODE_BRANCH_MODEL_ID = JINA_V2_EMBEDDING_MODEL if _embedding_model_choice == "hybrid" else JINA_V3_EMBEDDING_MODEL

BRANCH_SPECS: dict[str, BranchSpec] = {
    "jina_code": BranchSpec(
        key="jina_code",
        model_choice=JINA_CODE_BRANCH_MODEL_CHOICE,
        model_id=JINA_CODE_BRANCH_MODEL_ID,
        collection_name=COLLECTION_CODE_JINA,
        content_domain="code",
        label="Jina v2 Code" if _embedding_model_choice == "hybrid" else "Jina v3 Code",
    ),
    "bge_doc": BranchSpec(
        key="bge_doc",
        model_choice="bge",
        model_id=BGE_EMBEDDING_MODEL,
        collection_name=COLLECTION_DOC_BGE,
        content_domain="doc",
        label="BGE Docs",
    ),
}

DEFAULT_SINGLE_BRANCH_KEY = "bge_doc" if _embedding_model_choice == "bge" else "jina_code"


@dataclass
class RetrievedHit:
    key: str
    document: str
    metadata: dict[str, object]
    distance: float | None
    similarity: float | None
    branch: BranchSpec
    rank: int


@dataclass
class FusedHit:
    key: str
    document: str
    metadata: dict[str, object]
    rrf_score: float
    source_details: dict[str, dict[str, object]]
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Runtime caches (lazy loading para economizar RAM)
# ---------------------------------------------------------------------------

_chroma_client: chromadb.HttpClient | None = None
_collections: dict[str, chromadb.Collection] = {}
_models: dict[str, SentenceTransformer] = {}
_model_load_errors: dict[str, str] = {}
_splitter: RecursiveCharacterTextSplitter | None = None
_reranker: CrossEncoder | None = None
_reranker_error: str | None = None


# ---------------------------------------------------------------------------
# Chroma e modelos
# ---------------------------------------------------------------------------


def _model_cache_dir(base_dir: Path, model_id: str) -> Path:
    safe_name = model_id.replace("/", "__").replace(":", "_")
    return base_dir / safe_name


def _get_chroma_client() -> chromadb.HttpClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _chroma_client.heartbeat()
        log.info("Conectado ao ChromaDB em %s:%s", CHROMA_HOST, CHROMA_PORT)
    return _chroma_client


def get_chroma_collection(collection_name: str) -> chromadb.Collection:
    if collection_name in _collections:
        return _collections[collection_name]

    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        _collections[collection_name] = collection
        return collection
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível acessar a coleção '{collection_name}' no ChromaDB "
            f"({CHROMA_HOST}:{CHROMA_PORT}). Erro: {e}"
        )


def _load_sentence_transformer_from_local(model_id: str, local_model_dir: Path) -> SentenceTransformer:
    trust_remote_code = model_id.startswith("jinaai/")
    tokenizer_kwargs = {"fix_mistral_regex": True}

    def _instantiate_model() -> SentenceTransformer:
        return SentenceTransformer(
            str(local_model_dir),
            device="cpu",
            trust_remote_code=trust_remote_code,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    def _clear_hf_dynamic_modules_cache() -> None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        if cache_dir.exists():
            log.warning("Limpando cache dinâmico do Hugging Face em %s", cache_dir)
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _load_with_jina_patch() -> SentenceTransformer:
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
        return _load_with_jina_patch()
    except FileNotFoundError as e:
        if trust_remote_code and "transformers_modules" in str(e):
            log.warning("Cache dinâmico inconsistente detectado: %s", e)
            _clear_hf_dynamic_modules_cache()
            return _load_with_jina_patch()
        raise


def _apply_jina_quantization_if_needed(model: SentenceTransformer, model_id: str) -> SentenceTransformer:
    if model_id != JINA_V3_EMBEDDING_MODEL or JINA_QUANTIZATION == "default":
        return model

    try:
        import torch
        import warnings

        quantized_layers = 0
        for module in model.modules():
            if type(module).__name__ != "ParametrizedLinear":
                continue

            float_linear = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                float_linear.weight.copy_(module.weight.detach().to(torch.float32))
                if module.bias is not None:
                    float_linear.bias.copy_(module.bias.detach().to(torch.float32))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                quantized_linear = torch.quantization.quantize_dynamic(
                    torch.nn.Sequential(float_linear),
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )[0]

            module._dynamic_int8_linear = quantized_linear

            def _forward_dynamic_int8(self, input, task_id=None, residual=False):
                out = self._dynamic_int8_linear(input)
                if residual:
                    return out, input
                return out

            module.forward = _forward_dynamic_int8.__get__(module, module.__class__)
            quantized_layers += 1

        if quantized_layers == 0:
            log.warning("Nenhuma camada ParametrizedLinear encontrada para dynamic-int8 no Jina.")
            return model

        log.info("Quantizacao Jina aplicada: dynamic-int8 (CPU, %s camadas).", quantized_layers)
        return model
    except Exception as quant_error:
        log.warning("Falha ao aplicar dynamic-int8 no Jina (%s); usando modelo padrao.", quant_error)
        return model


def get_embedding_model(model_choice: str) -> SentenceTransformer:
    if model_choice in _models:
        return _models[model_choice]

    if model_choice in _model_load_errors:
        raise RuntimeError(_model_load_errors[model_choice])

    if model_choice == "jina":
        model_id = JINA_V3_EMBEDDING_MODEL
    elif model_choice == "jina_v2":
        model_id = JINA_V2_EMBEDDING_MODEL
    elif model_choice == "bge":
        model_id = BGE_EMBEDDING_MODEL
    else:
        raise RuntimeError(f"Modelo não suportado: {model_choice}")

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        preferred_model_cache_dir = _model_cache_dir(MODEL_DIR, model_id)
        log.info("Carregando embeddings '%s' em CPU (cache: %s)", model_id, preferred_model_cache_dir)

        selection = download_model_with_fallback(
            preferred_model_id=model_id,
            fallback_model_id=model_id,
            local_dir=MODEL_DIR,
        )
        model = _load_sentence_transformer_from_local(selection.model_id, selection.local_dir)
        if model_choice == "jina":
            model = _apply_jina_quantization_if_needed(model, selection.model_id)

        _models[model_choice] = model
        log.info(
            "Modelo de embeddings pronto: %s (provider=%s, path=%s)",
            selection.model_id,
            selection.provider,
            selection.local_dir,
        )
        return model
    except Exception as e:
        message = f"Falha ao carregar modelo '{model_choice}' ({model_id}): {e}"
        _model_load_errors[model_choice] = message
        raise RuntimeError(message)


def get_reranker() -> CrossEncoder | None:
    global _reranker, _reranker_error

    if not RERANK_ENABLED:
        return None
    if _reranker is not None:
        return _reranker
    if _reranker_error is not None:
        return None

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        selection = download_model_with_fallback(
            preferred_model_id=RERANK_MODEL_ID,
            fallback_model_id=RERANK_MODEL_ID,
            local_dir=MODEL_DIR,
        )

        reranker = CrossEncoder(
            str(selection.local_dir),
            device="cpu",
            max_length=RERANKER_MAX_LENGTH,
            trust_remote_code=False,
        )

        if RERANKER_QUANTIZATION == "dynamic-int8":
            try:
                import torch

                reranker.model = torch.quantization.quantize_dynamic(
                    reranker.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                log.info("Reranker com quantizacao dynamic-int8 habilitada.")
            except Exception as quant_error:
                log.warning("Falha ao quantizar reranker (%s). Seguindo sem quantizacao.", quant_error)

        _reranker = reranker
        log.info(
            "Reranker pronto: %s (provider=%s, path=%s)",
            selection.model_id,
            selection.provider,
            selection.local_dir,
        )
        return _reranker
    except Exception as e:
        _reranker_error = str(e)
        log.warning("Reranker indisponível. Busca seguirá sem reranking. Erro: %s", e)
        return None


def get_splitter() -> RecursiveCharacterTextSplitter:
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
# Indexação interna
# ---------------------------------------------------------------------------


def _make_chunk_id(file_path: str, chunk_index: int) -> str:
    raw = f"{file_path}::chunk::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _make_result_key(metadata: dict[str, object], fallback_id: str) -> str:
    file_path = str(metadata.get("file_path", ""))
    chunk_index = str(metadata.get("chunk_index", ""))
    if file_path and chunk_index:
        return f"{file_path}::chunk::{chunk_index}"
    return fallback_id


def _delete_file_chunks(collection: chromadb.Collection, file_path: str) -> int:
    # Pede apenas IDs para não materializar documentos/metadata desnecessários em RAM.
    results = collection.get(where={"file_path": file_path}, include=[])
    ids = results.get("ids", []) if results else []
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


def _scan_folder(folder_path: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not d.startswith(".")
        ]
        dirnames.sort()
        for filename in sorted(filenames):
            fp = Path(dirpath) / filename
            if fp.suffix.lower() in IGNORED_EXTENSIONS:
                continue
            try:
                if fp.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue
            yield fp


def _classify_file_targets(filepath: Path) -> list[BranchSpec]:
    suffix = filepath.suffix.lower()
    is_code = suffix in CODE_EXTENSIONS
    is_doc = suffix in DOC_EXTENSIONS

    if is_code and not is_doc:
        return [BRANCH_SPECS["jina_code"]]
    if is_doc and not is_code:
        return [BRANCH_SPECS["bge_doc"]]

    # Arquivos ambíguos/extensão desconhecida: indexa em ambas para não perder recall.
    return [BRANCH_SPECS["jina_code"], BRANCH_SPECS["bge_doc"]]


def _index_single_file_for_branch(
    filepath: Path,
    branch: BranchSpec,
    splitter: RecursiveCharacterTextSplitter,
    *,
    delete_existing: bool = True,
) -> int:
    content = _read_file_safe(filepath)
    if not content or not content.strip():
        return 0

    abs_path = str(filepath.resolve())
    model = get_embedding_model(branch.model_choice)
    collection = get_chroma_collection(branch.collection_name)

    chunks = splitter.split_text(content)
    if not chunks:
        return 0

    # Atualização idempotente por arquivo em cada coleção.
    if delete_existing:
        _delete_file_chunks(collection, abs_path)

    inserted_chunks = 0
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metadatas: list[dict[str, object]] = []

    def _flush_batch() -> None:
        nonlocal inserted_chunks
        if not batch_ids:
            return
        embeddings = model.encode(
            batch_docs,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        ).tolist()
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas,
        )
        inserted_chunks += len(batch_ids)
        del embeddings
        batch_ids.clear()
        batch_docs.clear()
        batch_metadatas.clear()

    for i, chunk in enumerate(chunks):
        batch_ids.append(_make_chunk_id(abs_path, i))
        batch_docs.append(chunk)
        batch_metadatas.append(
            {
                "file_path": abs_path,
                "file_name": filepath.name,
                "chunk_index": i,
                "source_collection": branch.collection_name,
                "source_model_choice": branch.model_choice,
                "source_model_id": branch.model_id,
                "content_domain": branch.content_domain,
            }
        )
        if len(batch_ids) >= EMBEDDING_BATCH_SIZE:
            _flush_batch()

    _flush_batch()
    return inserted_chunks


def _remove_file_from_all_collections(abs_path: str) -> tuple[dict[str, int], list[str]]:
    deleted_per_branch: dict[str, int] = {}
    errors: list[str] = []

    for branch in BRANCH_SPECS.values():
        try:
            collection = get_chroma_collection(branch.collection_name)
            deleted = _delete_file_chunks(collection, abs_path)
            deleted_per_branch[branch.key] = deleted
        except Exception as e:
            errors.append(f"{branch.key}: {e}")
    return deleted_per_branch, errors


# ---------------------------------------------------------------------------
# Busca semântica híbrida
# ---------------------------------------------------------------------------


def _query_branch(branch: BranchSpec, query: str, n_results: int) -> tuple[list[RetrievedHit], str | None]:
    try:
        collection = get_chroma_collection(branch.collection_name)
        model = get_embedding_model(branch.model_choice)
    except Exception as e:
        return [], f"{branch.key}: recurso indisponível ({e})"

    try:
        query_embedding = model.encode([query], show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return [], f"{branch.key}: falha na query ({e})"

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    hits: list[RetrievedHit] = []
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        metadata = meta or {}
        fallback_id = ids[idx - 1] if idx - 1 < len(ids) else f"{branch.key}:{idx}"
        key = _make_result_key(metadata, fallback_id)

        similarity = None
        if dist is not None:
            try:
                similarity = 1.0 - float(dist)
            except Exception:
                similarity = None

        hits.append(
            RetrievedHit(
                key=key,
                document=(doc or ""),
                metadata=metadata,
                distance=float(dist) if dist is not None else None,
                similarity=similarity,
                branch=branch,
                rank=idx,
            )
        )

    return hits, None


def _rrf_fuse(hits_by_branch: dict[str, list[RetrievedHit]], top_limit: int) -> list[FusedHit]:
    fused: dict[str, FusedHit] = {}

    for branch_key, hits in hits_by_branch.items():
        _ = branch_key
        for rank, hit in enumerate(hits, start=1):
            contribution = 1.0 / (RRF_K + rank)
            entry = fused.get(hit.key)

            if entry is None:
                entry = FusedHit(
                    key=hit.key,
                    document=hit.document,
                    metadata=dict(hit.metadata),
                    rrf_score=0.0,
                    source_details={},
                )
                fused[hit.key] = entry

            entry.rrf_score += contribution
            entry.source_details[hit.branch.key] = {
                "rank": rank,
                "distance": hit.distance,
                "similarity": hit.similarity,
                "collection": hit.branch.collection_name,
                "model_choice": hit.branch.model_choice,
                "model_id": hit.branch.model_id,
                "content_domain": hit.branch.content_domain,
            }

            # Usa metadados do hit com melhor similaridade local como base principal.
            current_sim = entry.metadata.get("_best_similarity", -10.0)
            candidate_sim = hit.similarity if hit.similarity is not None else -10.0
            if candidate_sim > current_sim:
                entry.document = hit.document
                entry.metadata = dict(hit.metadata)
                entry.metadata["_best_similarity"] = candidate_sim

    fused_hits = list(fused.values())
    fused_hits.sort(key=lambda h: h.rrf_score, reverse=True)

    # Limita o pool antes do reranking para reduzir CPU/RAM.
    return fused_hits[:top_limit]


def _apply_rerank(query: str, fused_hits: list[FusedHit], top_k: int) -> tuple[list[FusedHit], bool, str | None]:
    if not fused_hits:
        return [], False, None

    reranker = get_reranker()
    if reranker is None:
        reason = _reranker_error if _reranker_error else "reranker_desabilitado"
        return fused_hits[:top_k], False, reason

    try:
        pairs = [(query, hit.document) for hit in fused_hits]
        scores = reranker.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

        for hit, score in zip(fused_hits, scores):
            hit.rerank_score = float(score)

        fused_hits.sort(
            key=lambda h: (
                h.rerank_score if h.rerank_score is not None else -1e9,
                h.rrf_score,
            ),
            reverse=True,
        )
        return fused_hits[:top_k], True, None
    except Exception as e:
        return fused_hits[:top_k], False, str(e)


def _format_similarity(similarity: float | None) -> str:
    if similarity is None:
        return "n/a"
    return f"{round(similarity * 100, 1)}%"


def _format_fused_results(
    *,
    query: str,
    mode: str,
    hits: list[FusedHit],
    branch_errors: list[str],
    rerank_applied: bool,
    rerank_error: str | None,
) -> str:
    if not hits:
        msg = "Nenhum resultado encontrado. As coleções podem estar vazias."
        if branch_errors:
            msg += "\nFalhas detectadas: " + " | ".join(branch_errors)
        return msg

    lines: list[str] = [f"# Resultados para: '{query}'", f"**Modo:** {mode}"]

    if branch_errors:
        lines.append("**Avisos de branch:** " + " | ".join(branch_errors))

    if mode == "ensemble":
        if rerank_applied:
            lines.append(f"**Reranking:** ativo ({RERANK_MODEL_ID})")
        else:
            lines.append(f"**Reranking:** indisponível ({rerank_error or 'sem detalhes'})")

    lines.append("")

    for idx, hit in enumerate(hits, start=1):
        metadata = dict(hit.metadata)
        metadata.pop("_best_similarity", None)

        file_path = str(metadata.get("file_path", "desconhecido"))
        chunk_index = metadata.get("chunk_index", "?")
        file_name = str(metadata.get("file_name", Path(file_path).name if file_path != "desconhecido" else "?"))

        source_models = sorted({str(v.get("model_choice", "?")) for v in hit.source_details.values()})
        source_collections = sorted({str(v.get("collection", "?")) for v in hit.source_details.values()})

        source_parts: list[str] = []
        for source_key, details in sorted(
            hit.source_details.items(),
            key=lambda item: int(item[1].get("rank", 999999)),
        ):
            source_parts.append(
                f"{source_key}(rank={details.get('rank')}, sim={_format_similarity(details.get('similarity'))})"
            )

        snippet = hit.document.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + "\n... [truncado]"

        score_line = f"RRF={hit.rrf_score:.4f}"
        if hit.rerank_score is not None:
            score_line += f" | rerank={hit.rerank_score:.4f}"

        lines.append(f"## [{idx}] {file_path}")
        lines.append(f"**Scores:** {score_line}")
        lines.append(f"**Fontes de recuperação:** {', '.join(source_parts)}")
        lines.append(
            "**Metadados unificados:** "
            f"file_name={file_name} | chunk_index={chunk_index} | "
            f"source_models={source_models} | source_collections={source_collections}"
        )
        lines.append("")
        lines.append(f"```\n{snippet}\n```")
        lines.append("")

    return "\n".join(lines)


def _run_single_mode(query: str, top_k: int) -> tuple[list[FusedHit], list[str], bool, str | None]:
    primary_branch = BRANCH_SPECS[DEFAULT_SINGLE_BRANCH_KEY]

    hits, error = _query_branch(primary_branch, query, top_k)
    errors: list[str] = []
    if error:
        errors.append(error)

    # Fallback automático para a branch alternativa, preservando disponibilidade.
    if not hits:
        fallback_branch_key = "bge_doc" if primary_branch.key == "jina_code" else "jina_code"
        fallback_hits, fallback_error = _query_branch(BRANCH_SPECS[fallback_branch_key], query, top_k)
        if fallback_error:
            errors.append(fallback_error)
        if fallback_hits:
            hits = fallback_hits

    if not hits:
        return [], errors, False, None

    fused = _rrf_fuse({"single": hits}, top_k)
    return fused, errors, False, None


def _run_ensemble_mode(query: str, top_k: int) -> tuple[list[FusedHit], list[str], bool, str | None]:
    per_branch_k = min(MAX_QUERY_RESULTS, max(top_k * 2, top_k))
    branches = [BRANCH_SPECS["jina_code"], BRANCH_SPECS["bge_doc"]]

    hits_by_branch: dict[str, list[RetrievedHit]] = {}
    branch_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=len(branches)) as executor:
        futures = {
            executor.submit(_query_branch, branch, query, per_branch_k): branch
            for branch in branches
        }
        for future in as_completed(futures):
            branch = futures[future]
            try:
                hits, error = future.result()
                if error:
                    branch_errors.append(error)
                if hits:
                    hits_by_branch[branch.key] = hits
            except Exception as e:
                branch_errors.append(f"{branch.key}: falha inesperada ({e})")

    if not hits_by_branch:
        return [], branch_errors, False, None

    candidate_pool = min(RERANK_MAX_CANDIDATES, max(top_k, top_k * RERANK_CANDIDATE_MULTIPLIER))
    fused_candidates = _rrf_fuse(hits_by_branch, candidate_pool)
    reranked_hits, rerank_applied, rerank_error = _apply_rerank(query, fused_candidates, top_k)
    return reranked_hits, branch_errors, rerank_applied, rerank_error


# ---------------------------------------------------------------------------
# Servidor MCP via FastMCP
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="rag-codebase",
    instructions=(
        "Servidor RAG para busca semântica em código-fonte local com suporte a ensemble híbrido. "
        "No modo hybrid, a branch de código usa Jina v2 e a de documentação usa BGE. "
        "Use semantic_search_code(query, top_k, mode='ensemble') para combinar Jina+BGE com RRF e reranking. "
        "Use update_file_index após editar um arquivo para manter as duas coleções sincronizadas. "
        "Use index_specific_folder para indexação recursiva sob demanda."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: semantic_search_code
# ---------------------------------------------------------------------------

@mcp.tool()
def semantic_search_code(query: str, top_k: int = TOP_K_RESULTS, mode: str = SEARCH_MODE_DEFAULT) -> str:
    """
    Busca semântica no índice vetorial local.

    Modos:
    - single: usa apenas uma branch (Jina/BGE conforme MCP_EMBEDDING_MODEL; no hybrid, Jina v2).
    - ensemble: consulta em paralelo code_vectors_jina + doc_vectors_bge,
      faz fusão via Reciprocal Rank Fusion (RRF) e reranking leve.

    Args:
        query: Descrição do que procurar.
        top_k: Quantidade final de resultados.
        mode: "single" (padrão) ou "ensemble".

    Returns:
        Resultado textual formatado para consumo pelo LLM.
    """
    raw_query = (query or "").strip()
    search_mode = (mode or SEARCH_MODE_DEFAULT).strip().lower()

    _log_tool_usage(
        event="tool_call_start",
        tool_name="semantic_search_code",
        details={
            "query_preview": _safe_preview(raw_query),
            "query_len": len(raw_query),
            "top_k": top_k,
            "mode": search_mode,
        },
    )

    if not raw_query:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "error", "reason": "empty_query"},
        )
        return "Erro: a query não pode ser vazia."

    top_k = max(1, min(top_k, 20))
    if search_mode not in {"single", "ensemble"}:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "error", "reason": "invalid_mode", "mode": search_mode},
        )
        return "Erro: mode inválido. Use 'single' ou 'ensemble'."

    try:
        if search_mode == "ensemble":
            hits, branch_errors, rerank_applied, rerank_error = _run_ensemble_mode(raw_query, top_k)
        else:
            hits, branch_errors, rerank_applied, rerank_error = _run_single_mode(raw_query, top_k)

        result_text = _format_fused_results(
            query=raw_query,
            mode=search_mode,
            hits=hits,
            branch_errors=branch_errors,
            rerank_applied=rerank_applied,
            rerank_error=rerank_error,
        )

        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={
                "status": "ok",
                "mode": search_mode,
                "result_count": len(hits),
                "branch_errors": len(branch_errors),
                "rerank_applied": rerank_applied,
            },
        )
        return result_text
    except Exception as e:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="semantic_search_code",
            details={"status": "error", "reason": "search_failed", "error": str(e), "mode": search_mode},
        )
        return f"Erro ao executar busca semântica ({search_mode}): {e}"


# ---------------------------------------------------------------------------
# Tool 2: update_file_index
# ---------------------------------------------------------------------------

@mcp.tool()
def update_file_index(file_path: str) -> str:
    """
    Atualiza o índice RAG para um arquivo específico.

    O arquivo é classificado como código/doc e indexado na coleção apropriada.
    Para extensões ambíguas, indexa em ambas as coleções.
    """
    filepath = Path(file_path).resolve()
    abs_path = str(filepath)

    _log_tool_usage(
        event="tool_call_start",
        tool_name="update_file_index",
        details={"file_path": abs_path},
    )

    if not filepath.exists():
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "file_not_found", "file_path": abs_path},
        )
        return f"Erro: arquivo não encontrado: {filepath}"

    if not filepath.is_file():
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "not_a_file", "file_path": abs_path},
        )
        return f"Erro: o caminho não aponta para um arquivo: {filepath}"

    if filepath.stat().st_size > MAX_FILE_SIZE_BYTES:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="update_file_index",
            details={"status": "error", "reason": "file_too_large", "file_path": abs_path},
        )
        return f"Erro: arquivo muito grande (>{MAX_FILE_SIZE_BYTES // 1024}KB): {filepath}"

    splitter = get_splitter()
    targets = _classify_file_targets(filepath)

    deleted_per_branch, deletion_errors = _remove_file_from_all_collections(abs_path)

    inserted_per_branch: dict[str, int] = {}
    index_errors: list[str] = []
    for branch in targets:
        try:
            inserted = _index_single_file_for_branch(
                filepath,
                branch,
                splitter,
                delete_existing=False,  # já removido em todas as coleções acima
            )
            inserted_per_branch[branch.key] = inserted
        except Exception as e:
            index_errors.append(f"{branch.key}: {e}")

    success_branches = [k for k, v in inserted_per_branch.items() if v > 0]

    details = {
        "status": "ok" if success_branches else "error",
        "file_path": abs_path,
        "targets": [b.key for b in targets],
        "deleted_per_branch": deleted_per_branch,
        "inserted_per_branch": inserted_per_branch,
        "deletion_errors": len(deletion_errors),
        "index_errors": len(index_errors),
    }
    _log_tool_usage(event="tool_call_end", tool_name="update_file_index", details=details)

    if not success_branches and index_errors:
        return (
            "Erro: não foi possível reindexar o arquivo em nenhuma coleção.\n"
            f"Arquivo: {filepath}\n"
            "Falhas: " + " | ".join(index_errors)
        )

    lines = [
        "Arquivo reindexado.",
        f"  Arquivo : {filepath}",
        f"  Coleções alvo: {[b.collection_name for b in targets]}",
        f"  Remoções por coleção: {deleted_per_branch}",
        f"  Inserções por coleção: {inserted_per_branch}",
    ]
    if deletion_errors:
        lines.append("  Avisos na remoção: " + " | ".join(deletion_errors))
    if index_errors:
        lines.append("  Avisos na indexação: " + " | ".join(index_errors))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: delete_file_index
# ---------------------------------------------------------------------------

@mcp.tool()
def delete_file_index(file_path: str) -> str:
    """
    Remove um arquivo do índice em todas as coleções gerenciadas.
    """
    filepath = Path(file_path).resolve()
    abs_path = str(filepath)

    _log_tool_usage(
        event="tool_call_start",
        tool_name="delete_file_index",
        details={"file_path": abs_path},
    )

    deleted_per_branch, errors = _remove_file_from_all_collections(abs_path)
    total_deleted = sum(deleted_per_branch.values())

    _log_tool_usage(
        event="tool_call_end",
        tool_name="delete_file_index",
        details={
            "status": "ok" if total_deleted > 0 else "warning",
            "file_path": abs_path,
            "deleted_per_branch": deleted_per_branch,
            "errors": len(errors),
        },
    )

    if total_deleted == 0:
        base = f"Nenhum chunk encontrado para o arquivo: {abs_path}"
        if errors:
            base += "\nFalhas parciais: " + " | ".join(errors)
        return base

    out = [
        "Removido do índice com sucesso.",
        f"  Arquivo : {abs_path}",
        f"  Deleções por coleção: {deleted_per_branch}",
    ]
    if errors:
        out.append("  Avisos: " + " | ".join(errors))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tool 4: index_specific_folder
# ---------------------------------------------------------------------------

@mcp.tool()
def index_specific_folder(folder_path: str) -> str:
    """
    Indexa recursivamente uma pasta em coleções separadas por domínio.
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

    splitter = get_splitter()

    processed_files = 0
    branch_file_counts = {key: 0 for key in BRANCH_SPECS}
    branch_chunk_counts = {key: 0 for key in BRANCH_SPECS}
    error_count = 0
    error_samples: list[str] = []

    for filepath in _scan_folder(folder):
        processed_files += 1
        targets = _classify_file_targets(filepath)

        for branch in targets:
            try:
                n_chunks = _index_single_file_for_branch(filepath, branch, splitter)
                branch_file_counts[branch.key] += 1
                branch_chunk_counts[branch.key] += n_chunks
            except Exception as e:
                error_count += 1
                if len(error_samples) < 10:
                    error_samples.append(f"{filepath.name} [{branch.key}]: {e}")

    if processed_files == 0:
        _log_tool_usage(
            event="tool_call_end",
            tool_name="index_specific_folder",
            details={"status": "ok", "folder_path": str(folder), "files_processed": 0, "chunks": 0, "errors": 0},
        )
        return f"Nenhum arquivo indexável encontrado em: {folder}"

    total_chunks = sum(branch_chunk_counts.values())

    _log_tool_usage(
        event="tool_call_end",
        tool_name="index_specific_folder",
        details={
            "status": "ok",
            "folder_path": str(folder),
            "files_processed": processed_files,
            "chunks": total_chunks,
            "errors": error_count,
            "branch_file_counts": branch_file_counts,
            "branch_chunk_counts": branch_chunk_counts,
        },
    )

    report = [
        "Indexação da pasta concluída.",
        f"  Pasta: {folder}",
        f"  Arquivos processados: {processed_files}",
        f"  Total de chunks: {total_chunks}",
        f"  Arquivos por branch: {branch_file_counts}",
        f"  Chunks por branch: {branch_chunk_counts}",
    ]

    if error_count:
        report.append(f"  Erros ({error_count}):")
        for err in error_samples:
            report.append(f"    - {err}")
        if error_count > len(error_samples):
            report.append(f"    ... e mais {error_count - len(error_samples)} erros.")

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Iniciando servidor MCP RAG (stdio)...")
    log.info("ChromaDB: %s:%s", CHROMA_HOST, CHROMA_PORT)
    log.info(
        "Coleções: %s (%s), %s (%s)",
        COLLECTION_CODE_JINA,
        BRANCH_SPECS["jina_code"].model_id,
        COLLECTION_DOC_BGE,
        BRANCH_SPECS["bge_doc"].model_id,
    )
    log.info("Modo padrão de busca: %s", SEARCH_MODE_DEFAULT)
    log.info("Modelo single padrão: %s", BRANCH_SPECS[DEFAULT_SINGLE_BRANCH_KEY].model_id)
    log.info("Quantizacao Jina: %s", JINA_QUANTIZATION)
    log.info("Config de tuning carregada de: %s (found=%s)", INDEXER_CONFIG_PATH, bool(INDEXER_TUNING_CONFIG))
    log.info("Embedding batch size: %s", EMBEDDING_BATCH_SIZE)
    log.info("Chunk params: size=%s overlap=%s", CHUNK_SIZE, CHUNK_OVERLAP)
    log.info("Reranker: %s (enabled=%s, quant=%s)", RERANK_MODEL_ID, RERANK_ENABLED, RERANKER_QUANTIZATION)
    log.info("Pasta de modelos locais: %s", MODEL_DIR)
    log.info("Uso MCP será registrado em: %s", MCP_USAGE_LOG_PATH)

    # Pré-aquece somente conexão Chroma; modelos ficam lazy para poupar RAM.
    try:
        _get_chroma_client()
        get_chroma_collection(COLLECTION_CODE_JINA)
        get_chroma_collection(COLLECTION_DOC_BGE)
        log.info("Conexão Chroma inicializada. Modelos serão carregados sob demanda.")
    except Exception as e:
        log.error("Falha ao inicializar ChromaDB: %s", e)
        log.error("O servidor continuará, mas as ferramentas retornarão erro até o ChromaDB estar disponível.")

    mcp.run(transport="stdio")
