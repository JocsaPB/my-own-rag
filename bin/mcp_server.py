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
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP

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
EMBEDDING_MODEL = "BAAI/bge-m3"
_env_model_dir = os.environ.get("MCP_MODEL_DIR")
_cwd_model_dir = Path.cwd() / "model"
_script_model_dir = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR = (
    Path(_env_model_dir).expanduser()
    if _env_model_dir
    else _cwd_model_dir if _cwd_model_dir.exists()
    else _script_model_dir
)

# Parâmetros do splitter (idênticos ao indexer_full.py)
# BGE-M3 suporta até 8192 tokens — chunks maiores preservam funções inteiras
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
    global _model
    if _model is None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL} (CPU)...")
        log.info(f"Diretório local de modelo: {MODEL_DIR}")

        downloaded = False
        try:
            from huggingface_hub import snapshot_download

            log.info(f"Tentando baixar/atualizar '{EMBEDDING_MODEL}' via API do Hugging Face...")
            snapshot_download(
                repo_id=EMBEDDING_MODEL,
                local_dir=str(MODEL_DIR),
            )
            downloaded = True
            log.info("Download/atualização do modelo concluído.")
        except Exception as e:
            log.warning(f"Falha ao baixar '{EMBEDDING_MODEL}' via API: {e}")
            log.warning("Tentando usar o modelo local já salvo em 'model/'.")

        try:
            _model = SentenceTransformer(str(MODEL_DIR), device="cpu")
            if downloaded:
                log.info("Modelo carregado a partir da pasta local atualizada.")
            else:
                log.info("Modelo carregado a partir da pasta local (fallback).")
        except Exception as local_error:
            if downloaded:
                raise RuntimeError(
                    f"Modelo baixado mas falhou ao carregar da pasta local '{MODEL_DIR}': {local_error}"
                )
            raise RuntimeError(
                "Falha ao carregar modelo: download via API indisponível e modelo local não encontrado/válido "
                f"em '{MODEL_DIR}'. Erro: {local_error}"
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
    log.info(f"Modelo: {EMBEDDING_MODEL} (CPU)")
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
