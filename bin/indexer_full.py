#!/usr/bin/env python3
"""
indexer_full.py — Script standalone de indexação do RAG local.

Uso:
    python indexer_full.py [caminho_do_projeto]

Se nenhum caminho for passado, usa o diretório atual.
O ChromaDB deve estar rodando via Docker em localhost:8000.
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "codebase"

# Pastas e extensões ignoradas durante a varredura
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "out", ".next", ".nuxt", ".cache", "coverage",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", "bin", "obj",
    ".idea", ".vscode", ".DS_Store", "vendor", "tmp", "temp", "logs",
    ".rag_db",
}

IGNORED_EXTENSIONS = {
    # Binários e imagens
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp4", ".mp3", ".wav", ".ogg", ".avi", ".mov",
    # Pacotes e compilados
    ".zip", ".tar", ".gz", ".rar", ".7z", ".jar", ".war", ".ear",
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    # Lockfiles e gerados
    ".lock", ".sum",
    # Banco de dados
    ".sqlite", ".db", ".sqlite3",
    # Fontes
    ".ttf", ".woff", ".woff2", ".eot",
    # PDF/Documentos binários
    ".pdf", ".docx", ".xlsx", ".pptx",
}

# Tamanho máximo de arquivo (evita indexar arquivos enormes gerados)
MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB

# Parâmetros do splitter (aproximação de tokens via caracteres)
# Jina Code V2 suporta até 8192 tokens — ~6000 chars ≈ 1500 tokens (4 chars/token)
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 800

# Modelo de embeddings (roda na CPU)
# BGE-M3: multilingual, 1024D, 8192 token context, state-of-the-art no MTEB
EMBEDDING_MODEL = "BAAI/bge-m3"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Retorna o splitter compartilhado com as configurações padrão do projeto."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def load_embedding_model() -> SentenceTransformer:
    """Carrega o modelo de embeddings forçando uso de CPU."""
    print(f"[+] Carregando modelo de embeddings: {EMBEDDING_MODEL} (CPU)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    print("[+] Modelo carregado com sucesso.")
    return model


def connect_to_chroma() -> chromadb.HttpClient:
    """Conecta ao ChromaDB via HTTP e valida a conexão."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # Faz um heartbeat para confirmar que o servidor está no ar
        client.heartbeat()
        print(f"[+] Conectado ao ChromaDB em {CHROMA_HOST}:{CHROMA_PORT}")
        return client
    except Exception as e:
        print(f"[ERRO] Não foi possível conectar ao ChromaDB: {e}")
        print("       Verifique se o container Docker está rodando:")
        print("       docker compose up -d")
        sys.exit(1)


def scan_files(root_path: Path) -> list[Path]:
    """
    Varre recursivamente o diretório raiz, retornando apenas arquivos
    de texto relevantes para indexação.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Remove dirs ignorados in-place para que os.walk não desça neles
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            filepath = Path(dirpath) / filename

            # Ignora por extensão
            if filepath.suffix.lower() in IGNORED_EXTENSIONS:
                continue

            # Ignora arquivos muito grandes
            try:
                if filepath.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            files.append(filepath)

    return sorted(files)


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """Gera um ID determinístico para cada chunk baseado no caminho + índice."""
    raw = f"{file_path}::chunk::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def read_file_safe(filepath: Path) -> str | None:
    """Lê um arquivo de texto, tentando múltiplos encodings."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return filepath.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as e:
            print(f"  [AVISO] Não foi possível ler {filepath}: {e}")
            return None
    # Se nenhum encoding funcionou, é provavelmente binário disfarçado
    return None


def delete_file_chunks(collection: chromadb.Collection, file_path: str) -> None:
    """Remove todos os chunks de um arquivo específico da coleção."""
    try:
        results = collection.get(where={"file_path": file_path})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception as e:
        print(f"  [AVISO] Erro ao deletar chunks de {file_path}: {e}")


# ---------------------------------------------------------------------------
# Indexação de um único arquivo
# ---------------------------------------------------------------------------

def index_file(
    filepath: Path,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    splitter: RecursiveCharacterTextSplitter,
    root_path: Path,
) -> int:
    """
    Indexa um único arquivo: lê, divide em chunks, gera embeddings e faz upsert.
    Retorna o número de chunks indexados.
    """
    content = read_file_safe(filepath)
    if not content or not content.strip():
        return 0

    # Usa caminho absoluto como metadado
    abs_path = str(filepath.resolve())

    # Remove chunks antigos deste arquivo (atualização idempotente)
    delete_file_chunks(collection, abs_path)

    chunks = splitter.split_text(content)
    if not chunks:
        return 0

    ids = [make_chunk_id(abs_path, i) for i in range(len(chunks))]
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    metadatas = [
        {
            "file_path": abs_path,
            "chunk_index": i,
            "file_name": filepath.name,
            # Caminho relativo à raiz do projeto para exibição compacta
            "relative_path": str(filepath.relative_to(root_path)),
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


# ---------------------------------------------------------------------------
# Ponto de entrada principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Indexa um projeto de código no ChromaDB para RAG local."
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Caminho raiz do projeto a indexar (padrão: diretório atual)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Limpa toda a coleção antes de reindexar",
    )
    args = parser.parse_args()

    root_path = Path(args.project_path).resolve()
    if not root_path.is_dir():
        print(f"[ERRO] Caminho não existe ou não é um diretório: {root_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RAG Indexer — Projeto: {root_path}")
    print(f"{'='*60}\n")

    # Inicializa componentes
    client = connect_to_chroma()
    model = load_embedding_model()
    splitter = get_text_splitter()

    # Obtém ou cria a coleção
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Distância cosseno para embeddings de texto
    )

    if args.clear:
        print("[!] Limpando coleção existente...")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[+] Coleção limpa e recriada.")

    # Varre os arquivos
    print(f"\n[+] Varrendo arquivos em: {root_path}")
    files = scan_files(root_path)
    print(f"[+] {len(files)} arquivos encontrados para indexação.\n")

    if not files:
        print("[AVISO] Nenhum arquivo encontrado. Verifique o caminho e os filtros.")
        sys.exit(0)

    # Indexação com barra de progresso
    total_chunks = 0
    errors = 0

    with tqdm(
        total=len(files),
        desc="Indexando",
        unit="arquivo",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for filepath in files:
            try:
                n_chunks = index_file(filepath, collection, model, splitter, root_path)
                total_chunks += n_chunks
                pbar.set_postfix({"chunks": total_chunks, "atual": filepath.name[:20]})
            except Exception as e:
                errors += 1
                tqdm.write(f"  [ERRO] {filepath}: {e}")
            finally:
                pbar.update(1)

    print(f"\n{'='*60}")
    print(f"  Indexação concluída!")
    print(f"  Arquivos processados : {len(files)}")
    print(f"  Total de chunks      : {total_chunks}")
    print(f"  Erros                : {errors}")
    print(f"  Coleção ChromaDB     : '{COLLECTION_NAME}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
