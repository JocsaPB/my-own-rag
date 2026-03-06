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
import shutil
import logging
from pathlib import Path

# Evita avisos "advisory" ruidosos do transformers no fluxo interativo.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


class _TorchDtypeWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "`torch_dtype` is deprecated! Use `dtype` instead!" not in record.getMessage()


for _logger_name in ("transformers.configuration_utils", "transformers.modeling_utils"):
    logging.getLogger(_logger_name).addFilter(_TorchDtypeWarningFilter())

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from download_model_from_hugginface import download_model_with_fallback

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
# Jina Embeddings V3 suporta janelas longas — ~6000 chars ≈ 1500 tokens (4 chars/token)
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 800

# Modelo de embeddings (roda na CPU)
JINA_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL_CHOICE = "jina"
DEFAULT_JINA_QUANTIZATION = "default"
MODEL_CACHE_BASE_DIR = Path(
    os.environ.get("MCP_MODEL_DIR", str(Path.home() / ".cache" / "my-custom-rag-python" / "models"))
).expanduser()


def _resolve_model_id(model_choice: str) -> str:
    return JINA_EMBEDDING_MODEL if model_choice == "jina" else BGE_EMBEDDING_MODEL


def _resolve_fallback_model_id(model_choice: str) -> str:
    return BGE_EMBEDDING_MODEL if model_choice == "jina" else BGE_EMBEDDING_MODEL


def _model_cache_dir(base_dir: Path, model_id: str) -> Path:
    safe_name = model_id.replace("/", "__").replace(":", "_")
    return base_dir / safe_name


def _pick_with_prompt(
    *,
    current_value: str | None,
    default_value: str,
    title: str,
    options: list[tuple[str, str]],
) -> str:
    if current_value:
        return current_value
    if not sys.stdin.isatty():
        return default_value

    print(f"\n[CONFIG] {title}")
    for index, (_, description) in enumerate(options, start=1):
        print(f"  {index}) {description}")
    print(f"  Enter = padrão ({default_value})")

    answer = input("> Escolha: ").strip()
    if not answer:
        return default_value
    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    lowered = answer.lower()
    valid_keys = {k for k, _ in options}
    if lowered in valid_keys:
        return lowered
    print(f"[AVISO] Opção inválida '{answer}'. Usando padrão: {default_value}")
    return default_value


def resolve_embedding_config(
    model_choice_arg: str | None,
    jina_quantization_arg: str | None,
) -> tuple[str, str]:
    model_choice = model_choice_arg or os.environ.get("MCP_EMBEDDING_MODEL")
    if model_choice:
        model_choice = model_choice.strip().lower()
    model_choice = _pick_with_prompt(
        current_value=model_choice,
        default_value=DEFAULT_EMBEDDING_MODEL_CHOICE,
        title="Escolha do modelo de embeddings",
        options=[
            (
                "jina",
                f"jina ({JINA_EMBEDDING_MODEL}) - melhor performance em projetos majoritariamente de codigo.",
            ),
            (
                "bge",
                f"bge ({BGE_EMBEDDING_MODEL}) - melhor para conteudo misto (codigo + documentacao).",
            ),
        ],
    )
    if model_choice not in {"jina", "bge"}:
        print(f"[AVISO] MCP_EMBEDDING_MODEL inválido '{model_choice}'. Usando '{DEFAULT_EMBEDDING_MODEL_CHOICE}'.")
        model_choice = DEFAULT_EMBEDDING_MODEL_CHOICE

    jina_quantization = jina_quantization_arg or os.environ.get("MCP_JINA_QUANTIZATION")
    if jina_quantization:
        jina_quantization = jina_quantization.strip().lower().replace("_", "-")

    if model_choice == "jina":
        jina_quantization = _pick_with_prompt(
            current_value=jina_quantization,
            default_value=DEFAULT_JINA_QUANTIZATION,
            title="Quantizacao do Jina (apenas para CPU)",
            options=[
                ("default", "default (sem quantizacao) - maior qualidade, indexacao mais lenta."),
                ("dynamic-int8", "dynamic-int8 - indexacao mais rapida e menor uso de RAM, com pequena perda de qualidade."),
            ],
        )
        if jina_quantization not in {"default", "dynamic-int8"}:
            print(
                f"[AVISO] MCP_JINA_QUANTIZATION inválido '{jina_quantization}'. "
                f"Usando '{DEFAULT_JINA_QUANTIZATION}'."
            )
            jina_quantization = DEFAULT_JINA_QUANTIZATION
    else:
        jina_quantization = "default"

    return model_choice, jina_quantization


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


def load_embedding_model(model_choice: str, jina_quantization: str) -> SentenceTransformer:
    """Carrega o modelo de embeddings forçando uso de CPU."""
    embedding_model_id = _resolve_model_id(model_choice)
    fallback_model_id = _resolve_fallback_model_id(model_choice)

    model_base_dir = MODEL_CACHE_BASE_DIR
    model_base_dir.mkdir(parents=True, exist_ok=True)
    preferred_model_cache_dir = _model_cache_dir(model_base_dir, embedding_model_id)

    print(f"[+] Baixando modelo preferido: {embedding_model_id}")
    print(f"[+] Diretório de download/cache do modelo: {preferred_model_cache_dir}")
    selection = download_model_with_fallback(
        preferred_model_id=embedding_model_id,
        fallback_model_id=fallback_model_id,
        local_dir=model_base_dir,
    )
    selected_model_dir = selection.local_dir
    print(
        f"[+] Modelo selecionado: {selection.model_id} "
        f"(provider={selection.provider}, path={selected_model_dir})"
    )

    def _clear_hf_dynamic_modules_cache() -> None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        if cache_dir.exists():
            print(f"[!] Limpando cache de módulos dinâmicos do Hugging Face: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _load_from_local_dir(model_id: str) -> SentenceTransformer:
        # O modelo da Jina depende de código remoto; fallback normalmente não.
        trust_remote_code = model_id.startswith("jinaai/")
        tokenizer_kwargs = {"fix_mistral_regex": True}

        def _instantiate_model() -> SentenceTransformer:
            return SentenceTransformer(
                str(selected_model_dir),
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
            model_refs = {str(selected_model_dir), str(selected_model_dir.resolve())}

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

        print(f"[+] Carregando modelo de embeddings a partir de: {selected_model_dir} (CPU)...")
        try:
            return _load_with_mistral_regex_patch()
        except FileNotFoundError as e:
            # Corrige corrupção/incompletude no cache dinâmico do transformers.
            if trust_remote_code and "transformers_modules" in str(e):
                print(f"[!] Cache dinâmico inconsistente detectado: {e}")
                _clear_hf_dynamic_modules_cache()
                return _load_with_mistral_regex_patch()
            raise

    def _apply_jina_quantization_if_needed(model: SentenceTransformer, model_id: str) -> SentenceTransformer:
        if model_id != JINA_EMBEDDING_MODEL or jina_quantization == "default":
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
                    print("[+] Quantizacao Jina aplicada: dynamic-int8 (CPU).")
                    return model
            print("[AVISO] Nao foi possivel localizar auto_model para aplicar dynamic-int8; usando modelo padrao.")
            return model
        except Exception as quant_error:
            print(f"[AVISO] Falha ao aplicar dynamic-int8 ({quant_error}); usando modelo padrao.")
            return model

    try:
        model = _load_from_local_dir(selection.model_id)
        model = _apply_jina_quantization_if_needed(model, selection.model_id)
        print("[+] Modelo carregado com sucesso.")
        return model
    except Exception as first_error:
        if selection.model_id == fallback_model_id:
            raise RuntimeError(
                f"Falha ao carregar o modelo fallback '{fallback_model_id}': {first_error}"
            ) from first_error

        print(
            f"[!] Falha ao carregar '{selection.model_id}': {first_error}\n"
            f"    Tentando fallback de carregamento: {fallback_model_id}"
        )
        fallback_selection = download_model_with_fallback(
            preferred_model_id=fallback_model_id,
            fallback_model_id=fallback_model_id,
            local_dir=model_base_dir,
        )
        selected_model_dir = fallback_selection.local_dir
        print(
            f"[+] Modelo selecionado: {fallback_selection.model_id} "
            f"(provider={fallback_selection.provider}, path={selected_model_dir})"
        )
        model = _load_from_local_dir(fallback_selection.model_id)
        model = _apply_jina_quantization_if_needed(model, fallback_selection.model_id)
        print("[+] Modelo fallback carregado com sucesso.")
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
    parser.add_argument(
        "--embedding-model",
        choices=["jina", "bge"],
        help="Modelo de embeddings: 'jina' (codigo) ou 'bge' (conteudo misto).",
    )
    parser.add_argument(
        "--jina-quantization",
        choices=["default", "dynamic-int8"],
        help="Quantizacao para Jina: 'default' (mais qualidade) ou 'dynamic-int8' (mais velocidade).",
    )
    args = parser.parse_args()

    root_path = Path(args.project_path).resolve()
    if not root_path.is_dir():
        print(f"[ERRO] Caminho não existe ou não é um diretório: {root_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RAG Indexer — Projeto: {root_path}")
    print(f"{'='*60}\n")

    model_choice, jina_quantization = resolve_embedding_config(
        args.embedding_model,
        args.jina_quantization,
    )
    print(
        f"[CONFIG] Modelo escolhido: {model_choice} "
        f"({_resolve_model_id(model_choice)})"
    )
    if model_choice == "jina":
        print(f"[CONFIG] Quantizacao Jina: {jina_quantization}")
    else:
        print("[CONFIG] Quantizacao Jina: nao aplicavel (modelo BGE selecionado)")

    # Inicializa componentes
    client = connect_to_chroma()
    model = load_embedding_model(model_choice, jina_quantization)
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
