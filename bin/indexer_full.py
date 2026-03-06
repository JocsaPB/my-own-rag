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
import gc
import json
from time import perf_counter, time
from collections.abc import Iterator
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

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


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(min_value, default)
    try:
        return max(min_value, int(raw))
    except ValueError:
        return max(min_value, default)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_CODE_JINA = "code_vectors_jina"
COLLECTION_DOC_BGE = "doc_vectors_bge"

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

# Tamanho máximo de arquivo (evita indexar arquivos enormes gerados)
MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB

# Parâmetros do splitter e batch (perfil low-memory por padrão).
CHUNK_SIZE = _env_int("MCP_CHUNK_SIZE", 3000, min_value=256)
CHUNK_OVERLAP = min(CHUNK_SIZE - 1, _env_int("MCP_CHUNK_OVERLAP", 400, min_value=0))
EMBEDDING_BATCH_SIZE = _env_int("MCP_EMBEDDING_BATCH_SIZE", 4, min_value=1)
DEFAULT_PERF_PROFILE = "autotune"
INDEXER_CONFIG_PATH = Path(
    os.environ.get("MCP_INDEXER_CONFIG_FILE", str(Path.home() / ".rag_db" / "indexer_tuning.json"))
).expanduser()

# Modelo de embeddings (roda na CPU)
JINA_V3_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
JINA_V2_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL_CHOICE = "jina"
DEFAULT_JINA_QUANTIZATION = "dynamic-int8"
MODEL_CACHE_BASE_DIR = Path(
    os.environ.get("MCP_MODEL_DIR", str(Path.home() / ".cache" / "my-custom-rag-python" / "models"))
).expanduser()
JINA_RECOMMENDED_RAM_GB_DEFAULT = 64
JINA_RECOMMENDED_RAM_GB_DYNAMIC_INT8 = 48
JINA_RECOMMENDED_SWAP_GB = 16
JINA_MIN_AVAILABLE_RAM_GB_HINT = 12


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_memory_related_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    memory_markers = (
        "out of memory",
        "oom",
        "cannot allocate memory",
        "std::bad_alloc",
        "bad alloc",
        "insufficient memory",
    )
    return any(marker in msg for marker in memory_markers)


@dataclass(frozen=True)
class IndexTarget:
    model_choice: str
    collection_name: str
    label: str


def _resolve_model_id(model_choice: str) -> str:
    if model_choice == "jina":
        return JINA_V3_EMBEDDING_MODEL
    if model_choice == "jina-v2":
        return JINA_V2_EMBEDDING_MODEL
    if model_choice == "bge":
        return BGE_EMBEDDING_MODEL
    raise ValueError(f"Modelo não suportado: {model_choice}")


def _resolve_fallback_model_id(model_choice: str) -> str:
    return BGE_EMBEDDING_MODEL


def _describe_embedding_choice(model_choice: str) -> str:
    if model_choice == "jina":
        return f"jina ({JINA_V3_EMBEDDING_MODEL})"
    if model_choice == "bge":
        return f"bge ({BGE_EMBEDDING_MODEL})"
    if model_choice == "hybrid":
        return f"hybrid ({JINA_V2_EMBEDDING_MODEL} + {BGE_EMBEDDING_MODEL})"
    return model_choice


def _resolve_index_targets(model_choice: str) -> list[IndexTarget]:
    if model_choice == "jina":
        return [
            IndexTarget(
                model_choice="jina",
                collection_name=COLLECTION_CODE_JINA,
                label="Code/Jina",
            )
        ]
    if model_choice == "bge":
        return [
            IndexTarget(
                model_choice="bge",
                collection_name=COLLECTION_DOC_BGE,
                label="Doc/BGE",
            )
        ]
    if model_choice == "hybrid":
        return [
            IndexTarget(
                model_choice="jina-v2",
                collection_name=COLLECTION_CODE_JINA,
                label="Code/Jina v2",
            ),
            IndexTarget(
                model_choice="bge",
                collection_name=COLLECTION_DOC_BGE,
                label="Doc/BGE",
            ),
        ]
    raise ValueError(f"Modelo não suportado: {model_choice}")


def _classify_file_targets(filepath: Path, model_choice: str) -> set[str]:
    if model_choice != "hybrid":
        return {model_choice}

    suffix = filepath.suffix.lower()
    is_code = suffix in CODE_EXTENSIONS
    is_doc = suffix in DOC_EXTENSIONS

    if is_code and not is_doc:
        return {"jina-v2"}
    if is_doc and not is_code:
        return {"bge"}

    # Extensão desconhecida/ambígua: indexa nos dois ramos para manter recall.
    return {"jina-v2", "bge"}


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
    persisted_config: dict[str, object] | None = None,
) -> tuple[str, str]:
    persisted_config = persisted_config or {}
    model_choice_from_config = persisted_config.get("embedding_model")
    model_choice = model_choice_arg or os.environ.get("MCP_EMBEDDING_MODEL")
    if not model_choice and isinstance(model_choice_from_config, str):
        model_choice = model_choice_from_config
    if model_choice:
        model_choice = model_choice.strip().lower()
    model_choice = _pick_with_prompt(
        current_value=model_choice,
        default_value=DEFAULT_EMBEDDING_MODEL_CHOICE,
        title="Escolha do modelo de embeddings",
        options=[
            (
                "jina",
                f"jina ({JINA_V3_EMBEDDING_MODEL}) - foco em código.",
            ),
            (
                "bge",
                f"bge ({BGE_EMBEDDING_MODEL}) - conteúdo misto.",
            ),
            (
                "hybrid",
                f"hybrid (Jina v2 {JINA_V2_EMBEDDING_MODEL} + BGE) - duas coleções.",
            ),
        ],
    )
    if model_choice not in {"jina", "bge", "hybrid"}:
        print(f"[AVISO] MCP_EMBEDDING_MODEL inválido '{model_choice}'. Usando '{DEFAULT_EMBEDDING_MODEL_CHOICE}'.")
        model_choice = DEFAULT_EMBEDDING_MODEL_CHOICE

    quantization_from_config = persisted_config.get("jina_quantization")
    jina_quantization = jina_quantization_arg or os.environ.get("MCP_JINA_QUANTIZATION")
    if not jina_quantization and isinstance(quantization_from_config, str):
        jina_quantization = quantization_from_config
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


def load_indexer_tuning_config(force_reconfigure: bool) -> dict[str, object]:
    if force_reconfigure:
        return {}
    try:
        if not INDEXER_CONFIG_PATH.exists():
            return {}
        data = json.loads(INDEXER_CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_indexer_tuning_config(config: dict[str, object]) -> None:
    try:
        INDEXER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **config,
            "updated_at": int(time()),
        }
        INDEXER_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[CONFIG] Configuração persistida em: {INDEXER_CONFIG_PATH}")
    except Exception as e:
        print(f"[AVISO] Não foi possível persistir configuração em {INDEXER_CONFIG_PATH}: {e}")


def resolve_perf_profile(perf_profile_arg: str | None, persisted_config: dict[str, object]) -> str:
    profile_from_config = persisted_config.get("perf_profile")
    profile = perf_profile_arg or os.environ.get("MCP_PERF_PROFILE")
    if not profile and isinstance(profile_from_config, str):
        profile = profile_from_config
    if profile:
        profile = profile.strip().lower()

    profile = _pick_with_prompt(
        current_value=profile,
        default_value=DEFAULT_PERF_PROFILE,
        title="Perfil de performance da indexação",
        options=[
            (
                "autotune",
                "autotune - equilíbrio (recomendado).",
            ),
            (
                "max-performance",
                "max-performance - máximo throughput (mais RAM).",
            ),
        ],
    )
    if profile not in {"autotune", "max-performance"}:
        print(f"[AVISO] Perfil inválido '{profile}'. Usando '{DEFAULT_PERF_PROFILE}'.")
        profile = DEFAULT_PERF_PROFILE
    return profile


def _parse_config_int(config: dict[str, object], key: str) -> int | None:
    raw = config.get(key)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def _read_meminfo_gib() -> tuple[float | None, float | None, float | None]:
    """Retorna (mem_total, mem_available, swap_total) em GiB, quando disponível."""
    mem_total_kib: int | None = None
    mem_available_kib: int | None = None
    swap_total_kib: int | None = None

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kib = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
            elif line.startswith("SwapTotal:"):
                swap_total_kib = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None, None, None

    to_gib = lambda kib: (kib / (1024 * 1024)) if kib is not None else None
    return to_gib(mem_total_kib), to_gib(mem_available_kib), to_gib(swap_total_kib)


def warn_if_jina_memory_risk(model_choice: str, jina_quantization: str) -> None:
    """Mostra aviso de risco de OOM para o modelo Jina em máquinas com pouca memória."""
    if model_choice not in {"jina", "hybrid"}:
        return

    mem_total_gib, mem_available_gib, swap_total_gib = _read_meminfo_gib()
    if mem_total_gib is None:
        return

    recommended_ram_gib = (
        JINA_RECOMMENDED_RAM_GB_DEFAULT
        if jina_quantization == "default"
        else JINA_RECOMMENDED_RAM_GB_DYNAMIC_INT8
    )

    reasons: list[str] = []
    if mem_total_gib < recommended_ram_gib:
        reasons.append(
            f"RAM total detectada: {mem_total_gib:.1f} GiB (recomendado >= {recommended_ram_gib} GiB para Jina/{jina_quantization})."
        )
    if swap_total_gib is not None and swap_total_gib < JINA_RECOMMENDED_SWAP_GB:
        reasons.append(
            f"Swap detectada: {swap_total_gib:.1f} GiB (recomendado >= {JINA_RECOMMENDED_SWAP_GB} GiB)."
        )
    if mem_available_gib is not None and mem_available_gib < JINA_MIN_AVAILABLE_RAM_GB_HINT:
        reasons.append(
            f"RAM livre atual: {mem_available_gib:.1f} GiB (baixo para a carga inicial do Jina)."
        )

    if not reasons:
        return

    print("[AVISO] Alto risco de OOM com Jina nesta máquina/carga.")
    for reason in reasons:
        print(f"        - {reason}")
    print("        - Se ocorrer 'Killed' (exit 137), use BGE: --embedding-model bge")
    print("        - Ou rode o Jina em máquina com mais RAM/swap e menos processos concorrentes.")


@dataclass(frozen=True)
class RuntimeIndexingParams:
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    reasons: list[str]


def _resolve_max_performance_params(
    *,
    chunk_size_locked: bool,
    chunk_overlap_locked: bool,
    batch_size_locked: bool,
    chunk_size: int,
    chunk_overlap: int,
    embedding_batch_size: int,
) -> RuntimeIndexingParams:
    mem_total_gib, mem_available_gib, _ = _read_meminfo_gib()
    reasons = [
        "Perfil selecionado: max-performance.",
        "Modo pode elevar consideravelmente o consumo de memória e causar encerramento por OOM (exit 137).",
    ]

    tuned_chunk_size = chunk_size
    tuned_chunk_overlap = chunk_overlap
    tuned_batch = embedding_batch_size

    if not chunk_size_locked:
        if mem_total_gib is not None and mem_total_gib >= 64 and (mem_available_gib or 0) >= 16:
            tuned_chunk_size = 7000
        else:
            tuned_chunk_size = 6000
        reasons.append(f"chunk_size ajustado para {tuned_chunk_size} no perfil max-performance.")

    if not chunk_overlap_locked:
        tuned_chunk_overlap = min(tuned_chunk_size - 1, max(300, int(tuned_chunk_size * 0.15)))
        reasons.append(f"chunk_overlap ajustado para {tuned_chunk_overlap}.")

    if not batch_size_locked:
        if mem_total_gib is not None and mem_total_gib >= 64 and (mem_available_gib or 0) >= 16:
            tuned_batch = 24
        elif mem_total_gib is not None and mem_total_gib >= 32:
            tuned_batch = 16
        else:
            tuned_batch = 12
        reasons.append(f"embedding_batch_size ajustado para {tuned_batch}.")

    return RuntimeIndexingParams(
        chunk_size=tuned_chunk_size,
        chunk_overlap=tuned_chunk_overlap,
        embedding_batch_size=max(1, tuned_batch),
        reasons=reasons,
    )


def _resolve_autotuned_params(
    *,
    model: SentenceTransformer,
    chunk_size_locked: bool,
    chunk_overlap_locked: bool,
    batch_size_locked: bool,
    chunk_size: int,
    chunk_overlap: int,
    embedding_batch_size: int,
) -> RuntimeIndexingParams:
    reasons: list[str] = ["Perfil selecionado: autotune (custo-benefício)."]
    verbose_autotune = _env_bool("MCP_AUTOTUNE_VERBOSE", default=False)

    try:
        import psutil  # type: ignore
    except Exception:
        reasons.append("psutil indisponível; mantendo parâmetros atuais sem benchmark.")
        return RuntimeIndexingParams(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_batch_size=embedding_batch_size,
            reasons=reasons,
        )

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    mem_total_gib = vm.total / (1024**3)
    mem_available_gib = vm.available / (1024**3)
    swap_total_gib = swap.total / (1024**3)

    target_ram_percent = _clamp(
        float(os.environ.get("MCP_AUTOTUNE_TARGET_RAM_PERCENT", "68")),
        60.0,
        75.0,
    )
    if mem_available_gib < 6 or swap_total_gib < 4:
        target_ram_percent = min(target_ram_percent, 63.0)
    reasons.append(
        f"Memória detectada: total={mem_total_gib:.1f} GiB, livre={mem_available_gib:.1f} GiB, "
        f"swap={swap_total_gib:.1f} GiB, alvo={target_ram_percent:.1f}%."
    )

    tuned_chunk_size = chunk_size
    tuned_chunk_overlap = chunk_overlap
    tuned_batch = embedding_batch_size

    if not chunk_size_locked:
        if mem_total_gib < 8 or mem_available_gib < 3:
            tuned_chunk_size = 1800
        elif mem_total_gib < 16 or mem_available_gib < 6:
            tuned_chunk_size = 2400
        elif mem_total_gib < 32 or mem_available_gib < 12:
            tuned_chunk_size = 3200
        else:
            tuned_chunk_size = 4200
        reasons.append(f"chunk_size autotunado para {tuned_chunk_size}.")

    if not chunk_overlap_locked:
        tuned_chunk_overlap = min(tuned_chunk_size - 1, max(120, int(tuned_chunk_size * 0.15)))
        reasons.append(f"chunk_overlap autotunado para {tuned_chunk_overlap}.")

    if not batch_size_locked:
        max_candidate = 16
        if mem_total_gib < 8 or mem_available_gib < 3 or swap_total_gib < 2:
            max_candidate = 2
        elif mem_total_gib < 16 or mem_available_gib < 6:
            max_candidate = 4
        elif mem_total_gib < 32 or mem_available_gib < 10:
            max_candidate = 8

        candidates = [2, 4, 6, 8, 12, 16]
        candidates = [c for c in candidates if c <= max_candidate]
        if not candidates:
            candidates = [2]

        process = psutil.Process()
        sample_size = min(max(512, tuned_chunk_size), 3000)
        sample_text = ("# autotune-sample\n" + ("x" * sample_size))

        best_batch = candidates[0]
        best_score = -1.0
        best_memory_pct = 100.0
        selected_benchmark_line: str | None = None
        benchmark_lines: list[str] = []

        # Warmup curto para estabilizar cache interno.
        try:
            _ = model.encode([sample_text], show_progress_bar=False, batch_size=1)
        except Exception:
            pass

        for candidate in candidates:
            docs = [sample_text] * candidate
            gc.collect()
            before_vm = psutil.virtual_memory().percent
            before_rss = process.memory_info().rss / (1024**2)
            started = perf_counter()
            try:
                embeddings = model.encode(
                    docs,
                    show_progress_bar=False,
                    batch_size=candidate,
                )
            except Exception as e:
                benchmark_lines.append(f"batch={candidate}: erro ({e})")
                continue

            elapsed = max(perf_counter() - started, 1e-6)
            after_vm = psutil.virtual_memory().percent
            after_rss = process.memory_info().rss / (1024**2)
            del embeddings
            gc.collect()

            throughput = candidate / elapsed
            safe = after_vm <= (target_ram_percent + 3.0)
            benchmark_lines.append(
                f"batch={candidate}: {throughput:.2f} itens/s, vm={after_vm:.1f}%, rss_delta={after_rss - before_rss:+.1f} MiB"
            )

            if safe and throughput > best_score:
                best_score = throughput
                best_batch = candidate
                best_memory_pct = after_vm
                selected_benchmark_line = benchmark_lines[-1]
            elif best_score < 0 and after_vm < best_memory_pct:
                # Se nenhum candidato ficou "safe", escolhe o menos agressivo em memória.
                best_batch = candidate
                best_memory_pct = after_vm
                selected_benchmark_line = benchmark_lines[-1]

            # Se já passou muito do limite, evita tentar batches maiores.
            if after_vm > target_ram_percent + 8.0:
                break

            # Evita escolher candidato que já começou acima do limite.
            if before_vm > target_ram_percent + 5.0:
                break

        tuned_batch = max(1, best_batch)
        if verbose_autotune:
            reasons.extend(benchmark_lines)
        elif selected_benchmark_line:
            reasons.append(f"Micro-benchmark: {selected_benchmark_line}")
        reasons.append(
            f"embedding_batch_size autotunado para {tuned_batch} (alvo de memória: {target_ram_percent:.1f}%)."
        )

    return RuntimeIndexingParams(
        chunk_size=tuned_chunk_size,
        chunk_overlap=tuned_chunk_overlap,
        embedding_batch_size=max(1, tuned_batch),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Retorna o splitter compartilhado com as configurações padrão do projeto."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
        if model_id != JINA_V3_EMBEDDING_MODEL or jina_quantization == "default":
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
                print(
                    "[AVISO] Nenhuma camada ParametrizedLinear encontrada para dynamic-int8; usando modelo padrao."
                )
                return model

            print(f"[+] Quantizacao Jina aplicada: dynamic-int8 (CPU, {quantized_layers} camadas).")
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


def scan_files(root_path: Path) -> Iterator[Path]:
    """
    Varre recursivamente o diretório raiz, retornando em streaming
    os arquivos de texto relevantes para indexação.
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Remove dirs ignorados in-place para que os.walk não desça neles
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not d.startswith(".")
        ]
        dirnames.sort()

        for filename in sorted(filenames):
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

            yield filepath


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
        # Pede somente IDs para evitar materializar docs/metadata na memória.
        results = collection.get(where={"file_path": file_path}, include=[])
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
    embedding_batch_size: int,
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

    relative_path = str(filepath.relative_to(root_path))
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
            batch_size=embedding_batch_size,
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
        gc.collect()

    for i, chunk in enumerate(chunks):
        batch_ids.append(make_chunk_id(abs_path, i))
        batch_docs.append(chunk)
        batch_metadatas.append(
            {
                "file_path": abs_path,
                "chunk_index": i,
                "file_name": filepath.name,
                # Caminho relativo à raiz do projeto para exibição compacta
                "relative_path": relative_path,
            }
        )
        if len(batch_ids) >= embedding_batch_size:
            _flush_batch()

    _flush_batch()
    return inserted_chunks


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
        choices=["jina", "bge", "hybrid"],
        help=(
            "Modelo de embeddings: 'jina' (codigo), "
            "'bge' (conteudo misto) ou 'hybrid' (duas colecoes: Jina v2 + BGE)."
        ),
    )
    parser.add_argument(
        "--jina-quantization",
        choices=["default", "dynamic-int8"],
        help="Quantizacao para Jina: 'default' (mais qualidade) ou 'dynamic-int8' (mais velocidade).",
    )
    parser.add_argument(
        "--perf-profile",
        choices=["autotune", "max-performance"],
        help=(
            "Perfil de performance da indexação: "
            "'autotune' (custo-benefício) ou 'max-performance' (mais throughput, maior uso de RAM)."
        ),
    )
    args = parser.parse_args()

    root_path = Path(args.project_path).resolve()
    if not root_path.is_dir():
        print(f"[ERRO] Caminho não existe ou não é um diretório: {root_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RAG Indexer — Projeto: {root_path}")
    print(f"{'='*60}\n")
    index_started_at = datetime.now()
    print(f"[INFO] Início: {index_started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    force_model_reconfigure = _env_bool("MCP_FORCE_MODEL_RECONFIG", default=False)
    persisted_config = load_indexer_tuning_config(force_model_reconfigure)
    model_choice, jina_quantization = resolve_embedding_config(
        args.embedding_model,
        args.jina_quantization,
        persisted_config=persisted_config,
    )
    perf_profile = resolve_perf_profile(args.perf_profile, persisted_config)

    chunk_size_locked = "MCP_CHUNK_SIZE" in os.environ
    chunk_overlap_locked = "MCP_CHUNK_OVERLAP" in os.environ
    batch_size_locked = "MCP_EMBEDDING_BATCH_SIZE" in os.environ

    persisted_chunk_size = _parse_config_int(persisted_config, "chunk_size")
    persisted_chunk_overlap = _parse_config_int(persisted_config, "chunk_overlap")
    persisted_batch_size = _parse_config_int(persisted_config, "embedding_batch_size")

    effective_chunk_size = CHUNK_SIZE
    if not chunk_size_locked and persisted_chunk_size is not None:
        effective_chunk_size = max(256, persisted_chunk_size)

    effective_chunk_overlap = CHUNK_OVERLAP
    if not chunk_overlap_locked and persisted_chunk_overlap is not None:
        effective_chunk_overlap = max(0, min(effective_chunk_size - 1, persisted_chunk_overlap))

    effective_batch_size = EMBEDDING_BATCH_SIZE
    if not batch_size_locked and persisted_batch_size is not None:
        effective_batch_size = max(1, persisted_batch_size)

    print(
        f"[CONFIG] Modelo escolhido: {model_choice} "
        f"({_describe_embedding_choice(model_choice)})"
    )
    if model_choice == "jina":
        print(f"[CONFIG] Quantizacao Jina: {jina_quantization}")
    elif model_choice == "hybrid":
        print("[CONFIG] Quantizacao Jina: nao aplicavel no hybrid (Jina v2 + BGE)")
    else:
        print("[CONFIG] Quantizacao Jina: nao aplicavel (modelo BGE selecionado)")
    print(f"[CONFIG] Perfil de performance: {perf_profile}")
    if perf_profile == "max-performance":
        print(
            "[AVISO] Este modo pode elevar consideravelmente o consumo de memória "
            "e causar encerramento por OOM (exit 137)."
        )
    warn_if_jina_memory_risk(model_choice, jina_quantization)

    # Inicializa componentes
    client = connect_to_chroma()
    targets = _resolve_index_targets(model_choice)

    # Obtém ou recria as coleções envolvidas.
    collections: dict[str, chromadb.Collection] = {}
    for target in targets:
        if args.clear:
            try:
                client.delete_collection(target.collection_name)
                print(f"[!] Coleção '{target.collection_name}' removida para reindexação limpa.")
            except Exception:
                pass
        collections[target.collection_name] = client.get_or_create_collection(
            name=target.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # Carrega modelos de forma lazy e reaproveita por target.
    loaded_models: dict[str, SentenceTransformer] = {}
    total_chunks = 0
    errors = 0
    files_scanned = 0
    files_processed_total = 0
    chunks_by_collection = {target.collection_name: 0 for target in targets}
    files_by_collection = {target.collection_name: 0 for target in targets}
    files_eligible_by_collection = {target.collection_name: 0 for target in targets}
    errors_by_collection = {target.collection_name: 0 for target in targets}
    error_samples_by_collection: dict[str, list[str]] = {target.collection_name: [] for target in targets}
    target_by_model = {target.model_choice: target for target in targets}

    # Carrega o primeiro modelo antes para autotune com micro-benchmark.
    primary_target = targets[0]
    primary_quantization = jina_quantization if primary_target.model_choice == "jina" else "default"
    loaded_models[primary_target.model_choice] = load_embedding_model(primary_target.model_choice, primary_quantization)
    primary_model = loaded_models[primary_target.model_choice]

    if perf_profile == "autotune":
        tuned = _resolve_autotuned_params(
            model=primary_model,
            chunk_size_locked=chunk_size_locked,
            chunk_overlap_locked=chunk_overlap_locked,
            batch_size_locked=batch_size_locked,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            embedding_batch_size=effective_batch_size,
        )
    else:
        tuned = _resolve_max_performance_params(
            chunk_size_locked=chunk_size_locked,
            chunk_overlap_locked=chunk_overlap_locked,
            batch_size_locked=batch_size_locked,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            embedding_batch_size=effective_batch_size,
        )

    effective_chunk_size = max(256, tuned.chunk_size)
    effective_chunk_overlap = max(0, min(effective_chunk_size - 1, tuned.chunk_overlap))
    effective_batch_size = max(1, tuned.embedding_batch_size)

    for reason in tuned.reasons:
        print(f"[CONFIG] {reason}")

    print(
        f"[CONFIG] Parâmetros finais: "
        f"chunk_size={effective_chunk_size}, chunk_overlap={effective_chunk_overlap}, "
        f"embedding_batch={effective_batch_size}"
    )

    save_indexer_tuning_config(
        {
            "embedding_model": model_choice,
            "jina_quantization": jina_quantization,
            "perf_profile": perf_profile,
            "chunk_size": effective_chunk_size,
            "chunk_overlap": effective_chunk_overlap,
            "embedding_batch_size": effective_batch_size,
        }
    )

    splitter = get_text_splitter(effective_chunk_size, effective_chunk_overlap)

    print(f"\n[+] Varrendo e indexando arquivos em: {root_path}")
    files = list(scan_files(root_path))
    files_scanned = len(files)
    if files_scanned == 0:
        print("[AVISO] Nenhum arquivo encontrado. Verifique o caminho e os filtros.")
        sys.exit(0)

    print(f"[+] {files_scanned} arquivo(s) elegível(is) para indexação.")
    with tqdm(
        total=files_scanned,
        desc="Indexando",
        unit="arquivo",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]",
    ) as pbar:
        for filepath in files:
            target_models = _classify_file_targets(filepath, model_choice)

            for target_model in target_models:
                target = target_by_model.get(target_model)
                if target is None:
                    continue

                if target.model_choice not in loaded_models:
                    target_quantization = jina_quantization if target.model_choice == "jina" else "default"
                    try:
                        loaded_models[target.model_choice] = load_embedding_model(
                            target.model_choice,
                            target_quantization,
                        )
                    except Exception as load_error:
                        # Em hybrid, pode faltar RAM ao manter dois modelos grandes simultaneamente.
                        if model_choice == "hybrid" and loaded_models and _is_memory_related_error(load_error):
                            print(
                                "[AVISO] Falha ao carregar modelo adicional no hybrid por memória. "
                                "Liberando modelo anterior e tentando novamente."
                            )
                            loaded_models.clear()
                            gc.collect()
                            loaded_models[target.model_choice] = load_embedding_model(
                                target.model_choice,
                                target_quantization,
                            )
                        else:
                            raise

                model = loaded_models[target.model_choice]
                collection = collections[target.collection_name]
                files_eligible_by_collection[target.collection_name] += 1

                while True:
                    try:
                        n_chunks = index_file(
                            filepath,
                            collection,
                            model,
                            splitter,
                            root_path,
                            embedding_batch_size=effective_batch_size,
                        )
                        total_chunks += n_chunks
                        files_processed_total += 1
                        chunks_by_collection[target.collection_name] += n_chunks
                        files_by_collection[target.collection_name] += 1
                        break
                    except Exception as e:
                        # Fallback automático para evitar quebra total em máquinas no limite de RAM.
                        if (
                            not batch_size_locked
                            and effective_batch_size > 1
                            and _is_memory_related_error(e)
                        ):
                            new_batch = max(1, effective_batch_size // 2)
                            if new_batch < effective_batch_size:
                                tqdm.write(
                                    f"  [AJUSTE] Memória alta em {target.label}. "
                                    f"Batch reduzido {effective_batch_size} -> {new_batch}."
                                )
                                effective_batch_size = new_batch
                                gc.collect()
                                continue

                        errors += 1
                        errors_by_collection[target.collection_name] += 1
                        if len(error_samples_by_collection[target.collection_name]) < 3:
                            error_samples_by_collection[target.collection_name].append(f"{filepath.name}: {e}")
                        tqdm.write(f"  [ERRO] {filepath} [{target.label}]: {e}")
                        break

            pbar.set_postfix({"chunks": total_chunks, "atual": filepath.name[:20]})
            pbar.update(1)

    for target in targets:
        collection_name = target.collection_name
        eligible = files_eligible_by_collection[collection_name]
        processed = files_by_collection[collection_name]
        target_errors = errors_by_collection[collection_name]

        if eligible == 0:
            print(f"[AVISO] Nenhum arquivo elegível para {target.label}; etapa ignorada.")
        elif processed == 0 and target_errors > 0:
            print(
                f"[AVISO] {eligible} arquivo(s) elegível(is) para {target.label}, "
                "mas todos falharam."
            )

        if target_errors:
            print(f"[AVISO] {target_errors} erro(s) durante a indexação do target {target.label}.")
            for sample in error_samples_by_collection[collection_name]:
                print(f"        - {sample}")

    index_finished_at = datetime.now()
    elapsed_seconds = int((index_finished_at - index_started_at).total_seconds())
    elapsed_h = elapsed_seconds // 3600
    elapsed_m = (elapsed_seconds % 3600) // 60
    elapsed_s = elapsed_seconds % 60
    print(f"\n{'='*60}")
    print(f"  Indexação concluída!")
    print(f"  Início               : {index_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Fim                  : {index_finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duração              : {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}")
    print(f"  Arquivos varridos    : {files_scanned}")
    print(f"  Arquivos processados : {files_processed_total}")
    print(f"  Total de chunks      : {total_chunks}")
    print(f"  Erros                : {errors}")
    for target in targets:
        collection_name = target.collection_name
        print(
            f"  Coleção ChromaDB     : '{collection_name}' "
            f"(elegíveis={files_eligible_by_collection.get(collection_name, 0)}, "
            f"arquivos={files_by_collection.get(collection_name, 0)}, "
            f"chunks={chunks_by_collection.get(collection_name, 0)})"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print(
            "[ERRO] Falha de memória durante a indexação. "
            "Use --embedding-model bge ou execute o Jina em máquina com mais RAM/swap."
        )
        sys.exit(1)
