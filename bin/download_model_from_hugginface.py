#!/usr/bin/env python3
from __future__ import annotations

"""
download_model_from_hugginface.py

Camada de download de modelos com prioridade de provedores e fallback de modelo.
Fluxo padrão:
1) tenta baixar o modelo preferido via Hugging Face;
2) se falhar, tenta provedores alternativos (quando disponíveis);
3) se o modelo preferido falhar em todos os provedores, tenta modelo fallback.
"""

from dataclasses import dataclass
import getpass
import os
from pathlib import Path
import shutil
import sys
from typing import Protocol


class ModelDownloadStrategy(Protocol):
    name: str

    def download(self, model_id: str, local_dir: Path) -> None:
        """Baixa model_id para local_dir ou levanta exceção."""


class HuggingFaceDownloadStrategy:
    name = "huggingface"

    def download(self, model_id: str, local_dir: Path) -> None:
        from huggingface_hub import snapshot_download

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        _download_with_hf_token_recovery(
            repo_id=model_id,
            local_dir=local_dir,
            hf_token=hf_token,
            snapshot_download=snapshot_download,
        )


@dataclass(frozen=True)
class DownloadSelection:
    model_id: str
    provider: str
    local_dir: Path


def _load_optional_strategies() -> list[ModelDownloadStrategy]:
    strategies: list[ModelDownloadStrategy] = []

    try:
        from download_model_from_modelscope import ModelScopeDownloadStrategy

        strategies.append(ModelScopeDownloadStrategy())
    except Exception:
        # Provider opcional: ignora se não estiver disponível no ambiente.
        pass

    return strategies


def build_default_strategies() -> list[ModelDownloadStrategy]:
    """Factory simples: ordem de prioridade de provedores de download."""
    return [HuggingFaceDownloadStrategy(), *_load_optional_strategies()]


_MODEL_READY_MARKER = ".download_complete"


def _prepare_destination(local_dir: Path, *, clean: bool) -> None:
    if clean and local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)


def _model_cache_dir(base_dir: Path, model_id: str) -> Path:
    # Evita colisão de nomes e mantém diretório seguro em qualquer SO.
    safe_name = model_id.replace("/", "__").replace(":", "_")
    return base_dir / safe_name


def _cache_ready(local_dir: Path) -> bool:
    marker = local_dir / _MODEL_READY_MARKER
    if not marker.exists() or not local_dir.exists():
        return False
    return any(p.name != _MODEL_READY_MARKER for p in local_dir.iterdir())


def _mark_cache_ready(local_dir: Path) -> None:
    (local_dir / _MODEL_READY_MARKER).write_text("ok\n", encoding="utf-8")


def _status_code_from_error(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _is_invalid_hf_token_error(exc: Exception) -> bool:
    message = str(exc).lower()
    status_code = _status_code_from_error(exc)
    token_keywords = ("invalid token", "token is invalid", "unauthorized", "401")
    if status_code == 401:
        return True
    return any(keyword in message for keyword in token_keywords)


def _prompt_recover_invalid_hf_token() -> tuple[str, str | None]:
    if not sys.stdin.isatty():
        return ("no-token", None)

    while True:
        print(
            "[!] O token do HuggingFace parece inválido. Escolha: "
            "[1] informar novo token, [2] continuar sem token.",
            file=sys.stderr,
        )
        answer = input("> Escolha [1/2]: ").strip().lower()
        if answer in {"1", "novo", "new"}:
            new_token = getpass.getpass("Cole o novo HF_TOKEN: ").strip()
            if new_token:
                return ("new-token", new_token)
            print("[!] Token vazio. Tente novamente.", file=sys.stderr)
            continue
        if answer in {"2", "", "sem", "no"}:
            return ("no-token", None)
        print("[!] Opção inválida. Digite 1 ou 2.", file=sys.stderr)


def _download_with_hf_token_recovery(
    *,
    repo_id: str,
    local_dir: Path,
    hf_token: str | None,
    snapshot_download,
) -> None:
    attempt_token = hf_token

    while True:
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=attempt_token,
            )
            if attempt_token:
                os.environ["HF_TOKEN"] = attempt_token
            else:
                os.environ.pop("HF_TOKEN", None)
            return
        except Exception as exc:
            if attempt_token and _is_invalid_hf_token_error(exc):
                print(
                    "[!] Falha de autenticação no HuggingFace com o token atual. "
                    "Você pode informar outro token ou seguir sem token.",
                    file=sys.stderr,
                )
                action, replacement = _prompt_recover_invalid_hf_token()
                if action == "new-token" and replacement:
                    attempt_token = replacement
                    continue
                attempt_token = None
                continue
            raise


def download_model_with_fallback(
    preferred_model_id: str,
    fallback_model_id: str,
    local_dir: Path,
    strategies: list[ModelDownloadStrategy] | None = None,
) -> DownloadSelection:
    """
    Tenta baixar `preferred_model_id`; se falhar em todos os provedores,
    tenta `fallback_model_id`.
    """
    base_dir = local_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    providers = strategies or build_default_strategies()
    errors: list[str] = []

    for model_id in (preferred_model_id, fallback_model_id):
        model_local_dir = _model_cache_dir(base_dir, model_id)
        if _cache_ready(model_local_dir):
            return DownloadSelection(
                model_id=model_id,
                provider="local-cache",
                local_dir=model_local_dir,
            )

        for strategy in providers:
            try:
                print(
                    f"[+] Iniciando download do modelo '{model_id}' via {strategy.name} em: {model_local_dir}",
                    file=sys.stderr,
                )
                _prepare_destination(model_local_dir, clean=True)
                strategy.download(model_id=model_id, local_dir=model_local_dir)
                _mark_cache_ready(model_local_dir)
                return DownloadSelection(
                    model_id=model_id,
                    provider=strategy.name,
                    local_dir=model_local_dir,
                )
            except Exception as exc:
                errors.append(f"{strategy.name}:{model_id}: {exc}")

    raise RuntimeError(
        "Falha no download dos modelos em todos os provedores configurados. "
        + " | ".join(errors)
    )
