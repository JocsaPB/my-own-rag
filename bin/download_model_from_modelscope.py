#!/usr/bin/env python3
from __future__ import annotations

"""
Provider opcional de download via ModelScope.
Usado apenas se o pacote `modelscope` estiver instalado.
"""

from pathlib import Path


class ModelScopeDownloadStrategy:
    name = "modelscope"

    def download(self, model_id: str, local_dir: Path) -> None:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "Pacote `modelscope` indisponível para provider alternativo"
            ) from exc

        snapshot_download(
            model_id=model_id,
            local_dir=str(local_dir),
        )
