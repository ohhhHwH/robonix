"""BGE-M3 text embedder — 1024-dim bilingual dense vectors via ONNX.

Reuses ``memsearch``'s ONNX embedding provider (``OnnxEmbedding``,
``embedding_provider="onnx"``, model ``gpahal/bge-m3-onnx-int8``) so both
services share one implementation instead of drifting.  Falls back to the
deterministic ``_hash_embedding`` when onnxruntime / the model is missing, so
the service still boots (and ``vector_search`` still returns something,
just non-semantic) in a sandbox without the model.

The memory pipeline is synchronous (``remember``/``retrieve`` call
``VectorStore.encode``), so this calls ``OnnxEmbedding._encode`` directly —
the synchronous core that ``OnnxEmbedding.embed`` otherwise wraps in an
executor — rather than ``asyncio.run``, which would raise inside a running
event loop.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .embedding_config import EmbeddingModelConfig, _hash_embedding
from .onnx_compat import configure_onnxruntime

log = logging.getLogger("scribe_mem")


class BgeEmbedder:
    """Text → vector encoder backed by BGE-M3 ONNX (hash fallback)."""

    def __init__(self, config: EmbeddingModelConfig):
        self._config = config
        self._provider: Optional[object] = None
        self._try_load()

    def _try_load(self) -> None:
        """Load BGE-M3 via memsearch's ONNX provider; fall back to hash."""
        try:
            configure_onnxruntime()
            from memsearch.embeddings.onnx import OnnxEmbedding

            self._provider = OnnxEmbedding(
                model=self._config.model_name,
                batch_size=self._config.batch_size,
            )
            # The provider probes the real output dimension on construction;
            # keep the config in sync so _hash_embedding matches on fallback.
            if self._provider.dimension != self._config.dim:
                log.info(
                    "BgeEmbedder: actual dim %d (config said %d)",
                    self._provider.dimension,
                    self._config.dim,
                )
                self._config.dim = self._provider.dimension
            log.info(
                "BgeEmbedder: loaded %s (dim=%d)",
                self._config.model_name,
                self._config.dim,
            )
        except Exception as e:  # noqa: BLE001 — any load failure → hash
            log.warning(
                "BgeEmbedder: ONNX/BGE-M3 unavailable (%s); "
                "falling back to deterministic hash embedding",
                e,
            )
            self._provider = None

    @property
    def dim(self) -> int:
        return self._config.dim

    @property
    def is_semantic(self) -> bool:
        """True when the real BGE-M3 model is loaded (vs. hash fallback)."""
        return self._provider is not None

    def encode(self, text: str) -> List[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._provider is not None:
            try:
                # _encode is the synchronous core of memsearch's async embed().
                return self._provider._encode(texts)  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001 — runtime failure → hash
                log.warning("BgeEmbedder: encode failed (%s); hash fallback", e)
        return [_hash_embedding(t, dim=self._config.dim) for t in texts]
