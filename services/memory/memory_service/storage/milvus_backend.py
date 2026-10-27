"""Milvus Lite vector index — pure ANN, references back to MongoDB.

Collection ``memory_vectors`` holds only the vector plus a small scalar set
(``doc_id`` → Mongo ``_id``, ``kind``, ``node_type``, ``timestamp``).  MongoDB
is the single source of truth; this index is derived and rebuildable from it
(design doc §4).  ``doc_id`` is the primary key, so every write is an upsert.

Graceful degradation mirrors ``mongo_backend``: when ``pymilvus`` /
milvus-lite is absent, ``available`` is False and all methods no-op.  Writers
are already serialised by ``GraphStore``'s lock.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("scribe_mem")

COLLECTION = "memory_vectors"
DIM = 1024


def _resolve_uri(value: str) -> str:
    """Local filesystem path → absolute; remote endpoints passed through."""
    if "://" in value:
        return value
    if ":" in value and "/" not in value and not value.endswith(".db"):
        return value
    return str(Path(value).expanduser().resolve())


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class MilvusBackend:
    """Thin ``pymilvus.MilvusClient`` wrapper for the memory vector index."""

    def __init__(self, uri: str = "", dim: int = DIM):
        self._uri = uri or os.environ.get(
            "MEMGRAPH_MILVUS_URI", "./memory/robonix_memory.db"
        )
        self._dim = dim
        self._client: Optional[Any] = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            from pymilvus import MilvusClient  # noqa: F401
        except ImportError:
            log.warning(
                "milvus_backend: pymilvus not installed; vector index disabled"
            )
            return
        try:
            resolved = _resolve_uri(self._uri)
            if "://" not in resolved and not (
                ":" in resolved and "/" not in resolved and not resolved.endswith(".db")
            ):
                Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(uri=resolved)
            self._ensure_collection()
            self._available = True
            log.info("milvus_backend: opened %s", resolved)
        except Exception as e:  # noqa: BLE001 — any failure → disabled
            log.warning(
                "milvus_backend: Milvus unavailable at %s (%s); disabled",
                self._uri,
                e,
            )
            self._client = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_collection(self) -> None:
        if self._client.has_collection(COLLECTION):
            self._load()
            return

        from pymilvus import DataType

        schema = self._client.create_schema(enable_dynamic_field=False)
        schema.add_field(
            field_name="doc_id", datatype=DataType.VARCHAR,
            max_length=64, is_primary=True,
        )
        schema.add_field(
            field_name="kind", datatype=DataType.VARCHAR, max_length=16,
        )
        schema.add_field(
            field_name="node_type", datatype=DataType.VARCHAR, max_length=64,
        )
        schema.add_field(field_name="timestamp", datatype=DataType.INT64)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._dim,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="FLAT", metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=COLLECTION, schema=schema, index_params=index_params,
        )
        self._load()

    def _load(self) -> None:
        try:
            self._client.load_collection(COLLECTION)
        except TypeError:  # older pymilvus accepted a positional arg
            self._client.load_collection(collection_name=COLLECTION)

    # ── write ─────────────────────────────────────────────────────────

    def upsert(
        self,
        doc_id: str,
        vector: List[float],
        *,
        kind: str = "scene",
        node_type: str = "",
        timestamp: int = 0,
    ) -> None:
        if not self._available:
            return
        try:
            self._client.upsert(
                collection_name=COLLECTION,
                data=[{
                    "doc_id": str(doc_id),
                    "kind": kind,
                    "node_type": node_type or "",
                    "timestamp": int(timestamp or 0),
                    "vector": list(vector),
                }],
            )
        except Exception as e:  # noqa: BLE001 — Mongo is authoritative
            log.warning("milvus_backend: upsert %s failed: %s", doc_id, e)

    def delete(self, doc_id: str) -> None:
        if not self._available:
            return
        try:
            self._client.delete(collection_name=COLLECTION, ids=[str(doc_id)])
        except Exception as e:  # noqa: BLE001
            log.warning("milvus_backend: delete %s failed: %s", doc_id, e)

    # ── read ──────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        *,
        top_k: int = 5,
        kind: Optional[str] = None,
        node_type: Optional[str] = None,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """ANN search; returns ``[{doc_id, kind, node_type, timestamp, score}]``."""
        if not self._available:
            return []
        try:
            stats = self._client.get_collection_stats(COLLECTION)
            if int(stats.get("row_count", 0)) == 0:
                return []
        except Exception:  # noqa: BLE001
            pass

        expr = self._filter_expr(kind=kind, node_type=node_type, time_range=time_range)
        kwargs: Dict[str, Any] = {
            "collection_name": COLLECTION,
            "data": [list(query_vector)],
            "anns_field": "vector",
            "limit": int(top_k),
            "output_fields": ["doc_id", "kind", "node_type", "timestamp"],
        }
        if expr:
            kwargs["filter"] = expr

        try:
            hits = self._client.search(**kwargs)
        except Exception as e:  # noqa: BLE001
            log.warning("milvus_backend: search failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for group in hits or []:
            for hit in group:
                entity = hit.get("entity", {})
                results.append({
                    "doc_id": str(entity.get("doc_id", "")),
                    "kind": entity.get("kind", ""),
                    "node_type": entity.get("node_type", ""),
                    "timestamp": int(entity.get("timestamp", 0) or 0),
                    "score": float(hit.get("distance", 0.0)),
                })
        return results

    @staticmethod
    def _filter_expr(
        *,
        kind: Optional[str],
        node_type: Optional[str],
        time_range: Optional[Tuple[int, int]],
    ) -> str:
        parts: List[str] = []
        if kind:
            parts.append(f'kind == "{_escape(kind)}"')
        if node_type:
            parts.append(f'node_type == "{_escape(node_type)}"')
        if time_range is not None:
            start, end = time_range
            if start > 0:
                parts.append(f"timestamp >= {int(start)}")
            if end > 0:
                parts.append(f"timestamp <= {int(end)}")
        return " and ".join(parts)

    # ── maintenance ───────────────────────────────────────────────────

    def count(self) -> int:
        if not self._available:
            return 0
        try:
            stats = self._client.get_collection_stats(COLLECTION)
            return int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    def clear_kind(self, kind: str) -> None:
        """Delete all vectors of one kind (e.g. scene) — plan vectors kept."""
        if not self._available:
            return
        try:
            self._client.delete(
                collection_name=COLLECTION,
                filter=f'kind == "{_escape(kind)}"',
            )
        except Exception as e:  # noqa: BLE001
            log.warning("milvus_backend: clear_kind(%s) failed: %s", kind, e)

    def drop(self) -> None:
        if self._available and self._client.has_collection(COLLECTION):
            self._client.drop_collection(COLLECTION)
