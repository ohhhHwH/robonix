"""MongoDB backend — authoritative metadata + RGB images (BSON Binary).

Replaces ``graph_store.json`` / ``ptdl_store.json`` / ``data/images/`` as the
single source of truth (design doc §3).  Milvus is a derived index that
references back here by ``doc_id``.

Collections:
  scene_nodes   — one doc per MemoryNode, ``_id`` == node_id, images Binary[]
  causal_edges  — directed edges (denormalised for child/parent queries)
  plan_tasks    — one doc per saved RTDL plan tree, ``_id`` == sha1(query)

Graceful degradation: when ``pymongo`` is absent or ``mongod`` is unreachable,
``available`` is False and every method is a no-op, so callers fall back to
the legacy JSON files.  pymongo's connection pool is thread-safe; the
``GraphStore`` lock already serialises writers, so no extra locking is added
here.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from ..core.types import CausalEdge, MemoryNode

log = logging.getLogger("scribe_mem")

DEFAULT_URI = "mongodb://127.0.0.1:27017/robonix_memory"
DEFAULT_DB = "robonix_memory"

# BSON subtype for generic binary (RGB JPEG frames).
_BINARY_SUBTYPE = 0

# Default projection: exclude the heavy fields that are loaded lazily
# (design doc §6) — image BSON Binary and per-frame timestamps.
_NODE_PROJECTION = {"images": 0, "frame_ts": 0}


def _mongo_image_marker(node_id: int, index: int) -> str:
    """Opaque image ref for a node's *index*-th frame stored in Mongo."""
    return f"mongo:{node_id}:{index}"


class MongoBackend:
    """Thin pymongo wrapper for scene_nodes / causal_edges / plan_tasks."""

    def __init__(self, uri: str = "", db_name: str = DEFAULT_DB):
        self._uri = uri or os.environ.get("MEMGRAPH_MONGO_URI", DEFAULT_URI)
        self._db_name = db_name
        self._client: Optional[Any] = None
        self._db: Optional[Any] = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            import pymongo  # noqa: F401
        except ImportError:
            log.warning(
                "mongo_backend: pymongo not installed; MongoDB persistence "
                "disabled (JSON fallback active)"
            )
            return
        try:
            self._client = pymongo.MongoClient(
                self._uri, serverSelectionTimeoutMS=2000
            )
            self._client.admin.command("ping")
            self._db = self._client[self._db_name]
            self._available = True
            log.info("mongo_backend: connected %s", self._uri)
        except Exception as e:  # noqa: BLE001 — any connect failure → JSON
            log.warning(
                "mongo_backend: MongoDB unavailable at %s (%s); JSON fallback",
                self._uri,
                e,
            )
            self._client = None
            self._db = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _col(self, name: str):
        return self._db[name] if self._available else None

    # ── node serialization ────────────────────────────────────────────

    @staticmethod
    def _metadata(node: MemoryNode) -> Dict[str, Any]:
        """Node's Mongo metadata fields, excluding _id/images/embedding.

        ``embedding`` lives in Milvus; ``image_refs`` (disk paths) and
        ``images`` (Binary) are owned by ImageStore and must not be clobbered
        by a metadata-only upsert.
        """
        d = node.to_dict()
        d.pop("embedding", None)
        d.pop("image_refs", None)
        return d

    @staticmethod
    def doc_to_node(doc: Dict[str, Any]) -> MemoryNode:
        """Rebuild a MemoryNode from a Mongo doc (images left lazy).

        ``image_count`` is a scalar kept on the doc so the default projection
        (which excludes the ``images`` Binary array) still lets us emit the
        right number of opaque markers.
        """
        d = {k: v for k, v in doc.items()
             if k not in ("_id", "images", "image_count")}
        node = MemoryNode.from_dict(d)
        n_images = int(doc.get("image_count") or 0)
        # image_refs stays as opaque markers so callers can detect "has images"
        # and hand the refs back to ImageStore for lazy byte loading.
        node.image_refs = [
            _mongo_image_marker(node.node_id, i) for i in range(n_images)
        ]
        return node

    # ── scene_nodes ───────────────────────────────────────────────────

    def upsert_node(self, node: MemoryNode) -> None:
        col = self._col("scene_nodes")
        if col is None:
            return
        col.update_one(
            {"_id": int(node.node_id)},
            {"$set": self._metadata(node)},
            upsert=True,
        )

    def get_node(self, node_id: int) -> Optional[MemoryNode]:
        col = self._col("scene_nodes")
        if col is None:
            return None
        doc = col.find_one({"_id": int(node_id)}, _NODE_PROJECTION)
        return self.doc_to_node(doc) if doc else None

    def get_nodes(self, node_ids: List[int]) -> List[MemoryNode]:
        col = self._col("scene_nodes")
        if col is None:
            return []
        docs = col.find({"_id": {"$in": [int(n) for n in node_ids]}},
                        _NODE_PROJECTION)
        return [self.doc_to_node(d) for d in docs]

    def all_nodes(self) -> List[MemoryNode]:
        col = self._col("scene_nodes")
        if col is None:
            return []
        return [self.doc_to_node(d) for d in col.find({}, _NODE_PROJECTION)]

    def delete_node(self, node_id: int) -> None:
        col = self._col("scene_nodes")
        if col is not None:
            col.delete_one({"_id": int(node_id)})

    def count_scene(self) -> int:
        col = self._col("scene_nodes")
        return col.count_documents({}) if col is not None else 0

    def clear_scene(self) -> None:
        col = self._col("scene_nodes")
        if col is not None:
            col.delete_many({})

    # ── images (owned by ImageStore, stored on the node doc) ──────────

    def push_image(self, node_id: int, image_bytes: bytes) -> int:
        """Append one RGB frame to a node's ``images`` Binary array.

        Returns the new frame count (0 when Mongo is unavailable or the node
        doc does not exist).
        """
        col = self._col("scene_nodes")
        if col is None:
            return 0
        import bson

        result = col.update_one(
            {"_id": int(node_id)},
            {"$push": {"images": bson.Binary(image_bytes, subtype=_BINARY_SUBTYPE)},
             "$inc": {"image_count": 1}},
        )
        if result.matched_count == 0:
            return 0
        return self.image_count(node_id)

    def get_images(self, node_id: int) -> List[bytes]:
        col = self._col("scene_nodes")
        if col is None:
            return []
        doc = col.find_one({"_id": int(node_id)}, {"images": 1})
        return [bytes(b) for b in (doc.get("images") or [])] if doc else []

    def pop_oldest_image(self, node_id: int) -> None:
        """Drop the oldest frame (``images[0]``) for a node."""
        col = self._col("scene_nodes")
        if col is None:
            return
        col.update_one(
            {"_id": int(node_id)},
            {"$pop": {"images": -1}, "$inc": {"image_count": -1}},
        )

    def image_count(self, node_id: int) -> int:
        col = self._col("scene_nodes")
        if col is None:
            return 0
        doc = col.find_one({"_id": int(node_id)}, {"image_count": 1})
        return int(doc.get("image_count") or 0) if doc else 0

    def clear_images(self, node_id: int) -> None:
        col = self._col("scene_nodes")
        if col is not None:
            col.update_one(
                {"_id": int(node_id)},
                {"$unset": {"images": ""}, "$set": {"image_count": 0}},
            )

    # ── causal_edges ──────────────────────────────────────────────────

    def upsert_edges(self, edges: List[CausalEdge]) -> None:
        col = self._col("causal_edges")
        if col is None or not edges:
            return
        for e in edges:
            doc = {
                "_id": f"{e.from_node_id}->{e.to_node_id}",
                "from": int(e.from_node_id),
                "to": int(e.to_node_id),
                "relation": e.relation.value if e.relation else "enables",
            }
            col.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def delete_edges_for(self, node_id: int) -> None:
        col = self._col("causal_edges")
        if col is None:
            return
        col.delete_many({"$or": [{"from": int(node_id)}, {"to": int(node_id)}]})

    def all_edges(self) -> List[CausalEdge]:
        col = self._col("causal_edges")
        if col is None:
            return []
        edges: List[CausalEdge] = []
        for d in col.find({}):
            edges.append(CausalEdge(
                from_node_id=int(d["from"]),
                to_node_id=int(d["to"]),
                relation=d.get("relation", "enables"),
            ))
        return edges

    def clear_edges(self) -> None:
        col = self._col("causal_edges")
        if col is not None:
            col.delete_many({})

    # ── plan_tasks ────────────────────────────────────────────────────

    @staticmethod
    def _plan_key(query: str) -> str:
        return hashlib.sha1((query or "").strip().encode("utf-8")).hexdigest()

    def upsert_plan(self, entry: Dict[str, Any]) -> None:
        col = self._col("plan_tasks")
        if col is None:
            return
        doc = dict(entry)
        doc["_id"] = self._plan_key(entry.get("query", ""))
        col.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def get_plan(self, query: str) -> Optional[Dict[str, Any]]:
        col = self._col("plan_tasks")
        if col is None:
            return None
        doc = col.find_one({"_id": self._plan_key(query)})
        if doc:
            doc = {k: v for k, v in doc.items() if k != "_id"}
        return doc

    def get_plan_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        col = self._col("plan_tasks")
        if col is None:
            return None
        doc = col.find_one({"_id": str(key)})
        if doc:
            doc = {k: v for k, v in doc.items() if k != "_id"}
        return doc

    def all_plans(self) -> List[Dict[str, Any]]:
        col = self._col("plan_tasks")
        if col is None:
            return []
        return [
            {k: v for k, v in d.items() if k != "_id"} for d in col.find({})
        ]

    def delete_plan(self, query: str) -> None:
        col = self._col("plan_tasks")
        if col is not None:
            col.delete_one({"_id": self._plan_key(query)})

    def count_plans(self) -> int:
        col = self._col("plan_tasks")
        return col.count_documents({}) if col is not None else 0

    def clear_plans(self) -> None:
        col = self._col("plan_tasks")
        if col is not None:
            col.delete_many({})

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._available = False
