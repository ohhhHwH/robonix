"""Graph Store — in-memory dict + MongoDB persistence for MemoryNode.

MongoDB (``scene_nodes`` / ``causal_edges``) is the authoritative store when
available; the JSON file is kept as a graceful-degradation fallback (and a
one-time backfill source).  The derived Milvus vector index is synced on the
same write path ("先 Mongo 后 Milvus", design doc §5.1): every add/update/
promote upserts a ``memory_vectors`` row keyed by ``doc_id=str(node_id)``.

Interface is designed so a NetworkX / Neo4j backend can be swapped in later.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.types import CausalRelation, CausalEdge, MemoryNode, NodeType

log = logging.getLogger("scribe_mem")


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


def _scene_embed_text(node: MemoryNode) -> str:
    """Bilingual embed text for a scene node (design doc §4)."""
    zh = (node.summary_zh or "").strip()
    en = (node.summary_en or "").strip()
    if zh and en:
        return f"{zh} / {en}"
    return node.summary or zh or en


class GraphStore:
    """Persistent store for MemoryNode and causal edges.

    The scene hook writes from its own thread while MCP tools write from the
    event loop, so every public method holds one reentrant lock. Unlocked,
    two writers race on the shared graph_store.json.tmp and one os.replace
    finds its file already moved.

    ``mongo`` / ``milvus`` / ``embedder`` are optional backends injected by
    ``service.py``; when absent the store degrades to JSON-only with no
    vector index (``None`` → no-op).
    """

    # Local directory under CWD (matches service MEMORY_DIR default).
    # When run via rbnx boot, CWD = services/memory/ → data at ./memory/
    DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "memory")

    def __init__(self, data_dir: str = "", *, mongo=None, milvus=None, embedder=None):
        self._data_dir = Path(data_dir or self.DEFAULT_DATA_DIR)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self._nodes: Dict[int, MemoryNode] = {}
        # adjacency: node_id -> set of child node_ids
        self._children: Dict[int, Set[int]] = {}
        # reverse: node_id -> set of parent node_ids
        self._parents: Dict[int, Set[int]] = {}
        # Single monotonic counter. node_type is the sole long/short-term
        # discriminator; IDs carry no range semantics (stable across promote).
        self._next_id: int = 0

        self._mongo = mongo
        self._milvus = milvus
        self._embedder = embedder

        self._load()

    # ── Node CRUD ──────────────────────────────────────────────────────

    @_locked
    def add_node(self, node: MemoryNode) -> int:
        """Write a node, auto-assigning node_id if not set.

        Returns the assigned node_id.
        """
        if node.node_id <= 0:
            node.node_id = self._next_node_id()
        else:
            # Ensure counter stays ahead of explicit IDs
            self._next_id = max(self._next_id, node.node_id + 1)

        if node.created_at == 0:
            node.created_at = time.time_ns()

        self._nodes[node.node_id] = node
        if node.node_id not in self._children:
            self._children[node.node_id] = set()
        if node.node_id not in self._parents:
            self._parents[node.node_id] = set()

        self._persist()
        self._sync_milvus_upsert(node)
        return node.node_id

    @_locked
    def get_node(self, node_id: int) -> Optional[MemoryNode]:
        """Get a single node by ID."""
        return self._nodes.get(node_id)

    @_locked
    def get_nodes(self, node_ids: List[int]) -> List[MemoryNode]:
        """Batch query nodes."""
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    @_locked
    def update_node(self, node_id: int, node: MemoryNode) -> None:
        """Update an existing node with optimistic-lock version check.

        Raises ValueError if the node doesn't exist or version mismatch.
        """
        existing = self._nodes.get(node_id)
        if existing is None:
            raise ValueError(f"Node {node_id} not found")
        if node.version != existing.version:
            raise ValueError(
                f"Version conflict for node {node_id}: "
                f"expected {existing.version}, got {node.version}"
            )
        node.version += 1
        self._nodes[node_id] = node
        self._persist()
        self._sync_milvus_upsert(node)

    @_locked
    def list_by_type(self, node_type: NodeType, limit: int = 100) -> List[MemoryNode]:
        """List nodes by NodeType, sorted by created_at descending."""
        matching = [n for n in self._nodes.values() if n.node_type == node_type]
        matching.sort(key=lambda n: n.created_at, reverse=True)
        return matching[:limit]

    @_locked
    def list_by_time(self, start_ts: int, end_ts: int, limit: int = 100) -> List[MemoryNode]:
        """List nodes within a timestamp range.

        end_ts=0 means no upper bound (up to current time).
        """
        if end_ts == 0:
            end_ts = time.time_ns()
        matching = [n for n in self._nodes.values()
                    if start_ts <= n.timestamp <= end_ts]
        matching.sort(key=lambda n: n.timestamp, reverse=True)
        return matching[:limit]

    @_locked
    def remove_node(self, node_id: int) -> bool:
        """Remove a node and its edges. Returns True if the node existed."""
        if node_id not in self._nodes:
            return False
        # Remove edges
        for child in list(self._children.get(node_id, set())):
            self._parents.get(child, set()).discard(node_id)
        for parent in list(self._parents.get(node_id, set())):
            self._children.get(parent, set()).discard(node_id)
        del self._nodes[node_id]
        self._children.pop(node_id, None)
        self._parents.pop(node_id, None)
        if self._mongo is not None:
            self._mongo.delete_node(node_id)
            self._mongo.delete_edges_for(node_id)
        self._persist()
        self._sync_milvus_delete(node_id)
        return True

    @_locked
    def clear_all(self) -> None:
        """Remove all nodes, edges, and reset the ID counter."""
        self._nodes.clear()
        self._children.clear()
        self._parents.clear()
        self._next_id = 0
        if self._mongo is not None:
            self._mongo.clear_scene()
            self._mongo.clear_edges()
        if self._milvus is not None:
            self._milvus.clear_kind("scene")
        self._persist()

    @_locked
    def count(self) -> int:
        """Total number of nodes."""
        return len(self._nodes)

    @_locked
    def all_ids(self) -> List[int]:
        """Return all node IDs."""
        return list(self._nodes.keys())

    # ── Causal edges ───────────────────────────────────────────────────

    @_locked
    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Add a causal edge: parent → child."""
        if parent_id not in self._nodes or child_id not in self._nodes:
            raise ValueError(f"Edge endpoint not found: {parent_id} -> {child_id}")

        self._children.setdefault(parent_id, set()).add(child_id)
        self._parents.setdefault(child_id, set()).add(parent_id)

        # Update causal_chain on child node
        child_node = self._nodes[child_id]
        if parent_id not in child_node.causal_chain:
            child_node.causal_chain = list(child_node.causal_chain) + [parent_id]

        self._persist()

    @_locked
    def get_parents(self, node_id: int) -> List[int]:
        """Return parent node IDs for a given node."""
        return sorted(self._parents.get(node_id, set()))

    @_locked
    def get_children(self, node_id: int) -> List[int]:
        """Return child node IDs for a given node."""
        return sorted(self._children.get(node_id, set()))

    @_locked
    def get_all_edges(self) -> List[CausalEdge]:
        """Return all causal edges in the graph."""
        edges: List[CausalEdge] = []
        for parent_id, children in self._children.items():
            for child_id in children:
                edges.append(CausalEdge(
                    from_node_id=parent_id,
                    to_node_id=child_id,
                    relation=CausalRelation.ENABLES,
                ))
        return edges

    # ── Internal ───────────────────────────────────────────────────────

    def _next_node_id(self) -> int:
        """Auto-increment monotonic ID (no long/short range semantics)."""
        nid = self._next_id
        self._next_id += 1
        return nid

    @_locked
    def promote_to_long_term(self, node_id: int) -> Optional[int]:
        """Mark a short-term node long-term, keeping its stable ``node_id``.

        Returns the node_id, or None if not found / already long-term.
        ``node_type`` is the sole long/short discriminator now, so promote
        does not renumber — it flips the flag and re-syncs Milvus so the
        ``node_type`` scalar filter stops missing the freshly-promoted node.
        """
        node = self._nodes.get(node_id)
        if node is None or node.node_type == NodeType.LONG_TERM:
            return None

        node.node_type = NodeType.LONG_TERM
        node.version += 1
        self._persist()
        self._sync_milvus_upsert(node)
        return node_id

    def _milvus_vector(self, node: MemoryNode) -> List[float]:
        """Return the cached vector, computing + caching it when empty."""
        if node.embedding:
            return list(node.embedding)
        if self._embedder is None:
            return []
        try:
            vec = self._embedder.encode(_scene_embed_text(node))
            node.embedding = vec
            return vec
        except Exception as e:  # noqa: BLE001 — vector index is best-effort
            log.warning("graph_store: embed for node %d failed: %s", node.node_id, e)
            return []

    def _sync_milvus_upsert(self, node: MemoryNode) -> None:
        if self._milvus is None or not self._milvus.available:
            return
        vec = self._milvus_vector(node)
        if not vec:
            return
        self._milvus.upsert(
            str(node.node_id),
            vec,
            kind="scene",
            node_type=node.node_type.value,
            timestamp=node.timestamp,
        )

    def _sync_milvus_delete(self, node_id: int) -> None:
        if self._milvus is not None and self._milvus.available:
            self._milvus.delete(str(node_id))

    @_locked
    def rebuild_milvus(self) -> int:
        """Re-upsert every node into Milvus (on_init rebuild from Mongo).

        Nodes loaded from Mongo carry no cached vector (Mongo stores none),
        so ``_sync_milvus_upsert`` re-embeds each one via the BGE embedder.
        """
        if self._milvus is None or not self._milvus.available:
            return 0
        for node in self._nodes.values():
            self._sync_milvus_upsert(node)
        return len(self._nodes)

    def _persist(self) -> None:
        """Flush nodes and edges to the JSON fallback + MongoDB."""
        nodes_data = []
        for n in self._nodes.values():
            d = n.to_dict()
            d.pop("embedding", None)  # vector is derived → lives only in Milvus
            nodes_data.append(d)
        edges_data = []
        for parent_id, children in self._children.items():
            for child_id in children:
                edges_data.append({"from": parent_id, "to": child_id})

        payload = {
            "next_id": self._next_id,
            "nodes": nodes_data,
            "edges": edges_data,
        }
        filepath = self._data_dir / "graph_store.json"
        tmp = str(filepath) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(filepath))

        # Mongo (authoritative) — only metadata, never embedding/images.
        if self._mongo is not None and self._mongo.available:
            for node in self._nodes.values():
                self._mongo.upsert_node(node)
            self._mongo.upsert_edges(self.get_all_edges())

    def _load(self) -> None:
        """Restore from JSON first, then overlay / backfill MongoDB."""
        filepath = self._data_dir / "graph_store.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            # Back-compat: legacy JSON carried next_short_id / next_long_id.
            self._next_id = data.get(
                "next_id",
                max(data.get("next_short_id", 0), data.get("next_long_id", 1000)),
            )

            for nd in data.get("nodes", []):
                node = MemoryNode.from_dict(nd)
                self._nodes[node.node_id] = node
                self._children[node.node_id] = set()
                self._parents[node.node_id] = set()

            for ed in data.get("edges", []):
                pid, cid = ed["from"], ed["to"]
                self._children.setdefault(pid, set()).add(cid)
                self._parents.setdefault(cid, set()).add(pid)

        if self._mongo is not None and self._mongo.available:
            mongo_nodes = self._mongo.all_nodes()
            if mongo_nodes:
                # Mongo is authoritative — replace the JSON view.
                self._nodes = {n.node_id: n for n in mongo_nodes}
                self._children = {nid: set() for nid in self._nodes}
                self._parents = {nid: set() for nid in self._nodes}
                for e in self._mongo.all_edges():
                    self._children.setdefault(e.from_node_id, set()).add(e.to_node_id)
                    self._parents.setdefault(e.to_node_id, set()).add(e.from_node_id)
            else:
                # One-time backfill: JSON history → Mongo.
                for node in self._nodes.values():
                    self._mongo.upsert_node(node)
                self._mongo.upsert_edges(self.get_all_edges())

        if self._nodes:
            self._next_id = max(self._next_id, max(self._nodes) + 1)
