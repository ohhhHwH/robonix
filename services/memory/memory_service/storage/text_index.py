# SPDX-License-Identifier: MulanPSL-2.0
"""Lightweight text index for fast LLM-friendly memory scanning.

Every memory node is mirrored as a single text line in
``data/text_index.txt``.  The format is designed to be readable by an
LLM in a single prompt so Tier-1 retrieval can answer queries without
any BM25 / embedding / ranker cost.

Format (tab-separated for grep-friendliness, no JSON)::

    [node_id] | [node_type] | [timestamp_ns] | [summary] | [objects] | [region] | [parent_id]

The index is rebuilt from ``graph_store.json`` on restart so it never
drifts from the authoritative store.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger("scribe_mem")

_DATA_DIR = os.environ.get(
    "AGENT_MEMORY_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "memory"),
)
_DEFAULT_PATH = os.path.join(_DATA_DIR, "text_index.txt")


class TextIndex:
    """Append-only text index of all memory nodes.

    Thread-safe: :meth:`append`, :meth:`update`, and :meth:`remove`
    acquire a reentrant lock so the writer (remember pipeline) and
    readers (retrieve pipeline) do not step on each other.
    """

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        import threading
        self._path = path
        self._lock = threading.Lock()
        # In-memory cache: node_id → line text (excluding node_id prefix)
        self._lines: Dict[int, str] = {}
        self._loaded = False

    # ── file I/O ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load from disk on first access."""
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if not line.strip():
                        continue
                    nid = _parse_node_id(line)
                    if nid is not None:
                        self._lines[nid] = line
        except OSError as e:
            log.warning("text_index: could not read %s: %s", self._path, e)

    def _flush(self) -> None:
        """Write the full in-memory index to disk (overwrite)."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                for nid in sorted(self._lines):
                    f.write(self._lines[nid] + "\n")
        except OSError as e:
            log.warning("text_index: could not write %s: %s", self._path, e)

    # ── public API ─────────────────────────────────────────────────────

    def append(self, node) -> None:
        """Append a memory node to the index."""
        self._ensure_loaded()
        line = _format_line(node)
        with self._lock:
            self._lines[node.node_id] = line
            self._append_line_to_file(line)

    def update(self, node_id: int, node) -> None:
        """Update the index entry for *node_id* (e.g. after VLM enrichment)."""
        self._ensure_loaded()
        line = _format_line(node)
        with self._lock:
            self._lines[node_id] = line
            self._flush()

    def remove(self, node_id: int) -> None:
        """Remove an entry from the index."""
        self._ensure_loaded()
        with self._lock:
            self._lines.pop(node_id, None)
            self._flush()

    def rebuild(self, nodes) -> None:
        """Rebuild the entire index from a list of MemoryNodes."""
        with self._lock:
            self._lines.clear()
            for node in nodes:
                self._lines[node.node_id] = _format_line(node)
            self._flush()
            self._loaded = True

    def search(self, query: str, top_k: int = 20) -> List[int]:
        """Keyword search over index lines — returns matching node_ids.

        Case-insensitive substring match against the full line text.
        Returns the top *top_k* results (most recent first by default
        sort — higher node_ids are more recent).
        """
        self._ensure_loaded()
        q = query.lower()
        hits: List[int] = []
        with self._lock:
            for nid, line in self._lines.items():
                if q in line.lower():
                    hits.append(nid)
        # Most recent first (higher node_id ≈ newer)
        hits.sort(reverse=True)
        return hits[:top_k]

    def get_summary(self, node_ids: List[int]) -> str:
        """Return index lines for the given node_ids."""
        self._ensure_loaded()
        lines: List[str] = []
        with self._lock:
            for nid in node_ids:
                line = self._lines.get(nid)
                if line is not None:
                    lines.append(line)
        return "\n".join(lines)

    def full_text(self, max_lines: int = 200) -> str:
        """Return the complete index text (up to *max_lines* most recent).

        Designed to be passed directly into an LLM prompt as context.
        """
        self._ensure_loaded()
        with self._lock:
            nids = sorted(self._lines, reverse=True)[:max_lines]
            return "\n".join(self._lines[nid] for nid in nids)

    def count(self) -> int:
        """Number of indexed nodes."""
        self._ensure_loaded()
        with self._lock:
            return len(self._lines)

    # ── internal ───────────────────────────────────────────────────────

    def _append_line_to_file(self, line: str) -> None:
        """Append one line to the text file (fast path — no full rewrite)."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            log.warning("text_index: append failed: %s", e)


# ── singleton (module-level, like GraphStore) ─────────────────────────

_text_index: Optional[TextIndex] = None


def get_text_index(path: str = _DEFAULT_PATH) -> TextIndex:
    """Return the module-level TextIndex singleton."""
    global _text_index
    if _text_index is None:
        _text_index = TextIndex(path=path)
    return _text_index


# ── formatting ────────────────────────────────────────────────────────

def _format_line(node) -> str:
    """Format a MemoryNode as a single index line."""
    nid = node.node_id
    nt = node.node_type.value if node.node_type else "short_term"
    ts = node.timestamp
    summary = _truncate(node.summary or "", 120)
    objects = _objects_str(node)
    region = _region_str(node)
    parent_id = _parent_id_str(node)
    return f"{nid} | {nt} | {ts} | {summary} | {objects} | {region} | {parent_id}"


def _parse_node_id(line: str) -> Optional[int]:
    """Extract node_id from a formatted index line."""
    m = re.match(r"^(\d+)\s*\|", line)
    return int(m.group(1)) if m else None


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string, replacing the middle with '…' if needed."""
    s = s.replace("\n", " ").replace("\r", " ").replace("|", "/")
    if len(s) <= max_len:
        return s
    half = (max_len - 1) // 2
    return s[:half] + "…" + s[-half:]


def _objects_str(node) -> str:
    """Comma-separated object labels from tags or spatial_data."""
    if node.tags and node.tags.objects_present:
        return ",".join(node.tags.objects_present[:10])
    if node.spatial_data and node.spatial_data.objects:
        return ",".join(
            o.label for o in node.spatial_data.objects[:10] if o.label
        )
    return "-"


def _region_str(node) -> str:
    """Region from tags or kv."""
    if node.tags and node.tags.region:
        return node.tags.region
    return "-"


def _parent_id_str(node) -> str:
    """Parent node_id if available, else '-'."""
    # parent_node_id is not stored on MemoryNode directly; check causal chain
    if hasattr(node, "parent_node_id") and node.parent_node_id:
        return str(node.parent_node_id)
    return "-"
