"""Object Timeline — a per-object cross-time index (design §2).

Upgrades ``ObjectCoord.obj_id`` into a stable cross-time object key.  Where
``TagIndex`` answers "what happened here" and ``VectorStore`` answers "what
is semantically similar", this index answers "how did one object change over
time" — the primitive behind Demo 1's "was the globe moved since yesterday?".

It is derived and rebuildable from the node store: every node's
``spatial_data.objects[]`` already carries the coordinate snapshot, so this
index holds only ``obj_id -> [(timestamp, node_id, ObjectCoord), ...]`` and
can be reconstructed wholesale (like ``TagIndex.rebuild``) with no separate
persistence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Set

from ..core.types import MemoryNode, ObjectCoord


class ObjectTimeline:
    """Index ``obj_id`` → ascending list of ``(timestamp, node_id, coord)``."""

    def __init__(self) -> None:
        # obj_id -> list of (timestamp, node_id, ObjectCoord), ascending.
        self._entries: Dict[str, List[tuple[int, int, ObjectCoord]]] = defaultdict(list)

    # ── Write ──────────────────────────────────────────────────────────

    def record(self, obj_id: str, node_id: int, coord: ObjectCoord,
               timestamp: int) -> None:
        """Append one snapshot for *obj_id*, keeping the list sorted by time."""
        if not obj_id:
            return
        self._entries[obj_id].append((timestamp, node_id, coord))
        self._entries[obj_id].sort(key=lambda e: e[0])

    # ── Read ───────────────────────────────────────────────────────────

    def history(self, obj_id: str, limit: int = 0) -> List[Dict[str, Any]]:
        """Return the object's snapshots, most-recent-first.

        ``limit=0`` returns all.  Each entry carries the object key, node id,
        shot time and the 3D coordinate snapshot.
        """
        entries = self._entries.get(obj_id, [])
        out: List[Dict[str, Any]] = []
        for ts, node_id, coord in reversed(entries):
            out.append(self._snapshot(obj_id, ts, node_id, coord))
            if limit and len(out) >= limit:
                break
        return out

    def diff(self, obj_id: str) -> Dict[str, Any]:
        """Compare the two most recent snapshots of *obj_id*.

        Returns the latest and previous snapshots plus ``delta`` holding the
        per-axis displacement and Euclidean distance.  ``delta`` is ``None``
        when fewer than two snapshots exist.
        """
        entries = self._entries.get(obj_id, [])
        if len(entries) < 2:
            latest = self.history(obj_id, limit=1)
            return {
                "obj_id": obj_id,
                "latest": latest[0] if latest else None,
                "previous": None,
                "delta": None,
            }
        prev_ts, prev_nid, prev_coord = entries[-2]
        last_ts, last_nid, last_coord = entries[-1]
        dx = last_coord.x - prev_coord.x
        dy = last_coord.y - prev_coord.y
        dz = last_coord.z - prev_coord.z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        return {
            "obj_id": obj_id,
            "latest": self._snapshot(obj_id, last_ts, last_nid, last_coord),
            "previous": self._snapshot(obj_id, prev_ts, prev_nid, prev_coord),
            "delta": {
                "dx": round(dx, 6),
                "dy": round(dy, 6),
                "dz": round(dz, 6),
                "distance": round(distance, 6),
            },
        }

    def nodes_for_obj_id(self, obj_id: str) -> Set[int]:
        """Return every node id that has recorded *obj_id* (for search)."""
        return {node_id for _, node_id, _ in self._entries.get(obj_id, [])}

    def obj_ids(self) -> List[str]:
        """Return all known object ids."""
        return list(self._entries.keys())

    def count(self) -> int:
        """Number of distinct objects indexed."""
        return len(self._entries)

    # ── Rebuild ────────────────────────────────────────────────────────

    def rebuild(self, nodes: List[MemoryNode]) -> None:
        """Recompute the whole index from nodes (like ``TagIndex.rebuild``)."""
        self._entries.clear()
        for node in nodes:
            if node.spatial_data is None:
                continue
            for coord in node.spatial_data.objects:
                if coord.obj_id:
                    self.record(coord.obj_id, node.node_id, coord, node.timestamp)

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _snapshot(obj_id: str, ts: int, node_id: int, coord: ObjectCoord) -> Dict[str, Any]:
        return {
            "obj_id": obj_id,
            "node_id": node_id,
            "timestamp": ts,
            "label": coord.label,
            "x": coord.x,
            "y": coord.y,
            "z": coord.z,
        }
