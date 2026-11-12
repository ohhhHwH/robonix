"""Tag Store — in-memory inverted index for 4-dimension tag filtering.

Phase1: Python dict-based inverted index.
Phase2: swap to Redis with same logical interface.

The tag index is the FIRST retrieval stage: it reduces a candidate set
from O(1000s) to O(10s) before vector ranking is applied.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..core.types import TagFilter, TagSet, MemoryNode, difficulty_leq


class TagIndex:
    """Inverted index over TagSet dimensions.

    Each dimension value maps to a set of node_ids.
    Filter queries compute the intersection of dimension-indexed sets.
    """

    def __init__(self):
        # dimension name -> value -> set of node_ids
        self._inverted: Dict[str, Dict[str, Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # node_id -> TagSet (for rebuild / remove)
        self._node_tags: Dict[int, TagSet] = {}
        # obj_id -> set of node_ids (design §2.5 — stable cross-time object key)
        self._obj_id_index: Dict[str, Set[int]] = defaultdict(set)
        # node_id -> obj_ids (for remove / rebuild)
        self._node_obj_ids: Dict[int, Set[str]] = defaultdict(set)
        # All known node IDs (for "no filter" query)
        self._all_ids: Set[int] = set()

    # ── Public API ────────────────────────────────────────────────────

    def insert(self, node_id: int, tags: TagSet, obj_ids=None) -> None:
        """Index a node's tags into all dimension inverted lists."""
        self._all_ids.add(node_id)
        self._node_tags[node_id] = tags
        self._index_tags(node_id, tags)
        if obj_ids:
            self.index_obj_ids(node_id, obj_ids)

    def index_obj_ids(self, node_id: int, obj_ids) -> None:
        """Index a node's stable object ids (design §2.5 — cross-time key)."""
        ids = set(obj_ids or [])
        self._node_obj_ids[node_id] |= ids
        for oid in ids:
            if oid:
                self._obj_id_index[oid].add(node_id)

    def remove(self, node_id: int) -> None:
        """Remove a node from all indices."""
        self._all_ids.discard(node_id)
        tags = self._node_tags.pop(node_id, None)
        if tags is None:
            return
        self._deindex_tags(node_id, tags)
        for oid in self._node_obj_ids.pop(node_id, set()):
            self._obj_id_index[oid].discard(node_id)

    def query(self, tag_filter: TagFilter) -> Set[int]:
        """Return candidate node_ids matching the filter (AND semantics).

        Empty TagFilter returns all indexed node IDs.
        """
        if tag_filter.is_empty():
            return set(self._all_ids)

        result_sets: List[Set[int]] = []

        # Spatial: scene_type
        if tag_filter.scene_type is not None:
            result_sets.append(
                self._inverted["scene_type"].get(tag_filter.scene_type, set())
            )

        # Spatial: objects (intersection: node must contain ALL requested objects)
        # Each requested token matches either the free label (objects_present)
        # OR a stable obj_id (design §2.5 — cross-time object key).
        if tag_filter.objects is not None:
            obj_matches: List[Set[int]] = []
            for obj in tag_filter.objects:
                label_hits = self._inverted["objects_present"].get(obj, set())
                obj_id_hits = self._obj_id_index.get(obj, set())
                obj_matches.append(label_hits | obj_id_hits)
            if obj_matches:
                # Node must contain all requested objects
                obj_intersection = obj_matches[0]
                for s in obj_matches[1:]:
                    obj_intersection = obj_intersection & s
                result_sets.append(obj_intersection)

        # Behaviour: action_type
        if tag_filter.action_type is not None:
            result_sets.append(
                self._inverted["action_type"].get(tag_filter.action_type, set())
            )

        # Behaviour: success (bitmap-like)
        if tag_filter.success is not None:
            key = "success" if tag_filter.success else "failure"
            result_sets.append(self._inverted["success"].get(key, set()))

        # Cognitive: task_type
        if tag_filter.task_type is not None:
            result_sets.append(
                self._inverted["task_type"].get(tag_filter.task_type, set())
            )

        # Cognitive: difficulty_max (range filter)
        if tag_filter.difficulty_max is not None:
            diff_set: Set[int] = set()
            for level in ("easy", "medium", "hard"):
                if difficulty_leq(level, tag_filter.difficulty_max):
                    diff_set |= self._inverted["difficulty"].get(level, set())
                else:
                    break
            result_sets.append(diff_set)

        if not result_sets:
            return set(self._all_ids)

        # AND all result sets
        final = result_sets[0]
        for s in result_sets[1:]:
            final = final & s
        return final

    def rebuild(self, nodes: List[MemoryNode]) -> None:
        """Full rebuild from a list of nodes (e.g., after graph_store reload)."""
        self._inverted.clear()
        self._node_tags.clear()
        self._all_ids.clear()
        self._obj_id_index.clear()
        self._node_obj_ids.clear()
        for node in nodes:
            if node.tags is None:
                continue
            self._all_ids.add(node.node_id)
            self._node_tags[node.node_id] = node.tags
            self._index_tags(node.node_id, node.tags)
            if node.spatial_data is not None:
                obj_ids = [o.obj_id for o in node.spatial_data.objects if o.obj_id]
                if obj_ids:
                    self.index_obj_ids(node.node_id, obj_ids)

    def count(self) -> int:
        """Number of indexed nodes."""
        return len(self._all_ids)

    def get_tags(self, node_id: int) -> Optional[TagSet]:
        """Return the TagSet for a given node, or None."""
        return self._node_tags.get(node_id)

    # ── Internal ──────────────────────────────────────────────────────

    def _index_tags(self, node_id: int, tags: TagSet) -> None:
        """Add a node to all inverted index dimensions."""
        sid = {node_id}

        if tags.scene_type:
            self._inverted["scene_type"][tags.scene_type] |= sid
        for obj in tags.objects_present:
            if obj:
                self._inverted["objects_present"][obj] |= sid
        if tags.region:
            self._inverted["region"][tags.region] |= sid

        if tags.action_type:
            self._inverted["action_type"][tags.action_type] |= sid
        success_key = "success" if tags.success else "failure"
        self._inverted["success"][success_key] |= sid
        for tool in tags.tool_used:
            if tool:
                self._inverted["tool_used"][tool] |= sid

        if tags.task_type:
            self._inverted["task_type"][tags.task_type] |= sid
        if tags.difficulty:
            self._inverted["difficulty"][tags.difficulty] |= sid

    def _deindex_tags(self, node_id: int, tags: TagSet) -> None:
        """Remove a node from all inverted index dimensions."""
        sid = {node_id}

        if tags.scene_type:
            self._inverted["scene_type"][tags.scene_type] -= sid
        for obj in tags.objects_present:
            if obj:
                self._inverted["objects_present"][obj] -= sid
        if tags.region:
            self._inverted["region"][tags.region] -= sid

        if tags.action_type:
            self._inverted["action_type"][tags.action_type] -= sid
        success_key = "success" if tags.success else "failure"
        self._inverted["success"][success_key] -= sid
        for tool in tags.tool_used:
            if tool:
                self._inverted["tool_used"][tool] -= sid

        if tags.task_type:
            self._inverted["task_type"][tags.task_type] -= sid
        if tags.difficulty:
            self._inverted["difficulty"][tags.difficulty] -= sid
