"""Remember pipeline — transform LogRecord → MemoryNode and persist.

Pipeline:
  1. Extract tags from LogRecord + SpatialContext (rule-based for Phase1)
  2. Generate one-line summary (template-based for Phase1)
  3. Build MemoryNode
  4. graph_store.add_node(node)            — assign node_id + persist
  5. If image_base64 → ImageStore.save → update image_refs + persist
  6. tag_index.insert(node_id, tags)        — inverted index
  7. vector_store.insert(node_id, emb, summary) — vector + BM25
  8. If parent_node_id → add causal edge + persist
  9. Return RememberResponse
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import List, Optional, Tuple

from ..storage.graph_store import GraphStore
from ..storage.tag_index import TagIndex
from ..storage.vector_store import VectorStore
from .types import (
    LogRecord, MemoryNode, NodeType, SpatialContext, TagSet,
    RememberRequest, RememberResponse,
)

log = logging.getLogger("scribe_mem")

# ── Keyword sets for rule-based tag extraction (Phase1) ─────────────────

_SCENE_KEYWORDS = {
    "kitchen": ["kitchen", "sink", "stove", "fridge", "counter", "oven"],
    "living_room": ["living room", "sofa", "couch", "television", "coffee table"],
    "workshop": ["workshop", "workbench", "crafting table", "table saw", "tool rack"],
    "bedroom": ["bedroom", "bed", "wardrobe", "closet", "pillow"],
    "outdoor": ["outdoor", "garden", "yard", "street", "park", "outside"],
}

_ACTION_KEYWORDS = {
    "grasp": ["grasp", "grab", "pick", "hold", "take", "collect"],
    "place": ["place", "put", "set", "drop", "release", "leave"],
    "navigate": ["navigate", "move", "go", "walk", "drive", "travel", "approach"],
    "craft": ["craft", "make", "build", "create", "assemble", "combine"],
    "observe": ["observe", "see", "look", "watch", "scan", "inspect", "detect"],
}

_TASK_KEYWORDS = {
    "fetch": ["fetch", "get", "bring", "retrieve", "deliver"],
    "build": ["build", "construct", "assemble", "craft", "make", "create"],
    "explore": ["explore", "scan", "survey", "map", "search", "find"],
    "dialogue": ["ask", "tell", "say", "answer", "inform", "report"],
}

# ── Geometry helpers (even-odd polygon test, same algorithm as           ──
# ── system/scene/scene_service/geometry.py — duplicated here to avoid     ──
# ── a cross-package dependency)                                           ──

_Point = Tuple[float, float]


def _point_in_polygon(x: float, y: float, points: List[_Point]) -> bool:
    """Return whether (x, y) lies inside a polygon using even-odd crossings."""
    pts = [(float(px), float(py)) for px, py in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return False
    inside = False
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _polygon_area(points: List[_Point]) -> float:
    """Signed area of a polygon (shoelace formula); absolute value = area."""
    pts = [(float(px), float(py)) for px, py in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return 0.0
    twice = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        twice += x1 * y2 - x2 * y1
    return abs(twice) / 2.0


# ── Room annotation loader (reads Scene's per-map annotation JSON) ─────

_DEFAULT_ANNOTATIONS_DIR = os.path.expanduser("~/.robonix/scene/annotations")
_DEFAULT_SCENE_MAPS_DIR = os.path.expanduser("~/.robonix/scene/scene_maps")

# Module-level cache: {(annotations_dir, map_id): (max_mtime, rooms_list)}
_annotations_cache: dict = {}


def _resolve_map_id() -> str:
    """Resolve the current map_id using the same precedence as Scene.

    1. ``SCENE_MAP_ID`` environment variable
    2. Most recently saved map in ``~/.robonix/scene/scene_maps/``
    3. Empty string (loads ALL annotation files — backward compatible)
    """
    # 1. SCENE_MAP_ID env var (same as Scene's MapBinding precedence)
    env_id = os.environ.get("SCENE_MAP_ID", "").strip()
    if env_id:
        log.info("remember: map_id=%r from SCENE_MAP_ID env", env_id)
        return env_id

    # 2. Scan scene_maps/ for the most recently modified map file
    if os.path.isdir(_DEFAULT_SCENE_MAPS_DIR):
        try:
            map_files = [
                f for f in os.listdir(_DEFAULT_SCENE_MAPS_DIR)
                if f.endswith(".json") and not f.startswith(".")
            ]
            if map_files:
                best = max(
                    map_files,
                    key=lambda f: os.path.getmtime(
                        os.path.join(_DEFAULT_SCENE_MAPS_DIR, f)),
                )
                map_id = best[:-5]  # strip ".json"
                log.info("remember: map_id=%r from scene_maps/%s", map_id, best)
                return map_id
        except OSError:
            pass

    log.info("remember: no map_id resolved — loading all annotation files")
    return ""


def _load_room_annotations(annotations_dir: str,
                            map_id: str = "") -> List[Tuple[str, List[_Point]]]:
    """Scan annotation JSON files and return [(room_name, polygon_points), ...].

    Only loads annotations for the current ``map_id`` (i.e. the file
    ``{map_id}.json``) when ``map_id`` is provided.  When ``map_id`` is
    empty the legacy behaviour applies: every ``.json`` file in the
    directory is loaded.

    Only includes ``kind="room"`` annotations that are not stale.
    Results are cached by (directory, map_id) + mtime — re-scans only
    when files change.  Failures are logged and degrade to an empty list.
    """
    if not os.path.isdir(annotations_dir):
        return []

    # ── Determine which files to load ─────────────────────────────────
    target_files: set | None = None
    if map_id:
        target_files = {f"{map_id}.json"}

    # ── mtime-based cache check (keyed by dir + map_id) ──────────────
    cache_key = (annotations_dir, map_id)
    try:
        max_mtime = 0.0
        for entry in os.scandir(annotations_dir):
            if not entry.is_file(follow_symlinks=False):
                continue
            if not entry.name.endswith(".json"):
                continue
            if target_files and entry.name not in target_files:
                continue
            st = entry.stat()
            if st.st_mtime > max_mtime:
                max_mtime = st.st_mtime
    except OSError:
        max_mtime = 0.0

    cached = _annotations_cache.get(cache_key)
    if cached is not None:
        cached_mtime, cached_rooms = cached
        if cached_mtime >= max_mtime:
            return cached_rooms

    # ── scan & load ──────────────────────────────────────────────────
    rooms: List[Tuple[str, List[_Point]]] = []
    try:
        for entry in os.scandir(annotations_dir):
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name
            if not name.endswith(".json"):
                continue
            if name.startswith(".") or ".tmp" in name or ".corrupt-" in name:
                continue
            if target_files and name not in target_files:
                continue

            try:
                with open(entry.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("remember: skipping unreadable annotation file %s: %s",
                           entry.path, e)
                continue

            if not isinstance(data, dict):
                continue

            for ann in data.get("annotations", []):
                if not isinstance(ann, dict):
                    continue
                if ann.get("kind") != "room":
                    continue
                if ann.get("stale", False):
                    continue
                room_name = str(ann.get("name", "")).strip()
                pts = ann.get("points")
                if not room_name or not isinstance(pts, list) or len(pts) < 3:
                    continue
                try:
                    poly = [(float(p[0]), float(p[1])) for p in pts]
                except (TypeError, IndexError, ValueError):
                    continue
                rooms.append((room_name, poly))
    except OSError as e:
        log.warning("remember: failed to scan annotations dir %s: %s",
                   annotations_dir, e)

    _annotations_cache[cache_key] = (max_mtime, rooms)
    if rooms:
        log.info("remember: loaded %d room(s) from %s (map_id=%r)",
                 len(rooms), annotations_dir, map_id or "any")
    return rooms


def _point_to_segment_distance(px: float, py: float,
                                x1: float, y1: float,
                                x2: float, y2: float) -> float:
    """Minimum distance from (px, py) to line segment (x1,y1)→(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _point_to_polygon_distance(x: float, y: float,
                                points: List[_Point]) -> float:
    """Minimum distance from (x, y) to a polygon's boundary edges."""
    pts = [(float(px), float(py)) for px, py in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return float("inf")
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        d = _point_to_segment_distance(x, y, x1, y1, x2, y2)
        if d < best:
            best = d
    return best


def _match_room(x: float, y: float,
                rooms: List[Tuple[str, List[_Point]]]) -> Tuple[str, bool, float]:
    """Find the best room match for a point.

    When the point is inside one or more room polygons the smallest-area
    containing room wins (matching Scene's ``get_robot_context``
    tie-breaking).  When the point is outside every room, the nearest room
    by boundary distance is returned.

    Returns ``(room_name, inside, distance_m)``.  ``distance_m`` is 0.0
    when inside; ``inside`` is True only when the point is within the
    polygon.  Returns ``("", False, 0.0)`` when no rooms are registered.
    """
    if not rooms:
        return "", False, 0.0

    # ── inside any room? ──────────────────────────────────────────────
    containing = []
    for name, poly in rooms:
        if _point_in_polygon(x, y, poly):
            containing.append((name, poly))
    if containing:
        containing.sort(key=lambda t: _polygon_area(t[1]))
        return containing[0][0], True, 0.0

    # ── outside all rooms — find nearest by boundary distance ─────────
    best_name, best_dist = "", float("inf")
    for name, poly in rooms:
        d = _point_to_polygon_distance(x, y, poly)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, False, best_dist


def _rule_based_tag_extraction(log_record: LogRecord,
                                spatial: Optional[SpatialContext],
                                annotations_dir: str = "") -> TagSet:
    """Extract TagSet using keyword matching on LogRecord.msg, augmented
    with coordinate-based room classification when scene annotation files
    are available.

    Phase1 rule-based extraction. Phase2: upgrade to LLM-based extraction.
    """
    msg_lower = log_record.msg.lower()
    tags = TagSet()

    # ── Region / scene_type from spatial coordinates ──────────────────
    # Always runs BEFORE keyword matching — when the annotation store
    # has room polygons, their names are authoritative for region and
    # scene_type.  For points inside a polygon the room name is used
    # directly; for points outside every polygon the nearest room +
    # distance is recorded (e.g. "距大厅 2.3m").
    #
    # Only annotations matching the current map_id are loaded (resolved
    # via SCENE_MAP_ID env → scene_maps/ → fallback to all files).
    rooms: list = []
    room_inside: bool = True
    room_dist: float = 0.0
    if spatial and spatial.objects:
        ad = annotations_dir or os.environ.get(
            "MEMGRAPH_ANNOTATIONS_DIR", _DEFAULT_ANNOTATIONS_DIR)
        map_id = _resolve_map_id()
        rooms = _load_room_annotations(ad, map_id=map_id)
        if rooms:
            first = spatial.objects[0]
            room_name, room_inside, room_dist = _match_room(first.x, first.y, rooms)
            if room_name:
                if room_inside:
                    tags.region = room_name
                    tags.scene_type = room_name
                else:
                    tags.region = f"距{room_name} {room_dist:.1f}m"
                    tags.scene_type = room_name

    # ── Scene type (keyword fallback) ──
    if not tags.scene_type:
        for scene, keywords in _SCENE_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    tags.scene_type = scene
                    break
            if tags.scene_type:
                break

    # ── Action type ──
    for action, keywords in _ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                tags.action_type = action
                break
        if tags.action_type:
            break

    # ── Task type ──
    for task, keywords in _TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                tags.task_type = task
                break
        if tags.task_type:
            break

    # ── Success ──
    tags.success = (log_record.level.lower() not in ("error", "warn"))

    # ── Objects from spatial context ──
    if spatial:
        for obj in spatial.objects:
            if obj.label and obj.label not in tags.objects_present:
                tags.objects_present.append(obj.label)

    # ── Difficulty (heuristic: message length + action complexity) ──
    msg_len = len(log_record.msg)
    if msg_len > 200:
        tags.difficulty = "hard"
    elif msg_len > 80:
        tags.difficulty = "medium"
    else:
        tags.difficulty = "easy"

    # ── Tool / source ──
    if log_record.tag:
        tags.tool_used = [log_record.tag]

    return tags, room_inside, room_dist


def _generate_summary(log_record: LogRecord,
                       spatial: Optional[SpatialContext],
                       scene_hint: str = "",
                       scene_inside: bool = True,
                       scene_distance: float = 0.0) -> str:
    """Template-based summary generation. Phase2: upgrade to LLM.

    Format: "[{action}] {success/failure} in {scene}: {key objects}"

    When ``scene_hint`` is provided (from coordinate-based room
    classification), it is used directly instead of keyword matching.
    When the point is outside every room (``scene_inside=False``) the
    summary renders a distance suffix (e.g. "… near 大厅 (2.3m)").
    """
    msg_lower = log_record.msg.lower()

    # Determine action
    action = "did something"
    for act, keywords in _ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                action = act
                break
        if action != "did something":
            break

    # Determine outcome
    outcome = "successfully" if log_record.level.lower() not in ("error", "warn") else "failed to"

    # Determine scene — coordinate hint beats keyword matching
    if scene_hint:
        if scene_inside:
            scene_phrase = f"in {scene_hint}"
        else:
            scene_phrase = f"near {scene_hint} ({scene_distance:.1f}m)"
    else:
        scene = "unknown area"
        for s, keywords in _SCENE_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    scene = s.replace("_", " ")
                    break
            if scene != "unknown area":
                break
        scene_phrase = f"in {scene}"

    # Objects
    obj_names: List[str] = []
    if spatial:
        obj_names = [o.label for o in spatial.objects if o.label]
    obj_str = ", ".join(obj_names) if obj_names else ""

    if obj_str:
        return f"{outcome} {action} {obj_str} {scene_phrase}"
    else:
        return f"{outcome} {action} {scene_phrase}"


# ── Pipeline ────────────────────────────────────────────────────────────

class RememberPipeline:
    """Orchestrate the remember (write) path across all storage layers.

    If an ``ImageStore`` is provided and the request ``kv`` contains
    ``image_base64``, the pipeline saves the decoded image to
    ``data/images/{node_id}/`` and populates ``MemoryNode.image_refs``.
    """

    def __init__(self, graph_store: GraphStore, tag_index: TagIndex,
                 vector_store: VectorStore,
                 image_store: "ImageStore | None" = None):
        self._graph = graph_store
        self._tags = tag_index
        self._vectors = vector_store
        self._images = image_store

    async def execute(self, request: RememberRequest) -> RememberResponse:
        """Execute the remember pipeline.

        Returns:
            RememberResponse with the new node_id.
        """
        log_record = request.log_record
        spatial = request.spatial

        # 1. Extract tags (includes coordinate-based room classification)
        tags, room_inside, room_dist = _rule_based_tag_extraction(log_record, spatial)

        # 2. Generate summary — prefer the scene_type from tag extraction
        summary = _generate_summary(log_record, spatial,
                                    scene_hint=tags.scene_type,
                                    scene_inside=room_inside,
                                    scene_distance=room_dist)

        # 3. Build MemoryNode (without node_id — GraphStore assigns it)
        now = time.time_ns()
        embedding_text = summary  # Phase1: embed the summary text
        embedding = self._vectors.encode(embedding_text, modality="text")

        node = MemoryNode(
            node_id=0,  # GraphStore will assign
            summary=summary,
            raw_log=log_record,
            timestamp=log_record.ts or now,
            spatial_data=spatial,
            tags=tags,
            weight=0.5,
            embedding=embedding,
            node_type=NodeType.SHORT_TERM,
            created_at=now,
            version=1,
        )

        # 4. Persist to GraphStore first to get node_id
        node_id = self._graph.add_node(node)
        node.node_id = node_id

        # 5. Save image if provided (top-level image_base64 or kv fallback)
        img_b64 = request.image_base64 or request.kv.get("image_base64", "")
        if self._images and img_b64:
            import base64 as _b64
            t_img = time.time()
            b64_len = len(img_b64)
            log.info("remember: decoding base64 image (%d chars) for node %d",
                     b64_len, node_id)
            try:
                img_bytes = _b64.b64decode(img_b64)
                saved_path = self._images.save(node_id, img_bytes)
                node.image_refs = self._images.list(node_id)
                # Persist image_refs to JSON — add_node() already wrote
                # a snapshot without them.  update_node() bumps the
                # version and flushes; the node object is the same
                # reference so the in-memory state is already correct.
                self._graph.update_node(node_id, node)
                img_ms = (time.time() - t_img) * 1000
                log.info("remember: node %d → saved image %s (%.1f KB, %dms)",
                         node_id, saved_path, len(img_bytes) / 1024, round(img_ms))
            except Exception as e:
                img_ms = (time.time() - t_img) * 1000
                log.warning("remember: node %d image save FAILED after %dms: %s: %s",
                           node_id, round(img_ms), type(e).__name__, e)

        # 6. Tags first into inverted index
        self._tags.insert(node_id, tags)

        # 7. Vector + BM25 index
        self._vectors.insert(node_id, embedding, summary)

        # 8. Causal edge
        if request.parent_node_id is not None:
            try:
                self._graph.add_edge(request.parent_node_id, node_id)
            except ValueError:
                log.warning("remember: parent_node_id %d not found, skipping edge",
                           request.parent_node_id)

        log.info("remember: node %d → \"%s\"", node_id, summary)
        return RememberResponse(node_id=node_id, message=f"Memory saved as node {node_id}")
