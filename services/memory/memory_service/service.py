"""Scribe Mem service (memgraph) — structured CKG memory for Robonix.

Runs as a robonix-api Service with MCP tool registration.
Entrypoint for rbnx boot: `python -m memory_service.service`

MCP tools (registered under robonix/service/memory/ alongside memsearch):
  - robonix/service/memory/remember       — write structured MemoryNode
  - robonix/service/memory/hybrid_search  — 3-stage hybrid retrieval (BM25+Embedding)
  - robonix/service/memory/vector_search  — BGE-M3 semantic search (Milvus)
  - robonix/service/memory/promote        — promote short-term → long-term

Runs in parallel with memsearch (save/search/compact) under the same
robonix/service/memory/ namespace so Pilot discovers both sets of tools.

Phase1: JSON-over-std_msgs/String (avoids protobuf codegen changes).
Phase2: structured IDL when contract schema is frozen.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# ── 0. Scribe logging setup ──────────────────────────────────────────────
from .scribe_log import setup_scribe_logging  # noqa: E402

_LOG_LEVEL = os.environ.get("MEMORY_LOG_LEVEL", "INFO").upper()
_LOG_DIR = os.environ.get("SCRIBE_LOG_DIR",
                           os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))

log = setup_scribe_logging(log_dir=_LOG_DIR, tag="service_memory", level=_LOG_LEVEL)

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

log.info("scribe_mem: starting")


def _log_environment() -> None:
    log.info("python   : %s (%s)", sys.version.replace("\n", " "), sys.executable)
    log.info("cwd      : %s", os.getcwd())
    log.info("embedding: enabled=%s (set MEMGRAPH_ENABLE_EMBEDDING=1 to activate)",
             _ENABLE_EMBEDDING)
    log.info("clean_start: enabled=%s (graph/images only; plan memory always preserved)",
             not _KEEP_DATA)
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", "")
    if model_path:
        log.info("embedding model path: %s (exists=%s)",
                 model_path, os.path.isdir(model_path))
    else:
        log.info("embedding model source: BGE-M3 ONNX (gpahal/bge-m3-onnx-int8)")


# ── 1. Core backend imports (no rbnx-api dependency) ───────────────────

from .core.types import (  # noqa: E402
    LogRecord, SpatialContext, ObjectCoord,
    TagFilter, TimeRange, RememberRequest, SearchRequest,
    CameraParams,
)
from .storage.graph_store import GraphStore  # noqa: E402
from .storage.tag_index import TagIndex  # noqa: E402
from .storage.vector_store import VectorStore  # noqa: E402
from .storage.embedding_config import EmbeddingModelConfig  # noqa: E402
from .storage.image_store import ImageStore  # noqa: E402
from .storage.mongo_backend import MongoBackend  # noqa: E402
from .storage.milvus_backend import MilvusBackend  # noqa: E402
from .storage.bge_embedder import BgeEmbedder  # noqa: E402
from .core.remember import RememberPipeline  # noqa: E402
from .core.retrieve import RetrievePipeline  # noqa: E402
from .core.compact import CompactPipeline  # noqa: E402

# ── 2. Backend construction ─────────────────────────────────────────────

# Phase1: local directory anchored to the package root (this file is at
# memory_service/service.py, so parent.parent = services/memory/).
# Override via AGENT_MEMORY_DIR env var to point elsewhere.
_DEFAULT_MEMORY_DIR = str(Path(__file__).resolve().parent.parent / "memory")
MEMORY_DIR = str(
    Path(os.environ.get("AGENT_MEMORY_DIR", _DEFAULT_MEMORY_DIR))
    .resolve()
)
IMAGES_DIR = str(Path(__file__).resolve().parent.parent / "data" / "images")

# ── 2a. Clean slate: wipe ephemeral data on every boot ────────────────
# The observation graph is ephemeral per session — delete the prior graph
# and all observation images so each rbnx boot starts from a clean state.
# Saved RTDL plan memory (ptdl_store.json) is deliberately preserved: it is
# long-term knowledge (successful plans to replay for repeated questions) and
# must survive a restart. Set MEMGRAPH_KEEP_DATA=1 to additionally preserve
# the observation graph (e.g. debugging persistence bugs).
_KEEP_DATA = os.environ.get("MEMGRAPH_KEEP_DATA", "0") in ("1", "true", "yes")

def _clean_slate() -> None:
    """Delete graph_store.json and all image directories from prior runs.

    ``ptdl_store.json`` is intentionally left intact — saved plan memory is
    long-term and must survive a restart (see ptdl_store.py).
    """
    import shutil
    graph_json = os.path.join(MEMORY_DIR, "graph_store.json")
    if os.path.exists(graph_json):
        os.remove(graph_json)
        log.info("scribe_mem: cleaned %s", graph_json)

    images_dir = IMAGES_DIR
    if os.path.isdir(images_dir):
        count = 0
        for entry in os.listdir(images_dir):
            entry_path = os.path.join(images_dir, entry)
            try:
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    os.remove(entry_path)
                count += 1
            except OSError as e:
                log.debug("scribe_mem: clean_images: %s: %s", entry, e)
        if count:
            log.info("scribe_mem: cleaned %d entries from %s", count, images_dir)

def _reset_memory_state() -> None:
    """Clear all in-memory state AND persisted files (runtime reset).

    Used by POST /reset — wipes the graph, indices, images, and PTDL store
    without restarting the service.  Safe to call at any time (e.g. start
    of a recording session).
    """
    import shutil

    # 1. Clear graph in-memory + persist empty state
    _graph.clear_all()

    # 2. Rebuild empty tag/vector indices
    _tags.rebuild([])
    _vectors.rebuild([])

    # 3. Wipe persisted files (same as _clean_slate)
    graph_json = os.path.join(MEMORY_DIR, "graph_store.json")
    if os.path.exists(graph_json):
        os.remove(graph_json)

    ptdl_json = os.path.join(MEMORY_DIR, "ptdl_store.json")
    if os.path.exists(ptdl_json):
        os.remove(ptdl_json)
    try:
        from .storage import ptdl_store as _ps
        _ps._ptdl_store_reset()
        # Re-create the singleton wired to the shared backends.
        _ps.get_ptdl_store(mongo=_mongo, milvus=_milvus, embedder=_bge)
    except Exception:
        pass
    # PTDL deletion (explicit /reset) also clears plan vectors + Mongo plans.
    if _milvus is not None:
        _milvus.clear_kind("plan")
    if _mongo is not None:
        _mongo.clear_plans()

    images_dir = str(Path(__file__).resolve().parent.parent / "data" / "images")
    if os.path.isdir(images_dir):
        count = 0
        for entry in os.listdir(images_dir):
            entry_path = os.path.join(images_dir, entry)
            try:
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    os.remove(entry_path)
                count += 1
            except OSError as e:
                log.debug("scribe_mem: reset_images: %s: %s", entry, e)
        if count:
            log.info("scribe_mem: reset: cleaned %d entries from %s", count, images_dir)

    log.info("scribe_mem: reset complete — graph=%d nodes, tags=%d, vectors=%d",
             _graph.count(), _tags.count(), _vectors.count())


if not _KEEP_DATA:
    _clean_slate()

# ── 2b. Embedding toggle ──────────────────────────────────────────────
# Embedding (sentence-transformers / all-MiniLM-L6-v2) is OFF by default.
# Set MEMGRAPH_ENABLE_EMBEDDING=1 to load the model and enable the
# BM25+Cosine hybrid ranking path (Path B in retrieve.py).
_ENABLE_EMBEDDING = os.environ.get("MEMGRAPH_ENABLE_EMBEDDING", "0") in ("1", "true", "yes")

graph_type = "GraphStore"
idx_type = "TagIndex"
vec_type = "VectorStore(BM25 only)" if not _ENABLE_EMBEDDING else "VectorStore(BM25+Embedding)"
log.info("scribe_mem: graph_store → %s", os.path.join(MEMORY_DIR, "graph_store.json"))
log.info("scribe_mem: scribe_log  → %s", os.path.join(_LOG_DIR, "service_memory.log"))
log.info("scribe_mem: backend     → %s + %s + %s + ImageStore", graph_type, idx_type, vec_type)

_mongo = MongoBackend()
_milvus = MilvusBackend()
_bge = BgeEmbedder(EmbeddingModelConfig())

# Clean-start wipes the ephemeral scene data in every backend (Mongo scene
# nodes/edges + Milvus scene vectors). Plan memory (plan_tasks / kind="plan")
# is long-term and deliberately preserved.
if not _KEEP_DATA:
    _mongo.clear_scene()
    _mongo.clear_edges()
    _milvus.clear_kind("scene")

_graph = GraphStore(data_dir=MEMORY_DIR, mongo=_mongo, milvus=_milvus, embedder=_bge)
_tags = TagIndex()
_vectors = VectorStore(alpha=0.3, embedding_enabled=_ENABLE_EMBEDDING,
                       embedder=_bge if _ENABLE_EMBEDDING else None)
_images = ImageStore(mongo=_mongo)
_remember_pipe = RememberPipeline(_graph, _tags, _vectors, _images)
_retrieve_pipe = RetrievePipeline(_graph, _tags, _vectors, image_store=_images)
_compact_pipe = CompactPipeline(_graph)

# Eagerly create the PTDL singleton wired to Mongo/Milvus so plan vectors
# join vector_search; later get_ptdl_store() calls return this instance.
from .storage.ptdl_store import get_ptdl_store as _get_ptdl_store  # noqa: E402
_ptdl = _get_ptdl_store(mongo=_mongo, milvus=_milvus, embedder=_bge)

# ── 2c. Scene Hook HTTP server ───────────────────────────────────────
# Must start at module level — rbnx boot calls robonix_api.Service.run()
# directly, bypassing main().  The server listens for raw JSON POSTs
# from Scene's ObjectWatchdog and Scene Hook (mcp_tools.py).
# Called AFTER _start_scene_hook_server is defined below.
# See line ~640 for the actual call.

# ── 3. Standalone API (MemoryService) — works without rbnx ─────────────


class MemoryService:
    """Scribe Mem service — manages the full memory lifecycle.

    Works standalone (for scripts) and also used by the rbnx MCP entrypoint.
    When data_dir is provided, creates independent stores + pipelines.
    """

    def __init__(self, data_dir: str = "",
                 embedding_config=None, alpha: float = 0.3):
        if data_dir:
            self.graph = GraphStore(data_dir=data_dir)
            self.tags = TagIndex()
            ecfg = embedding_config or EmbeddingModelConfig()
            self.vectors = VectorStore(config=ecfg, alpha=alpha,
                                      embedding_enabled=_ENABLE_EMBEDDING)
        else:
            self.graph = _graph
            self.tags = _tags
            self.vectors = _vectors

        # Always create own pipelines bound to own stores
        self._images = ImageStore()
        self._remember_pipe = RememberPipeline(self.graph, self.tags, self.vectors, self._images)
        self._retrieve_pipe = RetrievePipeline(self.graph, self.tags, self.vectors)
        self._compact_pipe = CompactPipeline(self.graph)

    @property
    def is_initialized(self) -> bool:
        return True

    async def init(self, _cfg=None) -> str:
        nodes = [self.graph.get_node(nid) for nid in self.graph.all_ids()]
        nodes = [n for n in nodes if n is not None]
        self.tags.rebuild(nodes)
        vec_triples = [(n.node_id, n.embedding, n.summary) for n in nodes]
        self.vectors.rebuild(vec_triples)
        log.info("init: rebuilt indices for %d nodes", len(nodes))
        return "ok"

    async def remember(self, session_id: str, plan_id: str,
                       log_record, spatial=None, parent_node_id=None,
                       kv=None):
        req = RememberRequest(
            session_id=session_id, plan_id=plan_id, log_record=log_record,
            spatial=spatial, parent_node_id=parent_node_id, kv=kv or {},
        )
        return await self._remember_pipe.execute(req)

    async def search(self, query: str, tags=None, top_k: int = 5,
                     alpha=None, time_range=None, require_executable: bool = False):
        req = SearchRequest(
            query=query, tags=tags, top_k=top_k, alpha=alpha,
            time_range=time_range, require_executable=require_executable,
        )
        return await self._retrieve_pipe.execute(req)

    async def compact(self):
        return await self._compact_pipe.execute()

    async def remember_from_log(self, session_id: str, plan_id: str,
                                level: str, tag: str, msg: str,
                                objects=None, parent_node_id=None,
                                spatial_origin: str = ""):
        """Build a memory record while preserving explicit spatial origin."""
        lr = LogRecord(level=level, tag=tag, msg=msg)
        spatial = None
        if objects:
            spatial = SpatialContext(
                objects=[ObjectCoord(obj_id=o[0], label=o[1],
                                     x=o[2] if len(o)>2 else 0,
                                     y=o[3] if len(o)>3 else 0,
                                     z=o[4] if len(o)>4 else 0)
                         for o in objects],
                origin=spatial_origin,
            )
        resp = await self.remember(session_id, plan_id, lr,
                                   spatial=spatial, parent_node_id=parent_node_id)
        return resp.node_id

    async def search_as_dicts(self, query: str, top_k: int = 5,
                              tag_filter=None):
        tf = TagFilter.from_dict(tag_filter) if tag_filter else None
        resp = await self.search(query, tags=tf, top_k=top_k)
        return [n.to_dict() for n in resp.nodes]


# ── 4. rbnx MCP entrypoint ──────────────────────────────────────────────
# Register MCP tools via robonix-api at module level so get_type_hints()
# can resolve String/Empty from globalns. Wrapped in try/except so the
# MemoryService class is still importable without robonix_api installed.

_MCP_AVAILABLE = False
try:
    from robonix_api import Service, Ok  # noqa: E402, F811
    from std_msgs_mcp import Empty, String  # noqa: E402
    _MCP_AVAILABLE = True
except ImportError:
    log.info("robonix_api not available — MCP server disabled "
             "(standalone/script mode OK)")


if _MCP_AVAILABLE:
    _memory_svc = Service(id="memgraph", namespace="robonix/service/memory")
    _indices_initialized = False

    # ── JSON response helpers ──────────────────────────────────────────

    def _json_ok(obj: dict) -> String:
        """Encode a success response dict as JSON-over-String."""
        return String(data=json.dumps(obj, ensure_ascii=False))

    def _json_error(message: str, *, ctx: str = "") -> String:
        """Encode an error response as JSON-over-String."""
        if ctx:
            log.warning("MCP %s error: %s", ctx, message)
        else:
            log.warning("MCP error: %s", message)
        return String(data=json.dumps({"error": message}))

    def _parse_json(data: str, *, ctx: str = "") -> dict | None:
        """Parse JSON from the wire, returning None on failure (error already emitted)."""
        try:
            return json.loads(data)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            log.warning("MCP %s: invalid JSON (%d chars): %s",
                        ctx, len(data), e)
            return None

    def _relative_window_ns(relative: str) -> tuple:
        """Translate a relative-time keyword into a ``(start_ts, end_ts)``
        nanosecond window aligned to local calendar boundaries.

        Consistent with ``time.time_ns()`` (UTC epoch ns), so it filters the
        ``timestamp_ns`` values written by ptdl_store.  Supports the English
        keywords the VLM is told to use plus the common Chinese equivalents.
        """
        import datetime
        now = datetime.datetime.now()
        today_start = datetime.datetime.combine(now.date(), datetime.time.min)
        rel = (relative or "").strip().lower()
        if rel in ("today", "今天", "今日"):
            start, end = today_start, now
        elif rel in ("yesterday", "昨天", "昨日"):
            start = today_start - datetime.timedelta(days=1)
            end = today_start
        elif rel in ("last_week", "last week", "past_week", "上周"):
            start = now - datetime.timedelta(days=7)
            end = now
        elif rel in ("last_month", "last month", "上月"):
            start = now - datetime.timedelta(days=30)
            end = now
        else:
            raise ValueError(
                f"unsupported relative_time: {relative!r} "
                "(use today / yesterday / last_week)"
            )
        return int(start.timestamp() * 1e9), int(end.timestamp() * 1e9)

    def _lean_node(node) -> dict:
        """Serialize a node for vector_search — no images/frame_ts/embedding."""
        d = node.to_dict()
        for k in ("embedding", "image_refs", "frame_ts",
                  "depth_refs", "video_clip_refs"):
            d.pop(k, None)
        return d

    # ── MCP tool handlers ──────────────────────────────────────────────

    @_memory_svc.mcp("robonix/service/memory/remember")
    async def _mcp_remember(msg: String) -> String:
        """Write a structured memory node into the CKG (Causal Knowledge Graph).

        Observations are saved AUTOMATICALLY — when scene.list_objects detects
        objects, the Scene service captures the camera frame and saves
        everything to memory.  Do NOT call remember after list_objects
        or camera_snapshot — it is already handled for you.

        Use remember only for EXPLICIT saves: user preferences ("the user
        likes the red cup"), action results ("grasp succeeded"), navigation
        events, or any fact the robot should recall later.

        Each node is automatically tagged across four dimensions (spatial /
        behavioural / cognitive / optimisation) and embedded for hybrid
        BM25+vector search.

        Request JSON schema:
        {
          "session_id":    "<session uuid>",           // required
          "plan_id":       "<plan uuid>",              // required
          "log_record": {                              // required
            "ts":    1765432100123456789,  // ns timestamp (0 = now)
            "level": "Info",               // Debug|Info|Warn|Error
            "tag":   "exec",               // source provider_id
            "msg":   "grasped red cup in kitchen"
          },
          "spatial": {                               // optional
            "origin":  "frame_from_scene",
            "objects": [
              {"obj_id": "scene.obj.cup_001", "label": "red cup",
               "x": 1.0, "y": 2.0, "z": 0.8}
            ]
          },
          "parent_node_id": 5,           // optional — causal parent
          "image_base64": "<base64>",    // optional — manual image attach
          "depth_base64": "<base64>",    // optional — depth frame (P0: logged, not persisted)
          "camera_params": {             // optional — capture intrinsics + pose
            "fx": 554.0, "fy": 554.0, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480,
            "depth_scale": 0.001, "camera_type": "rgb",
            "camera_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
          },
          "time_range": {                 // optional — observation window (ns)
            "start_ts": 1765432100123456789,
            "end_ts":   1765432100123456789
          },
          "kv": {}                        // optional — extra metadata
        }

        Response JSON: {"node_id": 42, "message": "Memory saved as node 42"}
        On error:       {"error": "<description>"}
        """
        req_dict = _parse_json(msg.data, ctx="remember")
        if req_dict is None:
            return _json_error("Invalid JSON in request body", ctx="remember")

        session_id = req_dict.get("session_id")
        plan_id = req_dict.get("plan_id")
        b64_len = len(req_dict.get("image_base64", ""))
        n_objs = len((req_dict.get("spatial") or {}).get("objects", []))
        log.info("API remember: sid=%s pid=%s tag=%s msg=\"%s\" (%s chars) "
                 "has_image=%s b64len=%d objects=%d",
                 session_id, plan_id,
                 req_dict.get("log_record", {}).get("tag", "?"),
                 (req_dict.get("log_record", {}).get("msg", "") or "")[:60],
                 str(len(req_dict.get("log_record", {}).get("msg", ""))),
                 "yes" if req_dict.get("image_base64") else "no",
                 b64_len, n_objs)

        if not session_id or not plan_id:
            return _json_error(
                "Missing required fields: session_id and plan_id must be non-empty strings",
                ctx="remember",
            )

        raw = req_dict.get("log_record")
        if not raw or not isinstance(raw, dict):
            return _json_error("Missing required field: log_record must be a dict",
                              ctx="remember")

        try:
            lr = LogRecord.from_dict(raw)
        except Exception as e:
            return _json_error(f"Invalid log_record: {e}", ctx="remember")

        spatial = None
        if req_dict.get("spatial"):
            try:
                spatial = SpatialContext.from_dict(req_dict["spatial"])
            except Exception as e:
                return _json_error(f"Invalid spatial: {e}", ctx="remember")

        camera_params = None
        if req_dict.get("camera_params"):
            try:
                camera_params = CameraParams.from_dict(req_dict["camera_params"])
            except Exception as e:
                return _json_error(f"Invalid camera_params: {e}", ctx="remember")

        time_range = None
        if req_dict.get("time_range"):
            try:
                tr = req_dict["time_range"]
                time_range = TimeRange(
                    start_ts=int(tr.get("start_ts", 0)),
                    end_ts=int(tr.get("end_ts", 0)),
                )
            except (TypeError, ValueError) as e:
                return _json_error(f"Invalid time_range: {e}", ctx="remember")

        request = RememberRequest(
            session_id=session_id,
            plan_id=plan_id,
            log_record=lr,
            spatial=spatial,
            parent_node_id=req_dict.get("parent_node_id"),
            image_base64=req_dict.get("image_base64", ""),
            depth_base64=req_dict.get("depth_base64", ""),
            camera_params=camera_params,
            time_range=time_range,
            kv=req_dict.get("kv") if isinstance(req_dict.get("kv"), dict) else {},
        )
        resp = await _remember_pipe.execute(request)
        node = _graph.get_node(resp.node_id)
        summary = node.summary if node else "?"
        log.info("API remember: sid=%s pid=%s node_id=%s summary=\"%s\"",
                 request.session_id, request.plan_id, str(resp.node_id), summary)
        return _json_ok({"node_id": resp.node_id, "message": resp.message})

    @_memory_svc.mcp("robonix/service/memory/hybrid_search")
    async def _mcp_search(msg: String) -> String:
        """Search the CKG memory using a 3-stage pipeline: tag filter → hybrid
        BM25+embedding ranking → causal/time/weight filter.

        Returns ONLY memories that were previously saved via remember.  If no
        observations have been persisted (e.g. you looked but forgot to call
        remember), this will return empty results.

        Tags are used for O(1) inverted-index pre-filtering BEFORE vector
        ranking.  Always set as many tag fields as the user's intent
        implies — an empty TagFilter searches ALL nodes, which is slow
        and noisy.

        Request JSON schema:
        {
          "query":    "red cup in kitchen",          // required — semantic query
          "top_k":    5,                              // default 5
          "alpha":    0.3,                            // optional — BM25 weight (1.0=pure keyword, 0.0=pure semantic)
          "tags": {                                    // optional — all fields optional (AND semantics)
            "scene_type":     "kitchen",               // kitchen|living_room|workshop|bedroom|outdoor
            "objects":        ["red cup"],             // intersection match
            "action_type":    "grasp",                 // grasp|place|navigate|craft|observe
            "success":        true,                    // true = only successes, false = only failures
            "task_type":      "fetch",                 // fetch|build|explore|dialogue
            "difficulty_max": "medium"                 // easy|medium|hard — filter ≤ this level
          },
          "time_range": {                              // optional — ns timestamps
            "start_ts": 1765432100000000000,
            "end_ts":   1765432200000000000             // 0 = no upper bound
          },
          "require_executable": false,                  // default false (Phase2)
          "vlm_qa": false                                // if true, return VLM answer from node images
        }

        Response JSON:
        {
          "nodes": [
            ...
          ],
          "vlm_answer": "There was a red cup on the kitchen counter."  // only when vlm_qa=true
        }
            {
              "node_id": 42,
              "summary": "successfully grasp red cup in kitchen",
              "tags": {"scene_type": "kitchen", "action_type": "grasp", ...},
              "spatial_data": {"objects": [...], "origin": "frame_from_scene"},
              "weight": 0.5,
              "timestamp": 1765432100123456789,
              "raw_log": {"ts": ..., "level": "Info", "tag": "exec", "msg": "..."},
              "causal_chain": [...],
              "embedding": [...]
            }
          ]
        }
        On error: {"error": "<description>", "nodes": []}
        """
        req_dict = _parse_json(msg.data, ctx="search")
        if req_dict is None:
            return _json_error("Invalid JSON in request body", ctx="search")

        query = req_dict.get("query")
        log.info("API search: query=%r tags=%s top_k=%s alpha=%s time_range=%s",
                 (query or "")[:80],
                 "set" if req_dict.get("tags") else "none",
                 req_dict.get("top_k", "5"),
                 req_dict.get("alpha", "default"),
                 "set" if req_dict.get("time_range") else "none")

        if not query or not isinstance(query, str) or not query.strip():
            return _json_error("Missing or empty required field: query", ctx="search")

        tags = None
        if req_dict.get("tags"):
            try:
                tags = TagFilter.from_dict(req_dict["tags"])
            except Exception as e:
                return _json_error(f"Invalid tags: {e}", ctx="search")

        time_range = None
        if req_dict.get("time_range"):
            try:
                tr = req_dict["time_range"]
                time_range = TimeRange(
                    start_ts=int(tr.get("start_ts", 0)),
                    end_ts=int(tr.get("end_ts", 0)),
                )
            except (TypeError, ValueError) as e:
                return _json_error(f"Invalid time_range: {e}", ctx="search")

        try:
            top_k = int(req_dict.get("top_k", 5))
        except (TypeError, ValueError):
            return _json_error("Invalid top_k: must be an integer", ctx="search")

        try:
            alpha = float(req_dict["alpha"]) if req_dict.get("alpha") is not None else None
        except (TypeError, ValueError):
            return _json_error("Invalid alpha: must be a float between 0.0 and 1.0",
                              ctx="search")

        request = SearchRequest(
            query=query.strip(), tags=tags, top_k=max(1, min(top_k, 100)),
            alpha=alpha, time_range=time_range,
            require_executable=bool(req_dict.get("require_executable", False)),
            vlm_qa=bool(req_dict.get("vlm_qa", False)),
        )
        resp = await _retrieve_pipe.execute(request)
        log.info("API search: query=\"%s\" tags=%s top_k=%s alpha=%s vlm_qa=%s → %s results",
                 request.query[:80],
                 request.tags.to_dict() if request.tags else "{}",
                 str(request.top_k), str(request.alpha), str(request.vlm_qa),
                 str(len(resp.nodes)))
        result = {"nodes": [n.to_dict() for n in resp.nodes]}
        if resp.vlm_answer:
            result["vlm_answer"] = resp.vlm_answer
        return _json_ok(result)

    @_memory_svc.mcp("robonix/service/memory/promote")
    async def _mcp_compact(msg: Empty) -> String:
        """Promote overflowing ShortTerm memory nodes (oldest first) to LongTerm.

        The compaction threshold defaults to 50 ShortTerm nodes.  When the
        count exceeds the threshold the oldest ShortTerm nodes are promoted
        to the LongTerm ID range (1000+).  This keeps the retrieval index
        fast — ShortTerm nodes are a small hot set; LongTerm nodes are the
        persistent archive.

        Takes no parameters (Empty trigger).

        Response JSON:
        {
          "summary":          "Compacted 7 short-term nodes → long-term (50 remaining, 65 total).",
          "nodes_compacted":  7
        }
        """
        _ = msg
        log.info("API compact: trigger (%d nodes total, %d tags indexed)",
                 _graph.count(), _tags.count())
        resp = await _compact_pipe.execute()
        log.info("API compact: done — %d compacted (%d total)",
                 resp.nodes_compacted, _graph.count())
        return _json_ok({"summary": resp.summary, "nodes_compacted": resp.nodes_compacted})

    @_memory_svc.mcp("robonix/service/memory/ptdl_remember")
    async def _mcp_ptdl_remember(msg: String) -> String:
        """Save a successful RTDL plan to the standalone plan store (ptdl_store.json).

        Plans are stored INDEPENDENTLY of the CKG memory graph — they survive
        graph clean_start wipes and do not participate in graph search/compact.

        Request JSON schema:
        {
          "query":       "去厨房拿可乐",                 // required — user's original request
          "description": "navigate→grasp→return",         // optional — plan summary
          "steps": [                                      // required — ordered plan steps
            "1. [nav] navigate to kitchen",
            "2. [gripper] grasp cola",
            "3. [nav] return"
          ],
          "plan_count":    1,                              // optional — number of RTDL trees
          "canceled_count": 0,                             // optional — canceled tree count
          "rtdl_plan":     "{...}",                        // optional — full Plan AST JSON sent to Executor
          "raw_rtdl":      "{...}",                        // optional — raw LLM RTDL JSON before expansion
          "plan_id":       "1"                             // optional — per-round id; the child key in the query→sub-plans tree
        }

        Response JSON: {"ok": true, "message": "Plan saved: \"去厨房拿可乐\" (3 steps)"}
        """
        req_dict = _parse_json(msg.data, ctx="ptdl_remember")
        if req_dict is None:
            return _json_error("Invalid JSON", ctx="ptdl_remember")

        query = (req_dict.get("query") or "").strip()
        if not query:
            return _json_error("Missing required field: query", ctx="ptdl_remember")

        steps = req_dict.get("steps")
        if not steps or not isinstance(steps, list):
            steps = []

        from .storage.ptdl_store import get_ptdl_store
        ptdl = get_ptdl_store()
        ptdl.add(
            query=query,
            description=req_dict.get("description", ""),
            steps=steps,
            plan_count=int(req_dict.get("plan_count", 1)),
            canceled_count=int(req_dict.get("canceled_count", 0)),
            rtdl_plan=req_dict.get("rtdl_plan"),
            raw_rtdl=req_dict.get("raw_rtdl"),
            plan_id=req_dict.get("plan_id"),
        )
        log.info("API ptdl_remember: \"%s\" (%d steps)", query, len(steps))
        return _json_ok({"ok": True, "message": f"Plan saved: \"{query}\" ({len(steps)} steps)"})

    @_memory_svc.mcp("robonix/service/memory/ptdl_retrieve")
    async def _mcp_ptdl_retrieve(msg: String) -> String:
        """Search previously saved RTDL plans by keyword match.

        Returns the top-N matching plans from ptdl_store.json.  Does NOT
        search the main CKG graph — only standalone plan records.

        Explicit opt-in: set ``ROBONIX_MEMORY_PREFETCH_ENABLED=1`` to enable
        this path; when disabled (default), Pilot skips plan retrieval entirely.

        Request JSON schema:
        {
          "query":  "去厨房",     // required — search query
          "top_k":  5              // optional — max results (default 5)
        }

        Response JSON:
        {
          "plans": [
            {
              "query": "去厨房拿可乐",
              "description": "navigate→grasp→return",
              "steps": ["1. [nav] navigate to kitchen", ...],
              "timestamp_ns": 1785600000000000000,
              "plan_count": 1,
              "canceled_count": 0
            }
          ]
        }
        """
        req_dict = _parse_json(msg.data, ctx="ptdl_retrieve")
        if req_dict is None:
            return _json_error("Invalid JSON", ctx="ptdl_retrieve")

        query = (req_dict.get("query") or "").strip()
        if not query:
            return _json_error("Missing required field: query", ctx="ptdl_retrieve")

        try:
            top_k = max(1, min(int(req_dict.get("top_k", 5)), 20))
        except (TypeError, ValueError):
            return _json_error("Invalid top_k", ctx="ptdl_retrieve")

        from .storage.ptdl_store import get_ptdl_store
        ptdl = get_ptdl_store()
        plans = ptdl.search(query=query, top_k=top_k)
        log.info("API ptdl_retrieve: \"%s\" → %d plans", query[:80], len(plans))
        return _json_ok({"plans": plans})

    @_memory_svc.mcp("robonix/service/memory/search_history")
    async def _mcp_search_history(msg: String) -> String:
        """Search the robot's task history by time — answer questions like
        "我昨天/今天/上周做过什么任务？" (what tasks did I do …).

        Reads the standalone plan store (ptdl_store.json), where each record
        is one user task (its ``query``) saved with a ``timestamp_ns``.
        Returns a summary list (query / description / timestamp_ns /
        plan_count / canceled_count), most-recent-first.  It does NOT search
        the observation CKG graph.

        Time window — give exactly one of:
        - ``relative_time``: "today" | "yesterday" | "last_week" (also
          accepts "last_month").  Preferred: the server converts the keyword
          to a nanosecond window for you.  Map the user's words directly:
          "今天" → "today", "昨天" → "yesterday", "上周/最近一周" → "last_week".
        - ``time_range``: {"start_ts": <ns>, "end_ts": <ns>} — explicit
          nanosecond window; ``end_ts`` 0 means "until now".  Used only when
          you already have concrete epoch-ns timestamps.

        Request JSON schema (both time fields optional — omit both to list
        all tasks):
        {
          "relative_time": "yesterday"
          // or: "time_range": {"start_ts": 1792389000000000000, "end_ts": 0}
        }

        Response JSON:
        {
          "tasks": [
            {
              "query": "去厨房拿水杯",
              "description": "navigate→grasp→return",
              "timestamp_ns": 1792389521574177139,
              "plan_count": 18,
              "canceled_count": 0
            }
          ]
        }
        On error: {"error": "<description>"}
        """
        req_dict = _parse_json(msg.data, ctx="search_history")
        if req_dict is None:
            return _json_error("Invalid JSON in request body", ctx="search_history")

        start_ts = 0
        end_ts = 0
        tr = req_dict.get("time_range")
        rel = req_dict.get("relative_time")
        # The VLM sometimes puts the relative keyword inside ``time_range``
        # (e.g. {"time_range": "yesterday"}) instead of ``relative_time``.
        # Treat a bare string there as a relative keyword too.
        if isinstance(tr, str) and tr.strip():
            rel = tr
            tr = None
        if isinstance(tr, dict):
            try:
                start_ts = int(tr.get("start_ts", 0))
                end_ts = int(tr.get("end_ts", 0))
            except (TypeError, ValueError) as e:
                return _json_error(f"Invalid time_range: {e}", ctx="search_history")
        elif rel:
            try:
                start_ts, end_ts = _relative_window_ns(str(rel))
            except ValueError as e:
                return _json_error(str(e), ctx="search_history")

        from .storage.ptdl_store import get_ptdl_store
        ptdl = get_ptdl_store()
        entries = ptdl.search_by_time(start_ts=start_ts, end_ts=end_ts)

        tasks = [
            {
                "query": e.get("query", ""),
                "description": e.get("description", ""),
                "timestamp_ns": int(e.get("timestamp_ns") or 0),
                "plan_count": int(e.get("plan_count") or 0),
                "canceled_count": int(e.get("canceled_count") or 0),
            }
            for e in entries
        ]
        log.info("API search_history: window=[%d, %d] → %d tasks",
                 start_ts, end_ts, len(tasks))
        return _json_ok({"tasks": tasks})

    @_memory_svc.mcp("robonix/service/memory/vector_search")
    async def _mcp_vector_search(msg: String) -> String:
        """Semantic (vector) search across scene memories and saved plans.

        Runs a BGE-M3 ANN query against the Milvus ``memory_vectors`` index
        and returns the matching Mongo docs (scene nodes lean — no images /
        frame_ts / embedding — or plan records).  Use this for meaning-based
        recall ("find the scene where I placed the cup"); use hybrid_search
        for tag-filtered retrieval.

        Request JSON schema:
        {
          "query":    "去厨房拿可乐",            // required — semantic query
          "kind":     "all",                      // scene | plan | all (default all)
          "top_k":    5,                          // default 5
          "node_type": "short_term",              // optional — scene node_type filter
          "time_range": {"start_ts": <ns>, "end_ts": <ns>}  // optional
        }

        kind "all" (default) runs two INDEPENDENT top-k — scene objects and
        historical plans live in different semantic spaces, so they are never
        merged into one ranking; each kind returns its own top_k results:

        Response JSON (kind="all"):
        {
          "scene": {"count": 5, "results": [
            {"doc_id": "7", "kind": "scene", "node_type": "short_term",
             "timestamp": 1765432100123456789, "score": 0.83, "node": { ... }}, ...]},
          "plan": {"count": 5, "results": [
            {"doc_id": "a1b2...", "kind": "plan", "score": 0.71, "plan": { ... }}, ...]}
        }

        kind "scene" / "plan" return a single flat list:
        {
          "count": 2,
          "results": [ ...same item shape as above... ]
        }
        On error: {"error": "<description>"}
        """
        req_dict = _parse_json(msg.data, ctx="vector_search")
        if req_dict is None:
            return _json_error("Invalid JSON in request body", ctx="vector_search")

        query = (req_dict.get("query") or "").strip()
        if not query:
            return _json_error("Missing required field: query", ctx="vector_search")

        kind = (req_dict.get("kind") or "all").strip().lower()
        if kind not in ("scene", "plan", "all"):
            return _json_error("Invalid kind: must be scene | plan | all",
                              ctx="vector_search")

        try:
            top_k = max(1, min(int(req_dict.get("top_k", 5)), 100))
        except (TypeError, ValueError):
            return _json_error("Invalid top_k: must be an integer", ctx="vector_search")

        node_type = (req_dict.get("node_type") or "").strip() or None
        time_range = None
        if req_dict.get("time_range"):
            try:
                tr = req_dict["time_range"]
                time_range = (int(tr.get("start_ts", 0)), int(tr.get("end_ts", 0)))
            except (TypeError, ValueError):
                return _json_error("Invalid time_range", ctx="vector_search")

        qvec = _bge.encode(query)

        def _enrich(hits: list) -> list:
            results: list = []
            for h in hits:
                item = {
                    "doc_id": h["doc_id"],
                    "kind": h["kind"],
                    "node_type": h["node_type"],
                    "timestamp": h["timestamp"],
                    "score": round(h["score"], 4),
                }
                if h["kind"] == "scene" and h["doc_id"].isdigit():
                    node = _graph.get_node(int(h["doc_id"]))
                    if node is None and _mongo.available:
                        node = _mongo.get_node(int(h["doc_id"]))
                    if node is not None:
                        item["node"] = _lean_node(node)
                elif h["kind"] == "plan":
                    plan = _mongo.get_plan_by_key(h["doc_id"]) if _mongo.available else None
                    if plan is None:
                        plan = _ptdl.get_by_query_key(h["doc_id"])
                    if plan is not None:
                        item["plan"] = plan
                results.append(item)
            return results

        if kind == "all":
            # Scene objects and historical plans are different semantic
            # spaces; a single merged ranking would make their scores
            # non-comparable.  Run one ANN search per kind, each truncated to
            # its own top_k, and return the two lists separately.
            scene_hits = _milvus.search(
                qvec,
                top_k=top_k,
                kind="scene",
                node_type=node_type,
                time_range=time_range,
            )
            plan_hits = _milvus.search(
                qvec,
                top_k=top_k,
                kind="plan",
                node_type=None,
                time_range=time_range,
            )
            scene_results = _enrich(scene_hits)
            plan_results = _enrich(plan_hits)
            log.info("API vector_search: query=%r kind=all top_k=%d → scene=%d plan=%d",
                     query[:80], top_k, len(scene_results), len(plan_results))
            return _json_ok({
                "scene": {"count": len(scene_results), "results": scene_results},
                "plan": {"count": len(plan_results), "results": plan_results},
            })

        hits = _milvus.search(
            qvec,
            top_k=top_k,
            kind=kind,
            node_type=node_type,
            time_range=time_range,
        )
        results = _enrich(hits)

        log.info("API vector_search: query=%r kind=%s top_k=%d → %d hits",
                 query[:80], kind, top_k, len(results))
        return _json_ok({"count": len(results), "results": results})

    @_memory_svc.on_init
    def _on_init(cfg):
        _ = cfg
        global _indices_initialized
        log.info("on_init: MEMORY_DIR=%s initialized=%s", MEMORY_DIR, _indices_initialized)

        # Idempotent guard: only rebuild indices once per process lifetime.
        # Subsequent chat sessions trigger on_init again, but the in-memory
        # indices (_tags, _vectors) already contain all data written by
        # previous remember calls. Rebuilding from _graph would destroy
        # concurrent writes (clear-then-rebuild is not atomic).
        if _indices_initialized:
            log.info("on_init: indices already initialized (%d nodes), skipping rebuild",
                     _tags.count())
            return Ok()

        try:
            nodes = [_graph.get_node(nid) for nid in _graph.all_ids()]
            nodes = [n for n in nodes if n is not None]
            _tags.rebuild(nodes)
            vec_triples = [(n.node_id, n.embedding, n.summary) for n in nodes]
            _vectors.rebuild(vec_triples)

            # Rebuild the Milvus vector index from Mongo/JSON (re-embedding
            # scene nodes + plans) so a fresh/empty Milvus catches up.
            n_scene = _graph.rebuild_milvus()
            n_plan = _ptdl.rebuild_milvus()
            _indices_initialized = True
            log.info("on_init: indices rebuilt for %d nodes; milvus=%d scene + %d plan",
                     len(nodes), n_scene, n_plan)
        except Exception as e:
            log.warning("on_init: index rebuild failed (empty store?): %s: %s",
                        type(e).__name__, e)
        return Ok()


# ── 5. Entrypoint ───────────────────────────────────────────────────────

def main() -> int:
    """Run the MCP server (requires robonix_api)."""
    if not _MCP_AVAILABLE:
        log.error("robonix_api not available — cannot start MCP server")
        return 1

    _log_environment()
    log.info("memory_dir = %s", MEMORY_DIR)
    log.info("backend: GraphStore(%d nodes), embedding=%s (enabled=%s)",
             _graph.count(),
             _vectors._config.model_name if _ENABLE_EMBEDDING else "disabled",
             _ENABLE_EMBEDDING)
    # ── Scene Hook HTTP endpoint (raw, no MCP session needed) ─────
    _start_scene_hook_server()

    log.info("scribe_mem service ready; entering memory_svc.run()")
    _memory_svc.run()
    return 0


_scene_hook_started = False

def _start_scene_hook_server() -> None:
    """Start a tiny HTTP server on port 37798 for Scene Hook direct POST.

    FastMCP's Streamable HTTP transport requires SSE session negotiation
    which is too complex for a simple service-to-service Hook.  This
    endpoint accepts raw JSON POST directly — no MCP protocol overhead.

    Idempotent: safe to call multiple times (e.g. module-level + main()).
    """
    global _scene_hook_started
    if _scene_hook_started:
        return
    _scene_hook_started = True

    import json as _json
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class _HookHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            t0 = time.time()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

            # ── /reset — clear all memory ───────────────────────────
            if self.path == "/reset":
                try:
                    count_before = _graph.count()
                    _reset_memory_state()
                    log.info("scene_hook.http: /reset — cleared %d nodes → 0",
                             count_before)
                    self._reply(200, {"ok": True, "message": f"Reset complete: "
                                     f"cleared {count_before} nodes"})
                except Exception as e:
                    log.warning("scene_hook.http: /reset error: %s: %s", type(e).__name__, e)
                    self._reply(500, {"error": str(e)})
                return

            try:
                data = _json.loads(body)
            except Exception:
                log.warning("scene_hook.http: bad JSON body (%d bytes)", len(body))
                self._reply(400, {"error": "invalid json"})
                return

            kv = data.get("kv") if isinstance(data.get("kv"), dict) else {}

            # ── VLMObserver special ops ─────────────────────────────
            # These bypass the normal remember pipeline — they operate on
            # existing nodes rather than creating new ones.

            # link_to_existing: add edge parent_node_id → link_to_existing
            link_target = kv.get("link_to_existing")
            if link_target is not None:
                try:
                    parent_nid = data.get("parent_node_id")
                    if parent_nid is not None:
                        _graph.add_edge(int(parent_nid), int(link_target))
                        log.info("scene_hook.http: link_to_existing: %d → %d",
                                 int(parent_nid), int(link_target))
                        self._reply(200, {"node_id": int(link_target)})
                    else:
                        self._reply(400, {"error": "link_to_existing requires parent_node_id"})
                except Exception as e:
                    log.warning("scene_hook.http: link_to_existing failed: %s", e)
                    self._reply(500, {"error": str(e)})
                return

            # append_image_ref: append image to an existing node
            if kv.get("append_image_ref"):
                try:
                    existing_nid = data.get("parent_node_id")
                    if existing_nid is not None:
                        nid = int(existing_nid)
                        node = _graph.get_node(nid)
                        if node is not None:
                            img_b64 = data.get("image_base64", "")
                            if img_b64 and _images is not None:
                                import base64 as _b64
                                img_bytes = _b64.b64decode(img_b64)
                                _images.save(nid, img_bytes)
                                node.image_refs = _images.list(nid)
                                _graph.update_node(nid, node)
                                log.info("scene_hook.http: append_image_ref → node %d (%d images)",
                                         nid, len(node.image_refs))
                            self._reply(200, {"node_id": nid})
                        else:
                            self._reply(404, {"error": f"node {nid} not found"})
                    else:
                        self._reply(400, {"error": "append_image_ref requires parent_node_id"})
                except Exception as e:
                    log.warning("scene_hook.http: append_image_ref failed: %s", e)
                    self._reply(500, {"error": str(e)})
                return

            has_img = "yes" if data.get("image_base64") else "no"
            n_objs = len(data.get("spatial", {}).get("objects", []))
            b64_len = len(data.get("image_base64", ""))
            log.info("scene_hook.http: received — has_image=%s objects=%d b64len=%d body=%dB",
                     has_img, n_objs, b64_len, length)

            # Run the remember pipeline synchronously in this thread.
            import asyncio as _asyncio
            try:
                lr = LogRecord.from_dict(data.get("log_record", {}))
                spatial = None
                if data.get("spatial"):
                    spatial = SpatialContext.from_dict(data["spatial"])

                # ── dataset-spec fields (camera_params/time_range/depth) ──
                camera_params = None
                if data.get("camera_params"):
                    camera_params = CameraParams.from_dict(data["camera_params"])
                time_range = None
                if data.get("time_range"):
                    tr = data["time_range"]
                    time_range = TimeRange(
                        start_ts=int(tr.get("start_ts", 0)),
                        end_ts=int(tr.get("end_ts", 0)),
                    )

                # ── dedup: reuse existing node for same object+position ──
                existing_nid = _find_existing_node(spatial, data.get("image_base64", ""))
                if existing_nid is not None:
                    log.info("scene_hook.http: dedup → appending to node %d", existing_nid)
                    self._reply(200, {"node_id": existing_nid})
                    return

                req = RememberRequest(
                    session_id=data.get("session_id", "scene-hook"),
                    plan_id=data.get("plan_id", "scene-hook"),
                    log_record=lr,
                    spatial=spatial,
                    parent_node_id=data.get("parent_node_id"),
                    image_base64=data.get("image_base64", ""),
                    depth_base64=data.get("depth_base64", ""),
                    camera_params=camera_params,
                    time_range=time_range,
                    kv=data.get("kv") if isinstance(data.get("kv"), dict) else {},
                )
                t_pipe = time.time()
                resp = _asyncio.run(_remember_pipe.execute(req))
                pipe_ms = (time.time() - t_pipe) * 1000
                total_ms = (time.time() - t0) * 1000
                log.info("scene_hook.http: → node %d (pipe %dms, total %dms)",
                         resp.node_id, round(pipe_ms), round(total_ms))
                self._reply(200, {"node_id": resp.node_id})
            except Exception as e:
                total_ms = (time.time() - t0) * 1000
                log.warning("scene_hook.http: ! pipeline error after %dms: %s: %s",
                           round(total_ms), type(e).__name__, e)
                self._reply(500, {"error": str(e)})

        def _reply(self, code, obj):
            body = _json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            log.debug("scene_hook: %s", fmt % args)

    def _find_existing_node(spatial, img_b64: str):
        """Check whether a node for the same object+position already exists.

        Returns the existing node_id, or None.  Dedup key: first object's
        (label, x_grid, y_grid) at 1.0 m resolution.  When a match is
        found AND *img_b64* is non-empty, the new image is appended to the
        existing node's ``image_refs``.
        """
        if spatial is None or not spatial.objects:
            return None
        obj = spatial.objects[0]
        gx = round(float(obj.x))
        gy = round(float(obj.y))
        key = f"{obj.label}@{gx},{gy}"
        # Search existing nodes for the same grid key.
        for nid in _graph.all_ids():
            node = _graph.get_node(nid)
            if node is None or node.spatial_data is None:
                continue
            if not node.spatial_data.objects:
                continue
            no = node.spatial_data.objects[0]
            ngx = round(float(no.x))
            ngy = round(float(no.y))
            nkey = f"{no.label}@{ngx},{ngy}"
            if nkey == key:
                # Append image if provided.
                if img_b64 and _images is not None:
                    import base64 as _b64
                    try:
                        img_bytes = _b64.b64decode(img_b64)
                        _images.save(nid, img_bytes)
                        node.image_refs = _images.list(nid)
                        _graph.update_node(nid, node)
                        log.info("scene_hook.dedup: appended image to node %d (%d total)",
                                 nid, len(node.image_refs))
                    except Exception as e:
                        log.warning("scene_hook.dedup: image append failed: %s", e)
                return nid
        return None

    port = int(os.environ.get("SCENE_HOOK_PORT", "37798"))
    try:
        server = HTTPServer(("0.0.0.0", port), _HookHandler)
    except OSError as e:
        # A stale sibling process (e.g. one killed without cleanup) can still
        # hold this port. The Hook endpoint is a best-effort side channel, not
        # part of the MCP capability surface (remember/hybrid_search/promote)
        # — losing it must not crash the whole service.
        log.warning("scene_hook: could not bind 0.0.0.0:%d (%s); "
                    "Scene Hook POST endpoint disabled for this run", port, e)
        return
    thread = threading.Thread(target=server.serve_forever, name="scene-hook", daemon=True)
    thread.start()
    log.info("scene_hook: HTTP endpoint on 0.0.0.0:%d", port)


# Start at module level so it works regardless of whether the process
# enters through main() or robonix_api.Service.run().
_start_scene_hook_server()

if __name__ == "__main__":
    sys.exit(main())
