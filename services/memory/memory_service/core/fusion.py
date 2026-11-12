"""Multi-route retrieval fusion + router (design §6).

Each retrieval route (vector / keyword / tag / causal / llm) returns scores
in its own unit (cosine, BM25, Jaccard, LLM rank …), so they cannot be summed
directly.  This module fuses *rankings* instead of raw scores.

Default fusion is Reciprocal Rank Fusion (RRF)::

    RRF(d) = Σ_r  1 / (60 + rank_r(d))

``rank_r(d)`` is document *d*'s 1-based rank in route *r*; documents absent
from a route contribute nothing.  Optional per-route weights scale each
route's vote.  A weighted linear fusion (min-max normalised) is also provided
as a non-default alternative for callers with comparable score scales.

``FusedSearch`` is the router: it runs the configured routes over a
pre-filtered candidate set (domain + tags + time), fuses their rankings, and
resolves the fused top-k back to concrete items (memory nodes or saved plans).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .types import LogRecord, TagFilter, TagSet

# RRF smoothing constant — dampens the effect of high ranks.
RRF_K = 60

# Routes enabled by default (llm is expensive and opt-in).
DEFAULT_ROUTES = ("vector", "keyword", "tag", "causal")

# node_type values that belong to the "scene" domain.
_SCENE_NODE_TYPES = {"object_observation", "path_segment", "place"}


# ── Pure fusion ─────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    rankings: Dict[str, List[str]],
    weights: Optional[Dict[str, float]] = None,
    k: float = RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse per-route rankings with RRF.

    Args:
        rankings: ``{route_name: [doc_id, ...]}``, each list ordered
            best-first (rank 1 = the first element).
        weights: optional ``{route_name: weight}``; routes missing a weight
            default to ``1.0``.
        k: smoothing constant (default 60).

    Returns:
        ``[(doc_id, rrf_score), ...]`` sorted by score descending.
    """
    scores: Dict[str, float] = defaultdict(float)
    weights = weights or {}
    for route, ranked in rankings.items():
        w = float(weights.get(route, 1.0))
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_from_scores(scored: Iterable[Tuple[str, float]]) -> List[str]:
    """Turn a ``[(doc_id, score), ...]`` list (already sorted desc) into a
    ranking list of doc ids — the shape RRF consumes."""
    return [doc_id for doc_id, _ in scored]


def weighted_linear_fusion(
    score_maps: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Alternative fusion: min-max normalise each route, then weighted sum.

    Requires comparable score scales per route after normalisation and is
    more weight-sensitive than RRF, so it is *not* the default (design §6.3.2).
    """
    weights = weights or {}
    doc_ids: Set[str] = set()
    for m in score_maps.values():
        doc_ids.update(m.keys())

    # Per-route min-max normalisation to [0, 1].
    normalized: Dict[str, Dict[str, float]] = {}
    for route, m in score_maps.items():
        if not m:
            normalized[route] = {}
            continue
        vals = list(m.values())
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        normalized[route] = {d: (m[d] - lo) / span for d in m}

    fused: Dict[str, float] = defaultdict(float)
    for route, m in normalized.items():
        w = float(weights.get(route, 1.0))
        for doc_id, s in m.items():
            fused[doc_id] += w * s
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def fuse(
    rankings: Optional[Dict[str, List[str]]] = None,
    score_maps: Optional[Dict[str, Dict[str, float]]] = None,
    fusion: str = "rrf",
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Dispatch to the configured fusion strategy."""
    if fusion == "linear":
        return weighted_linear_fusion(score_maps or {}, weights)
    return reciprocal_rank_fusion(rankings or {}, weights)


# ── Tag-overlap helpers (tag route) ─────────────────────────────────────

def _tag_tokens(tags: Optional[TagSet]) -> Set[Tuple[str, str]]:
    if tags is None:
        return set()
    toks: Set[Tuple[str, str]] = set()
    if tags.scene_type:
        toks.add(("scene", tags.scene_type))
    if tags.action_type:
        toks.add(("action", tags.action_type))
    if tags.task_type:
        toks.add(("task", tags.task_type))
    for o in (tags.objects_present or []):
        if o:
            toks.add(("obj", o))
    return toks


def _jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ── Router ──────────────────────────────────────────────────────────────

class FusedSearch:
    """Run the multi-route retrieval + RRF re-rank (design §6.4)."""

    def __init__(self, *, graph=None, tags=None, vectors=None,
                 milvus=None, ptdl=None, bge=None):
        self._graph = graph
        self._tags = tags
        self._vectors = vectors
        self._milvus = milvus
        self._ptdl = ptdl
        self._bge = bge

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        domain: str = "all",
        routes: Optional[List[str]] = None,
        fusion: str = "rrf",
        route_weights: Optional[Dict[str, float]] = None,
        tags_filter: Optional[TagFilter] = None,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Run the fused search and return a ``{"results": [...]}`` dict."""
        routes = list(routes or DEFAULT_ROUTES)
        overfetch = max(top_k * 3, 20)

        node_ids, plan_entries = self._prefilter(domain, tags_filter, time_range)

        rankings: Dict[str, List[str]] = {}
        if "vector" in routes:
            v = self._vector_ranking(query, domain, time_range, overfetch)
            if v:
                rankings["vector"] = v
        if "keyword" in routes:
            kw = self._keyword_ranking(query, node_ids, plan_entries, overfetch)
            if kw:
                rankings["keyword"] = kw
        if "tag" in routes:
            tg = self._tag_ranking(query, node_ids)
            if tg:
                rankings["tag"] = tg
        if "causal" in routes:
            seeds = [d for ranked in rankings.values() for d in ranked]
            c = self._causal_ranking(seeds, node_ids)
            if c:
                rankings["causal"] = c
        if "llm" in routes:
            l = await self._llm_ranking(query, node_ids, overfetch)
            if l:
                rankings["llm"] = l

        fused = fuse(rankings=rankings, fusion=fusion, weights=route_weights)

        rank_map = {route: {doc: rank for rank, doc in enumerate(ranked, start=1)}
                    for route, ranked in rankings.items()}

        results: List[Dict[str, Any]] = []
        for doc_id, score in fused[:top_k]:
            item = self._resolve(doc_id)
            if item is None:
                continue
            route_scores = {r: rank_map[r].get(doc_id)
                            for r in rankings if doc_id in rank_map[r]}
            results.append({
                "doc_id": doc_id,
                "kind": item["kind"],
                "score": round(score, 6),
                "route_scores": route_scores,
                "timestamp": item["timestamp"],
                item["kind"]: item["payload"],
            })

        return {"results": results}

    # ── prefilter ──────────────────────────────────────────────────────

    def _prefilter(self, domain: str, tags_filter: Optional[TagFilter],
                   time_range: Optional[Tuple[int, int]]):
        """Return ``(node_ids, plan_entries)`` surviving domain/tag/time."""
        node_ids: Set[int] = set()
        plan_entries: List[Dict[str, Any]] = []

        start_ts, end_ts = time_range if time_range else (0, 0)

        want_nodes = domain in ("scene", "memory", "all")
        want_plans = domain in ("plan", "all")

        if want_nodes and self._tags is not None and self._graph is not None:
            candidates = self._tags.query(tags_filter or TagFilter())
            for nid in candidates:
                node = self._graph.get_node(nid)
                if node is None:
                    continue
                nt = node.node_type.value if node.node_type else ""
                if domain == "scene" and nt not in _SCENE_NODE_TYPES:
                    continue
                if domain == "memory" and nt in _SCENE_NODE_TYPES:
                    continue
                if start_ts and node.timestamp < start_ts:
                    continue
                if end_ts and node.timestamp > end_ts:
                    continue
                node_ids.add(nid)

        if want_plans and self._ptdl is not None:
            for entry in self._ptdl.list_all():
                ts = int(entry.get("timestamp_ns") or 0)
                if start_ts and ts < start_ts:
                    continue
                if end_ts and ts > end_ts:
                    continue
                plan_entries.append(entry)

        return node_ids, plan_entries

    # ── routes ─────────────────────────────────────────────────────────

    def _vector_ranking(self, query: str, domain: str,
                        time_range: Optional[Tuple[int, int]],
                        overfetch: int) -> List[str]:
        if self._milvus is None or self._bge is None:
            return []
        qvec = self._bge.encode(query)
        if not qvec:
            return []
        hits: List[Dict[str, Any]] = []
        if domain in ("scene", "all"):
            hits.extend(self._milvus.search(qvec, top_k=overfetch, kind="scene",
                                            time_range=time_range))
        if domain in ("plan", "all"):
            hits.extend(self._milvus.search(qvec, top_k=overfetch, kind="plan",
                                            time_range=time_range))
        hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return [h["doc_id"] for h in hits]

    def _keyword_ranking(self, query: str, node_ids: Set[int],
                         plan_entries: List[Dict[str, Any]],
                         overfetch: int) -> List[str]:
        ranking: List[str] = []
        if self._vectors is not None and node_ids:
            scored = self._vectors.search(query, node_ids, top_k=overfetch, alpha=1.0)
            ranking.extend(str(nid) for nid, _ in scored)
        if self._ptdl is not None:
            # plan_entries is the domain/time-prefiltered set; the store's own
            # search() scans every entry, so restrict to it or a time filter
            # would leak out-of-window plans back into the ranking.
            allowed = {self._plan_doc_id(e.get("query", "")) for e in plan_entries}
            for entry in self._ptdl.search(query, top_k=overfetch):
                key = self._plan_doc_id(entry.get("query", ""))
                if key and key in allowed:
                    ranking.append(key)
        return ranking

    def _tag_ranking(self, query: str, node_ids: Set[int]) -> List[str]:
        if self._graph is None:
            return []
        from .remember import _rule_based_tag_extraction
        query_tags = _rule_based_tag_extraction(LogRecord(level="Info", msg=query), None)
        q_tokens = _tag_tokens(query_tags)
        scored = []
        for nid in node_ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            scored.append((str(nid), _jaccard(q_tokens, _tag_tokens(node.tags))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, s in scored if s > 0.0]

    def _causal_ranking(self, seeds: List[str], node_ids: Set[int]) -> List[str]:
        """Boost pre-filtered candidates that are causally adjacent to the
        query-matching seeds.  Expansion stays inside the candidate set so a
        hard ``tags``/``domain``/``time`` filter is never leaked."""
        if self._graph is None:
            return []
        out: List[str] = []
        seen: Set[str] = set(seeds)
        for doc in seeds:
            if not doc.isdigit():
                continue
            nid = int(doc)
            for parent in self._graph.get_parents(nid):
                if parent in node_ids and str(parent) not in seen:
                    out.append(str(parent))
                    seen.add(str(parent))
            for child in self._graph.get_children(nid):
                if child in node_ids and str(child) not in seen:
                    out.append(str(child))
                    seen.add(str(child))
        return out

    async def _llm_ranking(self, query: str, node_ids: Set[int],
                           overfetch: int) -> List[str]:
        if not node_ids or self._graph is None:
            return []
        from . import llm_search
        if not llm_search.llm_search_available():
            return []
        ranked = await llm_search.llm_rank(query, node_ids, self._graph.get_node,
                                           top_k=overfetch)
        return [str(nid) for nid, _ in ranked]

    # ── resolve ────────────────────────────────────────────────────────

    @staticmethod
    def _plan_doc_id(query: str) -> str:
        import hashlib
        return hashlib.sha1((query or "").strip().encode("utf-8")).hexdigest()

    def _resolve(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if doc_id.isdigit() and self._graph is not None:
            node = self._graph.get_node(int(doc_id))
            if node is None:
                return None
            nt = node.node_type.value if node.node_type else ""
            kind = "scene" if nt in _SCENE_NODE_TYPES else "memory"
            d = node.to_dict()
            for k in ("embedding", "image_refs", "frame_ts",
                      "depth_refs", "video_clip_refs"):
                d.pop(k, None)
            return {"kind": kind, "timestamp": node.timestamp, "payload": d}

        if self._ptdl is not None:
            plan = self._ptdl.get_by_query_key(doc_id)
            if plan is not None:
                return {"kind": "plan",
                        "timestamp": int(plan.get("timestamp_ns") or 0),
                        "payload": plan}
        return None
