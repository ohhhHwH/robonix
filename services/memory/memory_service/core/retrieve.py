"""Retrieve pipeline — 3-tier search with text index, BM25+LLM rank, and VLM.

Tier 1 (text index scan):
   LLM scans the lightweight text_index.txt -> if it can answer, returns
   immediately (0 ranker cost).  Otherwise returns a shortlist of
   relevant node_ids for Tier 2.

Tier 2 (tag filter -> rank -> filters):
   Standard pipeline.  If Tier 1 provided a shortlist, candidate_ids are
   intersected with it.

Tier 3 (VLM feature extraction):
   When Tier 2 returns low-confidence results and the nodes have images,
   VLM extracts visual features -> nodes updated -> re-search.
   Hook point present; full implementation in Module 3+C.
"""

from __future__ import annotations

import json as _json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Set

from ..storage.graph_store import GraphStore
from ..storage.tag_index import TagIndex
from ..storage.vector_store import VectorStore
from . import llm_search
from .types import (
    MemoryNode, SearchRequest, SearchResponse, TagFilter,
)


def _extract_video_keyframes(
    video_refs: List[str],
    time_range=None,
    max_frames: int = 3,
    data_dir: str = "",
) -> List[str]:
    """Extract keyframes from video files using ffmpeg.

    Args:
        video_refs: Relative video paths (e.g. "videos/rgb_main.mp4")
        time_range: Optional TimeRange to seek to a specific segment.
        max_frames: Maximum number of frames to extract per video.
        data_dir: Base directory for resolving relative video paths.

    Returns:
        List of absolute paths to extracted PNG frames (temporary files).
    """
    extracted: List[str] = []
    # Find ffmpeg binary
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return extracted

    for vref in video_refs[:2]:  # Max 2 video files
        # Resolve path
        video_path = Path(vref)
        if not video_path.is_absolute() and data_dir:
            # Try session directory relative paths
            candidates = [
                Path(data_dir) / vref,
                Path(data_dir).parent / vref,  # data_dir is memory dir, videos are in session dir
            ]
            for c in candidates:
                if c.exists():
                    video_path = c
                    break

        if not video_path.exists():
            logging.getLogger("scribe_mem").warning(
                "retrieve: video not found: %s", vref
            )
            continue

        # Build ffmpeg command
        tmpdir = tempfile.mkdtemp(prefix="rbnx_vframes_")
        seek_offset = ""
        if time_range is not None:
            start_ts = getattr(time_range, "start_ts", 0)
            # Convert nanosecond timestamp to seconds from video start
            seek_s = 0.0  # Default: from beginning
            if start_ts > 0:
                # Timestamp is absolute; extract relative offset from filename or use 0
                seek_s = 0.0
            if seek_s > 0:
                seek_offset = str(seek_s)

        # ffmpeg: seek to position, extract N frames as PNG
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
        ]
        if seek_offset:
            cmd.extend(["-ss", seek_offset])
        cmd.extend([
            "-i", str(video_path),
            "-vframes", str(max_frames),
            "-vf", "fps=1/3",  # 1 frame every 3 seconds (sparse sampling)
            f"{tmpdir}/vframe_%03d.png",
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0:
                for p in sorted(Path(tmpdir).glob("vframe_*.png")):
                    extracted.append(str(p))
            else:
                stderr = result.stderr.decode(errors="replace")[:200]
                logging.getLogger("scribe_mem").warning(
                    "retrieve: ffmpeg failed for %s: %s", vref, stderr
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logging.getLogger("scribe_mem").warning(
                "retrieve: ffmpeg error for %s: %s", vref, e
            )

    return extracted


def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg binary on the system."""
    for candidate in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        try:
            result = subprocess.run(
                [candidate, "-version"], capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None

log = logging.getLogger("scribe_mem")

# Tier 1 text index scan -- toggle + LLM prompt
_TIER1_ENABLED = os.environ.get("MEMGRAPH_TIER1_ENABLED", "1") in ("1", "true", "yes")

_TIER1_PROMPT = (
    "You are a memory index scanner. Given the following text index of memory "
    "nodes and a question, determine:\n"
    "1. Can you answer the question directly from the index? If yes, give the answer.\n"
    "2. If not, which node_ids are most relevant? Return them as a list.\n\n"
    "Text Index:\n"
    "{text_index_lines}\n\n"
    "Question: {user_query}\n\n"
    "Return JSON: {{\"can_answer\": bool, \"answer\": \"...\", "
    "\"relevant_node_ids\": [...]}}"
)


async def _tier1_text_index_scan(query: str) -> Optional[dict]:
    """Scan the text index with an LLM to short-circuit retrieval.

    Returns ``None`` if Tier 1 is disabled, LLM is unavailable, or the
    call fails.  Otherwise returns a dict with keys ``can_answer``,
    ``answer``, and ``relevant_node_ids``.
    """
    if not _TIER1_ENABLED:
        return None
    if not llm_search.llm_search_available():
        return None

    try:
        from ..storage.text_index import get_text_index
        ti = get_text_index()
        full_text = ti.full_text(max_lines=200)
        if not full_text.strip():
            log.debug("tier1: text index empty -- skipping")
            return None
        if ti.count() < 3:
            log.debug("tier1: too few nodes (%d) -- skipping", ti.count())
            return None
    except Exception as e:
        log.debug("tier1: text_index unavailable: %s", e)
        return None

    prompt = _TIER1_PROMPT.format(
        text_index_lines=full_text,
        user_query=query,
    )

    try:
        response = await llm_search.llm_json_chat(
            system_msg="You are a helpful memory index scanner. Return valid JSON only.",
            user_msg=prompt,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as e:
        log.debug("tier1: LLM call failed: %s", e)
        return None

    if response is None:
        return None

    try:
        result = _json.loads(response) if isinstance(response, str) else response
        if isinstance(result, dict) and "can_answer" in result:
            log.info(
                "tier1: scan -> can_answer=%s, relevant_ids=%d",
                result.get("can_answer"),
                len(result.get("relevant_node_ids", [])),
            )
            return result
    except (_json.JSONDecodeError, TypeError):
        log.debug("tier1: could not parse LLM response: %r", str(response)[:200])

    return None


class RetrievePipeline:
    """Orchestrate the search (read) path across all storage layers."""

    def __init__(self, graph_store: GraphStore, tag_index: TagIndex,
                 vector_store: VectorStore,
                 vlm_extractor=None,   # VLMFeatureExtractor or None
                 data_dir: str = "",
                 ):
        self._graph = graph_store
        self._tags = tag_index
        self._vectors = vector_store  # kept for optional embedding path
        self._vlm_extractor = vlm_extractor
        self._data_dir = data_dir

    async def execute(self, request: SearchRequest) -> SearchResponse:
        """Execute the search pipeline: Tier1 text index -> Tier2 rank -> filters.

        Returns:
            SearchResponse with ranked MemoryNode list.
        """
        # ================================================================
        # Tier 1: Text index scan (LLM short-circuit)
        # ================================================================
        tier1_result = await _tier1_text_index_scan(request.query)
        if tier1_result is not None and tier1_result.get("can_answer"):
            answer = tier1_result.get("answer", "")
            if answer:
                log.info("search: tier1 answered directly -> %r", answer[:120])
                return SearchResponse(nodes=[], vlm_answer=answer)

        # Tier 1 shortlist -- constrain Tier 2 candidate pool
        tier1_ids: Optional[Set[int]] = None
        if tier1_result is not None:
            rids = tier1_result.get("relevant_node_ids", [])
            if rids:
                tier1_ids = {int(n) for n in rids if isinstance(n, (int, float))}

        # ================================================================
        # Tier 2: Tag filter (O(1) inverted index)
        # ================================================================
        tag_filter = request.tags or TagFilter()
        candidate_ids = self._tags.query(tag_filter)

        # Intersect with Tier 1 shortlist if available
        if tier1_ids:
            candidate_ids = candidate_ids & tier1_ids
            log.debug("search: tier1 shortlist -> %d candidates (was %d before intersect)",
                      len(candidate_ids), self._tags.query(tag_filter).__len__()
                      if False else len(candidate_ids))

        if not candidate_ids:
            log.debug("search: tag filter returned empty set")
            return SearchResponse(nodes=[])

        log.debug("search: tag filter -> %d candidates", len(candidate_ids))

        # -- Stage 2: LLM rank (default) -> embedding -> chronological --
        top_k = max(1, request.top_k)
        overfetch = max(top_k * 3, 10)

        if llm_search.llm_search_available():
            # Path A: LLM (default -- best quality, works without embedding)
            log.info("search: using LLM ranker")
            ranked = await llm_search.llm_rank(
                query=request.query,
                candidate_ids=candidate_ids,
                graph_get=self._graph.get_node,
                top_k=overfetch,
            )
        elif self._vectors.is_semantic:
            # Path B: BM25 + Cosine hybrid (embedding model installed)
            log.info("search: LLM unavailable -- using embedding ranker")
            ranked = self._vectors.search(
                query=request.query,
                candidate_ids=candidate_ids,
                top_k=overfetch,
                alpha=request.alpha,
            )
        else:
            # Path C: BM25 keyword match, fall back to chronological
            log.info("search: LLM and embedding unavailable -- BM25 keyword")
            ranked = self._vectors.search(
                query=request.query,
                candidate_ids=candidate_ids,
                top_k=overfetch,
                alpha=1.0,  # pure BM25 -- no embedding scores available
            )
            if not ranked:
                # Ultimate fallback: most-recent-first
                log.info("search: BM25 returned empty -- chronological")
                nodes = [
                    self._graph.get_node(nid) for nid in candidate_ids
                ]
                nodes = [n for n in nodes if n is not None]
                nodes.sort(key=lambda n: n.timestamp, reverse=True)
                ranked = [
                    (n.node_id, max(0.05, 1.0 - i * 0.1))
                    for i, n in enumerate(nodes[:overfetch])
                ]

        if not ranked:
            return SearchResponse(nodes=[])

        # Build (node_id -> hybrid_score) map
        score_map = {nid: score for nid, score in ranked}

        # -- Stage 3: Causal expansion --
        # For each ranked candidate, pull in its immediate causal parents
        # and children so the VLM sees richer context.
        #
        # Three tiers:
        #   - require_executable: full expansion (existing behaviour)
        #   - path_segment nodes: always expand children (object_observations)
        #   - object_observation nodes: always expand parents (path_segments)
        post_causal = set(nid for nid, _ in ranked)
        for nid in list(post_causal):
            node = self._graph.get_node(nid)
            if node is None:
                continue
            nt = node.node_type.value if node.node_type else ""

            # Full bidirectional expansion for executable queries
            if request.require_executable:
                for parent_id in self._graph.get_parents(nid):
                    post_causal.add(parent_id)
                for child_id in self._graph.get_children(nid):
                    post_causal.add(child_id)
            # path_segment -> include its VLM-recognized children
            elif nt == "path_segment":
                for child_id in self._graph.get_children(nid):
                    post_causal.add(child_id)
            # object_observation -> include its parent path_segment
            elif nt == "object_observation":
                for parent_id in self._graph.get_parents(nid):
                    post_causal.add(parent_id)

        if len(post_causal) > len(ranked):
            log.debug("search: causal expansion -> %d nodes (was %d)",
                      len(post_causal), len(ranked))

        # -- Stage 4: Time filter --
        if request.time_range is not None:
            tr = request.time_range
            now = time.time_ns()
            end_ts = tr.end_ts if tr.end_ts > 0 else now
            post_time: Set[int] = set()
            for nid in post_causal:
                node = self._graph.get_node(nid)
                if node is not None and tr.start_ts <= node.timestamp <= end_ts:
                    post_time.add(nid)
            post_causal = post_time
            if not post_causal:
                return SearchResponse(nodes=[])

        # -- Stage 5: Weight sort --
        final_scores: List[tuple[int, float]] = []
        for nid in post_causal:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            weight = node.weight
            hybrid = score_map.get(nid, 0.0)
            final_scores.append((nid, weight * hybrid))

        final_scores.sort(key=lambda x: x[1], reverse=True)

        # -- Fetch full nodes --
        result_nodes: List[MemoryNode] = []
        for nid, _ in final_scores[:top_k]:
            node = self._graph.get_node(nid)
            if node is not None:
                # Update access metadata (persist to GraphStore so it
                # survives reboots -- used by forget/compact scoring).
                node.last_access = time.time_ns()
                node.access_count += 1
                self._graph.update_node(nid, node)
                result_nodes.append(node)

        log.info("search: \"%s\" -> %d results", request.query[:60], len(result_nodes))

        # ================================================================
        # Tier 3: VLM feature extraction (Module 3+C)
        # ================================================================
        # Trigger VLM to look at node images, extract visual features,
        # update nodes, and re-search when Tier 1+2 couldn't answer.
        _tier3_enabled = os.environ.get("MEMGRAPH_TIER3_ENABLED", "0") in ("1", "true", "yes")
        if _tier3_enabled and not result_nodes and request.vlm_qa:
            if self._vlm_extractor is not None:
                log.info("search: tier3 -- triggering VLM feature extraction "
                         "for query %r", request.query[:80])
                # Collect candidate nodes that have images
                img_nodes = []
                for nid in list(candidate_ids)[:10]:
                    node = self._graph.get_node(nid)
                    if node is not None and node.image_refs:
                        img_nodes.append(node)

                if img_nodes:
                    # Extract from up to 3 nodes
                    for node in img_nodes[:3]:
                        try:
                            features = await self._vlm_extractor.extract_and_update(
                                node_id=node.node_id,
                                query=request.query,
                                image_paths=node.image_refs,
                            )
                            if features is not None and features.get("can_answer_query"):
                                answer = features.get("answer_to_query", "")
                                if answer:
                                    log.info("search: tier3 -> VLM answered: %r",
                                             answer[:120])
                                    # Return the enriched node + answer
                                    result_nodes = [node]
                                    return SearchResponse(
                                        nodes=result_nodes,
                                        vlm_answer=answer,
                                    )
                        except Exception as e:
                            log.debug("search: tier3 extract failed for node %d: %s",
                                      node.node_id, e)

                    # Re-rank after enrichment (if no direct answer)
                    if result_nodes:
                        pass  # already returned above
                    else:
                        log.info("search: tier3 -- enrichment done, no direct answer")
            else:
                log.info("search: tier3 hook -- VLMFeatureExtractor not configured")

        # -- Stage 6: VLM QA (optional) --
        vlm_answer = ""
        if request.vlm_qa and result_nodes:
            # Collect image paths and build node contexts
            all_image_refs: List[str] = []
            node_ctx: List[str] = []
            for n in result_nodes[:3]:
                parts = [f"summary: \"{n.summary}\""]
                if n.tags:
                    parts.append(f"scene={n.tags.scene_type or '?'}")
                    parts.append(f"action={n.tags.action_type or '?'}")
                    if n.tags.objects_present:
                        parts.append(f"objects={','.join(n.tags.objects_present)}")
                    parts.append(f"success={n.tags.success}")
                if n.spatial_data and n.spatial_data.objects:
                    coords = "; ".join(
                        f"{o.label}@{o.x:.1f},{o.y:.1f},{o.z:.1f}"
                        for o in n.spatial_data.objects
                    )
                    parts.append(f"spatial={coords}")
                node_ctx.append(" | ".join(parts))
                if n.image_refs:
                    all_image_refs.extend(n.image_refs)
                # Stage 6a: Fallback to video keyframe extraction
                if not n.image_refs and n.video_clip_refs:
                    vframes = _extract_video_keyframes(
                        video_refs=n.video_clip_refs,
                        time_range=n.time_range,
                        max_frames=3,
                        data_dir=self._data_dir,
                    )
                    if vframes:
                        all_image_refs.extend(vframes)
                        log.info(
                            "search: extracted %d keyframes from %d video(s) for node %d",
                            len(vframes), len(n.video_clip_refs), n.node_id,
                        )

            if all_image_refs or node_ctx:
                # -- LLM decides: can we answer from text alone? --
                need_image, text_answer = await llm_search.llm_decide_vlm(
                    query=request.query,
                    node_contexts=node_ctx,
                    has_images=bool(all_image_refs),
                )
                if not need_image and text_answer:
                    vlm_answer = text_answer
                    log.info("search: llm_decide_vlm -> answer from text (no VLM)")
                elif all_image_refs:
                    log.info("search: vlm_qa -> %d images, %d contexts for query %r",
                             len(all_image_refs), len(node_ctx), request.query[:60])
                    from .observe import vlm_answer_question
                    answer = await vlm_answer_question(
                        query=request.query,
                        image_paths=all_image_refs,
                        node_contexts=node_ctx,
                    )
                    if answer:
                        vlm_answer = answer
                        log.info("search: vlm_qa answer -> %r", answer[:120])
                    else:
                        log.info("search: vlm_qa -- VLM unavailable, skipping")
                else:
                    log.info("search: vlm_qa -- no images to show")

        return SearchResponse(nodes=result_nodes, vlm_answer=vlm_answer)
