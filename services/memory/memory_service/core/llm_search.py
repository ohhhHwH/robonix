"""LLM-based memory search — fallback when embeddings are unavailable.

When sentence-transformers is not installed (the default, lightweight deployment
path), the hybrid BM25+Embedding stage is replaced by an LLM call:

  1. TagIndex → candidate node set (unchanged)
  2. Format candidates (summary + tags + spatial + causal) as a prompt
  3. Send prompt + user query to an OpenAI-compatible LLM
  4. LLM returns a JSON array of node IDs ranked by relevance
  5. Downstream causal/time/weight filters apply as usual

LLM endpoint configuration (env vars, in precedence order):
  MEMGRAPH_LLM_BASE_URL  >  VLM_BASE_URL  >  OPENAI_BASE_URL
  MEMGRAPH_LLM_API_KEY   >  VLM_API_KEY   >  OPENAI_API_KEY
  MEMGRAPH_LLM_MODEL     >  VLM_MODEL     >  OPENAI_MODEL  >  "gpt-4.1"

Backup LLM (tried when primary returns empty after retries):
  MEMGRAPH_LLM_BACKUP_BASE_URL / API_KEY / MODEL

Retry behaviour:
  - Primary LLM: up to 3 attempts with exponential backoff (2s → 4s → 8s)
  - If primary returns empty/error after all retries → try backup LLM once
  - If backup also fails → caller falls back to BM25/chronological

If no LLM credentials are available the pipeline falls back to the
deterministic hash embedding (non-semantic, same-text → same-vector).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import MemoryNode

log = logging.getLogger("scribe_mem")

# ── LLM configuration ──────────────────────────────────────────────────────

# Retry settings (env-overridable)
_LLM_MAX_RETRIES = int(os.environ.get("MEMGRAPH_LLM_RETRIES", "3"))
_LLM_RETRY_BASE_DELAY_S = float(os.environ.get("MEMGRAPH_LLM_RETRY_BASE_S", "2.0"))


def _llm_config() -> dict:
    """Resolve LLM endpoint from env vars with standard fallbacks.

    Priority: MEMGRAPH_LLM_*  >  VLM_* (Pilot)  >  OPENAI_*  >  "gpt-4.1"
    """
    base_url = (
        os.environ.get("MEMGRAPH_LLM_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    )
    api_key = (
        os.environ.get("MEMGRAPH_LLM_API_KEY")
        or os.environ.get("VLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = (
        os.environ.get("MEMGRAPH_LLM_MODEL")
        or os.environ.get("VLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4.1"
    )
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _llm_backup_config() -> dict:
    """Resolve backup LLM endpoint (tried when primary returns empty).

    Priority:
      1. MEMGRAPH_BACKUP_LLM_*  — dedicated backup provider (new)
      2. MEMGRAPH_LLM_BACKUP_*  — legacy backup vars
      3. MEM_VLM_*              — Aliyun qwen VLM endpoint (last resort)
    """
    # Tier 1: dedicated backup
    base_url = os.environ.get("MEMGRAPH_BACKUP_LLM_URL", "")
    api_key = os.environ.get("MEMGRAPH_BACKUP_LLM_KEY", "")
    model = os.environ.get("MEMGRAPH_BACKUP_LLM_MODEL", "")
    if base_url and api_key:
        return {"base_url": base_url, "api_key": api_key,
                "model": model or "qwen3.6-flash"}

    # Tier 2: legacy backup vars
    base_url = os.environ.get("MEMGRAPH_LLM_BACKUP_BASE_URL", "")
    api_key = os.environ.get("MEMGRAPH_LLM_BACKUP_API_KEY", "")
    model = os.environ.get("MEMGRAPH_LLM_BACKUP_MODEL", "")
    if base_url and api_key:
        return {"base_url": base_url, "api_key": api_key,
                "model": model or "gpt-4.1"}

    # Tier 3: VLM endpoint as last resort (Aliyun qwen / compatible)
    vlm_url = os.environ.get("MEM_VLM_BASE_URL", "")
    vlm_key = os.environ.get("MEM_VLM_API_KEY", "")
    vlm_model = os.environ.get("MEM_VLM_MODEL", "")
    if vlm_url and vlm_key:
        log.debug("llm_search: using VLM endpoint as backup LLM (%s)", vlm_model)
        return {"base_url": vlm_url, "api_key": vlm_key,
                "model": vlm_model or "qwen3.6-flash"}

    return {"base_url": "", "api_key": "", "model": ""}


def _is_same_provider(cfg_a: dict, cfg_b: dict) -> bool:
    """Check whether two LLM configs point to the same API endpoint."""
    if not cfg_a.get("base_url") or not cfg_b.get("base_url"):
        return False
    return cfg_a["base_url"].rstrip("/").lower() == cfg_b["base_url"].rstrip("/").lower()


def llm_search_available() -> bool:
    """True if enough LLM credentials are set to attempt a search."""
    cfg = _llm_config()
    return bool(cfg["api_key"]) and bool(cfg["base_url"])


# ── Formatter ──────────────────────────────────────────────────────────────

def _format_node(node: MemoryNode) -> str:
    """Format one memory node as a structured text block for the LLM prompt.

    Plan-type nodes (task_type="plan") include their saved RTDL steps so
    the LLM can reuse or adapt them.
    """
    tags = node.tags
    lines = [
        f"  Node {node.node_id}:",
        f"    summary: \"{node.summary}\"",
    ]
    if tags:
        lines.append(f"    scene: {tags.scene_type or '?'}")
        lines.append(f"    action: {tags.action_type or '?'}")
        lines.append(f"    task: {tags.task_type or '?'}")
        lines.append(f"    success: {tags.success}")
        lines.append(f"    difficulty: {tags.difficulty or '?'}")
        if tags.objects_present:
            lines.append(f"    objects: {', '.join(tags.objects_present)}")

    # ── Plan-type: show saved RTDL steps for reuse ──
    if tags and tags.task_type == "plan" and node.raw_log:
        raw = node.raw_log
        if hasattr(raw, 'msg') and raw.msg:
            lines.append(f"    plan_query: \"{raw.msg}\"")
        # Extract plan steps from the log message or kv
        # (stored by Pilot's _build_plan_log_message)

    # Spatial coordinates
    if node.spatial_data and node.spatial_data.objects:
        coords = []
        for o in node.spatial_data.objects:
            coords.append(f"{o.label}({o.obj_id})@{o.x:.1f},{o.y:.1f},{o.z:.1f}")
        lines.append(f"    spatial: {'; '.join(coords)}")

    # Causal chain
    if node.causal_chain:
        lines.append(f"    causal_parents: {node.causal_chain}")

    # Image references
    if node.image_refs:
        lines.append(f"    has_images: {len(node.image_refs)} frame(s)")

    lines.append(f"    weight: {node.weight:.2f}")
    lines.append(f"    type: {node.node_type.value}")
    return "\n".join(lines)


def _build_prompt(query: str, nodes: List[MemoryNode]) -> str:
    """Build the LLM prompt from the query and the full memory graph."""
    node_blocks = "\n".join(_format_node(n) for n in nodes)

    return (
        "You are a memory retrieval system for an embodied robot agent.\n"
        "Below is the robot's Causal Knowledge Graph — every observation,\n"
        "action, and event the robot has experienced and remembered.\n"
        "\n"
        "The user asks: \"" + query + "\"\n"
        "\n"
        "Memory graph (" + str(len(nodes)) + " nodes):\n"
        + node_blocks + "\n"
        "\n"
        "Select the nodes most relevant to the query. Consider:\n"
        "  - Semantic match between query and summary\n"
        "  - Scene/action/task tag overlap\n"
        "  - Spatial proximity (if query mentions a location)\n"
        "  - Object presence (if query mentions specific objects)\n"
        "  - Success/failure (if query asks about lessons or errors)\n"
        "  - Temporal recency (prefer newer when query is open-ended)\n"
        "\n"
        "Return ONLY a JSON object with a \"nodes\" field containing an array\n"
        "of node IDs in descending relevance order, like: {\"nodes\": [3, 7, 1]}\n"
        "Return {\"nodes\": []} if no nodes are relevant."
    )


# ── LLM caller (with retry + backup) ────────────────────────────────────────

async def _call_llm_once(
    prompt: str, cfg: dict, max_tokens: int = 512, timeout: float = 30.0,
) -> tuple:
    """Single LLM call — no retry.

    Returns:
        (node_ids, reason) tuple:
        - On success: ([1, 3, 7], "ok")
        - On failure: (None, "timeout" | "http_429" | "http_500" |
                       "empty_response" | "parse_error" | "network_error")
    """
    import httpx

    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    # Granular timeouts: 10s connect, caller-specified read timeout
    http_timeout = httpx.Timeout(
        connect=min(10.0, timeout),
        read=timeout,
        write=10.0,
        pool=5.0,
    )

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            r = await client.post(url, json=body, headers=headers)
            if r.status_code >= 500:
                log.warning("llm_search: LLM returned %d: %s",
                            r.status_code, r.text[:200])
                return None, f"http_{r.status_code}"
            if r.status_code == 429:
                log.warning("llm_search: LLM rate-limited (429)")
                return None, "http_429"
            if r.status_code >= 400:
                log.warning("llm_search: LLM returned %d: %s",
                            r.status_code, r.text[:200])
                return None, f"http_{r.status_code}"
            data = r.json()
    except httpx.ConnectTimeout:
        log.warning("llm_search: LLM connect timeout (%.1fs)", timeout)
        return None, "connect_timeout"
    except httpx.ReadTimeout:
        log.warning("llm_search: LLM read timeout (%.1fs)", timeout)
        return None, "read_timeout"
    except httpx.TimeoutException as e:
        log.warning("llm_search: LLM timeout: %s", e)
        return None, "timeout"
    except Exception as e:
        log.warning("llm_search: LLM call failed: %s: %s", type(e).__name__, e)
        return None, "network_error"

    elapsed = time.monotonic() - t0
    content = _parse_llm_content(data, cfg["model"])
    if content is None:
        return None, "parse_error"

    log.info("llm_search: %s responded in %.2fs (%d chars): %s",
             cfg["model"], elapsed, len(content), content[:200])
    node_ids = _parse_node_ids(content)
    if node_ids is None:
        return None, "parse_error"
    if len(node_ids) == 0:
        return [], "empty_response"
    return node_ids, "ok"


def _parse_llm_content(data: dict, model: str) -> Optional[str]:
    """Extract message content from LLM response dict."""
    try:
        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except (KeyError, IndexError, TypeError):
        log.warning("llm_search: unexpected LLM response shape from %s", model)
        return None


def _parse_node_ids(content: str) -> Optional[List[int]]:
    """Parse a JSON array of node IDs from LLM text output.

    Tolerant: strips markdown fences, tries regex fallback.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    # Try direct JSON parse
    try:
        result = json.loads(text)
        ids = result.get("nodes", [])
        if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
            log.info("llm_search: LLM selected node IDs: %s", ids)
            return ids  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # Regex fallback: find the first JSON array of ints
    import re
    m = re.search(r'\[[\d,\s]+\]', text)
    if m:
        try:
            ids = json.loads(m.group())
            if isinstance(ids, list) and ids:
                log.info("llm_search: LLM selected node IDs (regex): %s", ids)
                return ids  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    log.warning("llm_search: could not parse node IDs from LLM response: %r",
                content[:200])
    return None


# ── Generic JSON chat (Tier 1 text index scan, etc.) ────────────────────

async def llm_json_chat(
    system_msg: str,
    user_msg: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Optional[str]:
    """Send a *system_msg* + *user_msg* to the LLM and return the raw
    text content.  Used by Tier 1 text index scanner and other
    lightweight LLM calls that don't need node formatting.

    Returns ``None`` on any failure.
    """
    cfg = _llm_config()
    if not cfg["api_key"] or not cfg["base_url"]:
        return None

    import httpx

    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=20.0),
        ) as client:
            r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            log.warning("llm_json_chat: LLM returned %d: %s",
                        r.status_code, r.text[:200])
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("llm_json_chat: LLM call failed: %s: %s",
                    type(e).__name__, e)
        return None


async def _call_llm(prompt: str, max_tokens: int = 512) -> Optional[List[int]]:
    """Send prompt to LLM with retry + backup fallback.

    Retry strategy:
      1. Primary LLM: up to _LLM_MAX_RETRIES attempts with exponential backoff
         (2s → 4s → 8s by default). Transient errors (timeout, http_429/500)
         retry; permanent errors (http_400/401/403) do not.
      2. If primary fails after all retries → try backup LLM once
         (skipped if backup uses same provider as primary).
      3. Returns None only if both primary and backup fail.

    The caller (llm_rank) falls back to chronological order on None.
    """
    primary_cfg = _llm_config()
    if not primary_cfg["api_key"] or not primary_cfg["base_url"]:
        log.debug("llm_search: no LLM credentials — skipping")
        return None

    # ── Primary LLM with retry ──
    last_reason: str = "unknown"
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        result, reason = await _call_llm_once(prompt, primary_cfg, max_tokens)
        last_reason = reason

        if result is not None and len(result) > 0:
            if attempt > 1:
                log.info("llm_search: primary LLM succeeded on attempt %d/%d "
                         "(prev errors: %s)",
                         attempt, _LLM_MAX_RETRIES, last_reason)
            return result

        if attempt < _LLM_MAX_RETRIES:
            # Don't retry permanent client errors (auth, bad request)
            if reason.startswith("http_4") and reason not in ("http_429",):
                log.warning("llm_search: primary LLM permanent error (%s) — "
                            "not retrying", reason)
                break

            delay = _LLM_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            log.info("llm_search: primary LLM attempt %d/%d failed (%s) — "
                     "retrying in %.1fs",
                     attempt, _LLM_MAX_RETRIES, reason, delay)
            await asyncio.sleep(delay)

    log.warning("llm_search: primary LLM (%s) failed after %d attempts "
                "(last: %s)",
                primary_cfg["model"], _LLM_MAX_RETRIES, last_reason)

    # ── Backup LLM ──
    backup_cfg = _llm_backup_config()
    if not (backup_cfg["api_key"] and backup_cfg["base_url"]):
        log.warning("llm_search: no backup LLM configured")
        return None

    # Skip backup if same provider as primary (shared failure mode)
    if _is_same_provider(primary_cfg, backup_cfg):
        log.warning("llm_search: backup LLM uses same provider as primary "
                    "(%s) — skipping redundant backup",
                    backup_cfg["base_url"])
        return None

    log.info("llm_search: trying backup LLM %s @ %s ...",
             backup_cfg["model"], backup_cfg["base_url"])
    result, reason = await _call_llm_once(
        prompt, backup_cfg, max_tokens, timeout=45.0,
    )
    if result is not None and len(result) > 0:
        log.info("llm_search: backup LLM (%s) succeeded", backup_cfg["model"])
        return result
    log.warning("llm_search: backup LLM (%s) also failed (%s)",
                backup_cfg["model"], reason)

    return None


# ── Search entry point ─────────────────────────────────────────────────────

async def llm_rank(
    query: str,
    candidate_ids: Set[int],
    graph_get,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Ask an LLM to rank the candidate nodes by relevance.

    Args:
        query: Natural language search query.
        candidate_ids: Node IDs to consider (from TagIndex pre-filter).
        graph_get: Callable node_id → Optional[MemoryNode] (GraphStore.get_node).
        top_k: Maximum nodes to return.

    Returns:
        List of (node_id, relevance_score) sorted descending, matching the
        same signature as VectorStore.search() so RetrievePipeline can
        consume both paths transparently.

        Scores are assigned as synthetic linear values:
          - 1st place → 1.0
          - 2nd place → 0.9
          - Nth place → max(0.05, 1.0 - (rank-1)*0.1)
    """
    # Fetch candidate nodes (no embeddings needed)
    nodes: List[MemoryNode] = []
    for nid in candidate_ids:
        node = graph_get(nid)
        if node is not None:
            nodes.append(node)

    if not nodes:
        return []

    if len(nodes) == 1:
        return [(nodes[0].node_id, 1.0)]

    # Build prompt and call LLM
    prompt = _build_prompt(query, nodes)
    ranked_ids = await _call_llm(prompt)

    if ranked_ids is None:
        log.info("llm_search: LLM unavailable — falling back to chronological order")
        # Fallback: most recent first
        nodes.sort(key=lambda n: n.timestamp, reverse=True)
        results: List[Tuple[int, float]] = []
        for i, n in enumerate(nodes[:top_k]):
            score = max(0.05, 1.0 - i * 0.1)
            results.append((n.node_id, score))
        return results

    # Build (node_id → score) from LLM ranking
    seen: Set[int] = set()
    results = []
    for rank, nid in enumerate(ranked_ids):
        if nid in seen or nid not in candidate_ids:
            continue
        seen.add(nid)
        score = max(0.05, 1.0 - rank * 0.1)
        results.append((nid, score))

    # Append any missing candidates at the end (low score)
    for n in nodes:
        if n.node_id not in seen:
            results.append((n.node_id, 0.05))

    log.info("llm_search: \"%s\" → %d ranked by LLM", query[:60], len(results))
    return results[:top_k]


_VLM_NEEDED_PROMPT = (
    "You are a memory retrieval system for an embodied robot.\n"
    "Below are memory nodes relevant to the user's query.\n"
    "Each node has a text summary, tags, and spatial positions.\n"
    "Some nodes also have images attached.\n\n"
    "Decide: can you answer the user's question from the TEXT alone,\n"
    "or do you NEED to examine the images?\n\n"
    "Reply with a single JSON object:\n"
    '  {"need_image": false, "answer": "your text answer"}  — if text is enough\n'
    '  {"need_image": true,  "answer": ""}                  — if images are required\n\n'
    "Only set need_image=true when the question explicitly asks about\n"
    "visual details (color, shape, count, whether something is present)\n"
    "that cannot be inferred from the text labels and spatial data alone.\n"
)


async def llm_decide_vlm(
    query: str,
    node_contexts: list[str],
    has_images: bool,
) -> tuple[bool, str]:
    """Ask the LLM whether VLM image analysis is needed to answer the query.

    Returns:
        (need_image, text_answer)
        - need_image=True  → proceed to VLM QA
        - need_image=False → use text_answer directly
    """
    if not has_images:
        return False, ""
    cfg = _llm_config()
    if not cfg["api_key"] or not cfg["base_url"]:
        return True, ""  # can't decide → default to VLM

    # Build prompt
    ctx_text = "\n\n".join(
        f"[{i+1}] {ctx}" for i, ctx in enumerate(node_contexts)
    )
    prompt = (
        f"{_VLM_NEEDED_PROMPT}\n\n"
        f"Memory nodes ({len(node_contexts)} total):\n{ctx_text}\n\n"
        f"User question: {query}"
    )

    import httpx
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, json=body, headers=headers)
            if r.status_code >= 400:
                log.debug("llm_decide_vlm: LLM returned %d", r.status_code)
                return True, ""
            data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.debug("llm_decide_vlm: call failed: %s", e)
        return True, ""

    # Parse LLM response
    import json as _json
    try:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        result = _json.loads(content)
        need = bool(result.get("need_image", True))
        answer = str(result.get("answer", ""))
        log.info("llm_decide_vlm: need_image=%s answer=%r", need, answer[:120])
        return need, answer
    except Exception as e:
        log.debug("llm_decide_vlm: parse failed: %s", e)
        return True, ""
