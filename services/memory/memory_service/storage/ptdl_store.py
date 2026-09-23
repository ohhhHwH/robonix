"""PTDL Store — standalone JSON file for successful RTDL plans.

Separate from the main graph_store.json so that:
  - Plan history survives graph_store clean_start wipes
  - Plans are easily inspectable (human-readable JSON array)
  - No dependency on TagIndex / VectorStore / MCP round-trip

Format — a **tree**, one record per user question (the root) whose ordered
planning rounds (sub-plans) are its children:

  [
    {
      "query": "向后移动2m",
      "description": "complete task (2 step(s) across 2 planning round(s))",
      "plan_count": 2,          # successful sub-plans (derived)
      "canceled_count": 0,      # canceled sub-plans (derived)
      "timestamp_ns": 1785600000000000000,
      "plans": [
        {
          "plan_id": "1",
          "status": "success",   # success | failed | canceled
          "description": "记录起点并开始旋转扫描",
          "steps": [
            "1. [mapping.map_get_pose] 读取当前位姿",
            "2. [tiago_camera.camera_snapshot] 拍摄初始图像"
          ],
          "rtdl_plan": "{\"plan_id\":\"1\",\"nodes\":[...]}",
          "raw_rtdl": "{\"op\":\"sequence\",...}",
          "timestamp_ns": 1785600000000000000
        },
        { "...": "second planning round (sub-plan #2)" }
      ]
    }
  ]

Each sub-plan carries its **own** ``rtdl_plan`` (the full RTDL Plan AST for
that round) and ``raw_rtdl`` (the raw LLM RTDL for that round), so the whole
per-round execution is preserved instead of flattened into one step list.

``add`` is an upsert keyed on the exact ``query`` (the root) plus ``plan_id``
(the child): a multi-round task reported once per planning round is
consolidated into a single root record whose children are the rounds.
``plan_id=None`` finalizes the root (a rollup summary) rather than adding a
child; ``plan_id=None`` with non-empty ``steps`` is the legacy "one anonymous
sub-plan" path used by the ``remember`` plan shortcut.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("scribe_mem")

_DEFAULT_PTDL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "memory" / "ptdl_store.json"
)


class PtdlStore:
    """Standalone JSON store for RTDL plan records (tree: query → sub-plans)."""

    def __init__(self, path: str = ""):
        self._path = Path(path) if path else Path(_DEFAULT_PTDL_PATH)
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def path(self) -> str:
        return str(self._path)

    def add(
        self,
        query: str,
        description: str,
        steps: List[str],
        plan_count: int = 1,
        canceled_count: int = 0,
        rtdl_plan: Optional[str] = None,
        raw_rtdl: Optional[str] = None,
        plan_id: Optional[str] = None,
    ) -> None:
        """Upsert a plan record keyed on ``query`` (root) + ``plan_id`` (child).

        One user question maps to exactly one root record; each successful
        planning round is a child sub-plan carrying its own ``rtdl_plan`` /
        ``raw_rtdl`` / ``steps``.

        Three call shapes:

        - ``plan_id`` given (Pilot's per-round immediate save) — upsert the
          child sub-plan keyed by ``plan_id``: same round merges steps (union,
          re-numbered), refreshes ``rtdl_plan``/``raw_rtdl`` (last-non-null)
          and its ``status``.
        - ``plan_id`` None + empty ``steps`` (Pilot's end-of-turn rollup) —
          finalize the root: refresh the summary ``description`` only; children
          and derived counts are left intact.
        - ``plan_id`` None + non-empty ``steps`` (legacy ``remember`` shortcut /
          direct callers) — upsert a single anonymous ``"legacy"`` sub-plan.
        """
        existing = self._find_exact(query)
        if existing is None:
            entry: Dict[str, Any] = {
                "query": query,
                "description": "",
                "plans": [],
                "timestamp_ns": time.time_ns(),
                "plan_count": 0,
                "canceled_count": 0,
            }
            self._entries.append(entry)
            existing = entry
        else:
            self._migrate_flat(existing)

        if plan_id is not None:
            self._upsert_plan(
                existing,
                str(plan_id),
                description,
                steps,
                rtdl_plan,
                raw_rtdl,
                canceled_count,
            )
        elif steps:
            # Legacy anonymous sub-plan (no plan_id supplied by caller).
            self._upsert_plan(
                existing,
                "legacy",
                description,
                steps,
                rtdl_plan,
                raw_rtdl,
                canceled_count,
            )
        else:
            # Rollup finalize: refresh the root summary only.
            existing["description"] = _pick_description(
                existing.get("description", ""), description
            )
            existing["timestamp_ns"] = time.time_ns()

        self._recompute_counts(existing)
        self._save()
        n_plans = len(existing.get("plans", []))
        log.info(
            "ptdl_store: saved plan \"%s\" (%d sub-plan(s))", query, n_plans
        )

    def _upsert_plan(
        self,
        entry: Dict[str, Any],
        plan_id: str,
        description: str,
        steps: List[str],
        rtdl_plan: Optional[str],
        raw_rtdl: Optional[str],
        canceled_count: int,
    ) -> None:
        """Upsert one child sub-plan into *entry*, keyed by *plan_id*."""
        status = _status_of(description, canceled_count)
        plans: List[Dict[str, Any]] = entry.setdefault("plans", [])
        for plan in plans:
            if str(plan.get("plan_id", "")) == plan_id:
                plan["steps"] = _merge_steps(plan.get("steps", []), steps)
                plan["description"] = _pick_description(
                    plan.get("description", ""), description
                )
                if rtdl_plan is not None:
                    plan["rtdl_plan"] = rtdl_plan
                if raw_rtdl is not None:
                    plan["raw_rtdl"] = raw_rtdl
                plan["status"] = status
                plan["timestamp_ns"] = time.time_ns()
                return

        child: Dict[str, Any] = {
            "plan_id": plan_id,
            "status": status,
            "description": description,
            "steps": list(steps),
            "timestamp_ns": time.time_ns(),
        }
        if rtdl_plan is not None:
            child["rtdl_plan"] = rtdl_plan
        if raw_rtdl is not None:
            child["raw_rtdl"] = raw_rtdl
        plans.append(child)

    def _migrate_flat(self, entry: Dict[str, Any]) -> None:
        """Upgrade a pre-tree flat record in place into a one-child tree."""
        if "plans" in entry:
            return
        steps = entry.pop("steps", []) or []
        rtdl_plan = entry.pop("rtdl_plan", None)
        raw_rtdl = entry.pop("raw_rtdl", None)
        description = entry.get("description", "")
        child: Dict[str, Any] = {
            "plan_id": "legacy",
            "status": "success",
            "description": description,
            "steps": steps,
            "timestamp_ns": entry.get("timestamp_ns", time.time_ns()),
        }
        if rtdl_plan is not None:
            child["rtdl_plan"] = rtdl_plan
        if raw_rtdl is not None:
            child["raw_rtdl"] = raw_rtdl
        entry["plans"] = [child]

    def _recompute_counts(self, entry: Dict[str, Any]) -> None:
        plans = entry.get("plans", [])
        entry["plan_count"] = sum(1 for p in plans if p.get("status") == "success")
        entry["canceled_count"] = sum(1 for p in plans if p.get("status") == "canceled")

    def _find_exact(self, query: str) -> Optional[Dict[str, Any]]:
        """Return the record whose ``query`` equals *query* field-by-field,
        or ``None`` when no identical question has been saved yet."""
        q = (query or "").strip()
        for entry in self._entries:
            if (entry.get("query") or "").strip() == q:
                return entry
        return None

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all saved plan records (most recent last)."""
        return list(self._entries)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Keyword-match plans against *query* and return the top *top_k*.

        A field-by-field identical question ("same-question detection") always
        leads the results — it is the definitive cached answer for a repeated
        request. The remaining slots are filled by fuzzy scoring:
        1. Substring containment — query appears as-is in plan text (high boost)
        2. Token overlap — whitespace-split BOW overlap (works for English)
        3. Character bigram overlap — CJK-friendly fallback

        No embedding model is required.
        """
        if not query or not self._entries:
            return []

        exact = self._find_exact(query)
        if exact is not None:
            rest = self._fuzzy_search(
                query, [e for e in self._entries if e is not exact], top_k - 1
            )
            return [exact] + rest
        return self._fuzzy_search(query, self._entries, top_k)

    def _fuzzy_search(
        self, query: str, entries: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for entry in entries:
            text = _entry_text(entry).lower()

            score = 0.0

            # 1. Substring match (strongest signal)
            if query_lower in text:
                score += 2.0

            # 2. Token overlap (whitespace-split, works for English)
            q_tokens = set(query_lower.split())
            t_tokens = set(text.split())
            overlap = len(q_tokens & t_tokens)
            score += overlap / (len(q_tokens) + 1.0)

            # 3. Character bigram overlap (CJK-friendly)
            q_bigrams = _char_bigrams(query_lower)
            t_bigrams = _char_bigrams(text)
            if q_bigrams:
                bg_overlap = len(q_bigrams & t_bigrams)
                score += bg_overlap / (len(q_bigrams) + 1.0)

            if score > 0.0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def remove(self, query: str) -> bool:
        """Remove the first entry whose query matches exactly. Returns True
        if an entry was removed."""
        for i, entry in enumerate(self._entries):
            if entry.get("query") == query:
                self._entries.pop(i)
                self._save()
                return True
        return False

    def count(self) -> int:
        return len(self._entries)

    # ── Internal ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = self._path.read_text(encoding="utf-8")
                self._entries = json.loads(raw) if raw.strip() else []
            except (json.JSONDecodeError, OSError) as e:
                log.warning("ptdl_store: failed to load %s: %s", self._path, e)
                self._entries = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._path) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp, str(self._path))
        except OSError as e:
            log.warning("ptdl_store: failed to save %s: %s", self._path, e)


def _entry_text(entry: Dict[str, Any]) -> str:
    """Flatten a tree record into a keyword-rich text block for fuzzy search."""
    parts = [entry.get("query", ""), entry.get("description", "")]
    for plan in entry.get("plans", []):
        parts.append(plan.get("description", ""))
        parts.extend(plan.get("steps", []))
    return " ".join(p for p in parts if p)


def _status_of(description: str, canceled_count: int) -> str:
    """Derive a child's status from its description / canceled flag."""
    if "(FAILED" in (description or ""):
        return "canceled" if canceled_count > 0 else "failed"
    return "success"


def _char_bigrams(s: str) -> set:
    """Extract character bigrams for CJK-friendly fuzzy matching."""
    stripped = "".join(c for c in s if c.isalnum())
    if len(stripped) < 2:
        return {stripped} if stripped else set()
    return {stripped[i:i + 2] for i in range(len(stripped) - 1)}


def _step_identity(step: str) -> str:
    """Strip the per-tree ``N. `` numbering so the same step reported by
    different planning rounds compares equal. Pilot numbers each round's
    steps from 1, so ``1. [navigate] …`` from one tree and ``5. [navigate]
    …`` from the merged rollup denote the same step."""
    s = (step or "").strip()
    if s and s[0].isdigit():
        dot = s.find(".")
        if dot != -1 and s[:dot].isdigit():
            return s[dot + 1:].strip()
    return s


def _merge_steps(existing: List[str], incoming: List[str]) -> List[str]:
    """Union *existing* and *incoming* steps, deduped by their un-numbered
    identity and re-numbered 1..N in first-seen order."""
    seen: Dict[str, None] = {}
    ordered: List[str] = []
    for step in list(existing) + list(incoming):
        ident = _step_identity(step)
        if not ident or ident in seen:
            continue
        seen[ident] = None
        ordered.append(ident)
    return [f"{i}. {ident}" for i, ident in enumerate(ordered, 1)]


def _pick_description(current: str, incoming: str) -> str:
    """Prefer the ``complete task …`` rollup summary; otherwise the newer
    non-empty value. Falls back to the current value when incoming is empty."""
    if incoming.startswith("complete task"):
        return incoming
    if current.startswith("complete task"):
        return current
    return incoming or current


# Module-level singleton, created on first import.
_ptdl_store: Optional[PtdlStore] = None


def get_ptdl_store(path: str = "") -> PtdlStore:
    global _ptdl_store
    if _ptdl_store is None:
        _ptdl_store = PtdlStore(path)
    return _ptdl_store


def _ptdl_store_reset() -> None:
    """Clear the module-level singleton (used by clean_start)."""
    global _ptdl_store
    _ptdl_store = None
