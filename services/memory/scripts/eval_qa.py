#!/usr/bin/env python3
"""Evaluate QA pairs against a memory graph using RAG + LLM-Judge.

For each QA pair:
  1. hybrid_search(query) → candidate memory nodes (context)
  2. RAG: LLM reads context and answers the question
  3. LLM-Judge compares RAG answer with ground truth → correct/incorrect
  4. Aggregate metrics per category + difficulty

Usage:
  uv run python scripts/eval_qa.py \
    --qa-dir ./qa_output \
    --memory-dir /tmp/scenes2_memory \
    --llm-model deepseek-v4-flash \
    --llm-url https://api.deepseek.com \
    --llm-key $VLM_API_KEY \
    --output metrics.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_SVC = _HERE.parent
sys.path.insert(0, str(_SVC))

from memory_service.core.types import TagFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate QA pairs against memory graph")
    p.add_argument("--qa-dir", required=True, help="Directory with qa_index.yaml + category JSONs")
    p.add_argument("--memory-dir", default="/tmp/scenes2_memory",
                   help="MemoryService data dir (pre-loaded via load_scenes2.py)")
    p.add_argument("--llm-model", default=os.environ.get("VLM_MODEL", "deepseek-v4-flash"))
    p.add_argument("--llm-url", default=os.environ.get("VLM_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--llm-key", default=os.environ.get("VLM_API_KEY", ""))
    p.add_argument("--output", default="metrics.json", help="Output metrics JSON")
    p.add_argument("--vlm-qa", action="store_true", help="Enable VLM-based image QA")
    p.add_argument("--no-rag", action="store_true", help="Skip RAG step, use raw search results")
    return p.parse_args()


def _call_llm(prompt: str, model: str, url: str, api_key: str,
              max_tokens: int = 1024, temperature: float = 0.0) -> Optional[str]:
    """Call an OpenAI-compatible chat completion API."""
    if not api_key:
        return None
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        req = urllib.request.Request(f"{url}/chat/completions", data=payload, headers=headers)
        resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
        content = resp["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def rag_answer(
    question: str, context: str, model: str, url: str, api_key: str,
) -> str:
    """Use LLM to answer a question based on retrieved memory context."""
    if not api_key:
        return context[:500]  # Fallback to raw context

    prompt = f"""You are a robot memory QA system. Answer the question based ONLY on the memory context below.
If the context doesn't contain enough information, say "insufficient information".
Keep your answer concise (1-2 sentences).

MEMORY CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    result = _call_llm(prompt, model, url, api_key, max_tokens=256, temperature=0.0)
    return result if result else context[:500]


def judge_answer(
    question: str, predicted: str, ground_truth: str,
    acceptable: List[str], model: str, url: str, api_key: str,
) -> Dict[str, Any]:
    """LLM-as-Judge: compare predicted answer vs ground truth."""
    if not api_key:
        matched = ground_truth.lower() in predicted.lower() or \
                  any(a.lower() in predicted.lower() for a in acceptable)
        return {"correct": matched, "confidence": 1.0 if matched else 0.0,
                "method": "keyword"}

    prompt = f"""You are a QA evaluator for an embodied memory benchmark. Judge whether the PREDICTED
answer correctly answers the QUESTION based on the GROUND TRUTH.

IMPORTANT: Be lenient with wording differences. If the predicted answer conveys the same
meaning/facts as the ground truth (even with different wording), mark it as CORRECT.

QUESTION: {question}
PREDICTED ANSWER: {predicted}
GROUND TRUTH: {ground_truth}
ACCEPTABLE ANSWERS: {json.dumps(acceptable, ensure_ascii=False)}

Reply with ONLY a JSON object: {{"correct": true/false, "reasoning": "brief reason", "confidence": 0.0-1.0}}"""

    result = _call_llm(prompt, model, url, api_key, max_tokens=256, temperature=0.0)
    if result is None:
        # Fallback to keyword matching
        matched = ground_truth.lower() in predicted.lower() or \
                  any(a.lower() in predicted.lower() for a in acceptable)
        return {"correct": matched, "confidence": 0.7 if matched else 0.3,
                "method": "keyword_fallback"}

    # Extract JSON
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```\w*\n?", "", result)
        result = re.sub(r"\n?```$", "", result)
    try:
        parsed = json.loads(result)
        return {"correct": parsed.get("correct", False),
                "confidence": parsed.get("confidence", 0.5),
                "reasoning": parsed.get("reasoning", ""),
                "method": "llm_judge"}
    except json.JSONDecodeError:
        matched = ground_truth.lower() in predicted.lower() or \
                  any(a.lower() in predicted.lower() for a in acceptable)
        return {"correct": matched, "confidence": 0.5, "method": "json_parse_fallback"}


def _aggregate(results: List[Dict]) -> Dict[str, Any]:
    """Compute per-category and overall metrics."""
    cats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"total": 0, "correct": 0, "easy": {"total": 0, "correct": 0},
                         "medium": {"total": 0, "correct": 0}, "hard": {"total": 0, "correct": 0}}
        cats[cat]["total"] += 1
        if r["correct"]:
            cats[cat]["correct"] += 1
        d = r.get("difficulty", "medium")
        if d in cats[cat]:
            cats[cat][d]["total"] += 1
            if r["correct"]:
                cats[cat][d]["correct"] += 1

    for cat, data in cats.items():
        data["accuracy"] = round(data["correct"] / data["total"], 3) if data["total"] else 0
        for d in ["easy", "medium", "hard"]:
            dd = data[d]
            dd["accuracy"] = round(dd["correct"] / dd["total"], 3) if dd["total"] else 0

    overall_correct = sum(1 for r in results if r["correct"])
    overall_total = len(results)
    rag_used = sum(1 for r in results if r.get("method") != "keyword")
    return {
        "overall": {"correct": overall_correct, "total": overall_total,
                     "accuracy": round(overall_correct/overall_total, 3) if overall_total else 0},
        "by_category": cats,
        "meta": {"rag_calls": rag_used, "total_qa": overall_total},
    }


async def _run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    """Main async evaluation routine."""
    from memory_service.service import MemoryService

    CATEGORIES = ["existence_recall", "detail_recall", "spatiotemporal",
                  "change_detection", "anomaly_judgment", "reasoning"]

    svc = MemoryService(data_dir=args.memory_dir)
    await svc.init()

    all_results: List[Dict] = []
    total_llm_calls = 0

    for cat_idx, cat in enumerate(CATEGORIES):
        qa_file = os.path.join(args.qa_dir, f"{cat}.json")
        if not os.path.exists(qa_file):
            print(f"  SKIP: {qa_file} not found")
            continue

        with open(qa_file) as f:
            pairs = json.load(f)

        print(f"\nEvaluating {cat} ({len(pairs)} pairs)...")

        for qa in pairs:
            query = qa.get("question_zh", qa.get("question_en", ""))

            # 1. Search memory
            tags = TagFilter(task_type="observe")
            try:
                search_resp = await svc.search(query, tags=tags, top_k=5)
                nodes = search_resp.nodes if search_resp.nodes else []
            except Exception:
                nodes = []

            # Build context from retrieved nodes
            context_parts = []
            for n in nodes[:5]:
                spatial = n.spatial_data
                objs_str = ""
                if spatial and spatial.objects:
                    objs_str = ", ".join(
                        f"{o.label}@({o.x:.1f},{o.y:.1f})" for o in spatial.objects[:3]
                    )
                context_parts.append(
                    f"[node_{n.node_id}] ts={n.timestamp} summary=\"{n.summary}\" "
                    f"objects=[{objs_str}] tags={n.tags.to_dict() if n.tags else {}}"
                )
            context = "\n".join(context_parts) if context_parts else "no memory found"

            # 2. RAG: answer question from context (or use raw context)
            if args.no_rag:
                predicted = context
            else:
                predicted = rag_answer(query, context, args.llm_model, args.llm_url, args.llm_key)
                if predicted != context[:500]:
                    total_llm_calls += 1

            # 3. LLM-Judge
            judge = judge_answer(
                question=query,
                predicted=predicted,
                ground_truth=qa.get("answer_zh", qa.get("answer_en", "")),
                acceptable=qa.get("acceptable_answers_zh", qa.get("acceptable_answers_en", [])),
                model=args.llm_model, url=args.llm_url, api_key=args.llm_key,
            )
            if judge.get("method") == "llm_judge":
                total_llm_calls += 1

            all_results.append({
                "qa_id": qa.get("qa_id", "?"),
                "category": cat,
                "difficulty": qa.get("difficulty", "medium"),
                "correct": judge["correct"],
                "confidence": judge["confidence"],
                "method": judge.get("method", "keyword"),
                "predicted": predicted[:300],
                "ground_truth": qa.get("answer_zh", "")[:200],
            })

        correct = sum(1 for r in all_results if r["category"] == cat and r["correct"])
        cat_total = sum(1 for r in all_results if r["category"] == cat)
        print(f"  {correct}/{cat_total} correct")

    metrics = _aggregate(all_results)
    metrics["meta"]["total_llm_calls"] = total_llm_calls

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Overall: {metrics['overall']['accuracy']:.1%} "
          f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    print(f"LLM calls: {total_llm_calls}")
    print(f"Saved: {args.output}")

    # Per-category summary
    for cat, data in metrics.get("by_category", {}).items():
        print(f"  {cat}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    return metrics


def main() -> None:
    args = parse_args()
    asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()
