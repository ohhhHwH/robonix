#!/usr/bin/env python3
"""Evaluate QA pairs against a memory graph using hybrid_search + LLM-Judge.

For each QA pair:
  1. hybrid_search(query) → candidate memory nodes
  2. If requires_image → vlm_qa(query, node images) → image-based answer
  3. LLM-Judge compares predicted answer with ground truth → correct/incorrect
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
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_SVC = _HERE.parent
sys.path.insert(0, str(_SVC))

from memory_service.core.types import SearchRequest, TagFilter


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
    return p.parse_args()


def search_memory(query: str, memory_dir: str, top_k: int = 10) -> List[Dict]:
    """Search memory graph via hybrid_search."""
    from memory_service.service import MemoryService

    svc = MemoryService(data_dir=memory_dir)
    tags = TagFilter(task_type="observe")

    async def _search():
        resp = await svc.search(query, tags=tags, top_k=top_k)
        return [n.to_dict() for n in resp.nodes]

    return asyncio.run(_search())


def llm_judge(
    question: str, predicted: str, ground_truth: str,
    acceptable: List[str], model: str, url: str, api_key: str,
) -> Dict[str, Any]:
    """LLM-as-Judge: compare predicted answer vs ground truth."""
    if not api_key:
        # No LLM → simple keyword match
        matched = ground_truth.lower() in predicted.lower() or \
                  any(a.lower() in predicted.lower() for a in acceptable)
        return {"correct": matched, "confidence": 1.0 if matched else 0.0,
                "method": "keyword"}

    prompt = f"""You are a QA evaluator. Judge whether the PREDICTED answer correctly answers the question
based on the GROUND TRUTH. Consider acceptable answers as equally valid.

QUESTION: {question}
PREDICTED ANSWER: {predicted}
GROUND TRUTH: {ground_truth}
ACCEPTABLE ANSWERS: {json.dumps(acceptable)}

Reply with ONLY a JSON object: {{"correct": true/false, "reasoning": "...", "confidence": 0.0-1.0}}"""

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        req = urllib.request.Request(f"{url}/chat/completions", data=payload, headers=headers)
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        result = json.loads(resp["choices"][0]["message"]["content"])
        return {"correct": result.get("correct", False),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "method": "llm_judge"}
    except Exception as e:
        print(f"  Judge error: {e}")
        return {"correct": False, "confidence": 0.0, "method": "error"}


def evaluate_category(
    qa_file: str, category: str, args: argparse.Namespace,
) -> List[Dict]:
    """Evaluate all QA pairs in one category file."""
    if not os.path.exists(qa_file):
        print(f"  SKIP: {qa_file} not found")
        return []

    with open(qa_file) as f:
        pairs = json.load(f)

    results = []
    svc = None
    from memory_service.service import MemoryService
    svc = MemoryService(data_dir=args.memory_dir)

    for qa in pairs:
        # 1. Search memory
        query = qa.get("question_zh", qa.get("question_en", ""))
        nodes = search_memory(query, args.memory_dir)

        # Collect relevant info
        node_summaries = [n.get("summary", "") for n in nodes[:5]]
        predicted = "\n".join(node_summaries) if node_summaries else "no memory found"

        # 2. LLM-Judge
        judge = llm_judge(
            question=query,
            predicted=predicted,
            ground_truth=qa.get("answer_zh", qa.get("answer_en", "")),
            acceptable=qa.get("acceptable_answers_zh", qa.get("acceptable_answers_en", [])),
            model=args.llm_model, url=args.llm_url, api_key=args.llm_key,
        )

        results.append({
            "qa_id": qa.get("qa_id", "?"),
            "category": category,
            "difficulty": qa.get("difficulty", "medium"),
            "correct": judge["correct"],
            "confidence": judge["confidence"],
            "method": judge.get("method", "keyword"),
            "predicted": predicted[:200],
            "ground_truth": qa.get("answer_zh", "")[:200],
        })

    return results


def aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
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

    # Compute rates
    for cat, data in cats.items():
        data["accuracy"] = round(data["correct"] / data["total"], 3) if data["total"] else 0
        for d in ["easy", "medium", "hard"]:
            dd = data[d]
            dd["accuracy"] = round(dd["correct"] / dd["total"], 3) if dd["total"] else 0

    overall_correct = sum(1 for r in results if r["correct"])
    overall_total = len(results)
    return {
        "overall": {"correct": overall_correct, "total": overall_total,
                     "accuracy": round(overall_correct/overall_total, 3) if overall_total else 0},
        "by_category": cats,
    }


async def main() -> None:
    args = parse_args()
    CATEGORIES = ["existence_recall", "detail_recall", "spatiotemporal",
                  "change_detection", "anomaly_judgment", "reasoning"]

    all_results = []
    for cat in CATEGORIES:
        qa_file = os.path.join(args.qa_dir, f"{cat}.json")
        print(f"Evaluating {cat}...")
        results = evaluate_category(qa_file, cat, args)
        all_results.extend(results)
        correct = sum(1 for r in results if r["correct"])
        print(f"  {correct}/{len(results)} correct")

    metrics = aggregate_metrics(all_results)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nOverall: {metrics['overall']['accuracy']:.1%} "
          f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
