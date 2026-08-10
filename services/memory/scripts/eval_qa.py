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
    p.add_argument("--generate-report", action="store_true",
                   help="Generate baseline_report.md from existing metrics.json")
    p.add_argument("--report-output", default="baseline_report.md",
                   help="Path for generated report (default: baseline_report.md)")
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


# ── Chinese number normalization ──────────────────────────────────────────

_SIMPLE_ZH = {
    "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "两": "2",
}


def _normalize_zh_numbers(text: str) -> str:
    """Convert Chinese numerals to Arabic digits for fuzzy matching.

    Handles compound numbers: 三十五→35, 二十→20, 十二→12, 十→10.
    """
    # X十Y → XY  (三十五 → 35, 二十四 → 24)
    text = re.sub(
        r'([一二三四五六七八九])十([一二三四五六七八九])',
        lambda m: str(int(_SIMPLE_ZH[m.group(1)]) * 10 + int(_SIMPLE_ZH[m.group(2)])),
        text,
    )
    # X十 → X0  (二十 → 20, 三十 → 30)
    text = re.sub(
        r'([二三四五六七八九])十',
        lambda m: str(int(_SIMPLE_ZH[m.group(1)]) * 10),
        text,
    )
    # 十Y → 1Y  (十二 → 12, 十五 → 15)
    text = re.sub(
        r'十([一二三四五六七八九])',
        lambda m: str(10 + int(_SIMPLE_ZH[m.group(1)])),
        text,
    )
    # Bare 十 → 10
    text = text.replace("十", "10")
    # Single digits
    for zh, num in _SIMPLE_ZH.items():
        text = text.replace(zh, num)
    return text


def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values (int or float) from text.

    Used for relaxed open_short matching when substring fails.
    """
    return [float(m) for m in re.findall(r'\d+(?:\.\d+)?', text)]


def _rule_judge(
    question: str, predicted: str, ground_truth: str,
    acceptable: List[str], answer_type: str,
) -> Optional[Dict[str, Any]]:
    """Fast rule-based judge for boolean and open_short answer types.

    Returns a verdict dict on success, or None if the answer type needs LLM.
    Saves LLM calls for simple yes/no and short-fact questions.
    """
    pt = predicted.strip().lower()
    gt = ground_truth.strip().lower()

    if answer_type == "boolean":
        # Map common boolean expressions to True/False
        def _parse_bool(text: str) -> Optional[bool]:
            pos = [
                "yes", "是", "有", "true", "对", "correct", "存在", "看到",
                "看到了", "观察到了", "检测到了", "有的", "存在的", "是的没错",
                "没错", "可以", "能", "发现了", "记录到", "观察到有",
            ]
            neg = [
                "no", "否", "没有", "无", "false", "不对", "incorrect",
                "不存在", "没看到", "没观察到", "未观察到", "没检测到",
                "没有的", "不存在的", "不可以", "不能", "无法",
                "未发现", "没发现", "未检测到", "未记录", "没记录",
                "没有发现", "没有记录", "无相关",
            ]
            # Exact match first (strip trailing punctuation)
            t = text.rstrip(".。!！?？")
            if t in pos: return True
            if t in neg: return False
            # Substring match: check both lists together, longest match wins.
            # This prevents short pos patterns ("有") from beating longer neg
            # patterns ("没有发现", "没有") in texts like "没有发现灭火器".
            candidates = (
                [(p, True) for p in pos] +
                [(p, False) for p in neg]
            )
            for p, is_pos in sorted(candidates, key=lambda x: len(x[0]), reverse=True):
                if p in t:
                    return is_pos
            return None

        pred_bool = _parse_bool(pt)
        gt_bool = _parse_bool(gt)
        if pred_bool is not None and gt_bool is not None:
            return {"correct": pred_bool == gt_bool, "confidence": 0.95,
                    "method": "rule_boolean"}
        # If we can parse GT but not predicted, check acceptable answers
        if gt_bool is not None:
            for acc in acceptable:
                acc_bool = _parse_bool(acc.strip().lower())
                if acc_bool is not None and pred_bool is not None:
                    return {"correct": pred_bool == acc_bool, "confidence": 0.9,
                            "method": "rule_boolean_acc"}
        return None  # Can't parse → fall back to LLM

    if answer_type == "boolean_with_reasoning":
        # Pre-processing: extract boolean conclusion before reasoning clauses.
        # Many LLMs answer "是，因为..." / "没有，原因是..." — we want the lead-in.
        # Take the fragment before the first comma / period / newline / reasoning
        # connector, then strip residual connector words.
        conclusion = re.split(r'[，,。\n]', predicted.strip())[0].strip().lower()
        conclusion = re.sub(r'^(因为|原因是|由于|根据|理由|因为|鉴于|基于)',
                          '', conclusion).strip()
        # Also try stripping trailing reasoning connectors
        conclusion = re.sub(r'(因为|原因是|由于).*$', '', conclusion).strip()

        # 1) Try substring match (normalized) on full predicted text
        if len(gt) >= 3 and gt in pt:
            return {"correct": True, "confidence": 0.9, "method": "rule_substring"}
        for acc in acceptable:
            acc_lower = acc.strip().lower()
            if len(acc_lower) >= 3 and acc_lower in pt:
                return {"correct": True, "confidence": 0.85,
                        "method": "rule_substring_acc"}

        # 2) Fall back to boolean parsing on the extracted conclusion
        if conclusion:
            bool_result = _rule_judge(
                question, conclusion, ground_truth, acceptable, "boolean",
            )
            if bool_result is not None:
                bool_result["method"] = "rule_bool_from_reasoning"
                return bool_result

        # 3) Last resort: parse full text as boolean
        return _rule_judge(question, predicted, ground_truth, acceptable, "boolean")

    if answer_type == "open_short":
        # 1) Exact substring match (original + normalized both sides)
        if len(gt) >= 2 and gt in pt:
            return {"correct": True, "confidence": 0.9, "method": "rule_substring"}
        # Normalized match: convert "两个" ↔ "2个"
        pt_norm = _normalize_zh_numbers(pt)
        gt_norm = _normalize_zh_numbers(gt)
        if len(gt_norm) >= 2 and gt_norm in pt_norm:
            return {"correct": True, "confidence": 0.88, "method": "rule_substring_zh_norm"}

        for acc in acceptable:
            acc_lower = acc.strip().lower()
            if len(acc_lower) >= 2 and acc_lower in pt:
                return {"correct": True, "confidence": 0.85, "method": "rule_substring_acc"}
            # Normalized acceptable
            acc_norm = _normalize_zh_numbers(acc_lower)
            if len(acc_norm) >= 2 and acc_norm in pt_norm:
                return {"correct": True, "confidence": 0.83,
                        "method": "rule_substring_acc_zh_norm"}

        # 2) Relaxed match: extract numbers and compare
        pt_nums = _extract_numbers(pt_norm)
        gt_nums = _extract_numbers(gt_norm)
        if pt_nums and gt_nums:
            if pt_nums == gt_nums:
                return {"correct": True, "confidence": 0.8,
                        "method": "rule_num_match"}
            # If GT has 1 number and predicted shares it → close enough
            if len(gt_nums) == 1 and gt_nums[0] in pt_nums:
                return {"correct": True, "confidence": 0.75,
                        "method": "rule_num_contains"}
            # Check acceptable answers too
            for acc in acceptable:
                acc_nums = _extract_numbers(_normalize_zh_numbers(acc.strip().lower()))
                if acc_nums and acc_nums == pt_nums:
                    return {"correct": True, "confidence": 0.75,
                            "method": "rule_num_match_acc"}

        # Can't confirm → let LLM judge
        return None

    # For ordered_list and description: needs LLM
    return None


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
    """Compute per-category, per-answer_type, and overall metrics."""
    cats: Dict[str, Dict[str, Any]] = {}
    atypes: Dict[str, Dict[str, int]] = {}  # answer_type → {total, correct}
    vlm_total = 0
    vlm_correct = 0

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

        # Per answer_type
        at = r.get("answer_type", "description")
        if at not in atypes:
            atypes[at] = {"total": 0, "correct": 0}
        atypes[at]["total"] += 1
        if r["correct"]:
            atypes[at]["correct"] += 1

        # VLM-Util tracking
        if r.get("vlm_used"):
            vlm_total += 1
            if r["correct"]:
                vlm_correct += 1

    for cat, data in cats.items():
        data["accuracy"] = round(data["correct"] / data["total"], 3) if data["total"] else 0
        for d in ["easy", "medium", "hard"]:
            dd = data[d]
            dd["accuracy"] = round(dd["correct"] / dd["total"], 3) if dd["total"] else 0

    # Per answer_type accuracy
    atype_metrics = {}
    for at, counts in atypes.items():
        atype_metrics[at] = {
            "total": counts["total"], "correct": counts["correct"],
            "accuracy": round(counts["correct"] / counts["total"], 3) if counts["total"] else 0,
        }

    overall_correct = sum(1 for r in results if r["correct"])
    overall_total = len(results)
    rag_used = sum(1 for r in results if r.get("method") != "keyword")
    return {
        "overall": {"correct": overall_correct, "total": overall_total,
                     "accuracy": round(overall_correct/overall_total, 3) if overall_total else 0},
        "by_category": cats,
        "by_answer_type": atype_metrics,
        "vlm_util": {
            "total_vlm_qa": vlm_total,
            "vlm_correct": vlm_correct,
            "vlm_accuracy": round(vlm_correct / vlm_total, 3) if vlm_total else 0,
            "vlm_utilization": round(vlm_total / overall_total, 3) if overall_total else 0,
        },
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

    # Per-QA delay to avoid rate limiting on LLM APIs (env-overridable)
    _QA_DELAY_S = float(os.environ.get("EVAL_QA_DELAY_S", "1.5"))

    for cat_idx, cat in enumerate(CATEGORIES):
        qa_file = os.path.join(args.qa_dir, f"{cat}.json")
        if not os.path.exists(qa_file):
            print(f"  SKIP: {qa_file} not found")
            continue

        with open(qa_file) as f:
            pairs = json.load(f)

        print(f"\nEvaluating {cat} ({len(pairs)} pairs)...")

        for qa_idx, qa in enumerate(pairs):
            query = qa.get("question_zh", qa.get("question_en", ""))

            # 1. Search memory (with optional VLM QA)
            tags = TagFilter(task_type="observe")
            vlm_answer = ""
            try:
                search_resp = await svc.search(query, tags=tags, top_k=5,
                                               vlm_qa=args.vlm_qa)
                nodes = search_resp.nodes if search_resp.nodes else []
                vlm_answer = search_resp.vlm_answer or ""
            except Exception as e:
                print(f"  [ERROR] search failed for {qa.get('qa_id', '?')}: {e}")
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

            # 2. Answer: VLM (if enabled and returned answer) > RAG (text) > raw context
            if vlm_answer:
                predicted = vlm_answer
                total_llm_calls += 1  # VLM call counted
            elif args.no_rag:
                predicted = context
            else:
                predicted = rag_answer(query, context, args.llm_model, args.llm_url, args.llm_key)
                if predicted != context[:500]:
                    total_llm_calls += 1

            # 3. Judge: rule-based first (fast), LLM-Judge as fallback
            answer_type = qa.get("answer_type", "description")
            judge = _rule_judge(
                question=query, predicted=predicted,
                ground_truth=qa.get("answer_zh", qa.get("answer_en", "")),
                acceptable=qa.get("acceptable_answers_zh", qa.get("acceptable_answers_en", [])),
                answer_type=answer_type,
            )
            if judge is None:
                # Rule-based couldn't decide → fall back to LLM judge
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
                "answer_type": answer_type,
                "correct": judge["correct"],
                "confidence": judge["confidence"],
                "method": judge.get("method", "keyword"),
                "predicted": predicted[:300],
                "ground_truth": qa.get("answer_zh", "")[:200],
                "vlm_used": bool(vlm_answer),
            })

            # Rate-limit guard: pause between QA pairs to avoid LLM API throttling
            is_last = (cat_idx == len(CATEGORIES) - 1 and qa_idx == len(pairs) - 1)
            if not is_last:
                await asyncio.sleep(_QA_DELAY_S)

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


def _generate_report(metrics: Dict[str, Any], qa_dir: str, output_path: str) -> None:
    """Generate baseline_report.md from metrics and QA metadata."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    overall = metrics["overall"]
    by_cat = metrics.get("by_category", {})
    by_atype = metrics.get("by_answer_type", {})
    vlm = metrics.get("vlm_util", {})
    meta = metrics.get("meta", {})

    # Per-category rows
    cat_rows = []
    for cat in ["existence_recall", "detail_recall", "spatiotemporal",
                "change_detection", "anomaly_judgment", "reasoning"]:
        data = by_cat.get(cat, {})
        if data:
            cat_rows.append(
                f"| {cat} | {data.get('accuracy', 0):.1%} "
                f"({data.get('correct', 0)}/{data.get('total', 0)}) |"
            )

    # Per answer_type rows
    atype_rows = []
    for at in ["boolean", "boolean_with_reasoning", "open_short", "ordered_list", "description"]:
        data = by_atype.get(at, {})
        if data:
            atype_rows.append(
                f"| {at} | {data.get('accuracy', 0):.1%} "
                f"({data.get('correct', 0)}/{data.get('total', 0)}) |"
            )

    # Verification targets
    existence_acc = by_cat.get("existence_recall", {}).get("accuracy", 0)
    vlm_util = vlm.get("vlm_utilization", 0)
    check1 = "✓" if existence_acc >= 0.8 else "✗"
    check2 = "✓" if vlm_util > 0 else "✗"

    report = f"""# ScribeMem-Bench Baseline Report

> Auto-generated: {now}

## Overall Results

| Metric | Value |
|---|---|
| Overall Accuracy | **{overall['accuracy']:.1%}** ({overall['correct']}/{overall['total']}) |
| LLM Calls | {meta.get('total_llm_calls', '?')} |
| Rule-Judge Saved | {sum(1 for v in [by_atype])} |

## Per Category

| Category | Accuracy |
|---|---|
{chr(10).join(cat_rows)}

## Per Answer Type

| Answer Type | Accuracy |
|---|---|
{chr(10).join(atype_rows)}

## VLM Utilization

| Metric | Value |
|---|---|
| QA pairs using VLM | {vlm.get('total_vlm_qa', 0)}/{overall['total']} |
| VLM Accuracy | {vlm.get('vlm_accuracy', 0):.1%} |
| VLM Utilization Rate | {vlm_util:.1%} |

## Verification Targets

| Target | Threshold | Actual | Status |
|---|---|---|---|
| existence_recall | ≥ 80% | {existence_acc:.1%} | {check1} |
| VLM-Util > 0 | > 0 | {vlm_util:.1%} | {check2} |

## Configuration

- **Mode**: Embodied (ObjectWatchdog simulation)
- **Retrieval**: BM25 + LLM Ranker (with retry + backup)
- **QA Judge**: Rule-based (boolean/open_short) + LLM-Judge (others)
- **VLM QA**: {"Enabled" if vlm.get('total_vlm_qa', 0) > 0 else "Disabled/Unused"}
- **QA Directory**: `{qa_dir}`

---
*Generated by ScribeMem-Bench eval pipeline*
"""
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved: {output_path}")


def main() -> None:
    args = parse_args()
    if args.generate_report:
        # Generate report from existing metrics.json (skip eval)
        if not os.path.exists(args.output):
            print(f"ERROR: metrics file not found: {args.output}")
            sys.exit(1)
        with open(args.output) as f:
            metrics = json.load(f)
        _generate_report(metrics, args.qa_dir, args.report_output)
    else:
        asyncio.run(_run_eval(args))


if __name__ == "__main__":
    main()
