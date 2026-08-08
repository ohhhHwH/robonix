#!/usr/bin/env python3
"""Generate 6-category bilingual QA pairs from memory_nodes.json via LLM.

Reads a memory graph, formats nodes as structured context, prompts an LLM
to generate QA pairs covering 6 cognitive dimensions, and outputs
per-category JSON files + qa_index.yaml following the ScribeMem-Bench spec.

Usage:
  uv run python scripts/generate_qa.py \
    --input memory_nodes.json \
    --output-dir ./qa_output \
    --llm-model deepseek-v4-flash \
    --llm-url https://api.deepseek.com \
    --llm-key $VLM_API_KEY \
    --max-qa-per-category 5

Format reference: datasets-struct.md Appendices B-G.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.request
import yaml

CATEGORIES = [
    "existence_recall",
    "detail_recall",
    "spatiotemporal",
    "change_detection",
    "anomaly_judgment",
    "reasoning",
]

CATEGORY_PREFIX = {"existence_recall": "E", "detail_recall": "D",
                   "spatiotemporal": "S", "change_detection": "C",
                   "anomaly_judgment": "A", "reasoning": "R"}

SYSTEM_PROMPT = """You are a QA generator for an embodied memory benchmark. Given memory nodes from a robot inspection patrol, generate bilingual (zh/en) question-answer pairs for the category "{category}".

Output ONLY valid JSON array. Each entry must have:
- qa_id: "{prefix}XXX" (sequential)
- category: "{category}"
- difficulty: "easy"/"medium"/"hard"
- session_ids: ["session_001"]
- related_node_ids: [node_ids]
- related_object_ids: [object_ids]
- requires_image: true/false
- question_zh / question_en: bilingual questions
- answer_zh / answer_en: bilingual answers
- answer_type: "boolean"/"boolean_with_reasoning"/"open_short"/"ordered_list"/"description"
- evidence_zh / evidence_en: citation to specific node(s) or "objects.yaml GT"
- acceptable_answers_zh / acceptable_answers_en: list of acceptable alternatives

Category-specific rules:
- existence_recall: binary yes/no about object presence. answer_type="boolean"
- detail_recall: object attributes (color, size, status). answer_type="description"
- spatiotemporal: where/when an object was seen. answer_type="description"
- change_detection: difference between sessions. answer_type="description"
- anomaly_judgment: abnormal equipment states. answer_type="description". severity: info|warning|critical
- reasoning: multi-step inference. answer_type="description"

Generate exactly {max_qa} QA pairs. Vary difficulty evenly."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QA pairs from memory nodes")
    p.add_argument("--input", required=True, help="Path to memory_nodes.json")
    p.add_argument("--output-dir", default="./qa_output", help="Output directory")
    p.add_argument("--llm-model", default=os.environ.get("VLM_MODEL", "deepseek-v4-flash"))
    p.add_argument("--llm-url", default=os.environ.get("VLM_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--llm-key", default=os.environ.get("VLM_API_KEY", ""))
    p.add_argument("--max-qa-per-category", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=50,
                   help="Max nodes per LLM call (controls prompt length)")
    p.add_argument("--require-image-ratio", type=float, default=0.0,
                   help="Target fraction of QA pairs with requires_image=true (0.0-1.0)")
    return p.parse_args()


def build_node_context(nodes: List[Dict], max_nodes: int = 50) -> str:
    """Format a subset of memory nodes as structured text for the LLM prompt."""
    selected = nodes[:max_nodes]
    lines = []
    for n in selected:
        tags = n.get("tags", {})
        spatial = n.get("spatial_data", {})
        objs = []
        for o in spatial.get("objects", []):
            objs.append(f"{o.get('label','?')}@({o.get('x',0):.1f},{o.get('y',0):.1f})")
        lines.append(
            f"node_{n['node_id']}: "
            f"summary=\"{n.get('summary','')}\" "
            f"type={n.get('node_type','?')} "
            f"ts={n.get('timestamp',0)} "
            f"objects=[{', '.join(objs)}] "
            f"scene={tags.get('scene_type','?')} "
            f"region={tags.get('region','?')} "
            f"action={tags.get('action_type','?')} "
            f"task={tags.get('task_type','?')} "
        )
    return "\n".join(lines)


async def call_llm(
    prompt: str, model: str, url: str, api_key: str,
) -> Optional[str]:
    """Call an OpenAI-compatible chat completion API (sync wrapper)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the QA pairs now."},
        ],
        "temperature": 0.7,
        "max_tokens": 16384,
    }).encode()
    req = urllib.request.Request(f"{url}/chat/completions", data=payload, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        if content is None:
            finish = data["choices"][0].get("finish_reason", "?")
            print(f"  LLM returned null content (finish_reason={finish})")
            return None
        return content
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def extract_json(text: str) -> Optional[List[Dict]]:
    """Extract JSON array from LLM output that may contain markdown fences."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON array between [ and ]
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Try to recover truncated JSON: close open brackets
    if text.strip().startswith('['):
        # Count brackets and close them
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        if open_braces > 0 or open_brackets > 0:
            # Truncated — try to close the last complete object
            # Remove the last incomplete object
            last_complete = text.rfind('},')
            if last_complete > 0:
                recovered = text[:last_complete+1] + '\n]'
                try:
                    return json.loads(recovered)
                except json.JSONDecodeError:
                    pass
            # Try last complete object without comma
            last_brace = text.rfind('}')
            if last_brace > 0:
                recovered = text[:last_brace+1] + '\n]'
                try:
                    result = json.loads(recovered)
                    if isinstance(result, list) and len(result) > 0:
                        print(f"  Recovered {len(result)} QA pairs from truncated JSON")
                        return result
                except json.JSONDecodeError:
                    pass
    return None


def _enforce_image_ratio(pairs: List[Dict], target_ratio: float) -> List[Dict]:
    """Post-process QA pairs so requires_image matches target_ratio.

    When the LLM doesn't produce enough (or too many) image-requiring questions,
    flip requires_image on the best candidates to hit the target.

    Priority for flipping to True: questions about visual details (color, shape,
    count, appearance).  Priority for flipping to False: existence / boolean
    questions that can be answered from tags alone.
    """
    if not pairs or target_ratio <= 0:
        return pairs

    target_count = max(1, round(len(pairs) * target_ratio))
    current = sum(1 for qa in pairs if qa.get("requires_image"))

    if current == target_count:
        return pairs

    visual_keywords = ["颜色", "color", "形状", "shape", "大小", "size", "几个",
                       "count", "状态", "status", "位置", "where", "长的", "look",
                       "appear", "多少", "how many", "什么样", "what does"]

    if current < target_count:
        # Need MORE image-requiring QAs — flip the best non-image ones
        candidates = [
            (i, qa) for i, qa in enumerate(pairs)
            if not qa.get("requires_image")
        ]
        # Prefer questions with visual keywords
        candidates.sort(key=lambda x: sum(
            1 for kw in visual_keywords
            if kw in x[1].get("question_zh", "") or kw in x[1].get("question_en", "")
        ), reverse=True)
        for i, qa in candidates[:target_count - current]:
            qa["requires_image"] = True
            if qa.get("answer_type") == "boolean":
                qa["answer_type"] = "boolean_with_reasoning"
    else:
        # Need FEWER image-requiring QAs — flip the weakest image ones
        candidates = [
            (i, qa) for i, qa in enumerate(pairs)
            if qa.get("requires_image")
        ]
        # Prefer flipping boolean/existence QAs back to non-image
        candidates.sort(key=lambda x: (
            0 if x[1].get("answer_type") in ("boolean",) else 1,
            sum(1 for kw in visual_keywords
                if kw in x[1].get("question_zh", "") or kw in x[1].get("question_en", "")),
        ))
        for i, qa in candidates[:current - target_count]:
            qa["requires_image"] = False

    return pairs


async def generate_category(
    args: argparse.Namespace, nodes: List[Dict],
    category: str, prefix: str,
) -> List[Dict]:
    """Generate QA pairs for one category."""
    context = build_node_context(nodes, args.batch_size)
    system = SYSTEM_PROMPT.format(
        category=category, prefix=prefix, max_qa=args.max_qa_per_category,
    )
    full_prompt = f"{system}\n\nMemory nodes:\n{context}"

    print(f"  Generating {category} ({args.max_qa_per_category} pairs)...")
    result = await call_llm(full_prompt, args.llm_model, args.llm_url, args.llm_key)
    if result is None:
        print(f"  FAILED: no LLM response")
        return []

    qa_pairs = extract_json(result)
    if qa_pairs is None:
        print(f"  FAILED: could not parse JSON from LLM output")
        print(f"  Raw (first 500): {result[:500]}")
        return []

    # Assign sequential IDs
    for i, qa in enumerate(qa_pairs):
        qa["qa_id"] = f"{prefix}{i+1:03d}"
        if "category" not in qa:
            qa["category"] = category

    print(f"  Generated {len(qa_pairs)} pairs")
    return qa_pairs


async def main() -> None:
    args = parse_args()
    nodes = json.loads(Path(args.input).read_text())
    print(f"Nodes: {len(nodes)}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_qa: Dict[str, List[Dict]] = {}
    total = 0
    difficulty_counts: Dict[str, Dict[str, int]] = {}

    for i, cat in enumerate(CATEGORIES):
        if i > 0:
            print(f"  (waiting 3s before next category...)")
            await asyncio.sleep(3)
        prefix = CATEGORY_PREFIX[cat]
        pairs = await generate_category(args, nodes, cat, prefix)
        if args.require_image_ratio > 0:
            pairs = _enforce_image_ratio(pairs, args.require_image_ratio)
        all_qa[cat] = pairs
        total += len(pairs)

        # Count difficulties
        counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        img_count = 0
        for qa in pairs:
            d = qa.get("difficulty", "medium")
            counts[d] = counts.get(d, 0) + 1
            if qa.get("requires_image"):
                img_count += 1
        pct = (img_count / len(pairs) * 100) if pairs else 0
        difficulty_counts[cat] = {**counts, "requires_image_pct": round(pct)}

        # Write per-category file
        out_path = os.path.join(args.output_dir, f"{cat}.json")
        with open(out_path, "w") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

    # Write qa_index.yaml
    entries = []
    for cat in CATEGORIES:
        for qa in all_qa[cat]:
            entries.append({
                "qa_id": qa["qa_id"],
                "category": cat,
                "difficulty": qa.get("difficulty", "medium"),
                "session_ids": qa.get("session_ids", ["session_001"]),
                "node_ids": qa.get("related_node_ids", []),
                "object_ids": qa.get("related_object_ids", []),
                "requires_image": qa.get("requires_image", False),
            })

    index = {
        "scene_id": "scenes2",
        "total_qa_pairs": total,
        "categories": difficulty_counts,
        "entries": entries,
    }
    index_path = os.path.join(args.output_dir, "qa_index.yaml")
    with open(index_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nDone. {total} QA pairs → {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
