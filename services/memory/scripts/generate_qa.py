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
- existence_recall: binary yes/no about object presence. answer_type="boolean". requires_image: usually false (can answer from tags), but set true when asking about visual confirmation ("did the robot SEE X?")
- detail_recall: object attributes (color, size, status). answer_type="description". requires_image: should be true for visual attributes (color, texture, condition), false for factual attributes already in text
- spatiotemporal: where/when an object was seen. answer_type="description". requires_image: true when asking about spatial layout, object positions relative to each other, or map-like understanding
- change_detection: difference between sessions. answer_type="description". requires_image: true when comparing visual states across observations
- anomaly_judgment: abnormal equipment states. answer_type="description". severity: info|warning|critical. requires_image: should be true (visual inspection of equipment/objects)
- reasoning: multi-step inference. answer_type="description". requires_image: true when reasoning chains involve visual evidence

{image_ratio_requirement}

Generate exactly {max_qa} QA pairs. Vary difficulty evenly."""

# Category-specific image dependency guidance
IMAGE_RATIO_REQUIREMENT = (
    "CRITICAL REQUIREMENT: At least {target_pct}% ({target_count} out of {max_qa}) "
    "questions MUST have requires_image=true. These questions must depend on image "
    "information (visual appearance, color, shape, texture, spatial relationships, "
    "object counts visible in the frame, equipment state/condition) that cannot be "
    "answered from text summary alone.\n\n"
    "Image-dependent question examples:\n"
    '- "灭火器是什么颜色的？" / "What color is the fire extinguisher?"\n'
    '- "桌子上放着几瓶水？" / "How many water bottles are on the table?"\n'
    '- "显示器的屏幕是亮着的还是关着的？" / "Is the monitor screen on or off?"\n'
    '- "从图片看，键盘周围还有什么物体？" / "What objects are visible around the keyboard?"\n'
    '- "盆栽植物的叶子状态是否健康？" / "Does the potted plant look healthy?"\n\n'
    "Text-only question examples (can answer from summary/tags):\n"
    '- "是否观察到了灭火器？" / "Was a fire extinguisher observed?"\n'
    '- "灭火器在哪个坐标？" / "At what coordinates was the fire extinguisher?"\n'
    '- "共观察到多少种物体？" / "How many object types were observed?"\n'
)


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


def _enforce_image_ratio(
    pairs: List[Dict], target_ratio: float, category: str = "",
) -> tuple:
    """Post-process QA pairs so requires_image matches target_ratio.

    When the LLM doesn't produce enough image-requiring questions, enrich
    the question text with visual prompts AND flip requires_image on the
    best candidates to hit the target.

    Returns (pairs, image_pct) — image_pct is the post-enforcement ratio.
    """
    if not pairs or target_ratio <= 0:
        pct = sum(1 for q in pairs if q.get("requires_image")) / len(pairs) if pairs else 0
        return pairs, pct

    target_count = max(1, int(len(pairs) * target_ratio + 0.5))
    current = sum(1 for qa in pairs if qa.get("requires_image"))

    visual_keywords = [
        "颜色", "color", "形状", "shape", "大小", "size", "几个",
        "count", "状态", "status", "位置", "where", "长的", "look",
        "appear", "多少", "how many", "什么样", "what does",
        "屏幕", "screen", "亮", "on/off", "周围", "around",
        "纹理", "texture", "表面", "surface", "损坏", "damage",
        "打开", "open", "关闭", "closed", "空的", "empty",
    ]

    # Category-specific visual question templates (zh, en)
    _VISUAL_TEMPLATES: Dict[str, tuple] = {
        "detail_recall": (
            "（请基于图片回答）", " (answer based on the image)",
        ),
        "spatiotemporal": (
            "（参考图片中的空间布局）", " (refer to spatial layout in the image)",
        ),
        "anomaly_judgment": (
            "（通过图片检查设备/物体状态）", " (inspect equipment state from the image)",
        ),
    }

    if current < target_count:
        # Need MORE image-requiring QAs
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
            # Enrich question text with visual prompt
            tmpl = _VISUAL_TEMPLATES.get(category, ("（请基于图片回答）", " (answer based on the image)"))
            if tmpl[0] not in (qa.get("question_zh") or ""):
                qa["question_zh"] = (qa.get("question_zh") or "") + tmpl[0]
            if tmpl[1] not in (qa.get("question_en") or ""):
                qa["question_en"] = (qa.get("question_en") or "") + tmpl[1]
    else:
        # Need FEWER image-requiring QAs — flip the weakest image ones
        candidates = [
            (i, qa) for i, qa in enumerate(pairs)
            if qa.get("requires_image")
        ]
        candidates.sort(key=lambda x: (
            0 if x[1].get("answer_type") in ("boolean",) else 1,
            sum(1 for kw in visual_keywords
                if kw in x[1].get("question_zh", "") or kw in x[1].get("question_en", "")),
        ))
        for i, qa in candidates[:current - target_count]:
            qa["requires_image"] = False

    new_current = sum(1 for q in pairs if q.get("requires_image"))
    new_pct = new_current / len(pairs) if pairs else 0
    return pairs, new_pct


async def generate_category(
    args: argparse.Namespace, nodes: List[Dict],
    category: str, prefix: str,
    target_image_ratio: float = 0.0,
    retry_boost: bool = False,
) -> List[Dict]:
    """Generate QA pairs for one category.

    Args:
        target_image_ratio: If > 0, inject image ratio requirement into prompt.
        retry_boost: If True, this is a regeneration with stronger image emphasis.
    """
    context = build_node_context(nodes, args.batch_size)

    # Build image ratio requirement text
    if target_image_ratio > 0:
        target_count = max(1, int(args.max_qa_per_category * target_image_ratio + 0.5))
        image_req = IMAGE_RATIO_REQUIREMENT.format(
            target_pct=int(target_image_ratio * 100),
            target_count=target_count,
            max_qa=args.max_qa_per_category,
        )
        if retry_boost:
            image_req = (
                "URGENT: Your previous generation had TOO FEW image-dependent questions. "
                + image_req
                + "\nThis time, ensure EVERY question where the answer depends on "
                "visual information (appearance, color, count, state, spatial layout) "
                "has requires_image=true. AIM FOR {target_pct}% OR HIGHER.\n"
            ).format(target_pct=int(target_image_ratio * 100), target_count=target_count, max_qa=args.max_qa_per_category)
    else:
        image_req = ""

    system = SYSTEM_PROMPT.format(
        category=category, prefix=prefix, max_qa=args.max_qa_per_category,
        image_ratio_requirement=image_req,
    )
    full_prompt = f"{system}\n\nMemory nodes:\n{context}"

    label = f"  Generating {category} ({args.max_qa_per_category} pairs)"
    if retry_boost:
        label += " [retry: boosting image ratio]"
    print(label + "...")
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

    img_ratio_flag = args.require_image_ratio if args.require_image_ratio > 0 else 0.0

    for i, cat in enumerate(CATEGORIES):
        if i > 0:
            print(f"  (waiting 3s before next category...)")
            await asyncio.sleep(3)
        prefix = CATEGORY_PREFIX[cat]
        pairs = await generate_category(args, nodes, cat, prefix,
                                        target_image_ratio=img_ratio_flag)

        # Post-process image ratio
        img_pct = 0.0
        if pairs and img_ratio_flag > 0:
            pairs, img_pct = _enforce_image_ratio(pairs, img_ratio_flag, cat)

            # If shortfall > 20% below target, retry with boosted prompt
            if img_pct < img_ratio_flag * 0.8:
                print(f"  Image ratio {img_pct:.0%} << target {img_ratio_flag:.0%} — regenerating...")
                await asyncio.sleep(2)
                pairs2 = await generate_category(args, nodes, cat, prefix,
                                                 target_image_ratio=img_ratio_flag,
                                                 retry_boost=True)
                if pairs2 and len(pairs2) >= len(pairs):
                    pairs2, img_pct2 = _enforce_image_ratio(pairs2, img_ratio_flag, cat)
                    if img_pct2 > img_pct:
                        pairs = pairs2
                        img_pct = img_pct2
                        print(f"  Retry improved: {img_pct2:.0%}")

        all_qa[cat] = pairs
        total += len(pairs)

        # Print per-category image ratio
        img_count = sum(1 for q in pairs if q.get("requires_image"))
        img_pct = (img_count / len(pairs) * 100) if pairs else 0
        tag = "✓" if img_pct >= img_ratio_flag * 100 * 0.8 else "⚠"
        print(f"  → {cat}: {len(pairs)} pairs, {img_count}/{len(pairs)} "
              f"requires_image ({img_pct:.0f}%) {tag}")

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

    # Build per-category image ratio stats for qa_index.yaml
    for cat in CATEGORIES:
        pairs = all_qa.get(cat, [])
        img_count = sum(1 for q in pairs if q.get("requires_image"))
        img_pct = round(img_count / len(pairs) * 100) if pairs else 0
        diff_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        for qa in pairs:
            d = qa.get("difficulty", "medium")
            diff_counts[d] = diff_counts.get(d, 0) + 1
        difficulty_counts[cat] = {**diff_counts, "requires_image_pct": img_pct}

    index = {
        "scene_id": "scenes2",
        "total_qa_pairs": total,
        "categories": difficulty_counts,
        "entries": entries,
    }
    index_path = os.path.join(args.output_dir, "qa_index.yaml")
    with open(index_path, "w") as f:
        yaml.dump(index, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Final image ratio summary
    print(f"\n{'='*55}")
    print(f"  Image Ratio Summary (target: {img_ratio_flag:.0%})")
    print(f"  {'Category':<24} {'Image QA':>8} {'Ratio':>8}")
    print(f"  {'-'*40}")
    for cat in CATEGORIES:
        data = difficulty_counts.get(cat, {})
        img_pct_val = data.get("requires_image_pct", 0)
        tag = " ✓" if img_pct_val >= img_ratio_flag * 100 * 0.8 else " ⚠"
        print(f"  {cat:<24} {data.get('total', 0):>8} {img_pct_val:>7}%{tag}")
    overall_img = sum(d.get("requires_image_pct", 0) for d in difficulty_counts.values())
    overall_avg = round(overall_img / len(CATEGORIES)) if CATEGORIES else 0
    print(f"  {'─'*40}")
    print(f"  {'OVERALL':<24} {total:>8} {overall_avg:>7}%")
    print(f"{'='*55}")
    print(f"\nDone. {total} QA pairs → {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
