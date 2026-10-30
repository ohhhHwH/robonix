#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Generate synthetic object-scene + plan memory (100 each) for retrieval tests.

Each object node keeps the design-doc "observe X in region" embedding text but
varies the *region* beyond "unknown area" (kitchen / living room / bedroom /
office / bathroom …). Each plan node is a natural Chinese task with an English
description. The two sets are written to one JSON file, matching the standard
summary format (embed text only, no vector).

Usage:
    cd services/memory
    rbnx-build/venv/bin/python scripts/generate_synthetic_memory.py \
        [--out ./memory/synthetic_memory_100.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# (English label, Chinese name)
OBJECTS = [
    ("potted_plant", "绿植"),
    ("chair", "椅子"),
    ("table", "桌子"),
    ("lamp", "灯"),
    ("laptop", "笔记本电脑"),
    ("keyboard", "键盘"),
    ("monitor", "显示器"),
    ("cabinet", "柜子"),
    ("couch", "沙发"),
    ("trash_bin", "垃圾桶"),
    ("door", "门"),
    ("shelf", "置物架"),
    ("bed", "床"),
    ("refrigerator", "冰箱"),
    ("microwave", "微波炉"),
    ("sink", "水槽"),
    ("wardrobe", "衣柜"),
    ("desk", "书桌"),
    ("window", "窗户"),
    ("television", "电视"),
]

# (English region, Chinese region)
REGIONS = [
    ("unknown area", "未知区域"),
    ("kitchen", "厨房"),
    ("living room", "客厅"),
    ("bedroom", "卧室"),
    ("office", "办公室"),
]

# Plan task templates: (query pattern, english description pattern). Query ``{}``
# is filled with (region_zh, obj_zh); description ``{}`` with (obj_en, region_en).
PLAN_TEMPLATES = [
    ("在{}找{}", "find the {} in the {}"),
    ("整理{}的{}", "tidy the {} in the {}"),
    ("检查{}的{}", "check the {} in the {}"),
    ("移动{}里的{}", "move the {} in the {}"),
    ("记录{}里{}的位置", "record the position of the {} in the {}"),
]


def _build_scene_nodes() -> list:
    nodes = []
    nid = 0
    for label, zh in OBJECTS:
        for region_en, region_zh in REGIONS:
            nid += 1
            summary_en = f"successfully observe {label} in {region_en}"
            nodes.append({
                "node_id": nid,
                "node_type": "short_term",
                "summary_en": summary_en,
                "summary_zh": f"成功地在{region_zh}观察到{zh}",
                "embed_text": summary_en,
                "objects_present": [label],
                "region": region_en,
                "region_zh": region_zh,
                "action_type": "observe",
                "success": True,
            })
    return nodes


def _build_plan_nodes() -> list:
    nodes = []
    for i, (label, zh) in enumerate(OBJECTS):
        for j, (tpl_q, tpl_d) in enumerate(PLAN_TEMPLATES):
            region_en, region_zh = REGIONS[(i + j) % len(REGIONS)]
            query = tpl_q.format(region_zh, zh)
            description = tpl_d.format(label, region_en)
            nodes.append({
                "doc_id": hashlib.sha1(query.encode("utf-8")).hexdigest(),
                "query": query,
                "description": description,
                "embed_text": f"{query} {description}",
                "plan_count": 1,
                "canceled_count": 0,
            })
    return nodes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic memory JSON.")
    ap.add_argument("--out", default="./memory/synthetic_memory_100.json")
    args = ap.parse_args(argv)

    scene = _build_scene_nodes()
    plan = _build_plan_nodes()
    result = {
        "meta": {
            "scene_nodes": len(scene),
            "plan_nodes": len(plan),
            "scene_embed_text": "successfully observe {object} in {region}",
            "plan_embed_text": "query + ' ' + description",
            "regions": [r[0] for r in REGIONS],
        },
        "scene_nodes": scene,
        "plan_nodes": plan,
    }
    out = os.path.abspath(os.path.expanduser(args.out))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"wrote {out}: scene={len(scene)} plan={len(plan)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
