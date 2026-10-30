#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Generate the three synthetic node/query files for the mem-node retrieval test.

Replaces the earlier ``scene_questions_100.json`` + ``query_100.json`` naming
and drops the real ``node_embed_texts.json`` data (per "不要真实场景"):

* ``mem_node_100_1030.json``   — 100 "mem node" (stored questions, 20 objects × 5
                                 question types). Embedding text = the question.
* ``scene_node_100_1030.json`` — 100 "scene node" (object observations, 20
                                 objects × 5 regions). Embedding text =
                                 ``successfully observe <obj> in <region>``.
* ``query_1030.json``          — 100 test queries, one paraphrase per mem node,
                                 linked by ``mem_node_id``.

Usage:
    cd services/memory
    rbnx-build/venv/bin/python scripts/generate_mem_scene_nodes.py
"""

from __future__ import annotations

import json
import os
import sys

OBJECTS = [
    ("potted_plant", "绿植"), ("chair", "椅子"), ("table", "桌子"),
    ("lamp", "灯"), ("laptop", "笔记本电脑"), ("keyboard", "键盘"),
    ("monitor", "显示器"), ("cabinet", "柜子"), ("couch", "沙发"),
    ("trash_bin", "垃圾桶"), ("door", "门"), ("shelf", "置物架"),
    ("bed", "床"), ("refrigerator", "冰箱"), ("microwave", "微波炉"),
    ("sink", "水槽"), ("wardrobe", "衣柜"), ("desk", "书桌"),
    ("window", "窗户"), ("television", "电视"),
]

REGIONS = [
    ("unknown area", "未知区域"), ("kitchen", "厨房"), ("living room", "客厅"),
    ("bedroom", "卧室"), ("office", "办公室"),
]

# (type, type_zh, intent, question template)
QTYPES = [
    ("where",     "位置",     "物体定位", "哪里有{}？"),
    ("existence", "存在",     "存在判断", "这里有没有{}？"),
    ("count",     "数量",     "数量统计", "{}有几个？"),
    ("state",     "状态",     "状态查询", "{}现在是什么状态？"),
    ("recall",    "记忆回顾", "记忆回顾", "我上次看到{}是什么时候？"),
]

# type -> query paraphrase template (近似但不同)
PARAPHRASES = {
    "where":     "{}在哪个位置？",
    "existence": "这个环境里是否存在{}？",
    "count":     "{}一共有多少个？",
    "state":     "{}目前处于什么状态？",
    "recall":    "我之前是在什么时间看到{}的？",
}


def _write(base, name, data):
    with open(os.path.join(base, "memory", name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"wrote {name}")


def main() -> int:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    mem, scene, queries = [], [], []
    for label, zh in OBJECTS:
        for key, tzh, intent, tpl in QTYPES:
            mem.append({
                "id": f"M{len(mem) + 1:03d}",
                "type": key,
                "type_zh": tzh,
                "intent": intent,
                "object": label,
                "object_zh": zh,
                "question": tpl.format(zh),
            })
    for label, zh in OBJECTS:
        for region_en, region_zh in REGIONS:
            scene.append({
                "id": f"SN{len(scene) + 1:03d}",
                "object": label,
                "object_zh": zh,
                "region": region_en,
                "region_zh": region_zh,
                "embed_text": f"successfully observe {label} in {region_en}",
            })
    for m in mem:
        queries.append({
            "id": f"Q{len(queries) + 1:03d}",
            "mem_node_id": m["id"],
            "type": m["type"],
            "object": m["object"],
            "object_zh": m["object_zh"],
            "query": PARAPHRASES[m["type"]].format(m["object_zh"]),
        })

    _write(base, "mem_node_100_1030.json", {
        "meta": {"total": len(mem), "node_type": "mem",
                 "embed_text": "question 本身（中文）",
                 "question_types": len(QTYPES), "objects": len(OBJECTS)},
        "mem_nodes": mem,
    })
    _write(base, "scene_node_100_1030.json", {
        "meta": {"total": len(scene), "node_type": "scene",
                 "embed_text": "successfully observe {object} in {region}",
                 "regions": len(REGIONS), "objects": len(OBJECTS)},
        "scene_nodes": scene,
    })
    _write(base, "query_1030.json", {
        "meta": {"total": len(queries), "linked_to": "mem_node_100_1030.json"},
        "queries": queries,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
