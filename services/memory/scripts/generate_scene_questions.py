#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Generate ``scene_questions_100.json`` + ``query_100.json``.

``scene_questions_100.json`` is the stored scene-question memory (100 distinct
questions = 20 objects x 5 question types); ``query_100.json`` is 100 queries
that are *similar but not identical* to those questions (one paraphrase each,
linked by ``question_id``). The ``embed_text``/``query`` duplication is removed:
each entry carries exactly one text field.

Usage:
    cd services/memory
    rbnx-build/venv/bin/python scripts/generate_scene_questions.py
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


def main() -> int:
    questions, queries = [], []
    for label, zh in OBJECTS:
        for key, tzh, intent, tpl in QTYPES:
            qid = f"SQ{len(questions) + 1:03d}"
            questions.append({
                "id": qid,
                "type": key,
                "type_zh": tzh,
                "intent": intent,
                "object": label,
                "object_zh": zh,
                "question": tpl.format(zh),
            })
            queries.append({
                "id": f"Q{len(queries) + 1:03d}",
                "question_id": qid,
                "type": key,
                "object": label,
                "object_zh": zh,
                "query": PARAPHRASES[key].format(zh),
            })

    q_data = {
        "meta": {
            "total": len(questions),
            "question_types": [
                {"type": k, "type_zh": z, "intent": it} for k, z, it, _t in QTYPES
            ],
            "objects": len(OBJECTS),
            "embed_text": "question 本身（中文）",
        },
        "scene_questions": questions,
    }
    u_data = {
        "meta": {
            "total": len(queries),
            "linked_to": "scene_questions_100.json",
            "query_types": [k for k, _z, _it, _t in QTYPES],
        },
        "queries": queries,
    }

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "memory", "scene_questions_100.json"),
              "w", encoding="utf-8") as f:
        json.dump(q_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(base, "memory", "query_100.json"),
              "w", encoding="utf-8") as f:
        json.dump(u_data, f, ensure_ascii=False, indent=2)
    print(f"wrote scene_questions_100.json ({len(questions)}) + "
          f"query_100.json ({len(queries)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
