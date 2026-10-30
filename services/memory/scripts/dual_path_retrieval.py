#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Dual-path retrieval: one query → two independent paths (single BGE-M3 model).

The three input files are the test architecture:

* ``memory/node_embed_texts.json``   — stored memory nodes (27 scene + 23 plan);
* ``memory/scene_questions_100.json``— stored scene-question memory (100);
* ``memory/query_100.json``          — 100 test queries (paraphrases of the
                                       scene questions, linked by ``question_id``).

Each query is embedded **once** with BGE-M3 (1024-dim), then ranked over two
paths, each returning its own FULL cosine-similarity vector (never merged):

* **node path**  — query × every memory node (object summaries + plan tasks);
* **scene path** — query × every stored scene question.

The scene path is self-verifiable: query ``Qxxx`` is a paraphrase of its linked
``question_id``, so a correct retrieval must rank that question first. The node
path is verified against the object label in the node's embed text
(``observe <object>``). Results are written to a JSON (full scores) and an HTML
summary.

Usage:
    cd services/memory
    HF_ENDPOINT=https://hf-mirror.com rbnx-build/venv/bin/python \
        scripts/dual_path_retrieval.py \
        [--node-json ./memory/node_embed_texts.json] \
        [--scene-questions ./memory/scene_questions_100.json] \
        [--queries ./memory/query_100.json] \
        [--out-json ./memory/dual_path_result.json] \
        [--out-html ./memory/dual_path_summary.html]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_embedder():
    from memory_service.storage.bge_embedder import BgeEmbedder
    from memory_service.storage.embedding_config import EmbeddingModelConfig

    return BgeEmbedder(EmbeddingModelConfig())


def _embed(emb, texts):
    out = []
    for t in texts:
        v = np.asarray(emb.encode(t), dtype=np.float32)
        out.append(v / (np.linalg.norm(v) + 1e-12))
    return out


def _load_nodes(path):
    """Return (nodes, order) where nodes = {key: {kind, text}}."""
    nodes, order = {}, []
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for n in d.get("scene_nodes", []):
        key = f"scene:{n['node_id']}"
        nodes[key] = {"kind": "scene", "text": n.get("embed_text", "").strip()}
        order.append(key)
    for n in d.get("plan_nodes", []):
        key = f"plan:{n['doc_id']}"
        nodes[key] = {"kind": "plan", "text": n.get("embed_text", "").strip()}
        order.append(key)
    return nodes, order


def _hit_at(scores, keys, top_k):
    """True if any of the top-k ranked keys is a ground-truth key."""
    return any(k in keys for k in scores[:top_k])


def _render_html(result, html_path) -> str:
    meta = result["meta"]
    rows = result["results"]
    scene_hits = [r["scene_hit"] for r in rows]
    node_hits = [r["node_hit"] for r in rows]
    covered = [r for r in rows if r["object_covered"]]
    covered_hits = [r["node_hit"] for r in covered]
    uncovered = [r for r in rows if not r["object_covered"]]

    def pct(xs):
        return f"{sum(xs) / len(xs) * 100:.1f}%" if xs else "—"

    trs = []
    for r in rows:
        st1 = r["scene_top1"]
        st1_id = st1["id"]
        st1_s = st1["cos"]
        st1_txt = st1["text"]
        ok = "hit" if r["scene_hit"] else "miss"
        nt1 = r["node_top1"]
        nt1_s = nt1["cos"]
        nt1_txt = nt1["text"][:46]
        nok = "hit" if r["node_hit"] else "miss"

        scene_cells = "".join(
            f"<span class='cell'>{_esc(sid)} <b>{s:.4f}</b></span>"
            for sid, s in r["scene_scores"].items()
        )
        node_cells = "".join(
            f"<span class='cell'>{_esc(k.split(':', 1)[0])}:"
            f"{_esc(k.split(':', 1)[1][:10])} <b>{s:.4f}</b></span>"
            for k, s in r["node_scores"].items()
        )

        trs.append(
            f"<tr><td>{r['query_id']}</td><td>{_esc(r['query'])}</td>"
            f"<td>{_esc(r['object_zh'])}</td>"
            f"<td class='{ok}'>{st1_id} <b>{st1_s:.4f}</b><br>"
            f"<span class='sub'>{_esc(st1_txt[:30])}</span></td>"
            f"<td class='{nok}'><b>{nt1_s:.4f}</b><br>"
            f"<span class='sub'>{_esc(nt1_txt)}</span></td></tr>"
            f"<tr class='detail'><td colspan='5'>"
            f"<details><summary>详细 · scene {len(r['scene_scores'])} / node "
            f"{len(r['node_scores'])} 全量降序</summary>"
            f"<div class='path'>scene 路径</div><div class='grid'>{scene_cells}</div>"
            f"<div class='path'>node 路径</div><div class='grid'>{node_cells}</div>"
            f"</details></td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双路径检索总结 — BGE-M3 单模型</title>
<style>
:root {{ --bg:#0f1420; --panel:#1a2130; --line:#2a3348; --fg:#e7ecf5;
        --dim:#8b96ab; --hit:#2ecc71; --miss:#e74c3c; --accent:#6ea8ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
       "PingFang SC","Microsoft YaHei",sans-serif; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:28px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:26px 0 10px; color:var(--accent); }}
.sub {{ color:var(--dim); font-size:12px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:12px; margin:16px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:14px; }}
.card .n {{ font-size:24px; font-weight:700; }}
.card .l {{ color:var(--dim); font-size:12px; margin-top:2px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border:1px solid var(--line); padding:6px 8px; text-align:left;
         vertical-align:top; }}
th {{ background:var(--panel); position:sticky; top:0; }}
td.hit {{ color:var(--hit); }}
td.miss {{ color:var(--miss); }}
.arch {{ background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:14px; font-family:ui-monospace,Menlo,
         Consolas,monospace; font-size:12.5px; white-space:pre-wrap; }}
.note {{ color:var(--dim); font-size:12.5px; }}
caption {{ caption-side:top; text-align:left; color:var(--dim); font-size:12px;
          padding:4px 0 8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
         gap:1px 12px; font-size:12px; margin:6px 0; }}
.cell {{ font-family:ui-monospace,Menlo,Consolas,monospace; color:var(--dim);
         white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.cell b {{ color:var(--fg); font-weight:600; }}
.path {{ color:var(--accent); font-size:12px; font-weight:700; margin-top:10px; }}
details {{ margin-top:2px; }}
details summary {{ cursor:pointer; color:var(--accent); font-size:12px; }}
tr.detail > td {{ background:var(--panel); }}
th .thsub {{ color:var(--dim); font-weight:400; font-size:11px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>双路径检索总结</h1>
<p class="sub">模型 {_esc(meta['embedder'])} · {meta['dim']} 维 · 余弦相似度 ·
生成于 {meta['generated_at'][:19]}</p>

<h2>架构</h2>
<div class="arch">query (100, 中文) ──BGE-M3 单次编码──▶ qv (1024)
        ├─ node 路径  × {meta['node_count']} 记忆节点 ({meta['scene_nodes']} 场景 + {meta['plan_nodes']} 规划) → 全量相似度
        └─ scene 路径 × {meta['scene_questions']} 场景问题 → 全量相似度</div>

<h2>命中统计</h2>
<div class="cards">
  <div class="card"><div class="n">{pct(scene_hits)}</div><div class="l">scene 路径 top-1 命中</div></div>
  <div class="card"><div class="n">{pct(node_hits)}</div><div class="l">node 路径 top-1 命中（全部 100 query）</div></div>
  <div class="card"><div class="n">{pct(covered_hits)}</div><div class="l">node 路径 top-1（{len(covered)} 个有节点覆盖的物体）</div></div>
  <div class="card"><div class="n">{len(uncovered)}</div><div class="l">无节点覆盖的物体查询数</div></div>
</div>
<p class="note">scene 路径：query 与其改写前的场景问题（question_id）应互为最近邻；
node 路径：query 应命中含 “observe &lt;object&gt;” 的场景节点。新增的 8 个物体
（床/冰箱/微波炉/水槽/衣柜/书桌/窗户/电视）在 node_embed_texts.json 中无对应观测节点，
故 node 路径对这些 query 无真值。</p>

<h2>逐 query 结果</h2>
<table>
<caption>模型 {meta['embedder']}（{meta['dim']} 维）· 余弦相似度</caption>
<thead><tr><th>query</th><th>查询</th><th>物体</th>
<th>scene 路径 top-1<br><span class='thsub'>BGE-M3 · cos</span></th>
<th>node 路径 top-1<br><span class='thsub'>BGE-M3 · cos</span></th></tr></thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
<p class="note">全量相似度值见 <code>{os.path.basename(html_path.replace('_summary.html', '_result.json'))}</code>。</p>
</div>
</body>
</html>"""
    return html


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-path retrieval (BGE-M3).")
    ap.add_argument("--node-json", default="./memory/node_embed_texts.json")
    ap.add_argument("--scene-questions", default="./memory/scene_questions_100.json")
    ap.add_argument("--queries", default="./memory/query_100.json")
    ap.add_argument("--out-json", default="./memory/dual_path_result.json")
    ap.add_argument("--out-html", default="./memory/dual_path_summary.html")
    args = ap.parse_args(argv)

    nodes, node_order = _load_nodes(args.node_json)
    with open(args.scene_questions, encoding="utf-8") as f:
        questions = json.load(f)["scene_questions"]
    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    emb = _load_embedder()
    t0 = time.time()

    node_vecs = _embed(emb, [nodes[k]["text"] for k in node_order])
    node_vecs = {k: v for k, v in zip(node_order, node_vecs)}
    scene_ids = [q["id"] for q in questions]
    scene_vecs = _embed(emb, [q["question"] for q in questions])
    scene_vecs = {k: v for k, v in zip(scene_ids, scene_vecs)}
    query_vecs = _embed(emb, [q["query"] for q in queries])

    # object label → set of node keys whose embed text observed it
    object_nodes = {}
    for k, info in nodes.items():
        if info["kind"] != "scene":
            continue
        for q in questions:
            lbl = q["object"]
            if f"observe {lbl}" in info["text"]:
                object_nodes.setdefault(lbl, set()).add(k)

    results = []
    for q, qv in zip(queries, query_vecs):
        # node path (full)
        ns = {k: float(np.dot(qv, v)) for k, v in node_vecs.items()}
        ns_sorted = sorted(ns.items(), key=lambda x: -x[1])
        node_top3 = [{"key": k, "cos": round(s, 6)} for k, s in ns_sorted[:3]]
        # scene path (full)
        ss = {sid: float(np.dot(qv, v)) for sid, v in scene_vecs.items()}
        ss_sorted = sorted(ss.items(), key=lambda x: -x[1])
        scene_top3 = [{"id": sid, "cos": round(s, 6)} for sid, s in ss_sorted[:3]]

        truth = object_nodes.get(q["object"], set())
        scene_top1_id = ss_sorted[0][0]
        scene_top1_q = next(x for x in questions if x["id"] == scene_top1_id)
        node_top1_key = ns_sorted[0][0]
        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "question_id": q["question_id"],
            "object": q["object"],
            "object_zh": q["object_zh"],
            "type": q["type"],
            "object_covered": bool(truth),
            "node_scores": {k: round(s, 6) for k, s in ns_sorted},
            "scene_scores": {sid: round(s, 6) for sid, s in ss_sorted},
            "node_top1": {
                "key": node_top1_key,
                "kind": nodes[node_top1_key]["kind"],
                "text": nodes[node_top1_key]["text"],
                "cos": round(ns[node_top1_key], 6),
            },
            "node_top3": node_top3,
            "node_hit": _hit_at([k for k, _ in ns_sorted], truth, 1),
            "scene_top1": {
                "id": scene_top1_id,
                "text": scene_top1_q["question"],
                "cos": round(ss[scene_top1_id], 6),
            },
            "scene_top3": scene_top3,
            "scene_hit": scene_top1_id == q["question_id"],
        })

    result = {
        "meta": {
            "embedder": "BGE-M3 (gpahal/bge-m3-onnx-int8)",
            "dim": 1024,
            "metric": "cosine",
            "node_count": len(nodes),
            "scene_nodes": sum(1 for v in nodes.values() if v["kind"] == "scene"),
            "plan_nodes": sum(1 for v in nodes.values() if v["kind"] == "plan"),
            "scene_questions": len(questions),
            "queries": len(queries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(_render_html(result, args.out_html))

    scene_ok = sum(1 for r in results if r["scene_hit"])
    node_ok = sum(1 for r in results if r["node_hit"])
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_html}")
    print(f"scene path top-1: {scene_ok}/{len(results)} "
          f"({scene_ok / len(results) * 100:.1f}%)")
    print(f"node  path top-1: {node_ok}/{len(results)} "
          f"({node_ok / len(results) * 100:.1f}%)  "
          f"[{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
