#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Dual-path retrieval over synthetic mem/scene nodes (single BGE-M3 model).

One query is embedded once, then ranked over **two independent paths**, each
returning its own FULL cosine-similarity vector (never merged):

* **mem node path**   — query × 100 stored questions (``mem_node_100_1030.json``);
* **scene node path** — query × 100 object observations
  (``scene_node_100_1030.json``).

No real scene data is used. Both paths are self-verifiable: a query is a
paraphrase of its ``mem_node_id`` (so the mem path must rank it first), and it
names an object that the matching scene node records as ``observe <object>``.

Inputs (see ``generate_mem_scene_nodes.py``):
* ``memory/mem_node_100_1030.json``   — 100 mem nodes (questions);
* ``memory/scene_node_100_1030.json`` — 100 scene nodes (object observations);
* ``memory/query_1030.json``          — 100 test queries.

Outputs: JSON (full scores) + HTML (left sidebar = mem/scene embeddings; top =
retrieval-path diagram; main = per-query table with both paths + full lists).

Usage:
    cd services/memory
    HF_ENDPOINT=https://hf-mirror.com rbnx-build/venv/bin/python \
        scripts/mem_node_retrieval.py \
        [--mem-json ./memory/mem_node_100_1030.json] \
        [--scene-json ./memory/scene_node_100_1030.json] \
        [--queries ./memory/query_1030.json] \
        [--out-json ./memory/mem_node_result.json] \
        [--out-html ./memory/mem_node_summary.html]
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


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _render_html(result, html_path) -> str:
    meta = result["meta"]
    rows = result["results"]
    mem_hits = [r["mem_hit"] for r in rows]
    scene_hits = [r["scene_hit"] for r in rows]

    def pct(xs):
        return f"{sum(xs) / len(xs) * 100:.1f}%" if xs else "—"

    mem_lis = "".join(
        f"<li><span class='eid'>{_esc(m['id'])}</span> {_esc(m['question'])}</li>"
        for m in result["mem_nodes"]
    )
    scene_lis = "".join(
        f"<li><span class='eid'>{_esc(s['id'])}</span> {_esc(s['embed_text'])}</li>"
        for s in result["scene_nodes"]
    )

    trs = []
    for r in rows:
        mt = r["mem_top1"]
        st = r["scene_top1"]
        mok = "hit" if r["mem_hit"] else "miss"
        sok = "hit" if r["scene_hit"] else "miss"
        mem_cells = "".join(
            f"<span class='cell'>{_esc(mid)} <b>{s:.4f}</b></span>"
            for mid, s in r["mem_scores"].items()
        )
        scene_cells = "".join(
            f"<span class='cell'>{_esc(sid)} <b>{s:.4f}</b></span>"
            for sid, s in r["scene_scores"].items()
        )
        trs.append(
            f"<tr><td>{r['query_id']}</td><td>{_esc(r['query'])}</td>"
            f"<td>{_esc(r['object_zh'])}</td>"
            f"<td class='{mok}'>{mt['id']} <b>{mt['cos']:.4f}</b><br>"
            f"<span class='sub'>{_esc(mt['text'][:30])}</span></td>"
            f"<td class='{sok}'>{st['id']} <b>{st['cos']:.4f}</b><br>"
            f"<span class='sub'>{_esc(st['text'][:40])}</span></td></tr>"
            f"<tr class='detail'><td colspan='5'>"
            f"<details><summary>详细 · mem {len(r['mem_scores'])} / scene "
            f"{len(r['scene_scores'])} 全量降序</summary>"
            f"<div class='path'>mem node 路径</div><div class='grid'>{mem_cells}</div>"
            f"<div class='path'>scene node 路径</div>"
            f"<div class='grid'>{scene_cells}</div>"
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
.layout {{ display:flex; align-items:flex-start; }}
.sidebar {{ width:360px; flex:0 0 360px; position:sticky; top:0; height:100vh;
           overflow-y:auto; border-right:1px solid var(--line); padding:18px 16px;
           background:var(--panel); }}
.sidebar h3 {{ font-size:13px; color:var(--accent); margin:18px 0 8px; }}
.sidebar h3:first-child {{ margin-top:0; }}
.sidebar ul {{ list-style:none; margin:0; padding:0; }}
.sidebar li {{ font-size:12px; color:var(--dim); padding:2px 0;
              border-bottom:1px dashed var(--line); word-break:break-all; }}
.eid {{ color:var(--fg); font-weight:600; font-family:ui-monospace,Menlo,
       Consolas,monospace; margin-right:6px; }}
.main {{ flex:1; padding:24px 20px; min-width:0; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:22px 0 10px; color:var(--accent); }}
.sub {{ color:var(--dim); font-size:12px; }}
.paths {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center;
         margin:14px 0; }}
.pn {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
       padding:8px 12px; font-size:13px; }}
.pn b {{ color:var(--accent); }}
.arrow {{ color:var(--dim); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:12px; margin:16px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:14px; }}
.card .n {{ font-size:24px; font-weight:700; }}
.card .l {{ color:var(--dim); font-size:12px; margin-top:2px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border:1px solid var(--line); padding:6px 8px; text-align:left;
         vertical-align:top; }}
th {{ background:var(--panel); }}
caption {{ caption-side:top; text-align:left; color:var(--dim); font-size:12px;
          padding:4px 0 8px; }}
td.hit {{ color:var(--hit); }}
td.miss {{ color:var(--miss); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
         gap:1px 12px; font-size:12px; margin:6px 0; }}
.cell {{ font-family:ui-monospace,Menlo,Consolas,monospace; color:var(--dim); }}
.cell b {{ color:var(--fg); font-weight:600; }}
.path {{ color:var(--accent); font-size:12px; font-weight:700; margin-top:10px; }}
details {{ margin-top:2px; }}
details summary {{ cursor:pointer; color:var(--accent); font-size:12px; }}
tr.detail > td {{ background:var(--panel); }}
th .thsub {{ color:var(--dim); font-weight:400; font-size:11px; }}
@media (max-width:760px) {{ .layout {{ flex-direction:column; }}
  .sidebar {{ width:100%; flex:none; position:static; height:auto; }} }}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <h3>mem node（{meta['mem_nodes']}）embedding</h3>
  <ul>{mem_lis}</ul>
  <h3>scene node（{meta['scene_nodes']}）embedding</h3>
  <ul>{scene_lis}</ul>
</aside>
<main class="main">
<h1>双路径检索总结</h1>
<p class="sub">模型 {_esc(meta['embedder'])} · {meta['dim']} 维 · 余弦相似度 ·
生成于 {meta['generated_at'][:19]}</p>

<div class="paths">
  <div class="pn">query（{meta['queries']}）</div>
  <div class="arrow">─BGE-M3→</div>
  <div class="pn"><b>mem node 路径</b> × {meta['mem_nodes']}（问题）</div>
  <div class="pn"><b>scene node 路径</b> × {meta['scene_nodes']}（物体观测）</div>
</div>

<div class="cards">
  <div class="card"><div class="n">{pct(mem_hits)}</div><div class="l">mem node 路径 top-1 命中</div></div>
  <div class="card"><div class="n">{pct(scene_hits)}</div><div class="l">scene node 路径 top-1 命中</div></div>
  <div class="card"><div class="n">{meta['queries']}</div><div class="l">测试 query</div></div>
  <div class="card"><div class="n">{meta['mem_nodes']}+{meta['scene_nodes']}</div><div class="l">mem + scene 节点</div></div>
</div>

<h2>逐 query 结果（两条路径各自 top-1）</h2>
<table>
<caption>模型 {_esc(meta['embedder'])}（{meta['dim']} 维）· 余弦相似度 · 双路径不合并</caption>
<thead><tr><th>query</th><th>查询</th><th>物体</th>
<th>mem node 路径 top-1<br><span class='thsub'>BGE-M3 · cos</span></th>
<th>scene node 路径 top-1<br><span class='thsub'>BGE-M3 · cos</span></th></tr></thead>
<tbody>
{''.join(trs)}
</tbody>
</table>
<p class="sub">全量相似度值见 <code>{os.path.basename(html_path.replace('_summary.html', '_result.json'))}</code>。</p>
</main>
</div>
</body>
</html>"""
    return html


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-path mem/scene retrieval (BGE-M3).")
    ap.add_argument("--mem-json", default="./memory/mem_node_100_1030.json")
    ap.add_argument("--scene-json", default="./memory/scene_node_100_1030.json")
    ap.add_argument("--queries", default="./memory/query_1030.json")
    ap.add_argument("--out-json", default="./memory/mem_node_result.json")
    ap.add_argument("--out-html", default="./memory/mem_node_summary.html")
    args = ap.parse_args(argv)

    with open(args.mem_json, encoding="utf-8") as f:
        mem = json.load(f)["mem_nodes"]
    with open(args.scene_json, encoding="utf-8") as f:
        scene = json.load(f)["scene_nodes"]
    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    emb = _load_embedder()
    t0 = time.time()

    mem_ids = [m["id"] for m in mem]
    mem_vecs = dict(zip(mem_ids, _embed(emb, [m["question"] for m in mem])))
    scene_ids = [s["id"] for s in scene]
    scene_vecs = dict(zip(scene_ids, _embed(emb, [s["embed_text"] for s in scene])))
    query_vecs = _embed(emb, [q["query"] for q in queries])

    # object label -> scene node ids that observed it (scene-path ground truth)
    object_scene_ids = {}
    for s in scene:
        object_scene_ids.setdefault(s["object"], set()).add(s["id"])

    results = []
    for q, qv in zip(queries, query_vecs):
        mem_scores = {mid: float(np.dot(qv, v)) for mid, v in mem_vecs.items()}
        mem_ranked = sorted(mem_scores.items(), key=lambda x: -x[1])
        scene_scores = {sid: float(np.dot(qv, v)) for sid, v in scene_vecs.items()}
        scene_ranked = sorted(scene_scores.items(), key=lambda x: -x[1])

        mem_top_id = mem_ranked[0][0]
        scene_top_id = scene_ranked[0][0]
        mem_top = next(m for m in mem if m["id"] == mem_top_id)
        scene_top = next(s for s in scene if s["id"] == scene_top_id)
        truth = object_scene_ids.get(q["object"], set())

        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "mem_node_id": q["mem_node_id"],
            "object": q["object"],
            "object_zh": q["object_zh"],
            "type": q["type"],
            "mem_scores": {mid: round(s, 6) for mid, s in mem_ranked},
            "scene_scores": {sid: round(s, 6) for sid, s in scene_ranked},
            "mem_top1": {"id": mem_top_id, "text": mem_top["question"],
                         "cos": round(mem_scores[mem_top_id], 6)},
            "mem_hit": mem_top_id == q["mem_node_id"],
            "scene_top1": {"id": scene_top_id, "text": scene_top["embed_text"],
                           "cos": round(scene_scores[scene_top_id], 6)},
            "scene_hit": scene_top_id in truth,
        })

    result = {
        "meta": {
            "embedder": "BGE-M3 (gpahal/bge-m3-onnx-int8)",
            "dim": 1024,
            "metric": "cosine",
            "mem_nodes": len(mem),
            "scene_nodes": len(scene),
            "queries": len(queries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "mem_nodes": mem,
        "scene_nodes": scene,
        "results": results,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(_render_html(result, args.out_html))

    mem_ok = sum(1 for r in results if r["mem_hit"])
    scene_ok = sum(1 for r in results if r["scene_hit"])
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_html}")
    print(f"mem   node path top-1: {mem_ok}/{len(results)} "
          f"({mem_ok / len(results) * 100:.1f}%)")
    print(f"scene node path top-1: {scene_ok}/{len(results)} "
          f"({scene_ok / len(results) * 100:.1f}%)  [{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
