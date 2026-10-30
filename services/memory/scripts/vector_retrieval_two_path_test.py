#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Standalone two-path retrieval test suite (scene objects vs historical plans).

Verifies the design-doc "补充：双路径 embedding" behaviour: a single query is
embedded once with BGE-M3 (1024-dim, bilingual) and ranked over **two separate
paths** — ``kind="scene"`` (English object summaries) and ``kind="plan"``
(Chinese task descriptions) — each returning its **own** top-k, never merged.

Each test case is a standardised (question, answer) pair:

- scene-path queries use English object words ("table", "chair", "door" …)
  whose expected answer is the matching ``scene`` node(s);
- plan-path queries use Chinese task phrases ("去厨房拿一个水杯" …) whose
  expected answer is the matching ``plan`` record.

The suite reads the raw Milvus on-disk files (parquet + arrow WAL, no lock, so
it runs while the service holds the LOCK) and joins text from the JSON
snapshots ``memory/graph_store.json`` / ``memory/ptdl_store.json`` (Mongo is
optional). Requires the BGE-M3 model (set ``HF_ENDPOINT=https://hf-mirror.com``
for first download; cached afterwards).

Usage:
    cd services/memory
    HF_ENDPOINT=https://hf-mirror.com rbnx-build/venv/bin/python \
        scripts/vector_retrieval_two_path_test.py \
        [--milvus-dir ./memory/robonix_memory.db] \
        [--scene-json ./memory/graph_store.json] \
        [--plan-json ./memory/ptdl_store.json] \
        [--top-k 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MILVUS_COLLECTION = "memory_vectors"


# ── standardised test cases: (id, question, expected scene objects, expected plan queries) ──

TEST_CASES = [
    # scene path — English object queries → English object summaries
    ("T01", "table",           ["table"], []),
    ("T02", "chair",           ["chair"], []),
    ("T03", "door",            ["door"],  []),
    ("T04", "monitor",         ["monitor"], []),
    ("T05", "laptop",          ["laptop"], []),
    ("T06", "trash bin",       ["trash_bin"], []),
    ("T07", "potted plant",    ["potted_plant"], []),
    ("T08", "couch",           ["couch"], []),
    ("T09", "lamp",            ["lamp"], []),
    ("T10", "keyboard",        ["keyboard"], []),
    # plan path — Chinese task queries → Chinese task descriptions
    ("T11", "去厨房拿一个水杯",     [], ["去厨房拿一个水杯"]),
    ("T12", "整理办公室的书桌",     [], ["整理办公室西侧的书桌"]),
    ("T13", "检查门口是否有障碍物", [], ["检查门口是否有障碍物"]),
    ("T14", "巡逻客厅并记录物品",   [], ["巡逻客厅并记录看到的物品"]),
    ("T15", "我昨天做过什么任务",   [], ["我昨天做过什么任务"]),
    ("T16", "探索到办公室西侧的门", [], ["手动探索到办公室西侧实体门"]),
]


# ── raw Milvus vector reader (no lock) ──────────────────────────────────

def _read_arrow_ipc(fp: str):
    import pyarrow.ipc as ipc

    last = None
    for _ in range(3):
        try:
            with open(fp, "rb") as f:
                return ipc.open_stream(f.read()).read_all()
        except Exception as e:  # torn append during live WAL rotation
            last = e
            time.sleep(0.05)
    raise last


def _read_raw_vectors(path: str) -> dict:
    """doc_id -> (kind, node_type, timestamp, vector). Last ``_seq`` wins."""
    import pyarrow.parquet as pq

    col_dir = os.path.join(path, "collections", MILVUS_COLLECTION)
    manifest = {}
    mpath = os.path.join(col_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)

    latest: dict = {}

    def note(seq: int, doc_id: str, row) -> None:
        cur = latest.get(doc_id)
        if cur is None or seq > cur[0]:
            latest[doc_id] = (seq, row)

    def apply_table(tbl, *, is_delete: bool) -> None:
        seqs = tbl.column("_seq").to_pylist()
        docs = tbl.column("doc_id").to_pylist()
        if is_delete:
            for s, d in zip(seqs, docs):
                note(int(s), str(d), None)
            return
        kinds = tbl.column("kind").to_pylist()
        nts = tbl.column("node_type").to_pylist()
        tss = tbl.column("timestamp").to_pylist()
        vecs = tbl.column("vector").to_pylist()
        for s, d, k, nt, ts, v in zip(seqs, docs, kinds, nts, tss, vecs):
            note(int(s), str(d), (k, nt, int(ts or 0),
                                  np.asarray(v, dtype=np.float32).reshape(-1)))

    part = manifest.get("partitions", {}).get("_default", {})
    pdir = os.path.join(col_dir, "partitions", "_default")
    for rel in part.get("data_files", []):
        fp = os.path.join(pdir, rel)
        if os.path.exists(fp):
            apply_table(pq.read_table(fp), is_delete=False)
    for rel in part.get("delta_files", []):
        fp = os.path.join(pdir, rel)
        if os.path.exists(fp):
            apply_table(pq.read_table(fp), is_delete=True)

    wal_dir = os.path.join(col_dir, "wal")
    if os.path.isdir(wal_dir):
        for name in sorted(os.listdir(wal_dir)):
            if not name.endswith(".arrow"):
                continue
            is_delete = name.startswith("wal_delta_")
            if not (name.startswith("wal_data_") or is_delete):
                continue
            apply_table(_read_arrow_ipc(os.path.join(wal_dir, name)),
                        is_delete=is_delete)

    return {d: r for d, (_s, r) in latest.items() if r is not None}


# ── JSON text join (Mongo-free fallback) ────────────────────────────────

def _read_json_texts(scene_json: str, plan_json: str) -> dict:
    """kind -> {doc_id_str: (text, extra)} for rendering each hit."""
    scene = {}
    if os.path.exists(scene_json):
        with open(scene_json, encoding="utf-8") as f:
            nodes = json.load(f).get("nodes", [])
        for n in nodes:
            sid = str(n.get("node_id"))
            summary = n.get("summary_en") or n.get("summary") or ""
            objs = (n.get("tags") or {}).get("objects_present") or []
            scene[sid] = (summary, objs)

    plan = {}
    if os.path.exists(plan_json):
        with open(plan_json, encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            q = (e.get("query") or "").strip()
            if not q:
                continue
            did = hashlib.sha1(q.encode("utf-8")).hexdigest()
            plan[did] = (q, e.get("description") or "")

    return {"scene": scene, "plan": plan}


# ── embedder ────────────────────────────────────────────────────────────

class _HashEmbedder:
    dim = 1024

    def encode(self, text: str):
        v = np.zeros(self.dim, dtype=np.float32)
        h = hashlib.sha256(text.encode("utf-8")).digest()
        i = 0
        while i < self.dim:
            for b in hashlib.sha256(h + bytes([i & 0xFF])).digest():
                if i >= self.dim:
                    break
                v[i] = (b / 127.5) - 1.0
                i += 1
        return (v / (np.linalg.norm(v) + 1e-12)).tolist()


def _load_embedder():
    try:
        from memory_service.storage.bge_embedder import BgeEmbedder
        from memory_service.storage.embedding_config import EmbeddingModelConfig

        emb = BgeEmbedder(EmbeddingModelConfig())
        return emb, getattr(emb, "is_semantic", False)
    except Exception as e:
        print(f"# BGE-M3 未加载（{type(e).__name__}: {e}），用内置 hash 兜底")
        return _HashEmbedder(), False


# ── main ────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Two-path vector retrieval test suite.")
    ap.add_argument("--milvus-dir", default="./memory/robonix_memory.db")
    ap.add_argument("--scene-json", default="./memory/graph_store.json")
    ap.add_argument("--plan-json", default="./memory/ptdl_store.json")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    path = os.path.abspath(os.path.expanduser(args.milvus_dir))
    if not os.path.isdir(path):
        print(f"milvus db not found: {path}")
        return 1

    vectors = _read_raw_vectors(path)
    n_scene = sum(1 for k in vectors.values() if k[0] == "scene")
    n_plan = sum(1 for k in vectors.values() if k[0] == "plan")
    print(f"# raw vectors: {len(vectors)} (scene={n_scene} plan={n_plan})")

    emb, semantic = _load_embedder()
    print(f"# embedder: {'BGE-M3 semantic' if semantic else 'hash fallback'} "
          f"(dim={getattr(emb, 'dim', 1024)})\n")

    texts = _read_json_texts(args.scene_json, args.plan_json)

    passed = 0
    for tid, question, scene_expect, plan_expect in TEST_CASES:
        qv = np.asarray(emb.encode(question), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        scene_hits, plan_hits = [], []
        for doc_id, (kind, nt, ts, vv) in vectors.items():
            v = vv / (np.linalg.norm(vv) + 1e-12)
            s = float(np.dot(qv, v))
            if kind == "scene":
                scene_hits.append((s, doc_id))
            else:
                plan_hits.append((s, doc_id))
        scene_hits.sort(key=lambda x: x[0], reverse=True)
        plan_hits.sort(key=lambda x: x[0], reverse=True)

        print(f"[{tid}] Q: {question}")
        # scene path
        sc_ok = not scene_expect
        print("  scene path:")
        for rank, (s, did) in enumerate(scene_hits[: args.top_k], 1):
            summary, objs = texts["scene"].get(did, ("", []))
            hit = any(o in objs or o in summary for o in scene_expect)
            sc_ok = sc_ok or hit
            print(f"    {rank}. cos={s:.4f} doc={did} objs={objs} | {summary[:50]}")
        # plan path
        pl_ok = not plan_expect
        print("  plan path:")
        for rank, (s, did) in enumerate(plan_hits[: args.top_k], 1):
            q, desc = texts["plan"].get(did, ("", ""))
            hit = any(e in q for e in plan_expect)
            pl_ok = pl_ok or hit
            print(f"    {rank}. cos={s:.4f} doc={did[:12]}… | {q[:46]}")
        verdict = "PASS" if (sc_ok and pl_ok) else "FAIL"
        passed += 1 if (sc_ok and pl_ok) else 0
        print(f"  => scene {'✓' if sc_ok else '✗'} | plan {'✓' if pl_ok else '✗'} "
              f"| {verdict}\n")

    print(f"=== {passed}/{len(TEST_CASES)} passed ===")
    return 0 if passed == len(TEST_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
