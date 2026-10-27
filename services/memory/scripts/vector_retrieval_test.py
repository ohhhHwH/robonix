#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Standalone vector-retrieval test over the memory service's stored vectors.

Embeds a fixed list of embodied-AI questions with the *same* BGE-M3 embedder the
memory service uses, reads the stored Milvus vectors straight from the raw
on-disk files (compacted parquet segments + arrow WAL — no milvus-lite lock, so
it works while the service is running; same trick as ``memory_db_viewer.py``),
and prints top-k cosine matches joined back to MongoDB for the human-readable
text.

Usage:
    cd services/memory
    HF_ENDPOINT=https://hf-mirror.com rbnx-build/venv/bin/python \
        scripts/vector_retrieval_test.py \
        [--milvus-uri ./memory/robonix_memory.db] \
        [--mongo-uri mongodb://127.0.0.1:27017/robonix_memory] \
        [--top-k 5] [--kind all|scene|plan]
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

# Embodied questions spanning the domains the memory actually stores: motion,
# door manipulation, grasping, navigation, perception, localisation, mapping.
QUESTIONS = [
    "向后移动两米",
    "开门并进入房间",
    "抓取桌上的红色杯子",
    "导航到厨房",
    "旋转扫描当前房间",
    "拍摄周围环境的图像",
    "记录机器人当前位姿",
    "观察房间里的布局",
]


# ── raw Milvus vector reader (no lock) ──────────────────────────────────

def _read_arrow_ipc(fp: str):
    """Read an arrow IPC file, tolerating a torn tail from a live WAL append."""
    import pyarrow.ipc as ipc

    last = None
    for _ in range(3):
        try:
            with open(fp, "rb") as f:
                return ipc.open_stream(f.read()).read_all()
        except Exception as e:  # torn append / empty during rotation
            last = e
            time.sleep(0.05)
    raise last


def _read_raw_vectors(path: str) -> dict:
    """Reconstruct current ``doc_id -> (kind, node_type, timestamp, vector)``.

    ``vector`` is the 1024-dim ``fixed_size_list<float>`` column; for each
    ``doc_id`` the latest ``_seq`` wins (upsert sets the row, delta deletes it).
    """
    import pyarrow.parquet as pq

    col_dir = os.path.join(path, "collections", MILVUS_COLLECTION)
    manifest = {}
    mpath = os.path.join(col_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)

    latest: dict = {}  # doc_id -> (seq, row | None)

    def note(seq: int, doc_id: str, row) -> None:
        cur = latest.get(doc_id)
        if cur is None or seq > cur[0]:
            latest[doc_id] = (seq, row)

    def vec_of(x) -> np.ndarray:
        return np.asarray(x, dtype=np.float32).reshape(-1)

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
            note(int(s), str(d), (k, nt, int(ts or 0), vec_of(v)))

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


# ── embedder (same one the service uses) ────────────────────────────────

class _HashEmbedder:
    """Local deterministic hash embedder (only if the package can't import)."""

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


# ── Mongo join (doc_id → readable text) ─────────────────────────────────

def _read_mongo_texts(uri: str) -> dict:
    """kind -> {doc_id_str: text} for rendering each hit."""
    from pymongo import MongoClient

    dbname = uri.rsplit("/", 1)[-1].split("?", 1)[0] or "robonix_memory"
    db = MongoClient(uri, serverSelectionTimeoutMS=2000)[dbname]

    scenes = {}
    for n in db.scene_nodes.find({}, {"summary_zh": 1, "summary": 1,
                                       "summary_en": 1}):
        text = (n.get("summary_zh") or n.get("summary")
                or n.get("summary_en") or "")
        scenes[str(n["_id"])] = text

    plans = {}
    for p in db.plan_tasks.find({}, {"query": 1, "description": 1}):
        text = " ".join(x for x in (p.get("query"), p.get("description")) if x)
        plans[str(p["_id"])] = text

    return {"scene": scenes, "plan": plans}


# ── main ────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Embodied-question vector retrieval test.")
    ap.add_argument("--milvus-uri",
                    default=os.environ.get("MEMGRAPH_MILVUS_URI",
                                           "./memory/robonix_memory.db"))
    ap.add_argument("--mongo-uri",
                    default=os.environ.get("MEMGRAPH_MONGO_URI",
                                           "mongodb://127.0.0.1:27017/robonix_memory"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--kind", choices=["all", "scene", "plan"], default="all")
    args = ap.parse_args(argv)

    path = os.path.abspath(os.path.expanduser(args.milvus_uri))
    if not os.path.isdir(path):
        print(f"milvus db not found: {path} (never initialised)")
        return 1

    vectors = _read_raw_vectors(path)
    n_scene = sum(1 for k in vectors.values() if k[0] == "scene")
    n_plan = sum(1 for k in vectors.values() if k[0] == "plan")
    print(f"# 原始文件读到 {len(vectors)} 条向量（scene={n_scene} plan={n_plan}）")

    emb, semantic = _load_embedder()
    dim = getattr(emb, "dim", 1024)
    print(f"# embedder: {'BGE-M3 语义' if semantic else 'hash 兜底'} (dim={dim})\n")

    texts = {}
    try:
        texts = _read_mongo_texts(args.mongo_uri)
    except Exception as e:
        print(f"# Mongo 不可用，只显示 doc_id（{type(e).__name__}: {e}）\n")

    for q in QUESTIONS:
        qv = np.asarray(emb.encode(q), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        scored = []
        for doc_id, (kind, nt, ts, vv) in vectors.items():
            if args.kind != "all" and kind != args.kind:
                continue
            v = vv / (np.linalg.norm(vv) + 1e-12)
            scored.append((float(np.dot(qv, v)), doc_id, kind, nt, ts))

        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"Q: {q}")
        for rank, (score, doc_id, kind, nt, ts) in enumerate(scored[: args.top_k], 1):
            text = texts.get(kind, {}).get(doc_id, "")
            print(f"  {rank}. cos={score:.4f} kind={kind} doc_id={doc_id} "
                  f"| {text[:60]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
