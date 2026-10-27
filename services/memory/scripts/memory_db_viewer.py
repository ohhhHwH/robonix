#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Standalone web viewer for the memory service's two stores.

Serves one HTML page (default http://127.0.0.1:8899/) that lists, live:

  - MongoDB  (authoritative):  scene_nodes / causal_edges / plan_tasks
  - Milvus   (derived index):  memory_vectors (doc_id / kind / node_type)

MongoDB is read directly and is always safe while the memory service runs.
Milvus (milvus-lite) is an embedded, single-process store guarded by an
exclusive flock — the running ``memory_service`` holds that lock, so a second
process cannot open it via ``MilvusClient`` (``DataDirLockedError``).  This
viewer works around that by reading milvus-lite's raw on-disk files (compacted
parquet segments + append-only arrow WAL) with pyarrow and reconstructing the
current rows without acquiring the lock — so it stays live while the service
runs.

Usage:

    cd services/memory
    rbnx-build/venv/bin/python scripts/memory_db_viewer.py \
        [--port 8899] [--host 127.0.0.1] \
        [--mongo-uri mongodb://127.0.0.1:27017/robonix_memory] \
        [--milvus-uri ./memory/robonix_memory.db]

Then open http://127.0.0.1:8899/ .  Only stdlib + pymongo + pyarrow are
required (no Flask / web framework); pymilvus is used only for remote
(non-lite) Milvus URIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    from bson import json_util  # serialises ObjectId / datetime / Int64
except Exception:  # pragma: no cover - bson ships with pymongo
    json_util = None

MILVUS_COLLECTION = "memory_vectors"
NODE_FIELDS = ["_id", "node_type", "summary", "summary_zh", "timestamp",
               "image_count", "weight", "causal_chain"]
PLAN_FIELDS = ["_id", "query", "description", "plan_count", "canceled_count",
               "timestamp_ns"]


# ── data readers ─────────────────────────────────────────────────────────

def _db_name(uri: str) -> str:
    path = uri.rsplit("/", 1)[-1]
    return path.split("?", 1)[0] or "robonix_memory"


def read_mongo(uri: str) -> dict:
    """Authoritative nodes / edges / plans.  Always readable while service runs."""
    from pymongo import MongoClient

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        db = client[_db_name(uri)]
        db.list_collection_names()  # force a connection / auth check
        nodes = list(db.scene_nodes.find(
            {}, {f: 1 for f in NODE_FIELDS}).sort("_id", 1))
        edges = list(db.causal_edges.find().sort("_id", 1))
        plans = list(db.plan_tasks.find(
            {}, {f: 1 for f in PLAN_FIELDS}).sort("timestamp_ns", -1))
        return {
            "available": True,
            "db": _db_name(uri),
            "counts": {
                "scene_nodes": len(nodes),
                "causal_edges": len(edges),
                "plan_tasks": len(plans),
            },
            "scene_nodes": nodes,
            "causal_edges": edges,
            "plan_tasks": plans,
        }
    except Exception as e:  # mongod down, etc.
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


def read_milvus(uri: str) -> dict:
    """Derived vector index, read live from milvus-lite's raw on-disk files.

    milvus-lite holds an exclusive flock while the memory_service runs, so a
    second process cannot open it via ``MilvusClient``.  For a local directory
    we therefore parse the raw storage (compacted parquet segments + append-only
    arrow WAL) with pyarrow and reconstruct the current
    ``doc_id → kind/node_type/timestamp`` rows — no lock, no MilvusClient, so it
    works while the service is up.
    """
    if "://" in uri:  # remote (non-lite) Milvus — go through the client
        return _read_milvus_client(uri)

    path = os.path.abspath(os.path.expanduser(uri))
    if not os.path.isdir(path):
        return {"available": False, "locked": False,
                "error": f"milvus db not found: {path} (never initialised)"}

    try:
        import pyarrow  # noqa: F401 — needed to read the raw files
    except ImportError:
        return {"available": False, "locked": False,
                "error": "pyarrow not installed (needed to read milvus-lite raw files)"}

    try:
        rows = _read_milvus_raw(path)
        return {"available": True, "locked": False,
                "count": len(rows), "rows": rows}
    except Exception as e:  # concurrent write may transiently truncate a read
        return {"available": False, "locked": False,
                "error": f"{type(e).__name__}: {e}"}


def _read_milvus_client(uri: str) -> dict:
    """Fallback reader for a remote Milvus endpoint (not milvus-lite)."""
    from pymilvus import MilvusClient

    try:
        client = MilvusClient(uri)
        try:
            if not client.has_collection(MILVUS_COLLECTION):
                return {"available": False, "locked": False,
                        "error": f"collection {MILVUS_COLLECTION} not found"}
            rows = client.query(
                collection_name=MILVUS_COLLECTION, filter="", limit=100000,
                output_fields=["doc_id", "kind", "node_type", "timestamp"])
            return {"available": True, "locked": False,
                    "count": len(rows), "rows": rows}
        finally:
            client.close()
    except Exception as e:
        return {"available": False, "locked": False,
                "error": f"{type(e).__name__}: {e}"}


def _read_arrow_ipc(fp: str):
    """Read one arrow IPC stream file, tolerating a torn tail from a live append.

    milvus-lite appends upserts/deletes to the WAL while the service runs; a read
    that lands mid-append sees a truncated record/footer and pyarrow raises.  Read
    the file's bytes (not a memory-map, which can SIGBUS if the file is truncated
    under us) and retry a few times — each append is microseconds, so the next
    snapshot parses clean.  Returns a pyarrow Table.
    """
    import time

    import pyarrow as pa
    import pyarrow.ipc as ipc

    last = None
    for _ in range(3):
        try:
            with open(fp, "rb") as f:
                data = f.read()
            return ipc.open_stream(pa.BufferReader(data)).read_all()
        except Exception as e:  # torn append / empty during rotation
            last = e
            time.sleep(0.05)
    raise last


def _read_milvus_raw(path: str) -> list:
    """Reconstruct current ``memory_vectors`` rows from milvus-lite raw files.

    Storage layout (``collections/memory_vectors/``): compacted parquet data
    segments under ``partitions/_default/data/`` plus an append-only arrow WAL
    (``wal/wal_data_*.arrow`` upserts, ``wal/wal_delta_*.arrow`` deletes).  Every
    record carries a ``_seq``; for each ``doc_id`` the latest ``_seq`` wins — an
    upsert sets the row, a delta marks it deleted.  Returns a best-effort
    snapshot (files may be mid-append), sorted by kind then doc_id.
    """
    import pyarrow.parquet as pq

    col_dir = os.path.join(path, "collections", MILVUS_COLLECTION)
    manifest = {}
    mpath = os.path.join(col_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)

    latest: dict = {}  # doc_id -> (seq, row or None)

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
        for s, d, k, nt, ts in zip(seqs, docs, kinds, nts, tss):
            note(int(s), str(d), {
                "doc_id": str(d), "kind": k, "node_type": nt,
                "timestamp": int(ts or 0),
            })

    part = manifest.get("partitions", {}).get("_default", {})
    part_dir = os.path.join(col_dir, "partitions", "_default")
    for rel in part.get("data_files", []):
        fp = os.path.join(part_dir, rel)
        if os.path.exists(fp):
            apply_table(pq.read_table(fp), is_delete=False)
    for rel in part.get("delta_files", []):
        fp = os.path.join(part_dir, rel)
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

    rows = [row for (_seq, row) in latest.values() if row is not None]
    rows.sort(key=lambda r: (r["kind"], r["doc_id"]))
    return rows


# ── HTTP handler ─────────────────────────────────────────────────────────

_PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory DB Viewer</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #0f1420; color: #e6e9ef; }
  header { padding: 14px 20px; background: #171d2b; border-bottom: 1px solid #2a3245; }
  header h1 { margin: 0 0 4px; font-size: 18px; }
  header .sub { color: #8b93a7; font-size: 12px; }
  .banner { margin: 12px 20px; padding: 10px 14px; border-radius: 8px; font-size: 13px; }
  .banner.warn { background: #3a2a12; border: 1px solid #7a5a1e; color: #f5c86a; }
  .banner.err  { background: #3a1216; border: 1px solid #7a1e28; color: #f58a94; }
  section { margin: 16px 20px; }
  section h2 { font-size: 15px; margin: 0 0 8px; color: #aeb6c8; }
  section h2 .cnt { color: #6fe3c1; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; background: #141a28; font-size: 12px; }
  th, td { border: 1px solid #232b3d; padding: 6px 8px; text-align: left; vertical-align: top; }
  th { background: #1b2234; color: #aeb6c8; position: sticky; top: 0; }
  td.mono { font-family: ui-monospace, monospace; font-size: 11px; }
  .scroll { max-height: 420px; overflow: auto; border: 1px solid #232b3d; border-radius: 8px; }
  .dim { color: #6b7280; }
</style></head><body>
<header>
  <h1>Memory DB Viewer</h1>
  <div class="sub">MongoDB（权威）＋ Milvus（派生向量索引 · 直读磁盘原始文件） · 每 3s 自动刷新</div>
</header>
<div id="banners"></div>
<section><h2>MongoDB · scene_nodes <span class="cnt" id="c-nodes"></span></h2>
  <div class="scroll"><table><thead><tr>
    <th>_id</th><th>node_type</th><th>summary</th><th>summary_zh</th>
    <th>imgs</th><th>weight</th><th>causal_chain</th>
  </tr></thead><tbody id="t-nodes"></tbody></table></div>
</section>
<section><h2>MongoDB · causal_edges <span class="cnt" id="c-edges"></span></h2>
  <div class="scroll"><table><thead><tr>
    <th>_id</th><th>from</th><th>to</th><th>relation</th>
  </tr></thead><tbody id="t-edges"></tbody></table></div>
</section>
<section><h2>MongoDB · plan_tasks <span class="cnt" id="c-plans"></span></h2>
  <div class="scroll"><table><thead><tr>
    <th>_id</th><th>query</th><th>plan_count</th><th>canceled</th>
  </tr></thead><tbody id="t-plans"></tbody></table></div>
</section>
<section><h2>Milvus · memory_vectors <span class="cnt" id="c-milvus"></span></h2>
  <div class="scroll"><table><thead><tr>
    <th>doc_id</th><th>kind</th><th>node_type</th><th>timestamp</th>
  </tr></thead><tbody id="t-milvus"></tbody></table></div>
</section>
<script>
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function banner(kind, text) {
  const d = document.getElementById('banners');
  d.innerHTML = text ? `<div class="banner ${kind}">${esc(text)}</div>` : '';
}
function fill(id, rows, mapper) {
  const tb = document.getElementById(id);
  tb.innerHTML = rows.map(mapper).join('');
}
function nodeRow(n) {
  return `<tr><td class="mono">${esc(n._id)}</td><td>${esc(n.node_type)}</td>` +
    `<td>${esc((n.summary||'').slice(0,60))}</td>` +
    `<td>${esc((n.summary_zh||'').slice(0,40))}</td>` +
    `<td>${esc(n.image_count ?? 0)}</td><td>${esc(n.weight ?? '')}</td>` +
    `<td class="mono">${esc(JSON.stringify(n.causal_chain ?? []))}</td></tr>`;
}
function edgeRow(e) {
  return `<tr><td class="mono">${esc(e._id)}</td><td>${esc(e.from)}</td>` +
    `<td>${esc(e.to)}</td><td>${esc(e.relation)}</td></tr>`;
}
function planRow(p) {
  return `<tr><td class="mono">${esc(String(p._id).slice(0,12))}</td>` +
    `<td>${esc((p.query||'').slice(0,80))}</td>` +
    `<td>${esc(p.plan_count ?? '')}</td><td>${esc(p.canceled_count ?? 0)}</td></tr>`;
}
function milvusRow(m) {
  return `<tr><td class="mono">${esc(m.doc_id)}</td><td>${esc(m.kind)}</td>` +
    `<td>${esc(m.node_type)}</td><td class="mono">${esc(m.timestamp ?? '')}</td></tr>`;
}
async function refresh() {
  try {
    const r = await fetch('/api/data');
    const d = await r.json();
    const b = [];
    if (d.mongo && !d.mongo.available) b.push(['err', 'MongoDB 不可用：' + (d.mongo.error || '')]);
    if (d.milvus && d.milvus.locked) b.push(['warn', d.milvus.hint || 'Milvus 被锁']);
    else if (d.milvus && d.milvus.error) b.push(['warn', 'Milvus：' + d.milvus.error]);
    banner(b.length ? b[0][0] : null, b.length ? b[0][1] : '');
    if (d.mongo && d.mongo.available) {
      document.getElementById('c-nodes').textContent = '· ' + d.mongo.counts.scene_nodes;
      document.getElementById('c-edges').textContent = '· ' + d.mongo.counts.causal_edges;
      document.getElementById('c-plans').textContent = '· ' + d.mongo.counts.plan_tasks;
      fill('t-nodes', d.mongo.scene_nodes, nodeRow);
      fill('t-edges', d.mongo.causal_edges, edgeRow);
      fill('t-plans', d.mongo.plan_tasks, planRow);
    }
    if (d.milvus && d.milvus.available) {
      document.getElementById('c-milvus').textContent = '· ' + d.milvus.count;
      fill('t-milvus', d.milvus.rows, milvusRow);
    } else {
      document.getElementById('c-milvus').textContent = d.milvus && d.milvus.locked ? '· 已锁' : '· —';
      document.getElementById('t-milvus').innerHTML = '';
    }
  } catch (e) {
    banner('err', '刷新失败：' + e);
  }
}
refresh();
setInterval(refresh, 3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MemoryDBViewer/1.0"

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/data":
            payload = {
                "mongo": read_mongo(self.server.mongo_uri),
                "milvus": read_milvus(self.server.milvus_uri),
            }
            dumps = json_util.dumps if json_util else json.dumps
            self._send(200, dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):  # keep console quiet
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve a web viewer for the memory DBs.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--mongo-uri",
                    default=os.environ.get("MEMGRAPH_MONGO_URI",
                                           "mongodb://127.0.0.1:27017/robonix_memory"))
    ap.add_argument("--milvus-uri",
                    default=os.environ.get("MEMGRAPH_MILVUS_URI",
                                           "./memory/robonix_memory.db"))
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.mongo_uri = args.mongo_uri
    server.milvus_uri = args.milvus_uri
    print(f"Memory DB viewer on http://{args.host}:{args.port}/")
    print(f"  mongo : {args.mongo_uri}")
    print(f"  milvus: {args.milvus_uri}")
    print("  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
