"""End-to-end tests for the real Mongo / Milvus / BGE-M3 backends.

These are the "有依赖环境" tests the Phase-0 TESTS.md flagged as 待补.  They
exercise the real storage paths the graceful-degradation smoke cannot reach:

  - MongoBackend   — connect, node/edge/plan persistence, image Binary round-trip
  - MilvusBackend  — collection create, upsert, ANN search, scalar filters
  - BgeEmbedder     — BGE-M3 (or hash fallback) encode → 1024-dim
  - GraphStore/PtdlStore/ImageStore wired to all three

Preconditions (run once, outside this file):

    pip install pymongo "memsearch[onnx]"
    docker run -d --name mongo-e2e -p 27017:27017 mongo:7

When any backend is unavailable the run SKIPs (exit 0) rather than failing, so
the file stays safe to run in a bare environment.

Run:

    cd services/memory
    ../rbnx-build/venv/bin/python tests/test_vector_backends.py
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core.types import MemoryNode, NodeType  # noqa: E402
from memory_service.storage.mongo_backend import MongoBackend  # noqa: E402
from memory_service.storage.milvus_backend import MilvusBackend  # noqa: E402
from memory_service.storage.bge_embedder import BgeEmbedder  # noqa: E402
from memory_service.storage.embedding_config import EmbeddingModelConfig  # noqa: E402
from memory_service.storage.graph_store import GraphStore  # noqa: E402
from memory_service.storage.image_store import ImageStore  # noqa: E402
from memory_service.storage.ptdl_store import PtdlStore  # noqa: E402

# A dedicated DB so the e2e run never touches the live `robonix_memory` data.
TEST_DB = "robonix_memory_e2e"

# Tiny stand-in for a JPEG frame — only byte-for-byte round-trip matters here.
_FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 128 + b"\xff\xd9"


def _clear_all(mongo, milvus):
    """Reset all three collections so each test starts clean."""
    if mongo.available:
        mongo.clear_scene()
        mongo.clear_edges()
        mongo.clear_plans()
    if milvus.available:
        milvus.clear_kind("scene")
        milvus.clear_kind("plan")


class TestBackendAvailability:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.mongo = MongoBackend(db_name=TEST_DB)
        self.milvus = MilvusBackend(uri=os.path.join(self.tmp, "e2e.db"))
        self.bge = BgeEmbedder(EmbeddingModelConfig())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mongo_available(self):
        assert self.mongo.available, "mongod not reachable at 127.0.0.1:27017"

    def test_milvus_available(self):
        assert self.milvus.available, "milvus-lite failed to open"

    def test_bge_dimension_1024(self):
        # Real model or hash fallback, the contract is a 1024-dim vector.
        assert self.bge.dim == 1024
        vec = self.bge.encode("pick up the red cup")
        assert len(vec) == 1024
        assert any(v != 0.0 for v in vec)


class TestMongoImageRoundtrip:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.mongo = MongoBackend(db_name=TEST_DB)
        self.milvus = MilvusBackend(uri=os.path.join(self.tmp, "e2e.db"))
        _clear_all(self.mongo, self.milvus)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_image_binary_roundtrip(self):
        if not self.mongo.available:
            return
        node = MemoryNode(summary="img node")
        self.mongo.upsert_node(node)
        nid = node.node_id

        n = self.mongo.push_image(nid, _FAKE_JPEG)
        assert n == 1
        assert self.mongo.image_count(nid) == 1

        imgs = self.mongo.get_images(nid)
        assert len(imgs) == 1
        assert imgs[0] == _FAKE_JPEG

        # The image field is excluded from the default node projection.
        lean = self.mongo.get_node(nid)
        assert lean is not None
        assert lean.image_refs == ["mongo:0:0"]


class TestGraphMilvusSync:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.mongo = MongoBackend(db_name=TEST_DB)
        self.milvus = MilvusBackend(uri=os.path.join(self.tmp, "e2e.db"))
        self.bge = BgeEmbedder(EmbeddingModelConfig())
        _clear_all(self.mongo, self.milvus)
        self.graph = GraphStore(
            data_dir=self.tmp, mongo=self.mongo, milvus=self.milvus, embedder=self.bge
        )

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_node_persists_mongo_and_milvus(self):
        if not (self.mongo.available and self.milvus.available):
            return
        nid = self.graph.add_node(MemoryNode(
            summary="pick up the red cup in the kitchen",
            summary_zh="在厨房拿起红色杯子",
            summary_en="pick up the red cup in the kitchen",
            node_type=NodeType.SHORT_TERM,
        ))
        assert self.mongo.count_scene() == 1
        got = self.mongo.get_node(nid)
        assert got is not None and got.summary.startswith("pick up")

        # ANN search (forces a flush) should surface the fresh node.
        hits = self.milvus.search(self.bge.encode("red cup kitchen"), top_k=5)
        assert any(h["doc_id"] == str(nid) for h in hits), hits

    def test_promote_stable_id_updates_node_type_scalar(self):
        if not (self.mongo.available and self.milvus.available):
            return
        nid = self.graph.add_node(MemoryNode(
            summary="short-lived observation", node_type=NodeType.SHORT_TERM,
        ))
        out = self.graph.promote_to_long_term(nid)
        assert out == nid  # stable _id — no renumbering
        assert self.graph.get_node(nid).node_type == NodeType.LONG_TERM

        # The node_type scalar filter must now see it under long_term.
        hits = self.milvus.search(
            self.bge.encode("observation"),
            top_k=5, kind="scene", node_type="long_term",
        )
        assert any(h["doc_id"] == str(nid) for h in hits), hits

    def test_rebuild_milvus_restores_vectors(self):
        if not (self.mongo.available and self.milvus.available):
            return
        nid = self.graph.add_node(MemoryNode(
            summary="rebuild me later",
            summary_zh="稍后重建", summary_en="rebuild me later",
        ))
        self.milvus.clear_kind("scene")
        # Fresh/empty Milvus must catch up from Mongo via re-embed.
        n_rebuilt = self.graph.rebuild_milvus()
        assert n_rebuilt >= 1
        hits = self.milvus.search(self.bge.encode("rebuild"), top_k=5)
        assert any(h["doc_id"] == str(nid) for h in hits), hits


class TestPtdlMilvusSync:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.mongo = MongoBackend(db_name=TEST_DB)
        self.milvus = MilvusBackend(uri=os.path.join(self.tmp, "e2e.db"))
        self.bge = BgeEmbedder(EmbeddingModelConfig())
        _clear_all(self.mongo, self.milvus)
        self.ptdl = PtdlStore(
            path=os.path.join(self.tmp, "ptdl_store.json"),
            mongo=self.mongo, milvus=self.milvus, embedder=self.bge,
        )

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_persists_mongo_and_milvus(self):
        if not (self.mongo.available and self.milvus.available):
            return
        self.ptdl.add(
            query="去厨房拿可乐",
            description="navigate then grasp",
            steps=["1. navigate to kitchen", "2. grasp cola"],
            plan_id="1",
        )
        assert self.mongo.count_plans() == 1
        key = self.ptdl._doc_id("去厨房拿可乐")
        assert self.ptdl.get_by_query_key(key) is not None

        hits = self.milvus.search(
            self.bge.encode("去厨房拿可乐"), top_k=5, kind="plan",
        )
        assert any(h["doc_id"] == key for h in hits), hits


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    # Report what the environment actually provides before running anything.
    probe_mongo = MongoBackend(db_name=TEST_DB)
    probe_milvus = MilvusBackend(uri=os.path.join(tempfile.mkdtemp(), "probe.db"))
    probe_bge = BgeEmbedder(EmbeddingModelConfig())
    print("backends: mongo=%s milvus=%s bge_semantic=%s (dim=%d)" % (
        probe_mongo.available,
        probe_milvus.available,
        probe_bge.is_semantic,
        probe_bge.dim,
    ))
    if not (probe_mongo.available and probe_milvus.available):
        print("SKIP: Mongo/Milvus unavailable — run mongod and install deps first")
        sys.exit(0)

    classes = [
        TestBackendAvailability(),
        TestMongoImageRoundtrip(),
        TestGraphMilvusSync(),
        TestPtdlMilvusSync(),
    ]
    passed = failed = skipped = 0
    for obj in classes:
        cls_name = type(obj).__name__
        for name in dir(obj):
            if not name.startswith("test_"):
                continue
            try:
                if hasattr(obj, "setup_method"):
                    obj.setup_method()
                getattr(obj, name)()
                if hasattr(obj, "teardown_method"):
                    obj.teardown_method()
                print(f"  PASS {cls_name}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL {cls_name}.{name}")
                traceback.print_exc()
                failed += 1
                if hasattr(obj, "teardown_method"):
                    obj.teardown_method()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
