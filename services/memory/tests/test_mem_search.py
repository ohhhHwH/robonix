"""Tests for FusedSearch (design §6) — multi-route recall + RRF re-rank.

The router is exercised against a real ``MemoryService`` for the node
(keyword/tag/causal) routes and a fake plan store for the plan route, so the
tests run without Milvus or an LLM key (those routes degrade to empty).
"""

import sys, os, tempfile, shutil, asyncio, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core.types import (
    LogRecord, ObjectCoord, SpatialContext, TagFilter,
)
from memory_service.core.fusion import FusedSearch
from memory_service.service import MemoryService


class FakePtdl:
    """Minimal plan store — the subset of PtdlStore FusedSearch touches."""

    def __init__(self):
        self._plans = {}

    @staticmethod
    def _key(query):
        return hashlib.sha1((query or "").strip().encode("utf-8")).hexdigest()

    def add(self, query, timestamp_ns=0, **kw):
        key = self._key(query)
        self._plans[key] = {"query": query, "timestamp_ns": timestamp_ns, **kw}

    def list_all(self):
        return list(self._plans.values())

    def search(self, query, top_k=5):
        return [p for p in self._plans.values()
                if query.lower() in p["query"].lower()][:top_k]

    def get_by_query_key(self, key):
        return self._plans.get(key)


class TestNodeRoutes:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = MemoryService(data_dir=self.tmp)
        asyncio.run(self.svc.init())
        asyncio.run(self.svc.remember_from_log(
            session_id="sess-1", plan_id="plan-1", level="Info", tag="exec",
            msg="robot grasped the red cup on the kitchen counter",
            objects=[("scene.obj.cup_001", "red cup", 1.0, 2.0, 0.8)],
            spatial_origin="t",
        ))
        asyncio.run(self.svc.remember_from_log(
            session_id="sess-1", plan_id="plan-1", level="Info", tag="exec",
            msg="robot placed the blue plate on the dining table",
            objects=[("scene.obj.plate_001", "blue plate", 0.0, 0.0, 0.0)],
            spatial_origin="t",
        ))
        self.search = FusedSearch(graph=self.svc.graph, tags=self.svc.tags,
                                  vectors=self.svc.vectors)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_keyword_route_ranks_matching_node_first(self):
        res = asyncio.run(self.search.search("red cup", top_k=3, domain="memory"))
        assert len(res["results"]) >= 1
        top = res["results"][0]
        assert top["kind"] == "memory"
        assert "red cup" in top["memory"]["summary"]
        assert top["route_scores"].get("keyword") is not None

    def test_tag_filter_narrows_candidates(self):
        res = asyncio.run(self.search.search(
            "cup", top_k=5, domain="memory",
            tags_filter=TagFilter(objects=["red cup"]),
        ))
        # only the cup node carries the "red cup" label
        assert len(res["results"]) >= 1
        for r in res["results"]:
            objs = r["memory"].get("tags", {}).get("objects_present", [])
            assert "red cup" in objs

    def test_tag_route_orders_by_jaccard(self):
        # "grasp in kitchen" overlaps the cup node's tags far more than the plate's
        res = asyncio.run(self.search.search(
            "grasp the cup in kitchen", top_k=5, domain="memory",
            routes=["tag"],
        ))
        assert len(res["results"]) >= 1
        assert "red cup" in res["results"][0]["memory"]["summary"]

    def test_scene_domain_excludes_short_term_nodes(self):
        # both seeded nodes are short_term → domain="scene" finds nothing
        res = asyncio.run(self.search.search("cup", top_k=5, domain="scene"))
        assert res["results"] == []

    def test_time_filter_excludes_nodes(self):
        far_future = (10 ** 18, 10 ** 18 + 1)  # ns range after every seeded node
        res = asyncio.run(self.search.search(
            "cup", top_k=5, domain="memory", time_range=far_future,
        ))
        assert res["results"] == []

    def test_milvus_absent_degrades_to_keyword(self):
        # vector route is empty (no milvus), keyword still yields results
        res = asyncio.run(self.search.search(
            "red cup", top_k=3, domain="memory", routes=["vector", "keyword"],
        ))
        assert len(res["results"]) >= 1


class TestPlanRoute:
    def test_plan_domain_resolves_plan_docs(self):
        ptdl = FakePtdl()
        ptdl.add("去厨房拿可乐", timestamp_ns=123)
        search = FusedSearch(vectors=None, ptdl=ptdl)
        res = asyncio.run(search.search("拿可乐", top_k=3, domain="plan"))
        assert len(res["results"]) == 1
        r = res["results"][0]
        assert r["kind"] == "plan"
        assert r["plan"]["query"] == "去厨房拿可乐"
        assert r["timestamp"] == 123

    def test_plan_domain_respects_time_filter(self):
        ptdl = FakePtdl()
        ptdl.add("去厨房拿可乐", timestamp_ns=100)
        ptdl.add("去客厅拿水杯", timestamp_ns=500)
        search = FusedSearch(vectors=None, ptdl=ptdl)
        res = asyncio.run(search.search("拿", top_k=5, domain="plan",
                                        time_range=(0, 200)))
        assert len(res["results"]) == 1
        assert res["results"][0]["plan"]["query"] == "去厨房拿可乐"
        assert res["results"][0]["timestamp"] == 100

    def test_memory_domain_excludes_plans(self):
        # domain discrimination: a plan store hit must not leak into a
        # node-only ("memory") search.
        ptdl = FakePtdl()
        ptdl.add("去厨房拿可乐", timestamp_ns=123)
        search = FusedSearch(vectors=None, ptdl=ptdl)
        res = asyncio.run(search.search("拿可乐", top_k=3, domain="memory"))
        assert res["results"] == []

    def test_plan_doc_id_is_stable_sha1(self):
        assert FakePtdl._key("去厨房") == FusedSearch._plan_doc_id("去厨房")


class TestSceneObjectRetrieval:
    """domain="scene" recall by object label and by stable obj_id (design §2)."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = MemoryService(data_dir=self.tmp)
        asyncio.run(self.svc.init())
        asyncio.run(self.svc.remember(
            session_id="sess-1", plan_id="plan-1",
            log_record=LogRecord(level="Info", tag="vlm",
                                 msg="observed globe on the desk"),
            spatial=SpatialContext(
                objects=[ObjectCoord(obj_id="obj.globe", label="globe",
                                     x=1.0, y=2.0, z=0.5)], origin="t"),
            kv={"node_type": "object_observation"},
        ))
        asyncio.run(self.svc.remember(
            session_id="sess-1", plan_id="plan-1",
            log_record=LogRecord(level="Info", tag="vlm",
                                 msg="observed cup on the table"),
            spatial=SpatialContext(
                objects=[ObjectCoord(obj_id="obj.cup", label="cup",
                                     x=0.0, y=0.0, z=0.0)], origin="t"),
            kv={"node_type": "object_observation"},
        ))
        self.search = FusedSearch(graph=self.svc.graph, tags=self.svc.tags,
                                  vectors=self.svc.vectors)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scene_domain_returns_object_observations(self):
        res = asyncio.run(self.search.search("globe", top_k=3, domain="scene"))
        assert len(res["results"]) >= 1
        assert all(r["kind"] == "scene" for r in res["results"])
        assert "globe" in res["results"][0]["scene"]["summary"]

    def test_scene_retrieval_by_obj_id_filter(self):
        # "obj.globe" matches the stable obj_id index, not the free-text label
        res = asyncio.run(self.search.search(
            "observe", top_k=5, domain="scene",
            tags_filter=TagFilter(objects=["obj.globe"]),
        ))
        assert len(res["results"]) == 1
        assert "globe" in res["results"][0]["scene"]["summary"]


class TestConstrainedCombination:
    """tags + time + domain + top_k applied together over the fused router."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = MemoryService(data_dir=self.tmp)
        asyncio.run(self.svc.init())
        # two same-object nodes at ts=100/500 and one other object at ts=100.
        asyncio.run(self.svc.remember(
            "sess-1", "plan-1",
            LogRecord(level="Info", tag="exec", ts=100,
                      msg="grasp the red cup in the kitchen"),
            spatial=SpatialContext(
                objects=[ObjectCoord(obj_id="obj.cup", label="red cup",
                                     x=1.0, y=2.0, z=0.0)], origin="t"),
        ))
        asyncio.run(self.svc.remember(
            "sess-1", "plan-1",
            LogRecord(level="Info", tag="exec", ts=100,
                      msg="grasp the blue plate in the kitchen"),
            spatial=SpatialContext(
                objects=[ObjectCoord(obj_id="obj.plate", label="blue plate",
                                     x=0.0, y=0.0, z=0.0)], origin="t"),
        ))
        asyncio.run(self.svc.remember(
            "sess-1", "plan-1",
            LogRecord(level="Info", tag="exec", ts=500,
                      msg="grasp the red cup in the kitchen"),
            spatial=SpatialContext(
                objects=[ObjectCoord(obj_id="obj.cup", label="red cup",
                                     x=1.0, y=2.0, z=0.0)], origin="t"),
        ))
        self.search = FusedSearch(graph=self.svc.graph, tags=self.svc.tags,
                                  vectors=self.svc.vectors)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_combined_tags_time_domain_topk(self):
        # tags(objects=["red cup"]) → 2 nodes; time(0,200) drops ts=500;
        # domain="memory" keeps short-term nodes → exactly the ts=100 cup node.
        res = asyncio.run(self.search.search(
            "grasp", top_k=5, domain="memory",
            tags_filter=TagFilter(objects=["red cup"]),
            time_range=(0, 200),
        ))
        assert len(res["results"]) == 1
        r = res["results"][0]
        assert r["kind"] == "memory"
        assert r["timestamp"] == 100
        assert "red cup" in r["memory"]["tags"]["objects_present"]

    def test_time_range_lower_bound(self):
        # only the ts=500 node falls inside (200, 1000)
        res = asyncio.run(self.search.search(
            "grasp", top_k=5, domain="memory", time_range=(200, 1000),
        ))
        assert len(res["results"]) == 1
        assert res["results"][0]["timestamp"] == 500

    def test_topk_truncates(self):
        res = asyncio.run(self.search.search("grasp", top_k=1, domain="memory"))
        assert len(res["results"]) == 1


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [TestNodeRoutes(), TestPlanRoute(), TestSceneObjectRetrieval(),
             TestConstrainedCombination()]
    passed = failed = 0
    for obj in tests:
        cls_name = type(obj).__name__
        for name in dir(obj):
            if name.startswith("test_"):
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
