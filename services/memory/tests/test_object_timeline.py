"""Tests for ObjectTimeline — per-object cross-time index (design §2)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core.types import ObjectCoord, SpatialContext, MemoryNode
from memory_service.storage.object_timeline import ObjectTimeline


def _coord(x, y, z):
    return ObjectCoord(obj_id="obj.globe", label="globe", x=x, y=y, z=z)


class TestRecordHistory:
    def setup_method(self):
        self.tl = ObjectTimeline()

    def test_history_most_recent_first(self):
        self.tl.record("obj.globe", 1, _coord(1.0, 2.0, 0.5), 100)
        self.tl.record("obj.globe", 2, _coord(1.5, 2.0, 0.5), 200)
        h = self.tl.history("obj.globe")
        assert [e["node_id"] for e in h] == [2, 1]
        assert [e["timestamp"] for e in h] == [200, 100]

    def test_history_limit(self):
        self.tl.record("obj.globe", 1, _coord(1.0, 2.0, 0.5), 100)
        self.tl.record("obj.globe", 2, _coord(1.5, 2.0, 0.5), 200)
        h = self.tl.history("obj.globe", limit=1)
        assert len(h) == 1
        assert h[0]["node_id"] == 2

    def test_record_sorts_ascending_despite_insert_order(self):
        self.tl.record("obj.globe", 2, _coord(1.5, 2.0, 0.5), 200)
        self.tl.record("obj.globe", 1, _coord(1.0, 2.0, 0.5), 100)
        h = self.tl.history("obj.globe")
        assert [e["node_id"] for e in h] == [2, 1]

    def test_history_unknown_object(self):
        assert self.tl.history("obj.none") == []

    def test_record_ignores_empty_obj_id(self):
        self.tl.record("", 1, _coord(1.0, 2.0, 0.5), 100)
        assert self.tl.count() == 0


class TestDiff:
    def setup_method(self):
        self.tl = ObjectTimeline()

    def test_diff_computes_delta(self):
        self.tl.record("obj.globe", 1, _coord(1.0, 2.0, 0.5), 100)
        self.tl.record("obj.globe", 2, _coord(1.5, 2.0, 0.5), 200)
        d = self.tl.diff("obj.globe")
        assert d["delta"] == {"dx": 0.5, "dy": 0.0, "dz": 0.0, "distance": 0.5}
        assert d["latest"]["node_id"] == 2
        assert d["previous"]["node_id"] == 1

    def test_diff_distance_is_euclidean(self):
        self.tl.record("obj.globe", 1, _coord(0.0, 0.0, 0.0), 100)
        self.tl.record("obj.globe", 2, _coord(3.0, 4.0, 0.0), 200)
        d = self.tl.diff("obj.globe")
        assert d["delta"]["distance"] == 5.0

    def test_diff_single_snapshot_returns_no_delta(self):
        self.tl.record("obj.globe", 1, _coord(1.0, 2.0, 0.5), 100)
        d = self.tl.diff("obj.globe")
        assert d["delta"] is None
        assert d["previous"] is None
        assert d["latest"]["node_id"] == 1

    def test_diff_empty_object(self):
        d = self.tl.diff("obj.none")
        assert d["delta"] is None and d["latest"] is None


class TestLookups:
    def setup_method(self):
        self.tl = ObjectTimeline()

    def test_nodes_for_obj_id(self):
        self.tl.record("obj.globe", 1, _coord(1.0, 2.0, 0.5), 100)
        self.tl.record("obj.globe", 2, _coord(1.5, 2.0, 0.5), 200)
        assert self.tl.nodes_for_obj_id("obj.globe") == {1, 2}

    def test_obj_ids_and_count(self):
        self.tl.record("obj.a", 1, _coord(0, 0, 0), 100)
        self.tl.record("obj.b", 2, _coord(0, 0, 0), 100)
        assert set(self.tl.obj_ids()) == {"obj.a", "obj.b"}
        assert self.tl.count() == 2


class TestRebuild:
    def test_rebuild_from_nodes(self):
        n1 = MemoryNode(node_id=1, timestamp=100,
                        spatial_data=SpatialContext(objects=[_coord(1.0, 2.0, 0.5)],
                                                    origin="t"))
        n2 = MemoryNode(node_id=2, timestamp=200,
                        spatial_data=SpatialContext(objects=[_coord(1.5, 2.0, 0.5)],
                                                    origin="t"))
        n3 = MemoryNode(node_id=3, timestamp=300, spatial_data=None)  # skipped

        tl = ObjectTimeline()
        tl.rebuild([n1, n2, n3])
        assert tl.count() == 1
        h = tl.history("obj.globe")
        assert [e["node_id"] for e in h] == [2, 1]
        assert tl.diff("obj.globe")["delta"]["dx"] == 0.5


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [TestRecordHistory(), TestDiff(), TestLookups(), TestRebuild()]
    passed = failed = 0
    for obj in tests:
        cls_name = type(obj).__name__
        for name in dir(obj):
            if name.startswith("test_"):
                try:
                    if hasattr(obj, "setup_method"):
                        obj.setup_method()
                    getattr(obj, name)()
                    print(f"  PASS {cls_name}.{name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL {cls_name}.{name}")
                    traceback.print_exc()
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
