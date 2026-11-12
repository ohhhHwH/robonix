"""Fixed-inspection-point "history vs now" VLM comparison (Demo 1 experiment).

Scenario: the robot patrols a *fixed inspection point* every day and records the
same object twice — yesterday and today.  To answer an explicit "what is the
difference between history and now?" question, we pull the object's two
most-recent snapshots from :class:`ObjectTimeline` and send **both** frames to
the VLM in one call, so it compares them visually rather than subtracting
coordinates by hand.

The VLM boundary (``observe.vlm_answer_question``) is monkeypatched in the tests
to record its inputs; the exact same orchestration runs the real VLM when left
unpatched (``python tests/test_history_vs_now.py --real``).
"""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core import observe
from memory_service.core.types import (
    ObjectCoord, SpatialContext, MemoryNode, NodeType,
)
from memory_service.storage.object_timeline import ObjectTimeline


# ── Orchestration ────────────────────────────────────────────────────────

async def ask_history_vs_now(timeline, nodes_by_id, obj_id, question):
    """Send one object's two most-recent snapshots to the VLM together.

    Returns the VLM's answer, or ``None`` when fewer than two snapshots exist.
    """
    snapshots = timeline.history(obj_id, limit=2)  # most-recent-first
    if len(snapshots) < 2:
        return None
    image_paths, contexts = [], []
    for snap in snapshots:
        node = nodes_by_id[snap["node_id"]]
        image_paths.extend(node.image_refs)
        contexts.append(
            f"{snap['label']} @ ({snap['x']:.2f}, {snap['y']:.2f}, {snap['z']:.2f}) "
            f"ts={snap['timestamp']} :: {node.summary}"
        )
    return await observe.vlm_answer_question(question, image_paths, contexts)


def _node(node_id, obj_id, label, x, y, z, ts, image_ref, summary):
    coord = ObjectCoord(obj_id=obj_id, label=label, x=x, y=y, z=z,
                        label_zh=label, label_en=label)
    return MemoryNode(
        node_id=node_id, timestamp=ts, summary=summary, summary_zh=summary,
        spatial_data=SpatialContext(objects=[coord], origin="t"),
        image_refs=[image_ref], node_type=NodeType.OBJECT_OBSERVATION,
    )


def _two_day_pair(node_id, obj_id, label, yesterday, today, summaries):
    """Two observations of one object at a fixed point: (yesterday_node, today_node)."""
    (yx, yy, yz), (tx, ty, tz) = yesterday, today
    day1 = _node(node_id, obj_id, label, yx, yy, yz, 1759000000000000000,
                 f"data/images/{node_id}/day1.png", summaries[0])
    day2 = _node(node_id + 1, obj_id, label, tx, ty, tz, 1760000000000000000,
                 f"data/images/{node_id + 1}/day2.png", summaries[1])
    return day1, day2


# ── Test (mock VLM) ──────────────────────────────────────────────────────

class TestHistoryVsNow:
    def setup_method(self):
        self._orig = observe.vlm_answer_question
        self.calls = []

        async def fake(query, image_paths, node_contexts=None, **_):
            self.calls.append({"query": query, "images": list(image_paths),
                               "contexts": list(node_contexts or [])})
            return "MOCK: the object moved."

        observe.vlm_answer_question = fake

    def teardown_method(self):
        observe.vlm_answer_question = self._orig

    def _run(self, obj_id, label, yesterday, today, summaries, question):
        d1, d2 = _two_day_pair(10, obj_id, label, yesterday, today, summaries)
        tl = ObjectTimeline()
        for n in (d1, d2):
            coord = n.spatial_data.objects[0]
            tl.record(coord.obj_id, n.node_id, coord, n.timestamp)
        nodes = {n.node_id: n for n in (d1, d2)}
        answer = asyncio.run(ask_history_vs_now(tl, nodes, obj_id, question))
        return answer

    def test_globe_moved_since_yesterday(self):
        ans = self._run(
            "exhibit.globe_001", "globe",
            (3.20, 1.5, 0.8), (3.25, 1.5, 0.8),
            ("globe at the centre of the display stand", "globe slightly off centre"),
            "这个地球仪和昨天相比有什么不同？",
        )
        call = self.calls[0]
        assert ans == "MOCK: the object moved."
        assert len(call["images"]) == 2          # both time points sent together
        assert len(call["contexts"]) == 2
        assert "不同" in call["query"]

    def test_extinguisher_lowered_since_yesterday(self):
        ans = self._run(
            "safety.extinguisher_001", "fire extinguisher",
            (1.0, 0.0, 1.0), (1.0, 0.0, 0.9),
            ("extinguisher upright by the door", "extinguisher leaning, lower"),
            "灭火器和昨天相比有什么变化？",
        )
        call = self.calls[0]
        assert ans == "MOCK: the object moved."
        assert len(call["images"]) == 2
        assert "变化" in call["query"]

    def test_chair_shifted_since_yesterday(self):
        ans = self._run(
            "furniture.chair_003", "chair",
            (0.5, 0.5, 0.0), (0.7, 0.5, 0.0),
            ("chair tucked under the desk", "chair pulled out from the desk"),
            "这把椅子和昨天有什么区别？",
        )
        call = self.calls[0]
        assert ans == "MOCK: the object moved."
        assert len(call["images"]) == 2
        assert "区别" in call["query"]


# ── Real VLM run (experiment) ────────────────────────────────────────────

async def _real_run():
    examples = [
        ("exhibit.globe_001", "globe",
         (3.20, 1.5, 0.8), (3.25, 1.5, 0.8),
         ("globe at the centre of the display stand", "globe slightly off centre"),
         "这个地球仪和昨天相比有什么不同？"),
        ("safety.extinguisher_001", "fire extinguisher",
         (1.0, 0.0, 1.0), (1.0, 0.0, 0.9),
         ("extinguisher upright by the door", "extinguisher leaning, lower"),
         "灭火器和昨天相比有什么变化？"),
        ("furniture.chair_003", "chair",
         (0.5, 0.5, 0.0), (0.7, 0.5, 0.0),
         ("chair tucked under the desk", "chair pulled out from the desk"),
         "这把椅子和昨天有什么区别？"),
    ]
    for i, (obj_id, label, y, t, summaries, question) in enumerate(examples, 1):
        d1, d2 = _two_day_pair(100 + i * 2, obj_id, label, y, t, summaries)
        tl = ObjectTimeline()
        for n in (d1, d2):
            c = n.spatial_data.objects[0]
            tl.record(c.obj_id, n.node_id, c, n.timestamp)
        diff = tl.diff(obj_id)
        nodes = {n.node_id: n for n in (d1, d2)}
        answer = await ask_history_vs_now(tl, nodes, obj_id, question)
        print(f"\n[{i}] {question}")
        print(f"    坐标差 Δ = {diff['delta']}")
        print(f"    VLM 回答 = {answer or '(VLM 未配置或调用失败)'}")


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--real" in sys.argv:
        asyncio.run(_real_run())
        sys.exit(0)

    import traceback
    obj = TestHistoryVsNow()
    passed = failed = 0
    for name in dir(obj):
        if name.startswith("test_"):
            try:
                obj.setup_method()
                getattr(obj, name)()
                obj.teardown_method()
                print(f"  PASS TestHistoryVsNow.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL TestHistoryVsNow.{name}")
                traceback.print_exc()
                failed += 1
                obj.teardown_method()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
