"""Bag placement + feature retrieval with multi-object VLM replies (experiment).

Scenario: the robot remembers several bags with *different features* (colour,
type, size) and where each was placed.  To answer "where is each bag and what
does it look like?", retrieval pulls every bag node by its common label and
sends **all** their frames to the VLM in one call, so it replies about each bag
separately instead of running one query per bag.

The VLM boundary (``observe.vlm_answer_question``) is monkeypatched in the tests
to record its inputs; unpatched, the same orchestration hits the real VLM
(``python tests/test_bag_features.py --real``).
"""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core import observe
from memory_service.core.types import (
    ObjectCoord, SpatialContext, MemoryNode, NodeType, TagSet, TagFilter,
)
from memory_service.storage.tag_index import TagIndex


# ── Orchestration ────────────────────────────────────────────────────────

async def ask_bag_features(tag_index, nodes_by_id, bag_label, question):
    """Retrieve every bag node by *bag_label* → all frames + contexts → VLM.

    Returns the VLM's answer, or ``None`` when nothing matches.
    """
    hits = sorted(tag_index.query(TagFilter(objects=[bag_label])))
    image_paths, contexts = [], []
    for nid in hits:
        node = nodes_by_id[nid]
        image_paths.extend(node.image_refs)
        coord = node.spatial_data.objects[0]
        contexts.append(
            f"{coord.label} @ ({coord.x:.2f}, {coord.y:.2f}, {coord.z:.2f}) "
            f"in {node.tags.scene_type} :: {node.summary}"
        )
    if not contexts:
        return None
    return await observe.vlm_answer_question(question, image_paths, contexts)


def _bag(node_id, obj_id, label, x, y, z, scene_type, image_ref, summary,
         bag_label):
    """One remembered bag: stable ``obj_id``, display ``label``, and the common
    retrieval token ``bag_label`` indexed under ``objects_present``."""
    coord = ObjectCoord(obj_id=obj_id, label=label, x=x, y=y, z=z,
                        label_zh=label, label_en=label)
    return MemoryNode(
        node_id=node_id, timestamp=1760000000000000000,
        summary=summary, summary_zh=summary,
        spatial_data=SpatialContext(objects=[coord], origin="t"),
        tags=TagSet(scene_type=scene_type, objects_present=[bag_label]),
        image_refs=[image_ref], node_type=NodeType.OBJECT_OBSERVATION,
    )


def _index(*bags):
    """Build a TagIndex over *bags* (with stable obj_id keys)."""
    idx = TagIndex()
    nodes = {}
    for n in bags:
        nodes[n.node_id] = n
        idx.insert(n.node_id, n.tags,
                   obj_ids=[n.spatial_data.objects[0].obj_id])
    return idx, nodes


# ── Test (mock VLM) ──────────────────────────────────────────────────────

class TestBagFeatures:
    def setup_method(self):
        self._orig = observe.vlm_answer_question
        self.calls = []

        async def fake(query, image_paths, node_contexts=None, **_):
            self.calls.append({"query": query, "images": list(image_paths),
                               "contexts": list(node_contexts or [])})
            return "MOCK: bag-by-bag reply."

        observe.vlm_answer_question = fake

    def teardown_method(self):
        observe.vlm_answer_question = self._orig

    def _ask(self, bags, bag_label, question):
        idx, nodes = _index(*bags)
        return asyncio.run(ask_bag_features(idx, nodes, bag_label, question))

    def test_three_backpacks_by_colour(self):
        bags = [
            _bag(1, "bag.backpack.red_001", "red backpack", 1.0, 0.0, 0.5,
                 "living_room", "data/images/1/red.png",
                 "backpack: red canvas, on the sofa", "backpack"),
            _bag(2, "bag.backpack.black_002", "black backpack", 2.0, 0.0, 0.5,
                 "office", "data/images/2/black.png",
                 "backpack: black nylon, under the desk", "backpack"),
            _bag(3, "bag.backpack.blue_003", "blue backpack", 3.0, 0.0, 0.5,
                 "bedroom", "data/images/3/blue.png",
                 "backpack: blue leather, by the door", "backpack"),
        ]
        ans = self._ask(bags, "backpack", "分别描述每个背包的特征和位置")
        call = self.calls[0]
        assert ans == "MOCK: bag-by-bag reply."
        assert len(call["images"]) == 3          # all three sent together
        assert len(call["contexts"]) == 3
        assert all("backpack" in c for c in call["contexts"])

    def test_two_bags_by_type(self):
        bags = [
            _bag(4, "bag.backpack_004", "backpack", 0.0, 0.0, 0.3,
                 "living_room", "data/images/4/backpack.png",
                 "bag: a hiking backpack, on the floor", "bag"),
            _bag(5, "bag.handbag_005", "handbag", 1.0, 1.0, 0.3,
                 "living_room", "data/images/5/handbag.png",
                 "bag: a small handbag, on the table", "bag"),
        ]
        ans = self._ask(bags, "bag", "分别说出每个包是什么包、放在哪里")
        call = self.calls[0]
        assert ans == "MOCK: bag-by-bag reply."
        assert len(call["images"]) == 2
        assert len(call["contexts"]) == 2
        assert any("backpack" in c for c in call["contexts"])
        assert any("handbag" in c for c in call["contexts"])

    def test_three_bags_by_use(self):
        bags = [
            _bag(6, "bag.sports_006", "sports bag", 0.5, 0.0, 0.4,
                 "bedroom", "data/images/6/sports.png",
                 "bag: gym duffel, near the wardrobe", "bag"),
            _bag(7, "bag.laptop_007", "laptop bag", 1.5, 0.0, 0.4,
                 "office", "data/images/7/laptop.png",
                 "bag: slim laptop sleeve, on the desk", "bag"),
            _bag(8, "bag.travel_008", "travel bag", 2.5, 0.0, 0.4,
                 "living_room", "data/images/8/travel.png",
                 "bag: wheeled suitcase, by the entrance", "bag"),
        ]
        ans = self._ask(bags, "bag", "每个包分别是什么用途、放在哪个房间")
        call = self.calls[0]
        assert ans == "MOCK: bag-by-bag reply."
        assert len(call["images"]) == 3
        assert len(call["contexts"]) == 3


# ── Real VLM run (experiment) ────────────────────────────────────────────

async def _real_run():
    examples = [
        ("backpack", "分别描述每个背包的特征和位置", [
            _bag(1, "bag.backpack.red_001", "red backpack", 1.0, 0.0, 0.5,
                 "living_room", "data/images/1/red.png",
                 "backpack: red canvas, on the sofa", "backpack"),
            _bag(2, "bag.backpack.black_002", "black backpack", 2.0, 0.0, 0.5,
                 "office", "data/images/2/black.png",
                 "backpack: black nylon, under the desk", "backpack"),
            _bag(3, "bag.backpack.blue_003", "blue backpack", 3.0, 0.0, 0.5,
                 "bedroom", "data/images/3/blue.png",
                 "backpack: blue leather, by the door", "backpack"),
        ]),
        ("bag", "分别说出每个包是什么包、放在哪里", [
            _bag(4, "bag.backpack_004", "backpack", 0.0, 0.0, 0.3,
                 "living_room", "data/images/4/backpack.png",
                 "bag: a hiking backpack, on the floor", "bag"),
            _bag(5, "bag.handbag_005", "handbag", 1.0, 1.0, 0.3,
                 "living_room", "data/images/5/handbag.png",
                 "bag: a small handbag, on the table", "bag"),
        ]),
    ]
    for i, (bag_label, question, bags) in enumerate(examples, 1):
        idx, nodes = _index(*bags)
        answer = await ask_bag_features(idx, nodes, bag_label, question)
        print(f"\n[{i}] {question}")
        print(f"    命中 {len(nodes)} 个包节点")
        print(f"    VLM 回答 = {answer or '(VLM 未配置或调用失败)'}")


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--real" in sys.argv:
        asyncio.run(_real_run())
        sys.exit(0)

    import traceback
    obj = TestBagFeatures()
    passed = failed = 0
    for name in dir(obj):
        if name.startswith("test_"):
            try:
                obj.setup_method()
                getattr(obj, name)()
                obj.teardown_method()
                print(f"  PASS TestBagFeatures.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL TestBagFeatures.{name}")
                traceback.print_exc()
                failed += 1
                obj.teardown_method()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
