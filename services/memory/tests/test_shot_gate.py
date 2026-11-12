"""Tests for the 10-minute photographic-memory time gate (design §5).

The gate lives in two places — the scene hook's dedup (``_find_existing_node``
/ ``append_image_ref``) and the VLM observer's per-segment dedup.  Both share
the same predicate: a re-observation of the same object+position within
``SHOT_MIN_INTERVAL_NS`` of the previous shot is the same shot and is skipped;
past that window the node is reused and a new frame is appended.

We test the observer side here — it is the unit with a clean seam (``_call_vlm``
and ``_post`` are monkeypatchable) and exercises the exact ``now - last_shot_ns
<= SHOT_MIN_INTERVAL_NS`` decision that the hook handler mirrors.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory_service.vlm_observer as vlm_observer
from memory_service.constants import SHOT_MIN_INTERVAL_NS, SHOT_MIN_INTERVAL_S
from memory_service.vlm_observer import VLMObserver


class TestShotGate:
    def setup_method(self):
        self.obs = VLMObserver("key", "http://x", image_store=None,
                               memgraph_url="http://x")
        # Stub the VLM + HTTP boundary; record what the observer POSTs.
        self._orig_call = vlm_observer._call_vlm
        vlm_observer._call_vlm = lambda *a, **k: [
            {"label": "cup", "confidence": 0.9},
        ]
        self.posts = []
        self.obs._post = lambda payload: (self.posts.append(payload), 100)[1]

    def teardown_method(self):
        vlm_observer._call_vlm = self._orig_call

    def _wp(self):
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    def test_constant_defaults_to_ten_minutes(self):
        assert SHOT_MIN_INTERVAL_S == 600.0
        assert SHOT_MIN_INTERVAL_NS == 600 * 1_000_000_000

    def test_same_shot_within_gate_is_skipped(self):
        wp = self._wp()
        self.obs._process(1, wp, "img")
        assert self.obs._nodes_created == 1

        # Immediate re-observation of the same label+segment → same shot.
        self.obs._process(1, wp, "img")
        assert self.obs._nodes_skipped == 1
        assert self.obs._nodes_created == 1   # unchanged — no new node
        assert self.obs._nodes_updated == 0   # and no image appended

    def test_past_gate_reuses_node_and_appends(self):
        wp = self._wp()
        self.obs._process(1, wp, "img")
        assert self.obs._nodes_created == 1

        # Rewind the recorded shot time past the gate.
        nid, _ = self.obs._seg_dedup[(1, "cup")]
        self.obs._seg_dedup[(1, "cup")] = (
            nid, time.time_ns() - SHOT_MIN_INTERVAL_NS - 1_000_000_000,
        )

        self.obs._process(1, wp, "img")
        assert self.obs._nodes_updated == 1   # reused + image appended
        assert self.obs._nodes_skipped == 0
        assert self.obs._nodes_created == 1   # still only one node

        # the append carried the parent node id of the existing node
        assert self.posts[-1]["parent_node_id"] == nid
        assert self.posts[-1]["kv"]["append_image_ref"] is True


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [TestShotGate()]
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
