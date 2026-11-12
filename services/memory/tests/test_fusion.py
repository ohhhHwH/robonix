"""Tests for core.fusion — RRF + weighted linear fusion + router dispatch."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core.fusion import (
    reciprocal_rank_fusion, rank_from_scores, weighted_linear_fusion, fuse,
    RRF_K,
)


class TestReciprocalRankFusion:
    def test_basic_rrf_aggregates_ranks(self):
        rankings = {"vector": ["a", "b"], "keyword": ["b", "a"]}
        fused = dict(reciprocal_rank_fusion(rankings))
        # a: 1/(60+1) + 1/(60+2); b: 1/(60+2) + 1/(60+1) → equal
        assert abs(fused["a"] - fused["b"]) < 1e-9
        assert set(fused) == {"a", "b"}

    def test_higher_rank_means_higher_score(self):
        fused = dict(reciprocal_rank_fusion({"r": ["top", "mid", "low"]}))
        assert fused["top"] > fused["mid"] > fused["low"]

    def test_doc_absent_from_route_contributes_nothing(self):
        rankings = {"a": ["x"], "b": ["y", "x"]}
        fused = dict(reciprocal_rank_fusion(rankings))
        # x appears in both routes, y only in b → x outranks y
        assert fused["x"] > fused["y"]

    def test_weights_scale_route_votes(self):
        rankings = {"a": ["x"], "b": ["x", "y"]}
        base = dict(reciprocal_rank_fusion(rankings))["x"]
        zero_b = dict(reciprocal_rank_fusion(rankings, weights={"b": 0.0}))["x"]
        # only route a contributes when b's weight is zero
        assert abs(zero_b - (1.0 / (RRF_K + 1))) < 1e-9
        assert zero_b < base

    def test_route_missing_weight_defaults_to_one(self):
        rankings = {"a": ["x"], "b": ["x"]}
        fused = dict(reciprocal_rank_fusion(rankings, weights={"a": 2.0}))
        # a: 2/61, b: 1/61 → total 3/61
        assert abs(fused["x"] - (3.0 / (RRF_K + 1))) < 1e-9

    def test_empty_rankings(self):
        assert reciprocal_rank_fusion({}) == []

    def test_sorted_descending(self):
        rankings = {"a": ["x", "y"], "b": ["x", "z"]}
        fused = reciprocal_rank_fusion(rankings)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)


class TestLinearFusion:
    def test_minmax_normalize_and_sum(self):
        score_maps = {"a": {"x": 1.0, "y": 0.0}, "b": {"x": 5.0, "y": 10.0}}
        fused = dict(weighted_linear_fusion(score_maps))
        # a: x=1, y=0; b: x=0, y=1 → x=1.0, y=1.0 (tie)
        assert abs(fused["x"] - fused["y"]) < 1e-9

    def test_single_element_route_normalizes_to_zero(self):
        fused = dict(weighted_linear_fusion({"a": {"only": 42.0}}))
        assert fused["only"] == 0.0

    def test_empty_score_maps(self):
        assert weighted_linear_fusion({}) == []


class TestFuseDispatch:
    def test_rrf_default(self):
        out = fuse(rankings={"r": ["x", "y"]})
        assert out[0][0] == "x"

    def test_linear_dispatch(self):
        out = fuse(score_maps={"r": {"x": 1.0, "y": 0.0}}, fusion="linear")
        assert out[0][0] == "x"


class TestRankFromScores:
    def test_strips_scores(self):
        assert rank_from_scores([("a", 0.9), ("b", 0.5)]) == ["a", "b"]


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [TestReciprocalRankFusion(), TestLinearFusion(),
             TestFuseDispatch(), TestRankFromScores()]
    passed = failed = 0
    for obj in tests:
        cls_name = type(obj).__name__
        for name in dir(obj):
            if name.startswith("test_"):
                try:
                    getattr(obj, name)()
                    print(f"  PASS {cls_name}.{name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL {cls_name}.{name}")
                    traceback.print_exc()
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
