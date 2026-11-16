"""Retrieval similarity + top-k disambiguation (select_ambiguous + LLM judgment).

The mechanism: embed a question, cosine-rank it against the scene-memory nodes,
take top-k, keep the candidates whose similarity is "not much different" from the
top-1 (within ``gap``), and send that ambiguous set together to the LLM in one
text-only call so it judges/answers.

This module tests the *pure* selection logic and the LLM orchestration with the
VLM/LLM boundary monkeypatched.  The real BGE-M3 retrieval + real LLM run is
rendered by ``memory/experiment_retrieval_html.py``.
"""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.core import observe


def select_ambiguous(scored, top_k=5, gap=0.05):
    """``scored`` = [(score, node_id), ...] sorted desc → ``(top, ambiguous)``.

    ``ambiguous`` is the longest prefix of ``top`` whose score stays within
    ``gap`` of the top-1 score — the "similarity not much different" set.
    """
    top = scored[:top_k]
    if not top:
        return [], []
    best = top[0][0]
    return top, [(s, n) for s, n in top if (best - s) <= gap]


async def ask_llm_judgment(question, contexts):
    """Send the ambiguous candidate set (as text context) to the LLM in one call."""
    return await observe.vlm_answer_question(question, [], contexts)


class TestSelectAmbiguous:
    def test_topk_caps(self):
        scored = [(0.9, 'a'), (0.8, 'b'), (0.7, 'c'), (0.6, 'd'),
                  (0.5, 'e'), (0.4, 'f')]
        top, _ = select_ambiguous(scored, top_k=5, gap=0.05)
        assert len(top) == 5
        assert top[4][1] == 'e'          # 'f' is dropped

    def test_tight_scores_all_ambiguous(self):
        scored = [(0.80, 'a'), (0.79, 'b'), (0.78, 'c'), (0.77, 'd'), (0.76, 'e')]
        _, amb = select_ambiguous(scored, top_k=5, gap=0.05)
        assert len(amb) == 5

    def test_clear_gap_single_ambiguous(self):
        scored = [(0.90, 'a'), (0.70, 'b'), (0.69, 'c'), (0.68, 'd'), (0.67, 'e')]
        _, amb = select_ambiguous(scored, top_k=5, gap=0.05)
        assert len(amb) == 1
        assert amb[0][1] == 'a'

    def test_partial_prefix(self):
        scored = [(0.80, 'a'), (0.78, 'b'), (0.76, 'c'), (0.70, 'd'), (0.69, 'e')]
        _, amb = select_ambiguous(scored, top_k=5, gap=0.05)
        assert [n for _, n in amb] == ['a', 'b', 'c']  # d/e fall outside gap


class TestLLMJudgment:
    def setup_method(self):
        self._orig = observe.vlm_answer_question
        self.calls = []

        async def fake(query, image_paths, node_contexts=None, **_):
            self.calls.append({"query": query, "images": list(image_paths or []),
                               "contexts": list(node_contexts or [])})
            return "MOCK judgment"

        observe.vlm_answer_question = fake

    def teardown_method(self):
        observe.vlm_answer_question = self._orig

    def test_sends_ambiguous_contexts_together(self):
        contexts = ["成功观察到绿植在卧室", "成功观察到绿植在客厅",
                    "成功观察到绿植在厨房"]
        ans = asyncio.run(ask_llm_judgment("绿植放在哪个房间？", contexts))
        assert ans == "MOCK judgment"
        assert self.calls[0]["images"] == []          # text-only
        assert len(self.calls[0]["contexts"]) == 3    # all ambiguous sent together


if __name__ == "__main__":
    import traceback
    tests = [TestSelectAmbiguous(), TestLLMJudgment()]
    passed = failed = 0
    for obj in tests:
        cls = type(obj).__name__
        for name in dir(obj):
            if name.startswith("test_"):
                try:
                    if hasattr(obj, "setup_method"):
                        obj.setup_method()
                    getattr(obj, name)()
                    if hasattr(obj, "teardown_method"):
                        obj.teardown_method()
                    print(f"  PASS {cls}.{name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL {cls}.{name}")
                    traceback.print_exc()
                    failed += 1
                    if hasattr(obj, "teardown_method"):
                        obj.teardown_method()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
