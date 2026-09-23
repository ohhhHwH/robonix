# SPDX-License-Identifier: MulanPSL-2.0
"""Tree-shape / same-question detection tests for PtdlStore.

The change under test: ``add`` is now an upsert keyed on the exact ``query``
(the root) plus ``plan_id`` (the child sub-plan). One user question maps to
exactly one root record; each successful planning round is a child that keeps
its own ``rtdl_plan`` / ``raw_rtdl`` / ``steps`` — instead of flattening every
round into one step list and dropping all but the last Plan AST.
"""

from pathlib import Path

from memory_service.storage.ptdl_store import PtdlStore


def _store(tmp_path: Path) -> PtdlStore:
    return PtdlStore(path=str(tmp_path / "ptdl_store.json"))


def _round(plan_id, desc, steps, rtdl=None):
    return dict(plan_id=plan_id, description=desc, steps=steps, rtdl_plan=rtdl)


def test_first_save_creates_one_root_with_one_child(tmp_path):
    store = _store(tmp_path)
    store.add("去厨房拿可乐", "记录起点并扫描",
              ["1. [nav] navigate", "2. [grip] grasp"], plan_id="1",
              rtdl_plan='{"plan":"r1"}')
    assert store.count() == 1
    entry = store._entries[0]
    assert entry["query"] == "去厨房拿可乐"
    assert entry["plan_count"] == 1
    assert len(entry["plans"]) == 1
    child = entry["plans"][0]
    assert child["plan_id"] == "1"
    assert child["rtdl_plan"] == '{"plan":"r1"}'
    assert child["steps"] == ["1. [nav] navigate", "2. [grip] grasp"]


def test_same_query_different_plan_ids_become_siblings(tmp_path):
    """A multi-round task saved once per round must collapse to a single root
    whose children are the rounds, each keeping its own Plan AST."""
    store = _store(tmp_path)
    store.add("旋转扫描并移动到最远物体再返回", "记录起点并扫描",
              ["1. [get_pose] 记录起点", "2. [list_objects] 列出物体"],
              plan_id="1", rtdl_plan='{"plan":"r1"}')
    store.add("旋转扫描并移动到最远物体再返回", "获取目标点",
              ["1. [goal_near] 获取目标点"],
              plan_id="2", rtdl_plan='{"plan":"r2"}')
    # Rollup finalizes the root (no child, no rtdl_plan).
    store.add("旋转扫描并移动到最远物体再返回",
              "complete task (3 step(s) across 2 planning round(s))",
              [], plan_id=None)

    assert store.count() == 1
    entry = store._entries[0]
    assert entry["plan_count"] == 2
    assert entry["description"].startswith("complete task")
    assert len(entry["plans"]) == 2
    # Each child retains its own rtdl_plan — not flattened / last-wins.
    assert entry["plans"][0]["rtdl_plan"] == '{"plan":"r1"}'
    assert entry["plans"][1]["rtdl_plan"] == '{"plan":"r2"}'
    assert entry["plans"][0]["plan_id"] == "1"
    assert entry["plans"][1]["plan_id"] == "2"


def test_same_plan_id_merges_steps(tmp_path):
    store = _store(tmp_path)
    store.add("向后移动2m", "第一步", ["1. [get_pose] 读位姿"], plan_id="1")
    store.add("向后移动2m", "第一步", ["1. [nav] 后退"], plan_id="1")
    assert store.count() == 1
    entry = store._entries[0]
    assert len(entry["plans"]) == 1
    assert entry["plans"][0]["steps"] == [
        "1. [get_pose] 读位姿",
        "2. [nav] 后退",
    ]


def test_rollup_without_plan_id_does_not_add_child(tmp_path):
    store = _store(tmp_path)
    store.add("任务A", "第一步", ["1. [nav] go"], plan_id="1")
    store.add("任务A", "complete task (1 step(s) across 1 planning round(s))",
              [], plan_id=None)
    assert store.count() == 1
    entry = store._entries[0]
    assert len(entry["plans"]) == 1  # rollup added no child
    assert entry["description"].startswith("complete task")


def test_status_and_counts(tmp_path):
    store = _store(tmp_path)
    store.add("任务B", "成功轮", ["1. [nav] go"], plan_id="1")
    store.add("任务B", "失败轮 (FAILED)", ["1. [nav] fail"], plan_id="2")
    store.add("任务B", "取消轮 (FAILED,canceled)", ["1. [nav] cancel"],
              plan_id="3", canceled_count=1)
    entry = store._entries[0]
    assert entry["plan_count"] == 1      # only the successful round
    assert entry["canceled_count"] == 1  # only the canceled round
    statuses = {p["plan_id"]: p["status"] for p in entry["plans"]}
    assert statuses == {"1": "success", "2": "failed", "3": "canceled"}


def test_legacy_flat_entry_migrates(tmp_path):
    store = _store(tmp_path)
    # Simulate a pre-tree record loaded from disk.
    store._entries = [{
        "query": "旧问题",
        "description": "navigate→return",
        "steps": ["1. [nav] go", "2. [nav] back"],
        "rtdl_plan": '{"plan":"old"}',
        "timestamp_ns": 1,
        "plan_count": 1,
        "canceled_count": 0,
    }]
    store.add("旧问题", "新轮", ["1. [nav] extra"], plan_id="2")
    entry = store._entries[0]
    assert "plans" in entry
    assert len(entry["plans"]) == 2
    assert entry["plans"][0]["plan_id"] == "legacy"
    assert entry["plans"][0]["rtdl_plan"] == '{"plan":"old"}'


def test_search_returns_exact_match_first(tmp_path):
    store = _store(tmp_path)
    store.add("去厨房检查灭火器", "inspect", ["1. [nav] go kitchen"], plan_id="1")
    store.add("巡逻走廊检查设备", "patrol", ["1. [nav] go corridor"], plan_id="1")
    results = store.search("去厨房检查灭火器", top_k=5)
    assert results[0]["query"] == "去厨房检查灭火器"


def test_search_exact_plus_fuzzy(tmp_path):
    store = _store(tmp_path)
    store.add("去厨房检查灭火器", "inspect fire extinguisher",
              ["1. [nav] go kitchen"], plan_id="1")
    store.add("巡逻走廊检查设备", "patrol corridor",
              ["1. [nav] go corridor"], plan_id="1")
    results = store.search("去厨房检查灭火器", top_k=5)
    assert results[0]["query"] == "去厨房检查灭火器"
    assert any(e["query"] == "巡逻走廊检查设备" for e in results[1:])


def test_different_queries_stay_separate(tmp_path):
    store = _store(tmp_path)
    store.add("去厨房拿可乐", "", ["1. [nav] go"], plan_id="1")
    store.add("去办公室关电脑", "", ["1. [nav] go"], plan_id="1")
    assert store.count() == 2
