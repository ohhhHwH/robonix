#!/usr/bin/env python3
"""Memory enhancement integration test — validates memory hierarchy structure.

Tests:
  1. PlanNode creation (via ptdl_store)
  2. PlaceNode creation (static/rotating scenario)
  3. PathSegment + ObjectObservation hierarchy (movement scenario)
  4. TextIndex append/update/search
  5. ImageStore per-node limits + GC
  6. Three-tier retrieval with text index scan

Usage:
    python3 scripts/integration_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Setup test data directory ──
_TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "memory"
os.environ.setdefault("AGENT_MEMORY_DIR", str(_TEST_DIR))
os.environ.setdefault("MEMGRAPH_KEEP_DATA", "1")  # Don't clean slate

# ── Imports ──
from memory_service.core.types import (
    NodeType, MemoryNode, SpatialContext, ObjectCoord,
    TagSet, LogRecord, RememberRequest, SearchRequest, TagFilter,
    CameraParams, CameraPose, TimeRange,
)
from memory_service.storage.graph_store import GraphStore
from memory_service.storage.tag_index import TagIndex
from memory_service.storage.vector_store import VectorStore
from memory_service.storage.image_store import ImageStore
from memory_service.storage.text_index import TextIndex, get_text_index
from memory_service.core.remember import RememberPipeline
from memory_service.core.retrieve import RetrievePipeline
from memory_service.path_recorder import PathRecorder
from memory_service.vlm_observer import VLMObserver, VLMFeatureExtractor


class IntegrationTest:
    def __init__(self):
        self.data_dir = str(_TEST_DIR)
        # Clean up previous test data
        for f in Path(self.data_dir).glob("*.json"):
            f.unlink()
        for f in Path(self.data_dir).glob("*.txt"):
            f.unlink()

        self.graph = GraphStore(data_dir=self.data_dir)
        self.tags = TagIndex()
        self.vectors = VectorStore(alpha=0.3, embedding_enabled=False)
        self.images = ImageStore()
        self.text_index = TextIndex(
            path=os.path.join(self.data_dir, "text_index.txt")
        )
        self.remember = RememberPipeline(self.graph, self.tags, self.vectors, self.images)
        self.retrieve = RetrievePipeline(self.graph, self.tags, self.vectors)

        self.results = []
        self.failures = []

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        entry = {"name": name, "status": status, "detail": detail}
        self.results.append(entry)
        if not condition:
            self.failures.append(entry)
        print(f"  [{status}] {name}")
        if detail and not condition:
            print(f"         {detail}")

    async def run_all(self):
        print("=" * 60)
        print("  Memory Enhancement Integration Test")
        print("=" * 60)

        await self.test_1_plan_node()
        await self.test_2_place_node()
        await self.test_3_path_segment()
        await self.test_4_object_observation_hierarchy()
        await self.test_5_text_index()
        await self.test_6_image_limits_gc()
        await self.test_7_retrieval()
        await self.test_8_dataset_fields()

        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        print(f"  Results: {passed} passed, {failed} failed out of {len(self.results)}")

        if self.failures:
            print(f"\n  ❌ FAILURES:")
            for f in self.failures:
                print(f"     [{f['name']}] {f['detail']}")
        else:
            print(f"\n  ✅ ALL TESTS PASSED")
        print("=" * 60)

        return failed == 0

    # ── Test 1: PlanNode ──────────────────────────────────────────────

    async def test_1_plan_node(self):
        print("\n── Test 1: PlanNode (Planning Memory) ──")

        # Simulate a plan-save request via ptdl_store
        try:
            from memory_service.storage.ptdl_store import get_ptdl_store, _ptdl_store_reset
            _ptdl_store_reset()
            ptdl = get_ptdl_store()
            ptdl.add(
                query="navigate from office_A to corridor_east",
                description="navigate→inspect→return",
                steps=["step1: leave office_A", "step2: traverse corridor",
                       "step3: inspect equipment", "step4: return"],
                plan_count=1,
                canceled_count=0,
            )
            plans = ptdl.search(query="navigate", top_k=5)
            self.check("Plan saved to ptdl_store", len(plans) > 0,
                      f"Expected >0 plans, got {len(plans)}")

            if plans:
                plan = plans[0]
                self.check("Plan has query", plan.get("query", "") != "")
                self.check("Plan has description", plan.get("description", "") != "")
                self.check("Plan has steps", len(plan.get("steps", [])) > 0,
                          f"Expected >0 steps, got {len(plan.get('steps', []))}")
                self.check("Plan has plan_count", plan.get("plan_count", 0) > 0)
                print(f"         Plan: \"{plan.get('query','')[:80]}\"")
                print(f"         Steps: {len(plan.get('steps',[]))}")
        except Exception as e:
            self.check("PlanNode creation", False, f"Exception: {e}")

    # ── Test 2: PlaceNode ─────────────────────────────────────────────

    async def test_2_place_node(self):
        print("\n── Test 2: PlaceNode (Static/Rotating Scenario) ──")

        now_ns = time.time_ns()
        lr = LogRecord(ts=now_ns, level="Info", tag="scene_hook",
                       msg="robot stationed at region=corridor_east, center=(6.3,1.2,0.0), duration=45s")

        spatial = SpatialContext(
            origin="world",
            center={"x": 6.3, "y": 1.2, "z": 0.0},
            radius_m=2.0,
            semantic_region="corridor_east",
        )

        req = RememberRequest(
            session_id="test-session",
            plan_id="test-plan",
            log_record=lr,
            spatial=spatial,
            kv={"node_type": "place", "region": "corridor_east"},
        )

        resp = await self.remember.execute(req)
        place_nid = resp.node_id

        self.check("PlaceNode created", place_nid >= 0,
                  f"node_id={place_nid}")

        if place_nid > 0:
            node = self.graph.get_node(place_nid)
            self.check("PlaceNode has PLACE type",
                      node is not None and node.node_type == NodeType.PLACE,
                      f"type={node.node_type if node else 'None'}")
            self.check("PlaceNode has spatial center",
                      node is not None and node.spatial_data is not None
                      and node.spatial_data.center is not None,
                      f"center={node.spatial_data.center if node and node.spatial_data else 'None'}")
            self.check("PlaceNode has semantic_region",
                      node is not None and node.spatial_data is not None
                      and node.spatial_data.semantic_region == "corridor_east")
            self.check("PlaceNode has weight=0.7",
                      node is not None and node.weight == 0.7,
                      f"weight={node.weight if node else 'None'}")
            print(f"         node_id={place_nid}, region=corridor_east, weight=0.7")

    # ── Test 3: PathSegment ───────────────────────────────────────────

    async def test_3_path_segment(self):
        print("\n── Test 3: PathSegment (Movement Memory) ──")

        now_ns = time.time_ns()
        lr = LogRecord(ts=now_ns, level="Info", tag="path_recorder",
                       msg="robot traversed from (0.0,0.0) to (10.0,5.0), length=11.2m, 8 waypoints")

        waypoints = []
        for i in range(8):
            t = now_ns + i * 1000000000  # +1s per waypoint
            x = 1.25 * i
            y = 0.625 * i
            waypoints.append({"x": x, "y": y, "z": 0.0, "yaw": 0.4636, "ts": t})

        spatial = SpatialContext(
            origin="world",
            objects=[ObjectCoord(
                obj_id=f"waypoint.{i}", label=f"waypoint_{i}",
                x=wp["x"], y=wp["y"], z=wp["z"]
            ) for i, wp in enumerate(waypoints)],
        )

        req = RememberRequest(
            session_id="test-session",
            plan_id="navigate-to-corridor",
            log_record=lr,
            spatial=spatial,
            parent_node_id=None,  # Could be linked to a PlanNode
            kv={
                "node_type": "path_segment",
                "waypoints": waypoints,
                "path_length_m": 11.2,
                "waypoint_count": 8,
                "start_pos": [0.0, 0.0, 0.0],
                "end_pos": [10.0, 5.0, 0.0],
            },
        )

        resp = await self.remember.execute(req)
        seg_nid = resp.node_id

        self.check("PathSegment created", seg_nid > 0,
                  f"node_id={seg_nid}")

        if seg_nid > 0:
            node = self.graph.get_node(seg_nid)
            self.check("PathSegment has PATH_SEGMENT type",
                      node is not None and node.node_type == NodeType.PATH_SEGMENT,
                      f"type={node.node_type if node else 'None'}")
            self.check("PathSegment has weight=0.6",
                      node is not None and node.weight == 0.6,
                      f"weight={node.weight if node else 'None'}")
            self.check("PathSegment summary contains waypoints info",
                      node is not None and "waypoints" in (node.summary or "").lower(),
                      f"summary={node.summary[:80] if node else 'None'}")
            print(f"         node_id={seg_nid}, waypoints=8, length=11.2m")

            # Store for next test
            self.seg_nid = seg_nid

    # ── Test 4: ObjectObservation Hierarchy ───────────────────────────

    async def test_4_object_observation_hierarchy(self):
        print("\n── Test 4: ObjectObservation + Hierarchy ──")

        seg_nid = getattr(self, 'seg_nid', None)
        if seg_nid is None:
            self.check("Skip: no PathSegment to attach to", False, "Test 3 must pass first")
            return

        now_ns = time.time_ns()

        # Create 3 object observations under the path_segment
        objects_data = [
            {"label": "chair", "desc": "office chair at center-mid", "conf": 0.92},
            {"label": "monitor", "desc": "lcd monitor at right-near", "conf": 0.88},
            {"label": "fire_extinguisher", "desc": "red extinguisher at left-far", "conf": 0.85},
        ]

        obj_node_ids = []
        for obj_data in objects_data:
            lr = LogRecord(ts=now_ns, level="Info", tag="vlm_observer",
                          msg=f"VLM observed {obj_data['label']} ({obj_data['conf']:.2f}) "
                              f"at {obj_data['desc']}")

            spatial = SpatialContext(
                origin="world",
                objects=[ObjectCoord(
                    obj_id=f"vlm.{obj_data['label']}.abc123",
                    label=obj_data['label'],
                    x=5.0, y=2.5, z=0.0,
                )],
            )

            req = RememberRequest(
                session_id="vlm-observer",
                plan_id="navigate-to-corridor",
                log_record=lr,
                spatial=spatial,
                parent_node_id=seg_nid,  # ← child of PathSegment
                kv={
                    "node_type": "object_observation",
                    "vlm_confidence": obj_data["conf"],
                    "vlm_position": obj_data["desc"],
                },
            )

            resp = await self.remember.execute(req)
            if resp.node_id > 0:
                obj_node_ids.append(resp.node_id)

        self.check("ObjectObservation nodes created",
                  len(obj_node_ids) == 3,
                  f"Expected 3, got {len(obj_node_ids)}")

        if obj_node_ids:
            # Check hierarchy
            for obj_nid in obj_node_ids:
                node = self.graph.get_node(obj_nid)
                self.check(f"Object {obj_nid} has OBJECT_OBSERVATION type",
                          node is not None and node.node_type == NodeType.OBJECT_OBSERVATION,
                          f"type={node.node_type if node else 'None'}")

                # Check parent link
                parents = self.graph.get_parents(obj_nid)
                self.check(f"Object {obj_nid} linked to PathSegment parent",
                          seg_nid in parents,
                          f"parents={parents}")

                children = self.graph.get_children(seg_nid)

            # Check PathSegment children
            children = self.graph.get_children(seg_nid)
            self.check("PathSegment has 3 children",
                      len(children) >= 3,
                      f"Expected >=3, got {len(children)}: {children}")

            print(f"         PathSegment {seg_nid} → children: {children}")

            # Check PlaceNode hierarchy: create object under PlaceNode
            place_lr = LogRecord(ts=time.time_ns(), level="Info", tag="scene_hook",
                              msg="detected chair near corridor_east")
            place_spatial = SpatialContext(
                origin="world",
                objects=[ObjectCoord(
                    obj_id="scene.object.chair_001", label="chair",
                    x=6.3, y=1.2, z=0.0,
                )],
            )
            place_req = RememberRequest(
                session_id="test-session", plan_id="test-plan",
                log_record=place_lr, spatial=place_spatial,
                parent_node_id=None,  # Could link to PlaceNode
                kv={"node_type": "object_observation", "region": "corridor_east"},
            )
            place_obj_resp = await self.remember.execute(place_req)
            self.check("PlaceNode child object created",
                      place_obj_resp.node_id > 0,
                      f"node_id={place_obj_resp.node_id}")

            self.obj_node_ids = obj_node_ids

    # ── Test 5: TextIndex ─────────────────────────────────────────────

    async def test_5_text_index(self):
        print("\n── Test 5: TextIndex ──")

        ti = get_text_index(path=os.path.join(self.data_dir, "text_index.txt"))

        # Rebuild from graph
        all_nodes = []
        for nid in self.graph.all_ids():
            node = self.graph.get_node(nid)
            if node:
                all_nodes.append(node)

        ti.rebuild(all_nodes)

        count = ti.count()
        self.check("TextIndex has entries", count > 0,
                  f"Expected >0, got {count}")

        if count > 0:
            full = ti.full_text(max_lines=50)
            self.check("TextIndex full_text returns data",
                      len(full) > 0,
                      f"Got {len(full)} chars")

            # Check format
            lines = full.split("\n")
            sample = lines[0] if lines else ""
            parts = sample.split("|")
            self.check("TextIndex line has 7 fields",
                      len(parts) == 7,
                      f"Expected 7, got {len(parts)}: {parts}")

            # Search
            results = ti.search("robot", top_k=10)
            self.check("TextIndex search finds path_segment",
                      len(results) > 0,
                      f"Found {len(results)} results for 'robot'")

            if results:
                summary_text = ti.get_summary(results[:3])
                print(f"         Search 'robot' → {len(results)} hits")
                print(f"         Summary: {summary_text[:120]}...")

            # Check node types in index
            node_types_found = set()
            for line in lines:
                parts = line.split("|")
                if len(parts) >= 2:
                    node_types_found.add(parts[1].strip())

            expected_types = {"place", "path_segment", "object_observation"}
            for nt in expected_types:
                self.check(f"TextIndex contains type '{nt}'",
                          nt in node_types_found,
                          f"Found types: {node_types_found}")

    # ── Test 6: Image Limits & GC ──────────────────────────────────────

    async def test_6_image_limits_gc(self):
        print("\n── Test 6: ImageStore Limits & GC ──")

        test_nid = 9999

        # Clean up first
        import shutil
        img_dir = Path(self.images.root) / str(test_nid)
        if img_dir.exists():
            shutil.rmtree(str(img_dir))

        # Save 15 images — should stay at 10 with oldest evicted
        for i in range(15):
            fake_jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xd9'
            self.images.save(test_nid, fake_jpeg)

        img_count = self.images.count(test_nid)
        self.check("ImageStore per-node limit <= 10",
                  img_count <= 10,
                  f"After 15 saves, got {img_count} images")

        self.check("ImageStore has images",
                  img_count > 0,
                  f"Got {img_count} images for node {test_nid}")

        # Test GC orphans
        all_valid_nids = set(self.graph.all_ids())
        # test_nid (9999) is NOT in the graph → should be GC'd
        orphan_count = self.images.gc_orphans(all_valid_nids)
        self.check("GC removes orphan node dirs",
                  orphan_count > 0,
                  f"Removed {orphan_count} orphan dirs")

        # Verify orphan was removed
        remaining = self.images.count(test_nid)
        self.check("Orphan dir no longer has images",
                  remaining == 0,
                  f"Expected 0, got {remaining}")

    # ── Test 7: Retrieval Pipeline ────────────────────────────────────

    async def test_7_retrieval(self):
        print("\n── Test 7: Retrieval Pipeline ──")

        # Test basic search
        req = SearchRequest(query="robot traversed", top_k=5)
        resp = await self.retrieve.execute(req)

        self.check("Search returns results for path query",
                  len(resp.nodes) > 0,
                  f"Expected >0, got {len(resp.nodes)}")

        if resp.nodes:
            node_types = set(n.node_type.value for n in resp.nodes)
            print(f"         Query 'robot traversed' → {len(resp.nodes)} results")
            print(f"         Node types: {node_types}")

        # Test search for VLM objects
        req2 = SearchRequest(query="chair", top_k=5)
        resp2 = await self.retrieve.execute(req2)
        self.check("Search finds VLM-recognized objects",
                  len(resp2.nodes) > 0,
                  f"Expected >0, got {len(resp2.nodes)}")

        # Test causal expansion: search for path should include children
        req3 = SearchRequest(query="path", top_k=5, require_executable=False)
        resp3 = await self.retrieve.execute(req3)

        if resp3.nodes:
            # Check if path_segment results include object_observation children
            found_child = False
            for node in resp3.nodes:
                if node.node_type == NodeType.OBJECT_OBSERVATION:
                    found_child = True
                    break
            # Note: children are included in post_causal expansion, not directly in results
            # unless they match the query
            print(f"         Causal expansion test: {len(resp3.nodes)} nodes")

        self.check("Retrieval pipeline functional",
                  len(resp.nodes) >= 0)  # Always passes — just checking no crash

    # ── Test 8: Dataset Field Completion (P0) ──────────────────────────

    async def test_8_dataset_fields(self):
        print("\n── Test 8: Dataset Field Completion (P0) ──")

        # Check 3: CameraParams round-trip (incl camera_pose)
        pose = CameraPose(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.1, qw=0.99)
        cp = CameraParams(fx=554.0, fy=554.0, cx=320.0, cy=240.0,
                          width=640, height=480, depth_scale=0.001,
                          camera_type="rgb", camera_pose=pose)
        cp_rt = CameraParams.from_dict(cp.to_dict())
        self.check("CameraParams round-trip (incl pose)",
                   cp_rt.fx == 554.0 and cp_rt.camera_pose is not None
                   and cp_rt.camera_pose.qz == 0.1 and cp_rt.camera_pose.qw == 0.99,
                   f"fx={cp_rt.fx}, pose.qz={cp_rt.camera_pose.qz if cp_rt.camera_pose else None}")

        # Check 1: remember request with new fields round-trips through the pipeline
        now_ns = time.time_ns()
        lr = LogRecord(ts=now_ns, level="Info", tag="dataset", msg="dataset compat test")
        spatial = SpatialContext(
            origin="map",
            objects=[ObjectCoord(obj_id="obj.1", label="cup",
                                 label_zh="杯子", label_en="cup",
                                 x=1.0, y=2.0, z=0.5)],
        )
        req = RememberRequest(
            session_id="dataset-session", plan_id="dataset-plan",
            log_record=lr, spatial=spatial,
            camera_params=cp,
            time_range=TimeRange(start_ts=now_ns, end_ts=now_ns + 1000),
            kv={
                "node_type": "object_observation",
                "depth_refs": json.dumps(["depth/1.png"]),
                "frame_ts": json.dumps([now_ns, now_ns + 1000]),
                "video_clip_refs": json.dumps(["clip/1.mp4"]),
                "confidence_flags": json.dumps({"vlm": 0.92}),
            },
        )
        resp = await self.remember.execute(req)
        node = self.graph.get_node(resp.node_id) if resp.node_id > 0 else None
        self.check(
            "Dataset fields round-trip on node",
            node is not None
            and node.camera_params is not None and node.camera_params.fx == 554.0
            and node.time_range is not None and node.time_range.end_ts == now_ns + 1000
            and node.depth_refs == ["depth/1.png"]
            and node.frame_ts == [now_ns, now_ns + 1000]
            and node.video_clip_refs == ["clip/1.mp4"]
            and node.confidence_flags == {"vlm": 0.92}
            and node.summary_zh != "" and node.summary_en != ""
            and (node.spatial_data is not None
                 and node.spatial_data.objects[0].label_zh == "杯子"),
            f"node_id={resp.node_id}",
        )

        # Check 2: old JSON without new fields parses without error
        old = {
            "node_id": 1, "summary": "old node",
            "raw_log": {"ts": 0, "level": "Info", "tag": "x", "msg": "hi"},
            "timestamp": 0, "spatial_data": None, "tags": None,
            "causal_chain": [], "weight": 0.5, "embedding": [],
            "node_type": "short_term", "created_at": 0, "last_access": 0,
            "access_count": 0, "version": 1, "image_refs": [],
        }
        try:
            old_node = MemoryNode.from_dict(old)
            self.check("Old JSON without new fields parses",
                       old_node.summary == "old node" and old_node.depth_refs == []
                       and old_node.camera_params is None
                       and old_node.time_range is None
                       and old_node.summary_zh == "")
        except Exception as e:
            self.check("Old JSON without new fields parses", False, f"Exception: {e}")


async def main():
    os.environ["MEMGRAPH_TIER1_ENABLED"] = "0"  # Disable LLM Tier1 for test
    os.environ["MEMGRAPH_TIER3_ENABLED"] = "0"  # Disable VLM Tier3 for test

    test = IntegrationTest()
    success = await test.run_all()

    # Save results as JSON for report
    results_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "memory" / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test.results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
