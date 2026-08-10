#!/usr/bin/env python3
"""Path Memory Enhancement — end-to-end integration demo.

Demonstrates the full 6-module path memory pipeline:
  1. PathRecorder records waypoints with spatial sampling
  2. VLMObserver recognizes objects from waypoint images
  3. PlaceNode created for static regions
  4. Three-tier retrieval (text index -> BM25+LLM -> VLM)

Usage:
    python3 scripts/path_memory_demo.py                  # dry-run (no memgraph)
    python3 scripts/path_memory_demo.py --live            # requires memgraph on :37798
    python3 scripts/path_memory_demo.py --live --vlm      # also require VLM API keys
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("path_memory_demo")

# Try to import Pillow for generating test images
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ═════════════════════════════════════════════════════════════════════════
# Test data generation
# ═════════════════════════════════════════════════════════════════════════

def _generate_test_image(label: str, width: int = 640, height: int = 480) -> bytes:
    """Generate a simple test JPEG image with text overlay."""
    if not HAS_PIL:
        # Return a minimal valid JPEG (1x1 grey pixel)
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
            "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
            "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCA"
            "ABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAA"
            "AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkK"
            "FhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWG"
            "h4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl"
            "5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
            "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYk"
            "NOEl8RcYIidU4pJjc6LT1NXWGSY2R1h2hJipOjZJWWl5iZmqSmq6Slpqeoqba3uLm6wsPEx"
            "cbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//Z"
        )

    img = Image.new("RGB", (width, height), color=(40, 40, 80))
    draw = ImageDraw.Draw(img)
    # Draw a colored rectangle as "object"
    colors = {
        "chair": (139, 90, 43),
        "monitor": (60, 60, 60),
        "fire_extinguisher": (200, 30, 30),
        "table": (160, 120, 60),
        "door": (100, 80, 60),
        "plant": (30, 130, 30),
        "default": (100, 100, 100),
    }
    color = colors.get(label, colors["default"])
    draw.rectangle([200, 150, 440, 380], fill=color, outline=(255, 255, 255), width=2)
    draw.text((220, 240), label.upper(), fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def _simulate_path(
    start: tuple, end: tuple, steps: int = 20,
) -> List[Dict[str, Any]]:
    """Generate a simulated straight-line path with waypoints."""
    waypoints = []
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        # Slight sinusoidal waviness
        if steps > 2 and 0 < i < steps - 1:
            y += 0.3 * math.sin(t * math.pi)
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        waypoints.append({
            "x": round(x, 3), "y": round(y, 3), "z": 0.0,
            "yaw": round(yaw, 4), "ts": _time.time(),
        })
    return waypoints


# ═════════════════════════════════════════════════════════════════════════
# Dry-run demo (no memgraph needed)
# ═════════════════════════════════════════════════════════════════════════

async def demo_dry_run():
    """Run the path memory pipeline without external services.

    Validates the local data structures and flow without requiring
    memgraph or VLM to be running.
    """
    print("═" * 60)
    print("  Path Memory Enhancement — Dry-Run Demo")
    print("═" * 60)

    # ── 1. Import modules ──
    print("\n── Step 1: Import & Setup ──")
    try:
        from memory_service.storage.text_index import TextIndex
        from memory_service.path_recorder import PathRecorder
        print("  ✅ path_recorder module loaded")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return 1

    # ── 2. TextIndex ──
    print("\n── Step 2: TextIndex Validation ──")
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        idx_path = tf.name

    try:
        ti = TextIndex(path=idx_path)
        assert ti.count() == 0, "empty index should have count=0"
        print("  ✅ TextIndex created (empty)")

        # Append a simulated node
        from memory_service.core.types import (
            MemoryNode, NodeType, LogRecord, TagSet, SpatialContext,
        )
        node = MemoryNode(
            node_id=1,
            summary="robot traversed from (0,0) to (5,3), length=5.8m, 20 waypoints",
            raw_log=LogRecord(ts=_time.time_ns(), level="Info", tag="test", msg="test"),
            timestamp=_time.time_ns(),
            tags=TagSet(region="corridor_east"),
            weight=0.6,
            node_type=NodeType.PATH_SEGMENT,
            created_at=_time.time_ns(),
            version=1,
        )
        ti.append(node)
        assert ti.count() == 1, "should have 1 node after append"
        print("  ✅ TextIndex.append() works")

        # full_text()
        ft = ti.full_text()
        assert "corridor_east" in ft, f"full_text should contain region: {ft}"
        print("  ✅ TextIndex.full_text() works")

        # search()
        hits = ti.search("corridor")
        assert len(hits) == 1 and hits[0] == 1
        print("  ✅ TextIndex.search() works")

        # update()
        node.summary = "updated summary with new info"
        ti.update(1, node)
        ft2 = ti.full_text()
        assert "updated" in ft2
        print("  ✅ TextIndex.update() works")

        # remove()
        ti.remove(1)
        assert ti.count() == 0
        print("  ✅ TextIndex.remove() works")
    finally:
        os.unlink(idx_path)

    # ── 3. PathRecorder (simulated) ──
    print("\n── Step 3: PathRecorder Simulation ──")
    rec = PathRecorder()
    assert rec.waypoint_count == 0
    print("  ✅ PathRecorder initialized (0 waypoints)")

    # Simulate a patrol path
    path = _simulate_path((0.0, 0.0), (5.0, 3.0), steps=25)
    recorded = 0
    for wp in path:
        img = _generate_test_image("chair")
        ok = rec.record_waypoint(wp["x"], wp["y"], wp["z"], wp["yaw"], img)
        if ok:
            recorded += 1
    print(f"  ✅ Recorded {recorded}/{len(path)} waypoints "
          f"(sampling gates: ≥0.5m, ≥15°, ≥1s)")

    # Check last segment data
    assert rec.waypoint_count > 0, "should have recorded waypoints"
    print(f"  ✅ PathRecorder has {rec.waypoint_count} uncommitted waypoints")

    # ── 4. VLMObserver simulation (no VLM) ──
    print("\n── Step 4: VLMObserver Structure Check ──")
    try:
        from memory_service.vlm_observer import VLMObserver, VLMFeatureExtractor
        print("  ✅ VLMObserver class loaded")
        print("  ✅ VLMFeatureExtractor class loaded")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return 1

    # ── 5. ImageStore GC ──
    print("\n── Step 5: ImageStore GC Check ──")
    try:
        from memory_service.storage.image_store import ImageStore
        store = ImageStore()
        # Save test images
        img_bytes = _generate_test_image("table")
        for _ in range(3):
            store.save(9999, img_bytes)
        cnt = store.count(9999)
        print(f"  ✅ ImageStore: {cnt} images saved for node 9999")

        # GC orphan check
        store.gc_orphans(valid_node_ids=set())  # no valid nodes -> should clean 9999
        cnt2 = store.count(9999)
        print(f"  ✅ GC orphans cleaned node 9999: {cnt} -> {cnt2}")

        # Clean up
        store.remove(9999)
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return 1

    # ── 6. NodeType check ──
    print("\n── Step 6: NodeType Enum Check ──")
    assert NodeType.PLACE.value == "place"
    assert NodeType.PATH_SEGMENT.value == "path_segment"
    assert NodeType.OBJECT_OBSERVATION.value == "object_observation"
    print("  ✅ PLACE, PATH_SEGMENT, OBJECT_OBSERVATION all present")

    # ── 7. Summary ──
    print(f"\n{'═' * 60}")
    print("  Dry-run PASSED — all local modules validated")
    print(f"{'═' * 60}")
    return 0


# ═════════════════════════════════════════════════════════════════════════
# Live demo (requires memgraph on :37798)
# ═════════════════════════════════════════════════════════════════════════

async def demo_live(use_vlm: bool = False):
    """Run the path memory pipeline against a live memgraph.

    Requires:
      - memgraph running on SCENE_HOOK_PORT (default :37798)
      - MEM_VLM_API_KEY + MEM_VLM_BASE_URL (if --vlm)
    """
    print("═" * 60)
    print("  Path Memory Enhancement — Live Demo")
    print("═" * 60)

    import httpx

    MEMGRAPH_URL = os.environ.get("MEMGRAPH_HOOK_URL", "http://127.0.0.1:37798")

    # ── 1. Reset memgraph ──
    print("\n── Step 1: Reset memgraph ──")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{MEMGRAPH_URL}/reset")
        print(f"  ✅ Reset: {r.json().get('message', r.text[:100])}")
    except Exception as e:
        print(f"  ⚠️  Could not reset memgraph: {e}")
        print("  (continuing anyway — results may be noisy)")

    # ── 2. Create a simulated patrol session ──
    print("\n── Step 2: Simulated Patrol Session ──")

    # Create a PathRecorder (just for waypoint generation)
    from memory_service.path_recorder import PathRecorder, _encode_jpeg

    rec = PathRecorder(memgraph_url=MEMGRAPH_URL)

    # Simulate 3 path segments through different regions
    segments = [
        ("office_A", (0.0, 0.0), (3.0, 0.5), "chair"),
        ("corridor", (3.0, 0.5), (7.0, 2.0), "fire_extinguisher"),
        ("corridor_east", (7.0, 2.0), (10.0, 5.0), "monitor"),
    ]

    segment_nodes: List[int] = []
    for region, start, end, obj_label in segments:
        path = _simulate_path(start, end, steps=15)
        recorded = 0
        for wp in path:
            img = _generate_test_image(obj_label)
            ok = rec.record_waypoint(wp["x"], wp["y"], wp["z"], wp["yaw"], img)
            if ok:
                recorded += 1
        print(f"  Region={region}: {recorded} waypoints, obj={obj_label}")

        # Close segment → POST to memgraph
        nid = rec.close_segment()
        if nid:
            segment_nodes.append(nid)
            print(f"  ✅ Segment closed → node {nid}")
        else:
            print(f"  ⚠️  Segment close failed (memgraph not running?)")

    print(f"  Total: {len(segment_nodes)} path_segment nodes created")

    # ── 3. Create PlaceNodes ──
    print("\n── Step 3: PlaceNode Creation ──")
    place_payloads = [
        {
            "session_id": "demo-place",
            "plan_id": "demo-place",
            "log_record": {
                "ts": _time.time_ns(),
                "level": "Info",
                "tag": "demo",
                "msg": "robot stationed at region=office_A, center=(0.5,0.2,0.0)",
            },
            "spatial": {
                "origin": "world",
                "center": {"x": 0.5, "y": 0.2, "z": 0.0},
                "radius_m": 2.0,
                "semantic_region": "office_A",
                "objects": [],
            },
            "kv": {"node_type": "place", "region": "office_A"},
        },
        {
            "session_id": "demo-place",
            "plan_id": "demo-place",
            "log_record": {
                "ts": _time.time_ns(),
                "level": "Info",
                "tag": "demo",
                "msg": "robot stationed at region=corridor_east, center=(9.5,4.5,0.0)",
            },
            "spatial": {
                "origin": "world",
                "center": {"x": 9.5, "y": 4.5, "z": 0.0},
                "radius_m": 2.0,
                "semantic_region": "corridor_east",
                "objects": [],
            },
            "kv": {"node_type": "place", "region": "corridor_east"},
        },
    ]

    place_nodes = []
    for payload in place_payloads:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(MEMGRAPH_URL, json=payload)
            if r.status_code < 400:
                nid = r.json().get("node_id")
                place_nodes.append(nid)
                region = payload["kv"]["region"]
                print(f"  ✅ PlaceNode created → node {nid} ({region})")
            else:
                print(f"  ❌ PlaceNode POST failed: {r.status_code}")
        except Exception as e:
            print(f"  ❌ PlaceNode POST error: {e}")

    # ── 4. VLMObserver (optional) ──
    if use_vlm:
        print("\n── Step 4: VLMObserver Object Recognition ──")
        api_key = os.environ.get("MEM_VLM_API_KEY", os.environ.get("VLM_API_KEY", ""))
        base_url = os.environ.get("MEM_VLM_BASE_URL", os.environ.get("VLM_BASE_URL", ""))
        if not api_key or not base_url:
            print("  ❌ VLM API credentials not set (MEM_VLM_API_KEY / MEM_VLM_BASE_URL)")
        else:
            from memory_service.vlm_observer import VLMObserver
            from memory_service.storage.image_store import ImageStore

            images = ImageStore()
            obs = VLMObserver(
                vlm_api_key=api_key,
                vlm_base_url=base_url,
                image_store=images,
                memgraph_url=MEMGRAPH_URL,
            )
            obs.start()
            print(f"  ✅ VLMObserver started (model={obs._model})")

            # Feed waypoint images from the last segment
            seg_data = rec.get_last_segment_data()
            if seg_data:
                nid = seg_data["node_id"]
                for wp, img_b64 in zip(seg_data["waypoints"], seg_data["images_b64"]):
                    obs.enqueue(nid, wp, img_b64)
                print(f"  ✅ Enqueued {len(seg_data['waypoints'])} waypoints for VLM")
                obs.join_empty(timeout=120)
                print(f"  ✅ VLMObserver stats: {obs.stats}")
            obs.stop()
            print("  ✅ VLMObserver stopped")
    else:
        print("\n── Step 4: VLMObserver (skipped — use --vlm to enable) ──")

    # ── 5. Search / retrieval ──
    print("\n── Step 5: Search Queries ──")
    try:
        from memory_service.service import MemoryService
        svc = MemoryService(data_dir="")
        await svc.init()

        queries = [
            "where was the fire extinguisher seen?",
            "what path did the robot take through the corridor?",
            "objects observed near office A",
        ]
        for q in queries:
            try:
                resp = await svc.search(q, top_k=3)
                print(f"  Query: \"{q}\" → {len(resp.nodes)} results")
                for n in resp.nodes[:2]:
                    print(f"    node {n.node_id}: \"{n.summary[:80]}\"")
            except Exception as e:
                print(f"  Query: \"{q}\" → error: {e}")
    except Exception as e:
        print(f"  ⚠️  Search not available (MemoryService may need memgraph): {e}")

    # ── 6. Summary ──
    print(f"\n{'═' * 60}")
    print(f"  Live demo complete")
    print(f"    Path segments: {len(segment_nodes)}")
    print(f"    PlaceNodes:    {len(place_nodes)}")
    print(f"{'═' * 60}")
    return 0


# ═════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═════════════════════════════════════════════════════════════════════════

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Path Memory Enhancement — integration demo"
    )
    parser.add_argument("--live", action="store_true",
                        help="Run against live memgraph on :37798")
    parser.add_argument("--vlm", action="store_true",
                        help="Enable VLM object recognition (requires API keys)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.live:
        return asyncio.run(demo_live(use_vlm=args.vlm))
    else:
        return asyncio.run(demo_dry_run())


if __name__ == "__main__":
    sys.exit(main())
