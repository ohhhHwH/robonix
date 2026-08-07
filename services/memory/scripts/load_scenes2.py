#!/usr/bin/env python3
"""Load scenes2 dataset into Robonix MemoryService and export memory_nodes.json.

Usage:
  uv run python scripts/load_scenes2.py \\
    --session /path/to/scenes2/session_001 \\
    --data-dir /tmp/memory_test \\
    --output memory_nodes.json

Reads images_index.csv, depth_index.csv, camera_calib.yaml, objects.yaml,
and base_link_pose.csv from the session directory, constructs a
RememberRequest per frame, feeds it through MemoryService, and exports
the full memory graph as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_SVC = _HERE.parent
sys.path.insert(0, str(_SVC))

from memory_service.service import MemoryService
from memory_service.core.types import (
    CameraParams, LogRecord, MemoryNode, ObjectCoord,
    RememberRequest, SpatialContext, TimeRange,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load scenes2 dataset into Robonix memory")
    p.add_argument("--session", required=True, help="Path to session directory")
    p.add_argument("--data-dir", default="/tmp/scenes2_memory", help="MemoryService data dir")
    p.add_argument("--output", default="memory_nodes.json", help="Output JSON path")
    p.add_argument("--limit", type=int, default=0, help="Max frames to load (0=all)")
    return p.parse_args()


def load_camera_params(calib_path: str) -> CameraParams:
    with open(calib_path) as f:
        c = yaml.safe_load(f)
    rgb = c["rgb"]
    ext = c.get("extrinsics", {}).get("rgb_to_base", {})
    return CameraParams(
        fx=float(rgb["fx"]), fy=float(rgb["fy"]),
        cx=float(rgb["cx"]), cy=float(rgb["cy"]),
        width=int(rgb["width"]), height=int(rgb["height"]),
        camera_pose=ObjectCoord(
            obj_id="camera_rgb", label="",
            x=float(ext.get("x", 0)), y=float(ext.get("y", 0)),
            z=float(ext.get("z", 0)),
        ),
        depth_scale=float(c.get("depth", {}).get("depth_scale", 1.0)),
        camera_type="rgb",
    )


def load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_frame_map(
    images: List[Dict], depths: List[Dict],
) -> Dict[int, Dict[str, Any]]:
    """Index frames by frame_id, attaching depth where aligned."""
    frame_map: Dict[int, Dict[str, Any]] = {}
    for row in images:
        fid = int(row["frame_id"])
        frame_map[fid] = {
            "ts": int(row["ts"]),
            "image_path": row["file_path"],
            "cam_x": float(row["cam_x"]),
            "cam_y": float(row["cam_y"]),
            "cam_z": float(row["cam_z"]),
            "cam_qx": float(row["cam_qx"]),
            "cam_qy": float(row["cam_qy"]),
            "cam_qz": float(row["cam_qz"]),
            "cam_qw": float(row["cam_qw"]),
        }
    for row in depths:
        fid = int(row["aligned_to_rgb_frame"])
        if fid in frame_map:
            frame_map[fid]["depth_path"] = row["file_path"]
            frame_map[fid]["depth_ts"] = int(row["ts"])
    return frame_map


async def main() -> None:
    args = parse_args()
    session_dir = Path(args.session)

    # 1. Load inputs
    calib_path = session_dir / "camera_calib.yaml"
    images_path = session_dir / "images_index.csv"
    depths_path = session_dir / "depth_index.csv"
    poses_path = session_dir / "base_link_pose.csv"
    objects_path = session_dir / "objects.yaml"

    camera_params = load_camera_params(str(calib_path))
    images = load_csv(str(images_path))
    depths = load_csv(str(depths_path))
    frame_map = build_frame_map(images, depths)

    objects_gt: Dict[str, Any] = {}
    if objects_path.exists():
        with open(objects_path) as f:
            data = yaml.safe_load(f)
            for obj in data.get("objects", []):
                objects_gt[obj["obj_id"]] = obj

    if args.limit and args.limit > 0:
        frame_map = dict(sorted(frame_map.items())[:args.limit])

    print(f"Session: {session_dir}")
    print(f"Frames: {len(frame_map)} images, {len(depths)} depth records")
    print(f"Objects (GT): {len(objects_gt)}")
    print(f"Camera: {camera_params.width}x{camera_params.height} "
          f"fx={camera_params.fx} depth_scale={camera_params.depth_scale}")

    # 2. Initialize MemoryService
    svc = MemoryService(data_dir=args.data_dir)
    await svc.init()

    # 3. Process each frame
    t0 = time.time()
    saved = 0
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None

    for fid in sorted(frame_map.keys()):
        fm = frame_map[fid]
        ts = fm["ts"]
        if first_ts is None:
            first_ts = ts
        last_ts = ts

        # Build spatial context: camera pose + matched objects
        spatial_objects = [ObjectCoord(
            obj_id=f"cam_frame_{fid:06d}",
            label="camera_frame",
            x=fm["cam_x"], y=fm["cam_y"], z=fm["cam_z"],
        )]
        # Match objects from objects.yaml to this frame via timestamp
        for obj_id, obj in objects_gt.items():
            first_ts = obj.get("first_seen_ts", 0)
            last_ts = obj.get("last_seen_ts", 0)
            # ts=0 means "visible throughout" (no temporal filtering)
            if first_ts == 0 and last_ts == 0:
                in_range = True
            else:
                in_range = (first_ts <= ts <= last_ts) if last_ts > 0 else (ts >= first_ts)
            if in_range:
                pos = obj.get("position", {})
                spatial_objects.append(ObjectCoord(
                    obj_id=obj_id,
                    label=obj.get("label_en", obj.get("label_zh", "unknown")),
                    x=float(pos.get("x", 0)),
                    y=float(pos.get("y", 0)),
                    z=float(pos.get("z", 0)),
                ))

        # Read image file as base64 (or use placeholder for speed)
        img_path = session_dir / fm["image_path"]
        img_b64 = ""
        depth_b64 = ""
        if img_path.exists():
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()

        depth_path_str = fm.get("depth_path", "")
        if depth_path_str:
            dp = session_dir / depth_path_str
            if dp.exists():
                depth_b64 = base64.b64encode(dp.read_bytes()).decode()

        # Build object label summary for kv
        obj_labels = list(set(o.label for o in spatial_objects if o.label != "camera_frame"))
        objects_str = ", ".join(sorted(obj_labels)[:30])

        spatial = SpatialContext(
            origin="world",
            objects=spatial_objects,
        )

        req = RememberRequest(
            session_id=f"scenes2-{session_dir.name}",
            plan_id=f"load-frame-{fid:06d}",
            log_record=LogRecord(
                ts=ts, level="Info", tag="scenes2",
                msg=f"loaded frame {fid:06d} with {len(spatial_objects)-1} objects: [{objects_str}]",
            ),
            spatial=spatial,
            image_base64=img_b64,
            depth_base64=depth_b64,
            camera_params=camera_params,
            time_range=TimeRange(start_ts=ts, end_ts=ts),
            kv={
                "frame_ts": str(ts),
                "video_clip_refs": "",
                "objects": objects_str,
            },
        )
        try:
            resp = await svc._remember_pipe.execute(req)
            if resp.node_id >= 0:
                saved += 1
        except Exception as e:
            print(f"  [{fid}] ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nSaved: {saved}/{len(frame_map)} nodes in {elapsed:.1f}s "
          f"({saved/elapsed:.1f} fps)" if elapsed > 0 else "")

    # 4. Export memory_nodes.json
    nodes = [
        svc.graph.get_node(nid).to_dict()
        for nid in svc.graph.all_ids()
        if svc.graph.get_node(nid) is not None
    ]
    with open(args.output, "w") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)

    print(f"Exported: {args.output} ({len(nodes)} nodes)")
    print(f"Data dir: {args.data_dir}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
