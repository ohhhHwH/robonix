#!/usr/bin/env python3
"""Load scenes2 dataset into Robonix MemoryService and export memory_nodes.json.

Usage:
  # Frame mode (default): one MemoryNode per image frame
  uv run python scripts/load_scenes2.py \\
    --session /path/to/scenes2/session_001 \\
    --data-dir /tmp/memory_test \\
    --output memory_nodes.json

  # Video mode: one MemoryNode per video clip (time-based segmentation)
  uv run python scripts/load_scenes2.py \\
    --session /path/to/scenes2/session_001 \\
    --data-dir /tmp/memory_test \\
    --output memory_nodes.json \\
    --mode video --clip-duration-sec 5.0

Reads images_index.csv, depth_index.csv, camera_calib.yaml, objects.yaml,
videos_index.csv, and base_link_pose.csv from the session directory.
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
    p.add_argument("--mode", choices=["frame", "video"], default="frame",
                   help="frame: one node per image (default); video: one node per video clip")
    p.add_argument("--clip-duration-sec", type=float, default=5.0,
                   help="Video mode: seconds per clip segment (default 5.0)")
    p.add_argument("--thumbnail-interval", type=int, default=15,
                   help="Video mode: encode 1 thumbnail image every N frames (0=none)")
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


def build_video_clips(
    frame_map: Dict[int, Dict[str, Any]],
    video_refs: List[str],
    clip_duration_sec: float,
    thumbnail_interval: int,
) -> List[Dict[str, Any]]:
    """Segment frames into time-based clips for video-mode loading.

    Each clip covers ``clip_duration_sec`` seconds and produces one MemoryNode.

    Returns a list of clip descriptors:
        {clip_id, start_ts, end_ts, start_fid, end_fid, frame_ts: [...],
         frame_ids: [...], video_clip_refs: [...], thumbnail_fid: int|None}
    """
    sorted_fids = sorted(frame_map.keys())
    if not sorted_fids:
        return []

    min_ts = frame_map[sorted_fids[0]]["ts"]
    max_ts = frame_map[sorted_fids[-1]]["ts"]
    duration_ns = max_ts - min_ts
    clip_duration_ns = int(clip_duration_sec * 1e9)

    if clip_duration_ns <= 0 or duration_ns <= 0:
        return []

    clips: List[Dict[str, Any]] = []
    clip_start_ts = min_ts
    clip_idx = 0
    frames_since_thumbnail = 0

    while clip_start_ts < max_ts:
        clip_end_ts = clip_start_ts + clip_duration_ns
        frame_ids = []
        frame_ts_list = []
        thumbnail_fid = None

        for fid in sorted_fids:
            ts = frame_map[fid]["ts"]
            if clip_start_ts <= ts < clip_end_ts:
                frame_ids.append(fid)
                frame_ts_list.append(ts)
                # Pick thumbnail frame every N frames
                if thumbnail_interval > 0 and thumbnail_fid is None:
                    thumbnail_fid = fid

        if frame_ids:
            clips.append({
                "clip_id": clip_idx,
                "start_ts": clip_start_ts,
                "end_ts": clip_end_ts,
                "start_fid": frame_ids[0],
                "end_fid": frame_ids[-1],
                "frame_ts": frame_ts_list,
                "frame_ids": frame_ids,
                "video_clip_refs": list(video_refs),
                "thumbnail_fid": thumbnail_fid,
            })
            clip_idx += 1

        clip_start_ts = clip_end_ts

    return clips


def match_objects_to_clip(
    objects_gt: Dict[str, Any],
    clip_start_ts: int,
    clip_end_ts: int,
) -> List[ObjectCoord]:
    """Return objects from objects.yaml whose time range overlaps the clip."""
    result: List[ObjectCoord] = []
    for obj_id, obj in objects_gt.items():
        first_ts = obj.get("first_seen_ts", 0)
        last_ts_obj = obj.get("last_seen_ts", 0)
        # ts=0 means "visible throughout" (no temporal filtering)
        if first_ts == 0 and last_ts_obj == 0:
            in_range = True
        else:
            in_range = (first_ts < clip_end_ts and
                        (last_ts_obj == 0 or last_ts_obj > clip_start_ts))
        if in_range:
            pos = obj.get("position", {})
            result.append(ObjectCoord(
                obj_id=obj_id,
                label=obj.get("label_en", obj.get("label_zh", "unknown")),
                x=float(pos.get("x", 0)),
                y=float(pos.get("y", 0)),
                z=float(pos.get("z", 0)),
            ))
    return result


async def main() -> None:
    args = parse_args()
    session_dir = Path(args.session)

    # 1. Load inputs (common)
    calib_path = session_dir / "camera_calib.yaml"
    images_path = session_dir / "images_index.csv"
    depths_path = session_dir / "depth_index.csv"
    objects_path = session_dir / "objects.yaml"
    videos_csv_path = session_dir / "videos_index.csv"

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
    print(f"Mode: {args.mode}")
    print(f"Frames: {len(frame_map)} images, {len(depths)} depth records")
    print(f"Objects (GT): {len(objects_gt)}")
    print(f"Camera: {camera_params.width}x{camera_params.height} "
          f"fx={camera_params.fx} depth_scale={camera_params.depth_scale}")

    # 2. Initialize MemoryService
    svc = MemoryService(data_dir=args.data_dir)
    await svc.init()

    if args.mode == "video":
        await _main_video_mode(args, session_dir, frame_map, objects_gt,
                               camera_params, svc)
    else:
        await _main_frame_mode(args, session_dir, frame_map, objects_gt,
                               camera_params, svc)

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


async def _main_frame_mode(
    args: argparse.Namespace,
    session_dir: Path,
    frame_map: Dict[int, Dict[str, Any]],
    objects_gt: Dict[str, Any],
    camera_params: CameraParams,
    svc: MemoryService,
) -> None:
    """Original per-frame loading: one MemoryNode per image."""
    t0 = time.time()
    saved = 0

    for fid in sorted(frame_map.keys()):
        fm = frame_map[fid]
        ts = fm["ts"]

        # Build spatial context: camera pose + matched objects
        spatial_objects = [ObjectCoord(
            obj_id=f"cam_frame_{fid:06d}",
            label="camera_frame",
            x=fm["cam_x"], y=fm["cam_y"], z=fm["cam_z"],
        )]
        for obj_id, obj in objects_gt.items():
            first_ts_o = obj.get("first_seen_ts", 0)
            last_ts_o = obj.get("last_seen_ts", 0)
            if first_ts_o == 0 and last_ts_o == 0:
                in_range = True
            else:
                in_range = (first_ts_o <= ts <= last_ts_o) if last_ts_o > 0 else (ts >= first_ts_o)
            if in_range:
                pos = obj.get("position", {})
                spatial_objects.append(ObjectCoord(
                    obj_id=obj_id,
                    label=obj.get("label_en", obj.get("label_zh", "unknown")),
                    x=float(pos.get("x", 0)),
                    y=float(pos.get("y", 0)),
                    z=float(pos.get("z", 0)),
                ))

        # Read image file as base64
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

        obj_labels = list(set(o.label for o in spatial_objects if o.label != "camera_frame"))
        objects_str = ", ".join(sorted(obj_labels)[:30])

        spatial = SpatialContext(origin="world", objects=spatial_objects)
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
            print(f"  [frame {fid}] ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nSaved: {saved}/{len(frame_map)} nodes in {elapsed:.1f}s "
          f"({saved/elapsed:.1f} fps)" if elapsed > 0 else "")


async def _main_video_mode(
    args: argparse.Namespace,
    session_dir: Path,
    frame_map: Dict[int, Dict[str, Any]],
    objects_gt: Dict[str, Any],
    camera_params: CameraParams,
    svc: MemoryService,
) -> None:
    """Video-clip mode: segment frames into time-based clips, one MemoryNode per clip."""
    videos_csv_path = session_dir / "videos_index.csv"

    # Load video references from videos_index.csv
    video_refs: List[str] = []
    if videos_csv_path.exists():
        vrows = load_csv(str(videos_csv_path))
        video_refs = [r["file_path"] for r in vrows if r.get("file_path")]
    if not video_refs:
        print("WARNING: no video files found in videos_index.csv, using empty video_clip_refs")
        video_refs = []

    print(f"Video files: {len(video_refs)}")
    for vf in video_refs:
        vp = session_dir / vf
        print(f"  {vf} ({vp.stat().st_size / 1e6:.1f} MB)" if vp.exists() else f"  {vf} (NOT FOUND)")

    # Build clips
    clips = build_video_clips(
        frame_map, video_refs,
        args.clip_duration_sec, args.thumbnail_interval,
    )
    print(f"Clips: {len(clips)} (duration={args.clip_duration_sec}s each)")

    t0 = time.time()
    saved = 0

    for clip in clips:
        clip_id = clip["clip_id"]
        start_ts = clip["start_ts"]
        end_ts = clip["end_ts"]

        # Build spatial objects from first frame + matched objects
        first_fid = clip["frame_ids"][0]
        fm = frame_map[first_fid]

        spatial_objects = [ObjectCoord(
            obj_id=f"clip_{clip_id:03d}_cam",
            label="camera_frame",
            x=fm["cam_x"], y=fm["cam_y"], z=fm["cam_z"],
        )]
        clip_objs = match_objects_to_clip(objects_gt, start_ts, end_ts)
        spatial_objects.extend(clip_objs)

        # Optional thumbnail: encode one representative frame
        img_b64 = ""
        depth_b64 = ""
        thumb_fid = clip.get("thumbnail_fid")
        if thumb_fid is not None and thumb_fid in frame_map:
            t_fm = frame_map[thumb_fid]
            img_path = session_dir / t_fm["image_path"]
            if img_path.exists():
                img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            depth_path_str = t_fm.get("depth_path", "")
            if depth_path_str:
                dp = session_dir / depth_path_str
                if dp.exists():
                    depth_b64 = base64.b64encode(dp.read_bytes()).decode()

        obj_labels = list(set(o.label for o in spatial_objects if o.label != "camera_frame"))
        objects_str = ", ".join(sorted(obj_labels)[:30])
        frame_count = len(clip["frame_ids"])
        duration_s = (end_ts - start_ts) / 1e9

        spatial = SpatialContext(origin="world", objects=spatial_objects)
        req = RememberRequest(
            session_id=f"scenes2-{session_dir.name}",
            plan_id=f"load-clip-{clip_id:03d}",
            log_record=LogRecord(
                ts=start_ts, level="Info", tag="scenes2",
                msg=(f"video clip {clip_id}: {frame_count} frames, {duration_s:.1f}s, "
                     f"{len(spatial_objects)-1} objects: [{objects_str}]"),
            ),
            spatial=spatial,
            image_base64=img_b64,           # thumbnail only (optional)
            depth_base64=depth_b64,         # thumbnail depth (optional)
            camera_params=camera_params,
            time_range=TimeRange(start_ts=start_ts, end_ts=end_ts),
            kv={
                "frame_ts": ",".join(str(ts) for ts in clip["frame_ts"][:50]),
                "video_clip_refs": ",".join(video_refs),
                "objects": objects_str,
                "frame_count": str(frame_count),
                "clip_duration_sec": f"{duration_s:.1f}",
            },
        )
        try:
            resp = await svc._remember_pipe.execute(req)
            if resp.node_id >= 0:
                saved += 1
        except Exception as e:
            print(f"  [clip {clip_id}] ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nSaved: {saved}/{len(clips)} clips in {elapsed:.1f}s "
          f"({saved/elapsed:.1f} clips/sec)" if elapsed > 0 else "")


if __name__ == "__main__":
    asyncio.run(main())
