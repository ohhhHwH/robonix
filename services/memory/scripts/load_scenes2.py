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
    p.add_argument("--mode", choices=["frame", "video", "embodied"], default="frame",
                   help="frame: one node per image (default); video: one node per video clip; "
                   "embodied: simulate ObjectWatchdog FOV filtering + spatial dedup")
    p.add_argument("--clip-duration-sec", type=float, default=5.0,
                   help="Video mode: seconds per clip segment (default 5.0)")
    p.add_argument("--thumbnail-interval", type=int, default=15,
                   help="Video mode: encode 1 thumbnail image every N frames (0=none)")
    p.add_argument("--embodied-cooldown-frames", type=int, default=30,
                   help="Embodied mode: min frames between angle appends for same object "
                   "(default 30 ≈ 2s at 15fps)")
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


# ═══════════════════════════════════════════════════════════════════════════
# Embodied-mode helpers: replicate ObjectWatchdog perception pipeline.
# See system/scene/scene_service/object_watchdog.py for the reference impl.
# ═══════════════════════════════════════════════════════════════════════════

def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw angle from quaternion (pure z-rotation assumption)."""
    import numpy as np
    return float(2.0 * np.arctan2(float(qz), float(qw)))


def _world_to_pixel(
    obj_x: float, obj_y: float, obj_z: float,
    cam_x: float, cam_y: float, cam_z: float,
    cam_yaw: float,
    camera_params,
):
    """Project 3D world point to 2D pixel using pinhole model.

    Matches ObjectWatchdog._project_to_pixel() Path B exactly:
      1. World-frame delta from camera to object
      2. Rotate by -yaw → body-frame coordinates
      3. Body → ROS optical frame (right, down, forward)
      4. Pinhole projection with camera intrinsics

    Returns:
        (u, v, depth) or (None, None, None) if behind camera.
    """
    import numpy as np

    dwx = obj_x - cam_x
    dwy = obj_y - cam_y
    dwz = obj_z - cam_z

    # Rotate by -yaw → body-frame (forward, left, up)
    cos_y = np.cos(cam_yaw)
    sin_y = np.sin(cam_yaw)
    bx = dwx * cos_y + dwy * sin_y   # body x (forward)
    by = -dwx * sin_y + dwy * cos_y  # body y (left)

    # Body → ROS optical frame: (right, down, forward)
    ox = -by        # optical x (right)
    oy = -dwz       # optical y (down)
    oz_val = bx     # optical z (forward / depth)

    if oz_val < 0.01:
        return None, None, None

    u = int(camera_params.fx * ox / oz_val + camera_params.cx)
    v = int(camera_params.fy * oy / oz_val + camera_params.cy)
    return u, v, oz_val


def _is_duplicate_by_class(
    cls: str,
    ox: float, oy: float,
    seen_positions: Dict[str, List[float]],
    radius: float,
) -> bool:
    """Check whether (ox, oy) is within *radius* of any previously-seen
    position for *cls*, using actual world-frame positions.

    Matches ObjectWatchdog._is_duplicate().
    """
    positions = seen_positions.get(cls)
    if not positions:
        return False
    r2 = radius ** 2
    for i in range(0, len(positions), 2):
        sx, sy = positions[i], positions[i + 1]
        if (ox - sx) ** 2 + (oy - sy) ** 2 <= r2:
            return True
    return False


# ── Dedup radii from ObjectWatchdog._DEDUP_RADII ──
_EMBODIED_DEDUP_RADII: Dict[str, float] = {
    "cabinet": 3.0, "shelf": 3.0, "table": 3.0, "desk": 3.0,
    "couch": 3.0, "sofa": 3.0, "chair": 2.5, "bed": 3.0,
    "door": 3.0, "window": 3.0, "refrigerator": 3.0,
    "monitor": 2.0, "tv": 2.0, "picture_frame": 2.0,
    "lamp": 2.0, "plant": 1.5, "potted_plant": 1.5,
    "keyboard": 1.5, "mouse": 1.0, "cup": 1.0, "bottle": 1.0,
    "mug": 1.0, "water_bottle": 1.0,
}
_EMBODIED_DEFAULT_RADIUS = 3.0
_EMBODIED_MAX_IMAGES = 3


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
    elif args.mode == "embodied":
        await _main_embodied_mode(args, session_dir, frame_map, objects_gt,
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


async def _main_embodied_mode(
    args: argparse.Namespace,
    session_dir: Path,
    frame_map: Dict[int, Dict[str, Any]],
    objects_gt: Dict[str, Any],
    camera_params: CameraParams,
    svc: MemoryService,
) -> None:
    """Embodied mode: simulate ObjectWatchdog perception pipeline.

    For each frame (tick):
      1. FOV filter: project objects to camera pixel space
      2. Spatial dedup: same-class objects within DEDUP_RADII → same identity
      3. New objects → create MemoryNode with current frame image
      4. Seen objects from new angle → append child node via parent_node_id
         (max 3 images per object, cooldown between appends).

    This is the closest approximation to how a real robot running
    ObjectWatchdog would build its memory — object-centric observations
    filtered through the camera frustum, with spatial deduplication.
    """
    import numpy as np

    # ── Load video references for video_clip_refs ──
    videos_csv_path = session_dir / "videos_index.csv"
    video_entries: List[Dict[str, Any]] = []
    if videos_csv_path.exists():
        vrows = load_csv(str(videos_csv_path))
        # Only keep RGB videos (not depth) for clip references
        video_entries = [r for r in vrows if r.get("camera_type", "") == "rgb"]
    if video_entries:
        print(f"Video refs: {len(video_entries)} RGB video(s)")
    else:
        print("WARNING: no RGB video entries in videos_index.csv — "
              "video_clip_refs will be empty")

    def _compute_video_refs(ts_ns: int) -> str:
        """Compute video_clip_refs string for a given timestamp.

        Returns a comma-separated list of "path#frame=N" references so the
        retrieve pipeline can extract keyframes via ffmpeg.
        """
        refs: List[str] = []
        for ve in video_entries:
            start_ts = int(ve.get("start_ts", 0))
            end_ts = int(ve.get("end_ts", 0))
            start_fid = int(ve.get("start_frame_id", 0))
            fps = float(ve.get("fps", 15.0))
            # Only include if timestamp falls within video range
            if start_ts <= ts_ns < end_ts:
                frame_offset = start_fid + int((ts_ns - start_ts) / 1e9 * fps)
                refs.append(f"{ve['file_path']}#frame={frame_offset}")
            elif ts_ns < start_ts:
                # Frame before video starts → use first frame
                refs.append(f"{ve['file_path']}#frame={start_fid}")
            else:
                # Frame after video ends → use last frame
                end_fid = int(ve.get("end_frame_id", 0))
                refs.append(f"{ve['file_path']}#frame={end_fid}")
        return ",".join(refs) if refs else ""

    dedup_radii = _EMBODIED_DEDUP_RADII
    default_radius = _EMBODIED_DEFAULT_RADIUS
    max_images = _EMBODIED_MAX_IMAGES
    cooldown = args.embodied_cooldown_frames

    # ── Per-tick state (mirrors ObjectWatchdog instance vars) ──
    seen_positions: Dict[str, List[float]] = {}  # cls → [x1,y1, x2,y2, …]
    grid_node: Dict[str, int] = {}               # "cls@gx,gy" → parent node_id
    grid_img_count: Dict[str, int] = {}           # "cls@gx,gy" → image count
    grid_last_frame: Dict[str, int] = {}           # "cls@gx,gy" → last frame idx

    t0 = time.time()
    saved_new = 0
    saved_append = 0
    total_checks = 0
    total_visible = 0

    sorted_fids = sorted(frame_map.keys())

    for frame_idx, fid in enumerate(sorted_fids):
        fm = frame_map[fid]
        ts = fm["ts"]
        cam_x, cam_y, cam_z = fm["cam_x"], fm["cam_y"], fm["cam_z"]
        qx, qy, qz, qw = fm["cam_qx"], fm["cam_qy"], fm["cam_qz"], fm["cam_qw"]
        cam_yaw = _yaw_from_quat(qx, qy, qz, qw)

        # ── Stage 1: FOV filter ──
        visible: list = []  # (obj_id, obj_dict, ox, oy, oz, u, v, depth)
        for obj_id, obj in objects_gt.items():
            pos = obj.get("position", {})
            ox = float(pos.get("x", 0))
            oy = float(pos.get("y", 0))
            oz = float(pos.get("z", 0))
            total_checks += 1

            u, v, depth = _world_to_pixel(
                ox, oy, oz, cam_x, cam_y, cam_z, cam_yaw, camera_params,
            )
            if u is not None and depth is not None and depth > 0.01:
                if 0 <= u < camera_params.width and 0 <= v < camera_params.height:
                    visible.append((obj_id, obj, ox, oy, oz, u, v, depth))

        total_visible += len(visible)
        if not visible:
            continue

        # ── Stage 2: Classify — new (not spatially deduped) vs append ──
        new_objects: list = []       # (obj_id, obj, ox, oy, oz, cls)
        append_candidates: list = []  # (grid_key, obj_id, obj, ox, oy, oz)

        for obj_id, obj, ox, oy, oz, u, v, depth in visible:
            cls = obj.get("label_en", obj.get("label_zh", "unknown"))
            radius = dedup_radii.get(cls, default_radius)
            grid_key = f"{cls}@{round(ox)},{round(oy)}"

            if _is_duplicate_by_class(cls, ox, oy, seen_positions, radius):
                cnt = grid_img_count.get(grid_key, 0)
                last_f = grid_last_frame.get(grid_key, -999)
                if 0 < cnt < max_images and (frame_idx - last_f) >= cooldown:
                    append_candidates.append((grid_key, obj_id, obj, ox, oy, oz))
            else:
                new_objects.append((obj_id, obj, ox, oy, oz, cls))
                seen_positions.setdefault(cls, []).extend([ox, oy])

        if not new_objects and not append_candidates:
            continue

        # ── Load frame image (one capture reused for the whole batch) ──
        img_b64 = ""
        img_path = session_dir / fm["image_path"]
        if img_path.exists():
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
        if not img_b64:
            continue

        # ── Stage 3: Save genuinely-new objects ──
        for obj_id, obj, ox, oy, oz, cls in new_objects:
            grid_key = f"{cls}@{round(ox)},{round(oy)}"

            spatial = SpatialContext(origin="world", objects=[
                ObjectCoord(obj_id=obj_id, label=cls, x=ox, y=oy, z=oz),
            ])
            req = RememberRequest(
                session_id=f"scenes2-{session_dir.name}",
                plan_id=f"load-embodied-{fid:06d}",
                log_record=LogRecord(
                    ts=ts, level="Info", tag="scenes2",
                    msg=f"observed new object: {cls} at ({ox:.1f},{oy:.1f},{oz:.1f})",
                ),
                spatial=spatial,
                image_base64=img_b64,
                camera_params=camera_params,
                time_range=TimeRange(start_ts=ts, end_ts=ts),
                kv={
                    "objects": cls,
                    "object_id": obj_id,
                    "frame_ts": str(ts),
                    "video_clip_refs": _compute_video_refs(ts),
                },
            )
            try:
                resp = await svc._remember_pipe.execute(req)
                if resp.node_id >= 0:
                    grid_node[grid_key] = resp.node_id
                    grid_img_count[grid_key] = 1
                    grid_last_frame[grid_key] = frame_idx
                    saved_new += 1
            except Exception as e:
                print(f"  [new {cls} fid={fid}] ERROR: {e}")

        # ── Stage 4: Append new viewing angles for known objects ──
        for grid_key, obj_id, obj, ox, oy, oz in append_candidates:
            parent_id = grid_node.get(grid_key)
            if parent_id is None:
                continue

            cls = obj.get("label_en", obj.get("label_zh", "unknown"))
            spatial = SpatialContext(origin="world", objects=[
                ObjectCoord(obj_id=obj_id, label=cls, x=ox, y=oy, z=oz),
            ])
            req = RememberRequest(
                session_id=f"scenes2-{session_dir.name}",
                plan_id=f"load-embodied-append-{fid:06d}",
                log_record=LogRecord(
                    ts=ts, level="Info", tag="scenes2",
                    msg=f"observed {cls} from another angle "
                    f"(img {grid_img_count.get(grid_key,0)+1}/{max_images})",
                ),
                spatial=spatial,
                image_base64=img_b64,
                camera_params=camera_params,
                time_range=TimeRange(start_ts=ts, end_ts=ts),
                parent_node_id=parent_id,
                kv={
                    "objects": cls,
                    "object_id": obj_id,
                    "frame_ts": str(ts),
                    "video_clip_refs": _compute_video_refs(ts),
                },
            )
            try:
                resp = await svc._remember_pipe.execute(req)
                if resp.node_id >= 0:
                    grid_img_count[grid_key] = grid_img_count.get(grid_key, 0) + 1
                    grid_last_frame[grid_key] = frame_idx
                    saved_append += 1
            except Exception as e:
                print(f"  [append {grid_key} fid={fid}] ERROR: {e}")

    # ── Report ──
    elapsed = time.time() - t0
    total_nodes = saved_new + saved_append
    unique = len(grid_node)
    print(f"\nEmbodied mode results:")
    print(f"  Frames processed:           {len(sorted_fids)}")
    print(f"  Total object FOV checks:    {total_checks}")
    print(f"  Objects in view (total):    {total_visible}")
    print(f"  New objects saved:          {saved_new}")
    print(f"  Angle appends saved:        {saved_append}")
    print(f"  Unique objects tracked:     {unique}")
    print(f"  Total nodes created:        {total_nodes}")
    print(f"  Time:                       {elapsed:.1f}s "
          f"({len(sorted_fids)/elapsed:.1f} fps)" if elapsed > 0 else "")

    if unique:
        print(f"  Avg images per object:      {(saved_new+saved_append)/unique:.1f}")

    # Per-class breakdown
    cls_nodes: Dict[str, int] = {}
    cls_imgs: Dict[str, int] = {}
    for gk, cnt in grid_img_count.items():
        c = gk.split("@")[0]
        cls_nodes[c] = cls_nodes.get(c, 0) + 1
        cls_imgs[c] = cls_imgs.get(c, 0) + cnt
    print(f"  Object nodes by class:")
    for c in sorted(cls_nodes):
        print(f"    {c}: {cls_nodes[c]} node(s), {cls_imgs[c]} total image(s)")


if __name__ == "__main__":
    asyncio.run(main())
