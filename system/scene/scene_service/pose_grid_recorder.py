# SPDX-License-Identifier: MulanPSL-2.0
"""Pose-grid image recorder — grid+angle-keyed memory snapshots.

Runs as a background asyncio task inside the Scene service.  At each tick
it reads the robot's map-frame pose from the hub (``pose`` = the
``robonix/service/map/pose`` PoseWithCovarianceStamped contract) and,
whenever the robot crosses into a new spatial grid cell OR its heading
rotates past an angle bucket boundary, captures the current RGB camera
frame and stores it in memgraph as a ``place`` MemoryNode keyed by grid
cell.

  - Same grid cell → same MemoryNode (images accumulate under it).
  - One image per (cell, heading-bucket): rotating 360° in place yields
    at most ``ceil(360 / angle_deg)`` images for that cell, and re-entering
    an already-captured (cell, bucket) does nothing — so duplicates are
    bounded by construction.
  - The previous (position, heading) are tracked internally to detect the
    grid/angle crossings.

Env vars:
  ``SCENE_POSE_GRID`` — set to ``"0"`` to disable (default ``"1"``).
  ``POSE_GRID_SIZE_M`` — spatial grid cell size in metres (default ``2.0``).
  ``POSE_GRID_ANGLE_DEG`` — heading bucket size in degrees (default ``90.0``).
  ``POSE_GRID_MIN_INTERVAL_S`` — min wall-clock between captures (default ``0.5``).
  ``POSE_GRID_POLL_INTERVAL_S`` — pose poll period (default ``0.25``).
  ``POSE_GRID_POSE_MAX_AGE_S`` — reject pose samples older than this (default ``1.0``).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

log = logging.getLogger(__name__)

# Set MEMGRAPH_HOOK_URL to the host-visible memgraph Scene Hook endpoint.
# Default works with --network host.
_MEMGRAPH_HOOK_URL = os.environ.get(
    "MEMGRAPH_HOOK_URL",
    "http://127.0.0.1:37798",
)

_GRID_SIZE_M = float(os.environ.get("POSE_GRID_SIZE_M", "2.0"))
_ANGLE_DEG = float(os.environ.get("POSE_GRID_ANGLE_DEG", "90.0"))
_MIN_INTERVAL_S = float(os.environ.get("POSE_GRID_MIN_INTERVAL_S", "0.5"))
_POLL_INTERVAL_S = float(os.environ.get("POSE_GRID_POLL_INTERVAL_S", "0.25"))
_POSE_MAX_AGE_S = float(os.environ.get("POSE_GRID_POSE_MAX_AGE_S", "1.0"))

_JPEG_QUALITY = 85


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (radians) from a quaternion, matching scene's ``_quat_to_yaw``."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class PoseGridRecorder:
    """Grid+angle image capture that persists to grid-keyed memory nodes.

    Usage::

        rec = PoseGridRecorder(hub=hub, memgraph_url=...)
        asyncio.create_task(rec.run())
    """

    def __init__(
        self,
        *,
        hub,  # SubscribersHub (for .latest("pose") / .latest("rgb"))
        memgraph_url: str = _MEMGRAPH_HOOK_URL,
        grid_size_m: float = _GRID_SIZE_M,
        angle_deg: float = _ANGLE_DEG,
    ) -> None:
        self._hub = hub
        self._url = memgraph_url
        self._grid_size_m = grid_size_m
        self._angle_rad = math.radians(angle_deg)
        self._min_interval_s = _MIN_INTERVAL_S
        self._poll_interval_s = _POLL_INTERVAL_S
        self._pose_max_age_s = _POSE_MAX_AGE_S

        # Grid cell (gx, gy) → set of heading buckets already captured.
        # A re-entered (cell, bucket) is skipped, so each cell yields at
        # most ceil(360 / angle_deg) images total (one per bucket).
        self._cell_buckets: Dict[Tuple[int, int], set] = {}
        # Grid cell (gx, gy) → node_id.  Same cell reuses its node.
        self._cell_node: Dict[Tuple[int, int], int] = {}
        self._last_capture_ts: float = 0.0
        self._running = False

    # ── lifecycle ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Blocking async entrypoint — use ``asyncio.create_task(rec.run())``."""
        if self._running:
            return
        self._running = True
        log.info(
            "pose_grid: started (grid=%.1fm, angle=%.0f°, poll=%.2fs)",
            self._grid_size_m, math.degrees(self._angle_rad),
            self._poll_interval_s,
        )
        while self._running:
            try:
                pose = self._read_pose()
                if pose is not None:
                    await self._maybe_capture(*pose)
            except Exception:
                log.debug("pose_grid: tick error", exc_info=True)
            await asyncio.sleep(self._poll_interval_s)

    def stop(self) -> None:
        """Signal the loop to exit at the next sleep boundary."""
        self._running = False

    # ── pose / capture sources ─────────────────────────────────────────

    def _read_pose(self) -> Optional[Tuple[float, float, float, float, str]]:
        """Read the robot's map-frame pose from the hub.

        Returns ``(x, y, z, yaw, frame_id)`` or ``None`` when no fresh
        map-frame pose is available.  Uses only the ``pose`` contract
        (SLAM-corrected map frame); odometry is deliberately not used for
        grid keying because it drifts in an odom-local frame.
        """
        if self._hub is None or not self._hub.has("pose"):
            return None
        msg, stamp_unix, _count = self._hub.latest("pose")
        if msg is None or stamp_unix <= 0 or time.time() - stamp_unix > self._pose_max_age_s:
            return None
        p = (
            msg.pose.pose
            if hasattr(msg, "pose") and hasattr(msg.pose, "pose")
            else msg.pose
        )
        q = p.orientation
        x = float(p.position.x)
        y = float(p.position.y)
        z = float(p.position.z)
        yaw = _quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        frame_id = getattr(getattr(msg, "header", None), "frame_id", None) or "map"
        return (x, y, z, yaw, frame_id)

    def _capture_encode(self) -> str:
        """Capture the latest RGB frame and return a base64 JPEG string.

        Returns ``""`` on failure.  Runs in executor thread (blocking cv2).
        """
        if self._hub is None or not self._hub.has("rgb"):
            return ""
        rgb_msg, _stamp, _count = self._hub.latest("rgb")
        if rgb_msg is None or cv2 is None:
            return ""
        try:
            raw = bytes(rgb_msg.data)
            h, w = rgb_msg.height, rgb_msg.width
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, -1)
            if rgb_msg.encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, jpg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:
                return ""
            return base64.b64encode(jpg.tobytes()).decode("ascii")
        except Exception:
            log.warning("pose_grid: frame capture/encode failed", exc_info=True)
            return ""

    # ── threshold logic ────────────────────────────────────────────────

    @staticmethod
    def _grid_cell(x: float, y: float, grid_size_m: float) -> Tuple[int, int]:
        """Map-frame (x, y) → grid cell indices, flooring negative coords."""
        return (math.floor(x / grid_size_m), math.floor(y / grid_size_m))

    @staticmethod
    def _yaw_bucket(yaw: float, angle_rad: float) -> int:
        """Normalise yaw to [0, 2π) and bucket it by *angle_rad*."""
        norm = yaw % (2.0 * math.pi)
        return int(math.floor(norm / angle_rad))

    async def _maybe_capture(self, x, y, z, yaw, frame_id) -> None:
        """Capture a frame when the robot enters a new (cell, bucket).

        Buckets already captured for a cell are skipped, so a 360° in-place
        rotation yields at most ``ceil(360 / angle_deg)`` images for that
        cell even if the heading oscillates across bucket boundaries.
        """
        gx, gy = self._grid_cell(x, y, self._grid_size_m)
        bucket = self._yaw_bucket(yaw, self._angle_rad)

        if bucket in self._cell_buckets.get((gx, gy), ()):
            return  # already captured this (cell, bucket)

        now = time.monotonic()
        if now - self._last_capture_ts < self._min_interval_s:
            return  # jitter guard — re-check on a later tick

        loop = asyncio.get_running_loop()
        img_b64 = await loop.run_in_executor(None, self._capture_encode)
        if not img_b64:
            return  # no frame yet; retry next tick (state not committed)

        node_id = self._cell_node.get((gx, gy))
        if node_id is not None:
            ok = await self._post_append(node_id, img_b64, gx, gy, bucket)
            if ok:
                self._cell_buckets.setdefault((gx, gy), set()).add(bucket)
                self._last_capture_ts = now
                log.info("pose_grid: appended image to node %d (cell %d,%d bucket %d)",
                         node_id, gx, gy, bucket)
        else:
            node_id = await self._post_create(
                x, y, z, yaw, frame_id, img_b64, gx, gy, bucket,
            )
            if node_id is not None:
                self._cell_node[(gx, gy)] = node_id
                self._cell_buckets.setdefault((gx, gy), set()).add(bucket)
                self._last_capture_ts = now
                log.info("pose_grid: created node %d (cell %d,%d bucket %d)",
                         node_id, gx, gy, bucket)

    # ── memgraph POST ──────────────────────────────────────────────────

    def _camera_pose(self, x, y, z, yaw) -> Dict[str, float]:
        """Robot pose as a camera_pose dict (yaw → unit quaternion about Z)."""
        half = yaw / 2.0
        return {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "qx": 0.0,
            "qy": 0.0,
            "qz": math.sin(half),
            "qw": math.cos(half),
        }

    async def _post_create(self, x, y, z, yaw, frame_id, img_b64,
                           gx, gy, bucket) -> Optional[int]:
        """Create a new grid-keyed ``place`` node and return its node_id."""
        import httpx

        now_ns = time.time_ns()
        cx = (gx + 0.5) * self._grid_size_m
        cy = (gy + 0.5) * self._grid_size_m
        summary = (
            f"grid cell ({gx},{gy}) center=({cx:.1f},{cy:.1f}) "
            f"heading bucket {bucket}"
        )
        payload: Dict[str, Any] = {
            "session_id": "scene-pose-grid",
            "plan_id": "scene-pose-grid",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "pose_grid",
                "msg": summary,
            },
            "spatial": {
                "origin": frame_id or "map",
                "center": {"x": cx, "y": cy, "z": 0.0},
                "radius_m": self._grid_size_m,
                "semantic_region": f"grid:{gx},{gy}",
                "objects": [],
            },
            "image_base64": img_b64,
            "camera_params": {
                "camera_pose": self._camera_pose(x, y, z, yaw),
            },
            "time_range": {"start_ts": now_ns, "end_ts": now_ns},
            "kv": {
                "node_type": "place",
                "grid_key": f"{gx},{gy}",
                "grid_cell": [gx, gy],
                "yaw_bucket": bucket,
                "pose_grid": "true",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self._url, json=payload)
            if r.status_code >= 400:
                log.warning("pose_grid: create POST returned %d: %s",
                            r.status_code, r.text[:200])
                return None
            nid = r.json().get("node_id")
            return int(nid) if isinstance(nid, int) else None
        except Exception as e:
            log.debug("pose_grid: create POST failed: %s", e)
            return None

    async def _post_append(self, node_id, img_b64, gx, gy, bucket) -> bool:
        """Append a new-angle frame to the grid cell's existing node."""
        import httpx

        now_ns = time.time_ns()
        payload: Dict[str, Any] = {
            "session_id": "scene-pose-grid",
            "plan_id": "scene-pose-grid",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "pose_grid",
                "msg": f"grid cell ({gx},{gy}) new heading bucket {bucket}",
            },
            "parent_node_id": node_id,
            "image_base64": img_b64,
            "kv": {"append_image_ref": True},
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self._url, json=payload)
            if r.status_code >= 400:
                log.debug("pose_grid: append POST for node %d returned %d: %s",
                          node_id, r.status_code, r.text[:200])
                return False
            return True
        except Exception as e:
            log.debug("pose_grid: append POST for node %d failed: %s", node_id, e)
            return False
