# SPDX-License-Identifier: MulanPSL-2.0
"""Object-level watchdog: save a memory snapshot for every newly-seen object.

Runs as a background asyncio task inside the Scene service.  At each tick
it polls ``ObjectRegistry.snapshot()``, diffs against previously-seen
object IDs, and for every genuinely new object:

  1. captures the latest RGB camera frame (JPEG-encoded)
  2. builds a ``remember`` request payload (one object, one image)
  3. HTTP POSTs to memgraph's Scene Hook endpoint on port 37798

This is separate from the existing Scene Hook in ``mcp_tools.py``
(which triggers on ``scene.list_objects`` MCP calls and saves the
*whole* scene).  The watchdog runs autonomously and does not require
Pilot to call ``list_objects`` — it is designed for the patrol /
exploration use-case where the robot moves and new objects appear in
view continuously.

The polling interval adapts to robot motion state detected from
odometry / IMU data (via ``SubscribersHub``) or set explicitly via
``set_motion_state()``:

  ==============  ========  ====
  moving          >0.2 m/s   2 s
  rotating        >10 °/s    5 s
  static          —         30 s
  static_long     >5 min   120 s
  ==============  ========  ====

Env vars:
  ``SCENE_OBJECT_WATCHDOG`` — set to ``"0"`` to disable (default ``"1"``).
  ``OBJECT_WATCHDOG_INTERVAL_MOVING_S`` — moving interval (default 2.0).
  ``OBJECT_WATCHDOG_INTERVAL_ROTATING_S`` — rotating interval (default 5.0).
  ``OBJECT_WATCHDOG_INTERVAL_STATIC_S`` — static interval (default 30.0).
  ``OBJECT_WATCHDOG_INTERVAL_STATIC_LONG_S`` — long-static interval (default 120.0).
  ``OBJECT_WATCHDOG_STATIC_LONG_THRESHOLD_S`` — long-static threshold (default 300.0).
  ``OBJECT_WATCHDOG_INTERVAL_S`` — fallback / legacy interval (default 2.0).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

log = logging.getLogger(__name__)

# Set MEMGRAPH_HOOK_URL to the host-visible memgraph Scene Hook
# endpoint.  Default works with --network host; for plain Docker
# use "http://172.17.0.1:37798" (bridge gateway).
_MEMGRAPH_HOOK_URL = os.environ.get(
    "MEMGRAPH_HOOK_URL",
    "http://127.0.0.1:37798",
)
_DEFAULT_INTERVAL_S = 2.0

# ── Dynamic interval env vars ──────────────────────────────────────────

_INTERVAL_MOVING_S = float(os.environ.get(
    "OBJECT_WATCHDOG_INTERVAL_MOVING_S", "2.0"))
_INTERVAL_ROTATING_S = float(os.environ.get(
    "OBJECT_WATCHDOG_INTERVAL_ROTATING_S", "5.0"))
_INTERVAL_STATIC_S = float(os.environ.get(
    "OBJECT_WATCHDOG_INTERVAL_STATIC_S", "30.0"))
_INTERVAL_STATIC_LONG_S = float(os.environ.get(
    "OBJECT_WATCHDOG_INTERVAL_STATIC_LONG_S", "120.0"))
_STATIC_LONG_THRESHOLD_S = float(os.environ.get(
    "OBJECT_WATCHDOG_STATIC_LONG_THRESHOLD_S", "300.0"))

# Motion state names
_MOVING = "moving"
_ROTATING = "rotating"
_STATIC = "static"
_STATIC_LONG = "static_long"

# Motion detection thresholds
_MOVING_LINEAR_VEL_THRESH = 0.2       # m/s
_ROTATING_ANGULAR_VEL_THRESH = 10.0   # deg/s, converted to rad/s below
_ROTATING_ANGULAR_VEL_THRESH_RAD = _ROTATING_ANGULAR_VEL_THRESH * (math.pi / 180.0)
# Linear vel below this AND angular vel below → static
_STATIC_LINEAR_VEL_MAX = 0.2
_STATIC_ANGULAR_VEL_MAX_RAD = 10.0 * (math.pi / 180.0)


def _compute_interval(motion_state: str) -> float:
    """Return the polling interval for a given motion state."""
    return {
        _MOVING: _INTERVAL_MOVING_S,
        _ROTATING: _INTERVAL_ROTATING_S,
        _STATIC: _INTERVAL_STATIC_S,
        _STATIC_LONG: _INTERVAL_STATIC_LONG_S,
    }.get(motion_state, _DEFAULT_INTERVAL_S)


class ObjectWatchdog:
    """Background observer that detects new objects and saves images.

    One object → one image (the full camera frame at the moment the
    object was first seen).  If multiple new objects appear in the
    same tick a single frame is captured and reused for efficiency;
    each object still gets its own MemoryNode.
    """

    def __init__(
        self,
        *,
        registry,          # ObjectRegistry
        hub,               # SubscribersHub (for .latest("rgb"))
        memgraph_url: str = _MEMGRAPH_HOOK_URL,
        interval_s: float = 0.0,
        anno_store = None, # Optional AnnotationStore for room lookup
    ) -> None:
        self._registry = registry
        self._hub = hub
        self._memgraph_url = memgraph_url
        self._anno_store = anno_store  # AnnotationStore or None
        self._fallback_interval = (
            interval_s
            if interval_s > 0
            else float(os.environ.get("OBJECT_WATCHDOG_INTERVAL_S", _DEFAULT_INTERVAL_S))
        )
        # Dedup by (cls, x, y) — physical identity, not Scene's
        # transient object_id which changes across tracking re-acquisitions.
        # Maps class → list of (x, y) world-frame positions.
        self._seen_objects: Dict[str, List[float]] = {}  # class → [x1,y1, x2,y2, ...]
        # How many images per object before we stop collecting.
        self._max_images_per_object = int(
            os.environ.get("OBJECT_WATCHDOG_MAX_IMAGES", "3")
        )
        # (cls, grid_x, grid_y) → node_id mapping for multi-angle append.
        self._grid_node: Dict[str, int] = {}
        # (cls, grid_x, grid_y) → image count for that node.
        self._grid_img_count: Dict[str, int] = {}
        self._task: asyncio.Task | None = None
        self._running = False

        # ── Motion state (dynamic interval) ──
        self._motion_state: str = _STATIC  # current state: moving/rotating/static/static_long
        self._static_start: Optional[float] = None  # when we entered static state
        self._motion_override: Optional[str] = None  # set via set_motion_state()
        # Odometry subscription name to poll from hub.
        self._odom_topic: str = os.environ.get(
            "OBJECT_WATCHDOG_ODOM_TOPIC", "/odom",
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def set_motion_state(self, state: str) -> None:
        """Override motion state from external source (e.g. Pilot / PathRecorder).

        Valid states: ``"moving"``, ``"rotating"``, ``"static"``.
        Set to ``None`` or ``""`` to resume auto-detection from odometry.
        """
        if state in (None, ""):
            self._motion_override = None
        elif state in (_MOVING, _ROTATING, _STATIC):
            self._motion_override = state
        else:
            log.warning("object_watchdog: unknown motion state %r — ignored", state)

    async def run(self) -> None:
        """Blocking async entrypoint — use ``asyncio.create_task(wd.run())``."""
        if self._running:
            return
        self._running = True

        # Seed the seen-objects map from whatever the registry already
        # holds so the first tick only reacts to genuinely new arrivals.
        try:
            objs, _surfs = await self._registry.snapshot()
            for o in objs.values():
                if not o.missing:
                    self._seen_objects.setdefault(o.cls, []).extend(
                        [float(o.pose.x), float(o.pose.y)]
                    )
            total = sum(len(v) // 2 for v in self._seen_objects.values())
        except Exception:
            log.debug("object_watchdog: initial snapshot failed", exc_info=True)
            self._seen_objects = {}
            total = 0

        log.info(
            "object_watchdog: started (dynamic intervals: "
            "moving=%.1fs, rotating=%.1fs, static=%.1fs, static_long=%.1fs, "
            "threshold=%.0fs, %d known, max_images=%d, dedup=per-class)",
            _INTERVAL_MOVING_S, _INTERVAL_ROTATING_S,
            _INTERVAL_STATIC_S, _INTERVAL_STATIC_LONG_S,
            _STATIC_LONG_THRESHOLD_S,
            total, self._max_images_per_object,
        )

        while self._running:
            start = time.monotonic()
            try:
                await self._tick()
            except Exception:
                log.debug("object_watchdog: tick error", exc_info=True)

            # ── Dynamic interval: detect motion → choose interval ──
            interval = self._select_interval()
            elapsed = time.monotonic() - start
            sleep_s = max(0.05, interval - elapsed)
            await asyncio.sleep(sleep_s)

    def stop(self) -> None:
        """Signal the loop to exit at the next sleep boundary."""
        self._running = False
        if self._task is not None:
            self._task.cancel()

    # ── dynamic interval ──────────────────────────────────────────────

    def _select_interval(self) -> float:
        """Detect motion state and return the appropriate interval."""
        # 1. External override
        if self._motion_override is not None:
            self._motion_state = self._motion_override
        else:
            # 2. Auto-detect from odometry
            self._motion_state = self._detect_motion()

        return _compute_interval(self._motion_state)

    def _detect_motion(self) -> str:
        """Detect robot motion state from hub odometry data.

        Falls back to ``_STATIC`` if odometry is unavailable.
        """
        linear_vel = 0.0
        angular_vel = 0.0

        # Try to read odometry from hub
        if self._hub is not None:
            for topic in (self._odom_topic, "/odom", "odom"):
                if self._hub.has(topic):
                    msg, _stamp, _count = self._hub.latest(topic)
                    if msg is not None:
                        # Try common odometry field names
                        twist = getattr(msg, "twist", None)
                        if twist is not None:
                            linear = getattr(twist, "linear", None)
                            if linear is not None:
                                linear_vel = max(
                                    abs(float(getattr(linear, "x", 0))),
                                    abs(float(getattr(linear, "y", 0))),
                                )
                            angular = getattr(twist, "angular", None)
                            if angular is not None:
                                angular_vel = abs(float(getattr(angular, "z", 0)))
                        # Alternative: direct speed fields
                        if linear_vel == 0.0 and angular_vel == 0.0:
                            linear_vel = abs(float(getattr(msg, "linear_velocity", 0)
                                                or getattr(msg, "linear_speed", 0)))
                            angular_vel = abs(float(getattr(msg, "angular_velocity", 0)
                                                 or getattr(msg, "angular_speed", 0)))
                    break

        # ── Classify ──
        if linear_vel > _MOVING_LINEAR_VEL_THRESH:
            self._static_start = None
            return _MOVING

        if angular_vel > _ROTATING_ANGULAR_VEL_THRESH_RAD:
            self._static_start = None
            return _ROTATING

        # Static — track duration for long-static transition
        now = time.monotonic()
        if self._static_start is None:
            self._static_start = now
        if now - self._static_start >= _STATIC_LONG_THRESHOLD_S:
            return _STATIC_LONG
        return _STATIC

    # ── tick logic ─────────────────────────────────────────────────────

    # Per-class dedup radii (metres).  Large furniture can have 2m+
    # position drift when re-detected from different angles; small
    # objects are more localised.  Default: 3.0 m.
    _DEDUP_RADII: Dict[str, float] = {
        "cabinet": 3.0, "shelf": 3.0, "table": 3.0, "desk": 3.0,
        "couch": 3.0, "sofa": 3.0, "chair": 2.5, "bed": 3.0,
        "door": 3.0, "window": 3.0, "refrigerator": 3.0,
        "monitor": 2.0, "tv": 2.0, "picture_frame": 2.0,
        "lamp": 2.0, "plant": 1.5, "potted_plant": 1.5,
        "keyboard": 1.5, "mouse": 1.0, "cup": 1.0, "bottle": 1.0,
    }
    _DEFAULT_DEDUP_RADIUS_M = float(
        os.environ.get("OBJECT_WATCHDOG_DEDUP_RADIUS_M", "3.0")
    )

    def _dedup_radius_for(self, cls: str) -> float:
        return self._DEDUP_RADII.get(cls, self._DEFAULT_DEDUP_RADIUS_M)

    @staticmethod
    def _grid_key(obj) -> str:
        """Coarse grid key for node/image-count tracking."""
        gx = round(float(obj.pose.x))
        gy = round(float(obj.pose.y))
        return f"{obj.cls}@{gx},{gy}"

    def _is_duplicate(self, obj) -> bool:
        """Check whether *obj* is within its per-class dedup radius of any
        already-seen object of the same class, using actual world-frame
        positions (not rounded grid cells)."""
        cls = obj.cls
        positions = self._seen_objects.get(cls)
        if not positions:
            return False
        ox, oy = float(obj.pose.x), float(obj.pose.y)
        r2 = self._dedup_radius_for(cls) ** 2
        for i in range(0, len(positions), 2):
            sx, sy = positions[i], positions[i + 1]
            if (ox - sx) ** 2 + (oy - sy) ** 2 <= r2:
                return True
        return False

    def _mark_seen(self, obj) -> None:
        """Record the actual world-frame position of a saved object."""
        self._seen_objects.setdefault(obj.cls, []).extend(
            [float(obj.pose.x), float(obj.pose.y)]
        )

    async def _tick(self) -> None:
        objs, _surfs = await self._registry.snapshot()
        visible: Dict[str, Any] = {
            o.object_id: o for o in objs.values() if not o.missing
        }

        # ── 1. Find genuinely new objects (spatial + temporal dedup) ──
        # Two-stage check:
        #   a) Cross-tick: against all previously-saved positions (per-class radius)
        #   b) Same-tick: new objects of the same class >1m apart → different
        new_objects: list = []
        same_tick_positions: Dict[str, list] = {}  # cls → [x1,y1, x2,y2, ...]
        for obj in visible.values():
            cls = obj.cls
            # Cross-tick dedup.
            if self._is_duplicate(obj):
                continue
            # Same-tick dedup: don't save two "new" objects of the same
            # class that are within 1m of each other in this tick.
            ox, oy = float(obj.pose.x), float(obj.pose.y)
            dup = False
            stp = same_tick_positions.get(cls)
            if stp:
                for i in range(0, len(stp), 2):
                    if (ox - stp[i]) ** 2 + (oy - stp[i + 1]) ** 2 <= 1.0:
                        dup = True
                        break
            if dup:
                continue
            same_tick_positions.setdefault(cls, []).extend([ox, oy])
            new_objects.append(obj)

        if new_objects:
            log.info(
                "object_watchdog: %d new object(s): %s",
                len(new_objects),
                ", ".join(f"{o.cls}@{o.pose.x:.1f},{o.pose.y:.1f}"
                          for o in new_objects),
            )

            # Step 1: capture ONE raw BGR frame for the whole batch.
            loop = asyncio.get_running_loop()
            bgr = await loop.run_in_executor(None, self._capture_raw_bgr)
            if bgr is None:
                log.warning("object_watchdog: raw frame capture failed — "
                            "skipping %d new object(s)", len(new_objects))
                return

            # Step 2: for each object, project→annotate a copy→encode→POST.
            h, w = bgr.shape[:2]
            saved = 0
            for obj in new_objects:
                px, py = self._project_to_pixel(obj, w, h)
                img_b64 = await loop.run_in_executor(
                    None, self._annotate_and_encode, bgr, obj, px, py,
                )
                if not img_b64:
                    log.warning("object_watchdog: annotate failed for %s — retry",
                                obj.cls)
                    continue
                self._mark_seen(obj)
                ok = await self._save_object(obj, img_b64, w, h)
                if ok:
                    key = self._grid_key(obj)
                    self._grid_img_count[key] = 1
                    saved += 1
            if saved:
                log.info("object_watchdog: saved %d/%d new object(s)",
                         saved, len(new_objects))

        # ── 2. Multi-angle: append images to recently-seen objects ────
        append_candidates = []
        for obj in visible.values():
            key = self._grid_key(obj)
            cnt = self._grid_img_count.get(key, 0)
            if 0 < cnt < self._max_images_per_object:
                append_candidates.append((key, obj))

        if append_candidates:
            loop = asyncio.get_running_loop()
            bgr = await loop.run_in_executor(None, self._capture_raw_bgr)
            if bgr is not None:
                h, w = bgr.shape[:2]
                for key, obj in append_candidates:
                    node_id = self._grid_node.get(key)
                    if node_id is None or node_id == 0:
                        continue
                    px, py = self._project_to_pixel(obj, w, h)
                    img_b64 = await loop.run_in_executor(
                        None, self._annotate_and_encode, bgr, obj, px, py,
                    )
                    if img_b64:
                        ok = await self._append_image(node_id, obj, img_b64)
                        if ok:
                            self._grid_img_count[key] += 1
                            log.info("object_watchdog: angle %d/%d for %s",
                                     self._grid_img_count[key],
                                     self._max_images_per_object, key)

    # ── frame capture ──────────────────────────────────────────────────

    def _capture_raw_bgr(self):
        """Capture latest RGB frame as a BGR numpy array (HxWx3, uint8).

        Returns ``None`` on any failure.  Runs in executor thread.
        """
        if self._hub is None or not self._hub.has("rgb"):
            return None
        rgb_msg, _stamp, _count = self._hub.latest("rgb")
        if rgb_msg is None or cv2 is None:
            return None
        try:
            raw = bytes(rgb_msg.data)
            h, w = rgb_msg.height, rgb_msg.width
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, -1)
            if rgb_msg.encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception:
            log.warning("object_watchdog: raw frame capture failed", exc_info=True)
            return None

    @staticmethod
    def _annotate_and_encode(bgr: "np.ndarray", obj, px: int, py: int) -> str:
        """Annotate a BGR frame for *obj* and return a base64 JPEG string.

        Runs in executor thread.  Returns ``""`` on failure.
        """
        try:
            arr = bgr.copy()
            # Red dot + label overlay.
            dot_r = max(4, min(arr.shape[0], arr.shape[1]) // 120)
            cv2.circle(arr, (px, py), dot_r, (0, 0, 255), -1)
            cv2.circle(arr, (px, py), dot_r + 2, (255, 255, 255), 1)

            label = str(obj.cls or "?")
            x, y, z = float(obj.pose.x), float(obj.pose.y), float(obj.pose.z)
            bb = getattr(obj, 'last_bbox_2d', None)
            if bb is not None and len(bb) == 4:
                line1 = f"{label}  bbox=[{int(bb[0])},{int(bb[1])},{int(bb[2])},{int(bb[3])}]"
                line2 = f"map: ({x:.2f}, {y:.2f}, {z:.2f})  px=({px},{py})"
            else:
                line1 = f"{label}"
                line2 = f"map: ({x:.2f}, {y:.2f}, {z:.2f})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.40
            thick = 2
            (tw1, th), _ = cv2.getTextSize(line1, font, scale, thick)
            (tw2, _), _ = cv2.getTextSize(line2, font, scale, thick)
            tw = max(tw1, tw2)
            pad = 6
            bar_h = th * 2 + pad * 3
            cv2.rectangle(arr, (0, 0), (tw + pad * 2, bar_h), (30, 30, 30), -1)
            cv2.putText(arr, line1, (pad, th + pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
            cv2.putText(arr, line2, (pad, th * 2 + pad * 2), font, scale, (180, 180, 180), thick, cv2.LINE_AA)

            ok, jpg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return ""
            return base64.b64encode(jpg.tobytes()).decode("ascii")
        except Exception:
            log.warning("object_watchdog: annotate+encode failed", exc_info=True)
            return ""

    def _project_to_pixel(self, obj, img_w: int, img_h: int):
        """Return the object's pixel position for annotation / masking.

        When ``obj.last_bbox_2d`` is populated (VLM detector path), the
        bbox centre is used directly — this is the actual detector output
        and needs no projection.  Otherwise falls back to approximate
        3D→2D pinhole projection.
        """
        # ── Path A: real pixel bbox from VLM detector ───────────────
        bb = getattr(obj, 'last_bbox_2d', None)
        if bb is not None and len(bb) == 4:
            cx = int((bb[0] + bb[2]) / 2)
            cy = int((bb[1] + bb[3]) / 2)
            return max(0, min(img_w - 1, cx)), max(0, min(img_h - 1, cy))

        # ── Path B: approximate 3D→2D projection (fallback) ─────────
        cam_x = cam_y = cam_z = 0.0
        cam_yaw = 0.0
        if self._hub is not None:
            cam_frame = os.environ.get(
                "SCENE_CAMERA_FRAME", "head_front_camera_rgb_optical_frame"
            )
            pose = self._hub.lookup_xy_yaw(cam_frame, "map")
            if pose is not None:
                cam_x, cam_y, cam_z, cam_yaw = pose

        obj_x, obj_y, obj_z = float(obj.pose.x), float(obj.pose.y), float(obj.pose.z)
        dwx, dwy, dwz = obj_x - cam_x, obj_y - cam_y, obj_z - cam_z

        cos_y, sin_y = np.cos(cam_yaw), np.sin(cam_yaw)
        bx = dwx * cos_y + dwy * sin_y
        by = -dwx * sin_y + dwy * cos_y
        ox, oy, oz = -by, -dwz, bx  # body → ROS optical

        fx = fy = 554.0
        cx, cy = img_w / 2, img_h / 2
        if self._hub is not None and self._hub.has("intrinsics"):
            intr_msg, _, _ = self._hub.latest("intrinsics")
            if intr_msg is not None:
                k = getattr(intr_msg, 'k', ())
                if len(k) >= 4:
                    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])

        if abs(oz) < 0.01:
            oz = 1.0
        u = int(fx * ox / oz + cx)
        v = int(fy * oy / oz + cy)
        return max(0, min(img_w - 1, u)), max(0, min(img_h - 1, v))

    def _build_camera_params(self, img_w: int, img_h: int) -> Dict[str, Any]:
        """Assemble camera_params for a remember payload (dataset spec).

        Intrinsics mirror ``_project_to_pixel`` (fallback fx=fy=554,
        cx=img_w/2, cy=img_h/2).  Pose uses camera→map tf; only yaw is
        available via ``lookup_xy_yaw``, so the quaternion is built as
        qz=sin(yaw/2), qw=cos(yaw/2) with the other components zero.
        """
        fx = fy = 554.0
        cx, cy = img_w / 2.0, img_h / 2.0
        if self._hub is not None and self._hub.has("intrinsics"):
            intr_msg, _, _ = self._hub.latest("intrinsics")
            if intr_msg is not None:
                k = getattr(intr_msg, 'k', ())
                if len(k) >= 4:
                    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])

        camera_pose: Optional[Dict[str, float]] = None
        if self._hub is not None:
            cam_frame = os.environ.get(
                "SCENE_CAMERA_FRAME", "head_front_camera_rgb_optical_frame"
            )
            pose = self._hub.lookup_xy_yaw(cam_frame, "map")
            if pose is not None:
                cam_x, cam_y, cam_z, cam_yaw = pose
                camera_pose = {
                    "x": float(cam_x), "y": float(cam_y), "z": float(cam_z),
                    "qx": 0.0, "qy": 0.0,
                    "qz": math.sin(cam_yaw / 2.0),
                    "qw": math.cos(cam_yaw / 2.0),
                }

        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": int(img_w), "height": int(img_h),
            "depth_scale": 0.001,
            "camera_type": "rgb",
            "camera_pose": camera_pose,
        }

    # ── Room/region lookup ─────────────────────────────────────────────

    def _lookup_region(self, x: float, y: float) -> str:
        """Return the room name that contains (x, y), or '' if none match."""
        if self._anno_store is None:
            return ""
        from .geometry import point_in_polygon
        for ann in self._anno_store.list():
            if ann.kind == "room" and not ann.stale:
                if point_in_polygon(x, y, ann.points):
                    return ann.name
        return ""

    # ── POST to memgraph ───────────────────────────────────────────────

    async def _save_object(self, obj, img_b64: str,
                           img_w: int = 0, img_h: int = 0) -> bool:
        """POST a single-object remember request to the memgraph Scene Hook."""
        import httpx

        frame_id = str(obj.pose.frame_id or "").strip()
        if not frame_id:
            log.warning(
                "object_watchdog: skip %s — spatial frame is unknown",
                obj.object_id,
            )
            return False
        now_ns = time.time_ns()
        ox, oy = float(obj.pose.x), float(obj.pose.y)
        region = self._lookup_region(ox, oy)
        if region:
            log.info("object_watchdog: %s@%.1f,%.1f → region=%s", obj.cls, ox, oy, region)
        camera_params = self._build_camera_params(img_w, img_h) if img_w and img_h else None
        payload: Dict[str, Any] = {
            "session_id": "scene-watchdog",
            "plan_id": "scene-watchdog",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "scene",
                "msg": f"observed new object: {obj.cls}",
            },
            "spatial": {
                "origin": frame_id,
                "objects": [
                    {
                        "obj_id": obj.object_id,
                        "label": obj.cls,
                        "x": ox,
                        "y": oy,
                        "z": float(obj.pose.z),
                    }
                ],
            },
            "image_base64": img_b64,
            "depth_base64": "",  # P0: no depth capture, field placeholder
            "camera_params": camera_params,
            "time_range": {"start_ts": now_ns, "end_ts": now_ns},
            "kv": {"region": region} if region else {},
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self._memgraph_url, json=payload)
            if r.status_code >= 400:
                log.warning(
                    "object_watchdog: memgraph returned %d for %s: %s",
                    r.status_code, obj.object_id, r.text[:200],
                )
                return False
            result = r.json()
            node_id = result.get("node_id", 0)
            if isinstance(node_id, int) and node_id >= 0:
                # Track for multi-angle: map grid_key → node_id.
                key = self._grid_key(obj)
                self._grid_node[key] = node_id
            log.info("object_watchdog: %s (%s) → node %s",
                     obj.object_id, obj.cls, node_id)
            return True
        except Exception:
            log.debug(
                "object_watchdog: POST failed for %s", obj.object_id,
                exc_info=True,
            )
            return False

    async def _append_image(self, node_id: int, obj, img_b64: str) -> bool:
        """POST an additional frame to an existing memory node.

        Uses memgraph's Scene Hook with ``parent_node_id`` to create a
        child node that shares the same object identity, or appends to
        the existing node's ``image_refs``.
        """
        import httpx

        now_ns = time.time_ns()
        payload: Dict[str, Any] = {
            "session_id": "scene-watchdog",
            "plan_id": "scene-watchdog",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "scene",
                "msg": f"observed {obj.cls} from another angle",
            },
            "spatial": {
                "origin": "world",
                "objects": [
                    {
                        "obj_id": obj.object_id,
                        "label": obj.cls,
                        "x": float(obj.pose.x),
                        "y": float(obj.pose.y),
                        "z": float(obj.pose.z),
                    }
                ],
            },
            "image_base64": img_b64,
            "parent_node_id": node_id,
            "time_range": {"start_ts": now_ns, "end_ts": now_ns},
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self._memgraph_url, json=payload)
            if r.status_code >= 400:
                log.debug("object_watchdog: append image failed for node %d: %d",
                         node_id, r.status_code)
                return False
            return True
        except Exception:
            log.debug("object_watchdog: append POST failed for node %d",
                     node_id, exc_info=True)
            return False


# ═════════════════════════════════════════════════════════════════════════
# PlaceNodeManager — creates/updates PlaceNode memory nodes for static regions
# ═════════════════════════════════════════════════════════════════════════

# Minimum distance (metres) to the nearest existing PlaceNode before
# creating a new one.
_PLACE_MIN_DIST_M = float(os.environ.get("PLACE_NODE_MIN_DIST_M", "2.0"))
# Minimum stay duration (seconds) before creating a PlaceNode.
_PLACE_MIN_STAY_S = float(os.environ.get("PLACE_NODE_MIN_STAY_S", "10.0"))
# Update interval for PlaceNode duration field (seconds).
_PLACE_UPDATE_INTERVAL_S = float(os.environ.get("PLACE_NODE_UPDATE_INTERVAL_S", "60.0"))


class PlaceNodeManager:
    """Manage PlaceNode creation for robot stationary periods.

    Tracks robot position and region, creates ``node_type="place"``
    MemoryNodes when the robot enters a new area, updates duration on
    each tick, and closes the PlaceNode when the robot leaves.

    Usage::

        mgr = PlaceNodeManager(memgraph_url="http://127.0.0.1:37798")
        # On each tick:
        nid = await mgr.tick(x, y, region, is_moving=False)
    """

    def __init__(self, memgraph_url: str = _MEMGRAPH_HOOK_URL) -> None:
        self._url = memgraph_url
        self._current_node_id: Optional[int] = None
        self._current_region: str = ""
        self._enter_ts: float = 0.0
        self._last_update_ts: float = 0.0
        self._last_x: float = 0.0
        self._last_y: float = 0.0
        # Track known PlaceNode positions for region transition detection.
        self._known_centers: List[tuple] = []  # [(x, y, region, node_id), ...]

    @property
    def current_node_id(self) -> Optional[int]:
        """The current PlaceNode's node_id, or None."""
        return self._current_node_id

    async def tick(
        self, x: float, y: float, region: str, *, is_moving: bool = False,
    ) -> Optional[int]:
        """Call on each watchdog tick.  Returns the current PlaceNode's
        node_id (for use as parent_node_id), or None.

        Creates, updates, or closes PlaceNodes based on position + region.
        """
        now = time.time()

        # ── Moving → close PlaceNode ──
        if is_moving:
            if self._current_node_id is not None:
                duration = now - self._enter_ts
                if duration >= _PLACE_MIN_STAY_S:
                    await self._update_place_node(self._current_node_id, duration)
                    log.info("place_node: closed node %d (duration=%.0fs, moving)",
                             self._current_node_id, duration)
                else:
                    # Too short — remove the node
                    await self._delete_place_node(self._current_node_id)
                    log.info("place_node: removed node %d (duration=%.0fs < %.0fs)",
                             self._current_node_id, duration, _PLACE_MIN_STAY_S)
                self._current_node_id = None
                self._current_region = ""
            return None

        # ── Static / rotating → maintain PlaceNode ──
        # Check if we've changed regions or moved far from current PlaceNode
        new_region = self._check_region_transition(x, y, region)

        if new_region or self._current_node_id is None:
            # Close previous PlaceNode if any
            if self._current_node_id is not None:
                duration = now - self._enter_ts
                if duration >= _PLACE_MIN_STAY_S:
                    await self._update_place_node(self._current_node_id, duration)
                else:
                    await self._delete_place_node(self._current_node_id)

            # Create new PlaceNode
            nid = await self._create_place_node(x, y, region)
            if nid is not None:
                self._current_node_id = nid
                self._current_region = region
                self._enter_ts = now
                self._last_update_ts = now
                self._last_x, self._last_y = x, y
                self._known_centers.append((x, y, region, nid))
                log.info("place_node: created node %d at region=%s (%.1f,%.1f)",
                         nid, region, x, y)
            return nid

        # Update position tracking and periodic duration update
        self._last_x, self._last_y = x, y
        if now - self._last_update_ts >= _PLACE_UPDATE_INTERVAL_S:
            duration = now - self._enter_ts
            await self._update_place_node(self._current_node_id, duration)
            self._last_update_ts = now

        return self._current_node_id

    def _check_region_transition(self, x: float, y: float, region: str) -> bool:
        """Return True if the robot has entered a genuinely new area."""
        if region and region != self._current_region:
            return True
        # Check distance to current PlaceNode center
        if self._current_node_id is not None:
            dist = ((x - self._last_x) ** 2 + (y - self._last_y) ** 2) ** 0.5
            if dist > _PLACE_MIN_DIST_M * 3:  # drifted far from tracking pos
                return True
        # Check distance to ALL known PlaceNode centers
        for cx, cy, _cr, _cnid in self._known_centers:
            if ((x - cx) ** 2 + (y - cy) ** 2) <= _PLACE_MIN_DIST_M ** 2:
                return False  # near known PlaceNode — reuse
        return True  # far from all known → new region

    # ── HTTP ──────────────────────────────────────────────────────────

    async def _create_place_node(self, x: float, y: float, region: str) -> Optional[int]:
        """POST a new PlaceNode to memgraph."""
        import httpx

        now_ns = time.time_ns()
        region_str = region or "unknown"
        summary = (
            f"robot stationed at region={region_str}, "
            f"center=({x:.1f},{y:.1f},0.0)"
        )
        payload = {
            "session_id": "scene-place",
            "plan_id": "scene-place",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "place_manager",
                "msg": summary,
            },
            "spatial": {
                "origin": "world",
                "center": {"x": x, "y": y, "z": 0.0},
                "radius_m": 2.0,
                "semantic_region": region_str,
                "objects": [],
            },
            "kv": {
                "node_type": "place",
                "region": region_str,
                "center_x": x,
                "center_y": y,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self._url, json=payload)
            if r.status_code >= 400:
                log.warning("place_node: create POST returned %d: %s",
                            r.status_code, r.text[:200])
                return None
            return r.json().get("node_id")
        except Exception as e:
            log.debug("place_node: create POST failed: %s", e)
            return None

    async def _update_place_node(self, node_id: int, duration_s: float) -> None:
        """Update a PlaceNode's duration (append-only update via POST)."""
        import httpx

        now_ns = time.time_ns()
        payload = {
            "session_id": "scene-place",
            "plan_id": "scene-place",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "place_manager",
                "msg": f"updated duration={duration_s:.0f}s",
            },
            "spatial": {"origin": "world", "objects": []},
            "parent_node_id": node_id,
            "kv": {
                "node_type": "place",
                "append_image_ref": True,
                "duration_s": round(duration_s, 1),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self._url, json=payload)
            if r.status_code >= 400:
                log.debug("place_node: update POST for node %d returned %d",
                          node_id, r.status_code)
        except Exception as e:
            log.debug("place_node: update POST for node %d failed: %s", node_id, e)

    async def _delete_place_node(self, node_id: int) -> None:
        """Remove a PlaceNode that was too short-lived."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{self._url.rstrip('/')}/delete/{node_id}",
                    json={"node_id": node_id},
                )
            if r.status_code >= 400:
                log.debug("place_node: delete POST for node %d returned %d",
                          node_id, r.status_code)
        except Exception as e:
            log.debug("place_node: delete POST for node %d failed: %s", node_id, e)
