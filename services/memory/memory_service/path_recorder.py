# SPDX-License-Identifier: MulanPSL-2.0
"""Path memory recorder — captures robot traversal paths as MemoryNodes.

Records waypoints during robot movement, applying spatial and temporal
sampling (≥0.5m displacement or ≥15° rotation, min 1s interval).  Each
waypoint captures a JPEG frame.  When the segment is closed, a
``path_segment`` MemoryNode is created via HTTP POST to memgraph's
Scene Hook endpoint.

Usage::

    rec = PathRecorder()
    while robot.moving:
        frame = camera.capture()  # raw RGB bytes
        rec.record_waypoint(x, y, z, yaw, frame)
    node_id = rec.close_segment()
    print(f"path segment → node {node_id}")

Auto-split: segments with 50+ waypoints or a cumulative turn ≥90°
are split automatically — ``close_segment()`` returns only the *last*
segment's node_id; earlier segments have already been committed.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

log = logging.getLogger("scribe_mem")

# Minimum distance (metres) between recorded waypoints.
_MIN_DIST_M = float(os.environ.get("PATH_RECORDER_MIN_DIST_M", "0.5"))

# Minimum rotation (degrees) between recorded waypoints.
_MIN_ROTATION_DEG = float(os.environ.get("PATH_RECORDER_MIN_ROTATION_DEG", "15.0"))

# Minimum wall-clock interval (seconds) between waypoints.
_MIN_INTERVAL_S = float(os.environ.get("PATH_RECORDER_MIN_INTERVAL_S", "1.0"))

# Auto-split at this many waypoints.
_MAX_WAYPOINTS_PER_SEGMENT = int(
    os.environ.get("PATH_RECORDER_MAX_WAYPOINTS", "50")
)

# Auto-split when cumulative turn exceeds this (degrees).
_MAX_CUMULATIVE_TURN_DEG = float(
    os.environ.get("PATH_RECORDER_MAX_TURN_DEG", "90.0")
)

# JPEG encode parameters.
_JPEG_SIZE = (640, 480)
_JPEG_QUALITY = 85

# Default memgraph Scene Hook endpoint.
_MEMGRAPH_HOOK_URL = os.environ.get(
    "MEMGRAPH_HOOK_URL", "http://127.0.0.1:37798",
)


def _encode_jpeg(frame_rgb: bytes, size: Tuple[int, int] = _JPEG_SIZE,
                 quality: int = _JPEG_QUALITY) -> Optional[str]:
    """Encode raw RGB bytes as a base64 JPEG string.

    Accepts raw RGB bytes (H×W×3) and returns a base64-encoded JPEG.
    If *frame_rgb* looks like already-encoded JPEG (starts with 0xFF 0xD8),
    it is returned as-is after base64 encoding.
    Returns ``None`` on failure.
    """
    if PILImage is None:
        log.warning("path_recorder: Pillow not installed — cannot encode JPEG")
        return None

    try:
        # Already JPEG?  (magic bytes FF D8 FF)
        if len(frame_rgb) >= 3 and frame_rgb[:2] == b'\xff\xd8':
            return base64.b64encode(frame_rgb).decode("ascii")

        # Try to decode from raw bytes
        img = PILImage.frombytes("RGB", size, frame_rgb, "raw")
        img = img.resize(_JPEG_SIZE, PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # Fallback: try opening as any PIL-supported format, resize, re-encode
        try:
            buf_in = io.BytesIO(frame_rgb)
            img = PILImage.open(buf_in).convert("RGB")
            img = img.resize(_JPEG_SIZE, PILImage.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            log.warning("path_recorder: JPEG encode failed: %s", e)
            return None


class PathRecorder:
    """Record robot traversal paths with spatial sampling and image capture.

    Accumulates waypoints in memory.  Each ``close_segment()`` call
    commits a ``path_segment`` MemoryNode to memgraph via HTTP POST
    and resets the internal buffer.  Auto-split rules (≥50 waypoints
    or ≥90° cumulative turn) trigger automatic intermediate commits.
    """

    def __init__(self, memgraph_url: str = _MEMGRAPH_HOOK_URL) -> None:
        self._url = memgraph_url
        self._waypoints: List[Dict[str, Any]] = []
        self._images_b64: List[str] = []
        self._frame_counter: int = 0
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._last_yaw: Optional[float] = None
        self._last_ts: float = 0.0
        self._cumulative_turn: float = 0.0
        self._prev_yaw: Optional[float] = None
        # Track committed segment node_ids (auto-split)
        self._committed_nodes: List[int] = []

    # ── public API ──────────────────────────────────────────────────────

    def record_waypoint(
        self, x: float, y: float, z: float, yaw: float,
        frame_rgb: bytes, *, ts: Optional[float] = None,
    ) -> bool:
        """Attempt to record a waypoint at *(x, y, z, yaw)*.

        The waypoint is recorded only when it passes the sampling gates
        (distance, rotation, time interval).  Returns ``True`` if the
        waypoint was stored, ``False`` if it was skipped.

        Parameters:
            x, y, z: world-frame position (metres).
            yaw: heading in radians.
            frame_rgb: raw camera frame bytes (RGB or JPEG).
            ts: unix timestamp (default: ``time.time()``).
        """
        now = ts if ts is not None else time.time()

        # ── Temporal gate ──
        if self._last_ts > 0 and (now - self._last_ts) < _MIN_INTERVAL_S:
            return False

        # ── Spatial gates ──
        dist = 0.0
        d_yaw = 0.0
        if self._last_x is not None:
            dist = math.hypot(x - self._last_x, y - self._last_y)
            d_yaw = abs(_angle_diff(yaw, self._last_yaw or 0.0))
            d_yaw_deg = math.degrees(d_yaw)

            if dist < _MIN_DIST_M and d_yaw_deg < _MIN_ROTATION_DEG:
                return False

        # ── Encode image ──
        img_b64 = _encode_jpeg(frame_rgb)
        if img_b64 is None:
            return False

        # ── Store ──
        wy = {
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "yaw": round(yaw, 4),
            "ts": now,
        }
        self._waypoints.append(wy)
        self._images_b64.append(img_b64)
        self._frame_counter += 1

        self._last_x, self._last_y, self._last_yaw = x, y, yaw
        self._last_ts = now

        # Track cumulative turn for auto-split
        if self._prev_yaw is not None:
            self._cumulative_turn += abs(math.degrees(_angle_diff(yaw, self._prev_yaw)))
        self._prev_yaw = yaw

        # ── Auto-split ──
        if len(self._waypoints) >= _MAX_WAYPOINTS_PER_SEGMENT:
            log.info("path_recorder: auto-split at %d waypoints", len(self._waypoints))
            self.close_segment()
        elif self._cumulative_turn >= _MAX_CUMULATIVE_TURN_DEG:
            log.info("path_recorder: auto-split at %.1f° cumulative turn",
                     self._cumulative_turn)
            self.close_segment()

        return True

    def close_segment(self) -> Optional[int]:
        """Commit the current path segment to memgraph and reset the buffer.

        Returns:
            The ``node_id`` of the created ``path_segment`` MemoryNode,
            or ``None`` if the segment is empty or the POST failed.
            If auto-split was triggered earlier, returns the *last*
            segment's node_id; earlier node_ids are available via
            :attr:`committed_nodes`.
        """
        if not self._waypoints:
            return None

        n = len(self._waypoints)
        start = self._waypoints[0]
        end = self._waypoints[-1]
        length = self._path_length()
        summary = (
            f"robot traversed from ({start['x']:.1f},{start['y']:.1f}) "
            f"to ({end['x']:.1f},{end['y']:.1f}), "
            f"length={length:.1f}m, {n} waypoints"
        )

        # Build spatial objects from waypoints
        spatial_objects: List[Dict[str, Any]] = []
        for i, wy in enumerate(self._waypoints):
            spatial_objects.append({
                "obj_id": f"waypoint.{i}",
                "label": f"waypoint_{i}",
                "x": wy["x"],
                "y": wy["y"],
                "z": wy["z"],
            })

        now_ns = time.time_ns()
        payload: Dict[str, Any] = {
            "session_id": "path-recorder",
            "plan_id": "path-recorder",
            "log_record": {
                "ts": now_ns,
                "level": "Info",
                "tag": "path_recorder",
                "msg": summary,
            },
            "spatial": {
                "origin": "world",
                "objects": spatial_objects,
            },
            "image_base64": self._images_b64[0] if self._images_b64 else "",
            "kv": {
                "node_type": "path_segment",
                "waypoints": self._waypoints,        # full waypoint data
                "path_length_m": round(length, 2),
                "waypoint_count": n,
                "start_pos": [start["x"], start["y"], start["z"]],
                "end_pos": [end["x"], end["y"], end["z"]],
            },
        }

        node_id = self._post(payload)
        if node_id is not None and node_id > 0:
            self._committed_nodes.append(node_id)
            log.info(
                "path_recorder: segment → node %d (%d waypoints, %.1fm)",
                node_id, n, length,
            )
        else:
            log.warning("path_recorder: POST failed — segment not saved")

        # Reset state for next segment
        self._waypoints.clear()
        self._images_b64.clear()
        self._cumulative_turn = 0.0
        self._prev_yaw = None

        return node_id

    # ── properties ──────────────────────────────────────────────────────

    @property
    def waypoint_count(self) -> int:
        """Number of waypoints in the current (uncommitted) segment."""
        return len(self._waypoints)

    @property
    def committed_nodes(self) -> List[int]:
        """node_ids of all committed path_segment nodes (includes auto-split)."""
        return list(self._committed_nodes)

    # ── helpers ─────────────────────────────────────────────────────────

    def _path_length(self) -> float:
        """Compute cumulative path length from waypoints (metres)."""
        if len(self._waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._waypoints)):
            a, b = self._waypoints[i - 1], self._waypoints[i]
            total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        return total

    def _post(self, payload: Dict[str, Any]) -> Optional[int]:
        """POST *payload* to memgraph Scene Hook.  Returns node_id or None."""
        import httpx
        import json as _json

        try:
            body = _json.dumps(payload).encode()
            with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=15.0)) as client:
                r = client.post(
                    self._url.rstrip("/"),
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
            if r.status_code >= 400:
                log.warning("path_recorder: memgraph returned %d: %s",
                            r.status_code, r.text[:200])
                return None
            resp = r.json()
            nid = resp.get("node_id")
            return int(nid) if nid is not None else None
        except Exception as e:
            log.warning("path_recorder: POST failed: %s: %s",
                        type(e).__name__, e)
            return None


# ── utilities ───────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference (radians) in [-π, π]."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d
