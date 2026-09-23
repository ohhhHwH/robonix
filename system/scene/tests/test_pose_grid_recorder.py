# SPDX-License-Identifier: MulanPSL-2.0
"""Pure grid/angle threshold logic tests for PoseGridRecorder.

The recorder keys memory snapshots by a 2m spatial grid cell plus a 90°
heading bucket.  These tests exercise the pure coordinate→(cell, bucket)
math without ROS, cv2, or a live camera so the threshold semantics are
verified in isolation.
"""

import math

from scene_service.pose_grid_recorder import PoseGridRecorder


def test_grid_cell_floors_and_handles_negatives():
    assert PoseGridRecorder._grid_cell(0.0, 0.0, 2.0) == (0, 0)
    assert PoseGridRecorder._grid_cell(1.99, 1.99, 2.0) == (0, 0)
    assert PoseGridRecorder._grid_cell(2.0, 0.0, 2.0) == (1, 0)
    # Negative coordinates floor toward -∞, not toward zero.
    assert PoseGridRecorder._grid_cell(-0.01, 0.0, 2.0) == (-1, 0)
    assert PoseGridRecorder._grid_cell(-4.8, 4.21, 2.0) == (-3, 2)


def test_grid_cell_scales_with_grid_size():
    assert PoseGridRecorder._grid_cell(4.0, 4.0, 4.0) == (1, 1)
    assert PoseGridRecorder._grid_cell(3.99, 3.99, 4.0) == (0, 0)


def test_yaw_bucket_four_quadrants():
    step = math.pi / 2  # 90°
    assert PoseGridRecorder._yaw_bucket(0.0, step) == 0
    assert PoseGridRecorder._yaw_bucket(math.pi / 4, step) == 0
    assert PoseGridRecorder._yaw_bucket(math.pi / 2, step) == 1
    assert PoseGridRecorder._yaw_bucket(math.pi, step) == 2
    assert PoseGridRecorder._yaw_bucket(3 * math.pi / 2, step) == 3


def test_yaw_bucket_wraps_negative_and_past_two_pi():
    step = math.pi / 2
    # -45° normalises to 315° → bucket 3.
    assert PoseGridRecorder._yaw_bucket(-math.pi / 4, step) == 3
    # Just under 360° → bucket 3.
    assert PoseGridRecorder._yaw_bucket(2 * math.pi - 0.01, step) == 3


def test_yaw_bucket_scales_with_angle():
    step = math.pi / 4  # 45°
    assert PoseGridRecorder._yaw_bucket(0.0, step) == 0
    assert PoseGridRecorder._yaw_bucket(math.pi / 2, step) == 2
    assert PoseGridRecorder._yaw_bucket(math.pi, step) == 4
