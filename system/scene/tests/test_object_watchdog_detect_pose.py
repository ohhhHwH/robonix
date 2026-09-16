# SPDX-License-Identifier: MulanPSL-2.0
"""Detection-time pose annotation tests for ObjectWatchdog.

The fix under test: the ObjectWatchdog no longer re-captures a single
"latest" frame for a batch of newly-seen objects (which, after the robot
has moved, projects each object against a camera pose it was never seen
from).  Instead each object carries the frame + camera→map transform
stamped at detection time, and the watchdog annotates against that pose.

These tests exercise the pure projection / camera-params helpers directly
so the math is verified without ROS, cv2, or a live camera.
"""

import numpy as np
import pytest

from scene_service.object_watchdog import ObjectWatchdog
from scene_service.state.object_registry import BBox3D, Pose3D, SceneObject


def _make_obj(x: float, y: float, z: float, cam_to_map) -> SceneObject:
    """Build a minimal SceneObject carrying a detection-time transform."""
    return SceneObject(
        object_id="scene.object.cup_001",
        cls="cup",
        pose=Pose3D(x=x, y=y, z=z, frame_id="map"),
        bbox=BBox3D(),
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.0,
        detect_frame=None,
        detect_cam_to_map=cam_to_map,
    )


@pytest.fixture()
def watchdog() -> ObjectWatchdog:
    # hub=None exercises the fallback intrinsics (fx=fy=554,
    # cx=img_w/2, cy=img_h/2), which keeps the test self-contained.
    return ObjectWatchdog(registry=None, hub=None)


def test_project_with_identity_pose_hits_center(watchdog):
    """Identity camera→map means the camera sits at map origin looking
    down +Z; a point on the optical axis lands at the principal point."""
    T = np.eye(4)
    obj = _make_obj(0.0, 0.0, 2.0, T)
    px, py = watchdog._project_with_detect_pose(obj, 640, 480)
    assert (px, py) == (320, 240)


def test_project_offsets_match_pinhole(watchdog):
    """A point offset from the optical axis projects off the principal
    point by f * (offset / depth)."""
    T = np.eye(4)
    # x=+1 → right; y=+0.4 → down (optical frame +Y is down).  Chosen
    # to stay well inside the image so no clamping is involved.
    obj = _make_obj(1.0, 0.4, 2.0, T)
    px, py = watchdog._project_with_detect_pose(obj, 640, 480)
    assert px == 320 + int(round(554.0 * 1.0 / 2.0))   # 597
    assert py == 240 + int(round(554.0 * 0.4 / 2.0))   # 351


def test_project_clamps_to_image(watchdog):
    """Off-axis points clamp to the image border instead of overflowing."""
    T = np.eye(4)
    obj = _make_obj(100.0, 0.0, 2.0, T)
    px, py = watchdog._project_with_detect_pose(obj, 640, 480)
    assert (px, py) == (639, 240)


def test_project_behind_camera_returns_none(watchdog):
    """A point behind the camera (negative Z) cannot be projected."""
    T = np.eye(4)
    obj = _make_obj(0.0, 0.0, -1.0, T)
    assert watchdog._project_with_detect_pose(obj, 640, 480) == (None, None)


def test_project_missing_transform_returns_none(watchdog):
    """Records without a detection-time transform fall back to the caller."""
    obj = _make_obj(0.0, 0.0, 2.0, None)
    assert watchdog._project_with_detect_pose(obj, 640, 480) == (None, None)


def test_project_inverts_cam_to_map_correctly(watchdog):
    """detect_cam_to_map is camera→map; projection must invert it to
    map→camera.  A camera translated to (1, 0, 0) in map, looking down
    +Z, sees a point at (1, 0, 2) exactly on its optical axis."""
    T = np.eye(4)
    T[0, 3] = 1.0  # camera at map (1, 0, 0)
    obj = _make_obj(1.0, 0.0, 2.0, T)
    px, py = watchdog._project_with_detect_pose(obj, 640, 480)
    assert (px, py) == (320, 240)


def test_camera_params_from_transform_identity(watchdog):
    """camera_pose mirrors the transform: identity translation + identity
    rotation → origin position, identity quaternion."""
    params = watchdog._camera_params_from_transform(640, 480, np.eye(4))
    pose = params["camera_pose"]
    assert (pose["x"], pose["y"], pose["z"]) == (0.0, 0.0, 0.0)
    assert (pose["qx"], pose["qy"], pose["qz"], pose["qw"]) == (0.0, 0.0, 0.0, 1.0)
    assert params["fx"] == 554.0 and params["cx"] == 320.0
    assert (params["width"], params["height"]) == (640, 480)


def test_camera_params_from_transform_yaw(watchdog):
    """A 90° yaw rotation around Z maps to a (0, 0, sin45, cos45) quaternion."""
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = 1.5
    T[1, 3] = -2.5
    T[2, 3] = 0.4
    params = watchdog._camera_params_from_transform(640, 480, T)
    pose = params["camera_pose"]
    assert (pose["x"], pose["y"], pose["z"]) == (1.5, -2.5, 0.4)
    assert pose["qx"] == pytest.approx(0.0, abs=1e-9)
    assert pose["qy"] == pytest.approx(0.0, abs=1e-9)
    assert pose["qz"] == pytest.approx(np.sin(np.pi / 4), abs=1e-9)
    assert pose["qw"] == pytest.approx(np.cos(np.pi / 4), abs=1e-9)


def test_camera_params_bad_transform_keeps_intrinsics(watchdog):
    """A malformed transform must not crash: camera_pose is dropped to
    None but intrinsics survive."""
    params = watchdog._camera_params_from_transform(640, 480, np.zeros((3, 3)))
    assert params["camera_pose"] is None
    assert params["fx"] == 554.0


def test_quat_from_rotmat_identity():
    assert ObjectWatchdog._quat_from_rotmat(np.eye(3)) == (0.0, 0.0, 0.0, 1.0)
