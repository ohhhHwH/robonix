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

import asyncio

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
    assert px == 320 + 554.0 * 1.0 / 2.0   # 597.0
    assert py == 240 + 554.0 * 0.4 / 2.0   # 350.8


def test_project_returns_unclamped(watchdog):
    """Off-axis points return their unclamped projection; the caller
    (via ``_center_in_frame``) decides whether to skip them."""
    T = np.eye(4)
    obj = _make_obj(100.0, 0.0, 2.0, T)
    px, py = watchdog._project_with_detect_pose(obj, 640, 480)
    # u = 320 + 554 * (100 / 2) = 28020 (far off-frame), v = 240.
    assert px == 320 + 554.0 * 100.0 / 2.0
    assert py == 240.0


def test_center_in_frame_rejects_offscreen(watchdog):
    """A centre projecting outside the frame is rejected; interior is kept."""
    assert watchdog._center_in_frame(320.0, 240.0, 640, 480) is True
    assert watchdog._center_in_frame(-1.0, 240.0, 640, 480) is False
    assert watchdog._center_in_frame(640.0, 240.0, 640, 480) is False
    assert watchdog._center_in_frame(320.0, -0.5, 640, 480) is False
    assert watchdog._center_in_frame(320.0, 480.0, 640, 480) is False


def test_clamp_to_pixel(watchdog):
    """Clamp maps an unclamped projection back into integer pixels."""
    assert watchdog._clamp_to_pixel(28020.0, 240.0, 640, 480) == (639, 240)
    assert watchdog._clamp_to_pixel(320.0, 240.0, 640, 480) == (320, 240)
    assert watchdog._clamp_to_pixel(-50.0, -50.0, 640, 480) == (0, 0)


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


def test_current_frame_projection_has_no_stale_bbox_fallback(watchdog):
    """A current frame without live geometry must be skipped, not mislabeled."""
    obj = _make_obj(0.0, 0.0, 2.0, None)
    obj.last_bbox_2d = (10.0, 20.0, 30.0, 40.0)
    assert watchdog._project_to_pixel(obj, 640, 480) == (None, None)
    assert watchdog._bbox_center(obj) == (20.0, 30.0)


def test_multi_angle_frame_uses_live_projection(monkeypatch):
    """A newly captured angle must use its matching live camera transform."""
    obj = _make_obj(0.0, 0.0, 2.0, np.eye(4))

    class Registry:
        async def snapshot(self):
            return {obj.object_id: obj}, {}

    watchdog = ObjectWatchdog(registry=Registry(), hub=None)
    watchdog._mark_seen(obj)
    key = watchdog._grid_key(obj)
    watchdog._grid_node[key] = 7
    watchdog._grid_img_count[key] = 1
    calls = []

    monkeypatch.setattr(
        watchdog, "_capture_raw_bgr", lambda: np.zeros((480, 640, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        watchdog,
        "_project_world_to_pixel",
        lambda _obj, _w, _h: calls.append("live") or (320.0, 240.0),
    )
    monkeypatch.setattr(
        watchdog,
        "_project_with_detect_pose",
        lambda *_args: pytest.fail("stale detection transform used for a current frame"),
    )
    monkeypatch.setattr(
        watchdog, "_annotate_and_encode", lambda *_args: "encoded-image"
    )

    async def append_image(node_id, saved_obj, image):
        assert (node_id, saved_obj, image) == (7, obj, "encoded-image")
        return True

    monkeypatch.setattr(watchdog, "_append_image", append_image)
    asyncio.run(watchdog._tick())

    assert calls == ["live"]
    assert watchdog._grid_img_count[key] == 2


def test_watchdog_excludes_picture_frame_and_robot(monkeypatch):
    """Noise and the Scene self-object never become long-term memories."""
    picture = _make_obj(0.0, 0.0, 2.0, np.eye(4))
    picture.object_id = "scene.object.picture_frame_001"
    picture.cls = "picture frame"
    robot = _make_obj(1.0, 0.0, 0.0, None)
    robot.object_id = "scene.object.robot_001"
    robot.cls = "robot"

    class Registry:
        async def snapshot(self):
            return {picture.object_id: picture, robot.object_id: robot}, {}

    watchdog = ObjectWatchdog(registry=Registry(), hub=None)
    monkeypatch.setattr(
        watchdog,
        "_capture_raw_bgr",
        lambda: pytest.fail("ignored objects triggered image capture"),
    )

    asyncio.run(watchdog._tick())

    assert watchdog._seen_objects == {}
    assert watchdog._grid_node == {}


def test_append_image_uses_memory_append_operation(monkeypatch, watchdog):
    """Additional angles update one node instead of creating child nodes."""
    import httpx

    captured = {}

    class Response:
        status_code = 200

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured.update(json)
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    obj = _make_obj(0.0, 0.0, 2.0, np.eye(4))

    assert asyncio.run(watchdog._append_image(7, obj, "encoded-image")) is True
    assert captured["parent_node_id"] == 7
    assert captured["kv"] == {"append_image_ref": True}


def _motion_watchdog(linear_m_s: float, angular_deg_s: float) -> ObjectWatchdog:
    """Build a watchdog with one nested nav_msgs-style odometry sample."""
    from types import SimpleNamespace

    twist = SimpleNamespace(
        linear=SimpleNamespace(x=linear_m_s, y=0.0),
        angular=SimpleNamespace(z=float(np.deg2rad(angular_deg_s))),
    )
    odom = SimpleNamespace(twist=SimpleNamespace(twist=twist))

    class Hub:
        @staticmethod
        def has(topic):
            return topic in {"/odom", "odom"}

        @staticmethod
        def latest(_topic):
            return odom, 0.0, 1

    return ObjectWatchdog(registry=None, hub=Hub())


def test_rotation_above_two_degrees_suppresses_curved_motion():
    """Rotation wins even when simultaneous translation exceeds 0.2 m/s."""
    watchdog = _motion_watchdog(linear_m_s=0.3, angular_deg_s=2.1)
    assert watchdog._detect_motion() == "rotating"


def test_translation_below_rotation_threshold_remains_moving():
    """Straight-enough translation remains eligible for object memory."""
    watchdog = _motion_watchdog(linear_m_s=0.3, angular_deg_s=1.9)
    assert watchdog._detect_motion() == "moving"


def test_rotation_threshold_is_strictly_greater_than_two_degrees():
    """Exactly 2 °/s does not cross the requested greater-than threshold."""
    watchdog = _motion_watchdog(linear_m_s=0.0, angular_deg_s=2.0)
    assert watchdog._detect_motion() == "static"


def _pose_watchdog(yaws, timestamps):
    """Build a watchdog whose map pose advances through the supplied samples."""
    from types import SimpleNamespace

    samples = iter(zip(yaws, timestamps))

    class Hub:
        @staticmethod
        def has(topic):
            return topic == "pose"

        @staticmethod
        def latest(_topic):
            yaw, stamp = next(samples)
            pose = SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=float(np.sin(yaw / 2.0)),
                    w=float(np.cos(yaw / 2.0)),
                ),
            )
            return SimpleNamespace(pose=SimpleNamespace(pose=pose)), stamp, 1

    return ObjectWatchdog(registry=None, hub=Hub())


def test_pose_delta_detects_rotation_when_odometry_twist_is_absent():
    """Consecutive map headings enforce the rotation gate without odometry twist."""
    import time

    now = time.time()
    watchdog = _pose_watchdog([0.0, np.deg2rad(3.0)], [now, now + 1.0])
    assert watchdog._detect_motion() == "static"
    assert watchdog._detect_motion() == "rotating"


def test_rotating_tick_does_not_snapshot_objects():
    """The watchdog exits before object capture while rotation is active."""
    class Registry:
        async def snapshot(self):
            raise AssertionError("rotation must suppress object snapshots")

    watchdog = ObjectWatchdog(registry=Registry(), hub=None)
    watchdog._motion_state = "rotating"
    asyncio.run(watchdog._tick())
