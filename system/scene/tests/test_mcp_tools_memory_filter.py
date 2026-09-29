# SPDX-License-Identifier: MulanPSL-2.0
"""Scene Hook object-memory filtering regressions."""

from scene_service.mcp_tools import _memory_visible_objects
from scene_service.state.object_registry import BBox3D, Pose3D, SceneObject


def _object(object_id: str, label: str) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        cls=label,
        pose=Pose3D(x=0.0, y=0.0, z=0.0, frame_id="map"),
        bbox=BBox3D(),
        confidence=1.0,
        first_seen=0.0,
        last_seen=0.0,
    )


def test_scene_hook_excludes_self_and_picture_frame_noise():
    """Whole-scene saves apply the same exclusions as ObjectWatchdog."""
    objects = [
        _object("scene.object.robot_001", "robot"),
        _object("scene.object.picture_frame_001", "picture frame"),
        _object("scene.object.chair_001", "chair"),
    ]
    assert [obj.cls for obj in _memory_visible_objects(objects)] == ["chair"]
