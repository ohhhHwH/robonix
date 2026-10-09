#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""Execute one odometry-bounded manual TIAGo chassis step in Webots."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def parse_args() -> argparse.Namespace:
    """Parse one exclusive distance or rotation command."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--forward-m", type=float)
    mode.add_argument("--rotate-deg", type=float)
    parser.add_argument("--linear-speed", type=float, default=0.25)
    parser.add_argument("--angular-speed", type=float, default=0.5)
    return parser.parse_args()


def normalize_angle(value: float) -> float:
    """Wrap an angle to the closed-open interval [-pi, pi)."""
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PoseTracker:
    """Track planar odometry and unwrapped rotation during one command."""

    x: float | None = None
    y: float | None = None
    yaw: float | None = None
    accumulated_yaw: float = 0.0

    def update(self, message: Odometry) -> None:
        """Store the latest pose and integrate the shortest yaw delta."""
        pose = message.pose.pose
        orientation = pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        if self.yaw is not None:
            self.accumulated_yaw += normalize_angle(yaw - self.yaw)
        self.x = float(pose.position.x)
        self.y = float(pose.position.y)
        self.yaw = yaw


def publish_stop(node, publisher) -> None:
    """Publish repeated zero commands so every subscriber observes the stop."""
    for _ in range(8):
        publisher.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.03)


def wait_until_ready(node, publisher, tracker, timeout_s: float = 5.0) -> None:
    """Wait for both `/cmd_vel` discovery and the first odometry sample."""
    deadline = time.monotonic() + timeout_s
    while publisher.get_subscription_count() == 0 or tracker.yaw is None:
        if time.monotonic() >= deadline:
            raise RuntimeError("manual step prerequisites unavailable")
        rclpy.spin_once(node, timeout_sec=0.1)


def run_forward(node, publisher, tracker, distance_m: float, speed_m_s: float) -> dict:
    """Drive straight until projected odometry displacement reaches the target."""
    start_x, start_y, start_yaw = tracker.x, tracker.y, tracker.yaw
    target = abs(distance_m)
    direction = 1.0 if distance_m >= 0.0 else -1.0
    deadline = time.monotonic() + target / speed_m_s * 3.0 + 5.0
    progress = 0.0
    while progress < max(0.0, target - 0.02):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"forward step blocked at {progress:.3f}/{target:.3f} m")
        rclpy.spin_once(node, timeout_sec=0.02)
        dx = float(tracker.x - start_x)
        dy = float(tracker.y - start_y)
        progress = direction * (dx * math.cos(start_yaw) + dy * math.sin(start_yaw))
        remaining = max(0.0, target - progress)
        command = Twist()
        command.linear.x = direction * min(speed_m_s, max(0.08, remaining * 0.8))
        heading_error = normalize_angle(start_yaw - tracker.yaw)
        command.angular.z = max(-0.3, min(0.3, heading_error * 1.5))
        publisher.publish(command)
        time.sleep(0.03)
    return {"actual_m": progress, "x": tracker.x, "y": tracker.y, "yaw": tracker.yaw}


def run_rotation(node, publisher, tracker, angle_deg: float, speed_rad_s: float) -> dict:
    """Rotate until integrated odometry yaw reaches the requested angle."""
    tracker.accumulated_yaw = 0.0
    target = abs(math.radians(angle_deg))
    direction = 1.0 if angle_deg >= 0.0 else -1.0
    deadline = time.monotonic() + target / speed_rad_s * 3.0 + 5.0
    progress = 0.0
    while progress < max(0.0, target - math.radians(1.5)):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"rotation step blocked at {math.degrees(progress):.1f}/{abs(angle_deg):.1f} deg"
            )
        rclpy.spin_once(node, timeout_sec=0.02)
        progress = direction * tracker.accumulated_yaw
        remaining = max(0.0, target - progress)
        command = Twist()
        command.angular.z = direction * min(speed_rad_s, max(0.12, remaining * 1.2))
        publisher.publish(command)
        time.sleep(0.03)
    return {
        "actual_deg": math.degrees(progress),
        "x": tracker.x,
        "y": tracker.y,
        "yaw": tracker.yaw,
    }


def main() -> int:
    """Run one closed-loop movement and always leave `/cmd_vel` stopped."""
    args = parse_args()
    if args.linear_speed <= 0.0 or args.angular_speed <= 0.0:
        raise ValueError("speeds must be positive")

    rclpy.init()
    node = rclpy.create_node("robonix_manual_chassis_step")
    publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    tracker = PoseTracker()
    node.create_subscription(Odometry, "/odom", tracker.update, 10)
    try:
        wait_until_ready(node, publisher, tracker)
        if args.forward_m is not None:
            result = run_forward(node, publisher, tracker, args.forward_m, args.linear_speed)
            result.update({"status": "done", "mode": "forward_m", "value": args.forward_m})
        else:
            result = run_rotation(node, publisher, tracker, args.rotate_deg, args.angular_speed)
            result.update({"status": "done", "mode": "rotate_deg", "value": args.rotate_deg})
        print(json.dumps(result))
        return 0
    finally:
        publish_stop(node, publisher)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
