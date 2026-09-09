from __future__ import annotations

import math
import os
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2

from aimdk_msgs.msg import JointStateArray, PmuState
from x2_stair_interfaces.msg import SafetyStatus
from x2_stair_interfaces.msg import StairEstimate as StairEstimateMsg

core = os.environ.get("X2_STAIR_CORE")
if core and core not in sys.path:
    sys.path.insert(0, core)
from x2_stair_autonomy.models import StairProfile
from x2_stair_autonomy.perception import estimate_stairs


def _euler(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    return roll, math.asin(sinp)


class ShadowObserver(Node):
    """Subscribes to robot state and publishes derived non-control status."""

    def __init__(self):
        super().__init__("x2_stair_shadow_observer")
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=2)
        self.profile = StairProfile()
        self.roll = 0.0
        self.pitch = 0.0
        self.last = {}
        self.stair_pub = self.create_publisher(StairEstimateMsg, "/x2/stairs/estimate", 10)
        self.safety_pub = self.create_publisher(SafetyStatus, "/x2/stairs/safety", 10)
        self.create_subscription(PointCloud2, "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud", self.cloud, qos)
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.imu, qos)
        self.create_subscription(JointStateArray, "/aima/hal/joint/leg/state", self.joints, qos)
        self.create_subscription(PmuState, "/aima/hal/pmu/state", self.pmu, qos)
        self.create_timer(0.5, self.publish_safety)
        self.get_logger().warning("SHADOW MODE: this node has no robot command publisher")

    def imu(self, msg):
        self.roll, self.pitch = _euler(msg.orientation)
        self.last["imu"] = time.monotonic()

    def joints(self, msg):
        self.last["joints"] = time.monotonic()
        self.last["joint_count"] = len(msg.joints)

    def pmu(self, msg):
        self.last["pmu"] = time.monotonic()
        self.last["battery_percent"] = int(msg.battery_remaining_capacity_percentage)

    def cloud(self, msg):
        raw_points = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        # Humble returns a structured NumPy array. Index by field name so
        # extra LiDAR fields with different datatypes cannot break XYZ reads.
        if getattr(getattr(raw_points, "dtype", None), "names", None):
            points = (
                (float(point["x"]), float(point["y"]), float(point["z"]))
                for point in raw_points
            )
        else:
            points = (
                (float(point[0]), float(point[1]), float(point[2]))
                for point in raw_points
            )
        result = estimate_stairs(points, self.profile, roll_rad=self.roll, pitch_rad=self.pitch)
        out = StairEstimateMsg()
        out.stamp = self.get_clock().now().to_msg()
        out.available = result.available
        out.supported = result.supported
        out.direction = result.direction or ""
        out.distance_m = float(result.distance_m or 0.0)
        out.yaw_deg = float(result.yaw_deg or 0.0)
        out.riser_m = float(result.riser_m or 0.0)
        out.tread_m = float(result.tread_m or 0.0)
        out.width_m = float(result.width_m or 0.0)
        out.step_count = result.step_count
        out.landing_detected = result.landing_detected
        out.confidence = result.confidence
        out.rejection_reason = result.rejection_reason or ""
        self.stair_pub.publish(out)
        self.last["lidar"] = time.monotonic()
        self.last["supported"] = result.supported

    def publish_safety(self):
        now = time.monotonic()
        required = ("imu", "joints", "pmu", "lidar")
        blockers = [f"{name} stale" for name in required if now - self.last.get(name, 0) > 0.25]
        if not self.last.get("supported", False):
            blockers.append("no supported stair estimate")
        blockers.extend(("vendor approval missing", "command adapter not compiled"))
        out = SafetyStatus()
        out.stamp = self.get_clock().now().to_msg()
        out.mission_state = "MONITORING"
        out.ready = False
        out.shadow_mode = True
        out.command_output_enabled = False
        out.blockers = blockers
        out.latched_fault = ""
        self.safety_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ShadowObserver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
