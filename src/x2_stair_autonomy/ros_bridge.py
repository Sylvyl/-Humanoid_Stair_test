from __future__ import annotations

import os
import shutil
from threading import Thread
import time


def start_ros_bridge(runtime):
    """Start the optional read-only ROS bridge in a daemon thread."""
    thread = Thread(target=_spin, args=(runtime,), name="x2-read-only-ros", daemon=True)
    thread.start()
    return thread


def _spin(runtime):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image, Imu, PointCloud2

    from aimdk_msgs.msg import CommonRequest, JointStateArray, PmuState
    from aimdk_msgs.srv import GetCurrentInputSource, GetMcAction
    from x2_stair_interfaces.msg import StairEstimate as StairEstimateMsg

    class ReadOnlyBridge(Node):
        def __init__(self):
            super().__init__("x2_stair_dashboard_bridge")
            qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=2)
            reliable = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                  history=QoSHistoryPolicy.KEEP_LAST, depth=2)
            self.last_robot_sample = 0.0
            self.mode_pending = False
            self.source_pending = False
            self.create_subscription(StairEstimateMsg, "/x2/stairs/estimate", self.stairs, reliable)
            self.create_subscription(PointCloud2, "/aima/hal/sensor/lidar_chest_front/lidar_pointcloud",
                                     lambda msg: self.touch_sensor("lidar"), qos)
            self.create_subscription(Image, "/aima/hal/sensor/rgbd_head_front/depth_image",
                                     lambda msg: self.touch_sensor("depth"), reliable)
            self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.imu, qos)
            self.create_subscription(JointStateArray, "/aima/hal/joint/leg/state", self.joints, qos)
            self.create_subscription(PmuState, "/aima/hal/pmu/state", self.pmu, qos)
            self.mode_client = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")
            self.source_client = self.create_client(GetCurrentInputSource, "/aimdk_5Fmsgs/srv/GetCurrentInputSource")
            self.create_timer(1.0, self.poll_services)
            self.create_timer(0.2, self.health)
            self.get_logger().warning("Dashboard ROS bridge is read-only; no robot command publisher exists")

        def touch_sensor(self, name):
            now = time.time()
            with runtime.lock:
                runtime.telemetry["sensors"][name] = {"stamp": now}
                self.last_robot_sample = time.monotonic()

        def imu(self, msg):
            self.touch_sensor("imu_torso")
            with runtime.lock:
                runtime.telemetry["imu"] = {
                    "angular_velocity": [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                    "linear_acceleration": [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                }

        def joints(self, msg):
            self.touch_sensor("joints")
            values = []
            for joint in msg.joints:
                values.append({"name": joint.name, "position": joint.position,
                               "velocity": joint.velocity, "effort": joint.effort})
            with runtime.lock:
                runtime.telemetry["joints"] = {"available": True, "count": len(values),
                                                "values": values, "violations": []}

        def pmu(self, msg):
            self.touch_sensor("pmu")
            with runtime.lock:
                runtime.telemetry["power"] = {
                    "available": True,
                    "battery_percent": int(msg.battery_remaining_capacity_percentage),
                    "battery_voltage": float(msg.battery_pack_voltage),
                    "battery_current": float(msg.battery_current),
                    "battery_temperature": float(msg.battery_temperature),
                    "pmu_temperature": float(msg.pmu_temperature),
                    "pmu_bool_status": int(msg.pmu_bool_status),
                    "bms_status_bits": int(msg.bms_status_bits),
                }

        def stairs(self, msg):
            from .models import StairEstimate
            with runtime.lock:
                runtime.stairs = StairEstimate(
                    available=bool(msg.available), supported=bool(msg.supported),
                    direction=msg.direction or None, distance_m=float(msg.distance_m),
                    yaw_deg=float(msg.yaw_deg), riser_m=float(msg.riser_m),
                    tread_m=float(msg.tread_m), width_m=float(msg.width_m),
                    step_count=int(msg.step_count), landing_detected=bool(msg.landing_detected),
                    confidence=float(msg.confidence), rejection_reason=msg.rejection_reason or None,
                    stamp=time.time(),
                )

        def request(self):
            request = CommonRequest()
            request.header.stamp = self.get_clock().now().to_msg()
            return request

        def poll_services(self):
            if self.mode_client.service_is_ready() and not self.mode_pending:
                self.mode_pending = True
                req = GetMcAction.Request()
                req.request = self.request()
                future = self.mode_client.call_async(req)
                future.add_done_callback(self.mode_result)
            if self.source_client.service_is_ready() and not self.source_pending:
                self.source_pending = True
                req = GetCurrentInputSource.Request()
                req.request = self.request()
                future = self.source_client.call_async(req)
                future.add_done_callback(self.source_result)

        def mode_result(self, future):
            self.mode_pending = False
            try:
                mode = future.result().info.action_desc
                with runtime.lock:
                    runtime.telemetry["robot"]["mode"] = mode
            except Exception as exc:
                self.get_logger().error(f"GetMcAction failed: {exc}")

        def source_result(self, future):
            self.source_pending = False
            try:
                source = future.result().input_source.name
                with runtime.lock:
                    runtime.telemetry["robot"]["input_source"] = source
            except Exception as exc:
                self.get_logger().error(f"GetCurrentInputSource failed: {exc}")

        def health(self):
            connected = time.monotonic() - self.last_robot_sample <= 1.0
            usage = shutil.disk_usage(os.path.expanduser("~"))
            with runtime.lock:
                runtime.telemetry["robot"]["connected"] = connected
                runtime.telemetry["host"] = {
                    "available": True,
                    "load_average": list(os.getloadavg()),
                    "disk_free_bytes": usage.free,
                    "disk_total_bytes": usage.total,
                }
            runtime.refresh_safety()

    rclpy.init()
    node = ReadOnlyBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

