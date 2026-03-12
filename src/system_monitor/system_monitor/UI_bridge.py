from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from turtlebot4_beep_interfaces.msg import TargetEvent


# ============================================================
# 상태 저장
# ============================================================
@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0


@dataclass
class RobotState:
    robot_id: str
    name: str
    status: str = "Idle"
    pose: Pose = field(default_factory=Pose)
    battery: int = 0
    current_goal: int = 0


robots: Dict[str, RobotState] = {
    "amr1": RobotState(robot_id="amr1", name="AMR_1", battery=90),
    "amr2": RobotState(robot_id="amr2", name="AMR_2", battery=70),
}

logs: Deque[dict] = deque(maxlen=1000)

latest_enemy_image_path: str | None = None
latest_enemy_image_lock = threading.Lock()

air_path_points: list[dict] = []
drop_points: list[dict] = []
map_points_lock = threading.Lock()

last_enemy_detect_time = 0.0
latest_amr2_alert: str = ""
latest_amr2_alert_time: float = 0.0

def add_log(robot: str, message: str) -> None:
    logs.append(
        {
            "id": int(time.time() * 1000),
            "time": time.strftime("%H:%M:%S"),
            "robot": robot.upper(),
            "message": message,
        }
    )


def get_recent_logs(limit: int = 10) -> list[dict]:
    return list(logs)[-limit:][::-1]


def get_all_logs() -> list[dict]:
    return list(logs)[::-1]


# ============================================================
# ROS Subscriber
# ============================================================
class UiBridgeNode(Node):
    def __init__(self):
        super().__init__("ui_bridge_node")

        # ----------------------------------------------------
        # webcam node 연결
        # ----------------------------------------------------
        self.create_subscription(
            TargetEvent,
            "/target_event",
            self.target_event_callback,
            10
        )

        self.create_subscription(
            String,
            "/webcam/enemy_image_path",
            self.enemy_image_callback,
            10
        )

        # ----------------------------------------------------
        # AMR 상태 / pose / battery
        # ----------------------------------------------------
        self.create_subscription(
            String,
            "/amr1/state",
            self.amr1_state_callback,
            10
        )

        self.create_subscription(
            String,
            "/amr2/state",
            self.amr2_state_callback,
            10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amr1/pose",
            self.amr1_pose_callback,
            10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amr2/pose",
            self.amr2_pose_callback,
            10
        )

        self.create_subscription(
            Int32,
            "/amr1/battery",
            self.amr1_battery_callback,
            10
        )

        self.create_subscription(
            Int32,
            "/amr2/battery",
            self.amr2_battery_callback,
            10
        )

        # ----------------------------------------------------
        # AMR_2 mission 관련
        # ----------------------------------------------------
        self.create_subscription(
            Int32,
            "/amr2/current_goal",
            self.amr2_goal_callback,
            10
        )

        self.create_subscription(
            Bool,
            "/amr2/object_detected",
            self.amr2_object_detected_callback,
            10
        )

        self.create_subscription(
            PointStamped,
            "/amr1/enemy_pose",
            self.air_pose_callback,
            10
        )

        self.create_subscription(
            PointStamped,
            "/amr2/object_pose",
            self.drop_pose_callback,
            10
        )

        self.create_subscription(
            String,
            "/amr2/alert",
            self.amr2_alert_callback,
            10
        )

    # --------------------------------------------------------
    # webcam callback
    # --------------------------------------------------------
    def target_event_callback(self, msg: TargetEvent):
        global last_enemy_detect_time

        if msg.detected:
            last_enemy_detect_time = time.time()
            robots["amr1"].status = "Enemy Detected"
            add_log("webcam", f"enemy 탐지 direction={msg.direction}")

    def enemy_image_callback(self, msg: String):
        global latest_enemy_image_path

        path = msg.data.strip()
        print(f"[UI_BRIDGE] received image path: {path}")

        if path and os.path.exists(path):
            with latest_enemy_image_lock:
                latest_enemy_image_path = path
            print(f"[UI_BRIDGE] image path updated: {path}")
            add_log("webcam", f"탐지 이미지 수신: {os.path.basename(path)}")
        else:
            print(f"[UI_BRIDGE] invalid image path: {path}")

    # --------------------------------------------------------
    # amr 상태 callback
    # --------------------------------------------------------
    def amr1_state_callback(self, msg: String):
        robots["amr1"].status = msg.data

    def amr2_state_callback(self, msg: String):
        robots["amr2"].status = msg.data

    def amr1_pose_callback(self, msg: PoseWithCovarianceStamped):
        robots["amr1"].pose.x = msg.pose.pose.position.x
        robots["amr1"].pose.y = msg.pose.pose.position.y

    def amr2_pose_callback(self, msg: PoseWithCovarianceStamped):
        robots["amr2"].pose.x = msg.pose.pose.position.x
        robots["amr2"].pose.y = msg.pose.pose.position.y

    def amr1_battery_callback(self, msg: Int32):
        robots["amr1"].battery = max(0, min(100, msg.data))

    def amr2_battery_callback(self, msg: Int32):
        robots["amr2"].battery = max(0, min(100, msg.data))

    # --------------------------------------------------------
    # amr2 mission callback
    # --------------------------------------------------------
    def amr2_goal_callback(self, msg: Int32):
        robots["amr2"].current_goal = msg.data

    def amr2_object_detected_callback(self, msg: Bool):
        if msg.data:
            add_log("amr2", "미확인 물체 탐지")

    def air_pose_callback(self, msg: PointStamped):
        with map_points_lock:
            air_path_points.append({"x": msg.point.x, "y": msg.point.y})
            if len(air_path_points) > 300:
                del air_path_points[:-300]

    def drop_pose_callback(self, msg: PointStamped):
        with map_points_lock:
            drop_points.append({"x": msg.point.x, "y": msg.point.y})
            if len(drop_points) > 100:
                del drop_points[:-100]

    def amr2_alert_callback(self, msg: String):
        global latest_amr2_alert, latest_amr2_alert_time

        latest_amr2_alert = msg.data
        latest_amr2_alert_time = time.time()

        add_log("amr2", f"경고: {msg.data}")


def ros_spin_worker():
    if not rclpy.ok():
        rclpy.init()

    node = UiBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


def start_bridge_thread():
    t = threading.Thread(target=ros_spin_worker, daemon=True)
    t.start()
    return t