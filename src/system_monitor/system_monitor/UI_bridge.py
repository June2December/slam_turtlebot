from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, String, Int32, Float32
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from amr_interfaces.msg import TargetEvent


# ============================================================
# topic 설정 (이전 버전 기준으로 수정)
# ============================================================
TARGET_EVENT = "/target_event"
WEBCAM_IMAGE_PATH = "/webcam/image_path"

AMR1_STATE = "/amr1/state"
AMR2_STATE = "/amr2/state"

AMR1_TRACK_POSE = "/amr1/track_pose"
AMR2_OBJECT_POSE = "/amr2/object_pose"
AMR2_CURRENT_GOAL = "/amr2/current_goal"
AMR2_UNKNOWN = "/amr2/unknown"

BATTERY_AMR1 = "/robot4/battery_state/percentage"
BATTERY_AMR2 = "/robot1/battery_state/percentage"

# 토픽명이랑 헷갈려서 매핑
ROS_TO_UI_ROBOT_MAP = {
    "robot4": "amr1",
    "robot1": "amr2",
}


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
    "amr1": RobotState(robot_id="amr1", name="AMR_1", battery= 90),
    "amr2": RobotState(robot_id="amr2", name="AMR_2", battery= 70),
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


def normalize_battery_percentage(value: float) -> int:
    """
    0.0~1.0 이면 0~100으로 변환
    이미 0~100 범위면 그대로 사용
    """
    if 0.0 <= value <= 1.0:
        value *= 100.0
    return int(max(0.0, min(100.0, value)))


# ============================================================
# ROS Subscriber Node
# ============================================================
class UiBridgeNode(Node):
    def __init__(self):
        super().__init__("ui_bridge_node")
        self.get_logger().info("UI bridge node started")

        # ----------------------------------------------------
        # webcam / trigger
        # ----------------------------------------------------
        self.create_subscription(
            TargetEvent,
            TARGET_EVENT,
            self.target_event_callback,
            10,
        )

        self.create_subscription(
            String,
            WEBCAM_IMAGE_PATH,
            self.enemy_image_callback,
            10,
        )

        # ----------------------------------------------------
        # AMR 상태 / battery
        # ----------------------------------------------------
        self.create_subscription(
            String,
            AMR1_STATE,
            self.amr1_state_callback,
            10,
        )

        self.create_subscription(
            String,
            AMR2_STATE,
            self.amr2_state_callback,
            10,
        )

        self.create_subscription(
            Float32,
            BATTERY_AMR1,
            self.amr1_battery_callback,
            10,
        )

        self.create_subscription(
            Float32,
            BATTERY_AMR2,
            self.amr2_battery_callback,
            10,
        )

        # ----------------------------------------------------
        # mission 관련
        # ----------------------------------------------------
        self.create_subscription(
            Int32,
            AMR2_CURRENT_GOAL,
            self.amr2_goal_callback,
            10,
        )

        self.create_subscription(
            PointStamped,
            AMR1_TRACK_POSE,
            self.air_pose_callback,
            10,
        )

        self.create_subscription(
            PointStamped,
            AMR2_OBJECT_POSE,
            self.drop_pose_callback,
            10,
        )

        self.create_subscription(
            Bool,
            AMR2_UNKNOWN,
            self.amr2_unknown_callback,
            10,
        )

    # ========================================================
    # webcam callback
    # ========================================================
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

    # ========================================================
    # amr 상태 callback
    # ========================================================
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

    def amr1_battery_callback(self, msg: Float32):
        robots["amr1"].battery = normalize_battery_percentage(float(msg.data))

    def amr2_battery_callback(self, msg: Float32):
        robots["amr2"].battery = normalize_battery_percentage(float(msg.data))

    # ========================================================
    # mission callback
    # ========================================================
    def amr2_goal_callback(self, msg: Int32):
        robots["amr2"].current_goal = msg.data

    def amr2_object_detected_callback(self, msg: Bool):
        if msg.data:
            add_log("amr2", "미확인 물체 탐지")

    def air_pose_callback(self, msg: PointStamped):
        robots["amr1"].pose.x = msg.point.x
        robots["amr1"].pose.y = msg.point.y
        robots["amr1"].pose.w = msg.point.z

        with map_points_lock:
            air_path_points.append({"x": msg.point.x, "y": msg.point.y})
            if len(air_path_points) > 300:
                del air_path_points[:-300]

    def drop_pose_callback(self, msg: PointStamped):
        robots["amr2"].pose.x = msg.point.x
        robots["amr2"].pose.y = msg.point.y
        robots["amr2"].pose.w = msg.point.z

        with map_points_lock:
            drop_points.append({"x": msg.point.x, "y": msg.point.y})
            if len(drop_points) > 100:
                del drop_points[:-100]

    def amr2_unknown_callback(self, msg: Bool):
        global latest_amr2_alert, latest_amr2_alert_time

        if msg.data:
            latest_amr2_alert = "Unknown detected"
            latest_amr2_alert_time = time.time()

            print("[UI_BRIDGE] unknown detected: True")
            add_log("amr2", "경고: Unknown detected")
        else:
            latest_amr2_alert = ""
            latest_amr2_alert_time = 0.0


# ============================================================
# background thread for flask
# ============================================================
def ros_spin_worker():
    node = None
    executor = None

    try:
        if not rclpy.ok():
            rclpy.init()

        node = UiBridgeNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        print("[UI_BRIDGE] ros_spin_worker started")
        print(f"[UI_BRIDGE] node name: {node.get_name()}")

        executor.spin()

    except Exception as e:
        print(f"[UI_BRIDGE] worker exception: {e}")

    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass

        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


def start_bridge_thread():
    t = threading.Thread(target=ros_spin_worker, daemon=True)
    t.start()
    return t


# ============================================================
# standalone execution
# ============================================================
def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    node = UiBridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()