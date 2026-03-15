from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque

import requests

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, String, Int32
from geometry_msgs.msg import PointStamped
from amr_interfaces.msg import TargetEvent


# ============================================================
# Firebase 설정
# ============================================================
FIREBASE_BASE_URL = "https://gas1-994d9-default-rtdb.asia-southeast1.firebasedatabase.app"
LATEST_FIREBASE_URL = f'{FIREBASE_BASE_URL}/gas_logs.json?orderBy="$key"&limitToLast=1'
GAS_THRESHOLD = 1000.0
CHECK_INTERVAL = 2.0

startup_latest_key: str | None = None
last_processed_key: str | None = None


# ============================================================
# Robot 상태 저장용 데이터 클래스
# ============================================================
@dataclass
class RobotState:
    robot_id: str
    name: str
    status: str = "Idle"
    battery: int = 0
    current_goal: int = 0


# ============================================================
# 로봇 상태 저장
# ============================================================
robots: Dict[str, RobotState] = {
    "amr1": RobotState(robot_id="amr1", name="AMR_1", battery=90),
    "amr2": RobotState(robot_id="amr2", name="AMR_2", battery=70),
}


# ============================================================
# 로그 저장
# ============================================================
logs: Deque[dict] = deque(maxlen=1000)


# ============================================================
# webcam 이미지 공유 변수
# ============================================================
latest_enemy_image_path: str | None = None
latest_enemy_image_lock = threading.Lock()


# ============================================================
# Map 좌표 저장
# ============================================================
air_path_points: list[dict] = []
drop_points: list[dict] = []
map_points_lock = threading.Lock()


# ============================================================
# detection / alert 상태
# ============================================================
last_enemy_detect_time = 0.0
latest_amr2_alert: str = ""
latest_amr2_alert_time: float = 0.0


# ============================================================
# 로그 유틸
# ============================================================
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
# ROS Subscriber / Publisher Node
# ============================================================
class UiBridgeNode(Node):
    def __init__(self):
        super().__init__("ui_bridge_node")
        self.get_logger().info("UI bridge node started")

        # robot4 target_event 로그 중복 방지
        self.robot4_target_logged = False
        self.prev_amr2_goal = None

        # ========================================================
        # Firebase -> ROS alert publisher
        # ========================================================
        self.amr2_alert_pub = self.create_publisher(
            Bool,
            "/amr2/unknown",
            10
        )

        self.initialize_startup_key()

        self.firebase_timer = self.create_timer(
            CHECK_INTERVAL,
            self.firebase_check_callback
        )

        # ========================================================
        # webcam detection 관련 토픽
        # robot4 것만 구독
        # ========================================================
        self.target_event_sub = self.create_subscription(
            TargetEvent,
            "/robot4/target_event",
            self.target_event_callback,
            10
        )

        self.enemy_image_sub = self.create_subscription(
            String,
            "/webcam/enemy_image_path",
            self.enemy_image_callback,
            10
        )

        # ========================================================
        # AMR 상태 토픽
        # ========================================================
        self.amr1_state_sub = self.create_subscription(
            String,
            "/amr1/state",
            self.amr1_state_callback,
            10
        )

        self.amr2_state_sub = self.create_subscription(
            String,
            "/amr2/state",
            self.amr2_state_callback,
            10
        )

        # ========================================================
        # 배터리 상태
        # ========================================================
        self.amr1_battery_sub = self.create_subscription(
            Int32,
            "/amr1/battery",
            self.amr1_battery_callback,
            10
        )

        self.amr2_battery_sub = self.create_subscription(
            Int32,
            "/amr2/battery",
            self.amr2_battery_callback,
            10
        )

        # ========================================================
        # AMR2 현재 탐색중인 goal 번호
        # ========================================================
        self.amr2_goal_sub = self.create_subscription(
            Int32,
            "/amr2/current_goal",
            self.amr2_goal_callback,
            10
        )

        # ========================================================
        # Map 표시용 좌표
        # ========================================================
        self.air_pose_sub = self.create_subscription(
            PointStamped,
            "/amr1/track_pose",
            self.air_pose_callback,
            10
        )

        self.drop_pose_sub = self.create_subscription(
            PointStamped,
            "/amr2/object_pose",
            self.drop_pose_callback,
            10
        )

        # ========================================================
        # AMR2 alert subscribe
        # ========================================================
        self.amr2_alert_sub = self.create_subscription(
            Bool,
            "/amr2/unknown",
            self.amr2_alert_callback,
            10
        )

    # ========================================================
    # Firebase 조회
    # ========================================================
    def get_latest_firebase_data(self):
        try:
            res = requests.get(LATEST_FIREBASE_URL, timeout=3)
            if res.status_code == 200:
                return res.json()
            self.get_logger().warning(f"Firebase HTTP error: {res.status_code}")
            return None
        except Exception as e:
            self.get_logger().error(f"Firebase read failed: {e}")
            return None

    # ========================================================
    # 시작 시점 기준 key 저장
    # ========================================================
    def initialize_startup_key(self):
        global startup_latest_key, last_processed_key

        data = self.get_latest_firebase_data()
        if not data:
            self.get_logger().info("Startup: Firebase latest data not found")
            startup_latest_key = None
            last_processed_key = None
            return

        key, _value = next(iter(data.items()))
        startup_latest_key = key
        last_processed_key = key

        self.get_logger().info(f"Startup baseline key: {startup_latest_key}")

    # ========================================================
    # Firebase 체크
    # ========================================================
    def firebase_check_callback(self):
        global last_processed_key

        data = self.get_latest_firebase_data()
        if not data:
            return

        key, value = next(iter(data.items()))

        if key == last_processed_key:
            return

        last_processed_key = key

        try:
            gas_value = float(value.get("gas_value", 0))
        except (TypeError, ValueError):
            gas_value = 0.0

        timestamp = str(value.get("timestamp", ""))

        self.get_logger().info(
            f"New Firebase value received: key={key}, gas={gas_value}, timestamp={timestamp}"
        )

        if gas_value > GAS_THRESHOLD:
            self.get_logger().warn(f"[GAS ALERT] {gas_value} ppm | {timestamp}")
            add_log("gas", f"가스 이상 감지: {gas_value} ppm")

            msg = Bool()
            msg.data = True

            self.amr2_alert_pub.publish(msg)
            time.sleep(0.1)
            self.amr2_alert_pub.publish(msg)

    # ========================================================
    # webcam callback
    # robot4 target_event만 받고, detected=True 로그는 1회만
    # detected=False 오면 다음 이벤트를 위해 리셋
    # ========================================================
    def target_event_callback(self, msg: TargetEvent):
        global last_enemy_detect_time

        if msg.detected:
            last_enemy_detect_time = time.time()
            robots["amr1"].status = "Enemy Detected"

            if not self.robot4_target_logged:
                add_log("webcam", f"robot4 enemy 탐지 direction={msg.direction}")
                self.robot4_target_logged = True
                self.get_logger().info(
                    f"robot4 target_event logged once | direction={msg.direction}"
                )
        else:
            # enemy 해제되면 다음 미션/다음 탐지에서 다시 1회 로그 가능
            if self.robot4_target_logged:
                self.get_logger().info("robot4 target_event reset")
            self.robot4_target_logged = False

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
    # AMR 상태 callback
    # ========================================================
    def amr1_state_callback(self, msg: String):
        robots["amr1"].status = msg.data

    def amr2_state_callback(self, msg: String):
        robots["amr2"].status = msg.data

    # ========================================================
    # 배터리 callback
    # ========================================================
    def amr1_battery_callback(self, msg: Int32):
        robots["amr1"].battery = max(0, min(100, msg.data))

    def amr2_battery_callback(self, msg: Int32):
        robots["amr2"].battery = max(0, min(100, msg.data))

    # ========================================================
    # current_goal callback
    # ========================================================
    def amr2_goal_callback(self, msg: Int32):
        robots["amr2"].current_goal = msg.data

        if self.prev_amr2_goal != msg.data:
            if msg.data > 0:
                add_log("amr2", f"({msg.data}/7) 탐색중")
            self.prev_amr2_goal = msg.data
    # ========================================================
    # Map 좌표 callback
    # ========================================================
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

        add_log("amr2", f"위험물 좌표 수신: x={msg.point.x:.2f}, y={msg.point.y:.2f}")

    # ========================================================
    # AMR2 alert callback
    # ========================================================
    def amr2_alert_callback(self, msg: Bool):
        global latest_amr2_alert, latest_amr2_alert_time

        if msg.data:
            latest_amr2_alert = "Hazardous gas detected"
            latest_amr2_alert_time = time.time()

            print("[UI_BRIDGE] alert received: True")
            add_log("amr2", "경고: Hazardous gas detected")
        else:
            latest_amr2_alert = ""

    # ========================================================
    # 종료 시 정리
    # ========================================================
    def destroy_node(self):
        self.get_logger().info("UI bridge node shutting down")
        super().destroy_node()


# ============================================================
# ROS spin thread
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


def start_bridge_thread():
    t = threading.Thread(target=ros_spin_worker, daemon=True)
    t.start()
    return t


# ============================================================
# standalone 실행
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