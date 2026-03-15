from __future__ import annotations

# ==========================================
# 기본 라이브러리 import
# ==========================================
import os
import time
from importlib import resources

# ==========================================
# ROS2 관련 import
# ==========================================
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ==========================================
# Flask 관련 import
# ==========================================
from flask import Flask, Response, jsonify, render_template_string, request, session, send_file

# UI 상태를 ROS topic으로부터 받아오는 bridge 모듈
from system_monitor import UI_bridge


# ============================================================
# Flask 앱 기본 설정
# ============================================================
app = Flask(__name__)
app.secret_key = "amr-monitor-demo-secret-key"

VALID_USERS = {
    "admin": {"password": "1234", "name": "관리자"},
    "rokey": {"password": "1234", "name": "상황근무자"},
}

HOST = "0.0.0.0"
PORT = 5000

MAP_PNG_PATH = "/home/sinya/slam_turtlebot/src/system_monitor/system_monitor/third_map.png"


# ============================================================
# ROS command node
# ============================================================
# 현재 UI에는 버튼만 남기고 실제 동작은 막을 예정이므로
# publisher 노드는 구조 유지용으로만 남겨둠
# ============================================================
class UICommandNode(Node):
    def __init__(self):
        super().__init__("ui_command_node")

        self.amr1_teleop_pub = self.create_publisher(String, "/amr1/teleop_cmd", 10)
        self.amr1_start_pub = self.create_publisher(String, "/amr1/start_cmd", 10)
        self.amr1_home_pub = self.create_publisher(String, "/amr1/home_cmd", 10)

        self.amr2_teleop_pub = self.create_publisher(String, "/amr2/teleop_cmd", 10)
        self.amr2_start_pub = self.create_publisher(String, "/amr2/start_cmd", 10)
        self.amr2_home_pub = self.create_publisher(String, "/amr2/home_cmd", 10)

    def send_teleop(self, robot_id: str, direction: str):
        # 실제 publish 막음
        self.get_logger().info(f"[NO-OP] teleop blocked: {robot_id} -> {direction}")

    def send_start(self, robot_id: str):
        # 실제 publish 막음
        self.get_logger().info(f"[NO-OP] start blocked: {robot_id}")

    def send_home(self, robot_id: str):
        # HOME도 막고 싶으면 여기처럼 유지
        self.get_logger().info(f"[NO-OP] home blocked: {robot_id}")


command_node: UICommandNode | None = None


def init_ui_command():
    global command_node

    if command_node is not None:
        return

    if not rclpy.ok():
        rclpy.init()

    command_node = UICommandNode()


# ============================================================
# HTML 로드
# ============================================================
def load_html(filename: str) -> str:
    return resources.files("system_monitor").joinpath(filename).read_text(encoding="utf-8")


PAGE_DASHBOARD = load_html("UI.html")
PAGE_ALL_LOG = load_html("UI_all_log.html")


# ============================================================
# 공통 유틸
# ============================================================
def is_logged_in() -> bool:
    return bool(session.get("logged_in"))


# ============================================================
# 페이지 라우팅
# ============================================================
@app.route("/")
@app.route("/login")
@app.route("/dashboard")
def dashboard_page():
    return render_template_string(PAGE_DASHBOARD)


@app.route("/logs")
def logs_page():
    return render_template_string(PAGE_ALL_LOG)


# ============================================================
# 지도(map) 관련 API
# ============================================================
@app.route("/api/map_info")
def api_map_info():
    return jsonify({
        "resolution": 0.05,
        "origin_x": -3.16,
        "origin_y": -0.174,
        "width": 128,
        "height": 123
    })


@app.route("/map_image")
def map_image():
    return send_file(
        MAP_PNG_PATH,
        mimetype="image/png",
        max_age=0
    )


# ============================================================
# Webcam snapshot API
# ============================================================
@app.route("/snapshot/webcam")
def snapshot_webcam():
    with UI_bridge.latest_enemy_image_lock:
        image_path = UI_bridge.latest_enemy_image_path

    if image_path and os.path.exists(image_path):
        return send_file(image_path, mimetype="image/jpeg", max_age=0)

    return Response(status=204)


# ============================================================
# 인증(auth) API
# ============================================================
@app.route("/api/session")
def api_session():
    return jsonify(
        {
            "logged_in": is_logged_in(),
            "username": session.get("username", ""),
        }
    )


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    user_id = (data.get("user_id") or "").strip()
    password = (data.get("password") or "").strip()

    user = VALID_USERS.get(user_id)
    if user is None or user["password"] != password:
        return jsonify({"ok": False, "error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

    session["logged_in"] = True
    session["user_id"] = user_id
    session["username"] = user["name"]

    UI_bridge.add_log(user_id, "사용자 로그인")

    return jsonify({"ok": True, "username": user["name"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    username = session.get("username", "사용자")

    if session.get("logged_in"):
        UI_bridge.add_log(session.get("user_id", "user"), f"{username} 로그아웃")

    session.clear()
    return jsonify({"ok": True})


# ============================================================
# 상태(state) API
# ============================================================
@app.route("/api/state")
def api_state():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    now = time.time()

    if now - UI_bridge.last_enemy_detect_time > 2.0:
        if UI_bridge.robots["amr1"].status == "Enemy Detected":
            UI_bridge.robots["amr1"].status = "Idle"

    with UI_bridge.map_points_lock:
        air_points = list(UI_bridge.air_path_points)
        drop_points = list(UI_bridge.drop_points)

    alert_active = bool(UI_bridge.latest_amr2_alert)

    return jsonify(
        {
            "amr1": {
                "status": UI_bridge.robots["amr1"].status,
                "battery": UI_bridge.robots["amr1"].battery,
            },
            "amr2": {
                "status": UI_bridge.robots["amr2"].status,
                "battery": UI_bridge.robots["amr2"].battery,
                "current_goal": UI_bridge.robots["amr2"].current_goal,
            },
            "alert": {
                "active": alert_active,
                "message": UI_bridge.latest_amr2_alert if alert_active else "",
                "time": UI_bridge.latest_amr2_alert_time,
            },
            "recent_logs": UI_bridge.get_recent_logs(10),
            "all_logs": UI_bridge.get_all_logs(),
            "air_path_points": air_points,
            "drop_points": drop_points,
        }
    )


# ============================================================
# alert clear API
# ============================================================
@app.route("/api/clear_alert", methods=["POST"])
def api_clear_alert():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    UI_bridge.latest_amr2_alert = ""
    UI_bridge.latest_amr2_alert_time = 0.0
    UI_bridge.add_log("system", "경고 해제")

    return jsonify({"ok": True})


# ============================================================
# command API
# ============================================================
# 화면에는 버튼 유지
# 하지만 실제 ROS publish는 하지 않음
# ============================================================

@app.route("/api/teleop", methods=["POST"])
def api_teleop():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(force=True)
    robot_id = data.get("robot_id")
    direction = data.get("direction")

    if robot_id not in {"amr1", "amr2"}:
        return jsonify({"ok": False, "error": "invalid robot id"}), 400

    if direction not in {"forward", "backward", "left", "right", "stop"}:
        return jsonify({"ok": False, "error": "invalid direction"}), 400

    # publish 안 함
    UI_bridge.add_log(robot_id, f"[비활성] 수동 조작 버튼 클릭: {direction}")

    return jsonify({"ok": True, "message": "teleop disabled"})


@app.route("/api/start", methods=["POST"])
def api_start():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(force=True)
    robot_id = data.get("robot_id")

    if robot_id not in {"amr1", "amr2"}:
        return jsonify({"ok": False, "error": "invalid robot id"}), 400

    # publish 안 함
    UI_bridge.add_log(robot_id, "[비활성] START 버튼 클릭")

    return jsonify({"ok": True, "message": "start disabled"})


@app.route("/api/home", methods=["POST"])
def api_home():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(force=True)
    robot_id = data.get("robot_id")

    if robot_id not in {"amr1", "amr2"}:
        return jsonify({"ok": False, "error": "invalid robot id"}), 400

    # HOME도 비활성화하려면 publish 안 함
    UI_bridge.add_log(robot_id, "[비활성] HOME 버튼 클릭")

    return jsonify({"ok": True, "message": "home disabled"})


# ============================================================
# 시작
# ============================================================
def bootstrap() -> None:
    UI_bridge.add_log("amr1", "UI 시스템 시작")
    UI_bridge.add_log("amr2", "UI 시스템 시작")

    UI_bridge.start_bridge_thread()

    # 버튼은 남기되 실제 명령 publish는 막을 거라면
    # publisher 노드 자체를 안 만들어도 됨
    # init_ui_command()


def main():
    bootstrap()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()