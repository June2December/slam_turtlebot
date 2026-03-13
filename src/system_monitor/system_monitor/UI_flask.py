from __future__ import annotations

import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, session, send_file

import UI_bridge
import UI_command

from amr_interfaces.msg import TargetEvent


app = Flask(__name__)
app.secret_key = "amr-monitor-demo-secret-key"

VALID_USERS = {
    "admin": {"password": "1234", "name": "상황근무자"},
    "donghyun": {"password": "1234", "name": "박동현"},
}

HOST = "0.0.0.0"
PORT = 5000

# alert가 들어온 뒤 이 시간 동안만 active로 표시
ALERT_TIMEOUT_SEC = 3.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# HTML 로드
# ============================================================
def load_html(filename: str) -> str:
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


PAGE_DASHBOARD = load_html("UI.html")
PAGE_ALL_LOG = load_html("UI_all_log.html")


# ============================================================
# 유틸
# ============================================================
def make_dummy_frame(text: str = "NO CAPTURE YET") -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 80, 140)

    cv2.putText(
        frame,
        "AMR DETECTION SNAPSHOT",
        (110, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (170, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (145, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def encode_jpg(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        fallback = make_dummy_frame("ENCODE ERROR")
        ok, buffer = cv2.imencode(".jpg", fallback)
    return buffer.tobytes()


def is_logged_in() -> bool:
    return bool(session.get("logged_in"))


# ============================================================
# 페이지
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
# snapshot
# ============================================================
@app.route("/snapshot/amr1")
def snapshot_amr1():
    with UI_bridge.latest_enemy_image_lock:
        image_path = UI_bridge.latest_enemy_image_path

    if image_path and os.path.exists(image_path):
        return send_file(image_path, mimetype="image/jpeg", max_age=0)

    return Response(
        encode_jpg(make_dummy_frame("NO DETECTION YET")),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.route("/snapshot/amr2")
def snapshot_amr2():
    return Response(
        encode_jpg(make_dummy_frame("NO CAMERA LINKED")),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# ============================================================
# auth api
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
# state api
# ============================================================
@app.route("/api/state")
def api_state():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    now = time.time()

    # enemy detect 상태 자동 복귀
    if now - UI_bridge.last_enemy_detect_time > 2.0:
        if UI_bridge.robots["amr1"].status == "Enemy Detected":
            UI_bridge.robots["amr1"].status = "Idle"

    with UI_bridge.map_points_lock:
        air_points = list(UI_bridge.air_path_points)
        drop_points = list(UI_bridge.drop_points)

    # 최근 alert만 활성 상태로 간주
    alert_active = (
        bool(UI_bridge.latest_amr2_alert)
        and (now - UI_bridge.latest_amr2_alert_time <= ALERT_TIMEOUT_SEC)
    )

    print(
        f"[API_STATE] alert_active={alert_active}, "
        f"message='{UI_bridge.latest_amr2_alert}', "
        f"time={UI_bridge.latest_amr2_alert_time}"
    )

    return jsonify(
        {
            "amr1": {
                "status": UI_bridge.robots["amr1"].status,
                "pose": {
                    "x": UI_bridge.robots["amr1"].pose.x,
                    "y": UI_bridge.robots["amr1"].pose.y,
                    "w": UI_bridge.robots["amr1"].pose.w,
                },
                "battery": UI_bridge.robots["amr1"].battery,
            },
            "amr2": {
                "status": UI_bridge.robots["amr2"].status,
                "pose": {
                    "x": UI_bridge.robots["amr2"].pose.x,
                    "y": UI_bridge.robots["amr2"].pose.y,
                    "w": UI_bridge.robots["amr2"].pose.w,
                },
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
# alert clear api
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
# command api
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

    UI_command.send_teleop(robot_id, direction)
    UI_bridge.add_log(robot_id, f"수동 조작: {direction}")

    return jsonify({"ok": True})


@app.route("/api/home", methods=["POST"])
def api_home():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(force=True)
    robot_id = data.get("robot_id")

    if robot_id not in {"amr1", "amr2"}:
        return jsonify({"ok": False, "error": "invalid robot id"}), 400

    UI_command.send_home(robot_id)
    UI_bridge.add_log(robot_id, "HOME 복귀 요청")

    return jsonify({"ok": True})


# ============================================================
# 테스트용 pose 변화
# ============================================================
def pose_simulator() -> None:
    tick = 0.0
    while True:
        tick += 0.05

        if UI_bridge.robots["amr1"].status in {"Idle", "Stopped", "Docked", "Enemy Detected"}:
            UI_bridge.robots["amr1"].pose.y = round(np.sin(tick) * 0.5, 2)

        if UI_bridge.robots["amr2"].status in {"Idle", "Stopped", "Docked"}:
            UI_bridge.robots["amr2"].pose.y = round(np.cos(tick) * 0.5, 2)

        time.sleep(0.2)


# ============================================================
# 시작
# ============================================================
def bootstrap() -> None:
    UI_bridge.add_log("amr1", "UI 시스템 시작")
    UI_bridge.add_log("amr2", "UI 시스템 시작")

    UI_bridge.start_bridge_thread()
    UI_command.init_ui_command()

    threading.Thread(target=pose_simulator, daemon=True).start()


def main():
    bootstrap()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
    