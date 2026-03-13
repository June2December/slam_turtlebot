from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Deque, Dict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from flask import Flask, Response, jsonify, render_template_string, request, session

app = Flask(__name__)
app.secret_key = "amr-monitor-demo-secret-key"

VALID_USERS = {
    "admin": {"password": "1234", "name": "상황근무자"},
    "donghyun": {"password": "1234", "name": "박동현"},
}

# ============================================================
# 설정값
# ============================================================
HOST = "0.0.0.0"
PORT = 5000

# ============================================================
# 데이터 구조
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


robots: Dict[str, RobotState] = {
    "amr1": RobotState(robot_id="amr1", name="방공지산 (AMR_1)"),
    "amr2": RobotState(robot_id="amr2", name="지상정찰 (AMR_2)"),
}

logs: Deque[dict] = deque(maxlen=500)

# detect 노드에서 받은 최신 적 탐지 이미지 경로
latest_enemy_image_path: str | None = None
latest_enemy_image_lock = threading.Lock()

# 마지막 탐지 시간
last_enemy_detect_time = 0.0
last_friend_detect_time = 0.0

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


def get_recent_logs(limit: int = 5) -> list[dict]:
    return list(logs)[-limit:][::-1]


def get_all_logs() -> list[dict]:
    return list(logs)[::-1]


# ============================================================
# ROS2 Subscriber Node
# ============================================================
class UiSubscriberNode(Node):
    def __init__(self):
        super().__init__("ui_subscriber_node")

        self.enemy_sub = self.create_subscription(
            Bool,
            "/enemy_detected",
            self.enemy_callback,
            10,
        )

        self.friend_sub = self.create_subscription(
            Bool,
            "/friend_detected",
            self.friend_callback,
            10,
        )

        self.enemy_image_sub = self.create_subscription(
            String,
            "/enemy_image_path",
            self.enemy_image_callback,
            10,
        )

    def enemy_callback(self, msg: Bool):
        global last_enemy_detect_time
        if msg.data:
            last_enemy_detect_time = time.time()
            robots["amr1"].status = "Enemy Detected"
            add_log("amr1", "enemy 탐지")

    def friend_callback(self, msg: Bool):
        global last_friend_detect_time
        if msg.data:
            last_friend_detect_time = time.time()
            robots["amr1"].status = "Friend Detected"
            add_log("amr1", "friend 탐지")

    def enemy_image_callback(self, msg: String):
        global latest_enemy_image_path
        path = msg.data.strip()

        if path and os.path.exists(path):
            with latest_enemy_image_lock:
                latest_enemy_image_path = path
            add_log("amr1", f"탐지 이미지 수신: {os.path.basename(path)}")


def ros_spin_worker():
    rclpy.init()
    node = UiSubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ============================================================
# 이미지 반환
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


# ============================================================
# HTML 템플릿
# ============================================================
PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AMR 통합 관제 UI</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f7fb;
      color: #1f2937;
    }
    .container {
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }
    .login-page {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: linear-gradient(135deg, #e8f0ff 0%, #f7faff 100%);
    }
    .login-card {
      width: 100%;
      max-width: 420px;
      background: white;
      border: 1px solid #dbe2ea;
      border-radius: 22px;
      padding: 28px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    }
    .login-card h1 {
      margin: 0 0 10px;
      font-size: 28px;
    }
    .login-card p {
      margin: 0 0 22px;
      color: #6b7280;
      line-height: 1.5;
    }
    .login-card label {
      display: block;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 700;
    }
    .login-card input {
      width: 100%;
      padding: 14px 15px;
      border: 1px solid #cfd8e3;
      border-radius: 14px;
      font-size: 15px;
      margin-bottom: 16px;
    }
    .login-card button {
      width: 100%;
      padding: 14px 16px;
      border: none;
      border-radius: 14px;
      background: #2563eb;
      color: white;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
    }
    .login-error {
      margin-top: 14px;
      color: #dc2626;
      font-size: 14px;
      min-height: 20px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
    }
    .title h1 {
      margin: 0 0 6px;
      font-size: 28px;
    }
    .title p {
      margin: 0;
      color: #6b7280;
      font-size: 14px;
    }
    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }
    .tab-btn {
      border: none;
      background: #dce7f5;
      color: #1f2937;
      padding: 12px 18px;
      border-radius: 14px;
      cursor: pointer;
      font-size: 15px;
      font-weight: 700;
    }
    .tab-btn.active {
      background: #2563eb;
      color: white;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 0.9fr;
      gap: 18px;
      align-items: stretch;
    }
    .bottom-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 0.9fr;
      gap: 18px;
      margin-top: 18px;
    }
    .card {
      background: white;
      border: 1px solid #dbe2ea;
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .card h2 {
      margin: 0 0 14px;
      font-size: 20px;
    }
    .status-badge {
      display: inline-block;
      background: #eef2ff;
      color: #4338ca;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 12px;
    }
    .snapshot-box {
      overflow: hidden;
      border-radius: 16px;
      border: 1px solid #dbe2ea;
      background: #111827;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .snapshot-box img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .pose-box {
      margin-top: 14px;
      border: 1px solid #dbe2ea;
      background: #f8fafc;
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 15px;
      font-weight: 600;
    }
    .button-row {
      margin-top: 14px;
      display: flex;
      gap: 10px;
    }
    .primary-btn, .secondary-btn {
      border: none;
      border-radius: 14px;
      cursor: pointer;
      padding: 12px 16px;
      font-size: 15px;
      font-weight: 700;
    }
    .primary-btn {
      background: #2563eb;
      color: white;
      width: 100%;
    }
    .secondary-btn {
      background: #e5e7eb;
      color: #111827;
    }
    .log-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .log-item {
      border: 1px solid #dbe2ea;
      border-radius: 14px;
      padding: 12px 14px;
      background: #fff;
    }
    .log-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      font-weight: 700;
    }
    .log-msg {
      color: #4b5563;
      font-size: 14px;
    }
    .teleop-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }
    .teleop-grid {
      display: grid;
      grid-template-columns: 70px 70px 70px;
      gap: 10px;
      align-items: center;
      justify-content: center;
    }
    .teleop-grid button {
      height: 64px;
      border-radius: 18px;
      border: 1px solid #cfd8e3;
      background: white;
      font-size: 24px;
      cursor: pointer;
      font-weight: 700;
    }
    .teleop-grid .stop {
      background: #dc2626;
      color: white;
      border: none;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #dbe2ea;
    }
    th, td {
      text-align: left;
      padding: 14px 16px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 14px;
    }
    th {
      background: #eff6ff;
      font-size: 15px;
    }
    .user-panel {
      display: flex;
      flex-direction: column;
      gap: 12px;
      justify-content: center;
      height: 100%;
    }
    .user-box {
      background: #f8fafc;
      border: 1px solid #dbe2ea;
      border-radius: 14px;
      padding: 14px 16px;
      font-weight: 700;
      text-align: center;
    }
    .logout-btn {
      border: none;
      border-radius: 14px;
      background: #111827;
      color: white;
      padding: 12px 16px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    .modal {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
    }
    .modal.show { display: flex; }
    .modal-content {
      width: 420px;
      max-width: calc(100vw - 32px);
      background: white;
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    .modal-content h3 {
      margin: 0 0 8px;
      font-size: 22px;
    }
    .modal-content p {
      margin: 0 0 18px;
      color: #4b5563;
      line-height: 1.5;
    }
    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
    }
    @media (max-width: 1200px) {
      .main-grid, .bottom-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div id="login-page" class="login-page">
    <div class="login-card">
      <h1>AMR 통합 관제 로그인</h1>
      <p>사용자 ID와 비밀번호를 입력한 뒤 관제 페이지로 이동하세요.</p>
      <label for="login-id">사용자 ID</label>
      <input id="login-id" type="text" placeholder="아이디 입력" />
      <label for="login-password">비밀번호</label>
      <input id="login-password" type="password" placeholder="비밀번호 입력" />
      <button onclick="login()">로그인</button>
      <div id="login-error" class="login-error"></div>
    </div>
  </div>

  <div id="app-page" class="container" style="display:none;">
    <div class="topbar">
      <div class="title">
        <h1>AMR 통합 관제 UI</h1>
        <p>detect 노드가 저장한 탐지 이미지를 ROS 토픽으로 받아 표시합니다.</p>
      </div>
    </div>

    <div class="tabs">
      <button class="tab-btn active" data-tab="realtime">실시간 확인</button>
      <button class="tab-btn" data-tab="logs">DB 확인</button>
    </div>

    <div id="realtime" class="tab-panel active">
      <div class="main-grid">
        <div class="card">
          <h2>방공지산 (AMR_1)</h2>
          <div class="status-badge" id="amr1-status">Idle</div>
          <div class="snapshot-box">
            <img id="amr1-snapshot" src="/snapshot/amr1" alt="AMR1 Snapshot" />
          </div>
          <div class="pose-box" id="amr1-pose">현재 위치 : x 0.00, y 0.00, w 0.00</div>
          <div class="button-row">
            <button class="primary-btn" onclick="openHomeModal('amr1', '방공지산 (AMR_1)')">HOME 복귀</button>
          </div>
        </div>

        <div class="card">
          <h2>지상정찰 (AMR_2)</h2>
          <div class="status-badge" id="amr2-status">Idle</div>
          <div class="snapshot-box">
            <img id="amr2-snapshot" src="/snapshot/amr2" alt="AMR2 Snapshot" />
          </div>
          <div class="pose-box" id="amr2-pose">현재 위치 : x 0.00, y 0.00, w 0.00</div>
          <div class="button-row">
            <button class="primary-btn" onclick="openHomeModal('amr2', '지상정찰 (AMR_2)')">HOME 복귀</button>
          </div>
        </div>

        <div class="card">
          <h2>실시간 상황일지</h2>
          <div class="log-list" id="recent-logs"></div>
        </div>
      </div>

      <div class="bottom-grid">
        <div class="card">
          <h2>AMR_1 수동 조작</h2>
          <div class="teleop-wrap">
            <div class="teleop-grid">
              <div></div>
              <button onclick="sendTeleop('amr1', 'forward')">↑</button>
              <div></div>

              <button onclick="sendTeleop('amr1', 'left')">←</button>
              <button class="stop" onclick="sendTeleop('amr1', 'stop')">■</button>
              <button onclick="sendTeleop('amr1', 'right')">→</button>

              <div></div>
              <button onclick="sendTeleop('amr1', 'backward')">↓</button>
              <div></div>
            </div>
          </div>
        </div>

        <div class="card">
          <h2>AMR_2 수동 조작</h2>
          <div class="teleop-wrap">
            <div class="teleop-grid">
              <div></div>
              <button onclick="sendTeleop('amr2', 'forward')">↑</button>
              <div></div>

              <button onclick="sendTeleop('amr2', 'left')">←</button>
              <button class="stop" onclick="sendTeleop('amr2', 'stop')">■</button>
              <button onclick="sendTeleop('amr2', 'right')">→</button>

              <div></div>
              <button onclick="sendTeleop('amr2', 'backward')">↓</button>
              <div></div>
            </div>
          </div>
        </div>

        <div class="card">
          <h2>사용자 정보</h2>
          <div class="user-panel">
            <div class="user-box" id="current-user">로그인 사용자 : -</div>
            <button class="logout-btn" onclick="logout()">로그아웃</button>
          </div>
        </div>
      </div>
    </div>

    <div id="logs" class="tab-panel">
      <div class="card">
        <h2>전체 로그</h2>
        <table>
          <thead>
            <tr>
              <th>시간</th>
              <th>로봇</th>
              <th>내용</th>
            </tr>
          </thead>
          <tbody id="all-logs"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div id="home-modal" class="modal">
    <div class="modal-content">
      <h3 id="modal-title">HOME 복귀</h3>
      <p id="modal-text">HOME 위치로 복귀한 뒤 Docking까지 진행할까요?</p>
      <div class="modal-actions">
        <button class="secondary-btn" onclick="closeHomeModal()">취소</button>
        <button class="primary-btn" style="width:auto" onclick="confirmHomeAction()">확인</button>
      </div>
    </div>
  </div>

  <script>
    let currentHomeRobotId = null;

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      });
    });

    function setAuthView(isLoggedIn, username='') {
      document.getElementById('login-page').style.display = isLoggedIn ? 'none' : 'flex';
      document.getElementById('app-page').style.display = isLoggedIn ? 'block' : 'none';
      document.getElementById('current-user').innerText = `로그인 사용자 : ${username || '-'}`;
    }

    async function checkSession() {
      try {
        const response = await fetch('/api/session');
        const data = await response.json();
        if (data.logged_in) {
          setAuthView(true, data.username);
          refreshState();
        } else {
          setAuthView(false);
        }
      } catch (err) {
        console.error('session check failed', err);
        setAuthView(false);
      }
    }

    async function login() {
      const userId = document.getElementById('login-id').value.trim();
      const password = document.getElementById('login-password').value.trim();
      const errorBox = document.getElementById('login-error');
      errorBox.innerText = '';

      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, password })
      });
      const data = await response.json();

      if (!data.ok) {
        errorBox.innerText = data.error || '로그인에 실패했습니다.';
        return;
      }

      document.getElementById('login-id').value = '';
      document.getElementById('login-password').value = '';
      setAuthView(true, data.username);
      refreshState();
    }

    async function logout() {
      await fetch('/api/logout', { method: 'POST' });
      setAuthView(false);
    }

    function openHomeModal(robotId, robotName) {
      currentHomeRobotId = robotId;
      document.getElementById('modal-title').innerText = `${robotName} HOME 복귀`;
      document.getElementById('modal-text').innerText = 'HOME 위치로 복귀한 뒤 Docking까지 진행할까요?';
      document.getElementById('home-modal').classList.add('show');
    }

    function closeHomeModal() {
      currentHomeRobotId = null;
      document.getElementById('home-modal').classList.remove('show');
    }

    async function confirmHomeAction() {
      if (!currentHomeRobotId) return;
      await fetch('/api/home', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ robot_id: currentHomeRobotId })
      });
      closeHomeModal();
    }

    async function sendTeleop(robotId, direction) {
      await fetch('/api/teleop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ robot_id: robotId, direction })
      });
    }

    function renderRecentLogs(logs) {
      const wrap = document.getElementById('recent-logs');
      wrap.innerHTML = '';
      if (!logs.length) {
        wrap.innerHTML = '<div class="log-item"><div class="log-msg">최근 로그가 없습니다.</div></div>';
        return;
      }
      logs.forEach(log => {
        const item = document.createElement('div');
        item.className = 'log-item';
        item.innerHTML = `
          <div class="log-top">
            <span>${log.robot}</span>
            <span>${log.time}</span>
          </div>
          <div class="log-msg">${log.message}</div>
        `;
        wrap.appendChild(item);
      });
    }

    function renderAllLogs(logs) {
      const tbody = document.getElementById('all-logs');
      tbody.innerHTML = '';
      logs.forEach(log => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${log.time}</td>
          <td>${log.robot}</td>
          <td>${log.message}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function refreshSnapshots() {
      const ts = Date.now();
      const amr1 = document.getElementById('amr1-snapshot');
      const amr2 = document.getElementById('amr2-snapshot');
      if (amr1) amr1.src = `/snapshot/amr1?t=${ts}`;
      if (amr2) amr2.src = `/snapshot/amr2?t=${ts}`;
    }

    async function refreshState() {
      try {
        const sessionResp = await fetch('/api/session');
        const sessionData = await sessionResp.json();
        if (!sessionData.logged_in) {
          setAuthView(false);
          return;
        }

        const response = await fetch('/api/state');
        if (!response.ok) return;
        const data = await response.json();

        document.getElementById('amr1-status').innerText = data.amr1.status;
        document.getElementById('amr2-status').innerText = data.amr2.status;

        document.getElementById('amr1-pose').innerText =
          `현재 위치 : x ${data.amr1.pose.x.toFixed(2)}, y ${data.amr1.pose.y.toFixed(2)}, w ${data.amr1.pose.w.toFixed(2)}`;
        document.getElementById('amr2-pose').innerText =
          `현재 위치 : x ${data.amr2.pose.x.toFixed(2)}, y ${data.amr2.pose.y.toFixed(2)}, w ${data.amr2.pose.w.toFixed(2)}`;

        renderRecentLogs(data.recent_logs || []);
        renderAllLogs(data.all_logs || []);
        refreshSnapshots();
      } catch (err) {
        console.error('state refresh failed', err);
      }
    }

    checkSession();
    setInterval(refreshState, 1000);
  </script>
</body>
</html>
"""


# ============================================================
# API
# ============================================================
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/snapshot/<robot_id>")
def snapshot(robot_id: str):
    if robot_id not in robots:
        return "invalid robot id", 404

    if robot_id == "amr1":
        with latest_enemy_image_lock:
            image_path = latest_enemy_image_path

        if image_path and os.path.exists(image_path):
            frame = cv2.imread(image_path)
            if frame is not None:
                return Response(encode_jpg(frame), mimetype="image/jpeg")

        return Response(encode_jpg(make_dummy_frame("NO DETECTION YET")), mimetype="image/jpeg")

    return Response(encode_jpg(make_dummy_frame("NO CAMERA LINKED")), mimetype="image/jpeg")


def is_logged_in() -> bool:
    return bool(session.get("logged_in"))


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
    add_log(user_id, "사용자 로그인")
    return jsonify({"ok": True, "username": user["name"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    username = session.get("username", "사용자")
    if session.get("logged_in"):
        add_log(session.get("user_id", "user"), f"{username} 로그아웃")
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/state")
def api_state():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    # 탐지 후 일정 시간 지나면 상태 복귀
    now = time.time()
    if now - last_enemy_detect_time > 2.0 and now - last_friend_detect_time > 2.0:
        if robots["amr1"].status in {"Enemy Detected", "Friend Detected"}:
            robots["amr1"].status = "Idle"

    return jsonify(
        {
            "amr1": {
                "status": robots["amr1"].status,
                "pose": asdict(robots["amr1"].pose),
            },
            "amr2": {
                "status": robots["amr2"].status,
                "pose": asdict(robots["amr2"].pose),
            },
            "recent_logs": get_recent_logs(5),
            "all_logs": get_all_logs(),
        }
    )


@app.route("/api/teleop", methods=["POST"])
def api_teleop():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(force=True)
    robot_id = data.get("robot_id")
    direction = data.get("direction")

    if robot_id not in robots:
        return jsonify({"ok": False, "error": "invalid robot id"}), 400

    if direction not in {"forward", "backward", "left", "right", "stop"}:
        return jsonify({"ok": False, "error": "invalid direction"}), 400

    if direction == "forward":
        robots[robot_id].status = "Manual Forward"
        robots[robot_id].pose.x += 0.10
    elif direction == "backward":
        robots[robot_id].status = "Manual Backward"
        robots[robot_id].pose.x -= 0.10
    elif direction == "left":
        robots[robot_id].status = "Rotate Left"
        robots[robot_id].pose.w += 5.0
    elif direction == "right":
        robots[robot_id].status = "Rotate Right"
        robots[robot_id].pose.w -= 5.0
    elif direction == "stop":
        robots[robot_id].status = "Stopped"

    add_log(robot_id, f"수동 조작: {direction}")
    return jsonify({"ok": True})


@app.route("/api/home", methods=["POST"])
def api_home():
    if not is_logged_in():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    data = request.get_json(force=True)
    robot_id = data.get("robot_id")

    if robot_id not in robots:
        return jsonify({"ok": False, "error": "invalid robot id"}), 400

    robots[robot_id].status = "Returning Home"
    add_log(robot_id, "HOME 복귀 요청")

    def go_home_and_dock():
        time.sleep(1.5)
        robots[robot_id].pose = Pose(0.0, 0.0, 0.0)
        robots[robot_id].status = "Docking"
        add_log(robot_id, "Docking 시작")
        time.sleep(2.0)
        robots[robot_id].status = "Docked"
        add_log(robot_id, "Docking 완료")

    threading.Thread(target=go_home_and_dock, daemon=True).start()
    return jsonify({"ok": True})


# ============================================================
# 테스트용 pose 변화
# ============================================================
def pose_simulator() -> None:
    tick = 0.0
    while True:
        tick += 0.05
        if robots["amr1"].status in {"Idle", "Stopped", "Docked"}:
            robots["amr1"].pose.y = round(np.sin(tick) * 0.5, 2)
        if robots["amr2"].status in {"Idle", "Stopped", "Docked"}:
            robots["amr2"].pose.y = round(np.cos(tick) * 0.5, 2)
        time.sleep(0.2)


# ============================================================
# 시작
# ============================================================
def bootstrap() -> None:
    add_log("amr1", "UI 시스템 시작")
    add_log("amr2", "UI 시스템 시작")

    threading.Thread(target=ros_spin_worker, daemon=True).start()
    threading.Thread(target=pose_simulator, daemon=True).start()


if __name__ == "__main__":
    bootstrap()