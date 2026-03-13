# obj_det_detph_dual.py
import threading
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────
NS             = "robot1"
YOLO_WEIGHTS   = "/home/sinya/slam_turtlebot/src/models/arm2/best.pt"
RESNET_WEIGHTS = "/home/sinya/slam_turtlebot/src/models/arm2/resnet18.pth"
CONF_THRESH    = 0.5
DX, DY         = 0, 5
LOG_INTERVAL   = 0.5
WIN_W, WIN_H   = 960, 540

# 공통 클래스 라벨
CLASSES = ["balloon", "bird", "enemy", "unknown"]

# 거리 로그 제외 클래스
# LOG_EXCLUDE = {"friend"}

# 클래스별 바운딩박스 색상 (BGR)
COLOR_MAP = {
    "balloon": (0, 165, 255),   # 주황
    "bird":    (255, 255,   0), # 하늘
    "enemy":   (0,   0, 255),   # 빨강
    "unknown": (0, 255,   0),   # 초록
}

DEPTH_TOPIC = f"/{NS}/oakd/stereo/image_raw/compressedDepth"
RGB_TOPIC   = f"/{NS}/oakd/rgb/image_raw/compressed"

# ── ResNet18 전처리 Transform ─────────────────────────────────────────────
RESNET_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── CompressedDepth 디코딩 ────────────────────────────────────────────────
def decode_compressed_depth(msg) -> np.ndarray | None:
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])
    idx = raw.tobytes().find(PNG_MAGIC)
    if idx < 0:
        idx = 12
    img = cv2.imdecode(raw[idx:], cv2.IMREAD_UNCHANGED)
    return img


# ── Pixel Alignment ───────────────────────────────────────────────────────
def apply_offset(depth, dx, dy):
    if dx == 0 and dy == 0:
        return depth
    h, w = depth.shape[:2]
    M = np.float64([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── ROI Median Depth → 미터 ───────────────────────────────────────────────
def get_depth_m(depth_al, x1, y1, x2, y2, shrink=0.5):
    h, w   = depth_al.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw     = (x2 - x1) * shrink / 2
    hh     = (y2 - y1) * shrink / 2
    rx1 = int(max(0, cx - hw));  rx2 = int(min(w, cx + hw))
    ry1 = int(max(0, cy - hh));  ry2 = int(min(h, cy + hh))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi   = depth_al[ry1:ry2, rx1:rx2]
    valid = roi[(roi > 0) & (roi < 8000)].astype(np.float32)
    if len(valid) < 5:
        return None
    z = float(np.median(valid))
    return z / 1000.0 if z > 100.0 else z


# ── ResNet18 단일 크롭 추론 ───────────────────────────────────────────────
def classify_crop(model_resnet, device, bgr_crop: np.ndarray):
    """BGR crop -> (클래스 인덱스, 신뢰도) 반환"""
    if bgr_crop is None or bgr_crop.size == 0:
        return None, 0.0
    rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    tensor   = RESNET_TRANSFORM(rgb_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model_resnet(tensor), dim=1)[0]
    idx = int(probs.argmax())
    return idx, float(probs[idx])


# ══════════════════════════════════════════════════════════════════════════
class DetectDepthNode(Node):
    def __init__(self):
        super().__init__("amr2_detector_dual")

        self._img_lock  = threading.Lock()
        self._disp_lock = threading.Lock()
        self._rgb       = None
        self._depth     = None
        self._disp      = None

        # ── YOLO 로드 ────────────────────────────────────────────────────
        self.yolo = YOLO(YOLO_WEIGHTS)
        self.get_logger().info(f"YOLO 로드: {YOLO_WEIGHTS}")

        # ── ResNet18 로드 ────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Linear(
            self.resnet.fc.in_features, len(CLASSES))

        state = torch.load(RESNET_WEIGHTS, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.resnet.load_state_dict(state)
        self.resnet.to(self.device).eval()
        self.get_logger().info(
            f"ResNet18 로드: {RESNET_WEIGHTS}  device={self.device}")

        # ── ROS 구독 ─────────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage, RGB_TOPIC,
            self._cb_rgb, qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage, DEPTH_TOPIC,
            self._cb_depth, qos_profile_sensor_data)

        self._stop     = threading.Event()
        self._last_log = 0.0

        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._gui,    daemon=True).start()

        self.get_logger().info(
            f"구독 토픽:\n  RGB  : {RGB_TOPIC}\n  Depth: {DEPTH_TOPIC}")

    # ── ROS 콜백 ──────────────────────────────────────────────────────────
    def _cb_rgb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self._img_lock:
                self._rgb = img

    def _cb_depth(self, msg):
        img = decode_compressed_depth(msg)
        if img is not None and img.dtype in (np.uint16, np.float32):
            with self._img_lock:
                self._depth = img
        elif img is not None:
            with self._img_lock:
                self._depth = img.astype(np.uint16)

    # ── Worker ────────────────────────────────────────────────────────────
    def _worker(self):
        while not self._stop.is_set():
            t0 = time.monotonic()

            with self._img_lock:
                rgb   = None if self._rgb   is None else self._rgb.copy()
                depth = None if self._depth is None else self._depth.copy()

            if rgb is None:
                time.sleep(0.03)
                continue

            frame       = rgb.copy()
            log_targets = {}   # {label: z_m} -- friend 제외

            if depth is not None:
                depth_al = apply_offset(depth, DX, DY)
            else:
                depth_al = None
                self.get_logger().warn(
                    "Depth 미수신 - RGB만 표시 중", throttle_duration_sec=5.0)

            results = self.yolo(rgb, verbose=False)[0]

            for det in results.boxes:
                yolo_label = self.yolo.names[int(det.cls[0])].lower()
                yolo_conf  = float(det.conf[0])
                if yolo_conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ── Depth ────────────────────────────────────────────────
                z_m = get_depth_m(depth_al, x1, y1, x2, y2) \
                    if depth_al is not None else None

                # ── ResNet18 분류 ────────────────────────────────────────
                crop          = rgb[max(0, y1):y2, max(0, x1):x2]
                r_idx, r_conf = classify_crop(self.resnet, self.device, crop)
                r_label       = CLASSES[r_idx] \
                                if r_idx is not None and r_idx < len(CLASSES) \
                                else "?"

                # ── 시각화 ───────────────────────────────────────────────
                col      = COLOR_MAP.get(yolo_label, (180, 180, 180))
                dist_str = f"{z_m:.2f}m" if z_m else "---"

                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

                # 윗줄: YOLO 결과
                cv2.putText(frame,
                            f"[YOLO] {yolo_label} {yolo_conf:.2f}  {dist_str}",
                            (x1, max(y1 - 22, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                # 아랫줄: ResNet18 결과
                cv2.putText(frame,
                            f"[ResNet] {r_label} {r_conf:.2f}",
                            (x1, max(y1 - 4, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

                cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

                # # friend 제외하고 거리 수집
                # if yolo_label not in LOG_EXCLUDE and z_m is not None:
                #     log_targets[yolo_label] = z_m

            # 0.5초마다 터미널 출력
            now = time.time()
            if now - self._last_log >= LOG_INTERVAL:
                self._last_log = now
                if log_targets:
                    for lbl, dist in log_targets.items():
                        self.get_logger().info(
                            f"[{lbl}]  {dist:.3f} m  "
                            f"({dist*1000:.0f} mm)  "
                            f"align dx={DX} dy={DY}")
                else:
                    self.get_logger().info("탐지 없음 (friend 제외)")

            with self._disp_lock:
                self._disp = frame

            time.sleep(max(0.0, 0.033 - (time.monotonic() - t0)))

    # ── GUI ───────────────────────────────────────────────────────────────
    def _gui(self):
        WIN = "Detection (yolo26n + resNet18) & Depth"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, WIN_W, WIN_H)

        while not self._stop.is_set():
            with self._disp_lock:
                img = self._disp

            if img is not None:
                cv2.imshow(WIN, img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._stop.set()
                break

        cv2.destroyAllWindows()

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = DetectDepthNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop.set()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()