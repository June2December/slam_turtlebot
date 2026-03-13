# obj_det_depth_unkonwn.py

import os
import sys
import time
import threading
from queue import Empty

import cv2
import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, CameraInfo

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
NS = "robot1"

YOLO_PT = "/home/sinya/slam_turtlebot/src/models/arm2/best.pt"
RESNET_PT = "/home/sinya/slam_turtlebot/src/models/arm2/resnet18.pth"

RGB_TOPIC = f"/{NS}/oakd/rgb/image_raw/compressed"
DEPTH_TOPIC = f"/{NS}/oakd/stereo/image_raw/compressedDepth"
CAMERA_INFO_TOPIC = f"/{NS}/oakd/stereo/camera_info"  

YOLO_CONF_THRESHOLD = 0.5
RESNET_CONF_THRESHOLD = 0.7

TARGET_LABEL = "unknown"   # 로그로 특히 보고 싶은 클래스(미확인 낙하물)
LOG_INTERVAL = 0.5

# RGB-Depth 정렬 보정값
DX = 0
DY = 5

# bbox 중앙 몇 % 영역으로 depth 거리 계산할지
DEPTH_ROI_RATIO = 0.4

# depth 값 유효 범위(mm)
DEPTH_MIN_MM = 1
DEPTH_MAX_MM = 8000

# depth 시각화 범위(m)
NORMALIZE_DEPTH_RANGE = 3.0
SHOW_DEPTH_WINDOW = True

WIN_W, WIN_H = 1280, 720


# 클래스별 bbox 색상 (BGR)
COLOR_MAP = {
    'balloon': (0, 255, 255),   # 노랑
    'bird':    (0, 255, 0),     # 초록
    'enemy':   (0, 0, 255),     # 빨강
    'unknown': (255, 0, 0),     # 파랑
}
DEFAULT_COLOR = (0, 200, 200)
REJECTED_COLOR = (128, 128, 128)


# ────────────────────────────────────────────────
# CompressedDepth 디코딩
# ────────────────────────────────────────────────
def decode_compressed_depth(msg) -> np.ndarray | None:
    """
    ROS2 compressedDepth 디코딩
    - 앞부분 헤더 뒤에 PNG 데이터가 붙어있는 경우가 많음
    - PNG 시그니처를 찾아서 그 위치부터 imdecode
    """
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    png_magic = bytes([0x89, 0x50, 0x4E, 0x47])

    idx = raw.tobytes().find(png_magic)
    if idx < 0:
        # PNG magic 못 찾으면 ROS compressedDepth 헤더 12바이트 스킵 시도
        idx = 12

    img = cv2.imdecode(raw[idx:], cv2.IMREAD_UNCHANGED)
    return img


def apply_offset(depth: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return depth

    h, w = depth.shape[:2]
    M = np.float64([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(
        depth, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


# ────────────────────────────────────────────────
# ResNet18 로더
# ────────────────────────────────────────────────
def load_resnet18(pth_path, device):
    checkpoint = torch.load(pth_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 형식이 dict가 아닙니다.")
    if 'class_names' not in checkpoint:
        raise KeyError("checkpoint 안에 'class_names' 키가 없습니다.")
    if 'model_state_dict' not in checkpoint:
        raise KeyError("checkpoint 안에 'model_state_dict' 키가 없습니다.")

    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    img_size = checkpoint.get('img_size', 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    print(f"ResNet18 로드 완료 | classes: {class_names}")
    return model, transform, class_names


# ────────────────────────────────────────────────
# ROS2 노드
# ────────────────────────────────────────────────
class DualDetectDepthNode(Node):
    def __init__(self, yolo_model, resnet_model, resnet_transform,
                yolo_classes, resnet_classes, device):
        super().__init__("amr2_detector_dual_unknown")

        self.yolo_model = yolo_model
        self.resnet_model = resnet_model
        self.resnet_transform = resnet_transform
        self.yolo_classes = yolo_classes
        self.resnet_classes = resnet_classes
        self.device = device

        self._img_lock = threading.Lock()
        self._disp_lock = threading.Lock()

        self._rgb = None
        self._depth = None
        self._disp = None
        self.K = None

        self._stop = threading.Event()
        self._last_log = 0.0

        self.create_subscription(
            CompressedImage,
            RGB_TOPIC,
            self.rgb_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            CompressedImage,
            DEPTH_TOPIC,
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("DualDetectDepthNode 시작")
        self.get_logger().info(f"YOLO classes   : {self.yolo_classes}")
        self.get_logger().info(f"ResNet classes : {self.resnet_classes}")
        self.get_logger().info(f"YOLO conf th   : {YOLO_CONF_THRESHOLD}")
        self.get_logger().info(f"ResNet conf th : {RESNET_CONF_THRESHOLD}")
        self.get_logger().info(f"RGB topic      : {RGB_TOPIC}")
        self.get_logger().info(f"Depth topic    : {DEPTH_TOPIC}")
        self.get_logger().info(f"CameraInfo     : {CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"Depth align    : dx={DX}, dy={DY}")

        threading.Thread(target=self.worker_loop, daemon=True).start()
        threading.Thread(target=self.gui_loop, daemon=True).start()

    # ────────────────────────────────────────────
    # 콜백
    # ────────────────────────────────────────────
    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo received: "
                f"fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
            )

    def rgb_callback(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn("Compressed RGB decode 실패")
                return

            with self._img_lock:
                self._rgb = img
        except Exception as e:
            self.get_logger().error(f"RGB decode failed: {e}")

    def depth_callback(self, msg):
        try:
            img = decode_compressed_depth(msg)
            if img is None:
                self.get_logger().warn("CompressedDepth decode 실패")
                return

            if img.dtype not in (np.uint16, np.float32, np.float64):
                img = img.astype(np.uint16)

            with self._img_lock:
                self._depth = img
        except Exception as e:
            self.get_logger().error(f"Depth decode failed: {e}")

    # ────────────────────────────────────────────
    # ResNet 분류
    # ────────────────────────────────────────────
    def classify_crop(self, crop_bgr):
        crop_pil = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            tensor = self.resnet_transform(crop_pil).unsqueeze(0).to(self.device)
            outputs = self.resnet_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            cls_id = probs.argmax().item()
            conf = probs[cls_id].item()

        return self.resnet_classes[cls_id], conf

    # ────────────────────────────────────────────
    # Depth 거리 계산
    # ────────────────────────────────────────────
    def get_depth_m(self, depth_al, x1, y1, x2, y2, shrink=DEPTH_ROI_RATIO):
        if depth_al is None:
            return None, None

        h, w = depth_al.shape[:2]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        hw = int((x2 - x1) * shrink / 2)
        hh = int((y2 - y1) * shrink / 2)

        rx1 = max(0, cx - hw)
        rx2 = min(w, cx + hw)
        ry1 = max(0, cy - hh)
        ry2 = min(h, cy + hh)

        if rx2 <= rx1 or ry2 <= ry1:
            return None, (cx, cy)

        roi = depth_al[ry1:ry2, rx1:rx2]

        if roi.dtype == np.uint16:
            valid = roi[(roi >= DEPTH_MIN_MM) & (roi < DEPTH_MAX_MM)].astype(np.float32)
            if len(valid) < 5:
                return None, (cx, cy)
            z_mm = float(np.median(valid))
            return z_mm / 1000.0, (cx, cy)

        elif roi.dtype in (np.float32, np.float64):
            valid = roi[np.isfinite(roi)]
            valid = valid[valid > 0]
            if len(valid) < 5:
                return None, (cx, cy)

            z = float(np.median(valid))

            # float인데 mm처럼 큰 수면 meter로 변환
            if z > 50.0:
                z = z / 1000.0

            return z, (cx, cy)

        return None, (cx, cy)

    # ────────────────────────────────────────────
    # Depth 시각화
    # ────────────────────────────────────────────
    def make_depth_visualization(self, depth):
        if depth is None:
            return None

        if depth.dtype == np.uint16:
            depth_vis = depth.astype(np.float32) / 1000.0
        else:
            depth_vis = depth.astype(np.float32)
            if np.nanmax(depth_vis) > 50:
                depth_vis = depth_vis / 1000.0

        depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = np.clip(depth_vis, 0, NORMALIZE_DEPTH_RANGE)
        depth_vis = (depth_vis / NORMALIZE_DEPTH_RANGE * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        return depth_colored

    # ────────────────────────────────────────────
    # 박스 그리기
    # ────────────────────────────────────────────
    def draw_box(self, img, x1, y1, x2, y2, label, color, thickness=2):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_top = max(0, y1 - th - 8)
        text_bottom = max(th + 2, y1)

        cv2.rectangle(img, (x1, text_top), (x1 + tw + 4, text_bottom), color, -1)
        cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ────────────────────────────────────────────
    # 메인 처리 루프
    # ────────────────────────────────────────────
    def worker_loop(self):
        while not self._stop.is_set():
            t0 = time.monotonic()

            with self._img_lock:
                rgb = None if self._rgb is None else self._rgb.copy()
                depth = None if self._depth is None else self._depth.copy()

            if rgb is None:
                time.sleep(0.03)
                continue

            frame = rgb.copy()
            best_target_z = None

            if depth is not None:
                depth_al = apply_offset(depth, DX, DY)
                depth_vis = self.make_depth_visualization(depth_al)
            else:
                depth_al = None
                depth_vis = None
                self.get_logger().warn("Depth 미수신 - RGB만 표시 중", throttle_duration_sec=5.0)

            try:
                results = self.yolo_model.predict(
                    rgb,
                    conf=YOLO_CONF_THRESHOLD,
                    verbose=False
                )[0]
            except Exception as e:
                self.get_logger().error(f"YOLO 추론 실패: {e}")
                with self._disp_lock:
                    self._disp = frame
                time.sleep(0.03)
                continue

            h, w = rgb.shape[:2]

            for det in results.boxes:
                try:
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    yolo_cls = int(det.cls[0])
                    yolo_conf = float(det.conf[0])

                    if yolo_conf < YOLO_CONF_THRESHOLD:
                        continue
                    if yolo_cls < 0 or yolo_cls >= len(self.yolo_classes):
                        continue

                    yolo_cls_name = self.yolo_classes[yolo_cls].lower()

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    resnet_cls_name, resnet_conf = self.classify_crop(crop)
                    resnet_cls_name = resnet_cls_name.lower()

                    z_m, depth_center = self.get_depth_m(depth_al, x1, y1, x2, y2)
                    dist_text = "D:N/A" if z_m is None else f"D:{z_m:.2f}m"

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 18, 2)

                    if depth_vis is not None and depth_center is not None:
                        du, dv = depth_center
                        cv2.circle(depth_vis, (du, dv), 4, (0, 0, 0), -1)

                    # 최종 판정
                    if resnet_conf < RESNET_CONF_THRESHOLD:
                        label = f"[LOW CONF] Y:{yolo_cls_name}({yolo_conf:.2f}) {dist_text}"
                        self.draw_box(frame, x1, y1, x2, y2, label, REJECTED_COLOR, thickness=1)

                    elif yolo_cls_name == resnet_cls_name:
                        color = COLOR_MAP.get(yolo_cls_name, DEFAULT_COLOR)
                        label = f"{yolo_cls_name} Y:{yolo_conf:.2f} R:{resnet_conf:.2f} {dist_text}"
                        self.draw_box(frame, x1, y1, x2, y2, label, color, thickness=2)

                        if yolo_cls_name == TARGET_LABEL and z_m is not None and best_target_z is None:
                            best_target_z = z_m

                    else:
                        label = (f"[MISMATCH] "
                                 f"Y:{yolo_cls_name}({yolo_conf:.2f}) "
                                 f"R:{resnet_cls_name}({resnet_conf:.2f}) "
                                 f"{dist_text}")
                        self.draw_box(frame, x1, y1, x2, y2, label, REJECTED_COLOR, thickness=1)

                except Exception as e:
                    self.get_logger().warn(f"개별 bbox 처리 실패: {e}")

            now = time.time()
            if now - self._last_log >= LOG_INTERVAL:
                self._last_log = now
                if best_target_z is not None:
                    self.get_logger().info(
                        f"[{TARGET_LABEL}] {best_target_z:.3f} m "
                        f"({best_target_z * 1000:.0f} mm) "
                        f"align dx={DX} dy={DY}"
                    )
                else:
                    self.get_logger().info(f"[{TARGET_LABEL}] 탐지 없음")

            if SHOW_DEPTH_WINDOW and depth_vis is not None:
                cv2.putText(depth_vis, f"Depth align dx={DX}, dy={DY}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            combined = frame
            if SHOW_DEPTH_WINDOW and depth_vis is not None:
                if depth_vis.shape[:2] != frame.shape[:2]:
                    depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
                combined = np.hstack([frame, depth_vis])

            with self._disp_lock:
                self._disp = combined

            time.sleep(max(0.0, 0.03 - (time.monotonic() - t0)))

    # ────────────────────────────────────────────
    # GUI
    # ────────────────────────────────────────────
    def gui_loop(self):
        win = "YOLO26n + ResNet18 + CompressedDepth"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WIN_W, WIN_H)

        while not self._stop.is_set():
            with self._disp_lock:
                img = None if self._disp is None else self._disp.copy()

            if img is not None:
                cv2.imshow(win, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 입력으로 종료")
                self._stop.set()
                break

        cv2.destroyAllWindows()

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


# ────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────
def main():
    for path, name in [(YOLO_PT, "YOLO pt"), (RESNET_PT, "ResNet18 pt")]:
        if not os.path.exists(path):
            print(f"❌ {name} 파일 없음: {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    yolo_model = YOLO(YOLO_PT)
    yolo_classes = [str(v).lower() for v in yolo_model.names.values()]
    print(f"YOLO 로드 완료 | classes: {yolo_classes}")

    resnet_model, resnet_transform, resnet_classes = load_resnet18(RESNET_PT, device)
    resnet_classes = [str(v).lower() for v in resnet_classes]

    rclpy.init()
    node = DualDetectDepthNode(
        yolo_model,
        resnet_model,
        resnet_transform,
        yolo_classes,
        resnet_classes,
        device
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C로 종료")
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
        cv2.destroyAllWindows()
        print("종료 완료")


if __name__ == "__main__":
    main()