# obj_det_depth.py
import os
import sys
import rclpy
import threading
import numpy as np
from queue import Queue, Empty

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

import cv2
from cv_bridge import CvBridge

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage


# ────────────────────────────────────────────────
# init
NAMESPACE = "robot1"

YOLO_PT = "/home/sinya/slam_turtlebot/src/models/arm2/best.pt"
RESNET_PT = "/home/sinya/slam_turtlebot/src/models/arm2/resnet18.pth"

RGB_TOPIC = f'/{NAMESPACE}/oakd/rgb/image_raw/compressed'
DEPTH_TOPIC = f'/{NAMESPACE}/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = f'/{NAMESPACE}/oakd/stereo/camera_info'

YOLO_CONF_THRESHOLD = 0.5
RESNET_CONF_THRESHOLD = 0.7

# depth 값이 mm 단위라고 가정
DEPTH_SCALE = 0.001   # mm -> m
NORMALIZE_DEPTH_RANGE = 3.0  # depth 시각화용 최대 3m
SHOW_DEPTH_WINDOW = True

# bbox 중앙 몇 % 영역을 거리 계산에 사용할지
DEPTH_ROI_RATIO = 0.4  # 중앙 40% 영역 사용
# ────────────────────────────────────────────────


# 클래스별 bbox 색상 (BGR)
COLOR_MAP = {
    'balloon': (0, 255, 255),   # 노랑
    'bird':    (0, 255, 0),     # 초록
    'enemy':   (0, 0, 255),     # 빨강
    'unknown': (255, 0, 0),     # 파랑
}
DEFAULT_COLOR = (0, 200, 200)
REJECTED_COLOR = (128, 128, 128)


# ════════════════════════════════════════════════
# ResNet18 로더
# ════════════════════════════════════════════════
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
                            [0.229, 0.224, 0.225]),])

    print(f"ResNet18 로드 완료 | classes: {class_names}")
    return model, transform, class_names


# ════════════════════════════════════════════════
# ROS2 노드
# ════════════════════════════════════════════════
class DualModelDepthDetector(Node):
    def __init__(self, yolo_model, resnet_model, resnet_transform,
                yolo_classes, resnet_classes, device):
        super().__init__('amr2_detector')

        self.yolo_model = yolo_model
        self.resnet_model = resnet_model
        self.resnet_transform = resnet_transform
        self.yolo_classes = yolo_classes
        self.resnet_classes = resnet_classes
        self.device = device

        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False

        self.depth_frame = None
        self.depth_shape = None
        self.K = None
        self.depth_lock = threading.Lock()

        # RGB subscriber
        self.rgb_subscription = self.create_subscription(
            CompressedImage,
            RGB_TOPIC,
            self.rgb_callback,
            10
        )

        # Depth subscriber
        self.depth_subscription = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10
        )

        # CameraInfo subscriber
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10
        )

        self.get_logger().info("DualModelDepthDetector 시작")
        self.get_logger().info(f"YOLO classes   : {self.yolo_classes}")
        self.get_logger().info(f"ResNet classes : {self.resnet_classes}")
        self.get_logger().info(f"YOLO conf th   : {YOLO_CONF_THRESHOLD}")
        self.get_logger().info(f"ResNet conf th : {RESNET_CONF_THRESHOLD}")
        self.get_logger().info(f"RGB topic      : {RGB_TOPIC}")
        self.get_logger().info(f"Depth topic    : {DEPTH_TOPIC}")
        self.get_logger().info(f"CameraInfo     : {CAMERA_INFO_TOPIC}")

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    # ── CameraInfo 수신 ──
    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo received: "
                f"fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
            )

    # ── RGB 수신 ──
    def rgb_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().error("Compressed RGB decode 실패")
                return

            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except Empty:
                    pass

            self.image_queue.put_nowait(img)

        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    # ── Depth 수신 ──
    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if len(depth.shape) != 2:
                self.get_logger().warn(f"Depth image shape 이상: {depth.shape}")
                return

            with self.depth_lock:
                self.depth_frame = depth.copy()
                self.depth_shape = depth.shape

        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    # ── 탐지 루프 ──
    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue
            except Exception:
                continue

            try:
                result_img, depth_vis = self._dual_detect(img)
            except Exception as e:
                self.get_logger().error(f"Detection failed: {e}")
                result_img = img
                depth_vis = None

            cv2.imshow("YOLO + ResNet18 + Depth", result_img)

            if SHOW_DEPTH_WINDOW and depth_vis is not None:
                cv2.imshow("Depth Image", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("'q' 입력으로 종료")
                self.should_shutdown = True
                break

    # ── 이중 검증 + depth 표시 ──
    def _dual_detect(self, img):
        rgb_h, rgb_w = img.shape[:2]

        results = self.yolo_model.predict(
            img,
            conf=YOLO_CONF_THRESHOLD,
            stream=True,
            verbose=False
        )

        # depth 시각화 프레임 생성
        depth_vis = self._make_depth_visualization(rgb_shape=img.shape)

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_cls = int(box.cls[0])
                yolo_conf = float(box.conf[0])

                if yolo_cls < 0 or yolo_cls >= len(self.yolo_classes):
                    continue

                yolo_cls_name = self.yolo_classes[yolo_cls]

                # 경계 클램핑
                x1c = max(0, min(x1, rgb_w - 1))
                y1c = max(0, min(y1, rgb_h - 1))
                x2c = max(0, min(x2, rgb_w))
                y2c = max(0, min(y2, rgb_h))

                if x2c <= x1c or y2c <= y1c:
                    continue

                # crop → ResNet18 분류
                crop = img[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                crop_pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                resnet_cls_name, resnet_conf = self._classify(crop_pil)

                # depth 거리 계산
                distance_m, depth_center = self._get_depth_distance_and_center(
                    x1c, y1c, x2c, y2c, img.shape
                )

                dist_text = "D:N/A" if distance_m is None else f"D:{distance_m:.2f}m"

                # depth 시각화에도 중심점 표시
                if depth_vis is not None and depth_center is not None:
                    du, dv = depth_center
                    cv2.circle(depth_vis, (du, dv), 4, (0, 0, 0), -1)

                # 결과에 따른 bbox 표시
                if resnet_conf < RESNET_CONF_THRESHOLD:
                    label = f"[LOW CONF] {yolo_cls_name} Y:{yolo_conf:.2f} {dist_text}"
                    self._draw_box(img, x1c, y1c, x2c, y2c, label, REJECTED_COLOR, thickness=1)

                elif yolo_cls_name == resnet_cls_name:
                    color = COLOR_MAP.get(yolo_cls_name, DEFAULT_COLOR)
                    label = f"{yolo_cls_name} Y:{yolo_conf:.2f} R:{resnet_conf:.2f} {dist_text}"
                    self._draw_box(img, x1c, y1c, x2c, y2c, label, color, thickness=2)

                else:
                    label = (f"[MISMATCH] "
                            f"Y:{yolo_cls_name}({yolo_conf:.2f}) "
                            f"R:{resnet_cls_name}({resnet_conf:.2f}) "
                            f"{dist_text}")
                    self._draw_box(img, x1c, y1c, x2c, y2c, label, REJECTED_COLOR, thickness=1)

        return img, depth_vis

    # ── ResNet18 분류 ──
    def _classify(self, crop_pil):
        with torch.no_grad():
            tensor = self.resnet_transform(crop_pil).unsqueeze(0).to(self.device)
            outputs = self.resnet_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            cls_id = probs.argmax().item()
            conf = probs[cls_id].item()

        return self.resnet_classes[cls_id], conf

    # ── bbox 중앙 ROI에서 depth median 추출 ──
    def _get_depth_distance_and_center(self, x1, y1, x2, y2, rgb_shape):
        with self.depth_lock:
            if self.depth_frame is None:
                return None, None

            depth = self.depth_frame.copy()

        depth_h, depth_w = depth.shape[:2]
        rgb_h, rgb_w = rgb_shape[:2]

        # RGB와 depth 해상도가 다를 수 있으므로 스케일링
        sx = depth_w / rgb_w
        sy = depth_h / rgb_h

        dx1 = int(x1 * sx)
        dy1 = int(y1 * sy)
        dx2 = int(x2 * sx)
        dy2 = int(y2 * sy)

        dx1 = max(0, min(dx1, depth_w - 1))
        dy1 = max(0, min(dy1, depth_h - 1))
        dx2 = max(0, min(dx2, depth_w))
        dy2 = max(0, min(dy2, depth_h))

        if dx2 <= dx1 or dy2 <= dy1:
            return None, None

        # bbox 중앙 ROI만 사용
        roi_w = dx2 - dx1
        roi_h = dy2 - dy1

        inner_w = max(1, int(roi_w * DEPTH_ROI_RATIO))
        inner_h = max(1, int(roi_h * DEPTH_ROI_RATIO))

        cx = (dx1 + dx2) // 2
        cy = (dy1 + dy2) // 2

        rx1 = max(0, cx - inner_w // 2)
        ry1 = max(0, cy - inner_h // 2)
        rx2 = min(depth_w, cx + inner_w // 2)
        ry2 = min(depth_h, cy + inner_h // 2)

        if rx2 <= rx1 or ry2 <= ry1:
            return None, (cx, cy)

        roi = depth[ry1:ry2, rx1:rx2]

        # depth 타입별 처리
        if roi.dtype == np.uint16:
            valid = roi[roi > 0]
            if len(valid) == 0:
                return None, (cx, cy)
            depth_raw = np.median(valid)
            distance_m = float(depth_raw) * DEPTH_SCALE

        elif roi.dtype == np.float32 or roi.dtype == np.float64:
            valid = roi[np.isfinite(roi)]
            valid = valid[valid > 0]
            if len(valid) == 0:
                return None, (cx, cy)

            depth_raw = np.median(valid)

            # float depth가 meter인지 mm인지 애매하면 값 범위로 대략 판정
            if depth_raw > 50:
                distance_m = float(depth_raw) * DEPTH_SCALE
            else:
                distance_m = float(depth_raw)
        else:
            valid = roi[roi > 0]
            if len(valid) == 0:
                return None, (cx, cy)
            depth_raw = np.median(valid)
            distance_m = float(depth_raw) * DEPTH_SCALE

        return distance_m, (cx, cy)

    # ── depth 시각화 이미지 생성 ──
    def _make_depth_visualization(self, rgb_shape=None):
        with self.depth_lock:
            if self.depth_frame is None:
                return None
            depth = self.depth_frame.copy()

        if depth.dtype == np.uint16:
            depth_vis = depth.astype(np.float32) * DEPTH_SCALE
        else:
            depth_vis = depth.astype(np.float32)

            # float인데 mm일 수도 있어서 매우 큰 값이면 m로 변환
            if np.nanmax(depth_vis) > 50:
                depth_vis = depth_vis * DEPTH_SCALE

        depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=0.0, neginf=0.0)
        depth_vis = np.clip(depth_vis, 0, NORMALIZE_DEPTH_RANGE)
        depth_vis = (depth_vis / NORMALIZE_DEPTH_RANGE * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        return depth_colored

    # ── bbox + 라벨 그리기 ──
    def _draw_box(self, img, x1, y1, x2, y2, label, color, thickness=2):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        text_top = max(0, y1 - th - 8)
        text_bottom = max(th + 2, y1)

        cv2.rectangle(img, (x1, text_top), (x1 + tw + 4, text_bottom), color, -1)
        cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
def main():
    for path, name in [(YOLO_PT, "YOLO pt"), (RESNET_PT, "ResNet18 pt")]:
        if not os.path.exists(path):
            print(f"❌ {name} 파일 없음: {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_PT)
    yolo_classes = list(yolo_model.names.values())
    print(f"YOLO 로드 완료 | classes: {yolo_classes}")

    resnet_model, resnet_transform, resnet_classes = load_resnet18(RESNET_PT, device)

    rclpy.init()
    node = DualModelDepthDetector(
        yolo_model, resnet_model, resnet_transform,
        yolo_classes, resnet_classes, device
    )

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C로 종료")
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("종료 완료")


if __name__ == '__main__':
    main()