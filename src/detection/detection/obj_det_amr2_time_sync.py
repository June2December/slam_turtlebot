# obj_det_arm2_time_sync.py
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer # 이거 timestamp synch 하는용 (depth & rgb) - 둘이 시간 맞춰서 callback 호출해주는거 (동기화) - 둘이 시간차이가 좀 있어도 같이 처리할 수 있게 해주는거

from cv_bridge import CvBridge

# ── 설정 ──────────────────────────────────────────────────────────────────
NS           = "/robot1"    
YOLO_WEIGHTS = "/home/sinya/slam_turtlebot/src/models/arm2/best.pt"
TARGET_LABEL = "unknown"
CONF_THRESH  = 0.5
DX, DY       = 0, 5
LOG_INTERVAL = 0.5
WIN_W, WIN_H = 704, 704

DEPTH_TOPIC = f"{NS}/oakd/stereo/image_raw/compressedDepth"
RGB_TOPIC   = f"{NS}/oakd/rgb/image_raw/compressed"


# ── CompressedDepth 디코딩 ────────────────────────────────────────────────
def decode_compressed_depth(msg) -> np.ndarray | None:
    """
    ROS2 CompressedDepth 디코딩
    포맷: 앞 12바이트 = depth 파라미터 헤더 (ROS 호환)
        나머지     = PNG 압축 데이터 (16UC1)

    헤더 스킵 후 cv2.imdecode(IMREAD_UNCHANGED) → uint16 depth (mm)
    """
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    # PNG 매직넘버 위치 탐색 (헤더 길이가 다를 수 있음)
    PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])
    idx = raw.tobytes().find(PNG_MAGIC)

    if idx < 0:
        # PNG 가 아닌 경우 (RVL 인코딩 등) - 12바이트 고정 스킵
        idx = 12

    img = cv2.imdecode(raw[idx:], cv2.IMREAD_UNCHANGED)
    return img  # uint16 (mm) 또는 None


# ── Pixel Alignment ───────────────────────────────────────────────────────
def apply_offset(depth, dx, dy): # rgb, depth 카메라 physical 거리 차이있어서
    # 
    if dx == 0 and dy == 0:
        return depth
    h, w = depth.shape[:2]
    M = np.float64([[1, 0, -dx], 
                    [0, 1, -dy]]) # 2x3
    # bordervalue=0 >> 빈공간 0으로 채움
    return cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)



# ── ROI Median Depth → 미터 ───────────────────────────────────────────────
def get_depth_to_m(depth_al, x1, y1, x2, y2, shrink=0.5):
    h, w   = depth_al.shape[:2]
    # center points
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


# ══════════════════════════════════════════════════════════════════════════
class DetectDepthNode(Node):
    def __init__(self):
        super().__init__("amr2_detector_w_depth")
        self.bridge = CvBridge()
        self.K = None  # Camera intrinsics
        self.lock = threading.Lock()

        self._img_lock  = threading.Lock() # lock 안걸면 interrupt 당하니까
        self._disp_lock = threading.Lock()
        self._pair_lock = threading.Lock() 
        self._rgb       = None
        self._depth     = None
        self._disp      = None  # display

        self.model = YOLO(YOLO_WEIGHTS)
        self.get_logger().info(f"YOLO 로드: {YOLO_WEIGHTS}")

        # 두 토픽 모두 CompressedImage 타입으로 구독 >> sensor_msgs.msg.CompressedImage
        '''
        qos_profile_sensor_data:
        QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability  = DurabilityPolicy.VOLATILE,
            history     = HistoryPolicy.KEEP_LAST,
            depth       = 5
        )
        '''
        self.rgb_sub = self.create_subscription(msg_type= CompressedImage, topic= RGB_TOPIC, callback= self.rgb_callback, qos_profile= qos_profile_sensor_data)

        self.depth_sub = self.create_subscription(msg_type= CompressedImage, topic= DEPTH_TOPIC, callback= self.depth_callback, qos_profile= qos_profile_sensor_data)

        self._stop = threading.Event()
        self._last_log = 0.0

        #
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        #

        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._gui,    daemon=True).start()

        self.get_logger().info(
            f"구독하는 토픽 타입 >> \n  RGB  : {RGB_TOPIC}\n  Depth: {DEPTH_TOPIC}")
        
        
    # origin
    # def sync_callback(self, rgb_msg, depth_msg):
    #     rgb_arr = np.frombuffer(rgb_msg.data, np.uint8)
    #     rgb_img = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)

    #     depth_img = decode_compressed_depth(depth_msg)
    #     if depth_img is not None and depth_img.dtype not in (np.uint16, np.float32):
    #         depth_img = depth_img.astype(np.uint16)

    #     if rgb_img is None or depth_img is None:
    #         return

    #     stamp_sec = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

    #     with self._pair_lock:
    #         # 최신 sync pair만 유지
    #         self._latest_pair = (rgb_img, depth_img, stamp_sec)

    # time sync 콜백
    def synced_callback(self, rgb_msg, depth_msg):
        """ Called when RGB and Depth frames arrive together (synchronized) """
        try:
            with self.lock:
                # Decode RGB image from compressed format
                np_arr = np.frombuffer(rgb_msg.data, np.uint8)
                rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if rgb is not None and rgb.size > 0:
                    if not self.logged_rgb_shape:
                        self.get_logger().info(f"RGB image shape: {rgb.shape}")
                        self.logged_rgb_shape = True
                    self.rgb_image = rgb

                # Convert Depth image from ROS format to OpenCV
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                if depth is not None and depth.size > 0:
                    if not self.logged_depth_shape:
                        self.get_logger().info(f"Depth image shape: {depth.shape}")
                        self.logged_depth_shape = True
                    self.depth_image = depth
                    self.camera_frame = depth_msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Synced callback failed: {e}")


    # ── ROS 콜백 ──────────────────────────────────────────────────────────
    def rgb_callback(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self._img_lock:
                self._rgb = img

    def depth_callback(self, msg):
        # CompressedDepth 전용 디코딩
        img = decode_compressed_depth(msg)
        if img is not None and img.dtype in (np.uint16, np.float32):
            with self._img_lock:
                self._depth = img
        elif img is not None:
            # uint16 아닌 경우 변환
            with self._img_lock:
                self._depth = img.astype(np.uint16)

    # ── Worker >> 
    def _worker(self):
        while not self._stop.is_set():
            t0 = time.monotonic()

            with self._img_lock:
                rgb   = None if self._rgb   is None else self._rgb.copy()
                depth = None if self._depth is None else self._depth.copy()

            # depth 없어도 RGB만 있으면 탐지는 표시
            if rgb is None:
                time.sleep(0.03)
                continue

            frame  = rgb.copy()
            best_z = None

            if depth is not None:
                depth_al = apply_offset(depth, DX, DY)
            else:
                depth_al = None
                self.get_logger().warn(
                    "Depth 미수신 - RGB만 표시 중", throttle_duration_sec=5.0)

            results = self.model(rgb, verbose=False)[0]

            for det in results.boxes:
                label = self.model.names[int(det.cls[0])].lower()
                conf  = float(det.conf[0])
                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                z_m = get_depth_to_m(depth_al, x1, y1, x2, y2) \
                    if depth_al is not None else None

                col      = (0, 255, 0) if label == TARGET_LABEL else (180, 180, 180)
                dist_str = f"{z_m:.2f}m" if z_m else "---"

                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame,
                            f"{label} {conf:.2f}  {dist_str}",
                            (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cv2.drawMarker(frame, (cx, cy),
                            (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

                if label == TARGET_LABEL and z_m and best_z is None:
                    best_z = z_m

            # 0.5초마다 터미널 출력
            now = time.time()
            if now - self._last_log >= LOG_INTERVAL:
                self._last_log = now
                if best_z is not None:
                    self.get_logger().info(
                        f"[{TARGET_LABEL}]  {best_z:.3f} m  "
                        f"({best_z*1000:.0f} mm)  "
                        f"align dx={DX} dy={DY}")
                else:
                    self.get_logger().info(f"[{TARGET_LABEL}]  탐지 없음")

            with self._disp_lock:
                self._disp = frame

            time.sleep(max(0.0, 0.033 - (time.monotonic() - t0)))

    # ── GUI ───────────────────────────────────────────────────────────────
    def _gui(self):
        WIN = "Detection + Depth"
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