import csv
import time

import cv2
import numpy as np

from rclpy.time import Time
from rclpy.duration import Duration
from ultralytics import YOLO

from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

# compressedDepth 헤더 스킵용 PNG 시그니처
_PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


class TargetTracker(Node):
    def __init__(self):
        super().__init__('amr1_tracking_aerial')

        self.bridge = CvBridge()
        self.tracking_enabled = False

        self.rotate_kp = 0.8
        self.max_angular_speed = 0.8
        self.min_angular_speed = 0.05
        self.center_tolerance_ratio = 0.05

        self.yolo_model_path = '/home/rokey/slam_turtlebot/db/best11.pt'
        self.model = YOLO(self.yolo_model_path)

        ns = self.get_namespace()
        self.rgb_topic   = f'{ns}/oakd/rgb/image_raw/compressed'
        self.depth_topic = f'{ns}/oakd/stereo/image_raw/compressedDepth'

        # pub / sub
        self.create_subscription(Bool, '/occupation', self.occupation_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ===================== [DEBUG 퍼블리셔 시작] =====================
        # rqt_image_view 에서 {ns}/dbg/image 토픽 선택해서 확인
        # 바운딩박스 + 중앙 십자선 + depth 수치 오버레이
        # 확인 끝나면 이 블록 + synced_callback 안의 DEBUG 블록 같이 삭제
        self._dbg_pub   = self.create_publisher(Image, f'{ns}/dbg/image', 1)
        self._dbg_timer = self.create_timer(0.05, self._dbg_publish)
        self._dbg_frame = None
        # ===================== [DEBUG 퍼블리셔 끝] =====================

        # sync
        self.rgb_sub   = Subscriber(self, CompressedImage, self.rgb_topic)
        self.depth_sub = Subscriber(self, CompressedImage, self.depth_topic)
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.logged_model_names = False
        self.logged_shape = False

        # tf
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 카메라 내부 파라미터
        self.K = None
        self.camera_frame = None
        self.camera_info_topic = f'{ns}/oakd/rgb/camera_info'
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # CSV
        self.csv_path = '/home/rokey/slam_turtlebot/src/amr_control/amr_control/data/depth_log.csv'
        self.csv_file   = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # depth MAF
        self.depth_history    = []
        self.depth_maf_window = 7

        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(['time', 'dt', 'u', 'v', 'depth_raw', 'depth_maf',
                                      'cam_x', 'cam_y', 'cam_z', 'map_x', 'map_y', 'map_z'])
            self.csv_file.flush()

        print('트래커 시작')

    def camera_info_callback(self, msg):
        self.K            = np.array(msg.k).reshape(3, 3)
        self.camera_frame = msg.header.frame_id

    def pixel_to_3d(self, u, v, depth_value):
        if self.K is None:
            return None
        z = float(depth_value) / 1000.0
        if z <= 0.0:
            return None
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return (u - cx) * z / fx, (v - cy) * z / fy, z

    def apply_depth_maf(self, depth_value):
        val = float(depth_value)
        # cold start: 첫 값으로 윈도우 전체를 채워서 초반 튀는 거 방지
        if len(self.depth_history) == 0:
            self.depth_history = [val] * self.depth_maf_window
        else:
            self.depth_history.append(val)
            if len(self.depth_history) > self.depth_maf_window:
                self.depth_history.pop(0)
        return float(np.mean(self.depth_history))

    def camera_point_to_map(self, x, y, z, stamp):
        if self.camera_frame is None:
            return None
        pt = PointStamped()
        pt.header.stamp    = stamp
        pt.header.frame_id = self.camera_frame
        pt.point.x, pt.point.y, pt.point.z = x, y, z
        try:
            # tf_buffer.transform 방식 — lookup + do_transform 한번에
            pt_map = self.tf_buffer.transform(pt, 'map', timeout=Duration(seconds=0.3))
            return pt_map.point.x, pt_map.point.y, pt_map.point.z
        except Exception as e:
            print(f'TF 실패: {e}')
            return None

    def occupation_callback(self, msg):
        self.tracking_enabled = msg.data
        print('추적 ON' if self.tracking_enabled else '추적 OFF')
        if not self.tracking_enabled:
            self.stop_rotation()

    def synced_callback(self, rgb_msg, depth_msg):
        rgb_t   = rgb_msg.header.stamp.sec   + rgb_msg.header.stamp.nanosec   * 1e-9
        depth_t = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        dt = abs(rgb_t - depth_t)

        print(f'sync dt={dt:.3f}s')

        if not self.tracking_enabled:
            return

        try:
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            rgb    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            depth  = self.decode_compressed_depth(depth_msg)
        except Exception as e:
            print(f'이미지 변환 실패: {e}')
            return

        if rgb is None or depth is None:
            print('rgb 또는 depth None')
            return

        if not self.logged_shape:
            print(f'rgb={rgb.shape}  depth={depth.shape} {depth.dtype}')
            self.logged_shape = True

        results = self.model.predict(source=rgb, conf=0.4, verbose=False)

        if not self.logged_model_names:
            print(f'클래스: {self.model.names}')
            self.logged_model_names = True

        if not results or len(results[0].boxes) == 0:
            print('감지 없음')
            self.stop_rotation()
            return

        best_box, best_conf, best_cls_name = None, -1.0, ''
        for box in results[0].boxes:
            conf     = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_id   = int(box.cls[0].item())    if box.cls  is not None else -1
            cls_name = self.model.names.get(cls_id, str(cls_id))
            if conf > best_conf:
                best_box      = box.xyxy[0].cpu().numpy().astype(int)
                best_conf     = conf
                best_cls_name = cls_name

        if best_box is None:
            self.stop_rotation()
            return

        x1, y1, x2, y2 = best_box.tolist()
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        print(f'감지: {best_cls_name}  conf={best_conf:.2f}  center=({u},{v})')

        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            print(f'중앙점 depth 범위 밖: ({u},{v})')
            self.stop_rotation()
            return

        y_min = max(0, v - 1);  y_max = min(depth.shape[0], v + 2)
        x_min = max(0, u - 1);  x_max = min(depth.shape[1], u + 2)
        patch = depth[y_min:y_max, x_min:x_max]
        valid = patch[patch > 0]

        if valid.size == 0:
            print(f'유효 depth 없음 ({u},{v})')
            self.stop_rotation()
            return

        depth_value_raw = float(np.median(valid))
        depth_value     = self.apply_depth_maf(depth_value_raw)
        print(f'depth  raw={depth_value_raw:.0f}mm  maf={depth_value:.0f}mm  ({depth_value/1000:.2f}m)')

        cam_xyz = self.pixel_to_3d(u, v, depth_value)
        if cam_xyz is None:
            print('pixel_to_3d 실패')
            self.stop_rotation()
            return

        map_xyz = self.camera_point_to_map(cam_xyz[0], cam_xyz[1], cam_xyz[2], rgb_msg.header.stamp)
        if map_xyz is None:
            print('TF 실패 — cam xyz만 저장')
            map_xyz = (np.nan, np.nan, np.nan)

        print(f'cam=({cam_xyz[0]:.3f},{cam_xyz[1]:.3f},{cam_xyz[2]:.3f})  '
              f'map=({map_xyz[0]:.3f},{map_xyz[1]:.3f},{map_xyz[2]:.3f})')

        self.csv_writer.writerow([
            time.time(), dt, u, v,
            depth_value_raw, depth_value,
            cam_xyz[0], cam_xyz[1], cam_xyz[2],
            map_xyz[0], map_xyz[1], map_xyz[2],
        ])
        self.csv_file.flush()

        # ===================== [DEBUG 오버레이 시작] =====================
        dbg = rgb.copy()
        h, w = dbg.shape[:2]
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (180, 180, 180), 1)          # 화면 중앙 기준선
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)               # 바운딩박스
        cv2.putText(dbg, f'{best_cls_name} {best_conf:.2f}',
                    (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cross = 12
        cv2.line(dbg, (u - cross, v), (u + cross, v), (0, 255, 255), 2)      # 십자선
        cv2.line(dbg, (u, v - cross), (u, v + cross), (0, 255, 255), 2)
        cv2.circle(dbg, (u, v), 4, (0, 255, 255), -1)
        cv2.putText(dbg, f'{depth_value/1000:.2f}m',
                    (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self._dbg_frame = dbg
        # ===================== [DEBUG 오버레이 끝] =====================

        angular_z = self.compute_angular_z(u, rgb.shape[1])
        self.publish_rotation(angular_z)
        print(f'angular.z={angular_z:.3f}')

    # ===================== [DEBUG 퍼블리셔 시작] =====================
    def _dbg_publish(self):
        if self._dbg_frame is None:
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(self._dbg_frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self._dbg_pub.publish(msg)
        except Exception as e:
            print(f'dbg 퍼블리시 실패: {e}')
    # ===================== [DEBUG 퍼블리셔 끝] =====================

    def compute_angular_z(self, u, image_width):
        center_x   = image_width / 2.0
        error      = u - center_x
        norm_error = error / center_x
        angular_z  = -self.rotate_kp * norm_error
        if abs(angular_z) > self.max_angular_speed:
            angular_z = self.max_angular_speed * np.sign(angular_z)
        if abs(norm_error) < self.center_tolerance_ratio:
            angular_z = 0.0
        elif 0.0 < abs(angular_z) < self.min_angular_speed:
            angular_z = self.min_angular_speed * np.sign(angular_z)
        return float(angular_z)

    def publish_rotation(self, angular_z):
        if not rclpy.ok():
            return
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = float(angular_z)
        try:
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            print(f'cmd_vel 발행 실패: {e}')

    def stop_rotation(self):
        self.publish_rotation(0.0)

    def decode_compressed_depth(self, msg: CompressedImage):
        raw    = bytes(msg.data)
        offset = raw.find(_PNG_MAGIC)
        if offset < 0:
            print(f'PNG 못 찾음  format={msg.format}  len={len(raw)}')
            return None
        png_bytes = np.frombuffer(raw[offset:], dtype=np.uint8)
        depth     = cv2.imdecode(png_bytes, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print('depth imdecode 실패')
        return depth  # 16UC1, mm 단위


def main():
    rclpy.init()
    node     = TargetTracker()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.stop_rotation()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        try:
            node.csv_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()