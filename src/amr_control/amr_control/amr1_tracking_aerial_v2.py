import csv
import math
import time

import cv2
import numpy as np
import struct
from ultralytics import YOLO

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


SCAN_SPEED = 0.5
LOST_TIMEOUT = 3.0
SWEEP_DURATION = 2 * math.pi / SCAN_SPEED   # ~12.6초


class TargetTracker(Node):
    def __init__(self):
        super().__init__('amr1_tracking_aerial_retracking')

        # -------- 기본값 --------
        self.bridge = CvBridge()
        self.tracking_enabled = False

        self.rotate_kp = 0.8
        self.max_angular_speed = 0.8
        self.min_angular_speed = 0.05
        self.center_tolerance_ratio = 0.05
        self.last_cmd_time = 0.0
        self.cmd_interval = 0.1  # 100ms — 네트워크+로봇 반응 지연

        # 소실 처리
        self.last_detected_time = None
        self.final_sweep_start = None
        self.last_angular_z = SCAN_SPEED  # 기본값

        self.yolo_model_path = './db/yolo11n.pt'
        self.model = YOLO(self.yolo_model_path)

        ns = self.get_namespace()
        self.rgb_topic = f'/robot4/oakd/rgb/image_raw/compressed'
        self.depth_topic = f'/robot4/oakd/stereo/image_raw/compressedDepth'

        # -------- pub / sub --------
        self.create_subscription(Bool, 'occupation', self.occupation_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot4/cmd_vel', 10)
        self.tracking_done_pub = self.create_publisher(Bool, 'tracking_done', 10)
        self.track_pose_pub = self.create_publisher(PointStamped, '/amr1/track_pose', 10)

        # -------- sync --------
        self.rgb_sub = Subscriber(self, CompressedImage, self.rgb_topic)
        self.depth_sub = Subscriber(self, CompressedImage, self.depth_topic)
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.logged_model_names = False
        self.logged_shape = False

        # -------- tf --------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- 카메라 내부 파라미터 --------
        self.K = None
        self.camera_frame = None
        self.camera_info_topic = f'robot4/oakd/rgb/camera_info'


        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.csv_path = './src/amr_control/amr_control/data/depth_log.csv'
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 파일이 비어 있으면 헤더 쓰기
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(['time', 'dt', 'u', 'v', 'depth_center', 'cam_x', 'cam_y', 'cam_z', 'map_x', 'map_y', 'map_z'])
            self.csv_file.flush()

        self.get_logger().info('tracker start')

    # -------- 3D 점으로 바꿀때 그 info 읽어야 --------
    def camera_info_callback(self, msg):
        self.K = msg.k
        self.camera_frame = msg.header.frame_id
    
    def pixel_to_3d(self, u, v, depth_value):
        if self.K is None:
            return None

        fx = self.K[0]
        fy = self.K[4]
        cx = self.K[2]
        cy = self.K[5]

        z = float(depth_value) / 1000.0  # mm -> m
        if z <= 0.0:
            return None

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return x, y, z
    # -------- 카메라 점을 map 으로 tf 변환 --------
    def camera_point_to_map(self, x, y, z, stamp):
        if self.camera_frame is None:
            return None

        pt = PointStamped()
        pt.header.stamp = stamp
        pt.header.frame_id = self.camera_frame
        pt.point.x = x
        pt.point.y = y
        pt.point.z = z

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                'map',
                self.camera_frame,
                stamp
            )
        except Exception as e:
            self.get_logger().warn(f'tf lookup fail: {e}')
            return None

        # PointStamped 변환
        import tf2_geometry_msgs
        map_pt = tf2_geometry_msgs.do_transform_point(pt, tf_msg)
        return map_pt.point.x, map_pt.point.y, map_pt.point.z

    # -------- 추적 on/off --------
    def occupation_callback(self, msg):
        self.tracking_enabled = msg.data

        if self.tracking_enabled:
            self.last_detected_time = None
            self.final_sweep_start = None
            self.get_logger().info('tracking start')
        else:
            self.stop_rotation()
            self.get_logger().info('tracking stop')

    # -------- rgb/depth sync 확인 + yolo 확인 + 회전만 테스트 --------
    def synced_callback(self, rgb_msg, depth_msg):
        rgb_t = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        depth_t = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec * 1e-9
        dt = abs(rgb_t - depth_t)

        print(f'sync는 되나? | rgb, depth 의 dt={dt:.3f}s')

        if not self.tracking_enabled:
            return

        try:
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            depth = self.decode_compressed_depth(depth_msg)
        except Exception as e:
            self.get_logger().error(f'image convert fail: {e}')
            return

        if rgb is None or depth is None:
            self.get_logger().warn('rgb or depth is none')
            return

        if not self.logged_shape:
            self.get_logger().info(f'rgb shape={rgb.shape}, depth shape={depth.shape}')
            self.logged_shape = True

        results = self.model.predict(source=rgb, conf=0.4, verbose=False)

        if not self.logged_model_names:
            self.get_logger().info(f'class names: {self.model.names}')
            self.logged_model_names = True

        if not results or len(results) == 0:
            self.get_logger().info('no result')
            self._handle_lost()
            return

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.get_logger().info('no box')
            self._handle_lost()
            return

        # 지금은 가장 conf 높은 박스 1개만
        best_box = None
        best_conf = -1.0
        best_cls_name = ''

        for box in boxes:
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            cls_name = self.model.names.get(cls_id, str(cls_id))

            if cls_name != 'enemy':
                continue

            if conf > best_conf:
                best_box = box.xyxy[0].cpu().numpy().astype(int)
                best_conf = conf
                best_cls_name = cls_name

        if best_box is None:
            self.get_logger().info('best box none')
            self._handle_lost()
            return

        self.last_detected_time = time.time()
        self.final_sweep_start = None

        x1, y1, x2, y2 = best_box.tolist()
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        print(f'detect cls={best_cls_name}, conf={best_conf:.2f}, center=({u},{v})')
        depth_value = depth[v, u]

        cam_xyz = self.pixel_to_3d(u, v, depth_value)
        if cam_xyz is None:
            self.get_logger().warn('pixel_to_3d fail')
            return

        map_xyz = self.camera_point_to_map(
            cam_xyz[0], cam_xyz[1], cam_xyz[2], rgb_msg.header.stamp
        )
        if map_xyz is None:
            self.get_logger().warn('camera_point_to_map fail')
            return

        print(f'map xyz = ({map_xyz[0]:.3f}, {map_xyz[1]:.3f}, {map_xyz[2]:.3f})')

        track_pose_msg = PointStamped()
        track_pose_msg.header.stamp = rgb_msg.header.stamp
        track_pose_msg.header.frame_id = 'map'
        track_pose_msg.point.x = map_xyz[0]
        track_pose_msg.point.y = map_xyz[1]
        track_pose_msg.point.z = map_xyz[2]
        self.track_pose_pub.publish(track_pose_msg)

        self.csv_writer.writerow([
            time.time(),
            dt,
            u,
            v,
            float(depth_value),
            cam_xyz[0], cam_xyz[1], cam_xyz[2],
            map_xyz[0], map_xyz[1], map_xyz[2],
        ])
        self.csv_file.flush()

        print(f"픽셀 중앙: ({u},{v}) depth={depth_value}")

        angular_z = self.compute_angular_z(u, rgb.shape[1])
        self.last_angular_z = angular_z
        self.publish_rotation(angular_z)

        self.get_logger().info(f'cmd angular.z = {angular_z:.3f}')

    # -------- 화면 중앙 오차로 회전값 계산 --------
    def compute_angular_z(self, u, image_width):
        center_x = image_width / 2.0
        error = u - center_x
        norm_error = error / center_x

        angular_z = -self.rotate_kp * norm_error

        if abs(angular_z) > self.max_angular_speed:
            angular_z = self.max_angular_speed * np.sign(angular_z)

        if abs(norm_error) < self.center_tolerance_ratio:
            angular_z = 0.0
        elif 0.0 < abs(angular_z) < self.min_angular_speed:
            angular_z = self.min_angular_speed * np.sign(angular_z)

        return float(angular_z)

    # -------- 회전 명령 보냄 --------
    def publish_rotation(self, angular_z):
        if not rclpy.ok():
            return

        now = time.time()
        if now - self.last_cmd_time < self.cmd_interval:
            return
        self.last_cmd_time = now

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = float(angular_z)

        try:
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            self.get_logger().warn(f'cmd_vel publish skip: {e}')

    # -------- 회전 멈춤 --------
    def stop_rotation(self):
        self.publish_rotation(0.0)

    # -------- 소실 처리 --------
    def _handle_lost(self):
        now = time.time()
        print('handle lost 진입')

        if self.final_sweep_start is not None:
            elapsed = now - self.final_sweep_start
            print(f'리트레킹 | elapsed={elapsed:.1f}s / {SWEEP_DURATION:.1f}s')
            if elapsed < SWEEP_DURATION:
                print('회전 명령 발행')
                self.publish_rotation(self.last_angular_z)
            else:
                print('360 sweep 완료 → 정지')
                self.stop_rotation()
                self.tracking_enabled = False
                self.final_sweep_start = None
                self.last_detected_time = None
                self.get_logger().info('360 sweep done, tracking end')
                msg = Bool()
                msg.data = True
                self.tracking_done_pub.publish(msg)
            return

        if self.last_detected_time is None:
            print('멈춤1 : 객체탐지된 적 없음')
            self.stop_rotation()
            return

        if now - self.last_detected_time >= LOST_TIMEOUT:
            self.final_sweep_start = now
            self.get_logger().info('lost 3s → start 360 sweep')
            self.publish_rotation(SCAN_SPEED)
            return

    # -------- 압축 풀기 --------
    def decode_compressed_depth(self, msg: CompressedImage):
        try:
            # 예: "16UC1; compressedDepth png"
            # 예: "32FC1; compressedDepth png"
            fmt = msg.format
            if ';' not in fmt:
                self.get_logger().warn(f'invalid compressedDepth format: {fmt}')
                return None

            depth_fmt, compr_type = fmt.split(';')
            depth_fmt = depth_fmt.strip()
            compr_type = compr_type.strip()

            if 'compressedDepth' not in compr_type:
                self.get_logger().warn(f'not compressedDepth: {fmt}')
                return None

            # compressedDepth는 앞부분 헤더 + 뒤쪽 PNG 데이터 구조
            # 일반적으로 12바이트 헤더를 제거 후 PNG decode
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if raw.size <= 12:
                self.get_logger().warn('compressedDepth data too short')
                return None

            png_data = raw[12:]
            depth = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)

            if depth is None:
                self.get_logger().warn('cv2.imdecode failed for compressedDepth')
                return None

            # 16UC1이면 보통 여기서 끝
            if depth_fmt == '16UC1':
                return depth

            # 32FC1은 compressedDepth에서 quantization header를 해석해야 하는 경우가 있음
            # 현장에서는 raw depth topic 쓰는 것이 더 안전
            if depth_fmt == '32FC1':
                self.get_logger().warn(
                    '32FC1 compressedDepth detected. '
                    'PNG decode only may be insufficient; raw depth topic is recommended.'
                )
                return depth

            self.get_logger().warn(f'unknown depth format: {depth_fmt}')
            return depth

        except Exception as e:
            self.get_logger().error(f'decode_compressed_depth fail: {e}')
            return None


def main():
    rclpy.init()
    node = TargetTracker()
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