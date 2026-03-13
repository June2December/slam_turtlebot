import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import (PointStamped, PoseStamped, 
                               Quaternion, PoseWithCovariance, 
                               PoseWithCovarianceStamped, Twist
                               )
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import threading

from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

import cv2
from ultralytics import YOLO

import numpy as np
import math
from time import time
from collections import deque

# ── 설정 ──────────────────────────────────────────────────────────────────
NS             = "/robot4"
YOLO_WEIGHTS   = "./src/models/amr1/yolo11n_amr1_v2.pt"
CONF_THRESH    = 0.3
DX, DY         = 0, 5
LOG_INTERVAL   = 0.5

CLASSES = ["balloon", "bird", "enemy", "friend"]

# 거리 로그 제외 클래스
LOG_EXCLUDE = {"friend"}

# 클래스별 바운딩박스 색상 (BGR)
COLOR_MAP = {
    "balloon": (0, 165, 255),   # 주황
    "bird":    (255, 255,   0), # 하늘
    "enemy":   (0,   0, 255),   # 빨강
    "friend":  (0, 255,   0),   # 초록
}

RGB_TOPIC   = "/robot4/oakd/rgb/image_raw/compressed"
DEPTH_TOPIC = "/robot4/oakd/stereo/image_raw/compressedDepth"
ROBOT_POSE = "/robot4/amcl_pose"
ROBOT_VELOCITY = "/robot4/cmd_vel"

ROBOT_DETECT_TOPIC = "/robot4/occupation"
ENEMY_POS_TOPIC = "/AMR_1/enemy_pos"
ROBOT_PULLOUT_TOPIC = "/AMR_1/tracking_done"

class AmrObserve(Node):
    def __init__(self):
        super().__init__('amr_observe')

        # 상태머신
        # WAIT : 노드 대기상태
        # DETECT : 최초 발견
        # TRACK : 객체 추적(회전)
        self.mode = 'WAIT'
        self.yolo = YOLO(YOLO_WEIGHTS)
        self.lock = threading.Lock()

        self.rgb_img = None
        self.annotated_img = None
        self.enemy_centers = deque(maxlen=3)
        self.current_yaw = 0.0
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.navigator = TurtleBot4Navigator()
        # self.goal_dir = TurtleBot4Directions.EAST

        # self.gui_thread_stop = threading.Event()
        # self.gui_thread = threading.Thread(target=self.pop_gui, daemon=True)
        # self.gui_thread.start()

        # self.create_subscription(
        #     CompressedImage, RGB_TOPIC,
        #     self.decode_rgb_cb, qos_profile_sensor_data
        #     )    
        # self.create_subscription(
        #     CompressedImage, DEPTH_TOPIC,
        #     self.depth_cb, qos_profile_sensor_data
        #     )

        self.create_subscription(
            PoseWithCovarianceStamped, ROBOT_POSE,
            self.check_amcl_cb, 10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist, ROBOT_VELOCITY, 10
        )

        # self.create_subscription(
        #     Bool, ROBOT_DETECT_TOPIC, 
        #     self.check_yolo_start_cb, 10
        #     )
        self.create_publisher(
            Bool, ENEMY_POS_TOPIC, 10
            )
        # self.create_publisher(
        #     Bool, ROBOT_PULLOUT_TOPIC, 10
        # )

    def check_amcl_cb(self, msg):
        '''AMCL의 위치, 방향'''
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )
        with self.lock:
            self.current_yaw = yaw

    def rotate_control_cb(self):
        with self.lock:
            current = self.current_yaw

        error = math.atan2(
            math.sin(self.target_yaw - current),
            math.cos(self.target_yaw - current)
        )

        twist = Twist()
        if abs(error) > self.rotate_tol:
            twist.angular.z = self.angular_speed if error > 0 else -self.angular_speed
        else:
            twist.angular.z = 0.0
            self.rotate_timer.cancel()
            self.get_logger().info(f'회전 완료: yaw={math.degrees(current):.1f}°')

        self.cmd_vel_pub.publish(twist)

    # def decode_rgb_cb(self, msg):
    #     arr = np.frombuffer(msg.data, np.uint8)
    #     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #     if img is not None:
    #         with self.lock:
    #             self.rgb_img = img

    # def depth_cb(self):
    #     pass

    # def check_yolo_start_cb(self, msg):
    #     '''객체탐지 시작 메시치 처리 cb'''
    #     if msg.data == True:
    #         self.mode = 'DETECT'
    #         self.detect_timer = self.create_timer(0.2, self.detect_object_yolo)
    #         self.get_logger().info('탐지 상태 전환')
    #     else:
    #         self.mode = 'WAIT'
    #         if hasattr(self, 'detect_timer'):
    #             self.detect_timer.cancel()
    #         self.get_logger().info('대기 상태 유지')
    
    # def draw_bbox(self, imgs):
    #     '''bbox 선택'''
    #     results = self.yolo(imgs)
    #     # best_conf = 0.0
    #     best_bbox = None
    #     best_center = None
    #     annotated_img = imgs.copy()

    #     for result in results:
    #         for box in result.boxes:
    #             cls_id = int(box.cls[0])
    #             cls_name = self.yolo.names[cls_id]
    #             conf = float(box.conf[0])
                
    #             if conf < CONF_THRESH:
    #                 continue

    #             # if cls_name.lower() == 'enemy' and conf > best_conf:
    #             if cls_name.lower() == 'enemy':
    #                 # best_conf = conf
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 best_bbox   = (x1, y1, x2, y2)
    #                 best_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    #                 cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #                 label = f'enemy {conf:.2f}'
    #                 cv2.putText(
    #                     annotated_img, label, (x1, y1 - 8),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    #                 )

    #     detected = best_bbox is not None
    #     return detected, best_bbox, best_center, annotated_img

    # def detect_object_yolo(self):
    #     '''객체탐지 : cls, bbox, center.'''
    #     with self.lock:
    #         if self.rgb_img is None: return
    #         rgb_img = self.rgb_img.copy()
        
    #     # rgb_img = cv2.resize(rgb_img, (640, 540))
    #     detected, bbox, center, annotated_img = self.draw_bbox(rgb_img)

    #     with self.lock:
    #         self.enemy_detected = detected
    #         self.enemy_bbox = bbox
    #         self.enemy_center = center
    #         self.annotated_img = annotated_img
    #         if center is not None:
    #             self.enemy_centers.append(center)

    #     if self.enemy_detected:
    #         self.get_logger().info(f'적 항체 bbox={bbox}, center={center}')

    # def pop_gui(self):
    #     '''화면 표시'''
    #     while not self.gui_thread_stop.is_set():
    #         with self.lock:
    #             img = self.annotated_img.copy() if self.annotated_img is not None else None

    #         if img is not None:
    #             cv2.imshow('AMR1', img)
    #             key = cv2.waitKey(1)
    #             if key == ord('q'):
    #                 self.get_logger().info('종료 버튼 입력')
    #                 self.shutdown_requested = True
    #                 self.gui_thread_stop.set()
    #                 rclpy.shutdown()
    #         else:
    #             cv2.waitKey(10)  # 이미지가 없으면 잠시 대기

    # def calculate_rotate_degree(self):
    #     '''방향 계산 : 터틀봇의 화면 중심점과 bbox의 중심점간 차이 계산.'''
    #     '''중심점이 없는 경우(객체 소실) 이전 계산값([가장 마지막 탐지 시점]과 
    #     [가장 최근 소실 시점]) 적용.'''
    #     '''만약 t-2에도 없으면 10도 회전.'''
    #     # latest = self.enemy_centers[-1]  # 가장 최신 중심점
    #     with self.lock:
    #         yaw = self.current_yaw
    #         if not self.enemy_centers: return
    #         cx, cy = self.enemy_centers[-1]

    #     img_cx = 640 // 2
    #     dx = cx - img_cx

    def rotate_turtle(self, degrees, angular_speed=0.3, tol=0.05):
        '''로봇방향 회전 : 터틀봇 회전'''
        with self.lock:
            start_yaw = self.current_yaw

        delta_rad = math.radians(degrees)
        self.target_yaw = math.atan2(
        math.sin(start_yaw - delta_rad),
        math.cos(start_yaw - delta_rad)
        )
        self.angular_speed = angular_speed
        self.rotate_tol = tol
        self.rotate_timer = self.create_timer(0.05, self.rotate_control_cb)
        self.get_logger().info(f'회전 시작: {degrees}° 시계방향, 목표 yaw={math.degrees(self.target_yaw):.1f}°')

                        # def publish_object_img_cb():
                        #     '객체 탐지 img 발행'
                        #     pass

    # def calculate_pixtomap():
    #     '계산 pixel to map for UI'
    #     pass

    # def publish_map_pos_cb():
    #     '객체 위치 전송'
    #     pass

def main():
    rclpy.init()
    node = AmrObserve()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # node.gui_thread_stop.set()   # GUI 스레드 종료 신호
    # node.gui_thread.join()       # GUI 스레드 종료 대기
    node.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()

if __name__=='__main__':
    main()
