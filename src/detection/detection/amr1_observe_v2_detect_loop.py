import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, PoseWithCovariance
from std_msgs.msg import Bool, msg
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


# ── 설정 ──────────────────────────────────────────────────────────────────
NS             = "/robot4"
YOLO_WEIGHTS   = "./slam_turtlebot/src/models/amr1/yolo11n_amr1_v2.pt"
CONF_THRESH    = 0.5
DX, DY         = 0, 5
LOG_INTERVAL   = 0.5
WIN_W, WIN_H   = 960, 540

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

RGB_TOPIC   = "robot4/oakd/rgb/image_raw/compressed"
DEPTH_TOPIC = "robot4/oakd/stereo/image_raw/compressedDepth"
ROBOT_START_TOPIC = "/robot4/occupation"
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

        self.lock = threading.Lock()

        self.rgb_img = None
        self.yolo = YOLO(YOLO_WEIGHTS)
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.navigator = TurtleBot4Navigator()
        # self.goal_dir = TurtleBot4Directions.EAST

        self.gui_thread_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.pop_gui, daemon=True)
        self.gui_thread.start()



        self.create_subscription(
            CompressedImage, RGB_TOPIC,
            self.decode_rgb_cb, qos_profile_sensor_data
            )    
        self.create_subscription(
            CompressedImage, DEPTH_TOPIC,
            self._cb_depth, qos_profile_sensor_data
            )    
        self.create_subscription(
            Bool, ROBOT_START_TOPIC, 
            self.check_yolo_start_cb
            )
        self.create_publisher(
            Bool, ENEMY_POS_TOPIC
            )    
        self.create_publisher(
            Bool, ROBOT_PULLOUT_TOPIC
        )

        self.start_timer = self.create_timer(5.0, self.start_trnsform)

    def start_transform(self):
        """5초 대기 후 호출. 0.2초 주기의 메인 처리 타이머를 시작한다."""
        self.get_logger().info('TF Tree 안정화 완료. 루프 시작합니다.')
        self.timer = self.create_timer(0.2, self.display_images)
        self.start_timer.cancel()  # 1회성 타이머 해제

    def check_amcl_cb(self, msg):
        '''AMCL의 위치, 방향'''
        pass


    def decode_rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self.img_lock:
                self.rgb_img = img

    def check_yolo_start_cb(self, msg):
        '''객체탐지 시작 메시치 처리 cb'''
        if msg.data == True:
            self.mode = 'detect'
            self.detect_yolo()
            self.get_logger().info('탐지 상태 전환')
        else:
            self.get_logger().info('대기 상태 유지')

    def draw_bbox(self, imgs):
        '''bbox 선택'''
        results = self.yolo(imgs)
        best_conf = 0.0
        best_bbox = None
        best_center = None
        annotated_img = imgs.copy()

        for result in results:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names[cls_id]
                conf = float(box.conf[0])

                if cls_name.lower() == 'enemy' and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_bbox   = (x1, y1, x2, y2)
                    best_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        detected = best_bbox is not None
        return detected, best_bbox, best_center, annotated_img

    def display_image(self):
        '''객체탐지 : cls, bbox, center.'''
        with self.lock:
            rgb_img = self.rgb_img.copy()
        
        detected, bbox, center, rgb_img = self.draw_bbox(rgb_img)

        with self.lock:
            self.enemy_detected = detected
            self.enemy_bbox = bbox
            self.enemy_center = center

        if self.enemy_detected:
            self.get_logger().info(f'enemy detected at bbox={bbox}, center={center}')

    def pop_gui(self):
        while not self.gui_thread_stop.is_set():
            with self.lock:
                img = self.display_image().copy() if self.display_image is not None else None

            if img is not None:
                cv2.imshow('AMR1', img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.get_logger().info('Shutdown requested by user.')
                    self.shutdown_requested = True
                    self.gui_thread_stop.set()
                    rclpy.shutdown()
            else:
                cv2.waitKey(10)  # 이미지가 없으면 잠시 대기


        


    # def contain_img():
    #     '이미지 저장 : 5개. 6번째 이미지가 있는 경우 가장 먼저 들어온 이미지 버림queue, 이미지 처리 단계인 이미지의 경우 다음 이미지 선택'
    #     pass

    def check_bbox_center():
        '''방향 계산 : 터틀봇의 화면 중심점과 bbox의 중심점간 차이 계산.'''
        '''중심점이 없는 경우(객체 소실) 이전 계산값([가장 마지막 탐지 시점]과 [가장 최근 소실 시점]) 적용.'''
        '''만약 t-2에도 없으면 10도 회전.'''
        pass

    def rotate_turtle1():
        '''로봇방향 회전 : 터틀봇 회전'''
        pass

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

    node.gui_thread_stop.set()   # GUI 스레드 종료 신호
    node.gui_thread.join()       # GUI 스레드 종료 대기
    node.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()


if __name__=='main':
    main()
