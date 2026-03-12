import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, PoseWithCovariance
from std_msgs.msg import Bool
import threading

from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

import cv2
from ultralytics import YOLO


import numpy as np
import math

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
    super().__init__('amr_observe')
    self.navigator = TurtleBot4Directions()
    # 상태머신
    # WAIT
    # DETECT : 발견
    # TRACK : 추적(회전)
    self.mode = 'WAIT'

    self.lock = threading.Lock()

    self.rgb_img = None
    self.yolo = YOLO(YOLO_WEIGHTS)





    self.create_subscription(
        CompressedImage, RGB_TOPIC,
        self.cb, qos_profile_sensor_data
        )    
    self.create_subscription(
        CompressedImage, DEPTH_TOPIC,
        self._cb_depth, qos_profile_sensor_data
        )    
    self.create_subscription(
        Bool, ROBOT_START_TOPIC, 
        self.check_yolo_time_cb
        )
    self.create_publisher(
        Bool, ENEMY_POS_TOPIC
        )    
    self.create_publisher(
        Bool, ROBOT_PULLOUT_TOPIC
    )

    def decode_rgb_cb():
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self.img_lock:
                self.rgb_img = img


    def check_yolo_time_cb():
        '객체탐지 시작 메시치 처리 cb'
        pass
    
    def detect_yolo():
        '객체탐지 : cls, bbox, center.'
        pass

    # def contain_img():
    #     '이미지 저장 : 5개. 6번째 이미지가 있는 경우 가장 먼저 들어온 이미지 버림queue, 이미지 처리 단계인 이미지의 경우 다음 이미지 선택'
    #     pass

    def check_bbox_center():
        '방향 계산 : 터틀봇의 화면 중심점과 bbox의 중심점간 차이 계산.'
        '중심점이 없는 경우(객체 소실) 이전 계산값([가장 마지막 탐지 시점]과 [가장 최근 소실 시점]) 적용.'
        '만약 t-2에도 없으면 10도 회전.' 
        pass

    def rotate_turtle1():
        '로봇방향 회전 : 터틀봇 회전'
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
    rclpy.spin(node)

if __name__=='main':
    main()
