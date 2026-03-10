import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, PoseWithCovarianceStamped
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions
import tf2_geometry_msgs

import numpy as np
import cv2
import threading
import math
from ultralytics import YOLO


class DepthToMap(Node):
    """
    깊이(Depth) 카메라 이미지를 이용해 탐지된 자동차의 위치를
    ROS2 맵 좌표로 변환하고, TurtleBot4가 해당 목표를 향해
    자율 주행하도록 제어하는 노드.
    """

    def __init__(self):
        super().__init__('depth_to_map_node')

        # OpenCV <-> ROS 이미지 메시지 변환기
        self.bridge = CvBridge()

        # 카메라 내부 파라미터 행렬 (camera_info 수신 후 초기화)
        self.K = None

        # 멀티스레드 접근 보호를 위한 락
        self.lock = threading.Lock()

        # -------------------------------------------------------
        # 토픽 이름 설정 (네임스페이스 기반)
        # -------------------------------------------------------
        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'          # 깊이 이미지 토픽
        self.rgb_topic   = f'{ns}/oakd/rgb/image_raw/compressed'  # RGB 압축 이미지 토픽
        self.info_topic  = f'{ns}/oakd/rgb/camera_info'           # 카메라 내부 파라미터 토픽

        # -------------------------------------------------------
        # 이미지 / 상태 변수 초기화
        # -------------------------------------------------------
        self.depth_image       = None   # 최신 깊이 이미지
        self.rgb_image         = None   # 최신 RGB 이미지
        self.camera_frame      = None   # 깊이 이미지의 TF 프레임 ID
        self.display_image     = None   # GUI 표시용 합성 이미지
        self.shutdown_requested = False  # 종료 요청 플래그

        # 로봇의 현재 맵 좌표 (AMCL 콜백에서 갱신)
        self.robot_x           = None
        self.robot_y           = None
        self.robot_orientation = None

        # =========================
        # 상태 머신 (State Machine)
        # =========================
        # WAIT_TRIGGER    : /detect_success 신호 대기
        # UNDOCK          : 도킹 스테이션에서 분리
        # MOVE_TO_FIXED_GOAL : 사전에 지정된 고정 목표 지점으로 이동
        # WAIT_FIXED_GOAL : 고정 목표 도달 대기
        # FOLLOW_TARGET   : YOLO로 탐지된 자동차를 추적
        self.mode = 'WAIT_TRIGGER'

        # 고정 목표 좌표 및 방향
        self.goal_xy  = [-2.6, 0.6]
        self.goal_dir = TurtleBot4Directions.EAST

        # =========================
        # YOLO 모델 로드
        # =========================
        model_path = '/home/rokey/rgb_train_results/11n_bat_8/car_box_detection/weights/best.pt'
        self.yolo_model = YOLO(model_path)
        self.get_logger().info(f'YOLO model loaded from: {model_path}')

        # YOLO 탐지 결과 상태
        self.car_detected        = False  # 자동차 탐지 여부
        self.car_bbox            = None   # 바운딩 박스 (x1,y1,x2,y2)
        self.car_center          = None   # 바운딩 박스 중심 픽셀 좌표
        self.detection_goal_sent = False  # 이번 탐지에서 goal을 이미 보냈는지 여부

        # -------------------------------------------------------
        # GUI 스레드 시작 (메인 스핀과 분리)
        # -------------------------------------------------------
        self.gui_thread_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()

        # -------------------------------------------------------
        # TF 버퍼 및 리스너 초기화
        # -------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------------------------------------------------------
        # TurtleBot4 네비게이터 초기화
        # -------------------------------------------------------
        self.navigator = TurtleBot4Navigator()

        # 시작 전 도킹 상태 확인 및 도킹
        if not self.navigator.getDockedStatus():
            self.get_logger().info('Docking before initializing pose')
            self.navigator.dock()

        # 초기 위치(0,0) 설정 후 Nav2 활성화 대기
        initial_pose = self.navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()

        # -------------------------------------------------------
        # 중복 로그 방지 플래그
        # -------------------------------------------------------
        self.logged_intrinsics  = False  # 카메라 내부 파라미터 로그 출력 여부
        self.logged_rgb_shape   = False  # RGB 이미지 shape 로그 출력 여부
        self.logged_depth_shape = False  # 깊이 이미지 shape 로그 출력 여부

        # -------------------------------------------------------
        # 구독자(Subscriber) 등록
        # -------------------------------------------------------
        self.create_subscription(CameraInfo,               self.info_topic,       self.camera_info_callback, 1)
        self.create_subscription(Image,                    self.depth_topic,      self.depth_callback,       1)
        self.create_subscription(CompressedImage,          self.rgb_topic,        self.rgb_callback,         1)
        self.create_subscription(Bool,                     '/detect_success',     self.listener_cb,          10)
        self.create_subscription(PoseWithCovarianceStamped,'/robot1/amcl_pose',   self.amcl_callback,        1)

        # TF 트리가 안정화될 때까지 5초 대기 후 메인 루프 타이머 시작
        self.get_logger().info('TF Tree 안정화 대기. 5초 후 루프 시작합니다.')
        self.start_timer = self.create_timer(5.0, self.start_transform)

    def start_transform(self):
        """5초 대기 후 호출. 0.2초 주기의 메인 처리 타이머를 시작한다."""
        self.get_logger().info('TF Tree 안정화 완료. 루프 시작합니다.')
        self.timer = self.create_timer(0.2, self.display_images)
        self.start_timer.cancel()  # 1회성 타이머 해제

    # =======================================================
    # 콜백 함수들
    # =======================================================

    def camera_info_callback(self, msg):
        """카메라 내부 파라미터(K 행렬)를 수신하여 저장한다."""
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f'Camera intrinsics received: '
                    f'fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, '
                    f'cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}'
                )
                self.logged_intrinsics = True

    def listener_cb(self, msg):
        """
        /detect_success 토픽 수신 콜백.
        True 신호를 받으면 WAIT_TRIGGER → UNDOCK 상태로 전환하여 미션을 시작한다.
        """
        if msg.data and self.mode == 'WAIT_TRIGGER':
            self.get_logger().info('Car detected -> start mission')
            self.mode = 'UNDOCK'

    def depth_callback(self, msg):
        """깊이 이미지(16비트, mm 단위)를 수신하여 저장한다."""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0:
                if not self.logged_depth_shape:
                    self.get_logger().info(f'Depth image received: {depth.shape}')
                    self.logged_depth_shape = True
                with self.lock:
                    self.depth_image  = depth
                    self.camera_frame = msg.header.frame_id  # TF 프레임 ID 저장
        except Exception as e:
            self.get_logger().error(f'Depth CV bridge conversion failed: {e}')

    def rgb_callback(self, msg):
        """압축(JPEG) RGB 이미지를 수신하여 OpenCV 포맷으로 디코딩 후 저장한다."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is not None and rgb.size > 0:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f'RGB image decoded: {rgb.shape}')
                    self.logged_rgb_shape = True
                with self.lock:
                    self.rgb_image = rgb
        except Exception as e:
            self.get_logger().error(f'Compressed RGB decode failed: {e}')

    def amcl_callback(self, msg):
        """AMCL 위치 추정 결과를 수신하여 로봇의 현재 맵 좌표를 갱신한다."""
        self.robot_x           = msg.pose.pose.position.x
        self.robot_y           = msg.pose.pose.position.y
        self.robot_orientation = msg.pose.pose.orientation

    # =======================================================
    # YOLO 탐지
    # =======================================================

    def run_yolo_detection(self, rgb_img):
        """
        RGB 이미지에서 YOLO를 실행하여 'car' 클래스 중 가장 신뢰도 높은 객체를 반환한다.

        Returns:
            detected (bool)      : 자동차 탐지 여부
            best_bbox (tuple)    : 최고 신뢰도 박스 (x1, y1, x2, y2)
            best_center (tuple)  : 박스 중심 픽셀 (cx, cy)
            annotated_img (ndarray): 바운딩 박스가 그려진 이미지
        """
        results = self.yolo_model(rgb_img, verbose=False)

        best_conf   = 0.0
        best_bbox   = None
        best_center = None
        annotated_img = rgb_img.copy()

        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                conf     = float(box.conf[0])

                # 'car' 클래스 중 가장 높은 confidence를 best로 선택
                if cls_name.lower() == 'car' and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_bbox   = (x1, y1, x2, y2)
                    best_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # 모든 'car' 박스를 이미지에 시각화
                if cls_name.lower() == 'car':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f'car {conf:.2f}'
                    cv2.putText(
                        annotated_img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

        detected = best_bbox is not None
        return detected, best_bbox, best_center, annotated_img

    # =======================================================
    # 유틸리티 함수
    # =======================================================

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV 마우스 이벤트 콜백 (현재 미사용, 확장용 플레이스홀더)."""
        pass

    def send_fixed_goal_once(self):
        """사전에 지정된 고정 좌표(goal_xy)로 이동 명령을 한 번 전송한다."""
        goal_pose = self.navigator.getPoseStamped(self.goal_xy, self.goal_dir)
        self.navigator.startToPose(goal_pose)
        self.get_logger().info(f'Fixed goal sent: {self.goal_xy}')

    def send_follow_goal(self, goal_x, goal_y, goal_o):
        """
        맵 좌표 (goal_x, goal_y)와 방향(goal_o)으로 목표 Pose를 전송한다.
        yaw=0으로 고정하거나 로봇 현재 방향(goal_o)을 그대로 사용할 수 있다.
        """
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp    = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = 0.0

        # yaw 각도로 쿼터니언 생성 (현재는 로봇 현재 방향 사용)
        yaw = 0.0
        qz  = math.sin(yaw / 2.0)
        qw  = math.cos(yaw / 2.0)
        goal_pose.pose.orientation = goal_o  # 로봇 현재 방향 유지

        self.navigator.goToPose(goal_pose)
        self.get_logger().info(f'Follow goal sent: ({goal_x:.2f}, {goal_y:.2f})')

    def compute_follow_goal(self, target_map_x, target_map_y, target_depth):
        """
        탐지된 자동차의 맵 좌표를 받아, 로봇이 자동차로부터
        keep_distance(0.5m) 앞에 정지할 수 있는 중간 goal 좌표를 계산한다.

        Returns:
            (goal_x, goal_y) : 목표 좌표, 로봇 위치 미확인 시 None 반환
        """
        if self.robot_x is None or self.robot_y is None:
            return None

        keep_distance = 0.5  # 자동차와 유지할 거리 (단위: m)

        dx   = target_map_x - self.robot_x
        dy   = target_map_y - self.robot_y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            return None  # 로봇과 자동차가 거의 같은 위치면 이동 불필요

        # 로봇 → 자동차 방향으로 keep_distance만큼 앞에 goal 설정
        goal_x = target_map_x - keep_distance * dx / dist
        goal_y = target_map_y - keep_distance * dy / dist
        return goal_x, goal_y

    # =======================================================
    # 메인 처리 루프 (0.2초 주기 타이머 콜백)
    # =======================================================

    def display_images(self):
        """
        상태 머신을 처리하고, YOLO 탐지 결과를 바탕으로
        내비게이션 목표를 전송하며, GUI 표시용 이미지를 갱신한다.
        """
        # 공유 자원을 안전하게 복사
        with self.lock:
            rgb      = self.rgb_image.copy()   if self.rgb_image   is not None else None
            depth    = self.depth_image.copy() if self.depth_image is not None else None
            frame_id = self.camera_frame
            K        = self.K.copy()           if self.K           is not None else None

        # ---------------------------
        # 상태 머신 처리
        # ---------------------------

        # UNDOCK: 도킹 스테이션에서 분리 후 고정 목표 이동 상태로 전환
        if self.mode == 'UNDOCK':
            self.get_logger().info('Undocking...')
            self.navigator.undock()
            self.mode = 'MOVE_TO_FIXED_GOAL'
            return

        # MOVE_TO_FIXED_GOAL: 고정 좌표로 이동 명령 전송 후 대기 상태로 전환
        if self.mode == 'MOVE_TO_FIXED_GOAL':
            self.send_fixed_goal_once()
            self.mode = 'WAIT_FIXED_GOAL'
            return

        # WAIT_FIXED_GOAL: 고정 목표 도달 여부 확인
        if self.mode == 'WAIT_FIXED_GOAL':
            if self.navigator.isTaskComplete():
                self.get_logger().info('Fixed goal reached. Start follow mode.')
                self.mode = 'FOLLOW_TARGET'
            return

        # WAIT_TRIGGER 상태 또는 데이터가 준비되지 않은 경우 조기 반환
        if rgb is None or depth is None or frame_id is None or K is None:
            return

        try:
            # ---------------------------
            # YOLO 탐지 실행
            # ---------------------------
            detected, bbox, center, rgb_display = self.run_yolo_detection(rgb)

            # 탐지 결과를 공유 변수에 저장 (다른 스레드에서 참조 가능)
            with self.lock:
                self.car_detected = detected
                self.car_bbox     = bbox
                self.car_center   = center

            # 깊이 이미지 시각화: 정규화 후 컬러맵 적용
            depth_display    = depth.copy()
            depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored    = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            if detected:
                self.get_logger().info(f'Car detected at bbox={bbox}, center={center}')

            # ---------------------------
            # FOLLOW_TARGET 상태: 탐지된 자동차를 향해 이동 goal 전송
            # ---------------------------
            if self.mode == 'FOLLOW_TARGET' and self.robot_x is not None and self.robot_y is not None:
                # 자동차가 탐지되고, 이번 탐지에서 아직 goal을 보내지 않은 경우
                if detected and not self.detection_goal_sent and self.K is not None:
                    cx_car, cy_car = center

                    # 픽셀 좌표에서 깊이값 읽기 (mm → m 변환)
                    z_car = float(depth[cy_car, cx_car]) / 1000.0

                    # 유효한 깊이 범위(0.2m ~ 5.0m) 내에서만 처리
                    if 0.2 < z_car < 5.0:
                        fx, fy     = K[0, 0], K[1, 1]
                        cx_k, cy_k = K[0, 2], K[1, 2]

                        # 핀홀 카메라 모델로 카메라 좌표계 3D 위치 계산
                        # X = (u - cx) * Z / fx
                        X = (cx_car - cx_k) * z_car / fx
                        Y = (cy_car - cy_k) * z_car / fy
                        Z = z_car

                        # 카메라 좌표계 PointStamped 생성
                        pt_camera = PointStamped()
                        pt_camera.header.stamp    = Time().to_msg()
                        pt_camera.header.frame_id = frame_id
                        pt_camera.point.x = X
                        pt_camera.point.y = Y
                        pt_camera.point.z = Z

                        # TF를 사용하여 카메라 좌표 → 맵 좌표로 변환
                        pt_map = self.tf_buffer.transform(
                            pt_camera, 'map', timeout=Duration(seconds=1.0)
                        )

                        self.get_logger().info(
                            f'[Car] Map coord: ({pt_map.point.x:.2f}, {pt_map.point.y:.2f})'
                        )

                        # 자동차 앞 keep_distance 지점을 goal로 계산
                        follow_goal = self.compute_follow_goal(
                            pt_map.point.x,
                            pt_map.point.y,
                            z_car
                        )

                        # 현재 로봇 방향을 goal orientation으로 사용
                        self.goal_orientation = self.robot_orientation

                        if follow_goal is not None:
                            goal_x, goal_y = follow_goal
                            self.send_follow_goal(goal_x, goal_y, self.goal_orientation)
                            self.detection_goal_sent = True  # 중복 전송 방지 플래그 설정

                # 자동차가 탐지되지 않으면 다음 탐지 시 goal을 다시 보낼 수 있도록 초기화
                if not detected:
                    self.detection_goal_sent = False

            # 현재 상태를 이미지에 오버레이
            mode_text = f'MODE: {self.mode}'
            cv2.putText(
                rgb_display, mode_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )

            # RGB + 깊이 컬러맵을 가로로 합성하여 GUI용 이미지 저장
            combined = np.hstack((rgb_display, depth_colored))
            with self.lock:
                self.display_image = combined.copy()

        except Exception as e:
            self.get_logger().warn(f'TF or goal error: {e}')

    # =======================================================
    # GUI 루프 (별도 스레드에서 실행)
    # =======================================================

    def gui_loop(self):
        """
        OpenCV 윈도우에 RGB + 깊이 합성 이미지를 실시간으로 표시한다.
        'q' 키 입력 시 로봇을 도킹하고 ROS 종료를 요청한다.
        """
        cv2.namedWindow('RGB (left) | Depth (right)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB (left) | Depth (right)', 1280, 480)
        cv2.moveWindow('RGB (left) | Depth (right)', 100, 100)

        while not self.gui_thread_stop.is_set():
            with self.lock:
                img = self.display_image.copy() if self.display_image is not None else None

            if img is not None:
                cv2.imshow('RGB (left) | Depth (right)', img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    # 'q' 키: 도킹 후 종료
                    self.get_logger().info('Shutdown requested by user (via GUI).')
                    self.navigator.dock()
                    self.shutdown_requested = True
                    self.gui_thread_stop.set()
                    rclpy.shutdown()
            else:
                cv2.waitKey(10)  # 이미지가 없으면 잠시 대기


def main():
    rclpy.init()
    node = DepthToMap()

    # 멀티스레드 실행자 사용 (콜백 병렬 처리)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # 종료 시 정리
    node.gui_thread_stop.set()   # GUI 스레드 종료 신호
    node.gui_thread.join()       # GUI 스레드 종료 대기
    node.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()