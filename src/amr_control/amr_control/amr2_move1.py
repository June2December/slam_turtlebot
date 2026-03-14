"""
사용법:
  ros2 run amr_control amr2_move1 \
    --ros-args --params-file \
    ~/turtle4_ws/slam_turtlebot/src/amr_control/config/amr2_move1_params.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from amr_interfaces.msg import TargetEvent               # 외부 웹캠 감지 트리거 메시지
from geometry_msgs.msg import PointStamped, PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped  # AMCL 로봇 위치 메시지
from irobot_create_msgs.msg import AudioNoteVector, AudioNote  # 로봇 스피커 제어
from builtin_interfaces.msg import Duration as BuiltinDuration
from std_msgs.msg import Bool, Int32                     # 센서 감지 신호 / 순찰 번호

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from ultralytics import YOLO
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import numpy as np
import cv2
import threading
import math
from rclpy.time import Time


class PatrolInspectNode(Node):
    def __init__(self):
        super().__init__('patrol_inspect_node')

        # ── ROS2 파라미터 선언 ──────────────────────────────
        self.declare_parameter('model_path',
            '/home/rokey/slam_turtlebot/src/models/arm2/best.pt')  # YOLO 모델 경로
        self.declare_parameter('inspect_frame_required', 30)        # 추론할 총 프레임 수
        self.declare_parameter('inspect_threshold',      0.5)       # 이상개체 판정 비율 (50%)
        self.declare_parameter('confidence_min',         0.6)       # YOLO 최소 신뢰도
        self.declare_parameter('target_class',      'unknown')      # 이상개체 클래스명
        self.declare_parameter('approach_distance',      0.5)       # 1차 approach 정지 거리 (m)
        self.declare_parameter('approach_distance_final',0.3)       # 2차 approach 정지 거리 (m)
        self.declare_parameter('approach_timeout',      10.0)       # approach 타임아웃 (초)
        self.declare_parameter('depth_valid_min',        0.2)       # 유효 depth 최솟값 (m)
        self.declare_parameter('depth_valid_max',        5.0)       # 유효 depth 최댓값 (m)
        self.declare_parameter('sync_tolerance_sec',     0.05)      # RGB-Depth 싱크 허용 오차 (50ms)
        self.declare_parameter('sync_timeout_sec',       2.0)       # 싱크 대기 타임아웃 (초)

        self.bridge = CvBridge()        # ROS Image ↔ OpenCV 변환기
        self.lock   = threading.Lock() # 멀티스레드 공유 자원 보호용 락

        # ── 토픽 이름 (네임스페이스 포함) ──
        ns          = self.get_namespace()
        depth_topic = f'{ns}/oakd/stereo/image_raw'
        rgb_topic   = f'{ns}/oakd/rgb/image_raw/compressed'
        info_topic  = f'{ns}/oakd/rgb/camera_info'

        # ── 이미지 버퍼 ──
        self.rgb_image    = None
        self.depth_image  = None
        self.K            = None
        self.camera_frame = None

        # ── 타임스탬프 버퍼 (RGB-Depth 싱크 판단용) ──
        self.rgb_stamp   = None
        self.depth_stamp = None

        # ── 로그 중복 방지 플래그 ──
        self.logged_rgb   = False
        self.logged_depth = False
        self.logged_K     = False

        # ── 외부 웹캠 트리거 수신 플래그 ──
        self.object_detected = False

        # ── YOLO 추론 상태 관리 ──
        self.is_inspecting       = False  # 현재 추론 진행 중 여부
        self.inspect_frame_count = 0      # 현재까지 추론한 프레임 수
        self.anomaly_hits        = 0      # 이상개체 탐지된 프레임 수
        self.inspect_done        = False  # 추론 완료 여부
        self.anomaly_result      = False  # 최종 이상개체 판정 결과

        # ── approach 목표 픽셀 좌표 ──
        self.target_cx   = None
        self.target_cy   = None
        self.target_conf = 0.0

        # ── 별도 센서 감지 플래그 ──
        self.sensor_detected = False

        # ── AMCL 로봇 위치 (map 기준) ──
        self.robot_x           = 0.0
        self.robot_y           = 0.0
        self.robot_orientation = None

        # ── 재시도 횟수 ──
        self.retry = 0

        # ── YOLO 모델 로드 ──
        model_path = self.p('model_path')
        self.get_logger().info(f'YOLO 모델 로드 중: {model_path}')
        self.yolo = YOLO(model_path)
        self.get_logger().info('YOLO 모델 로드 완료')

        # ── TF 버퍼/리스너 ──
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── 경보음 publisher ──
        self.audio_pub = self.create_publisher(
            AudioNoteVector,
            f'{ns}/cmd_audio',
            10
        )

        # ── 감지 위치 publisher ──
        self.object_pose_pub = self.create_publisher(
            PointStamped,
            '/amr2/object_pose',
            10
        )

        # ── 현재 순찰 번호 publisher ──
        self.current_goal_pub = self.create_publisher(
            Int32,
            '/amr2/current_goal',
            10
        )

        # ── 토픽 구독 ──
        self.create_subscription(TargetEvent,              '/robot1/target_event', self.trigger_cb, 1)
        self.create_subscription(CameraInfo,               info_topic,             self.info_cb,    1)
        self.create_subscription(Image,                    depth_topic,            self.depth_cb,   1)
        self.create_subscription(CompressedImage,          rgb_topic,              self.rgb_cb,     1)
        self.create_subscription(Bool,                     '/amr2/unknown',        self.sensor_cb,  1)
        self.create_subscription(PoseWithCovarianceStamped,'/robot1/amcl_pose',    self.amcl_cb,    1)

        self.get_logger().info('PatrolInspectNode 초기화 완료')

        # ── GUI 스레드 시작 ──
        self.gui_thread_stop = threading.Event()
        self.gui_thread      = threading.Thread(target=self._gui_loop, daemon=True)
        self.gui_thread.start()

    # ── 파라미터 헬퍼 ──────────────────────────────────────
    def p(self, name):
        return self.get_parameter(name).value

    # ── 콜백 ──────────────────────────────────────────────

    def trigger_cb(self, msg):
        if msg.detected and not self.object_detected:
            self.get_logger().info('트리거 수신: 객체 감지됨')
        self.object_detected = msg.detected

    def info_cb(self, msg):
        """
        카메라 내부 파라미터 콜백
        K 행렬(초점거리, 주점)을 저장 → depth → 3D 좌표 계산에 사용
        """
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
        if not self.logged_K:
            self.get_logger().info('CameraInfo 수신 완료')
            self.logged_K = True

    def sensor_cb(self, msg):
        """
        /amr2/unknown 콜백 (외부 센서)
        True 수신 시 sensor_detected 플래그 설정
        → 2차 approach 후 wait_sensor_and_alarm()에서 확인
        """
        if msg.data:
            self.get_logger().info('센서 수신: 센서값 초과 감지')
            self.sensor_detected = True

    def amcl_cb(self, msg):
        """
        /robot1/amcl_pose 콜백
        AMCL이 추정한 로봇의 map 기준 위치/방향을 저장
        → 이상개체 감지 위치 발행 시 사용
        """
        self.robot_x           = msg.pose.pose.position.x
        self.robot_y           = msg.pose.pose.position.y
        self.robot_orientation = msg.pose.pose.orientation

    def depth_cb(self, msg):
        """
        Depth 이미지 콜백
        - depth_image: approach 시 3D 좌표 계산에 사용
        - camera_frame: TF 변환의 source frame
        - depth_stamp: RGB-Depth 싱크 판단용 타임스탬프
        """
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0:
                with self.lock:
                    self.depth_image  = depth
                    self.camera_frame = msg.header.frame_id
                    self.depth_stamp  = msg.header.stamp.sec + \
                                        msg.header.stamp.nanosec * 1e-9
                if not self.logged_depth:
                    self.get_logger().info(f'Depth 이미지 수신: {depth.shape}')
                    self.logged_depth = True
        except Exception as e:
            self.get_logger().error(f'Depth 변환 오류: {e}')

    def rgb_cb(self, msg):
        """
        RGB 압축 이미지 콜백
        - rgb_image: YOLO 추론 및 GUI 표시에 사용
        - rgb_stamp: RGB-Depth 싱크 판단용 타임스탬프
        """
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is not None and rgb.size > 0:
                with self.lock:
                    self.rgb_image = rgb
                    self.rgb_stamp = msg.header.stamp.sec + \
                                     msg.header.stamp.nanosec * 1e-9
                if not self.logged_rgb:
                    self.get_logger().info(f'RGB 이미지 수신: {rgb.shape}')
                    self.logged_rgb = True
        except Exception as e:
            self.get_logger().error(f'RGB 디코딩 오류: {e}')

    def _gui_loop(self):
        """
        별도 스레드에서 실행되는 카메라 창 표시 루프
        - 최신 RGB 프레임에 YOLO bbox를 실시간으로 그려서 표시
        - 이상개체(unknown): 빨간색 / 나머지: 초록색
        - 화면 좌상단에 현재 추론 상태 오버레이
        """
        while not self.gui_thread_stop.is_set():
            with self.lock:
                frame = self.rgb_image.copy() if self.rgb_image is not None else None

            if frame is not None:
                # YOLO 추론 후 bbox 그리기
                results = self.yolo(frame, conf=self.p('confidence_min'), verbose=False)

                for r in results:
                    for box in r.boxes:
                        cls_name        = self.yolo.names[int(box.cls)]
                        conf            = float(box.conf[0])
                        x1, y1, x2, y2  = map(int, box.xyxy[0])
                        cx              = (x1 + x2) // 2
                        cy              = (y1 + y2) // 2

                        # 이상개체(unknown) → 빨간색 / 나머지 → 초록색
                        color = (0, 0, 255) if cls_name == self.p('target_class') \
                                            else (0, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame, f'{cls_name} {conf:.2f}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )
                        cv2.circle(frame, (cx, cy), 5, color, -1)

                # 현재 추론 상태 오버레이 (Inspecting 중이면 진행률, 아니면 Waiting)
                status = f'Inspecting: {self.inspect_frame_count}/{self.p("inspect_frame_required")}' \
                         if self.is_inspecting else 'Waiting...'
                cv2.putText(
                    frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )

                cv2.imshow('Patrol Camera', frame)

            cv2.waitKey(30)

        cv2.destroyAllWindows()

    # ── 경보음 ────────────────────────────────────────────

    def play_alarm(self):
        """
        로봇 스피커로 경보음 발행
        880Hz → 440Hz → 880Hz → 440Hz, 각 0.3초씩 재생
        """
        msg = AudioNoteVector()
        msg.append = False  # 기존 재생 중인 음 덮어쓰기

        for freq in [880, 440, 880, 440]:
            note = AudioNote()
            note.frequency = int(freq)
            dur = BuiltinDuration()
            dur.sec = 0
            dur.nanosec = 300_000_000  # 0.3초
            note.max_runtime = dur
            msg.notes.append(note)

        self.audio_pub.publish(msg)
        self.get_logger().info('경보음 발행!')

    # ── 센서 신호 대기 + 경보음 + pose 발행 ───────────────

    def wait_sensor_and_alarm(self, executor, timeout_sec=10.0):
        """
        2차 approach 완료 후 호출.
        최대 timeout_sec 동안 /amr2/unknown True 신호 대기.
        - True 수신 시: 경보음 + /amr2/object_pose 발행
        - 타임아웃 시: 조용히 다음 진행
        """
        self.sensor_detected = False  # 이전 감지 플래그 초기화
        start = self.get_clock().now()
        self.get_logger().info(f'센서 신호 대기 중... (최대 {timeout_sec}초)')

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            if self.sensor_detected:
                self.get_logger().warn('센서 True 수신! 경보음 발행 및 위치 전송.')
                self.play_alarm()
                self._publish_object_pose()
                return True

            if elapsed > timeout_sec:
                self.get_logger().info('센서 신호 타임아웃. 경보음 없이 다음 진행.')
                return False

        return False

    def _publish_object_pose(self):
        """
        현재 AMCL 로봇 위치를 /amr2/object_pose 토픽으로 발행
        이상개체가 감지된 위치(map 기준 x, y, z=0)를 외부에 알림
        """
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.point.x = self.robot_x
        msg.point.y = self.robot_y
        msg.point.z = 0.0
        self.object_pose_pub.publish(msg)
        self.get_logger().info(
            f'감지 위치 발행: x={self.robot_x:.2f}, y={self.robot_y:.2f}'
        )

    # ── 현재 순찰 번호 발행 ────────────────────────────────

    def publish_current_goal(self, goal_index):
        """골 도착 시 현재 순찰 번호(1~7) 발행"""
        msg = Int32()
        msg.data = goal_index + 1  # 0-indexed → 1-indexed
        self.current_goal_pub.publish(msg)
        self.get_logger().info(f'현재 순찰 번호 발행: {msg.data}')

    # ── 추론 제어 ──────────────────────────────────────────

    def start_inspect(self):
        """
        YOLO 추론 시작 초기화
        랠리포인트 도착 후 1.5초 대기 뒤 호출됨
        모든 카운터/플래그/박스 중심 좌표를 초기화하고 추론 시작
        """
        self.is_inspecting       = True
        self.inspect_frame_count = 0
        self.anomaly_hits        = 0
        self.inspect_done        = False
        self.anomaly_result      = False
        self.target_cx           = None
        self.target_cy           = None
        self.target_conf         = 0.0
        self.get_logger().info('추론 시작...')

    def run_inspect_once(self):
        """
        메인 루프에서 매 iteration마다 호출되는 추론 함수
        - 현재 RGB 프레임으로 YOLO 추론 1회 수행
        - target_class(unknown) 탐지 시 anomaly_hits 증가 및 박스 중심 갱신
        - inspect_frame_required 프레임 완료 시:
          - anomaly_hits / total >= inspect_threshold → 이상개체 확정
          - 아니면 이상없음 처리
        """
        if not self.is_inspecting:
            return

        with self.lock:
            frame = self.rgb_image.copy() if self.rgb_image is not None else None

        if frame is None:
            return

        results  = self.yolo(frame, conf=self.p('confidence_min'), verbose=False)
        detected = False

        for r in results:
            for box in r.boxes:
                cls_name = self.yolo.names[int(box.cls)]
                conf     = float(box.conf[0])
                if cls_name == self.p('target_class'):
                    detected = True
                    # 가장 높은 신뢰도의 박스 중심 좌표 갱신 (approach용)
                    if conf > self.target_conf:
                        self.target_conf = conf
                        x1, y1, x2, y2  = box.xyxy[0]
                        self.target_cx   = int((x1 + x2) / 2)
                        self.target_cy   = int((y1 + y2) / 2)
                    break

        self.inspect_frame_count += 1
        if detected:
            self.anomaly_hits += 1

        self.get_logger().info(
            f'  프레임 {self.inspect_frame_count}/{self.p("inspect_frame_required")} '
            f'-> {"탐지 O" if detected else "탐지 X"} '
            f'(누적 {self.anomaly_hits}회)'
        )

        # 목표 프레임 수 도달 시 최종 판정
        if self.inspect_frame_count >= self.p('inspect_frame_required'):
            ratio               = self.anomaly_hits / self.p('inspect_frame_required')
            self.anomaly_result = ratio >= self.p('inspect_threshold')
            self.is_inspecting  = False
            self.inspect_done   = True

            if self.anomaly_result:
                self.get_logger().warn(
                    f'이상개체 감지! ({self.anomaly_hits}/{self.p("inspect_frame_required")} '
                    f'프레임, {ratio*100:.0f}%) '
                    f'박스 중심: ({self.target_cx}, {self.target_cy})'
                )
            else:
                self.get_logger().info(
                    f'이상없음 ({self.anomaly_hits}/{self.p("inspect_frame_required")} '
                    f'프레임, {ratio*100:.0f}%)'
                )

    # ── 싱크 맞춘 프레임 획득 ─────────────────────────────

    def get_synced_frame(self, executor):
        """RGB-Depth 타임스탬프 차이가 sync_tolerance_sec 이내인 프레임 쌍 반환"""
        timeout   = self.p('sync_timeout_sec')
        tolerance = self.p('sync_tolerance_sec')
        start     = self.get_clock().now()

        self.get_logger().info(f'싱크 대기 중... (허용오차: {tolerance*1000:.0f}ms)')

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            with self.lock:
                rgb_s   = self.rgb_stamp
                depth_s = self.depth_stamp
                rgb     = self.rgb_image
                depth   = self.depth_image

            # 두 타임스탬프 차이 확인
            if rgb_s is not None and depth_s is not None:
                diff = abs(rgb_s - depth_s)
                if diff < tolerance:
                    self.get_logger().info(
                        f'싱크 완료! (타임스탬프 차이: {diff*1000:.1f}ms)'
                    )
                    return rgb.copy(), depth.copy()

            if elapsed > timeout:
                self.get_logger().warn(f'싱크 타임아웃 ({timeout}초 초과)')
                return None, None

        return None, None

    # ── 재검증 (싱크 맞춘 프레임으로 YOLO 1회 추론) ────────

    def verify_target(self, executor):
        """
        1차 approach 완료 후 오탐지 여부를 재확인하는 함수
        - 싱크된 RGB + Depth 프레임으로 YOLO 1회 추론
        - 이상개체 재발견 시: (True, cx, cy, depth) 반환
        - 미발견(오탐지) 시: (False, None, None, None) 반환
        """
        self.get_logger().info('=== 재검증 시작 ===')

        # 싱크 맞춘 프레임 획득
        rgb, depth = self.get_synced_frame(executor)
        if rgb is None or depth is None:
            self.get_logger().warn('재검증 실패: 싱크된 프레임 없음')
            return False, None, None, None

        # YOLO 추론 1회 - 가장 높은 신뢰도의 이상개체 박스 선택
        results  = self.yolo(rgb, conf=self.p('confidence_min'), verbose=False)
        best_cx, best_cy, best_conf = None, None, 0.0

        for r in results:
            for box in r.boxes:
                cls_name = self.yolo.names[int(box.cls)]
                conf     = float(box.conf[0])
                if cls_name == self.p('target_class') and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0]
                    best_cx = int((x1 + x2) / 2)
                    best_cy = int((y1 + y2) / 2)

        if best_cx is None:
            self.get_logger().info('재검증 결과: 이상개체 없음 (오탐지)')
            return False, None, None, None

        self.get_logger().warn(
            f'재검증 결과: 이상개체 확인! '
            f'(conf: {best_conf:.2f}, 박스 중심: ({best_cx}, {best_cy}))'
        )
        return True, best_cx, best_cy, depth

    # ── Approach 공통 로직 ────────────────────────────────

    def _do_approach(self, navigator, executor, cx, cy, depth, stop_distance):
        """
        픽셀 좌표(cx, cy)와 depth로 3D 좌표를 계산한 뒤
        stop_distance(m) 앞 지점까지 Nav2로 이동하는 공통 함수

        동작 순서:
        1. depth[cy, cx]로 거리(z_m) 획득 (mm → m 변환)
        2. K 행렬로 카메라 좌표계 3D 점 계산
        3. TF 변환으로 map 좌표계로 변환
        4. map 좌표에서 stop_distance 앞 목표점 계산
        5. Nav2로 이동 후 완료 대기
        """
        with self.lock:
            K     = self.K.copy() if self.K is not None else None
            frame = self.camera_frame

        if K is None or frame is None:
            self.get_logger().warn('Approach 실패: K / frame 없음')
            return False

        # ── 1. depth 읽기 (mm → m 변환) ──
        z_m   = float(depth[cy, cx]) / 1000.0
        d_min = self.p('depth_valid_min')
        d_max = self.p('depth_valid_max')

        if not (d_min < z_m < d_max):
            self.get_logger().warn(f'Approach 실패: depth 범위 초과 ({z_m:.2f}m)')
            return False

        self.get_logger().info(f'거리: {z_m:.2f}m → 목표 정지거리: {stop_distance}m')

        # ── 2. K 행렬로 카메라 좌표계 3D 점 계산 ──
        fx, fy     = K[0, 0], K[1, 1]  # 초점 거리
        cx_k, cy_k = K[0, 2], K[1, 2]  # 주점(principal point)
        X = (cx - cx_k) * z_m / fx     # 카메라 X (좌우)
        Y = (cy - cy_k) * z_m / fy     # 카메라 Y (상하)
        Z = z_m                         # 카메라 Z (깊이)

        # ── 3. TF 변환 (카메라 좌표계 → map 좌표계) ──
        pt_cam = PointStamped()
        pt_cam.header.frame_id = frame
        pt_cam.header.stamp    = Time().to_msg()
        pt_cam.point.x = X
        pt_cam.point.y = Y
        pt_cam.point.z = Z

        try:
            pt_map = self.tf_buffer.transform(
                pt_cam, 'map', timeout=Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f'TF 변환 실패: {e}')
            return False

        mx, my = pt_map.point.x, pt_map.point.y
        self.get_logger().info(f'지도 좌표: x={mx:.2f} y={my:.2f}')

        # ── 4. 목표점에서 stop_distance 앞 지점 계산 ──
        # 원점(0,0)에서 목표 방향 각도를 구해 stop_distance만큼 앞에 정지
        angle  = math.atan2(my, mx)
        goal_x = mx - stop_distance * math.cos(angle)
        goal_y = my - stop_distance * math.sin(angle)


        # ── 5. Nav2로 이동 ──
        approach_goal = navigator.getPoseStamped(
            [goal_x, goal_y],
            TurtleBot4Directions.NORTH
        )
        navigator.startToPose(approach_goal)

        # 완료 대기 (approach_timeout 초 이내)
        timeout    = self.p('approach_timeout')
        start_time = self.get_clock().now()

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9

            if navigator.isTaskComplete():
                self.get_logger().info(f'Approach 완료! ({elapsed:.1f}초)')
                return True

            if elapsed > timeout:
                self.get_logger().warn('Approach 타임아웃')
                navigator.cancelTask()
                return False

        return False

    def approach_target(self, navigator, executor):
        """1차 approach: approach_distance(0.5m)까지 접근 (싱크 없음)"""
        self.get_logger().info('=== 1차 Approach 시작 (싱크 없음) ===')

        with self.lock:
            cx    = self.target_cx
            cy    = self.target_cy
            depth = self.depth_image.copy() if self.depth_image is not None else None

        if cx is None or cy is None or depth is None:
            self.get_logger().warn('1차 Approach 실패: 데이터 없음')
            return False

        return self._do_approach(
            navigator, executor, cx, cy, depth,
            stop_distance=self.p('approach_distance')
        )

    def approach_target_final(self, navigator, executor, cx, cy, depth):
        """2차 approach: approach_distance_final(0.3m)까지 접근 (싱크 있음)"""
        self.get_logger().info('=== 2차 Approach 시작 (싱크 있음) ===')

        return self._do_approach(
            navigator, executor, cx, cy, depth,
            stop_distance=self.p('approach_distance_final')
        )


# ── main ──────────────────────────────────────────────────

def main():
    rclpy.init()

    node     = PatrolInspectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    navigator = TurtleBot4Navigator()

    # 도킹 상태 확인 후 도킹
    if not navigator.getDockedStatus():
        navigator.dock()

    initial_pose = navigator.getPoseStamped([0.776, 0.819], math.degrees(-1.8))
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()

    # 순찰할 랠리포인트 목록 (map 기준 좌표)
    goal_pose = [
        navigator.getPoseStamped([1.39,  2.35], TurtleBot4Directions.EAST ),
        navigator.getPoseStamped([1.82,  4.22], TurtleBot4Directions.WEST ),
        navigator.getPoseStamped([1.05,  4.6 ], TurtleBot4Directions.WEST ),
        navigator.getPoseStamped([0.23,  2.8 ], TurtleBot4Directions.EAST ),
        navigator.getPoseStamped([-2.06, 1.84], TurtleBot4Directions.SOUTH),
        navigator.getPoseStamped([-0.42, 4.4 ], TurtleBot4Directions.NORTH),
        navigator.getPoseStamped([-1.36, 4.8 ], TurtleBot4Directions.WEST ),
    ]
    home_pose = navigator.getPoseStamped([0.63, 0.87], TurtleBot4Directions.EAST)

    try:
        # ── 1. 외부 웹캠 트리거 대기 ──
        # /target_event 토픽에서 detected=True 수신 시 순찰 시작
        navigator.get_logger().info('웹캠 감지 신호 대기 중...')
        while rclpy.ok() and not node.object_detected:
            executor.spin_once(timeout_sec=0.1)

        if not rclpy.ok():
            return

        # ── 2. 언독 후 순찰 시작 ──
        navigator.get_logger().info('객체 감지됨. 언독 후 순찰 시작.')
        navigator.undock()

        # ── 3. 순찰 + 추론 메인루프 ──
        goal_index = 0
        node.retry = 0
        navigator.startToPose(goal_pose[goal_index])
        navigator.get_logger().info(f'[{goal_index+1}/{len(goal_pose)}] 이동 중...')

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            node.run_inspect_once()

            if navigator.isTaskComplete():
                result = navigator.getResult()

                # ── 골 실패 시 최대 3번 재시도 ──
                if result == TaskResult.FAILED and node.retry < 3:
                    node.retry += 1
                    navigator.get_logger().warn(
                        f'[{goal_index+1}/{len(goal_pose)}] 이동 실패! '
                        f'재시도 {node.retry}/3'
                    )
                    navigator.startToPose(goal_pose[goal_index])
                    continue  # 메인 while로 복귀하여 완료 대기

                # 3번 모두 실패한 경우
                if result == TaskResult.FAILED:
                    navigator.get_logger().warn(
                        f'[{goal_index+1}/{len(goal_pose)}] 3회 재시도 모두 실패. '
                        f'다음 포인트로.'
                    )

                # 성공 or 3회 실패 → 다음 골을 위해 retry 초기화
                node.retry = 0

                # ── 골 도착 시 현재 순찰 번호 발행 ──
                node.publish_current_goal(goal_index)

                # 아직 추론 시작 전이면 1.5초 대기 후 추론 시작
                if not node.is_inspecting and not node.inspect_done:
                    navigator.get_logger().info(
                        f'[{goal_index+1}/{len(goal_pose)}] 도착. 1.5초 대기 후 추론 시작.'
                    )
                    wait_start = node.get_clock().now()
                    while rclpy.ok():
                        executor.spin_once(timeout_sec=0.1)
                        if (node.get_clock().now() - wait_start).nanoseconds / 1e9 >= 1.5:
                            break
                    node.start_inspect()

                # 추론 완료 시 결과 처리
                if node.inspect_done:

                    if node.anomaly_result:
                        navigator.get_logger().info('이상개체 감지!')

                        # ── 4. 1차 approach (싱크 없이 0.5m) ──
                        success = node.approach_target(navigator, executor)

                        if success:
                            # ── 5. 재검증 (싱크된 프레임으로 YOLO 1회) ──
                            verified, cx, cy, depth = node.verify_target(executor)

                            if verified:
                                # ── 6. 2차 approach (싱크된 프레임으로 0.3m) ──
                                node.approach_target_final(
                                    navigator, executor, cx, cy, depth
                                )
                            else:
                                # 재검증에서 미탐지 → 오탐지로 판단
                                navigator.get_logger().info('오탐지 확인. 다음 포인트로.')

                        # ── 7. 센서 신호 최대 10초 대기 ──
                        # True 수신 시: 경보음 + 감지 위치 발행
                        # 타임아웃 시: 조용히 다음 포인트로
                        node.wait_sensor_and_alarm(executor, timeout_sec=10.0)

                    # ── 8. 다음 랠리포인트로 이동 ──
                    goal_index       += 1
                    node.inspect_done = False

                    if goal_index >= len(goal_pose):
                        navigator.get_logger().info('모든 랠리포인트 순찰 완료.')
                        break

                    navigator.get_logger().info(f'[{goal_index+1}/{len(goal_pose)}] 이동 중...')
                    navigator.startToPose(goal_pose[goal_index])

    except KeyboardInterrupt:
        node.get_logger().info('종료 요청.')

    # ── 9. 홈 복귀 및 도킹 ──
    navigator.get_logger().info('순찰 완료. 홈 위치로 이동.')
    navigator.startToPose(home_pose)

    navigator.dock()
    navigator.get_logger().info('미션 완료. 도킹 복귀.')
    node.gui_thread_stop.set()
    node.gui_thread.join()
    rclpy.shutdown()


if __name__ == '__main__':
    main()