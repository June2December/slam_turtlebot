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
from amr_interfaces.msg import TargetEvent
from geometry_msgs.msg import PointStamped
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration as BuiltinDuration

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from ultralytics import YOLO

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
            '/home/rokey/slam_turtlebot/src/models/arm2/best.pt')
        self.declare_parameter('inspect_frame_required', 30)
        self.declare_parameter('inspect_threshold',      0.5)
        self.declare_parameter('confidence_min',         0.5)
        self.declare_parameter('target_class',      'unknown')
        self.declare_parameter('approach_distance',      0.5)   # 1차 approach 거리
        self.declare_parameter('approach_distance_final',0.3)   # 2차 approach 거리
        self.declare_parameter('approach_timeout',      10.0)
        self.declare_parameter('depth_valid_min',        0.2)
        self.declare_parameter('depth_valid_max',        5.0)
        self.declare_parameter('sync_tolerance_sec',     0.05)  # 싱크 허용 오차 (50ms)
        self.declare_parameter('sync_timeout_sec',       2.0)   # 싱크 타임아웃 (초)

        self.bridge = CvBridge()
        self.lock   = threading.Lock()

        # ── 토픽 이름 ──
        ns          = self.get_namespace()
        depth_topic = f'{ns}/oakd/stereo/image_raw'
        rgb_topic   = f'{ns}/oakd/rgb/image_raw/compressed'
        info_topic  = f'{ns}/oakd/rgb/camera_info'

        # ── 이미지 버퍼 ──
        self.rgb_image    = None
        self.depth_image  = None
        self.K            = None
        self.camera_frame = None

        # ── 타임스탬프 버퍼 (싱크용) ──
        self.rgb_stamp   = None   # RGB 수신 시각 (초)
        self.depth_stamp = None   # Depth 수신 시각 (초)

        # ── 로그 중복 방지 ──
        self.logged_rgb   = False
        self.logged_depth = False
        self.logged_K     = False

        # ── 트리거 ──
        self.object_detected = False

        # ── 추론 상태 ──
        self.is_inspecting       = False
        self.inspect_frame_count = 0
        self.anomaly_hits        = 0
        self.inspect_done        = False
        self.anomaly_result      = False

        # ── approach용 박스 중심 픽셀 ──
        self.target_cx   = None
        self.target_cy   = None
        self.target_conf = 0.0

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

        # ── 토픽 구독 ──
        self.create_subscription(TargetEvent,     '/detected', self.trigger_cb, 1)
        self.create_subscription(CameraInfo,      info_topic,  self.info_cb,    1)
        self.create_subscription(Image,           depth_topic, self.depth_cb,   1)
        self.create_subscription(CompressedImage, rgb_topic,   self.rgb_cb,     1)

        self.get_logger().info('PatrolInspectNode 초기화 완료')

        # ── GUI 스레드 시작 ──────────────────────
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
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
        if not self.logged_K:
            self.get_logger().info('CameraInfo 수신 완료')
            self.logged_K = True

# 수정
    def depth_cb(self, msg):
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
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is not None and rgb.size > 0:
                with self.lock:
                    self.rgb_image = rgb
                    # 타임스탬프 저장 (싱크용)
                    self.rgb_stamp = msg.header.stamp.sec + \
                                     msg.header.stamp.nanosec * 1e-9
                if not self.logged_rgb:
                    self.get_logger().info(f'RGB 이미지 수신: {rgb.shape}')
                    self.logged_rgb = True

                 
        except Exception as e:
            self.get_logger().error(f'RGB 디코딩 오류: {e}')

    def _gui_loop(self):
        """별도 스레드에서 카메라 창 + YOLO bbox 실시간 표시"""
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

                # 현재 추론 상태 오버레이
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
        """경보음 발행 (880Hz↔440Hz 반복)"""
        msg = AudioNoteVector()
        msg.append = False

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

    # ── 추론 제어 ──────────────────────────────────────────

    def start_inspect(self):
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
        """
        RGB와 Depth 타임스탬프 차이가 sync_tolerance_sec 이내인
        싱크된 프레임 쌍을 반환. 타임아웃 시 None 반환.
        """
        timeout   = self.p('sync_timeout_sec')
        tolerance = self.p('sync_tolerance_sec')
        start     = self.get_clock().now()

        self.get_logger().info(f'싱크 대기 중... (허용오차: {tolerance*1000:.0f}ms)')

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.05)  # 콜백 처리 (이미지 수신)
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
        1차 approach 완료 후 호출
        싱크 맞춘 RGB + Depth로 YOLO 1회 추론하여 이상개체 재확인
        Returns: (verified, cx, cy, depth_image) or (False, None, None, None)
        """
        self.get_logger().info('=== 재검증 시작 ===')

        # 싱크 맞춘 프레임 획득
        rgb, depth = self.get_synced_frame(executor)
        if rgb is None or depth is None:
            self.get_logger().warn('재검증 실패: 싱크된 프레임 없음')
            return False, None, None, None

        # YOLO 추론 1회
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

    # ── Approach (공통 로직) ──────────────────────────────

    def _do_approach(self, navigator, executor, cx, cy, depth, stop_distance):
        """
        주어진 픽셀(cx, cy)과 depth로 3D 좌표 계산 후
        stop_distance 앞까지 Nav2로 이동
        """
        with self.lock:
            K     = self.K.copy()           if self.K           is not None else None
            frame = self.camera_frame

        if K is None or frame is None:
            self.get_logger().warn('Approach 실패: K / frame 없음')
            return False

        # depth 읽기 (mm → m)
        z_m   = float(depth[cy, cx]) / 1000.0
        d_min = self.p('depth_valid_min')
        d_max = self.p('depth_valid_max')

        if not (d_min < z_m < d_max):
            self.get_logger().warn(
                f'Approach 실패: depth 범위 초과 ({z_m:.2f}m)'
            )
            return False

        self.get_logger().info(f'거리: {z_m:.2f}m → 목표 정지거리: {stop_distance}m')

        # K 행렬로 3D 좌표 계산
        fx, fy     = K[0, 0], K[1, 1]
        cx_k, cy_k = K[0, 2], K[1, 2]
        X = (cx - cx_k) * z_m / fx
        Y = (cy - cy_k) * z_m / fy
        Z = z_m

        # TF 변환 (카메라 → 지도)
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

        # 정지 목표 좌표 계산
        angle  = math.atan2(my, mx)
        goal_x = mx - stop_distance * math.cos(angle)
        goal_y = my - stop_distance * math.sin(angle)

        # Nav2 이동
        approach_goal = navigator.getPoseStamped(
            [goal_x, goal_y],
            TurtleBot4Directions.NORTH
        )
        navigator.startToPose(approach_goal)

        # 완료 대기
        timeout    = self.p('approach_timeout')
        start_time = self.get_clock().now()

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9

            if navigator.isTaskComplete():
                self.get_logger().info(f'Approach 완료! ({elapsed:.1f}초)')
                return True

            if elapsed > timeout:
                self.get_logger().warn(f'Approach 타임아웃')
                navigator.cancelTask()
                return False

        return False

    def approach_target(self, navigator, executor):
        """1차 approach: 싱크 없이 approach_distance(0.5m)까지 접근"""
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
            stop_distance=self.p('approach_distance')  # 0.5m
        )

    def approach_target_final(self, navigator, executor, cx, cy, depth):
        """2차 approach: 싱크된 프레임으로 approach_distance_final(0.3m)까지 접근"""
        self.get_logger().info('=== 2차 Approach 시작 (싱크 있음) ===')

        return self._do_approach(
            navigator, executor, cx, cy, depth,
            stop_distance=self.p('approach_distance_final')  # 0.3m
        )


# ── main ──────────────────────────────────────────────────

def main():
    rclpy.init()

    node     = PatrolInspectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    navigator = TurtleBot4Navigator()

    if not navigator.getDockedStatus():
        navigator.dock()

    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.SOUTH_WEST)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()

    goal_pose = [
        navigator.getPoseStamped([1.39,  2.35], TurtleBot4Directions.EAST ),
        navigator.getPoseStamped([1.82, 4.22], TurtleBot4Directions.EAST),
        navigator.getPoseStamped([1.05, 4.6], TurtleBot4Directions.WEST),
        navigator.getPoseStamped([0.23, 2.8], TurtleBot4Directions.EAST),
        navigator.getPoseStamped([-2.06, 1.84], TurtleBot4Directions.SOUTH),
        navigator.getPoseStamped([-0.42, 4.4], TurtleBot4Directions.SOUTH),
        navigator.getPoseStamped([-1.53, 4.82], TurtleBot4Directions.WEST ),
    ]
    home_pose = navigator.getPoseStamped([0.63,  0.87], TurtleBot4Directions.EAST)

    try:
        # ── 1. 트리거 대기 ──
        navigator.get_logger().info('웹캠 감지 신호 대기 중...')
        while rclpy.ok() and not node.object_detected:
            executor.spin_once(timeout_sec=0.1)

        if not rclpy.ok():
            return

        # ── 2. 언독 ──
        navigator.get_logger().info('객체 감지됨. 언독 후 순찰 시작.')
        navigator.undock()

        # ── 3. 순찰 + 추론 메인루프 ──
        goal_index = 0
        navigator.startToPose(goal_pose[goal_index])
        navigator.get_logger().info(f'[{goal_index+1}/{len(goal_pose)}] 이동 중...')

        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            node.run_inspect_once()

            if navigator.isTaskComplete():

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

                if node.inspect_done:

                    if node.anomaly_result:
                        navigator.get_logger().info('이상개체 감지!')

                        # ── 4. 1차 approach (싱크 없이 0.5m) ──
                        success = node.approach_target(navigator, executor)

                        # ── 5. 경보음 발행 ──
                        node.play_alarm()

                        if success:
                            # ── 6. 재검증 (싱크 맞춰서 YOLO 1회) ──
                            verified, cx, cy, depth = node.verify_target(executor)

                            if verified:
                                # ── 7. 2차 approach (싱크된 프레임으로 0.3m) ──
                                node.approach_target_final(
                                    navigator, executor, cx, cy, depth
                                )
                            else:
                                navigator.get_logger().info('오탐지 확인. 다음 포인트로.')

                    # ── 8. 다음 랠리포인트로 이동 ──
                    goal_index        += 1
                    node.inspect_done  = False

                    if goal_index >= len(goal_pose):
                        navigator.get_logger().info('모든 랠리포인트 순찰 완료.')
                        break

                    navigator.get_logger().info(f'[{goal_index+1}/{len(goal_pose)}] 이동 중...')
                    navigator.startToPose(goal_pose[goal_index])

    except KeyboardInterrupt:
        node.get_logger().info('종료 요청.')

    # ── 9. 도킹 복귀 ──
    navigator.get_logger().info('순찰 완료. 홈 위치로 이동.')
    navigator.startToPose(home_pose)

    navigator.dock()
    navigator.get_logger().info('미션 완료. 도킹 복귀.')
    node.gui_thread_stop.set()
    node.gui_thread.join()
    rclpy.shutdown()


if __name__ == '__main__':
    main()