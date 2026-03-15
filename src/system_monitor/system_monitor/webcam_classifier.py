import os
import cv2
import rclpy
import time

from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
from amr_interfaces.msg import TargetEvent


class YoloDetectPublisher(Node):
    def __init__(self):
        super().__init__('yolo_detect_publisher')
        self.first_enemy_captured = False
        self.last_image_pub_time = 0.0
        self.image_pub_interval = 0.3
        # enemy가 처음 안정적으로 감지된 시작 시각
        self.enemy_detect_start_time = None

        # 이번 enemy에 대해 이미 저장했는지 여부
        self.first_enemy_captured = False

        # 처음 감지 후 몇 초 뒤 저장할지
        self.capture_delay = 1.0
        # 이전 프레임에서 안정적으로 enemy가 탐지되었는지 저장
        # False -> True 로 바뀌는 순간에만 이미지 저장하기 위해 사용
        self.prev_enemy_detected = False

        # 마지막으로 enemy를 본 시간
        self.last_enemy_seen_time = 0.0

        # enemy bbox가 잠깐 풀려도 이 시간 안에는 같은 enemy로 간주
        # 예: 1초 이내에 다시 잡히면 새로운 enemy가 아니라 같은 enemy로 처리
        self.last_enemy_timeout = 1.0

        # YOLO detection 모델
        self.model = YOLO(
            "/home/rokey/rokey_ws/src/turtlebot4_beep/turtlebot4_beep/yolo11_v6_n/object_detection/weights/best.pt",
            task="detect"
        )

        # 카메라 열기
        self.cap = cv2.VideoCapture(2)

        # 카메라 연결 실패할 경우 Log 띄우기
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam index 0")

        # confidence 0.65이상만 나오게하기
        self.conf_thres = 0.65

        # publish topic
        self.target_event_pub_robot1 = self.create_publisher(TargetEvent, "/robot1/target_event", 10)
        self.target_event_pub_robot4 = self.create_publisher(TargetEvent, "/robot4/target_event", 10)
        self.enemy_image_pub = self.create_publisher(String, "/webcam/enemy_image_path", 10)

        # 이미지 저장 폴더 위치
        self.save_dir = "/home/rokey/rokey_ws/src/turtlebot4_beep/turtlebot4_beep/captured_images2"
        os.makedirs(self.save_dir, exist_ok=True)

        # 30Hz로 실행
        self.timer = self.create_timer(0.03, self.detect)

    """
    TargetEvent 메시지를 publish
    detected : True / False   (enemy 탐지 결과 publish)
    direction : N  (현재는 방향 계산을 하지 않아서 항상 N 고정)
    """
    def publish_target_event(self, detected: bool):
        msg = TargetEvent()
        msg.detected = detected
        msg.direction = "N"

        self.target_event_pub_robot1.publish(msg)
        self.target_event_pub_robot4.publish(msg)

    def detect(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame, conf=self.conf_thres, verbose=False)
        annotated_frame = frame.copy()
        enemy_detected = False

        if results and len(results) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = str(self.model.names[cls_id]).strip().lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                if label == "enemy":
                    enemy_detected = True
                    color = (0, 0, 255)
                elif label == "friend":
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 0)

                draw_text = f"{label} {conf:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    draw_text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        if enemy_detected:
            self.last_enemy_seen_time = time.time()

        now = time.time()
        stable_enemy_detected = enemy_detected or (
            self.last_enemy_seen_time > 0 and
            (now - self.last_enemy_seen_time) < self.last_enemy_timeout
        )

        # -------------------------------
        # 상태 변화가 있을 때만 publish
        # -------------------------------
        if stable_enemy_detected and not self.prev_enemy_detected:
            self.publish_target_event(True)
            self.get_logger().info("Enemy appeared")

        elif not stable_enemy_detected and self.prev_enemy_detected:
            self.publish_target_event(False)
            self.get_logger().info("Enemy disappeared")

        # -------------------------------
        # enemy가 보이는 동안 캡처 타이머 동작
        # -------------------------------
        if stable_enemy_detected:
            if self.enemy_detect_start_time is None and enemy_detected:
                self.enemy_detect_start_time = now
                self.get_logger().info("Enemy first detected, capture timer started")

            if (
                self.enemy_detect_start_time is not None and
                not self.first_enemy_captured and
                (now - self.enemy_detect_start_time) >= self.capture_delay and
                enemy_detected
            ):
                timestamp = int(time.time() * 1000)

                history_path = os.path.join(self.save_dir, f"enemy_capture_{timestamp}.jpg")
                latest_path = os.path.join(self.save_dir, "enemy_capture_latest.jpg")

                cv2.imwrite(history_path, annotated_frame)
                cv2.imwrite(latest_path, annotated_frame)

                self.enemy_image_pub.publish(String(data=latest_path))
                self.first_enemy_captured = True

                self.get_logger().info(
                    f"Enemy image saved after 1s | history={history_path} | latest={latest_path}"
                )

        else:
            # enemy가 완전히 사라지면 다음 enemy를 위해 초기화
            self.enemy_detect_start_time = None
            self.first_enemy_captured = False

        # 현재 상태 저장
        self.prev_enemy_detected = stable_enemy_detected

        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()