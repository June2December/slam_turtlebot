# capture by turtle4.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
# capture by turtle4.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os

class ImageCaptureNode(Node):
    """
    OAK-D 카메라(RGB)에서 압축 이미지를 수신하여
    실시간으로 화면에 표시하고, 키 입력으로 저장하는 ROS2 노드.

    - 'c': 현재 프레임을 JPG로 저장
    - 'q': 프로그램 종료
    """

    def __init__(self):
        super().__init__('image_capture_node')

        # ROS2 파라미터 선언 (ros2 run 또는 launch 파일에서 불러오기도 가능)
        self.declare_parameter('save_directory', '/home/sinya/slam_turtlebot/src/detection/captured_imgs')
        self.declare_parameter('file_prefix', 'turtle01')

        # 파라미터 값 읽기
        save_directory = self.get_parameter('save_directory').get_parameter_value().string_value
        file_prefix    = self.get_parameter('file_prefix').get_parameter_value().string_value

        # OAK-D RGB 압축 이미지 토픽 구독
        # robot1의 카메라에서 CompressedImage 메시지를 수신
        self.subscription = self.create_subscription(
            CompressedImage,
            '/robot1/oakd/rgb/image_raw/compressed',
            self.listener_callback,
            10  # QoS depth: 큐에 최대 10개 메시지 유지
        )

        self.frame = None          # 가장 최근에 수신한 디코딩된 프레임
        self.save_directory = save_directory
        self.file_prefix = f"{file_prefix}_"  # 저장 파일명 접두사 (예를 들면: "turtle01_")
        self.image_count = 0       # 저장된 이미지 수 (파일명 인덱스로 사용)

        # 저장 디렉토리가 없으면 자동 생성
        os.makedirs(self.save_directory, exist_ok=True)

        self.get_logger().info(f"저장 경로: {self.save_directory}")
        self.get_logger().info(f"파일 접두사: {self.file_prefix}")

    def listener_callback(self, msg: CompressedImage):
        """
        ROS2 토픽에서 압축 이미지 메시지를 수신할 때마다 호출되는 콜백.
        JPEG/PNG 압축 데이터를 OpenCV BGR 이미지로 디코딩하여 self.frame에 저장.
        """
        # CompressedImage.data(bytes) → NumPy 1D 배열로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)

        # JPEG/PNG 압축 데이터를 BGR 컬러 이미지로 디코딩
        self.frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.frame is None:
            self.get_logger().warn("이미지 디코딩 실패: 손상된 메시지일 수 있습니다.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()

    try:
        while rclpy.ok():
            # 논블로킹 방식으로 ROS 콜백 1회 처리 (최대 0.1초 대기)
            # → OpenCV 창과 ROS 콜백을 단일 스레드에서 번갈아 처리
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.frame is not None:
                # 실시간 카메라 피드를 윈도우에 표시
                cv2.imshow("Live Feed", node.frame)

                # 키 입력 감지 (1ms 대기, 0xFF 마스크로 ASCII 코드 추출)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    # 'c' 키: 현재 프레임을 JPG 파일로 저장
                    # 파일명 예시: turtle01_img_0.jpg, turtle01_img_1.jpg ...
                    file_name = os.path.join(
                        node.save_directory,
                        f"{node.file_prefix}img_{node.image_count}.jpg"
                    )
                    cv2.imwrite(file_name, node.frame)
                    print(f"이미지 저장 완료: {file_name}")
                    node.image_count += 1

                elif key == ord('q'):
                    # 'q' 키: 루프 종료 → 프로그램 정상 종료
                    break

    except KeyboardInterrupt:
        
        pass    # ctrl + c 도 예외없이 종료

    finally:
        # 노드 및 OpenCV 자원 해제 (예외 발생 여부와 무관하게 항상 실행)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()