# capture by turtle4.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')

        # input() 대신 ROS2 파라미터로 변경
        self.declare_parameter('save_directory', '/home/sinya/slam_turtlebot/src/detection/captured_imgs')
        self.declare_parameter('file_prefix', 'turtle01')

        save_directory = self.get_parameter('save_directory').get_parameter_value().string_value
        file_prefix = self.get_parameter('file_prefix').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            CompressedImage,
            '/robot1/oakd/rgb/image_raw/compressed',
            self.listener_callback,
            10)
        self.frame = None
        self.save_directory = save_directory
        self.file_prefix = f"{file_prefix}_"
        self.image_count = 0
        os.makedirs(self.save_directory, exist_ok=True)

        self.get_logger().info(f"저장 경로: {self.save_directory}")
        self.get_logger().info(f"파일 접두사: {self.file_prefix}")

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.frame is None:
            self.get_logger().warn("이미지 디코딩 실패")

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.frame is not None:
                cv2.imshow("Live Feed", node.frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    file_name = os.path.join(
                        node.save_directory,
                        f"{node.file_prefix}img_{node.image_count}.jpg"
                    )
                    cv2.imwrite(file_name, node.frame)
                    print(f"Image saved: {file_name}")
                    node.image_count += 1

                elif key == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()