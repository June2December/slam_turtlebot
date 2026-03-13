import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


ROTATION_SPEED = 0.5  # rad/s
ROTATION_DURATION = 2 * math.pi / ROTATION_SPEED  # ~12.6초


class RotateOnce(Node):
    def __init__(self):
        super().__init__('rotate_once')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('제자리 360도 회전 시작')

    def tick(self):
        elapsed = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        msg = Twist()

        if elapsed < ROTATION_DURATION:
            msg.angular.z = ROTATION_SPEED
            self.pub.publish(msg)
        else:
            self.pub.publish(msg)  # 정지
            self.get_logger().info('회전 완료')
            self.timer.cancel()
            rclpy.shutdown()


def main():
    rclpy.init()
    rclpy.spin(RotateOnce())


if __name__ == '__main__':
    main()
