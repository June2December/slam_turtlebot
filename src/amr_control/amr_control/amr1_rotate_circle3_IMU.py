import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

ROTATION_SPEED = 0.5  # rad/s
TARGET_ANGLE = 2 * math.pi  # 360도


class RotateOnceIMU(Node):
    def __init__(self):
        super().__init__('rotate_once_imu')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.accumulated = 0.0
        self.prev_time = None
        self.done = False
        self.get_logger().info('IMU 기반 360도 회전 시작')

    def imu_callback(self, msg):
        if self.done:
            return

        now = self.get_clock().now().nanoseconds / 1e9

        if self.prev_time is None:
            self.prev_time = now
            return

        dt = now - self.prev_time
        self.prev_time = now

        self.accumulated += abs(msg.angular_velocity.z * dt)

        if self.accumulated < TARGET_ANGLE:
            twist = Twist()
            twist.angular.z = ROTATION_SPEED
            self.pub.publish(twist)
        else:
            self.pub.publish(Twist())  # 정지
            self.get_logger().info(f'회전 완료 (누적: {math.degrees(self.accumulated):.1f}도)')
            self.done = True
            rclpy.shutdown()


def main():
    rclpy.init()
    rclpy.spin(RotateOnceIMU())


if __name__ == '__main__':
    main()
