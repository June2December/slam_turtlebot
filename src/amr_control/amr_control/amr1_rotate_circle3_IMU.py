import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

ROTATION_SPEED = 0.5
TARGET_ANGLE = 2 * math.pi


class RotateOnceIMU(Node):
    def __init__(self):
        super().__init__('rotate_once_imu')
        self.pub = self.create_publisher(Twist, '/robot4/cmd_vel', 10)
        self.create_subscription(Imu, '/robot4/imu', self.imu_callback, 10)
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
            self.pub.publish(Twist())
            self.get_logger().info(f'회전 완료 ({math.degrees(self.accumulated):.1f}도)')
            self.done = True  # ✅ 플래그만 세우고 shutdown은 밖에서


def main():
    rclpy.init()
    node = RotateOnceIMU()
    try:
        while rclpy.ok() and not node.done:  # ✅ done 되면 루프 탈출
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()  # ✅ spin 완전히 끝난 후 호출


if __name__ == '__main__':
    main()