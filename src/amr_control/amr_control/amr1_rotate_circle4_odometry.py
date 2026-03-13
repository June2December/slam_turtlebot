import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

ROTATION_SPEED = 0.5  # rad/s
TARGET_ANGLE = 2 * math.pi  # 360도


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RotateOnceOdom(Node):
    def __init__(self):
        super().__init__('rotate_once_odom')
        self.pub = self.create_publisher(Twist, '/robot4/cmd_vel', 10)
        self.create_subscription(Odometry, '/robot4/odom', self.odom_callback, 10)
        self.accumulated = 0.0
        self.prev_yaw = None
        self.done = False
        self.get_logger().info('Odometry 기반 360도 회전 시작')

    def odom_callback(self, msg):
        if self.done:
            return

        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if self.prev_yaw is None:
            self.prev_yaw = yaw
            return

        diff = yaw - self.prev_yaw
        # -π ~ +π wraparound 처리
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi

        self.accumulated += abs(diff)
        self.prev_yaw = yaw

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
    node = RotateOnceOdom()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
