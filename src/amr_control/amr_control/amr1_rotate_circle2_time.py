import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys


class RotateOnce(Node):
    def __init__(self, namespace=''):
        super().__init__('rotate_once')  # ✅ 네임스페이스 주입
        self.pub = self.create_publisher(Twist, '/robot4/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.count = 0

    def tick(self):
        if self.count < 126:
            msg = Twist()
            msg.angular.z = 0.5
            self.pub.publish(msg)
            self.count += 1
        else:
            self.pub.publish(Twist())
            self.timer.cancel()
            self.get_logger().info('완료')
            raise SystemExit


def main():
    rclpy.init()
    # 실행 시: python rotate.py robot1  (네임스페이스 인자)
    ns = sys.argv[1] if len(sys.argv) > 1 else ''
    node = RotateOnce(namespace=ns)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()