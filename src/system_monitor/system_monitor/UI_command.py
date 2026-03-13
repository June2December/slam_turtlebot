from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.executors import SingleThreadedExecutor

class UICommandNode(Node):
    def __init__(self):
        super().__init__("ui_controller_node")

        self.amr1_teleop_pub = self.create_publisher(String, "/amr1/teleop_cmd", 10)
        self.amr1_home_pub = self.create_publisher(String, "/amr1/home_cmd", 10)

        self.amr2_teleop_pub = self.create_publisher(String, "/amr2/teleop_cmd", 10)
        self.amr2_home_pub = self.create_publisher(String, "/amr2/home_cmd", 10)

    def send_teleop(self, robot_id: str, direction: str):
        msg = String()
        msg.data = direction

        if robot_id == "amr1":
            self.amr1_teleop_pub.publish(msg)
            self.get_logger().info(f"/amr1/teleop_cmd -> {direction}")

        elif robot_id == "amr2":
            self.amr2_teleop_pub.publish(msg)
            self.get_logger().info(f"/amr2/teleop_cmd -> {direction}")

    def send_home(self, robot_id: str):
        msg = String()
        msg.data = "home"

        if robot_id == "amr1":
            self.amr1_home_pub.publish(msg)
            self.get_logger().info("/amr1/home_cmd -> home")

        elif robot_id == "amr2":
            self.amr2_home_pub.publish(msg)
            self.get_logger().info("/amr2/home_cmd -> home")


_command_node: UICommandNode | None = None
_executor_started = False
_executor = None

def init_ui_command():
    global _command_node, _executor_started, _executor

    if _executor_started:
        return

    if not rclpy.ok():
        rclpy.init()

    _command_node = UICommandNode()
    _executor = SingleThreadedExecutor()
    _executor.add_node(_command_node)
    _executor_started = True

def spin_ui_command():
    global _executor

    if _executor is not None:
        _executor.spin()

def send_teleop(robot_id: str, direction: str):
    if _command_node is not None:
        _command_node.send_teleop(robot_id, direction)


def send_home(robot_id: str):
    if _command_node is not None:
        _command_node.send_home(robot_id)


def main(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)

    node = UICommandNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()