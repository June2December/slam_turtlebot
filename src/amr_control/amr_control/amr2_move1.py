import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from amr_interfaces.msg import TargetEvent
from geometry_msgs.msg import PoseWithCovarianceStamped


class WebcamDetect(Node):
    def __init__(self):
        super().__init__('webcam_detect')
        self.object_detected = False
        self.create_subscription(TargetEvent, '/detected', self.listener_cb, 1)

    def listener_cb(self, msg):
        self.object_detected = msg.detected


def main():
    rclpy.init()

    navigator = TurtleBot4Navigator()
    webcam    = WebcamDetect()

    executor = MultiThreadedExecutor()
    executor.add_node(webcam)

    # 도킹 확인
    if not navigator.getDockedStatus():
        navigator.dock()

    # 초기 위치 설정
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()

    # 웹캠 신호 수신 대기
    navigator.get_logger().info('웹캠 감지 신호 대기 중...')
    while not webcam.object_detected:
        executor.spin_once(timeout_sec=0.1)
    navigator.get_logger().info('객체 감지됨. 순찰 시작.')

    # 랠리 포인트 7개
    goal_pose = [
        navigator.getPoseStamped([-1.55,  0.07], TurtleBot4Directions.EAST ),
        navigator.getPoseStamped([-1.00,  1.00], TurtleBot4Directions.NORTH),
        navigator.getPoseStamped([ 0.00,  1.50], TurtleBot4Directions.NORTH),
        navigator.getPoseStamped([ 1.00,  1.00], TurtleBot4Directions.EAST ),
        navigator.getPoseStamped([ 1.50,  0.00], TurtleBot4Directions.SOUTH),
        navigator.getPoseStamped([ 1.00, -1.00], TurtleBot4Directions.SOUTH),
        navigator.getPoseStamped([ 0.00, -1.50], TurtleBot4Directions.WEST ),
    ]
    if not navigator.undock():
        navigator.undock()

    for i, goal in enumerate(goal_pose):
        navigator.startToPose(goal)
        navigator.get_logger().info(f'[{i+1}/{len(goal_pose)}] 이동 중...')

        while not navigator.isTaskComplete():
            executor.spin_once(timeout_sec=0.1)

        navigator.get_logger().info(f'[{i+1}/{len(goal_pose)}] 도착 완료')

    navigator.dock()
    navigator.get_logger().info('미션 완료. 도킹 복귀.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()