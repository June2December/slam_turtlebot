"""
/tracking_done 토픽 받으면 (Bool : true)
철수하도록 해야지
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from turtlebot4_navigation.turtlebot4_navigator import (
    TurtleBot4Directions,
    TurtleBot4Navigator,
)


class AmrPullout(Node):
    def __init__(self):
        super().__init__("amr_pullout")

        # Nav 객체
        self.navigator = TurtleBot4Navigator()

        # 중복 철수 방지 랑 철수 완료 일단 상태 머신으로 
        self.started = False
        self.done = False

        # 완료 확인용 타이머 저장
        self.result_timer = None

        # tracking_done == True 이면 철수 시작
        self.create_subscription(Bool, "/tracking_done", self.tracking_done_cb, 10)

        # Nav2 서버가 켜지면 초기화 끝나는걸로
        self.navigator.waitUntilNav2Active()

        print("amr_pullout node 복귀 시작 입감완료")

    def tracking_done_cb(self, msg):
        # False면 무시
        if not msg.data:
            print("/tracking_done=False -> 무시")
            return
        # 중복 명령 방지  ? 다시 호출되면 가야 되는거 같긴 한데
        if self.started:
            print("이미 철수 중이므로 중복 명령 무시")
            return

        self.started = True
        print("/tracking_done=True 수신 -> 철수 시작")

        self.start_pullout()

    def start_pullout(self):
        goal_pose = []
        # 1. 경유지
        goal_pose.append(self.navigator.getPoseStamped([0.423, 1.86], TurtleBot4Directions.NORTH_EAST))

        # 2. 차양대(복귀 위치)
        goal_pose.append(self.navigator.getPoseStamped([-0.2, 1.0], TurtleBot4Directions.EAST))

        # 철수 시작
        self.navigator.startFollowWaypoints(goal_pose)
        print("철수 waypoint 전송 완료")

        # 완료 확인 타이머 시이작
        self.result_timer = self.create_timer(0.5, self.check_result)

    def check_result(self):
        if self.done:
            print("복귀 완료 했지롱")
            return

        # 아직 진행 중이면 계속 타이머 셈 하면서 대기
        if not self.navigator.isTaskComplete():
            return

        self.done = True

        if self.result_timer is not None:
            self.result_timer.cancel()

        result = self.navigator.getResult()
        print(f"철수 작업 종료 result={result}")

        # 노드 종료
        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = AmrPullout()
    rclpy.spin(node)


if __name__ == "__main__":
    main()