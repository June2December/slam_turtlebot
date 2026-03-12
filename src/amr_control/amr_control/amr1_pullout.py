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

        # [수정] 앞에 / 제거
        # 절대 토픽 "/tracking_done" 이 아니라 상대 토픽 "tracking_done" 으로 받음
        # 따라서 노드를 /robot4 네임스페이스로 띄우면 실제 구독 토픽은 /robot4/tracking_done 이 됨
        self.create_subscription(Bool, "tracking_done", self.tracking_done_cb, 10)

        # Nav2 서버가 켜지면 초기화 끝나는걸로
        self.navigator.waitUntilNav2Active()

        # [수정] print 대신 logger 사용
        self.get_logger().info("amr_pullout node 복귀 시작 입력완료")

    def tracking_done_cb(self, msg):
        # [추가] 콜백이 실제로 들어오는지 확인용 로그
        self.get_logger().info(f"tracking_done_cb 들어옴: {msg.data}")

        # False면 무시
        if not msg.data:
            # [수정] logger 사용
            self.get_logger().info("tracking_done=False -> 무시")
            return

        # 중복 명령 방지
        if self.started:
            # [수정] logger 사용
            self.get_logger().info("이미 철수 중이므로 중복 명령 무시")
            return

        self.started = True
        # [수정] logger 사용
        self.get_logger().info("tracking_done=True 수신 -> 철수 시작")

        self.start_pullout()

    def start_pullout(self):
        goal_pose = []
        # 경유점
        goal_pose.append(
            self.navigator.getPoseStamped(
                [0.07312959625695838, 1.7249514652651532],
                TurtleBot4Directions.EAST
            )
        )
        # 차양대 좌표
        goal_pose.append(
            self.navigator.getPoseStamped(
                [-0.09283551329186988, 1.0193024901658334],
                TurtleBot4Directions.EAST
            )
        )

        # 철수 시작
        self.navigator.startFollowWaypoints(goal_pose)
        # [수정] logger 사용
        self.get_logger().info("철수 waypoint 전송 완료")

        # 완료 확인 타이머 시작
        self.result_timer = self.create_timer(0.5, self.check_result)

    def check_result(self):
        if self.done:
            # [수정] 완료 후에는 매 타이머마다 불필요한 로그가 계속 찍히지 않게 return만
            return

        # 아직 진행 중이면 계속 대기
        if not self.navigator.isTaskComplete():
            # [수정] 필요 시 확인하려고 유지
            self.get_logger().info("아직 차양대로 복귀 중입니다..")
            return

        # [수정] 타이머 정리 후 None 처리
        if self.result_timer is not None:
            self.result_timer.cancel()
            self.result_timer = None

        result = self.navigator.getResult()
        # [수정] logger 사용
        self.get_logger().info(f"철수 작업 종료 result={result}")

        # 이제 차양대 근처까지 왔으니 마지막 도킹 수행
        # [추가] 도킹 단계 로그
        self.get_logger().info("도킹 시작")
        self.navigator.dock()
        self.get_logger().info("도킹 완료")

        self.done = True

        # 노드 종료
        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = AmrPullout()
    rclpy.spin(node)


if __name__ == "__main__":
    main()