"""
1. 객체 탐지 여부에 대한 토픽(커스텀 메세지) 받음
    - /target_event 토픽 하나 만들게        
2. 진지 점령 / 이때 바로 주사격 방향 지향
    - 점령 했다면 이 노드는 죽어야지

============ 아래가 커스텀 메세지 임============
토픽 이름은 /target_event 잉
bool detected           True(적기든 아군기든 일단 True 일지도?) / False
string direction        N, S, W, E  중 하나만 꼭 보내라잉 여기선 다른거 못한다  
"""

import rclpy
from rclpy.node import Node
from amr_interfaces.msg import TargetEvent
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

class AmrMoveout(Node):
    def __init__(self):
        super().__init__("amr_moveout")
        
        # Nav 는 만들어주고
        self.navigator = TurtleBot4Navigator()
        # 종복 방지용 : topic 계속 받으면 현재 추적하고 있는애 추적이 우선이지
        self.started = False
        # 얘가 해당 이름의 토픽을 국룰 인 10까지만 쌓아두는걸로 하고, 어차피 중복 방지는 started 있으니까
        self.create_subscription(TargetEvent, '/target_event', self.target_cb, 10)
    
    
    def target_cb(self, msg):
        # 중복 명령 방지 / 출력문은 나중에 삭제해라잉
        if self.started:
            print("중복 명령은 가볍게 거부한다.")
            return
        
        # 애초에 감지된게 없는데 보내질수도  / 출력문은 나중에 삭제해라잉
        if not msg.detected:
            print("감지된게 없는데 토픽 왜 아직도 보내고 있냐")
            return
        ######################################################################
        # 방향 매핑
        direction_map = {
            "N": TurtleBot4Directions.NORTH_WEST,
            "S": TurtleBot4Directions.SOUTH_EAST,
            "E": TurtleBot4Directions.NORTH_EAST,
            "W": TurtleBot4Directions.SOUTH_WEST
        }
        # 나중에 웹캠 방향 바뀌면 이거 바꾸면 될듯
        if msg.direction not in direction_map:
            print(f"잘못된 direction: {msg.direction}")
            return
        self.started = True
        navigator = self.navigator
        target_dir = direction_map.get(msg.direction)
        ######################################################################
        # (임시) Start on dock
        if not navigator.getDockedStatus():
            navigator.info('Docking before intialising pose')
            navigator.dock()
        
        # Set initial pose
        initial_pose = navigator.getPoseStamped([-0.1, 0.7], TurtleBot4Directions.EAST)
        # initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.EAST)
        navigator.setInitialPose(initial_pose)

        # Wait for Nav2
        navigator.waitUntilNav2Active()
        
        goal_pose = []
        # 경유지
        goal_pose.append(navigator.getPoseStamped([0.423, 1.86], TurtleBot4Directions.SOUTH_WEST))
        # 최종 진지 (방향만 동적)
        goal_pose.append(navigator.getPoseStamped([-0.584, 2.2], target_dir))
        
        # 이제 진짜 출동
        navigator.undock()
        navigator.startFollowWaypoints(goal_pose)
        
        # 점령 다 했으면 이 노드는 사라져 줘야지?
        # 죽기전에 topic 보내야 하나? 이제 추적 하라고?

def main():
    rclpy.init()
    
    node = AmrMoveout()
    
    rclpy.spin(node)

if __name__ == '__main__':
    main()