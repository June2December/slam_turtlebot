"""
1. 객체 탐지 여부에 대한 토픽(커스텀 메세지) 받음
    - /target_event 토픽 하나 만들게
        bool detected / Ture False
        string direction / N E S W
        
2. 진지 점령 / 이때 바로 주사격 방향 지향
    - 점령 했다면 이 노드는 죽어야지
"""

import rclpy

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator


def main():
    rclpy.init()

    navigator = TurtleBot4Navigator()

    # Start on dock
    if not navigator.getDockedStatus():
        navigator.info('Docking before intialising pose')
        navigator.dock()

    # Set initial pose
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.SOUTH)
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2
    navigator.waitUntilNav2Active()

    # Set goal poses
    goal_pose = []

    # 진지 점령까지 경유 1곳, 주사격방향 북쪽 지향0도
    goal_pose.append(navigator.getPoseStamped([-1.55069, 0.0668084], TurtleBot4Directions.WEST))
    goal_pose.append(navigator.getPoseStamped([-0.761671, -0.852567], TurtleBot4Directions.NORTH))

    # Undock
    navigator.undock()

    # Follow Waypoints
    navigator.startFollowWaypoints(goal_pose)

    # # Finished navigating, dock
    # navigator.dock()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
