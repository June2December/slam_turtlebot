#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator


def main():
    rclpy.init()
    navigator = TurtleBot4Navigator()

    # 도킹 상태 확인
    if not navigator.getDockedStatus():
        navigator.info('Docking before initialising pose')
        navigator.dock()

    # 초기 위치 설정
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()

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

    # Undock 후 순찰
    navigator.undock()

    for i, goal in enumerate(goal_pose):
        navigator.startToPose(goal)
        navigator.get_logger().info(f'[{i+1}/{len(goal_pose)}] 이동 중...')

        while not navigator.isTaskComplete():
            pass

        navigator.get_logger().info(f'[{i+1}/{len(goal_pose)}] 도착 완료')

    # 모든 포인트 완료 후 복귀
    navigator.dock()
    navigator.get_logger().info('미션 완료. 도킹 복귀.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()