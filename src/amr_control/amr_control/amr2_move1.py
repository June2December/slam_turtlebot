#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from amr_interfaces.msg import TargetEvent
from geometry_msgs.msg import PoseWithCovarianceStamped

class WebcamDetect(Node):
    def __init__(self):
        super().__init__('webcame_detect')
        self.object_detected = False
        
        self.create_subscription(TargetEvent,'/detected', self.listener_cb, 1)
        self.create_subscription(PoseWithCovarianceStamped,'/robot1/amcl_pose', self.amcl_callback, 1)


    def listener_cb(self, msg):
        # 웹캠 객체 감지됨 여부
        self.object_detected = msg.detected
        
        
    def amcl_callback(self, msg):
        """AMCL 위치 추정 결과를 수신하여 로봇의 현재 맵 좌표를 갱신"""
        self.robot_x           = msg.pose.pose.position.x
        self.robot_y           = msg.pose.pose.position.y
        self.robot_orientation = msg.pose.pose.orientation

def main():
    rclpy.init()

    navigator = TurtleBot4Navigator()

    webcam= WebcamDetect()

    # Start on dock
    if not navigator.getDockedStatus():
        navigator.info('Docking before intialising pose')
        navigator.dock()

    # Set initial pose
    initial_pose = navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2
    navigator.waitUntilNav2Active()

    # Set goal poses
    goal_pose = []

   
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
    goal_pose.append(navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST))
     # goal_pose = navigator.getPoseStamped([-1.55, 0.066], TurtleBot4Directions.EAST)
    
    if webcam.object_detected :
        # Undock
        navigator.undock()

        # Follow Waypoints
        navigator.startFollowWaypoints(goal_pose)

        # Finished navigating, dock
        navigator.dock()


        # Go to each goal pose
        # navigator.startToPose(goal_pose)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
