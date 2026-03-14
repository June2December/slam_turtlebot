import csv
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from irobot_create_msgs.msg import WheelVelocities


class MotionDelayTester(Node):
    def __init__(self):
        super().__init__('amr1_motion_delay_tester')

        # 출발점 / 도착점 오도메트리 x 좌표 (나중에 실측값으로 교체)
        self.start_x = 0.0
        self.goal_x = 2.0

        # 직진 3단계 속도 (m/s)
        self.linear_speeds = [0.1, 0.2, 0.3]

        # 회전 3단계 속도 (rad/s)
        self.angular_speeds = [0.2, 0.4, 0.6]

        # 상태
        self.current_odom_x = 0.0
        self.current_yaw = 0.0
        self.t_pub = None
        self.last_wheel_vel = None

        # pub
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot4/cmd_vel', 10)

        # sub
        self.create_subscription(WheelVelocities, '/robot4/wheel_vels', self.wheel_vels_callback, 10)
        self.create_subscription(Odometry, '/robot4/odom', self.odom_callback, 10)
        self.create_subscription(Imu, '/robot4/imu', self.imu_callback, 10)

        # CSV
        self.csv_path = './src/amr_control/amr_control/data/motion_delay_log.csv'
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow([
                'motion_type', 'speed_cmd',
                't_pub', 't_recv', 'response_delay_ms'
            ])
            self.csv_file.flush()

        self.get_logger().info('MotionDelayTester start')

    def publish_cmd_vel(self, linear_x: float, angular_z: float):
        # cmd_vel 발행 + T_pub 기록
        # 반환값: t_pub (float)
        pass

    def wheel_vels_callback(self, msg: WheelVelocities):
        # wheel_vels 수신 → 속도 변화 감지 → T_recv 기록 → save_csv 호출
        pass

    def odom_callback(self, msg: Odometry):
        # 현재 x 위치 갱신 (drive_straight 도착 판단에 사용)
        pass

    def imu_callback(self, msg: Imu):
        # 현재 yaw 갱신 (rotate_180 완료 판단에 사용)
        pass

    def drive_straight(self, target_x: float):
        # linear_speeds 3단계로 속도 변화하며 직진
        # odom x가 target_x에 도달하면 정지
        # 각 속도 변화 시점마다 publish_cmd_vel 호출
        pass

    def rotate_180(self):
        # angular_speeds 3단계로 속도 변화하며 회전
        # IMU yaw 기준 180도 달성 시 정지
        # 각 속도 변화 시점마다 publish_cmd_vel 호출
        pass

    def run_mission(self):
        # 전체 시나리오 루프
        # drive_straight(goal_x) → rotate_180() → drive_straight(start_x) → rotate_180() → 반복
        pass

    def save_csv(self, motion_type: str, speed_cmd: float, t_pub: float, t_recv: float):
        # 한 행 기록
        # response_delay_ms = (t_recv - t_pub) * 1000
        pass


def main():
    rclpy.init()
    node = MotionDelayTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.csv_file.close()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
