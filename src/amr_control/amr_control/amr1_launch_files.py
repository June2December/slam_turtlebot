"""
Phase 1: AmrMoveout  → 웨이포인트 완료 대기
Phase 2: TargetTracker → 추적 시작
ROS2 통신 없이 Python 단에서 순차 실행
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from amr_control.amr1_moveout_follow_waypoints import AmrMoveout
from amr_control.amr1_tracking_aerial import TargetTracker


# ────────────────────────────────────────────
# Phase 1 : 이동 / 웨이포인트 추종
# ────────────────────────────────────────────
def run_moveout():
    rclpy.init()
    node = AmrMoveout()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


# ────────────────────────────────────────────
# Phase 2 : 추적
# ────────────────────────────────────────────
def run_tracking():
    rclpy.init()
    node = TargetTracker()

    # occupation 토픽 없이 바로 추적 시작
    node.tracking_enabled = True

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.stop_rotation()
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
        try:
            node.csv_file.close()
        except Exception:
            pass


# ────────────────────────────────────────────
def main():
    print('[launcher] Phase 1 시작: 이동 / 웨이포인트 추종')
    run_moveout()
    print('[launcher] Phase 1 완료 → Phase 2 시작: 추적')
    run_tracking()
    print('[launcher] 종료')


if __name__ == '__main__':
    main()
