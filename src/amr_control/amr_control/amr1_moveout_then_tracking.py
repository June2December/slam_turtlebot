"""
Phase 1: AmrMoveout  → 웨이포인트 완료 대기
Phase 2: TargetTracker → 추적 시작
ROS2 통신 없이 Python 단에서 순차 실행 (단일 rclpy 컨텍스트)
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from amr_control.amr1_moveout_follow_waypoints import AmrMoveout
from amr_control.amr1_tracking_aerial import TargetTracker


def main():
    rclpy.init()

    # ────────────────────────────────────────────
    # Phase 1 : 이동 / 웨이포인트 추종
    # ────────────────────────────────────────────
    print('[launcher] Phase 1 시작: 이동 / 웨이포인트 추종')
    moveout_node = AmrMoveout()
    executor1 = MultiThreadedExecutor()
    executor1.add_node(moveout_node)

    try:
        while rclpy.ok() and not moveout_node.done:
            executor1.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        print('[launcher] 인터럽트 → 종료')
        moveout_node.destroy_node()
        rclpy.shutdown()
        return

    # Phase 1 정리 (TurtleBot4Navigator 포함)
    executor1.remove_node(moveout_node)
    try:
        moveout_node.navigator.destroy_node()
    except Exception:
        pass
    moveout_node.destroy_node()
    print('[launcher] Phase 1 완료')

    if not rclpy.ok():
        return

    # ────────────────────────────────────────────
    # Phase 2 : 추적
    # ────────────────────────────────────────────
    print('[launcher] Phase 2 시작: 추적')
    tracker_node = TargetTracker()

    # occupation 토픽 없이 바로 추적 시작
    tracker_node.tracking_enabled = True

    executor2 = MultiThreadedExecutor()
    executor2.add_node(tracker_node)

    try:
        executor2.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                tracker_node.stop_rotation()
        except Exception:
            pass
        try:
            tracker_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        try:
            tracker_node.csv_file.close()
        except Exception:
            pass

    print('[launcher] 종료')


if __name__ == '__main__':
    main()
