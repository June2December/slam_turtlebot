# gas_db_log.py
#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from system_monitor.system_monitor.dummy.firebase_config import init_firebase, get_reference


class GasFirebaseLoggerNode(Node):
    def __init__(self):
        super().__init__("gas_write_to_firebase_node")
        self.get_logger().info("Gas Firebase Logger Node 시작")

        # ===================
        # Firebase 초기화
        # ===================
        try:
            init_firebase()
            self.get_logger().info("Firebase 초기화 완")
        except Exception as e:
            self.get_logger().error(f"Firebase 초기화 실패: {e}")
            raise

        # ============================================================
        # Firebase reference
        # ============================================================
        self.detection_ref = get_reference("/gas_monitor/detection")
        self.violation_ref = get_reference("/gas_monitor/violation")
        self.stats_ref = get_reference("/gas_monitor/stats")

        # ============================================================
        # 설정값
        # ============================================================
        self.threshold = 1000.0
        self.min_log_interval = 2.0
        self.last_logged_time = 0.0

        # ============================================================
        # ROS2 Subscriber
        # ============================================================
        self.subscription = self.create_subscription(
            Float32,
            "/gas_ppm",
            self.gas_callback,
            10
        )

        self.get_logger().info("구독 시작: /gas_ppm")

    def read_stats(self) -> Dict[str, int]:
        try:
            stats = self.stats_ref.get()
            if not stats:
                return {"total": 0, "alert": 0}

            return {
                "total": int(stats.get("total", 0)),
                "alert": int(stats.get("alert", 0)),
            }
        except Exception as e:
            self.get_logger().error(f"stats 읽기 실패: {e}") #
            return {"total": 0, "alert": 0}

    def is_valid_ppm(self, ppm: float) -> bool:
        if ppm is None:
            return False
        if ppm < 0:
            return False
        if ppm > 1000000:
            return False
        return True

    def gas_callback(self, msg: Float32):
        try:
            ppm = float(msg.data)

            if not self.is_valid_ppm(ppm):
                self.get_logger().warn(f"유효하지 않은 ppm 값 무시: {ppm}")
                return

            now = time.time()

            # 너무 잦은 로그는 트래픽 증가될 수 있음
            if now - self.last_logged_time < self.min_log_interval:
                return

            stats = self.read_stats()
            total = stats["total"] + 1
            alert = stats["alert"]

            # 1) 전체 측정 로그 저장
            detection_payload = {
                "ppm": ppm,
                "timestamp": now,
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            }
            self.detection_ref.push(detection_payload)

            # 2) 기준치 초과 시 violation 저장
            if ppm >= self.threshold:
                violation_payload = {
                    "name": f"Gas Alert: {ppm:.1f} ppm",
                    "ppm": ppm,
                    "timestamp": now,
                    "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                }
                self.violation_ref.push(violation_payload)
                alert += 1

                self.get_logger().warn(
                    f"[ALERT] ppm={ppm:.1f} / total={total} / alert={alert}"
                )
            else:
                self.get_logger().info(
                    f"[NORMAL] ppm={ppm:.1f} / total={total} / alert={alert}"
                )

            # 3) 통계 저장
            self.stats_ref.set({
                "total": total,
                "alert": alert,
                "last_update": now,
                "last_update_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "threshold": self.threshold,
            })

            self.last_logged_time = now

        except Exception as e:
            self.get_logger().error(f"gas_callback 처리 실패: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GasFirebaseLoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt로 종료")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()