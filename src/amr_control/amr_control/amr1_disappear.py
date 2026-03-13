# ================================================================
# [삽입 디버깅 분석] ObjectLossDetector → amr1_tracking_aerial.py
# ================================================================
#
# 문제 1 | on_detection_success() 호출 위치 (심각도: 낮음)
#   - (u, v) 계산 직후(L207) 호출하면 pixel_to_3d / camera_point_to_map
#     실패 시에도 last_detected_time이 갱신됨
#   - 의도대로 맞음: 중심점은 얻었으므로 객체 소실이 아님
#
# 문제 2 | occupation=False 시 리셋 누락 (심각도: 높음) ← 최우선 처리
#   - occupation_callback의 else 분기에 on_tracking_stop() 없으면
#     추적 재시작 시 이전 타임스탬프 잔류 → 소실 판단 전체가 틀어짐
#   - 수정 위치: amr1_tracking_aerial.py L138
#     else:
#         self.stop_rotation()
#         self.loss_detector.on_tracking_stop()  ← 추가 필수
#
# 문제 3 | 이미지 디코드 실패 분기 포함 여부 (심각도: 중간)
#   - L156-162 image convert fail / rgb or depth is None 시
#     on_detection_fail() 호출 없음
#   - 카메라 문제인지 객체 소실인지 애매 → 포함시키는 게 안전
#
# 문제 4 | Thread safety (심각도: 낮음)
#   - MultiThreadedExecutor 환경에서 occupation_callback / synced_callback이
#     다른 스레드에서 last_detected_time 동시 접근 가능
#   - Python GIL로 현재는 커버되나 구조적으로 존재하는 리스크
#
# 결론: 최소기능으로 하나만 넣는다면 문제 2 (occupation_callback 리셋)
# ================================================================

"""
객체 소실 판단 로직 단위 테스트
ROS2 / YOLO / 카메라 없이 순수 Python으로 실행 가능

실행 방법:
    python3 amr1_disappear.py
"""

import time


# ----------------------------------------------------------------
# 테스트 대상 로직 (amr1_tracking_aerial.py 에 들어갈 실제 코드)
# ----------------------------------------------------------------

class ObjectLossDetector:
    def __init__(self, loss_timeout=3.0):
        self.loss_timeout = loss_timeout
        self.last_detected_time = None
        self.object_lost = False

    def on_detection_success(self):
        """YOLO로 객체 중심점(u, v)을 얻었을 때 호출"""
        self.last_detected_time = time.time()
        self.object_lost = False

    def on_detection_fail(self):
        """YOLO 탐지 실패 / boxes 없음 등 중심점을 못 얻었을 때 호출"""
        if self.last_detected_time is None:
            # 한 번도 탐지된 적 없음 → 소실 아님
            return False

        elapsed = time.time() - self.last_detected_time
        if elapsed > self.loss_timeout:
            self.object_lost = True

        return self.object_lost

    def on_tracking_stop(self):
        """occupation=False 로 추적 중단 시 상태 리셋"""
        self.last_detected_time = None
        self.object_lost = False


# ----------------------------------------------------------------
# 테스트 시나리오
# ----------------------------------------------------------------

def test_never_detected():
    """시나리오 1: 한 번도 탐지 안 된 상태에서 탐지 실패 → 소실 아님"""
    print("\n[시나리오 1] 탐지 이력 없음 + 탐지 실패")
    d = ObjectLossDetector(loss_timeout=3.0)

    result = d.on_detection_fail()
    assert result == False, "FAIL: 탐지 이력 없으면 소실 아님"
    assert d.object_lost == False
    print("  PASS: 소실 판단 안 함")


def test_short_frame_drop():
    """시나리오 2: 탐지 성공 후 1초 드랍 → 소실 아님"""
    print("\n[시나리오 2] 탐지 성공 후 1초 프레임 드랍")
    d = ObjectLossDetector(loss_timeout=3.0)

    d.on_detection_success()
    time.sleep(1.0)

    result = d.on_detection_fail()
    assert result == False, "FAIL: 1초 드랍은 소실 아님"
    assert d.object_lost == False
    print("  PASS: 소실 아님 (1초 드랍)")


def test_loss_timeout():
    """시나리오 3: 탐지 성공 후 3초 초과 → 소실"""
    print("\n[시나리오 3] 탐지 성공 후 3초 초과 드랍")
    d = ObjectLossDetector(loss_timeout=3.0)

    d.on_detection_success()
    time.sleep(3.2)

    result = d.on_detection_fail()
    assert result == True, "FAIL: 3초 초과는 소실이어야 함"
    assert d.object_lost == True
    print("  PASS: 소실 판단 (3.2초 드랍)")


def test_recovery_after_loss():
    """시나리오 4: 소실 판단 후 객체 재탐지 → 소실 해제"""
    print("\n[시나리오 4] 소실 판단 후 객체 재탐지")
    d = ObjectLossDetector(loss_timeout=3.0)

    d.on_detection_success()
    time.sleep(3.2)
    d.on_detection_fail()
    assert d.object_lost == True

    d.on_detection_success()
    assert d.object_lost == False, "FAIL: 재탐지 후 소실 플래그 해제되어야 함"
    print("  PASS: 재탐지 후 소실 해제")


def test_tracking_stop_reset():
    """시나리오 5: occupation=False 로 추적 중단 → 상태 리셋"""
    print("\n[시나리오 5] 추적 중단(occupation=False) 후 리셋")
    d = ObjectLossDetector(loss_timeout=3.0)

    d.on_detection_success()
    time.sleep(3.2)
    d.on_detection_fail()
    assert d.object_lost == True

    d.on_tracking_stop()
    assert d.last_detected_time is None, "FAIL: last_detected_time 리셋 안 됨"
    assert d.object_lost == False, "FAIL: object_lost 리셋 안 됨"
    print("  PASS: 추적 중단 후 상태 리셋")


def test_rapid_detection():
    """시나리오 6: 연속 탐지 성공 → 소실 판단 안 함"""
    print("\n[시나리오 6] 연속 탐지 성공 (정상 추적 중)")
    d = ObjectLossDetector(loss_timeout=3.0)

    for i in range(5):
        d.on_detection_success()
        time.sleep(0.1)
        result = d.on_detection_fail()
        # 0.1초 드랍이므로 소실 아님
        assert result == False, f"FAIL: {i}번째 반복에서 소실 오판"

    print("  PASS: 연속 탐지 중 소실 오판 없음")


# ----------------------------------------------------------------
# 메인
# ----------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 50)
    print("객체 소실 판단 로직 단위 테스트")
    print("=" * 50)

    test_never_detected()
    test_short_frame_drop()
    test_loss_timeout()          # 3초 대기 있음
    test_recovery_after_loss()   # 3초 대기 있음
    test_tracking_stop_reset()   # 3초 대기 있음
    test_rapid_detection()

    print("\n" + "=" * 50)
    print("전체 테스트 통과")
    print("=" * 50)
