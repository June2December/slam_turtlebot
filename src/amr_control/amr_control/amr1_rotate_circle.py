# ================================================================
# [삽입 제안] RotationAfterLoss + ObjectLossDetector → amr1_tracking_aerial.py
# ================================================================
#
# 삽입 위치: amr1_tracking_aerial.py
#
# 1. import 추가 (파일 상단)
#      import math
#
# 2. 상수 + 클래스 2개 (TargetTracker 클래스 정의 바로 위)
#      ROTATION_SPEED = 0.5
#      ROTATION_DURATION = 2 * math.pi / ROTATION_SPEED
#      class ObjectLossDetector: ...
#      class RotationAfterLoss: ...
#
# 3. __init__ 에 인스턴스 생성
#      self.loss_detector = ObjectLossDetector(loss_timeout=3.0)
#      self.rotation_handler = RotationAfterLoss()
#
# 4. occupation_callback else 분기에 리셋 추가 (L138)
#      self.loss_detector.on_tracking_stop()
#      self.rotation_handler.reset()
#
# 5. synced_callback 분기 수정
#
#    [tracking_enabled 체크 직후 — YOLO 전에 삽입]
#      if self.rotation_handler.is_rotating_360 or self.rotation_handler.rotation_done:
#          cmd = self.rotation_handler.get_cmd()
#          if cmd == 'rotating':
#              self.publish_rotation(ROTATION_SPEED)
#          elif cmd == 'stopping':
#              self.publish_rotation(0.0)
#          return   ← YOLO 스킵
#
#    [YOLO 탐지 실패 분기들 (no result / no box / best_box None)]
#      기존: self.stop_rotation() + return
#      변경: if self.loss_detector.on_detection_fail():
#                self.rotation_handler.start()
#            else:
#                self.stop_rotation()
#            return
#
#    [탐지 성공 후 (u, v) 계산 직후 L207)]
#      self.loss_detector.on_detection_success()
#
# ================================================================

"""
객체 소실 → 360도 회전 → 정지 단위기능
ROS2 / 카메라 없이 순수 Python으로 실행 가능

실행 방법:
    python3 amr1_rotate_circle.py
"""

import time
import math


ROTATION_SPEED = 0.5                        # rad/s 고정
ROTATION_DURATION = 2 * math.pi / ROTATION_SPEED  # ≈ 12.6초
STOP_DURATION = 0.5                         # 정지 명령 퍼블리시 유지 시간


# ----------------------------------------------------------------
# 테스트 대상 로직
# ----------------------------------------------------------------

class RotationAfterLoss:
    def __init__(self):
        self.is_rotating_360 = False
        self.rotation_done = False
        self.rotation_start_time = None

    def start(self):
        """객체 소실 판단 직후 호출. 이미 회전 중이거나 완료된 경우 무시"""
        if self.is_rotating_360 or self.rotation_done:
            return
        self.is_rotating_360 = True
        self.rotation_done = False
        self.rotation_start_time = time.time()

    def get_cmd(self):
        """
        synced_callback 매 프레임마다 호출
        반환값:
            'rotating'  → angular_z = ROTATION_SPEED 퍼블리시
            'stopping'  → angular_z = 0.0 퍼블리시 (0.5초간)
            'done'      → 종료, 외부 모션 대기
        """
        if not self.is_rotating_360:
            return 'done'

        elapsed = time.time() - self.rotation_start_time

        if elapsed < ROTATION_DURATION:
            return 'rotating'
        elif elapsed < ROTATION_DURATION + STOP_DURATION:
            return 'stopping'
        else:
            self.is_rotating_360 = False
            self.rotation_done = True
            return 'done'

    def reset(self):
        """occupation=False 또는 외부 모션 수신 시 호출"""
        self.is_rotating_360 = False
        self.rotation_done = False
        self.rotation_start_time = None


# ----------------------------------------------------------------
# 테스트 시나리오
# ----------------------------------------------------------------

def test_start_state():
    """시나리오 1 — start() 직후 상태 확인"""
    print('\n[시나리오 1] start() 직후 상태')
    r = RotationAfterLoss()
    r.start()
    assert r.is_rotating_360 == True
    assert r.rotation_done == False
    assert r.rotation_start_time is not None
    print('  PASS: is_rotating_360=True, rotation_done=False')


def test_rotating_phase():
    """시나리오 2 — 회전 중간 시점에서 get_cmd() = rotating"""
    print('\n[시나리오 2] 회전 중간 (6초 경과)')
    r = RotationAfterLoss()
    r.start()
    time.sleep(6.0)
    cmd = r.get_cmd()
    assert cmd == 'rotating', f'FAIL: expected rotating, got {cmd}'
    print(f'  PASS: elapsed≈6s → {cmd}')


def test_stopping_phase():
    """시나리오 3 — 360도 완료 직후 정지 명령 구간"""
    print('\n[시나리오 3] 360도 완료 후 정지 구간 (12.7초 경과)')
    r = RotationAfterLoss()
    r.start()
    time.sleep(ROTATION_DURATION + 0.1)
    cmd = r.get_cmd()
    assert cmd == 'stopping', f'FAIL: expected stopping, got {cmd}'
    print(f'  PASS: elapsed≈{ROTATION_DURATION+0.1:.1f}s → {cmd}')


def test_done_phase():
    """시나리오 4 — 정지 구간 완료 후 done, 재트리거 없음"""
    print('\n[시나리오 4] 정지 구간 완료 후 done')
    r = RotationAfterLoss()
    r.start()
    time.sleep(ROTATION_DURATION + STOP_DURATION + 0.1)

    cmd = r.get_cmd()
    assert cmd == 'done', f'FAIL: expected done, got {cmd}'
    assert r.rotation_done == True
    assert r.is_rotating_360 == False

    # 추가 호출해도 done 유지
    cmd2 = r.get_cmd()
    assert cmd2 == 'done', 'FAIL: 재호출 시 done 유지 안 됨'
    print(f'  PASS: done 상태 유지, 재트리거 없음')


def test_double_start():
    """시나리오 5 — start() 중복 호출 시 타이머 리셋 방지"""
    print('\n[시나리오 5] start() 중복 호출 (매 프레임 소실 판단 시뮬레이션)')
    r = RotationAfterLoss()
    r.start()
    t1 = r.rotation_start_time

    time.sleep(0.1)
    r.start()  # 두 번째 호출 → 무시되어야 함
    t2 = r.rotation_start_time

    assert t1 == t2, 'FAIL: 중복 start()로 타이머가 리셋됨'
    print('  PASS: 중복 start() 무시, 타이머 유지')


def test_reset():
    """시나리오 5 — reset() 후 상태 초기화"""
    print('\n[시나리오 5] reset() 후 상태 초기화')
    r = RotationAfterLoss()
    r.start()
    time.sleep(ROTATION_DURATION + STOP_DURATION + 0.1)
    r.get_cmd()
    assert r.rotation_done == True

    r.reset()
    assert r.is_rotating_360 == False
    assert r.rotation_done == False
    assert r.rotation_start_time is None
    print('  PASS: reset 후 전체 상태 초기화')


# ----------------------------------------------------------------
# 메인
# ----------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 50)
    print('360도 회전 후 정지 단위 테스트')
    print(f'ROTATION_DURATION = {ROTATION_DURATION:.2f}s')
    print('=' * 50)

    test_start_state()
    test_rotating_phase()        # 6초 대기
    test_stopping_phase()        # 12.7초 대기
    test_done_phase()            # 13.2초 대기
    test_double_start()          # 0.1초 대기
    test_reset()                 # 13.2초 대기

    print('\n' + '=' * 50)
    print('전체 테스트 통과')
    print('=' * 50)
