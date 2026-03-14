# amr1_tracking_aerial_v2.py 작업 노트

## 하려는 작업
`amr1_tracking_aerial_v2.py` 리팩토링 (기능 동일, 코드 품질 개선)

---

## 주요 흐름

```
[occupation=True 수신]
        ↓
[synced_callback] — 3프레임에 1번 처리
  RGB(CompressedImage) + Depth(compressedDepth) 동기화 수신
        ↓
  이미지 디코딩
  - RGB: cv2.imdecode (JPEG)
  - Depth: decode_compressed_depth() → 12바이트 헤더 제거 후 PNG decode
        ↓
  YOLO 추론 → 'enemy' 클래스 중 conf 최고 박스 선택
        ↓
  [탐지 성공]                         [탐지 실패]
  박스 중심 픽셀 (u, v)                _handle_lost()
        ↓                              ↓
  pixel_to_3d (카메라 내부 파라미터)   1단계: 탐지된 적 없음 → 정지
        ↓                              2단계: 3초 경과 → 360° 스윕 시작
  camera_point_to_map (TF → map)       3단계: 스윕 완료 → 정지 + tracking_done 발행
        ↓
  CSV 로그 기록
        ↓
  compute_angular_z → publish_rotation
  (P제어, kp=0.8, 데드밴드 5%, min/max 클램핑)
```

---

## 리팩토링 주요 포인트

| # | 항목 | 현황 | 방향 |
|---|------|------|------|
| 1 | `print` / `get_logger()` 혼용 | 디버그 출력이 혼재 | 전부 `get_logger()`로 통일 |
| 2 | 토픽 하드코딩 | `robot4` 네임스페이스 직접 박힘, `ns` 변수 선언만 하고 미사용 | `ns` 또는 파라미터로 통일 |
| 3 | `import tf2_geometry_msgs` | `camera_point_to_map` 내부에서 매번 import | 상단으로 이동 |
| 4 | CSV 파일 관리 | 리소스 정리가 `finally` 블록에만 의존 | `destroy_node()` 오버라이드 또는 try/finally 보강 |
| 5 | `_handle_lost` 분기 순서 | `final_sweep_start` → `last_detected_time is None` → timeout 순으로 의존성 있음 | 상태머신으로 명확화 고려 |
| 6 | 프레임 스킵 하드코딩 | `frame_count % 3` 고정 | 파라미터화 |
| 7 | 상수 선언 위치 | `SCAN_SPEED`, `LOST_TIMEOUT` 등 모듈 레벨에 있음 | 클래스 상수 또는 ROS 파라미터로 이동 고려 |
| 8 | depth/TF 실패 시 제어 공백 | enemy 탐지 후 `pixel_to_3d` 또는 `camera_point_to_map` 실패 시 `return`만 하고 회전 명령 없음 | 실패해도 픽셀 기반 회전은 유지하거나, 명시적으로 이전 명령 유지/정지 처리 |
| 9 | 탐지 실패 1프레임부터 즉시 SCAN_SPEED 회전 | `_handle_lost`에서 3초 미만 구간도 마지막 분기(`publish_rotation(SCAN_SPEED)`)로 빠짐. 직전 프레임에 정지 명령이었어도 다음 프레임 탐지 실패 시 즉시 0.5 rad/s 전환 | 3초 미만 구간은 별도 명령 없이 마지막 명령 유지. 3초 경과 시에만 SCAN_SPEED 회전 진입 |
