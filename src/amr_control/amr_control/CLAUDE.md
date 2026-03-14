# 작업 컨텍스트

## 대상 파일
`amr1_tracking_aerial_v2.py` — TurtleBot4(robot4) 객체 추적 노드

## 핵심 구조
- Node: `TargetTracker`
- 구독: `occupation`(Bool) → 추적 on/off
- 동기화: RGB + Depth (`ApproximateTimeSynchronizer`)
- 발행: `/robot4/cmd_vel`(Twist), `tracking_done`(Bool)
- YOLO 클래스: `enemy` 만 필터링
- 제어: P제어 회전만 (linear.x = 0)

## 소실 상태 흐름
```
탐지 없음(첫 시작) → 정지
탐지 후 3초 경과(LOST_TIMEOUT) → 360° 스윕(~12.6s)
스윕 완료 → tracking_enabled=False, tracking_done 발행
```

## 리팩토링 시 주의
- 기능 동일하게 유지 (테스트 완료 코드)
- `frame_count % 3` 건드리면 타이밍 영향
- `decode_compressed_depth`의 12바이트 헤더 스킵은 ROS compressedDepth 포맷 특성 (건드리지 말 것)
- 토픽명 변경 시 launch 파일도 확인 필요

## 주요 수정 목록 (MARKCH.md 참조)
1. print → get_logger() 통일
2. 토픽 하드코딩(robot4) → ns 파라미터 활용
3. import tf2_geometry_msgs 상단으로 이동
4. CSV 리소스 정리 보강
5. _handle_lost 가독성 개선
6. frame_count % 3 파라미터화
7. 상수 파라미터화 고려
8. depth/TF 실패 시 제어 공백 → 탐지는 됐지만 회전 명령 없는 어중간한 상태 발생 (last_detected_time은 갱신됨)
9. _handle_lost 3초 미만 구간 오동작 → LOST_TIMEOUT 분기에 걸리지 않으면 마지막 else로 즉시 SCAN_SPEED 회전. 수정 방향: 3초 미만은 명령 미발행(마지막 명령 유지), 3초 이상부터 스캔 회전
