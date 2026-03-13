# app_v1.py
from __future__ import annotations

# import firebase_admin
# from firebase_admin import credentials, db

from flask import Flask, request, jsonify, render_template
from system_monitor.system_monitor.dummy.firebase_config import init_firebase, get_reference


# Initialize the Flask application
app = Flask(__name__)

# 1. Firebase Admin SDK 초기화
# 1단계 5번에서 다운로드한 json 키 파일 경로
# SERVICE_ACCOUNT_KEY_PATH = "/home/sinya/slam_turtlebot/src/system_monitor/config/defense-db_PrivateKey.json" 
# # 1단계 4번의 databaseURL 값 (Firebase 콘솔 -> Realtime Database에서 확인)
# DATABASE_URL = "https://defense-db-cea4d-default-rtdb.asia-southeast1.firebasedatabase.app"

# try:
#     cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': DATABASE_URL
#     })
#     print("[Firebase] Realtime Database 연결 성공!")
# except ValueError:
#     print("Firebase 앱이 이미 초기화되었습니다.")

try:
    init_firebase()
    print("Firebase 초기화 완료")
except Exception as e:
    print(f" Firebase 초기화 실패: {e}")
    raise

# Firebase reference -> stats_ref, violation_ref, detection_ref

stats_ref = get_reference("/gas_monitor/stats")
violation_ref = get_reference("/gas_monitor/violation")
detection_ref = get_reference("/gas_monitor/detection")

# 현재 상태 뽑기
def safe_get_stats() -> dict:
    try:
        stats = stats_ref.get()
        if not stats:
            return {
                "total": 0,
                "alert": 0,
                "last_update": None,
                "last_update_datetime": "-",
                "threshold": None,
            }

        return {
            "total": int(stats.get("total", 0)),
            "alert": int(stats.get("alert", 0)),
            "last_update": stats.get("last_update"),
            "last_update_datetime": stats.get("last_update_datetime", "-"),
            "threshold": stats.get("threshold"),
        }
    except Exception as e:
        print(f"stats 읽기 실패! : {e}")
        return {
            "total": 0,
            "alert": 0,
            "last_update": None,
            "last_update_datetime": "-",
            "threshold": None,
        }


def safe_get_violations(limit: int = 20) -> list[dict]:
    try:
        raw = violation_ref.get()
        if not raw:
            return []

        items = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue

            items.append({
                "key": key,
                "name": value.get("name", "-"),
                "ppm": value.get("ppm", "-"),
                "timestamp": value.get("timestamp", 0),
                "datetime": value.get("datetime", "-"),
            })

        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return items[:limit]

    except Exception as e:
        print(f" violation 읽기 실패: {e}")
        return []


# 본 web page(=root)
@app.route("/")
def index():
    return render_template('app.html')



@app.route("/api/dashboard")
def api_dashboard():
    '''
    메인 대시보드를 지정하는 함수
    stats = safe_get_stats
    violations = safe_get_violations
    '''
    stats = safe_get_stats()
    violations = safe_get_violations(limit=20)

    ## 디버깅용 테스트
    # print("stats =", stats)
    # print("count =", len(violations)) # 다 나옴. 아주 잘 나옴

    return jsonify({
        '''
        딕셔너리 데이터를 전부 json 타입으로 변환 
        '''
        "stats": stats,
        "violations": violations,
    })


@app.route("/api/stats")
def api_stats():
    return jsonify(safe_get_stats())


@app.route("/api/violations") 

def api_violations():
    '''
    violation condition 을 만족한 경우의 수일 때, 파이썬 딕셔너리에서 json 타입으로 변환할 데이터의 개수 제한시키는 함수
    '''
    return jsonify({
        "violations": safe_get_violations(limit=20)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



# app_v1.py
from __future__ import annotations

# Flask 기본 모듈들 import
from flask import Flask, request, jsonify, render_template 

# Firebase 관련 설정 (더미 모듈로 대체)
from system_monitor.system_monitor.dummy.firebase_config import init_firebase, get_reference
# 실제로는 firebase_admin과 credentials, db 모듈 필요하지만 더미로 대체함

# Flask 애플리케이션 인스턴스 생성
app = Flask(__name__)  # 현재 파일명을 기반으로 Flask 앱 초기화


# ================================
# 1. Firebase 연결 초기화 
# ================================
try:
    init_firebase()  # Firebase Admin SDK 초기화 (더미 함수 호출)
    print("Firebase 초기화 완료")  # 성공 로그 출력
except Exception as e:
    print(f"Firebase 초기화 실패: {e}")  # 에러 발생 시 상세 로그 출력
    raise  # 예외를 다시 발생시켜 앱 시작 차단


# Firebase 데이터베이스 참조 경로 설정
# 실시간 데이터베이스(Realtime Database)의 특정 경로를 참조하는 객체들
stats_ref = get_reference("/gas_monitor/stats")       # 가스 모니터링 통계 데이터 경로
violation_ref = get_reference("/gas_monitor/violation")  # 가스 농도 위반 기록 경로
detection_ref = get_reference("/gas_monitor/detection")   # 감지 이벤트 기록 경로 (미사용)


# ============================================================================
# 2. 안전한 데이터 조회 헬퍼 함수들
# ============================================================================
def safe_get_stats() -> dict:
    """
    Firebase stats 경로에서 최신 통계 데이터를 안전하게 조회
    
    Returns:
        dict: 통계 데이터 (total, alert, last_update, threshold 등)
    """
    try:
        # Firebase에서 stats 데이터 전체 조회
        stats = stats_ref.get()
        if not stats:  # 데이터가 없으면 기본값 반환
            return {
                "total": 0,                    # 총 감지 횟수
                "alert": 0,                    # 경고 발생 횟수
                "last_update": None,           # 마지막 업데이트 타임스탬프
                "last_update_datetime": "-",   # 읽기 쉬운 날짜/시간 문자열
                "threshold": None,             # 경고 기준값 (ppm)
            }

        # Firebase 데이터에서 필요한 필드만 추출 (안전한 기본값 처리)
        return {
            "total": int(stats.get("total", 0)),           # 문자열->정수 변환
            "alert": int(stats.get("alert", 0)),           # 문자열->정수 변환
            "last_update": stats.get("last_update"),       # 타임스탬프
            "last_update_datetime": stats.get("last_update_datetime", "-"),
            "threshold": stats.get("threshold"),           # 경고 기준 ppm 값
        }
    except Exception as e:
        # 모든 예외 상황에서 기본값 반환 (웹 요청이 실패하더라도 서버는 계속 동작)
        print(f"stats 읽기 실패! : {e}")
        return {
            "total": 0,
            "alert": 0,
            "last_update": None,
            "last_update_datetime": "-",
            "threshold": None,
        }


def safe_get_violations(limit: int = 20) -> list[dict]:
    """
    Firebase violation 경로에서 최근 위반 기록 목록 조회
    
    Args:
        limit: 반환할 최대 항목 수 (기본값: 20)
    
    Returns:
        list[dict]: 위반 기록 리스트 (key, name, ppm, timestamp, datetime)
    """
    try:
        # Firebase에서 violation 데이터 전체 조회
        raw = violation_ref.get()
        if not raw:  # 데이터가 없으면 빈 리스트 반환
            return []

        # 각 위반 항목을 파이썬 딕셔너리로 변환
        items = []
        for key, value in raw.items():
            # value가 딕셔너리가 아닌 경우 건너뛰기 (데이터 무결성 검증)
            if not isinstance(value, dict):
                continue

            # 필요한 필드만 안전하게 추출
            items.append({
                "key": key,                        # Firebase 고유 키
                "name": value.get("name", "-"),    # 센서/위치 이름
                "ppm": value.get("ppm", "-"),      # 가스 농도 (parts per million)
                "timestamp": value.get("timestamp", 0),  # 유닉스 타임스탬프
                "datetime": value.get("datetime", "-"),  # 읽기 쉬운 날짜/시간
            })

        # 타임스탬프 기준 내림차순 정렬 (최신순)
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        # 최근 limit개만 반환 (성능 최적화)
        return items[:limit]

    except Exception as e:
        # 모든 예외 상황에서 빈 리스트 반환
        print(f"violation 읽기 실패: {e}")
        return []


# ============================================================================
# 3. 웹 라우트 (URL 경로) 정의
# ============================================================================

# 메인 페이지 (HTML 템플릿 렌더링)
@app.route("/")
def index():
    """
    루트 경로(/) 요청 시 웹 대시보드 HTML 페이지 반환
    """
    return render_template('app.html')  # templates/app.html 파일 렌더링


# 대시보드 API (통합 데이터)
@app.route("/api/dashboard")
def api_dashboard():
    """
    메인 대시보드용 통합 API
    - 통계 데이터 + 최근 위반 기록을 한 번에 제공
    - 프론트엔드에서 한 번의 요청으로 모든 데이터 획득 가능
    """
    # 헬퍼 함수로 안전하게 데이터 조회
    stats = safe_get_stats()           # 현재 통계 상태
    violations = safe_get_violations(limit=20)  # 최근 20개 위반 기록

    # 디버깅용 (배포 시 주석 처리 권장)
    # print("stats =", stats)
    # print("count =", len(violations))

    # JSON 응답 생성 (Flask jsonify가 자동으로 Content-Type: application/json 설정)
    return jsonify({
        "stats": stats,        # 통계 데이터
        "violations": violations,  # 위반 기록 리스트
    })


# 통계 전용 API
@app.route("/api/stats")
def api_stats():
    """
    통계 데이터만 제공하는 전용 API
    - 실시간 차트 업데이트 등에 사용
    """
    return jsonify(safe_get_stats())  # 간단히 통계 데이터만 JSON으로 반환


# 위반 기록 전용 API
@app.route("/api/violations")
def api_violations():
    """
    위반 기록 목록만 제공하는 전용 API
    - 페이지네이션, 필터링 등에 유용
    - 최대 20개 최근 기록만 반환 (성능 최적화)
    """
    return jsonify({
        "violations": safe_get_violations(limit=20)  # violations 키로 감싸서 반환
    })


# ============================================================================
# 4. 애플리케이션 실행
# ============================================================================
if __name__ == "__main__":
    """
    직접 실행 시 Flask 개발 서버 시작
    - host="0.0.0.0": 외부 네트워크에서 접근 가능
    - port=5000: 기본 포트
    - debug=True: 개발 모드 (코드 변경 시 자동 재시작, 에러 페이지 상세 표시)
    """
    app.run(host="0.0.0.0", port=5000, debug=True)
