# fb_config.py
from __future__ import annotations

import os

import firebase_admin
from firebase_admin import credentials, db

# 1. Firebase Admin SDK 초기화
# 1단계 5번에서 다운로드한 json 키 파일 경로
SERVICE_ACCOUNT_KEY_PATH = "/home/sinya/slam_turtlebot/src/system_monitor/config/defense-db_PrivateKey.json" 
# 1단계 4번의 databaseURL 값 (Firebase 콘솔 -> Realtime Database에서 확인)
DATABASE_URL = "https://defense-db-cea4d-default-rtdb.asia-southeast1.firebasedatabase.app"


# try:
#     cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': DATABASE_URL
#     })
#     print("[Firebase] Realtime Database 연결 성공!")
# except ValueError:
#     print("Firebase 앱이 이미 초기화되었습니다.")


def init_firebase():
    try:
        app = firebase_admin.get_app()
        print("초기화되어있는 앱 >> 재사용")
        return app
    except ValueError:
        pass

    if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        raise FileNotFoundError(
            f"{SERVICE_ACCOUNT_KEY_PATH} 없음"
        )

    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    app = firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })
    print(" FB Realtime DB 연결 완")
    return app


def get_reference(path: str):
    if not path:
        raise ValueError("No reference from firebase.")

    normalized_path = path if path.startswith("/") else f"/{path}"
    return db.reference(normalized_path)