# 방공 로키
군 내의 방공 진지 저궤도 적군기 탐지 및 대응 시스템 프로젝트 

> 프로젝트 기간 : 2026. 03. 06 ~ 2026. 03. 16

Turtlebot4 두 대를 활용하여 군 방공 진지에서 저궤도 적군기를 탐지하고 대응을 수행하는 자율 로봇 시스템입니다. SLAM 기반 자율주행, Computer Vision 기반 객체 탐지, ROS2 통신, IoT 센서 및 Firebase를 통한 실시간 모니터링을 포함합니다.


![Turtlebot4](https://img.shields.io/badge/Turtlebot4-00599C?style=for-the-badge&logo=ros&logoColor=white) ![Nav2](https://img.shields.io/badge/Nav2-Navigation-0A66C2?style=for-the-badge)![SLAM](https://img.shields.io/badge/SLAM-Autonomous%20Mapping-blue?style=for-the-badge) ![ROS2](https://img.shields.io/badge/ROS2-22314E?style=for-the-badge&logo=ros&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-FF6F00?style=for-the-badge)![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-111111?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)![Arduino](https://img.shields.io/badge/Arduino-00979D?style=for-the-badge&logo=arduino&logoColor=white)![Firebase](https://img.shields.io/badge/Firebase-DB-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

---

## System Overview

본 시스템은 멀티 로봇 구조로 구성되어 있으며 각 로봇은 서로 다른 역할을 수행합니다.

##### AMR 1

- 저고도 적군기 탐지

- Computer Vision 기반 객체 탐지

- 타겟 위치 추적

##### AMR 2

- 위험 지역 이동 및 순찰

- 아두이노 센서 기반 기체 이상 감지

- 관리자에게 상태 보고 및 알림 시스템 가동

</br>

## System Architecture

> 첨부 예정

## Key Features

1. **Multi-Robot System**
: 두 대의 Turtlebot4가 서로 다른 역할을 수행하며 협력하여 작동합니다.

2. **Computer Vision Object Detection**
: Ultralytics YOLO 모델을 이용해 카메라 영상에서 적군기 및 무동력 비행물체 (새, 풍선) 탐지합니다.

3. **Autonomous Robot Navigation**
: SLAM과 Nav2를 이용한 자율 주행을 수행합니다.

4. **Real-Time Monitoring**
: Firebase Realtime Database를 이용하여 시스템 상태를 실시간으로 모니터링합니다.

5. **Web Dashboard**
: 웹 UI를 통해 로봇 상태 및 이벤트를 확인할 수 있습니다.

 </br>

## Project Directory

```
.slam_turtlebot
│
├── detection (CV Object Detection 관련 파일)
│   ├── obj_det_amr2.py
│   ├── obj_det_amr2_time_sync.py
│   ├── amr1_depth_aligned_dual.py
│   ├── amr1_observe_v2_detect.py
│
├── system_monitor (Web UI 모니터링 관련 파일)
│   ├── UI_bridge.py
│   ├── UI_flask.py
│   ├── UI.html
|   ├── UI_command.py
|   ├── webcam_classifier.py
│
├── amr_control (AMR 컨트롤 관련 파일)
│   ├── amr1_moveout_follow_waypoints.py
│   ├── amr1_tracking_aerial.py
│   ├── amr1_pullout.py
│   ├── amr2_move1.py
|   
├── amr_interfaces (ROS2 사용자 인터페이스)
│   ├── msg
|   │   ├── TargetEvent.msg
|
├── models (YOLO 학습 모델 .pt 관련 파일)
│   ├── amr1
|   │   ├── yolo11n_arm1__v2.pt
|   ├── amr2
|   │   ├── yolo26n.pt
|   |   ├── resnet18.pth
|   ├── webcam
|   │   ├── 
|
│
└── README.md
```