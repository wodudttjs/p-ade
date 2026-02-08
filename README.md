# P-ADE (Physical AI Data Engine)

웹 비디오 자원을 자동 수집하여 로봇 모방학습용 (State, Action) 데이터셋으로 변환하는 End-to-End 파이프라인

## 🎯 프로젝트 개요

- **목표**: 웹에서 로봇팔/2족보행 동작 비디오를 자동 발견하고, 로봇이 모방학습 가능한 형태로 변환하여 클라우드에 저장
- **핵심 가치**: 데이터 부족 해결, 완전 자동화, 클라우드 네이티브 확장성

## 🏗️ 시스템 아키텍처

```
[Crawl] → [Download] → [Detect] → [Transform] → [Upload] → [Monitor]
   │          │           │            │           │          │
   ├── YouTube/Google    ├── yt-dlp   ├── YOLO   ├── NPZ    ├── S3
   └── Multi-source      └── Parallel └── MediaPipe         └── Dashboard
```

## ✅ 구현 완료 기능 (v1.0.0)

### 🔍 1단계: 크롤링 (Crawl)
- **다국어 키워드 생성기** (`ingestion/keyword_generator.py`)
  - 영어/한국어/일본어/중국어 자동 키워드 생성
  - 로봇팔, 2족보행, 매니퓰레이터 등 카테고리별 키워드
- **멀티소스 크롤러** (`ingestion/multi_source_crawler.py`)
  - YouTube, Google Videos, Vimeo, Dailymotion 지원
  - 병렬 크롤링 (4 workers)
  - 레이트 리미터 및 재시도 매니저

### 📥 2단계: 다운로드 (Download)
- **병렬 다운로드** (`parallel_download.py`)
  - yt-dlp 기반 고속 다운로드
  - 6 workers 병렬 처리
  - 720p 품질, 30초~20분 필터링

### 🔍 3단계: 객체 검출 (Detect)
- **YOLO + MediaPipe 파이프라인** (`extraction/detect_to_episodes.py`)
  - 프레임 단위 객체 검출
  - 바운딩 박스, 신뢰도 점수 추출
  - 에피소드 단위 NPZ 저장

### 📦 4단계: 데이터 변환 (Transform)
- **모방학습 데이터 생성** (`build_imitation_data.py`) - **⭐ NEW!**
  - MediaPipe Tasks API 기반 비디오 → 포즈 추출
  - 33개 관절 + 21개 손 랜드마크 추출
  - State-Action 인코딩 (state_dim=199, action_dim=100)
  - 그리퍼(손 오므림) 상태 자동 추정
  - 관절 속도(velocity), 정규화, 중앙차분 계산
  ```
  states:       [T, 199]    # 관절위치(99) + 속도(99) + 신뢰도(1)
  actions:      [T-1, 100]  # 관절 delta(99) + gripper(1)
  poses:        [T, 33, 3]  # 정규화된 관절 좌표
  velocity:     [T, 33, 3]  # 관절 속도
  left_hand:    [T, 21, 3]  # 왼손 랜드마크
  right_hand:   [T, 21, 3]  # 오른손 랜드마크
  gripper_state:[T]          # 그리퍼 상태 (0=열림, 1=닫힘)
  confidence:   [T]          # 포즈 검출 신뢰도
  ```
- **레거시 객체 검출** (`extraction/detect_to_episodes.py`)
  - YOLO 프레임 단위 객체 검출
  - 바운딩 박스, 신뢰도 점수 추출

### ☁️ 5단계: 클라우드 업로드 (Upload)
- **AWS S3 업로드** (`upload_to_s3.py`)
  - 자동 버킷 경로: `s3://p-ade-datasets/episodes/YYYY/MM/DD/`
  - SHA256 중복 체크, 멱등성 보장
  - Multipart 업로드 지원

### 📊 6단계: 모니터링 (Monitor)
- **웹 대시보드** (`dashboard/web_app.py`) - **⭐ NEW!**
  - Flask 기반 실시간 웹 UI
  - 파이프라인 진행률 시각화
  - Start/Stop 제어
  - DB 통계, 로그 스트리밍
  - **IL Data 페이지**: 모방학습 데이터 현황/품질 시각화
- **데스크톱 대시보드** (`dashboard/app.py`)
  - PySide6 기반 GUI (레거시)

## 📦 모듈 구조

```
p-ade-master/
├── ingestion/           # 크롤링 및 다운로드
│   ├── keyword_generator.py  # 다국어 키워드 생성
│   ├── multi_source_crawler.py  # 멀티소스 크롤러
│   ├── rate_limiter.py  # 레이트 리미터
│   └── downloader.py    # 다운로드 매니저
├── extraction/          # 객체 검출 및 에피소드 생성
│   ├── detect_to_episodes.py  # YOLO 검출 → NPZ
│   ├── object_detector.py  # YOLO 래퍼
│   └── pose_estimator.py  # MediaPipe 래퍼
├── storage/             # 클라우드 저장소
│   └── providers/s3_provider.py  # S3 업로드
├── dashboard/           # 대시보드
│   ├── web_app.py       # Flask 웹 대시보드 ⭐
│   ├── app.py           # PySide6 데스크톱 앱
│   └── data_service.py  # DB 서비스
├── models/              # SQLAlchemy 모델 + MediaPipe 모델
│   └── mediapipe/       # pose_landmarker.task, hand_landmarker.task
├── core/                # 공통 유틸리티
├── config/              # 설정
├── tests/               # 테스트
├── data/                # 데이터 디렉토리
│   ├── raw/             # 다운로드된 mp4
│   ├── episodes/        # 생성된 npz
│   └── pade.db          # SQLite DB
├── mass_collector.py    # 전체 파이프라인 오케스트레이터 ⭐
├── build_imitation_data.py # 모방학습 데이터 생성 ⭐
├── parallel_download.py # 병렬 다운로드
├── upload_to_s3.py      # S3 업로드
└── requirements.txt     # 의존성
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경설정

```bash
# AWS S3 (선택: 기본값 내장)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=ap-northeast-2
```

### 3. 전체 파이프라인 실행 (권장)

```bash
# 500개 영상 수집 (크롤링 → 다운로드 → 검출 → 업로드)
python mass_collector.py --target 500

# 단계별 실행
python mass_collector.py --target 100 --stage crawl
python mass_collector.py --target 100 --stage download
python mass_collector.py --target 100 --stage detect
python mass_collector.py --target 100 --stage upload

# 드라이런 (실행 계획만 확인)
python mass_collector.py --target 500 --dry-run
```

### 3-1. 모방학습 데이터 생성

```bash
# 전체 비디오 → IL 데이터 변환 (이미 있으면 스킵)
python build_imitation_data.py

# 10개만 테스트
python build_imitation_data.py --limit 10 --fps 5 --max-frames 50

# 옵션
python build_imitation_data.py --fps 10 --max-frames 200 --limit 100
```

### 4. 웹 대시보드 실행

```bash
# Flask 웹 대시보드 (http://localhost:5000)
python dashboard/web_app.py --port 5000

# 또는 데스크톱 GUI (PySide6)
python run_dashboard.py
```

## 📊 실행 결과 (2026-02-08 최종)

| 단계 | 결과 |
|------|------|
| 크롤링 | 467개 URL 수집 |
| 다운로드 | 466개 mp4 (720p) |
| 객체 검출 | 466개 에피소드 (.npz) |
| **모방학습 데이터** | **464개 IL 에피소드 (states/actions/poses)** ✅ |
| S3 업로드 | 454개 전량 업로드 완료 |

### 🤖 모방학습 데이터 품질 (IL Data) — 최종

| 항목 | 값 |
|------|-----|
| IL 에피소드 수 | **464** |
| State 차원 | 199 (관절99 + 속도99 + 신뢰도1) |
| Action 차원 | 100 (관절delta99 + gripper1) |
| 총 프레임 수 | **46,005** |
| 총 액션 수 | **45,541** |
| 평균 Confidence | 0.2387 |
| 평균 Gripper | 0.5075 |
| 고품질 (conf>0.3) | **158/464 (34%)** |
| 학습가능 (conf>0.1) | **333/464 (72%)** |
| 손 검출 비율 | **186/464 (40%)** |
| NaN/Inf | **0건** ✅ |
| 디스크 용량 | **45.4 MB** |

**S3 경로**: `s3://p-ade-datasets/episodes/2026/02/08/`

## 🛠️ 기술 스택

| 영역 | 기술 |
|------|------|
| **언어** | Python 3.10+ |
| **크롤링** | yt-dlp, requests, concurrent.futures |
| **비디오** | OpenCV, ffmpeg |
| **AI/ML** | YOLOv8 (ultralytics), MediaPipe Tasks API |
| **데이터** | NumPy, Pandas |
| **클라우드** | AWS S3 (boto3) |
| **DB** | SQLite (SQLAlchemy) |
| **웹 UI** | Flask, Bootstrap 5 |
| **데스크톱 UI** | PySide6 (Qt) |

## 📁 데이터 포맷

### 모방학습 Episode NPZ 구조 (IL Data)
```python
import numpy as np
data = np.load('episode.npz', allow_pickle=True)

# 핵심 모방학습 데이터
data['states']        # [T, 199]   - 관절위치(99) + 속도(99) + 신뢰도(1)
data['actions']       # [T-1, 100] - 관절 delta(99) + gripper(1)

# 포즈 데이터
data['poses']         # [T, 33, 3] - 정규화된 관절 좌표 (hip 중심, 어깨너비 스케일)
data['poses_raw']     # [T, 33, 3] - 원시 관절 좌표
data['poses_world']   # [T, 33, 3] - 월드 좌표계

# 손 & 그리퍼
data['left_hand']     # [T, 21, 3] - 왼손 랜드마크
data['right_hand']    # [T, 21, 3] - 오른손 랜드마크
data['gripper_state'] # [T]        - 그리퍼 상태 (0=열림, 1=닫힘)

# 메타
data['velocity']      # [T, 33, 3] - 관절 속도 (중앙차분)
data['timestamps']    # [T]        - 타임스탬프 (초)
data['confidence']    # [T]        - 포즈 검출 신뢰도
data['video_id']      # str        - 원본 비디오 ID
data['fps']           # float      - 추출 FPS
```

### 빠른 사용 예시
```python
# 모방학습 학습 루프
data = np.load('data/episodes/video_episode.npz', allow_pickle=True)
states = data['states']    # [T, 199]
actions = data['actions']  # [T-1, 100]

for t in range(len(actions)):
    state = states[t]      # 현재 상태
    action = actions[t]    # 취해야 할 행동
    next_state = states[t+1]  # 다음 상태
    # policy.train(state, action)
```

## 🔧 설정 옵션

### mass_collector.py
```bash
--target N          # 목표 영상 수 (기본: 100)
--stage STAGE       # 실행 단계: crawl/download/detect/upload/all
--sources SRC       # 소스: youtube,google_videos (기본: youtube,google_videos)
--languages LANG    # 언어: en,ko,ja,zh (기본: en,ko)
--quality Q         # 다운로드 품질: 360p/480p/720p/1080p
--workers N         # 병렬 워커 수
--dry-run           # 실행 계획만 출력
--resume            # 이어서 실행
```

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR 환영합니다!
