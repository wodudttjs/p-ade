# P-ADE (Physical AI Data Engine)

웹 비디오 자원을 자동 수집하여 로봇 모방학습용 (State, Action) 데이터셋으로 변환하는 End-to-End 파이프라인

## 🎯 프로젝트 개요

- **목표**: 웹에서 로봇팔/2족보행 동작 비디오를 자동 발견하고, 로봇이 모방학습 가능한 형태로 변환하여 클라우드에 저장
- **핵심 가치**: 데이터 부족 해결, 완전 자동화, 클라우드 네이티브 확장성, 24/7 무인 운영

## 🏗️ 시스템 아키텍처

```
                          ┌─── main.py (run-forever / systemd) ───┐
                          │                                        │
[Crawl] → [Download] → [Detect] → [Quality] → [Upload] → [Monitor/Alert]
   │          │           │           │           │            │
   ├── YouTube yt-dlp    ├── GPU     ├── 5-dim   ├── S3      ├── Dashboard
   ├── Google Videos     │  3-Stream │  scoring   │           ├── Slack/Email
   ├── Lambda Serverless │  CUDA     │  A~F 등급  │           └── Alert Rules
   └── Multi-source      └── CPU     └── Redis    └── SHA256
       (keyword expansion)  fallback    cache       dedup
```

## ✅ 구현 완료 기능 (v2.0.0)

### 🔍 1단계: 크롤링 (Crawl)
- **다국어 키워드 생성기** (`ingestion/keyword_generator.py`)
  - 영어/한국어/일본어/중국어/독일어 자동 키워드 생성
  - **카르테시안 곱 조합**: action(40) × robot(25) × object(30) × context(15) = 수만 키워드
  - **MultilingualExpander**: googletrans 기반 자동 번역 확장
  - **LongtailDiscovery**: YouTube autocomplete 기반 롱테일 키워드 탐색
- **멀티소스 크롤러** (`ingestion/multi_source_crawler.py`)
  - YouTube, Google Videos, Vimeo, Dailymotion 지원
  - 병렬 크롤링 (4 workers)
  - 레이트 리미터 및 재시도 매니저
- **AWS Lambda 서버리스 크롤러** (`lambda_/`)
  - `crawler_function.py`: Lambda 핸들러 (yt-dlp + YouTube API 폴백)
  - `invoke_lambda.py`: 로컬에서 배치 병렬 호출
  - `dynamodb_sync.py`: DynamoDB → 로컬 SQLite 동기화
- **멀티프로세스 워커** (`workers/crawl_worker.py`)
  - Redis 큐 기반 독립 워커 프로세스
  - `task_queue/task_queue.py`: CrawlTaskQueue, ProcessingQueue

### 📥 2단계: 다운로드 (Download)
- **병렬 다운로드** (`parallel_download.py`)
  - yt-dlp **Python API** 기반 고속 다운로드 (subprocess 대신 직접 호출)
  - 6 workers 병렬 처리
  - 720p 품질, 30초~20분 필터링
  - Deno JS 런타임 자동 연동 (YouTube 추출 지원)

### 🔍 3단계: 객체 검출 (Detect)
- **YOLO + MediaPipe 파이프라인** (`extraction/detect_to_episodes.py`)
  - 프레임 단위 객체 검출
  - 바운딩 박스, 신뢰도 점수 추출
  - 에피소드 단위 NPZ 저장
- **GPU 3-Stream 병렬 처리** (`gpu/stream_manager.py`)
  - CUDA Stream 활용 3개 영상 동시 처리
  - VRAM 자동 관리 (9GB 제한), CPU 폴백
  - 긴 영상 자동 FPS 조절 (>60초 → 15fps)

### 📊 4단계: 품질 평가 (Quality)
- **실시간 품질 평가** (`quality/evaluator.py`)
  - 5가지 메트릭: 관절 검출(30), 동작 자연스러움(25), 파지 동작(20), 안정성(15), 커버리지(10)
  - A~F 등급 분류 (pass threshold: 60점)
  - Redis 연동 실시간 통계 퍼블리싱
  - `QualityStats`: 통과율, 등급별 분포 추적

### 📦 5단계: 데이터 변환 (Transform)
- **모방학습 데이터 생성** (`build_imitation_data.py`)
  - MediaPipe Tasks API 기반 비디오 → 포즈 추출
  - 33개 관절 + 21개 손 랜드마크 추출
  - State-Action 인코딩 (state_dim=199, action_dim=100)
  - 그리퍼(손 오므림) 상태 자동 추정
  ```
  states:       [T, 199]    # 관절위치(99) + 속도(99) + 신뢰도(1)
  actions:      [T-1, 100]  # 관절 delta(99) + gripper(1)
  poses:        [T, 33, 3]  # 정규화된 관절 좌표
  left_hand:    [T, 21, 3]  # 왼손 랜드마크
  right_hand:   [T, 21, 3]  # 오른손 랜드마크
  gripper_state:[T]          # 그리퍼 상태 (0=열림, 1=닫힘)
  confidence:   [T]          # 포즈 검출 신뢰도
  ```

### ☁️ 6단계: 클라우드 업로드 (Upload)
- **AWS S3 업로드** (`upload_to_s3.py`)
  - 자동 버킷 경로: `s3://p-ade-datasets/episodes/YYYY/MM/DD/`
  - SHA256 중복 체크, 멱등성 보장
  - Multipart 업로드 지원

### 📊 7단계: 모니터링 & 알림 (Monitor)
- **웹 대시보드** (`dashboard/web_app.py`)
  - Flask 기반 실시간 웹 UI
  - 파이프라인 진행률 시각화, Start/Stop 제어
  - SSE 실시간 로그 스트리밍
  - DB 통계, Jobs/Quality/Settings 페이지
- **알림 시스템** (`alerts/`)
  - Slack + Email 알림 (AlertManager)
  - 8개 알림 규칙 (기본 4 + Task4 4)
  - 쿨다운, silence, force fire 지원
- **알림 모니터링 루프** (`monitor/alert_loop.py`)
  - GPU 사용률 < 30% → 경고
  - 큐 잔여 < 100개 → 경고
  - 실패율 > 40% → 에러
  - 18시 기준 일일 목표 < 40% → 에러

### 🔄 운영 자동화 (Operations)
- **메인 엔트리포인트** (`main.py`)
  - `run-once`: 단일 파이프라인 실행
  - `run-forever`: 무한 루프 (일일 목표 + 자정 리셋 + 자동 재시도)
  - `monitor-alerts`: 알림 모니터링 전용 모드
- **systemd 배포** (`deploy/`)
  - `robot-collector.service`: run-forever 서비스
  - `robot-alert-monitor.service`: 알림 모니터 서비스
  - `robot-collector.logrotate`: 로그 로테이션 (7일 보관)

## 📦 모듈 구조

```
p-ade/
├── main.py              # 🚀 통합 엔트리포인트 (run-once/run-forever/monitor-alerts)
├── mass_collector.py    # 전체 파이프라인 오케스트레이터
├── parallel_download.py # 병렬 다운로드 (yt-dlp Python API)
├── build_imitation_data.py # 모방학습 데이터 생성
├── upload_to_s3.py      # S3 업로드
│
├── ingestion/           # 크롤링 및 키워드
│   ├── keyword_generator.py  # 다국어 키워드 + 카르테시안 곱 + 롱테일
│   ├── multi_source_crawler.py  # 멀티소스 크롤러
│   ├── rate_limiter.py  # 레이트 리미터
│   └── downloader.py    # 다운로드 매니저
│
├── extraction/          # 객체 검출 및 에피소드 생성
│   ├── detect_to_episodes.py  # YOLO 검출 → NPZ
│   ├── object_detector.py     # YOLO 래퍼
│   └── pose_estimator.py      # MediaPipe 래퍼
│
├── quality/             # 품질 평가
│   └── evaluator.py     # 5-dim 품질 평가 + A~F 등급
│
├── gpu/                 # GPU 처리
│   └── stream_manager.py  # 3-Stream CUDA 병렬 처리
│
├── task_queue/          # 작업 큐
│   └── task_queue.py    # Redis 기반 CrawlTaskQueue / ProcessingQueue
│
├── workers/             # 멀티프로세스 워커
│   └── crawl_worker.py  # Redis 큐 기반 크롤링 워커
│
├── lambda_/             # AWS Lambda 서버리스 크롤러
│   ├── crawler_function.py  # Lambda 핸들러
│   ├── invoke_lambda.py     # 배치 호출
│   └── dynamodb_sync.py     # DynamoDB → SQLite 동기화
│
├── alerts/              # 알림 시스템
│   ├── manager.py       # AlertManager + Rules
│   ├── slack.py         # Slack 알림
│   └── email.py         # Email 알림
│
├── monitor/             # 모니터링
│   ├── stats_collector.py  # 시스템 메트릭 수집
│   └── alert_loop.py       # 알림 모니터링 루프 (4개 규칙)
│
├── dashboard/           # 대시보드
│   ├── web_app.py       # Flask 웹 대시보드 (SSE + 제어)
│   └── data_service.py  # DB 서비스
│
├── storage/             # 클라우드 저장소
│   └── s3_uploader.py   # S3 업로드
│
├── core/                # 공통 유틸리티
│   ├── database.py      # DB 연결
│   ├── logging_config.py  # loguru 설정
│   └── worker.py        # 워커 베이스
│
├── config/              # 설정
│   └── settings.py      # 환경변수 로드
│
├── deploy/              # 배포 설정
│   ├── robot-collector.service       # systemd 서비스
│   ├── robot-alert-monitor.service   # 알림 모니터 서비스
│   └── robot-collector.logrotate     # 로그 로테이션
│
├── tests/               # 테스트 (181개)
│   ├── test_task2_integration.py  # Task 2: 63개
│   ├── test_task3_integration.py  # Task 3: 64개
│   ├── test_task4_integration.py  # Task 4: 54개
│   └── ... (28개 테스트 파일)
│
├── data/                # 데이터 디렉토리
│   ├── raw/             # 다운로드된 mp4
│   ├── episodes/        # 생성된 npz
│   └── pade.db          # SQLite DB
│
└── requirements.txt     # 의존성
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# Deno 설치 (yt-dlp YouTube 추출에 필요)
# Windows:
irm https://deno.land/install.ps1 | iex
# Linux/Mac:
curl -fsSL https://deno.land/install.sh | sh
```

### 2. 환경설정

```bash
# .env 파일 생성 (.env.example 참고)
cp .env.example .env

# 필수 항목
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=p-ade-datasets
```

### 3. 전체 파이프라인 실행

```bash
# 단일 실행: 크롤링 → 다운로드 → 검출 → 업로드
python main.py run-once --target 500

# 무한 루프 (서버 운영용): 일일 500개 목표, 60초 간격
python main.py run-forever --target 500 --interval 60

# 단계별 실행
python mass_collector.py --target 100 --stage crawl
python mass_collector.py --target 100 --stage download
python mass_collector.py --target 100 --stage detect
python mass_collector.py --target 100 --stage upload

# 키워드 지정 실행
python mass_collector.py --target 10 --keywords "robot arm,pick and place"

# 드라이런 (실행 계획만 확인)
python mass_collector.py --target 500 --dry-run
```

### 4. 모방학습 데이터 생성

```bash
# 전체 비디오 → IL 데이터 변환
python build_imitation_data.py

# 10개만 테스트
python build_imitation_data.py --limit 10 --fps 5 --max-frames 50
```

### 5. 모니터링

```bash
# Flask 웹 대시보드 (http://localhost:5000)
python dashboard/web_app.py --port 5000

# 알림 모니터링 (Slack/Email)
python main.py monitor-alerts --interval 300
```

### 6. 서버 배포 (systemd)

```bash
# 서비스 파일 복사
sudo cp deploy/robot-collector.service /etc/systemd/system/
sudo cp deploy/robot-alert-monitor.service /etc/systemd/system/
sudo cp deploy/robot-collector.logrotate /etc/logrotate.d/

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl enable robot-collector robot-alert-monitor
sudo systemctl start robot-collector robot-alert-monitor

# 상태 확인
sudo systemctl status robot-collector
sudo journalctl -u robot-collector -f
```

## 🛠️ 기술 스택

| 영역 | 기술 |
|------|------|
| **언어** | Python 3.13+ |
| **크롤링** | yt-dlp (Python API), Scrapy, aiohttp, requests |
| **키워드** | 카르테시안 곱, googletrans, YouTube autocomplete |
| **비디오** | OpenCV, ffmpeg, Deno (JS runtime) |
| **AI/ML** | YOLOv8 (ultralytics), MediaPipe Tasks, PyTorch |
| **GPU** | CUDA 3-Stream 병렬, CPU 폴백 |
| **데이터** | NumPy, Polars, SciPy |
| **클라우드** | AWS S3 (boto3), Lambda, DynamoDB |
| **DB** | SQLite (SQLAlchemy) |
| **큐/캐시** | Redis, Celery |
| **웹 UI** | Flask, SSE, Bootstrap 5 |
| **알림** | Slack WebHook, SMTP Email |
| **모니터링** | loguru, psutil, AlertManager |
| **배포** | systemd, logrotate |
| **테스트** | pytest (181개 테스트) |

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

## 🔧 CLI 옵션

### main.py (통합 엔트리포인트)
```bash
# run-once: 단일 실행
python main.py run-once --target 500 --stage crawl

# run-forever: 무한 루프 (systemd용)
python main.py run-forever --target 500 --interval 60 --error-wait 300

# monitor-alerts: 알림 모니터링
python main.py monitor-alerts --interval 300
```

### mass_collector.py (파이프라인 오케스트레이터)
```bash
--target N          # 목표 영상 수 (기본: 500)
--stage STAGE       # 실행 단계: crawl/download/detect/upload
--keywords KW       # 키워드 (콤마 구분)
--sources SRC       # 소스: youtube,google_videos
--languages LANG    # 언어: en,ko,ja,zh
--quality Q         # 다운로드 품질: 360p/480p/720p/1080p
--workers N         # 병렬 워커 수
--dry-run           # 실행 계획만 출력
--detect-device DEV # 검출 디바이스: cuda/cpu
```

## 🧪 테스트

```bash
# 전체 테스트 실행 (181개)
pytest tests/ -v

# Task별 실행
pytest tests/test_task2_integration.py -v    # 63개 (멀티프로세스, GPU, 품질, 대시보드)
pytest tests/test_task3_integration.py -v    # 64개 (Lambda, 키워드 확장)
pytest tests/test_task4_integration.py -v    # 54개 (main.py, 알림, 배포)
```

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR 환영합니다!
