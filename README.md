# P-ADE (Physical AI Data Engine)

웹 비디오 자원을 자동 수집하여 로봇 학습용 (State, Action) 데이터셋으로 변환하는 End-to-End 파이프라인

## 🎯 프로젝트 개요

- **목표**: 웹에서 로봇팔/2족보행 동작 비디오를 자동 발견하고, 로봇이 모방학습 가능한 형태로 변환하여 클라우드에 저장
- **핵심 가치**: 데이터 부족 해결, 완전 자동화, 클라우드 네이티브 확장성, 24/7 무인 운영

## 🏗️ 시스템 아키텍처

```
              ┌─── main.py serve (Flask + 백그라운드 파이프라인) ───┐
              │                                                      │
[Crawl] → [Download] → [Detect] → [Build IL] → [Quality] → [Upload]
   │          │            │           │            │           │
   ├── YouTube          ├── GPU       ├── MediaPipe ├── 5-dim  ├── S3
   ├── Google Videos    │  3-Stream   │  Tasks API  │  scoring  │  SHA256
   ├── Redis 캐시       │  CUDA       │  States +   │  A~F 등급 │  dedup
   └── Multi-source     └── CPU       │  Actions    └── Redis   └── Multipart
       (keyword expansion)  fallback   └── NPZ 저장    cache

   ※ 실패한 단계가 있어도 기존 데이터로 다음 단계 계속 진행
   ※ 서버가 켜져 있으면 30초 간격으로 무한 반복 실행
```

## ✅ 구현 완료 기능 (v2.1.0)

### 🔍 1단계: 크롤링 (Crawl)
- **다국어 키워드 생성기** (`ingestion/keyword_generator.py`)
  - 영어/한국어/일본어/중국어/독일어 자동 키워드 생성
  - **카르테시안 곱 조합**: action × robot × object × context = 수만 키워드
  - **MultilingualExpander**: googletrans 기반 자동 번역 확장
  - **LongtailDiscovery**: YouTube autocomplete 기반 롱테일 키워드 탐색
- **멀티소스 크롤러** (`ingestion/multi_source_crawler.py`)
  - YouTube, Google Videos 지원
  - 병렬 크롤링 (4 workers)
  - **Redis 캐시 연동**: 키워드별 검색 결과 캐싱 → 반복 크롤링 시 즉시 응답
  - 레이트 리미터 및 재시도 매니저
- **멀티프로세스 워커** (`workers/crawl_worker.py`)
  - Redis 큐 기반 독립 워커 프로세스
  - `task_queue/task_queue.py`: CrawlTaskQueue, ProcessingQueue

### 📥 2단계: 다운로드 (Download)
- **병렬 다운로드** (`scripts/pipeline/parallel_download.py`)
  - yt-dlp **Python API** 기반 고속 다운로드
  - 6 workers 병렬 처리
  - 720p 품질, 30초~20분 필터링

### 🔍 3단계: 객체 검출 (Detect)
- **YOLO + MediaPipe 파이프라인** (`extraction/detect_to_episodes.py`)
  - YOLOv8 기반 프레임 단위 객체 검출 (**GPU cuda:0 사용**)
  - 바운딩 박스, 신뢰도 점수 추출
  - 에피소드 단위 NPZ 저장
- **GPU 3-Stream 병렬 처리** (`gpu/stream_manager.py`)
  - CUDA Stream 활용 3개 영상 동시 처리
  - VRAM 자동 관리 (9GB 제한), CPU 폴백
  - 긴 영상 자동 FPS 조절 (>60초 → 15fps)

### 📦 4단계: 모방학습 데이터 생성 (Build IL)
- **모방학습 데이터 생성** (`scripts/pipeline/build_imitation_data.py`)
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

### 📊 5단계: 품질 평가 (Quality)
- **실시간 품질 평가** (`quality/evaluator.py`)
  - 5가지 메트릭: 관절 검출(30점), 동작 품질(25점), 파지 동작(20점), 안정성(15점), 커버리지(10점)
  - A~F 등급 분류 (pass threshold: 60점)
  - 실제 NPZ 구조에 맞춤 최적화 (`evaluate_npz`, `_npz_to_sequence`)
  - `QualityStats`: 통과율, 등급별 분포 추적

### ☁️ 6단계: 클라우드 업로드 (Upload)
- **AWS S3 업로드** (`scripts/pipeline/upload_to_s3.py`)
  - 자동 버킷 경로: `s3://p-ade-datasets/episodes/YYYY/MM/DD/`
  - SHA256 중복 체크, 멱등성 보장
  - Multipart 업로드 지원

### 📊 웹 대시보드 & 모니터링
- **웹 대시보드** (`dashboard/web_app.py`)
  - Flask 기반 실시간 웹 UI (http://0.0.0.0:5000)
  - 파이프라인 진행률 시각화, Start/Stop 제어
  - **실시간 로그 스트리밍**
  - **GPU/VRAM 사용량 모니터링**
  - **Redis 캐시 통계**
  - DB 통계, Jobs/Quality/Videos/Episodes/IL Data 페이지
  - **작업 히스토리 영속화** (`data/jobs_history.json`) — 재시작 후에도 유지
- **알림 시스템** (`alerts/`)
  - Slack + Email 알림 (AlertManager)
  - GPU, 큐, 실패율 기반 알림 규칙

### 🔄 운영 자동화
- **메인 엔트리포인트** (`main.py`)
  - `serve`: 웹 대시보드 + 백그라운드 파이프라인 자동 반복 (**기본 모드**)
  - `run-once`: 단일 파이프라인 실행
  - `run-forever`: CLI만으로 무한 루프
- **파이프라인 복원력**: 단계 실패 시에도 기존 데이터로 다음 단계 계속 진행
- **무한 반복**: 서버가 켜져 있으면 30초 간격으로 전체 파이프라인 계속 순환

## 📦 모듈 구조

```
p-ade-master/
├── main.py                    # 🚀 통합 엔트리포인트 (serve/run-once/run-forever)
├── mass_collector.py          # 파이프라인 오케스트레이터 (6단계)
│
├── scripts/                   # 파이프라인 스크립트
│   └── pipeline/
│       ├── parallel_download.py   # 병렬 다운로드 (yt-dlp)
│       ├── build_imitation_data.py # 모방학습 데이터 생성
│       ├── upload_to_s3.py        # S3 업로드
│       ├── collect_youtube.py     # YouTube 수집
│       ├── encode_actions.py      # 액션 인코딩
│       ├── extract_poses.py       # 포즈 추출
│       ├── filter_quality.py      # 품질 필터
│       └── segment_episodes.py    # 에피소드 분할
│
├── ingestion/                 # 크롤링 및 키워드
│   ├── keyword_generator.py       # 다국어 키워드 생성
│   ├── multi_source_crawler.py    # 멀티소스 크롤러 + Redis 캐시
│   └── downloader.py              # 다운로드 매니저
│
├── extraction/                # 객체 검출 및 에피소드 생성
│   ├── detect_to_episodes.py      # YOLO 검출 → NPZ (GPU)
│   ├── object_detector.py         # YOLOv8 래퍼
│   └── pose_estimator.py          # MediaPipe 래퍼
│
├── quality/                   # 품질 평가
│   └── evaluator.py               # 5-dim 품질 평가 + A~F 등급
│
├── gpu/                       # GPU 처리
│   └── stream_manager.py          # 3-Stream CUDA 병렬 처리
│
├── cache/                     # 캐시
│   └── redis_cache.py             # Redis 캐시 클라이언트
│
├── task_queue/                # 작업 큐
│   └── task_queue.py              # Redis 기반 작업 큐
│
├── workers/                   # 멀티프로세스 워커
│   └── crawl_worker.py            # 크롤링 워커
│
├── dashboard/                 # 웹 대시보드
│   └── web_app.py                 # Flask 대시보드 (실시간 모니터링)
│
├── alerts/                    # 알림 시스템
│   ├── manager.py                 # AlertManager
│   ├── slack.py                   # Slack 알림
│   └── email.py                   # Email 알림
│
├── storage/                   # 클라우드 저장소
│   └── s3_uploader.py             # S3 업로드
│
├── core/                      # 공통 유틸리티
│   ├── database.py                # DB 연결
│   ├── logging_config.py          # loguru 설정
│   └── worker.py                  # 워커 베이스
│
├── config/                    # 설정
│   └── settings.py                # 환경변수 로드
│
├── tests/                     # 테스트
│   └── ... (28개 테스트 파일)
│
├── deploy/                    # 배포 설정
│   ├── docker-compose.yml         # Docker Compose
│   ├── Dockerfile                 # 컨테이너 이미지
│   └── redis.conf                 # Redis 설정
│
├── data/                      # 데이터 디렉토리
│   ├── raw/                       # 다운로드된 mp4
│   ├── episodes/                  # 생성된 npz (검출 + IL 데이터)
│   └── jobs_history.json          # 작업 히스토리 (영속화)
│
└── requirements.txt           # 의존성
```

## 🚀 빠른 시작

### 설치

```bash
# 패키지 설치
pip install -r requirements.txt

# PostgreSQL 설치 & 설정
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo -u postgres psql -c "CREATE USER pade WITH PASSWORD 'pade' CREATEDB;"
sudo -u postgres psql -c "CREATE DATABASE pade OWNER pade;"

# Redis 설치 (캐싱/큐)
sudo apt install -y redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

### 환경설정

```bash
# .env 파일 생성 (.env.example 참고)
cp .env.example .env

# 필수: PostgreSQL (기본값으로 동작)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=pade
POSTGRES_USER=pade
POSTGRES_PASSWORD=pade

# 필수: Redis (기본값으로 동작)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# 선택: AWS S3 업로드 시
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET=p-ade-datasets
```

### 3. 서버 모드 실행 (권장)

```bash
# 웹 대시보드 + 자동 파이프라인 무한 반복
# crawl → download → detect → build_il → quality → upload → 30초 대기 → 반복
source venv/bin/activate
python main.py serve --target 500 --port 5000

# 대시보드 접속: http://localhost:5000
```

### 4. 단일 실행

```bash
# 파이프라인 1회 실행
python main.py run-once --target 500

# 특정 단계만 실행
python mass_collector.py --target 100 --stage crawl
python mass_collector.py --target 100 --stage detect
```

## 🔧 CLI 옵션

### main.py
```bash
# serve: 웹 대시보드 + 자동 파이프라인 (기본 모드)
python main.py serve --target 500 --port 5000

# run-once: 단일 실행
python main.py run-once --target 500 --stage crawl

# run-forever: CLI 무한 루프
python main.py run-forever --target 500
```

### mass_collector.py
```bash
--target N          # 목표 영상 수 (기본: 500)
--stage STAGE       # 실행 단계: crawl/download/detect/build_il/quality/upload
--keywords KW       # 키워드 (콤마 구분)
--sources SRC       # 소스: youtube,google_videos
--quality Q         # 다운로드 품질: 360p/480p/720p/1080p
--workers N         # 병렬 워커 수
--dry-run           # 실행 계획만 출력
--detect-device DEV # 검출 디바이스: cuda:0/cpu (기본: auto)
```

## 📁 데이터 포맷

### 모방학습 Episode NPZ 구조
```python
import numpy as np
data = np.load('episode.npz', allow_pickle=True)

data['states']        # [T, 199]   - 관절위치(99) + 속도(99) + 신뢰도(1)
data['actions']       # [T-1, 100] - 관절 delta(99) + gripper(1)
data['poses']         # [T, 33, 3] - 정규화된 관절 좌표
data['left_hand']     # [T, 21, 3] - 왼손 랜드마크
data['right_hand']    # [T, 21, 3] - 오른손 랜드마크
data['gripper_state'] # [T]        - 그리퍼 상태 (0=열림, 1=닫힘)
data['confidence']    # [T]        - 포즈 검출 신뢰도
data['velocity']      # [T, 33, 3] - 관절 속도
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

## 🛠️ 기술 스택

| 영역 | 기술 |
|------|------|
| **언어** | Python 3.13+ |
| **크롤링** | yt-dlp 2026.2, aiohttp, requests |
| **AI/ML** | YOLOv8 (ultralytics 8.4), MediaPipe Tasks, PyTorch 2.7 |
| **GPU** | CUDA 3-Stream 병렬 (듀얼 GPU), CPU 폴백 |
| **데이터** | NumPy, Polars, SciPy, h5py |
| **클라우드** | AWS S3 (boto3) |
| **DB** | PostgreSQL 14 (SQLAlchemy 2.0) |
| **큐/캐시** | Redis 6+ |
| **웹 UI** | Flask 3.1, Bootstrap 5, 실시간 로그 |
| **모니터링** | loguru, psutil, GPU/VRAM 모니터 |
| **테스트** | pytest |

## 🖥️ 시스템 요구사항

| 항목 | 최소 사양 |
|------|-----------|
| **OS** | Ubuntu 22.04+ (x86_64) |
| **Python** | 3.13+ |
| **PostgreSQL** | 14+ |
| **Redis** | 6+ |
| **GPU** | CUDA 지원 GPU (선택, CPU 폴백 가능) |
| **RAM** | 16GB+ |
| **디스크** | 100GB+ (영상 다운로드/처리용) |

## 📝 라이선스

MIT License
