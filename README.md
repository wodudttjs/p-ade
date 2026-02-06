# P-ADE (Physical AI Data Engine)

웹 비디오 자원을 자동 수집하여 로봇 학습용 (State, Action) 데이터셋으로 변환하는 End-to-End 파이프라인

## 🎯 프로젝트 개요

- **목표**: 웹에서 동작 비디오를 자동 발견하고, 로봇이 학습 가능한 형태로 변환하여 클라우드에 저장
- **핵심 가치**: 데이터 부족 해결, 완전 자동화, 클라우드 네이티브 확장성

## 🏗️ 시스템 아키텍처

```
[Discovery] → [Ingestion] → [Extraction] → [Transform] → [Storage] → [Monitor]
  (Scrapy)     (yt-dlp)     (MediaPipe)    (Pipeline)   (Cloud)     (Dashboard)
```

## 📦 모듈 구조

- `spiders/` - 웹 크롤링 (Scrapy 기반)
- `ingestion/` - 비디오 다운로드 관리
- `extraction/` - 포즈 및 객체 추출 (MediaPipe, YOLO)
- `transformation/` - 데이터 변환 및 정규화
- `storage/` - 클라우드 저장소 관리
- `monitoring/` - 모니터링 및 대시보드
- `models/` - 데이터베이스 모델
- `core/` - 공통 유틸리티
- `config/` - 설정 파일
- `tests/` - 테스트 코드

## 🚀 빠른 시작

### 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# (객체 검출 사용 시) YOLO 의존성
pip install ultralytics

# (선택) ffmpeg 설치 후 PATH 등록
# Windows: https://ffmpeg.org/download.html
```

### 환경설정

```bash
# DB (SQLite 기본 경로 사용 시 생략 가능)
set P_ADE_DB_PATH=path\to\pade.db

# (공유 DB 사용 시) 네트워크 공유 경로
# 예: \\호스트이름\pade-db\pade.db
# set P_ADE_DB_PATH=\\HOSTNAME\pade-db\pade.db

# S3 업로드
set AWS_ACCESS_KEY_ID=...
set AWS_SECRET_ACCESS_KEY=...
set AWS_REGION=ap-northeast-2
set AWS_S3_BUCKET=p-ade-datasets

# (선택) Redis 큐
set REDIS_URL=redis://localhost:6379/0
```

### 공유 DB 설정 (팀 협업)

- 공유 이름: pade-db
- 공유 경로: \\HOSTNAME\pade-db
- 실제 DB 파일: \\HOSTNAME\pade-db\pade.db

```bash
# 공유 DB 사용
set P_ADE_DB_PATH=\\HOSTNAME\pade-db\pade.db
```

### MVP Phase 1 실행

```bash
# 1. YouTube 크롤링
python -m spiders.youtube_spider --keywords "robot arm pick and place" --max-results 100 --overwrite

# 2. 비디오 다운로드 (필터 포함, 최대 10개)
python -m ingestion.downloader --input data/urls.csv --max-downloads 10

# 3. 객체 검출 → episodes 저장 (로봇팔 영상 상위 10개)
python -m extraction.detect_to_episodes --limit 10 --output-fps 5

# 4. 클라우드 업로드 (episodes)
python upload_to_s3.py --input data/episodes --prefix episodes

# 5. GUI 실행
python run_dashboard.py

# (선택) 로봇팔 외 데이터 정리
python cleanup_robot_arm_data.py --apply
```

## 📊 MVP Phase 1 목표

- ✅ 100개 비디오 URL 수집
- ✅ 비디오 다운로드 및 저장
- ✅ 로봇팔 객체 검출 및 episodes 저장
- ✅ AWS S3 업로드

## 🛠️ 기술 스택

- **언어**: Python 3.10+
- **크롤링**: Scrapy, Playwright
- **비디오**: yt-dlp, OpenCV
- **AI**: MediaPipe, YOLOv8
- **데이터**: NumPy, Pandas
- **클라우드**: AWS S3, boto3
- **DB**: PostgreSQL

## 📝 라이선스

MIT License
