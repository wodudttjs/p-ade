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
```

### MVP Phase 1 실행

```bash
# 1. YouTube 크롤링
python -m spiders.youtube_spider --keywords "robot assembly" --max-results 100

# 2. 비디오 다운로드
python -m ingestion.downloader --input data/urls.csv

# 3. 포즈 추출
python -m extraction.pose_estimator --video data/raw/video.mp4

# 4. 클라우드 업로드
python -m storage.s3_uploader --input data/processed/
```

## 📊 MVP Phase 1 목표

- ✅ 100개 비디오 URL 수집
- ✅ 비디오 다운로드 및 저장
- ✅ MediaPipe로 포즈 데이터 추출
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
