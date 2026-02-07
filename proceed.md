# P-ADE 대량 수집 시스템 실행 가이드

## 🚀 원클릭 대량 수집 (500개 이상)

```bash
# 전체 파이프라인 실행 (500개 목표, 크롤링→다운로드→검출→업로드)
python mass_collector.py --target 500

# 1000개 수집 + 6개 소스 + GPU 검출
python mass_collector.py --target 1000 --sources youtube,google_videos,vimeo,dailymotion,bilibili,rutube --detect-device cuda:0

# 드라이런 (실행 계획만 확인)
python mass_collector.py --target 500 --dry-run
```

## 📋 단계별 실행

### 1. 크롤링만 (URL 수집)
```bash
python mass_collector.py --target 500 --stage crawl
# 또는 직접 실행
python -m ingestion.multi_source_crawler --keywords "robot arm,pick and place,cobot,FANUC robot,UR5" --sources youtube,google_videos,vimeo,dailymotion --max-results 1500
```

### 2. 다운로드만
```bash
python mass_collector.py --target 500 --stage download
# 또는 직접 실행
python parallel_download.py --urls data/urls_mass.csv --workers 6 --timeout 600
```

### 3. 객체 검출 & Episode 생성
```bash
python mass_collector.py --target 500 --stage detect
# 또는 직접 실행
python -m extraction.detect_to_episodes --limit 500 --output-fps 5 --device cuda:0
```

### 4. S3 업로드
```bash
python mass_collector.py --stage upload
# 또는 직접 실행
python upload_to_s3.py --input data/episodes --prefix episodes
```

## 🔑 키워드 생성기 확인
```bash
python -m ingestion.keyword_generator
```

## 📊 대시보드
```bash
python run_dashboard.py
```

## 🧹 정리
```bash
python cleanup_robot_arm_data.py --apply
```

## ⚙️ 환경 변수 (.env)
```
MASS_COLLECT_TARGET=500
MASS_COLLECT_SOURCES=youtube,google_videos,vimeo,dailymotion
MASS_COLLECT_DOWNLOAD_WORKERS=6
MASS_COLLECT_DETECT_DEVICE=cuda:0
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=p-ade-datasets
```