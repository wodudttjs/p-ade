Phase 4: Docker & 배포 (Day 7-8)
🐳 Task C4-1: Docker 환경 구축
우선순위: P1
시간: 1일
파일: deploy/docker-compose.yml, deploy/Dockerfile
작업 내용:
1. docker-compose.yml 작성
yamlservices:
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: pade
      POSTGRES_USER: pade
      POSTGRES_PASSWORD: pade
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    command: redis-server /usr/local/etc/redis/redis.conf
    volumes:
      - ./redis.conf:/usr/local/etc/redis/redis.conf
      - redis_data:/data
    ports:
      - "6379:6379"
  
  pade:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://pade:pade@postgres:5432/pade
      REDIS_HOST: redis
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
2. Dockerfile 작성
dockerfileFROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.10 설치
# PyTorch, MediaPipe, YOLO 설치
# P-ADE 소스 복사
# requirements.txt 설치

CMD ["python", "main.py", "--target", "5000"]
3. 실행 및 검증
bashdocker-compose up -d
docker-compose logs -f pade
```

**검증**:
- 컨테이너 정상 시작
- GPU 인식 확인
- PostgreSQL, Redis 연결 확인

---

### 🚀 Task C4-2: systemd 서비스 업데이트
**우선순위**: P2  
**시간**: 0.5일  
**파일**: `deploy/robot-collector.service`

**작업 내용**:

#### 1. 서비스 파일 수정
```
변경:
- ExecStart: --target 500 → --target 5000
- MemoryMax: 8G → 16G
- Environment: CUDA_VISIBLE_DEVICES=0 → CUDA_VISIBLE_DEVICES=0,1
2. 설치 및 활성화
bashsudo cp deploy/robot-collector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable robot-collector
sudo systemctl start robot-collector
```

**검증**:
- 서비스 정상 시작
- 로그 확인 (journalctl -u robot-collector)
- 재부팅 후 자동 시작 확인

---

### 📝 Task C4-3: 환경 변수 정리
**우선순위**: P2  
**시간**: 0.5일  
**파일**: `.env.example`

**작업 내용**:

#### 1. 전체 환경변수 정리
```
# Database
DATABASE_URL=postgresql://pade:pade@localhost:5432/pade

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AWS S3
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET=robot-arm-dataset
S3_REGION=us-east-1

# YouTube API
YOUTUBE_API_KEYS=key1,key2,key3

# Pipeline
TARGET_COUNT=5000
CRAWL_WORKERS=16
DOWNLOAD_WORKERS=12
GPU_STREAMS=6

# Monitoring
DASHBOARD_PORT=8000
ALERT_EMAIL=
SLACK_WEBHOOK=
```

#### 2. README 업데이트
```
섹션 추가:
- 환경 설정 가이드
- PostgreSQL 설치 가이드
- Redis 설정 가이드
- Docker 실행 가이드
```

---

## 📊 Developer C 최종 체크리스트

### 필수 완료 항목
- [ ] PostgreSQL 마이그레이션 완료
- [ ] video_registry, pipeline_runs 테이블 생성
- [ ] Redis 설정 최적화
- [ ] DiskPolicy 구현 및 연동
- [ ] docker-compose.yml 작성
- [ ] systemd 서비스 업데이트

### 성능/안정성 목표
- [ ] PostgreSQL 동시 쓰기 20개 안정
- [ ] Redis 메모리 2GB 이하
- [ ] 디스크 사용량 < 50GB (파이프라인 후)
- [ ] Docker 환경 정상 동작

---

# 🔴 Developer D: 대시보드/모니터링 담당

## 📋 작업 목표
- 대시보드 5,000개 스케일 대응
- 실시간 모니터링 강화
- 크로스-런 모니터링 추가
- 알림 시스템 확장

---

## Phase 0: 사전 준비 (4시간)

### 📊 Task D0-1: 현재 대시보드 성능 측정
**우선순위**: P1  
**시간**: 2시간

**측정 항목**:
1. **파일 스캔 성능**:
   - `glob("*.mp4")` 소요 시간 (파일 수별)
   - `glob("*.npz")` 소요 시간
   - 5,000개일 때 예상 응답 시간
   
2. **페이지 로드 시간**:
   - 메인 페이지
   - 통계 페이지
   - 로그 페이지
   
3. **메모리 사용량**:
   - 로그 2,000줄 메모리
   - 파일 목록 메모리

**산출물**:
- `dashboard_performance_report.md`
- 병목 지점 3가지 이상
- 최적화 우선순위

---

### 🔍 Task D0-2: 대시보드 요구사항 정의
**우선순위**: P1  
**시간**: 2시간

**요구사항 정리**:
1. **5K 스케일 대응**:
   - 페이지네이션 필수
   - DB 쿼리 기반 통계
   - 실시간 스트리밍
   
2. **크로스-런 모니터링**:
   - 전체 실행 이력
   - 중복 방지 통계
   - 비디오 레지스트리 검색
   
3. **새 기능**:
   - 실시간 진행률 (SSE)
   - 알림 규칙 6개 추가
   - 디스크 사용량 모니터링

---

## Phase 1: 대시보드 5K 대응 (Day 1-3)

### ⭐⭐⭐⭐⭐ Task D1-1: DB 기반 통계로 전환
**우선순위**: P0  
**시간**: 1.5일  
**파일**: `dashboard/web_app.py`

**작업 내용**:

#### 1. 파일 시스템 스캔 제거
```
AS-IS:
- glob("data/raw/*.mp4")
- glob("data/episodes/*.npz")
- 파일 목록 전체 로드

TO-BE:
- PostgreSQL 쿼리
- COUNT(*) 집계
- 페이지네이션
2. 통계 API 수정
python# /api/stats 엔드포인트

AS-IS:
{
  "total_videos": len(glob("*.mp4")),
  "total_episodes": len(glob("*.npz")),
  ...
}

TO-BE:
{
  "total_videos": db.query("SELECT COUNT(*) FROM videos"),
  "total_episodes": db.query("SELECT COUNT(*) FROM episodes"),
  "today_collected": db.query("SELECT COUNT(*) WHERE date = today"),
  ...
}
```

#### 3. 페이지네이션 추가
```
엔드포인트:
- /api/videos?page=1&per_page=50
- /api/episodes?page=1&per_page=50
- /api/rejected?page=1&per_page=50

응답:
{
  "items": [...],
  "total": 5000,
  "page": 1,
  "per_page": 50,
  "total_pages": 100
}
검증:

5,000개 데이터로 응답 시간 < 100ms
페이지네이션 정상 동작
메모리 사용량 안정


🔄 Task D1-2: 실시간 진행률 스트리밍
우선순위: P0
시간: 1일
파일: dashboard/web_app.py, dashboard/pages.py
작업 내용:
1. SSE 엔드포인트 추가
python# /api/stream/progress

구현:
1. Redis pub/sub 구독
2. 파이프라인 진행률 실시간 수신
3. SSE로 클라이언트에 스트리밍
2. 프론트엔드 연동
javascript// dashboard/pages.py 또는 static/app.js

const eventSource = new EventSource('/api/stream/progress');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateProgress(data);
};
```

#### 3. Redis 진행률 발행
```
위치: mass_collector.py 각 스테이지

추가:
redis.publish('pade:progress', json.dumps({
  'stage': 'crawl',
  'current': 1000,
  'total': 20000,
  'percent': 5.0
}))
```

**검증**:
- 실시간 진행률 업데이트 확인
- 브라우저 여러 개 동시 접속 테스트
- SSE 연결 안정성 확인

---

### 📊 Task D1-3: 로그 시스템 개선
**우선순위**: P1  
**시간**: 0.5일  
**파일**: `dashboard/web_app.py`

**작업 내용**:

#### 1. 메모리 로그 → Redis Stream
```
AS-IS:
- 메모리 list (2,000줄 제한)
- 서버 재시작 시 소실

TO-BE:
- Redis Stream ("pade:logs")
- MAXLEN ~ 10000 (자동 만료)
- 영구 보존 (선택적)
```

#### 2. 로그 레벨 필터링
```
기능:
- DEBUG, INFO, WARNING, ERROR 선택
- 실시간 필터링
- 검색 기능 (키워드)
```

#### 3. 로그 다운로드
```
기능:
- 특정 run_id 로그 전체 다운로드
- 텍스트 파일 형식
- 디버깅 용도
검증:

10,000줄 로그 부드러운 스크롤
필터링 즉시 적용
로그 다운로드 정상 동작


Phase 2: 크로스-런 모니터링 (Day 4-5)
📈 Task D2-1: 실행 이력 페이지
우선순위: P1
시간: 1일
파일: dashboard/web_app.py, dashboard/pages.py
작업 내용:
1. 실행 이력 API
python# /api/runs - 전체 실행 이력

응답:
{
  "runs": [
    {
      "run_id": "20260303-123456",
      "started_at": "2026-03-03 12:34:56",
      "completed_at": "2026-03-03 20:45:12",
      "target_count": 5000,
      "crawled": 20000,
      "downloaded": 10000,
      "processed": 5000,
      "passed": 3250,
      "uploaded": 3250,
      "status": "completed"
    },
    ...
  ]
}
2. 실행 상세 API
python# /api/runs/<run_id> - 특정 실행 상세

응답:
{
  "run_id": "20260303-123456",
  "stages": {
    "crawl": {"duration": 600, "count": 20000},
    "download": {"duration": 14400, "count": 10000},
    ...
  },
  "errors": [...],
  "warnings": [...],
  "stats": {...}
}
```

#### 3. Runs 페이지 UI
```
구성:
- 테이블: run_id, 날짜, 타겟, 수집량, 성공률, 소요시간
- 필터: 날짜 범위, 상태 (running, completed, failed)
- 정렬: 날짜순, 수집량순
- 상세 보기: 클릭 시 run_id 상세 페이지
검증:

100개 실행 이력 로드 < 200ms
페이지네이션 정상 동작
상세 페이지 모든 정보 표시


🔍 Task D2-2: 중복 방지 통계 페이지
우선순위: P1
시간: 1일
파일: dashboard/web_app.py, dashboard/pages.py
작업 내용:
1. 중복 통계 API
python# /api/dedup/stats

응답:
{
  "total_collected": 50000,
  "unique_videos": 48500,
  "duplicate_blocked": 1500,
  "duplicate_rate": 3.0,
  "rejected_count": 5000,
  "by_platform": {
    "youtube": 40000,
    "google_videos": 8500
  }
}
2. 레지스트리 검색 API
python# /api/registry/search?q=video_id

응답:
{
  "found": true,
  "video_id": "dQw4w9WgXcQ",
  "url": "https://...",
  "status": "collected",
  "collected_at": "2026-03-03 14:23:11",
  "quality_score": 72.5,
  "s3_path": "s3://..."
}
```

#### 3. Dedup Stats 위젯
```
위치: 메인 대시보드

표시:
- 총 수집 영상 수
- 고유 영상 수
- 중복 차단 수
- 중복률 (%)
- 그래프: 일별 중복률 추이
```

**검증**:
- 50,000개 레지스트리에서 검색 < 50ms
- 통계 정확성 확인
- 그래프 정상 표시

---

## Phase 3: 알림 시스템 강화 (Day 6-7)

### 🔔 Task D3-1: 알림 규칙 확장
**우선순위**: P1  
**시간**: 1일  
**파일**: `monitor/alert_loop.py`

**작업 내용**:

#### 1. 신규 알림 규칙 6개 추가
```
규칙:
1. disk_space_low
   - 조건: 여유 < 50GB
   - 레벨: CRITICAL
   - 메시지: "Disk space low: {free}GB remaining"

2. download_stall
   - 조건: 10분간 다운로드 0건
   - 레벨: WARNING
   - 메시지: "Download stalled for 10 minutes"

3. dedup_rate_high
   - 조건: 중복률 > 70%
   - 레벨: WARNING
   - 메시지: "High duplicate rate: {rate}% (keyword exhaustion?)"

4. pipeline_timeout
   - 조건: 단일 스테이지 > 2시간
   - 레벨: ERROR
   - 메시지: "Stage {stage} timeout: {duration} hours"

5. quality_drop
   - 조건: 통과율 < 30%
   - 레벨: ERROR
   - 메시지: "Quality pass rate dropped to {rate}%"

6. run_complete
   - 조건: 파이프라인 완료
   - 레벨: INFO
   - 메시지: "Pipeline completed: {passed}/{target} collected"
```

#### 2. 알림 발송 로직
```
채널:
- Email (SMTP)
- Slack Webhook
- 대시보드 알림 배지

throttling:
- 동일 규칙: 1시간에 1회만
- CRITICAL: 즉시 발송
- WARNING: 5분 지연 후 발송
- INFO: 10분 지연 후 발송
```

#### 3. 알림 이력
```
저장:
- PostgreSQL alerts 테이블
- 알림 발송 기록
- 확인 여부

UI:
- /alerts 페이지
- 미확인 알림 표시
- 확인 버튼
```

**검증**:
- 각 규칙별 트리거 테스트
- 이메일/Slack 정상 발송 확인
- throttling 동작 확인

---

### 📊 Task D3-2: 메트릭 수집 강화
**우선순위**: P1  
**시간**: 1일  
**파일**: `monitor/stats_collector.py`

**작업 내용**:

#### 1. 신규 메트릭 추가
```
메트릭:
1. disk_usage
   - data/raw 사용량
   - data/episodes 사용량
   - 여유 공간
   - 1분마다 수집

2. download_speed
   - 현재 다운로드 속도 (videos/min)
   - 10초마다 수집
   - 5분 이동 평균

3. duplicate_rate
   - 최근 100개 중 중복 수
   - 1분마다 수집
   - 실시간 중복률

4. gpu_memory
   - GPU별 VRAM 사용량
   - 10초마다 수집
   - 2개 GPU 독립 추적

5. queue_depth
   - 크롤링 큐
   - 다운로드 큐
   - 처리 큐
   - 30초마다 수집
```

#### 2. Redis 저장
```
키:
- pade:stats:disk → Hash
- pade:stats:download_speed → Time Series
- pade:stats:duplicate_rate → Time Series
- pade:stats:gpu → Hash
- pade:stats:queues → Hash

TTL: 24시간 (일일 통계)
```

#### 3. 대시보드 위젯
```
위치: 메인 대시보드

위젯:
- 디스크 사용량 게이지
- 다운로드 속도 그래프
- 중복률 그래프
- GPU 메모리 게이지
- 큐 깊이 막대 그래프
검증:

모든 메트릭 정상 수집 확인
대시보드에 실시간 업데이트 확인
Redis 메모리 사용량 안정


Phase 4: Airflow DAG 업데이트 (Day 8)
🔄 Task D4-1: DAG 설정 업데이트
우선순위: P2
시간: 0.5일
파일: dags/robot_collection_dag.py
작업 내용:
1. Task 설정 변경
python# target 5000으로 변경
task_args = {
    'target_count': 5000,
    'timeout': timedelta(hours=8),  # 전체 타임아웃
}

# 개별 태스크 타임아웃
crawl_task = PythonOperator(
    ...,
    execution_timeout=timedelta(minutes=30)
)

download_task = PythonOperator(
    ...,
    execution_timeout=timedelta(hours=5)
)

process_task = PythonOperator(  # detect+il 통합
    ...,
    execution_timeout=timedelta(hours=5)
)

quality_task = PythonOperator(
    ...,
    execution_timeout=timedelta(minutes=30)
)

upload_task = PythonOperator(
    ...,
    execution_timeout=timedelta(hours=2)
)
2. Task 의존성 수정
python# AS-IS
crawl >> download >> detect >> build_il >> quality >> upload

# TO-BE
crawl >> download >> process >> quality >> upload >> cleanup
3. 재시도 정책
pythondefault_args = {
    'retries': 1,  # 1회 재시도
    'retry_delay': timedelta(minutes=10),
    'retry_exponential_backoff': True,
}
```

**검증**:
- DAG 구문 오류 없음 확인
- 테스트 실행 (--dry-run)
- Airflow UI에서 DAG 그래프 확인

---

## 📊 Developer D 최종 체크리스트

### 필수 완료 항목
- [ ] 대시보드 DB 기반 전환
- [ ] 페이지네이션 구현
- [ ] 실시간 진행률 SSE
- [ ] 실행 이력 페이지
- [ ] 중복 통계 페이지
- [ ] 알림 규칙 6개 추가
- [ ] 메트릭 수집 강화
- [ ] Airflow DAG 업데이트

### 성능 목표
- [ ] 5,000개 데이터 응답 < 100ms
- [ ] SSE 실시간 업데이트 안정
- [ ] 알림 정상 발송
- [ ] 대시보드 메모리 안정

---

# 🔄 Phase 4: 통합 테스트 (Day 11-12, 전체)

## 통합 테스트 계획

### 🧪 Test 1: 소규모 통합 테스트
**담당**: 전체  
**시간**: 0.5일

**테스트 시나리오**:
```
1. 타겟 100개로 전체 파이프라인 실행
2. 모든 기능 정상 동작 확인:
   - GlobalVideoRegistry 중복 방지
   - Unified Processing (Detect+IL 통합)
   - Dual-GPU 6-Stream
   - 품질 평가 배치
   - S3 업로드 후 로컬 정리
   - 대시보드 실시간 업데이트
```

**예상 결과**:
- 100개 수집 완료: ~10분
- 중복 0개
- 모든 스테이지 정상
- 대시보드 정상 표시

---

### 🧪 Test 2: 1,000개 중규모 테스트
**담당**: 전체  
**시간**: 0.5일

**테스트 시나리오**:
```
1. 타겟 1,000개 실행
2. 성능 측정:
   - 크롤링 속도
   - 다운로드 속도
   - GPU 처리 속도
   - 전체 소요 시간
3. 리소스 사용량:
   - GPU VRAM
   - 디스크 사용량
   - PostgreSQL 연결 수
   - Redis 메모리
```

**예상 결과**:
- 1,000개 수집: ~1.5시간
- GPU 안정 동작
- 디스크 사용량 < 10GB
- 에러율 < 5%

---

### 🧪 Test 3: 5,000개 풀 스케일 테스트
**담당**: 전체  
**시간**: 1일

**테스트 시나리오**:
```
1. 타겟 5,000개 실행 (1회차)
2. 동일 키워드로 재실행 (2회차)
   - 중복 100% 차단 확인
3. 새 키워드로 재실행 (3회차)
   - 신규 5,000개 수집
```

**측정 항목**:
- 총 소요 시간: 목표 8시간 이내
- 1회차 중복률: 0%
- 2회차 중복 차단: 100%
- 3회차 신규 수집: 5,000개
- 디스크 사용량: < 50GB (정리 후)
- 에러율: < 5%

**성공 기준**:
- [ ] 5,000개 수집 완료
- [ ] 8시간 이내 완료
- [ ] 실행 간 중복 0%
- [ ] 모든 시스템 안정

---

# 📝 최종 정리

## 핵심 성공 지표

| 지표 | 현재 | 목표 | 담당 |
|------|------|------|------|
| 크롤링 속도 | 456개/235초 | 20,000개/10분 | A |
| 다운로드 속도 | 71개/30분 | 10,000개/4시간 | A |
| GPU 처리 속도 | 1,012개/6,254초 | 5,000개/4시간 | B |
| 통과율 | 48.8% | 65% | B |
| 실행 간 중복 | 체크 없음 | 0% | A, C |
| 전체 소요 시간 | 3.5시간/500개 | 8시간/5,000개 | 전체 |

## 의존성 그래프
```
Phase 0 (전체)
    ↓
Phase 1
    A1-1 (Registry) ←→ C1-1 (PostgreSQL)
    ↓                  ↓
Phase 2
    A2-1 (키워드)      B1-1 (Unified)
    A2-2 (크롤링)      B2-1 (6-Stream)
    A2-3 (다운로드)    C2-1 (Redis)
    ↓                  ↓
Phase 3
    D1-1 (대시보드)    C3-1 (디스크)
    D2-1 (모니터링)    C4-1 (Docker)
    D3-1 (알림)
    ↓
Phase 4 (통합 테스트, 전체)