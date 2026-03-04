# P-ADE 5000 Scale-Up MVP: 프로젝트 리모델링 & 분업 계획서

> **목표**: 파이프라인 1회 실행 시 500개 → **5,000개** 영상 수집, **매 실행마다 이전 파이프라인과 영상 중복 0%** 보장
> **현재 기준**: 1회 실행 ~3.5시간, 456 크롤 → 71 다운로드 → 1,012 에피소드 → 533 통과 (48.8%)

---

## 0. 현재 병목 진단

| 단계 | 현재 설정 | 병목 원인 | 5,000개 시 예상 문제 |
|------|----------|----------|-------------------|
| **Crawl** | 50 키워드, 4 workers, sync 모드 | 키워드 부족 + 동기식 크롤링 | 동일 키워드 반복 → 중복 URL 폭증 |
| **Download** | 6 workers, 720p, timeout 600s | yt-dlp 단일 프로세스 병목 | 5,000개 순차 다운로드 시 ~14시간 |
| **Detect** | GPU 3-Stream, batch 4 | VRAM 9GB 제한, FPS 5.0 고정 | 5,000개 처리 시 ~17시간 (현재 6,254초/1,012개) |
| **Build IL** | 순차 or GPU 3-Stream | ~~MediaPipe CPU 바운드~~ → RTMPose GPU ONNX로 해결 | Detect+IL 통합 1-Pass 처리 |
| **Quality** | 순차 평가, 48.8% 통과율 | 통과율 낮아 2배 이상 크롤 필요 | 10,000+ 크롤 → 5,000 통과 필요 |
| **Upload** | 순차 S3 업로드 | 단일 스레드 업로드 | 5,000 NPZ 업로드 ~2시간 |
| **중복 방지** | CSV stem 비교 (현재 실행 내) | **실행 간(cross-run) 중복 체크 없음** | 2회차부터 동일 영상 대량 재수집 |
| **DB** | SQLite 단일 파일 | 동시 쓰기 락 | 멀티프로세스 충돌 |
| **디스크** | data/raw에 MP4 무한 축적 | 정리 정책 없음 | 5,000 x 50MB = ~250GB/회 |

---

## 1. 아키텍처 리모델링 개요

### AS-IS (500개)
```
main.py → MassCollector → 6단계 순차 실행 → SQLite → S3
                          (단일 프로세스)
```

### TO-BE (5,000개)
```
main.py → PipelineOrchestrator
           │
           ├─ [Phase 1: Crawl]  ── AsyncCrawlerPool(16 workers) + Redis Dedup
           │                        + GlobalVideoRegistry (cross-run 중복 방지)
           │
           ├─ [Phase 2: Download] ── DownloadPool(12 workers) + bandwidth throttle
           │                          + disk quota management
           │
           ├─ [Phase 3: Detect+IL] ── GPU Pipeline (dual-GPU 6-stream)
           │                           + CPU worker pool (fallback)
           │
           ├─ [Phase 4: Quality] ── batch evaluator (vectorized)
           │
           └─ [Phase 5: Upload] ── S3 multipart parallel (8 workers)

           PostgreSQL ←──→ Redis (cache + dedup + queue)
```

---

## 2. 팀 분업 구조

| 역할 | 담당 영역 | 핵심 모듈 |
|------|----------|----------|
| **A. 크롤링/수집 담당** | Crawl + Download 최적화, 중복 방지 | `ingestion/`, `workers/`, `cache/` |
| **B. GPU/처리 담당** | Detect + Build IL + Quality 최적화 | `extraction/`, `gpu/`, `quality/`, `scripts/pipeline/` |
| **C. 인프라/데이터 담당** | DB 마이그레이션, Redis, 스토리지, 배포 | `storage/`, `task_queue/`, `config/`, `deploy/`, `core/` |
| **D. 대시보드/모니터링 담당** | 대시보드 5K 대응, 알림 강화 | `dashboard/`, `monitor/`, `api/`, `dags/` |

---

## A. 크롤링/수집 담당

### A-1. 글로벌 중복 방지 시스템 (최우선)

**문제**: 현재 `mass_collector.py:479`에서 `existing = set(p.stem for p in output_dir.glob("*.mp4"))`로 현재 실행 내 다운로드 중복만 체크. **이전 파이프라인 실행**의 영상과 중복 체크하는 로직 없음.

**해결**: `GlobalVideoRegistry` 구현

```
파일: cache/video_registry.py (신규)

class GlobalVideoRegistry:
    """
    모든 파이프라인 실행에 걸친 글로벌 영상 중복 방지 레지스트리.
    Redis Persistent Set + SQLite fallback.
    """

    핵심 구조:
    - Redis Set: "pade:collected_videos" → 모든 수집 완료 video_id
    - Redis Set: "pade:collected_urls" → 모든 수집 완료 URL (정규화)
    - Redis Set: "pade:rejected_videos" → 품질 탈락 video_id (재수집 방지)
    - SQLite 테이블: video_registry(video_id, url_hash, collected_at, run_id, status)

    핵심 메서드:
    - is_collected(video_id) → bool
    - is_url_collected(url) → bool
    - register(video_id, url, run_id)
    - register_rejected(video_id, reason)
    - filter_new_only(video_list) → List  # 미수집 영상만 필터
    - get_run_stats(run_id) → Dict
```

**적용 위치**:
- `mass_collector.py` `_stage_crawl()`: 크롤링 결과를 `filter_new_only()`로 필터
- `mass_collector.py` `_stage_download()`: 다운로드 전 `is_collected()` 체크
- `mass_collector.py` `_stage_quality()`: 탈락 시 `register_rejected()` 호출
- `mass_collector.py` `_stage_upload()`: 성공 시 `register()` 호출

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `cache/video_registry.py` | **신규** - GlobalVideoRegistry 클래스 |
| `cache/redis_cache.py` | Bloom filter 크기 확장 (100K → 1M), TTL 제거 (영구 보존) |
| `mass_collector.py` | 모든 스테이지에 Registry 연동 |
| `ingestion/multi_source_crawler.py` | crawl() 내부에 Registry 필터 삽입 |

---

### A-2. 크롤링 10배 확장

**현재**: `KeywordGenerator.get_flat_keywords(max_count=50)` → 50개 키워드로 `crawl_target = 500 * 3.0 = 1,500`개 크롤 시도

**변경**:

| 항목 | 현재 | 변경 |
|------|------|------|
| 키워드 수 | 50개 | **500개** (카테시안 풀 6,000개 중 상위 500) |
| 크롤 배수 | 3.0x | **4.0x** (통과율 48% 감안 → 20,000 크롤 → 10,000 다운 → 5,000 통과) |
| 크롤 모드 | sync (ThreadPool 4) | **async (100 concurrent)** |
| 소스 | youtube, google_videos | **+ vimeo, bilibili** (4 소스) |
| 언어 | en, ko | **+ ja, zh, de** (5 언어) |
| 중복 필터 | 실행 내 video_id | **GlobalVideoRegistry** (cross-run) |

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `mass_collector.py:62-123` | `PipelineConfig` 기본값 업데이트 |
| `mass_collector.py:336-340` | `max_keywords=500`, `get_flat_keywords(max_count=500)` |
| `mass_collector.py:362-371` | `async_mode=True`, `max_workers=16` |
| `ingestion/keyword_generator.py` | `generate_cartesian_all()` 호출 + 성과 기반 키워드 우선순위 |
| `ingestion/multi_source_crawler.py` | async 모드 기본화, 소스 4개 확장 |
| `ingestion/async_crawler.py` | `max_concurrent=100` → `200`, 레이트 리밋 소스별 분리 |
| `ingestion/youtube_batch.py` | 배치 사이즈 50 유지, 쿼터 모니터링 강화 |

---

### A-3. 다운로드 시스템 강화

**현재**: `parallel_download()` → 6 workers, 동기식 yt-dlp

**변경**:

| 항목 | 현재 | 변경 |
|------|------|------|
| workers | 6 | **12** (네트워크 I/O 바운드) |
| timeout | 600초 | **300초** (빠른 실패 → 재시도) |
| retry | 없음 | **3회 재시도** (지수 백오프) |
| bandwidth | 무제한 | **영상당 5MB/s 제한** (총 60MB/s) |
| 디스크 관리 | 없음 | **다운로드 전 디스크 여유 체크** |
| 품질 | 720p | **480p 옵션** (대량 수집 시 용량 절약) |
| 진행률 | stdout print | **Redis pub/sub 진행률** |

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `scripts/pipeline/parallel_download.py` | **merge conflict 해결** + workers 12, retry 3, bandwidth limit |
| `mass_collector.py:487` | `videos[:self.config.target_count]` → Registry 필터 후 슬라이스 |
| `ingestion/downloader.py` | bandwidth throttle 옵션, 480p 모드 추가 |
| `storage/storage_manager.py` | `ensure_space()` 다운로드 전 호출 연동 |

---

## B. GPU/처리 담당

### B-1. GPU 파이프라인 최적화

**현재**: `GPU3StreamManager` → 3 streams, VRAM 9GB 제한, batch 2~4

**변경**:

| 항목 | 현재 | 변경 |
|------|------|------|
| Stream 수 | 3 | **6** (dual-GPU: GPU0 x3 + GPU1 x3) |
| VRAM 제한 | 9GB | **GPU별 동적 (총 VRAM의 85%)** |
| batch 전략 | auto_adjust (2~4) | **영상 길이별 분류 후 동적 배칭** |
| 장시간 영상 | 15fps CPU 폴백 | **별도 CPU worker pool** 에 위임 |
| Detect+IL | 별도 스테이지 | **Detect → IL 파이프라인 통합** (1-pass) |
| 메모리 관리 | `torch.cuda.empty_cache()` | **프레임 단위 스트리밍 + 명시적 GC** |

**핵심 변경: Detect+IL 통합 1-Pass 처리 (RTMPose WholeBody GPU)**

현재 문제: Detect(YOLO)로 NPZ 저장 → Build IL(MediaPipe)로 다시 읽어 처리 = **2번 비디오 디코딩**

해결: 비디오 1회 디코딩 → YOLOX(사람검출) + DWPose(WholeBody 133 keypoints) GPU 동시 처리 → NPZ 1회 저장

> **⚠️ 포즈 프레임워크 마이그레이션 완료**:
> MediaPipe (CPU, 33 keypoints) → **RTMPose WholeBody (GPU ONNX, 133 keypoints)**
> - body: 17 COCO keypoints → State dim 103, Action dim 52
> - hand: 21 × 2 → 그리퍼 열림/닫힘 추정
> - `torch` 사전 임포트로 cuDNN 9 DLL 자동 로딩
> - 핵심 모듈: `extraction/rtmpose_wholebody.py`

```
파일: gpu/unified_processor.py (신규)

class UnifiedVideoProcessor:
    """
    1-Pass 통합 처리: 비디오 → (YOLOX + DWPose WholeBody) → State-Action NPZ

    기존 2단계를 1단계로 통합:
      AS-IS: detect(video→npz) + build_il(video→npz) = 2x 비디오 디코딩
      TO-BE: unified(video→npz) = 1x 비디오 디코딩 (GPU 가속)

    예상 성능 개선: 처리 시간 ~40% 단축
    """

    def process(self, video_path, output_dir):
        # 1. cv2로 비디오 1회 디코딩 (프레임 스트리밍)
        # 2. 각 프레임에 YOLOX + DWPose GPU 동시 적용
        # 3. Detection + Pose + State-Action 통합 NPZ 저장
```

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `gpu/unified_processor.py` | **신규** - 1-Pass 통합 프로세서 |
| `gpu/stream_manager.py:36-41` | `StreamConfig` — `num_streams=6`, VRAM 동적 계산 |
| `gpu/stream_manager.py:55-73` | multi-GPU 초기화 (`cuda:0`, `cuda:1`) |
| `gpu/stream_manager.py:115-128` | `auto_adjust_batch_size()` — GPU별 독립 VRAM 체크 |
| `gpu/stream_manager.py:140-195` | `process_batch()` — 영상 길이별 분류 + 큐잉 |
| `mass_collector.py:236-319` | `_stage_detect` + `_stage_build_il` → `_stage_process` 통합 |
| `extraction/detect_to_episodes.py` | **merge conflict 해결** + unified_processor 호출 |
| `extraction/object_detector.py` | batch inference 모드 추가 (여러 프레임 동시 추론) |

---

### B-2. CPU Worker Pool (긴 영상 전용)

```
파일: gpu/cpu_worker_pool.py (신규)

class CPUWorkerPool:
    """
    60초 이상 긴 영상을 CPU 멀티프로세스로 처리.
    GPU 3-Stream에서 긴 영상이 병목이 되지 않도록 분리.

    - ProcessPoolExecutor(max_workers=CPU_COUNT // 2)
    - RTMPose ONNX CPU 모드 (CPUExecutionProvider 폴백)
    - 15fps 다운샘플링
    """
```

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `gpu/cpu_worker_pool.py` | **신규** - CPU 전용 워커 풀 |
| `gpu/stream_manager.py` | `process_batch()`에서 긴 영상 자동 분류 → CPUWorkerPool 위임 |

---

### B-3. 품질 평가 벡터화

**현재**: `RobotArmQualityEvaluator` — 파일당 순차 평가

**변경**:
| 항목 | 현재 | 변경 |
|------|------|------|
| 평가 방식 | 파일별 순차 | **배치 벡터화** (numpy 연산) |
| 통과 기준 | 60점 | **50점** (통과율 48% → 목표 65%) |
| rejected 처리 | 파일 이동 | **DB 마킹만** (디스크 I/O 절약) |
| early reject | 없음 | **프레임 수 < 30이면 즉시 탈락** |

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `quality/evaluator.py` | `evaluate_batch()` 메서드 추가, threshold 50점, early reject |
| `quality/evaluator.py` | 벡터화 점수 계산 (numpy broadcast) |
| `mass_collector.py:798-883` | `_stage_quality()` — 배치 평가 호출 |
| `scripts/pipeline/filter_quality.py` | 배치 모드 연동 |

---

## C. 인프라/데이터 담당

### C-1. SQLite → PostgreSQL 마이그레이션

**현재 문제**: SQLite는 동시 쓰기를 지원하지 않음. 12 download workers + 6 GPU streams가 동시에 DB 업데이트하면 `database is locked` 오류.

**변경**:

```
파일: config/settings.py 변경

DATABASE_URL 기본값:
  AS-IS: sqlite:///data/pade.db
  TO-BE: postgresql://pade:pade@localhost:5432/pade
  FALLBACK: sqlite:///data/pade.db (개발 환경)
```

**마이그레이션 스크립트**:
```
파일: scripts/tools/migrate_to_postgres.py (신규)

기능:
1. PostgreSQL DB 생성 & 스키마 초기화
2. SQLite → PostgreSQL 데이터 복사
3. video_registry 테이블 추가 (cross-run 중복 방지)
4. 인덱스 생성: video_id, url_hash, status, created_at
5. 검증 (행 수 비교)
```

**새 테이블 추가**:
```sql
-- 글로벌 비디오 레지스트리 (cross-run 중복 방지)
CREATE TABLE video_registry (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(64) UNIQUE NOT NULL,
    url_hash VARCHAR(64) NOT NULL,
    url TEXT,
    platform VARCHAR(32),
    status VARCHAR(16) DEFAULT 'collected',  -- collected, rejected, failed
    run_id VARCHAR(64),
    collected_at TIMESTAMP DEFAULT NOW(),
    quality_score FLOAT,
    rejection_reason TEXT,
    s3_path TEXT
);

CREATE INDEX idx_video_registry_video_id ON video_registry(video_id);
CREATE INDEX idx_video_registry_url_hash ON video_registry(url_hash);
CREATE INDEX idx_video_registry_status ON video_registry(status);

-- 파이프라인 실행 이력
CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    target_count INT,
    crawled INT DEFAULT 0,
    downloaded INT DEFAULT 0,
    processed INT DEFAULT 0,
    passed INT DEFAULT 0,
    uploaded INT DEFAULT 0,
    status VARCHAR(16) DEFAULT 'running'
);
```

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `scripts/tools/migrate_to_postgres.py` | **신규** - 마이그레이션 스크립트 |
| `config/settings.py` | DATABASE_URL PostgreSQL 기본값 |
| `core/database.py` | PostgreSQL 연결 풀 (pool_size=20, max_overflow=30) |
| `.env.example` | PostgreSQL 관련 환경변수 추가 |
| `requirements.txt` | **merge conflict 해결** + `psycopg2-binary` 추가 |

---

### C-2. Redis 인프라 강화

**현재**: 단일 Redis 인스턴스, 캐시 전용

**변경**: Redis를 3가지 역할로 분리

| 역할 | 키 prefix | 용도 |
|------|----------|------|
| **Dedup** | `pade:registry:*` | 영구 보존. 수집된 모든 video_id/url |
| **Cache** | `pade:cache:*` | TTL 6h. 검색 결과 캐시 |
| **Queue** | `pade:queue:*` | 작업 큐. 크롤/다운로드/처리 작업 분배 |

**Redis 설정 변경**:
```
파일: deploy/redis.conf (신규)

maxmemory 2gb
maxmemory-policy allkeys-lru  # cache는 LRU 삭제
save 900 1                     # registry는 RDB 영구 보존
save 300 10
appendonly yes                 # AOF로 registry 보호
```

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `deploy/redis.conf` | **신규** - Redis 설정 |
| `cache/redis_cache.py` | Bloom filter 크기 1M, 키 prefix 분리, 연결 풀 |
| `cache/video_registry.py` | **신규** (A-1 참조) |
| `task_queue/task_queue.py` | 큐 크기 모니터링, dead letter queue 추가 |

---

### C-3. 디스크 & 스토리지 관리

**문제**: 5,000개 x 50MB = 250GB/회. 반복 실행 시 디스크 폭발.

**해결**:
```
파일: storage/disk_policy.py (신규)

class DiskPolicy:
    """
    디스크 사용 정책:
    1. 다운로드 전: 여유 공간 체크 (최소 100GB)
    2. 업로드 완료 후: 원본 MP4 삭제 (data/raw → 정리)
    3. 품질 탈락 에피소드: 즉시 삭제 (DB에만 기록)
    4. 성공 에피소드: S3 업로드 확인 후 로컬 삭제
    5. 파이프라인 시작 시: 이전 실행 잔여파일 정리
    """
```

**라이프사이클**:
```
[다운로드] → data/raw/{video_id}.mp4
                ↓ (처리 완료)
[GPU 처리] → data/episodes/{video_id}_episode.npz
                ↓ (S3 업로드 완료 확인)
[업로드]   → S3에 존재 확인 후 로컬 삭제
                ↓
[정리]     → data/raw/{video_id}.mp4 삭제
             data/episodes/{video_id}_episode.npz 삭제 (S3 확인 후)
```

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `storage/disk_policy.py` | **신규** - 디스크 정책 매니저 |
| `storage/storage_manager.py` | `cleanup_after_upload()` 메서드 추가 |
| `mass_collector.py` | 각 스테이지 후 디스크 정리 호출 |
| `storage/s3_uploader.py` | 업로드 성공 확인 후 로컬 삭제 옵션 |

---

### C-4. 배포 설정 업데이트

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `deploy/robot-collector.service` | `--target 5000`, 메모리 16GB, CUDA 듀얼 GPU |
| `deploy/docker-compose.yml` | **신규** - PostgreSQL + Redis + P-ADE 컨테이너 |
| `deploy/Dockerfile` | **신규** - P-ADE 이미지 (CUDA base) |
| `.env.example` | 전체 환경변수 업데이트 |
| `requirements.txt` | **merge conflict 해결** + 전체 의존성 정리 |

---

## D. 대시보드/모니터링 담당

### D-1. 대시보드 5K 대응

**현재 문제**: `web_app.py`에서 파일 시스템 직접 스캔 (`glob("*.mp4")`, `glob("*.npz")`) → 5,000+개 파일 시 응답 ~5초

**변경**:

| 항목 | 현재 | 변경 |
|------|------|------|
| 통계 소스 | 파일 시스템 glob | **DB 쿼리 전용** (PostgreSQL) |
| 파일 목록 | 전체 로드 | **페이지네이션** (50개/페이지) |
| 진행률 | `pipeline_state` dict | **Redis pub/sub** 실시간 스트리밍 |
| 로그 | 메모리 list (2000줄) | **Redis Stream** (자동 만료) |
| 새로고침 | 5초 폴링 | **Server-Sent Events (SSE)** |
| run 이력 | `jobs_history.json` (200개) | **pipeline_runs 테이블** |

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `dashboard/web_app.py` | DB 기반 통계 쿼리, 파일 glob 제거, 페이지네이션 API |
| `dashboard/web_app.py` | SSE 엔드포인트 (`/api/stream/progress`) |
| `dashboard/web_app.py` | `pipeline_runs` 테이블 연동 |
| `dashboard/pages.py` | 5K 스케일 UI (진행 바, 실행 이력 테이블) |

---

### D-2. 크로스-런 모니터링

**신규 대시보드 페이지 추가**:

```
/api/runs               → 전체 파이프라인 실행 이력
/api/runs/<run_id>      → 특정 실행 상세
/api/dedup/stats        → 중복 방지 통계 (총 수집, 중복 차단, 고유 영상)
/api/registry/search    → 비디오 레지스트리 검색
```

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `dashboard/web_app.py` | 위 4개 API 엔드포인트 추가 |
| `dashboard/pages.py` | "Runs" 페이지, "Dedup Stats" 위젯 |

---

### D-3. 알림 강화

**현재**: GPU < 30%, 큐 고갈, 실패율 > 40%

**추가 알림 규칙**:

| 규칙 | 조건 | 레벨 |
|------|------|------|
| `disk_space_low` | 여유 < 50GB | CRITICAL |
| `download_stall` | 10분간 다운로드 0건 | WARNING |
| `dedup_rate_high` | 중복률 > 70% (키워드 고갈 신호) | WARNING |
| `pipeline_timeout` | 단일 스테이지 > 2시간 | ERROR |
| `quality_drop` | 통과율 < 30% | ERROR |
| `run_complete` | 파이프라인 완료 | INFO |

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `monitor/alert_loop.py` | 위 규칙 추가 |
| `monitor/stats_collector.py` | 디스크/중복률/다운로드 속도 메트릭 추가 |

---

### D-4. Airflow DAG 업데이트

**파일 변경 목록**:
| 파일 | 변경 내용 |
|------|----------|
| `dags/robot_collection_dag.py` | target 5000, 태스크 타임아웃 8시간, 통합 Detect+IL 태스크 |

---

## 3. mass_collector.py 핵심 변경 사항

### PipelineConfig 변경

```python
# AS-IS
target_count: int = 500
crawl_workers: int = 4
download_workers: int = 6
use_async: bool = False

# TO-BE
target_count: int = 5000
crawl_workers: int = 16
download_workers: int = 12
upload_workers: int = 8
use_async: bool = True           # 기본 async
use_multiprocess: bool = True    # 기본 멀티프로세스
crawl_multiplier: float = 4.0   # 3.0 → 4.0
max_keywords: int = 500          # 50 → 500
languages: List[str] = ["en", "ko", "ja", "zh", "de"]
sources: List[str] = ["youtube", "google_videos", "vimeo", "bilibili"]

# 신규 옵션
run_id: str = ""                 # 파이프라인 실행 ID (자동 생성)
cleanup_after_upload: bool = True # S3 업로드 후 로컬 삭제
unified_processing: bool = True  # Detect+IL 통합 처리
download_quality: str = "480p"   # 대량 수집 시 용량 절약
```

### STAGES 변경

```python
# AS-IS
STAGES = ["crawl", "download", "detect", "build_il", "quality", "upload"]

# TO-BE
STAGES = ["crawl", "download", "process", "quality", "upload", "cleanup"]
#                                ^^^^^^^^                        ^^^^^^^
#                          detect+build_il 통합        S3 업로드 후 로컬 정리
```

---

## 4. 실행 계획 (우선순위)

### Phase 0: 기반 정리 (모든 담당)
- [ ] `requirements.txt` merge conflict 해결
- [ ] `README.md` merge conflict 해결
- [ ] `extraction/detect_to_episodes.py` merge conflict 해결
- [ ] `scripts/pipeline/parallel_download.py` merge conflict 해결

### Phase 1: 중복 방지 + DB (C, A 담당)
- [ ] **C-1**: PostgreSQL 마이그레이션 스크립트 작성 + 스키마 생성
- [ ] **C-2**: Redis 설정 + 키 prefix 분리
- [ ] **A-1**: `GlobalVideoRegistry` 구현
- [ ] **A-1**: `mass_collector.py` 전 스테이지에 Registry 연동

### Phase 2: 크롤링 확장 (A 담당)
- [ ] **A-2**: 키워드 500개 확장 + 5언어 + 4소스
- [ ] **A-2**: async 크롤링 기본화 (200 concurrent)
- [ ] **A-3**: 다운로드 workers 12 + retry + bandwidth limit

### Phase 3: GPU 처리 최적화 (B 담당)
- [ ] **B-1**: `UnifiedVideoProcessor` (Detect+IL 1-Pass 통합)
- [ ] **B-1**: dual-GPU 6-stream 지원
- [ ] **B-2**: `CPUWorkerPool` (긴 영상 분리 처리)
- [ ] **B-3**: 품질 평가 배치 벡터화 + threshold 50점

### Phase 4: 대시보드 + 운영 (D, C 담당)
- [ ] **D-1**: 대시보드 DB 기반 전환 + 페이지네이션
- [ ] **D-2**: 크로스-런 모니터링 페이지
- [ ] **D-3**: 신규 알림 규칙 6개 추가
- [ ] **C-3**: 디스크 정책 매니저 + S3 업로드 후 자동 삭제
- [ ] **C-4**: Docker Compose + systemd 업데이트

---

## 5. 예상 성능 목표

| 지표 | 현재 (500개) | 목표 (5,000개) | 개선율 |
|------|------------|--------------|-------|
| 크롤링 속도 | 456개/235초 | 20,000개/300초 | ~30x |
| 다운로드 | 71개/~30분 | 10,000개/~3시간 | ~10x |
| GPU 처리 | 1,012개/6,254초 | 5,000개/~4시간 | ~2x (통합 1-pass) |
| 품질 통과율 | 48.8% | 65%+ | threshold 조정 |
| 업로드 | 533개/순차 | 5,000개/~1시간 | 8 workers |
| **총 파이프라인** | **~3.5시간/500개** | **~8시간/5,000개** | **10x 수집, 2.3x 시간** |
| 일일 최대 | 500개/일 | 15,000개/일 (3회) | 30x |
| 크로스-런 중복 | 체크 없음 | **0% 중복** | - |
| 디스크 사용 | 무한 축적 | 업로드 후 자동 삭제 | - |

---

## 6. 신규 파일 목록 (전체)

| 파일 | 담당 | 설명 |
|------|------|------|
| `cache/video_registry.py` | A | 글로벌 비디오 중복 방지 레지스트리 |
| `gpu/unified_processor.py` | B | Detect+IL 1-Pass 통합 프로세서 |
| `gpu/cpu_worker_pool.py` | B | CPU 전용 긴 영상 워커 풀 |
| `storage/disk_policy.py` | C | 디스크 사용 정책 매니저 |
| `scripts/tools/migrate_to_postgres.py` | C | SQLite → PostgreSQL 마이그레이션 |
| `deploy/redis.conf` | C | Redis 설정 파일 |
| `deploy/docker-compose.yml` | C | 컨테이너 오케스트레이션 |
| `deploy/Dockerfile` | C | P-ADE Docker 이미지 |

---

## 7. 수정 파일 목록 (전체, 담당별)

### A. 크롤링/수집 담당
| 파일 | 변경 요약 |
|------|----------|
| `ingestion/keyword_generator.py` | 카테시안 풀 500개, 성과 기반 우선순위 |
| `ingestion/multi_source_crawler.py` | async 기본, 4소스, Registry 필터 |
| `ingestion/async_crawler.py` | concurrent 200, 소스별 레이트 리밋 |
| `ingestion/downloader.py` | bandwidth throttle, 480p 모드 |
| `ingestion/quality_filter.py` | Registry rejected 연동 |
| `workers/crawl_worker.py` | 16 workers, Registry 체크 |
| `cache/redis_cache.py` | Bloom filter 1M, 키 prefix 분리 |
| `scripts/pipeline/parallel_download.py` | merge conflict 해결, 12 workers, retry 3 |

### B. GPU/처리 담당
| 파일 | 변경 요약 |
|------|----------|
| `gpu/stream_manager.py` | 6-stream dual-GPU, 동적 VRAM, 영상 분류 배칭 |
| `extraction/detect_to_episodes.py` | merge conflict 해결, unified_processor 호출 |
| `extraction/object_detector.py` | batch inference 모드 |
| `quality/evaluator.py` | evaluate_batch(), threshold 50, early reject |
| `scripts/pipeline/build_imitation_data.py` | unified_processor 연동 |
| `scripts/pipeline/filter_quality.py` | 배치 모드 연동 |
| `scripts/pipeline/encode_actions.py` | unified_processor 연동 |

### C. 인프라/데이터 담당
| 파일 | 변경 요약 |
|------|----------|
| `config/settings.py` | PostgreSQL 기본, Redis 설정 확장 |
| `core/database.py` | PostgreSQL 연결 풀 (pool_size=20) |
| `storage/storage_manager.py` | cleanup_after_upload() 추가 |
| `storage/s3_uploader.py` | 업로드 후 로컬 삭제 옵션 |
| `task_queue/task_queue.py` | 큐 모니터링, dead letter queue |
| `deploy/robot-collector.service` | target 5000, 메모리 16GB |
| `.env.example` | PostgreSQL, 확장 설정 |
| `requirements.txt` | merge conflict 해결, psycopg2 추가 |
| `README.md` | merge conflict 해결 |

### D. 대시보드/모니터링 담당
| 파일 | 변경 요약 |
|------|----------|
| `dashboard/web_app.py` | DB 기반 통계, SSE, runs API, dedup API |
| `dashboard/pages.py` | Runs 페이지, Dedup 위젯, 페이지네이션 |
| `monitor/alert_loop.py` | 6개 신규 알림 규칙 |
| `monitor/stats_collector.py` | 디스크/중복률/다운로드 속도 메트릭 |
| `dags/robot_collection_dag.py` | target 5000, 통합 태스크 |

### 공통 (모든 담당)
| 파일 | 변경 요약 |
|------|----------|
| `mass_collector.py` | PipelineConfig 전면 업데이트, STAGES 변경, Registry 연동, 통합 처리 |
| `main.py` | run_id 생성, 기본 target 5000 |

---

## 8. 검증 체크리스트

### 중복 방지 검증
- [ ] 동일 video_id로 2회 크롤 시 2회차에서 필터링 확인
- [ ] 동일 URL (다른 video_id) 크롤 시 url_hash로 필터링 확인
- [ ] 품질 탈락 영상 재수집 차단 확인
- [ ] Redis 재시작 후 PostgreSQL fallback 동작 확인
- [ ] 10,000개 이상 Registry에서 `filter_new_only()` 응답 < 100ms 확인

### 성능 검증
- [ ] 5,000개 크롤링 < 10분 확인
- [ ] 10,000개 다운로드 < 4시간 확인
- [ ] GPU 6-stream 안정 동작 확인 (OOM 없음)
- [ ] 통합 1-pass 처리 vs 분리 처리 시간 비교
- [ ] 디스크 사용량 파이프라인 전후 차이 < 50GB 확인

### 인프라 검증
- [ ] PostgreSQL 동시 쓰기 20 connections 안정성
- [ ] Redis 메모리 2GB 제한 내 동작
- [ ] Docker Compose up/down 정상
- [ ] systemd 서비스 재시작 + 로그 확인
