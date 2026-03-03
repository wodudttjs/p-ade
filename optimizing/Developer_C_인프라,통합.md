🟡 Developer C: 인프라/데이터 담당
📋 작업 목표

SQLite → PostgreSQL 마이그레이션
Redis 인프라 강화
디스크 자동 관리 시스템
Docker/배포 환경 구축


Phase 0: 사전 준비 (4시간)
✅ Task C0-1: Merge Conflict 해결
우선순위: P0
시간: 2시간
파일: requirements.txt, README.md
작업 내용:

requirements.txt 충돌 해결

모든 팀의 의존성 통합
버전 충돌 해결


README.md 충돌 해결

설치 가이드 통합



검증:

pip install -r requirements.txt 정상 완료
README 설치 가이드 따라하기 테스트


📊 Task C0-2: 현재 인프라 상태 점검
우선순위: P1
시간: 2시간
점검 항목:

SQLite 현황:

DB 파일 크기
테이블 구조 파악
동시 쓰기 제한 확인


Redis 현황:

사용 중인 키 패턴
메모리 사용량
영속성 설정 (RDB/AOF)


디스크 사용량:

data/raw 크기
data/episodes 크기
축적 속도 계산



산출물:

infrastructure_audit.md
개선 포인트 3가지 이상 식별


Phase 1: PostgreSQL 마이그레이션 (Day 1-2)
⭐⭐⭐⭐⭐ Task C1-1: PostgreSQL 설치 및 초기화
우선순위: P0
시간: 0.5일
작업 내용:
1. PostgreSQL 설치
OS별 설치:
- Ubuntu: apt install postgresql-14
- macOS: brew install postgresql@14
- Docker: postgres:14-alpine 이미지
2. DB 및 사용자 생성
작업:
1. CREATE DATABASE pade;
2. CREATE USER pade WITH PASSWORD 'pade';
3. GRANT ALL PRIVILEGES ON DATABASE pade TO pade;
3. 연결 설정
파일: config/settings.py

변경:
DATABASE_URL:
  AS-IS: sqlite:///data/pade.db
  TO-BE: postgresql://pade:pade@localhost:5432/pade

환경변수:
  POSTGRES_HOST=localhost
  POSTGRES_PORT=5432
  POSTGRES_DB=pade
  POSTGRES_USER=pade
  POSTGRES_PASSWORD=pade
검증:

psql로 연결 확인
테이블 생성 권한 확인


🔧 Task C1-2: 스키마 정의 및 마이그레이션 스크립트
우선순위: P0
시간: 1일
파일: scripts/tools/migrate_to_postgres.py (신규)
작업 내용:
1. 새 테이블 추가
sql-- video_registry (A팀 중복 방지용)
CREATE TABLE video_registry (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(64) UNIQUE NOT NULL,
    url_hash VARCHAR(64) NOT NULL,
    url TEXT,
    platform VARCHAR(32),
    status VARCHAR(16) DEFAULT 'collected',
    run_id VARCHAR(64),
    collected_at TIMESTAMP DEFAULT NOW(),
    quality_score FLOAT,
    rejection_reason TEXT,
    s3_path TEXT
);

CREATE INDEX idx_video_registry_video_id ON video_registry(video_id);
CREATE INDEX idx_video_registry_url_hash ON video_registry(url_hash);
CREATE INDEX idx_video_registry_status ON video_registry(status);

-- pipeline_runs (D팀 모니터링용)
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

#### 2. 마이그레이션 스크립트 작성
```
기능:
1. SQLite에서 기존 데이터 읽기
2. PostgreSQL 스키마 생성
3. 데이터 복사
4. 인덱스 생성
5. 검증 (행 수 비교)
```

#### 3. 실행 및 검증
```
명령:
python scripts/tools/migrate_to_postgres.py \
  --sqlite data/pade.db \
  --postgres postgresql://pade:pade@localhost:5432/pade \
  --verify
```

**검증**:
- 모든 테이블 데이터 정상 복사
- 인덱스 생성 확인
- 쿼리 성능 테스트

---

### 🔗 Task C1-3: core/database.py 수정
**우선순위**: P0  
**시간**: 0.5일  
**파일**: `core/database.py`

**작업 내용**:

#### 1. 연결 풀 설정
```
설정:
- pool_size=20 (기본 연결 20개)
- max_overflow=30 (최대 50개 연결)
- pool_timeout=30
- pool_recycle=3600 (1시간마다 재연결)
```

#### 2. 자동 재연결
```
작업:
- 연결 끊김 감지
- 자동 재연결 시도
- 에러 로깅
```

#### 3. 트랜잭션 관리
```
추가:
- 컨텍스트 매니저 (with session:)
- 자동 commit/rollback
- 데드락 재시도
```

**검증**:
- 20개 동시 쓰기 테스트
- 연결 끊김 시나리오 테스트

---

## Phase 2: Redis 인프라 강화 (Day 3-4)

### 🔧 Task C2-1: Redis 설정 최적화
**우선순위**: P1  
**시간**: 0.5일  
**파일**: `deploy/redis.conf` (신규)

**작업 내용**:

#### 1. redis.conf 작성
```
핵심 설정:
- maxmemory 2gb
- maxmemory-policy allkeys-lru
- save 900 1 (Registry 영구 보존)
- save 300 10
- appendonly yes (AOF 활성화)
- appendfsync everysec
```

#### 2. 키 prefix 정책
```
분류:
- pade:registry:* → 영구 보존 (중복 방지)
- pade:cache:* → TTL 6시간 (검색 결과)
- pade:queue:* → 작업 큐
- pade:stats:* → 실시간 통계
```

#### 3. 영속성 설정
```
방법:
- RDB: 15분마다 변경 시 저장
- AOF: 1초마다 fsync
- Registry는 양쪽 모두 적용
```

**검증**:
- Redis 재시작 후 데이터 복구 확인
- 메모리 사용량 2GB 이하 유지

---

### 🔗 Task C2-2: cache/redis_cache.py 수정
**우선순위**: P1  
**시간**: 0.5일  
**파일**: `cache/redis_cache.py`

**작업 내용**:

#### 1. Bloom Filter 확장
```
변경:
- 크기: 100K → 1M
- False positive rate: 0.01 유지
- TTL 제거 (영구 보존)
```

#### 2. 연결 풀 설정
```
설정:
- max_connections=50
- socket_timeout=5
- socket_connect_timeout=5
- retry_on_timeout=True
```

#### 3. 키 prefix 분리
```
메서드:
- set_registry(key, value)
- set_cache(key, value, ttl)
- set_queue(key, value)
각각 prefix 자동 추가
```

**검증**:
- 1M개 Bloom Filter 성능 테스트
- 연결 풀 안정성 확인

---

## Phase 3: 디스크 관리 시스템 (Day 5-6)

### 🗂️ Task C3-1: DiskPolicy 구현
**우선순위**: P1  
**시간**: 1일  
**파일**: `storage/disk_policy.py` (신규)

**작업 내용**:

#### 1. 디스크 정책 정의
```
정책:
1. 다운로드 전: 여유 공간 체크 (최소 100GB)
2. 처리 완료 후: 원본 MP4 삭제 대기 (data/raw)
3. S3 업로드 확인 후: 로컬 NPZ 삭제
4. 품질 탈락: NPZ 즉시 삭제
5. 파이프라인 시작 시: 이전 실행 잔여파일 정리
```

#### 2. DiskPolicy 클래스 구현
```
메서드:
- ensure_space(required_gb) → bool
- cleanup_after_upload(run_id)
- cleanup_rejected()
- cleanup_old_runs(days=7)
- get_disk_usage() → Dict
```

#### 3. 라이프사이클 관리
```
흐름:
[다운로드] → data/raw/{video_id}.mp4
[처리] → data/episodes/{video_id}_episode.npz
[업로드] → S3 확인 후 로컬 삭제
[정리] → 원본 MP4 삭제
```

**검증**:
- 디스크 공간 부족 시나리오
- S3 업로드 후 자동 삭제 확인
- 파이프라인 재시작 시 정리 확인

---

### 🔗 Task C3-2: storage_manager.py 연동
**우선순위**: P1  
**시간**: 0.5일  
**파일**: `storage/storage_manager.py`

**작업 내용**:

#### 1. cleanup_after_upload() 추가
```
메서드:
- S3 업로드 성공 확인
- 로컬 파일 존재 확인
- 안전하게 삭제
- 로그 기록
```

#### 2. mass_collector 연동
```
호출 지점:
1. 다운로드 전: ensure_space() 체크
2. 업로드 후: cleanup_after_upload() 호출
3. 파이프라인 시작: cleanup_old_runs() 호출
검증:

5,000개 처리 후 디스크 사용량 < 50GB
S3와 로컬 파일 일치 확인