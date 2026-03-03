🔵 Developer A: 크롤링/수집 담당
📋 작업 목표

크롤링: 456개 → 20,000개 (44배)
다운로드: 71개 → 10,000개 (141배)
핵심: 실행 간 중복 0% 보장


Phase 0: 사전 준비 (4시간)
✅ Task A0-1: Merge Conflict 해결
우선순위: P0 (긴급)
시간: 2시간
파일: scripts/pipeline/parallel_download.py, requirements.txt
작업 내용:

parallel_download.py Git 충돌 해결
workers 수, timeout 설정 병합
테스트 실행 (샘플 10개 다운로드)
requirements.txt에서 크롤링 관련 라이브러리 확인

aiohttp, playwright, yt-dlp 버전 체크



검증:

샘플 다운로드 10개 정상 완료
pip install -r requirements.txt 에러 없음


📊 Task A0-2: 현재 성능 프로파일링
우선순위: P1
시간: 2시간
측정 항목:

크롤링 성능:

50 키워드 → 456개 URL 수집 시간: 235초
키워드당 평균 수집량 계산
YouTube API vs 스크래핑 속도 비교


다운로드 성능:

6 workers로 71개 다운로드 시간
평균 영상 크기 측정 (MB)
네트워크 대역폭 사용률



산출물:

profiling_report.md 작성
최적 workers 수 계산 결과
병목 지점 3가지 이상 식별


Phase 1: 글로벌 중복 방지 시스템 (Day 1-2)
⭐⭐⭐⭐⭐ Task A1-1: GlobalVideoRegistry 구현 (최우선)
우선순위: P0
시간: 1.5일
파일: cache/video_registry.py (신규)
의존성: C팀의 Redis 설정 완료 필요
구현 내용:
1. Redis 레지스트리 구조 설계
Redis Keys:
- "pade:registry:videos" → Set[video_id]
- "pade:registry:urls" → Set[url_hash]  
- "pade:registry:rejected" → Set[video_id]

특징:
- TTL 없음 (영구 보존)
- RDB + AOF 백업
2. 핵심 메서드 구현

is_collected(video_id) → bool 체크
is_url_collected(url) → URL 정규화 후 체크
filter_new_only(video_list) → 미수집 영상만 필터링
register(video_id, url, run_id) → 수집 완료 등록
register_rejected(video_id, reason) → 품질 탈락 등록
get_stats() → 전체 통계 (총 수집, 중복 차단)

3. PostgreSQL Fallback

Redis 장애 시 DB에서 자동 조회
video_registry 테이블 쿼리
성능: Redis 조회 < 1ms, DB 조회 < 10ms

4. 동기화 로직

Redis → PostgreSQL 비동기 동기화 (1분마다)
시스템 시작 시 DB → Redis 로드
데이터 일관성 보장

검증:

10,000개 video_id 등록 후 조회 속도 < 5ms
Redis 재시작 후 PostgreSQL에서 자동 복구
중복 영상 100% 차단 확인


🔗 Task A1-2: MassCollector 연동
우선순위: P0
시간: 0.5일
파일: mass_collector.py
변경 지점:
1. _stage_crawl() 수정
위치: mass_collector.py:336-371

작업:
1. 크롤링 결과 받은 직후 Registry 필터 적용
2. videos = registry.filter_new_only(crawled_videos)
3. 필터링 통계 로깅 (중복 차단 수, 신규 수)
2. _stage_download() 수정
위치: mass_collector.py:479-487

작업:
1. 다운로드 전 video_id 재확인
2. if registry.is_collected(video_id): skip
3. 기존 파일 시스템 glob 제거 (성능 개선)
3. _stage_quality() 수정
위치: mass_collector.py:798-883

작업:
1. 품질 탈락 시 Registry에 등록
2. registry.register_rejected(video_id, reason)
3. 재수집 방지
4. _stage_upload() 수정
위치: mass_collector.py:885-950

작업:
1. S3 업로드 성공 시 Registry 등록
2. registry.register(video_id, url, run_id, s3_path)
3. 수집 완료 마킹
검증:

1회차 실행: 500개 수집
2회차 실행: 동일 키워드 사용 → 중복 500개 모두 차단
로그에서 "Already collected: 500, New: 0" 확인


Phase 2: 크롤링 10배 확장 (Day 3-5)
🚀 Task A2-1: 키워드 500개 확장
우선순위: P0
시간: 1일
파일: ingestion/keyword_generator.py, mass_collector.py
작업 내용:
1. 키워드 생성 확장
현재: 50개 키워드
목표: 500개 키워드

방법:
1. generate_cartesian_all() 사용
   - 6,000개 카테시안 풀 생성
   - 상위 500개 선택
   
2. 우선순위 기준:
   - 과거 성공률 높은 키워드
   - 다양한 카테고리 분포
   - 5개 언어 균등 분배
2. mass_collector.py 설정 변경
위치: mass_collector.py:62-123

변경:
- max_keywords: 50 → 500
- crawl_multiplier: 3.0 → 4.0
- languages: ["en", "ko"] → ["en", "ko", "ja", "zh", "de"]
- sources: ["youtube", "google_videos"] → 
           ["youtube", "google_videos", "vimeo", "bilibili"]
3. 크롤 타겟 계산
목표: 5,000개 수집
통과율: 50% 예상
필요 다운로드: 10,000개
필요 크롤: 20,000개 (여유분 포함)

계산:
- 500 키워드 × 40개/키워드 = 20,000개
- crawl_multiplier = 4.0
검증:

키워드 500개 생성 확인
5개 언어, 4개 소스 고르게 분포
크롤링 타겟 20,000개 설정 확인


⚡ Task A2-2: 비동기 크롤링 구현
우선순위: P0
시간: 1일
파일: ingestion/async_crawler.py, ingestion/multi_source_crawler.py
작업 내용:
1. 비동기 크롤러 설정 강화
파일: ingestion/async_crawler.py

변경:
- max_concurrent: 100 → 200
- timeout: 30초 유지
- 소스별 독립 레이트 리밋
  - YouTube API: 10 req/sec
  - Google Videos: 5 req/sec
  - Vimeo: 3 req/sec
  - Bilibili: 2 req/sec
2. MultiSourceCrawler 기본 모드 변경
파일: ingestion/multi_source_crawler.py

변경:
1. crawl() 메서드에서 async_mode=True 기본값
2. 4개 소스 병렬 크롤링
3. 각 소스별 결과 병합
4. Registry 필터 적용
3. YouTube Batch API 활용
파일: ingestion/youtube_batch.py

최적화:
- 배치 사이즈 50 유지
- video.list API 배치 요청
- 쿼터 사용량 모니터링 강화
성능 목표:

500 키워드 → 20,000개 URL 크롤링
소요 시간: < 10분 (현재 235초 → 600초)
크롤링 속도: 2,000개/분

검증:

100 키워드로 테스트 (4,000개 크롤 기대)
10분 내 완료 확인
에러율 < 5%


📥 Task A2-3: 다운로드 시스템 강화
우선순항: P0
시간: 1일
파일: scripts/pipeline/parallel_download.py, ingestion/downloader.py
작업 내용:
1. workers 수 증가
파일: scripts/pipeline/parallel_download.py

변경:
- num_workers: 6 → 12
- ThreadPoolExecutor max_workers=12
- 동시 다운로드 12개
2. Retry 로직 추가
작업:
1. 다운로드 실패 시 3회 재시도
2. 지수 백오프: 2초, 4초, 8초
3. 영구 실패 시 건너뛰기 (무한 대기 방지)
3. Timeout 단축
변경:
- timeout: 600초 → 300초
- 빠른 실패 → 재시도로 안정성 확보
- 멈춘 다운로드 조기 감지
4. Bandwidth 제한
파일: ingestion/downloader.py

추가 옵션:
- yt-dlp 옵션에 rate_limit 추가
- 영상당 5MB/s 제한
- 총 대역폭: 12 workers × 5MB/s = 60MB/s
5. 품질 옵션 조정
변경:
- 기본 품질: 720p → 480p
- 대량 수집 시 용량 절약
- 설정으로 전환 가능 (config.download_quality)
6. 디스크 체크 추가
작업:
1. 다운로드 전 여유 공간 확인
2. 최소 100GB 필요
3. 부족 시 다운로드 중단 + 알림
성능 목표:

10,000개 다운로드
소요 시간: < 4시간
다운로드 속도: 42개/분

검증:

100개 샘플 다운로드 (12 workers)
평균 소요 시간 측정
에러율 < 10%
디스크 공간 부족 시나리오 테스트


Phase 3: 기타 최적화 (Day 6-7)
🔧 Task A3-1: 품질 필터 연동
우선순위: P1
시간: 0.5일
파일: ingestion/quality_filter.py
작업 내용:

Registry에 rejected 영상 등록
재크롤 시 자동 제외
rejected 사유 로깅


📝 Task A3-2: 크롤링 통계 강화
우선순위: P2
시간: 0.5일
작업 내용:

키워드별 성공률 추적
소스별 수집량 통계
언어별 분포 확인
성과 낮은 키워드 자동 비활성화


📊 Developer A 최종 체크리스트
필수 완료 항목

 GlobalVideoRegistry 구현 및 테스트
 mass_collector.py 전 스테이지 연동
 키워드 500개 확장
 비동기 크롤링 200 concurrent
 다운로드 12 workers + retry
 중복 방지 100% 검증

성능 목표

 크롤링: 20,000개 / 10분
 다운로드: 10,000개 / 4시간
 실행 간 중복: 0%