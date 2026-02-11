
로봇팔 영상 수집 시스템 최적화 Task 로드맵

📋 Task 우선순위 매트릭스
우선순위영향도난이도소요시간P0 (긴급)매우 높음낮음-중간1-3일P1 (높음)높음중간3-7일P2 (중간)중간중간-높음1-2주P3 (낮음)낮음높음2주+

🚀 Phase 1: 즉시 성과 (Quick Wins) - 1주일
Task 1.1: YouTube API 배치 요청 전환 ⭐⭐⭐⭐⭐
해결 문제: 웹크롤링 속도 (문제 1)
우선순위: P0 (긴급)
예상 소요: 1-2일
난이도: ★☆☆☆☆
작업 내용:
1. 기존 코드 분석
   - 현재 API 호출 방식 파악
   - 단일 호출 → 배치 호출 변환 지점 식별

2. 배치 요청 구현
   - videos.list API 수정
   - id 파라미터에 최대 50개 video_id 쉼표로 연결
   - 예: id="dQw4w9WgXcQ,jNQXAC9IVRw,..."

3. 코드 수정
   # 기존
   for video_id in video_ids:
       metadata = youtube.videos().list(id=video_id).execute()
   
   # 개선
   batch_size = 50
   for i in range(0, len(video_ids), batch_size):
       batch = video_ids[i:i+batch_size]
       metadata = youtube.videos().list(id=','.join(batch)).execute()

4. 할당량 계산 로직 수정
   - 기존: N개 × 1 unit = N units
   - 개선: N개 / 50 × 1 unit = N/50 units
필요 기술:

YouTube Data API v3
Python (기존 코드 수정)

예상 효과:

API 할당량 50배 절감
크롤링 속도 5-10배 향상
500개 크롤링: 1.5시간 → 15분

검증 방법:
bash# 테스트 실행
python test_batch_crawl.py --keywords "robot arm" --count 100

# 예상 결과
# - 소요 시간: < 3분
# - API units 사용: ~2 units (기존 100 units)
```

**의존성**: 없음 (독립 실행 가능)

---

### Task 1.2: 비동기 병렬 크롤링 구현 ⭐⭐⭐⭐⭐

**해결 문제**: 웹크롤링 속도 (문제 1)  
**우선순위**: P0 (긴급)  
**예상 소요**: 2-3일  
**난이도**: ★★★☆☆

**작업 내용**:
```
1. 라이브러리 설치
   pip install aiohttp aiodns asyncio

2. 비동기 크롤러 클래스 작성
   # crawler/async_youtube_crawler.py
   
   import asyncio
   import aiohttp
   
   class AsyncYouTubeCrawler:
       def __init__(self, max_concurrent=100):
           self.semaphore = asyncio.Semaphore(max_concurrent)
           self.session = None
       
       async def search_video(self, keyword):
           async with self.semaphore:
               # API 호출 또는 HTML 파싱
               async with self.session.get(url) as response:
                   return await response.json()
       
       async def crawl_batch(self, keywords):
           async with aiohttp.ClientSession() as session:
               self.session = session
               tasks = [self.search_video(k) for k in keywords]
               results = await asyncio.gather(*tasks)
           return results

3. 기존 동기 코드 교체
   # 기존
   results = [crawl(k) for k in keywords]
   
   # 개선
   results = asyncio.run(crawler.crawl_batch(keywords))

4. 에러 처리 강화
   - 타임아웃 설정 (30초)
   - 재시도 로직 (3회)
   - 실패한 키워드만 재수집
필요 기술:

Python asyncio
aiohttp
비동기 프로그래밍 개념

예상 효과:

동시 100개 요청 처리
네트워크 대기 시간 최소화
500개 크롤링: 15분 → 5분
Task 1.1과 결합 시: 1.5시간 → 2-3분

검증 방법:
bash# 성능 비교 테스트
python benchmark_crawler.py

# 예상 출력
# Sync crawler: 100 keywords in 180s
# Async crawler: 100 keywords in 15s
# Speedup: 12x
```

**의존성**: Task 1.1 완료 후 작업 권장 (병렬 효과 극대화)

---

### Task 1.3: Redis 캐싱 시스템 구축 ⭐⭐⭐⭐

**해결 문제**: 웹크롤링 속도 + 중복 방지 (문제 1, 6)  
**우선순위**: P0 (긴급)  
**예상 소요**: 1일  
**난이도**: ★★☆☆☆

**작업 내용**:
```
1. Redis 설치 및 설정
   # Ubuntu
   sudo apt install redis-server
   sudo systemctl start redis
   
   # Python 클라이언트
   pip install redis

2. 캐싱 레이어 구현
   # cache/redis_cache.py
   
   import redis
   import json
   import hashlib
   
   class CrawlCache:
       def __init__(self):
           self.r = redis.Redis(host='localhost', port=6379, db=0)
       
       def get_search_results(self, keyword):
           """검색 결과 캐시 조회"""
           key = f"search:{keyword}"
           cached = self.r.get(key)
           if cached:
               return json.loads(cached)
           return None
       
       def save_search_results(self, keyword, results, ttl=21600):
           """검색 결과 캐시 저장 (6시간 TTL)"""
           key = f"search:{keyword}"
           self.r.setex(key, ttl, json.dumps(results))
       
       def is_video_collected(self, video_id):
           """비디오 수집 여부 체크 (Bloom Filter)"""
           return self.r.getbit('collected_videos', self._hash(video_id))
       
       def mark_video_collected(self, video_id):
           """비디오 수집 마킹"""
           self.r.setbit('collected_videos', self._hash(video_id), 1)
       
       def _hash(self, video_id):
           """해시 함수"""
           return int(hashlib.sha256(video_id.encode()).hexdigest(), 16) % (10**9)

3. 크롤러에 캐시 통합
   cache = CrawlCache()
   
   # 캐시 먼저 확인
   results = cache.get_search_results(keyword)
   if not results:
       results = api_search(keyword)
       cache.save_search_results(keyword, results)
   
   # 중복 필터링
   new_videos = [v for v in results if not cache.is_video_collected(v['id'])]

4. 캐시 통계 모니터링
   - 히트율 추적
   - 캐시 크기 모니터링
   - 만료 정책 조정
필요 기술:

Redis
Python redis 라이브러리
캐싱 전략 이해

예상 효과:

재검색 시 즉시 반환 (0.001초)
중복 수집 완전 제거
API 할당량 추가 절감
2차 실행부터 10배 빠름

검증 방법:
bash# 캐시 테스트
python test_cache.py

# 1차 실행 (캐시 없음)
# Time: 180s

# 2차 실행 (캐시 히트)
# Time: 5s
# Cache hit rate: 95%
```

**의존성**: Task 1.1, 1.2와 독립적 (병렬 작업 가능)

---

### Task 1.4: Airflow DAG 기본 구조 생성 ⭐⭐⭐⭐

**해결 문제**: 파이프라인 자동화 (문제 3)  
**우선순위**: P1 (높음)  
**예상 소요**: 2-3일  
**난이도**: ★★★☆☆

**작업 내용**:
```
1. Airflow 설치 (로컬 환경)
   pip install apache-airflow==2.8.1
   
   # DB 초기화
   airflow db init
   
   # 관리자 계정 생성
   airflow users create \
       --username admin \
       --password admin \
       --firstname Admin \
       --lastname User \
       --role Admin \
       --email admin@example.com

2. DAG 디렉토리 설정
   mkdir -p ~/airflow/dags
   export AIRFLOW_HOME=~/airflow

3. 기본 DAG 작성
   # dags/robot_collection_dag.py
   
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta
   
   default_args = {
       'owner': 'robot-team',
       'retries': 3,
       'retry_delay': timedelta(minutes=5),
   }
   
   dag = DAG(
       'robot_arm_collection',
       default_args=default_args,
       description='Robot arm video collection pipeline',
       schedule_interval='0 6 * * *',  # 매일 6시
       start_date=datetime(2026, 2, 7),
       catchup=False,
   )
   
   # Task 정의 (간단한 버전)
   def crawl():
       from crawler import crawl_videos
       crawl_videos()
   
   def download():
       from downloader import download_batch
       download_batch()
   
   def process_gpu():
       from gpu_processor import process_all
       process_all()
   
   # Task 생성
   crawl_task = PythonOperator(
       task_id='crawl_videos',
       python_callable=crawl,
       dag=dag,
   )
   
   download_task = PythonOperator(
       task_id='download_videos',
       python_callable=download,
       dag=dag,
   )
   
   gpu_task = PythonOperator(
       task_id='gpu_processing',
       python_callable=process_gpu,
       dag=dag,
   )
   
   # 의존성
   crawl_task >> download_task >> gpu_task

4. Airflow 서비스 시작
   # 스케줄러
   airflow scheduler &
   
   # 웹서버
   airflow webserver --port 8080 &
   
   # 브라우저로 접속
   # http://localhost:8080

5. DAG 테스트
   # 수동 실행
   airflow dags test robot_arm_collection 2026-02-07
필요 기술:

Apache Airflow
Python
기본 DAG 개념

예상 효과:

파이프라인 시각화
스케줄링 자동화
실패 시 자동 재시도
웹 UI로 모니터링

검증 방법:
bash# DAG 확인
airflow dags list

# 수동 트리거
airflow dags trigger robot_arm_collection

# 로그 확인
airflow tasks logs robot_arm_collection crawl_videos 2026-02-07
```

**의존성**: 기존 파이프라인 코드가 모듈화되어 있어야 함