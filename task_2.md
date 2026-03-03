## 🏗️ Phase 2: 핵심 기능 구축 - 1-2주

### Task 2.1: 멀티프로세스 크롤러 구현 ⭐⭐⭐⭐

**해결 문제**: 웹크롤링 속도 (문제 1)  
**우선순위**: P1 (높음)  
**예상 소요**: 2-3일  
**난이도**: ★★★☆☆

**작업 내용**:
```
1. 멀티프로세싱 아키텍처 설계
   
   [Master Process]
        ├─ 키워드 분배
        ├─ 작업 큐 관리 (Redis)
        └─ 결과 수집
        
   [Worker Process 1-8]
        ├─ 독립적 크롤링
        ├─ 결과 Redis에 저장
        └─ 진행률 보고

2. Redis 작업 큐 구현
   # queue/task_queue.py
   
   import redis
   import json
   
   class CrawlTaskQueue:
       def __init__(self):
           self.r = redis.Redis()
       
       def enqueue_keywords(self, keywords):
           """키워드를 큐에 추가"""
           for keyword in keywords:
               self.r.rpush('crawl_queue', keyword)
       
       def dequeue_keyword(self, timeout=5):
           """키워드 가져오기 (블로킹)"""
           result = self.r.blpop('crawl_queue', timeout)
           return result[1].decode() if result else None
       
       def mark_complete(self, keyword, results):
           """완료 마킹"""
           self.r.hset('crawl_results', keyword, json.dumps(results))

3. 워커 프로세스 구현
   # workers/crawl_worker.py
   
   import multiprocessing as mp
   from queue import CrawlTaskQueue
   from crawler import AsyncYouTubeCrawler
   
   def worker_loop(worker_id):
       """워커 메인 루프"""
       print(f"Worker {worker_id} started")
       
       queue = CrawlTaskQueue()
       crawler = AsyncYouTubeCrawler()
       
       while True:
           # 키워드 가져오기
           keyword = queue.dequeue_keyword(timeout=5)
           
           if not keyword:
               break  # 큐가 비면 종료
           
           # 크롤링
           results = crawler.search(keyword)
           
           # 결과 저장
           queue.mark_complete(keyword, results)
           
           print(f"Worker {worker_id}: {keyword} -> {len(results)} videos")

4. 마스터 프로세스 구현
   # master.py
   
   import multiprocessing as mp
   from queue import CrawlTaskQueue
   
   def run_multiprocess_crawl(keywords, num_workers=8):
       """멀티프로세스 크롤링 실행"""
       
       # 큐에 키워드 추가
       queue = CrawlTaskQueue()
       queue.enqueue_keywords(keywords)
       
       # 워커 프로세스 시작
       processes = []
       for i in range(num_workers):
           p = mp.Process(target=worker_loop, args=(i,))
           p.start()
           processes.append(p)
       
       # 완료 대기
       for p in processes:
           p.join()
       
       print("All workers completed")

5. CPU 코어 수 자동 감지
   num_workers = mp.cpu_count()  # 예: 8 코어
   run_multiprocess_crawl(keywords, num_workers)
필요 기술:

Python multiprocessing
Redis (프로세스 간 통신)
동시성 개념

예상 효과:

8코어 기준 8배 속도 향상
500개 크롤링: 3분 → 30초
누적 효과 (Task 1.1+1.2+2.1): 1.5시간 → 20초

검증 방법:
bash# 성능 테스트
python benchmark_multiprocess.py --workers 8 --keywords 1000

# 예상 출력
# Single process: 180s
# 8 workers: 25s
# Speedup: 7.2x
```

**의존성**: Task 1.3 (Redis) 필수

---

### Task 2.2: GPU 3-Stream 병렬 처리 최적화 ⭐⭐⭐⭐⭐

**해결 문제**: GPU 활용률 극대화  
**우선순위**: P0 (긴급)  
**예상 소요**: 3-4일  
**난이도**: ★★★★☆

**작업 내용**:
```
1. CUDA Stream 기반 병렬 처리 구현
   # gpu/stream_manager.py
   
   import torch
   import torch.cuda as cuda
   from concurrent.futures import ThreadPoolExecutor
   
   class GPU3StreamManager:
       def __init__(self):
           self.device = torch.device('cuda:0')
           self.num_streams = 3
           
           # CUDA Stream 생성
           self.streams = [cuda.Stream() for _ in range(3)]
           
           # MediaPipe 인스턴스 (각 스트림별)
           self.estimators = [
               MediaPipePoseEstimator(device='cuda:0')
               for _ in range(3)
           ]
       
       def process_batch(self, video_paths):
           """3개 영상 동시 처리"""
           
           # 3개씩 묶음
           batch_size = 3
           results = []
           
           for i in range(0, len(video_paths), batch_size):
               batch = video_paths[i:i+batch_size]
               
               # 병렬 처리
               with ThreadPoolExecutor(max_workers=3) as executor:
                   futures = []
                   
                   for stream_id, video_path in enumerate(batch):
                       future = executor.submit(
                           self._process_single,
                           video_path,
                           stream_id
                       )
                       futures.append(future)
                   
                   # 결과 수집
                   batch_results = [f.result() for f in futures]
                   results.extend(batch_results)
           
           return results
       
       def _process_single(self, video_path, stream_id):
           """단일 스트림에서 처리"""
           
           # 해당 스트림 사용
           with cuda.stream(self.streams[stream_id]):
               estimator = self.estimators[stream_id]
               sequence = estimator.process_video(video_path)
               return sequence

2. VRAM 사용량 모니터링
   def monitor_vram():
       """VRAM 사용량 체크"""
       import torch
       
       allocated = torch.cuda.memory_allocated() / 1024**3  # GB
       reserved = torch.cuda.memory_reserved() / 1024**3
       
       print(f"VRAM Allocated: {allocated:.2f} GB")
       print(f"VRAM Reserved: {reserved:.2f} GB")
       
       if allocated > 9.0:  # 10GB 중 9GB 이상
           print("WARNING: High VRAM usage!")

3. 배치 크기 자동 조정
   def auto_adjust_batch_size():
       """VRAM 여유에 따라 배치 크기 조정"""
       
       allocated = torch.cuda.memory_allocated() / 1024**3
       
       if allocated < 6.0:
           return 4  # 여유 있음, 4-stream 시도
       elif allocated < 8.0:
           return 3  # 정상, 3-stream
       else:
           return 2  # 부족, 2-stream으로 축소

4. 프레임 샘플링 최적화
   # 60초 영상 @ 30fps = 1800 프레임
   # → 너무 많으면 VRAM 부족
   
   # 해결: 적응형 샘플링
   if video_duration > 60:
       target_fps = 15  # 긴 영상은 15fps로 다운샘플
   else:
       target_fps = 30

5. 처리 큐 관리
   # queue/processing_queue.py
   
   class ProcessingQueue:
       def __init__(self):
           self.r = redis.Redis()
       
       def pop_batch(self, batch_size=3):
           """3개 배치 가져오기"""
           batch = []
           for _ in range(batch_size):
               item = self.r.lpop('processing_queue')
               if item:
                   batch.append(json.loads(item))
           return batch
필요 기술:

PyTorch CUDA
CUDA Stream 개념
ThreadPoolExecutor
메모리 관리

예상 효과:

GPU 사용률: 60% → 90%
처리 속도: 60개/시간 → 90개/시간 (1.5배)
일일 처리: 1,440개 → 2,160개
목표 1,500개 안정적 달성

검증 방법:
bash# GPU 모니터링
nvidia-smi dmon -s um -c 60

# 처리 벤치마크
python benchmark_gpu.py --videos 100

# 예상 결과
# Single stream: 100 videos in 150 min
# 3-stream: 100 videos in 67 min
# GPU utilization: 88%
```

**의존성**: 기존 MediaPipe 코드 필요

---

### Task 2.3: 품질 평가 시스템 구현 ⭐⭐⭐⭐⭐

**해결 문제**: 데이터 품질 테스트 (문제 5)  
**우선순위**: P0 (긴급)  
**예상 소요**: 4-5일  
**난이도**: ★★★★☆

**작업 내용**:
```
1. 품질 평가자 클래스 구현
   # quality/evaluator.py
   
   (앞서 작성한 RobotArmQualityEvaluator 코드 사용)

2. 4-DOF 관절 검출 로직 구현
   def check_4dof_joints(sequence):
       """
       4-DOF 체크:
       - Shoulder (어깨)
       - Elbow (팔꿈치)
       - Wrist (손목)
       - Gripper (그리퍼/손)
       """
       
       joint_indices = {
           'shoulder': [11, 12],  # 양쪽 어깨
           'elbow': [13, 14],     # 양쪽 팔꿈치
           'wrist': [15, 16],     # 양쪽 손목
           'gripper': [19, 20]    # 양쪽 손끝
       }
       
       results = {}
       for joint_name, indices in joint_indices.items():
           confidences = []
           
           for frame in sequence.frames:
               if not frame.body_landmarks:
                   continue
               
               # 양쪽 중 높은 confidence 사용
               conf = max([
                   frame.body_landmarks[idx].visibility
                   for idx in indices
               ])
               confidences.append(conf)
           
           # 평균 > 0.5면 검출 성공
           results[joint_name] = np.mean(confidences) > 0.5
       
       return results

3. 파지 동작 감지
   def detect_grasping(sequence):
       """손가락 간격 변화로 파지 감지"""
       
       if not sequence.frames[0].right_hand_landmarks:
           return False
       
       finger_distances = []
       
       for frame in sequence.frames:
           hand = frame.right_hand_landmarks or frame.left_hand_landmarks
           if not hand:
               continue
           
           # 엄지-검지 거리
           thumb = hand[4]
           index = hand[8]
           dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
           finger_distances.append(dist)
       
       if not finger_distances:
           return False
       
       # 거리 변화 > 0.15 이면 파지 동작
       variation = max(finger_distances) - min(finger_distances)
       return variation > 0.15

4. 품질 필터 파이프라인 통합
   # pipeline/quality_filter.py
   
   def process_with_quality_filter(video_path, sequence):
       """품질 평가 및 필터링"""
       
       evaluator = RobotArmQualityEvaluator()
       result = evaluator.evaluate(video_path, sequence)
       
       print(f"Quality Score: {result['total_score']}/100")
       print(f"Grade: {result['grade']}")
       
       if result['passed']:  # 60점 이상
           # .npz로 저장
           save_as_npz(sequence, result)
           return True
       else:
           # 삭제
           os.remove(video_path)
           return False

5. 품질 통계 수집
   # stats/quality_stats.py
   
   class QualityStats:
       def __init__(self):
           self.grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
           self.total = 0
       
       def record(self, grade):
           self.grades[grade] += 1
           self.total += 1
       
       def print_report(self):
           print("\n=== Quality Report ===")
           for grade in ['A', 'B', 'C', 'D', 'F']:
               count = self.grades[grade]
               pct = count / self.total * 100 if self.total > 0 else 0
               print(f"Grade {grade}: {count} ({pct:.1f}%)")
           
           passed = sum(self.grades[g] for g in ['A', 'B', 'C'])
           pass_rate = passed / self.total * 100 if self.total > 0 else 0
           print(f"\nPass Rate (≥60): {pass_rate:.1f}%")
필요 기술:

NumPy (배열 연산)
MediaPipe 랜드마크 이해
통계 분석

예상 효과:

4-DOF 학습에 적합한 데이터만 선별
불량 데이터 자동 제거 (30-40%)
학습 효율 향상
저장 공간 절약

검증 방법:
bash# 샘플 테스트
python test_quality.py --sample 100

# 예상 출력
# Processed: 100 videos
# Grade A: 15 (15%)
# Grade B: 30 (30%)
# Grade C: 25 (25%)
# Grade D: 15 (15%)
# Grade F: 15 (15%)
# Pass rate: 70%
```

**의존성**: MediaPipe 처리 완료 후

---

### Task 2.4: FastAPI 웹 대시보드 구축 ⭐⭐⭐⭐

**해결 문제**: 파이프라인 모니터링 (문제 3)  
**우선순위**: P1 (높음)  
**예상 소요**: 3-4일  
**난이도**: ★★★☆☆

**작업 내용**:
```
1. FastAPI 서버 구현
   # api/server.py
   
   from fastapi import FastAPI, WebSocket
   from fastapi.responses import HTMLResponse
   from fastapi.staticfiles import StaticFiles
   import asyncio
   import redis
   
   app = FastAPI()
   app.mount("/static", StaticFiles(directory="static"), name="static")
   
   r = redis.Redis()
   
   @app.get("/")
   async def dashboard():
       with open("static/index.html") as f:
           return HTMLResponse(f.read())
   
   @app.get("/api/stats")
   async def get_stats():
       """실시간 통계"""
       return {
           "collected_today": int(r.get("collected_today") or 0),
           "target": 1500,
           "crawl_speed": float(r.get("crawl_speed") or 0),
           "download_speed": float(r.get("download_speed") or 0),
           "gpu_util": float(r.get("gpu_util") or 0),
           "queue_download": r.llen("download_queue"),
           "queue_processing": r.llen("processing_queue"),
       }
   
   @app.post("/api/control/{action}")
   async def control(action: str):
       """파이프라인 제어"""
       if action == "start":
           r.set("pipeline_status", "running")
       elif action == "stop":
           r.set("pipeline_status", "stopped")
       return {"status": "ok"}
   
   @app.websocket("/ws/logs")
   async def logs_stream(websocket: WebSocket):
       """실시간 로그"""
       await websocket.accept()
       pubsub = r.pubsub()
       pubsub.subscribe("logs")
       
       try:
           while True:
               message = pubsub.get_message()
               if message and message['type'] == 'message':
                   await websocket.send_text(message['data'].decode())
               await asyncio.sleep(0.1)
       except:
           await websocket.close()

2. HTML 대시보드 작성
   # static/index.html
   
   <!DOCTYPE html>
   <html>
   <head>
       <title>Robot Arm Collection Dashboard</title>
       <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
       <style>
           body { font-family: Arial; margin: 20px; }
           .card { border: 1px solid #ccc; padding: 20px; margin: 10px 0; }
           .progress-bar { width: 100%; height: 30px; background: #f0f0f0; }
           .progress-fill { height: 100%; background: #4CAF50; }
           button { padding: 10px 20px; margin: 5px; }
       </style>
   </head>
   <body>
       <h1>🤖 Robot Arm Collection Dashboard</h1>
       
       <div class="card">
           <h2>Progress</h2>
           <div>
               <span id="collected">0</span> / <span id="target">1500</span>
               (<span id="percent">0%</span>)
           </div>
           <div class="progress-bar">
               <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
           </div>
       </div>
       
       <div class="card">
           <h2>Control</h2>
           <button onclick="control('start')">▶️ Start</button>
           <button onclick="control('stop')">⏹️ Stop</button>
           <button onclick="control('restart')">🔄 Restart</button>
       </div>
       
       <div class="card">
           <h2>Speeds</h2>
           <div>Crawling: <span id="crawl-speed">0</span> videos/min</div>
           <div>Download: <span id="download-speed">0</span> videos/min</div>
           <div>GPU Utilization: <span id="gpu-util">0</span>%</div>
       </div>
       
       <div class="card">
           <h2>Live Logs</h2>
           <pre id="logs" style="height: 300px; overflow-y: scroll; background: #000; color: #0f0; padding: 10px;"></pre>
       </div>
       
       <script>
           // 1초마다 통계 업데이트
           setInterval(async () => {
               const res = await fetch('/api/stats');
               const data = await res.json();
               
               document.getElementById('collected').innerText = data.collected_today;
               document.getElementById('target').innerText = data.target;
               
               const percent = (data.collected_today / data.target * 100).toFixed(1);
               document.getElementById('percent').innerText = percent + '%';
               document.getElementById('progress-fill').style.width = percent + '%';
               
               document.getElementById('crawl-speed').innerText = data.crawl_speed.toFixed(1);
               document.getElementById('download-speed').innerText = data.download_speed.toFixed(1);
               document.getElementById('gpu-util').innerText = data.gpu_util.toFixed(1);
           }, 1000);
           
           // 제어 함수
           async function control(action) {
               await fetch(`/api/control/${action}`, { method: 'POST' });
               alert(`Pipeline ${action} command sent`);
           }
           
           // WebSocket 로그 스트리밍
           const ws = new WebSocket('ws://localhost:8000/ws/logs');
           ws.onmessage = (event) => {
               const logs = document.getElementById('logs');
               logs.innerText += event.data + '\n';
               logs.scrollTop = logs.scrollHeight;
           };
       </script>
   </body>
   </html>

3. 통계 수집 워커
   # monitor/stats_collector.py
   
   import redis
   import time
   import psutil
   import subprocess
   
   r = redis.Redis()
   
   def collect_stats():
       """주기적 통계 수집"""
       while True:
           # 크롤링 속도 계산
           prev_count = int(r.get("prev_crawled") or 0)
           curr_count = int(r.get("total_crawled") or 0)
           crawl_speed = (curr_count - prev_count) * 60  # per minute
           r.set("crawl_speed", crawl_speed)
           r.set("prev_crawled", curr_count)
           
           # GPU 사용률
           gpu_util = get_gpu_utilization()
           r.set("gpu_util", gpu_util)
           
           time.sleep(1)
   
   def get_gpu_utilization():
       """nvidia-smi로 GPU 사용률 조회"""
       try:
           output = subprocess.check_output([
               'nvidia-smi',
               '--query-gpu=utilization.gpu',
               '--format=csv,noheader,nounits'
           ])
           return float(output.decode().strip())
       except:
           return 0.0

4. 서버 실행
   # 백그라운드 실행
   nohup uvicorn api.server:app --host 0.0.0.0 --port 8000 > dashboard.log 2>&1 &
   
   # 통계 수집 워커
   nohup python monitor/stats_collector.py > stats.log 2>&1 &
필요 기술:

FastAPI
WebSocket
HTML/CSS/JavaScript
Chart.js (선택)

예상 효과:

실시간 진행률 확인
원격 모니터링 가능
수동 제어 가능
문제 조기 발견