"""
Redis 기반 크롤링 작업 큐

멀티프로세스 크롤링을 위한 작업 배분 및 결과 수집을 담당합니다.
"""

import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CrawlTask:
    """크롤링 작업"""
    keyword: str
    source: str = "youtube"
    max_results: int = 50
    priority: int = 0  # 높을수록 우선


class CrawlTaskQueue:
    """
    크롤링 작업 큐
    
    Redis를 사용하여 멀티프로세스 간 작업을 배분합니다.
    
    사용법:
        queue = CrawlTaskQueue()
        
        # 마스터: 키워드 추가
        queue.enqueue_keywords(["robot arm", "pick and place"])
        
        # 워커: 작업 가져오기
        task = queue.dequeue_keyword(timeout=5)
        if task:
            results = crawl(task)
            queue.mark_complete(task, results)
    """
    
    # Redis 키 이름
    QUEUE_KEY = "pade:crawl_queue"
    RESULTS_KEY = "pade:crawl_results"
    STATS_KEY = "pade:crawl_stats"
    PROGRESS_KEY = "pade:crawl_progress"
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._connected = False
        self._client: Optional[redis.Redis] = None
        
        if REDIS_AVAILABLE:
            try:
                self._client = redis.Redis(
                    host=host, port=port, db=db,
                    decode_responses=True,
                    socket_timeout=5,
                )
                self._client.ping()
                self._connected = True
            except:
                pass
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # =========================================================================
    # 큐 관리
    # =========================================================================
    
    def enqueue_keywords(self, keywords: List[str], source: str = "youtube"):
        """키워드들을 큐에 추가"""
        if not self._connected:
            return 0
        
        count = 0
        for keyword in keywords:
            task = json.dumps({
                "keyword": keyword,
                "source": source,
                "enqueued_at": time.time(),
            })
            self._client.rpush(self.QUEUE_KEY, task)
            count += 1
        
        # 총 작업 수 기록
        self._client.hincrby(self.STATS_KEY, "total_enqueued", count)
        return count
    
    def enqueue_task(self, task: CrawlTask):
        """단일 작업 추가"""
        if not self._connected:
            return False
        
        task_data = json.dumps({
            "keyword": task.keyword,
            "source": task.source,
            "max_results": task.max_results,
            "priority": task.priority,
            "enqueued_at": time.time(),
        })
        
        if task.priority > 0:
            # 우선순위 높으면 앞에 추가
            self._client.lpush(self.QUEUE_KEY, task_data)
        else:
            self._client.rpush(self.QUEUE_KEY, task_data)
        
        self._client.hincrby(self.STATS_KEY, "total_enqueued", 1)
        return True
    
    def dequeue_keyword(self, timeout: int = 5) -> Optional[Dict]:
        """
        큐에서 작업 가져오기 (블로킹)
        
        Args:
            timeout: 대기 시간 (초)
            
        Returns:
            작업 정보 dict 또는 None
        """
        if not self._connected:
            return None
        
        try:
            result = self._client.blpop(self.QUEUE_KEY, timeout=timeout)
            if result:
                _, task_data = result
                return json.loads(task_data)
        except:
            pass
        
        return None
    
    def mark_complete(self, keyword: str, results: List[Dict], source: str = "youtube"):
        """작업 완료 마킹 및 결과 저장"""
        if not self._connected:
            return
        
        key = f"{source}:{keyword}"
        self._client.hset(self.RESULTS_KEY, key, json.dumps({
            "keyword": keyword,
            "source": source,
            "count": len(results),
            "results": results,
            "completed_at": time.time(),
        }))
        
        # 통계 업데이트
        self._client.hincrby(self.STATS_KEY, "total_completed", 1)
        self._client.hincrby(self.STATS_KEY, "total_results", len(results))
    
    def mark_failed(self, keyword: str, error: str, source: str = "youtube"):
        """작업 실패 마킹 + Dead Letter Queue 기록"""
        if not self._connected:
            return
        
        key = f"{source}:{keyword}"
        fail_payload = json.dumps({
            "keyword": keyword,
            "source": source,
            "error": error,
            "failed_at": time.time(),
        })
        self._client.hset(self.RESULTS_KEY, key, fail_payload)
        
        # Dead Letter Queue에 실패 작업 추가
        dlq_key = f"pade:dlq:{self.QUEUE_KEY}"
        self._client.lpush(dlq_key, fail_payload)
        
        self._client.hincrby(self.STATS_KEY, "total_failed", 1)
    
    def get_dlq(self, limit: int = 100) -> List[Dict]:
        """Dead Letter Queue 조회"""
        if not self._connected:
            return []
        dlq_key = f"pade:dlq:{self.QUEUE_KEY}"
        items = self._client.lrange(dlq_key, 0, limit - 1)
        return [json.loads(item) for item in items]
    
    def retry_dlq(self, count: int = 0) -> int:
        """DLQ 항목을 다시 큐에 넣기. count=0이면 전체"""
        if not self._connected:
            return 0
        dlq_key = f"pade:dlq:{self.QUEUE_KEY}"
        total = self._client.llen(dlq_key)
        to_retry = total if count == 0 else min(count, total)
        retried = 0
        for _ in range(to_retry):
            item = self._client.rpop(dlq_key)
            if item is None:
                break
            data = json.loads(item)
            self.push(data.get("keyword", ""), data.get("source", "youtube"))
            retried += 1
        return retried
    
    # =========================================================================
    # 상태 조회
    # =========================================================================
    
    def queue_length(self) -> int:
        """큐에 남은 작업 수"""
        if not self._connected:
            return 0
        return self._client.llen(self.QUEUE_KEY)
    
    def get_stats(self) -> Dict[str, int]:
        """통계 조회"""
        if not self._connected:
            return {}
        
        stats = self._client.hgetall(self.STATS_KEY)
        return {k: int(v) for k, v in stats.items()}
    
    def get_results(self, keyword: str, source: str = "youtube") -> Optional[Dict]:
        """특정 키워드 결과 조회"""
        if not self._connected:
            return None
        
        key = f"{source}:{keyword}"
        data = self._client.hget(self.RESULTS_KEY, key)
        return json.loads(data) if data else None
    
    def get_all_results(self) -> Dict[str, Dict]:
        """모든 결과 조회"""
        if not self._connected:
            return {}
        
        all_data = self._client.hgetall(self.RESULTS_KEY)
        return {k: json.loads(v) for k, v in all_data.items()}
    
    # =========================================================================
    # 진행률 추적
    # =========================================================================
    
    def report_progress(self, worker_id: int, status: str, current_keyword: str = ""):
        """워커 진행 상태 보고"""
        if not self._connected:
            return
        
        self._client.hset(self.PROGRESS_KEY, f"worker:{worker_id}", json.dumps({
            "status": status,
            "keyword": current_keyword,
            "updated_at": time.time(),
        }))
    
    def get_workers_status(self) -> Dict[str, Dict]:
        """모든 워커 상태 조회"""
        if not self._connected:
            return {}
        
        all_data = self._client.hgetall(self.PROGRESS_KEY)
        return {k: json.loads(v) for k, v in all_data.items()}
    
    # =========================================================================
    # 정리
    # =========================================================================
    
    def clear_queue(self):
        """큐 비우기"""
        if self._connected:
            self._client.delete(self.QUEUE_KEY)
    
    def clear_results(self):
        """결과 비우기"""
        if self._connected:
            self._client.delete(self.RESULTS_KEY)
    
    def clear_all(self):
        """모든 데이터 비우기"""
        if self._connected:
            self._client.delete(self.QUEUE_KEY, self.RESULTS_KEY, 
                              self.STATS_KEY, self.PROGRESS_KEY)


class ProcessingQueue(CrawlTaskQueue):
    """
    GPU 처리 작업 큐
    
    다운로드 완료된 영상의 GPU 처리를 위한 큐입니다.
    """
    
    QUEUE_KEY = "pade:processing_queue"
    RESULTS_KEY = "pade:processing_results"
    STATS_KEY = "pade:processing_stats"
    PROGRESS_KEY = "pade:processing_progress"
    
    def enqueue_video(self, video_path: str, video_id: str, priority: int = 0):
        """비디오를 처리 큐에 추가"""
        if not self._connected:
            return False
        
        task = json.dumps({
            "video_path": video_path,
            "video_id": video_id,
            "priority": priority,
            "enqueued_at": time.time(),
        })
        
        if priority > 0:
            self._client.lpush(self.QUEUE_KEY, task)
        else:
            self._client.rpush(self.QUEUE_KEY, task)
        
        return True
    
    def pop_batch(self, batch_size: int = 3) -> List[Dict]:
        """배치로 가져오기 (GPU 3-stream용)"""
        if not self._connected:
            return []
        
        batch = []
        for _ in range(batch_size):
            item = self._client.lpop(self.QUEUE_KEY)
            if item:
                batch.append(json.loads(item))
            else:
                break
        
        return batch
    
    def mark_video_complete(self, video_id: str, output_path: str, quality_score: float):
        """비디오 처리 완료"""
        if not self._connected:
            return
        
        self._client.hset(self.RESULTS_KEY, video_id, json.dumps({
            "video_id": video_id,
            "output_path": output_path,
            "quality_score": quality_score,
            "completed_at": time.time(),
        }))
        
        self._client.hincrby(self.STATS_KEY, "total_processed", 1)
