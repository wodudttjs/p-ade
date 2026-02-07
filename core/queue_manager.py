"""
Queue Manager - Redis 기반 작업 큐 시스템

MVP Phase 2 Week 5: Queue System
- Redis 작업 큐 구현
- 병렬 처리 (4 workers)
- 처리 속도 2배 개선
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """작업 단위"""
    task_id: str
    task_type: str  # download, extract, transform, upload
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "worker_id": self.worker_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            payload=data["payload"],
            priority=TaskPriority(data.get("priority", 1)),
            status=TaskStatus(data.get("status", "pending")),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            worker_id=data.get("worker_id"),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Task":
        return cls.from_dict(json.loads(json_str))


class QueueBackend(ABC):
    """큐 백엔드 추상 클래스"""
    
    @abstractmethod
    def enqueue(self, queue_name: str, task: Task) -> bool:
        """작업을 큐에 추가"""
        pass
    
    @abstractmethod
    def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[Task]:
        """큐에서 작업 가져오기"""
        pass
    
    @abstractmethod
    def get_queue_length(self, queue_name: str) -> int:
        """큐 길이 조회"""
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """작업 상태 조회"""
        pass
    
    @abstractmethod
    def update_task_status(self, task: Task) -> bool:
        """작업 상태 업데이트"""
        pass


class InMemoryQueue(QueueBackend):
    """인메모리 큐 (테스트/개발용)"""
    
    def __init__(self):
        self._queues: Dict[str, List[Task]] = {}
        self._tasks: Dict[str, Task] = {}
        logger.info("InMemoryQueue 초기화 완료")
    
    def enqueue(self, queue_name: str, task: Task) -> bool:
        if queue_name not in self._queues:
            self._queues[queue_name] = []
        
        task.status = TaskStatus.QUEUED
        self._queues[queue_name].append(task)
        self._tasks[task.task_id] = task
        
        # 우선순위 정렬 (높은 우선순위가 앞으로)
        self._queues[queue_name].sort(key=lambda t: -t.priority.value)
        
        logger.debug(f"Task {task.task_id} 큐에 추가됨 (queue={queue_name})")
        return True
    
    def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[Task]:
        if queue_name not in self._queues or not self._queues[queue_name]:
            return None
        
        task = self._queues[queue_name].pop(0)
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        return task
    
    def get_queue_length(self, queue_name: str) -> int:
        if queue_name not in self._queues:
            return 0
        return len(self._queues[queue_name])
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)
    
    def update_task_status(self, task: Task) -> bool:
        self._tasks[task.task_id] = task
        return True
    
    def clear(self, queue_name: str = None):
        """큐 비우기"""
        if queue_name:
            self._queues[queue_name] = []
        else:
            self._queues.clear()
            self._tasks.clear()


class RedisQueue(QueueBackend):
    """Redis 기반 큐"""
    
    QUEUE_PREFIX = "pade:queue:"
    TASK_PREFIX = "pade:task:"
    PROCESSING_PREFIX = "pade:processing:"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis 패키지가 설치되지 않았습니다: pip install redis")
        
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )
        
        # 연결 테스트
        try:
            self.client.ping()
            logger.info(f"Redis 연결 성공: {host}:{port}/{db}")
        except redis.ConnectionError as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise
    
    def _queue_key(self, queue_name: str) -> str:
        return f"{self.QUEUE_PREFIX}{queue_name}"
    
    def _task_key(self, task_id: str) -> str:
        return f"{self.TASK_PREFIX}{task_id}"
    
    def _processing_key(self, queue_name: str) -> str:
        return f"{self.PROCESSING_PREFIX}{queue_name}"
    
    def enqueue(self, queue_name: str, task: Task) -> bool:
        try:
            task.status = TaskStatus.QUEUED
            
            # 작업 데이터 저장
            self.client.set(
                self._task_key(task.task_id),
                task.to_json(),
                ex=86400 * 7  # 7일 TTL
            )
            
            # 우선순위 큐에 추가 (score = priority, 높을수록 먼저 처리)
            self.client.zadd(
                self._queue_key(queue_name),
                {task.task_id: task.priority.value}
            )
            
            logger.debug(f"Task {task.task_id} Redis 큐에 추가됨")
            return True
            
        except Exception as e:
            logger.error(f"Redis enqueue 실패: {e}")
            return False
    
    def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[Task]:
        try:
            queue_key = self._queue_key(queue_name)
            processing_key = self._processing_key(queue_name)
            
            if timeout > 0:
                # Blocking pop with timeout
                result = self.client.bzpopmax(queue_key, timeout=timeout)
                if not result:
                    return None
                _, task_id, _ = result
            else:
                # Non-blocking pop (가장 높은 우선순위)
                result = self.client.zpopmax(queue_key, count=1)
                if not result:
                    return None
                task_id, _ = result[0]
            
            # 작업 데이터 가져오기
            task_data = self.client.get(self._task_key(task_id))
            if not task_data:
                logger.warning(f"Task {task_id} 데이터 없음")
                return None
            
            task = Task.from_json(task_data)
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            
            # 처리 중 목록에 추가
            self.client.sadd(processing_key, task_id)
            
            # 상태 업데이트
            self.update_task_status(task)
            
            return task
            
        except Exception as e:
            logger.error(f"Redis dequeue 실패: {e}")
            return None
    
    def get_queue_length(self, queue_name: str) -> int:
        try:
            return self.client.zcard(self._queue_key(queue_name))
        except Exception:
            return 0
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        try:
            task_data = self.client.get(self._task_key(task_id))
            if task_data:
                return Task.from_json(task_data)
            return None
        except Exception as e:
            logger.error(f"Task 조회 실패: {e}")
            return None
    
    def update_task_status(self, task: Task) -> bool:
        try:
            self.client.set(
                self._task_key(task.task_id),
                task.to_json(),
                ex=86400 * 7
            )
            return True
        except Exception as e:
            logger.error(f"Task 업데이트 실패: {e}")
            return False
    
    def complete_task(self, queue_name: str, task: Task, result: Dict[str, Any] = None):
        """작업 완료 처리"""
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().isoformat()
        task.result = result
        
        self.update_task_status(task)
        self.client.srem(self._processing_key(queue_name), task.task_id)
    
    def fail_task(self, queue_name: str, task: Task, error: str, retry: bool = True):
        """작업 실패 처리"""
        task.error = error
        
        if retry and task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            
            # 재시도 큐에 다시 추가
            self.client.srem(self._processing_key(queue_name), task.task_id)
            self.enqueue(queue_name, task)
            logger.info(f"Task {task.task_id} 재시도 예약 ({task.retry_count}/{task.max_retries})")
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now().isoformat()
            self.update_task_status(task)
            self.client.srem(self._processing_key(queue_name), task.task_id)
    
    def get_processing_count(self, queue_name: str) -> int:
        """처리 중인 작업 수"""
        return self.client.scard(self._processing_key(queue_name))
    
    def get_stats(self, queue_name: str) -> Dict[str, Any]:
        """큐 통계"""
        return {
            "queue_name": queue_name,
            "pending": self.get_queue_length(queue_name),
            "processing": self.get_processing_count(queue_name),
        }
    
    def clear(self, queue_name: str = None):
        """큐 비우기"""
        if queue_name:
            self.client.delete(self._queue_key(queue_name))
            self.client.delete(self._processing_key(queue_name))
        else:
            # 모든 P-ADE 관련 키 삭제
            for key in self.client.scan_iter(f"{self.QUEUE_PREFIX}*"):
                self.client.delete(key)
            for key in self.client.scan_iter(f"{self.TASK_PREFIX}*"):
                self.client.delete(key)
            for key in self.client.scan_iter(f"{self.PROCESSING_PREFIX}*"):
                self.client.delete(key)


class QueueManager:
    """
    큐 매니저
    
    Redis가 없으면 InMemoryQueue로 폴백
    """
    
    QUEUE_DOWNLOAD = "download"
    QUEUE_EXTRACT = "extract"
    QUEUE_TRANSFORM = "transform"
    QUEUE_UPLOAD = "upload"
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_redis: bool = True,
    ):
        self.backend: QueueBackend
        
        # Redis URL 파싱 또는 환경변수에서 가져오기
        if use_redis:
            redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
            try:
                # redis://host:port/db 파싱
                if redis_url.startswith("redis://"):
                    parts = redis_url.replace("redis://", "").split("/")
                    host_port = parts[0].split(":")
                    host = host_port[0]
                    port = int(host_port[1]) if len(host_port) > 1 else 6379
                    db = int(parts[1]) if len(parts) > 1 else 0
                else:
                    host, port, db = "localhost", 6379, 0
                
                self.backend = RedisQueue(host=host, port=port, db=db)
                logger.info("Redis 큐 백엔드 사용")
                
            except Exception as e:
                logger.warning(f"Redis 연결 실패, InMemoryQueue 사용: {e}")
                self.backend = InMemoryQueue()
        else:
            self.backend = InMemoryQueue()
            logger.info("InMemory 큐 백엔드 사용")
    
    def create_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> Task:
        """새 작업 생성"""
        return Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
        )
    
    def submit_download(self, video_id: str, url: str, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """다운로드 작업 제출"""
        task = self.create_task(
            task_type="download",
            payload={"video_id": video_id, "url": url},
            priority=priority,
        )
        self.backend.enqueue(self.QUEUE_DOWNLOAD, task)
        return task.task_id
    
    def submit_extract(self, video_id: str, video_path: str, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """포즈 추출 작업 제출"""
        task = self.create_task(
            task_type="extract",
            payload={"video_id": video_id, "video_path": video_path},
            priority=priority,
        )
        self.backend.enqueue(self.QUEUE_EXTRACT, task)
        return task.task_id
    
    def submit_transform(self, video_id: str, pose_path: str, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """변환 작업 제출"""
        task = self.create_task(
            task_type="transform",
            payload={"video_id": video_id, "pose_path": pose_path},
            priority=priority,
        )
        self.backend.enqueue(self.QUEUE_TRANSFORM, task)
        return task.task_id
    
    def submit_upload(self, video_id: str, file_path: str, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """업로드 작업 제출"""
        task = self.create_task(
            task_type="upload",
            payload={"video_id": video_id, "file_path": file_path},
            priority=priority,
        )
        self.backend.enqueue(self.QUEUE_UPLOAD, task)
        return task.task_id
    
    def get_next_task(self, queue_name: str, timeout: int = 0) -> Optional[Task]:
        """다음 작업 가져오기"""
        return self.backend.dequeue(queue_name, timeout)
    
    def complete_task(self, queue_name: str, task: Task, result: Dict[str, Any] = None):
        """작업 완료 처리"""
        if isinstance(self.backend, RedisQueue):
            self.backend.complete_task(queue_name, task, result)
        else:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = result
            self.backend.update_task_status(task)
    
    def fail_task(self, queue_name: str, task: Task, error: str, retry: bool = True):
        """작업 실패 처리"""
        if isinstance(self.backend, RedisQueue):
            self.backend.fail_task(queue_name, task, error, retry)
        else:
            if retry and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.backend.enqueue(queue_name, task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now().isoformat()
                task.error = error
                self.backend.update_task_status(task)
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """작업 상태 조회"""
        return self.backend.get_task_status(task_id)
    
    def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """모든 큐 통계"""
        queues = [
            self.QUEUE_DOWNLOAD,
            self.QUEUE_EXTRACT,
            self.QUEUE_TRANSFORM,
            self.QUEUE_UPLOAD,
        ]
        
        stats = {}
        for queue_name in queues:
            if isinstance(self.backend, RedisQueue):
                stats[queue_name] = self.backend.get_stats(queue_name)
            else:
                stats[queue_name] = {
                    "queue_name": queue_name,
                    "pending": self.backend.get_queue_length(queue_name),
                    "processing": 0,
                }
        
        return stats
    
    def clear_all(self):
        """모든 큐 비우기"""
        self.backend.clear()


# 편의 함수
def get_queue_manager(use_redis: bool = True) -> QueueManager:
    """큐 매니저 싱글톤 가져오기"""
    if not hasattr(get_queue_manager, "_instance"):
        get_queue_manager._instance = QueueManager(use_redis=use_redis)
    return get_queue_manager._instance


if __name__ == "__main__":
    # 테스트
    print("=== Queue Manager 테스트 ===\n")
    
    # InMemory 테스트
    qm = QueueManager(use_redis=False)
    
    # 작업 제출
    task_id = qm.submit_download("test_video_1", "https://youtube.com/watch?v=test1")
    print(f"✅ 작업 제출됨: {task_id}")
    
    # 큐 상태
    stats = qm.get_queue_stats()
    print(f"📊 큐 상태: {stats}")
    
    # 작업 가져오기
    task = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
    if task:
        print(f"📥 작업 가져옴: {task.task_type} - {task.payload}")
        
        # 완료 처리
        qm.complete_task(QueueManager.QUEUE_DOWNLOAD, task, {"success": True})
        print(f"✅ 작업 완료 처리됨")
    
    # 최종 상태
    final_status = qm.get_task_status(task_id)
    print(f"📋 최종 상태: {final_status.status.value if final_status else 'Unknown'}")
    
    print("\n✅ 테스트 완료!")
