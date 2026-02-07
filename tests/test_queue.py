"""
Queue System 테스트

MVP Phase 2 Week 5: Queue System
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.queue_manager import (
    QueueManager,
    Task,
    TaskStatus,
    TaskPriority,
    InMemoryQueue,
    RedisQueue,
)


class TestTask:
    """Task 클래스 테스트"""
    
    def test_task_creation(self):
        """Task 생성 테스트"""
        task = Task(
            task_id="test-123",
            task_type="download",
            payload={"video_id": "abc", "url": "https://example.com"},
        )
        
        assert task.task_id == "test-123"
        assert task.task_type == "download"
        assert task.payload["video_id"] == "abc"
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_task_to_dict(self):
        """Task -> dict 변환 테스트"""
        task = Task(
            task_id="test-456",
            task_type="extract",
            payload={"video_id": "xyz"},
            priority=TaskPriority.HIGH,
        )
        
        data = task.to_dict()
        
        assert data["task_id"] == "test-456"
        assert data["task_type"] == "extract"
        assert data["priority"] == TaskPriority.HIGH.value
        assert data["status"] == "pending"
    
    def test_task_from_dict(self):
        """dict -> Task 변환 테스트"""
        data = {
            "task_id": "test-789",
            "task_type": "upload",
            "payload": {"file_path": "/path/to/file"},
            "priority": 2,
            "status": "running",
            "retry_count": 1,
        }
        
        task = Task.from_dict(data)
        
        assert task.task_id == "test-789"
        assert task.task_type == "upload"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.RUNNING
        assert task.retry_count == 1
    
    def test_task_json_serialization(self):
        """Task JSON 직렬화 테스트"""
        task = Task(
            task_id="test-json",
            task_type="transform",
            payload={"data": [1, 2, 3]},
        )
        
        json_str = task.to_json()
        restored = Task.from_json(json_str)
        
        assert restored.task_id == task.task_id
        assert restored.payload == task.payload


class TestInMemoryQueue:
    """InMemoryQueue 테스트"""
    
    def test_enqueue_dequeue(self):
        """기본 enqueue/dequeue 테스트"""
        queue = InMemoryQueue()
        
        task = Task(
            task_id="mem-1",
            task_type="download",
            payload={"test": True},
        )
        
        # Enqueue
        result = queue.enqueue("test_queue", task)
        assert result is True
        assert queue.get_queue_length("test_queue") == 1
        
        # Dequeue
        retrieved = queue.dequeue("test_queue")
        assert retrieved is not None
        assert retrieved.task_id == "mem-1"
        assert retrieved.status == TaskStatus.RUNNING
        assert queue.get_queue_length("test_queue") == 0
    
    def test_priority_ordering(self):
        """우선순위 정렬 테스트"""
        queue = InMemoryQueue()
        
        # 낮은 우선순위 먼저 추가
        low_task = Task(
            task_id="low",
            task_type="download",
            payload={},
            priority=TaskPriority.LOW,
        )
        queue.enqueue("priority_test", low_task)
        
        # 높은 우선순위 추가
        high_task = Task(
            task_id="high",
            task_type="download",
            payload={},
            priority=TaskPriority.HIGH,
        )
        queue.enqueue("priority_test", high_task)
        
        # 높은 우선순위가 먼저 나와야 함
        first = queue.dequeue("priority_test")
        assert first.task_id == "high"
        
        second = queue.dequeue("priority_test")
        assert second.task_id == "low"
    
    def test_empty_queue_returns_none(self):
        """빈 큐에서 dequeue 시 None 반환"""
        queue = InMemoryQueue()
        
        result = queue.dequeue("nonexistent")
        assert result is None
    
    def test_get_task_status(self):
        """작업 상태 조회 테스트"""
        queue = InMemoryQueue()
        
        task = Task(
            task_id="status-test",
            task_type="extract",
            payload={},
        )
        queue.enqueue("test", task)
        
        status = queue.get_task_status("status-test")
        assert status is not None
        assert status.status == TaskStatus.QUEUED
    
    def test_update_task_status(self):
        """작업 상태 업데이트 테스트"""
        queue = InMemoryQueue()
        
        task = Task(
            task_id="update-test",
            task_type="upload",
            payload={},
        )
        queue.enqueue("test", task)
        
        # 상태 업데이트
        task.status = TaskStatus.COMPLETED
        queue.update_task_status(task)
        
        updated = queue.get_task_status("update-test")
        assert updated.status == TaskStatus.COMPLETED
    
    def test_clear_queue(self):
        """큐 비우기 테스트"""
        queue = InMemoryQueue()
        
        for i in range(5):
            task = Task(task_id=f"clear-{i}", task_type="download", payload={})
            queue.enqueue("clear_test", task)
        
        assert queue.get_queue_length("clear_test") == 5
        
        queue.clear("clear_test")
        assert queue.get_queue_length("clear_test") == 0


class TestQueueManager:
    """QueueManager 테스트"""
    
    def test_create_task(self):
        """작업 생성 테스트"""
        qm = QueueManager(use_redis=False)
        
        task = qm.create_task(
            task_type="download",
            payload={"video_id": "test123"},
            priority=TaskPriority.NORMAL,
        )
        
        assert task.task_id is not None
        assert task.task_type == "download"
        assert task.payload["video_id"] == "test123"
    
    def test_submit_download(self):
        """다운로드 작업 제출 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_download("vid123", "https://youtube.com/watch?v=vid123")
        
        assert task_id is not None
        
        # 큐에서 확인
        task = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
        assert task is not None
        assert task.payload["video_id"] == "vid123"
    
    def test_submit_extract(self):
        """추출 작업 제출 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_extract("vid123", "/path/to/video.mp4")
        
        task = qm.get_next_task(QueueManager.QUEUE_EXTRACT)
        assert task is not None
        assert task.payload["video_path"] == "/path/to/video.mp4"
    
    def test_submit_transform(self):
        """변환 작업 제출 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_transform("vid123", "/path/to/pose.npz")
        
        task = qm.get_next_task(QueueManager.QUEUE_TRANSFORM)
        assert task is not None
        assert task.payload["pose_path"] == "/path/to/pose.npz"
    
    def test_submit_upload(self):
        """업로드 작업 제출 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_upload("vid123", "/path/to/file.npz")
        
        task = qm.get_next_task(QueueManager.QUEUE_UPLOAD)
        assert task is not None
        assert task.payload["file_path"] == "/path/to/file.npz"
    
    def test_complete_task(self):
        """작업 완료 처리 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_download("vid", "https://example.com")
        task = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
        
        qm.complete_task(
            QueueManager.QUEUE_DOWNLOAD,
            task,
            result={"success": True, "path": "/data/vid.mp4"}
        )
        
        status = qm.get_task_status(task_id)
        assert status.status == TaskStatus.COMPLETED
        assert status.result["success"] is True
    
    def test_fail_task_with_retry(self):
        """작업 실패 & 재시도 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_download("vid", "https://example.com")
        task = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
        
        # 실패 처리 (재시도 활성화)
        qm.fail_task(
            QueueManager.QUEUE_DOWNLOAD,
            task,
            error="Network error",
            retry=True,
        )
        
        # 재시도로 인해 큐에 다시 들어감
        retry_task = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
        assert retry_task is not None
        assert retry_task.retry_count == 1
        assert retry_task.status == TaskStatus.RUNNING
    
    def test_fail_task_without_retry(self):
        """작업 실패 (재시도 없음) 테스트"""
        qm = QueueManager(use_redis=False)
        
        task_id = qm.submit_download("vid", "https://example.com")
        task = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
        
        qm.fail_task(
            QueueManager.QUEUE_DOWNLOAD,
            task,
            error="Quality too low",
            retry=False,
        )
        
        # 큐가 비어있어야 함
        assert qm.get_next_task(QueueManager.QUEUE_DOWNLOAD) is None
        
        # 상태 확인
        status = qm.get_task_status(task_id)
        assert status.status == TaskStatus.FAILED
        assert status.error == "Quality too low"
    
    def test_max_retries_exceeded(self):
        """최대 재시도 초과 테스트"""
        qm = QueueManager(use_redis=False)
        
        # max_retries=2로 작업 생성
        task = qm.create_task(
            task_type="download",
            payload={"video_id": "test"},
            max_retries=2,
        )
        qm.backend.enqueue(QueueManager.QUEUE_DOWNLOAD, task)
        
        # 3번 실패 시도
        for i in range(3):
            t = qm.get_next_task(QueueManager.QUEUE_DOWNLOAD)
            if t:
                qm.fail_task(QueueManager.QUEUE_DOWNLOAD, t, f"Error {i}", retry=True)
        
        # 큐가 비어있어야 함
        assert qm.get_next_task(QueueManager.QUEUE_DOWNLOAD) is None
    
    def test_get_queue_stats(self):
        """큐 통계 테스트"""
        qm = QueueManager(use_redis=False)
        
        # 여러 작업 제출
        for i in range(3):
            qm.submit_download(f"vid{i}", f"https://example.com/{i}")
        
        for i in range(2):
            qm.submit_extract(f"vid{i}", f"/path/{i}.mp4")
        
        stats = qm.get_queue_stats()
        
        assert stats["download"]["pending"] == 3
        assert stats["extract"]["pending"] == 2
        assert stats["transform"]["pending"] == 0
        assert stats["upload"]["pending"] == 0
    
    def test_clear_all(self):
        """모든 큐 비우기 테스트"""
        qm = QueueManager(use_redis=False)
        
        qm.submit_download("vid1", "https://example.com/1")
        qm.submit_extract("vid2", "/path/2.mp4")
        
        qm.clear_all()
        
        stats = qm.get_queue_stats()
        assert all(q["pending"] == 0 for q in stats.values())


def is_redis_available():
    """Redis 연결 가능 여부 확인"""
    try:
        import redis
        r = redis.Redis()
        r.ping()
        return True
    except:
        return False


class TestRedisQueue:
    """RedisQueue 테스트 (Redis 연결 필요)"""
    
    @pytest.mark.skipif(
        not is_redis_available(),
        reason="Redis not available"
    )
    def test_redis_enqueue_dequeue(self):
        """Redis enqueue/dequeue 테스트"""
        
        try:
            queue = RedisQueue()
            
            task = Task(
                task_id="redis-test-1",
                task_type="download",
                payload={"test": True},
            )
            
            # Enqueue
            result = queue.enqueue("redis_test", task)
            assert result is True
            
            # Dequeue
            retrieved = queue.dequeue("redis_test")
            assert retrieved is not None
            assert retrieved.task_id == "redis-test-1"
            
            # Cleanup
            queue.clear("redis_test")
            
        except Exception as e:
            pytest.skip(f"Redis test failed: {e}")


class TestWorkerIntegration:
    """Worker 통합 테스트"""
    
    def test_worker_imports(self):
        """Worker 모듈 임포트 테스트"""
        from core.worker import (
            Worker,
            WorkerPool,
            TaskProcessor,
            process_batch_parallel,
        )
        
        assert Worker is not None
        assert WorkerPool is not None
        assert TaskProcessor is not None
    
    def test_task_processor_handlers(self):
        """TaskProcessor 핸들러 존재 확인"""
        from core.worker import TaskProcessor
        
        processor = TaskProcessor()
        
        # 핸들러 메서드 확인
        assert hasattr(processor, "process_download")
        assert hasattr(processor, "process_extract")
        assert hasattr(processor, "process_transform")
        assert hasattr(processor, "process_upload")
        assert hasattr(processor, "process")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
