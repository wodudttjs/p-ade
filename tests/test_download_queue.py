"""
다운로드 큐 테스트
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch
from ingestion.download_queue import (
    DownloadQueue,
    DownloadJob,
    Priority
)


# Redis 연결 테스트를 위한 픽스처
@pytest.fixture
def mock_redis():
    """Mock Redis 클라이언트"""
    with patch('ingestion.download_queue.redis.Redis') as mock:
        redis_instance = Mock()
        redis_instance.ping.return_value = True
        redis_instance.set.return_value = True
        redis_instance.get.return_value = None
        redis_instance.zadd.return_value = 1
        redis_instance.zpopmax.return_value = []
        redis_instance.sadd.return_value = 1
        redis_instance.srem.return_value = 1
        redis_instance.smembers.return_value = set()
        redis_instance.zcard.return_value = 0
        redis_instance.scard.return_value = 0
        redis_instance.zrange.return_value = []
        redis_instance.zrevrange.return_value = []
        redis_instance.delete.return_value = 1
        
        mock.return_value = redis_instance
        yield redis_instance


@pytest.fixture
def queue(mock_redis):
    """DownloadQueue 인스턴스"""
    return DownloadQueue()


@pytest.fixture
def sample_job():
    """샘플 다운로드 작업"""
    return DownloadJob(
        job_id="test_job_1",
        video_id="abc123",
        video_url="https://youtube.com/watch?v=abc123",
        platform="youtube",
        priority=Priority.NORMAL,
        quality="1080p"
    )


def test_download_job_creation():
    """DownloadJob 생성 테스트"""
    job = DownloadJob(
        job_id="job1",
        video_id="vid1",
        video_url="https://example.com/video",
        platform="youtube"
    )
    
    assert job.job_id == "job1"
    assert job.video_id == "vid1"
    assert job.platform == "youtube"
    assert job.priority == Priority.NORMAL
    assert job.status == "pending"
    assert job.retry_count == 0
    assert job.max_retries == 3
    assert job.created_at is not None


def test_download_job_to_dict(sample_job):
    """DownloadJob 딕셔너리 변환"""
    job_dict = sample_job.to_dict()
    
    assert job_dict['job_id'] == "test_job_1"
    assert job_dict['video_id'] == "abc123"
    assert job_dict['platform'] == "youtube"
    assert job_dict['priority'] == Priority.NORMAL


def test_download_job_from_dict():
    """딕셔너리에서 DownloadJob 생성"""
    data = {
        'job_id': 'job2',
        'video_id': 'vid2',
        'video_url': 'https://example.com/video2',
        'platform': 'vimeo',
        'priority': Priority.HIGH,
        'quality': '720p',
        'created_at': '2024-01-01T00:00:00',
        'started_at': None,
        'completed_at': None,
        'status': 'pending',
        'retry_count': 0,
        'max_retries': 3,
        'error_message': None,
        'metadata': None,
    }
    
    job = DownloadJob.from_dict(data)
    
    assert job.job_id == 'job2'
    assert job.platform == 'vimeo'
    assert job.priority == Priority.HIGH


def test_priority_enum():
    """Priority enum 값 검증"""
    assert Priority.LOW == 1
    assert Priority.NORMAL == 2
    assert Priority.HIGH == 3
    assert Priority.URGENT == 4
    
    assert Priority.URGENT > Priority.HIGH
    assert Priority.HIGH > Priority.NORMAL


def test_queue_initialization(mock_redis):
    """큐 초기화 테스트"""
    queue = DownloadQueue(
        redis_host="localhost",
        redis_port=6379,
        redis_db=0
    )
    
    assert queue.redis_client is not None
    mock_redis.ping.assert_called_once()


def test_queue_connection_failure():
    """Redis 연결 실패"""
    with patch('ingestion.download_queue.redis.Redis') as mock:
        redis_instance = Mock()
        redis_instance.ping.side_effect = Exception("Connection refused")
        mock.return_value = redis_instance
        
        with pytest.raises(Exception):
            DownloadQueue()


def test_add_job(queue, mock_redis, sample_job):
    """작업 추가"""
    result = queue.add_job(sample_job)
    
    assert result is True
    
    # Redis 호출 검증
    assert mock_redis.set.called
    assert mock_redis.zadd.called


def test_get_next_job_empty_queue(queue, mock_redis):
    """빈 큐에서 작업 가져오기"""
    mock_redis.zpopmax.return_value = []
    
    job = queue.get_next_job()
    
    assert job is None


def test_get_next_job_success(queue, mock_redis, sample_job):
    """작업 가져오기 성공"""
    import json
    
    job_data = json.dumps(sample_job.to_dict())
    mock_redis.zpopmax.return_value = [("test_job_1", Priority.NORMAL)]
    mock_redis.get.return_value = job_data
    
    job = queue.get_next_job()
    
    assert job is not None
    assert job.job_id == "test_job_1"
    assert job.status == "processing"
    assert job.started_at is not None


def test_complete_job(queue, mock_redis, sample_job):
    """작업 완료 처리"""
    import json
    
    sample_job.status = "processing"
    job_data = json.dumps(sample_job.to_dict())
    mock_redis.get.return_value = job_data
    
    metadata = {'filepath': '/path/to/video.mp4', 'size': 1024000}
    result = queue.complete_job("test_job_1", metadata=metadata)
    
    assert result is True
    assert mock_redis.set.called
    assert mock_redis.srem.called
    assert mock_redis.sadd.called


def test_complete_job_not_found(queue, mock_redis):
    """존재하지 않는 작업 완료 처리"""
    mock_redis.get.return_value = None
    
    result = queue.complete_job("nonexistent_job")
    
    assert result is False


def test_fail_job_with_retry(queue, mock_redis, sample_job):
    """작업 실패 - 재시도"""
    import json
    
    sample_job.retry_count = 0
    job_data = json.dumps(sample_job.to_dict())
    mock_redis.get.return_value = job_data
    
    result = queue.fail_job("test_job_1", "Network timeout", retry=True)
    
    assert result is True
    # 재시도를 위해 큐에 다시 추가
    assert mock_redis.zadd.called


def test_fail_job_max_retries(queue, mock_redis, sample_job):
    """작업 실패 - 최대 재시도 초과"""
    import json
    
    sample_job.retry_count = 3  # 최대치 도달
    job_data = json.dumps(sample_job.to_dict())
    mock_redis.get.return_value = job_data
    
    result = queue.fail_job("test_job_1", "Permanent error", retry=True)
    
    assert result is True
    # 실패 목록에 추가
    assert mock_redis.sadd.called


def test_get_job(queue, mock_redis, sample_job):
    """작업 조회"""
    import json
    
    job_data = json.dumps(sample_job.to_dict())
    mock_redis.get.return_value = job_data
    
    job = queue.get_job("test_job_1")
    
    assert job is not None
    assert job.job_id == "test_job_1"


def test_get_job_not_found(queue, mock_redis):
    """존재하지 않는 작업 조회"""
    mock_redis.get.return_value = None
    
    job = queue.get_job("nonexistent")
    
    assert job is None


def test_get_queue_size(queue, mock_redis):
    """대기 중 작업 수"""
    mock_redis.zcard.return_value = 5
    
    size = queue.get_queue_size()
    
    assert size == 5


def test_get_processing_count(queue, mock_redis):
    """처리 중 작업 수"""
    mock_redis.scard.return_value = 3
    
    count = queue.get_processing_count()
    
    assert count == 3


def test_get_stats(queue, mock_redis):
    """큐 통계"""
    mock_redis.zcard.return_value = 5  # pending
    mock_redis.scard.side_effect = [3, 10, 2]  # processing, completed, failed
    
    stats = queue.get_stats()
    
    assert stats['pending'] == 5
    assert stats['processing'] == 3
    assert stats['completed'] == 10
    assert stats['failed'] == 2


def test_list_jobs_by_status_pending(queue, mock_redis, sample_job):
    """대기 중 작업 목록"""
    import json
    
    mock_redis.zrevrange.return_value = ["test_job_1"]
    mock_redis.get.return_value = json.dumps(sample_job.to_dict())
    
    jobs = queue.list_jobs_by_status("pending", limit=10)
    
    assert len(jobs) == 1
    assert jobs[0].job_id == "test_job_1"


def test_list_jobs_by_status_invalid(queue, mock_redis):
    """잘못된 상태로 조회"""
    jobs = queue.list_jobs_by_status("invalid_status")
    
    assert len(jobs) == 0


def test_clear_completed(queue, mock_redis, sample_job):
    """완료된 작업 정리"""
    import json
    
    sample_job.status = "completed"
    sample_job.completed_at = datetime.now().isoformat()
    
    mock_redis.smembers.return_value = {"test_job_1"}
    mock_redis.get.return_value = json.dumps(sample_job.to_dict())
    
    deleted = queue.clear_completed()
    
    assert deleted == 1
    assert mock_redis.delete.called


def test_reset_all(queue, mock_redis):
    """모든 큐 초기화"""
    mock_redis.zrange.return_value = ["job1"]
    mock_redis.smembers.side_effect = [{"job2"}, {"job3"}, {"job4"}]
    
    result = queue.reset_all()
    
    assert result is True
    assert mock_redis.delete.called
