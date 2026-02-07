"""
다운로드 큐 관리자

Redis 기반 우선순위 큐를 사용한 비디오 다운로드 작업 관리
"""

import redis
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import IntEnum

from core.logging_config import logger


class Priority(IntEnum):
    """작업 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class DownloadJob:
    """다운로드 작업"""
    job_id: str
    video_id: str
    video_url: str
    platform: str
    priority: int = Priority.NORMAL
    quality: str = "1080p"
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DownloadJob':
        """딕셔너리에서 생성"""
        return cls(**data)


class DownloadQueue:
    """Redis 기반 다운로드 큐"""
    
    QUEUE_KEY = "p-ade:download:queue"
    PROCESSING_KEY = "p-ade:download:processing"
    COMPLETED_KEY = "p-ade:download:completed"
    FAILED_KEY = "p-ade:download:failed"
    JOB_PREFIX = "p-ade:download:job:"
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
    ):
        """
        Args:
            redis_host: Redis 호스트
            redis_port: Redis 포트
            redis_db: Redis DB 번호
            redis_password: Redis 비밀번호
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
        )
        
        # 연결 테스트
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def add_job(self, job: DownloadJob) -> bool:
        """
        작업 추가
        
        Args:
            job: 다운로드 작업
            
        Returns:
            성공 여부
        """
        try:
            # 작업 데이터 저장
            job_key = f"{self.JOB_PREFIX}{job.job_id}"
            self.redis_client.set(job_key, json.dumps(job.to_dict()))
            
            # 우선순위 큐에 추가 (score가 높을수록 우선순위 높음)
            score = job.priority
            self.redis_client.zadd(self.QUEUE_KEY, {job.job_id: score})
            
            logger.info(f"Added job to queue: {job.job_id} (priority: {job.priority})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add job: {e}")
            return False
    
    def get_next_job(self) -> Optional[DownloadJob]:
        """
        다음 작업 가져오기 (우선순위 높은 순)
        
        Returns:
            다음 작업 또는 None
        """
        try:
            # 우선순위가 가장 높은 작업 가져오기
            result = self.redis_client.zpopmax(self.QUEUE_KEY)
            
            if not result:
                return None
            
            job_id, _ = result[0]
            
            # 작업 데이터 로드
            job_key = f"{self.JOB_PREFIX}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                logger.warning(f"Job data not found: {job_id}")
                return None
            
            job = DownloadJob.from_dict(json.loads(job_data))
            
            # 처리 중 상태로 변경
            job.status = "processing"
            job.started_at = datetime.now().isoformat()
            self.redis_client.set(job_key, json.dumps(job.to_dict()))
            
            # 처리 중 목록에 추가
            self.redis_client.sadd(self.PROCESSING_KEY, job_id)
            
            logger.info(f"Retrieved job from queue: {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Failed to get next job: {e}")
            return None
    
    def complete_job(self, job_id: str, metadata: Optional[Dict] = None) -> bool:
        """
        작업 완료 처리
        
        Args:
            job_id: 작업 ID
            metadata: 추가 메타데이터
            
        Returns:
            성공 여부
        """
        try:
            job_key = f"{self.JOB_PREFIX}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                logger.warning(f"Job not found: {job_id}")
                return False
            
            job = DownloadJob.from_dict(json.loads(job_data))
            job.status = "completed"
            job.completed_at = datetime.now().isoformat()
            
            if metadata:
                job.metadata = metadata
            
            # 작업 데이터 업데이트
            self.redis_client.set(job_key, json.dumps(job.to_dict()))
            
            # 처리 중에서 제거
            self.redis_client.srem(self.PROCESSING_KEY, job_id)
            
            # 완료 목록에 추가
            self.redis_client.sadd(self.COMPLETED_KEY, job_id)
            
            logger.info(f"Job completed: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete job: {e}")
            return False
    
    def fail_job(
        self,
        job_id: str,
        error_message: str,
        retry: bool = True
    ) -> bool:
        """
        작업 실패 처리
        
        Args:
            job_id: 작업 ID
            error_message: 에러 메시지
            retry: 재시도 여부
            
        Returns:
            성공 여부
        """
        try:
            job_key = f"{self.JOB_PREFIX}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                logger.warning(f"Job not found: {job_id}")
                return False
            
            job = DownloadJob.from_dict(json.loads(job_data))
            job.retry_count += 1
            job.error_message = error_message
            
            # 재시도 가능한 경우
            if retry and job.retry_count < job.max_retries:
                job.status = "pending"
                job.started_at = None
                
                # 작업 데이터 업데이트
                self.redis_client.set(job_key, json.dumps(job.to_dict()))
                
                # 처리 중에서 제거
                self.redis_client.srem(self.PROCESSING_KEY, job_id)
                
                # 큐에 다시 추가 (우선순위 낮춤)
                score = max(1, job.priority - 1)
                self.redis_client.zadd(self.QUEUE_KEY, {job_id: score})
                
                logger.info(
                    f"Job requeued for retry: {job_id} "
                    f"(attempt {job.retry_count}/{job.max_retries})"
                )
                return True
            
            # 재시도 불가능한 경우 실패 처리
            else:
                job.status = "failed"
                job.completed_at = datetime.now().isoformat()
                
                # 작업 데이터 업데이트
                self.redis_client.set(job_key, json.dumps(job.to_dict()))
                
                # 처리 중에서 제거
                self.redis_client.srem(self.PROCESSING_KEY, job_id)
                
                # 실패 목록에 추가
                self.redis_client.sadd(self.FAILED_KEY, job_id)
                
                logger.error(
                    f"Job failed permanently: {job_id} - {error_message}"
                )
                return True
            
        except Exception as e:
            logger.error(f"Failed to handle job failure: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[DownloadJob]:
        """
        작업 조회
        
        Args:
            job_id: 작업 ID
            
        Returns:
            작업 객체 또는 None
        """
        try:
            job_key = f"{self.JOB_PREFIX}{job_id}"
            job_data = self.redis_client.get(job_key)
            
            if not job_data:
                return None
            
            return DownloadJob.from_dict(json.loads(job_data))
            
        except Exception as e:
            logger.error(f"Failed to get job: {e}")
            return None
    
    def get_queue_size(self) -> int:
        """대기 중인 작업 수"""
        return self.redis_client.zcard(self.QUEUE_KEY)
    
    def get_processing_count(self) -> int:
        """처리 중인 작업 수"""
        return self.redis_client.scard(self.PROCESSING_KEY)
    
    def get_completed_count(self) -> int:
        """완료된 작업 수"""
        return self.redis_client.scard(self.COMPLETED_KEY)
    
    def get_failed_count(self) -> int:
        """실패한 작업 수"""
        return self.redis_client.scard(self.FAILED_KEY)
    
    def list_jobs_by_status(self, status: str, limit: int = 100) -> List[DownloadJob]:
        """
        상태별 작업 목록
        
        Args:
            status: 작업 상태 (pending, processing, completed, failed)
            limit: 최대 개수
            
        Returns:
            작업 리스트
        """
        try:
            if status == "pending":
                # 우선순위 큐에서 가져오기 (높은 우선순위부터)
                job_ids = self.redis_client.zrevrange(self.QUEUE_KEY, 0, limit - 1)
            elif status == "processing":
                job_ids = list(self.redis_client.smembers(self.PROCESSING_KEY))[:limit]
            elif status == "completed":
                job_ids = list(self.redis_client.smembers(self.COMPLETED_KEY))[:limit]
            elif status == "failed":
                job_ids = list(self.redis_client.smembers(self.FAILED_KEY))[:limit]
            else:
                return []
            
            jobs = []
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job:
                    jobs.append(job)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def clear_completed(self, older_than_seconds: Optional[int] = None) -> int:
        """
        완료된 작업 정리
        
        Args:
            older_than_seconds: 지정된 시간보다 오래된 작업만 삭제 (None이면 모두 삭제)
            
        Returns:
            삭제된 작업 수
        """
        try:
            job_ids = list(self.redis_client.smembers(self.COMPLETED_KEY))
            deleted_count = 0
            
            for job_id in job_ids:
                job = self.get_job(job_id)
                if not job:
                    continue
                
                # 시간 체크
                if older_than_seconds is not None and job.completed_at:
                    completed_time = datetime.fromisoformat(job.completed_at)
                    age = (datetime.now() - completed_time).total_seconds()
                    
                    if age < older_than_seconds:
                        continue
                
                # 작업 삭제
                job_key = f"{self.JOB_PREFIX}{job_id}"
                self.redis_client.delete(job_key)
                self.redis_client.srem(self.COMPLETED_KEY, job_id)
                deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} completed jobs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear completed jobs: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, int]:
        """
        큐 통계
        
        Returns:
            통계 딕셔너리
        """
        return {
            'pending': self.get_queue_size(),
            'processing': self.get_processing_count(),
            'completed': self.get_completed_count(),
            'failed': self.get_failed_count(),
        }
    
    def reset_all(self) -> bool:
        """
        모든 큐 초기화 (테스트용)
        
        Returns:
            성공 여부
        """
        try:
            # 모든 작업 ID 수집
            all_job_ids = set()
            
            all_job_ids.update(self.redis_client.zrange(self.QUEUE_KEY, 0, -1))
            all_job_ids.update(self.redis_client.smembers(self.PROCESSING_KEY))
            all_job_ids.update(self.redis_client.smembers(self.COMPLETED_KEY))
            all_job_ids.update(self.redis_client.smembers(self.FAILED_KEY))
            
            # 작업 데이터 삭제
            for job_id in all_job_ids:
                job_key = f"{self.JOB_PREFIX}{job_id}"
                self.redis_client.delete(job_key)
            
            # 큐 삭제
            self.redis_client.delete(self.QUEUE_KEY)
            self.redis_client.delete(self.PROCESSING_KEY)
            self.redis_client.delete(self.COMPLETED_KEY)
            self.redis_client.delete(self.FAILED_KEY)
            
            logger.warning("All queues have been reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset queues: {e}")
            return False
