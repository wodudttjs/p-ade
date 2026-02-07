"""
Processing Job Manager

Feedback #4: 멱등성(Idempotency) 시스템
- 같은 비디오가 큐에 2번 들어와도 결과가 꼬이지 않음
- Job Key = (platform, video_id, processing_version)
- 이미 완료된 job이면 skip
"""

import hashlib
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.logging_config import setup_logger

logger = setup_logger(__name__)


class JobStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStage(Enum):
    """작업 단계"""
    DISCOVER = "discover"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    UPLOAD = "upload"


class FailureType(Enum):
    """
    실패 분류 (Feedback: 최소 3종)
    
    - network: 재시도 가능
    - quality: 스킵 (품질 미달)
    - system: 즉시 알림 필요
    """
    NETWORK = "network"  # 재시도
    QUALITY = "quality"  # 스킵
    SYSTEM = "system"    # 즉시 알림


@dataclass
class JobKey:
    """
    Idempotent Job Key
    
    규칙: {platform}_{video_id}_{processing_version}
    """
    platform: str
    video_id: str
    processing_version: str
    
    def __str__(self) -> str:
        return f"{self.platform}_{self.video_id}_{self.processing_version}"
    
    @classmethod
    def from_string(cls, key_str: str) -> "JobKey":
        """문자열에서 JobKey 파싱"""
        parts = key_str.split("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid job key format: {key_str}")
        return cls(platform=parts[0], video_id=parts[1], processing_version=parts[2])
    
    def get_result_path(self, base_dir: str = "data") -> Path:
        """Deterministic 결과 경로"""
        return Path(base_dir) / "episodes" / self.platform / self.video_id / self.processing_version


@dataclass
class JobResult:
    """작업 결과"""
    job_key: str
    status: JobStatus
    stage: JobStage
    result_path: Optional[str] = None
    result_hash: Optional[str] = None
    failure_type: Optional[FailureType] = None
    failure_reason: Optional[str] = None
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_key": self.job_key,
            "status": self.status.value,
            "stage": self.stage.value,
            "result_path": self.result_path,
            "result_hash": self.result_hash,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "failure_reason": self.failure_reason,
            "processing_time_sec": self.processing_time_sec,
        }


class JobManager:
    """
    작업 관리자
    
    Feedback #4: Idempotent Job Key 시스템
    - 중복 작업 방지
    - 재시도 관리
    - 실패 분류
    """
    
    def __init__(self, db_session=None, processing_version: Optional[str] = None):
        self.db_session = db_session
        self._processing_version = processing_version or self._get_git_commit()
        self._jobs_cache: Dict[str, JobResult] = {}
    
    @property
    def processing_version(self) -> str:
        return self._processing_version
    
    @staticmethod
    def _get_git_commit() -> str:
        """현재 Git commit hash 가져오기"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def create_job_key(self, platform: str, video_id: str) -> JobKey:
        """Job Key 생성"""
        return JobKey(
            platform=platform,
            video_id=video_id,
            processing_version=self.processing_version
        )
    
    def is_job_completed(self, job_key: JobKey) -> bool:
        """작업이 이미 완료되었는지 확인"""
        key_str = str(job_key)
        
        # 캐시 확인
        if key_str in self._jobs_cache:
            return self._jobs_cache[key_str].status == JobStatus.COMPLETED
        
        # DB 확인 (있는 경우)
        if self.db_session:
            try:
                from models.database import ProcessingJob
                job = self.db_session.query(ProcessingJob).filter(
                    ProcessingJob.job_key == key_str
                ).first()
                if job and job.status == "completed":
                    return True
            except Exception as e:
                logger.warning(f"DB 조회 실패: {e}")
        
        # 결과 파일 존재 확인
        result_path = job_key.get_result_path()
        manifest_path = result_path / "manifest.json"
        if manifest_path.exists():
            logger.info(f"이미 완료된 작업: {key_str}")
            return True
        
        return False
    
    def should_skip_job(self, job_key: JobKey) -> tuple:
        """
        작업 스킵 여부 확인
        
        Returns:
            (should_skip: bool, reason: str)
        """
        if self.is_job_completed(job_key):
            return True, "already_completed"
        
        # DB에서 실패 기록 확인
        if self.db_session:
            try:
                from models.database import ProcessingJob
                job = self.db_session.query(ProcessingJob).filter(
                    ProcessingJob.job_key == str(job_key)
                ).first()
                
                if job:
                    # 품질 문제로 실패한 경우 스킵
                    if job.failure_type == FailureType.QUALITY.value:
                        return True, "quality_failure"
                    
                    # 재시도 횟수 초과
                    if job.retry_count >= job.max_retries:
                        return True, "max_retries_exceeded"
                        
            except Exception as e:
                logger.warning(f"DB 조회 실패: {e}")
        
        return False, ""
    
    def start_job(self, job_key: JobKey, stage: JobStage) -> JobResult:
        """작업 시작"""
        key_str = str(job_key)
        
        result = JobResult(
            job_key=key_str,
            status=JobStatus.RUNNING,
            stage=stage,
        )
        
        self._jobs_cache[key_str] = result
        
        # DB 기록
        if self.db_session:
            try:
                from models.database import ProcessingJob
                
                job = self.db_session.query(ProcessingJob).filter(
                    ProcessingJob.job_key == key_str
                ).first()
                
                if job:
                    job.status = JobStatus.RUNNING.value
                    job.stage = stage.value
                    job.started_at = datetime.utcnow()
                else:
                    job = ProcessingJob(
                        job_key=key_str,
                        platform=job_key.platform,
                        video_id=job_key.video_id,
                        processing_version=job_key.processing_version,
                        status=JobStatus.RUNNING.value,
                        stage=stage.value,
                        started_at=datetime.utcnow(),
                    )
                    self.db_session.add(job)
                
                self.db_session.commit()
            except Exception as e:
                logger.error(f"DB 기록 실패: {e}")
                self.db_session.rollback()
        
        logger.info(f"작업 시작: {key_str} (stage={stage.value})")
        return result
    
    def complete_job(
        self,
        job_key: JobKey,
        stage: JobStage,
        result_path: str,
        processing_time_sec: float = 0.0
    ) -> JobResult:
        """작업 완료"""
        key_str = str(job_key)
        
        # 결과 해시 계산
        result_hash = self._compute_result_hash(result_path)
        
        result = JobResult(
            job_key=key_str,
            status=JobStatus.COMPLETED,
            stage=stage,
            result_path=result_path,
            result_hash=result_hash,
            processing_time_sec=processing_time_sec,
        )
        
        self._jobs_cache[key_str] = result
        
        # DB 업데이트
        if self.db_session:
            try:
                from models.database import ProcessingJob
                
                job = self.db_session.query(ProcessingJob).filter(
                    ProcessingJob.job_key == key_str
                ).first()
                
                if job:
                    job.status = JobStatus.COMPLETED.value
                    job.stage = stage.value
                    job.result_path = result_path
                    job.result_hash = result_hash
                    job.completed_at = datetime.utcnow()
                    self.db_session.commit()
            except Exception as e:
                logger.error(f"DB 업데이트 실패: {e}")
                self.db_session.rollback()
        
        logger.info(f"작업 완료: {key_str}")
        return result
    
    def fail_job(
        self,
        job_key: JobKey,
        stage: JobStage,
        failure_type: FailureType,
        failure_reason: str,
    ) -> JobResult:
        """작업 실패 기록"""
        key_str = str(job_key)
        
        result = JobResult(
            job_key=key_str,
            status=JobStatus.FAILED,
            stage=stage,
            failure_type=failure_type,
            failure_reason=failure_reason,
        )
        
        self._jobs_cache[key_str] = result
        
        # DB 업데이트
        if self.db_session:
            try:
                from models.database import ProcessingJob
                
                job = self.db_session.query(ProcessingJob).filter(
                    ProcessingJob.job_key == key_str
                ).first()
                
                if job:
                    job.status = JobStatus.FAILED.value
                    job.stage = stage.value
                    job.failure_type = failure_type.value
                    job.failure_reason = failure_reason
                    job.retry_count += 1
                    job.completed_at = datetime.utcnow()
                    self.db_session.commit()
            except Exception as e:
                logger.error(f"DB 업데이트 실패: {e}")
                self.db_session.rollback()
        
        # 실패 유형별 처리
        if failure_type == FailureType.SYSTEM:
            logger.error(f"시스템 오류 - 즉시 알림 필요: {key_str} - {failure_reason}")
        elif failure_type == FailureType.NETWORK:
            logger.warning(f"네트워크 오류 - 재시도 예정: {key_str} - {failure_reason}")
        else:
            logger.info(f"품질 미달 - 스킵: {key_str} - {failure_reason}")
        
        return result
    
    @staticmethod
    def _compute_result_hash(result_path: str) -> str:
        """결과 파일 해시 계산"""
        path = Path(result_path)
        if not path.exists():
            return ""
        
        hasher = hashlib.sha256()
        
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        elif path.is_dir():
            # 디렉토리인 경우 manifest.json 해시
            manifest = path / "manifest.json"
            if manifest.exists():
                with open(manifest, "rb") as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()[:16]
    
    def get_pending_jobs(self, stage: Optional[JobStage] = None) -> List[str]:
        """대기 중인 작업 목록"""
        if not self.db_session:
            return []
        
        try:
            from models.database import ProcessingJob
            
            query = self.db_session.query(ProcessingJob.job_key).filter(
                ProcessingJob.status == JobStatus.PENDING.value
            )
            
            if stage:
                query = query.filter(ProcessingJob.stage == stage.value)
            
            return [job.job_key for job in query.all()]
        except Exception as e:
            logger.error(f"대기 작업 조회 실패: {e}")
            return []
    
    def get_failed_jobs_for_retry(
        self,
        failure_type: FailureType = FailureType.NETWORK
    ) -> List[str]:
        """재시도 가능한 실패 작업 목록"""
        if not self.db_session:
            return []
        
        try:
            from models.database import ProcessingJob
            
            jobs = self.db_session.query(ProcessingJob.job_key).filter(
                ProcessingJob.status == JobStatus.FAILED.value,
                ProcessingJob.failure_type == failure_type.value,
                ProcessingJob.retry_count < ProcessingJob.max_retries
            ).all()
            
            return [job.job_key for job in jobs]
        except Exception as e:
            logger.error(f"재시도 작업 조회 실패: {e}")
            return []


# 전역 인스턴스
_job_manager: Optional[JobManager] = None


def get_job_manager(db_session=None) -> JobManager:
    """JobManager 싱글톤"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager(db_session=db_session)
    return _job_manager
