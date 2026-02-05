"""
Upload Manager

클라우드 업로드 관리자
- 업로드 큐 관리
- 진행 상황 추적
- 일괄 업로드
- 재시도 로직
"""

import os
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any, Callable, Union
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logging_config import setup_logger
from storage.providers.base import (
    StorageProvider,
    UploadResult,
    MultipartConfig,
    UploadStatus,
    ErrorType,
)
from storage.providers.s3_provider import S3Provider
from storage.providers.gcs_provider import GCSProvider

logger = setup_logger(__name__)


class QueuePriority(Enum):
    """업로드 우선순위"""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class UploadJob:
    """업로드 작업 정보"""
    job_id: str
    local_path: str
    remote_key: str
    bucket: str
    provider: str = "s3"
    priority: QueuePriority = QueuePriority.NORMAL
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    storage_class: Optional[str] = None
    overwrite: bool = False
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    status: UploadStatus = UploadStatus.PENDING
    result: Optional[UploadResult] = None
    
    def __lt__(self, other):
        """우선순위 비교 (Queue 정렬용)"""
        return self.priority.value < other.priority.value


@dataclass
class BatchProgress:
    """배치 업로드 진행 상황"""
    batch_id: str
    total_files: int
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    in_progress: int = 0
    total_bytes: int = 0
    uploaded_bytes: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[UploadResult] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        return self.completed + self.failed + self.skipped >= self.total_files
    
    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.completed / self.total_files * 100
    
    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    @property
    def bytes_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.uploaded_bytes / self.elapsed_seconds


class UploadManager:
    """
    클라우드 업로드 관리자
    
    Task 5.1.2: Upload Queue
    - 우선순위 큐
    - 동시 업로드 제어
    - 재시도 로직
    - 진행 상황 추적
    """
    
    def __init__(
        self,
        default_provider: str = "s3",
        max_workers: int = 4,
        max_queue_size: int = 1000,
        s3_config: Optional[Dict[str, Any]] = None,
        gcs_config: Optional[Dict[str, Any]] = None,
    ):
        """
        UploadManager 초기화
        
        Args:
            default_provider: 기본 클라우드 프로바이더 ("s3" 또는 "gcs")
            max_workers: 최대 동시 업로드 수
            max_queue_size: 최대 큐 크기
            s3_config: S3 Provider 설정
            gcs_config: GCS Provider 설정
        """
        self.default_provider = default_provider
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # 프로바이더 초기화
        self._providers: Dict[str, StorageProvider] = {}
        self._s3_config = s3_config or {}
        self._gcs_config = gcs_config or {}
        
        # 큐 및 스레드풀
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        
        # 작업 추적
        self._jobs: Dict[str, UploadJob] = {}
        self._batches: Dict[str, BatchProgress] = {}
        self._lock = threading.Lock()
        
        # 콜백
        self._progress_callbacks: List[Callable[[UploadJob], None]] = []
        self._completion_callbacks: List[Callable[[UploadResult], None]] = []
        
    def get_provider(self, provider_name: str) -> StorageProvider:
        """프로바이더 가져오기 (lazy init)"""
        if provider_name not in self._providers:
            if provider_name == "s3":
                self._providers[provider_name] = S3Provider(**self._s3_config)
            elif provider_name == "gcs":
                self._providers[provider_name] = GCSProvider(**self._gcs_config)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
        return self._providers[provider_name]
    
    def start(self):
        """워커 스레드 시작"""
        if self._running:
            return
            
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 워커 스레드 시작
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"UploadManager started with {self.max_workers} workers")
    
    def stop(self, wait: bool = True):
        """워커 스레드 중지"""
        self._running = False
        
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
            
        logger.info("UploadManager stopped")
    
    def enqueue(
        self,
        local_path: str,
        remote_key: str,
        bucket: str,
        provider: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
        overwrite: bool = False,
        max_retries: int = 3,
    ) -> str:
        """업로드 작업을 큐에 추가"""
        job_id = str(uuid.uuid4())
        
        job = UploadJob(
            job_id=job_id,
            local_path=local_path,
            remote_key=remote_key,
            bucket=bucket,
            provider=provider or self.default_provider,
            priority=priority,
            metadata=metadata or {},
            tags=tags or {},
            storage_class=storage_class,
            overwrite=overwrite,
            max_retries=max_retries,
        )
        
        with self._lock:
            self._jobs[job_id] = job
            
        self._queue.put((job.priority.value, job))
        logger.debug(f"Enqueued job {job_id}: {local_path} -> {remote_key}")
        
        return job_id
    
    def enqueue_batch(
        self,
        files: List[Dict[str, Any]],
        bucket: str,
        provider: Optional[str] = None,
        base_prefix: str = "",
        **kwargs,
    ) -> str:
        """여러 파일을 일괄 큐에 추가"""
        batch_id = str(uuid.uuid4())
        
        total_bytes = 0
        job_ids = []
        
        for file_info in files:
            local_path = file_info.get("local_path") or file_info.get("path")
            remote_key = file_info.get("remote_key")
            
            if not remote_key:
                # remote_key가 없으면 base_prefix + 파일명 사용
                remote_key = os.path.join(base_prefix, Path(local_path).name)
                
            # 파일 크기 계산
            if os.path.exists(local_path):
                total_bytes += os.path.getsize(local_path)
                
            job_id = self.enqueue(
                local_path=local_path,
                remote_key=remote_key,
                bucket=bucket,
                provider=provider,
                metadata=file_info.get("metadata", {}),
                tags=file_info.get("tags", {}),
                storage_class=file_info.get("storage_class"),
                **kwargs,
            )
            job_ids.append(job_id)
            
        # 배치 진행 상황 추적
        batch = BatchProgress(
            batch_id=batch_id,
            total_files=len(files),
            total_bytes=total_bytes,
        )
        
        with self._lock:
            self._batches[batch_id] = batch
            
        logger.info(f"Enqueued batch {batch_id}: {len(files)} files, {total_bytes} bytes")
        
        return batch_id
    
    def upload_sync(
        self,
        local_path: str,
        remote_key: str,
        bucket: str,
        provider: Optional[str] = None,
        **kwargs,
    ) -> UploadResult:
        """동기적으로 파일 업로드 (큐 사용 안함)"""
        provider_name = provider or self.default_provider
        storage_provider = self.get_provider(provider_name)
        
        return storage_provider.upload_file(
            local_path=local_path,
            remote_key=remote_key,
            bucket=bucket,
            **kwargs,
        )
    
    def upload_batch_sync(
        self,
        files: List[Dict[str, Any]],
        bucket: str,
        provider: Optional[str] = None,
        base_prefix: str = "",
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> BatchProgress:
        """동기적으로 여러 파일 업로드"""
        batch_id = str(uuid.uuid4())
        max_workers = max_workers or self.max_workers
        provider_name = provider or self.default_provider
        storage_provider = self.get_provider(provider_name)
        
        total_bytes = sum(
            os.path.getsize(f.get("local_path") or f.get("path"))
            for f in files
            if os.path.exists(f.get("local_path") or f.get("path"))
        )
        
        batch = BatchProgress(
            batch_id=batch_id,
            total_files=len(files),
            total_bytes=total_bytes,
        )
        
        def upload_one(file_info: Dict) -> UploadResult:
            local_path = file_info.get("local_path") or file_info.get("path")
            remote_key = file_info.get("remote_key")
            
            if not remote_key:
                remote_key = os.path.join(base_prefix, Path(local_path).name)
                
            return storage_provider.upload_file(
                local_path=local_path,
                remote_key=remote_key,
                bucket=bucket,
                metadata=file_info.get("metadata", {}),
                tags=file_info.get("tags", {}),
                storage_class=file_info.get("storage_class"),
                **kwargs,
            )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(upload_one, f): f for f in files}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    batch.results.append(result)
                    
                    if result.status == UploadStatus.COMPLETED:
                        batch.completed += 1
                        batch.uploaded_bytes += result.size_bytes or 0
                    elif result.status == UploadStatus.SKIPPED:
                        batch.skipped += 1
                    else:
                        batch.failed += 1
                        
                    if progress_callback:
                        progress_callback(
                            batch.completed + batch.failed + batch.skipped,
                            batch.total_files,
                        )
                        
                except Exception as e:
                    batch.failed += 1
                    logger.error(f"Upload failed: {e}")
                    
        batch.end_time = datetime.now()
        
        logger.info(
            f"Batch {batch_id} complete: "
            f"{batch.completed}/{batch.total_files} uploaded, "
            f"{batch.failed} failed, {batch.skipped} skipped, "
            f"{batch.bytes_per_second:.0f} bytes/sec"
        )
        
        return batch
    
    def get_job(self, job_id: str) -> Optional[UploadJob]:
        """작업 정보 조회"""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_batch(self, batch_id: str) -> Optional[BatchProgress]:
        """배치 진행 상황 조회"""
        with self._lock:
            return self._batches.get(batch_id)
    
    def get_queue_size(self) -> int:
        """현재 큐 크기 반환"""
        return self._queue.qsize()
    
    def on_progress(self, callback: Callable[[UploadJob], None]):
        """진행 상황 콜백 등록"""
        self._progress_callbacks.append(callback)
    
    def on_completion(self, callback: Callable[[UploadResult], None]):
        """완료 콜백 등록"""
        self._completion_callbacks.append(callback)
    
    def _process_queue(self):
        """큐 처리 워커"""
        while self._running:
            try:
                # 큐에서 작업 가져오기
                priority, job = self._queue.get(timeout=1.0)
                
                if not self._running:
                    break
                    
                # 작업 실행
                self._executor.submit(self._execute_job, job)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    def _execute_job(self, job: UploadJob):
        """개별 작업 실행"""
        try:
            job.status = UploadStatus.UPLOADING
            
            # 진행 콜백 호출
            for callback in self._progress_callbacks:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
                    
            # 업로드 실행
            provider = self.get_provider(job.provider)
            result = provider.upload_file(
                local_path=job.local_path,
                remote_key=job.remote_key,
                bucket=job.bucket,
                metadata=job.metadata,
                tags=job.tags,
                storage_class=job.storage_class,
                overwrite=job.overwrite,
            )
            
            job.result = result
            job.status = result.status
            
            # 재시도 로직
            if result.status == UploadStatus.FAILED:
                if result.error_type == ErrorType.NETWORK_TRANSIENT and job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = UploadStatus.PENDING
                    
                    # 지수 백오프
                    delay = min(2 ** job.retry_count, 60)
                    logger.warning(f"Retrying job {job.job_id} in {delay}s (attempt {job.retry_count})")
                    
                    time.sleep(delay)
                    self._queue.put((job.priority.value, job))
                    return
                    
            # 완료 콜백 호출
            for callback in self._completion_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Completion callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Job execution error: {e}")
            job.status = UploadStatus.FAILED
            job.result = UploadResult(
                provider=job.provider,
                bucket=job.bucket,
                key=job.remote_key,
                uri=f"{job.provider}://{job.bucket}/{job.remote_key}",
                status=UploadStatus.FAILED,
                error_type=ErrorType.UNKNOWN,
                error_message=str(e),
            )
        finally:
            with self._lock:
                self._jobs[job.job_id] = job


class UploadQueue:
    """
    Celery 기반 비동기 업로드 큐 인터페이스
    
    UploadManager를 Celery 태스크로 래핑
    """
    
    def __init__(self, manager: Optional[UploadManager] = None):
        self.manager = manager or UploadManager()
        
    def enqueue_async(
        self,
        local_path: str,
        remote_key: str,
        bucket: str,
        **kwargs,
    ) -> str:
        """Celery 태스크로 업로드 큐잉"""
        # Celery 임포트 시도
        try:
            from storage.tasks import upload_file_task
            
            result = upload_file_task.delay(
                local_path=local_path,
                remote_key=remote_key,
                bucket=bucket,
                **kwargs,
            )
            return result.id
            
        except ImportError:
            # Celery 없으면 동기 실행
            logger.warning("Celery not available, falling back to sync upload")
            return self.manager.enqueue(local_path, remote_key, bucket, **kwargs)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Celery 태스크 상태 조회"""
        try:
            from celery.result import AsyncResult
            from storage.tasks import celery_app
            
            result = AsyncResult(task_id, app=celery_app)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "traceback": result.traceback if result.failed() else None,
            }
            
        except ImportError:
            # Celery 없으면 로컬 매니저에서 조회
            job = self.manager.get_job(task_id)
            if job:
                return {
                    "task_id": task_id,
                    "status": job.status.value,
                    "result": job.result,
                }
            return {"task_id": task_id, "status": "UNKNOWN"}
