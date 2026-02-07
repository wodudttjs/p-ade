"""
Celery Tasks for Cloud Upload

백그라운드 업로드 태스크
- Redis 브로커
- 재시도 로직
- 우선순위 큐
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime

from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from kombu import Queue

from core.logging_config import setup_logger

logger = setup_logger(__name__)


# Celery 앱 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "p-ade-storage",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Celery 설정
celery_app.conf.update(
    # 직렬화
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # 시간대
    timezone="Asia/Seoul",
    enable_utc=True,
    
    # 재시도
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # 결과 만료
    result_expires=3600,  # 1시간
    
    # 동시성
    worker_concurrency=4,
    worker_prefetch_multiplier=2,
    
    # 태스크 우선순위
    task_queues=(
        Queue("high", routing_key="high"),
        Queue("normal", routing_key="normal"),
        Queue("low", routing_key="low"),
    ),
    task_default_queue="normal",
    task_default_routing_key="normal",
)


def get_upload_manager():
    """UploadManager 싱글톤 가져오기"""
    from storage.upload_manager import UploadManager
    
    # 환경변수에서 설정 로드
    s3_config = {
        "region": os.getenv("AWS_REGION", "ap-northeast-2"),
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "endpoint_url": os.getenv("AWS_ENDPOINT_URL"),  # LocalStack 등
    }
    
    gcs_config = {
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "location": os.getenv("GCS_LOCATION", "asia-northeast3"),
    }
    
    return UploadManager(
        default_provider=os.getenv("DEFAULT_STORAGE_PROVIDER", "s3"),
        max_workers=int(os.getenv("UPLOAD_MAX_WORKERS", "4")),
        s3_config=s3_config,
        gcs_config=gcs_config,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=10,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def upload_file_task(
    self,
    local_path: str,
    remote_key: str,
    bucket: str,
    provider: str = "s3",
    metadata: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    storage_class: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    파일 업로드 태스크
    
    Args:
        local_path: 로컬 파일 경로
        remote_key: 원격 키 (경로)
        bucket: 버킷 이름
        provider: 스토리지 프로바이더 ("s3" 또는 "gcs")
        metadata: 메타데이터
        tags: 태그
        storage_class: 스토리지 클래스
        overwrite: 덮어쓰기 여부
        
    Returns:
        업로드 결과 딕셔너리
    """
    task_id = self.request.id
    logger.info(f"Starting upload task {task_id}: {local_path} -> {provider}://{bucket}/{remote_key}")
    
    # 업로드 진행 상황 업데이트
    self.update_state(
        state="UPLOADING",
        meta={
            "local_path": local_path,
            "remote_key": remote_key,
            "bucket": bucket,
            "provider": provider,
            "started_at": datetime.now().isoformat(),
        },
    )
    
    try:
        manager = get_upload_manager()
        result = manager.upload_sync(
            local_path=local_path,
            remote_key=remote_key,
            bucket=bucket,
            provider=provider,
            metadata=metadata,
            tags=tags,
            storage_class=storage_class,
            overwrite=overwrite,
        )
        
        result_dict = {
            "task_id": task_id,
            "provider": result.provider,
            "bucket": result.bucket,
            "key": result.key,
            "uri": result.uri,
            "etag": result.etag,
            "version_id": result.version_id,
            "size_bytes": result.size_bytes,
            "sha256": result.sha256,
            "sha256_verified": result.sha256_verified,
            "status": result.status.value,
            "error_type": result.error_type.value if result.error_type else None,
            "error_message": result.error_message,
            "completed_at": datetime.now().isoformat(),
        }
        
        if result.status.value == "FAILED":
            # 일시적 오류면 재시도
            from storage.providers.base import ErrorType
            if result.error_type == ErrorType.NETWORK_TRANSIENT:
                raise Exception(f"Transient error: {result.error_message}")
                
        logger.info(f"Completed upload task {task_id}: {result.status.value}")
        return result_dict
        
    except MaxRetriesExceededError:
        logger.error(f"Max retries exceeded for task {task_id}")
        raise
    except Exception as e:
        logger.error(f"Upload task {task_id} failed: {e}")
        raise


@celery_app.task(
    bind=True,
    max_retries=3,
)
def upload_batch_task(
    self,
    files: list,
    bucket: str,
    provider: str = "s3",
    base_prefix: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """
    배치 업로드 태스크
    
    Args:
        files: 파일 정보 리스트 [{"local_path": ..., "remote_key": ...}, ...]
        bucket: 버킷 이름
        provider: 스토리지 프로바이더
        base_prefix: 기본 경로 prefix
        
    Returns:
        배치 결과 딕셔너리
    """
    task_id = self.request.id
    logger.info(f"Starting batch upload task {task_id}: {len(files)} files")
    
    self.update_state(
        state="UPLOADING",
        meta={
            "total_files": len(files),
            "completed": 0,
            "failed": 0,
            "started_at": datetime.now().isoformat(),
        },
    )
    
    try:
        manager = get_upload_manager()
        
        def progress_callback(completed: int, total: int):
            self.update_state(
                state="UPLOADING",
                meta={
                    "total_files": total,
                    "completed": completed,
                    "progress": completed / total * 100 if total > 0 else 0,
                },
            )
        
        batch = manager.upload_batch_sync(
            files=files,
            bucket=bucket,
            provider=provider,
            base_prefix=base_prefix,
            progress_callback=progress_callback,
            **kwargs,
        )
        
        return {
            "task_id": task_id,
            "batch_id": batch.batch_id,
            "total_files": batch.total_files,
            "completed": batch.completed,
            "failed": batch.failed,
            "skipped": batch.skipped,
            "total_bytes": batch.total_bytes,
            "uploaded_bytes": batch.uploaded_bytes,
            "success_rate": batch.success_rate,
            "elapsed_seconds": batch.elapsed_seconds,
            "bytes_per_second": batch.bytes_per_second,
            "completed_at": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Batch upload task {task_id} failed: {e}")
        raise


@celery_app.task
def cleanup_old_uploads(days: int = 30) -> Dict[str, Any]:
    """
    오래된 업로드 기록 정리
    
    Args:
        days: 정리할 기준 일수
        
    Returns:
        정리 결과
    """
    # TODO: 데이터베이스에서 오래된 업로드 기록 정리
    logger.info(f"Cleaning up upload records older than {days} days")
    
    return {
        "cleaned_records": 0,
        "cleaned_at": datetime.now().isoformat(),
    }


@celery_app.task
def verify_upload_integrity(
    bucket: str,
    key: str,
    expected_sha256: str,
    provider: str = "s3",
) -> Dict[str, Any]:
    """
    업로드 무결성 검증
    
    Args:
        bucket: 버킷 이름
        key: 오브젝트 키
        expected_sha256: 예상 SHA256 해시
        provider: 스토리지 프로바이더
        
    Returns:
        검증 결과
    """
    logger.info(f"Verifying upload integrity: {provider}://{bucket}/{key}")
    
    manager = get_upload_manager()
    storage_provider = manager.get_provider(provider)
    
    obj_info = storage_provider.head_object(key, bucket)
    
    if not obj_info:
        return {
            "verified": False,
            "error": "Object not found",
        }
        
    actual_sha256 = obj_info.metadata.get("sha256")
    
    if actual_sha256 == expected_sha256:
        return {
            "verified": True,
            "sha256": actual_sha256,
        }
    else:
        return {
            "verified": False,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "error": "SHA256 mismatch",
        }


# 주기적 태스크 스케줄
celery_app.conf.beat_schedule = {
    "cleanup-old-uploads": {
        "task": "storage.tasks.cleanup_old_uploads",
        "schedule": 86400.0,  # 매일
        "args": (30,),
    },
}
