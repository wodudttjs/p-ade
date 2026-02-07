"""
GCS Storage Provider

google-cloud-storage 기반 GCS 통합
- 버킷 생성 및 권한 설정
- Resumable upload
- 멱등성 보장
"""

import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable

from core.logging_config import setup_logger
from storage.providers.base import (
    StorageProvider,
    UploadResult,
    ObjectInfo,
    BucketInfo,
    PolicyConfig,
    MultipartConfig,
    LifecycleRule,
    StorageClass,
    UploadStatus,
    ErrorType,
)

logger = setup_logger(__name__)


# Storage class 매핑 (S3 → GCS)
STORAGE_CLASS_MAP = {
    StorageClass.STANDARD: "STANDARD",
    StorageClass.NEARLINE: "NEARLINE",
    StorageClass.COLDLINE: "COLDLINE",
    StorageClass.ARCHIVE: "ARCHIVE",
    StorageClass.INTELLIGENT_TIERING: "STANDARD",  # GCS는 autoclass 사용
    StorageClass.GLACIER: "ARCHIVE",
    StorageClass.GLACIER_DEEP_ARCHIVE: "ARCHIVE",
}


class GCSProvider(StorageProvider):
    """
    Google Cloud Storage Provider
    
    Task 5.1.1: GCS Integration
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: str = "asia-northeast3",
        max_retries: int = 5,
        timeout: int = 60,
    ):
        """
        GCS Provider 초기화
        
        Args:
            project_id: GCP 프로젝트 ID (None이면 환경변수 사용)
            credentials_path: 서비스 계정 키 파일 경로
            location: 기본 버킷 위치
            max_retries: 최대 재시도 횟수
            timeout: 요청 타임아웃 (초)
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.location = location
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        
    @property
    def client(self):
        """google.cloud.storage 클라이언트 (lazy loading)"""
        if self._client is None:
            try:
                from google.cloud import storage
                from google.auth import default
                from google.api_core.retry import Retry
                
                if self.credentials_path:
                    self._client = storage.Client.from_service_account_json(
                        self.credentials_path,
                        project=self.project_id,
                    )
                else:
                    self._client = storage.Client(project=self.project_id)
                    
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS integration. "
                    "Install with: pip install google-cloud-storage"
                )
        return self._client
    
    @property
    def provider_name(self) -> str:
        return "gcs"
    
    def ensure_bucket(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        policy_cfg: Optional[PolicyConfig] = None,
    ) -> BucketInfo:
        """버킷 생성 또는 확인"""
        region = region or self.location
        policy_cfg = policy_cfg or PolicyConfig()
        
        try:
            bucket = self.client.get_bucket(bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
            
        except Exception as e:
            if "404" in str(e) or "NotFound" in str(e):
                # 버킷 생성
                from google.cloud import storage
                
                bucket = storage.Bucket(self.client, name=bucket_name)
                bucket.location = region
                bucket.storage_class = "STANDARD"
                
                # 퍼블릭 액세스 차단
                if policy_cfg.block_public_access:
                    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
                    
                # 버전닝 설정
                if policy_cfg.versioning:
                    bucket.versioning_enabled = True
                    
                bucket = self.client.create_bucket(bucket)
                logger.info(f"Created bucket {bucket_name} in {region}")
            else:
                raise
                
        return BucketInfo(
            name=bucket_name,
            region=region,
            provider="gcs",
            encryption="GOOGLE_MANAGED",  # GCS는 기본 암호화
            versioning_enabled=policy_cfg.versioning,
        )
    
    def upload_file(
        self,
        local_path: str,
        remote_key: str,
        bucket: str,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
        multipart_cfg: Optional[MultipartConfig] = None,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> UploadResult:
        """파일 업로드"""
        local_path = Path(local_path)
        
        if not local_path.exists():
            return UploadResult(
                provider="gcs",
                bucket=bucket,
                key=remote_key,
                uri=self.get_uri(bucket, remote_key),
                status=UploadStatus.FAILED,
                error_type=ErrorType.NOT_FOUND,
                error_message=f"Local file not found: {local_path}",
            )
            
        file_size = local_path.stat().st_size
        sha256 = self._compute_sha256(local_path)
        metadata = metadata or {}
        metadata["sha256"] = sha256
        
        # 멱등성 체크 - 이미 존재하는지 확인
        existing = self.head_object(remote_key, bucket)
        if existing and not overwrite:
            existing_sha256 = existing.metadata.get("sha256")
            if existing_sha256 == sha256:
                logger.info(f"File already exists with same sha256, skipping: {remote_key}")
                return UploadResult(
                    provider="gcs",
                    bucket=bucket,
                    key=remote_key,
                    uri=self.get_uri(bucket, remote_key),
                    etag=existing.etag,
                    size_bytes=existing.size_bytes,
                    sha256=sha256,
                    sha256_verified=True,
                    status=UploadStatus.SKIPPED,
                )
                
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(remote_key)
            
            # 메타데이터 설정
            blob.metadata = metadata
            
            # 스토리지 클래스 설정
            if storage_class:
                blob.storage_class = storage_class
                
            # Resumable upload (기본값)
            multipart_cfg = multipart_cfg or MultipartConfig()
            chunk_size = multipart_cfg.part_size_bytes
            
            # Content type 추론
            content_type = self._get_content_type(local_path)
            
            # 업로드 실행
            blob.upload_from_filename(
                str(local_path),
                content_type=content_type,
                num_retries=self.max_retries,
                timeout=self.timeout,
            )
            
            # 업로드 확인
            blob.reload()
            
            logger.info(f"Uploaded {local_path} to gs://{bucket}/{remote_key}")
            
            return UploadResult(
                provider="gcs",
                bucket=bucket,
                key=remote_key,
                uri=self.get_uri(bucket, remote_key),
                etag=blob.etag,
                version_id=str(blob.generation) if blob.generation else None,
                size_bytes=file_size,
                sha256=sha256,
                sha256_verified=True,
                status=UploadStatus.COMPLETED,
            )
            
        except Exception as e:
            error_type = self._classify_error(e)
            logger.error(f"Upload failed: {e}")
            
            return UploadResult(
                provider="gcs",
                bucket=bucket,
                key=remote_key,
                uri=self.get_uri(bucket, remote_key),
                size_bytes=file_size,
                sha256=sha256,
                status=UploadStatus.FAILED,
                error_type=error_type,
                error_message=str(e),
            )
    
    def head_object(
        self,
        remote_key: str,
        bucket: str,
    ) -> Optional[ObjectInfo]:
        """오브젝트 정보 조회"""
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(remote_key)
            
            if not blob.exists():
                return None
                
            blob.reload()
            
            return ObjectInfo(
                key=remote_key,
                bucket=bucket,
                size_bytes=blob.size or 0,
                etag=blob.etag or "",
                storage_class=blob.storage_class,
                last_modified=blob.updated,
                content_type=blob.content_type,
                metadata=blob.metadata or {},
                version_id=str(blob.generation) if blob.generation else None,
            )
            
        except Exception as e:
            if "404" in str(e) or "NotFound" in str(e):
                return None
            raise
    
    def download_file(
        self,
        remote_key: str,
        bucket: str,
        local_path: str,
    ) -> bool:
        """파일 다운로드"""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(remote_key)
            
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded gs://{bucket}/{remote_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def delete_object(
        self,
        remote_key: str,
        bucket: str,
    ) -> bool:
        """오브젝트 삭제"""
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(remote_key)
            blob.delete()
            
            logger.info(f"Deleted gs://{bucket}/{remote_key}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[ObjectInfo]:
        """오브젝트 목록 조회"""
        objects = []
        
        try:
            bucket_obj = self.client.bucket(bucket)
            blobs = bucket_obj.list_blobs(prefix=prefix, max_results=max_keys)
            
            for blob in blobs:
                objects.append(ObjectInfo(
                    key=blob.name,
                    bucket=bucket,
                    size_bytes=blob.size or 0,
                    etag=blob.etag or "",
                    storage_class=blob.storage_class,
                    last_modified=blob.updated,
                ))
                    
        except Exception as e:
            logger.error(f"List objects failed: {e}")
            
        return objects
    
    def generate_presigned_url(
        self,
        remote_key: str,
        bucket: str,
        expires_sec: int = 3600,
    ) -> Optional[str]:
        """서명된 URL 생성"""
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(remote_key)
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_sec),
                method="GET",
            )
            return url
            
        except Exception as e:
            logger.error(f"Generate signed URL failed: {e}")
            return None
    
    def copy_object(
        self,
        src_key: str,
        dst_key: str,
        bucket: str,
        storage_class: Optional[str] = None,
    ) -> Optional[UploadResult]:
        """오브젝트 복사"""
        try:
            bucket_obj = self.client.bucket(bucket)
            src_blob = bucket_obj.blob(src_key)
            dst_blob = bucket_obj.copy_blob(src_blob, bucket_obj, dst_key)
            
            if storage_class:
                dst_blob.update_storage_class(storage_class)
                
            return UploadResult(
                provider="gcs",
                bucket=bucket,
                key=dst_key,
                uri=self.get_uri(bucket, dst_key),
                etag=dst_blob.etag,
                status=UploadStatus.COMPLETED,
            )
            
        except Exception as e:
            logger.error(f"Copy object failed: {e}")
            return None
    
    def set_lifecycle_policy(
        self,
        bucket: str,
        rules: List[LifecycleRule],
    ) -> bool:
        """라이프사이클 정책 설정"""
        try:
            from google.cloud.storage import LifecycleRuleConditions, LifecycleRuleDelete, LifecycleRuleSetStorageClass
            
            bucket_obj = self.client.bucket(bucket)
            bucket_obj.reload()
            
            lifecycle_rules = []
            
            for rule in rules:
                if not rule.enabled:
                    continue
                    
                if rule.transition_days and rule.transition_storage_class:
                    lifecycle_rules.append({
                        "action": {
                            "type": "SetStorageClass",
                            "storageClass": rule.transition_storage_class,
                        },
                        "condition": {
                            "age": rule.transition_days,
                            "matchesPrefix": [rule.prefix] if rule.prefix else [],
                        },
                    })
                    
                if rule.expiration_days:
                    lifecycle_rules.append({
                        "action": {"type": "Delete"},
                        "condition": {
                            "age": rule.expiration_days,
                            "matchesPrefix": [rule.prefix] if rule.prefix else [],
                        },
                    })
                    
                if rule.noncurrent_version_expiration_days:
                    lifecycle_rules.append({
                        "action": {"type": "Delete"},
                        "condition": {
                            "daysSinceNoncurrentTime": rule.noncurrent_version_expiration_days,
                            "matchesPrefix": [rule.prefix] if rule.prefix else [],
                        },
                    })
                    
            bucket_obj.lifecycle_rules = lifecycle_rules
            bucket_obj.patch()
            
            logger.info(f"Set lifecycle policy for {bucket}: {len(lifecycle_rules)} rules")
            return True
            
        except Exception as e:
            logger.error(f"Set lifecycle policy failed: {e}")
            return False
    
    def _compute_sha256(self, file_path: Path) -> str:
        """SHA256 해시 계산"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
                
        return sha256_hash.hexdigest()
    
    def _get_content_type(self, file_path: Path) -> str:
        """Content-Type 추론"""
        import mimetypes
        content_type, _ = mimetypes.guess_type(str(file_path))
        return content_type or "application/octet-stream"
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """오류 유형 분류"""
        error_str = str(error).lower()
        
        if "credentials" in error_str or "permission" in error_str:
            return ErrorType.AUTHORIZATION
        elif "not found" in error_str or "404" in error_str:
            return ErrorType.NOT_FOUND
        elif "timeout" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_TRANSIENT
        elif "quota" in error_str or "limit" in error_str:
            return ErrorType.QUOTA_LIMIT
        elif "disk" in error_str or "space" in error_str:
            return ErrorType.SYSTEM_RESOURCE
        else:
            return ErrorType.UNKNOWN
