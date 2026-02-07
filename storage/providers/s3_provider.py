"""
AWS S3 Storage Provider

boto3 기반 S3 통합
- 버킷 생성 및 권한 설정
- Multipart upload (>5MB)
- 멱등성 보장
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
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


class S3Provider(StorageProvider):
    """
    AWS S3 Storage Provider
    
    Task 5.1.1: AWS S3 Integration
    """
    
    def __init__(
        self,
        region: str = "ap-northeast-2",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_retries: int = 5,
        connect_timeout: int = 10,
        read_timeout: int = 60,
    ):
        """
        S3 Provider 초기화
        
        Args:
            region: AWS 리전
            access_key_id: AWS Access Key (None이면 환경변수/IAM 사용)
            secret_access_key: AWS Secret Key
            endpoint_url: 커스텀 엔드포인트 (LocalStack 등)
            max_retries: 최대 재시도 횟수
            connect_timeout: 연결 타임아웃 (초)
            read_timeout: 읽기 타임아웃 (초)
        """
        self.region = region
        self.endpoint_url = endpoint_url
        self._client = None
        self._resource = None
        
        # boto3 설정
        self._config_kwargs = {
            "region_name": region,
            "config": self._create_config(max_retries, connect_timeout, read_timeout),
        }
        
        if access_key_id and secret_access_key:
            self._config_kwargs["aws_access_key_id"] = access_key_id
            self._config_kwargs["aws_secret_access_key"] = secret_access_key
            
        if endpoint_url:
            self._config_kwargs["endpoint_url"] = endpoint_url
            
    def _create_config(
        self,
        max_retries: int,
        connect_timeout: int,
        read_timeout: int,
    ):
        """boto3 Config 생성"""
        try:
            from botocore.config import Config
            return Config(
                retries={
                    "max_attempts": max_retries,
                    "mode": "adaptive",
                },
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        except ImportError:
            return None
    
    @property
    def client(self):
        """boto3 S3 클라이언트 (lazy loading)"""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3", **self._config_kwargs)
            except ImportError:
                raise ImportError("boto3 is required for S3 integration. Install with: pip install boto3")
        return self._client
    
    @property
    def resource(self):
        """boto3 S3 리소스 (lazy loading)"""
        if self._resource is None:
            try:
                import boto3
                self._resource = boto3.resource("s3", **self._config_kwargs)
            except ImportError:
                raise ImportError("boto3 is required for S3 integration")
        return self._resource
    
    @property
    def provider_name(self) -> str:
        return "s3"
    
    def ensure_bucket(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        policy_cfg: Optional[PolicyConfig] = None,
    ) -> BucketInfo:
        """버킷 생성 또는 확인"""
        region = region or self.region
        policy_cfg = policy_cfg or PolicyConfig()
        
        try:
            # 버킷 존재 확인
            self.client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
            
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            
            if error_code == "404":
                # 버킷 생성
                create_params = {"Bucket": bucket_name}
                if region != "us-east-1":
                    create_params["CreateBucketConfiguration"] = {
                        "LocationConstraint": region
                    }
                    
                self.client.create_bucket(**create_params)
                logger.info(f"Created bucket {bucket_name} in {region}")
                
                # 퍼블릭 액세스 차단
                if policy_cfg.block_public_access:
                    self.client.put_public_access_block(
                        Bucket=bucket_name,
                        PublicAccessBlockConfiguration={
                            "BlockPublicAcls": True,
                            "IgnorePublicAcls": True,
                            "BlockPublicPolicy": True,
                            "RestrictPublicBuckets": True,
                        },
                    )
                    
                # 암호화 설정
                self.client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration={
                        "Rules": [{
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": policy_cfg.encryption,
                            },
                        }],
                    },
                )
                
                # 버전닝 설정
                if policy_cfg.versioning:
                    self.client.put_bucket_versioning(
                        Bucket=bucket_name,
                        VersioningConfiguration={"Status": "Enabled"},
                    )
                    
            else:
                raise
                
        return BucketInfo(
            name=bucket_name,
            region=region,
            provider="s3",
            encryption=policy_cfg.encryption,
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
                provider="s3",
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
                    provider="s3",
                    bucket=bucket,
                    key=remote_key,
                    uri=self.get_uri(bucket, remote_key),
                    etag=existing.etag,
                    size_bytes=existing.size_bytes,
                    sha256=sha256,
                    sha256_verified=True,
                    status=UploadStatus.SKIPPED,
                )
                
        # 멀티파트 설정
        multipart_cfg = multipart_cfg or MultipartConfig()
        
        try:
            from boto3.s3.transfer import TransferConfig
            
            transfer_config = TransferConfig(
                multipart_threshold=multipart_cfg.threshold_bytes,
                multipart_chunksize=multipart_cfg.part_size_bytes,
                max_concurrency=multipart_cfg.max_concurrency,
            )
            
            # 업로드 파라미터
            extra_args = {"Metadata": metadata}
            
            if storage_class:
                extra_args["StorageClass"] = storage_class
                
            if tags:
                tag_set = "&".join(f"{k}={v}" for k, v in tags.items())
                extra_args["Tagging"] = tag_set
                
            # 업로드 실행
            callback = None
            if progress_callback:
                class ProgressCallback:
                    def __init__(self, callback):
                        self._callback = callback
                        self._bytes_sent = 0
                    def __call__(self, bytes_amount):
                        self._bytes_sent += bytes_amount
                        self._callback(self._bytes_sent)
                callback = ProgressCallback(progress_callback)
                
            self.client.upload_file(
                str(local_path),
                bucket,
                remote_key,
                ExtraArgs=extra_args,
                Config=transfer_config,
                Callback=callback,
            )
            
            # 업로드 확인
            uploaded = self.head_object(remote_key, bucket)
            
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{remote_key}")
            
            return UploadResult(
                provider="s3",
                bucket=bucket,
                key=remote_key,
                uri=self.get_uri(bucket, remote_key),
                etag=uploaded.etag if uploaded else None,
                version_id=uploaded.version_id if uploaded else None,
                size_bytes=file_size,
                sha256=sha256,
                sha256_verified=True,
                status=UploadStatus.COMPLETED,
            )
            
        except Exception as e:
            error_type = self._classify_error(e)
            logger.error(f"Upload failed: {e}")
            
            return UploadResult(
                provider="s3",
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
            response = self.client.head_object(Bucket=bucket, Key=remote_key)
            
            return ObjectInfo(
                key=remote_key,
                bucket=bucket,
                size_bytes=response.get("ContentLength", 0),
                etag=response.get("ETag", "").strip('"'),
                storage_class=response.get("StorageClass"),
                last_modified=response.get("LastModified"),
                content_type=response.get("ContentType"),
                metadata=response.get("Metadata", {}),
                version_id=response.get("VersionId"),
            )
            
        except self.client.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
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
            
            self.client.download_file(bucket, remote_key, str(local_path))
            logger.info(f"Downloaded s3://{bucket}/{remote_key} to {local_path}")
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
            self.client.delete_object(Bucket=bucket, Key=remote_key)
            logger.info(f"Deleted s3://{bucket}/{remote_key}")
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
            paginator = self.client.get_paginator("list_objects_v2")
            
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            ):
                for obj in page.get("Contents", []):
                    objects.append(ObjectInfo(
                        key=obj["Key"],
                        bucket=bucket,
                        size_bytes=obj["Size"],
                        etag=obj.get("ETag", "").strip('"'),
                        storage_class=obj.get("StorageClass"),
                        last_modified=obj.get("LastModified"),
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
        """사전 서명된 URL 생성"""
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": remote_key},
                ExpiresIn=expires_sec,
            )
            return url
        except Exception as e:
            logger.error(f"Generate presigned URL failed: {e}")
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
            copy_source = {"Bucket": bucket, "Key": src_key}
            extra_args = {}
            
            if storage_class:
                extra_args["StorageClass"] = storage_class
                extra_args["MetadataDirective"] = "COPY"
                
            response = self.client.copy_object(
                Bucket=bucket,
                Key=dst_key,
                CopySource=copy_source,
                **extra_args,
            )
            
            return UploadResult(
                provider="s3",
                bucket=bucket,
                key=dst_key,
                uri=self.get_uri(bucket, dst_key),
                etag=response.get("CopyObjectResult", {}).get("ETag", "").strip('"'),
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
            lifecycle_rules = []
            
            for rule in rules:
                rule_config = {
                    "ID": rule.id,
                    "Status": "Enabled" if rule.enabled else "Disabled",
                    "Filter": {"Prefix": rule.prefix},
                }
                
                if rule.transition_days and rule.transition_storage_class:
                    rule_config["Transitions"] = [{
                        "Days": rule.transition_days,
                        "StorageClass": rule.transition_storage_class,
                    }]
                    
                if rule.expiration_days:
                    rule_config["Expiration"] = {"Days": rule.expiration_days}
                    
                if rule.noncurrent_version_expiration_days:
                    rule_config["NoncurrentVersionExpiration"] = {
                        "NoncurrentDays": rule.noncurrent_version_expiration_days,
                    }
                    
                lifecycle_rules.append(rule_config)
                
            self.client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration={"Rules": lifecycle_rules},
            )
            
            logger.info(f"Set lifecycle policy for {bucket}: {len(rules)} rules")
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
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """오류 유형 분류"""
        error_str = str(error).lower()
        
        if "credentials" in error_str or "access denied" in error_str:
            return ErrorType.AUTHORIZATION
        elif "not found" in error_str or "nosuchbucket" in error_str:
            return ErrorType.NOT_FOUND
        elif "timeout" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_TRANSIENT
        elif "quota" in error_str or "limit" in error_str:
            return ErrorType.QUOTA_LIMIT
        elif "disk" in error_str or "space" in error_str:
            return ErrorType.SYSTEM_RESOURCE
        else:
            return ErrorType.UNKNOWN
