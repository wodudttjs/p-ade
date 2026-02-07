"""
Storage Provider Base Interface

클라우드 스토리지 제공자 공통 인터페이스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class StorageClass(Enum):
    """스토리지 클래스"""
    # AWS S3
    STANDARD = "STANDARD"
    INTELLIGENT_TIERING = "INTELLIGENT_TIERING"
    STANDARD_IA = "STANDARD_IA"
    ONEZONE_IA = "ONEZONE_IA"
    GLACIER = "GLACIER"
    GLACIER_IR = "GLACIER_IR"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"
    GLACIER_DEEP_ARCHIVE = "DEEP_ARCHIVE"  # Alias
    
    # GCS
    NEARLINE = "NEARLINE"
    COLDLINE = "COLDLINE"
    ARCHIVE = "ARCHIVE"


class UploadStatus(Enum):
    """업로드 상태"""
    PENDING = "pending"
    UPLOADING = "uploading"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorType(Enum):
    """오류 유형"""
    NETWORK_TRANSIENT = "network_transient"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    INTEGRITY_FAIL = "integrity_fail"
    QUOTA_LIMIT = "quota_limit"
    SYSTEM_RESOURCE = "system_resource"
    UNKNOWN = "unknown"


@dataclass
class MultipartConfig:
    """멀티파트 업로드 설정"""
    threshold_bytes: int = 5 * 1024 * 1024  # 5MB
    part_size_bytes: int = 8 * 1024 * 1024  # 8MB
    max_concurrency: int = 10


@dataclass
class BucketInfo:
    """버킷 정보"""
    name: str
    region: str
    provider: str
    created_at: Optional[datetime] = None
    encryption: Optional[str] = None
    versioning_enabled: bool = False


@dataclass
class ObjectInfo:
    """오브젝트 정보"""
    key: str
    bucket: str
    size_bytes: int
    etag: Optional[str] = None
    crc32c: Optional[str] = None
    sha256: Optional[str] = None
    storage_class: Optional[str] = None
    last_modified: Optional[datetime] = None
    content_type: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    version_id: Optional[str] = None


@dataclass
class UploadResult:
    """업로드 결과"""
    provider: str
    bucket: str
    key: str
    uri: str
    etag: Optional[str] = None
    crc32c: Optional[str] = None
    version_id: Optional[str] = None
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    sha256: Optional[str] = None
    sha256_verified: bool = False
    status: UploadStatus = UploadStatus.COMPLETED
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None


@dataclass
class LifecycleRule:
    """라이프사이클 규칙"""
    id: str
    prefix: str
    enabled: bool = True
    transition_days: Optional[int] = None
    transition_storage_class: Optional[str] = None
    expiration_days: Optional[int] = None
    noncurrent_version_expiration_days: Optional[int] = None


@dataclass
class PolicyConfig:
    """정책 설정"""
    block_public_access: bool = True
    encryption: str = "AES256"  # AES256 or aws:kms
    versioning: bool = False
    lifecycle_rules: List[LifecycleRule] = field(default_factory=list)


class StorageProvider(ABC):
    """
    클라우드 스토리지 제공자 추상 인터페이스
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """제공자 이름 (s3, gcs)"""
        pass
    
    @abstractmethod
    def ensure_bucket(
        self,
        bucket_name: str,
        region: str,
        policy_cfg: Optional[PolicyConfig] = None,
    ) -> BucketInfo:
        """
        버킷 생성 또는 확인
        
        Args:
            bucket_name: 버킷 이름
            region: 리전
            policy_cfg: 정책 설정
            
        Returns:
            BucketInfo: 버킷 정보
        """
        pass
    
    @abstractmethod
    def upload_file(
        self,
        local_path: str,
        remote_key: str,
        bucket: str,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        storage_class: Optional[str] = None,
        multipart_cfg: Optional[MultipartConfig] = None,
    ) -> UploadResult:
        """
        파일 업로드
        
        Args:
            local_path: 로컬 파일 경로
            remote_key: 원격 키
            bucket: 버킷 이름
            metadata: 메타데이터
            tags: 태그
            storage_class: 스토리지 클래스
            multipart_cfg: 멀티파트 설정
            
        Returns:
            UploadResult: 업로드 결과
        """
        pass
    
    @abstractmethod
    def head_object(
        self,
        remote_key: str,
        bucket: str,
    ) -> Optional[ObjectInfo]:
        """
        오브젝트 정보 조회
        
        Args:
            remote_key: 원격 키
            bucket: 버킷 이름
            
        Returns:
            ObjectInfo 또는 None (없으면)
        """
        pass
    
    @abstractmethod
    def download_file(
        self,
        remote_key: str,
        bucket: str,
        local_path: str,
    ) -> bool:
        """
        파일 다운로드
        
        Args:
            remote_key: 원격 키
            bucket: 버킷 이름
            local_path: 로컬 저장 경로
            
        Returns:
            성공 여부
        """
        pass
    
    @abstractmethod
    def delete_object(
        self,
        remote_key: str,
        bucket: str,
    ) -> bool:
        """
        오브젝트 삭제
        
        Args:
            remote_key: 원격 키
            bucket: 버킷 이름
            
        Returns:
            성공 여부
        """
        pass
    
    @abstractmethod
    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[ObjectInfo]:
        """
        오브젝트 목록 조회
        
        Args:
            bucket: 버킷 이름
            prefix: 접두사
            max_keys: 최대 개수
            
        Returns:
            ObjectInfo 리스트
        """
        pass
    
    def generate_presigned_url(
        self,
        remote_key: str,
        bucket: str,
        expires_sec: int = 3600,
    ) -> Optional[str]:
        """
        사전 서명된 URL 생성 (선택)
        
        Args:
            remote_key: 원격 키
            bucket: 버킷 이름
            expires_sec: 만료 시간 (초)
            
        Returns:
            서명된 URL 또는 None
        """
        return None
    
    def copy_object(
        self,
        src_key: str,
        dst_key: str,
        bucket: str,
        storage_class: Optional[str] = None,
    ) -> Optional[UploadResult]:
        """
        오브젝트 복사 (선택)
        
        Args:
            src_key: 소스 키
            dst_key: 대상 키
            bucket: 버킷 이름
            storage_class: 스토리지 클래스
            
        Returns:
            UploadResult 또는 None
        """
        return None
    
    def set_lifecycle_policy(
        self,
        bucket: str,
        rules: List[LifecycleRule],
    ) -> bool:
        """
        라이프사이클 정책 설정 (선택)
        
        Args:
            bucket: 버킷 이름
            rules: 라이프사이클 규칙 리스트
            
        Returns:
            성공 여부
        """
        return False
    
    def get_uri(self, bucket: str, key: str) -> str:
        """URI 생성"""
        return f"{self.provider_name}://{bucket}/{key}"
