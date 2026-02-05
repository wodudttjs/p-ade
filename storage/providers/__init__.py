"""
Storage Providers

클라우드 스토리지 제공자 통합 모듈
- S3 Provider (AWS)
- GCS Provider (GCP)
"""

from storage.providers.base import (
    StorageProvider,
    UploadResult,
    ObjectInfo,
    BucketInfo,
    MultipartConfig,
    LifecycleRule,
    PolicyConfig,
    StorageClass,
    UploadStatus,
    ErrorType,
)
from storage.providers.s3_provider import S3Provider
from storage.providers.gcs_provider import GCSProvider

__all__ = [
    "StorageProvider",
    "UploadResult",
    "ObjectInfo",
    "BucketInfo",
    "MultipartConfig",
    "LifecycleRule",
    "PolicyConfig",
    "StorageClass",
    "UploadStatus",
    "ErrorType",
    "S3Provider",
    "GCSProvider",
]
