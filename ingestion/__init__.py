"""
P-ADE Ingestion 패키지

비디오 수집, 다운로드, 메타데이터 추출, 품질 필터링 등
데이터 수집 파이프라인의 핵심 모듈들을 제공합니다.
"""

from ingestion.downloader import (
    VideoDownloader,
    DownloadResult,
    VideoQuality,
)
from ingestion.download_queue import (
    DownloadQueue,
    DownloadJob,
    Priority,
)
from ingestion.metadata_extractor import (
    MetadataExtractor,
    VideoMetadata,
)
from ingestion.quality_filter import (
    QualityFilter,
    QualityScore,
    QualityLevel,
)
from ingestion.keyword_manager import KeywordManager
from ingestion.storage_manager import StorageManager

__all__ = [
    # downloader
    "VideoDownloader",
    "DownloadResult",
    "VideoQuality",
    # download_queue
    "DownloadQueue",
    "DownloadJob",
    "Priority",
    # metadata_extractor
    "MetadataExtractor",
    "VideoMetadata",
    # quality_filter
    "QualityFilter",
    "QualityScore",
    "QualityLevel",
    # keyword_manager
    "KeywordManager",
    # storage_manager
    "StorageManager",
]
