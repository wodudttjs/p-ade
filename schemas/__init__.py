"""
데이터 계약(Contract) 스키마

Feedback #2: 모듈 간 데이터 계약을 먼저 고정
- raw_candidates.jsonl: Discovery 결과
- download_manifest.json: Download 결과
- features.npz: Feature 추출 결과
- episodes.npz: Episode 결과
"""

from schemas.contracts import (
    RawCandidate,
    DownloadManifest,
    FeatureResult,
    EpisodeResult,
    ProcessingManifest,
)

__all__ = [
    "RawCandidate",
    "DownloadManifest", 
    "FeatureResult",
    "EpisodeResult",
    "ProcessingManifest",
]
