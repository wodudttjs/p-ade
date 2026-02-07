"""
데이터 계약(Contract) 정의

Feedback: 모듈 간 데이터 계약을 확정하여 병렬 개발 가능

A. raw_candidates.jsonl - Discovery 결과 (크롤러 출력)
B. download_manifest.json - Download 결과
C. features.npz - Feature 추출 결과
D. episodes.npz - Episode 결과
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import hashlib


@dataclass
class RawCandidate:
    """
    A. Discovery 결과 (크롤러 출력)
    
    파일 형식: raw_candidates.jsonl
    경로: data/raw_candidates/{keyword_id}/{date}/candidates.jsonl
    """
    # 필수 필드
    platform: str  # youtube, vimeo
    video_id: str
    source_url: str
    
    # 메타데이터
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    thumbnail_url: Optional[str] = None
    
    # 검색 정보
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    query: Optional[str] = None
    keyword_id: Optional[int] = None
    
    # 라이선스/권리 (Feedback #2)
    license_hint: Optional[str] = None  # CC-BY, Standard License 등
    channel_id: Optional[str] = None
    uploader: Optional[str] = None
    
    # 중복 감지
    url_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.url_hash:
            self.url_hash = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]
    
    def to_jsonl(self) -> str:
        """JSONL 형식으로 변환"""
        return json.dumps(asdict(self), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawCandidate":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DownloadManifest:
    """
    B. Download 결과
    
    파일 형식: download_manifest.json
    경로: data/videos/{platform}/{video_id}/manifest.json
    """
    # 식별
    video_id: str
    platform: str
    local_path: str
    
    # 비디오 정보
    duration_sec: float
    resolution: str  # "1920x1080"
    fps: float
    
    # 다운로드 정보
    yt_dlp_format_id: str
    downloaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # 무결성 (Feedback #3: 재현성)
    checksum_sha256: str = ""
    file_size_bytes: int = 0
    
    # 상태
    status: str = "success"  # success, failed, skipped
    failure_reason: Optional[str] = None
    
    # 재현성 (Feedback #3)
    yt_dlp_version: Optional[str] = None
    ffmpeg_version: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass  
class FeatureResult:
    """
    C. Feature 추출 결과
    
    파일 형식: features.npz
    경로: data/features/{platform}/{video_id}/features.npz
    
    NPZ 내용:
    - pose: [T, 33, 3/4] (x, y, z, visibility)
    - hands: [T, 42, 3] (optional, left+right hands)
    - objects: [T, K, 5] (optional, x, y, w, h, class_id)
    - conf_pose: [T] 
    - conf_hand: [T] (optional)
    - timestamps: [T]
    """
    # 식별
    video_id: str
    platform: str
    
    # 배열 정보
    num_frames: int
    pose_shape: List[int]  # [T, 33, 4]
    has_hands: bool = False
    has_objects: bool = False
    
    # 타임스탬프
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    fps: float = 30.0
    
    # 재현성 (Feedback #3)
    model_versions: Dict[str, str] = field(default_factory=dict)
    # 예: {"mediapipe_pose": "0.10.9", "yolo": "v8.0.123"}
    
    processing_params: Dict[str, Any] = field(default_factory=dict)
    # 예: {"sampling_fps": 30, "conf_threshold": 0.5, "model_complexity": 1}
    
    git_commit_hash: Optional[str] = None
    
    # 누락 프레임 정책 (Feedback #6)
    missing_frame_policy: str = "interpolate"  # interpolate, drop, mask
    missing_frame_indices: List[int] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


@dataclass
class EpisodeResult:
    """
    D. Episode 결과
    
    파일 형식: episodes.npz
    경로: data/episodes/{platform}/{video_id}/episodes.npz
    
    Episode ID 규칙: "{video_id}_ep{idx:03d}"
    
    NPZ 내용:
    - states: [T, J, 3] (정규화된 포즈)
    - actions: [T-1, J, 3] (delta position)
    - timestamps: [T]
    - confidence: [T]
    """
    # 식별
    episode_id: str  # "{video_id}_ep{idx:03d}" 규칙
    video_id: str
    platform: str
    episode_index: int
    
    # 프레임 범위
    start_frame: int
    end_frame: int
    num_frames: int
    
    # 배열 형태
    states_shape: List[int]  # [T, J, 3]
    actions_shape: List[int]  # [T-1, J, 3]
    
    # 품질 메트릭 (Feedback #5)
    quality_score: float = 0.0
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    jitter_score: float = 0.0
    nan_ratio: float = 0.0
    
    # 액션 정의 (Feedback #7)
    action_type: str = "kinematic_delta"  # kinematic_delta = 다음 프레임 델타
    # 참고: 로봇 action으로 매핑하려면 별도 IK/trajectory fitting 필요
    
    # 재현성 (Feedback #3)
    processing_version: str = ""
    model_versions: Dict[str, str] = field(default_factory=dict)
    processing_params: Dict[str, Any] = field(default_factory=dict)
    
    # Job Key (Feedback #4)
    job_key: str = ""
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @classmethod
    def generate_episode_id(cls, video_id: str, episode_index: int) -> str:
        """Episode ID 생성 규칙"""
        return f"{video_id}_ep{episode_index:03d}"
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


@dataclass
class ProcessingManifest:
    """
    전체 처리 매니페스트 (재현성)
    
    Feedback #3: 재현성 키
    파일 형식: processing_manifest.json
    경로: data/episodes/{platform}/{video_id}/manifest.json
    """
    # 식별
    video_id: str
    platform: str
    job_key: str
    
    # 버전 정보
    pipeline_version: str  # 전체 파이프라인 버전
    git_commit_hash: str
    
    # 도구 버전
    python_version: str = ""
    yt_dlp_version: str = ""
    ffmpeg_version: str = ""
    mediapipe_version: str = ""
    opencv_version: str = ""
    numpy_version: str = ""
    
    # 모델 버전
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    # 처리 파라미터
    processing_params: Dict[str, Any] = field(default_factory=dict)
    
    # 품질 설정 (Feedback #5)
    quality_config_name: str = "default"
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # 결과 요약
    total_episodes: int = 0
    total_frames: int = 0
    processing_time_sec: float = 0.0
    
    # 타임스탬프
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
    
    def save(self, path: str):
        """매니페스트 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "ProcessingManifest":
        """매니페스트 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ===== Job Key 유틸리티 (Feedback #4) =====

def generate_job_key(platform: str, video_id: str, processing_version: str) -> str:
    """
    Idempotent Job Key 생성
    
    규칙: {platform}_{video_id}_{processing_version}
    """
    return f"{platform}_{video_id}_{processing_version}"


def generate_result_path(
    base_dir: str,
    platform: str, 
    video_id: str, 
    processing_version: str,
    file_type: str = "episodes"
) -> str:
    """
    Deterministic 결과 경로 생성
    
    Feedback #4: 결과 저장 경로도 동일 키 기반으로 deterministic
    """
    return f"{base_dir}/{file_type}/{platform}/{video_id}/{processing_version}/"


def parse_episode_id(episode_id: str) -> tuple:
    """
    Episode ID 파싱
    
    입력: "abc123_ep001"
    출력: ("abc123", 1)
    """
    parts = episode_id.rsplit("_ep", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid episode_id format: {episode_id}")
    return parts[0], int(parts[1])
