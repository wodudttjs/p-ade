"""
데이터베이스 모델 정의

P-ADE 시스템의 모든 데이터 모델을 정의합니다.
Feedback 반영:
- 재현성(reproducibility) 필드 추가
- 라이선스/권리 메타데이터 필드 추가
- Idempotent Job Key 지원
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, JSON, Index
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class KeywordCategory(Base):
    """키워드 카테고리"""
    __tablename__ = 'keyword_categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    keywords = relationship("Keyword", back_populates="category")


class Keyword(Base):
    """검색 키워드"""
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    keyword = Column(String(200), unique=True, nullable=False)
    category_id = Column(Integer, ForeignKey('keyword_categories.id'))
    
    language = Column(String(10), default='en')
    priority = Column(Integer, default=5)
    weight = Column(Float, default=1.0)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    category = relationship("KeywordCategory", back_populates="keywords")
    performance = relationship("KeywordPerformance", back_populates="keyword", uselist=False)


class KeywordPerformance(Base):
    """키워드 성능 지표"""
    __tablename__ = 'keyword_performance'
    
    id = Column(Integer, primary_key=True)
    keyword_id = Column(Integer, ForeignKey('keywords.id'), unique=True)
    
    total_searches = Column(Integer, default=0)
    total_videos_found = Column(Integer, default=0)
    total_videos_downloaded = Column(Integer, default=0)
    total_high_quality_episodes = Column(Integer, default=0)
    
    avg_video_quality = Column(Float, default=0.0)
    avg_relevance_score = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    
    videos_per_search = Column(Float, default=0.0)
    quality_episodes_per_video = Column(Float, default=0.0)
    
    last_calculated_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    keyword = relationship("Keyword", back_populates="performance")


class Video(Base):
    """비디오 정보"""
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(50), unique=True, nullable=False)
    platform = Column(String(50), nullable=False)
    url = Column(String(500), nullable=False)
    
    title = Column(String(500))
    description = Column(Text)
    duration_sec = Column(Integer)
    upload_date = Column(DateTime)
    
    channel_id = Column(String(100))
    channel_name = Column(String(200))
    view_count = Column(Integer)
    like_count = Column(Integer)
    
    thumbnail_url = Column(String(500))
    tags = Column(JSON)
    
    # ===== 라이선스/권리 메타데이터 (Feedback #2) =====
    license = Column(String(100))  # CC-BY, CC-BY-SA, Standard YouTube License, etc.
    copyright_owner = Column(String(200))
    permission_proof = Column(String(500))  # 허가 증명 링크/문서 경로
    attribution_required = Column(Boolean, default=False)
    is_download_allowed = Column(Boolean, default=True)  # 내부 정책 플래그
    source_terms_snapshot = Column(String(100))  # 수집 시점 ToS 해시/링크
    
    discovered_at = Column(DateTime, default=datetime.utcnow)
    downloaded_at = Column(DateTime)
    processed_at = Column(DateTime)
    
    status = Column(String(50), default='discovered')
    local_path = Column(String(500))
    
    # ===== 재현성 (Feedback #3) =====
    download_tool_version = Column(String(50))  # yt-dlp 버전
    download_format_id = Column(String(50))  # 다운로드 포맷 ID
    checksum_sha256 = Column(String(64))  # 파일 무결성 해시
    
    episodes = relationship("Episode", back_populates="video")
    fingerprint = relationship("VideoFingerprint", back_populates="video", uselist=False)
    history = relationship("VideoHistory", back_populates="video")


class VideoFingerprint(Base):
    """비디오 중복 감지용 지문"""
    __tablename__ = 'video_fingerprints'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'), unique=True)
    
    url_hash = Column(String(64), unique=True, nullable=False)
    thumbnail_hash = Column(String(64))
    title_hash = Column(String(64))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    video = relationship("Video", back_populates="fingerprint")


class VideoHistory(Base):
    """비디오 처리 이력"""
    __tablename__ = 'video_history'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'))
    
    action = Column(String(50), nullable=False)
    status = Column(String(50))
    message = Column(Text)
    meta_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    video = relationship("Video", back_populates="history")


class Episode(Base):
    """동작 에피소드"""
    __tablename__ = 'episodes'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('videos.id'))
    
    episode_id = Column(String(100), unique=True, nullable=False)  # {video_id}_ep{idx:03d}
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    duration_frames = Column(Integer)
    
    action_type = Column(String(100))
    confidence_score = Column(Float)
    quality_score = Column(Float)
    jittering_score = Column(Float)
    
    cloud_path = Column(String(500))
    local_path = Column(String(500))
    filesize_bytes = Column(Integer)
    
    # ===== 재현성 (Feedback #3) =====
    processing_version = Column(String(50))  # 처리 코드 git commit hash
    model_versions = Column(JSON)  # {"mediapipe": "0.10.x", "yolo": "v8.x", ...}
    processing_params = Column(JSON)  # {"fps": 30, "conf_threshold": 0.5, ...}
    ffmpeg_version = Column(String(50))
    
    # ===== Idempotent Job Key (Feedback #4) =====
    job_key = Column(String(200), unique=True, index=True)  # {platform}_{video_id}_{processing_version}
    
    meta_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    uploaded_at = Column(DateTime)
    
    video = relationship("Video", back_populates="episodes")
    
    # 복합 인덱스
    __table_args__ = (
        Index('ix_episodes_video_processing', 'video_id', 'processing_version'),
    )


class DatasetVersion(Base):
    """데이터셋 버전 관리"""
    __tablename__ = 'dataset_versions'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(20), unique=True, nullable=False)
    
    total_videos = Column(Integer)
    total_episodes = Column(Integer)
    total_size_bytes = Column(Integer)
    
    description = Column(Text)
    manifest_path = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # FR-5.3 관계
    files = relationship("CloudFile", back_populates="dataset_version")


class CloudFile(Base):
    """
    클라우드 파일 메타데이터
    
    FR-5.2: Metadata Database
    - 클라우드 업로드 파일 추적
    - SHA256 해시로 무결성 검증
    - 버전 관리
    """
    __tablename__ = 'cloud_files'
    
    id = Column(Integer, primary_key=True)
    
    # 파일 식별
    file_id = Column(String(36), unique=True, nullable=False)  # UUID
    episode_id = Column(Integer, ForeignKey('episodes.id'), nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    dataset_version_id = Column(Integer, ForeignKey('dataset_versions.id'), nullable=True)
    
    # 파일 정보
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # episode_npz, video_mp4, manifest_json, etc.
    file_size_bytes = Column(Integer, nullable=False)
    
    # 해시 (무결성)
    sha256 = Column(String(64), nullable=False, index=True)
    md5 = Column(String(32))
    
    # 클라우드 위치
    provider = Column(String(20), nullable=False)  # s3, gcs
    bucket = Column(String(255), nullable=False)
    key = Column(String(1000), nullable=False)
    uri = Column(String(1500), nullable=False)
    
    # 클라우드 메타데이터
    etag = Column(String(255))
    version_id = Column(String(255))
    storage_class = Column(String(50))
    
    # 압축
    compression = Column(String(20))  # None, gzip, lz4, zstd
    original_size_bytes = Column(Integer)
    compression_ratio = Column(Float)
    
    # 상태
    status = Column(String(20), default='uploaded')  # uploaded, verified, archived, deleted
    verified_at = Column(DateTime)
    
    # 메타데이터
    meta_data = Column(JSON)
    tags = Column(JSON)
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 관계
    episode = relationship("Episode", backref="cloud_files")
    video = relationship("Video", backref="cloud_files")
    dataset_version = relationship("DatasetVersion", back_populates="files")


class UploadTask(Base):
    """
    업로드 태스크 추적
    
    Celery 태스크 상태 및 결과 추적
    """
    __tablename__ = 'upload_tasks'
    
    id = Column(Integer, primary_key=True)
    
    # 태스크 정보
    task_id = Column(String(36), unique=True, nullable=False)  # Celery task ID
    task_type = Column(String(50), nullable=False)  # upload_file, upload_batch
    
    # 파일 정보
    local_path = Column(String(1000), nullable=False)
    remote_key = Column(String(1000), nullable=False)
    bucket = Column(String(255), nullable=False)
    provider = Column(String(20), nullable=False)
    
    # 상태
    status = Column(String(20), default='pending')  # pending, uploading, completed, failed
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # 결과
    cloud_file_id = Column(Integer, ForeignKey('cloud_files.id'), nullable=True)
    error_type = Column(String(50))
    error_message = Column(Text)
    
    # 우선순위
    priority = Column(Integer, default=2)  # 1=high, 2=normal, 3=low
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 관계
    cloud_file = relationship("CloudFile", backref="upload_task")


class StorageCost(Base):
    """
    스토리지 비용 추적
    
    FR-5.4: Cost Optimization
    """
    __tablename__ = 'storage_costs'
    
    id = Column(Integer, primary_key=True)
    
    # 기간
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # 프로바이더
    provider = Column(String(20), nullable=False)
    bucket = Column(String(255))
    
    # 용량
    total_bytes = Column(Integer, default=0)
    storage_class = Column(String(50))
    
    # 비용 (USD)
    storage_cost = Column(Float, default=0.0)
    request_cost = Column(Float, default=0.0)
    transfer_cost = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # API 호출
    put_requests = Column(Integer, default=0)
    get_requests = Column(Integer, default=0)
    list_requests = Column(Integer, default=0)
    
    # 메타데이터
    meta_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessingJob(Base):
    """
    처리 작업 (Idempotent Job Key)
    
    Feedback #4: 멱등성 지원
    - 같은 비디오가 큐에 2번 들어와도 결과가 꼬이지 않음
    - Job Key = (platform, video_id, processing_version)
    """
    __tablename__ = 'processing_jobs'
    
    id = Column(Integer, primary_key=True)
    
    # ===== Idempotent Job Key =====
    job_key = Column(String(200), unique=True, nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    video_id = Column(String(50), nullable=False)
    processing_version = Column(String(50), nullable=False)  # git commit hash
    
    # 상태
    status = Column(String(30), default='pending')  # pending, running, completed, failed, skipped
    stage = Column(String(50))  # discover, download, extract, transform, upload
    
    # 재시도
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # 실패 분류 (Feedback 3종)
    failure_type = Column(String(30))  # network, quality, system
    failure_reason = Column(Text)
    
    # 결과
    result_path = Column(String(500))  # 결과 저장 경로 (deterministic)
    result_hash = Column(String(64))
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 복합 인덱스
    __table_args__ = (
        Index('ix_job_platform_video', 'platform', 'video_id'),
        Index('ix_job_status_stage', 'status', 'stage'),
    )
    
    @classmethod
    def generate_job_key(cls, platform: str, video_id: str, processing_version: str) -> str:
        """Job Key 생성"""
        return f"{platform}_{video_id}_{processing_version}"
    
    @classmethod  
    def generate_result_path(cls, platform: str, video_id: str, processing_version: str) -> str:
        """Deterministic 결과 경로 생성"""
        return f"data/episodes/{platform}/{video_id}/{processing_version}/"


class QualityConfig(Base):
    """
    품질 필터링 설정 (동적 관리)
    
    Feedback #5: 기준을 코드 상수로 박지 않고 DB에서 관리
    """
    __tablename__ = 'quality_configs'
    
    id = Column(Integer, primary_key=True)
    
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # 임계값 설정
    min_confidence = Column(Float, default=0.5)
    max_jitter_score = Column(Float, default=0.3)
    min_episode_frames = Column(Integer, default=30)
    max_nan_ratio = Column(Float, default=0.1)
    
    # 포즈 품질
    min_visible_joints = Column(Integer, default=15)
    min_pose_completeness = Column(Float, default=0.7)
    
    # 프로파일
    is_active = Column(Boolean, default=False)
    profile = Column(String(50), default='default')  # dev, prod, strict
    
    # 예상 통과율 (주간 리포트용)
    expected_pass_rate = Column(Float)
    last_calculated_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
