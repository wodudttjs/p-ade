"""
데이터베이스 모델 정의

P-ADE 시스템의 모든 데이터 모델을 정의합니다.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, JSON
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
    
    discovered_at = Column(DateTime, default=datetime.utcnow)
    downloaded_at = Column(DateTime)
    processed_at = Column(DateTime)
    
    status = Column(String(50), default='discovered')
    local_path = Column(String(500))
    
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
    
    episode_id = Column(String(100), unique=True, nullable=False)
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
    
    meta_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    uploaded_at = Column(DateTime)
    
    video = relationship("Video", back_populates="episodes")


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
