"""
P-ADE 설정 관리

환경 변수 및 애플리케이션 설정을 관리합니다.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    """애플리케이션 설정"""
    
    # 프로젝트 경로
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_VIDEO_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    EPISODES_DIR = DATA_DIR / "episodes"
    LOGS_DIR = BASE_DIR / "logs"
    CONFIG_DIR = BASE_DIR / "config"
    
    # API Keys
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")
    VIMEO_ACCESS_TOKEN: Optional[str] = os.getenv("VIMEO_ACCESS_TOKEN")
    
    # AWS 설정
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "p-ade-datasets")
    
    # GCP 설정
    GCP_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID")
    GCP_CREDENTIALS_PATH: Optional[str] = os.getenv("GCP_CREDENTIALS_PATH")
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "p-ade-datasets")
    
    # 데이터베이스 설정
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "pade_db")
    DB_USER: str = os.getenv("DB_USER", "pade_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    @property
    def DATABASE_URL(self) -> str:
        """SQLAlchemy 데이터베이스 URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Redis 설정
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    @property
    def REDIS_URL(self) -> str:
        """Redis 연결 URL"""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # 프록시 설정
    PROXY_FILE: Path = CONFIG_DIR / os.getenv("PROXY_FILE", "proxies.txt")
    USE_PROXY: bool = os.getenv("USE_PROXY", "false").lower() == "true"
    
    # 처리 설정
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    VIDEO_QUALITY: str = os.getenv("VIDEO_QUALITY", "1080p")
    FRAME_RATE: int = int(os.getenv("FRAME_RATE", "30"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.7"))
    
    # 크롤링 설정
    CRAWL_DELAY: float = 2.0  # 요청 간 대기 시간 (초)
    MAX_RETRIES: int = 3
    CONCURRENT_REQUESTS: int = 8
    
    # 비디오 설정
    MIN_DURATION_SEC: int = 10
    MAX_DURATION_SEC: int = 1200  # 20분
    MIN_RESOLUTION_HEIGHT: int = 720
    MIN_FPS: int = 24
    
    # MediaPipe 설정
    MEDIAPIPE_MODEL_COMPLEXITY: int = 1  # 0, 1, 2
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # 모니터링
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # ===== 대량 수집 파이프라인 설정 =====
    MASS_COLLECT_TARGET: int = int(os.getenv("MASS_COLLECT_TARGET", "500"))
    MASS_COLLECT_SOURCES: list = os.getenv(
        "MASS_COLLECT_SOURCES", "youtube,google_videos"
    ).split(",")
    MASS_COLLECT_LANGUAGES: list = os.getenv(
        "MASS_COLLECT_LANGUAGES", "en,ko"
    ).split(",")
    MASS_COLLECT_CRAWL_WORKERS: int = int(os.getenv("MASS_COLLECT_CRAWL_WORKERS", "4"))
    MASS_COLLECT_DOWNLOAD_WORKERS: int = int(os.getenv("MASS_COLLECT_DOWNLOAD_WORKERS", "6"))
    MASS_COLLECT_DOWNLOAD_TIMEOUT: int = int(os.getenv("MASS_COLLECT_DOWNLOAD_TIMEOUT", "600"))
    MASS_COLLECT_DETECT_FPS: float = float(os.getenv("MASS_COLLECT_DETECT_FPS", "5.0"))
    MASS_COLLECT_DETECT_DEVICE: Optional[str] = os.getenv("MASS_COLLECT_DETECT_DEVICE")
    
    # 레이트 리밋 설정
    RATE_LIMIT_YOUTUBE_RPM: int = int(os.getenv("RATE_LIMIT_YOUTUBE_RPM", "20"))
    RATE_LIMIT_GOOGLE_RPM: int = int(os.getenv("RATE_LIMIT_GOOGLE_RPM", "15"))
    RATE_LIMIT_MIN_DELAY: float = float(os.getenv("RATE_LIMIT_MIN_DELAY", "2.0"))
    RATE_LIMIT_MAX_DELAY: float = float(os.getenv("RATE_LIMIT_MAX_DELAY", "5.0"))
    
    @classmethod
    def ensure_directories(cls):
        """필수 디렉토리 생성"""
        for directory in [cls.DATA_DIR, cls.RAW_VIDEO_DIR, cls.PROCESSED_DIR, 
                         cls.EPISODES_DIR, cls.LOGS_DIR, cls.CONFIG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    TESTING = False


class TestConfig(Config):
    """테스트 환경 설정"""
    DEBUG = True
    TESTING = True
    DB_NAME = "pade_test_db"
    
    @property
    def DATABASE_URL(self) -> str:
        """테스트용 SQLite 데이터베이스 URL"""
        return "sqlite:///:memory:"


# 환경별 설정 선택
_env = os.getenv("ENVIRONMENT", "development").lower()
if _env == "production":
    config = ProductionConfig()
elif _env == "test":
    config = TestConfig()
else:
    config = DevelopmentConfig()

# 디렉토리 생성
config.ensure_directories()
