"""
설정 파일 테스트

config 모듈의 설정 관리 기능을 테스트합니다.
"""

import pytest
import os
from pathlib import Path


def test_config_module_import():
    """config 모듈 import 가능 확인"""
    from config import config, Config
    assert config is not None
    assert Config is not None


def test_config_has_required_attributes():
    """필수 설정 속성 존재 확인"""
    from config import config
    
    required_attrs = [
        'BASE_DIR',
        'DATA_DIR',
        'RAW_VIDEO_DIR',
        'PROCESSED_DIR',
        'EPISODES_DIR',
        'LOGS_DIR',
        'CONFIG_DIR',
        'MAX_WORKERS',
        'VIDEO_QUALITY',
        'FRAME_RATE',
        'MIN_CONFIDENCE'
    ]
    
    for attr in required_attrs:
        assert hasattr(config, attr), f"설정 속성 누락: {attr}"


def test_config_paths_are_pathlib():
    """설정 경로가 pathlib.Path 타입인지 확인"""
    from config import config
    
    path_attrs = [
        'BASE_DIR',
        'DATA_DIR',
        'RAW_VIDEO_DIR',
        'PROCESSED_DIR',
        'EPISODES_DIR',
        'LOGS_DIR',
        'CONFIG_DIR'
    ]
    
    for attr in path_attrs:
        value = getattr(config, attr)
        assert isinstance(value, Path), f"{attr}가 Path 타입이 아닙니다: {type(value)}"


def test_config_database_url_format():
    """데이터베이스 URL 형식 확인"""
    from config import config
    
    db_url = config.DATABASE_URL
    assert isinstance(db_url, str)
    # 테스트 환경에서는 SQLite를 사용
    if config.TESTING:
        assert db_url.startswith('sqlite://'), "테스트 환경에서 SQLite URL 형식이 아닙니다"
    else:
        assert db_url.startswith('postgresql://'), "PostgreSQL URL 형식이 아닙니다"
        assert '@' in db_url, "데이터베이스 URL에 호스트 정보가 없습니다"


def test_config_redis_url_format():
    """Redis URL 형식 확인"""
    from config import config
    
    redis_url = config.REDIS_URL
    assert isinstance(redis_url, str)
    assert redis_url.startswith('redis://'), "Redis URL 형식이 아닙니다"


def test_config_numeric_values():
    """숫자 설정값 타입 및 범위 확인"""
    from config import config
    
    # 정수 값 확인
    assert isinstance(config.MAX_WORKERS, int)
    assert config.MAX_WORKERS > 0, "MAX_WORKERS는 양수여야 합니다"
    
    assert isinstance(config.FRAME_RATE, int)
    assert config.FRAME_RATE > 0, "FRAME_RATE는 양수여야 합니다"
    
    assert isinstance(config.DB_PORT, int)
    assert 1 <= config.DB_PORT <= 65535, "DB_PORT는 유효한 포트 번호여야 합니다"
    
    assert isinstance(config.REDIS_PORT, int)
    assert 1 <= config.REDIS_PORT <= 65535, "REDIS_PORT는 유효한 포트 번호여야 합니다"
    
    # 실수 값 확인
    assert isinstance(config.MIN_CONFIDENCE, float)
    assert 0.0 <= config.MIN_CONFIDENCE <= 1.0, "MIN_CONFIDENCE는 0.0~1.0 범위여야 합니다"
    
    assert isinstance(config.CRAWL_DELAY, float)
    assert config.CRAWL_DELAY > 0, "CRAWL_DELAY는 양수여야 합니다"


def test_config_video_settings():
    """비디오 관련 설정값 확인"""
    from config import config
    
    assert config.VIDEO_QUALITY in ['360p', '720p', '1080p', '1440p', '2160p'], \
        f"유효하지 않은 VIDEO_QUALITY: {config.VIDEO_QUALITY}"
    
    assert isinstance(config.MIN_DURATION_SEC, int)
    assert config.MIN_DURATION_SEC > 0
    
    assert isinstance(config.MAX_DURATION_SEC, int)
    assert config.MAX_DURATION_SEC > config.MIN_DURATION_SEC, \
        "MAX_DURATION_SEC는 MIN_DURATION_SEC보다 커야 합니다"
    
    assert isinstance(config.MIN_RESOLUTION_HEIGHT, int)
    assert config.MIN_RESOLUTION_HEIGHT >= 360
    
    assert isinstance(config.MIN_FPS, int)
    assert config.MIN_FPS >= 1


def test_config_mediapipe_settings():
    """MediaPipe 관련 설정값 확인"""
    from config import config
    
    assert config.MEDIAPIPE_MODEL_COMPLEXITY in [0, 1, 2], \
        f"유효하지 않은 MEDIAPIPE_MODEL_COMPLEXITY: {config.MEDIAPIPE_MODEL_COMPLEXITY}"
    
    assert isinstance(config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE, float)
    assert 0.0 <= config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE <= 1.0
    
    assert isinstance(config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE, float)
    assert 0.0 <= config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE <= 1.0


def test_config_boolean_values():
    """불린 설정값 확인"""
    from config import config
    
    boolean_attrs = ['USE_PROXY', 'ENABLE_METRICS', 'DEBUG', 'TESTING']
    
    for attr in boolean_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            assert isinstance(value, bool), f"{attr}가 bool 타입이 아닙니다: {type(value)}"


def test_config_environment_selection():
    """환경별 설정 클래스 선택 확인"""
    from config.settings import DevelopmentConfig, ProductionConfig, TestConfig
    
    # 각 환경 설정 클래스가 존재하는지 확인
    assert DevelopmentConfig is not None
    assert ProductionConfig is not None
    assert TestConfig is not None
    
    # 기본 Config 클래스를 상속하는지 확인
    from config.settings import Config
    assert issubclass(DevelopmentConfig, Config)
    assert issubclass(ProductionConfig, Config)
    assert issubclass(TestConfig, Config)


def test_development_config_settings():
    """개발 환경 설정 확인"""
    from config.settings import DevelopmentConfig
    
    config = DevelopmentConfig()
    assert config.DEBUG is True
    assert config.TESTING is False


def test_production_config_settings():
    """프로덕션 환경 설정 확인"""
    from config.settings import ProductionConfig
    
    config = ProductionConfig()
    assert config.DEBUG is False
    assert config.TESTING is False


def test_test_config_settings():
    """테스트 환경 설정 확인"""
    from config.settings import TestConfig
    
    config = TestConfig()
    assert config.DEBUG is True
    assert config.TESTING is True
    assert 'test' in config.DB_NAME.lower(), "테스트 DB 이름에 'test'가 포함되어야 합니다"


def test_settings_file_structure():
    """settings.py 파일 구조 확인"""
    project_root = Path(__file__).parent.parent
    settings_file = project_root / "config" / "settings.py"
    
    assert settings_file.exists()
    
    content = settings_file.read_text(encoding='utf-8')
    
    # 필수 import 확인
    assert 'from dotenv import load_dotenv' in content
    assert 'from pathlib import Path' in content
    
    # 클래스 정의 확인
    assert 'class Config:' in content
    assert 'class DevelopmentConfig' in content
    assert 'class ProductionConfig' in content
    assert 'class TestConfig' in content


def test_env_example_file_exists():
    """.env.example 파일 존재 확인"""
    project_root = Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    
    assert env_example.exists()


def test_env_example_has_required_variables():
    """.env.example 파일에 필수 환경 변수 포함 확인"""
    project_root = Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    
    content = env_example.read_text(encoding='utf-8')
    
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_S3_BUCKET',
        'DB_HOST',
        'DB_PORT',
        'DB_NAME',
        'DB_USER',
        'DB_PASSWORD',
        'REDIS_HOST',
        'REDIS_PORT',
        'MAX_WORKERS',
        'VIDEO_QUALITY'
    ]
    
    for var in required_vars:
        assert var in content, f"환경 변수 누락: {var}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
