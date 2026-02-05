"""
테스트 설정 및 픽스처
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 테스트 환경 설정 - 모든 import 이전에 설정
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from models.database import Base


@pytest.fixture(scope="function")
def test_db():
    """테스트용 인메모리 데이터베이스 (각 테스트마다 새로 생성)"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def temp_dir():
    """임시 디렉토리 생성"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_video_metadata():
    """샘플 비디오 메타데이터"""
    return {
        "video_id": "test_video_123",
        "platform": "youtube",
        "url": "https://youtube.com/watch?v=test_video_123",
        "title": "Robot Assembly Tutorial",
        "description": "Learn how to assemble a robot",
        "duration_sec": 300,
        "view_count": 10000,
        "tags": ["robot", "assembly", "tutorial"]
    }
