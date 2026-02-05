"""
데이터베이스 모델 테스트

models 모듈의 데이터베이스 모델을 테스트합니다.
"""

import pytest
from datetime import datetime


def test_models_module_import():
    """models 모듈 import 가능 확인"""
    from models import (
        Base, KeywordCategory, Keyword, KeywordPerformance,
        Video, VideoFingerprint, VideoHistory, Episode, DatasetVersion
    )
    
    assert Base is not None
    assert KeywordCategory is not None
    assert Keyword is not None
    assert Video is not None


def test_keyword_category_model():
    """KeywordCategory 모델 테스트"""
    from models import KeywordCategory
    
    # 테이블 이름 확인
    assert KeywordCategory.__tablename__ == 'keyword_categories'
    
    # 필수 컬럼 확인
    assert hasattr(KeywordCategory, 'id')
    assert hasattr(KeywordCategory, 'name')
    assert hasattr(KeywordCategory, 'description')
    assert hasattr(KeywordCategory, 'created_at')
    
    # 관계 확인
    assert hasattr(KeywordCategory, 'keywords')


def test_keyword_model():
    """Keyword 모델 테스트"""
    from models import Keyword
    
    assert Keyword.__tablename__ == 'keywords'
    
    # 필수 컬럼
    required_columns = ['id', 'keyword', 'category_id', 'language', 
                       'priority', 'weight', 'is_active', 'created_at']
    for col in required_columns:
        assert hasattr(Keyword, col), f"컬럼 누락: {col}"
    
    # 관계
    assert hasattr(Keyword, 'category')
    assert hasattr(Keyword, 'performance')


def test_keyword_performance_model():
    """KeywordPerformance 모델 테스트"""
    from models import KeywordPerformance
    
    assert KeywordPerformance.__tablename__ == 'keyword_performance'
    
    # 통계 컬럼
    stats_columns = ['total_searches', 'total_videos_found', 'total_videos_downloaded',
                     'avg_video_quality', 'success_rate']
    for col in stats_columns:
        assert hasattr(KeywordPerformance, col), f"통계 컬럼 누락: {col}"


def test_video_model():
    """Video 모델 테스트"""
    from models import Video
    
    assert Video.__tablename__ == 'videos'
    
    # 필수 컬럼
    required_columns = ['id', 'video_id', 'platform', 'url', 'title',
                       'duration_sec', 'discovered_at', 'status']
    for col in required_columns:
        assert hasattr(Video, col), f"컬럼 누락: {col}"
    
    # 관계
    assert hasattr(Video, 'episodes')
    assert hasattr(Video, 'fingerprint')
    assert hasattr(Video, 'history')


def test_video_fingerprint_model():
    """VideoFingerprint 모델 테스트"""
    from models import VideoFingerprint
    
    assert VideoFingerprint.__tablename__ == 'video_fingerprints'
    
    # 해시 컬럼
    assert hasattr(VideoFingerprint, 'url_hash')
    assert hasattr(VideoFingerprint, 'thumbnail_hash')
    assert hasattr(VideoFingerprint, 'title_hash')


def test_video_history_model():
    """VideoHistory 모델 테스트"""
    from models import VideoHistory
    
    assert VideoHistory.__tablename__ == 'video_history'
    
    assert hasattr(VideoHistory, 'action')
    assert hasattr(VideoHistory, 'status')
    assert hasattr(VideoHistory, 'meta_data')


def test_episode_model():
    """Episode 모델 테스트"""
    from models import Episode
    
    assert Episode.__tablename__ == 'episodes'
    
    # 필수 컬럼
    required_columns = ['id', 'video_id', 'episode_id', 'start_frame',
                       'end_frame', 'confidence_score', 'quality_score']
    for col in required_columns:
        assert hasattr(Episode, col), f"컬럼 누락: {col}"


def test_dataset_version_model():
    """DatasetVersion 모델 테스트"""
    from models import DatasetVersion
    
    assert DatasetVersion.__tablename__ == 'dataset_versions'
    
    assert hasattr(DatasetVersion, 'version')
    assert hasattr(DatasetVersion, 'total_videos')
    assert hasattr(DatasetVersion, 'total_episodes')


def test_database_init(test_db):
    """데이터베이스 초기화 테스트"""
    from models import KeywordCategory, Keyword
    
    # 카테고리 생성
    category = KeywordCategory(
        name='test_category',
        description='Test category'
    )
    test_db.add(category)
    test_db.commit()
    
    # 조회 테스트
    result = test_db.query(KeywordCategory).filter_by(name='test_category').first()
    assert result is not None
    assert result.name == 'test_category'


def test_keyword_creation(test_db):
    """키워드 생성 테스트"""
    from models import KeywordCategory, Keyword
    
    # 카테고리 생성
    category = KeywordCategory(name='assembly', description='Assembly tasks')
    test_db.add(category)
    test_db.commit()
    
    # 키워드 생성
    keyword = Keyword(
        keyword='robot assembly',
        category_id=category.id,
        language='en',
        priority=8,
        weight=1.5,
        is_active=True
    )
    test_db.add(keyword)
    test_db.commit()
    
    # 검증
    result = test_db.query(Keyword).filter_by(keyword='robot assembly').first()
    assert result is not None
    assert result.priority == 8
    assert result.weight == 1.5
    assert result.is_active is True


def test_video_creation(test_db):
    """비디오 생성 테스트"""
    from models import Video
    
    video = Video(
        video_id='test_video_123',
        platform='youtube',
        url='https://youtube.com/watch?v=test_video_123',
        title='Test Video',
        duration_sec=300,
        status='discovered'
    )
    test_db.add(video)
    test_db.commit()
    
    # 검증
    result = test_db.query(Video).filter_by(video_id='test_video_123').first()
    assert result is not None
    assert result.platform == 'youtube'
    assert result.duration_sec == 300


def test_video_fingerprint_creation(test_db):
    """비디오 지문 생성 테스트"""
    from models import Video, VideoFingerprint
    import hashlib
    
    # 비디오 생성
    video = Video(
        video_id='test_video_456',
        platform='youtube',
        url='https://youtube.com/watch?v=test_video_456',
        title='Test Video 2',
        status='discovered'
    )
    test_db.add(video)
    test_db.commit()
    
    # 지문 생성
    url_hash = hashlib.sha256(video.url.encode()).hexdigest()
    fingerprint = VideoFingerprint(
        video_id=video.id,
        url_hash=url_hash
    )
    test_db.add(fingerprint)
    test_db.commit()
    
    # 검증
    result = test_db.query(VideoFingerprint).filter_by(video_id=video.id).first()
    assert result is not None
    assert result.url_hash == url_hash


def test_episode_creation(test_db):
    """에피소드 생성 테스트"""
    from models import Video, Episode
    
    # 비디오 생성
    video = Video(
        video_id='test_video_789',
        platform='youtube',
        url='https://youtube.com/watch?v=test_video_789',
        title='Test Video 3',
        status='processed'
    )
    test_db.add(video)
    test_db.commit()
    
    # 에피소드 생성
    episode = Episode(
        video_id=video.id,
        episode_id='ep_001',
        start_frame=100,
        end_frame=500,
        duration_frames=400,
        confidence_score=0.85,
        quality_score=0.9
    )
    test_db.add(episode)
    test_db.commit()
    
    # 검증
    result = test_db.query(Episode).filter_by(episode_id='ep_001').first()
    assert result is not None
    assert result.start_frame == 100
    assert result.end_frame == 500
    assert result.confidence_score == 0.85


def test_relationships(test_db):
    """모델 관계 테스트"""
    from models import Video, Episode
    
    # 비디오 생성
    video = Video(
        video_id='test_video_rel',
        platform='youtube',
        url='https://youtube.com/watch?v=test_video_rel',
        title='Relationship Test',
        status='processed'
    )
    test_db.add(video)
    test_db.commit()
    
    # 여러 에피소드 생성
    for i in range(3):
        episode = Episode(
            video_id=video.id,
            episode_id=f'ep_rel_{i:03d}',  # 고유한 ID 사용
            start_frame=i * 100,
            end_frame=(i + 1) * 100,
            confidence_score=0.8
        )
        test_db.add(episode)
    test_db.commit()
    
    # 관계 검증
    video = test_db.query(Video).filter_by(video_id='test_video_rel').first()
    assert len(video.episodes) == 3
    assert all(ep.video_id == video.id for ep in video.episodes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
