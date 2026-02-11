"""
Redis 캐시 테스트

캐시 기능의 정상 동작을 검증합니다.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.redis_cache import CrawlCache, CacheConfig, REDIS_AVAILABLE


class TestCrawlCache:
    """CrawlCache 테스트"""
    
    def test_init_without_redis(self):
        """Redis 없이 초기화 테스트"""
        with patch("cache.redis_cache.REDIS_AVAILABLE", False):
            cache = CrawlCache()
            assert not cache._connected
    
    def test_cache_config(self):
        """캐시 설정 테스트"""
        config = CacheConfig(
            host="localhost",
            port=6379,
            search_ttl=3600,
        )
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.search_ttl == 3600
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    def test_search_cache_miss(self):
        """검색 캐시 미스 테스트"""
        cache = CrawlCache()
        if not cache.is_connected:
            pytest.skip("Redis not connected")
        
        result = cache.get_search_results("nonexistent_keyword_xyz", "youtube")
        assert result is None
        assert cache.stats["search_misses"] >= 1
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    def test_search_cache_hit(self):
        """검색 캐시 히트 테스트"""
        cache = CrawlCache()
        if not cache.is_connected:
            pytest.skip("Redis not connected")
        
        test_keyword = "test_robot_arm_keyword"
        test_results = [
            {"video_id": "abc123", "title": "Robot Arm Demo"},
            {"video_id": "def456", "title": "Pick and Place"},
        ]
        
        # 캐시 저장
        cache.save_search_results(test_keyword, test_results, "youtube", ttl=60)
        
        # 캐시 조회
        cached = cache.get_search_results(test_keyword, "youtube")
        assert cached is not None
        assert len(cached) == 2
        assert cached[0]["video_id"] == "abc123"
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    def test_bloom_filter(self):
        """Bloom filter 중복 체크 테스트"""
        cache = CrawlCache()
        if not cache.is_connected:
            pytest.skip("Redis not connected")
        
        test_video_id = "test_video_bloom_12345"
        
        # 초기에는 수집되지 않음
        assert not cache.is_video_collected(test_video_id)
        
        # 수집 완료 마킹
        cache.mark_video_collected(test_video_id)
        
        # 이제 수집됨으로 표시
        assert cache.is_video_collected(test_video_id)
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    def test_batch_video_cache(self):
        """배치 비디오 캐시 테스트"""
        cache = CrawlCache()
        if not cache.is_connected:
            pytest.skip("Redis not connected")
        
        # 비디오 정보 저장
        cache.save_video_info("vid1", {"title": "Video 1"}, ttl=60)
        cache.save_video_info("vid2", {"title": "Video 2"}, ttl=60)
        
        # 배치 조회
        results = cache.get_videos_batch(["vid1", "vid2", "vid3"])
        
        assert results["vid1"]["title"] == "Video 1"
        assert results["vid2"]["title"] == "Video 2"
        assert results["vid3"] is None
    
    def test_stats(self):
        """통계 기능 테스트"""
        cache = CrawlCache()
        stats = cache.stats
        
        assert "search_hits" in stats
        assert "search_misses" in stats
        assert "search_hit_rate" in stats
        assert "connected" in stats


class TestCrawlCacheMocked:
    """Mock을 사용한 캐시 테스트"""
    
    def test_get_search_results_with_mock(self):
        """Mock Redis로 검색 캐시 테스트"""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = '{"video_id": "abc123"}'
        
        with patch("cache.redis_cache.redis.Redis", return_value=mock_redis):
            with patch("cache.redis_cache.REDIS_AVAILABLE", True):
                cache = CrawlCache()
                result = cache.get_search_results("robot arm", "youtube")
                # Mock이 호출되었는지 확인
                assert mock_redis.get.called
    
    def test_filter_uncollected(self):
        """수집되지 않은 비디오 필터링 테스트"""
        cache = CrawlCache()
        
        # 연결되지 않은 경우 모든 비디오가 반환되어야 함
        if not cache.is_connected:
            video_ids = ["v1", "v2", "v3"]
            result = cache.filter_uncollected(video_ids)
            assert result == video_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
