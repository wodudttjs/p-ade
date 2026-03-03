"""
YouTube 배치 API 테스트

YouTubeBatchClient의 배치 요청, 할당량 추적, ISO duration 파싱을 검증합니다.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.youtube_batch import (
    YouTubeBatchClient,
    QuotaTracker,
    BatchVideoInfo,
    _parse_iso_duration,
    _safe_int,
    MAX_BATCH_SIZE,
    DAILY_QUOTA_LIMIT,
)


class TestISODurationParser:
    """ISO 8601 duration 파서 테스트"""
    
    def test_hours_minutes_seconds(self):
        assert _parse_iso_duration("PT1H2M3S") == 3723
    
    def test_minutes_seconds(self):
        assert _parse_iso_duration("PT5M30S") == 330
    
    def test_seconds_only(self):
        assert _parse_iso_duration("PT45S") == 45
    
    def test_hours_only(self):
        assert _parse_iso_duration("PT2H") == 7200
    
    def test_empty_string(self):
        assert _parse_iso_duration("") is None
    
    def test_invalid_format(self):
        assert _parse_iso_duration("invalid") is None
    
    def test_none_input(self):
        assert _parse_iso_duration(None) is None


class TestSafeInt:
    """안전한 정수 변환 테스트"""
    
    def test_string_number(self):
        assert _safe_int("12345") == 12345
    
    def test_integer(self):
        assert _safe_int(42) == 42
    
    def test_none(self):
        assert _safe_int(None) is None
    
    def test_invalid_string(self):
        assert _safe_int("not_a_number") is None


class TestQuotaTracker:
    """할당량 추적기 테스트"""
    
    def test_initial_state(self):
        tracker = QuotaTracker()
        assert tracker.used == 0
        assert tracker.remaining == DAILY_QUOTA_LIMIT
        assert tracker.usage_percent == 0
    
    def test_record_usage(self):
        tracker = QuotaTracker()
        tracker.record("videos.list")  # 1 unit
        assert tracker.used == 1
        assert tracker.calls["videos.list"] == 1
    
    def test_record_search(self):
        tracker = QuotaTracker()
        tracker.record("search.list")  # 100 units
        assert tracker.used == 100
        assert tracker.remaining == DAILY_QUOTA_LIMIT - 100
    
    def test_can_afford(self):
        tracker = QuotaTracker(daily_limit=50)
        assert tracker.can_afford("videos.list")  # 1 unit
        assert not tracker.can_afford("search.list") is False  # need to check
        
    def test_estimate_batch_calls(self):
        tracker = QuotaTracker()
        est = tracker.estimate_batch_calls(100)
        
        assert est["video_count"] == 100
        assert est["batch_calls_needed"] == 2  # 100/50 = 2
        assert est["total_quota_cost"] == 2  # 2 calls * 1 unit
        assert est["individual_cost"] == 100  # 100 * 1 unit
        assert est["savings"] == 98
        assert est["savings_percent"] == 98.0
        assert est["can_afford"] is True
    
    def test_estimate_single_batch(self):
        tracker = QuotaTracker()
        est = tracker.estimate_batch_calls(50)
        assert est["batch_calls_needed"] == 1
    
    def test_estimate_partial_batch(self):
        tracker = QuotaTracker()
        est = tracker.estimate_batch_calls(75)
        assert est["batch_calls_needed"] == 2  # ceil(75/50)
    
    def test_summary(self):
        tracker = QuotaTracker()
        tracker.record("videos.list")
        tracker.record("search.list")
        summary = tracker.summary()
        assert "할당량" in summary
        assert "videos.list" in summary


class TestBatchVideoInfo:
    """BatchVideoInfo 테스트"""
    
    def test_from_api_item(self):
        item = {
            "id": "abc123",
            "snippet": {
                "title": "Robot Arm Demo",
                "description": "A demo video",
                "channelId": "ch123",
                "channelTitle": "TestChannel",
                "tags": ["robot", "arm"],
                "publishedAt": "2026-01-01T00:00:00Z",
                "categoryId": "28",
                "thumbnails": {
                    "high": {"url": "https://example.com/thumb.jpg"},
                },
            },
            "contentDetails": {
                "duration": "PT5M30S",
            },
            "statistics": {
                "viewCount": "1000",
                "likeCount": "50",
                "commentCount": "10",
            },
        }
        
        video = BatchVideoInfo.from_api_item(item)
        
        assert video.video_id == "abc123"
        assert video.title == "Robot Arm Demo"
        assert video.duration_sec == 330
        assert video.view_count == 1000
        assert video.like_count == 50
        assert video.channel_id == "ch123"
        assert video.tags == ["robot", "arm"]
    
    def test_to_dict(self):
        video = BatchVideoInfo(
            video_id="test123",
            title="Test Video",
            duration_sec=120,
        )
        d = video.to_dict()
        
        assert d["video_id"] == "test123"
        assert d["title"] == "Test Video"
        assert d["duration_sec"] == 120
        assert "url" in d
        assert "test123" in d["url"]


class TestYouTubeBatchClient:
    """YouTubeBatchClient 테스트"""
    
    def test_init_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            client = YouTubeBatchClient(api_key="")
            assert client.api_key == ""
    
    def test_init_with_key(self):
        client = YouTubeBatchClient(api_key="test_key")
        assert client.api_key == "test_key"
    
    def test_get_videos_batch_no_key(self):
        client = YouTubeBatchClient(api_key="")
        result = client.get_videos_batch(["id1", "id2"])
        assert result == []
    
    def test_get_videos_batch_empty_ids(self):
        client = YouTubeBatchClient(api_key="test_key")
        result = client.get_videos_batch([])
        assert result == []
    
    @patch("requests.get")
    def test_get_videos_batch_success(self, mock_get):
        """배치 조회 성공 테스트"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "vid1",
                    "snippet": {
                        "title": "Video 1",
                        "description": "",
                        "channelId": "",
                        "channelTitle": "",
                        "thumbnails": {},
                    },
                    "contentDetails": {"duration": "PT1M"},
                    "statistics": {},
                },
                {
                    "id": "vid2",
                    "snippet": {
                        "title": "Video 2",
                        "description": "",
                        "channelId": "",
                        "channelTitle": "",
                        "thumbnails": {},
                    },
                    "contentDetails": {"duration": "PT2M"},
                    "statistics": {},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        client = YouTubeBatchClient(api_key="test_key")
        result = client.get_videos_batch(["vid1", "vid2"])
        
        assert len(result) == 2
        assert result[0].video_id == "vid1"
        assert result[1].video_id == "vid2"
        assert client.quota.used == 1  # 1 batch call = 1 unit
    
    def test_quota_usage_property(self):
        client = YouTubeBatchClient(api_key="test")
        usage = client.quota_usage
        
        assert "used" in usage
        assert "remaining" in usage
        assert "daily_limit" in usage
        assert usage["used"] == 0
    
    def test_batch_size_constant(self):
        assert MAX_BATCH_SIZE == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
