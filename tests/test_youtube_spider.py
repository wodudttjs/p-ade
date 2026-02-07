"""
YouTube Spider 테스트

YouTubeSpider 클래스의 기능을 테스트합니다.
"""

import pytest
import json
from scrapy.http import HtmlResponse, Request


def test_youtube_spider_import():
    """YouTubeSpider import 확인"""
    from spiders.youtube_spider import YouTubeSpider
    assert YouTubeSpider is not None


def test_youtube_spider_initialization():
    """YouTubeSpider 초기화 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    spider = YouTubeSpider(keywords=['robot assembly'], max_results=50)
    
    assert spider.name == 'youtube'
    assert spider.keywords == ['robot assembly']
    assert spider.max_results == 50


def test_build_search_url():
    """검색 URL 생성 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    spider = YouTubeSpider(keywords=['test'])
    
    url = spider.build_search_url('robot assembly')
    assert 'youtube.com/results' in url
    assert 'search_query=' in url
    assert 'robot' in url or 'assembly' in url or 'robot+assembly' in url


def test_build_search_url_with_special_characters():
    """특수 문자가 있는 키워드로 URL 생성 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    spider = YouTubeSpider(keywords=['test'])
    
    url = spider.build_search_url('C++ programming')
    assert 'youtube.com/results' in url
    assert 'search_query=' in url


def test_parse_duration_text_minutes():
    """분:초 형식 duration 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_duration_text('10:30') == 630
    assert YouTubeSpider._parse_duration_text('1:05') == 65
    assert YouTubeSpider._parse_duration_text('0:45') == 45


def test_parse_duration_text_hours():
    """시:분:초 형식 duration 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_duration_text('1:30:00') == 5400
    assert YouTubeSpider._parse_duration_text('2:15:30') == 8130


def test_parse_duration_text_invalid():
    """잘못된 duration 형식 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_duration_text('invalid') == 0
    assert YouTubeSpider._parse_duration_text('') == 0


def test_parse_view_count_simple():
    """간단한 조회수 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_view_count('1,234 views') == 1234
    assert YouTubeSpider._parse_view_count('567 views') == 567
    assert YouTubeSpider._parse_view_count('1 view') == 1


def test_parse_view_count_thousands():
    """천 단위 조회수 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_view_count('1.5K views') == 1500
    assert YouTubeSpider._parse_view_count('850K views') == 850000
    assert YouTubeSpider._parse_view_count('12K views') == 12000


def test_parse_view_count_millions():
    """백만 단위 조회수 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_view_count('1.2M views') == 1200000
    assert YouTubeSpider._parse_view_count('5M views') == 5000000
    assert YouTubeSpider._parse_view_count('10.5M views') == 10500000


def test_parse_view_count_billions():
    """십억 단위 조회수 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_view_count('1B views') == 1000000000
    assert YouTubeSpider._parse_view_count('2.5B views') == 2500000000


def test_parse_view_count_invalid():
    """잘못된 조회수 형식 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert YouTubeSpider._parse_view_count('invalid') == 0
    assert YouTubeSpider._parse_view_count('') == 0


def test_parse_video_renderer():
    """videoRenderer 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    spider = YouTubeSpider(keywords=['test'])
    
    # 샘플 videoRenderer 데이터
    video_renderer = {
        'videoId': 'test_video_123',
        'title': {
            'runs': [{'text': 'Test Video Title'}]
        },
        'lengthText': {
            'simpleText': '10:30'
        },
        'viewCountText': {
            'simpleText': '1.5K views'
        },
        'ownerText': {
            'runs': [{
                'text': 'Test Channel',
                'navigationEndpoint': {
                    'browseEndpoint': {
                        'browseId': 'channel_123'
                    }
                }
            }]
        },
        'publishedTimeText': {
            'simpleText': '2 days ago'
        },
        'thumbnail': {
            'thumbnails': [
                {'url': 'https://i.ytimg.com/vi/test/default.jpg'},
                {'url': 'https://i.ytimg.com/vi/test/hqdefault.jpg'}
            ]
        },
        'detailedMetadataSnippets': [{
            'snippetText': {
                'runs': [{'text': 'Test description'}]
            }
        }]
    }
    
    result = spider._parse_video_renderer(video_renderer)
    
    assert result is not None
    assert result['video_id'] == 'test_video_123'
    assert result['title'] == 'Test Video Title'
    assert result['duration_sec'] == 630
    assert result['view_count'] == 1500
    assert result['channel_name'] == 'Test Channel'
    assert result['channel_id'] == 'channel_123'
    assert 'youtube.com/watch?v=test_video_123' in result['url']


def test_parse_video_renderer_minimal():
    """최소한의 데이터가 있는 videoRenderer 파싱 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    spider = YouTubeSpider(keywords=['test'])
    
    video_renderer = {
        'videoId': 'minimal_video',
        'title': {
            'runs': [{'text': 'Minimal Video'}]
        }
    }
    
    result = spider._parse_video_renderer(video_renderer)
    
    assert result is not None
    assert result['video_id'] == 'minimal_video'
    assert result['title'] == 'Minimal Video'
    assert result['duration_sec'] == 0
    assert result['view_count'] == 0


def test_parse_video_renderer_no_video_id():
    """videoId가 없는 경우 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    spider = YouTubeSpider(keywords=['test'])
    
    video_renderer = {
        'title': {
            'runs': [{'text': 'No ID Video'}]
        }
    }
    
    result = spider._parse_video_renderer(video_renderer)
    
    assert result is None


def test_video_url_template():
    """비디오 URL 템플릿 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert 'youtube.com/watch?v=' in YouTubeSpider.VIDEO_URL_TEMPLATE
    assert '{video_id}' in YouTubeSpider.VIDEO_URL_TEMPLATE


def test_search_url_template():
    """검색 URL 템플릿 테스트"""
    from spiders.youtube_spider import YouTubeSpider
    
    assert 'youtube.com/results' in YouTubeSpider.SEARCH_URL_TEMPLATE
    assert 'search_query=' in YouTubeSpider.SEARCH_URL_TEMPLATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
