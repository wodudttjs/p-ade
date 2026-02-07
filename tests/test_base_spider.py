"""
Base Spider 테스트

BaseSpider 클래스의 기능을 테스트합니다.
"""

import pytest
from typing import Generator
from scrapy.http import Response, Request, HtmlResponse


def test_base_spider_import():
    """BaseSpider import 확인"""
    from spiders import BaseSpider
    assert BaseSpider is not None


def test_base_spider_is_abstract():
    """BaseSpider가 추상 클래스인지 확인"""
    from spiders import BaseSpider
    
    # 직접 인스턴스화 시도하면 에러 발생해야 함
    with pytest.raises(TypeError):
        BaseSpider(keywords=['test'])


def test_base_spider_initialization():
    """BaseSpider 초기화 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test_spider"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return f"https://example.com/search?q={keyword}&page={page}"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test1', 'test2'], max_results=50)
    
    assert spider.keywords == ['test1', 'test2']
    assert spider.max_results == 50
    assert spider.results_count == 0
    assert len(spider.user_agents) > 0


def test_base_spider_single_keyword():
    """단일 키워드 초기화 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return f"https://example.com/search?q={keyword}"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords='single_keyword', max_results=100)
    assert spider.keywords == ['single_keyword']


def test_user_agents_loaded():
    """User-Agent 목록 로드 확인"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'])
    user_agents = spider.user_agents
    
    assert isinstance(user_agents, list)
    assert len(user_agents) >= 3
    assert all('Mozilla' in ua for ua in user_agents)


def test_get_random_headers():
    """랜덤 헤더 생성 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'])
    headers = spider._get_random_headers()
    
    assert 'User-Agent' in headers
    assert 'Accept' in headers
    assert 'Accept-Language' in headers
    assert headers['User-Agent'] in spider.user_agents


def test_should_continue_crawling():
    """크롤링 계속 여부 판단 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'], max_results=10)
    
    # 초기 상태
    assert spider.should_continue_crawling() is True
    
    # 결과 수 증가
    spider.results_count = 5
    assert spider.should_continue_crawling() is True
    
    # 최대치 도달
    spider.results_count = 10
    assert spider.should_continue_crawling() is False
    
    # 최대치 초과
    spider.results_count = 15
    assert spider.should_continue_crawling() is False


def test_increment_results_count():
    """결과 카운트 증가 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'])
    
    assert spider.results_count == 0
    
    spider.increment_results_count()
    assert spider.results_count == 1
    
    spider.increment_results_count()
    assert spider.results_count == 2


def test_normalize_video_data():
    """비디오 데이터 정규화 테스트"""
    from spiders.base_spider import BaseSpider
    from datetime import datetime
    
    class TestSpider(BaseSpider):
        name = "test_platform"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'])
    
    raw_data = {
        'video_id': 'abc123',
        'url': 'https://example.com/video/abc123',
        'title': 'Test Video Title',
        'description': 'This is a test video description',
        'duration_sec': 300,
        'view_count': 10000,
        'channel_id': 'channel_123',
        'channel_name': 'Test Channel',
        'thumbnail_url': 'https://example.com/thumb.jpg',
        'tags': ['test', 'video'],
    }
    
    normalized = spider.normalize_video_data(raw_data)
    
    assert normalized['video_id'] == 'abc123'
    assert normalized['platform'] == 'test_platform'
    assert normalized['url'] == 'https://example.com/video/abc123'
    assert normalized['title'] == 'Test Video Title'
    assert normalized['duration_sec'] == 300
    assert normalized['view_count'] == 10000
    assert normalized['tags'] == ['test', 'video']
    assert 'discovered_at' in normalized


def test_normalize_video_data_with_long_description():
    """긴 설명 정규화 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'])
    
    long_description = 'x' * 1000  # 1000자
    raw_data = {
        'video_id': 'test',
        'url': 'https://example.com/video',
        'title': 'Test',
        'description': long_description,
    }
    
    normalized = spider.normalize_video_data(raw_data)
    
    # 설명이 500자로 제한되어야 함
    assert len(normalized['description']) == 500


def test_normalize_video_data_with_missing_fields():
    """누락된 필드가 있는 데이터 정규화 테스트"""
    from spiders.base_spider import BaseSpider
    
    class TestSpider(BaseSpider):
        name = "test"
        
        def build_search_url(self, keyword: str, page: int = 1) -> str:
            return "https://example.com"
        
        def parse_search_results(self, response: Response) -> Generator:
            yield {}
    
    spider = TestSpider(keywords=['test'])
    
    minimal_data = {
        'video_id': 'test123',
        'url': 'https://example.com/video/test123',
        'title': 'Minimal Video',
    }
    
    normalized = spider.normalize_video_data(minimal_data)
    
    # 기본값이 설정되어야 함
    assert normalized['video_id'] == 'test123'
    assert normalized['description'] == ''
    assert normalized['duration_sec'] == 0
    assert normalized['view_count'] == 0
    assert normalized['tags'] == []


def test_custom_settings():
    """커스텀 설정 확인"""
    from spiders.base_spider import BaseSpider
    
    settings = BaseSpider.custom_settings
    
    assert settings['CONCURRENT_REQUESTS'] == 8
    assert settings['DOWNLOAD_DELAY'] == 2.0
    assert settings['RANDOMIZE_DOWNLOAD_DELAY'] is True
    assert settings['RETRY_TIMES'] == 3
    assert 500 in settings['RETRY_HTTP_CODES']
    assert 429 in settings['RETRY_HTTP_CODES']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
