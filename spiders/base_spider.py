"""
Base Spider 클래스

모든 비디오 플랫폼 크롤러의 추상 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Generator
import time
import random
from datetime import datetime

import scrapy
from scrapy.http import Response
from loguru import logger


class BaseSpider(scrapy.Spider, ABC):
    """모든 비디오 플랫폼 스파이더의 베이스 클래스"""
    
    name = "base_spider"
    
    custom_settings = {
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_DELAY': 2.0,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    def __init__(self, keywords: List[str], max_results: int = 100, pages: int = 1, *args, **kwargs):
        """
        Args:
            keywords: 검색할 키워드 리스트
            max_results: 최대 결과 수
            pages: 검색 페이지 수 (페이지네이션을 지원하는 스파이더에서 사용)
        """
        super().__init__(*args, **kwargs)
        self.keywords = keywords if isinstance(keywords, list) else [keywords]
        self.max_results = max_results
        self.pages = max(1, int(pages))
        self.results_count = 0
        self.user_agents = self._load_user_agents()
        
        logger.info(f"Spider initialized: {self.name}")
        logger.info(f"Keywords: {self.keywords}")
        logger.info(f"Max results: {max_results}")
        logger.info(f"Pages: {self.pages}")
    
    def _load_user_agents(self) -> List[str]:
        """다양한 User-Agent 문자열 로드"""
        return [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        ]
    
    def start_requests(self) -> Generator[scrapy.Request, None, None]:
        """초기 요청 생성"""
        for keyword in self.keywords:
            if not self.should_continue_crawling():
                break
                
            logger.info(f"Starting search for keyword: {keyword}")

            # 페이지네이션 지원: 페이지 인덱스를 meta로 전달
            for page in range(1, self.pages + 1):
                if not self.should_continue_crawling():
                    break

                url = self.build_search_url(keyword, page=page)
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_search_results,
                    errback=self.handle_error,
                    headers=self._get_random_headers(),
                    meta={'keyword': keyword, 'page': page}
                )
    
    @abstractmethod
    def build_search_url(self, keyword: str, page: int = 1) -> str:
        """
        플랫폼별 검색 URL 생성 (서브클래스에서 구현)
        
        Args:
            keyword: 검색 키워드
            page: 페이지 번호
            
        Returns:
            검색 URL
        """
        pass
    
    @abstractmethod
    def parse_search_results(self, response: Response) -> Generator:
        """
        검색 결과 파싱 (서브클래스에서 구현)
        
        Args:
            response: Scrapy Response 객체
            
        Yields:
            비디오 메타데이터 딕셔너리
        """
        pass
    
    def _get_random_headers(self) -> Dict[str, str]:
        """랜덤 헤더 생성"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ko;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        }
    
    def handle_error(self, failure):
        """에러 핸들링"""
        logger.error(f"Request failed: {failure.request.url}")
        logger.error(f"Reason: {failure.value}")
    
    def should_continue_crawling(self) -> bool:
        """크롤링 계속 여부 판단"""
        return self.results_count < self.max_results
    
    def increment_results_count(self) -> None:
        """결과 카운트 증가"""
        self.results_count += 1
        if self.results_count % 10 == 0:
            logger.info(f"Progress: {self.results_count}/{self.max_results} results collected")
    
    def normalize_video_data(self, data: Dict) -> Dict:
        """
        비디오 데이터 정규화
        
        Args:
            data: 원본 비디오 데이터
            
        Returns:
            정규화된 비디오 데이터
        """
        normalized = {
            'video_id': data.get('video_id', ''),
            'platform': self.name,
            'url': data.get('url', ''),
            'title': data.get('title', ''),
            'description': data.get('description', '')[:500] if data.get('description') else '',
            'duration_sec': data.get('duration_sec', 0),
            'upload_date': data.get('upload_date'),
            'view_count': data.get('view_count', 0),
            'channel_id': data.get('channel_id', ''),
            'channel_name': data.get('channel_name', ''),
            'thumbnail_url': data.get('thumbnail_url', ''),
            'tags': data.get('tags', []),
            'discovered_at': datetime.utcnow().isoformat(),
        }
        
        return normalized
