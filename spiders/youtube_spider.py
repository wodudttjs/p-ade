"""
YouTube Spider

YouTube 플랫폼 크롤러 - HTML 스크래핑 방식
"""

import json
import re
from typing import Dict, Optional, Generator
from urllib.parse import urlencode, quote_plus
from datetime import datetime

import scrapy
from scrapy.http import Response

from .base_spider import BaseSpider
from core.logging_config import logger


class YouTubeSpider(BaseSpider):
    """YouTube 검색 및 비디오 메타데이터 수집"""
    
    name = "youtube"
    allowed_domains = ["youtube.com", "www.youtube.com"]
    
    # YouTube 검색 URL 템플릿
    SEARCH_URL_TEMPLATE = "https://www.youtube.com/results?search_query={query}"
    VIDEO_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"
    
    def build_search_url(self, keyword: str, page: int = 1) -> str:
        """
        YouTube 검색 URL 생성
        
        Args:
            keyword: 검색 키워드
            page: 페이지 번호 (YouTube는 무한 스크롤이지만 첫 페이지만 사용)
            
        Returns:
            검색 URL
        """
        encoded_query = quote_plus(keyword)
        return self.SEARCH_URL_TEMPLATE.format(query=encoded_query)
    
    def parse_search_results(self, response: Response) -> Generator:
        """
        검색 결과 페이지 파싱
        
        YouTube는 JavaScript로 렌더링되므로 HTML에서 직접 데이터 추출
        """
        keyword = response.meta.get('keyword', '')
        logger.info(f"Parsing search results for keyword: {keyword}")
        
        # ytInitialData에서 JSON 데이터 추출
        yt_initial_data = self._extract_yt_initial_data(response)
        
        if not yt_initial_data:
            logger.warning(f"Could not extract ytInitialData from {response.url}")
            return
        
        # 검색 결과에서 비디오 항목 추출
        videos = self._extract_videos_from_search(yt_initial_data)
        
        for video_data in videos:
            if not self.should_continue_crawling():
                break
            
            # 비디오 데이터 정규화 및 반환
            normalized_data = self.normalize_video_data(video_data)
            self.increment_results_count()
            
            yield normalized_data
    
    def _extract_yt_initial_data(self, response: Response) -> Optional[Dict]:
        """
        ytInitialData JSON 추출
        
        YouTube 페이지의 JavaScript에서 JSON 데이터를 추출합니다.
        """
        # ytInitialData 패턴 찾기
        pattern = r'var ytInitialData = ({.*?});'
        match = re.search(pattern, response.text)
        
        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ytInitialData JSON: {e}")
                return None
        
        # 대체 패턴 시도
        pattern2 = r'window\["ytInitialData"\] = ({.*?});'
        match2 = re.search(pattern2, response.text)
        
        if match2:
            try:
                json_str = match2.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ytInitialData JSON (pattern 2): {e}")
                return None
        
        return None
    
    def _extract_videos_from_search(self, yt_data: Dict) -> Generator[Dict, None, None]:
        """
        검색 결과에서 비디오 정보 추출
        
        Args:
            yt_data: ytInitialData JSON
            
        Yields:
            비디오 메타데이터 딕셔너리
        """
        try:
            # 검색 결과 컨텐츠 탐색
            contents = yt_data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
            
            for content in contents:
                item_section = content.get('itemSectionRenderer', {})
                for item in item_section.get('contents', []):
                    video_renderer = item.get('videoRenderer')
                    
                    if video_renderer:
                        video_data = self._parse_video_renderer(video_renderer)
                        if video_data:
                            yield video_data
        
        except Exception as e:
            logger.error(f"Error extracting videos from search results: {e}")
    
    def _parse_video_renderer(self, video_renderer: Dict) -> Optional[Dict]:
        """
        videoRenderer에서 메타데이터 추출
        
        Args:
            video_renderer: videoRenderer JSON 객체
            
        Returns:
            비디오 메타데이터 또는 None
        """
        try:
            video_id = video_renderer.get('videoId')
            if not video_id:
                return None
            
            # 제목 추출
            title_runs = video_renderer.get('title', {}).get('runs', [])
            title = title_runs[0].get('text', '') if title_runs else ''
            
            # 설명 추출
            description_snippet = video_renderer.get('detailedMetadataSnippets', [{}])[0].get('snippetText', {}).get('runs', [])
            description = ''.join([run.get('text', '') for run in description_snippet])
            
            # 길이 추출
            length_text = video_renderer.get('lengthText', {}).get('simpleText', '0:00')
            duration_sec = self._parse_duration_text(length_text)
            
            # 조회수 추출
            view_count_text = video_renderer.get('viewCountText', {}).get('simpleText', '0 views')
            view_count = self._parse_view_count(view_count_text)
            
            # 채널 정보
            owner_text = video_renderer.get('ownerText', {}).get('runs', [{}])[0]
            channel_name = owner_text.get('text', '')
            channel_id = owner_text.get('navigationEndpoint', {}).get('browseEndpoint', {}).get('browseId', '')
            
            # 업로드 날짜 (상대적 시간만 제공됨)
            published_time_text = video_renderer.get('publishedTimeText', {}).get('simpleText', '')
            
            # 썸네일
            thumbnails = video_renderer.get('thumbnail', {}).get('thumbnails', [])
            thumbnail_url = thumbnails[-1].get('url', '') if thumbnails else ''
            
            return {
                'video_id': video_id,
                'url': self.VIDEO_URL_TEMPLATE.format(video_id=video_id),
                'title': title,
                'description': description,
                'duration_sec': duration_sec,
                'view_count': view_count,
                'channel_id': channel_id,
                'channel_name': channel_name,
                'thumbnail_url': thumbnail_url,
                'published_time_text': published_time_text,
                'tags': [],  # HTML 스크래핑에서는 태그를 얻을 수 없음
            }
        
        except Exception as e:
            logger.error(f"Error parsing video renderer: {e}")
            return None
    
    @staticmethod
    def _parse_duration_text(duration_text: str) -> int:
        """
        텍스트 duration을 초 단위로 변환
        
        Args:
            duration_text: "10:30" 또는 "1:05:20" 형식
            
        Returns:
            초 단위 시간
        """
        try:
            parts = duration_text.split(':')
            parts = [int(p) for p in parts]
            
            if len(parts) == 2:  # MM:SS
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3:  # HH:MM:SS
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                return 0
        except (ValueError, AttributeError):
            return 0
    
    @staticmethod
    def _parse_view_count(view_text: str) -> int:
        """
        조회수 텍스트 파싱
        
        Args:
            view_text: "1.2M views", "850K views", "1,234 views" 등
            
        Returns:
            조회수 (정수)
        """
        try:
            # "views" 제거
            text = view_text.lower().replace('views', '').replace('view', '').strip()
            
            # 콤마 제거
            text = text.replace(',', '')
            
            # K, M, B 처리
            multiplier = 1
            if 'k' in text:
                multiplier = 1_000
                text = text.replace('k', '')
            elif 'm' in text:
                multiplier = 1_000_000
                text = text.replace('m', '')
            elif 'b' in text:
                multiplier = 1_000_000_000
                text = text.replace('b', '')
            
            number = float(text.strip())
            return int(number * multiplier)
        
        except (ValueError, AttributeError):
            return 0
