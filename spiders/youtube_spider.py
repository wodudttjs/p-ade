"""
YouTube Spider

YouTube 플랫폼 크롤러 - HTML 스크래핑 방식
"""

import json
import re
import argparse
from typing import Dict, Optional, Generator, List
from urllib.parse import quote_plus
from datetime import datetime
from pathlib import Path
import yt_dlp

import scrapy
from scrapy.http import Response
from scrapy.crawler import CrawlerProcess

from .base_spider import BaseSpider
from core.logging_config import logger


class SQLiteVideoPipeline:
    """Scrapy 아이템을 SQLite DB에 저장하는 파이프라인"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._engine = None
        self._session_factory = None
        self._session = None

    @classmethod
    def from_crawler(cls, crawler):
        db_path = crawler.settings.get("P_ADE_DB_PATH", "data/pade.db")
        return cls(db_path)

    def open_spider(self, spider):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models.database import Base

        db_path = Path(self.db_path)
        if not db_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            db_path = project_root / db_path

        self._engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        self._session = self._session_factory()

    def close_spider(self, spider):
        if self._session:
            self._session.close()
            self._session = None

    def process_item(self, item, spider):
        from models.database import Video, ProcessingJob

        video_id = item.get("video_id")
        if not video_id:
            return item

        session = self._session
        existing = session.query(Video).filter_by(video_id=video_id).first()

        discovered_at = None
        discovered_at_raw = item.get("discovered_at")
        if discovered_at_raw:
            try:
                discovered_at = datetime.fromisoformat(discovered_at_raw)
            except ValueError:
                discovered_at = datetime.utcnow()
        else:
            discovered_at = datetime.utcnow()

        if existing:
            if not existing.title and item.get("title"):
                existing.title = item.get("title")
            if not existing.description and item.get("description"):
                existing.description = item.get("description")
            if not existing.duration_sec and item.get("duration_sec"):
                existing.duration_sec = item.get("duration_sec")
            if not existing.channel_id and item.get("channel_id"):
                existing.channel_id = item.get("channel_id")
            if not existing.channel_name and item.get("channel_name"):
                existing.channel_name = item.get("channel_name")
            if not existing.thumbnail_url and item.get("thumbnail_url"):
                existing.thumbnail_url = item.get("thumbnail_url")
            if not existing.tags and item.get("tags"):
                existing.tags = item.get("tags")

            processing_version = "local"
            job_key = ProcessingJob.generate_job_key(existing.platform, video_id, processing_version)
            job = session.query(ProcessingJob).filter_by(job_key=job_key).first()
            now = datetime.utcnow()
            if not job:
                job = ProcessingJob(
                    job_key=job_key,
                    platform=existing.platform,
                    video_id=video_id,
                    processing_version=processing_version,
                    stage="discover",
                    status="completed",
                    started_at=now,
                    completed_at=now,
                )
                session.add(job)
            else:
                job.stage = "discover"
                job.status = "completed"
                if not job.started_at:
                    job.started_at = now
                job.completed_at = now

            session.commit()
            return item

        video = Video(
            video_id=video_id,
            platform=item.get("platform") or "youtube",
            url=item.get("url", ""),
            title=item.get("title"),
            description=item.get("description"),
            duration_sec=item.get("duration_sec"),
            channel_id=item.get("channel_id"),
            channel_name=item.get("channel_name"),
            view_count=item.get("view_count"),
            like_count=item.get("like_count"),
            thumbnail_url=item.get("thumbnail_url"),
            tags=item.get("tags"),
            status="discovered",
            discovered_at=discovered_at,
        )
        session.add(video)
        processing_version = "local"
        job_key = ProcessingJob.generate_job_key(video.platform, video_id, processing_version)
        job = ProcessingJob(
            job_key=job_key,
            platform=video.platform,
            video_id=video_id,
            processing_version=processing_version,
            stage="discover",
            status="completed",
            started_at=discovered_at,
            completed_at=discovered_at,
        )
        session.add(job)
        session.commit()
        return item


class UrlCsvPipeline:
    """크롤링 결과 URL을 CSV에 기록하는 파이프라인"""

    def __init__(self, csv_path: str, overwrite: bool = False):
        self.csv_path = csv_path
        self.overwrite = overwrite
        self._file = None
        self._writer = None

    @classmethod
    def from_crawler(cls, crawler):
        csv_path = crawler.settings.get("P_ADE_URLS_PATH", "data/urls.csv")
        overwrite = crawler.settings.get("P_ADE_URLS_OVERWRITE", False)
        return cls(csv_path, overwrite=overwrite)

    def open_spider(self, spider):
        csv_path = Path(self.csv_path)
        if not csv_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            csv_path = project_root / csv_path

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.overwrite else "a"
        self._file = csv_path.open(mode, encoding="utf-8", newline="")
        import csv
        self._writer = csv.DictWriter(self._file, fieldnames=["url", "video_id", "title"])
        if self._file.tell() == 0:
            self._writer.writeheader()

    def close_spider(self, spider):
        if self._file:
            self._file.close()
            self._file = None

    def process_item(self, item, spider):
        if not self._writer:
            return item
        self._writer.writerow({
            "url": item.get("url", ""),
            "video_id": item.get("video_id", ""),
            "title": item.get("title", ""),
        })
        return item


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
            for video_data in self._fallback_search_with_ytdlp(keyword):
                if not self.should_continue_crawling():
                    break
                normalized_data = self.normalize_video_data(video_data)
                self.increment_results_count()
                yield normalized_data
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
        try:
            body_text = response.text
        except AttributeError:
            body_text = response.body.decode("utf-8", errors="ignore")

        # ytInitialData 패턴 찾기
        pattern = r'var ytInitialData = ({.*?});'
        match = re.search(pattern, body_text)
        
        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ytInitialData JSON: {e}")
                return None
        
        # 대체 패턴 시도
        pattern2 = r'window\["ytInitialData"\] = ({.*?});'
        match2 = re.search(pattern2, body_text)
        
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

    def _fallback_search_with_ytdlp(self, keyword: str) -> Generator[Dict, None, None]:
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "force_generic_extractor": False,
            }
            search_url = f"ytsearch{self.max_results}:{keyword}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(search_url, download=False)

            if result and "entries" in result:
                for entry in result["entries"]:
                    if not entry:
                        continue
                    yield {
                        "video_id": entry.get("id"),
                        "url": entry.get("url") or self.VIDEO_URL_TEMPLATE.format(video_id=entry.get("id")),
                        "title": entry.get("title"),
                        "description": entry.get("description", ""),
                        "duration_sec": entry.get("duration"),
                        "view_count": entry.get("view_count"),
                        "channel_id": entry.get("channel_id"),
                        "channel_name": entry.get("channel"),
                        "thumbnail_url": entry.get("thumbnail"),
                        "tags": entry.get("tags", []),
                    }
        except Exception as e:
            logger.error(f"yt-dlp fallback failed: {e}")
    
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


def _parse_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    if "," in raw:
        return [k.strip() for k in raw.split(",") if k.strip()]
    return [raw.strip()]


def run_cli():
    parser = argparse.ArgumentParser(description="YouTube Spider 실행")
    parser.add_argument("--keywords", required=True, help="검색 키워드 (콤마로 다중 지정)")
    parser.add_argument("--max-results", type=int, default=100, help="최대 결과 수")
    parser.add_argument("--db", default="data/pade.db", help="SQLite DB 경로")
    parser.add_argument("--output", default="data/urls.csv", help="URL CSV 출력 경로")
    parser.add_argument("--overwrite", action="store_true", help="URL CSV 덮어쓰기")
    args = parser.parse_args()

    keywords = _parse_keywords(args.keywords)
    if not keywords:
        print("❌ 키워드를 지정해주세요.")
        return 1

    settings = {
        "LOG_LEVEL": "INFO",
        "P_ADE_DB_PATH": args.db,
        "P_ADE_URLS_PATH": args.output,
        "P_ADE_URLS_OVERWRITE": args.overwrite,
        "ITEM_PIPELINES": {
            "spiders.youtube_spider.SQLiteVideoPipeline": 300,
            "spiders.youtube_spider.UrlCsvPipeline": 400,
        },
    }

    process = CrawlerProcess(settings=settings)
    process.crawl(YouTubeSpider, keywords=keywords, max_results=args.max_results)
    process.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
