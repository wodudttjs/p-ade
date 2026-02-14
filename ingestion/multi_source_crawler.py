"""
다중 플랫폼 병렬 크롤러

YouTube, Google Videos, Vimeo, Dailymotion에서 병렬로 영상을 크롤링합니다.

기능:
- 소스별 독립 레이트 리미팅
- 병렬 크롤링 (ThreadPoolExecutor)
- 중복 URL/video_id 제거
- 품질/관련성 사전 필터링
- 진행률 콜백 & 통계 리포트
- DB 동기화 (Video/ProcessingJob upsert)
"""

import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Callable, Set, Any
from dataclasses import dataclass, field, asdict

import yt_dlp
import requests
import re
from urllib.parse import quote_plus

from core.logging_config import setup_logger
from ingestion.rate_limiter import SourceRateLimiter, RetryManager, RetryConfig

# 캐시 (선택적 임포트)
try:
    from cache.redis_cache import CrawlCache, get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logger = setup_logger(__name__)


# ============================================================
# 데이터 모델
# ============================================================

@dataclass
class CrawlResult:
    """크롤링 결과"""
    video_id: str
    url: str
    title: str = ""
    description: str = ""
    duration_sec: Optional[int] = None
    view_count: Optional[int] = None
    channel_id: str = ""
    channel_name: str = ""
    thumbnail_url: str = ""
    tags: List[str] = field(default_factory=list)
    platform: str = ""
    keyword: str = ""
    discovered_at: str = ""


@dataclass
class CrawlStats:
    """크롤링 통계"""
    total_searched: int = 0
    total_found: int = 0
    total_filtered: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    by_keyword: Dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"📊 크롤링 결과 요약",
            f"{'='*60}",
            f"  총 검색: {self.total_searched}개",
            f"  총 발견: {self.total_found}개",
            f"  중복 제거: {self.total_duplicates}개",
            f"  필터링됨: {self.total_filtered}개",
            f"  에러: {self.total_errors}개",
            f"  소요 시간: {self.elapsed_sec:.1f}초",
        ]
        if self.by_source:
            lines.append(f"\n  소스별:")
            for src, cnt in sorted(self.by_source.items()):
                lines.append(f"    {src}: {cnt}개")
        return "\n".join(lines)


# ============================================================
# 필터 설정
# ============================================================

# 관련성 키워드 (제목/설명에 포함되어야 함)
RELEVANCE_KEYWORDS = [
    "robot", "robotic", "arm", "gripper", "manipulator",
    "pick", "place", "grasping", "cobot", "industrial",
    "FANUC", "ABB", "KUKA", "UR5", "UR10", "UR3",
    "automation", "manufacturing", "assembly",
    "로봇", "로봇팔", "그리퍼", "매니퓰레이터",
]

# 거부 키워드
REJECT_KEYWORDS = [
    "animation", "cartoon", "cgi", "3d render", "toy",
    "lego", "surgery", "medical", "prosthetic", "game",
    "minecraft", "roblox", "fortnite",
    "unboxing", "review", "price", "how much",
]

# 길이 필터
MIN_DURATION_SEC = 30
MAX_DURATION_SEC = 1200  # 20분


def _passes_content_filter(title: str, description: str = "") -> bool:
    """콘텐츠 관련성 필터"""
    text = f"{title} {description}".lower()

    # 거부 키워드 체크
    for rk in REJECT_KEYWORDS:
        if rk in text:
            return False

    # 관련성 키워드 체크 (최소 1개)
    for kw in RELEVANCE_KEYWORDS:
        if kw.lower() in text:
            return True

    return False


def _passes_duration_filter(
    duration: Optional[int],
    min_sec: int = MIN_DURATION_SEC,
    max_sec: int = MAX_DURATION_SEC,
) -> bool:
    """길이 필터"""
    if duration is None:
        return True  # 길이 정보 없으면 통과 (나중에 다운로드 시 필터)
    return min_sec <= duration <= max_sec


# ============================================================
# 소스별 크롤러
# ============================================================

class _YtDlpSearcher:
    """yt-dlp 기반 검색기 (YouTube, Google Videos)"""

    EXTRACTOR_MAP = {
        "youtube": "ytsearch",
        "google_videos": "gvsearch",
    }

    def __init__(self, rate_limiter: SourceRateLimiter):
        self._limiter = rate_limiter
        self._retry = RetryManager(RetryConfig(max_retries=3))

    def search(
        self,
        source: str,
        keyword: str,
        max_results: int = 50,
        get_full_info: bool = False,
    ) -> List[CrawlResult]:
        """yt-dlp 검색 실행"""
        extractor = self.EXTRACTOR_MAP.get(source)
        if not extractor:
            logger.warning(f"지원하지 않는 소스: {source}")
            return []

        self._limiter.wait_for(source)

        query = f"{extractor}{max_results}:{keyword}"
        ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
        results: List[CrawlResult] = []

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                raw = ydl.extract_info(query, download=False)

            if not raw or "entries" not in raw:
                return []

            for entry in raw.get("entries", []):
                if not entry:
                    continue

                duration = entry.get("duration")

                # flat 모드에서 duration이 없으면 상세 정보 요청
                if get_full_info and duration is None:
                    self._limiter.wait_for(source)
                    try:
                        with yt_dlp.YoutubeDL(
                            {"quiet": True, "no_warnings": True, "skip_download": True}
                        ) as ydl2:
                            full = ydl2.extract_info(
                                entry.get("url") or entry.get("webpage_url"),
                                download=False,
                            )
                            if full:
                                duration = full.get("duration")
                                entry.update({
                                    k: v for k, v in full.items()
                                    if k in ("description", "view_count", "channel_id",
                                             "uploader", "thumbnail", "tags", "duration")
                                })
                    except Exception:
                        pass

                cr = CrawlResult(
                    video_id=entry.get("id", ""),
                    url=entry.get("url") or entry.get("webpage_url") or "",
                    title=entry.get("title", ""),
                    description=entry.get("description", ""),
                    duration_sec=duration,
                    view_count=entry.get("view_count"),
                    channel_id=entry.get("channel_id", ""),
                    channel_name=entry.get("uploader", ""),
                    thumbnail_url=entry.get("thumbnail", ""),
                    tags=entry.get("tags", []) or [],
                    platform=source,
                    keyword=keyword,
                    discovered_at=datetime.now(timezone.utc).isoformat(),
                )
                results.append(cr)

        except Exception as e:
            logger.error(f"검색 실패 [{source}] '{keyword}': {e}")

        return results


class _SiteScraper:
    """사이트 스크래핑 기반 검색기 (Vimeo, Dailymotion)"""

    def __init__(self, rate_limiter: SourceRateLimiter):
        self._limiter = rate_limiter

    def search(
        self,
        source: str,
        keyword: str,
        max_results: int = 20,
    ) -> List[CrawlResult]:
        """사이트 스크래핑으로 검색"""
        self._limiter.wait_for(source)

        urls = self._scrape_urls(source, keyword, max_results)
        results: List[CrawlResult] = []

        for url in urls:
            self._limiter.wait_for(source)
            try:
                with yt_dlp.YoutubeDL(
                    {"quiet": True, "no_warnings": True, "skip_download": True}
                ) as ydl:
                    info = ydl.extract_info(url, download=False)

                if not info:
                    continue

                cr = CrawlResult(
                    video_id=info.get("id", ""),
                    url=info.get("webpage_url") or url,
                    title=info.get("title", ""),
                    description=info.get("description", ""),
                    duration_sec=info.get("duration"),
                    view_count=info.get("view_count"),
                    channel_id=info.get("channel_id", ""),
                    channel_name=info.get("uploader") or info.get("uploader_id", ""),
                    thumbnail_url=info.get("thumbnail", ""),
                    tags=info.get("tags", []) or [],
                    platform=source,
                    keyword=keyword,
                    discovered_at=datetime.now(timezone.utc).isoformat(),
                )
                results.append(cr)
            except Exception as e:
                logger.debug(f"메타데이터 추출 실패 [{source}] {url}: {e}")

        return results

    def _scrape_urls(self, source: str, keyword: str, limit: int) -> List[str]:
        """검색 페이지에서 URL 스크래핑"""
        urls: List[str] = []
        try:
            if source == "vimeo":
                search_url = f"https://vimeo.com/search?q={quote_plus(keyword)}"
                pattern = r'https?://vimeo\.com/\d+'
            elif source == "dailymotion":
                search_url = f"https://www.dailymotion.com/search/{quote_plus(keyword)}/videos"
                pattern = r'https?://www\.dailymotion\.com/video/[a-zA-Z0-9]+'
            elif source == "bilibili":
                search_url = f"https://search.bilibili.com/all?keyword={quote_plus(keyword)}"
                pattern = r'https?://www\.bilibili\.com/video/[A-Za-z0-9]+'
            elif source == "rutube":
                search_url = f"https://rutube.ru/api/search/video/?query={quote_plus(keyword)}"
                pattern = r'https?://rutube\.ru/video/[a-f0-9]+'
            else:
                return []

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(search_url, headers=headers, timeout=15)
            found = re.findall(pattern, resp.text)

            seen: Set[str] = set()
            for u in found:
                u = u.rstrip("/")
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
                if len(urls) >= limit:
                    break
        except Exception as e:
            logger.error(f"스크래핑 실패 [{source}] '{keyword}': {e}")

        return urls


# ============================================================
# 다중 소스 병렬 크롤러
# ============================================================

class MultiSourceCrawler:
    """다중 플랫폼 병렬 크롤러

    사용법:
        crawler = MultiSourceCrawler(
            sources=["youtube", "google_videos", "vimeo"],
            max_results=500,
            max_workers=4,
        )
        results, stats = crawler.crawl(keywords=["robot arm", "pick and place"])
    """

    SUPPORTED_SOURCES = ["youtube", "google_videos", "vimeo", "dailymotion", "bilibili", "rutube"]

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        max_results: int = 500,
        max_workers: int = 4,
        get_full_info: bool = False,
        min_duration_sec: int = MIN_DURATION_SEC,
        max_duration_sec: int = MAX_DURATION_SEC,
        content_filter: bool = True,
        use_cache: bool = True,  # 캐시 사용 여부
        async_mode: bool = False,  # 비동기 모드 사용 여부
    ):
        self.sources = sources or ["youtube", "google_videos"]
        self.max_results = max_results
        self.max_workers = max_workers
        self.get_full_info = get_full_info
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.content_filter = content_filter
        self.use_cache = use_cache and CACHE_AVAILABLE
        self.async_mode = async_mode

        self._rate_limiter = SourceRateLimiter()
        self._ydl_searcher = _YtDlpSearcher(self._rate_limiter)
        self._site_scraper = _SiteScraper(self._rate_limiter)

        self._seen_ids: Set[str] = set()
        self._lock = threading.Lock()
        
        # 캐시 초기화
        self._cache: Optional[CrawlCache] = None
        if self.use_cache:
            try:
                self._cache = get_cache()
                logger.info(f"🔍 캐시 상태: _connected={self._cache._connected}, client={self._cache._client is not None}")
                if self._cache._connected or self._cache.is_connected:
                    logger.info("✅ Redis 캐시 연결됨")
                else:
                    # 재연결 시도
                    logger.info("🔄 Redis 재연결 시도 중...")
                    if self._cache.reconnect():
                        logger.info("✅ Redis 캐시 재연결 성공")
                    else:
                        logger.warning("⚠️ Redis 연결 실패, 캐시 없이 진행")
                        self._cache = None
            except Exception as e:
                logger.warning(f"⚠️ 캐시 초기화 실패: {type(e).__name__}: {e}")
                self._cache = None

    def crawl(
        self,
        keywords: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple:  # (List[CrawlResult], CrawlStats)
        """
        다중 소스에서 병렬 크롤링 실행
        
        async_mode=True인 경우 비동기 크롤러로 위임합니다.

        Args:
            keywords: 검색 키워드 리스트
            progress_callback: 진행률 콜백 (현재, 전체)

        Returns:
            (결과 리스트, 통계 객체)
        """
        # 비동기 모드 위임
        if self.async_mode:
            return self._crawl_async(keywords, progress_callback)
        
        start_time = time.time()
        stats = CrawlStats()
        all_results: List[CrawlResult] = []

        # 작업 목록 생성: (source, keyword, per_task_limit)
        tasks = []
        per_task = max(1, self.max_results // max(1, len(self.sources) * len(keywords)))

        for source in self.sources:
            if source not in self.SUPPORTED_SOURCES:
                logger.warning(f"미지원 소스: {source}")
                continue
            for kw in keywords:
                tasks.append((source, kw, per_task))

        total_tasks = len(tasks)
        completed_tasks = 0

        logger.info(
            f"🚀 크롤링 시작: {len(keywords)}개 키워드 × {len(self.sources)}개 소스 = {total_tasks}개 작업"
        )

        # 소스 타입별로 그룹화하여 병렬 실행
        # yt-dlp 기반과 scraper 기반을 분리
        ydl_tasks = [(s, kw, lim) for s, kw, lim in tasks if s in ("youtube", "google_videos")]
        scrape_tasks = [(s, kw, lim) for s, kw, lim in tasks if s in ("vimeo", "dailymotion", "bilibili", "rutube")]

        # yt-dlp 태스크 병렬 실행 (소스별 레이트 리밋 적용되므로 워커 수 제한)
        ydl_workers = min(self.max_workers, len(ydl_tasks)) if ydl_tasks else 0
        scrape_workers = min(2, len(scrape_tasks)) if scrape_tasks else 0

        def _execute_task(task_info):
            source, kw, limit = task_info
            
            # 캐시 조회 시도
            if self._cache:
                cache_key = f"{source}:{kw}"
                cached = self._cache.get_search_results(kw, source)
                if cached:
                    logger.debug(f"📋 캐시 히트: [{source}] '{kw}' ({len(cached)}개)")
                    # 캐시된 결과를 CrawlResult로 변환
                    results = []
                    for item in cached:
                        results.append(CrawlResult(
                            video_id=item.get("video_id", ""),
                            url=item.get("url", ""),
                            title=item.get("title", ""),
                            description=item.get("description", ""),
                            duration_sec=item.get("duration_sec"),
                            view_count=item.get("view_count"),
                            channel_id=item.get("channel_id", ""),
                            channel_name=item.get("channel_name", ""),
                            thumbnail_url=item.get("thumbnail_url", ""),
                            tags=item.get("tags", []),
                            platform=source,
                            keyword=kw,
                            discovered_at=item.get("discovered_at", ""),
                        ))
                    return results
            
            # 실제 검색 실행
            if source in ("youtube", "google_videos"):
                results = self._ydl_searcher.search(source, kw, limit, self.get_full_info)
            else:
                results = self._site_scraper.search(source, kw, limit)
            
            # 캐시에 저장
            if self._cache and results:
                cache_data = [asdict(cr) for cr in results]
                self._cache.save_search_results(kw, cache_data, source)
                logger.debug(f"💾 캐시 저장: [{source}] '{kw}' ({len(results)}개)")
            
            return results

        # 병렬 실행
        with ThreadPoolExecutor(max_workers=max(1, ydl_workers + scrape_workers)) as executor:
            future_map = {
                executor.submit(_execute_task, t): t for t in tasks
            }

            for future in as_completed(future_map):
                task_info = future_map[future]
                source, kw, _ = task_info
                completed_tasks += 1

                try:
                    results = future.result()
                    stats.total_searched += 1

                    for cr in results:
                        # 중복 체크
                        with self._lock:
                            if cr.video_id in self._seen_ids:
                                stats.total_duplicates += 1
                                continue
                            self._seen_ids.add(cr.video_id)

                        # 콘텐츠 필터
                        if self.content_filter:
                            if not _passes_content_filter(cr.title, cr.description):
                                stats.total_filtered += 1
                                continue

                        # 길이 필터
                        if not _passes_duration_filter(
                            cr.duration_sec,
                            self.min_duration_sec,
                            self.max_duration_sec,
                        ):
                            stats.total_filtered += 1
                            continue

                        all_results.append(cr)
                        stats.total_found += 1

                        # 소스별 카운트
                        stats.by_source[source] = stats.by_source.get(source, 0) + 1
                        stats.by_keyword[kw] = stats.by_keyword.get(kw, 0) + 1

                        # 목표 도달 시 중단
                        if stats.total_found >= self.max_results:
                            break

                except Exception as e:
                    stats.total_errors += 1
                    logger.error(f"크롤링 실패 [{source}] '{kw}': {e}")

                if progress_callback:
                    progress_callback(completed_tasks, total_tasks)

                # 목표 도달 시
                if stats.total_found >= self.max_results:
                    break

        stats.elapsed_sec = time.time() - start_time
        logger.info(stats.summary())

        return all_results, stats

    def save_csv(
        self,
        results: List[CrawlResult],
        output_path: Path,
        overwrite: bool = True,
    ):
        """결과를 CSV로 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"

        fieldnames = [
            "url", "video_id", "title", "platform", "keyword",
            "duration_sec", "view_count", "channel_name", "discovered_at",
        ]

        with output_path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for cr in results:
                writer.writerow({fn: getattr(cr, fn, "") for fn in fieldnames})

        logger.info(f"💾 {len(results)}개 결과 저장: {output_path}")

    def save_to_db(self, results: List[CrawlResult], db_path: str = "data/pade.db"):
        """결과를 DB에 저장"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from models.database import Base, Video, ProcessingJob

            db_file = Path(db_path)
            if not db_file.is_absolute():
                project_root = Path(__file__).resolve().parent.parent
                db_file = project_root / db_file

            engine = create_engine(f"sqlite:///{db_file}")
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            session = Session()

            saved = 0
            for cr in results:
                if not cr.video_id:
                    continue

                existing = session.query(Video).filter_by(video_id=cr.video_id).first()
                if existing:
                    continue

                video = Video(
                    video_id=cr.video_id,
                    platform=cr.platform or "unknown",
                    url=cr.url,
                    title=cr.title,
                    description=cr.description[:2000] if cr.description else None,
                    duration_sec=cr.duration_sec,
                    channel_id=cr.channel_id,
                    channel_name=cr.channel_name,
                    view_count=cr.view_count,
                    thumbnail_url=cr.thumbnail_url,
                    tags=cr.tags,
                    status="discovered",
                    discovered_at=datetime.now(timezone.utc),
                )
                session.add(video)
                saved += 1

                # ProcessingJob
                processing_version = "mass_collect_v1"
                job_key = ProcessingJob.generate_job_key(
                    video.platform, cr.video_id, processing_version
                )
                job = ProcessingJob(
                    job_key=job_key,
                    platform=video.platform,
                    video_id=cr.video_id,
                    processing_version=processing_version,
                    stage="discover",
                    status="completed",
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                session.add(job)

            session.commit()
            session.close()
            logger.info(f"💾 DB에 {saved}개 신규 비디오 저장")
            return saved

        except Exception as e:
            logger.error(f"DB 저장 실패: {e}")
            return 0

    def _crawl_async(
        self,
        keywords: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple:
        """
        비동기 크롤러를 사용한 크롤링
        
        AsyncMultiSourceCrawler에 위임하여 asyncio+aiohttp 기반
        비동기 크롤링을 수행합니다.
        """
        import asyncio
        
        try:
            from ingestion.async_crawler import AsyncMultiSourceCrawler, AsyncCrawlConfig
        except ImportError:
            logger.error("비동기 크롤러 임포트 실패, 동기 모드로 전환")
            self.async_mode = False
            return self.crawl(keywords, progress_callback)
        
        start_time = time.time()
        
        config = AsyncCrawlConfig(
            max_concurrent=self.max_workers * 25,
            timeout_sec=30.0,
            retry_count=3,
            retry_delay_sec=2.0,
        )
        
        async_crawler = AsyncMultiSourceCrawler(config=config)
        
        # asyncio 이벤트 루프 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    async_results = pool.submit(
                        asyncio.run,
                        async_crawler.crawl_all_sources(
                            keywords=keywords,
                            sources=self.sources,
                            progress_callback=progress_callback,
                        )
                    ).result()
            else:
                async_results = loop.run_until_complete(
                    async_crawler.crawl_all_sources(
                        keywords=keywords,
                        sources=self.sources,
                        progress_callback=progress_callback,
                    )
                )
        except RuntimeError:
            async_results = asyncio.run(
                async_crawler.crawl_all_sources(
                    keywords=keywords,
                    sources=self.sources,
                    progress_callback=progress_callback,
                )
            )
        
        # AsyncCrawlResult → CrawlResult 변환
        stats = CrawlStats()
        all_results: List[CrawlResult] = []
        
        for async_result in async_results:
            stats.total_searched += 1
            
            if async_result.error:
                stats.total_errors += 1
                continue
            
            for item in async_result.results:
                video_id = item.get("video_id", "")
                
                # 중복 체크
                if video_id in self._seen_ids:
                    stats.total_duplicates += 1
                    continue
                self._seen_ids.add(video_id)
                
                title = item.get("title", "")
                description = item.get("description", "")
                duration_sec = item.get("duration_sec")
                
                # 콘텐츠 필터
                if self.content_filter and not _passes_content_filter(title, description):
                    stats.total_filtered += 1
                    continue
                
                # 길이 필터
                if not _passes_duration_filter(duration_sec, self.min_duration_sec, self.max_duration_sec):
                    stats.total_filtered += 1
                    continue
                
                cr = CrawlResult(
                    video_id=video_id,
                    url=item.get("url", ""),
                    title=title,
                    description=description,
                    duration_sec=duration_sec,
                    view_count=item.get("view_count"),
                    channel_id=item.get("channel_id", ""),
                    channel_name=item.get("channel_name", ""),
                    thumbnail_url=item.get("thumbnail_url", ""),
                    tags=item.get("tags", []),
                    platform=async_result.source,
                    keyword=async_result.keyword,
                    discovered_at=datetime.now(timezone.utc).isoformat(),
                )
                all_results.append(cr)
                stats.total_found += 1
                stats.by_source[async_result.source] = stats.by_source.get(async_result.source, 0) + 1
                stats.by_keyword[async_result.keyword] = stats.by_keyword.get(async_result.keyword, 0) + 1
                
                if stats.total_found >= self.max_results:
                    break
            
            if stats.total_found >= self.max_results:
                break
        
        stats.elapsed_sec = time.time() - start_time
        logger.info(f"⚡ [async] {stats.summary()}")
        
        return all_results, stats


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="다중 소스 병렬 크롤러")
    parser.add_argument(
        "--keywords", required=True,
        help="검색 키워드 (콤마 구분)"
    )
    parser.add_argument(
        "--sources", default="youtube,google_videos",
        help="소스 목록 (콤마 구분: youtube,google_videos,vimeo,dailymotion)"
    )
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="data/urls_mass.csv")
    parser.add_argument("--db", default="data/pade.db")
    parser.add_argument("--min-duration", type=int, default=30)
    parser.add_argument("--max-duration", type=int, default=1200)
    parser.add_argument("--full-info", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument(
        "--async", dest="async_mode", action="store_true",
        help="비동기 크롤링 모드 사용 (aiohttp 기반, 더 빠름)"
    )

    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    crawler = MultiSourceCrawler(
        sources=sources,
        max_results=args.max_results,
        max_workers=args.workers,
        get_full_info=args.full_info,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        content_filter=not args.no_filter,
        async_mode=args.async_mode,
    )

    results, stats = crawler.crawl(keywords)

    if results:
        crawler.save_csv(results, Path(args.output))
        crawler.save_to_db(results, args.db)

    print(stats.summary())


if __name__ == "__main__":
    main()
