"""
YT-DLP 기반 다중 소스 비디오 검색 스크립트

- yt-dlp의 검색(extractor) 기능을 활용해 YouTube 및 Google Videos에서 키워드 기반으로 비디오 메타데이터를 추출합니다.
- 추출 결과를 CSV(`data/urls.csv` 기본)로 저장하고, 기존 프로젝트 DB(ORM)를 사용해 Video/ProcessingJob을 업데이트합니다.
- 지원 소스: youtube, google_videos

사용 예:
python -m spiders.yt_dlp_search_spider --keywords "robot arm" --max-results 200 --sources youtube,google_videos --output data/urls.csv --db data/pade.db --overwrite
"""

import argparse
import csv
from pathlib import Path
from typing import List
import yt_dlp
from datetime import datetime
from loguru import logger
import requests
import re
from urllib.parse import quote_plus


EXTRACTOR_MAP = {
    "youtube": "ytsearch",
    "google_videos": "gvsearch",
    "vimeo": None,       # handled via site scraping
    "dailymotion": None,  # handled via site scraping
    "bilibili": None,     # handled via site scraping
    "rutube": None,       # handled via site scraping
}


def _parse_keywords(raw: str) -> List[str]:
    if not raw:
        return []
    if "," in raw:
        return [k.strip() for k in raw.split(",") if k.strip()]
    return [raw.strip()]


def save_to_csv(path: Path, rows: List[dict], overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["url", "video_id", "title"])
        if mode == "w":
            writer.writeheader()
        for r in rows:
            writer.writerow({"url": r.get("url", ""), "video_id": r.get("video_id", ""), "title": r.get("title", "")})


def upsert_db(db_path: str, rows: List[dict]):
    """간단한 DB 업데이트: models.database.Video 및 ProcessingJob 사용"""
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

    for item in rows:
        video_id = item.get("video_id")
        if not video_id:
            continue
        existing = session.query(Video).filter_by(video_id=video_id).first()
        discovered_at = None
        discovered_at_raw = item.get("discovered_at")
        if discovered_at_raw:
            try:
                discovered_at = datetime.fromisoformat(discovered_at_raw)
            except Exception:
                discovered_at = datetime.utcnow()
        else:
            discovered_at = datetime.utcnow()

        if existing:
            # 최소한의 필드 업데이트
            if not existing.title and item.get("title"):
                existing.title = item.get("title")
            if not existing.thumbnail_url and item.get("thumbnail_url"):
                existing.thumbnail_url = item.get("thumbnail_url")
            # ProcessingJob 업데이트
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
            continue

        video = Video(
            video_id=video_id,
            platform=item.get("platform") or "yt_dlp",
            url=item.get("url", ""),
            title=item.get("title"),
            description=item.get("description"),
            duration_sec=item.get("duration_sec"),
            channel_id=item.get("channel_id"),
            channel_name=item.get("channel_name"),
            view_count=item.get("view_count"),
            thumbnail_url=item.get("thumbnail_url"),
            tags=item.get("tags", []),
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

    session.close()


def _scrape_site_search(source: str, keyword: str, limit: int):
    """간단한 검색 결과 스크래핑으로 비디오 URL 수집
    source: 'vimeo', 'dailymotion', 'bilibili', 'rutube'
    """
    urls = []
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        
        if source == 'vimeo':
            search_url = f"https://vimeo.com/search?q={quote_plus(keyword)}"
            pattern = r'https?://vimeo\.com/\d+'
        elif source == 'dailymotion':
            search_url = f"https://www.dailymotion.com/search/{quote_plus(keyword)}/videos"
            pattern = r'https?://www\.dailymotion\.com/video/[a-zA-Z0-9]+'
        elif source == 'bilibili':
            search_url = f"https://search.bilibili.com/all?keyword={quote_plus(keyword)}"
            pattern = r'https?://www\.bilibili\.com/video/[A-Za-z0-9]+'
        elif source == 'rutube':
            search_url = f"https://rutube.ru/api/search/video/?query={quote_plus(keyword)}"
            pattern = r'https?://rutube\.ru/video/[a-f0-9]+'
        else:
            return urls

        resp = requests.get(search_url, headers=headers, timeout=15)
        html = resp.text

        found = re.findall(pattern, html)

        # dedupe and limit
        seen = set()
        for u in found:
            u = u.rstrip("/")
            if u not in seen:
                seen.add(u)
                urls.append(u)
            if len(urls) >= limit:
                break
    except Exception as exc:
        logger.warning(f"Scraping failed for {source}: {exc}")
        return urls

    return urls


def run_search(keywords: List[str], max_results: int, sources: List[str]):
    all_rows = []
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}

    remaining = max_results
    per_source = max(1, int(max_results / max(1, len(sources))))

    for source in sources:
        extractor = EXTRACTOR_MAP.get(source.lower())
        if extractor is None and source.lower() not in ("vimeo", "dailymotion", "bilibili", "rutube"):
            logger.warning(f"Unsupported source: {source}. Skipping.")
            continue

        per = min(per_source, remaining)
        if per <= 0:
            break

        for kw in keywords:
            if remaining <= 0:
                break

            # site-scrape based sources
            if source.lower() in ("vimeo", "dailymotion", "bilibili", "rutube"):
                urls = _scrape_site_search(source.lower(), kw, per)
                logger.info(f"Scraped {len(urls)} URLs from {source} for keyword '{kw}'")
                for u in urls:
                    if remaining <= 0:
                        break
                    try:
                        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl2:
                            e = ydl2.extract_info(u, download=False)
                    except Exception as exc:
                        logger.error(f"Failed to extract info for {u}: {exc}")
                        continue

                    duration = e.get("duration") if isinstance(e, dict) else None
                    if duration is not None and (duration < 30 or duration > 1200):
                        continue

                    row = {
                        "video_id": e.get("id"),
                        "url": e.get("webpage_url") or u,
                        "title": e.get("title"),
                        "description": e.get("description"),
                        "duration_sec": duration,
                        "view_count": e.get("view_count"),
                        "channel_id": e.get("channel_id"),
                        "channel_name": e.get("uploader") or e.get("uploader_id"),
                        "thumbnail_url": e.get("thumbnail"),
                        "tags": e.get("tags", []),
                        "platform": source.lower(),
                        "discovered_at": datetime.utcnow().isoformat(),
                    }
                    all_rows.append(row)
                    remaining -= 1
                continue

            # 기존 yt-dlp 검색(extractor) 사용
            query = f"{extractor}{per}:{kw}"
            logger.info(f"Running search: {query}")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(query, download=False)
                if not result:
                    continue
                entries = result.get("entries", []) if isinstance(result, dict) else []
                for e in entries:
                    if not e:
                        continue
                    # 필터: 길이 (초 단위) 60~300 초 사이만 허용
                    duration = e.get("duration")
                    if duration is None:
                        # extract_flat 모드일 때 상세 정보가 없을 수 있으므로 실제 메타데이터 요청
                        try:
                            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl2:
                                full = ydl2.extract_info(e.get("url") or e.get("webpage_url"), download=False)
                                duration = full.get("duration")
                        except Exception:
                            duration = None

                    if duration is not None and (duration < 30 or duration > 1200):
                        # 길이 조건 불충족 시 건너뜀
                        continue
                    row = {
                        "video_id": e.get("id"),
                        "url": e.get("url") or e.get("webpage_url"),
                        "title": e.get("title"),
                        "description": e.get("description"),
                        "duration_sec": e.get("duration"),
                        "view_count": e.get("view_count"),
                        "channel_id": e.get("channel_id"),
                        "channel_name": e.get("uploader"),
                        "thumbnail_url": e.get("thumbnail"),
                        "tags": e.get("tags", []),
                        "platform": source.lower(),
                        "discovered_at": datetime.utcnow().isoformat(),
                    }
                    all_rows.append(row)
                    remaining -= 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
            except Exception as exc:
                logger.error(f"Search failed for {query}: {exc}")
        if remaining <= 0:
            break

    return all_rows


def run_cli():
    parser = argparse.ArgumentParser(description="yt-dlp 기반 다중 소스 비디오 검색")
    parser.add_argument("--keywords", required=True, help="검색 키워드 (콤마로 다중 지정)")
    parser.add_argument("--max-results", type=int, default=500, help="최대 결과 수 (기본: 500)")
    parser.add_argument("--sources", default="youtube,google_videos", help="소스 목록 (콤마 구분). 지원: youtube,google_videos,vimeo,dailymotion,bilibili,rutube")
    parser.add_argument("--db", default="data/pade.db", help="SQLite DB 경로")
    parser.add_argument("--output", default="data/urls.csv", help="CSV 출력 경로")
    parser.add_argument("--overwrite", action="store_true", help="CSV 덮어쓰기")

    args = parser.parse_args()
    keywords = _parse_keywords(args.keywords)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    rows = run_search(keywords=keywords, max_results=args.max_results, sources=sources)
    if not rows:
        logger.info("No results found.")
        return 0

    save_to_csv(Path(args.output), rows, overwrite=args.overwrite)
    try:
        upsert_db(args.db, rows)
    except Exception as e:
        logger.error(f"DB upsert failed: {e}")

    logger.info(f"Saved {len(rows)} results to {args.output} and updated DB {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
