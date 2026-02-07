"""
비디오 다운로드 관리자

yt-dlp를 사용한 비디오 다운로드 기능을 제공합니다.
"""

import yt_dlp
import argparse
import csv
import hashlib
from pathlib import Path
from typing import Optional, Dict, Callable, List, Iterable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

from core.logging_config import logger


class VideoQuality(Enum):
    """비디오 품질 설정"""
    LOW = "360p"
    MEDIUM = "720p"
    HIGH = "1080p"
    ULTRA = "1440p"


@dataclass
class DownloadResult:
    """다운로드 결과"""
    success: bool
    filepath: Optional[str] = None
    video_id: Optional[str] = None
    filesize_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    format_id: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    error_message: Optional[str] = None


class VideoDownloader:
    """비디오 다운로드 클래스"""
    
    # 품질별 포맷 코드
    QUALITY_FORMATS = {
        VideoQuality.LOW: "best[height<=360][ext=mp4]/best[height<=360]",
        VideoQuality.MEDIUM: "best[height<=720][ext=mp4]/best[height<=720]",
        VideoQuality.HIGH: "best[height<=1080][ext=mp4]/best[height<=1080]",
        VideoQuality.ULTRA: "best[height<=1440][ext=mp4]/best[height<=1440]",
    }
    
    def __init__(
        self,
        output_dir: Path,
        preferred_quality: VideoQuality = VideoQuality.HIGH,
        max_retries: int = 3,
    ):
        """
        Args:
            output_dir: 다운로드 디렉토리
            preferred_quality: 선호 품질
            max_retries: 최대 재시도 횟수
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preferred_quality = preferred_quality
        self.max_retries = max_retries
        
        # 기본 yt-dlp 옵션
        self.base_opts = {
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'format': self.QUALITY_FORMATS[preferred_quality],
            'quiet': False,
            'no_warnings': False,
            'retries': max_retries,
            'fragment_retries': max_retries,
            'skip_unavailable_fragments': False,
            'keepvideo': False,
            'nocheckcertificate': True,
            'prefer_ffmpeg': True,
        }
    
    def download(
        self,
        url: str,
        progress_callback: Optional[Callable] = None
    ) -> DownloadResult:
        """
        비디오 다운로드
        
        Args:
            url: 비디오 URL
            progress_callback: 진행률 콜백 함수
            
        Returns:
            DownloadResult 객체
        """
        opts = self.base_opts.copy()
        
        if progress_callback:
            opts['progress_hooks'] = [progress_callback]
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                logger.info(f"Downloading video: {url}")
                info = ydl.extract_info(url, download=True)
                
                video_id = info['id']
                ext = info.get('ext', 'mp4')
                filepath = self.output_dir / f"{video_id}.{ext}"
                
                result = DownloadResult(
                    success=True,
                    filepath=str(filepath),
                    video_id=video_id,
                    filesize_bytes=info.get('filesize') or info.get('filesize_approx'),
                    duration_sec=info.get('duration'),
                    format_id=info.get('format_id'),
                    resolution=info.get('resolution'),
                    fps=info.get('fps'),
                )
                
                logger.info(f"Downloaded successfully: {filepath}")
                return result
                
        except Exception as e:
            logger.error(f"Download failed: {url} - {str(e)}")
            return DownloadResult(
                success=False,
                error_message=str(e)
            )
    
    def download_with_format(
        self,
        url: str,
        format_code: str = "best[ext=mp4]/best"
    ) -> DownloadResult:
        """
        커스텀 포맷으로 다운로드
        
        Args:
            url: 비디오 URL
            format_code: yt-dlp 포맷 코드
            
        Returns:
            DownloadResult 객체
        """
        opts = self.base_opts.copy()
        opts['format'] = format_code
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                video_id = info['id']
                ext = info.get('ext', 'mp4')
                filepath = self.output_dir / f"{video_id}.{ext}"
                
                return DownloadResult(
                    success=True,
                    filepath=str(filepath),
                    video_id=video_id,
                    filesize_bytes=info.get('filesize'),
                    duration_sec=info.get('duration'),
                )
        except Exception as e:
            return DownloadResult(
                success=False,
                error_message=str(e)
            )
    
    def get_video_info(self, url: str) -> Optional[Dict]:
        """
        메타데이터만 추출 (다운로드 X)
        
        Args:
            url: 비디오 URL
            
        Returns:
            비디오 정보 딕셔너리
        """
        opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            logger.error(f"Failed to extract info: {e}")
            return None
    
    def list_formats(self, url: str) -> List[Dict]:
        """
        사용 가능한 포맷 목록 조회
        
        Args:
            url: 비디오 URL
            
        Returns:
            포맷 정보 리스트
        """
        info = self.get_video_info(url)
        if not info:
            return []
        
        formats = info.get('formats', [])
        
        # 정리된 형태로 변환
        format_list = []
        for fmt in formats:
            format_list.append({
                'format_id': fmt.get('format_id'),
                'ext': fmt.get('ext'),
                'resolution': fmt.get('resolution'),
                'fps': fmt.get('fps'),
                'vcodec': fmt.get('vcodec'),
                'acodec': fmt.get('acodec'),
                'filesize': fmt.get('filesize'),
                'filesize_approx': fmt.get('filesize_approx'),
            })
        
        return format_list


REQUIRED_KEYWORDS = [
    "robot arm", "robotic arm", "robot gripper",
    "pick and place", "pick & place", "grasping",
    "manipulation", "object manipulation",
    "assembly", "bin picking",
]

REJECT_KEYWORDS = [
    "animation", "simulation", "cgi", "3d render",
    "toy", "lego", "surgery", "medical",
]

MIN_DURATION_SEC = 60
MAX_DURATION_SEC = 300
MIN_HEIGHT = 480
MIN_FPS = 24
MIN_BITRATE_KBPS = 2000


def _detect_url_field(fieldnames: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    lowered = [f.lower() for f in fieldnames]
    for candidate in ("url", "video_url", "video", "link"):
        if candidate in lowered:
            return fieldnames[lowered.index(candidate)]
    return fieldnames[0]


def _iter_urls_from_csv(csv_path: Path) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        url_field = _detect_url_field(reader.fieldnames or [])
        if url_field:
            for row in reader:
                url = (row.get(url_field) or "").strip()
                if url:
                    yield url
            return

    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            url = line.strip().split(",")[0]
            if url:
                yield url


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _matches_keywords(text: str) -> bool:
    lowered = text.lower()
    if any(bad in lowered for bad in REJECT_KEYWORDS):
        return False
    return any(key in lowered for key in REQUIRED_KEYWORDS)


def _passes_constraints(info: Dict) -> bool:
    duration = info.get("duration") or 0
    if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
        return False

    formats = info.get("formats") or []
    video_formats = [
        fmt for fmt in formats
        if (fmt.get("vcodec") not in (None, "none"))
    ]

    info_height = info.get("height") or 0
    max_height = max((fmt.get("height") or 0) for fmt in video_formats) if video_formats else 0
    height = max(info_height, max_height)
    if height < MIN_HEIGHT:
        return False

    info_fps = info.get("fps") or 0
    max_fps = max((fmt.get("fps") or 0) for fmt in video_formats) if video_formats else 0
    fps = max(info_fps, max_fps)
    if fps < MIN_FPS:
        return False

    info_tbr = info.get("tbr") or 0
    max_tbr = max((fmt.get("tbr") or 0) for fmt in video_formats) if video_formats else 0
    tbr = max(info_tbr, max_tbr)
    if tbr and tbr < MIN_BITRATE_KBPS:
        return False

    title = info.get("title") or ""
    desc = info.get("description") or ""
    if not _matches_keywords(f"{title} {desc}"):
        return False

    return True


def _get_db_session(db_path: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models.database import Base

    db_path_obj = Path(db_path)
    if not db_path_obj.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        db_path_obj = project_root / db_path_obj

    engine = create_engine(f"sqlite:///{db_path_obj}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _infer_platform(url: str) -> str:
    lowered = url.lower()
    if "youtube" in lowered or "youtu.be" in lowered:
        return "youtube"
    return "unknown"


def _extract_video_id_from_url(url: str) -> Optional[str]:
    lowered = url.lower()
    if "youtu.be" in lowered:
        return url.split("youtu.be/")[-1].split("?")[0]
    if "youtube" in lowered and "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    return None


def _record_job(session, platform: str, video_id: str, status: str, stage: str, error: Optional[str] = None):
    from models.database import ProcessingJob

    processing_version = "local"
    job_key = ProcessingJob.generate_job_key(platform, video_id, processing_version)

    job = session.query(ProcessingJob).filter_by(job_key=job_key).first()
    now = datetime.utcnow()

    if not job:
        job = ProcessingJob(
            job_key=job_key,
            platform=platform,
            video_id=video_id,
            processing_version=processing_version,
            stage=stage,
            status=status,
            started_at=now if status == "running" else None,
            completed_at=now if status in ("completed", "failed", "skipped") else None,
            failure_reason=error,
        )
        session.add(job)
    else:
        job.stage = stage
        job.status = status
        if status == "running" and not job.started_at:
            job.started_at = now
        if status in ("completed", "failed", "skipped"):
            job.completed_at = now
        if error:
            job.failure_reason = error


def _update_video_after_download(session, url: str, result: DownloadResult, info: Dict):
    from models.database import Video

    platform = _infer_platform(url)
    video_id = result.video_id
    if not video_id:
        return

    video = session.query(Video).filter_by(video_id=video_id).first()

    if not video:
        video = Video(
            video_id=video_id,
            platform=platform,
            url=url,
            status="downloaded" if result.success else "failed",
        )
        session.add(video)

    if result.success:
        video.downloaded_at = datetime.utcnow()
        video.local_path = result.filepath
        if result.duration_sec:
            video.duration_sec = result.duration_sec
        video.status = "downloaded"
        video.download_tool_version = getattr(yt_dlp.version, "__version__", None)
        video.download_format_id = result.format_id
        video.view_count = info.get("view_count")
        video.thumbnail_url = info.get("thumbnail")
        video.title = info.get("title")
        video.description = info.get("description")
        video.tags = info.get("tags")
        video.channel_id = info.get("channel_id")
        video.channel_name = info.get("channel")
    else:
        video.status = "failed"


def run_cli():
    parser = argparse.ArgumentParser(description="비디오 다운로드 (CSV 입력)")
    parser.add_argument("--input", required=True, help="CSV 파일 경로")
    parser.add_argument("--output", default="data/raw", help="다운로드 출력 디렉토리")
    parser.add_argument("--quality", default="1080p", choices=["360p", "720p", "1080p", "1440p"], help="다운로드 품질")
    parser.add_argument("--db", default="data/pade.db", help="SQLite DB 경로")
    parser.add_argument("--max-downloads", type=int, default=10, help="다운로드 최대 개수")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"❌ CSV 파일 없음: {args.input}")
        return 1

    output_dir = Path(args.output)
    quality_map = {
        "360p": VideoQuality.LOW,
        "720p": VideoQuality.MEDIUM,
        "1080p": VideoQuality.HIGH,
        "1440p": VideoQuality.ULTRA,
    }
    downloader = VideoDownloader(output_dir=output_dir, preferred_quality=quality_map[args.quality])

    session = _get_db_session(args.db)
    total = 0
    success = 0
    seen = set()

    try:
        for url in _iter_urls_from_csv(csv_path):
            if success >= args.max_downloads:
                break

            url_hash = _hash_url(url)
            if url_hash in seen:
                continue
            seen.add(url_hash)

            info = downloader.get_video_info(url)
            if not info:
                continue

            if not _passes_constraints(info):
                continue

            total += 1
            platform = _infer_platform(url)
            inferred_video_id = _extract_video_id_from_url(url) or f"unknown_{abs(hash(url))}"

            _record_job(session, platform, inferred_video_id, "running", "download")

            result = downloader.download(url)
            if result.success:
                success += 1
                _update_video_after_download(session, url, result, info)
                _record_job(session, platform, result.video_id, "completed", "download")
                logger.info(f"✅ 다운로드 완료: {result.video_id}")
            else:
                _record_job(session, platform, inferred_video_id, "failed", "download", result.error_message)
                logger.error(f"❌ 다운로드 실패: {url} - {result.error_message}")

            session.commit()
    finally:
        session.close()

    print(f"✅ 완료: {success}/{total} 성공")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
    
    def check_available(self, url: str) -> bool:
        """
        비디오 다운로드 가능 여부 확인
        
        Args:
            url: 비디오 URL
            
        Returns:
            다운로드 가능 여부
        """
        info = self.get_video_info(url)
        return info is not None
    
    def estimate_size(self, url: str) -> Optional[int]:
        """
        다운로드 예상 크기 (bytes)
        
        Args:
            url: 비디오 URL
            
        Returns:
            예상 크기 (bytes)
        """
        info = self.get_video_info(url)
        if not info:
            return None
        
        # 선호 포맷의 크기 찾기
        formats = info.get('formats', [])
        format_code = self.QUALITY_FORMATS[self.preferred_quality]
        
        # 가장 근접한 포맷의 크기 반환
        for fmt in formats:
            if fmt.get('filesize'):
                return fmt['filesize']
            elif fmt.get('filesize_approx'):
                return fmt['filesize_approx']
        
        return None


def progress_hook(d: Dict):
    """
    기본 진행률 콜백
    
    Args:
        d: yt-dlp progress 딕셔너리
    """
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        downloaded = d.get('downloaded_bytes', 0)
        
        if total > 0:
            percentage = (downloaded / total) * 100
            speed = d.get('speed', 0)
            eta = d.get('eta', 0)
            
            speed_mb = speed / 1024 / 1024 if speed else 0
            
            logger.debug(
                f"Download progress: {percentage:.1f}% "
                f"({downloaded}/{total} bytes) "
                f"Speed: {speed_mb:.2f} MB/s "
                f"ETA: {eta}s"
            )
    
    elif d['status'] == 'finished':
        logger.info(f"Download finished: {d.get('filename')}")
