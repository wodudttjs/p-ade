"""
비디오 다운로드 관리자

yt-dlp를 사용한 비디오 다운로드 기능을 제공합니다.
"""

import yt_dlp
from pathlib import Path
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass
from enum import Enum
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
        VideoQuality.LOW: "bestvideo[height<=360]+bestaudio/best[height<=360]",
        VideoQuality.MEDIUM: "bestvideo[height<=720]+bestaudio/best[height<=720]",
        VideoQuality.HIGH: "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        VideoQuality.ULTRA: "bestvideo[height<=1440]+bestaudio/best[height<=1440]",
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
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
            'retries': max_retries,
            'fragment_retries': max_retries,
            'skip_unavailable_fragments': False,
            'keepvideo': False,
            'nocheckcertificate': True,
            'prefer_ffmpeg': True,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
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
        format_code: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
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
