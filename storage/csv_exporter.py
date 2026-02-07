"""
CSV Export 기능

비디오 메타데이터를 CSV 파일로 저장합니다.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.database import Video
from core.logging_config import logger


class CSVExporter:
    """CSV 파일 저장 클래스"""
    
    # 기본 CSV 필드
    DEFAULT_FIELDS = [
        'video_id',
        'platform',
        'url',
        'title',
        'description',
        'duration_sec',
        'upload_date',
        'channel_id',
        'channel_name',
        'view_count',
        'like_count',
        'thumbnail_url',
        'tags',
        'discovered_at',
        'downloaded_at',
        'processed_at',
        'status',
        'local_path'
    ]
    
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: CSV 파일을 저장할 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_videos(
        self,
        videos: List[Video],
        filename: Optional[str] = None,
        fields: Optional[List[str]] = None
    ) -> Path:
        """
        비디오 목록을 CSV 파일로 저장
        
        Args:
            videos: Video 객체 리스트
            filename: 저장할 파일명 (기본값: videos_YYYYMMDD_HHMMSS.csv)
            fields: 저장할 필드 리스트 (기본값: DEFAULT_FIELDS)
            
        Returns:
            저장된 CSV 파일 경로
        """
        if not videos:
            logger.warning("No videos to export")
            return None
        
        # 기본 파일명 생성
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"videos_{timestamp}.csv"
        
        # 파일 경로
        filepath = self.output_dir / filename
        
        # 필드 설정
        if not fields:
            fields = self.DEFAULT_FIELDS
        
        # CSV 작성
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            
            for video in videos:
                row = self._video_to_dict(video, fields)
                writer.writerow(row)
        
        logger.info(f"Exported {len(videos)} videos to {filepath}")
        return filepath
    
    def _video_to_dict(self, video: Video, fields: List[str]) -> Dict[str, Any]:
        """
        Video 객체를 딕셔너리로 변환
        
        Args:
            video: Video 객체
            fields: 포함할 필드 리스트
            
        Returns:
            비디오 데이터 딕셔너리
        """
        data = {}
        
        for field in fields:
            value = getattr(video, field, None)
            
            # 날짜 객체를 문자열로 변환
            if isinstance(value, datetime):
                value = value.isoformat()
            
            # 리스트/딕셔너리를 문자열로 변환
            elif isinstance(value, (list, dict)):
                value = str(value)
            
            data[field] = value if value is not None else ''
        
        return data
    
    def export_videos_by_platform(
        self,
        videos: List[Video],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        플랫폼별로 비디오를 분리하여 CSV 파일로 저장
        
        Args:
            videos: Video 객체 리스트
            fields: 저장할 필드 리스트
            
        Returns:
            플랫폼: 파일경로 딕셔너리
        """
        # 플랫폼별로 비디오 그룹화
        platform_videos = {}
        for video in videos:
            platform = video.platform.lower()
            if platform not in platform_videos:
                platform_videos[platform] = []
            platform_videos[platform].append(video)
        
        # 각 플랫폼별 CSV 파일 생성
        result = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for platform, platform_video_list in platform_videos.items():
            filename = f"{platform}_videos_{timestamp}.csv"
            filepath = self.export_videos(
                videos=platform_video_list,
                filename=filename,
                fields=fields
            )
            result[platform] = filepath
        
        logger.info(f"Exported videos for {len(platform_videos)} platforms")
        return result
    
    def export_videos_by_status(
        self,
        videos: List[Video],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        상태별로 비디오를 분리하여 CSV 파일로 저장
        
        Args:
            videos: Video 객체 리스트
            fields: 저장할 필드 리스트
            
        Returns:
            상태: 파일경로 딕셔너리
        """
        # 상태별로 비디오 그룹화
        status_videos = {}
        for video in videos:
            status = video.status or 'unknown'
            if status not in status_videos:
                status_videos[status] = []
            status_videos[status].append(video)
        
        # 각 상태별 CSV 파일 생성
        result = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for status, status_video_list in status_videos.items():
            filename = f"videos_{status}_{timestamp}.csv"
            filepath = self.export_videos(
                videos=status_video_list,
                filename=filename,
                fields=fields
            )
            result[status] = filepath
        
        logger.info(f"Exported videos for {len(status_videos)} statuses")
        return result
    
    def append_video(
        self,
        video: Video,
        filename: str,
        fields: Optional[List[str]] = None
    ) -> bool:
        """
        기존 CSV 파일에 비디오 추가
        
        Args:
            video: Video 객체
            filename: CSV 파일명
            fields: 필드 리스트 (기본값: DEFAULT_FIELDS)
            
        Returns:
            성공 여부
        """
        filepath = self.output_dir / filename
        
        if not fields:
            fields = self.DEFAULT_FIELDS
        
        # 파일이 존재하지 않으면 새로 생성
        if not filepath.exists():
            return self.export_videos([video], filename, fields) is not None
        
        # 기존 파일에 추가
        try:
            with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                row = self._video_to_dict(video, fields)
                writer.writerow(row)
            
            logger.debug(f"Appended video {video.video_id} to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to append video to CSV: {e}")
            return False
    
    def read_csv(
        self,
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        CSV 파일 읽기
        
        Args:
            filename: CSV 파일명
            
        Returns:
            비디오 데이터 딕셔너리 리스트
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            logger.warning(f"CSV file not found: {filepath}")
            return []
        
        videos = []
        try:
            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    videos.append(dict(row))
            
            logger.info(f"Read {len(videos)} videos from {filepath}")
            return videos
        
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return []
    
    def get_csv_files(self) -> List[Path]:
        """
        저장 디렉토리의 모든 CSV 파일 목록
        
        Returns:
            CSV 파일 경로 리스트
        """
        csv_files = list(self.output_dir.glob("*.csv"))
        return sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)
