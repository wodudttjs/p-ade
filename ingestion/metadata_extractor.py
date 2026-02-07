"""
비디오 메타데이터 추출기

ffprobe를 사용한 비디오 메타데이터 추출
"""

import ffmpeg
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from core.logging_config import logger


@dataclass
class VideoMetadata:
    """비디오 메타데이터"""
    filepath: str
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    codec_name: Optional[str] = None
    codec_long_name: Optional[str] = None
    format_name: Optional[str] = None
    format_long_name: Optional[str] = None
    size_bytes: Optional[int] = None
    num_streams: Optional[int] = None
    has_video: bool = False
    has_audio: bool = False
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_json(self, output_path: Path):
        """JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        logger.info(f"Metadata saved to: {output_path}")


class MetadataExtractor:
    """메타데이터 추출 클래스"""
    
    @staticmethod
    def extract(video_path: Path) -> Optional[VideoMetadata]:
        """
        비디오 메타데이터 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            VideoMetadata 객체 또는 None
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            # ffprobe로 메타데이터 추출
            probe = ffmpeg.probe(str(video_path))
            
            # 기본 정보
            format_info = probe.get('format', {})
            
            metadata = VideoMetadata(
                filepath=str(video_path),
                duration_sec=float(format_info.get('duration', 0)) or None,
                format_name=format_info.get('format_name'),
                format_long_name=format_info.get('format_long_name'),
                size_bytes=int(format_info.get('size', 0)) or None,
                bitrate=int(format_info.get('bit_rate', 0)) or None,
                num_streams=len(probe.get('streams', [])),
            )
            
            # 스트림별 정보 추출
            for stream in probe.get('streams', []):
                codec_type = stream.get('codec_type')
                
                if codec_type == 'video' and not metadata.has_video:
                    metadata.has_video = True
                    metadata.width = stream.get('width')
                    metadata.height = stream.get('height')
                    metadata.codec_name = stream.get('codec_name')
                    metadata.codec_long_name = stream.get('codec_long_name')
                    
                    # FPS 계산
                    fps_str = stream.get('r_frame_rate', '0/1')
                    try:
                        num, den = map(int, fps_str.split('/'))
                        if den > 0:
                            metadata.fps = num / den
                    except:
                        pass
                
                elif codec_type == 'audio' and not metadata.has_audio:
                    metadata.has_audio = True
                    metadata.audio_codec = stream.get('codec_name')
                    metadata.audio_sample_rate = stream.get('sample_rate')
                    metadata.audio_channels = stream.get('channels')
            
            logger.info(f"Extracted metadata from: {video_path}")
            return metadata
            
        except ffmpeg.Error as e:
            logger.error(f"FFprobe error: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return None
    
    @staticmethod
    def extract_to_json(
        video_path: Path,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        메타데이터 추출 후 JSON 파일로 저장
        
        Args:
            video_path: 비디오 파일 경로
            output_path: JSON 출력 경로 (None이면 자동 생성)
            
        Returns:
            저장된 JSON 파일 경로 또는 None
        """
        metadata = MetadataExtractor.extract(video_path)
        
        if metadata is None:
            return None
        
        # 출력 경로 자동 생성
        if output_path is None:
            output_path = video_path.with_suffix('.metadata.json')
        
        metadata.save_json(output_path)
        return output_path
    
    @staticmethod
    def batch_extract(
        video_dir: Path,
        pattern: str = "*.mp4",
        output_dir: Optional[Path] = None
    ) -> Dict[str, Optional[VideoMetadata]]:
        """
        디렉토리 내 여러 비디오 메타데이터 일괄 추출
        
        Args:
            video_dir: 비디오 디렉토리
            pattern: 파일 패턴
            output_dir: JSON 출력 디렉토리
            
        Returns:
            {video_path: metadata} 딕셔너리
        """
        if not video_dir.exists():
            logger.error(f"Directory not found: {video_dir}")
            return {}
        
        results = {}
        video_files = list(video_dir.glob(pattern))
        
        logger.info(f"Found {len(video_files)} video files")
        
        for video_path in video_files:
            metadata = MetadataExtractor.extract(video_path)
            results[str(video_path)] = metadata
            
            # JSON 저장
            if metadata and output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                json_path = output_dir / f"{video_path.stem}.metadata.json"
                metadata.save_json(json_path)
        
        return results
    
    @staticmethod
    def get_video_info_summary(video_path: Path) -> Optional[str]:
        """
        비디오 정보 요약 텍스트
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            요약 텍스트 또는 None
        """
        metadata = MetadataExtractor.extract(video_path)
        
        if metadata is None:
            return None
        
        summary_parts = [
            f"File: {Path(metadata.filepath).name}",
        ]
        
        if metadata.duration_sec:
            minutes = int(metadata.duration_sec // 60)
            seconds = int(metadata.duration_sec % 60)
            summary_parts.append(f"Duration: {minutes}m {seconds}s")
        
        if metadata.width and metadata.height:
            summary_parts.append(f"Resolution: {metadata.width}x{metadata.height}")
        
        if metadata.fps:
            summary_parts.append(f"FPS: {metadata.fps:.2f}")
        
        if metadata.codec_name:
            summary_parts.append(f"Video Codec: {metadata.codec_name}")
        
        if metadata.audio_codec:
            summary_parts.append(f"Audio Codec: {metadata.audio_codec}")
        
        if metadata.size_bytes:
            size_mb = metadata.size_bytes / 1024 / 1024
            summary_parts.append(f"Size: {size_mb:.2f} MB")
        
        return " | ".join(summary_parts)
    
    @staticmethod
    def validate_video(video_path: Path) -> bool:
        """
        비디오 파일 유효성 검증
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            유효 여부
        """
        metadata = MetadataExtractor.extract(video_path)
        
        if metadata is None:
            return False
        
        # 기본 검증 조건
        if not metadata.has_video:
            logger.warning(f"No video stream found: {video_path}")
            return False
        
        if metadata.duration_sec is None or metadata.duration_sec <= 0:
            logger.warning(f"Invalid duration: {video_path}")
            return False
        
        if metadata.width is None or metadata.height is None:
            logger.warning(f"Invalid resolution: {video_path}")
            return False
        
        return True
    
    @staticmethod
    def get_resolution_string(metadata: VideoMetadata) -> Optional[str]:
        """
        해상도 문자열 생성
        
        Args:
            metadata: 비디오 메타데이터
            
        Returns:
            해상도 문자열 (예: "1920x1080", "720p")
        """
        if metadata.width is None or metadata.height is None:
            return None
        
        # 표준 해상도 매핑
        height = metadata.height
        if height == 2160:
            return "4K"
        elif height == 1440:
            return "1440p"
        elif height == 1080:
            return "1080p"
        elif height == 720:
            return "720p"
        elif height == 480:
            return "480p"
        elif height == 360:
            return "360p"
        else:
            return f"{metadata.width}x{metadata.height}"
