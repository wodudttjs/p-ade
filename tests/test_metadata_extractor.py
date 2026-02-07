"""
메타데이터 추출기 테스트
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ingestion.metadata_extractor import (
    MetadataExtractor,
    VideoMetadata
)


@pytest.fixture
def sample_metadata():
    """샘플 메타데이터"""
    return VideoMetadata(
        filepath="/path/to/video.mp4",
        duration_sec=120.5,
        width=1920,
        height=1080,
        fps=30.0,
        bitrate=5000000,
        codec_name="h264",
        codec_long_name="H.264 / AVC / MPEG-4 AVC",
        format_name="mp4",
        format_long_name="MP4 (MPEG-4 Part 14)",
        size_bytes=75000000,
        num_streams=2,
        has_video=True,
        has_audio=True,
        audio_codec="aac",
        audio_sample_rate=48000,
        audio_channels=2,
    )


@pytest.fixture
def mock_probe_data():
    """Mock ffprobe 데이터"""
    return {
        'format': {
            'duration': '120.5',
            'format_name': 'mp4',
            'format_long_name': 'MP4 (MPEG-4 Part 14)',
            'size': '75000000',
            'bit_rate': '5000000',
        },
        'streams': [
            {
                'codec_type': 'video',
                'codec_name': 'h264',
                'codec_long_name': 'H.264 / AVC / MPEG-4 AVC',
                'width': 1920,
                'height': 1080,
                'r_frame_rate': '30/1',
            },
            {
                'codec_type': 'audio',
                'codec_name': 'aac',
                'sample_rate': '48000',
                'channels': 2,
            }
        ]
    }


def test_video_metadata_creation():
    """VideoMetadata 생성 테스트"""
    metadata = VideoMetadata(
        filepath="/test/video.mp4",
        duration_sec=60.0,
        width=1280,
        height=720,
        fps=24.0
    )
    
    assert metadata.filepath == "/test/video.mp4"
    assert metadata.duration_sec == 60.0
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.fps == 24.0


def test_video_metadata_to_dict(sample_metadata):
    """딕셔너리 변환"""
    data = sample_metadata.to_dict()
    
    assert isinstance(data, dict)
    assert data['filepath'] == "/path/to/video.mp4"
    assert data['duration_sec'] == 120.5
    assert data['width'] == 1920
    assert data['has_video'] is True


def test_video_metadata_to_json(sample_metadata):
    """JSON 문자열 변환"""
    json_str = sample_metadata.to_json()
    
    assert isinstance(json_str, str)
    
    # JSON 파싱 가능한지 확인
    data = json.loads(json_str)
    assert data['filepath'] == "/path/to/video.mp4"
    assert data['duration_sec'] == 120.5


def test_video_metadata_save_json(sample_metadata, tmp_path):
    """JSON 파일 저장"""
    output_path = tmp_path / "metadata.json"
    sample_metadata.save_json(output_path)
    
    assert output_path.exists()
    
    # 파일 내용 확인
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    assert data['filepath'] == "/path/to/video.mp4"
    assert data['width'] == 1920


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_success(mock_probe, tmp_path, mock_probe_data):
    """메타데이터 추출 성공"""
    video_path = tmp_path / "test.mp4"
    video_path.write_text("fake video")
    
    mock_probe.return_value = mock_probe_data
    
    metadata = MetadataExtractor.extract(video_path)
    
    assert metadata is not None
    assert metadata.duration_sec == 120.5
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == 30.0
    assert metadata.has_video is True
    assert metadata.has_audio is True


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_file_not_found(mock_probe, tmp_path):
    """존재하지 않는 파일"""
    video_path = tmp_path / "nonexistent.mp4"
    
    metadata = MetadataExtractor.extract(video_path)
    
    assert metadata is None
    mock_probe.assert_not_called()


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_ffprobe_error(mock_probe, tmp_path):
    """ffprobe 에러"""
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_text("corrupt")
    
    import ffmpeg
    mock_probe.side_effect = ffmpeg.Error('ffprobe', '', b'Invalid data')
    
    metadata = MetadataExtractor.extract(video_path)
    
    assert metadata is None


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_video_only(mock_probe, tmp_path):
    """비디오만 있는 파일"""
    video_path = tmp_path / "video_only.mp4"
    video_path.write_text("fake")
    
    mock_probe.return_value = {
        'format': {'duration': '60.0', 'size': '10000000'},
        'streams': [
            {
                'codec_type': 'video',
                'codec_name': 'h264',
                'width': 1280,
                'height': 720,
                'r_frame_rate': '24/1',
            }
        ]
    }
    
    metadata = MetadataExtractor.extract(video_path)
    
    assert metadata.has_video is True
    assert metadata.has_audio is False
    assert metadata.audio_codec is None


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_fps_calculation(mock_probe, tmp_path):
    """FPS 계산"""
    video_path = tmp_path / "test.mp4"
    video_path.write_text("fake")
    
    # 복잡한 FPS (29.97)
    mock_probe.return_value = {
        'format': {'duration': '60.0'},
        'streams': [
            {
                'codec_type': 'video',
                'codec_name': 'h264',
                'width': 1920,
                'height': 1080,
                'r_frame_rate': '30000/1001',
            }
        ]
    }
    
    metadata = MetadataExtractor.extract(video_path)
    
    assert metadata.fps is not None
    assert 29.9 < metadata.fps < 30.0


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_to_json(mock_probe, tmp_path, mock_probe_data):
    """메타데이터 추출 후 JSON 저장"""
    video_path = tmp_path / "test.mp4"
    video_path.write_text("fake")
    
    mock_probe.return_value = mock_probe_data
    
    json_path = MetadataExtractor.extract_to_json(video_path)
    
    assert json_path is not None
    assert json_path.exists()
    assert json_path.name == "test.metadata.json"


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_extract_to_json_custom_output(mock_probe, tmp_path, mock_probe_data):
    """커스텀 출력 경로"""
    video_path = tmp_path / "test.mp4"
    video_path.write_text("fake")
    output_path = tmp_path / "custom_metadata.json"
    
    mock_probe.return_value = mock_probe_data
    
    json_path = MetadataExtractor.extract_to_json(video_path, output_path)
    
    assert json_path == output_path
    assert output_path.exists()


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_batch_extract(mock_probe, tmp_path, mock_probe_data):
    """일괄 추출"""
    # 여러 비디오 파일 생성
    for i in range(3):
        (tmp_path / f"video_{i}.mp4").write_text("fake")
    
    mock_probe.return_value = mock_probe_data
    
    results = MetadataExtractor.batch_extract(tmp_path)
    
    assert len(results) == 3
    assert all(metadata is not None for metadata in results.values())


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_batch_extract_with_output_dir(mock_probe, tmp_path, mock_probe_data):
    """일괄 추출 with 출력 디렉토리"""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    output_dir = tmp_path / "metadata"
    
    for i in range(2):
        (video_dir / f"video_{i}.mp4").write_text("fake")
    
    mock_probe.return_value = mock_probe_data
    
    results = MetadataExtractor.batch_extract(video_dir, output_dir=output_dir)
    
    assert len(results) == 2
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.json"))) == 2


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_get_video_info_summary(mock_probe, tmp_path, mock_probe_data):
    """비디오 정보 요약"""
    video_path = tmp_path / "test.mp4"
    video_path.write_text("fake")
    
    mock_probe.return_value = mock_probe_data
    
    summary = MetadataExtractor.get_video_info_summary(video_path)
    
    assert summary is not None
    assert "test.mp4" in summary
    assert "1920x1080" in summary
    assert "h264" in summary


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_validate_video_valid(mock_probe, tmp_path, mock_probe_data):
    """유효한 비디오 검증"""
    video_path = tmp_path / "valid.mp4"
    video_path.write_text("fake")
    
    mock_probe.return_value = mock_probe_data
    
    is_valid = MetadataExtractor.validate_video(video_path)
    
    assert is_valid is True


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_validate_video_no_video_stream(mock_probe, tmp_path):
    """비디오 스트림 없음"""
    video_path = tmp_path / "audio_only.mp4"
    video_path.write_text("fake")
    
    mock_probe.return_value = {
        'format': {'duration': '60.0'},
        'streams': [
            {'codec_type': 'audio', 'codec_name': 'aac'}
        ]
    }
    
    is_valid = MetadataExtractor.validate_video(video_path)
    
    assert is_valid is False


@patch('ingestion.metadata_extractor.ffmpeg.probe')
def test_validate_video_invalid_duration(mock_probe, tmp_path):
    """유효하지 않은 duration"""
    video_path = tmp_path / "zero_duration.mp4"
    video_path.write_text("fake")
    
    mock_probe.return_value = {
        'format': {'duration': '0'},
        'streams': [
            {
                'codec_type': 'video',
                'width': 1920,
                'height': 1080,
                'r_frame_rate': '30/1',
            }
        ]
    }
    
    is_valid = MetadataExtractor.validate_video(video_path)
    
    assert is_valid is False


def test_get_resolution_string_standard():
    """표준 해상도 문자열"""
    metadata_1080p = VideoMetadata(filepath="test", width=1920, height=1080)
    metadata_720p = VideoMetadata(filepath="test", width=1280, height=720)
    metadata_4k = VideoMetadata(filepath="test", width=3840, height=2160)
    
    assert MetadataExtractor.get_resolution_string(metadata_1080p) == "1080p"
    assert MetadataExtractor.get_resolution_string(metadata_720p) == "720p"
    assert MetadataExtractor.get_resolution_string(metadata_4k) == "4K"


def test_get_resolution_string_custom():
    """비표준 해상도"""
    metadata = VideoMetadata(filepath="test", width=1600, height=900)
    
    resolution = MetadataExtractor.get_resolution_string(metadata)
    assert resolution == "1600x900"


def test_get_resolution_string_none():
    """해상도 없음"""
    metadata = VideoMetadata(filepath="test")
    
    resolution = MetadataExtractor.get_resolution_string(metadata)
    assert resolution is None
