"""
비디오 다운로더 테스트
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ingestion.downloader import (
    VideoDownloader,
    VideoQuality,
    DownloadResult,
    progress_hook
)


@pytest.fixture
def temp_download_dir(tmp_path):
    """임시 다운로드 디렉토리"""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    return download_dir


@pytest.fixture
def downloader(temp_download_dir):
    """VideoDownloader 인스턴스"""
    return VideoDownloader(
        output_dir=temp_download_dir,
        preferred_quality=VideoQuality.HIGH,
        max_retries=3
    )


def test_downloader_initialization(downloader, temp_download_dir):
    """다운로더 초기화 테스트"""
    assert downloader.output_dir == temp_download_dir
    assert downloader.preferred_quality == VideoQuality.HIGH
    assert downloader.max_retries == 3
    assert temp_download_dir.exists()


def test_quality_formats():
    """품질별 포맷 코드 검증"""
    assert VideoQuality.LOW.value == "360p"
    assert VideoQuality.MEDIUM.value == "720p"
    assert VideoQuality.HIGH.value == "1080p"
    assert VideoQuality.ULTRA.value == "1440p"
    
    assert "360" in VideoDownloader.QUALITY_FORMATS[VideoQuality.LOW]
    assert "720" in VideoDownloader.QUALITY_FORMATS[VideoQuality.MEDIUM]
    assert "1080" in VideoDownloader.QUALITY_FORMATS[VideoQuality.HIGH]
    assert "1440" in VideoDownloader.QUALITY_FORMATS[VideoQuality.ULTRA]


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_download_success(mock_ydl_class, downloader, temp_download_dir):
    """다운로드 성공 케이스"""
    # Mock 설정
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_info = {
        'id': 'test_video_123',
        'ext': 'mp4',
        'filesize': 1024000,
        'filesize_approx': 1024000,
        'duration': 120.5,
        'format_id': '137+140',
        'resolution': '1920x1080',
        'fps': 30.0,
    }
    mock_ydl.extract_info.return_value = mock_info
    
    # 다운로드 실행
    result = downloader.download("https://youtube.com/watch?v=test123")
    
    # 검증
    assert result.success is True
    assert result.video_id == "test_video_123"
    assert result.filesize_bytes == 1024000
    assert result.duration_sec == 120.5
    assert result.format_id == "137+140"
    assert result.resolution == "1920x1080"
    assert result.fps == 30.0
    assert result.error_message is None
    
    # filepath 검증
    expected_path = temp_download_dir / "test_video_123.mp4"
    assert result.filepath == str(expected_path)


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_download_failure(mock_ydl_class, downloader):
    """다운로드 실패 케이스"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.side_effect = Exception("Network error")
    
    result = downloader.download("https://youtube.com/watch?v=test123")
    
    assert result.success is False
    assert result.error_message == "Network error"
    assert result.filepath is None


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_download_with_progress_callback(mock_ydl_class, downloader):
    """진행률 콜백 포함 다운로드"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_info = {
        'id': 'test123',
        'ext': 'mp4',
        'duration': 60,
    }
    mock_ydl.extract_info.return_value = mock_info
    
    progress_called = []
    def callback(d):
        progress_called.append(d)
    
    result = downloader.download(
        "https://youtube.com/watch?v=test123",
        progress_callback=callback
    )
    
    assert result.success is True
    # progress_hooks가 전달됐는지 확인
    call_args = mock_ydl_class.call_args
    assert 'progress_hooks' in call_args[0][0]


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_download_with_custom_format(mock_ydl_class, downloader):
    """커스텀 포맷 다운로드"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_info = {
        'id': 'custom123',
        'ext': 'mp4',
        'filesize': 5000000,
        'duration': 180,
    }
    mock_ydl.extract_info.return_value = mock_info
    
    custom_format = "bestvideo[height=720]+bestaudio"
    result = downloader.download_with_format(
        "https://youtube.com/watch?v=test123",
        format_code=custom_format
    )
    
    assert result.success is True
    assert result.video_id == "custom123"
    
    # 커스텀 포맷이 사용됐는지 확인
    call_args = mock_ydl_class.call_args
    opts = call_args[0][0]
    assert opts['format'] == custom_format


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_get_video_info(mock_ydl_class, downloader):
    """메타데이터 추출 테스트"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_info = {
        'id': 'info123',
        'title': 'Test Video',
        'duration': 300,
        'view_count': 10000,
    }
    mock_ydl.extract_info.return_value = mock_info
    
    info = downloader.get_video_info("https://youtube.com/watch?v=info123")
    
    assert info is not None
    assert info['id'] == 'info123'
    assert info['title'] == 'Test Video'
    assert info['duration'] == 300
    
    # download=False로 호출됐는지 확인
    mock_ydl.extract_info.assert_called_once_with(
        "https://youtube.com/watch?v=info123",
        download=False
    )


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_get_video_info_failure(mock_ydl_class, downloader):
    """메타데이터 추출 실패"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.side_effect = Exception("Video not found")
    
    info = downloader.get_video_info("https://youtube.com/watch?v=invalid")
    
    assert info is None


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_list_formats(mock_ydl_class, downloader):
    """포맷 목록 조회"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_info = {
        'id': 'format123',
        'formats': [
            {
                'format_id': '137',
                'ext': 'mp4',
                'resolution': '1920x1080',
                'fps': 30,
                'vcodec': 'avc1',
                'acodec': 'none',
                'filesize': 10000000,
            },
            {
                'format_id': '140',
                'ext': 'm4a',
                'resolution': None,
                'fps': None,
                'vcodec': 'none',
                'acodec': 'mp4a',
                'filesize': 2000000,
            }
        ]
    }
    mock_ydl.extract_info.return_value = mock_info
    
    formats = downloader.list_formats("https://youtube.com/watch?v=format123")
    
    assert len(formats) == 2
    assert formats[0]['format_id'] == '137'
    assert formats[0]['resolution'] == '1920x1080'
    assert formats[1]['format_id'] == '140'
    assert formats[1]['acodec'] == 'mp4a'


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_check_available_true(mock_ydl_class, downloader):
    """비디오 가용성 확인 - 성공"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_return = mock_ydl
    mock_ydl.extract_info.return_value = {'id': 'available123'}
    
    with patch.object(downloader, 'get_video_info', return_value={'id': 'test'}):
        available = downloader.check_available("https://youtube.com/watch?v=available123")
        assert available is True


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_check_available_false(mock_ydl_class, downloader):
    """비디오 가용성 확인 - 실패"""
    with patch.object(downloader, 'get_video_info', return_value=None):
        available = downloader.check_available("https://youtube.com/watch?v=unavailable")
        assert available is False


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_estimate_size(mock_ydl_class, downloader):
    """다운로드 크기 예측"""
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_info = {
        'id': 'size123',
        'formats': [
            {'filesize': 15000000},
            {'filesize_approx': 14500000},
        ]
    }
    mock_ydl.extract_info.return_value = mock_info
    
    size = downloader.estimate_size("https://youtube.com/watch?v=size123")
    
    assert size == 15000000


@patch('ingestion.downloader.yt_dlp.YoutubeDL')
def test_estimate_size_no_info(mock_ydl_class, downloader):
    """크기 예측 실패"""
    with patch.object(downloader, 'get_video_info', return_value=None):
        size = downloader.estimate_size("https://youtube.com/watch?v=nosize")
        assert size is None


def test_progress_hook_downloading():
    """진행률 콜백 - 다운로드 중"""
    progress_data = {
        'status': 'downloading',
        'total_bytes': 10000000,
        'downloaded_bytes': 5000000,
        'speed': 1024000,
        'eta': 5,
    }
    
    # 로그만 생성하므로 예외가 발생하지 않으면 성공
    progress_hook(progress_data)


def test_progress_hook_finished():
    """진행률 콜백 - 완료"""
    progress_data = {
        'status': 'finished',
        'filename': 'test_video.mp4',
    }
    
    progress_hook(progress_data)


def test_download_result_dataclass():
    """DownloadResult 데이터클래스 검증"""
    result = DownloadResult(
        success=True,
        filepath="/path/to/video.mp4",
        video_id="abc123",
        filesize_bytes=5000000,
        duration_sec=180.5,
        format_id="137+140",
        resolution="1920x1080",
        fps=30.0,
        error_message=None
    )
    
    assert result.success is True
    assert result.video_id == "abc123"
    assert result.filesize_bytes == 5000000
    assert result.duration_sec == 180.5
