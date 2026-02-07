"""
CSVExporter 테스트
"""

import pytest
from pathlib import Path
from datetime import datetime

from storage.csv_exporter import CSVExporter
from models.database import Video


# ============ Fixtures ============

@pytest.fixture
def sample_videos():
    """샘플 비디오 데이터"""
    return [
        Video(
            video_id="vid1",
            platform="youtube",
            url="https://youtube.com/watch?v=vid1",
            title="Robot Assembly Tutorial",
            description="Learn robot assembly",
            duration_sec=300,
            channel_name="RoboticsLab",
            view_count=10000,
            status="discovered"
        ),
        Video(
            video_id="vid2",
            platform="youtube",
            url="https://youtube.com/watch?v=vid2",
            title="Pick and Place Demo",
            description="Pick and place demonstration",
            duration_sec=120,
            channel_name="RoboChannel",
            view_count=5000,
            status="downloaded"
        ),
        Video(
            video_id="vid3",
            platform="vimeo",
            url="https://vimeo.com/vid3",
            title="Industrial Robot Programming",
            description="Programming guide",
            duration_sec=600,
            channel_name="IndustrialRobotics",
            view_count=8000,
            status="discovered"
        )
    ]


# ============ Basic Export Tests ============

def test_csv_exporter_import():
    """CSVExporter import 확인"""
    from storage.csv_exporter import CSVExporter
    assert CSVExporter is not None


def test_csv_exporter_initialization(temp_dir):
    """CSVExporter 초기화 테스트"""
    exporter = CSVExporter(temp_dir)
    
    assert exporter.output_dir == temp_dir
    assert exporter.output_dir.exists()


def test_export_videos(temp_dir, sample_videos):
    """비디오 CSV 저장 테스트"""
    exporter = CSVExporter(temp_dir)
    
    filepath = exporter.export_videos(sample_videos)
    
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix == '.csv'
    
    # 파일 내용 확인
    content = filepath.read_text(encoding='utf-8')
    assert 'video_id' in content
    assert 'vid1' in content
    assert 'vid2' in content
    assert 'vid3' in content


def test_export_videos_with_custom_filename(temp_dir, sample_videos):
    """커스텀 파일명으로 저장 테스트"""
    exporter = CSVExporter(temp_dir)
    
    filepath = exporter.export_videos(
        videos=sample_videos,
        filename="custom_export.csv"
    )
    
    assert filepath.name == "custom_export.csv"
    assert filepath.exists()


def test_export_videos_with_custom_fields(temp_dir, sample_videos):
    """커스텀 필드로 저장 테스트"""
    exporter = CSVExporter(temp_dir)
    
    custom_fields = ['video_id', 'title', 'platform']
    filepath = exporter.export_videos(
        videos=sample_videos,
        fields=custom_fields
    )
    
    # CSV 헤더 확인
    content = filepath.read_text(encoding='utf-8')
    lines = content.split('\n')
    header = lines[0]
    
    assert 'video_id' in header
    assert 'title' in header
    assert 'platform' in header
    # 기본 필드는 포함되지 않아야 함
    assert 'description' not in header


def test_export_empty_videos(temp_dir):
    """빈 비디오 리스트 저장 테스트"""
    exporter = CSVExporter(temp_dir)
    
    result = exporter.export_videos([])
    
    assert result is None


# ============ Grouping Tests ============

def test_export_videos_by_platform(temp_dir, sample_videos):
    """플랫폼별 CSV 저장 테스트"""
    exporter = CSVExporter(temp_dir)
    
    result = exporter.export_videos_by_platform(sample_videos)
    
    assert 'youtube' in result
    assert 'vimeo' in result
    assert result['youtube'].exists()
    assert result['vimeo'].exists()
    
    # YouTube CSV 확인
    youtube_content = result['youtube'].read_text(encoding='utf-8')
    assert 'vid1' in youtube_content
    assert 'vid2' in youtube_content
    assert 'vid3' not in youtube_content
    
    # Vimeo CSV 확인
    vimeo_content = result['vimeo'].read_text(encoding='utf-8')
    assert 'vid3' in vimeo_content
    assert 'vid1' not in vimeo_content


def test_export_videos_by_status(temp_dir, sample_videos):
    """상태별 CSV 저장 테스트"""
    exporter = CSVExporter(temp_dir)
    
    result = exporter.export_videos_by_status(sample_videos)
    
    assert 'discovered' in result
    assert 'downloaded' in result
    assert result['discovered'].exists()
    assert result['downloaded'].exists()
    
    # discovered 상태 CSV 확인
    discovered_content = result['discovered'].read_text(encoding='utf-8')
    assert 'vid1' in discovered_content
    assert 'vid3' in discovered_content
    assert 'vid2' not in discovered_content


# ============ Append Tests ============

def test_append_video_to_new_file(temp_dir, sample_videos):
    """새 파일에 비디오 추가 테스트"""
    exporter = CSVExporter(temp_dir)
    
    result = exporter.append_video(
        video=sample_videos[0],
        filename="append_test.csv"
    )
    
    assert result is True
    filepath = temp_dir / "append_test.csv"
    assert filepath.exists()
    
    content = filepath.read_text(encoding='utf-8')
    assert 'vid1' in content


def test_append_video_to_existing_file(temp_dir, sample_videos):
    """기존 파일에 비디오 추가 테스트"""
    exporter = CSVExporter(temp_dir)
    
    # 첫 번째 비디오로 파일 생성
    exporter.export_videos([sample_videos[0]], filename="append_test.csv")
    
    # 두 번째 비디오 추가
    result = exporter.append_video(
        video=sample_videos[1],
        filename="append_test.csv"
    )
    
    assert result is True
    
    # 파일 내용 확인
    filepath = temp_dir / "append_test.csv"
    content = filepath.read_text(encoding='utf-8')
    assert 'vid1' in content
    assert 'vid2' in content


# ============ Read Tests ============

def test_read_csv(temp_dir, sample_videos):
    """CSV 파일 읽기 테스트"""
    exporter = CSVExporter(temp_dir)
    
    # CSV 파일 생성
    filename = "read_test.csv"
    exporter.export_videos(sample_videos, filename=filename)
    
    # 읽기
    videos = exporter.read_csv(filename)
    
    assert len(videos) == 3
    assert videos[0]['video_id'] == 'vid1'
    assert videos[1]['video_id'] == 'vid2'
    assert videos[2]['video_id'] == 'vid3'


def test_read_nonexistent_csv(temp_dir):
    """존재하지 않는 CSV 파일 읽기 테스트"""
    exporter = CSVExporter(temp_dir)
    
    videos = exporter.read_csv("nonexistent.csv")
    
    assert videos == []


# ============ Utility Tests ============

def test_get_csv_files_empty(temp_dir):
    """빈 디렉토리에서 CSV 파일 목록 조회"""
    exporter = CSVExporter(temp_dir)
    
    files = exporter.get_csv_files()
    
    assert files == []


def test_get_csv_files(temp_dir, sample_videos):
    """CSV 파일 목록 조회 테스트"""
    exporter = CSVExporter(temp_dir)
    
    # 여러 CSV 파일 생성
    exporter.export_videos([sample_videos[0]], filename="file1.csv")
    exporter.export_videos([sample_videos[1]], filename="file2.csv")
    exporter.export_videos([sample_videos[2]], filename="file3.csv")
    
    files = exporter.get_csv_files()
    
    assert len(files) == 3
    # 최신 파일부터 정렬되어야 함
    assert all(f.suffix == '.csv' for f in files)


def test_video_to_dict(temp_dir, sample_videos):
    """Video 객체를 딕셔너리로 변환 테스트"""
    exporter = CSVExporter(temp_dir)
    
    video = sample_videos[0]
    fields = ['video_id', 'title', 'platform']
    
    result = exporter._video_to_dict(video, fields)
    
    assert result['video_id'] == 'vid1'
    assert result['title'] == 'Robot Assembly Tutorial'
    assert result['platform'] == 'youtube'
    assert 'description' not in result


def test_datetime_conversion(temp_dir):
    """날짜 변환 테스트"""
    exporter = CSVExporter(temp_dir)
    
    video = Video(
        video_id="test",
        platform="youtube",
        url="http://test.com",
        title="Test",
        discovered_at=datetime(2024, 1, 1, 12, 0, 0)
    )
    
    filepath = exporter.export_videos([video])
    
    content = filepath.read_text(encoding='utf-8')
    # ISO 8601 형식으로 저장되어야 함
    assert '2024-01-01' in content


def test_list_field_conversion(temp_dir):
    """리스트 필드 변환 테스트"""
    exporter = CSVExporter(temp_dir)
    
    video = Video(
        video_id="test",
        platform="youtube",
        url="http://test.com",
        title="Test",
        tags=["robot", "assembly", "tutorial"]
    )
    
    filepath = exporter.export_videos([video])
    
    content = filepath.read_text(encoding='utf-8')
    # 리스트가 문자열로 변환되어야 함
    assert "['robot', 'assembly', 'tutorial']" in content or \
           '["robot", "assembly", "tutorial"]' in content
