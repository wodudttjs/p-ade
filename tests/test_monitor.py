#!/usr/bin/env python
"""
CLI 모니터 테스트

MVP Phase 2 Week 8: Basic Dashboard (CLI 모니터링)
"""

import sys
import io
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestColors:
    """Colors 클래스 테스트"""
    
    def test_colors_defined(self):
        """색상 코드가 정의되어 있는지"""
        from monitor import Colors
        
        assert hasattr(Colors, "RESET")
        assert hasattr(Colors, "RED")
        assert hasattr(Colors, "GREEN")
        assert hasattr(Colors, "YELLOW")
        assert hasattr(Colors, "BLUE")
        assert hasattr(Colors, "BOLD")
    
    def test_colors_are_strings(self):
        """색상 코드가 문자열인지"""
        from monitor import Colors
        
        assert isinstance(Colors.RESET, str)
        assert isinstance(Colors.GREEN, str)


class TestFormatFunctions:
    """포맷 함수 테스트"""
    
    def test_format_duration_seconds(self):
        """초 단위 포맷"""
        from monitor import format_duration
        
        assert format_duration(30) == "30.0s"
        assert format_duration(59.5) == "59.5s"
    
    def test_format_duration_minutes(self):
        """분 단위 포맷"""
        from monitor import format_duration
        
        assert format_duration(60) == "1.0m"
        assert format_duration(120) == "2.0m"
        assert format_duration(300) == "5.0m"
    
    def test_format_duration_hours(self):
        """시간 단위 포맷"""
        from monitor import format_duration
        
        assert format_duration(3600) == "1.0h"
        assert format_duration(7200) == "2.0h"
    
    def test_format_size_bytes(self):
        """바이트 단위 포맷"""
        from monitor import format_size
        
        assert "B" in format_size(100)
        assert "500.00 B" == format_size(500)
    
    def test_format_size_kb(self):
        """KB 단위 포맷"""
        from monitor import format_size
        
        result = format_size(1024)
        assert "KB" in result
    
    def test_format_size_mb(self):
        """MB 단위 포맷"""
        from monitor import format_size
        
        result = format_size(1024 * 1024)
        assert "MB" in result
    
    def test_format_size_gb(self):
        """GB 단위 포맷"""
        from monitor import format_size
        
        result = format_size(1024 * 1024 * 1024)
        assert "GB" in result


class TestProgressBar:
    """진행률 바 테스트"""
    
    def test_progress_bar_empty(self):
        """빈 진행률 바"""
        from monitor import progress_bar, Colors
        
        result = progress_bar(0, width=10)
        assert "░" * 10 in result
    
    def test_progress_bar_full(self):
        """가득 찬 진행률 바"""
        from monitor import progress_bar, Colors
        
        result = progress_bar(1.0, width=10)
        assert "█" * 10 in result
    
    def test_progress_bar_half(self):
        """50% 진행률 바"""
        from monitor import progress_bar, Colors
        
        result = progress_bar(0.5, width=10)
        assert "█" * 5 in result
        assert "░" * 5 in result


class TestCLIMonitor:
    """CLIMonitor 클래스 테스트"""
    
    @pytest.fixture
    def mock_data_service(self):
        """Mock DataService"""
        service = Mock()
        service.is_connected.return_value = True
        
        # Mock KPI
        kpi = Mock()
        kpi.total_videos = 100
        kpi.downloaded = 80
        kpi.episodes = 1000
        kpi.high_quality = 700
        kpi.success_rate = 0.9
        kpi.avg_processing_time_sec = 45.0
        kpi.queue_depth = 5
        kpi.active_workers = 2
        kpi.storage_gb = 10.5
        kpi.monthly_cost = 5.25
        service.get_kpi.return_value = kpi
        
        # Mock Job Stats
        service.get_job_stats.return_value = {
            "total": 100,
            "pending": 5,
            "running": 10,
            "success": 75,
            "fail": 8,
            "skip": 2,
        }
        
        # Mock Jobs
        job = Mock()
        job.status = "completed"
        job.stage = "download"
        job.video_id = "vid_001"
        job.duration_ms = 5000
        job.error_type = None
        job.log_snippet = None
        service.get_jobs.return_value = [job]
        
        # Mock Quality Stats
        quality = Mock()
        quality.pass_rate = 0.85
        quality.passed = 850
        quality.failed = 150
        quality.confidence_mean = 0.82
        quality.confidence_std = 0.08
        quality.jitter_mean = 0.1
        quality.jitter_std = 0.05
        quality.length_mean = 120.0
        service.get_quality_stats.return_value = quality
        
        return service
    
    def test_monitor_creation(self):
        """CLIMonitor 생성 테스트"""
        from monitor import CLIMonitor
        # CLIMonitor 인스턴스는 생성 시 연결 실패해도 예외 없이 생성됨
        monitor = CLIMonitor()
        assert hasattr(monitor, 'connected')
        assert hasattr(monitor, 'data_service')
    
    def test_print_header_output(self, capsys):
        """헤더 출력 테스트"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.print_header("Test Dashboard")
        
        captured = capsys.readouterr()
        assert "Test Dashboard" in captured.out
        assert "=" in captured.out
    
    def test_print_connection_status_disconnected(self, capsys):
        """연결 끊김 상태 출력"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.connected = False
        monitor.print_connection_status()
        
        captured = capsys.readouterr()
        assert "Not Connected" in captured.out or "mock" in captured.out.lower() or "❌" in captured.out
    
    def test_print_connection_status_connected(self, capsys):
        """연결됨 상태 출력"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.connected = True
        monitor.print_connection_status()
        
        captured = capsys.readouterr()
        assert "Connected" in captured.out or "✅" in captured.out
    
    def test_print_kpi_with_mock(self, mock_data_service, capsys):
        """Mock DataService로 KPI 출력"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.data_service = mock_data_service
        monitor.connected = True
        
        monitor.print_kpi()
        
        captured = capsys.readouterr()
        assert "KPI" in captured.out
        assert "Videos" in captured.out
    
    def test_print_job_stats_with_mock(self, mock_data_service, capsys):
        """Mock DataService로 작업 통계 출력"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.data_service = mock_data_service
        monitor.connected = True
        
        monitor.print_job_stats()
        
        captured = capsys.readouterr()
        assert "Job Statistics" in captured.out or "Job" in captured.out
    
    def test_print_recent_jobs_with_mock(self, mock_data_service, capsys):
        """Mock DataService로 최근 작업 출력"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.data_service = mock_data_service
        monitor.connected = True
        
        monitor.print_recent_jobs(limit=5)
        
        captured = capsys.readouterr()
        assert "Recent Jobs" in captured.out or "Jobs" in captured.out
    
    def test_print_errors_empty(self, mock_data_service, capsys):
        """오류 없음 출력"""
        mock_data_service.get_jobs.return_value = []
        
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.data_service = mock_data_service
        monitor.connected = True
        
        monitor.print_errors()
        
        captured = capsys.readouterr()
        assert "No errors found" in captured.out or "Error" in captured.out
    
    def test_print_quality_stats_with_mock(self, mock_data_service, capsys):
        """Mock DataService로 품질 통계 출력"""
        from monitor import CLIMonitor
        monitor = CLIMonitor()
        monitor.data_service = mock_data_service
        monitor.connected = True
        
        monitor.print_quality_stats()
        
        captured = capsys.readouterr()
        assert "Quality" in captured.out
    
    def test_print_pipeline_status(self, capsys, tmp_path):
        """파이프라인 상태 출력"""
        # 임시 디렉토리 생성
        (tmp_path / "data" / "raw").mkdir(parents=True)
        (tmp_path / "data" / "poses").mkdir()
        
        from monitor import CLIMonitor
        import monitor as monitor_module
        
        original_root = monitor_module.project_root
        monitor_module.project_root = tmp_path
        
        try:
            monitor = CLIMonitor()
            monitor.print_pipeline_status()
            
            captured = capsys.readouterr()
            assert "Pipeline Status" in captured.out
        finally:
            monitor_module.project_root = original_root


class TestCLIArgs:
    """CLI 인자 파싱 테스트"""
    
    def test_main_default(self):
        """기본 실행"""
        with patch("sys.argv", ["monitor.py", "--once"]):
            with patch("monitor.CLIMonitor") as MockMonitor:
                mock_instance = Mock()
                MockMonitor.return_value = mock_instance
                
                from monitor import main
                main()
                
                mock_instance.print_full_dashboard.assert_called_once()
    
    def test_main_jobs(self):
        """--jobs 옵션"""
        with patch("sys.argv", ["monitor.py", "--jobs"]):
            with patch("monitor.CLIMonitor") as MockMonitor:
                mock_instance = Mock()
                MockMonitor.return_value = mock_instance
                
                from monitor import main
                main()
                
                mock_instance.print_recent_jobs.assert_called()
    
    def test_main_errors(self):
        """--errors 옵션"""
        with patch("sys.argv", ["monitor.py", "--errors"]):
            with patch("monitor.CLIMonitor") as MockMonitor:
                mock_instance = Mock()
                MockMonitor.return_value = mock_instance
                
                from monitor import main
                main()
                
                mock_instance.print_errors.assert_called()
    
    def test_main_no_color(self):
        """--no-color 옵션"""
        from monitor import Colors
        original_reset = Colors.RESET
        
        with patch("sys.argv", ["monitor.py", "--once", "--no-color"]):
            with patch("monitor.CLIMonitor") as MockMonitor:
                mock_instance = Mock()
                MockMonitor.return_value = mock_instance
                
                from monitor import main
                main()
        
        # 테스트 후 색상 복원
        Colors.RESET = original_reset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
