"""
Dashboard 모듈 테스트

PySide6 GUI 컴포넌트 테스트
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestDashboardModels:
    """데이터 모델 테스트"""
    
    def test_job_row_creation(self):
        """JobRow 생성 테스트"""
        from dashboard.models import JobRow
        
        job = JobRow(
            job_key="test-job-001",
            run_id="run-123",
            stage="download",
            status="success",
            error_type=None,
            started_at=datetime.now(),
            duration_ms=5000,
            video_id="vid123",
            episode_id="ep001",
        )
        
        assert job.job_key == "test-job-001"
        assert job.stage == "download"
        assert job.status == "success"
        assert job.video_id == "vid123"
    
    def test_kpi_data_creation(self):
        """KPIData 생성 테스트"""
        from dashboard.models import KPIData
        
        kpi = KPIData(
            total_videos=100,
            downloaded=80,
            episodes=500,
            high_quality=450,
        )
        
        assert kpi.total_videos == 100
        assert kpi.downloaded == 80
        assert kpi.high_quality == 450
    
    def test_quality_stats_creation(self):
        """QualityStats 생성 테스트"""
        from dashboard.models import QualityStats
        
        quality = QualityStats(
            total_episodes=1000,
            passed=950,
            failed=50,
            pass_rate=0.95,  # 직접 설정
        )
        
        assert quality.pass_rate == 0.95
    
    def test_make_mock_jobs(self):
        """mock jobs 생성 테스트"""
        from dashboard.models import make_mock_jobs
        
        jobs = make_mock_jobs(10)
        
        assert len(jobs) == 10
        for job in jobs:
            assert job.job_key is not None
            assert job.stage is not None
            assert job.status is not None
    
    def test_make_mock_kpi(self):
        """mock KPI 생성 테스트"""
        from dashboard.models import make_mock_kpi
        
        kpi = make_mock_kpi()
        
        assert kpi.total_videos > 0
        assert 0 <= kpi.success_rate <= 1
    
    def test_make_mock_quality(self):
        """mock quality 생성 테스트"""
        from dashboard.models import make_mock_quality
        
        quality = make_mock_quality()
        
        assert quality.total_episodes > 0
        assert quality.passed + quality.failed == quality.total_episodes
    
    def test_make_mock_system(self):
        """mock system stats 생성 테스트"""
        from dashboard.models import make_mock_system
        
        system = make_mock_system()
        
        assert 0 <= system.cpu_percent <= 100
        assert 0 <= system.memory_percent <= 100


class TestStyles:
    """스타일 테스트"""
    
    def test_dark_theme_exists(self):
        """다크 테마 존재 확인"""
        from dashboard.styles import DARK_THEME
        
        assert DARK_THEME is not None
        assert len(DARK_THEME) > 0
        assert "QMainWindow" in DARK_THEME
    
    def test_light_theme_exists(self):
        """라이트 테마 존재 확인"""
        from dashboard.styles import LIGHT_THEME
        
        assert LIGHT_THEME is not None
        assert len(LIGHT_THEME) > 0
    
    def test_colors_class(self):
        """Colors 클래스 테스트"""
        from dashboard.styles import Colors
        
        assert Colors.BG_DARK is not None
        assert Colors.TEXT_PRIMARY is not None
        assert Colors.ACCENT_BLUE is not None
        assert Colors.SUCCESS is not None
        assert Colors.ERROR is not None
    
    def test_get_status_color(self):
        """상태별 색상 함수 테스트"""
        from dashboard.styles import get_status_color, Colors
        
        assert get_status_color("success") == Colors.SUCCESS
        assert get_status_color("fail") == Colors.ERROR
        assert get_status_color("running") == Colors.RUNNING
        assert get_status_color("skip") == Colors.WARNING


class TestTableModels:
    """테이블 모델 테스트"""
    
    def test_jobs_table_model_creation(self):
        """JobsTableModel 생성 테스트"""
        from dashboard.table_models import JobsTableModel
        
        model = JobsTableModel()
        
        assert model is not None
        assert model.rowCount() == 0
    
    def test_jobs_table_model_with_data(self):
        """데이터가 있는 JobsTableModel 테스트"""
        from dashboard.table_models import JobsTableModel
        from dashboard.models import make_mock_jobs
        
        jobs = make_mock_jobs(5)
        model = JobsTableModel(jobs)
        
        assert model.rowCount() == 5
    
    def test_jobs_table_model_replace_all(self):
        """replaceAll 테스트"""
        from dashboard.table_models import JobsTableModel
        from dashboard.models import make_mock_jobs
        
        model = JobsTableModel()
        assert model.rowCount() == 0
        
        jobs = make_mock_jobs(10)
        model.replaceAll(jobs)
        
        assert model.rowCount() == 10
    
    def test_jobs_table_model_filter(self):
        """필터 테스트"""
        from dashboard.table_models import JobsTableModel
        from dashboard.models import JobRow
        from datetime import datetime
        
        jobs = [
            JobRow(job_key="download-1", run_id="r1", stage="download", status="success", 
                   error_type=None, started_at=datetime.now(), duration_ms=1000, 
                   video_id="v1", episode_id="e1"),
            JobRow(job_key="extract-1", run_id="r2", stage="extract", status="fail", 
                   error_type="NETWORK_TRANSIENT", started_at=datetime.now(), duration_ms=2000, 
                   video_id="v2", episode_id="e2"),
            JobRow(job_key="download-2", run_id="r3", stage="download", status="running", 
                   error_type=None, started_at=datetime.now(), duration_ms=None, 
                   video_id="v3", episode_id="e3"),
        ]
        
        model = JobsTableModel(jobs)
        
        # Stage 필터
        model.setFilter("", "download", "all")
        assert model.rowCount() == 2
        
        # Status 필터
        model.setFilter("", "all", "success")
        assert model.rowCount() == 1
        
        # 검색어 필터
        model.setFilter("extract", "all", "all")
        assert model.rowCount() == 1
    
    def test_jobs_table_model_row_at(self):
        """rowAt 테스트"""
        from dashboard.table_models import JobsTableModel
        from dashboard.models import make_mock_jobs
        
        jobs = make_mock_jobs(5)
        model = JobsTableModel(jobs)
        
        job = model.rowAt(0)
        assert job is not None
        assert job == jobs[0]
    
    def test_jobs_table_model_get_stats(self):
        """getStats 테스트"""
        from dashboard.table_models import JobsTableModel
        from dashboard.models import JobRow
        from datetime import datetime
        
        jobs = [
            JobRow(job_key="j1", run_id="r1", stage="download", status="success", 
                   error_type=None, started_at=datetime.now(), duration_ms=1000, 
                   video_id="v1", episode_id="e1"),
            JobRow(job_key="j2", run_id="r2", stage="extract", status="fail", 
                   error_type="TIMEOUT", started_at=datetime.now(), duration_ms=2000, 
                   video_id="v2", episode_id="e2"),
            JobRow(job_key="j3", run_id="r3", stage="qc", status="success", 
                   error_type=None, started_at=datetime.now(), duration_ms=500, 
                   video_id="v3", episode_id="e3"),
        ]
        
        model = JobsTableModel(jobs)
        stats = model.getStats()
        
        assert stats["total"] == 3
        assert stats.get("success", 0) == 2
        assert stats.get("fail", 0) == 1


class TestModuleExports:
    """모듈 export 테스트"""
    
    def test_dashboard_module_imports(self):
        """대시보드 모듈 import 테스트"""
        from dashboard import (
            Stage, JobStatus, ErrorType,
            JobRow, KPIData, QualityStats, SystemStats,
            make_mock_jobs, make_mock_kpi, make_mock_quality, make_mock_system,
            JobsTableModel,
            KPICard, StatusBar, ProgressCard, ResourceMeter,
            SectionHeader, Separator,
            DashboardApp, run_dashboard,
        )
        
        # 모든 import가 성공하면 테스트 통과
        assert True


class TestIntegration:
    """통합 테스트 (GUI 없이)"""
    
    def test_jobs_table_model_with_mock_data(self):
        """mock 데이터로 테이블 모델 테스트"""
        from dashboard.table_models import JobsTableModel
        from dashboard.models import make_mock_jobs
        
        jobs = make_mock_jobs(100)
        model = JobsTableModel(jobs)
        
        # 모든 열 헤더 확인
        for col in range(model.columnCount()):
            header = model.headerData(col, orientation=1)  # Qt.Horizontal = 1
            assert header is not None
        
        # 데이터 접근
        for row in range(min(10, model.rowCount())):
            for col in range(model.columnCount()):
                data = model.data(model.index(row, col))
                # 데이터가 있어야 함 (None도 허용)
    
    def test_full_workflow(self):
        """전체 워크플로우 테스트"""
        from dashboard.models import (
            make_mock_jobs, make_mock_kpi, 
            make_mock_quality, make_mock_system
        )
        from dashboard.table_models import JobsTableModel
        
        # 1. mock 데이터 생성
        jobs = make_mock_jobs(50)
        kpi = make_mock_kpi()
        quality = make_mock_quality()
        system = make_mock_system()
        
        # 2. 테이블 모델 생성
        model = JobsTableModel(jobs)
        
        # 3. 필터링
        model.setFilter("", "download", "all")
        filtered_count = model.rowCount()
        
        model.setFilter("", "all", "all")
        assert model.rowCount() == 50
        
        # 4. 통계 확인
        stats = model.getStats()
        assert stats["total"] == 50
        
        # 5. KPI 확인
        assert kpi.total_videos > 0
        assert kpi.success_rate >= 0
        
        # 6. 품질 확인
        assert quality.pass_rate >= 0
        
        # 7. 시스템 리소스 확인
        assert 0 <= system.cpu_percent <= 100
