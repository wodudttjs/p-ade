"""
P-ADE Dashboard

PySide6 기반 데스크톱 대시보드 애플리케이션
"""

from dashboard.models import (
    Stage, JobStatus, ErrorType,
    JobRow, KPIData, QualityStats, SystemStats,
    make_mock_jobs, make_mock_kpi, make_mock_quality, make_mock_system,
)
from dashboard.table_models import JobsTableModel
from dashboard.widgets import (
    KPICard, StatusBar, ProgressCard, ResourceMeter,
    SectionHeader, Separator,
)
from dashboard.pages import OverviewPage, JobsPage, QualityPage, SettingsPage
from dashboard.app import DashboardApp, run_dashboard
from dashboard.data_service import DataService, get_data_service

__all__ = [
    # Models
    "Stage", "JobStatus", "ErrorType",
    "JobRow", "KPIData", "QualityStats", "SystemStats",
    "make_mock_jobs", "make_mock_kpi", "make_mock_quality", "make_mock_system",
    # Table Models
    "JobsTableModel",
    # Widgets
    "KPICard", "StatusBar", "ProgressCard", "ResourceMeter",
    "SectionHeader", "Separator",
    # Pages
    "OverviewPage", "JobsPage", "QualityPage", "SettingsPage",
    # App
    "DashboardApp", "run_dashboard",
    # Data Service
    "DataService", "get_data_service",
]
