"""
대시보드 메인 애플리케이션

PySide6 기반 GUI 대시보드 메인 윈도우
"""

import sys
from typing import Optional
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QFrame, QLabel, QMessageBox
)
from PySide6.QtGui import QIcon, QFont

from dashboard.models import make_mock_jobs
from dashboard.table_models import JobsTableModel
from dashboard.pages import OverviewPage, JobsPage, QualityPage, SettingsPage
from dashboard.widgets import StatusBar
from dashboard.styles import DARK_THEME, LIGHT_THEME, Colors


class SidebarButton(QPushButton):
    """사이드바 네비게이션 버튼"""
    
    def __init__(self, icon: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(f"{icon}  {text}", parent)
        self.setCheckable(True)
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)


class Sidebar(QFrame):
    """사이드바 네비게이션"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # 로고
        logo = QLabel("🎬 P-ADE")
        logo.setStyleSheet(
            f"font-size: 24px; font-weight: 800; "
            f"color: {Colors.ACCENT_BLUE}; padding: 15px 0;"
        )
        layout.addWidget(logo)
        
        # 구분선
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background: {Colors.BORDER};")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        
        # 네비게이션 버튼
        self.buttons = []
        nav_items = [
            ("📊", "Overview"),
            ("📋", "Jobs"),
            ("📈", "Quality"),
            ("⚙️", "Settings"),
        ]
        
        for icon, text in nav_items:
            btn = SidebarButton(icon, text)
            layout.addWidget(btn)
            self.buttons.append(btn)
        
        layout.addStretch()
        
        # 상태 표시
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        # 버전
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)
        
        # 첫 번째 버튼 선택
        self.buttons[0].setChecked(True)


class DashboardApp(QMainWindow):
    """대시보드 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("P-ADE Dashboard")
        self.setMinimumSize(1200, 800)
        
        self._dark_mode = True
        self._auto_refresh = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._on_refresh)
        
        self._setup_ui()
        self._connect_signals()
        self._apply_theme()
    
    def _setup_ui(self):
        """UI 구성"""
        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 사이드바
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)
        
        # 메인 콘텐츠
        content_frame = QFrame()
        content_frame.setObjectName("content")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # 상단 툴바
        toolbar = QWidget()
        toolbar.setFixedHeight(50)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(20, 0, 20, 0)
        
        self.page_title = QLabel("Overview")
        self.page_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        toolbar_layout.addWidget(self.page_title)
        
        toolbar_layout.addStretch()
        
        # 마지막 업데이트 시간
        self.last_update = QLabel("Last update: —")
        self.last_update.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        toolbar_layout.addWidget(self.last_update)
        
        # 새로고침 버튼
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setFixedSize(36, 36)
        self.btn_refresh.setToolTip("Refresh")
        toolbar_layout.addWidget(self.btn_refresh)
        
        # 테마 토글
        self.btn_theme = QPushButton("🌙")
        self.btn_theme.setFixedSize(36, 36)
        self.btn_theme.setToolTip("Toggle Theme")
        toolbar_layout.addWidget(self.btn_theme)
        
        content_layout.addWidget(toolbar)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {Colors.BORDER};")
        line.setFixedHeight(1)
        content_layout.addWidget(line)
        
        # 스택 위젯
        self.stack = QStackedWidget()
        
        # 데이터 모델
        self.jobs_model = JobsTableModel()
        self.jobs_model.replaceAll(make_mock_jobs())
        
        # 페이지들
        self.overview_page = OverviewPage()
        self.jobs_page = JobsPage(self.jobs_model)
        self.quality_page = QualityPage()
        self.settings_page = SettingsPage()
        
        self.stack.addWidget(self.overview_page)
        self.stack.addWidget(self.jobs_page)
        self.stack.addWidget(self.quality_page)
        self.stack.addWidget(self.settings_page)
        
        content_layout.addWidget(self.stack)
        
        main_layout.addWidget(content_frame)
        
        # 상태바
        self.statusBar().showMessage("Ready")
    
    def _connect_signals(self):
        """시그널 연결"""
        for i, btn in enumerate(self.sidebar.buttons):
            btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx))
        
        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_theme.clicked.connect(self._toggle_theme)
        
        # 설정 페이지 새로고침 간격
        self.settings_page.refresh_combo.currentTextChanged.connect(self._on_refresh_interval_changed)
    
    def _switch_page(self, index: int):
        """페이지 전환"""
        self.stack.setCurrentIndex(index)
        
        # 버튼 체크 상태 업데이트
        for i, btn in enumerate(self.sidebar.buttons):
            btn.setChecked(i == index)
        
        # 페이지 타이틀 업데이트
        titles = ["Overview", "Jobs", "Quality", "Settings"]
        self.page_title.setText(titles[index])
    
    def _on_refresh(self):
        """새로고침"""
        # 현재 페이지 새로고침
        current = self.stack.currentWidget()
        if hasattr(current, 'refresh'):
            current.refresh()
        
        # 타임스탬프 업데이트
        now = datetime.now().strftime("%H:%M:%S")
        self.last_update.setText(f"Last update: {now}")
        self.statusBar().showMessage(f"Refreshed at {now}")
    
    def _toggle_theme(self):
        """테마 토글"""
        self._dark_mode = not self._dark_mode
        self._apply_theme()
        self.btn_theme.setText("☀️" if self._dark_mode else "🌙")
    
    def _apply_theme(self):
        """테마 적용"""
        theme = DARK_THEME if self._dark_mode else LIGHT_THEME
        self.setStyleSheet(theme)
    
    def _on_refresh_interval_changed(self, text: str):
        """자동 새로고침 간격 변경"""
        self._refresh_timer.stop()
        
        intervals = {
            "Off": 0,
            "5s": 5000,
            "10s": 10000,
            "30s": 30000,
            "1m": 60000,
            "5m": 300000,
        }
        
        interval = intervals.get(text, 0)
        if interval > 0:
            self._refresh_timer.start(interval)
            self.statusBar().showMessage(f"Auto-refresh enabled: {text}")
        else:
            self.statusBar().showMessage("Auto-refresh disabled")


def run_dashboard():
    """대시보드 실행"""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("P-ADE Dashboard")
    app.setFont(QFont("Segoe UI", 10))
    
    window = DashboardApp()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_dashboard())
