"""
대시보드 메인 애플리케이션

PySide6 기반 GUI 대시보드 메인 윈도우
"""

import os
import sys
import subprocess
import threading
from typing import Optional
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QFrame, QLabel, QMessageBox,
    QProgressBar, QGroupBox, QLineEdit, QSpinBox
)
from PySide6.QtGui import QIcon, QFont

from dashboard.models import make_mock_jobs
from dashboard.table_models import JobsTableModel
from dashboard.pages import OverviewPage, JobsPage, QualityPage, SettingsPage
from dashboard.widgets import StatusBar, ProgressCard
from dashboard.styles import DARK_THEME, LIGHT_THEME, Colors

# DataService import (실제 DB 연동)
try:
    from dashboard.data_service import get_data_service
    HAS_DATA_SERVICE = True
except ImportError:
    HAS_DATA_SERVICE = False


class WorkerSignals(QObject):
    """워커 스레드 시그널"""
    started = Signal()
    stopped = Signal()
    progress = Signal(str, int, int)  # stage, current, total
    error = Signal(str)
    log = Signal(str)


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
    
    def __init__(self, db_url: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("P-ADE Dashboard")
        self.setMinimumSize(1200, 800)
        
        self._dark_mode = True
        self._auto_refresh = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._on_refresh)
        
        # 프로젝트 루트
        self._project_root = Path(__file__).parent.parent
        
        # 수집 프로세스 관리
        self._collection_process = None
        self._is_collecting = False
        self._worker_signals = WorkerSignals()
        
        # DataService 초기화
        self._data_service = None
        self._use_real_data = False
        if HAS_DATA_SERVICE:
            try:
                self._data_service = get_data_service(db_url)
                self._use_real_data = self._data_service.is_connected()
            except Exception:
                pass
        
        self._setup_ui()
        self._connect_signals()
        self._apply_theme()
        
        # DB 데이터 자동 새로고침 (5초마다)
        self._db_refresh_timer = QTimer(self)
        self._db_refresh_timer.timeout.connect(self._refresh_db_stats)
        self._db_refresh_timer.start(5000)
        
        # 초기 DB 통계 로드
        self._refresh_db_stats()
        
        # 상태 표시
        if self._use_real_data:
            self.statusBar().showMessage("✓ Connected to database")
            self.sidebar.status_bar.setConnected(True)
        else:
            self.statusBar().showMessage("⚠ Using mock data (DB not connected)")
            self.sidebar.status_bar.setConnected(False)
    
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
        
        # ===== 상단 컨트롤 패널 (Start/Stop 버튼 + 진행 상황) =====
        control_panel = QWidget()
        control_panel.setFixedHeight(120)
        control_panel.setStyleSheet(f"background: {Colors.BG_CARD}; border-bottom: 1px solid {Colors.BORDER};")
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 10, 20, 10)
        
        # Start/Stop 버튼 그룹
        btn_group = QGroupBox("Pipeline Control")
        btn_group.setFixedWidth(200)
        btn_layout = QVBoxLayout(btn_group)
        
        self.btn_start = QPushButton("▶ Start Collection")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.SUCCESS};
                color: white;
                font-weight: 600;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: #27ae60; }}
            QPushButton:disabled {{ background: {Colors.TEXT_MUTED}; }}
        """)
        btn_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ERROR};
                color: white;
                font-weight: 600;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: #c0392b; }}
            QPushButton:disabled {{ background: {Colors.TEXT_MUTED}; }}
        """)
        btn_layout.addWidget(self.btn_stop)
        
        # 검색어 입력
        query_layout = QHBoxLayout()
        query_label = QLabel("Query:")
        query_label.setStyleSheet("font-size: 11px;")
        query_layout.addWidget(query_label)
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("2족보행, 로봇, 걷기...")
        self.query_input.setText("2족보행 로봇")
        self.query_input.setFixedHeight(28)
        self.query_input.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_MAIN};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        query_layout.addWidget(self.query_input)
        btn_layout.addLayout(query_layout)
        
        # 수집 설정 (Videos, Workers)
        settings_layout = QHBoxLayout()
        
        count_label = QLabel("Videos:")
        count_label.setStyleSheet("font-size: 11px;")
        settings_layout.addWidget(count_label)
        
        self.video_count_spin = QSpinBox()
        self.video_count_spin.setRange(1, 100)
        self.video_count_spin.setValue(5)
        self.video_count_spin.setFixedWidth(50)
        settings_layout.addWidget(self.video_count_spin)
        
        workers_label = QLabel("Workers:")
        workers_label.setStyleSheet("font-size: 11px;")
        settings_layout.addWidget(workers_label)
        
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(2)
        self.workers_spin.setFixedWidth(50)
        settings_layout.addWidget(self.workers_spin)
        
        btn_layout.addLayout(settings_layout)
        
        control_layout.addWidget(btn_group)
        
        # 진행 상황 표시
        progress_group = QGroupBox("Pipeline Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # 각 단계별 프로그레스 바
        self.progress_bars = {}
        stages = [
            ("download", "📥 Download", Colors.ACCENT_BLUE),
            ("extract", "🔍 Extract", Colors.ACCENT_PURPLE),
            ("filter", "✨ Filter", Colors.ACCENT_GREEN),
            ("encode", "🔧 Encode", Colors.ACCENT_YELLOW),
            ("upload", "☁️ Upload", Colors.ACCENT_BLUE),
        ]
        
        progress_grid = QHBoxLayout()
        for stage_id, stage_name, color in stages:
            stage_widget = QWidget()
            stage_layout = QVBoxLayout(stage_widget)
            stage_layout.setContentsMargins(5, 0, 5, 0)
            stage_layout.setSpacing(2)
            
            label = QLabel(stage_name)
            label.setStyleSheet(f"font-size: 11px; color: {Colors.TEXT_SECONDARY};")
            stage_layout.addWidget(label)
            
            pbar = QProgressBar()
            pbar.setMaximum(100)
            pbar.setValue(0)
            pbar.setTextVisible(True)
            pbar.setFixedHeight(20)
            pbar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {Colors.BORDER};
                    border-radius: 4px;
                    text-align: center;
                    background: {Colors.BG_MAIN};
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 3px;
                }}
            """)
            stage_layout.addWidget(pbar)
            
            self.progress_bars[stage_id] = pbar
            progress_grid.addWidget(stage_widget)
        
        progress_layout.addLayout(progress_grid)
        
        # 전체 진행률
        total_layout = QHBoxLayout()
        total_label = QLabel("Total Progress:")
        total_label.setStyleSheet("font-weight: 600;")
        total_layout.addWidget(total_label)
        
        self.total_progress = QProgressBar()
        self.total_progress.setMaximum(100)
        self.total_progress.setValue(0)
        self.total_progress.setFixedHeight(24)
        self.total_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Colors.ACCENT_BLUE}, stop:1 {Colors.ACCENT_GREEN});
                border-radius: 3px;
            }}
        """)
        total_layout.addWidget(self.total_progress)
        
        self.progress_status = QLabel("Ready")
        self.progress_status.setFixedWidth(150)
        self.progress_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        total_layout.addWidget(self.progress_status)
        
        progress_layout.addLayout(total_layout)
        
        control_layout.addWidget(progress_group, 1)
        
        # DB 통계 요약
        db_group = QGroupBox("Database Stats")
        db_group.setFixedWidth(200)
        db_layout = QVBoxLayout(db_group)
        
        self.db_stats_labels = {}
        db_items = [
            ("videos", "📹 Collected:"),
            ("episodes", "🎬 Episodes:"),
            ("jobs", "📋 Jobs:"),
            ("storage", "💾 Storage:"),
            ("db", "🗄️ DB:"),
            ("s3", "☁️ S3:"),
        ]
        
        for key, label_text in db_items:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setStyleSheet(f"font-size: 11px;")
            row.addWidget(label)
            
            value = QLabel("—")
            value.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {Colors.ACCENT_BLUE};")
            value.setAlignment(Qt.AlignRight)
            row.addWidget(value)
            
            self.db_stats_labels[key] = value
            db_layout.addLayout(row)
        
        control_layout.addWidget(db_group)
        
        content_layout.addWidget(control_panel)
        
        # ===== 상단 툴바 =====
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
        
        # 데이터 로드
        if self._use_real_data and self._data_service:
            # 실제 DB에서 jobs 로드
            jobs = self._data_service.get_jobs(limit=100)
        else:
            jobs = make_mock_jobs()
        
        self.jobs_model = JobsTableModel()
        self.jobs_model.replaceAll(jobs)
        
        # 페이지들 (DataService 주입)
        self.overview_page = OverviewPage(data_service=self._data_service)
        self.jobs_page = JobsPage(self.jobs_model)
        self.quality_page = QualityPage(data_service=self._data_service)
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
        
        # Start/Stop 버튼
        self.btn_start.clicked.connect(self._start_collection)
        self.btn_stop.clicked.connect(self._stop_collection)
        
        # 워커 시그널
        self._worker_signals.started.connect(self._on_collection_started)
        self._worker_signals.stopped.connect(self._on_collection_stopped)
        self._worker_signals.progress.connect(self._on_progress_update)
        self._worker_signals.log.connect(self._on_log_message)
        
        # 설정 페이지 새로고침 간격
        self.settings_page.refresh_combo.currentTextChanged.connect(self._on_refresh_interval_changed)
    
    def _start_collection(self):
        """영상 수집 시작"""
        if self._is_collecting:
            return
        
        self._is_collecting = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_status.setText("Starting...")
        self.statusBar().showMessage("🚀 Collection started")
        
        # 프로그레스 바 리셋
        for pbar in self.progress_bars.values():
            pbar.setValue(0)
        self.total_progress.setValue(0)
        
        # 백그라운드 스레드에서 수집 실행
        def run_collection():
            try:
                self._worker_signals.started.emit()
                
                # 설정값 가져오기
                query = self.query_input.text().strip() or "2족보행 로봇"
                video_count = self.video_count_spin.value()
                workers = self.workers_spin.value()
                
                # 1. Download 단계
                self._worker_signals.progress.emit("download", 0, 100)
                self._run_stage_script("parallel_download.py", [
                    "--query", query,
                    "--workers", str(workers),
                    "--limit", str(video_count)
                ])
                self._worker_signals.progress.emit("download", 100, 100)
                
                if not self._is_collecting:
                    return
                
                # 2. Extract 단계
                self._worker_signals.progress.emit("extract", 0, 100)
                self._run_stage_script("extract_poses.py", ["--all"])
                self._worker_signals.progress.emit("extract", 100, 100)
                
                if not self._is_collecting:
                    return
                
                # 3. Filter 단계
                self._worker_signals.progress.emit("filter", 0, 100)
                self._run_stage_script("filter_quality.py", ["--all", "--update-db"])
                self._worker_signals.progress.emit("filter", 100, 100)
                
                if not self._is_collecting:
                    return
                
                # 4. Encode 단계
                self._worker_signals.progress.emit("encode", 0, 100)
                self._run_stage_script("encode_actions.py", ["--all"])
                self._worker_signals.progress.emit("encode", 100, 100)
                
                if not self._is_collecting:
                    return
                
                # 5. Upload 단계 (클라우드)
                self._worker_signals.progress.emit("upload", 0, 100)
                self._run_stage_script("upload_to_s3.py", ["--all"])
                self._worker_signals.progress.emit("upload", 100, 100)
                
                self._worker_signals.stopped.emit()
                
            except Exception as e:
                self._worker_signals.log.emit(f"Error: {e}")
                self._worker_signals.stopped.emit()
        
        self._collection_thread = threading.Thread(target=run_collection, daemon=True)
        self._collection_thread.start()
    
    def _run_stage_script(self, script_name: str, args: list):
        """스테이지 스크립트 실행"""
        script_path = self._project_root / script_name
        if not script_path.exists():
            self._worker_signals.log.emit(f"Script not found: {script_name}")
            return
        
        cmd = [sys.executable, str(script_path)] + args
        self._worker_signals.log.emit(f"Running: {script_name}")
        
        # 환경 변수 설정 (인코딩 문제 해결)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            self._collection_process = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            
            # 출력 읽기
            for line in self._collection_process.stdout:
                if not self._is_collecting:
                    self._collection_process.terminate()
                    break
                line = line.strip()
                if line:
                    self._worker_signals.log.emit(line)
            
            self._collection_process.wait()
            
        except Exception as e:
            self._worker_signals.log.emit(f"Error running {script_name}: {e}")
    
    def _stop_collection(self):
        """영상 수집 중지"""
        self._is_collecting = False
        self.progress_status.setText("Stopping...")
        self.statusBar().showMessage("⏹ Stopping collection...")
        
        if self._collection_process:
            try:
                self._collection_process.terminate()
            except Exception:
                pass
    
    def _on_collection_started(self):
        """수집 시작됨"""
        self.progress_status.setText("Running...")
        self.progress_status.setStyleSheet(f"color: {Colors.SUCCESS};")
    
    def _on_collection_stopped(self):
        """수집 중지됨"""
        self._is_collecting = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_status.setText("Completed" if self.total_progress.value() >= 100 else "Stopped")
        self.progress_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        self.statusBar().showMessage("✓ Collection finished")
        self._refresh_db_stats()
    
    def _on_progress_update(self, stage: str, current: int, total: int):
        """진행 상황 업데이트"""
        if stage in self.progress_bars:
            pct = int(current * 100 / max(total, 1))
            self.progress_bars[stage].setValue(pct)
        
        # 전체 진행률 계산
        total_pct = sum(pb.value() for pb in self.progress_bars.values()) // len(self.progress_bars)
        self.total_progress.setValue(total_pct)
    
    def _on_log_message(self, message: str):
        """로그 메시지"""
        self.statusBar().showMessage(message)
    
    def _refresh_db_stats(self):
        """DB 통계 새로고침"""
        if not self._data_service:
            return

        if self.db_stats_labels.get("db"):
            self.db_stats_labels["db"].setText(self._data_service.get_db_status())
        if self.db_stats_labels.get("s3"):
            self.db_stats_labels["s3"].setText(self._data_service.get_s3_status())

        if not self._use_real_data:
            return
        
        try:
            kpi = self._data_service.get_kpi()
            self.db_stats_labels["videos"].setText(f"{kpi.total_videos}")
            self.db_stats_labels["episodes"].setText(f"{kpi.episodes:,}")
            self.db_stats_labels["jobs"].setText(f"{kpi.queue_depth} pending")
            self.db_stats_labels["storage"].setText(f"{kpi.storage_gb:.1f} GB")
            
            # 파이프라인 진행률 계산 (파일 기반)
            self._update_pipeline_progress()

            # 페이지 데이터 새로고침
            self.overview_page.refresh()
            self.quality_page.refresh()
            if self.jobs_model:
                jobs = self._data_service.get_jobs(limit=100)
                self.jobs_model.replaceAll(jobs)
            
        except Exception as e:
            pass
    
    def _update_pipeline_progress(self):
        """파이프라인 진행률 업데이트 (파일 기반)"""
        data_dir = self._project_root / "data"
        
        # 각 단계 파일 수 확인
        try:
            raw_count = len(list((data_dir / "raw").glob("*.mp4"))) if (data_dir / "raw").exists() else 0
            poses_count = len(list((data_dir / "poses").glob("*.npz"))) if (data_dir / "poses").exists() else 0
            filtered_count = len(list((data_dir / "filtered").glob("*.npz"))) if (data_dir / "filtered").exists() else 0
            episodes_count = len(list((data_dir / "episodes").glob("*.npz"))) if (data_dir / "episodes").exists() else 0
            
            # 진행률 계산 (이전 단계 대비)
            if raw_count > 0:
                self.progress_bars["download"].setValue(100)
                self.progress_bars["extract"].setValue(min(100, int(poses_count * 100 / raw_count)))
            
            if poses_count > 0:
                self.progress_bars["filter"].setValue(min(100, int(filtered_count * 100 / poses_count)))
            
            if filtered_count > 0:
                self.progress_bars["encode"].setValue(min(100, int(episodes_count * 100 / filtered_count)))
            
            # 전체 진행률
            total_pct = sum(pb.value() for pb in self.progress_bars.values()) // len(self.progress_bars)
            self.total_progress.setValue(total_pct)
            
        except Exception:
            pass
    
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
