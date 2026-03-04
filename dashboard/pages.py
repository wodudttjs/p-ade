"""
페이지 뷰

대시보드 각 페이지 위젯
"""

from typing import Optional, Callable
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTableView, QGroupBox, QTextEdit,
    QSplitter, QGridLayout, QScrollArea, QFrame, QTabWidget,
    QHeaderView, QProgressBar, QTableWidget, QTableWidgetItem
)

from dashboard.models import (
    STAGES, STATUSES, JobRow,
    make_mock_kpi, make_mock_quality, make_mock_system
)

# DataService import (실제 DB 연동)
try:
    from dashboard.data_service import get_data_service, DataService
    HAS_DATA_SERVICE = True
except ImportError:
    HAS_DATA_SERVICE = False
from dashboard.table_models import JobsTableModel
from dashboard.widgets import (
    KPICard, ProgressCard, ResourceMeter, 
    SectionHeader, Separator
)
from dashboard.styles import Colors


class OverviewPage(QWidget):
    """Overview 페이지"""
    
    def __init__(self, data_service: Optional['DataService'] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_service = data_service
        self._use_real_data = data_service is not None and data_service.is_connected()
        self._setup_ui()
        self._load_data()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)
        
        # 헤더
        header = SectionHeader("개요", "파이프라인 전체 현황")
        content_layout.addWidget(header)
        
        # ── 파이프라인 진행률 바 (5K 스케일) ──
        progress_group = QGroupBox("파이프라인 진행률")
        progress_layout = QVBoxLayout(progress_group)
        
        self.pipeline_progress_bar = QProgressBar()
        self.pipeline_progress_bar.setRange(0, 5000)
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setFormat("%v / 5,000 (%p%)")
        self.pipeline_progress_bar.setMinimumHeight(28)
        progress_layout.addWidget(self.pipeline_progress_bar)
        
        self.pipeline_stage_label = QLabel("대기 중")
        self.pipeline_stage_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 13px;")
        progress_layout.addWidget(self.pipeline_stage_label)
        
        content_layout.addWidget(progress_group)
        
        # KPI 그리드
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(15)
        
        self.kpi_cards = {}
        kpis = [
            ("total_videos", "전체 비디오", Colors.ACCENT_BLUE),
            ("downloaded", "다운로드됨", Colors.ACCENT_GREEN),
            ("episodes", "에피소드", Colors.ACCENT_PURPLE),
            ("high_quality", "고품질", Colors.SUCCESS),
            ("storage_gb", "저장소 (GB)", Colors.ACCENT_YELLOW),
            ("monthly_cost", "월간 비용 (원)", Colors.WARNING),
        ]
        
        for i, (key, title, color) in enumerate(kpis):
            card = KPICard(title, "—", color=color)
            kpi_grid.addWidget(card, i // 3, i % 3)
            self.kpi_cards[key] = card
        
        content_layout.addLayout(kpi_grid)
        
        # 구분선
        content_layout.addWidget(Separator())
        
        # 처리 통계
        stats_header = SectionHeader("처리 통계", "처리 성능 지표")
        content_layout.addWidget(stats_header)
        
        stats_grid = QGridLayout()
        stats_grid.setSpacing(15)
        
        self.stats_cards = {}
        stats = [
            ("success_rate", "성공률", Colors.SUCCESS),
            ("avg_time", "평균 처리 시간", Colors.ACCENT_BLUE),
            ("queue_depth", "큐 깊이", Colors.ACCENT_PURPLE),
            ("active_workers", "활성 작업자", Colors.ACCENT_GREEN),
        ]
        
        for i, (key, title, color) in enumerate(stats):
            card = KPICard(title, "—", color=color)
            stats_grid.addWidget(card, 0, i)
            self.stats_cards[key] = card
        
        content_layout.addLayout(stats_grid)
        
        # 구분선
        content_layout.addWidget(Separator())
        
        # 시스템 리소스
        resource_header = SectionHeader("시스템 리소스", "리소스 사용량")
        content_layout.addWidget(resource_header)
        
        resource_grid = QGridLayout()
        resource_grid.setSpacing(10)
        
        self.resource_meters = {}
        resources = [
            ("cpu", "CPU"),
            ("memory", "메모리"),
            ("disk", "디스크"),
            ("gpu", "GPU"),
        ]
        
        for i, (key, title) in enumerate(resources):
            meter = ResourceMeter(title, max_value=100, unit="%")
            resource_grid.addWidget(meter, 0, i)
            self.resource_meters[key] = meter
        
        content_layout.addLayout(resource_grid)
        
        # 품질 분포
        content_layout.addWidget(Separator())
        quality_header = SectionHeader("품질 분포", "품질 메트릭 분포")
        content_layout.addWidget(quality_header)
        
        quality_grid = QGridLayout()
        quality_grid.setSpacing(15)
        
        self.quality_cards = {}
        quality_items = [
            ("pass_rate", "합격률", Colors.SUCCESS),
            ("confidence", "신뢰도 (평균)", Colors.ACCENT_BLUE),
            ("jitter", "지터 (평균)", Colors.ACCENT_PURPLE),
            ("nan_ratio", "NaN 비율", Colors.WARNING),
        ]
        
        for i, (key, title, color) in enumerate(quality_items):
            card = KPICard(title, "—", color=color)
            quality_grid.addWidget(card, 0, i)
            self.quality_cards[key] = card
        
        content_layout.addLayout(quality_grid)
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def _load_data(self):
        """데이터 로드"""
        self.refresh()
    
    def refresh(self):
        """데이터 새로고침"""
        if self._use_real_data and self._data_service:
            kpi = self._data_service.get_kpi()
            quality = self._data_service.get_quality_stats()
            system = self._data_service.get_system_stats()
        else:
            kpi = make_mock_kpi()
            quality = make_mock_quality()
            system = make_mock_system()
        
        # KPI 업데이트
        self.kpi_cards["total_videos"].setValue(str(kpi.total_videos))
        self.kpi_cards["downloaded"].setValue(str(kpi.downloaded))
        self.kpi_cards["episodes"].setValue(f"{kpi.episodes:,}")
        self.kpi_cards["high_quality"].setValue(f"{kpi.high_quality:,}")
        self.kpi_cards["storage_gb"].setValue(f"{kpi.storage_gb:.1f}")
        self.kpi_cards["monthly_cost"].setValue(f"${kpi.monthly_cost:.2f}")
        
        # 통계 업데이트
        self.stats_cards["success_rate"].setValue(f"{kpi.success_rate*100:.1f}%")
        self.stats_cards["avg_time"].setValue(f"{kpi.avg_processing_time_sec:.1f}s")
        self.stats_cards["queue_depth"].setValue(str(kpi.queue_depth))
        self.stats_cards["active_workers"].setValue(str(kpi.active_workers))
        
        # 리소스 업데이트
        self.resource_meters["cpu"].setValue(system.cpu_percent)
        self.resource_meters["memory"].setValue(system.memory_percent)
        self.resource_meters["disk"].setValue(system.disk_percent)
        self.resource_meters["gpu"].setValue(system.gpu_util_percent)
        
        # 품질 업데이트
        self.quality_cards["pass_rate"].setValue(f"{quality.pass_rate*100:.1f}%")
        self.quality_cards["confidence"].setValue(f"{quality.confidence_mean:.2f}")
        self.quality_cards["jitter"].setValue(f"{quality.jitter_mean:.3f}")
        self.quality_cards["nan_ratio"].setValue(f"{quality.nan_ratio_mean:.3f}")


class JobsPage(QWidget):
    """Jobs 페이지"""
    
    jobSelected = Signal(object)  # JobRow
    
    def __init__(self, model: JobsTableModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = model
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 헤더
        header = SectionHeader("작업", "작업 목록 및 상태")
        layout.addWidget(header)
        
        # 필터 박스
        filter_box = QGroupBox("필터")
        filter_layout = QHBoxLayout(filter_box)
        
        # 검색
        filter_layout.addWidget(QLabel("검색:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("job_key, video_id, episode_id, error...")
        self.search_input.setMinimumWidth(200)
        filter_layout.addWidget(self.search_input, 3)
        
        # Stage 필터
        filter_layout.addWidget(QLabel("단계:"))
        self.stage_combo = QComboBox()
        self.stage_combo.addItems(["all"] + STAGES)
        filter_layout.addWidget(self.stage_combo, 1)
        
        # Status 필터
        filter_layout.addWidget(QLabel("상태:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["all"] + STATUSES)
        filter_layout.addWidget(self.status_combo, 1)
        
        # 버튼
        self.btn_apply = QPushButton("적용")
        self.btn_apply.setObjectName("primary")
        filter_layout.addWidget(self.btn_apply)
        
        self.btn_clear = QPushButton("초기화")
        filter_layout.addWidget(self.btn_clear)
        
        layout.addWidget(filter_box)
        
        # 스플리터: 테이블 + 상세 패널
        splitter = QSplitter(Qt.Horizontal)
        
        # 테이블
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setShowGrid(False)
        
        # 컬럼 너비 설정
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, self.model.columnCount()):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        table_layout.addWidget(self.table)
        
        # 통계 바
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("총합: 0")
        self.stats_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        self.btn_refresh = QPushButton("🔄 새로고침")
        stats_layout.addWidget(self.btn_refresh)
        
        table_layout.addLayout(stats_layout)
        
        splitter.addWidget(table_widget)
        
        # 상세 패널
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        
        self.detail_title = QLabel("작업 상세")
        self.detail_title.setStyleSheet("font-size: 14px; font-weight: 700;")
        detail_layout.addWidget(self.detail_title)
        
        self.detail_info = QLabel("작업을 선택하면 상세 정보가 표시됩니다")
        self.detail_info.setWordWrap(True)
        self.detail_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        detail_layout.addWidget(self.detail_info)
        
        # 탭: Logs / Metrics
        self.detail_tabs = QTabWidget()
        
        # Logs 탭
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setPlaceholderText("Logs will appear here...")
        self.detail_tabs.addTab(self.logs_text, "로그")
        
        # Metrics 탭
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        self.detail_tabs.addTab(metrics_widget, "메트릭")
        
        detail_layout.addWidget(self.detail_tabs)
        
        # 액션 버튼
        action_layout = QHBoxLayout()
        self.btn_retry = QPushButton("🔁 재시도")
        self.btn_retry.setEnabled(False)
        action_layout.addWidget(self.btn_retry)
        
        self.btn_cancel = QPushButton("⏹ 취소")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        action_layout.addWidget(self.btn_cancel)
        
        action_layout.addStretch()
        detail_layout.addLayout(action_layout)
        
        splitter.addWidget(detail_widget)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        
        self._update_stats()
    
    def _connect_signals(self):
        """시그널 연결"""
        self.btn_apply.clicked.connect(self._apply_filter)
        self.btn_clear.clicked.connect(self._clear_filter)
        self.btn_refresh.clicked.connect(self._refresh_data)
        self.search_input.returnPressed.connect(self._apply_filter)
        
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
    
    def _apply_filter(self):
        """필터 적용"""
        query = self.search_input.text()
        stage = self.stage_combo.currentText()
        status = self.status_combo.currentText()
        self.model.setFilter(query, stage, status)
        self._update_stats()
    
    def _clear_filter(self):
        """필터 초기화"""
        self.search_input.clear()
        self.stage_combo.setCurrentIndex(0)
        self.status_combo.setCurrentIndex(0)
        self.model.setFilter("", "all", "all")
        self._update_stats()
    
    def _refresh_data(self):
        """데이터 새로고침"""
        from dashboard.models import make_mock_jobs
        self.model.replaceAll(make_mock_jobs())
        self._update_stats()
    
    def _update_stats(self):
        """통계 업데이트"""
        stats = self.model.getStats()
        self.stats_label.setText(
            f"총합: {stats['total']} | "
            f"✓ {stats.get('success', 0)} | "
            f"▶ {stats.get('running', 0)} | "
            f"✗ {stats.get('fail', 0)} | "
            f"⊘ {stats.get('skip', 0)}"
        )
    
    def _on_selection_changed(self, selected, deselected):
        """선택 변경 처리"""
        indexes = selected.indexes()
        if not indexes:
            return
        
        row_idx = indexes[0].row()
        job = self.model.rowAt(row_idx)
        
        if job:
            self._show_job_detail(job)
            self.jobSelected.emit(job)
    
    def _show_job_detail(self, job: JobRow):
        """작업 상세 정보 표시"""
        self.detail_title.setText(f"Job: {job.job_key}")
        
        info_lines = [
            f"<b>Stage:</b> {job.stage.upper()}",
            f"<b>Status:</b> {job.status.upper()}",
            f"<b>Video ID:</b> {job.video_id or '—'}",
            f"<b>Episode ID:</b> {job.episode_id or '—'}",
            f"<b>Started:</b> {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"<b>Duration:</b> {job.duration_ms//1000 if job.duration_ms else '—'}s",
        ]
        
        if job.error_type:
            info_lines.append(f"<b>Error:</b> <span style='color: {Colors.ERROR}'>{job.error_type}</span>")
            info_lines.append(f"<b>Retries:</b> {job.retry_count}")
        
        self.detail_info.setText("<br>".join(info_lines))
        
        # Logs
        self.logs_text.setText(job.log_snippet or "No logs available")
        
        # Metrics
        metrics_lines = [
            f"Processing Time: {job.duration_ms or 0}ms",
            f"Retry Count: {job.retry_count}",
            f"Run ID: {job.run_id}",
        ]
        self.metrics_text.setText("\n".join(metrics_lines))
        
        # 버튼 활성화
        self.btn_retry.setEnabled(job.status == "fail")
        self.btn_cancel.setEnabled(job.status == "running")


class QualityPage(QWidget):
    """Quality 페이지"""
    
    def __init__(self, data_service: Optional['DataService'] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_service = data_service
        self._use_real_data = data_service is not None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        header = SectionHeader("품질", "데이터 품질 상세 분석")
        layout.addWidget(header)
        
        # 품질 요약
        summary_box = QGroupBox("품질 요약")
        summary_layout = QGridLayout(summary_box)
        
        self.quality_labels = {}
        labels = [
            ("total", "전체 에피소드"),
            ("passed", "합격"),
            ("failed", "불합격"),
            ("pass_rate", "합격률"),
        ]
        
        for i, (key, title) in enumerate(labels):
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
            summary_layout.addWidget(title_lbl, 0, i)
            
            value_lbl = QLabel("—")
            value_lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
            summary_layout.addWidget(value_lbl, 1, i)
            self.quality_labels[key] = value_lbl
        
        layout.addWidget(summary_box)
        
        # 상세 메트릭
        metrics_box = QGroupBox("상세 메트릭")
        metrics_layout = QGridLayout(metrics_box)
        
        metric_items = [
            ("신뢰도 평균", "0.85"),
            ("신뢰도 표준편차", "0.08"),
            ("지터 평균", "0.12"),
            ("지터 P95", "0.25"),
            ("에피소드 길이 평균", "120"),
            ("NaN 비율 평균", "0.02"),
        ]
        
        for i, (name, value) in enumerate(metric_items):
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
            metrics_layout.addWidget(name_lbl, i // 3, (i % 3) * 2)
            
            val_lbl = QLabel(value)
            val_lbl.setStyleSheet("font-weight: 600;")
            metrics_layout.addWidget(val_lbl, i // 3, (i % 3) * 2 + 1)
        
        layout.addWidget(metrics_box)
        
        # 플레이스홀더: 차트 영역
        chart_box = QGroupBox("분포 차트")
        chart_layout = QVBoxLayout(chart_box)
        
        placeholder = QLabel("📊 차트는 pyqtgraph 또는 matplotlib로 렌더링됩니다")
        placeholder.setStyleSheet(f"color: {Colors.TEXT_MUTED}; padding: 40px;")
        placeholder.setAlignment(Qt.AlignCenter)
        chart_layout.addWidget(placeholder)
        
        layout.addWidget(chart_box)
        layout.addStretch()
        
        self.refresh()
    
    def refresh(self):
        """데이터 새로고침"""
        if self._use_real_data and self._data_service:
            quality = self._data_service.get_quality_stats()
        else:
            quality = make_mock_quality()
        
        self.quality_labels["total"].setText(str(quality.total_episodes))
        self.quality_labels["passed"].setText(str(quality.passed))
        self.quality_labels["failed"].setText(str(quality.failed))
        self.quality_labels["pass_rate"].setText(f"{quality.pass_rate*100:.1f}%")


class SettingsPage(QWidget):
    """Settings 페이지"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        header = SectionHeader("설정", "대시보드 설정")
        layout.addWidget(header)
        
        # 새로고침 간격
        refresh_box = QGroupBox("자동 새로고침")
        refresh_layout = QHBoxLayout(refresh_box)
        
        refresh_layout.addWidget(QLabel("간격:"))
        self.refresh_combo = QComboBox()
        self.refresh_combo.addItems(["Off", "5s", "10s", "30s", "1m", "5m"])
        refresh_layout.addWidget(self.refresh_combo)
        refresh_layout.addStretch()
        
        layout.addWidget(refresh_box)
        
        # 테마
        theme_box = QGroupBox("테마")
        theme_layout = QHBoxLayout(theme_box)
        
        theme_layout.addWidget(QLabel("색상 테마:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        
        layout.addWidget(theme_box)
        
        # 알림
        alert_box = QGroupBox("알림")
        alert_layout = QVBoxLayout(alert_box)
        
        alert_layout.addWidget(QLabel("알림 설정이 여기에 표시됩니다..."))
        
        layout.addWidget(alert_box)
        
        layout.addStretch()


# ============================================================
# Runs 페이지 (D-2: 크로스-런 모니터링)
# ============================================================

class RunsPage(QWidget):
    """파이프라인 실행 이력 페이지"""
    
    def __init__(self, data_service: Optional['DataService'] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data_service = data_service
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        header = SectionHeader("실행 이력", "파이프라인 실행 기록")
        layout.addWidget(header)
        
        # 실행 이력 테이블
        self.runs_table = QTableWidget()
        self.runs_table.setColumnCount(8)
        self.runs_table.setHorizontalHeaderLabels([
            "Run ID", "시작", "완료", "목표", "크롤링", "처리", "통과", "상태"
        ])
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_table.setAlternatingRowColors(True)
        self.runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.runs_table)
        
        # 새로고침 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        refresh_btn = QPushButton("새로고침")
        refresh_btn.clicked.connect(self.refresh)
        btn_layout.addWidget(refresh_btn)
        layout.addLayout(btn_layout)
    
    def refresh(self):
        """실행 이력 로드 (API /api/runs 호출)"""
        try:
            import urllib.request
            import json
            resp = urllib.request.urlopen("http://localhost:5000/api/runs?per_page=50")
            data = json.loads(resp.read().decode())
            runs = data.get("items", [])
            
            self.runs_table.setRowCount(len(runs))
            for i, run in enumerate(runs):
                self.runs_table.setItem(i, 0, QTableWidgetItem(str(run.get("run_id", ""))))
                self.runs_table.setItem(i, 1, QTableWidgetItem(str(run.get("started_at", ""))))
                self.runs_table.setItem(i, 2, QTableWidgetItem(str(run.get("completed_at", ""))))
                self.runs_table.setItem(i, 3, QTableWidgetItem(str(run.get("target_count", ""))))
                self.runs_table.setItem(i, 4, QTableWidgetItem(str(run.get("crawled", ""))))
                self.runs_table.setItem(i, 5, QTableWidgetItem(str(run.get("processed", ""))))
                self.runs_table.setItem(i, 6, QTableWidgetItem(str(run.get("passed", ""))))
                self.runs_table.setItem(i, 7, QTableWidgetItem(str(run.get("status", ""))))
        except Exception:
            pass


# ============================================================
# Dedup 위젯 (D-2: 중복 방지 통계)
# ============================================================

class DedupStatsWidget(QWidget):
    """중복 방지 통계 위젯"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        header = SectionHeader("중복 방지 통계", "cross-run 중복 방지 현황")
        layout.addWidget(header)
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        self.dedup_cards = {}
        items = [
            ("total_collected", "총 수집", Colors.ACCENT_BLUE),
            ("unique_videos", "고유 영상", Colors.ACCENT_GREEN),
            ("duplicate_blocked", "중복 차단", Colors.WARNING),
            ("duplicate_rate", "중복률 (%)", Colors.DANGER if hasattr(Colors, 'DANGER') else Colors.WARNING),
        ]
        
        for i, (key, title, color) in enumerate(items):
            card = KPICard(title, "—", color=color)
            grid.addWidget(card, 0, i)
            self.dedup_cards[key] = card
        
        layout.addLayout(grid)
    
    def refresh(self):
        """중복 방지 통계 로드 (API /api/dedup/stats 호출)"""
        try:
            import urllib.request
            import json
            resp = urllib.request.urlopen("http://localhost:5000/api/dedup/stats")
            data = json.loads(resp.read().decode())
            
            self.dedup_cards["total_collected"].setValue(str(data.get("total_collected", 0)))
            self.dedup_cards["unique_videos"].setValue(str(data.get("unique_videos", 0)))
            self.dedup_cards["duplicate_blocked"].setValue(str(data.get("duplicate_blocked", 0)))
            rate = data.get("duplicate_rate", 0)
            self.dedup_cards["duplicate_rate"].setValue(f"{rate:.1f}%")
        except Exception:
            pass
