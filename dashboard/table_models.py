"""
테이블 모델

Qt 테이블 뷰용 모델 클래스들
"""

from typing import List, Optional, Any
from datetime import datetime

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QColor, QBrush

from dashboard.models import JobRow, STAGES, STATUSES
from dashboard.styles import get_status_color


class JobsTableModel(QAbstractTableModel):
    """작업 테이블 모델"""
    
    HEADERS = [
        "job_key", "stage", "status", "error_type", 
        "started_at", "duration", "video_id", "episode_id"
    ]
    
    HEADER_DISPLAY = {
        "job_key": "Job Key",
        "stage": "Stage",
        "status": "Status",
        "error_type": "Error Type",
        "started_at": "Started At",
        "duration": "Duration",
        "video_id": "Video ID",
        "episode_id": "Episode ID",
    }

    def __init__(self, rows: Optional[List[JobRow]] = None):
        super().__init__()
        self._all = rows or []
        self._view = self._all[:]
        
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._view)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(
        self, 
        section: int, 
        orientation: Qt.Orientation, 
        role: int = Qt.DisplayRole
    ) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            col = self.HEADERS[section]
            return self.HEADER_DISPLAY.get(col, col)
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = self._view[index.row()]
        col = self.HEADERS[index.column()]

        if role == Qt.DisplayRole:
            return self._get_display_value(row, col)

        if role == Qt.TextAlignmentRole:
            if col in ("stage", "status", "duration"):
                return Qt.AlignCenter
            return Qt.AlignVCenter | Qt.AlignLeft

        if role == Qt.ForegroundRole:
            if col == "status":
                color_hex = get_status_color(row.status)
                return QColor(color_hex)
            if col == "error_type" and row.error_type:
                return QColor("#f38ba8")

        if role == Qt.BackgroundRole:
            if row.status == "running":
                return QBrush(QColor("#1e3a5f"))

        if role == Qt.ToolTipRole:
            if col == "job_key":
                return row.job_key
            if col == "error_type" and row.error_type:
                return f"Error: {row.error_type}\nRetries: {row.retry_count}"

        return None
    
    def _get_display_value(self, row: JobRow, col: str) -> str:
        """컬럼별 표시 값 반환"""
        if col == "job_key":
            # 긴 키는 축약
            if len(row.job_key) > 40:
                return row.job_key[:37] + "..."
            return row.job_key
        if col == "stage":
            return row.stage.upper()
        if col == "status":
            return row.status.upper()
        if col == "error_type":
            return row.error_type or "—"
        if col == "started_at":
            return row.started_at.strftime("%Y-%m-%d %H:%M:%S")
        if col == "duration":
            if row.duration_ms is None:
                return "..."
            secs = row.duration_ms // 1000
            if secs < 60:
                return f"{secs}s"
            mins = secs // 60
            secs = secs % 60
            return f"{mins}m {secs}s"
        if col == "video_id":
            return row.video_id or "—"
        if col == "episode_id":
            return row.episode_id or "—"
        return ""

    def setFilter(self, query: str, stage: str, status: str):
        """필터 적용"""
        q = query.strip().lower()
        self.beginResetModel()
        self._view = []
        
        for r in self._all:
            # Stage 필터
            if stage != "all" and r.stage != stage:
                continue
            # Status 필터
            if status != "all" and r.status != status:
                continue
            # 검색어 필터
            if q:
                searchable = " ".join([
                    r.job_key, r.run_id, r.stage, r.status,
                    r.error_type or "", r.video_id or "", r.episode_id or ""
                ]).lower()
                if q not in searchable:
                    continue
            self._view.append(r)
            
        self.endResetModel()

    def rowAt(self, row_idx: int) -> Optional[JobRow]:
        """특정 행의 데이터 반환"""
        if 0 <= row_idx < len(self._view):
            return self._view[row_idx]
        return None

    def replaceAll(self, new_rows: List[JobRow]):
        """전체 데이터 교체"""
        self.beginResetModel()
        self._all = new_rows
        self._view = new_rows[:]
        self.endResetModel()
    
    def appendRow(self, row: JobRow):
        """행 추가"""
        self.beginInsertRows(QModelIndex(), len(self._all), len(self._all))
        self._all.append(row)
        self._view.append(row)
        self.endInsertRows()
    
    def getStats(self) -> dict:
        """통계 반환"""
        stats = {s: 0 for s in STATUSES}
        for row in self._all:
            if row.status in stats:
                stats[row.status] += 1
        stats["total"] = len(self._all)
        return stats
