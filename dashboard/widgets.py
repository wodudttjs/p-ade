"""
위젯 컴포넌트

재사용 가능한 UI 위젯들
"""

from typing import Optional, Dict, Any, Callable
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QFrame, QProgressBar, QGridLayout
)

from dashboard.styles import Colors


class KPICard(QGroupBox):
    """KPI 카드 위젯"""
    
    def __init__(
        self, 
        title: str, 
        value: str = "0", 
        subtitle: str = "",
        color: str = Colors.ACCENT_BLUE,
        parent: Optional[QWidget] = None
    ):
        super().__init__(title, parent)
        self.setMinimumWidth(140)
        self.setMaximumHeight(120)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # 값
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: 700;
            color: {color};
        """)
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
        # 부제목
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
            self.subtitle_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.subtitle_label)
        else:
            self.subtitle_label = None
        
        layout.addStretch()
    
    def setValue(self, value: str):
        """값 업데이트"""
        self.value_label.setText(value)
    
    def setSubtitle(self, subtitle: str):
        """부제목 업데이트"""
        if self.subtitle_label:
            self.subtitle_label.setText(subtitle)


class StatusBar(QWidget):
    """상태 표시줄 위젯"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 상태 표시등
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 14px;")
        layout.addWidget(self.status_dot)
        
        # 상태 텍스트
        self.status_text = QLabel("Connected")
        self.status_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        
        # 타임스탬프
        self.timestamp = QLabel("")
        self.timestamp.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        layout.addWidget(self.timestamp)
    
    def setStatus(self, connected: bool, message: str = ""):
        """상태 업데이트"""
        if connected:
            self.status_dot.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 14px;")
            self.status_text.setText(message or "Connected")
        else:
            self.status_dot.setStyleSheet(f"color: {Colors.ERROR}; font-size: 14px;")
            self.status_text.setText(message or "Disconnected")
    
    def setConnected(self, connected: bool):
        """연결 상태 설정 (setStatus 호환)"""
        self.setStatus(connected)
    
    def setTimestamp(self, timestamp: str):
        """타임스탬프 업데이트"""
        self.timestamp.setText(f"Last updated: {timestamp}")


class ProgressCard(QWidget):
    """진행률 카드 위젯"""
    
    def __init__(
        self, 
        title: str,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # 헤더
        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label)
        
        self.value_label = QLabel("0%")
        self.value_label.setStyleSheet(f"color: {Colors.ACCENT_BLUE}; font-weight: 600;")
        header.addWidget(self.value_label)
        
        layout.addLayout(header)
        
        # 프로그레스바
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        layout.addWidget(self.progress)
    
    def setValue(self, value: int, label: str = None):
        """값 업데이트"""
        self.progress.setValue(value)
        self.value_label.setText(label or f"{value}%")


class ResourceMeter(QWidget):
    """리소스 사용량 미터 위젯"""
    
    def __init__(
        self, 
        title: str,
        max_value: float = 100.0,
        unit: str = "%",
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.max_value = max_value
        self.unit = unit
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # 헤더
        header = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 11px;")
        header.addWidget(self.title_label)
        
        header.addStretch()
        
        self.value_label = QLabel(f"0{unit}")
        self.value_label.setStyleSheet(f"font-size: 11px; color: {Colors.ACCENT_BLUE};")
        header.addWidget(self.value_label)
        
        layout.addLayout(header)
        
        # 프로그레스바
        self.progress = QProgressBar()
        self.progress.setMaximum(int(max_value))
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(6)
        layout.addWidget(self.progress)
    
    def setValue(self, value: float):
        """값 업데이트"""
        self.progress.setValue(int(value))
        self.value_label.setText(f"{value:.1f}{self.unit}")
        
        # 임계값에 따른 색상 변경
        if value >= self.critical_threshold:
            color = Colors.ERROR
        elif value >= self.warning_threshold:
            color = Colors.WARNING
        else:
            color = Colors.ACCENT_BLUE
        
        self.value_label.setStyleSheet(f"font-size: 11px; color: {color};")
        self.progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)


class SectionHeader(QWidget):
    """섹션 헤더 위젯"""
    
    def __init__(
        self, 
        title: str,
        subtitle: str = "",
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(2)
        
        self.title = QLabel(title)
        self.title.setStyleSheet("""
            font-size: 18px;
            font-weight: 600;
        """)
        layout.addWidget(self.title)
        
        if subtitle:
            self.subtitle = QLabel(subtitle)
            self.subtitle.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px;")
            layout.addWidget(self.subtitle)


class Separator(QFrame):
    """구분선 위젯"""
    
    def __init__(self, horizontal: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if horizontal:
            self.setFrameShape(QFrame.HLine)
        else:
            self.setFrameShape(QFrame.VLine)
        self.setStyleSheet(f"background-color: {Colors.BG_OVERLAY};")
