"""
대시보드 스타일 정의

Qt 위젯용 테마, 색상 상수, 상태별 색상 매핑 등을 제공합니다.
"""


class Colors:
    """앱 전체에서 사용하는 색상 상수"""

    # 배경
    BG_DARK = "#1e1e2e"
    BG_MEDIUM = "#2b2b3d"
    BG_LIGHT = "#363649"
    BG_CARD = "#303044"

    # 텍스트
    TEXT_PRIMARY = "#cdd6f4"
    TEXT_SECONDARY = "#a6adc8"
    TEXT_MUTED = "#6c7086"

    # 액센트
    ACCENT_BLUE = "#89b4fa"
    ACCENT_PURPLE = "#cba6f7"
    ACCENT_TEAL = "#94e2d5"

    # 시맨틱
    SUCCESS = "#a6e3a1"
    ERROR = "#f38ba8"
    WARNING = "#f9e2af"
    INFO = "#89dceb"
    RUNNING = "#74c7ec"

    # 보더 / 구분선
    BORDER = "#45475a"
    SEPARATOR = "#585b70"


def get_status_color(status: str) -> str:
    """상태 문자열에 대응하는 색상을 반환합니다.

    Args:
        status: 상태 문자열 (success, fail, error, running, skip, pending 등)

    Returns:
        CSS 색상 문자열
    """
    mapping = {
        "success": Colors.SUCCESS,
        "completed": Colors.SUCCESS,
        "done": Colors.SUCCESS,
        "fail": Colors.ERROR,
        "failed": Colors.ERROR,
        "error": Colors.ERROR,
        "running": Colors.RUNNING,
        "processing": Colors.RUNNING,
        "in_progress": Colors.RUNNING,
        "skip": Colors.WARNING,
        "skipped": Colors.WARNING,
        "warning": Colors.WARNING,
        "pending": Colors.TEXT_MUTED,
        "queued": Colors.TEXT_MUTED,
        "waiting": Colors.TEXT_MUTED,
        "info": Colors.INFO,
    }
    return mapping.get(status.lower(), Colors.TEXT_SECONDARY)


# ---------------------------------------------------------------------------
# Qt StyleSheet 테마
# ---------------------------------------------------------------------------

DARK_THEME = """
QMainWindow {
    background-color: %(bg_dark)s;
    color: %(text_primary)s;
}
QWidget {
    background-color: %(bg_dark)s;
    color: %(text_primary)s;
    font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: %(accent_blue)s;
}
QLabel {
    background: transparent;
}
QPushButton {
    background-color: %(accent_blue)s;
    color: %(bg_dark)s;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: %(accent_purple)s;
}
QPushButton:pressed {
    background-color: %(accent_teal)s;
}
QProgressBar {
    background-color: %(bg_light)s;
    border: 1px solid %(border)s;
    border-radius: 4px;
    text-align: center;
    color: %(text_primary)s;
    height: 18px;
}
QProgressBar::chunk {
    background-color: %(accent_blue)s;
    border-radius: 3px;
}
QTableView {
    background-color: %(bg_medium)s;
    alternate-background-color: %(bg_light)s;
    gridline-color: %(border)s;
    selection-background-color: %(accent_blue)s;
    selection-color: %(bg_dark)s;
    border: 1px solid %(border)s;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: %(bg_card)s;
    color: %(text_secondary)s;
    border: 1px solid %(border)s;
    padding: 6px;
    font-weight: bold;
}
QTabWidget::pane {
    border: 1px solid %(border)s;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: %(bg_medium)s;
    color: %(text_secondary)s;
    border: 1px solid %(border)s;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: %(bg_card)s;
    color: %(accent_blue)s;
    border-bottom: 2px solid %(accent_blue)s;
}
QScrollBar:vertical {
    background-color: %(bg_dark)s;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: %(border)s;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: %(text_muted)s;
}
""" % {
    "bg_dark": Colors.BG_DARK,
    "bg_medium": Colors.BG_MEDIUM,
    "bg_light": Colors.BG_LIGHT,
    "bg_card": Colors.BG_CARD,
    "text_primary": Colors.TEXT_PRIMARY,
    "text_secondary": Colors.TEXT_SECONDARY,
    "text_muted": Colors.TEXT_MUTED,
    "accent_blue": Colors.ACCENT_BLUE,
    "accent_purple": Colors.ACCENT_PURPLE,
    "accent_teal": Colors.ACCENT_TEAL,
    "border": Colors.BORDER,
}

LIGHT_THEME = """
QMainWindow {
    background-color: #f5f5f5;
    color: #1e1e2e;
}
QWidget {
    background-color: #f5f5f5;
    color: #1e1e2e;
    font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dce0e8;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #1e66f5;
}
QLabel {
    background: transparent;
}
QPushButton {
    background-color: #1e66f5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #7287fd;
}
QPushButton:pressed {
    background-color: #179299;
}
QProgressBar {
    background-color: #e6e9ef;
    border: 1px solid #dce0e8;
    border-radius: 4px;
    text-align: center;
    color: #1e1e2e;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #1e66f5;
    border-radius: 3px;
}
QTableView {
    background-color: #ffffff;
    alternate-background-color: #eff1f5;
    gridline-color: #dce0e8;
    selection-background-color: #1e66f5;
    selection-color: #ffffff;
    border: 1px solid #dce0e8;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #e6e9ef;
    color: #4c4f69;
    border: 1px solid #dce0e8;
    padding: 6px;
    font-weight: bold;
}
QTabWidget::pane {
    border: 1px solid #dce0e8;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #e6e9ef;
    color: #4c4f69;
    border: 1px solid #dce0e8;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1e66f5;
    border-bottom: 2px solid #1e66f5;
}
QScrollBar:vertical {
    background-color: #f5f5f5;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #bcc0cc;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #9ca0b0;
}
"""

__all__ = [
    "Colors",
    "get_status_color",
    "DARK_THEME",
    "LIGHT_THEME",
]
