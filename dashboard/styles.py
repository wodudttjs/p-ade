"""
스타일 정의

대시보드 테마 및 스타일시트
"""

# 다크 테마 스타일시트
DARK_THEME = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 12px;
}

QLabel {
    color: #cdd6f4;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #89b4fa;
}

QPushButton {
    background-color: #45475a;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    color: #cdd6f4;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #585b70;
}

QPushButton:pressed {
    background-color: #313244;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}

QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QPushButton#primary:hover {
    background-color: #b4befe;
}

QPushButton#danger {
    background-color: #f38ba8;
    color: #1e1e2e;
}

QPushButton#success {
    background-color: #a6e3a1;
    color: #1e1e2e;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #89b4fa;
}

QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 10px;
    color: #cdd6f4;
    min-width: 80px;
}

QComboBox:hover {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #cdd6f4;
    margin-right: 5px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    border: 1px solid #45475a;
    selection-background-color: #45475a;
}

QTableView {
    background-color: #1e1e2e;
    alternate-background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    gridline-color: #313244;
    selection-background-color: #45475a;
}

QTableView::item {
    padding: 4px;
    border: none;
}

QTableView::item:selected {
    background-color: #45475a;
}

QHeaderView::section {
    background-color: #313244;
    color: #89b4fa;
    border: none;
    border-right: 1px solid #45475a;
    border-bottom: 1px solid #45475a;
    padding: 6px;
    font-weight: bold;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 6px;
    min-width: 30px;
}

QListWidget {
    background-color: #181825;
    border: none;
    border-radius: 4px;
    outline: none;
}

QListWidget::item {
    padding: 10px 15px;
    border-radius: 4px;
    margin: 2px 4px;
}

QListWidget::item:hover {
    background-color: #313244;
}

QListWidget::item:selected {
    background-color: #45475a;
    color: #89b4fa;
}

QSplitter::handle {
    background-color: #45475a;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #313244;
    text-align: center;
    color: #cdd6f4;
}

QProgressBar::chunk {
    border-radius: 4px;
    background-color: #89b4fa;
}

QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #1e1e2e;
}

QTabBar::tab {
    background-color: #313244;
    border: none;
    padding: 8px 20px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #45475a;
    color: #89b4fa;
}

QTabBar::tab:hover:!selected {
    background-color: #3c3c54;
}

QToolTip {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
"""

# 라이트 테마 스타일시트
LIGHT_THEME = """
QMainWindow {
    background-color: #eff1f5;
}

QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    font-size: 12px;
}

QLabel {
    color: #4c4f69;
}

QGroupBox {
    border: 1px solid #ccd0da;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #1e66f5;
}

QPushButton {
    background-color: #dce0e8;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    color: #4c4f69;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #ccd0da;
}

QPushButton:pressed {
    background-color: #bcc0cc;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 6px;
    color: #4c4f69;
}

QComboBox {
    background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 6px 10px;
    color: #4c4f69;
}

QTableView {
    background-color: #eff1f5;
    alternate-background-color: #e6e9ef;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    gridline-color: #ccd0da;
    selection-background-color: #dce0e8;
}

QHeaderView::section {
    background-color: #e6e9ef;
    color: #1e66f5;
    border: none;
    border-right: 1px solid #ccd0da;
    border-bottom: 1px solid #ccd0da;
    padding: 6px;
    font-weight: bold;
}

QListWidget {
    background-color: #e6e9ef;
    border: none;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #ccd0da;
    color: #1e66f5;
}
"""

# 색상 상수
class Colors:
    """색상 팔레트"""
    # 상태별 색상
    SUCCESS = "#a6e3a1"
    WARNING = "#f9e2af"
    ERROR = "#f38ba8"
    INFO = "#89b4fa"
    RUNNING = "#74c7ec"
    
    # 배경
    BG_DARK = "#1e1e2e"
    BG_MAIN = "#1e1e2e"
    BG_CARD = "#313244"
    BG_SURFACE = "#313244"
    BG_OVERLAY = "#45475a"
    
    # 테두리
    BORDER = "#45475a"
    
    # 텍스트
    TEXT_PRIMARY = "#cdd6f4"
    TEXT_SECONDARY = "#a6adc8"
    TEXT_MUTED = "#6c7086"
    
    # 강조
    ACCENT_BLUE = "#89b4fa"
    ACCENT_PURPLE = "#cba6f7"
    ACCENT_GREEN = "#a6e3a1"
    ACCENT_RED = "#f38ba8"
    ACCENT_YELLOW = "#f9e2af"


def get_status_color(status: str) -> str:
    """상태에 따른 색상 반환"""
    status_colors = {
        "success": Colors.SUCCESS,
        "fail": Colors.ERROR,
        "running": Colors.RUNNING,
        "skip": Colors.WARNING,
        "pending": Colors.TEXT_MUTED,
    }
    return status_colors.get(status.lower(), Colors.TEXT_PRIMARY)
