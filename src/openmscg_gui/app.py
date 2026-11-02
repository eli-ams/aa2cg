from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import APP_NAME, __version__
from .main_window import MainWindow


STYLE = """
QWidget {
    color: #172321;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
}
QMainWindow { background: #f5f7f9; }
QWidget#workflowScrollContent, QScrollArea { background: #f8fafb; }
QFrame#systemBar {
    background: #f5f7f6;
    border-bottom: 1px solid #d9e0de;
}
QFrame#speciesPreview {
    background: #ffffff;
    border: 1px solid #d9e0de;
    border-radius: 8px;
}
QWidget#stagePreviewPane, QWidget#stageControlsPane { background: transparent; }
QWidget#stageHeader {
    background: #ffffff;
    border-bottom: 1px solid #e2e8ed;
}
QLabel#pageTitle { font-size: 16pt; font-weight: 650; color: #172f37; }
QWidget#stageHeader QLabel#pageTitle { font-size: 13pt; }
QWidget#stageHeader QLabel#stageStatus { padding: 3px 8px; }
QWidget#stageHeader QPushButton#primaryButton { padding: 5px 10px; }
QLabel#pageDescription { color: #4e625e; font-size: 10.5pt; }
QLabel#sectionTitle { font-size: 11pt; font-weight: 650; color: #284d46; }
QLabel#note {
    background: transparent;
    border: none;
    padding: 4px 0;
    color: #667781;
}
QWidget[tutorialTarget="true"] {
    border: 2px solid #247866;
    border-radius: 5px;
}
QLabel#canvasSummary { color: #51635f; padding: 4px; }
QFrame#mappingStatusPanel {
    background: #f8fbfa;
    border: 1px solid #d9e0de;
    border-radius: 7px;
}
QLabel#tileTitle { color: #173f38; font-size: 9pt; font-weight: 650; }
QLabel#tileMeta { color: #657672; font-size: 8pt; }
QLabel#stageOutput { color: #4e625e; }
QLabel#stageStatus {
    background: #e6eae9;
    border-radius: 11px;
    padding: 4px 10px;
    font-weight: 600;
}
QLabel#stageStatus[state="complete"] { background: #d8eee7; color: #17634f; }
QLabel#stageStatus[state="pending"] { background: #e6eae9; color: #51635f; }
QLabel#stageStatus[state="running"] { background: #fff0c7; color: #785b05; }
QLabel#stageStatus[state="failed"] { background: #f9dddd; color: #8a2929; }
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5d2;
    border-radius: 6px;
    padding: 6px 11px;
}
QPushButton:hover { border-color: #699d92; background: #f8fbfa; }
QPushButton:checked { color: #173f38; background: #d8eae5; border-color: #247866; }
QPushButton:pressed { background: #e1ede9; }
QPushButton:disabled { color: #93a19e; background: #edf0ef; }
QPushButton#primaryButton {
    color: white;
    background: #247866;
    border-color: #247866;
    padding: 8px 18px;
    font-weight: 650;
}
QPushButton#primaryButton:hover { background: #1c6656; }
QPushButton#primaryButton:disabled { color: #6f7d78; background: #e4eae7; border-color: #c4ceca; }
QGroupBox {
    background: white;
    border: none;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 650;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QFrame#tableTile {
    background: #ffffff;
    border: 1px solid #d9e0de;
    border-radius: 7px;
}
QFrame#tableTile:hover { border-color: #86ada5; background: #f8fbfa; }
QFrame#tableTile[selected="true"] {
    border: 2px solid #247866;
    background: #f3faf7;
}
QLineEdit, QAbstractSpinBox, QComboBox, QListWidget, QTableWidget, QPlainTextEdit {
    background: white;
    border: 1px solid #cdd8d5;
    border-radius: 5px;
    padding: 5px;
    selection-background-color: #6ba99b;
}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    color: #6f7d78;
    background: #edf1ef;
    border-color: #c4ceca;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus { border-color: #247866; }
QComboBox { padding-right: 28px; }
QAbstractSpinBox { padding-right: 22px; }
QComboBox::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    border-left: 1px solid #c4ceca;
}
QComboBox::down-arrow {
    image: url("@THEME_ASSETS@/chevron-down.svg"); width: 12px; height: 12px;
}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    width: 18px;
    border: none;
    border-left: 1px solid #c4ceca;
    background: #edf3f1;
}
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: #d8eae5; }
QAbstractSpinBox::up-button:disabled, QAbstractSpinBox::down-button:disabled { background: #e4eae7; }
QAbstractSpinBox::up-arrow { image: url("@THEME_ASSETS@/chevron-up.svg"); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url("@THEME_ASSETS@/chevron-down.svg"); width: 10px; height: 10px; }
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f4f7f6;
    gridline-color: #dce3e1;
}
QTableWidget::item { background-color: #ffffff; }
QTableWidget::item:alternate { background-color: #f4f7f6; }
QTableWidget::item:selected { background-color: #d8eae5; color: #173f38; }
QHeaderView::section {
    background-color: #eef3f1;
    color: #284d46;
    border: 0;
    border-right: 1px solid #d5dfdc;
    border-bottom: 1px solid #cbd7d4;
    padding: 6px;
    font-weight: 600;
}
QTableCornerButton::section {
    background-color: #eef3f1;
    border: 0;
    border-right: 1px solid #d5dfdc;
    border-bottom: 1px solid #cbd7d4;
}
QCheckBox, QRadioButton { background-color: transparent; color: #172321; spacing: 7px; }
QCheckBox:disabled, QRadioButton:disabled { color: #6f7d78; }
QCheckBox::indicator, QRadioButton::indicator, QGroupBox::indicator, QAbstractItemView::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #71827b;
    border-radius: 3px;
    background: #ffffff;
    image: none;
}
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QAbstractItemView::indicator:checked {
    image: url("@THEME_ASSETS@/check.svg");
    border-color: #247866;
    background: #d8eae5;
}
QCheckBox::indicator:indeterminate, QAbstractItemView::indicator:indeterminate {
    image: url("@THEME_ASSETS@/partial.svg");
    border-color: #247866;
    background: #d8eae5;
}
QRadioButton::indicator:checked {
    image: url("@THEME_ASSETS@/radio.svg"); border-color: #247866; background: #d8eae5;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover, QGroupBox::indicator:hover,
QAbstractItemView::indicator:hover, QCheckBox::indicator:focus, QRadioButton::indicator:focus {
    border-color: #172321;
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled,
QGroupBox::indicator:disabled, QAbstractItemView::indicator:disabled {
    border-color: #a1afa8;
    background: #e9eeeb;
}
QMenu { background: #ffffff; color: #172321; border: 1px solid #c4ceca; }
QMenu::item:selected { background: #d8eae5; color: #173f38; }
QMenu::item:disabled { color: #6f7d78; }
QMenu::indicator { width: 16px; height: 16px; }
QMenu::indicator:non-exclusive:unchecked, QMenu::indicator:non-exclusive:checked,
QMenu::indicator:exclusive:unchecked, QMenu::indicator:exclusive:checked {
    border: 1px solid #71827b; border-radius: 3px; background: #ffffff; image: none;
}
QMenu::indicator:exclusive:unchecked, QMenu::indicator:exclusive:checked { border-radius: 9px; }
QMenu::indicator:non-exclusive:checked {
    image: url("@THEME_ASSETS@/check.svg"); border-color: #247866; background: #d8eae5;
}
QMenu::indicator:exclusive:checked {
    image: url("@THEME_ASSETS@/radio.svg"); border-color: #247866; background: #d8eae5;
}
QMenu::indicator:disabled { border-color: #a1afa8; background: #e9eeeb; }
QListWidget::item { padding: 6px; border-radius: 4px; }
QListWidget::item:selected { background: #d8eae5; color: #173f38; }
QTabWidget::pane { background: #f8fafb; border: none; border-top: 1px solid #e2e8ed; }
QTabBar::tab { background: transparent; color: #667781; padding: 10px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #137b70; border-bottom: 2px solid #168477; font-weight: 650; }
QTabWidget#stageSubTabs::pane {
    background: #ffffff;
    border: none;
    border-radius: 7px;
}
QTabWidget#stageSubTabs QTabBar::tab {
    background: transparent;
    padding: 7px 12px;
    margin-right: 2px;
}
QTabWidget#stageSubTabs QTabBar::tab:selected {
    background: #ffffff;
    color: #17634f;
    font-weight: 650;
}
QPlainTextEdit#console {
    background: #15211f;
    color: #d5e3df;
    border: 0;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 9pt;
}
QSplitter::handle { background: #dce3e1; width: 2px; }
QSplitter#workspaceSplitter::handle { height: 4px; }
QSplitter#stageWorkspaceSplitter::handle { width: 4px; }
QToolButton { border: 1px solid #d9e2e7; border-radius: 5px; padding: 5px 8px; background: #ffffff; }
QToolButton:hover { background: #edf7f5; border-color: #168477; }
QToolButton:checked { background: #e1f1ed; color: #17634f; }
QToolButton:disabled { color: #6f7d78; background: #edf1ef; border-color: #c4ceca; }
QSlider::groove:horizontal { height: 4px; background: #d9e2e7; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: #168477; }
QStatusBar { background: #ffffff; border-top: 1px solid #e2e8ed; color: #667781; }
""".replace('@THEME_ASSETS@', (Path(__file__).resolve().parent / 'assets' / 'theme').as_posix())


def _light_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f5f7f6"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#172321"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#f4f7f6"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#172321"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#172321"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#172321"))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#d8eae5"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#173f38"))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#7c8d89"))
    # Explicit light/shadow roles keep native details legible on dark OS themes
    # too; disabled text must remain distinguishable from the light background.
    for role, color in (
        (QtGui.QPalette.ColorRole.Light, '#ffffff'),
        (QtGui.QPalette.ColorRole.Midlight, '#e6eeeb'),
        (QtGui.QPalette.ColorRole.Mid, '#9aaba5'),
        (QtGui.QPalette.ColorRole.Dark, '#64766f'),
        (QtGui.QPalette.ColorRole.Shadow, '#172321'),
    ):
        palette.setColor(role, QtGui.QColor(color))
    for role in (QtGui.QPalette.ColorRole.WindowText, QtGui.QPalette.ColorRole.Text,
                 QtGui.QPalette.ColorRole.ButtonText, QtGui.QPalette.ColorRole.PlaceholderText):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor('#6f7d78'))
    return palette


def main() -> int:
    # Set the format before QApplication, including when ModernGL is unavailable.
    surface = QtGui.QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(surface)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("eli-ams")
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    QtCore.QTimer.singleShot(0, window.show_welcome_on_startup)
    return app.exec()



