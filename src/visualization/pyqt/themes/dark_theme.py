"""
Dark Theme for PyQt TEC Display Application
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


def apply_dark_theme(app: QApplication):
    """
    Apply a dark theme to the PyQt application.

    Args:
        app: QApplication instance
    """
    app.setStyle("Fusion")

    palette = QPalette()

    # Window background
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)

    # Base (input fields, etc.)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))

    # Text
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)

    # Buttons
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)

    # Highlights
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    # Bright text (for contrast)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)

    # Links
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))

    # Disabled colors
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))

    app.setPalette(palette)

    # Additional stylesheet for fine-tuning
    app.setStyleSheet("""
        QToolTip {
            background-color: #2a2a2a;
            color: white;
            border: 1px solid #3a3a3a;
            padding: 4px;
        }

        QGroupBox {
            border: 1px solid #3a3a3a;
            margin-top: 1ex;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }

        QStatusBar {
            background-color: #2a2a2a;
            border-top: 1px solid #3a3a3a;
        }

        QSplitter::handle {
            background-color: #3a3a3a;
        }

        QSplitter::handle:horizontal {
            width: 4px;
        }

        QSplitter::handle:vertical {
            height: 4px;
        }

        QTabWidget::pane {
            border: 1px solid #3a3a3a;
        }

        QTabBar::tab {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            padding: 6px 12px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background-color: #4a4a4a;
        }

        QScrollBar:vertical {
            background-color: #2a2a2a;
            width: 12px;
        }

        QScrollBar::handle:vertical {
            background-color: #5a5a5a;
            min-height: 20px;
            border-radius: 4px;
        }

        QScrollBar:horizontal {
            background-color: #2a2a2a;
            height: 12px;
        }

        QScrollBar::handle:horizontal {
            background-color: #5a5a5a;
            min-width: 20px;
            border-radius: 4px;
        }
    """)
