"""
Solar Imaging Main Window

Tabbed main window displaying all solar image sources organized by category.
"""

from datetime import datetime
from typing import Dict

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGridLayout, QLabel, QPushButton,
    QStatusBar, QToolBar, QMessageBox, QScrollArea,
    QSizePolicy
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, pyqtSlot

from .sources import SOLAR_SOURCES
from .widgets import SolarImagePanel, StatusIndicator


class SolarImagingMainWindow(QMainWindow):
    """
    Main window with tabbed categories of solar images.

    Tabs: GOES SUVI | SDO AIA | SDO HMI | SOHO LASCO | SOHO EIT
    Each tab contains a grid of SolarImagePanel widgets.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solar Imaging - Real-Time Multi-Source Display")
        self.setMinimumSize(1200, 900)
        self.resize(1400, 1000)

        # Map source_id -> panel widget for signal routing
        self.panels: Dict[str, SolarImagePanel] = {}

        # Track last update time
        self.last_update_time: datetime = None

        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_ui(self):
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget for categories
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #353535;
                color: #cccccc;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #454545;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #404040;
            }
        """)

        # Create a tab for each source category
        for category_key, category in SOLAR_SOURCES.items():
            tab = self._create_category_tab(category_key, category)
            self.tabs.addTab(tab, category['name'])

        layout.addWidget(self.tabs)

    def _create_category_tab(self, category_key: str, category: Dict) -> QWidget:
        """Create a tab widget for a source category."""
        # Use scroll area for categories with many images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        tab = QWidget()
        images = category['images']
        count = len(images)

        # Determine grid layout based on image count
        # Goal: large panels, 4-6 visible at once
        if count <= 2:
            cols = 2
        elif count <= 4:
            cols = 2
        elif count <= 6:
            cols = 3
        elif count <= 9:
            cols = 3
        else:
            cols = 5  # For AIA's 10 images

        grid = QGridLayout(tab)
        grid.setSpacing(12)
        grid.setContentsMargins(10, 10, 10, 10)

        for i, img in enumerate(images):
            panel = SolarImagePanel(
                source_id=img['id'],
                source_name=category['name'],
                wavelength=img['display_name'],
                description=img.get('description', '')
            )
            self.panels[img['id']] = panel

            row = i // cols
            col = i % cols
            grid.addWidget(panel, row, col)

        # Add stretch to bottom to prevent panels from expanding too much vertically
        grid.setRowStretch(grid.rowCount(), 1)

        scroll.setWidget(tab)
        return scroll

    def _setup_toolbar(self):
        """Setup the toolbar with refresh and info actions."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2a2a2a;
                border: none;
                spacing: 10px;
                padding: 5px;
            }
            QToolButton {
                background-color: transparent;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QToolButton:hover {
                background-color: #3a3a3a;
            }
        """)
        self.addToolBar(toolbar)

        # Refresh action
        self.refresh_action = QAction("Refresh All", self)
        self.refresh_action.setToolTip("Manually refresh all images")
        self.refresh_action.triggered.connect(self._on_refresh_clicked)
        toolbar.addAction(self.refresh_action)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # About action
        about_action = QAction("About", self)
        about_action.setToolTip("About this application")
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status indicator (connection state)
        self.status_indicator = StatusIndicator()
        self.status_bar.addWidget(self.status_indicator)

        # Separator
        sep = QLabel(" | ")
        sep.setStyleSheet("color: #666666;")
        self.status_bar.addWidget(sep)

        # Last update label
        self.update_label = QLabel("Last update: --")
        self.update_label.setStyleSheet("color: #aaaaaa;")
        self.status_bar.addWidget(self.update_label)

        # Separator
        sep2 = QLabel(" | ")
        sep2.setStyleSheet("color: #666666;")
        self.status_bar.addWidget(sep2)

        # Fetch status
        self.fetch_label = QLabel("")
        self.fetch_label.setStyleSheet("color: #aaaaaa;")
        self.status_bar.addWidget(self.fetch_label)

        # Permanent message on right
        self.permanent_label = QLabel("24 sources across 5 observatories")
        self.permanent_label.setStyleSheet("color: #888888;")
        self.status_bar.addPermanentWidget(self.permanent_label)

    @pyqtSlot(str, bytes, str)
    def on_image_received(self, source_id: str, data: bytes, timestamp: str):
        """Handle incoming image data."""
        if source_id in self.panels:
            self.panels[source_id].update_image(data, timestamp)

        # Update last update time
        self.last_update_time = datetime.utcnow()
        self.update_label.setText(
            f"Last update: {self.last_update_time.strftime('%H:%M:%S')} UTC"
        )

    @pyqtSlot(str, str)
    def on_error(self, source_id: str, error_msg: str):
        """Handle fetch error."""
        if source_id in self.panels:
            self.panels[source_id].set_error(error_msg)

    @pyqtSlot()
    def on_connected(self):
        """Handle connection established."""
        self.status_indicator.set_connected()

    @pyqtSlot()
    def on_disconnected(self):
        """Handle disconnection."""
        self.status_indicator.set_disconnected()

    @pyqtSlot()
    def on_fetch_started(self):
        """Handle fetch cycle start."""
        self.status_indicator.set_fetching()
        self.fetch_label.setText("Fetching images...")

    @pyqtSlot(int, int)
    def on_fetch_completed(self, success_count: int, total_count: int):
        """Handle fetch cycle completion."""
        self.status_indicator.set_connected()
        self.fetch_label.setText(f"Loaded {success_count}/{total_count} images")

    def _on_refresh_clicked(self):
        """Handle manual refresh request."""
        # This will be connected to the client's refresh method
        self.refresh_requested = True
        self.fetch_label.setText("Manual refresh requested...")

    def _show_about(self):
        """Show about dialog."""
        about_text = """
<h2>Solar Imaging Display</h2>
<p>Version 1.0.0</p>

<p>Real-time solar imagery from multiple space-based observatories:</p>

<h3>Data Sources:</h3>
<ul>
<li><b>GOES SUVI</b> - NOAA's geostationary solar imager (6 EUV channels)</li>
<li><b>SDO AIA</b> - NASA's high-resolution solar telescope (10 wavelengths)</li>
<li><b>SDO HMI</b> - Magnetic field and white-light imaging</li>
<li><b>SOHO LASCO</b> - Coronagraph for CME detection</li>
<li><b>SOHO EIT</b> - EUV imaging from L1</li>
</ul>

<h3>Update Cadence:</h3>
<ul>
<li>SUVI, AIA, HMI: Every 60 seconds</li>
<li>LASCO, EIT: Every 15 minutes</li>
</ul>

<p><b>Data provided by:</b><br>
NOAA Space Weather Prediction Center<br>
NASA/ESA via Helioviewer.org</p>

<p>Part of the AutoNVIS Project</p>
"""
        QMessageBox.about(self, "About Solar Imaging", about_text)

    def set_client(self, client):
        """Connect to a data client."""
        self.client = client

        # Connect signals
        client.image_received.connect(self.on_image_received)
        client.error.connect(self.on_error)
        client.connected.connect(self.on_connected)
        client.disconnected.connect(self.on_disconnected)
        client.fetch_started.connect(self.on_fetch_started)
        client.fetch_completed.connect(self.on_fetch_completed)

        # Connect refresh action
        self.refresh_action.triggered.connect(client.refresh_all)

    def closeEvent(self, event):
        """Handle window close."""
        if hasattr(self, 'client') and self.client:
            self.client.stop()
        event.accept()
