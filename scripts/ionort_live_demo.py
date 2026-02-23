#!/usr/bin/env python3
"""
IONORT Live Dashboard Demo

Runs the IONORT-style ray tracing homing algorithm and displays results
in the three-panel visualization dashboard:
- Altitude vs Ground Range
- 3D Geographic View
- Synthetic Oblique Ionogram

Usage:
    python scripts/ionort_live_demo.py [--tx LAT,LON] [--rx LAT,LON] [--freq MIN,MAX]

Examples:
    # Default: Boulder, CO to Albuquerque, NM
    python scripts/ionort_live_demo.py

    # Custom path: New York to Washington DC
    python scripts/ionort_live_demo.py --tx 40.7,-74.0 --rx 38.9,-77.0

    # NVIS mode (short range, high elevation)
    python scripts/ionort_live_demo.py --tx 40.0,-105.0 --rx 40.5,-104.5 --nvis

    # Custom frequency range
    python scripts/ionort_live_demo.py --freq 5,12
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGroupBox, QProgressBar,
    QStatusBar, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from src.raytracer import (
    IonosphericModel,
    HaselgroveSolver,
    HomingAlgorithm,
    HomingSearchSpace,
    HomingConfig,
    HomingResult,
    create_integrator,
)
from src.visualization.pyqt.raytracer import IONORTVisualizationPanel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HomingWorker(QThread):
    """Background worker for ray tracing homing algorithm."""

    progress = pyqtSignal(int, int)  # rays_done, total_rays
    finished = pyqtSignal(object)    # HomingResult
    error = pyqtSignal(str)          # error message

    def __init__(
        self,
        ionosphere: IonosphericModel,
        tx_lat: float, tx_lon: float,
        rx_lat: float, rx_lon: float,
        search_space: HomingSearchSpace,
        config: HomingConfig,
        integrator_name: str = "rk4",
    ):
        super().__init__()
        self.ionosphere = ionosphere
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        self.rx_lat = rx_lat
        self.rx_lon = rx_lon
        self.search_space = search_space
        self.config = config
        self.integrator_name = integrator_name

    def run(self):
        print("DEBUG: HomingWorker.run() started")
        try:
            # Create solver with selected integrator
            print(f"DEBUG: Creating solver with integrator: {self.integrator_name}")
            solver = HaselgroveSolver(
                self.ionosphere,
                integrator_name=self.integrator_name
            )
            print("DEBUG: Solver created")

            # Create homing algorithm
            homing = HomingAlgorithm(solver, self.config)

            # Run homing with progress callback
            def progress_cb(done, total):
                self.progress.emit(done, total)

            result = homing.find_paths(
                self.tx_lat, self.tx_lon,
                self.rx_lat, self.rx_lon,
                search_space=self.search_space,
                progress_callback=progress_cb,
            )

            self.finished.emit(result)

        except Exception as e:
            logger.exception("Homing failed")
            self.error.emit(str(e))


class ControlPanel(QWidget):
    """Control panel for configuring and running homing."""

    run_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("IONORT Ray Tracing Control")
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #ffffff;")
        layout.addWidget(title)

        # Transmitter group
        tx_group = QGroupBox("Transmitter")
        tx_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        tx_layout = QHBoxLayout(tx_group)

        tx_layout.addWidget(QLabel("Lat:"))
        self.tx_lat = QDoubleSpinBox()
        self.tx_lat.setRange(-90, 90)
        self.tx_lat.setValue(40.0)
        self.tx_lat.setDecimals(2)
        tx_layout.addWidget(self.tx_lat)

        tx_layout.addWidget(QLabel("Lon:"))
        self.tx_lon = QDoubleSpinBox()
        self.tx_lon.setRange(-180, 180)
        self.tx_lon.setValue(-105.0)
        self.tx_lon.setDecimals(2)
        tx_layout.addWidget(self.tx_lon)

        layout.addWidget(tx_group)

        # Receiver group
        rx_group = QGroupBox("Receiver")
        rx_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        rx_layout = QHBoxLayout(rx_group)

        rx_layout.addWidget(QLabel("Lat:"))
        self.rx_lat = QDoubleSpinBox()
        self.rx_lat.setRange(-90, 90)
        self.rx_lat.setValue(35.0)
        self.rx_lat.setDecimals(2)
        rx_layout.addWidget(self.rx_lat)

        rx_layout.addWidget(QLabel("Lon:"))
        self.rx_lon = QDoubleSpinBox()
        self.rx_lon.setRange(-180, 180)
        self.rx_lon.setValue(-106.0)
        self.rx_lon.setDecimals(2)
        rx_layout.addWidget(self.rx_lon)

        layout.addWidget(rx_group)

        # Frequency range
        freq_group = QGroupBox("Frequency Range (MHz)")
        freq_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        freq_layout = QHBoxLayout(freq_group)

        freq_layout.addWidget(QLabel("Min:"))
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(1, 30)
        self.freq_min.setValue(3.0)
        self.freq_min.setDecimals(1)
        freq_layout.addWidget(self.freq_min)

        freq_layout.addWidget(QLabel("Max:"))
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(1, 30)
        self.freq_max.setValue(15.0)
        self.freq_max.setDecimals(1)
        freq_layout.addWidget(self.freq_max)

        freq_layout.addWidget(QLabel("Step:"))
        self.freq_step = QDoubleSpinBox()
        self.freq_step.setRange(0.1, 5)
        self.freq_step.setValue(1.0)  # Larger step for faster demo
        self.freq_step.setDecimals(1)
        freq_layout.addWidget(self.freq_step)

        layout.addWidget(freq_group)

        # Elevation range
        elev_group = QGroupBox("Elevation Range (degrees)")
        elev_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        elev_layout = QHBoxLayout(elev_group)

        elev_layout.addWidget(QLabel("Min:"))
        self.elev_min = QDoubleSpinBox()
        self.elev_min.setRange(0, 89)
        self.elev_min.setValue(10.0)
        self.elev_min.setDecimals(0)
        elev_layout.addWidget(self.elev_min)

        elev_layout.addWidget(QLabel("Max:"))
        self.elev_max = QDoubleSpinBox()
        self.elev_max.setRange(1, 90)
        self.elev_max.setValue(80.0)
        self.elev_max.setDecimals(0)
        elev_layout.addWidget(self.elev_max)

        elev_layout.addWidget(QLabel("Step:"))
        self.elev_step = QDoubleSpinBox()
        self.elev_step.setRange(1, 20)
        self.elev_step.setValue(10.0)  # Larger step for faster demo
        self.elev_step.setDecimals(0)
        elev_layout.addWidget(self.elev_step)

        layout.addWidget(elev_group)

        # Options
        opts_group = QGroupBox("Options")
        opts_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        opts_layout = QVBoxLayout(opts_group)

        # Integrator selection
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Integrator:"))
        self.integrator = QComboBox()
        self.integrator.addItems(["rk4", "abm", "rk45"])
        self.integrator.setCurrentText("rk4")
        int_layout.addWidget(self.integrator)
        int_layout.addStretch()
        opts_layout.addLayout(int_layout)

        # Options row
        opt_row = QHBoxLayout()
        self.trace_both_modes = QCheckBox("Both modes (O+X)")
        self.trace_both_modes.setChecked(True)
        self.trace_both_modes.setStyleSheet("color: #ffffff;")
        opt_row.addWidget(self.trace_both_modes)

        self.store_paths = QCheckBox("Store ray paths")
        self.store_paths.setChecked(True)
        self.store_paths.setStyleSheet("color: #ffffff;")
        opt_row.addWidget(self.store_paths)
        opts_layout.addLayout(opt_row)

        # Workers
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Workers:"))
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(4)
        workers_layout.addWidget(self.workers)
        workers_layout.addStretch()
        opts_layout.addLayout(workers_layout)

        layout.addWidget(opts_group)

        # Ionosphere parameters
        iono_group = QGroupBox("Ionosphere Model")
        iono_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        iono_layout = QVBoxLayout(iono_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("foF2 (MHz):"))
        self.foF2 = QDoubleSpinBox()
        self.foF2.setRange(2, 15)
        self.foF2.setValue(7.0)
        self.foF2.setDecimals(1)
        row1.addWidget(self.foF2)

        row1.addWidget(QLabel("hmF2 (km):"))
        self.hmF2 = QDoubleSpinBox()
        self.hmF2.setRange(200, 500)
        self.hmF2.setValue(300.0)
        self.hmF2.setDecimals(0)
        row1.addWidget(self.hmF2)
        iono_layout.addLayout(row1)

        layout.addWidget(iono_group)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Run button
        self.run_btn = QPushButton("Run Homing Algorithm")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4488ff;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5599ff;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)

    def _on_run_clicked(self):
        """Handle run button click with debug output."""
        print("DEBUG: Run button clicked!")
        self.status_label.setText("Button clicked - emitting signal...")
        self.status_label.setStyleSheet("color: #ffaa00;")
        self.run_requested.emit()

        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)

        # Results summary
        self.results_label = QLabel("")
        self.results_label.setStyleSheet("color: #44ff44; font-size: 12px;")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

        layout.addStretch()

    def get_search_space(self) -> HomingSearchSpace:
        """Get search space from UI values."""
        space = HomingSearchSpace(
            freq_range=(self.freq_min.value(), self.freq_max.value()),
            freq_step=self.freq_step.value(),
            elevation_range=(self.elev_min.value(), self.elev_max.value()),
            elevation_step=self.elev_step.value(),
            azimuth_deviation_range=(-5.0, 5.0),  # Smaller range for speed
            azimuth_step=5.0,
        )
        print(f"DEBUG: Search space has {space.total_triplets} triplets")
        return space

    def get_config(self) -> HomingConfig:
        """Get homing config from UI values."""
        return HomingConfig(
            distance_tolerance_km=50.0,
            use_distance_tolerance=True,
            trace_both_modes=self.trace_both_modes.isChecked(),
            store_ray_paths=self.store_paths.isChecked(),
            max_workers=self.workers.value(),
        )

    def get_ionosphere(self) -> IonosphericModel:
        """Create ionosphere model from UI values."""
        model = IonosphericModel()
        model.update_from_realtime(
            foF2=self.foF2.value(),
            hmF2=self.hmF2.value(),
        )
        return model

    def set_running(self, running: bool):
        """Set UI state for running/stopped."""
        self.run_btn.setEnabled(not running)
        if running:
            self.status_label.setText("Running homing algorithm...")
            self.status_label.setStyleSheet("color: #ffaa00;")
        else:
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: #888888;")

    def set_progress(self, done: int, total: int):
        """Update progress bar."""
        if total > 0:
            pct = int(100 * done / total)
            self.progress.setValue(pct)
            self.status_label.setText(f"Tracing rays: {done}/{total}")

    def show_result(self, result: HomingResult):
        """Display result summary."""
        self.progress.setValue(100)
        self.status_label.setText("Complete")
        self.status_label.setStyleSheet("color: #44ff44;")

        text = (
            f"Range: {result.great_circle_range_km:.0f} km\n"
            f"Winners: {result.num_winners} "
            f"(O: {len(result.o_mode_winners)}, X: {len(result.x_mode_winners)})\n"
            f"MUF: {result.muf:.1f} MHz, LUF: {result.luf:.1f} MHz, FOT: {result.fot:.1f} MHz\n"
            f"Rays traced: {result.total_rays_traced}\n"
            f"Time: {result.computation_time_s:.1f}s"
        )
        self.results_label.setText(text)


class IONORTLiveWindow(QMainWindow):
    """Main window for IONORT live demo."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("IONORT Live Ray Tracing Dashboard")
        self.setMinimumSize(1600, 1000)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                color: #ffffff;
                padding: 3px;
            }
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4488ff;
            }
        """)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout with splitter
        layout = QHBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Control panel (left)
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(350)
        self.control_panel.run_requested.connect(self._run_homing)
        splitter.addWidget(self.control_panel)

        # Visualization panel (right)
        self.viz_panel = IONORTVisualizationPanel()
        splitter.addWidget(self.viz_panel)

        # Set splitter sizes (control: 300px, viz: rest)
        splitter.setSizes([300, 1300])

        layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready - Configure parameters and click Run")

    def _run_homing(self):
        """Start homing algorithm in background thread."""
        print("DEBUG: _run_homing called!")

        if self.worker is not None and self.worker.isRunning():
            print("DEBUG: Worker already running")
            QMessageBox.warning(self, "Busy", "Homing already running")
            return

        # Get parameters from control panel
        search_space = self.control_panel.get_search_space()
        config = self.control_panel.get_config()
        ionosphere = self.control_panel.get_ionosphere()

        tx_lat = self.control_panel.tx_lat.value()
        tx_lon = self.control_panel.tx_lon.value()
        rx_lat = self.control_panel.rx_lat.value()
        rx_lon = self.control_panel.rx_lon.value()
        integrator = self.control_panel.integrator.currentText()

        # Log
        logger.info(f"Starting homing: ({tx_lat}, {tx_lon}) -> ({rx_lat}, {rx_lon})")
        logger.info(f"Search space: {search_space.total_triplets} triplets")
        logger.info(f"Integrator: {integrator}")

        # Update UI
        self.control_panel.set_running(True)
        self.viz_panel.clear()
        self.statusBar().showMessage("Running homing algorithm...")

        # Create and start worker
        self.worker = HomingWorker(
            ionosphere,
            tx_lat, tx_lon,
            rx_lat, rx_lon,
            search_space,
            config,
            integrator,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, done: int, total: int):
        """Handle progress update."""
        self.control_panel.set_progress(done, total)

    def _on_finished(self, result: HomingResult):
        """Handle homing completion."""
        self.control_panel.set_running(False)
        self.control_panel.show_result(result)

        # Update visualization
        self.viz_panel.update_from_homing_result(result)

        # Status
        msg = (f"Found {result.num_winners} winner triplets - "
               f"MUF: {result.muf:.1f} MHz, LUF: {result.luf:.1f} MHz")
        self.statusBar().showMessage(msg)

        logger.info(f"Homing complete: {result}")

    def _on_error(self, error_msg: str):
        """Handle homing error."""
        self.control_panel.set_running(False)
        self.statusBar().showMessage(f"Error: {error_msg}")
        QMessageBox.critical(self, "Homing Error", error_msg)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IONORT Live Ray Tracing Dashboard"
    )
    parser.add_argument(
        "--tx", type=str, default="40.0,-105.0",
        help="Transmitter lat,lon (default: 40.0,-105.0 Boulder, CO)"
    )
    parser.add_argument(
        "--rx", type=str, default="35.0,-106.0",
        help="Receiver lat,lon (default: 35.0,-106.0 Albuquerque, NM)"
    )
    parser.add_argument(
        "--freq", type=str, default="3,15",
        help="Frequency range min,max MHz (default: 3,15)"
    )
    parser.add_argument(
        "--nvis", action="store_true",
        help="NVIS mode (short range, high elevation)"
    )
    parser.add_argument(
        "--foF2", type=float, default=7.0,
        help="F2 critical frequency MHz (default: 7.0)"
    )
    parser.add_argument(
        "--hmF2", type=float, default=300.0,
        help="F2 peak height km (default: 300)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse coordinates
    tx_lat, tx_lon = map(float, args.tx.split(","))
    rx_lat, rx_lon = map(float, args.rx.split(","))
    freq_min, freq_max = map(float, args.freq.split(","))

    # Create application
    app = QApplication(sys.argv)

    # Create and configure window
    window = IONORTLiveWindow()

    # Apply command line args to UI
    window.control_panel.tx_lat.setValue(tx_lat)
    window.control_panel.tx_lon.setValue(tx_lon)
    window.control_panel.rx_lat.setValue(rx_lat)
    window.control_panel.rx_lon.setValue(rx_lon)
    window.control_panel.freq_min.setValue(freq_min)
    window.control_panel.freq_max.setValue(freq_max)
    window.control_panel.foF2.setValue(args.foF2)
    window.control_panel.hmF2.setValue(args.hmF2)

    # NVIS mode adjustments
    if args.nvis:
        window.control_panel.elev_min.setValue(60.0)
        window.control_panel.elev_max.setValue(89.0)
        window.control_panel.elev_step.setValue(2.0)

    window.show()

    logger.info("IONORT Live Dashboard started")
    logger.info(f"Tx: ({tx_lat}, {tx_lon}), Rx: ({rx_lat}, {rx_lon})")
    logger.info(f"Frequency: {freq_min}-{freq_max} MHz")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
