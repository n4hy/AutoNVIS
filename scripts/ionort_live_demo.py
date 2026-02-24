#!/usr/bin/env python3
"""
IONORT Live Dashboard Demo with Real-Time Ionospheric Data

Runs the IONORT-style ray tracing homing algorithm with LIVE ionospheric data
from the GIRO ionosonde network and NOAA space weather services.

Features:
- Real-time foF2/hmF2 from nearest ionosonde stations
- Space weather monitoring (X-ray flux, Kp index)
- Automatic ionosphere model updates
- Three-panel visualization dashboard

Usage:
    python scripts/ionort_live_demo.py [--tx LAT,LON] [--rx LAT,LON] [--live] [options]

Examples:
    # Default with live data
    python scripts/ionort_live_demo.py --live

    # Custom path: New York to Washington DC
    python scripts/ionort_live_demo.py --tx 40.7,-74.0 --rx 38.9,-77.0 --live

    # NYC to Chicago with FT8 SNR threshold
    python scripts/ionort_live_demo.py --tx 40.71,-74.00 --rx 41.88,-87.63 --snr-cutoff -20

    # NVIS mode (short range, high elevation)
    python scripts/ionort_live_demo.py --tx 40.0,-105.0 --rx 40.5,-104.5 --nvis --live

    # Long path with relaxed tolerance and voice-quality SNR
    python scripts/ionort_live_demo.py --tx 34.0,-118.0 --rx 40.7,-74.0 --tolerance 200 --snr-cutoff 10

    # Simulated live data (for testing without network)
    python scripts/ionort_live_demo.py --live --simulated
"""

import sys
import os
import argparse
import logging
import multiprocessing
from datetime import datetime, timezone
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGroupBox, QProgressBar,
    QStatusBar, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSplitter, QFrame, QScrollArea, QTextEdit
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

# Import live data client
try:
    from src.ingestion.live_iono_client import LiveIonoClient, LiveIonosphericState
    HAS_LIVE_CLIENT = True
except ImportError as e:
    HAS_LIVE_CLIENT = False
    print(f"Warning: Live data client not available: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HomingWorker(QThread):
    """Background worker for ray tracing homing algorithm."""

    progress = pyqtSignal(int, int)  # rays_done, total_rays
    finished = pyqtSignal(object)    # HomingResult
    cancelled = pyqtSignal()         # emitted when cancelled
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
        self._homing: Optional[HomingAlgorithm] = None

    def cancel(self):
        """Cancel the running homing algorithm."""
        if self._homing:
            self._homing.cancel()
            self._homing.cleanup()
            logger.info("Homing worker: cancellation requested")

    def run(self):
        try:
            # Create solver with selected integrator
            solver = HaselgroveSolver(
                self.ionosphere,
                integrator_name=self.integrator_name
            )

            # Create homing algorithm
            self._homing = HomingAlgorithm(solver, self.config)

            # Run homing with progress callback
            def progress_cb(done, total):
                self.progress.emit(done, total)

            result = self._homing.find_paths(
                self.tx_lat, self.tx_lon,
                self.rx_lat, self.rx_lon,
                search_space=self.search_space,
                progress_callback=progress_cb,
            )

            # Check if cancelled
            if self._homing.is_cancelled():
                self.cancelled.emit()
            else:
                # Diagnostic info about paths - use print for immediate visibility
                paths_with_data = sum(1 for w in result.winner_triplets
                                      if w.ray_path and w.ray_path.states)
                print(f"\n{'='*60}")
                print(f"HOMING RESULT DIAGNOSTICS:")
                print(f"  Winners: {result.num_winners}")
                print(f"  Paths with data: {paths_with_data}")
                print(f"  store_ray_paths: {self.config.store_ray_paths}")
                print(f"  use_multiprocessing: {self.config.use_multiprocessing}")
                if result.winner_triplets:
                    # Show first few winners
                    print(f"  First 5 winners:")
                    for w in result.winner_triplets[:5]:
                        has_path = "YES" if (w.ray_path and w.ray_path.states) else "NO"
                        print(f"    {w.frequency_mhz:.1f} MHz, {w.elevation_deg:.0f}°, path={has_path}")
                print(f"{'='*60}\n", flush=True)
                self.finished.emit(result)

        except Exception as e:
            logger.exception("Homing failed")
            self.error.emit(str(e))


class LiveDataPanel(QWidget):
    """Panel showing live ionospheric data status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Status indicator
        status_row = QHBoxLayout()
        self.status_indicator = QLabel("OFFLINE")
        self.status_indicator.setStyleSheet("""
            QLabel {
                background-color: #aa3333;
                color: white;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 3px;
            }
        """)
        status_row.addWidget(self.status_indicator)
        status_row.addStretch()
        layout.addLayout(status_row)

        # Data source
        self.source_label = QLabel("Source: --")
        self.source_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.source_label)

        # Ionospheric parameters
        params_layout = QHBoxLayout()

        self.fof2_label = QLabel("foF2: --")
        self.fof2_label.setStyleSheet("color: #44ff44;")
        params_layout.addWidget(self.fof2_label)

        self.hmf2_label = QLabel("hmF2: --")
        self.hmf2_label.setStyleSheet("color: #44ff44;")
        params_layout.addWidget(self.hmf2_label)

        layout.addLayout(params_layout)

        # Space weather
        wx_layout = QHBoxLayout()

        self.kp_label = QLabel("Kp: --")
        self.kp_label.setStyleSheet("color: #ffaa44;")
        wx_layout.addWidget(self.kp_label)

        self.xray_label = QLabel("R: --")
        self.xray_label.setStyleSheet("color: #ffaa44;")
        wx_layout.addWidget(self.xray_label)

        layout.addLayout(wx_layout)

        # Data age and confidence
        self.age_label = QLabel("Age: --")
        self.age_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(self.age_label)

    def set_online(self, is_online: bool):
        """Update online/offline status."""
        if is_online:
            self.status_indicator.setText("LIVE")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #33aa33;
                    color: white;
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 3px;
                }
            """)
        else:
            self.status_indicator.setText("OFFLINE")
            self.status_indicator.setStyleSheet("""
                QLabel {
                    background-color: #aa3333;
                    color: white;
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 3px;
                }
            """)

    def update_state(self, state: 'LiveIonosphericState'):
        """Update display from live state."""
        self.set_online(True)

        # Source
        source = state.source_station or "Default"
        if state.source_distance_km > 0:
            source += f" ({state.source_distance_km:.0f} km)"
        self.source_label.setText(f"Source: {source}")

        # Ionospheric parameters
        self.fof2_label.setText(f"foF2: {state.foF2:.2f} MHz")
        self.hmf2_label.setText(f"hmF2: {state.hmF2:.0f} km")

        # Space weather
        self.kp_label.setText(f"Kp: {state.kp_index:.1f}")
        r_text = f"R{state.r_scale}" if state.r_scale > 0 else "R0"
        self.xray_label.setText(r_text)

        # Confidence color
        if state.overall_confidence >= 0.7:
            color = "#44ff44"  # Green
        elif state.overall_confidence >= 0.4:
            color = "#ffaa44"  # Orange
        else:
            color = "#ff4444"  # Red

        self.fof2_label.setStyleSheet(f"color: {color};")
        self.hmf2_label.setStyleSheet(f"color: {color};")

        # Age
        if state.data_age_seconds < 60:
            age_text = f"Age: {state.data_age_seconds:.0f}s"
        elif state.data_age_seconds < 3600:
            age_text = f"Age: {state.data_age_seconds/60:.0f}m"
        else:
            age_text = f"Age: {state.data_age_seconds/3600:.1f}h"
        age_text += f" | Conf: {state.overall_confidence:.0%}"
        self.age_label.setText(age_text)


class ControlPanel(QWidget):
    """Control panel for configuring and running homing."""

    run_requested = pyqtSignal()
    live_data_toggled = pyqtSignal(bool)
    location_changed = pyqtSignal(float, float, float, float)  # tx_lat, tx_lon, rx_lat, rx_lon

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Title
        title = QLabel("IONORT Ray Tracing Control")
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #ffffff;")
        layout.addWidget(title)

        # Live Data Section
        live_group = QGroupBox("Live Ionospheric Data")
        live_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        live_layout = QVBoxLayout(live_group)

        # Enable checkbox
        enable_row = QHBoxLayout()
        self.live_enabled = QCheckBox("Enable Live Data")
        self.live_enabled.setStyleSheet("color: #ffffff;")
        self.live_enabled.toggled.connect(self._on_live_toggled)
        enable_row.addWidget(self.live_enabled)

        self.simulated_mode = QCheckBox("Simulated")
        self.simulated_mode.setStyleSheet("color: #888888;")
        self.simulated_mode.setToolTip("Use simulated data (for testing without network)")
        enable_row.addWidget(self.simulated_mode)

        live_layout.addLayout(enable_row)

        # Live data status panel
        self.live_panel = LiveDataPanel()
        live_layout.addWidget(self.live_panel)

        # Auto-apply checkbox
        self.auto_apply = QCheckBox("Auto-apply to model")
        self.auto_apply.setChecked(True)
        self.auto_apply.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        live_layout.addWidget(self.auto_apply)

        layout.addWidget(live_group)

        # Transmitter group
        tx_group = QGroupBox("Transmitter")
        tx_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        tx_layout = QHBoxLayout(tx_group)

        tx_layout.addWidget(QLabel("Lat:"))
        self.tx_lat = QDoubleSpinBox()
        self.tx_lat.setRange(-90, 90)
        self.tx_lat.setValue(40.0)
        self.tx_lat.setDecimals(2)
        self.tx_lat.valueChanged.connect(self._on_location_changed)
        tx_layout.addWidget(self.tx_lat)

        tx_layout.addWidget(QLabel("Lon:"))
        self.tx_lon = QDoubleSpinBox()
        self.tx_lon.setRange(-180, 180)
        self.tx_lon.setValue(-105.0)
        self.tx_lon.setDecimals(2)
        self.tx_lon.valueChanged.connect(self._on_location_changed)
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
        self.rx_lat.valueChanged.connect(self._on_location_changed)
        rx_layout.addWidget(self.rx_lat)

        rx_layout.addWidget(QLabel("Lon:"))
        self.rx_lon = QDoubleSpinBox()
        self.rx_lon.setRange(-180, 180)
        self.rx_lon.setValue(-106.0)
        self.rx_lon.setDecimals(2)
        self.rx_lon.valueChanged.connect(self._on_location_changed)
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
        self.freq_step.setValue(1.0)
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
        self.elev_step.setValue(10.0)
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
        # Order: fastest to slowest (abm > rk45 > rk4)
        # Accuracy: rk45 (5th order) > rk4/abm (4th order)
        self.integrator.addItem("abm", "abm")      # Fastest: 2 evals/step, 4th order
        self.integrator.addItem("rk45", "rk45")    # Medium: 7 evals/step, 5th order (most accurate)
        self.integrator.addItem("rk4", "rk4")      # Slow: 12 evals/step, 4th order
        self.integrator.setCurrentIndex(0)  # Default to fastest (abm)
        self.integrator.setToolTip(
            "Integrator selection:\n"
            "  abm  - Fastest (2 evals/step), 4th order\n"
            "  rk45 - Medium (7 evals/step), 5th order (most accurate)\n"
            "  rk4  - Slow (12 evals/step), 4th order with error tracking"
        )
        int_layout.addWidget(self.integrator)
        int_layout.addStretch()
        opts_layout.addLayout(int_layout)

        # Options row 1
        opt_row1 = QHBoxLayout()
        self.trace_both_modes = QCheckBox("Both modes (O+X)")
        self.trace_both_modes.setChecked(True)
        self.trace_both_modes.setStyleSheet("color: #ffffff;")
        opt_row1.addWidget(self.trace_both_modes)
        opt_row1.addStretch()
        opts_layout.addLayout(opt_row1)

        # Options row 2 - ray path storage (required for visualization)
        opt_row2 = QHBoxLayout()
        self.store_paths = QCheckBox("Store ray paths (required for visualization)")
        self.store_paths.setChecked(True)
        self.store_paths.setStyleSheet("color: #ffff44; font-weight: bold;")
        opt_row2.addWidget(self.store_paths)
        opt_row2.addStretch()
        opts_layout.addLayout(opt_row2)

        # Workers
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Workers:"))
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(4)
        workers_layout.addWidget(self.workers)
        workers_layout.addStretch()
        opts_layout.addLayout(workers_layout)

        # Tolerance (landing distance tolerance in km)
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance (km):"))
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(10, 500)
        self.tolerance.setValue(100.0)
        self.tolerance.setDecimals(0)
        self.tolerance.setToolTip("Landing tolerance - how close ray must land to receiver (km)")
        tol_layout.addWidget(self.tolerance)
        tol_layout.addStretch()
        opts_layout.addLayout(tol_layout)

        # SNR Cutoff (minimum usable SNR threshold)
        snr_cutoff_layout = QHBoxLayout()
        snr_cutoff_layout.addWidget(QLabel("SNR Cutoff (dB):"))
        self.snr_cutoff = QDoubleSpinBox()
        self.snr_cutoff.setRange(-20, 60)
        self.snr_cutoff.setValue(0.0)
        self.snr_cutoff.setDecimals(0)
        self.snr_cutoff.setToolTip("Minimum SNR threshold - paths below this are filtered out\n"
                                    "-20 dB for FT8/digital, 0 dB for CW, 10+ dB for voice")
        snr_cutoff_layout.addWidget(self.snr_cutoff)
        snr_cutoff_layout.addStretch()
        opts_layout.addLayout(snr_cutoff_layout)

        layout.addWidget(opts_group)

        # Radio Configuration (for link budget / SNR calculation)
        radio_group = QGroupBox("Radio Configuration (Link Budget)")
        radio_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        radio_layout = QVBoxLayout(radio_group)

        # TX Power
        tx_power_row = QHBoxLayout()
        tx_power_row.addWidget(QLabel("TX Power (W):"))
        self.tx_power = QDoubleSpinBox()
        self.tx_power.setRange(1, 1500)
        self.tx_power.setValue(100.0)
        self.tx_power.setDecimals(0)
        self.tx_power.setToolTip("Transmitter power in watts")
        tx_power_row.addWidget(self.tx_power)
        tx_power_row.addStretch()
        radio_layout.addLayout(tx_power_row)

        # Antenna gains
        antenna_row = QHBoxLayout()
        antenna_row.addWidget(QLabel("TX Ant (dBi):"))
        self.tx_antenna_gain = QDoubleSpinBox()
        self.tx_antenna_gain.setRange(-10, 20)
        self.tx_antenna_gain.setValue(0.0)
        self.tx_antenna_gain.setDecimals(1)
        self.tx_antenna_gain.setToolTip("TX antenna gain (0=isotropic)")
        antenna_row.addWidget(self.tx_antenna_gain)

        antenna_row.addWidget(QLabel("RX Ant (dBi):"))
        self.rx_antenna_gain = QDoubleSpinBox()
        self.rx_antenna_gain.setRange(-10, 20)
        self.rx_antenna_gain.setValue(0.0)
        self.rx_antenna_gain.setDecimals(1)
        self.rx_antenna_gain.setToolTip("RX antenna gain (0=isotropic)")
        antenna_row.addWidget(self.rx_antenna_gain)
        radio_layout.addLayout(antenna_row)

        # RX bandwidth
        bw_row = QHBoxLayout()
        bw_row.addWidget(QLabel("RX BW (Hz):"))
        self.rx_bandwidth = QComboBox()
        self.rx_bandwidth.addItems(["500", "2400", "3000", "6000"])
        self.rx_bandwidth.setCurrentText("3000")
        self.rx_bandwidth.setToolTip("Receiver bandwidth (500=CW, 3000=SSB)")
        bw_row.addWidget(self.rx_bandwidth)
        bw_row.addStretch()
        radio_layout.addLayout(bw_row)

        layout.addWidget(radio_group)

        # Ionosphere parameters (manual override)
        iono_group = QGroupBox("Ionosphere Model (Manual Override)")
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

        # Note about live data override
        self.override_note = QLabel("(Using manual values - enable Live Data for real-time)")
        self.override_note.setStyleSheet("color: #888888; font-size: 10px;")
        iono_layout.addWidget(self.override_note)

        layout.addWidget(iono_group)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Run/Cancel buttons in a row
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Run Homing")
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
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc4444;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #dd5555;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self.cancel_btn)

        layout.addLayout(btn_row)

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

    cancel_requested = pyqtSignal()

    def _on_run_clicked(self):
        """Handle run button click."""
        self.run_requested.emit()

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit()

    def _on_live_toggled(self, enabled: bool):
        """Handle live data toggle."""
        if enabled:
            self.override_note.setText("(Using LIVE data - manual values ignored)")
            self.override_note.setStyleSheet("color: #44ff44; font-size: 10px;")
            self.foF2.setEnabled(False)
            self.hmF2.setEnabled(False)
        else:
            self.override_note.setText("(Using manual values - enable Live Data for real-time)")
            self.override_note.setStyleSheet("color: #888888; font-size: 10px;")
            self.foF2.setEnabled(True)
            self.hmF2.setEnabled(True)

        self.live_data_toggled.emit(enabled)

    def _on_location_changed(self):
        """Handle location change for live data reference point update."""
        self.location_changed.emit(
            self.tx_lat.value(),
            self.tx_lon.value(),
            self.rx_lat.value(),
            self.rx_lon.value()
        )

    def update_from_live_state(self, state: 'LiveIonosphericState'):
        """Update UI from live ionospheric state."""
        self.live_panel.update_state(state)

        # Update ionosphere values if auto-apply is enabled
        if self.auto_apply.isChecked() and self.live_enabled.isChecked():
            # Block signals to avoid feedback loop
            self.foF2.blockSignals(True)
            self.hmF2.blockSignals(True)
            self.foF2.setValue(state.foF2)
            self.hmF2.setValue(state.hmF2)
            self.foF2.blockSignals(False)
            self.hmF2.blockSignals(False)

    def get_search_space(self) -> HomingSearchSpace:
        """Get search space from UI values."""
        return HomingSearchSpace(
            freq_range=(self.freq_min.value(), self.freq_max.value()),
            freq_step=self.freq_step.value(),
            elevation_range=(self.elev_min.value(), self.elev_max.value()),
            elevation_step=self.elev_step.value(),
            azimuth_deviation_range=(-5.0, 5.0),
            azimuth_step=5.0,
        )

    def get_config(self) -> HomingConfig:
        """Get homing config from UI values."""
        return HomingConfig(
            distance_tolerance_km=self.tolerance.value(),
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
        self.cancel_btn.setEnabled(running)
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

        if result.num_winners == 0:
            # FAILURE - no paths found
            self.status_label.setText("NO PATHS FOUND")
            self.status_label.setStyleSheet("""
                color: #ffffff;
                background-color: #cc3333;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            """)
            text = (
                f"*** NO PROPAGATION PATHS FOUND ***\n\n"
                f"Range: {result.great_circle_range_km:.0f} km\n"
                f"Rays traced: {result.total_rays_traced}\n"
                f"Time: {result.computation_time_s:.1f}s\n\n"
                f"Possible causes:\n"
                f"- Frequency too high for ionosphere (try lower freq)\n"
                f"- foF2 too low (check live data or increase)\n"
                f"- Path too long for max hops setting\n"
                f"- Tolerance too tight (increase distance_tolerance)"
            )
        else:
            # SUCCESS
            self.status_label.setText("Complete")
            self.status_label.setStyleSheet("color: #44ff44;")

            # Count multi-hop winners
            single_hop = sum(1 for w in result.winner_triplets if w.hop_count <= 1)
            multi_hop = sum(1 for w in result.winner_triplets if w.hop_count > 1)
            max_hops = max((w.hop_count for w in result.winner_triplets), default=0)

            hop_info = f"Hops: 1-hop={single_hop}"
            if multi_hop > 0:
                hop_info += f", multi-hop={multi_hop} (max {max_hops})"

            # SNR statistics
            snr_values = [w.snr_db for w in result.winner_triplets if w.snr_db is not None]
            if snr_values:
                best_snr = max(snr_values)
                avg_snr = sum(snr_values) / len(snr_values)
                # Find best winner for details
                best_winner = max(
                    (w for w in result.winner_triplets if w.snr_db is not None),
                    key=lambda w: w.snr_db
                )
                snr_info = (
                    f"\n--- Link Budget ---\n"
                    f"Best SNR: {best_snr:.0f} dB @ {best_winner.frequency_mhz:.1f} MHz\n"
                    f"Signal: {best_winner.signal_strength_dbm:.0f} dBm, "
                    f"Loss: {best_winner.path_loss_db:.0f} dB\n"
                    f"Avg SNR: {avg_snr:.0f} dB"
                )
            else:
                snr_info = ""

            # Count paths with visualization data
            paths_with_viz = sum(1 for w in result.winner_triplets if w.ray_path and w.ray_path.states)
            paths_without_viz = result.num_winners - paths_with_viz
            viz_info = f"Paths visualized: {paths_with_viz}/{result.num_winners}"
            if paths_without_viz > 0:
                viz_info += f" ({paths_without_viz} missing)"

            text = (
                f"Range: {result.great_circle_range_km:.0f} km\n"
                f"Winners: {result.num_winners} "
                f"(O: {len(result.o_mode_winners)}, X: {len(result.x_mode_winners)})\n"
                f"{hop_info}\n"
                f"MUF: {result.muf:.1f} MHz, LUF: {result.luf:.1f} MHz, FOT: {result.fot:.1f} MHz\n"
                f"{viz_info}\n"
                f"Rays traced: {result.total_rays_traced}\n"
                f"Time: {result.computation_time_s:.1f}s"
                f"{snr_info}"
            )

        self.results_label.setText(text)


class IONORTLiveWindow(QMainWindow):
    """Main window for IONORT live demo with real-time data."""

    def __init__(self, use_live: bool = False, use_simulated: bool = False):
        super().__init__()
        self.worker = None
        self.live_client: Optional[LiveIonoClient] = None
        self.use_live = use_live
        self.use_simulated = use_simulated
        self.step_km = 1.0  # Default integration step, can be overridden
        self.last_result: Optional[HomingResult] = None  # Store last result for re-filtering
        self.excluded_frequencies: set = set()  # Currently excluded frequency ranges
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
            QScrollArea {
                border: none;
            }
        """)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout with splitter
        layout = QHBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Control panel in scroll area (left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.control_panel = ControlPanel()
        self.control_panel.setMinimumWidth(320)
        self.control_panel.setMaximumWidth(400)
        self.control_panel.run_requested.connect(self._run_homing)
        self.control_panel.cancel_requested.connect(self._cancel_homing)
        self.control_panel.live_data_toggled.connect(self._on_live_data_toggled)
        self.control_panel.location_changed.connect(self._on_location_changed)

        scroll.setWidget(self.control_panel)
        splitter.addWidget(scroll)

        # Visualization panel (right)
        self.viz_panel = IONORTVisualizationPanel()
        splitter.addWidget(self.viz_panel)

        # Connect frequency filter signal from altitude widget
        self.viz_panel.altitude_widget.frequency_filter_changed.connect(
            self._on_frequency_filter_changed
        )

        # Set splitter sizes (control: 350px, viz: rest)
        splitter.setSizes([350, 1250])

        # Vertical splitter for main content + log console
        vsplitter = QSplitter(Qt.Orientation.Vertical)
        vsplitter.addWidget(splitter)

        # Log console at bottom
        log_group = QGroupBox("Diagnostic Console (copyable)")
        log_group.setStyleSheet("QGroupBox { color: #ffffff; }")
        log_layout = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #00ff00;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        self.log_console.setMaximumHeight(150)
        log_layout.addWidget(self.log_console)
        vsplitter.addWidget(log_group)

        vsplitter.setSizes([850, 150])
        layout.addWidget(vsplitter)

        # Status bar
        self.statusBar().showMessage("Ready - Configure parameters and click Run")

        # Initialize live data if requested
        if self.use_live:
            self.control_panel.live_enabled.setChecked(True)
            self.control_panel.simulated_mode.setChecked(self.use_simulated)

    def _init_live_client(self):
        """Initialize live data client."""
        if not HAS_LIVE_CLIENT:
            QMessageBox.warning(
                self, "Live Data Unavailable",
                "Live data client not available. Using manual mode."
            )
            self.control_panel.live_enabled.setChecked(False)
            return

        # Calculate midpoint for reference location
        mid_lat = (self.control_panel.tx_lat.value() + self.control_panel.rx_lat.value()) / 2
        mid_lon = (self.control_panel.tx_lon.value() + self.control_panel.rx_lon.value()) / 2

        self.live_client = LiveIonoClient(
            reference_lat=mid_lat,
            reference_lon=mid_lon,
            update_interval_ms=60000,  # 1 minute updates
            use_simulated=self.control_panel.simulated_mode.isChecked(),
        )

        # Connect signals
        self.live_client.state_updated.connect(self._on_live_state_update)
        self.live_client.connected.connect(self._on_live_connected)
        self.live_client.disconnected.connect(self._on_live_disconnected)
        self.live_client.error.connect(self._on_live_error)

        self.live_client.start()
        logger.info("Live data client started")

    def _stop_live_client(self):
        """Stop live data client."""
        if self.live_client:
            self.live_client.stop()
            self.live_client = None
            logger.info("Live data client stopped")

    def _on_live_data_toggled(self, enabled: bool):
        """Handle live data toggle."""
        if enabled:
            self._init_live_client()
        else:
            self._stop_live_client()
            self.control_panel.live_panel.set_online(False)

    def _on_location_changed(self, tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float):
        """Handle location change - update live client reference point."""
        if self.live_client:
            mid_lat = (tx_lat + rx_lat) / 2
            mid_lon = (tx_lon + rx_lon) / 2
            self.live_client.set_reference_location(mid_lat, mid_lon)

    def _on_live_state_update(self, state: 'LiveIonosphericState'):
        """Handle live state update."""
        self.control_panel.update_from_live_state(state)

        # Update status bar with live data info
        source = state.source_station or "Default"
        self.statusBar().showMessage(
            f"Live Data: foF2={state.foF2:.2f} MHz, hmF2={state.hmF2:.0f} km "
            f"(from {source}, Kp={state.kp_index:.1f})"
        )

    def _on_live_connected(self):
        """Handle live client connected."""
        logger.info("Live data client connected")
        self.control_panel.live_panel.set_online(True)

    def _on_live_disconnected(self):
        """Handle live client disconnected."""
        logger.info("Live data client disconnected")
        self.control_panel.live_panel.set_online(False)

    def _on_live_error(self, error_msg: str):
        """Handle live client error."""
        logger.warning(f"Live data error: {error_msg}")
        self.statusBar().showMessage(f"Live data error: {error_msg}")

    def _on_frequency_filter_changed(self, excluded_freqs: set):
        """Handle frequency filter button toggle - re-render with filtered frequencies."""
        self.excluded_frequencies = excluded_freqs
        self.log(f"Frequency filter changed: excluding {sorted(excluded_freqs)}")

        if self.last_result and self.last_result.winner_triplets:
            self._render_filtered_result()

    # Button frequencies for mapping traces to buttons
    BUTTON_FREQUENCIES = [2, 5, 8, 12, 15, 20, 25, 30]

    def _get_nearest_button_freq(self, freq_mhz: float) -> int:
        """Find the nearest button frequency for a given trace frequency."""
        nearest = self.BUTTON_FREQUENCIES[0]
        min_dist = abs(freq_mhz - nearest)
        for btn_freq in self.BUTTON_FREQUENCIES[1:]:
            dist = abs(freq_mhz - btn_freq)
            if dist < min_dist:
                min_dist = dist
                nearest = btn_freq
        return nearest

    def _render_filtered_result(self):
        """Re-render the last result with current frequency filters applied."""
        if not self.last_result:
            return

        # Clear and re-render with filtered winners
        result = self.last_result

        # Filter winners based on excluded frequencies
        # Map each trace to its nearest button, then check if that button is excluded
        filtered_winners = []
        for w in result.winner_triplets:
            nearest_button = self._get_nearest_button_freq(w.frequency_mhz)
            if nearest_button not in self.excluded_frequencies:
                filtered_winners.append(w)

        self.log(f"Filtering: {len(result.winner_triplets)} total, {len(filtered_winners)} shown")

        # Clear visualization
        self.viz_panel.altitude_widget.clear()
        self.viz_panel.geographic_widget.clear()

        if not filtered_winners:
            self.statusBar().showMessage("All frequencies filtered out")
            return

        # Get Tx/Rx positions
        tx_lat, tx_lon, tx_alt = result.tx_position
        rx_lat, rx_lon, rx_alt = result.rx_position

        # Re-add markers
        self.viz_panel.geographic_widget.add_marker(tx_lat, tx_lon, tx_alt, '#ff4444', 150)
        self.viz_panel.geographic_widget.add_marker(rx_lat, rx_lon, rx_alt, '#44ff44', 150)

        # Re-add filtered ray paths
        # Note: Color mapping uses fixed 2-30 MHz range (set in widget constants)
        paths_added = 0
        for w in filtered_winners:
            if w.ray_path and w.ray_path.states:
                positions = [s.lat_lon_alt() for s in w.ray_path.states]
                is_reflected = w.ray_path.termination.value == 'ground'
                is_o_mode = w.mode.value == 'O'

                self.viz_panel.altitude_widget.add_ray_path_from_positions(
                    positions, tx_lat, tx_lon,
                    w.frequency_mhz, is_reflected, is_o_mode
                )
                self.viz_panel.geographic_widget.add_ray_path(
                    positions, w.frequency_mhz
                )
                paths_added += 1

        # Update status
        excluded_count = len(result.winner_triplets) - len(filtered_winners)
        self.statusBar().showMessage(
            f"Showing {len(filtered_winners)} paths ({excluded_count} filtered out)"
        )

    def _run_homing(self):
        """Start homing algorithm in background thread."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Homing already running")
            return

        # Get parameters from control panel
        search_space = self.control_panel.get_search_space()
        config = self.control_panel.get_config()
        ionosphere = self.control_panel.get_ionosphere()

        # Apply runtime params from command line (stored in window)
        if hasattr(self, 'step_km'):
            config.step_km = self.step_km
        if hasattr(self, 'max_hops'):
            config.max_hops = self.max_hops
        # Tolerance is now set directly from control_panel.get_config()

        tx_lat = self.control_panel.tx_lat.value()
        tx_lon = self.control_panel.tx_lon.value()
        rx_lat = self.control_panel.rx_lat.value()
        rx_lon = self.control_panel.rx_lon.value()
        integrator = self.control_panel.integrator.currentData()

        # Log
        logger.info(f"Starting homing: ({tx_lat}, {tx_lon}) -> ({rx_lat}, {rx_lon})")
        logger.info(f"Search space: {search_space.total_triplets} triplets")
        logger.info(f"Integrator: {integrator}, step_km: {config.step_km}, workers: {config.max_workers}")
        logger.info(f"Ionosphere: foF2={self.control_panel.foF2.value():.2f} MHz, "
                   f"hmF2={self.control_panel.hmF2.value():.0f} km")

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
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel_homing(self):
        """Cancel running homing algorithm."""
        if self.worker and self.worker.isRunning():
            logger.info("Cancelling homing algorithm...")
            self.statusBar().showMessage("Cancelling...")
            self.worker.cancel()

    def _on_progress(self, done: int, total: int):
        """Handle progress update."""
        self.control_panel.set_progress(done, total)

    def log(self, text: str):
        """Write to the diagnostic console."""
        self.log_console.append(text)
        # Also print to terminal
        print(text, flush=True)

    def _on_finished(self, result: HomingResult):
        """Handle homing completion."""
        self.control_panel.set_running(False)

        # Store result for re-filtering when frequency buttons are clicked
        self.last_result = result

        # Calculate SNR for winners FIRST
        if result.num_winners > 0:
            self._calculate_snr_for_result(result)

        # Log diagnostics to console (now with SNR)
        paths_with_data = sum(1 for w in result.winner_triplets
                              if w.ray_path and w.ray_path.states)
        self.log("=" * 50)
        self.log(f"HOMING COMPLETE: {result.num_winners} winners, {paths_with_data} with path data")
        if result.winner_triplets:
            # Sort by SNR descending
            sorted_winners = sorted(result.winner_triplets,
                                   key=lambda w: w.snr_db if w.snr_db else -999,
                                   reverse=True)
            for w in sorted_winners[:10]:
                has_path = "✓" if (w.ray_path and w.ray_path.states) else "✗"
                snr = f"SNR={w.snr_db:.0f}dB" if w.snr_db else "SNR=?"
                sig = f"Sig={w.signal_strength_dbm:.0f}dBm" if w.signal_strength_dbm else ""
                self.log(f"  {w.frequency_mhz:.1f}MHz el={w.elevation_deg:.0f}° {w.mode.value} {snr} {sig} path={has_path}")

        self.control_panel.show_result(result)

        # Update visualization
        self.viz_panel.update_from_homing_result(result)

        # Status - show clear failure or success message
        if result.num_winners == 0:
            msg = (f"NO PATHS FOUND - {result.total_rays_traced} rays traced, "
                   f"range {result.great_circle_range_km:.0f} km - "
                   f"try lower freq or higher foF2")
            self.statusBar().setStyleSheet("background-color: #883333; color: white;")
            logger.warning(f"Homing FAILED: no paths found for {result.great_circle_range_km:.0f} km range")
        else:
            # Include SNR info in status
            best_snr = max((w.snr_db for w in result.winner_triplets if w.snr_db is not None), default=None)
            snr_info = f", Best SNR: {best_snr:.0f} dB" if best_snr is not None else ""
            msg = (f"Found {result.num_winners} winner triplets - "
                   f"MUF: {result.muf:.1f} MHz, LUF: {result.luf:.1f} MHz{snr_info}")
            self.statusBar().setStyleSheet("")  # Reset to default
            logger.info(f"Homing complete: {result}")

        self.statusBar().showMessage(msg)

    def _calculate_snr_for_result(self, result: HomingResult):
        """Calculate SNR for all winners using physics-based link budget.

        SNR is computed from first principles:
        - Free space path loss (FSPL)
        - D-layer absorption (ITU-R P.533)
        - Ground reflection losses
        - Atmospheric noise (ITU-R P.372)

        Winners with invalid/uncomputable SNR are REMOVED from the result.
        """
        import numpy as np
        from src.raytracer.link_budget import (
            LinkBudgetCalculator,
            TransmitterConfig,
            ReceiverConfig,
            AntennaConfig,
            NoiseEnvironment,
            calculate_solar_zenith_angle,
            is_nighttime,
        )

        # Get radio config from UI
        tx_power = self.control_panel.tx_power.value()
        tx_gain = self.control_panel.tx_antenna_gain.value()
        rx_gain = self.control_panel.rx_antenna_gain.value()
        rx_bw = float(self.control_panel.rx_bandwidth.currentText())

        self.log(f"Link Budget: TX={tx_power:.0f}W (+{tx_gain:.1f}dBi), RX={rx_gain:.1f}dBi, BW={rx_bw:.0f}Hz")

        # Space weather
        xray_flux = 1e-6
        kp_index = 2.0
        if self.live_client and hasattr(self.live_client, 'state'):
            xray_flux = getattr(self.live_client.state, 'xray_flux', 1e-6)
            kp_index = getattr(self.live_client.state, 'kp_index', 2.0)

        # Path geometry
        mid_lat = (result.tx_position[0] + result.rx_position[0]) / 2
        mid_lon = (result.tx_position[1] + result.rx_position[1]) / 2
        gc_range = result.great_circle_range_km

        try:
            is_night = is_nighttime(mid_lat, mid_lon)
        except Exception:
            is_night = False

        try:
            solar_zenith = calculate_solar_zenith_angle(mid_lat, mid_lon)
        except Exception:
            solar_zenith = 45.0

        self.log(f"Path: {gc_range:.0f}km, Solar zenith={solar_zenith:.0f}°, night={is_night}")

        calc = LinkBudgetCalculator()
        tx_config = TransmitterConfig(power_watts=tx_power, antenna=AntennaConfig(gain_dbi=tx_gain))
        rx_config = ReceiverConfig(antenna=AntennaConfig(gain_dbi=rx_gain), bandwidth_hz=rx_bw,
                                   noise_environment=NoiseEnvironment.RURAL)

        # Process winners - REMOVE any that fail SNR calculation
        valid_winners = []
        invalid_count = 0

        for idx, winner in enumerate(result.winner_triplets):
            # Extract and validate physical parameters
            freq = winner.frequency_mhz
            hop_count = max(winner.hop_count, 1)

            # Reflection height from ray path (actual physics)
            h = winner.reflection_height_km
            if h <= 0 and winner.ray_path and winner.ray_path.states:
                h = max(s.altitude() for s in winner.ray_path.states)
            if h <= 0:
                # Cannot compute without reflection height - skip this winner
                self.log(f"  #{idx}: {freq:.1f}MHz SKIPPED - no reflection height data")
                invalid_count += 1
                continue

            # Ground range from ray path or result
            ground_range = winner.ground_range_km
            if ground_range <= 0 and winner.ray_path and winner.ray_path.states:
                states = winner.ray_path.states
                if len(states) >= 2:
                    lat1, lon1, _ = states[0].lat_lon_alt()
                    lat2, lon2, _ = states[-1].lat_lon_alt()
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    ground_range = 6371.0 * c
            if ground_range <= 0:
                ground_range = gc_range
            if ground_range <= 0:
                self.log(f"  #{idx}: {freq:.1f}MHz SKIPPED - no range data")
                invalid_count += 1
                continue

            # Calculate path length from geometry (physics)
            hop_range = ground_range / hop_count
            path_per_hop = 2 * np.sqrt((hop_range/2)**2 + h**2)
            path_length = path_per_hop * hop_count

            # Compute SNR using physics-based link budget
            try:
                link_result = calc.calculate(
                    frequency_mhz=freq,
                    path_length_km=path_length,
                    hop_count=hop_count,
                    reflection_height_km=h,
                    solar_zenith_angle_deg=solar_zenith,
                    xray_flux=xray_flux,
                    kp_index=kp_index,
                    tx_config=tx_config,
                    rx_config=rx_config,
                    latitude_deg=mid_lat,
                    is_night=is_night,
                )

                snr = link_result.snr_db

                # Validate: SNR must be a real number from physics
                if np.isnan(snr) or np.isinf(snr):
                    raise ValueError(f"Non-physical SNR: {snr}")

                # Filter: SNR must be usable for practical communication
                # User-configurable threshold: -20 dB for FT8, 0 dB for CW, 10+ dB for voice
                snr_cutoff = self.control_panel.snr_cutoff.value()
                if snr < snr_cutoff:
                    self.log(f"  #{idx}: {freq:.1f}MHz REJECTED - SNR {snr:.0f}dB below {snr_cutoff:.0f}dB cutoff")
                    invalid_count += 1
                    continue

                winner.snr_db = snr
                winner.signal_strength_dbm = link_result.signal_power_dbw + 30
                winner.path_loss_db = link_result.total_path_loss_db
                valid_winners.append(winner)

                if idx < 5:
                    self.log(f"  #{idx}: {freq:.1f}MHz path={path_length:.0f}km h={h:.0f}km "
                             f"loss={link_result.total_path_loss_db:.0f}dB SNR={snr:.0f}dB")

            except Exception as e:
                self.log(f"  #{idx}: {freq:.1f}MHz FAILED: {e}")
                invalid_count += 1
                continue

        # Replace winner list with only valid winners
        result.winner_triplets[:] = valid_winners

        if invalid_count > 0:
            self.log(f"Removed {invalid_count} winners with uncomputable SNR")

    def _on_cancelled(self):
        """Handle homing cancellation."""
        self.control_panel.set_running(False)
        self.control_panel.progress.setValue(0)
        self.statusBar().showMessage("Homing algorithm cancelled")
        logger.info("Homing cancelled by user")

    def _on_error(self, error_msg: str):
        """Handle homing error."""
        self.control_panel.set_running(False)
        self.statusBar().showMessage(f"Error: {error_msg}")
        QMessageBox.critical(self, "Homing Error", error_msg)

    def closeEvent(self, event):
        """Handle window close - ensure all workers are killed."""
        logger.info("Window closing - cleaning up...")

        # Stop visualization timers
        if hasattr(self, 'viz_panel') and self.viz_panel is not None:
            self.viz_panel.cleanup()

        # Stop live data client
        self._stop_live_client()

        # Cancel and cleanup homing worker
        if self.worker is not None:
            logger.info("Terminating homing worker...")
            self.worker.cancel()
            self.worker.quit()
            if not self.worker.wait(2000):  # Wait 2 seconds
                logger.warning("Worker didn't stop, terminating...")
                self.worker.terminate()
                self.worker.wait(1000)

        # Gracefully terminate remaining child processes to avoid semaphore leaks
        import os
        import time
        try:
            import psutil
            current_pid = os.getpid()
            current_process = psutil.Process(current_pid)
            children = current_process.children(recursive=True)

            # First, send SIGTERM to allow graceful cleanup
            for child in children:
                try:
                    logger.info(f"Terminating child process {child.pid}")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Wait briefly for graceful shutdown
            time.sleep(0.5)

            # Kill any remaining processes
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    if child.is_running():
                        logger.info(f"Force killing child process {child.pid}")
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            # psutil not available, try basic approach
            pass
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")

        logger.info("Cleanup complete")
        super().closeEvent(event)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IONORT Live Ray Tracing Dashboard with Real-Time Data"
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
        help="F2 critical frequency MHz (default: 7.0, ignored if --live)"
    )
    parser.add_argument(
        "--hmF2", type=float, default=300.0,
        help="F2 peak height km (default: 300, ignored if --live)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live ionospheric data from GIRO network"
    )
    parser.add_argument(
        "--simulated", action="store_true",
        help="Use simulated live data (for testing without network)"
    )
    # Performance tuning
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    parser.add_argument(
        "--step-km", type=float, default=1.0,
        help="Ray integration step size in km (default: 1.0, larger=faster but less accurate)"
    )
    parser.add_argument(
        "--freq-step", type=float, default=1.0,
        help="Frequency search step in MHz (default: 1.0)"
    )
    parser.add_argument(
        "--elev-step", type=float, default=10.0,
        help="Elevation search step in degrees (default: 10.0)"
    )
    parser.add_argument(
        "--max-hops", type=int, default=3,
        help="Maximum ground reflections for multi-hop paths (default: 3)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=100.0,
        help="Landing tolerance in km (default: 100, try 200-500 for long paths)"
    )
    parser.add_argument(
        "--snr-cutoff", type=float, default=0.0,
        help="Minimum SNR threshold in dB (default: 0, range: -20 to 60). "
             "Use -20 for FT8/digital, 0 for CW, 10+ for voice"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse coordinates
    tx_lat, tx_lon = map(float, args.tx.split(","))
    rx_lat, rx_lon = map(float, args.rx.split(","))
    freq_min, freq_max = map(float, args.freq.split(","))

    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    num_workers = args.workers if args.workers else cpu_count

    # Create application
    app = QApplication(sys.argv)

    # Create and configure window
    window = IONORTLiveWindow(
        use_live=args.live,
        use_simulated=args.simulated
    )

    # Store runtime params for use in homing
    window.step_km = args.step_km
    window.max_hops = args.max_hops

    # Apply command line args to UI
    window.control_panel.tx_lat.setValue(tx_lat)
    window.control_panel.tx_lon.setValue(tx_lon)
    window.control_panel.rx_lat.setValue(rx_lat)
    window.control_panel.rx_lon.setValue(rx_lon)
    window.control_panel.freq_min.setValue(freq_min)
    window.control_panel.freq_max.setValue(freq_max)
    window.control_panel.freq_step.setValue(args.freq_step)
    window.control_panel.elev_step.setValue(args.elev_step)
    window.control_panel.foF2.setValue(args.foF2)
    window.control_panel.hmF2.setValue(args.hmF2)

    # Set workers (allow up to 64 in UI, default to num_workers)
    window.control_panel.workers.setRange(1, 64)
    window.control_panel.workers.setValue(num_workers)

    # Set tolerance from command line
    window.control_panel.tolerance.setValue(args.tolerance)

    # Set SNR cutoff from command line (clamp to valid range)
    snr_cutoff = max(-20.0, min(60.0, args.snr_cutoff))
    window.control_panel.snr_cutoff.setValue(snr_cutoff)

    # NVIS mode adjustments
    if args.nvis:
        window.control_panel.elev_min.setValue(60.0)
        window.control_panel.elev_max.setValue(89.0)
        window.control_panel.elev_step.setValue(2.0)

    window.show()

    logger.info("IONORT Live Dashboard started")
    logger.info(f"Tx: ({tx_lat}, {tx_lon}), Rx: ({rx_lat}, {rx_lon})")
    logger.info(f"Frequency: {freq_min}-{freq_max} MHz, step={args.freq_step} MHz")
    logger.info(f"Elevation step: {args.elev_step}°")
    logger.info(f"Workers: {num_workers} (CPUs: {cpu_count})")
    logger.info(f"Integration step: {args.step_km} km")
    logger.info(f"Max hops: {args.max_hops}")
    logger.info(f"Landing tolerance: {args.tolerance} km")
    if args.live:
        logger.info("Live ionospheric data ENABLED")
        if args.simulated:
            logger.info("Using SIMULATED live data")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
