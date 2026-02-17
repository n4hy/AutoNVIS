"""
HF Ray Tracer Main Window

Visualizes ionospheric ray tracing for NVIS propagation analysis.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QToolBar, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
import pyqtgraph as pg
import numpy as np
import logging
from typing import Dict, Optional

from .widgets import FrequencyIndicator, RayPathWidget, CoverageMapWidget, ControlPanel


class RayTraceWorker(QThread):
    """Worker thread for ray tracing calculations."""

    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, params: Dict, parent=None):
        super().__init__(parent)
        self.params = params
        self.logger = logging.getLogger("raytracer")

    def run(self):
        """Run ray tracing calculation."""
        try:
            self.logger.info("Starting ray trace calculation...")
            result = self._run_demo_trace()
            self.logger.info("Ray trace calculation complete, emitting result...")
            self.finished.emit(result)
            self.logger.info("Result emitted")
        except Exception as e:
            self.logger.error(f"Ray trace error: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _run_demo_trace(self) -> Dict:
        """Run demo ray trace with Chapman layer model."""
        self.logger.info("Creating Chapman layer ionosphere...")

        # Create Chapman layer ionosphere
        lat = np.linspace(-20, 20, 15)
        lon = np.linspace(-20, 20, 15)
        alt = np.linspace(60, 600, 30)

        NmF2 = self.params.get('nmf2', 5e11)
        hmF2 = self.params.get('hmf2', 300.0)
        H = 80.0  # Scale height

        self.logger.info(f"NmF2={NmF2}, hmF2={hmF2}")

        # Create electron density grid
        n_lat, n_lon, n_alt = len(lat), len(lon), len(alt)
        ne_grid = np.zeros((n_lat, n_lon, n_alt))

        for i in range(n_lat):
            for j in range(n_lon):
                for k in range(n_alt):
                    z = (alt[k] - hmF2) / H
                    ne_grid[i, j, k] = NmF2 * np.exp(0.5 * (1 - z - np.exp(-z)))

        self.logger.info("Grid created, running simulation...")

        # Always use simulation for now (C++ module may cause segfaults)
        self.logger.info("Using simulated ray tracing")
        return self._simulate_trace(NmF2, hmF2)

    def _trace_with_engine(self, tracer) -> Dict:
        """Trace with real ray tracer engine."""
        tx_lat = self.params['tx_lat']
        tx_lon = self.params['tx_lon']
        freq_min = self.params['freq_min']
        freq_max = self.params['freq_max']
        freq_step = self.params['freq_step']

        # Calculate coverage
        coverage = tracer.calculate_coverage(
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_step=freq_step
        )

        # Trace rays for visualization
        all_paths = []
        frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
        for freq in frequencies:
            paths = tracer.trace_nvis(tx_lat, tx_lon, freq,
                                      elevation_step=5.0, azimuth_step=45.0)
            for p in paths:
                p['freq_mhz'] = freq
            all_paths.extend(paths)

        return {
            'luf': coverage['luf'],
            'muf': coverage['muf'],
            'fot': coverage['optimal_freq'],
            'blackout': coverage['blackout'],
            'paths': all_paths,
            'tx_lat': tx_lat,
            'tx_lon': tx_lon,
        }

    def _simulate_trace(self, NmF2: float, hmF2: float) -> Dict:
        """Simulate ray trace results for demo purposes."""
        self.logger.info("Starting simulation...")

        tx_lat = self.params['tx_lat']
        tx_lon = self.params['tx_lon']
        freq_min = self.params['freq_min']
        freq_max = self.params['freq_max']
        freq_step = self.params['freq_step']

        self.logger.info(f"Params: tx=({tx_lat}, {tx_lon}), freq={freq_min}-{freq_max}")

        # Calculate critical frequency from NmF2
        # foF2 = 9 * sqrt(NmF2) / 1e6 (MHz)
        foF2 = 9.0 * np.sqrt(NmF2) / 1e6
        self.logger.info(f"foF2 = {foF2:.2f} MHz")

        # MUF ~ foF2 * sec(zenith) ~ foF2 * 1.2 for NVIS
        muf = foF2 * 1.15

        # LUF depends on absorption (simplified model)
        luf = max(freq_min, muf * 0.3)

        # FOT = 0.85 * MUF
        fot = 0.85 * muf

        self.logger.info(f"LUF={luf:.2f}, MUF={muf:.2f}, FOT={fot:.2f}")

        # Generate simulated ray paths
        paths = []
        frequencies = list(np.arange(freq_min, min(freq_max, muf + 2) + freq_step, freq_step))
        self.logger.info(f"Simulating {len(frequencies)} frequencies...")

        for freq in frequencies:
            for elev in [75, 80, 85, 90]:  # Simplified
                for azim in [0, 90, 180, 270]:  # Simplified
                    # Simple ray path simulation
                    path = self._simulate_single_ray(
                        tx_lat, tx_lon, freq, elev, azim, hmF2, foF2
                    )
                    paths.append(path)

        self.logger.info(f"Generated {len(paths)} paths")

        result = {
            'luf': float(luf),
            'muf': float(muf),
            'fot': float(fot),
            'blackout': luf > muf,
            'paths': paths,
            'tx_lat': float(tx_lat),
            'tx_lon': float(tx_lon),
            'simulated': True,
        }

        self.logger.info("Simulation complete")
        return result

    def _simulate_single_ray(self, tx_lat: float, tx_lon: float,
                              freq: float, elevation: float, azimuth: float,
                              hmF2: float, foF2: float) -> Dict:
        """Simulate a single ray path."""
        # Will this ray reflect?
        # Simplified: reflects if freq < MUF for this angle
        cos_val = np.cos(np.radians(90 - elevation))
        if abs(cos_val) < 0.01:
            cos_val = 0.01
        muf_angle = foF2 / cos_val
        reflected = freq < muf_angle

        # Generate path points - use simple Python lists to avoid numpy issues
        n_points = 20

        if reflected:
            # Parabolic path
            max_alt = hmF2 + (90 - elevation) * 2
            ground_range_max = (90 - elevation) * 8  # km

            positions = []
            for i in range(n_points):
                t = i / (n_points - 1)
                alt = max_alt * 4 * t * (1 - t)  # Parabola
                rng = ground_range_max * t

                # Convert range to lat/lon offset
                dlat = rng * np.cos(np.radians(azimuth)) / 111.0
                dlon = rng * np.sin(np.radians(azimuth)) / 111.0

                positions.append([tx_lat + dlat, tx_lon + dlon, alt])

            apex_alt = max_alt
            apex_lat = tx_lat + (ground_range_max * 0.5 * np.cos(np.radians(azimuth)) / 111.0)
            apex_lon = tx_lon + (ground_range_max * 0.5 * np.sin(np.radians(azimuth)) / 111.0)
            ground_range = ground_range_max

        else:
            # Escaping ray - goes straight up
            positions = []
            for i in range(n_points):
                alt = 800 * i / (n_points - 1)
                positions.append([tx_lat, tx_lon, alt])

            apex_alt = 800
            apex_lat = tx_lat
            apex_lon = tx_lon
            ground_range = 0

        # Simulated absorption (higher at lower frequencies)
        absorption = max(0, (10 - freq) * 5)

        return {
            'positions': positions,  # Now a list of lists, not numpy array
            'freq_mhz': float(freq),
            'elevation': float(elevation),
            'azimuth': float(azimuth),
            'reflected': bool(reflected),
            'escaped': not reflected,
            'absorbed': False,
            'apex_altitude': float(apex_alt),
            'apex_lat': float(apex_lat),
            'apex_lon': float(apex_lon),
            'ground_range': float(ground_range),
            'absorption_db': float(absorption),
        }


class RayTracerMainWindow(QMainWindow):
    """
    Main window for HF Ray Tracer visualization.

    Layout:
    +----------------------------------------------------------+
    | Toolbar                                                   |
    +----------------------------------------------------------+
    | +--------+ +------------------------------------------+  |
    | |        | |                                          |  |
    | | Control| |        Frequency Indicator               |  |
    | | Panel  | |                                          |  |
    | |        | +------------------------------------------+  |
    | |        | +------------------------------------------+  |
    | |        | |                                          |  |
    | |        | |        Ray Path Display                  |  |
    | |        | |                                          |  |
    | |        | +------------------------------------------+  |
    | |        | +------------------------------------------+  |
    | |        | |                                          |  |
    | |        | |        Coverage Map                      |  |
    | |        | |                                          |  |
    | +--------+ +------------------------------------------+  |
    +----------------------------------------------------------+
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger("raytracer")
        self.setWindowTitle("AutoNVIS HF Ray Tracer")
        self.setMinimumSize(1200, 900)

        pg.setConfigOptions(antialias=True)

        self.worker: Optional[RayTraceWorker] = None

        self._setup_ui()
        self._setup_toolbar()
        self._apply_dark_theme()

        # Don't run initial trace - let user click the button

    def _setup_ui(self):
        """Create and arrange all UI elements."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: Control panel
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(250)
        self.control_panel.trace_requested.connect(self._on_trace_requested)
        layout.addWidget(self._wrap_widget(self.control_panel))

        # Right: Visualization panels
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        # Frequency indicator (top)
        self.freq_indicator = FrequencyIndicator()
        self.freq_indicator.setFixedHeight(180)
        right_layout.addWidget(self._wrap_widget(self.freq_indicator))

        # Ray paths (middle)
        self.ray_path_widget = RayPathWidget()
        right_layout.addWidget(self._wrap_widget(self.ray_path_widget), stretch=1)

        # Coverage map (bottom)
        self.coverage_widget = CoverageMapWidget()
        right_layout.addWidget(self._wrap_widget(self.coverage_widget), stretch=1)

        layout.addLayout(right_layout, stretch=1)

        # Status bar
        self.statusBar().showMessage("Ready - Click 'Calculate Coverage' to start")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2d2d2d;
                color: #aaa;
            }
        """)

    def _wrap_widget(self, widget: QWidget) -> QFrame:
        """Wrap a widget in a styled frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.Box)
        frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(widget)

        return frame

    def _setup_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 5px;
                padding: 5px;
            }
            QToolButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 10px;
                color: #ddd;
            }
            QToolButton:hover {
                background-color: #4d4d4d;
            }
        """)
        self.addToolBar(toolbar)

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ddd;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 3px;
            }
        """)

    @pyqtSlot(dict)
    def _on_trace_requested(self, params: Dict):
        """Handle trace request from control panel."""
        if self.worker and self.worker.isRunning():
            return

        self.control_panel.set_busy(True)
        self.statusBar().showMessage("Calculating ray traces...")

        self.worker = RayTraceWorker(params)
        self.worker.finished.connect(self._on_trace_complete)
        self.worker.error.connect(self._on_trace_error)
        self.worker.start()

    @pyqtSlot(dict)
    def _on_trace_complete(self, result: Dict):
        """Handle trace completion."""
        try:
            self.logger.info("Trace complete, updating UI...")
            self.control_panel.set_busy(False)

            # Update frequency indicator
            self.logger.info(f"LUF={result['luf']}, MUF={result['muf']}, FOT={result['fot']}")
            self.freq_indicator.update_frequencies(
                result['luf'],
                result['muf'],
                result['fot'],
                result.get('blackout', False)
            )

            # Update ray path display
            self.logger.info("Clearing rays...")
            self.ray_path_widget.clear_rays()
            paths = result.get('paths', [])
            self.logger.info(f"Got {len(paths)} paths")

            # Group by frequency and show subset
            freq_paths = {}
            for p in paths:
                freq = p.get('freq_mhz', 0)
                if freq not in freq_paths:
                    freq_paths[freq] = []
                freq_paths[freq].append(p)

            # Show a few rays per frequency
            ray_count = 0
            for freq, freq_p in freq_paths.items():
                for p in freq_p[:4]:  # Max 4 rays per frequency
                    if 'positions' in p and p['positions'] is not None:
                        self.ray_path_widget.add_ray_path(
                            p['positions'],
                            freq,
                            p.get('reflected', False)
                        )
                        ray_count += 1

            self.logger.info(f"Added {ray_count} ray curves")

            # Update scale to fit all rays
            self.ray_path_widget.update_scale()

            n_reflected = sum(1 for p in paths if p.get('reflected', False))
            self.ray_path_widget.set_info(
                f"{len(paths)} rays traced, {n_reflected} reflected"
            )

            # Update coverage map
            self.logger.info("Updating coverage map...")
            self.coverage_widget.set_transmitter(result['tx_lat'], result['tx_lon'])
            self.coverage_widget.update_coverage(paths)

            # Status
            mode = "simulated" if result.get('simulated') else "ray traced"
            self.statusBar().showMessage(
                f"Complete ({mode}): LUF={result['luf']:.1f} MHz, "
                f"MUF={result['muf']:.1f} MHz, FOT={result['fot']:.1f} MHz"
            )
            self.logger.info("UI update complete")

        except Exception as e:
            self.logger.error(f"Error updating UI: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Error: {e}")

    @pyqtSlot(str)
    def _on_trace_error(self, error: str):
        """Handle trace error."""
        self.control_panel.set_busy(False)
        self.statusBar().showMessage(f"Error: {error}")
        QMessageBox.warning(self, "Ray Trace Error", error)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About HF Ray Tracer",
            "AutoNVIS HF Ray Tracer\n\n"
            "Ionospheric ray tracing for NVIS propagation analysis.\n\n"
            "Features:\n"
            "  - LUF/MUF calculation\n"
            "  - Ray path visualization\n"
            "  - NVIS coverage mapping\n\n"
            "Uses native C++ ray tracer (no MATLAB required).\n"
            "Falls back to simulated mode if C++ module not built.\n\n"
            "To build C++ ray tracer:\n"
            "  cd src/propagation\n"
            "  cmake -B build && cmake --build build\n\n"
            "Built with PyQt6 and pyqtgraph."
        )

    def closeEvent(self, event):
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1000)
        event.accept()


# Add default params method to ControlPanel
def _get_default_params(self):
    return {
        'tx_lat': self.tx_lat.value(),
        'tx_lon': self.tx_lon.value(),
        'freq_min': self.freq_min.value(),
        'freq_max': self.freq_max.value(),
        'freq_step': self.freq_step.value(),
        'nmf2': self.nmf2.value(),
        'hmf2': self.hmf2.value(),
    }

ControlPanel._get_default_params = _get_default_params
