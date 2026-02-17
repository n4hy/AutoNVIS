#!/usr/bin/env python3
"""
Ray Tracer Visualization Display

PyQt6 application for visualizing HF ray tracing through the ionosphere.
Shows ray paths, electron density profiles, and NVIS optimization results.

Usage:
    python -m src.raytracer.display
"""

import sys
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QGroupBox,
    QSpinBox, QDoubleSpinBox, QSplitter, QStatusBar, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from .electron_density import IonosphericModel, create_test_profile
from .haselgrove import HaselgroveSolver, RayPath, RayMode, RayTermination


class RayTraceWorker(QThread):
    """Background worker for ray tracing."""

    ray_complete = pyqtSignal(object, str)  # path, color
    all_complete = pyqtSignal(int, int)  # total, ground_hits
    progress = pyqtSignal(int, int)  # current, total

    def __init__(self, solver, params_list):
        super().__init__()
        self.solver = solver
        self.params_list = params_list  # List of (freq, elev, azim, lat, lon, color)

    def run(self):
        ground_hits = 0
        total = len(self.params_list)

        for i, (freq, elev, azim, lat, lon, color) in enumerate(self.params_list):
            path = self.solver.trace_ray(
                tx_lat=lat, tx_lon=lon, tx_alt=0.0,
                elevation=elev, azimuth=azim,
                frequency_mhz=freq,
                mode=RayMode.ORDINARY,
                step_km=3.0,  # Larger steps for speed
                max_path_km=2000.0,  # Shorter max path
            )

            if path.termination == RayTermination.GROUND_HIT:
                ground_hits += 1

            self.ray_complete.emit(path, color)
            self.progress.emit(i + 1, total)

        self.all_complete.emit(total, ground_hits)


class ElectronDensityWidget(QWidget):
    """Widget showing electron density profile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Electron Density Profile")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#1a1a2e')
        self.plot.setLabel('left', 'Altitude (km)')
        self.plot.setLabel('bottom', 'Ne (el/cm³)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLogMode(x=True, y=False)
        layout.addWidget(self.plot)

        # Curve
        self.curve = self.plot.plot(pen=pg.mkPen('#00ffff', width=2))

        # F2 peak marker
        self.f2_marker = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen('#ff0000'), brush=pg.mkBrush('#ff0000')
        )
        self.plot.addItem(self.f2_marker)

        # Info
        info_layout = QHBoxLayout()
        self.fof2_label = QLabel("foF2: --")
        self.fof2_label.setStyleSheet("color: #00ffff;")
        self.hmf2_label = QLabel("hmF2: --")
        self.hmf2_label.setStyleSheet("color: #ffff00;")
        info_layout.addWidget(self.fof2_label)
        info_layout.addWidget(self.hmf2_label)
        layout.addLayout(info_layout)

    def update_profile(self, model: IonosphericModel, lat: float, lon: float):
        """Update the electron density profile display."""
        altitudes = np.linspace(80, 450, 80)
        densities = np.array([
            model.get_electron_density(lat, lon, alt) for alt in altitudes
        ])

        self.curve.setData(densities, altitudes)

        # Find and mark F2 peak
        peak_idx = np.argmax(densities)
        peak_alt = altitudes[peak_idx]
        peak_ne = densities[peak_idx]
        self.f2_marker.setData([peak_ne], [peak_alt])

        # Update labels
        status = model.get_correction_status()
        foF2 = status.get('current_foF2', 0)
        hmF2 = status.get('current_hmF2', 0)
        self.fof2_label.setText(f"foF2: {foF2:.1f} MHz")
        self.hmf2_label.setText(f"hmF2: {hmF2:.0f} km")


class RayPathWidget(QWidget):
    """Widget showing ray paths through ionosphere."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rays = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Ray Paths (Altitude vs Ground Range)")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#1a1a2e')
        self.plot.setLabel('left', 'Altitude (km)')
        self.plot.setLabel('bottom', 'Ground Range (km)')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setXRange(0, 600)
        self.plot.setYRange(0, 350)
        layout.addWidget(self.plot)

        # Ground line
        ground = self.plot.plot([0, 1000], [0, 0], pen=pg.mkPen('#00ff00', width=3))

        # Ionosphere shading
        self.iono_fill = pg.FillBetweenItem(
            pg.PlotDataItem([0, 1000], [150, 150]),
            pg.PlotDataItem([0, 1000], [350, 350]),
            brush=pg.mkBrush(50, 50, 150, 40)
        )
        self.plot.addItem(self.iono_fill)

        # Legend/info
        self.info_label = QLabel("Click 'Trace Ray' or 'Trace Fan' to begin")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.info_label)

    def clear_rays(self):
        for ray in self.rays:
            self.plot.removeItem(ray)
        self.rays.clear()
        self.info_label.setText("Cleared")

    def add_ray(self, path: RayPath, color: str = '#ffff00'):
        if not path.states or len(path.states) < 2:
            return

        # Calculate ground range incrementally
        ranges = [0.0]
        altitudes = [path.states[0].altitude()]

        for i in range(1, len(path.states)):
            s0 = path.states[i-1]
            s1 = path.states[i]

            lat0, lon0, alt0 = s0.lat_lon_alt()
            lat1, lon1, alt1 = s1.lat_lon_alt()

            # Approximate horizontal distance
            dlat = (lat1 - lat0) * 111.0  # km per degree
            dlon = (lon1 - lon0) * 111.0 * np.cos(np.radians(lat0))
            d_horiz = np.sqrt(dlat**2 + dlon**2)

            ranges.append(ranges[-1] + d_horiz)
            altitudes.append(alt1)

        # Plot
        curve = self.plot.plot(ranges, altitudes, pen=pg.mkPen(color, width=2))
        self.rays.append(curve)

        # Update info
        max_alt = max(altitudes)
        final_range = ranges[-1]
        term = path.termination.value

        self.info_label.setText(
            f"{path.frequency_mhz:.1f} MHz @ {path.start_direction[0]:.0f}° → "
            f"{term}, range={final_range:.0f} km, peak={max_alt:.0f} km"
        )


class ControlPanel(QWidget):
    """Control panel for ray tracing parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Ionosphere
        iono_group = QGroupBox("Ionosphere")
        iono_layout = QGridLayout(iono_group)

        iono_layout.addWidget(QLabel("foF2 (MHz):"), 0, 0)
        self.fof2_spin = QDoubleSpinBox()
        self.fof2_spin.setRange(3.0, 15.0)
        self.fof2_spin.setValue(8.0)
        self.fof2_spin.setSingleStep(0.5)
        iono_layout.addWidget(self.fof2_spin, 0, 1)

        iono_layout.addWidget(QLabel("hmF2 (km):"), 1, 0)
        self.hmf2_spin = QSpinBox()
        self.hmf2_spin.setRange(200, 400)
        self.hmf2_spin.setValue(300)
        iono_layout.addWidget(self.hmf2_spin, 1, 1)

        layout.addWidget(iono_group)

        # Ray parameters
        ray_group = QGroupBox("Ray Parameters")
        ray_layout = QGridLayout(ray_group)

        ray_layout.addWidget(QLabel("Freq (MHz):"), 0, 0)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(2.0, 20.0)
        self.freq_spin.setValue(6.0)
        self.freq_spin.setSingleStep(0.5)
        ray_layout.addWidget(self.freq_spin, 0, 1)

        ray_layout.addWidget(QLabel("Elevation (°):"), 1, 0)
        self.elev_spin = QSpinBox()
        self.elev_spin.setRange(10, 90)
        self.elev_spin.setValue(60)
        ray_layout.addWidget(self.elev_spin, 1, 1)

        layout.addWidget(ray_group)

        # Buttons
        self.trace_btn = QPushButton("Trace Single Ray")
        self.trace_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 8px; font-weight: bold; }"
        )
        layout.addWidget(self.trace_btn)

        self.fan_btn = QPushButton("Trace Fan (5 rays)")
        self.fan_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 8px; font-weight: bold; }"
        )
        layout.addWidget(self.fan_btn)

        self.clear_btn = QPushButton("Clear All")
        layout.addWidget(self.clear_btn)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)

        self.nvis_btn = QPushButton("NVIS (5 MHz, 80°)")
        self.nvis_btn.setStyleSheet("color: #00ffff;")
        preset_layout.addWidget(self.nvis_btn)

        self.skip_btn = QPushButton("Show Skip Zone")
        self.skip_btn.setStyleSheet("color: #ffaa00;")
        preset_layout.addWidget(self.skip_btn)

        self.reflect_btn = QPushButton("Reflect vs Escape")
        self.reflect_btn.setStyleSheet("color: #ff66ff;")
        preset_layout.addWidget(self.reflect_btn)

        layout.addWidget(preset_group)
        layout.addStretch()


class RayTracerDisplay(QMainWindow):
    """Main ray tracer visualization window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoNVIS Ray Tracer")
        self.setMinimumSize(1100, 650)
        self.setStyleSheet("background-color: #0f0f1a; color: #ffffff;")

        # Model and solver
        self.model = create_test_profile()
        self.model.update_from_realtime(foF2=8.0, hmF2=300.0)
        self.solver = HaselgroveSolver(self.model)

        self.worker = None

        self.setup_ui()
        self.connect_signals()
        self.update_display()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Controls
        self.controls = ControlPanel()
        self.controls.setMaximumWidth(220)
        layout.addWidget(self.controls)

        # Plots
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.ray_widget = RayPathWidget()
        splitter.addWidget(self.ray_widget)

        self.density_widget = ElectronDensityWidget()
        splitter.addWidget(self.density_widget)

        splitter.setSizes([400, 250])
        layout.addWidget(splitter, stretch=1)

        # Status
        self.status = QStatusBar()
        self.status.setStyleSheet("color: #aaaaaa;")
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def connect_signals(self):
        self.controls.fof2_spin.valueChanged.connect(self.on_iono_changed)
        self.controls.hmf2_spin.valueChanged.connect(self.on_iono_changed)

        self.controls.trace_btn.clicked.connect(self.trace_single)
        self.controls.fan_btn.clicked.connect(self.trace_fan)
        self.controls.clear_btn.clicked.connect(self.clear_rays)

        self.controls.nvis_btn.clicked.connect(self.preset_nvis)
        self.controls.skip_btn.clicked.connect(self.preset_skip)
        self.controls.reflect_btn.clicked.connect(self.preset_reflect)

    def on_iono_changed(self):
        foF2 = self.controls.fof2_spin.value()
        hmF2 = self.controls.hmf2_spin.value()
        self.model.update_from_realtime(foF2=foF2, hmF2=hmF2)
        self.update_display()

    def update_display(self):
        self.density_widget.update_profile(self.model, 40.0, -105.0)

    def trace_single(self):
        freq = self.controls.freq_spin.value()
        elev = self.controls.elev_spin.value()

        self.status.showMessage("Tracing...")
        QApplication.processEvents()

        path = self.solver.trace_ray(
            tx_lat=40.0, tx_lon=-105.0, tx_alt=0.0,
            elevation=elev, azimuth=0.0,
            frequency_mhz=freq,
            mode=RayMode.ORDINARY,
            step_km=2.0,
            max_path_km=2000.0,
        )

        self.ray_widget.add_ray(path, '#ffff00')
        self.status.showMessage(f"Ray traced: {path.termination.value}")

    def trace_fan(self):
        freq = self.controls.freq_spin.value()

        # Prepare ray parameters - just 5 rays for speed
        params = []
        colors = ['#ff0000', '#ff8800', '#ffff00', '#00ff00', '#00ffff']
        elevations = [20, 35, 50, 65, 80]

        for elev, color in zip(elevations, colors):
            params.append((freq, elev, 0.0, 40.0, -105.0, color))

        self.controls.progress.setVisible(True)
        self.controls.progress.setMaximum(len(params))
        self.controls.progress.setValue(0)
        self.controls.fan_btn.setEnabled(False)
        self.controls.trace_btn.setEnabled(False)

        self.worker = RayTraceWorker(self.solver, params)
        self.worker.ray_complete.connect(self.on_ray_complete)
        self.worker.progress.connect(self.on_progress)
        self.worker.all_complete.connect(self.on_fan_complete)
        self.worker.start()

    def on_ray_complete(self, path, color):
        self.ray_widget.add_ray(path, color)

    def on_progress(self, current, total):
        self.controls.progress.setValue(current)
        self.status.showMessage(f"Tracing ray {current}/{total}...")

    def on_fan_complete(self, total, ground_hits):
        self.controls.progress.setVisible(False)
        self.controls.fan_btn.setEnabled(True)
        self.controls.trace_btn.setEnabled(True)
        self.status.showMessage(f"Fan complete: {ground_hits}/{total} rays hit ground")

    def clear_rays(self):
        self.ray_widget.clear_rays()
        self.status.showMessage("Cleared")

    def preset_nvis(self):
        """NVIS preset - high angle, low frequency."""
        self.controls.freq_spin.setValue(5.0)
        self.controls.elev_spin.setValue(80)
        self.controls.fof2_spin.setValue(8.0)
        self.on_iono_changed()
        self.clear_rays()
        self.trace_single()

    def preset_skip(self):
        """Show skip zone with fan of rays."""
        self.controls.freq_spin.setValue(7.0)
        self.controls.fof2_spin.setValue(10.0)
        self.on_iono_changed()
        self.clear_rays()
        self.trace_fan()

    def preset_reflect(self):
        """Show reflection vs escape comparison."""
        self.clear_rays()
        self.controls.fof2_spin.setValue(8.0)
        self.on_iono_changed()

        # Below foF2 - will reflect
        path1 = self.solver.trace_ray(
            tx_lat=40.0, tx_lon=-105.0, tx_alt=0.0,
            elevation=45, azimuth=0.0, frequency_mhz=6.0,
            mode=RayMode.ORDINARY, step_km=2.0,
        )
        self.ray_widget.add_ray(path1, '#00ff00')

        # Above foF2 - will escape
        path2 = self.solver.trace_ray(
            tx_lat=40.0, tx_lon=-105.0, tx_alt=0.0,
            elevation=45, azimuth=0.0, frequency_mhz=12.0,
            mode=RayMode.ORDINARY, step_km=2.0,
        )
        self.ray_widget.add_ray(path2, '#ff0000')

        self.status.showMessage("Green=6MHz (reflects), Red=12MHz (escapes)")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(15, 15, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 40))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(40, 40, 60))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = RayTracerDisplay()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
