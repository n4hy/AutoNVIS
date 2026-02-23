#!/usr/bin/env python3
"""
Simple IONORT Demo - All in one window, no separate classes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from src.raytracer import (
    IonosphericModel, HaselgroveSolver, HomingAlgorithm,
    HomingSearchSpace, HomingConfig
)
from src.visualization.pyqt.raytracer import IONORTVisualizationPanel


class Worker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, int)

    def __init__(self, tx_lat, tx_lon, rx_lat, rx_lon, foF2):
        super().__init__()
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        self.rx_lat = rx_lat
        self.rx_lon = rx_lon
        self.foF2 = foF2

    def run(self):
        print("Worker started")
        iono = IonosphericModel()
        iono.update_from_realtime(foF2=self.foF2)

        solver = HaselgroveSolver(iono)
        homing = HomingAlgorithm(solver, HomingConfig(
            store_ray_paths=True,
            max_workers=4,
            distance_tolerance_km=150.0,  # More forgiving landing tolerance
        ))

        # Simple search: 3 frequencies, 3 elevations, both modes = 18 rays
        search = HomingSearchSpace(
            freq_range=(5.0, 9.0),
            freq_step=2.0,
            elevation_range=(40.0, 60.0),
            elevation_step=10.0,
            azimuth_deviation_range=(0.0, 0.0),
            azimuth_step=10.0,
        )

        print(f"Tracing {search.total_triplets * 2} rays (O+X modes)...")

        def prog(done, total):
            self.progress.emit(done, total)

        result = homing.find_paths(
            self.tx_lat, self.tx_lon,
            self.rx_lat, self.rx_lon,
            search_space=search,
            progress_callback=prog
        )

        print(f"Done: {result.num_winners} winners")
        self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("IONORT Simple Demo")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: controls
        left = QWidget()
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Tx Lat:"))
        self.tx_lat = QDoubleSpinBox()
        self.tx_lat.setRange(-90, 90)
        self.tx_lat.setValue(40.0)
        left_layout.addWidget(self.tx_lat)

        left_layout.addWidget(QLabel("Tx Lon:"))
        self.tx_lon = QDoubleSpinBox()
        self.tx_lon.setRange(-180, 180)
        self.tx_lon.setValue(-105.0)
        left_layout.addWidget(self.tx_lon)

        left_layout.addWidget(QLabel("Rx Lat:"))
        self.rx_lat = QDoubleSpinBox()
        self.rx_lat.setRange(-90, 90)
        self.rx_lat.setValue(35.0)
        left_layout.addWidget(self.rx_lat)

        left_layout.addWidget(QLabel("Rx Lon:"))
        self.rx_lon = QDoubleSpinBox()
        self.rx_lon.setRange(-180, 180)
        self.rx_lon.setValue(-106.0)
        left_layout.addWidget(self.rx_lon)

        left_layout.addWidget(QLabel("foF2 (MHz):"))
        self.foF2 = QDoubleSpinBox()
        self.foF2.setRange(2, 15)
        self.foF2.setValue(7.0)
        left_layout.addWidget(self.foF2)

        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)

        self.run_btn = QPushButton("RUN")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4488ff;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        self.run_btn.clicked.connect(self.on_run)
        left_layout.addWidget(self.run_btn)

        self.status = QLabel("Ready")
        left_layout.addWidget(self.status)

        self.results = QLabel("")
        self.results.setWordWrap(True)
        left_layout.addWidget(self.results)

        left_layout.addStretch()
        layout.addWidget(left)

        # Right: visualization
        self.viz = IONORTVisualizationPanel()
        layout.addWidget(self.viz)

    def on_run(self):
        print("RUN CLICKED!")
        self.status.setText("Running...")
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)

        self.worker = Worker(
            self.tx_lat.value(),
            self.tx_lon.value(),
            self.rx_lat.value(),
            self.rx_lon.value(),
            self.foF2.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, done, total):
        if total > 0:
            self.progress.setValue(int(100 * done / total))

    def on_finished(self, result):
        print(f"Finished: {result}")
        self.run_btn.setEnabled(True)
        self.status.setText("Done!")
        self.progress.setValue(100)

        self.results.setText(
            f"Range: {result.great_circle_range_km:.0f} km\n"
            f"Winners: {result.num_winners}\n"
            f"MUF: {result.muf:.1f} MHz\n"
            f"LUF: {result.luf:.1f} MHz"
        )

        self.viz.update_from_homing_result(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    print("Window shown - click RUN")
    sys.exit(app.exec())
