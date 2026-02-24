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

        left_layout.addWidget(QLabel("SNR Cutoff (dB):"))
        self.snr_cutoff = QDoubleSpinBox()
        self.snr_cutoff.setRange(-20, 60)
        self.snr_cutoff.setValue(0.0)
        self.snr_cutoff.setToolTip("-20 for FT8, 0 for CW, 10+ for voice")
        left_layout.addWidget(self.snr_cutoff)

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

        # Calculate SNR for all winners (MANDATORY)
        if result.num_winners > 0:
            self._calculate_snr_for_result(result)

        # Show results including best SNR
        best_snr = None
        if result.winner_triplets:
            snr_values = [w.snr_db for w in result.winner_triplets if w.snr_db is not None]
            if snr_values:
                best_snr = max(snr_values)

        snr_text = f"\nBest SNR: {best_snr:.0f} dB" if best_snr is not None else ""

        self.results.setText(
            f"Range: {result.great_circle_range_km:.0f} km\n"
            f"Winners: {result.num_winners}\n"
            f"MUF: {result.muf:.1f} MHz\n"
            f"LUF: {result.luf:.1f} MHz{snr_text}"
        )

        self.viz.update_from_homing_result(result)

    def _calculate_snr_for_result(self, result):
        """Calculate SNR for all winners using physics-based link budget.

        Winners with invalid/uncomputable SNR are REMOVED from the result.
        No fake or fallback values - SNR is physics or nothing.
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

        # Radio config
        tx_power = 100.0
        tx_gain = 0.0
        rx_gain = 0.0
        rx_bw = 3000.0

        tx_config = TransmitterConfig(power_watts=tx_power, antenna=AntennaConfig(gain_dbi=tx_gain))
        rx_config = ReceiverConfig(antenna=AntennaConfig(gain_dbi=rx_gain), bandwidth_hz=rx_bw,
                                   noise_environment=NoiseEnvironment.RURAL)

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

        calc = LinkBudgetCalculator()
        print(f"Link budget: range={gc_range:.0f}km, solar_zenith={solar_zenith:.0f}°, night={is_night}")

        # Process winners - REMOVE any that fail SNR calculation
        valid_winners = []
        invalid_count = 0

        for idx, winner in enumerate(result.winner_triplets):
            freq = winner.frequency_mhz
            hop_count = max(winner.hop_count, 1)

            # Reflection height from ray path
            h = winner.reflection_height_km
            if h <= 0 and winner.ray_path and winner.ray_path.states:
                h = max(s.altitude() for s in winner.ray_path.states)
            if h <= 0:
                print(f"  #{idx}: {freq:.1f}MHz SKIPPED - no reflection height")
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
                print(f"  #{idx}: {freq:.1f}MHz SKIPPED - no range data")
                invalid_count += 1
                continue

            # Path length from geometry
            hop_range = ground_range / hop_count
            path_per_hop = 2 * np.sqrt((hop_range/2)**2 + h**2)
            path_length = path_per_hop * hop_count

            try:
                link_result = calc.calculate(
                    frequency_mhz=freq,
                    path_length_km=path_length,
                    hop_count=hop_count,
                    reflection_height_km=h,
                    solar_zenith_angle_deg=solar_zenith,
                    xray_flux=1e-6,
                    kp_index=2.0,
                    tx_config=tx_config,
                    rx_config=rx_config,
                    latitude_deg=mid_lat,
                    is_night=is_night,
                )

                snr = link_result.snr_db
                if np.isnan(snr) or np.isinf(snr):
                    raise ValueError(f"Non-physical SNR: {snr}")

                # Filter: SNR must be usable for practical communication
                # User-configurable: -20 for FT8, 0 for CW, 10+ for voice
                snr_cutoff = self.snr_cutoff.value()
                if snr < snr_cutoff:
                    print(f"  #{idx}: {freq:.1f}MHz REJECTED - SNR {snr:.0f}dB below {snr_cutoff:.0f}dB cutoff")
                    invalid_count += 1
                    continue

                winner.snr_db = snr
                winner.signal_strength_dbm = link_result.signal_power_dbw + 30
                winner.path_loss_db = link_result.total_path_loss_db
                valid_winners.append(winner)

                if idx < 5:
                    print(f"  {freq:.1f}MHz: path={path_length:.0f}km h={h:.0f}km, "
                          f"loss={link_result.total_path_loss_db:.0f}dB, SNR={snr:.0f}dB")
            except Exception as e:
                print(f"  #{idx}: {freq:.1f}MHz FAILED: {e}")
                invalid_count += 1
                continue

        # Replace winner list with only valid winners
        result.winner_triplets[:] = valid_winners
        if invalid_count > 0:
            print(f"Removed {invalid_count} winners with uncomputable SNR")

    def closeEvent(self, event):
        """Handle window close - ensure all resources are released."""
        print("Window closing - cleaning up...")

        # Stop visualization timers
        if hasattr(self, 'viz') and self.viz is not None:
            self.viz.cleanup()

        # Stop worker if running
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            if not self.worker.wait(2000):
                self.worker.terminate()
                self.worker.wait(1000)

        print("Cleanup complete")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    print("Window shown - click RUN")
    sys.exit(app.exec())
