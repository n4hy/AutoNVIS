"""
Historical Event Replayer for Auto-NVIS Validation

Replays historical ionospheric events through the assimilation filter
to validate system behavior against known ground truth.

Supports:
- Loading event configurations from YAML files
- Replaying observations through filter in sequence
- Recording filter response (state, mode switches, statistics)
- Comparing against expected behavior

Example usage:
    replayer = EventReplayer("events/x9_flare_2017.yaml")
    replayer.load_event()
    results = replayer.run()
    metrics = replayer.compute_metrics()
"""

import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging
from enum import Enum
import csv

logger = logging.getLogger(__name__)


class ObservationType(Enum):
    """Types of observations supported"""
    IONOSONDE = "ionosonde"
    TEC = "tec"
    GOES_XRAY = "goes_xray"
    DSCOVR_SOLAR_WIND = "solar_wind"
    COSMIC_GPS_RO = "gps_ro"


@dataclass
class Observation:
    """Single observation data point"""
    timestamp: datetime
    obs_type: ObservationType
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    values: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedBehavior:
    """Expected filter behavior during event"""
    mode_switches: List[Tuple[datetime, str]] = field(default_factory=list)
    fof2_change: Dict[str, Any] = field(default_factory=dict)
    hmf2_change: Dict[str, Any] = field(default_factory=dict)
    tec_change: Dict[str, Any] = field(default_factory=dict)
    response_time_minutes: Optional[float] = None


@dataclass
class EventConfig:
    """Configuration for a historical event"""
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    event_type: str  # "flare", "storm", "quiet", etc.
    observations: Dict[ObservationType, Path] = field(default_factory=dict)
    expected_behavior: ExpectedBehavior = field(default_factory=ExpectedBehavior)
    ground_truth: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterSnapshot:
    """Snapshot of filter state at a point in time"""
    timestamp: datetime
    cycle_index: int
    mode: str
    ne_grid: Optional[np.ndarray] = None
    fof2_estimate: Optional[float] = None
    hmf2_estimate: Optional[float] = None
    reff: Optional[float] = None
    uncertainty: Optional[float] = None
    nis: Optional[float] = None
    observations_processed: int = 0


@dataclass
class ReplayResult:
    """Results from replaying an event"""
    event_config: EventConfig
    snapshots: List[FilterSnapshot] = field(default_factory=list)
    mode_switches: List[Tuple[datetime, str, str]] = field(default_factory=list)  # time, from, to
    observations_processed: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class EventReplayer:
    """
    Replays historical ionospheric events through the Auto-NVIS filter.

    Loads event configuration and observation data, feeds observations
    through the filter in chronological order, and records the filter's
    response for validation.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        event_config: Optional[EventConfig] = None,
        snapshot_interval: timedelta = timedelta(minutes=5)
    ):
        """
        Initialize event replayer.

        Args:
            config_path: Path to YAML event configuration file
            event_config: EventConfig object (alternative to config_path)
            snapshot_interval: How often to record filter state
        """
        self.config_path = Path(config_path) if config_path else None
        self.event_config = event_config
        self.snapshot_interval = snapshot_interval

        self.observations: List[Observation] = []
        self.filter = None
        self._last_snapshot_time: Optional[datetime] = None
        self._current_mode: str = "QUIET"

    def load_event(self) -> EventConfig:
        """
        Load event configuration from YAML file.

        Returns:
            EventConfig object
        """
        if self.event_config is not None:
            return self.event_config

        if self.config_path is None:
            raise ValueError("No config_path or event_config provided")

        logger.info(f"Loading event configuration from {self.config_path}")

        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        event_data = config_data.get('event', {})
        obs_data = config_data.get('observations', {})
        expected_data = config_data.get('expected_behavior', {})
        truth_data = config_data.get('ground_truth', {})

        # Parse timestamps
        start_time = self._parse_timestamp(event_data.get('start_time'))
        end_time = self._parse_timestamp(event_data.get('end_time'))

        # Parse observations files
        obs_files = {}
        base_dir = self.config_path.parent
        for obs_name, obs_info in obs_data.items():
            obs_type = ObservationType(obs_name.lower())
            file_path = base_dir / obs_info.get('file', '')
            if file_path.exists():
                obs_files[obs_type] = file_path

        # Parse expected behavior
        mode_switches = []
        for switch in expected_data.get('mode_switches', []):
            switch_time = self._parse_timestamp(switch.get('time'))
            switch_mode = switch.get('mode', 'QUIET')
            mode_switches.append((switch_time, switch_mode))

        expected = ExpectedBehavior(
            mode_switches=mode_switches,
            fof2_change=expected_data.get('fof2_change', {}),
            hmf2_change=expected_data.get('hmf2_change', {}),
            tec_change=expected_data.get('tec_change', {}),
            response_time_minutes=expected_data.get('response_time_minutes')
        )

        # Parse ground truth files
        truth_files = {}
        for name, info in truth_data.items():
            file_path = base_dir / info.get('file', '')
            if file_path.exists():
                truth_files[name] = file_path

        self.event_config = EventConfig(
            name=event_data.get('name', 'Unknown Event'),
            description=event_data.get('description', ''),
            start_time=start_time,
            end_time=end_time,
            event_type=event_data.get('type', 'unknown'),
            observations=obs_files,
            expected_behavior=expected,
            ground_truth=truth_files,
            metadata=event_data.get('metadata', {})
        )

        logger.info(f"Loaded event: {self.event_config.name}")
        logger.info(f"  Time range: {start_time} to {end_time}")
        logger.info(f"  Observation sources: {list(obs_files.keys())}")

        return self.event_config

    def load_observations(self) -> List[Observation]:
        """
        Load all observations from configured files.

        Returns:
            List of Observation objects sorted by timestamp
        """
        if self.event_config is None:
            self.load_event()

        self.observations = []

        for obs_type, file_path in self.event_config.observations.items():
            logger.info(f"Loading {obs_type.value} observations from {file_path}")

            try:
                if obs_type == ObservationType.IONOSONDE:
                    obs = self._load_ionosonde_csv(file_path)
                elif obs_type == ObservationType.TEC:
                    obs = self._load_tec_csv(file_path)
                elif obs_type == ObservationType.GOES_XRAY:
                    obs = self._load_goes_xray_csv(file_path)
                else:
                    logger.warning(f"Unknown observation type: {obs_type}")
                    continue

                self.observations.extend(obs)
                logger.info(f"  Loaded {len(obs)} observations")

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        # Sort by timestamp
        self.observations.sort(key=lambda x: x.timestamp)

        # Filter to event time range
        self.observations = [
            obs for obs in self.observations
            if self.event_config.start_time <= obs.timestamp <= self.event_config.end_time
        ]

        logger.info(f"Total observations in event window: {len(self.observations)}")

        return self.observations

    def _load_ionosonde_csv(self, file_path: Path) -> List[Observation]:
        """Load ionosonde observations from CSV"""
        observations = []

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = self._parse_timestamp(row.get('timestamp', row.get('Time', '')))
                    if timestamp is None:
                        continue

                    values = {}
                    errors = {}

                    if 'foF2' in row and row['foF2']:
                        values['foF2'] = float(row['foF2'])
                        errors['foF2'] = float(row.get('foF2_error', 0.3))

                    if 'hmF2' in row and row['hmF2']:
                        values['hmF2'] = float(row['hmF2'])
                        errors['hmF2'] = float(row.get('hmF2_error', 15.0))

                    if 'foE' in row and row['foE']:
                        values['foE'] = float(row['foE'])

                    if not values:
                        continue

                    obs = Observation(
                        timestamp=timestamp,
                        obs_type=ObservationType.IONOSONDE,
                        latitude=float(row.get('latitude', row.get('lat', 0))),
                        longitude=float(row.get('longitude', row.get('lon', 0))),
                        values=values,
                        errors=errors,
                        metadata={'station': row.get('station', 'unknown')}
                    )
                    observations.append(obs)

                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid row: {e}")

        return observations

    def _load_tec_csv(self, file_path: Path) -> List[Observation]:
        """Load TEC observations from CSV"""
        observations = []

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = self._parse_timestamp(row.get('timestamp', row.get('Time', '')))
                    if timestamp is None:
                        continue

                    tec_value = float(row.get('tec', row.get('STEC', row.get('VTEC', 0))))
                    if tec_value <= 0:
                        continue

                    obs = Observation(
                        timestamp=timestamp,
                        obs_type=ObservationType.TEC,
                        latitude=float(row.get('latitude', row.get('rx_lat', 0))),
                        longitude=float(row.get('longitude', row.get('rx_lon', 0))),
                        values={'tec': tec_value},
                        errors={'tec': float(row.get('tec_error', 1.0))},
                        metadata={
                            'elevation': float(row.get('elevation', 45.0)),
                            'azimuth': float(row.get('azimuth', 0.0)),
                            'prn': row.get('prn', row.get('satellite', 'unknown'))
                        }
                    )
                    observations.append(obs)

                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid TEC row: {e}")

        return observations

    def _load_goes_xray_csv(self, file_path: Path) -> List[Observation]:
        """Load GOES X-ray flux observations from CSV"""
        observations = []

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamp = self._parse_timestamp(row.get('timestamp', row.get('time_tag', '')))
                    if timestamp is None:
                        continue

                    # GOES X-ray flux typically in W/m^2
                    xray_long = float(row.get('xray_long', row.get('A_FLUX', 0)))
                    xray_short = float(row.get('xray_short', row.get('B_FLUX', 0)))

                    if xray_long <= 0 and xray_short <= 0:
                        continue

                    obs = Observation(
                        timestamp=timestamp,
                        obs_type=ObservationType.GOES_XRAY,
                        values={
                            'xray_long': xray_long,  # 1-8 Angstrom
                            'xray_short': xray_short  # 0.5-4 Angstrom
                        },
                        metadata={'satellite': row.get('satellite', 'GOES')}
                    )
                    observations.append(obs)

                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid GOES row: {e}")

        return observations

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime"""
        if not ts_str:
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        logger.warning(f"Could not parse timestamp: {ts_str}")
        return None

    def set_filter(self, filter_instance) -> None:
        """
        Set the filter instance to use for replay.

        Args:
            filter_instance: AutoNVISFilter instance
        """
        self.filter = filter_instance
        logger.info("Filter instance set for replay")

    def run(self) -> ReplayResult:
        """
        Run the event replay.

        Feeds all observations through the filter in chronological order
        and records the filter's response.

        Returns:
            ReplayResult with snapshots and metrics
        """
        import time
        start_time = time.time()

        if self.event_config is None:
            self.load_event()

        if not self.observations:
            self.load_observations()

        result = ReplayResult(event_config=self.event_config)

        logger.info(f"Starting replay of {self.event_config.name}")
        logger.info(f"Processing {len(self.observations)} observations")

        self._last_snapshot_time = None
        self._current_mode = "QUIET"

        for i, obs in enumerate(self.observations):
            try:
                self._process_observation(obs, result)

                # Take snapshot if interval elapsed
                if self._should_snapshot(obs.timestamp):
                    snapshot = self._take_snapshot(obs.timestamp, result.observations_processed)
                    result.snapshots.append(snapshot)
                    self._last_snapshot_time = obs.timestamp

                result.observations_processed += 1

                if (i + 1) % 100 == 0:
                    logger.debug(f"Processed {i + 1}/{len(self.observations)} observations")

            except Exception as e:
                error_msg = f"Error processing observation at {obs.timestamp}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Final snapshot
        if self.observations:
            final_snapshot = self._take_snapshot(
                self.observations[-1].timestamp,
                result.observations_processed
            )
            result.snapshots.append(final_snapshot)

        result.processing_time_seconds = time.time() - start_time

        logger.info(f"Replay complete in {result.processing_time_seconds:.1f}s")
        logger.info(f"Processed {result.observations_processed} observations")
        logger.info(f"Recorded {len(result.snapshots)} snapshots")
        logger.info(f"Mode switches: {len(result.mode_switches)}")

        return result

    def _process_observation(self, obs: Observation, result: ReplayResult) -> None:
        """Process a single observation through the filter"""
        if self.filter is None:
            return

        # Feed observation to filter based on type
        if obs.obs_type == ObservationType.IONOSONDE:
            if 'foF2' in obs.values and 'hmF2' in obs.values:
                self.filter.process_ionosonde_observation(
                    latitude=obs.latitude,
                    longitude=obs.longitude,
                    fof2=obs.values['foF2'],
                    hmf2=obs.values['hmF2'],
                    fof2_error=obs.errors.get('foF2', 0.3),
                    hmf2_error=obs.errors.get('hmF2', 15.0),
                    timestamp=obs.timestamp
                )

        elif obs.obs_type == ObservationType.TEC:
            self.filter.process_tec_observation(
                latitude=obs.latitude,
                longitude=obs.longitude,
                tec=obs.values['tec'],
                tec_error=obs.errors.get('tec', 1.0),
                elevation=obs.metadata.get('elevation', 45.0),
                azimuth=obs.metadata.get('azimuth', 0.0),
                timestamp=obs.timestamp
            )

        elif obs.obs_type == ObservationType.GOES_XRAY:
            self.filter.process_xray_flux(
                xray_long=obs.values.get('xray_long', 0.0),
                xray_short=obs.values.get('xray_short', 0.0),
                timestamp=obs.timestamp
            )

        # Check for mode switch
        new_mode = self._get_current_mode()
        if new_mode != self._current_mode:
            result.mode_switches.append((obs.timestamp, self._current_mode, new_mode))
            logger.info(f"Mode switch at {obs.timestamp}: {self._current_mode} -> {new_mode}")
            self._current_mode = new_mode

    def _should_snapshot(self, current_time: datetime) -> bool:
        """Check if it's time to take a snapshot"""
        if self._last_snapshot_time is None:
            return True

        elapsed = current_time - self._last_snapshot_time
        return elapsed >= self.snapshot_interval

    def _take_snapshot(self, timestamp: datetime, obs_count: int) -> FilterSnapshot:
        """Take a snapshot of the current filter state"""
        snapshot = FilterSnapshot(
            timestamp=timestamp,
            cycle_index=0,
            mode=self._current_mode,
            observations_processed=obs_count
        )

        if self.filter is not None:
            try:
                stats = self.filter.get_statistics()
                snapshot.cycle_index = stats.get('cycle_count', 0)
                snapshot.nis = stats.get('avg_nis')
                snapshot.reff = self.filter.get_effective_ssn()

                # Get state estimate at a reference location
                ref_lat, ref_lon = 40.0, -105.0  # Example reference
                fof2, hmf2 = self.filter.get_f2_parameters(ref_lat, ref_lon)
                snapshot.fof2_estimate = fof2
                snapshot.hmf2_estimate = hmf2

            except Exception as e:
                logger.debug(f"Could not get filter state: {e}")

        return snapshot

    def _get_current_mode(self) -> str:
        """Get current filter mode"""
        if self.filter is None:
            return "QUIET"

        try:
            stats = self.filter.get_statistics()
            return stats.get('current_mode', 'QUIET')
        except Exception:
            return "QUIET"


def create_sample_event_config(event_dir: Path) -> None:
    """
    Create a sample event configuration file.

    Args:
        event_dir: Directory to create sample files in
    """
    event_dir.mkdir(parents=True, exist_ok=True)

    # Sample event configuration
    config = {
        'event': {
            'name': 'X9.3 Flare - September 6, 2017',
            'description': 'Intense X-class flare causing significant SID',
            'type': 'flare',
            'start_time': '2017-09-06T11:53:00Z',
            'end_time': '2017-09-06T14:00:00Z',
            'metadata': {
                'flare_class': 'X9.3',
                'active_region': 'AR2673',
                'peak_time': '2017-09-06T12:02:00Z'
            }
        },
        'observations': {
            'goes_xray': {'file': 'goes_xray.csv'},
            'ionosonde': {'file': 'ionosonde_fof2.csv'}
        },
        'expected_behavior': {
            'mode_switches': [
                {'time': '2017-09-06T11:55:00Z', 'mode': 'FLARE'},
                {'time': '2017-09-06T13:30:00Z', 'mode': 'QUIET'}
            ],
            'fof2_change': {
                'type': 'decrease',
                'magnitude': '10-30%',
                'start_time': '2017-09-06T11:55:00Z',
                'recovery_time_minutes': 60
            },
            'response_time_minutes': 5
        },
        'ground_truth': {
            'digisonde': {'file': 'digisonde_truth.csv'}
        }
    }

    config_path = event_dir / 'event_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Sample GOES X-ray data
    goes_data = """timestamp,xray_long,xray_short,satellite
2017-09-06T11:50:00Z,1.5e-5,1.2e-6,GOES-15
2017-09-06T11:55:00Z,5.0e-4,4.5e-5,GOES-15
2017-09-06T12:00:00Z,8.5e-4,7.8e-5,GOES-15
2017-09-06T12:02:00Z,9.3e-4,8.9e-5,GOES-15
2017-09-06T12:05:00Z,7.2e-4,6.5e-5,GOES-15
2017-09-06T12:15:00Z,3.5e-4,3.0e-5,GOES-15
2017-09-06T12:30:00Z,1.2e-4,9.8e-6,GOES-15
2017-09-06T13:00:00Z,4.5e-5,3.2e-6,GOES-15
2017-09-06T13:30:00Z,2.1e-5,1.5e-6,GOES-15
"""
    (event_dir / 'goes_xray.csv').write_text(goes_data)

    # Sample ionosonde data
    ionosonde_data = """timestamp,station,latitude,longitude,foF2,hmF2,foF2_error,hmF2_error
2017-09-06T11:45:00Z,BC840,40.0,-105.3,8.5,285,0.3,15
2017-09-06T12:00:00Z,BC840,40.0,-105.3,7.2,295,0.3,15
2017-09-06T12:15:00Z,BC840,40.0,-105.3,6.8,305,0.4,20
2017-09-06T12:30:00Z,BC840,40.0,-105.3,7.0,300,0.3,15
2017-09-06T12:45:00Z,BC840,40.0,-105.3,7.5,290,0.3,15
2017-09-06T13:00:00Z,BC840,40.0,-105.3,8.0,285,0.3,15
2017-09-06T13:30:00Z,BC840,40.0,-105.3,8.3,282,0.3,15
"""
    (event_dir / 'ionosonde_fof2.csv').write_text(ionosonde_data)

    logger.info(f"Created sample event configuration in {event_dir}")


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    # Create sample event
    sample_dir = Path(__file__).parent / 'historical_events' / 'x9_flare_2017'
    create_sample_event_config(sample_dir)

    # Load and inspect
    replayer = EventReplayer(sample_dir / 'event_config.yaml')
    config = replayer.load_event()
    observations = replayer.load_observations()

    print(f"\nEvent: {config.name}")
    print(f"Duration: {config.end_time - config.start_time}")
    print(f"Observations: {len(observations)}")
