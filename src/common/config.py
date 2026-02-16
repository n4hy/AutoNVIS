"""
Centralized Configuration Management for Auto-NVIS

This module provides a unified interface for loading and accessing
system configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GridConfig:
    """Configuration for the 4D ionospheric grid"""

    # Latitude grid
    lat_min: float = -90.0  # degrees
    lat_max: float = 90.0   # degrees
    lat_step: float = 2.5   # degrees

    # Longitude grid
    lon_min: float = -180.0  # degrees
    lon_max: float = 180.0   # degrees
    lon_step: float = 5.0    # degrees

    # Altitude grid
    alt_min: float = 60.0    # km
    alt_max: float = 600.0   # km
    alt_step: float = 10.0   # km

    @property
    def n_lat(self) -> int:
        """Number of latitude grid points"""
        return int((self.lat_max - self.lat_min) / self.lat_step) + 1

    @property
    def n_lon(self) -> int:
        """Number of longitude grid points"""
        return int((self.lon_max - self.lon_min) / self.lon_step) + 1

    @property
    def n_alt(self) -> int:
        """Number of altitude grid points"""
        return int((self.alt_max - self.alt_min) / self.alt_step) + 1

    @property
    def total_points(self) -> int:
        """Total number of grid points"""
        return self.n_lat * self.n_lon * self.n_alt

    def get_lat_grid(self) -> np.ndarray:
        """Get latitude grid as numpy array"""
        return np.arange(self.lat_min, self.lat_max + self.lat_step/2, self.lat_step)

    def get_lon_grid(self) -> np.ndarray:
        """Get longitude grid as numpy array"""
        return np.arange(self.lon_min, self.lon_max + self.lon_step/2, self.lon_step)

    def get_alt_grid(self) -> np.ndarray:
        """Get altitude grid as numpy array"""
        return np.arange(self.alt_min, self.alt_max + self.alt_step/2, self.alt_step)


@dataclass
class DataSourceConfig:
    """Configuration for external data sources"""

    # GOES X-ray
    goes_xray_url: str = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    goes_update_interval: int = 60  # seconds

    # ACE Solar Wind
    ace_swepam_url: str = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1m.json"
    ace_mag_url: str = "https://services.swpc.noaa.gov/json/ace/mag/ace_mag_1m.json"
    ace_update_interval: int = 60  # seconds

    # GNSS-TEC (IGS Ntrip)
    ntrip_host: str = "www.igs-ip.net"
    ntrip_port: int = 2101
    ntrip_mountpoint: str = "RTCM3"
    ntrip_username: Optional[str] = None
    ntrip_password: Optional[str] = None

    # GIRO Ionosonde
    giro_base_url: str = "https://giro.uml.edu/didbase"
    giro_stations: list = field(default_factory=lambda: ["MHJ45", "AT138", "EA036"])
    giro_update_interval: int = 300  # seconds (5 minutes)

    # GloTEC (NOAA SWPC Global TEC Maps)
    glotec_index_url: str = "https://services.swpc.noaa.gov/products/glotec/geojson_2d_urt.json"
    glotec_base_url: str = "https://services.swpc.noaa.gov"
    glotec_update_interval: int = 600  # seconds (10 minutes, matches data cadence)


@dataclass
class ServiceConfig:
    """Configuration for microservice endpoints"""

    # Message queue
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_vhost: str = "/"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0

    # Assimilation service
    assimilation_host: str = "assimilation"
    assimilation_port: int = 50051  # gRPC port

    # Supervisor service
    supervisor_host: str = "supervisor"
    supervisor_port: int = 8000

    # Output service
    output_host: str = "output"
    output_port: int = 8080


@dataclass
class SRUKFConfig:
    """Configuration for Square-Root Unscented Kalman Filter"""

    # Unscented transform parameters
    alpha: float = 1e-3  # Spread parameter (1e-4 to 1)
    beta: float = 2.0    # Prior knowledge parameter (2 for Gaussian)
    kappa: float = 0.0   # Secondary scaling (usually 0 or 3-n)

    # Process noise
    process_noise_ne: float = 1e10  # Electron density process noise (el/m³)²
    process_noise_reff: float = 10.0  # Effective sunspot number variance

    # Observation noise
    obs_noise_tec: float = 2.0  # TEC observation error (TECU)
    obs_noise_fof2: float = 0.5  # foF2 observation error (MHz)
    obs_noise_hmf2: float = 10.0  # hmF2 observation error (km)

    # Numerical stability
    min_eigenvalue: float = 1e-10
    covariance_inflation: float = 1.0  # Multiplicative inflation factor


@dataclass
class SupervisorConfig:
    """Configuration for Supervisor logic"""

    # Mode switching
    xray_threshold_m1: float = 1e-5  # M1 flare threshold (W/m²)
    mode_switch_hysteresis_sec: int = 600  # 10 minutes

    # Cycle timing
    update_cycle_sec: int = 900  # 15 minutes
    max_cycle_duration_sec: int = 1200  # 20 minutes (allow overrun)

    # Health monitoring
    health_check_interval_sec: int = 30
    data_staleness_threshold_sec: int = 300  # 5 minutes


@dataclass
class PropagationConfig:
    """Configuration for propagation prediction (ray tracing)"""

    # Transmitter location
    tx_lat: float = 40.0  # Boulder, CO
    tx_lon: float = -105.0
    tx_alt_km: float = 0.0

    # Frequency scan parameters
    freq_min_mhz: float = 2.0
    freq_max_mhz: float = 15.0
    freq_step_mhz: float = 0.5

    # NVIS ray geometry
    elevation_min_deg: float = 70.0
    elevation_max_deg: float = 90.0
    elevation_step_deg: float = 2.0
    azimuth_step_deg: float = 15.0

    # Quality thresholds
    absorption_threshold_db: float = 50.0
    snr_threshold_db: float = 10.0

    # Performance
    enable_parallel: bool = True
    max_workers: int = 4


@dataclass
class NVISAdapterConfig:
    """Configuration for a single NVIS protocol adapter"""
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8001
    topic_prefix: Optional[str] = None
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None


@dataclass
class NVISQualityTierConfig:
    """Configuration for a single quality tier"""
    signal_error_db: float = 2.0
    delay_error_ms: float = 0.1
    min_score: float = 0.8


@dataclass
class NVISIngestionConfig:
    """Configuration for NVIS sounder data ingestion"""

    # Protocol adapters
    tcp: NVISAdapterConfig = field(default_factory=lambda: NVISAdapterConfig(enabled=True, port=8001))
    http: NVISAdapterConfig = field(default_factory=lambda: NVISAdapterConfig(enabled=True, port=8002))
    mqtt: NVISAdapterConfig = field(default_factory=lambda: NVISAdapterConfig(enabled=False, port=1883))
    email: NVISAdapterConfig = field(default_factory=lambda: NVISAdapterConfig(enabled=False))

    # Quality tiers
    platinum: NVISQualityTierConfig = field(default_factory=lambda: NVISQualityTierConfig(
        signal_error_db=2.0, delay_error_ms=0.1, min_score=0.80
    ))
    gold: NVISQualityTierConfig = field(default_factory=lambda: NVISQualityTierConfig(
        signal_error_db=4.0, delay_error_ms=0.5, min_score=0.60
    ))
    silver: NVISQualityTierConfig = field(default_factory=lambda: NVISQualityTierConfig(
        signal_error_db=8.0, delay_error_ms=2.0, min_score=0.40
    ))
    bronze: NVISQualityTierConfig = field(default_factory=lambda: NVISQualityTierConfig(
        signal_error_db=15.0, delay_error_ms=5.0, min_score=0.0
    ))

    # Aggregation settings
    window_seconds: int = 60
    rate_threshold: int = 60  # obs/hour triggers aggregation

    # Rate limiting (max observations per 15-min cycle)
    max_obs_per_cycle_platinum: int = 50
    max_obs_per_cycle_gold: int = 30
    max_obs_per_cycle_silver: int = 15
    max_obs_per_cycle_bronze: int = 5

    # Quality component weights (sum to 1.0)
    weight_signal_quality: float = 0.25
    weight_calibration_quality: float = 0.20
    weight_temporal_quality: float = 0.15
    weight_spatial_quality: float = 0.15
    weight_equipment_quality: float = 0.15
    weight_historical_quality: float = 0.10


@dataclass
class AutoNVISConfig:
    """Master configuration for Auto-NVIS system"""

    grid: GridConfig = field(default_factory=GridConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    sr_ukf: SRUKFConfig = field(default_factory=SRUKFConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    nvis_ingestion: NVISIngestionConfig = field(default_factory=NVISIngestionConfig)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("/data"))
    config_dir: Path = field(default_factory=lambda: Path("/config"))

    @property
    def grid_output_dir(self) -> Path:
        """Directory for electron density grids"""
        return self.data_dir / "grids"

    @property
    def state_checkpoint_dir(self) -> Path:
        """Directory for filter state checkpoints"""
        return self.data_dir / "state"

    @property
    def observation_cache_dir(self) -> Path:
        """Directory for cached observations"""
        return self.data_dir / "observations"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AutoNVISConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse NVIS ingestion config with nested structures
        nvis_config_dict = config_dict.get('nvis_ingestion', {})
        adapters = nvis_config_dict.get('adapters', {})
        quality_tiers = nvis_config_dict.get('quality_tiers', {})
        aggregation = nvis_config_dict.get('aggregation', {})
        rate_limiting = nvis_config_dict.get('rate_limiting', {}).get('max_obs_per_cycle', {})
        quality_weights = nvis_config_dict.get('quality_weights', {})

        nvis_ingestion = NVISIngestionConfig(
            tcp=NVISAdapterConfig(**adapters.get('tcp', {})),
            http=NVISAdapterConfig(**adapters.get('http', {})),
            mqtt=NVISAdapterConfig(**adapters.get('mqtt', {})),
            email=NVISAdapterConfig(**adapters.get('email', {})),
            platinum=NVISQualityTierConfig(**quality_tiers.get('platinum', {})),
            gold=NVISQualityTierConfig(**quality_tiers.get('gold', {})),
            silver=NVISQualityTierConfig(**quality_tiers.get('silver', {})),
            bronze=NVISQualityTierConfig(**quality_tiers.get('bronze', {})),
            window_seconds=aggregation.get('window_seconds', 60),
            rate_threshold=aggregation.get('rate_threshold', 60),
            max_obs_per_cycle_platinum=rate_limiting.get('platinum', 50),
            max_obs_per_cycle_gold=rate_limiting.get('gold', 30),
            max_obs_per_cycle_silver=rate_limiting.get('silver', 15),
            max_obs_per_cycle_bronze=rate_limiting.get('bronze', 5),
            weight_signal_quality=quality_weights.get('signal_quality', 0.25),
            weight_calibration_quality=quality_weights.get('calibration_quality', 0.20),
            weight_temporal_quality=quality_weights.get('temporal_quality', 0.15),
            weight_spatial_quality=quality_weights.get('spatial_quality', 0.15),
            weight_equipment_quality=quality_weights.get('equipment_quality', 0.15),
            weight_historical_quality=quality_weights.get('historical_quality', 0.10)
        )

        return cls(
            grid=GridConfig(**config_dict.get('grid', {})),
            data_sources=DataSourceConfig(**config_dict.get('data_sources', {})),
            services=ServiceConfig(**config_dict.get('services', {})),
            sr_ukf=SRUKFConfig(**config_dict.get('sr_ukf', {})),
            supervisor=SupervisorConfig(**config_dict.get('supervisor', {})),
            nvis_ingestion=nvis_ingestion,
            data_dir=Path(config_dict.get('data_dir', '/data')),
            config_dir=Path(config_dict.get('config_dir', '/config'))
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'grid': self.grid.__dict__,
            'data_sources': self.data_sources.__dict__,
            'services': self.services.__dict__,
            'sr_ukf': self.sr_ukf.__dict__,
            'supervisor': self.supervisor.__dict__,
            'nvis_ingestion': self.nvis_ingestion.__dict__,
            'data_dir': str(self.data_dir),
            'config_dir': str(self.config_dir)
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_config(config_path: Optional[str] = None) -> AutoNVISConfig:
    """
    Get system configuration

    Priority:
    1. Provided config_path
    2. AUTONVIS_CONFIG environment variable
    3. config/production.yml
    4. Default configuration
    """
    if config_path is None:
        config_path = os.getenv('AUTONVIS_CONFIG')

    if config_path is None:
        # Try default paths
        default_paths = [
            Path(__file__).parent.parent.parent / 'config' / 'production.yml',
            Path('/config/production.yml'),
            Path('config/production.yml')
        ]

        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        return AutoNVISConfig.from_yaml(config_path)

    # Return default configuration
    return AutoNVISConfig()
