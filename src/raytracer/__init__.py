"""
AutoNVIS Ray Tracing Package

3D magnetoionic HF ray tracing with real-time ionospheric data integration.

This package provides ray tracing capabilities that exceed standard climatological
predictions by assimilating real-time ionosonde data into the ionospheric model.

Core Components:
- IonosphericModel: Electron density profiles with real-time correction
- IRICorrection: Apply real-time foF2/hmF2 to IRI baseline
- HaselgroveSolver: 6-coupled ODE ray path integration
- PHaRLAPInterface: High-level ray tracing API
- NVISOptimizer: Find optimal NVIS frequencies via homing

Key Innovation:
Standard ray tracers use IRI climatological (monthly median) profiles.
This package corrects IRI using real-time GIRO ionosonde data:
    foF2_real → α = foF2_real / foF2_IRI
    Ne_corrected = α² × Ne_IRI

Based on concepts from:
- IONORT (INGV) - Haselgrove's equations implementation
- PHaRLAP - High-level ray tracing framework
"""

__version__ = "0.1.0"
__author__ = "AutoNVIS Project"

# Physical constants
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT = 299792458.0  # m/s
ELECTRON_CHARGE = 1.60217663e-19  # C
ELECTRON_MASS = 9.1093837e-31  # kg
PERMITTIVITY_0 = 8.8541878e-12  # F/m
MU_0 = 1.25663706e-6  # H/m

# Plasma frequency constant: fp = 9 * sqrt(Ne) Hz where Ne in el/m³
PLASMA_FREQ_CONSTANT = 8.98  # MHz when Ne in el/cm³

# Core ionospheric model
from .electron_density import (
    IonosphericModel,
    IonosphericProfile,
    ChapmanLayer,
    AppletonHartree,
)

# Real-time correction
from .iri_correction import (
    IRICorrection,
    IRICorrectionCallback,
    IonosondeStation,
    IonosondeMeasurement,
    CorrectionFactors,
)

# Ray tracing
from .haselgrove import (
    HaselgroveSolver,
    RayState,
    RayPath,
    RayMode,
    RayTermination,
)

# High-level interface
from .pharlap_interface import (
    PHaRLAPInterface,
    PropagationResult,
    MUFResult,
    IonogramPoint,
)

# NVIS optimization
from .nvis_optimizer import (
    NVISOptimizer,
    NVISResult,
    NVISFrequency,
    SignalQuality,
)

# Numerical integrators (IONORT-style)
from .integrators import (
    BaseIntegrator,
    IntegrationStep,
    IntegrationStats,
    RK4Integrator,
    AdamsBashforthMoultonIntegrator,
    RK45Integrator,
    IntegratorFactory,
    create_integrator,
)

# IONORT-style homing algorithm
from .homing_algorithm import (
    HomingAlgorithm,
    HomingResult,
    HomingSearchSpace,
    HomingConfig,
    WinnerTriplet,
    PropagationMode,
)

# Link budget calculator
from .link_budget import (
    LinkBudgetCalculator,
    LinkBudgetResult,
    PropagationLosses,
    TransmitterConfig,
    ReceiverConfig,
    AntennaConfig,
    NoiseEnvironment,
    calculate_solar_zenith_angle,
    is_nighttime,
)

__all__ = [
    # Constants
    'EARTH_RADIUS_KM',
    'SPEED_OF_LIGHT',
    'ELECTRON_CHARGE',
    'ELECTRON_MASS',
    'PERMITTIVITY_0',
    'MU_0',
    'PLASMA_FREQ_CONSTANT',
    # Ionospheric model
    'IonosphericModel',
    'IonosphericProfile',
    'ChapmanLayer',
    'AppletonHartree',
    # IRI correction
    'IRICorrection',
    'IRICorrectionCallback',
    'IonosondeStation',
    'IonosondeMeasurement',
    'CorrectionFactors',
    # Ray tracing
    'HaselgroveSolver',
    'RayState',
    'RayPath',
    'RayMode',
    'RayTermination',
    # PHaRLAP interface
    'PHaRLAPInterface',
    'PropagationResult',
    'MUFResult',
    'IonogramPoint',
    # NVIS optimizer
    'NVISOptimizer',
    'NVISResult',
    'NVISFrequency',
    'SignalQuality',
    # Numerical integrators
    'BaseIntegrator',
    'IntegrationStep',
    'IntegrationStats',
    'RK4Integrator',
    'AdamsBashforthMoultonIntegrator',
    'RK45Integrator',
    'IntegratorFactory',
    'create_integrator',
    # Homing algorithm
    'HomingAlgorithm',
    'HomingResult',
    'HomingSearchSpace',
    'HomingConfig',
    'WinnerTriplet',
    'PropagationMode',
    # Link budget
    'LinkBudgetCalculator',
    'LinkBudgetResult',
    'PropagationLosses',
    'TransmitterConfig',
    'ReceiverConfig',
    'AntennaConfig',
    'NoiseEnvironment',
    'calculate_solar_zenith_angle',
    'is_nighttime',
]
