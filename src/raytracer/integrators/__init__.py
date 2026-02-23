"""
IONORT-Style Numerical Integrators for Ray Tracing

This package provides multiple integration methods for solving the
Haselgrove ray equations (6 coupled first-order ODEs).

Available Integrators:
- RK4Integrator: Classical 4th-order Runge-Kutta with error tracking
- AdamsBashforthMoultonIntegrator: 4-step predictor / 3-step corrector
- RK45Integrator: Dormand-Prince adaptive step

Reference: IONORT paper Section 2.2 (remotesensing-15-05111-v2.pdf)
"""

from .base import (
    BaseIntegrator,
    IntegrationStep,
    IntegrationStats,
)
from .rk4 import RK4Integrator
from .adams_bashforth import AdamsBashforthMoultonIntegrator
from .rk45 import RK45Integrator
from .factory import IntegratorFactory, create_integrator

__all__ = [
    # Base classes
    'BaseIntegrator',
    'IntegrationStep',
    'IntegrationStats',
    # Integrators
    'RK4Integrator',
    'AdamsBashforthMoultonIntegrator',
    'RK45Integrator',
    # Factory
    'IntegratorFactory',
    'create_integrator',
]
