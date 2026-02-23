"""
Integrator Factory

Provides factory functions for creating integrators by name or configuration.
Simplifies integrator selection in configuration files and command-line tools.

Example:
    # By name
    integrator = create_integrator('rk45', derivative_func, tolerance=1e-7)

    # By enum
    integrator = IntegratorFactory.create(
        IntegratorType.ADAMS_BASHFORTH,
        derivative_func,
        tolerance=1e-6
    )

    # List available integrators
    for name in IntegratorFactory.available():
        print(name)
"""

from enum import Enum, auto
from typing import Callable, Dict, Optional, Type
import numpy as np

from .base import BaseIntegrator
from .rk4 import RK4Integrator, RK4IntegratorFast
from .adams_bashforth import AdamsBashforthMoultonIntegrator
from .rk45 import RK45Integrator, RK45IntegratorFast


class IntegratorType(Enum):
    """Available integrator types."""
    RK4 = auto()
    RK4_FAST = auto()
    ADAMS_BASHFORTH = auto()
    RK45 = auto()
    RK45_FAST = auto()


# Mapping from type enum to class
_INTEGRATOR_CLASSES: Dict[IntegratorType, Type[BaseIntegrator]] = {
    IntegratorType.RK4: RK4Integrator,
    IntegratorType.RK4_FAST: RK4IntegratorFast,
    IntegratorType.ADAMS_BASHFORTH: AdamsBashforthMoultonIntegrator,
    IntegratorType.RK45: RK45Integrator,
    IntegratorType.RK45_FAST: RK45IntegratorFast,
}

# Mapping from string names to type enum
_NAME_TO_TYPE: Dict[str, IntegratorType] = {
    # RK4 variants
    'rk4': IntegratorType.RK4,
    'rk4_error': IntegratorType.RK4,
    'rk4_fast': IntegratorType.RK4_FAST,
    'rk4_simple': IntegratorType.RK4_FAST,

    # Adams-Bashforth/Moulton
    'abm': IntegratorType.ADAMS_BASHFORTH,
    'adams': IntegratorType.ADAMS_BASHFORTH,
    'adams_bashforth': IntegratorType.ADAMS_BASHFORTH,
    'adams_bashforth_moulton': IntegratorType.ADAMS_BASHFORTH,
    'predictor_corrector': IntegratorType.ADAMS_BASHFORTH,

    # RK45 Dormand-Prince
    'rk45': IntegratorType.RK45,
    'dopri': IntegratorType.RK45,
    'dormand_prince': IntegratorType.RK45,
    'adaptive': IntegratorType.RK45,
    'rk45_fast': IntegratorType.RK45_FAST,
}


class IntegratorFactory:
    """
    Factory for creating ray equation integrators.

    Supports creation by enum type or string name. String names are
    case-insensitive and support multiple aliases.

    Example:
        factory = IntegratorFactory()

        # Create by type
        integrator = factory.create(
            IntegratorType.RK45,
            my_derivative_func,
            tolerance=1e-7
        )

        # Create by name
        integrator = factory.from_name(
            'adams_bashforth',
            my_derivative_func,
            min_step=0.1
        )
    """

    @staticmethod
    def create(
        integrator_type: IntegratorType,
        derivative_func: Callable[[np.ndarray, float], np.ndarray],
        **kwargs,
    ) -> BaseIntegrator:
        """
        Create integrator by type enum.

        Args:
            integrator_type: Type of integrator to create
            derivative_func: Function computing dy/ds given (state, freq)
            **kwargs: Additional arguments passed to integrator constructor
                     (tolerance, min_step, max_step, etc.)

        Returns:
            Configured integrator instance

        Raises:
            ValueError: If integrator type is not recognized
        """
        if integrator_type not in _INTEGRATOR_CLASSES:
            raise ValueError(f"Unknown integrator type: {integrator_type}")

        integrator_class = _INTEGRATOR_CLASSES[integrator_type]
        return integrator_class(derivative_func, **kwargs)

    @staticmethod
    def from_name(
        name: str,
        derivative_func: Callable[[np.ndarray, float], np.ndarray],
        **kwargs,
    ) -> BaseIntegrator:
        """
        Create integrator by string name.

        Args:
            name: Integrator name (case-insensitive). Supported names:
                  - 'rk4', 'rk4_error': RK4 with error tracking
                  - 'rk4_fast', 'rk4_simple': RK4 without error tracking
                  - 'abm', 'adams', 'adams_bashforth': Adams-Bashforth/Moulton
                  - 'rk45', 'dopri', 'adaptive': RK45 Dormand-Prince
            derivative_func: Function computing dy/ds given (state, freq)
            **kwargs: Additional arguments for integrator

        Returns:
            Configured integrator instance

        Raises:
            ValueError: If name is not recognized
        """
        name_lower = name.lower().strip()

        if name_lower not in _NAME_TO_TYPE:
            available = ', '.join(sorted(set(_NAME_TO_TYPE.keys())))
            raise ValueError(
                f"Unknown integrator name: '{name}'. "
                f"Available: {available}"
            )

        integrator_type = _NAME_TO_TYPE[name_lower]
        return IntegratorFactory.create(integrator_type, derivative_func, **kwargs)

    @staticmethod
    def available() -> list:
        """
        List available integrator names.

        Returns:
            List of canonical integrator names
        """
        return ['rk4', 'rk4_fast', 'adams_bashforth', 'rk45', 'rk45_fast']

    @staticmethod
    def available_aliases() -> Dict[str, str]:
        """
        List all available names with their canonical form.

        Returns:
            Dict mapping alias -> canonical name
        """
        canonical = {
            IntegratorType.RK4: 'rk4',
            IntegratorType.RK4_FAST: 'rk4_fast',
            IntegratorType.ADAMS_BASHFORTH: 'adams_bashforth',
            IntegratorType.RK45: 'rk45',
            IntegratorType.RK45_FAST: 'rk45_fast',
        }
        return {
            alias: canonical[itype]
            for alias, itype in _NAME_TO_TYPE.items()
        }

    @staticmethod
    def get_description(name: str) -> str:
        """
        Get description for an integrator.

        Args:
            name: Integrator name

        Returns:
            Human-readable description
        """
        descriptions = {
            'rk4': (
                "Classical 4th-order Runge-Kutta with step doubling for error "
                "estimation. Uses 12 derivative evaluations per step. "
                "Good for debugging and when error tracking is important."
            ),
            'rk4_fast': (
                "Fast RK4 without error tracking. Uses only 4 derivative "
                "evaluations per step. Best for quick computations where "
                "error monitoring is not needed."
            ),
            'adams_bashforth': (
                "Adams-Bashforth 4-step / Adams-Moulton 3-step predictor-corrector. "
                "Uses only 2 derivative evaluations per step after startup. "
                "Most efficient for long, smooth ray paths. Requires reset() "
                "before each new ray trace."
            ),
            'rk45': (
                "Dormand-Prince RK45 adaptive integrator. Automatically adjusts "
                "step size based on local error. Uses 7 evaluations per step. "
                "Best for paths with varying curvature (e.g., near reflection)."
            ),
            'rk45_fast': (
                "Dormand-Prince RK45 with relaxed tolerance and larger step limits. "
                "Faster but less accurate. Good for initial exploration."
            ),
        }

        name_lower = name.lower().strip()
        if name_lower in _NAME_TO_TYPE:
            itype = _NAME_TO_TYPE[name_lower]
            canonical = {
                IntegratorType.RK4: 'rk4',
                IntegratorType.RK4_FAST: 'rk4_fast',
                IntegratorType.ADAMS_BASHFORTH: 'adams_bashforth',
                IntegratorType.RK45: 'rk45',
                IntegratorType.RK45_FAST: 'rk45_fast',
            }[itype]
            return descriptions.get(canonical, "No description available.")

        return f"Unknown integrator: {name}"


def create_integrator(
    name: str,
    derivative_func: Callable[[np.ndarray, float], np.ndarray],
    tolerance: float = 1e-6,
    min_step: float = 0.01,
    max_step: float = 10.0,
    **kwargs,
) -> BaseIntegrator:
    """
    Convenience function to create an integrator by name.

    This is a shorthand for IntegratorFactory.from_name() with common
    default parameters.

    Args:
        name: Integrator name (see IntegratorFactory.from_name for options)
        derivative_func: Function computing dy/ds given (state, freq)
        tolerance: Local error tolerance (default 1e-6)
        min_step: Minimum step size in km (default 0.01)
        max_step: Maximum step size in km (default 10.0)
        **kwargs: Additional integrator-specific arguments

    Returns:
        Configured integrator instance

    Example:
        def haselgrove_derivs(state, freq):
            # ... compute derivatives ...
            return derivs

        # Create adaptive integrator
        integrator = create_integrator('rk45', haselgrove_derivs, tolerance=1e-7)

        # Create efficient predictor-corrector
        integrator = create_integrator('abm', haselgrove_derivs)
    """
    return IntegratorFactory.from_name(
        name,
        derivative_func,
        tolerance=tolerance,
        min_step=min_step,
        max_step=max_step,
        **kwargs,
    )


def get_recommended_integrator(
    path_length_km: float,
    derivative_func: Callable[[np.ndarray, float], np.ndarray],
    accuracy: str = 'normal',
) -> BaseIntegrator:
    """
    Get recommended integrator based on path characteristics.

    Provides sensible defaults for different use cases.

    Args:
        path_length_km: Expected total path length
        derivative_func: Haselgrove derivative function
        accuracy: 'fast', 'normal', or 'high'

    Returns:
        Configured integrator instance

    Recommendations:
        - Short paths (<500 km): RK4 (error tracking useful)
        - Medium paths (500-2000 km): RK45 (adaptive handles curvature)
        - Long paths (>2000 km): Adams-Bashforth (efficiency matters)

        - fast: Larger tolerance, bigger steps
        - normal: Default settings
        - high: Tight tolerance, smaller steps
    """
    # Tolerance based on accuracy level
    tolerances = {
        'fast': 1e-5,
        'normal': 1e-6,
        'high': 1e-8,
    }
    tol = tolerances.get(accuracy, 1e-6)

    # Step limits based on accuracy
    if accuracy == 'fast':
        min_step, max_step = 0.1, 20.0
    elif accuracy == 'high':
        min_step, max_step = 0.001, 5.0
    else:
        min_step, max_step = 0.01, 10.0

    # Select integrator based on path length
    if path_length_km < 500:
        # Short paths: error tracking is useful
        return create_integrator(
            'rk4', derivative_func,
            tolerance=tol, min_step=min_step, max_step=max_step
        )
    elif path_length_km < 2000:
        # Medium paths: adaptive step handles reflection regions
        return create_integrator(
            'rk45', derivative_func,
            tolerance=tol, min_step=min_step, max_step=max_step
        )
    else:
        # Long paths: efficiency matters
        return create_integrator(
            'adams_bashforth', derivative_func,
            tolerance=tol, min_step=min_step, max_step=max_step
        )
