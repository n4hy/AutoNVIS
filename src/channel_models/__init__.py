"""
Channel Models Package for AutoNVIS

Provides HF channel model implementations for realistic communications
simulation alongside ionospheric ray tracing.

This package integrates:
    - Vogler-Hoffmeyer (NTIA 90-255) wideband HF channel model
    - Ray-to-channel parameter mapping
    - Abstract base class for pluggable channel models

The channel models apply statistical fading effects that complement
the deterministic ray tracing, enabling:
    - Realistic multipath fading simulation
    - Doppler spread and shift effects
    - Spread-F condition modeling
    - Communications system performance evaluation

Example Usage:
    >>> from channel_models import (
    ...     VoglerHoffmeyerModel,
    ...     ChannelConditions,
    ...     IonosphericRegion,
    ...     DisturbanceLevel
    ... )
    >>>
    >>> # Create and configure channel model
    >>> model = VoglerHoffmeyerModel(sample_rate=1e6)
    >>> conditions = ChannelConditions(
    ...     region=IonosphericRegion.MIDLATITUDE,
    ...     disturbance_level=DisturbanceLevel.MODERATE,
    ...     kp_index=3.0
    ... )
    >>> model.configure(conditions)
    >>>
    >>> # Process I/Q samples
    >>> output = model.process_samples(input_iq)

Using with Ray Tracing:
    >>> from channel_models import RayToChannelMapper
    >>> from propagation.services import PropagationService
    >>>
    >>> # Get ray paths from propagation service
    >>> coverage = propagation_service.calculate_nvis_coverage(7.0)
    >>> ray_paths = coverage['ray_paths']
    >>>
    >>> # Map rays to channel model
    >>> mapper = RayToChannelMapper(sample_rate=1e6)
    >>> config = mapper.map_rays_to_channel(ray_paths, kp_index=3.0)
    >>>
    >>> # Create channel from mapped configuration
    >>> from channel_models.hifi import VoglerHoffmeyerChannel
    >>> channel = VoglerHoffmeyerChannel(config.config)
    >>> output = channel.process(input_iq)

Available Presets:
    - equatorial: Trans-equatorial path with large delay spread
    - polar: High-latitude path with E+F modes
    - midlatitude: Stable mid-latitude conditions
    - auroral: Auroral zone with exponential Doppler
    - auroral_spread_f: Auroral with spread-F enabled
    - benign: Minimal distortion for testing
    - severe: Maximum distortion for stress testing

Author: AutoNVIS Project
"""

# Base classes and enums
from .base import (
    BaseChannelModel,
    StaticChannelModel,
    ChannelConditions,
    ChannelResponse,
    ChannelState,
    FadingType,
    IonosphericRegion,
    DisturbanceLevel
)

# Vogler-Hoffmeyer model adapter
from .vogler_hoffmeyer import (
    VoglerHoffmeyerModel,
    create_channel_model
)

# Ray-to-channel mapping
from .ray_channel_mapper import (
    RayToChannelMapper,
    RayModeParams,
    MappedChannelConfig
)

# Version
__version__ = '0.1.0'

# Public API
__all__ = [
    # Version
    '__version__',

    # Base classes
    'BaseChannelModel',
    'StaticChannelModel',
    'ChannelConditions',
    'ChannelResponse',
    'ChannelState',

    # Enums
    'FadingType',
    'IonosphericRegion',
    'DisturbanceLevel',

    # Vogler-Hoffmeyer
    'VoglerHoffmeyerModel',
    'create_channel_model',

    # Ray mapping
    'RayToChannelMapper',
    'RayModeParams',
    'MappedChannelConfig',
]


def get_available_presets() -> list:
    """
    Get list of available channel model presets.

    Returns:
        List of preset name strings
    """
    return VoglerHoffmeyerModel.available_presets()


def quick_channel(
    preset: str = "midlatitude",
    sample_rate: float = 1e6
) -> VoglerHoffmeyerModel:
    """
    Quick factory for creating a configured channel model.

    Args:
        preset: Preset name (default: "midlatitude")
        sample_rate: Sample rate in Hz (default: 1 MHz)

    Returns:
        Configured VoglerHoffmeyerModel ready for use

    Example:
        >>> channel = quick_channel("auroral")
        >>> output = channel.process_samples(input_iq)
    """
    return create_channel_model(sample_rate=sample_rate, preset=preset)
