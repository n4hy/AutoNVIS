"""
Ray Tracing to Channel Model Parameter Mapper

Bridges AutoNVIS ray tracing output to Vogler-Hoffmeyer channel model input.
Uses ray geometry (path lengths, apex altitudes, mode structure) to configure
statistical channel parameters.

This enables a hybrid approach:
    1. Ray Tracing: Provides deterministic geometry (path delays, angles, modes)
    2. Channel Model: Applies realistic time-varying fading statistics

The mapping uses physical relationships between:
    - Path length → Group delay
    - Apex altitude → Layer classification (E, F1, F2)
    - Kp index → Doppler spread
    - foF2 proximity → Delay spread

Author: AutoNVIS Project
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .base import ChannelConditions, IonosphericRegion, DisturbanceLevel
from .hifi.vogler_hoffmeyer_channel import (
    ChannelConfig,
    ModeParameters,
    CorrelationType
)


# Speed of light in km/μs
C_KM_PER_US = 299.792


@dataclass
class RayModeParams:
    """
    Parameters derived from a single ray path.

    These are extracted from ray tracing output and used to configure
    the Vogler-Hoffmeyer channel model.
    """
    name: str                           # Mode descriptive name
    delay_us: float                     # Group delay from path length
    amplitude: float                    # Relative amplitude [0, 1]
    doppler_shift_hz: float = 0.0       # Doppler shift from ionospheric motion
    layer: str = "F2"                   # Ionospheric layer: "E", "F1", "F2"
    apex_altitude_km: float = 300.0     # Ray apex altitude
    ground_range_km: float = 0.0        # Ground range
    elevation_deg: float = 90.0         # Launch elevation angle

    # Ray tracing quality
    reflected: bool = True              # Ray was reflected
    absorption_db: float = 0.0          # D-region absorption


@dataclass
class MappedChannelConfig:
    """
    Channel configuration derived from ray tracing.

    Contains both the Vogler-Hoffmeyer configuration and metadata
    about the mapping process.
    """
    config: ChannelConfig               # VH channel configuration
    source_rays: int                    # Number of rays used
    conditions: ChannelConditions       # Ionospheric conditions applied
    mapping_quality: float = 1.0        # Quality metric [0, 1]
    notes: List[str] = field(default_factory=list)


class RayToChannelMapper:
    """
    Maps ray tracing results to Vogler-Hoffmeyer channel parameters.

    Uses ray geometry to configure the statistical channel model, enabling
    realistic fading while preserving the physical mode structure from
    ray tracing.

    The mapping process:
        1. Classify rays by layer (E, F1, F2) based on apex altitude
        2. Compute group delay from path length
        3. Estimate delay spread from layer conditions
        4. Estimate Doppler spread from Kp index
        5. Create VH ModeParameters for each propagation mode
        6. Combine into ChannelConfig

    Example:
        >>> mapper = RayToChannelMapper(sample_rate=1e6)
        >>> ray_paths = propagation_service.trace_nvis(lat, lon, freq)
        >>> config = mapper.map_rays_to_channel(ray_paths, conditions)
        >>> channel = VoglerHoffmeyerChannel(config.config)
    """

    # Layer altitude boundaries (km)
    E_LAYER_MAX = 150.0
    F1_LAYER_MAX = 220.0
    # Above F1_LAYER_MAX is F2

    # Base delay spread by layer (μs) - from NTIA 90-255 Table 1
    BASE_DELAY_SPREAD = {
        "E": 30.0,      # E-layer: narrow delay spread
        "F1": 80.0,     # F1-layer: moderate
        "F2": 150.0     # F2-layer: largest spread
    }

    # Base Doppler spread by layer (Hz)
    BASE_DOPPLER_SPREAD = {
        "E": 2.0,       # E-layer: moderate
        "F1": 0.5,      # F1-layer: small
        "F2": 1.0       # F2-layer: moderate
    }

    def __init__(
        self,
        sample_rate: float = 1e6,
        max_modes: int = 3,
        min_amplitude: float = 0.1
    ):
        """
        Initialize the ray-to-channel mapper.

        Args:
            sample_rate: Sample rate in Hz for channel model
            max_modes: Maximum number of modes to include in channel
            min_amplitude: Minimum relative amplitude to include a mode
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.max_modes = max_modes
        self.min_amplitude = min_amplitude

    def map_rays_to_channel(
        self,
        ray_paths: List[Dict[str, Any]],
        conditions: Optional[ChannelConditions] = None,
        kp_index: float = 2.0,
        fof2_mhz: Optional[float] = None
    ) -> MappedChannelConfig:
        """
        Convert ray tracing output to channel model configuration.

        Args:
            ray_paths: List of ray path dictionaries from RayTracer
            conditions: Optional ionospheric conditions (or construct from kp_index)
            kp_index: Planetary K index (0-9) for Doppler scaling
            fof2_mhz: F2-layer critical frequency for delay spread scaling

        Returns:
            MappedChannelConfig with VH configuration and metadata

        Example:
            >>> paths = tracer.trace_nvis(40.0, -105.0, 7.0)
            >>> config = mapper.map_rays_to_channel(paths, kp_index=3.0)
            >>> channel = VoglerHoffmeyerChannel(config.config)
        """
        notes = []

        # Build conditions if not provided
        if conditions is None:
            conditions = self._infer_conditions(ray_paths, kp_index, fof2_mhz)
            notes.append("Conditions inferred from ray tracing")

        # Extract mode parameters from rays
        ray_modes = self._extract_ray_modes(ray_paths)
        notes.append(f"Extracted {len(ray_modes)} modes from {len(ray_paths)} rays")

        if not ray_modes:
            # No valid modes - use fallback
            notes.append("No reflected rays - using fallback configuration")
            mode_params = [self._create_fallback_mode(conditions)]
        else:
            # Convert ray modes to VH ModeParameters
            mode_params = self._convert_to_vh_modes(
                ray_modes, conditions, kp_index
            )

        # Limit number of modes
        if len(mode_params) > self.max_modes:
            mode_params = mode_params[:self.max_modes]
            notes.append(f"Truncated to {self.max_modes} modes")

        # Determine spread-F and correlation type
        spread_f = conditions.spread_f_present or kp_index >= 5
        correlation = (
            CorrelationType.EXPONENTIAL if spread_f
            else CorrelationType.GAUSSIAN
        )

        # Compute quality metric
        quality = self._compute_mapping_quality(ray_paths, ray_modes)

        # Create channel configuration
        config = ChannelConfig(
            sample_rate=self.sample_rate,
            modes=mode_params,
            spread_f_enabled=spread_f,
            k_factor=None  # Pure Rayleigh for multipath
        )

        self.logger.info(
            f"Mapped {len(ray_paths)} rays to {len(mode_params)} channel modes, "
            f"quality={quality:.2f}, spread_f={spread_f}"
        )

        return MappedChannelConfig(
            config=config,
            source_rays=len(ray_paths),
            conditions=conditions,
            mapping_quality=quality,
            notes=notes
        )

    def map_rays_to_modes(
        self,
        ray_paths: List[Dict[str, Any]],
        ionospheric_conditions: Dict[str, Any]
    ) -> List[ModeParameters]:
        """
        Convert ray tracing output to channel mode parameters.

        Simplified interface that returns ModeParameters directly.

        Args:
            ray_paths: List of ray path dicts from RayTracer
            ionospheric_conditions: Dict with foF2, hmF2, Kp, etc.

        Returns:
            List of ModeParameters for VoglerHoffmeyerChannel
        """
        kp = ionospheric_conditions.get('kp', 2.0)
        fof2 = ionospheric_conditions.get('fof2_mhz')

        conditions = ChannelConditions(
            kp_index=kp,
            fof2_mhz=fof2,
            hmf2_km=ionospheric_conditions.get('hmf2_km', 300.0)
        )

        mapped = self.map_rays_to_channel(
            ray_paths=ray_paths,
            conditions=conditions,
            kp_index=kp,
            fof2_mhz=fof2
        )

        return mapped.config.modes

    def _extract_ray_modes(
        self,
        ray_paths: List[Dict[str, Any]]
    ) -> List[RayModeParams]:
        """
        Extract propagation modes from ray paths.

        Groups rays by layer and computes aggregate mode parameters.

        Args:
            ray_paths: Ray path dictionaries from RayTracer

        Returns:
            List of RayModeParams, one per distinct mode
        """
        # Group reflected rays by layer
        layer_rays: Dict[str, List[Dict]] = {"E": [], "F1": [], "F2": []}

        for ray in ray_paths:
            if not ray.get('reflected', False):
                continue

            apex = ray.get('apex_altitude', ray.get('apex_altitude_km', 300))
            layer = self._classify_layer(apex)
            layer_rays[layer].append(ray)

        # Create mode for each populated layer
        modes = []
        total_rays = sum(len(rays) for rays in layer_rays.values())

        for layer, rays in layer_rays.items():
            if not rays:
                continue

            # Compute mean parameters
            mean_path_length = np.mean([
                r.get('path_length', r.get('path_length_km', 500))
                for r in rays
            ])
            mean_apex = np.mean([
                r.get('apex_altitude', r.get('apex_altitude_km', 300))
                for r in rays
            ])
            mean_range = np.mean([
                r.get('ground_range', r.get('ground_range_km', 0))
                for r in rays
            ])
            mean_elevation = np.mean([
                r.get('elevation', r.get('elevation_deg', 90))
                for r in rays
            ])
            mean_absorption = np.mean([
                r.get('absorption_db', 0)
                for r in rays
            ])

            # Compute delay from path length
            delay_us = (mean_path_length / C_KM_PER_US)

            # Amplitude proportional to number of rays in this layer
            amplitude = len(rays) / total_rays if total_rays > 0 else 1.0

            mode = RayModeParams(
                name=f"{layer}-layer",
                delay_us=delay_us,
                amplitude=amplitude,
                layer=layer,
                apex_altitude_km=mean_apex,
                ground_range_km=mean_range,
                elevation_deg=mean_elevation,
                reflected=True,
                absorption_db=mean_absorption
            )
            modes.append(mode)

        # Sort by delay (earliest first)
        modes.sort(key=lambda m: m.delay_us)

        return modes

    def _classify_layer(self, apex_altitude_km: float) -> str:
        """
        Classify ray by ionospheric layer based on apex altitude.

        Args:
            apex_altitude_km: Ray apex altitude in km

        Returns:
            Layer string: "E", "F1", or "F2"
        """
        if apex_altitude_km < self.E_LAYER_MAX:
            return "E"
        elif apex_altitude_km < self.F1_LAYER_MAX:
            return "F1"
        else:
            return "F2"

    def _convert_to_vh_modes(
        self,
        ray_modes: List[RayModeParams],
        conditions: ChannelConditions,
        kp_index: float
    ) -> List[ModeParameters]:
        """
        Convert extracted ray modes to Vogler-Hoffmeyer ModeParameters.

        Args:
            ray_modes: Extracted ray mode parameters
            conditions: Ionospheric conditions
            kp_index: Kp index for Doppler scaling

        Returns:
            List of VH ModeParameters
        """
        vh_modes = []

        # Find reference delay (earliest mode)
        min_delay = min(m.delay_us for m in ray_modes) if ray_modes else 0.0

        for ray_mode in ray_modes:
            # Skip weak modes
            if ray_mode.amplitude < self.min_amplitude:
                continue

            # Get base parameters for this layer
            base_delay_spread = self.BASE_DELAY_SPREAD.get(ray_mode.layer, 100.0)
            base_doppler_spread = self.BASE_DOPPLER_SPREAD.get(ray_mode.layer, 1.0)

            # Scale delay spread by conditions
            delay_spread = self._estimate_delay_spread(
                ray_mode.layer, conditions, base_delay_spread
            )

            # Scale Doppler spread by Kp
            doppler_spread = self._estimate_doppler_spread(kp_index, base_doppler_spread)

            # Compute tau_L relative to first arrival
            tau_L = ray_mode.delay_us - min_delay

            # Sigma_c is typically 1/4 to 1/2 of delay spread
            sigma_c = delay_spread / 4.0

            # Determine correlation type
            correlation = (
                CorrelationType.EXPONENTIAL if kp_index >= 4
                else CorrelationType.GAUSSIAN
            )

            vh_mode = ModeParameters(
                name=ray_mode.name,
                amplitude=ray_mode.amplitude,
                floor_amplitude=0.01,
                tau_L=tau_L,
                sigma_tau=delay_spread,
                sigma_c=sigma_c,
                sigma_D=doppler_spread,
                doppler_shift=ray_mode.doppler_shift_hz,
                doppler_shift_min_delay=0.0,
                correlation_type=correlation
            )
            vh_modes.append(vh_mode)

        # Normalize amplitudes
        total_amp = sum(m.amplitude for m in vh_modes)
        if total_amp > 0:
            for m in vh_modes:
                m.amplitude = m.amplitude / total_amp

        return vh_modes

    def _estimate_delay_spread(
        self,
        layer: str,
        conditions: ChannelConditions,
        base_spread: float
    ) -> float:
        """
        Estimate delay spread in microseconds.

        Scales base delay spread by:
            - Kp index (higher = more spread)
            - Disturbance level

        Args:
            layer: Ionospheric layer
            conditions: Ionospheric conditions
            base_spread: Base delay spread for this layer

        Returns:
            Estimated delay spread (μs)
        """
        kp_factor = 1.0 + conditions.kp_index / 10.0

        # Disturbance multiplier
        disturbance_factor = {
            DisturbanceLevel.QUIET: 0.7,
            DisturbanceLevel.MODERATE: 1.0,
            DisturbanceLevel.DISTURBED: 1.5,
            DisturbanceLevel.STORM: 2.5
        }.get(conditions.disturbance_level, 1.0)

        # Spread-F increases delay spread significantly
        spread_f_factor = 3.0 if conditions.spread_f_present else 1.0

        return base_spread * kp_factor * disturbance_factor * spread_f_factor

    def _estimate_doppler_spread(
        self,
        kp_index: float,
        base_spread: float
    ) -> float:
        """
        Estimate Doppler spread in Hz from Kp index.

        Empirical relationship:
            - Quiet (Kp 0-1): ~0.1 Hz
            - Moderate (Kp 2-3): ~0.5-2 Hz
            - Disturbed (Kp 4-5): ~2-5 Hz
            - Storm (Kp 6+): ~5-20 Hz

        Args:
            kp_index: Planetary K index (0-9)
            base_spread: Base Doppler spread for this layer

        Returns:
            Estimated Doppler spread (Hz)
        """
        # Exponential scaling: 0.1 Hz at Kp=0, ~10 Hz at Kp=8
        return base_spread * (10 ** (kp_index / 4))

    def _create_fallback_mode(
        self,
        conditions: ChannelConditions
    ) -> ModeParameters:
        """
        Create fallback mode when no rays are available.

        Uses conditions to select appropriate preset-like parameters.
        """
        # Default F2-layer mode
        return ModeParameters(
            name="F2-layer (fallback)",
            amplitude=1.0,
            floor_amplitude=0.01,
            tau_L=0.0,
            sigma_tau=100.0,
            sigma_c=25.0,
            sigma_D=self._estimate_doppler_spread(conditions.kp_index, 1.0),
            doppler_shift=0.0,
            doppler_shift_min_delay=0.0,
            correlation_type=CorrelationType.GAUSSIAN
        )

    def _infer_conditions(
        self,
        ray_paths: List[Dict[str, Any]],
        kp_index: float,
        fof2_mhz: Optional[float]
    ) -> ChannelConditions:
        """
        Infer ionospheric conditions from ray paths.

        Args:
            ray_paths: Ray path data
            kp_index: Kp index
            fof2_mhz: Optional foF2

        Returns:
            Inferred ChannelConditions
        """
        # Determine disturbance level from Kp
        if kp_index < 2:
            disturbance = DisturbanceLevel.QUIET
        elif kp_index < 4:
            disturbance = DisturbanceLevel.MODERATE
        elif kp_index < 6:
            disturbance = DisturbanceLevel.DISTURBED
        else:
            disturbance = DisturbanceLevel.STORM

        # Check for spread-F indicators
        spread_f = kp_index >= 5

        # Compute mean apex altitude for region inference
        apex_alts = [
            r.get('apex_altitude', r.get('apex_altitude_km', 300))
            for r in ray_paths if r.get('reflected', False)
        ]
        mean_apex = np.mean(apex_alts) if apex_alts else 300.0

        return ChannelConditions(
            region=IonosphericRegion.MIDLATITUDE,  # Default
            disturbance_level=disturbance,
            spread_f_present=spread_f,
            kp_index=kp_index,
            fof2_mhz=fof2_mhz,
            apex_altitude_km=mean_apex
        )

    def _compute_mapping_quality(
        self,
        ray_paths: List[Dict[str, Any]],
        ray_modes: List[RayModeParams]
    ) -> float:
        """
        Compute quality metric for the ray-to-channel mapping.

        Quality is based on:
            - Number of reflected rays
            - Mode diversity (multiple layers)
            - Ray coverage (elevation angles)

        Returns:
            Quality score [0, 1]
        """
        if not ray_paths:
            return 0.0

        reflected = sum(1 for r in ray_paths if r.get('reflected', False))
        reflection_rate = reflected / len(ray_paths)

        # Mode diversity: more layers = better
        layers = set(m.layer for m in ray_modes)
        diversity_score = len(layers) / 3.0  # Max 3 layers

        # Combined quality
        quality = 0.6 * reflection_rate + 0.4 * diversity_score

        return min(quality, 1.0)

    def estimate_channel_from_luf_muf(
        self,
        luf_mhz: float,
        muf_mhz: float,
        freq_mhz: float,
        kp_index: float = 2.0
    ) -> MappedChannelConfig:
        """
        Estimate channel parameters from LUF/MUF without ray tracing.

        Useful when detailed ray tracing is not available but LUF/MUF
        predictions exist.

        Args:
            luf_mhz: Lowest Usable Frequency
            muf_mhz: Maximum Usable Frequency
            freq_mhz: Operating frequency
            kp_index: Kp index

        Returns:
            MappedChannelConfig with estimated parameters
        """
        notes = ["Estimated from LUF/MUF (no ray tracing)"]

        # Proximity to MUF affects delay spread
        if muf_mhz > luf_mhz:
            muf_proximity = (freq_mhz - luf_mhz) / (muf_mhz - luf_mhz)
        else:
            muf_proximity = 0.5

        # Near MUF: larger delay spread (oblique paths)
        # Near LUF: smaller delay spread (steep paths)
        delay_spread = 50.0 + 150.0 * muf_proximity

        # Create single-mode configuration
        mode = ModeParameters(
            name="F2-layer (estimated)",
            amplitude=1.0,
            floor_amplitude=0.01,
            tau_L=0.0,
            sigma_tau=delay_spread,
            sigma_c=delay_spread / 4.0,
            sigma_D=self._estimate_doppler_spread(kp_index, 1.0),
            doppler_shift=0.0,
            doppler_shift_min_delay=0.0,
            correlation_type=CorrelationType.GAUSSIAN
        )

        conditions = ChannelConditions(kp_index=kp_index)

        config = ChannelConfig(
            sample_rate=self.sample_rate,
            modes=[mode],
            spread_f_enabled=(kp_index >= 5)
        )

        return MappedChannelConfig(
            config=config,
            source_rays=0,
            conditions=conditions,
            mapping_quality=0.5,  # Lower quality without rays
            notes=notes
        )
