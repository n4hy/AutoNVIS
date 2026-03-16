"""
Vogler-Hoffmeyer HF Channel Model Adapter

Wraps the NTIA 90-255 channel model implementation from the hifi package
to implement the BaseChannelModel interface for AutoNVIS integration.

The Vogler-Hoffmeyer model is a stochastic wideband HF channel model that
simulates:
    - Dispersion: Frequency-dependent group delay from ionospheric refraction
    - Scattering: From ionospheric irregularities (including spread-F)
    - Doppler spread: Frequency spreading from ionospheric motion
    - Multipath: Multiple propagation modes (E-layer, F-layer)

References:
    - NTIA Report 90-255: "A Model for Wideband HF Propagation Channels"
      by L.E. Vogler and J.A. Hoffmeyer (1990)
    - CCIR Report 549: Watterson model (narrowband baseline)

Author: AutoNVIS Project
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np

from .base import (
    BaseChannelModel,
    ChannelConditions,
    ChannelResponse,
    ChannelState,
    FadingType,
    IonosphericRegion,
    DisturbanceLevel
)

# Import from the hifi package
from .hifi.vogler_hoffmeyer_channel import (
    VoglerHoffmeyerChannel,
    ChannelConfig,
    ModeParameters,
    CorrelationType,
    get_preset,
    list_presets
)


# Mapping from IonosphericRegion to preset names
REGION_TO_PRESET = {
    IonosphericRegion.EQUATORIAL: "equatorial",
    IonosphericRegion.MIDLATITUDE: "midlatitude",
    IonosphericRegion.POLAR: "polar",
    IonosphericRegion.AURORAL: "auroral"
}

# Mapping from DisturbanceLevel to preset names (fallback)
DISTURBANCE_TO_PRESET = {
    DisturbanceLevel.QUIET: "benign",
    DisturbanceLevel.MODERATE: "midlatitude",
    DisturbanceLevel.DISTURBED: "auroral",
    DisturbanceLevel.STORM: "auroral_spread_f"
}


class VoglerHoffmeyerModel(BaseChannelModel):
    """
    Vogler-Hoffmeyer HF channel model adapter.

    Implements the BaseChannelModel interface using the NTIA 90-255
    stochastic channel model. This model is appropriate for:
        - Wideband HF signals (up to 1 MHz+ bandwidth)
        - Communications simulation with realistic fading
        - Evaluation of adaptive equalizers and modems

    The model supports multiple ionospheric condition presets from
    NTIA Report 90-255 Table 1:
        - equatorial: Trans-equatorial path with large delay spread
        - polar: High-latitude with E+F layer modes
        - midlatitude: Stable mid-latitude conditions
        - auroral: Auroral zone with exponential Doppler
        - auroral_spread_f: Auroral with spread-F enabled
        - benign: Minimal distortion for testing

    Example:
        >>> model = VoglerHoffmeyerModel(sample_rate=1e6)
        >>> conditions = ChannelConditions(
        ...     region=IonosphericRegion.MIDLATITUDE,
        ...     disturbance_level=DisturbanceLevel.MODERATE
        ... )
        >>> model.configure(conditions)
        >>> output = model.process_samples(input_iq)
    """

    def __init__(
        self,
        sample_rate: float = 1e6,
        random_seed: Optional[int] = None,
        k_factor: Optional[float] = None
    ):
        """
        Initialize the Vogler-Hoffmeyer channel model.

        Args:
            sample_rate: Sample rate in Hz (default 1 MHz for wideband HF)
            random_seed: Optional seed for reproducible fading sequences
            k_factor: Rician K-factor (None=Rayleigh, >0=Rician with LOS)
        """
        self.logger = logging.getLogger(__name__)

        self._sample_rate = sample_rate
        self._random_seed = random_seed
        self._k_factor = k_factor

        # Internal channel instance (created on configure)
        self._channel: Optional[VoglerHoffmeyerChannel] = None
        self._config: Optional[ChannelConfig] = None
        self._conditions: Optional[ChannelConditions] = None
        self._preset_name: Optional[str] = None

        self.logger.debug(
            f"VoglerHoffmeyerModel initialized: fs={sample_rate/1e6:.3f} MHz, "
            f"seed={random_seed}, K={k_factor}"
        )

    def configure(self, conditions: ChannelConditions) -> None:
        """
        Configure the channel model for given ionospheric conditions.

        Maps conditions to a Vogler-Hoffmeyer preset and initializes
        the internal channel model.

        Args:
            conditions: Ionospheric conditions from ray tracing or space weather

        Raises:
            ValueError: If conditions result in invalid configuration
        """
        self._conditions = conditions
        preset_name = self._select_preset(conditions)
        self._configure_from_preset_internal(preset_name)

        self.logger.info(
            f"Channel configured: preset='{preset_name}', "
            f"region={conditions.region.value}, "
            f"disturbance={conditions.disturbance_level.value}, "
            f"spread_f={conditions.spread_f_present}"
        )

    def configure_from_preset(self, preset_name: str) -> None:
        """
        Configure using a named preset from NTIA 90-255 Table 1.

        Args:
            preset_name: One of: equatorial, polar, midlatitude, auroral,
                        auroral_spread_f, auroral_complex, benign, severe

        Raises:
            ValueError: If preset_name is unknown
        """
        available = list_presets()
        if preset_name not in available:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {', '.join(available)}"
            )

        self._configure_from_preset_internal(preset_name)
        self._conditions = None  # Clear conditions since using preset directly

        self.logger.info(f"Channel configured from preset: '{preset_name}'")

    def _configure_from_preset_internal(self, preset_name: str) -> None:
        """Internal preset configuration"""
        self._preset_name = preset_name

        # Get preset configuration
        self._config = get_preset(preset_name, self._sample_rate)

        # Apply overrides
        if self._random_seed is not None:
            self._config.random_seed = self._random_seed
        if self._k_factor is not None:
            self._config.k_factor = self._k_factor

        # Create channel instance
        self._channel = VoglerHoffmeyerChannel(self._config)

    def _select_preset(self, conditions: ChannelConditions) -> str:
        """
        Map ionospheric conditions to Vogler-Hoffmeyer preset.

        Implements intelligent preset selection based on:
            1. Spread-F presence (highest priority)
            2. Geographic region
            3. Disturbance level
            4. Kp index

        Args:
            conditions: Ionospheric conditions

        Returns:
            Preset name string
        """
        # Spread-F takes precedence
        if conditions.spread_f_present:
            return "auroral_spread_f"

        # Check for storm conditions based on Kp
        if conditions.kp_index >= 6:
            return "auroral_spread_f"
        elif conditions.kp_index >= 4:
            return "auroral"

        # Map by region
        if conditions.region == IonosphericRegion.EQUATORIAL:
            return "equatorial"
        elif conditions.region == IonosphericRegion.POLAR:
            return "polar"
        elif conditions.region == IonosphericRegion.AURORAL:
            return "auroral"

        # Mid-latitude: use disturbance level
        if conditions.disturbance_level == DisturbanceLevel.QUIET:
            return "benign"
        elif conditions.disturbance_level == DisturbanceLevel.DISTURBED:
            return "auroral"
        elif conditions.disturbance_level == DisturbanceLevel.STORM:
            return "auroral_spread_f"

        # Default to midlatitude
        return "midlatitude"

    def process_samples(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Apply Vogler-Hoffmeyer channel effects to I/Q samples.

        Processes input through the configured channel model, applying:
            - Tapped delay line multipath
            - Time-varying Rayleigh/Rician fading
            - Doppler shift and spread
            - Optional spread-F random multiplication

        Args:
            input_samples: Complex I/Q samples

        Returns:
            Complex output samples with channel effects

        Raises:
            RuntimeError: If channel not configured
        """
        if self._channel is None:
            raise RuntimeError("Channel not configured. Call configure() first.")

        # Ensure input is complex128
        samples = np.asarray(input_samples, dtype=np.complex128)

        # Process through channel
        output = self._channel.process(samples)

        return output

    def get_channel_response(self) -> ChannelResponse:
        """
        Get channel response for the dominant (primary) mode.

        For multi-mode channels, returns the first/strongest mode.

        Returns:
            ChannelResponse with delay, Doppler, and fading parameters

        Raises:
            RuntimeError: If channel not configured
        """
        responses = self.get_all_mode_responses()
        if not responses:
            raise RuntimeError("Channel not configured")
        return responses[0]

    def get_all_mode_responses(self) -> List[ChannelResponse]:
        """
        Get channel response for all configured propagation modes.

        Returns:
            List of ChannelResponse objects, one per mode
        """
        if self._config is None:
            return []

        responses = []
        for mode in self._config.modes:
            # Determine fading type
            if self._config.k_factor is not None and self._config.k_factor > 0:
                fading_type = FadingType.RICIAN
                k_factor = self._config.k_factor
            else:
                fading_type = FadingType.RAYLEIGH
                k_factor = None

            # Map mode to layer
            layer = self._infer_layer(mode.name, mode.tau_L)

            response = ChannelResponse(
                delay_us=mode.tau_c,
                delay_spread_us=mode.sigma_tau,
                min_delay_us=mode.tau_L,
                max_delay_us=mode.tau_U,
                doppler_shift_hz=mode.doppler_shift,
                doppler_spread_hz=mode.sigma_D,
                path_loss_db=-20 * np.log10(mode.amplitude) if mode.amplitude > 0 else float('inf'),
                amplitude=mode.amplitude,
                fading_type=fading_type,
                k_factor=k_factor,
                mode_name=mode.name,
                layer=layer
            )
            responses.append(response)

        return responses

    def _infer_layer(self, mode_name: str, min_delay_us: float) -> str:
        """Infer ionospheric layer from mode name or delay"""
        name_lower = mode_name.lower()
        if 'e-layer' in name_lower or 'e layer' in name_lower:
            return "E"
        elif 'f1' in name_lower or 'f-1' in name_lower:
            return "F1"
        elif 'f2' in name_lower or 'f-2' in name_lower:
            return "F2"
        elif 'f-layer' in name_lower or 'f layer' in name_lower:
            # Distinguish F1/F2 by delay (rough heuristic)
            return "F2" if min_delay_us > 200 else "F1"
        return ""

    def reset(self) -> None:
        """Reset channel state for processing a new signal."""
        if self._channel is not None:
            self._channel.reset()
            self.logger.debug("Channel state reset")

    def get_state(self) -> ChannelState:
        """
        Get current channel state for checkpointing.

        Returns:
            ChannelState snapshot including time index and mode states
        """
        if self._channel is None:
            return ChannelState()

        # Extract mode states from internal channel
        mode_states = []
        for md in self._channel.mode_data:
            mode_state = {
                'buffer': md['buffer'].tobytes().hex(),
                'gauss_C_state': md['gauss_C_state'].tobytes().hex() if 'gauss_C_state' in md else None,
            }
            if 'exp_x_real_state' in md:
                mode_state['exp_x_real_state'] = md['exp_x_real_state'].tobytes().hex()
                mode_state['exp_x_imag_state'] = md['exp_x_imag_state'].tobytes().hex()
            mode_states.append(mode_state)

        return ChannelState(
            time_index=self._channel.time_index,
            mode_states=mode_states
        )

    def set_state(self, state: ChannelState) -> None:
        """
        Restore channel state from checkpoint.

        Args:
            state: Previously saved ChannelState
        """
        if self._channel is None:
            self.logger.warning("Cannot restore state: channel not configured")
            return

        self._channel.time_index = state.time_index

        # Restore mode states
        for i, mode_state in enumerate(state.mode_states):
            if i >= len(self._channel.mode_data):
                break
            md = self._channel.mode_data[i]

            if 'buffer' in mode_state and mode_state['buffer']:
                md['buffer'] = np.frombuffer(
                    bytes.fromhex(mode_state['buffer']),
                    dtype=np.complex128
                ).copy()

            if mode_state.get('gauss_C_state'):
                md['gauss_C_state'] = np.frombuffer(
                    bytes.fromhex(mode_state['gauss_C_state']),
                    dtype=np.complex128
                ).copy()

            if mode_state.get('exp_x_real_state'):
                md['exp_x_real_state'] = np.frombuffer(
                    bytes.fromhex(mode_state['exp_x_real_state']),
                    dtype=np.float64
                ).copy()
                md['exp_x_imag_state'] = np.frombuffer(
                    bytes.fromhex(mode_state['exp_x_imag_state']),
                    dtype=np.float64
                ).copy()

        self.logger.debug(f"Channel state restored: time_index={state.time_index}")

    @property
    def name(self) -> str:
        """Model name for logging"""
        return "Vogler-Hoffmeyer (NTIA 90-255)"

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz"""
        return self._sample_rate

    @property
    def is_configured(self) -> bool:
        """Check if channel is configured and ready"""
        return self._channel is not None

    @property
    def preset_name(self) -> Optional[str]:
        """Currently active preset name"""
        return self._preset_name

    @property
    def conditions(self) -> Optional[ChannelConditions]:
        """Ionospheric conditions used for configuration"""
        return self._conditions

    @property
    def config(self) -> Optional[ChannelConfig]:
        """Internal channel configuration"""
        return self._config

    def compute_scattering_function(
        self,
        num_delay_bins: int = 64,
        num_doppler_bins: int = 64
    ) -> tuple:
        """
        Compute the theoretical channel scattering function S(τ, f_D).

        The scattering function shows the distribution of signal power
        in delay-Doppler space. It's the primary verification tool from
        NTIA 90-255.

        Args:
            num_delay_bins: Number of delay bins
            num_doppler_bins: Number of Doppler frequency bins

        Returns:
            Tuple of (delay_axis_us, doppler_axis_hz, scattering_function)

        Raises:
            RuntimeError: If channel not configured
        """
        if self._channel is None:
            raise RuntimeError("Channel not configured")

        return self._channel.compute_scattering_function(
            num_delay_bins=num_delay_bins,
            num_doppler_bins=num_doppler_bins
        )

    def set_k_factor(self, k_factor: Optional[float]) -> None:
        """
        Set Rician K-factor for fading control.

        Args:
            k_factor: K-factor value:
                - None or 0: Pure Rayleigh fading
                - 1: Equal direct and scattered power
                - 10: Strong direct path (mild fading)
                - infinity: No fading (static)

        Note:
            Requires reconfiguration to take effect on existing channel.
        """
        self._k_factor = k_factor
        if self._config is not None:
            self._config.k_factor = k_factor
            # Recreate channel with new K-factor
            self._channel = VoglerHoffmeyerChannel(self._config)

    def enable_spread_f(self, enabled: bool = True) -> None:
        """
        Enable or disable spread-F simulation.

        Spread-F applies additional random amplitude modulation to
        simulate diffuse ionospheric scattering during disturbed conditions.

        Args:
            enabled: True to enable spread-F effects

        Note:
            Requires reconfiguration to take effect on existing channel.
        """
        if self._config is not None:
            self._config.spread_f_enabled = enabled
            self._channel = VoglerHoffmeyerChannel(self._config)
            self.logger.info(f"Spread-F {'enabled' if enabled else 'disabled'}")

    @staticmethod
    def available_presets() -> List[str]:
        """
        List available preset configurations.

        Returns:
            List of preset names
        """
        return list_presets()

    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current channel configuration.

        Returns:
            Dictionary with configuration details
        """
        if self._config is None:
            return {'configured': False}

        return {
            'configured': True,
            'preset': self._preset_name,
            'sample_rate_hz': self._sample_rate,
            'num_modes': len(self._config.modes),
            'spread_f_enabled': self._config.spread_f_enabled,
            'k_factor': self._config.k_factor,
            'modes': [
                {
                    'name': m.name,
                    'delay_spread_us': m.sigma_tau,
                    'doppler_spread_hz': m.sigma_D,
                    'correlation': m.correlation_type.value
                }
                for m in self._config.modes
            ]
        }


def create_channel_model(
    sample_rate: float = 1e6,
    preset: Optional[str] = None,
    conditions: Optional[ChannelConditions] = None,
    k_factor: Optional[float] = None,
    random_seed: Optional[int] = None
) -> VoglerHoffmeyerModel:
    """
    Factory function to create and configure a Vogler-Hoffmeyer channel model.

    Args:
        sample_rate: Sample rate in Hz
        preset: Optional preset name (overrides conditions if both given)
        conditions: Optional ionospheric conditions
        k_factor: Optional Rician K-factor
        random_seed: Optional random seed

    Returns:
        Configured VoglerHoffmeyerModel

    Example:
        >>> model = create_channel_model(
        ...     sample_rate=1e6,
        ...     preset="midlatitude"
        ... )
    """
    model = VoglerHoffmeyerModel(
        sample_rate=sample_rate,
        random_seed=random_seed,
        k_factor=k_factor
    )

    if preset is not None:
        model.configure_from_preset(preset)
    elif conditions is not None:
        model.configure(conditions)
    else:
        # Default to midlatitude
        model.configure_from_preset("midlatitude")

    return model
