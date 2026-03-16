"""
Base Channel Model Interface for AutoNVIS

Provides an abstract base class for HF channel models, enabling pluggable
channel models for communications simulation alongside ray tracing.

This interface supports:
    - Vogler-Hoffmeyer (NTIA 90-255) statistical channel model
    - Watterson narrowband model (legacy compatibility)
    - Future channel model implementations

Author: AutoNVIS Project
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np


class FadingType(Enum):
    """Type of fading applied by the channel"""
    STATIC = "static"           # No fading (ideal channel)
    RAYLEIGH = "rayleigh"       # Pure scattered multipath
    RICIAN = "rician"           # Direct + scattered components
    LOGNORMAL = "lognormal"     # Slow fading (shadowing)


class IonosphericRegion(Enum):
    """Ionospheric region classification"""
    EQUATORIAL = "equatorial"   # ±20° latitude
    MIDLATITUDE = "midlatitude" # 20-60° latitude
    POLAR = "polar"             # >60° latitude
    AURORAL = "auroral"         # Auroral oval (variable)


class DisturbanceLevel(Enum):
    """Ionospheric disturbance level"""
    QUIET = "quiet"             # Kp 0-2, calm conditions
    MODERATE = "moderate"       # Kp 2-4, typical conditions
    DISTURBED = "disturbed"     # Kp 4-6, elevated activity
    STORM = "storm"             # Kp 6+, geomagnetic storm


@dataclass
class ChannelResponse:
    """
    Channel response parameters for a propagation path.

    Describes the statistical properties of the channel including
    delay spread, Doppler spread, and fading characteristics.

    All delay values are in microseconds, Doppler values in Hz.
    """

    # Delay parameters (microseconds)
    delay_us: float = 0.0               # Mean group delay
    delay_spread_us: float = 0.0        # RMS delay spread (multipath extent)
    min_delay_us: float = 0.0           # Minimum delay (first arrival)
    max_delay_us: float = 0.0           # Maximum delay (last arrival)

    # Doppler parameters (Hz)
    doppler_shift_hz: float = 0.0       # Mean Doppler shift
    doppler_spread_hz: float = 0.0      # RMS Doppler spread (fading rate)

    # Amplitude parameters
    path_loss_db: float = 0.0           # Total path loss
    amplitude: float = 1.0              # Relative amplitude [0, 1]

    # Fading characteristics
    fading_type: FadingType = FadingType.RAYLEIGH
    k_factor: Optional[float] = None    # Rician K-factor (if applicable)

    # Mode identification
    mode_name: str = ""                 # E.g., "F-layer low-ray"
    layer: str = ""                     # "E", "F1", "F2"

    def __post_init__(self):
        """Validate parameters"""
        if self.delay_spread_us < 0:
            raise ValueError("Delay spread must be non-negative")
        if self.doppler_spread_hz < 0:
            raise ValueError("Doppler spread must be non-negative")
        if self.amplitude < 0 or self.amplitude > 1:
            raise ValueError("Amplitude must be in [0, 1]")


@dataclass
class ChannelConditions:
    """
    Ionospheric conditions for channel model configuration.

    Maps environmental parameters to channel model presets and parameters.
    These conditions can be derived from:
        - Space weather data (Kp, solar flux)
        - Ray tracing output (mode structure, apex altitudes)
        - User configuration (region, expected conditions)
    """

    # Geographic classification
    region: IonosphericRegion = IonosphericRegion.MIDLATITUDE

    # Disturbance state
    disturbance_level: DisturbanceLevel = DisturbanceLevel.MODERATE
    spread_f_present: bool = False      # Spread-F condition detected

    # Space weather indices
    kp_index: float = 2.0               # Planetary K index (0-9)
    solar_flux: float = 100.0           # F10.7 solar flux (SFU)
    xray_flux: float = 0.0              # GOES X-ray flux (W/m²)

    # Ionospheric parameters (from ray tracing or ionosonde)
    fof2_mhz: Optional[float] = None    # F2-layer critical frequency
    hmf2_km: Optional[float] = None     # F2-layer peak height
    foe_mhz: Optional[float] = None     # E-layer critical frequency

    # Path characteristics
    path_length_km: float = 0.0         # Ground distance
    apex_altitude_km: float = 300.0     # Ray apex altitude

    def get_preset_name(self) -> str:
        """
        Map conditions to Vogler-Hoffmeyer preset name.

        Returns:
            Preset name string for channel configuration
        """
        if self.spread_f_present:
            return "auroral_spread_f"

        if self.region == IonosphericRegion.EQUATORIAL:
            return "equatorial"

        if self.region == IonosphericRegion.POLAR:
            return "polar"

        if self.region == IonosphericRegion.AURORAL:
            return "auroral"

        if self.disturbance_level in (DisturbanceLevel.DISTURBED, DisturbanceLevel.STORM):
            return "auroral"

        if self.disturbance_level == DisturbanceLevel.QUIET:
            return "benign"

        return "midlatitude"


@dataclass
class ChannelState:
    """
    Snapshot of channel model internal state.

    Used for checkpointing and state restoration.
    """

    time_index: int = 0                 # Sample counter for phase continuity
    random_state: Optional[bytes] = None  # Serialized RNG state
    mode_states: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            'time_index': self.time_index,
            'random_state': self.random_state.hex() if self.random_state else None,
            'mode_states': self.mode_states
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelState':
        """Deserialize state from dictionary"""
        random_state = bytes.fromhex(data['random_state']) if data.get('random_state') else None
        return cls(
            time_index=data.get('time_index', 0),
            random_state=random_state,
            mode_states=data.get('mode_states', [])
        )


class BaseChannelModel(ABC):
    """
    Abstract base class for HF channel models.

    Defines the interface for all channel model implementations in AutoNVIS.
    Channel models apply realistic time-varying fading effects to I/Q samples,
    complementing the deterministic ray tracing with statistical channel behavior.

    Subclasses must implement:
        - configure(): Set up model for given ionospheric conditions
        - process_samples(): Apply channel effects to I/Q samples
        - get_channel_response(): Return current channel parameters
        - reset(): Reset channel state

    Example usage:
        >>> model = VoglerHoffmeyerModel(sample_rate=1e6)
        >>> conditions = ChannelConditions(region=IonosphericRegion.MIDLATITUDE)
        >>> model.configure(conditions)
        >>> output = model.process_samples(input_iq)
    """

    @abstractmethod
    def configure(self, conditions: ChannelConditions) -> None:
        """
        Configure the channel model for given ionospheric conditions.

        This method maps ionospheric conditions to internal model parameters,
        selecting appropriate presets and configuring mode structure.

        Args:
            conditions: Ionospheric conditions including region, disturbance
                       level, and space weather indices

        Raises:
            ValueError: If conditions are invalid or unsupported
        """
        pass

    @abstractmethod
    def configure_from_preset(self, preset_name: str) -> None:
        """
        Configure the channel model using a named preset.

        Presets correspond to NTIA 90-255 Table 1 conditions:
            - "equatorial": Trans-equatorial path
            - "polar": High-latitude path
            - "midlatitude": Mid-latitude short path
            - "auroral": Auroral zone
            - "auroral_spread_f": Auroral with spread-F
            - "benign": Quiet conditions

        Args:
            preset_name: Name of the preset configuration

        Raises:
            ValueError: If preset_name is unknown
        """
        pass

    @abstractmethod
    def process_samples(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Apply channel effects to I/Q samples.

        Processes input samples through the configured channel model,
        applying multipath delay, Doppler spread, and fading effects.

        The channel maintains state between calls for phase continuity
        and consistent fading statistics.

        Args:
            input_samples: Complex I/Q samples (numpy array, dtype=complex128)

        Returns:
            Complex output samples after channel effects

        Raises:
            RuntimeError: If channel not configured
        """
        pass

    @abstractmethod
    def get_channel_response(self) -> ChannelResponse:
        """
        Get current channel response parameters.

        Returns aggregate channel characteristics for the configured mode.
        For multi-mode channels, returns the dominant mode parameters.

        Returns:
            ChannelResponse with delay, Doppler, and fading parameters

        Raises:
            RuntimeError: If channel not configured
        """
        pass

    @abstractmethod
    def get_all_mode_responses(self) -> List[ChannelResponse]:
        """
        Get channel response for all configured modes.

        Returns individual mode parameters for multi-mode channels
        (e.g., E-layer + F-layer paths).

        Returns:
            List of ChannelResponse objects, one per mode
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset channel state for processing a new signal.

        Clears internal buffers, resets time indices, and reinitializes
        random processes. Configuration is preserved.
        """
        pass

    @abstractmethod
    def get_state(self) -> ChannelState:
        """
        Get current channel state for checkpointing.

        Returns:
            ChannelState snapshot
        """
        pass

    @abstractmethod
    def set_state(self, state: ChannelState) -> None:
        """
        Restore channel state from checkpoint.

        Args:
            state: Previously saved ChannelState
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable model name for logging.

        Returns:
            Model name string (e.g., "Vogler-Hoffmeyer (NTIA 90-255)")
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """
        Sample rate in Hz.

        Returns:
            Sample rate used by this channel model
        """
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """
        Check if channel is configured and ready.

        Returns:
            True if configure() has been called successfully
        """
        pass

    def get_impulse_response(self, num_samples: int = 1024) -> np.ndarray:
        """
        Get the instantaneous channel impulse response.

        Useful for visualization and verification. This processes an
        impulse through the channel to show the multipath structure.

        Args:
            num_samples: Length of impulse response

        Returns:
            Complex impulse response

        Note:
            This temporarily modifies and restores channel state.
        """
        if not self.is_configured:
            raise RuntimeError("Channel not configured")

        # Save current state
        saved_state = self.get_state()

        # Reset for clean impulse response
        self.reset()

        # Create impulse
        impulse = np.zeros(num_samples, dtype=np.complex128)
        impulse[0] = 1.0

        # Process through channel
        response = self.process_samples(impulse)

        # Restore original state
        self.set_state(saved_state)

        return response

    def get_frequency_response(self, num_bins: int = 1024) -> tuple:
        """
        Get the instantaneous channel frequency response.

        Args:
            num_bins: Number of frequency bins (FFT size)

        Returns:
            Tuple of (frequency_axis_hz, complex_response)
        """
        h = self.get_impulse_response(num_bins)
        H = np.fft.fft(h)
        f = np.fft.fftfreq(num_bins, 1 / self.sample_rate)
        return f, H


class StaticChannelModel(BaseChannelModel):
    """
    Static (ideal) channel model for baseline comparisons.

    Applies only fixed attenuation and delay, no fading effects.
    Useful for isolating other system effects during testing.
    """

    def __init__(self, sample_rate: float = 1e6, attenuation_db: float = 0.0):
        self._sample_rate = sample_rate
        self._attenuation_db = attenuation_db
        self._attenuation_linear = 10 ** (-attenuation_db / 20)
        self._configured = False

    def configure(self, conditions: ChannelConditions) -> None:
        self._configured = True

    def configure_from_preset(self, preset_name: str) -> None:
        self._configured = True

    def process_samples(self, input_samples: np.ndarray) -> np.ndarray:
        if not self._configured:
            raise RuntimeError("Channel not configured")
        return input_samples * self._attenuation_linear

    def get_channel_response(self) -> ChannelResponse:
        return ChannelResponse(
            path_loss_db=self._attenuation_db,
            fading_type=FadingType.STATIC,
            mode_name="Static"
        )

    def get_all_mode_responses(self) -> List[ChannelResponse]:
        return [self.get_channel_response()]

    def reset(self) -> None:
        pass

    def get_state(self) -> ChannelState:
        return ChannelState()

    def set_state(self, state: ChannelState) -> None:
        pass

    @property
    def name(self) -> str:
        return "Static Channel"

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def is_configured(self) -> bool:
        return self._configured
