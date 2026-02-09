"""
synthetic_csi.py

Simulates synthetic Wi-Fi-like channel state information (CSI) derived from
a continuous-space world model.

This module represents the measurement layer, bridging:
- Continuous ground-truth physics (world.world)
- Downstream discretization and inference

CSI measurements are intentionally lossy, ambiguous, and noisy.
They do NOT encode explicit object identity.

All spatial units are meters.
All time units are seconds.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np

from world.world import World


@dataclass(frozen=True)
class SyntheticCSIFrame:
    """
    A single synthetic CSI measurement frame.

    This is a lightweight, immutable data structure representing
    channel measurements at a specific point in time.

    Attributes
    ----------
    timestamp : float
        Simulation time in seconds when this frame was captured.
    amplitudes : np.ndarray
        Per-subcarrier amplitude values. Shape: (num_subcarriers,).
        Units are arbitrary (not calibrated to real power).
    phases : np.ndarray
        Per-subcarrier phase values in radians. Shape: (num_subcarriers,).
        Range is unbounded (not wrapped to [-pi, pi]).
    """
    timestamp: float
    amplitudes: np.ndarray
    phases: np.ndarray

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers in this frame."""
        return len(self.amplitudes)

    def get_complex(self) -> np.ndarray:
        """
        Get CSI as complex values (amplitude * exp(j * phase)).

        Returns
        -------
        np.ndarray
            Complex-valued CSI. Shape: (num_subcarriers,).
        """
        return self.amplitudes * np.exp(1j * self.phases)


class SyntheticCSIGenerator:
    """
    Generates synthetic CSI frames from a continuous-space world model.

    The generator simulates Wi-Fi-like channel measurements by:
    1. Computing baseline path loss between transmitter and receiver
    2. Querying the world's disturbance field along the signal path
    3. Mapping disturbances to amplitude and phase perturbations
    4. Adding controlled measurement noise

    The resulting CSI is a lossy, indirect measurement that does not
    preserve object identity or exact spatial information.

    Parameters
    ----------
    world : World
        Reference to the world model providing disturbance information.
    tx_position : Tuple[float, float, float]
        Transmitter position (x, y, z) in meters.
    rx_position : Tuple[float, float, float]
        Receiver position (x, y, z) in meters.
    num_subcarriers : int
        Number of frequency subcarriers to simulate.
    center_frequency_hz : float, optional
        Center frequency in Hz. Defaults to 5.8 GHz (Wi-Fi 5 GHz band).
    bandwidth_hz : float, optional
        Total bandwidth in Hz. Defaults to 40 MHz.
    amplitude_noise_std : float, optional
        Standard deviation of amplitude noise. Defaults to 0.01.
    phase_noise_std : float, optional
        Standard deviation of phase noise in radians. Defaults to 0.05.
    num_sample_points : int, optional
        Number of points to sample along TX-RX path for disturbance
        integration. Defaults to 10.
    random_seed : Optional[int], optional
        Seed for reproducible noise generation. If None, uses system entropy.

    Attributes
    ----------
    tx_position : Tuple[float, float, float]
        Transmitter position in meters.
    rx_position : Tuple[float, float, float]
        Receiver position in meters.
    num_subcarriers : int
        Number of frequency subcarriers.
    """

    # Speed of light in meters per second
    SPEED_OF_LIGHT: float = 299792458.0

    def __init__(
        self,
        world: World,
        tx_position: Tuple[float, float, float],
        rx_position: Tuple[float, float, float],
        num_subcarriers: int,
        center_frequency_hz: float = 5.8e9,
        bandwidth_hz: float = 40e6,
        amplitude_noise_std: float = 0.01,
        phase_noise_std: float = 0.05,
        num_sample_points: int = 10,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the synthetic CSI generator."""
        self._world = world
        self._tx_position = tx_position
        self._rx_position = rx_position
        self._num_subcarriers = num_subcarriers
        self._center_frequency_hz = center_frequency_hz
        self._bandwidth_hz = bandwidth_hz
        self._amplitude_noise_std = amplitude_noise_std
        self._phase_noise_std = phase_noise_std
        self._num_sample_points = num_sample_points

        # Initialize random number generator for reproducibility
        self._rng = np.random.default_rng(random_seed)

        # Precompute geometry
        self._tx_rx_distance = self._compute_distance(tx_position, rx_position)
        self._subcarrier_frequencies = self._compute_subcarrier_frequencies()
        self._wavelengths = self.SPEED_OF_LIGHT / self._subcarrier_frequencies

    @property
    def tx_position(self) -> Tuple[float, float, float]:
        """Transmitter position (x, y, z) in meters."""
        return self._tx_position

    @property
    def rx_position(self) -> Tuple[float, float, float]:
        """Receiver position (x, y, z) in meters."""
        return self._rx_position

    @property
    def num_subcarriers(self) -> int:
        """Number of frequency subcarriers."""
        return self._num_subcarriers

    @property
    def tx_rx_distance(self) -> float:
        """Distance between transmitter and receiver in meters."""
        return self._tx_rx_distance

    def _compute_distance(
        self,
        point_a: Tuple[float, float, float],
        point_b: Tuple[float, float, float],
    ) -> float:
        """Compute Euclidean distance between two 3D points."""
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        dz = point_b[2] - point_a[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _compute_subcarrier_frequencies(self) -> np.ndarray:
        """
        Compute the frequency of each subcarrier.

        Subcarriers are evenly spaced across the bandwidth,
        centered on the center frequency.

        Returns
        -------
        np.ndarray
            Frequencies in Hz. Shape: (num_subcarriers,).
        """
        half_bandwidth = self._bandwidth_hz / 2.0
        start_freq = self._center_frequency_hz - half_bandwidth
        end_freq = self._center_frequency_hz + half_bandwidth

        return np.linspace(start_freq, end_freq, self._num_subcarriers)

    def _sample_path_points(self) -> np.ndarray:
        """
        Generate sample points along the TX-RX path.

        Points are evenly distributed along the line segment
        from transmitter to receiver.

        Returns
        -------
        np.ndarray
            Sample points. Shape: (num_sample_points, 3).
        """
        tx = np.array(self._tx_position)
        rx = np.array(self._rx_position)

        # Parametric points along the line: p(t) = tx + t * (rx - tx)
        # Sample t in (0, 1) to exclude endpoints (TX and RX themselves)
        t_values = np.linspace(0.1, 0.9, self._num_sample_points)

        points = np.zeros((self._num_sample_points, 3))
        for i, t in enumerate(t_values):
            points[i] = tx + t * (rx - tx)

        return points

    def _query_integrated_disturbance(self) -> float:
        """
        Query and integrate disturbance along the TX-RX path.

        Samples the world's disturbance field at multiple points
        along the signal path and computes a weighted sum.

        Returns
        -------
        float
            Integrated disturbance value (arbitrary units).
        """
        sample_points = self._sample_path_points()
        total_disturbance = 0.0

        for point in sample_points:
            x, y, z = point[0], point[1], point[2]
            disturbance = self._world.disturbance_at(x, y, z)
            total_disturbance += disturbance

        # Normalize by number of samples to get average disturbance
        average_disturbance = total_disturbance / self._num_sample_points

        return average_disturbance

    def _compute_path_loss(self, distance: float, wavelength: float) -> float:
        """
        Compute free-space path loss.

        Uses simplified Friis transmission equation for path loss.
        Returns amplitude attenuation factor (not dB).

        Parameters
        ----------
        distance : float
            Distance in meters.
        wavelength : float
            Signal wavelength in meters.

        Returns
        -------
        float
            Amplitude attenuation factor (0 to 1).
        """
        if distance < 1e-6:
            # Avoid division by zero for co-located TX/RX
            return 1.0

        # Friis path loss: (wavelength / (4 * pi * distance))^2
        # We return amplitude factor, so take square root
        amplitude_factor = wavelength / (4.0 * math.pi * distance)

        # Clamp to reasonable range
        return min(amplitude_factor, 1.0)

    def _disturbance_to_amplitude_factor(self, disturbance: float) -> float:
        """
        Map disturbance to an amplitude attenuation factor.

        Uses a smooth, bounded exponential decay function.
        Higher disturbance leads to more attenuation.

        Parameters
        ----------
        disturbance : float
            Integrated disturbance value.

        Returns
        -------
        float
            Amplitude factor in range (0, 1].
        """
        # Exponential decay: exp(-k * |disturbance|)
        # k controls sensitivity to disturbance
        sensitivity = 0.5
        attenuation = math.exp(-sensitivity * abs(disturbance))
        return attenuation

    def _disturbance_to_phase_shift(
        self,
        disturbance: float,
        wavelength: float,
    ) -> float:
        """
        Map disturbance to a phase shift.

        Uses a smooth, bounded function that maps disturbance
        to additional path length equivalent.

        Parameters
        ----------
        disturbance : float
            Integrated disturbance value.
        wavelength : float
            Signal wavelength in meters.

        Returns
        -------
        float
            Phase shift in radians.
        """
        # Model disturbance as equivalent path length change
        # Use tanh to bound the effect smoothly
        max_path_change = 0.1  # meters
        equivalent_path_change = max_path_change * math.tanh(disturbance)

        # Convert path length to phase: phase = 2 * pi * distance / wavelength
        phase_shift = 2.0 * math.pi * equivalent_path_change / wavelength

        return phase_shift

    def _compute_baseline_phase(self, distance: float, wavelength: float) -> float:
        """
        Compute baseline phase from TX-RX distance.

        Parameters
        ----------
        distance : float
            Distance in meters.
        wavelength : float
            Signal wavelength in meters.

        Returns
        -------
        float
            Phase in radians.
        """
        # Phase accumulation over distance
        return 2.0 * math.pi * distance / wavelength

    def generate(self, timestamp: Optional[float] = None) -> SyntheticCSIFrame:
        """
        Generate a synthetic CSI frame at the current or specified time.

        This method:
        1. Queries the world's disturbance field along the TX-RX path
        2. Computes baseline amplitude from path loss
        3. Applies disturbance-induced amplitude attenuation
        4. Computes phase from distance and disturbance
        5. Adds controlled Gaussian noise

        Parameters
        ----------
        timestamp : Optional[float], optional
            Timestamp for the frame. If None, uses the world's current time.

        Returns
        -------
        SyntheticCSIFrame
            The generated CSI frame.
        """
        # Use world time if no timestamp specified
        if timestamp is None:
            timestamp = self._world.time

        # Query integrated disturbance along the signal path
        integrated_disturbance = self._query_integrated_disturbance()

        # Compute disturbance-induced factors
        disturbance_amplitude_factor = self._disturbance_to_amplitude_factor(
            integrated_disturbance
        )

        # Compute per-subcarrier values
        amplitudes = np.zeros(self._num_subcarriers)
        phases = np.zeros(self._num_subcarriers)

        for i in range(self._num_subcarriers):
            wavelength = self._wavelengths[i]

            # Baseline path loss
            path_loss_amplitude = self._compute_path_loss(
                self._tx_rx_distance, wavelength
            )

            # Combined amplitude: path loss * disturbance attenuation
            amplitude = path_loss_amplitude * disturbance_amplitude_factor

            # Baseline phase from distance
            baseline_phase = self._compute_baseline_phase(
                self._tx_rx_distance, wavelength
            )

            # Disturbance-induced phase shift
            disturbance_phase = self._disturbance_to_phase_shift(
                integrated_disturbance, wavelength
            )

            # Combined phase
            phase = baseline_phase + disturbance_phase

            amplitudes[i] = amplitude
            phases[i] = phase

        # Add measurement noise
        amplitude_noise = self._rng.normal(
            loc=0.0,
            scale=self._amplitude_noise_std,
            size=self._num_subcarriers,
        )
        phase_noise = self._rng.normal(
            loc=0.0,
            scale=self._phase_noise_std,
            size=self._num_subcarriers,
        )

        # Apply noise (ensure amplitudes stay non-negative)
        amplitudes = np.maximum(amplitudes + amplitude_noise, 0.0)
        phases = phases + phase_noise

        return SyntheticCSIFrame(
            timestamp=timestamp,
            amplitudes=amplitudes,
            phases=phases,
        )

    def generate_sequence(
        self,
        num_frames: int,
        time_interval: float,
        start_timestamp: Optional[float] = None,
    ) -> List[SyntheticCSIFrame]:
        """
        Generate a sequence of CSI frames at regular time intervals.

        Note: This does NOT advance the world's time. It generates frames
        at specified timestamps assuming the world state is static or
        has already been advanced externally.

        Parameters
        ----------
        num_frames : int
            Number of frames to generate.
        time_interval : float
            Time between frames in seconds.
        start_timestamp : Optional[float], optional
            Timestamp for the first frame. If None, uses world's current time.

        Returns
        -------
        List[SyntheticCSIFrame]
            List of generated CSI frames.
        """
        if start_timestamp is None:
            start_timestamp = self._world.time

        frames = []
        for i in range(num_frames):
            timestamp = start_timestamp + i * time_interval
            frame = self.generate(timestamp=timestamp)
            frames.append(frame)

        return frames

    def set_noise_parameters(
        self,
        amplitude_noise_std: Optional[float] = None,
        phase_noise_std: Optional[float] = None,
    ) -> None:
        """
        Update noise parameters.

        Parameters
        ----------
        amplitude_noise_std : Optional[float], optional
            New amplitude noise standard deviation. If None, unchanged.
        phase_noise_std : Optional[float], optional
            New phase noise standard deviation. If None, unchanged.
        """
        if amplitude_noise_std is not None:
            self._amplitude_noise_std = amplitude_noise_std
        if phase_noise_std is not None:
            self._phase_noise_std = phase_noise_std

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator with a new seed.

        Parameters
        ----------
        seed : Optional[int], optional
            New seed value. If None, uses system entropy.
        """
        self._rng = np.random.default_rng(seed)
