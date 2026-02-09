"""
objects.py

Disturbance object definitions for the world model.

Each object implements the DisturbanceObject protocol:
- step(dt: float) -> None: Advance internal state
- disturbance_at(x: float, y: float, z: float) -> float: Query contribution

All spatial quantities are in meters.
All time quantities are in seconds.
"""

from typing import Tuple
import math


class StaticDisturbance:
    """
    A static point disturbance with Gaussian falloff.

    Does not move. Useful for testing and representing fixed obstacles.

    Parameters
    ----------
    x, y, z : float
        Position in meters.
    radius : float
        Characteristic radius in meters.
    intensity : float
        Peak intensity at center.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 1.0,
        intensity: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.intensity = intensity

    def step(self, dt: float) -> None:
        """Static objects do not change over time."""
        pass

    def disturbance_at(self, x: float, y: float, z: float) -> float:
        """Compute Gaussian disturbance at query point."""
        dx = x - self.x
        dy = y - self.y
        dz = z - self.z
        dist_sq = dx*dx + dy*dy + dz*dz

        # Early exit for distant points
        if dist_sq > (self.radius * 3) ** 2:
            return 0.0

        # Gaussian falloff
        return self.intensity * math.exp(-dist_sq / (2 * self.radius**2))


class MovingDisturbance:
    """
    A moving point disturbance with Gaussian falloff.

    Moves with constant velocity, bouncing off defined boundaries.

    Parameters
    ----------
    x, y, z : float
        Initial position in meters.
    vx, vy : float
        Velocity components in m/s.
    radius : float
        Characteristic radius in meters.
    intensity : float
        Peak intensity at center.
    bounds : Tuple[float, float, float, float]
        Movement boundaries (x_min, x_max, y_min, y_max).
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        vx: float = 0.0,
        vy: float = 0.0,
        radius: float = 1.0,
        intensity: float = 1.0,
        bounds: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.intensity = intensity
        self.bounds = bounds

    def step(self, dt: float) -> None:
        """
        Advance position by dt seconds.

        Bounces off boundaries defined by bounds.
        """
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off boundaries
        x_min, x_max, y_min, y_max = self.bounds
        if self.x < x_min or self.x > x_max:
            self.vx *= -1
            self.x = max(x_min, min(x_max, self.x))
        if self.y < y_min or self.y > y_max:
            self.vy *= -1
            self.y = max(y_min, min(y_max, self.y))

    def disturbance_at(self, x: float, y: float, z: float) -> float:
        """Compute Gaussian disturbance at query point."""
        dx = x - self.x
        dy = y - self.y
        dz = z - self.z
        dist_sq = dx*dx + dy*dy + dz*dz

        # Early exit for distant points
        if dist_sq > (self.radius * 3) ** 2:
            return 0.0

        # Gaussian falloff
        return self.intensity * math.exp(-dist_sq / (2 * self.radius**2))


class OscillatingDisturbance:
    """
    A disturbance that oscillates in intensity over time.

    Useful for simulating periodic activity patterns.

    Parameters
    ----------
    x, y, z : float
        Position in meters.
    radius : float
        Characteristic radius in meters.
    base_intensity : float
        Base intensity level.
    amplitude : float
        Oscillation amplitude.
    frequency : float
        Oscillation frequency in Hz.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        radius: float = 1.0,
        base_intensity: float = 0.5,
        amplitude: float = 0.5,
        frequency: float = 0.5,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.base_intensity = base_intensity
        self.amplitude = amplitude
        self.frequency = frequency
        self._time = 0.0

    def step(self, dt: float) -> None:
        """Advance internal time."""
        self._time += dt

    @property
    def current_intensity(self) -> float:
        """Compute current intensity based on time."""
        oscillation = math.sin(2 * math.pi * self.frequency * self._time)
        return self.base_intensity + self.amplitude * oscillation

    def disturbance_at(self, x: float, y: float, z: float) -> float:
        """Compute Gaussian disturbance at query point."""
        dx = x - self.x
        dy = y - self.y
        dz = z - self.z
        dist_sq = dx*dx + dy*dy + dz*dz

        # Early exit for distant points
        if dist_sq > (self.radius * 3) ** 2:
            return 0.0

        # Gaussian falloff with time-varying intensity
        intensity = max(0.0, self.current_intensity)
        return intensity * math.exp(-dist_sq / (2 * self.radius**2))
