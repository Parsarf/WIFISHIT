"""
world.py

Continuous-space environment representing the authoritative ground-truth world model.

This module aggregates disturbance contributions from multiple objects without
discretizing space, performing sensing, or inferring meaning.

All spatial quantities are in meters.
All time quantities are in seconds.
"""

from typing import List, Tuple, Protocol, runtime_checkable


@runtime_checkable
class DisturbanceObject(Protocol):
    """
    Protocol defining the interface that disturbance objects must implement.

    Objects are expected to:
    - Advance their internal state when stepped forward in time
    - Compute their disturbance contribution at arbitrary spatial points
    """

    def step(self, dt: float) -> None:
        """
        Advance the object's internal state by dt seconds.

        Parameters
        ----------
        dt : float
            Time increment in seconds. Must be non-negative.
        """
        ...

    def disturbance_at(self, x: float, y: float, z: float) -> float:
        """
        Compute the disturbance contribution at a continuous world-space point.

        Parameters
        ----------
        x : float
            X coordinate in meters.
        y : float
            Y coordinate in meters.
        z : float
            Z coordinate in meters.

        Returns
        -------
        float
            Disturbance contribution at the specified point.
            Units and interpretation are object-specific.
        """
        ...


class World:
    """
    Continuous-space environment containing multiple disturbance objects.

    The World maintains a collection of disturbance objects and tracks
    simulation time. It provides methods to:
    - Add and remove objects
    - Advance simulation time
    - Query total disturbance at arbitrary spatial points

    The total disturbance at any point is the sum of individual object
    contributions. No normalization, clipping, or thresholding is applied.

    Attributes
    ----------
    time : float
        Current simulation time in seconds. Advances monotonically.

    Invariants
    ----------
    - All spatial quantities are in meters
    - All time quantities are in seconds
    - Space is not discretized
    - Object order does not affect query results
    - Behavior is deterministic given initial conditions
    """

    def __init__(self, initial_time: float = 0.0) -> None:
        """
        Initialize the world.

        Parameters
        ----------
        initial_time : float, optional
            Starting simulation time in seconds. Defaults to 0.0.
        """
        self._time: float = initial_time
        self._objects: List[DisturbanceObject] = []

    @property
    def time(self) -> float:
        """
        Current simulation time in seconds.

        Returns
        -------
        float
            The current simulation time.
        """
        return self._time

    @property
    def object_count(self) -> int:
        """
        Number of disturbance objects currently in the world.

        Returns
        -------
        int
            Count of objects.
        """
        return len(self._objects)

    def add_object(self, obj: DisturbanceObject) -> None:
        """
        Add a disturbance object to the world.

        The object will contribute to disturbance queries and will be
        stepped forward when the world advances in time.

        Parameters
        ----------
        obj : DisturbanceObject
            The disturbance object to add. Must implement the
            DisturbanceObject protocol (step and disturbance_at methods).

        Raises
        ------
        TypeError
            If the object does not implement the required protocol.
        """
        if not isinstance(obj, DisturbanceObject):
            raise TypeError(
                f"Object must implement DisturbanceObject protocol "
                f"(step and disturbance_at methods). Got {type(obj).__name__}."
            )
        self._objects.append(obj)

    def remove_object(self, obj: DisturbanceObject) -> bool:
        """
        Remove a disturbance object from the world.

        Parameters
        ----------
        obj : DisturbanceObject
            The disturbance object to remove.

        Returns
        -------
        bool
            True if the object was found and removed, False otherwise.
        """
        try:
            self._objects.remove(obj)
            return True
        except ValueError:
            return False

    def clear_objects(self) -> None:
        """
        Remove all disturbance objects from the world.
        """
        self._objects.clear()

    def step(self, dt: float) -> None:
        """
        Advance the world forward in time by dt seconds.

        This advances the world's simulation time and steps each
        disturbance object forward independently using its own motion model.

        Parameters
        ----------
        dt : float
            Time increment in seconds. Must be non-negative.

        Raises
        ------
        ValueError
            If dt is negative (time must advance monotonically).
        """
        if dt < 0.0:
            raise ValueError(
                f"Time step dt must be non-negative. Got {dt}."
            )

        # Step each object forward independently
        for obj in self._objects:
            obj.step(dt)

        # Advance world time
        self._time += dt

    def disturbance_at(self, x: float, y: float, z: float) -> float:
        """
        Query the total disturbance at an arbitrary continuous world-space point.

        The total disturbance is computed as the sum of individual
        contributions from all objects. No normalization, clipping,
        or thresholding is applied.

        Parameters
        ----------
        x : float
            X coordinate in meters.
        y : float
            Y coordinate in meters.
        z : float
            Z coordinate in meters.

        Returns
        -------
        float
            Total disturbance at the specified point.
            Returns 0.0 if no objects are present.
        """
        total_disturbance = 0.0

        for obj in self._objects:
            contribution = obj.disturbance_at(x, y, z)
            total_disturbance += contribution

        return total_disturbance

    def disturbance_at_point(self, point: Tuple[float, float, float]) -> float:
        """
        Query the total disturbance at a point specified as a tuple.

        Convenience method that unpacks a 3-tuple into x, y, z coordinates.

        Parameters
        ----------
        point : Tuple[float, float, float]
            The (x, y, z) coordinates in meters.

        Returns
        -------
        float
            Total disturbance at the specified point.
        """
        x, y, z = point
        return self.disturbance_at(x, y, z)

    def get_objects(self) -> List[DisturbanceObject]:
        """
        Get a copy of the list of disturbance objects.

        Returns a copy to prevent external modification of the internal list.

        Returns
        -------
        List[DisturbanceObject]
            A copy of the list of disturbance objects.
        """
        return list(self._objects)
