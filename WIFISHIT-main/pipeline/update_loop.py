"""
update_loop.py

Time-stepped execution loop coordinating world evolution, CSI generation,
and downstream data flow.

This module does not perform physics, inference, or visualization.
It orchestrates the sequence of operations at each timestep.

All time units are seconds.
"""

from typing import Optional, List, Protocol, runtime_checkable

from world.world import World
from csi.synthetic_csi import SyntheticCSIGenerator
from csi.csi_frame import CSIFrame


@runtime_checkable
class VoxelGridProtocol(Protocol):
    """
    Protocol defining the interface for optional voxel grid updates.

    A voxel grid that integrates with the update loop must provide
    a method to update its state given a CSI frame.
    """

    def update_from_csi(self, csi_frame: CSIFrame) -> None:
        """
        Update the voxel grid state from a CSI measurement.

        Parameters
        ----------
        csi_frame : CSIFrame
            The CSI frame to incorporate into the grid state.
        """
        ...


class UpdateLoop:
    """
    Time-stepped execution loop for the simulation pipeline.

    Coordinates the sequence of operations at each timestep:
    1. Advance the world forward in time
    2. Generate a CSI measurement
    3. Optionally update the voxel grid

    Parameters
    ----------
    world : World
        The world model to evolve.
    csi_generator : SyntheticCSIGenerator
        The CSI generator producing measurements.
    dt : float
        Fixed timestep in seconds. Must be positive.
    voxel_grid : Optional[VoxelGridProtocol], optional
        Optional voxel grid to update each step. Defaults to None.

    Attributes
    ----------
    world : World
        Reference to the world model.
    csi_generator : SyntheticCSIGenerator
        Reference to the CSI generator.
    dt : float
        Fixed timestep in seconds.
    step_count : int
        Number of steps executed since initialization.
    """

    def __init__(
        self,
        world: World,
        csi_generator: SyntheticCSIGenerator,
        dt: float,
        voxel_grid: Optional[VoxelGridProtocol] = None,
    ) -> None:
        """Initialize the update loop."""
        if dt <= 0.0:
            raise ValueError(
                f"Timestep dt must be positive. Got {dt}."
            )

        self._world = world
        self._csi_generator = csi_generator
        self._dt = dt
        self._voxel_grid = voxel_grid
        self._step_count = 0

    @property
    def world(self) -> World:
        """Reference to the world model."""
        return self._world

    @property
    def csi_generator(self) -> SyntheticCSIGenerator:
        """Reference to the CSI generator."""
        return self._csi_generator

    @property
    def dt(self) -> float:
        """Fixed timestep in seconds."""
        return self._dt

    @property
    def voxel_grid(self) -> Optional[VoxelGridProtocol]:
        """Reference to the optional voxel grid."""
        return self._voxel_grid

    @property
    def step_count(self) -> int:
        """Number of steps executed since initialization."""
        return self._step_count

    @property
    def current_time(self) -> float:
        """Current simulation time in seconds (from world)."""
        return self._world.time

    def step(self) -> CSIFrame:
        """
        Execute one simulation step.

        This method performs the following operations in order:
        1. Advance the world forward by dt seconds
        2. Generate a CSI frame at the current world time
        3. If a voxel grid is attached, update it with the CSI frame
        4. Increment the step counter

        Returns
        -------
        CSIFrame
            The CSI frame generated at this timestep.
        """
        # Advance world state
        self._world.step(self._dt)

        # Generate CSI measurement at current world time
        synthetic_frame = self._csi_generator.generate()

        # Convert to CSIFrame
        csi_frame = CSIFrame(
            timestamp=synthetic_frame.timestamp,
            amplitude=synthetic_frame.amplitudes,
            phase=synthetic_frame.phases,
        )

        # Optionally update voxel grid
        if self._voxel_grid is not None:
            self._voxel_grid.update_from_csi(csi_frame)

        # Increment step counter
        self._step_count += 1

        return csi_frame

    def run(self, n_steps: int) -> List[CSIFrame]:
        """
        Execute multiple simulation steps.

        Repeatedly calls step() for the specified number of iterations
        and collects the generated CSI frames.

        Parameters
        ----------
        n_steps : int
            Number of steps to execute. Must be non-negative.

        Returns
        -------
        List[CSIFrame]
            List of CSI frames generated during the run.
            Length equals n_steps.

        Raises
        ------
        ValueError
            If n_steps is negative.
        """
        if n_steps < 0:
            raise ValueError(
                f"n_steps must be non-negative. Got {n_steps}."
            )

        frames: List[CSIFrame] = []

        for _ in range(n_steps):
            frame = self.step()
            frames.append(frame)

        return frames

    def reset_step_count(self) -> None:
        """Reset the step counter to zero."""
        self._step_count = 0

    def set_voxel_grid(self, voxel_grid: Optional[VoxelGridProtocol]) -> None:
        """
        Attach or detach a voxel grid.

        Parameters
        ----------
        voxel_grid : Optional[VoxelGridProtocol]
            The voxel grid to attach, or None to detach.
        """
        self._voxel_grid = voxel_grid
