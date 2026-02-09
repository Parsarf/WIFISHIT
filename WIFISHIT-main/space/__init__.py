"""
space package

Spatial discretization and projection utilities.

Includes voxel grid representation and 2D projections.
"""

from space.voxel_grid import VoxelGrid
from space.projections import (
    floor_projection,
    vertical_projection_xz,
    vertical_projection_yz,
    horizontal_slice,
    vertical_slice_x,
    vertical_slice_y,
    all_projections,
)

# Re-export contract types for convenience
try:
    from contracts import Field2D, Field3D, FieldMetadata
    _HAS_CONTRACTS = True
except ImportError:
    _HAS_CONTRACTS = False

__all__ = [
    "VoxelGrid",
    "floor_projection",
    "vertical_projection_xz",
    "vertical_projection_yz",
    "horizontal_slice",
    "vertical_slice_x",
    "vertical_slice_y",
    "all_projections",
]

if _HAS_CONTRACTS:
    __all__.extend([
        "Field2D",
        "Field3D",
        "FieldMetadata",
    ])
