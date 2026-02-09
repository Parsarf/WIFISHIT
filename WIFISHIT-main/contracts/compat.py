"""
contracts/compat.py

Compatibility layer for migrating from legacy types to contracts.

Provides adapters to convert between legacy types (e.g., csi.csi_frame.CSIFrame)
and the new contract types (contracts.CSIFrame).

Usage
-----
>>> from contracts.compat import csi_frame_to_contract, contract_to_csi_frame
>>> from csi.csi_frame import CSIFrame as LegacyCSIFrame
>>> from contracts import CSIFrame

# Convert legacy to contract
>>> legacy = LegacyCSIFrame(timestamp=0.0, amplitude=amp, phase=phase)
>>> contract = csi_frame_to_contract(legacy, link_id="default")

# Convert contract to legacy
>>> legacy = contract_to_csi_frame(contract)
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from csi.csi_frame import CSIFrame as LegacyCSIFrame
    from contracts.csi import CSIFrame as ContractCSIFrame


def csi_frame_to_contract(
    legacy_frame: "LegacyCSIFrame",
    link_id: str = "default",
) -> "ContractCSIFrame":
    """
    Convert a legacy CSIFrame to a contract CSIFrame.

    Parameters
    ----------
    legacy_frame : csi.csi_frame.CSIFrame
        The legacy CSI frame.
    link_id : str
        Link identifier to assign (legacy frames don't have this).

    Returns
    -------
    contracts.CSIFrame
        The new contract-based frame.
    """
    from contracts.csi import CSIFrame

    return CSIFrame(
        link_id=link_id,
        timestamp=legacy_frame.timestamp,
        amplitude=np.array(legacy_frame.amplitude),
        phase=np.array(legacy_frame.phase),
        meta={},
    )


def contract_to_csi_frame(
    contract_frame: "ContractCSIFrame",
) -> "LegacyCSIFrame":
    """
    Convert a contract CSIFrame to a legacy CSIFrame.

    Parameters
    ----------
    contract_frame : contracts.CSIFrame
        The contract-based frame.

    Returns
    -------
    csi.csi_frame.CSIFrame
        The legacy frame (link_id and meta are discarded).
    """
    from csi.csi_frame import CSIFrame as LegacyCSIFrame

    return LegacyCSIFrame(
        timestamp=contract_frame.timestamp,
        amplitude=np.array(contract_frame.amplitude),
        phase=np.array(contract_frame.phase),
    )


def synthetic_frame_to_contract(
    synthetic_frame,
    link_id: str = "synthetic",
) -> "ContractCSIFrame":
    """
    Convert a SyntheticCSIFrame to a contract CSIFrame.

    Parameters
    ----------
    synthetic_frame : csi.synthetic_csi.SyntheticCSIFrame
        The synthetic frame.
    link_id : str
        Link identifier to assign.

    Returns
    -------
    contracts.CSIFrame
        The new contract-based frame.
    """
    from contracts.csi import CSIFrame

    return CSIFrame(
        link_id=link_id,
        timestamp=synthetic_frame.timestamp,
        amplitude=np.array(synthetic_frame.amplitudes),
        phase=np.array(synthetic_frame.phases),
        meta={"source": "synthetic"},
    )
