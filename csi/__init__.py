"""
csi package

Channel State Information data structures and generators.

Note: For new code, prefer using contracts.CSIFrame which includes
link_id for multi-link setups. The legacy CSIFrame is maintained
for backward compatibility.
"""

from csi.csi_frame import CSIFrame
from csi.synthetic_csi import SyntheticCSIGenerator, SyntheticCSIFrame

# Re-export contract types for convenience
try:
    from contracts import CSIFrame as ContractCSIFrame
    from contracts import ConditionedCSIFrame
    from contracts.compat import (
        csi_frame_to_contract,
        contract_to_csi_frame,
        synthetic_frame_to_contract,
    )
    _HAS_CONTRACTS = True
except ImportError:
    _HAS_CONTRACTS = False

__all__ = ['CSIFrame', 'SyntheticCSIGenerator', 'SyntheticCSIFrame']

if _HAS_CONTRACTS:
    __all__.extend([
        'ContractCSIFrame',
        'ConditionedCSIFrame',
        'csi_frame_to_contract',
        'contract_to_csi_frame',
        'synthetic_frame_to_contract',
    ])
