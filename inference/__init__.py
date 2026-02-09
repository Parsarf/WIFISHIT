"""
inference package

Inference pipeline components for CSI-based sensing.

Includes preprocessing, feature extraction, representation learning,
inference engine, and clustering.
"""

from inference.preprocessing import (
    unwrap_phase,
    unwrap_phase_temporal,
    normalize_amplitude,
    normalize_amplitude_minmax,
    normalize_amplitude_zscore,
    frames_to_arrays,
    sliding_window,
)
from inference.feature_extraction import (
    extract_temporal_features,
    extract_spectral_features,
    first_order_difference,
    second_order_difference,
    short_time_energy,
    windowed_energy,
    windowed_variance,
    windowed_mean,
    windowed_std,
    subcarrier_correlation_matrix,
    aggregate_statistics,
)
from inference.representation import LinearEncoder, PCAEncoder, MLPEncoder
from inference.inference_engine import InferenceEngine
from inference.clustering import (
    SpatialCluster,
    LatentCluster,
    cluster_spatial_activity,
    cluster_latent_vectors,
)

# Re-export contract types for convenience
try:
    from contracts import (
        Measurement,
        QualityMetrics,
        ConditionedCSIFrame,
    )
    _HAS_CONTRACTS = True
except ImportError:
    _HAS_CONTRACTS = False

__all__ = [
    # Preprocessing
    "unwrap_phase",
    "unwrap_phase_temporal",
    "normalize_amplitude",
    "normalize_amplitude_minmax",
    "normalize_amplitude_zscore",
    "frames_to_arrays",
    "sliding_window",
    # Feature extraction
    "extract_temporal_features",
    "extract_spectral_features",
    "first_order_difference",
    "second_order_difference",
    "short_time_energy",
    "windowed_energy",
    "windowed_variance",
    "windowed_mean",
    "windowed_std",
    "subcarrier_correlation_matrix",
    "aggregate_statistics",
    # Representation
    "LinearEncoder",
    "PCAEncoder",
    "MLPEncoder",
    # Inference
    "InferenceEngine",
    # Clustering
    "SpatialCluster",
    "LatentCluster",
    "cluster_spatial_activity",
    "cluster_latent_vectors",
]

if _HAS_CONTRACTS:
    __all__.extend([
        "Measurement",
        "QualityMetrics",
        "ConditionedCSIFrame",
    ])
