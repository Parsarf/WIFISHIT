"""
representation.py

Latent representation layer mapping extracted features into a lower-dimensional
latent space where similar physical situations produce similar representations.

Does not perform inference, classification, or decision-making.

All mappings are deterministic given model parameters.
No semantic labels are used.
"""

from typing import Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np


class RepresentationEncoder(ABC):
    """
    Abstract base class for representation encoders.

    An encoder maps feature vectors to fixed-dimensional latent vectors.

    Attributes
    ----------
    input_dim : int
        Expected input feature dimension.
    latent_dim : int
        Output latent vector dimension.
    """

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Expected input feature dimension."""
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Output latent vector dimension."""
        ...

    @abstractmethod
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode feature vectors to latent representations.

        Parameters
        ----------
        features : np.ndarray
            Input features. Shape: (n_samples, input_dim) or (input_dim,).

        Returns
        -------
        np.ndarray
            Latent vectors. Shape: (n_samples, latent_dim) or (latent_dim,).
        """
        ...


class LinearEncoder(RepresentationEncoder):
    """
    Linear projection encoder.

    Maps features to latent space via a learned or random linear projection:
        z = (x - bias) @ W

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    latent_dim : int
        Output latent dimension.
    weights : Optional[np.ndarray], optional
        Projection matrix. Shape: (input_dim, latent_dim).
        If None, initialized with orthogonal random projection.
    bias : Optional[np.ndarray], optional
        Bias to subtract from input. Shape: (input_dim,).
        If None, no bias subtraction.
    random_seed : Optional[int], optional
        Seed for reproducible weight initialization.

    Attributes
    ----------
    weights : np.ndarray
        Projection matrix. Shape: (input_dim, latent_dim).
    bias : Optional[np.ndarray]
        Input bias. Shape: (input_dim,) or None.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize linear encoder."""
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive. Got {input_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive. Got {latent_dim}.")

        self._input_dim = input_dim
        self._latent_dim = latent_dim

        if weights is not None:
            if weights.shape != (input_dim, latent_dim):
                raise ValueError(
                    f"weights shape must be ({input_dim}, {latent_dim}). "
                    f"Got {weights.shape}."
                )
            self._weights = weights.copy()
        else:
            self._weights = self._init_orthogonal_weights(
                input_dim, latent_dim, random_seed
            )

        if bias is not None:
            if bias.shape != (input_dim,):
                raise ValueError(
                    f"bias shape must be ({input_dim},). Got {bias.shape}."
                )
            self._bias = bias.copy()
        else:
            self._bias = None

    @staticmethod
    def _init_orthogonal_weights(
        input_dim: int,
        latent_dim: int,
        seed: Optional[int],
    ) -> np.ndarray:
        """
        Initialize weights using orthogonal random projection.

        Uses QR decomposition to obtain orthonormal columns.
        """
        rng = np.random.default_rng(seed)
        random_matrix = rng.standard_normal((input_dim, latent_dim))

        # QR decomposition for orthonormal columns
        q, _ = np.linalg.qr(random_matrix)

        # Handle case where latent_dim > input_dim
        if latent_dim <= input_dim:
            return q[:, :latent_dim]
        else:
            # Pad with additional random orthogonal vectors
            return q

    @property
    def input_dim(self) -> int:
        """Expected input feature dimension."""
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        """Output latent vector dimension."""
        return self._latent_dim

    @property
    def weights(self) -> np.ndarray:
        """Projection matrix. Shape: (input_dim, latent_dim)."""
        return self._weights

    @property
    def bias(self) -> Optional[np.ndarray]:
        """Input bias. Shape: (input_dim,) or None."""
        return self._bias

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode feature vectors via linear projection.

        Parameters
        ----------
        features : np.ndarray
            Input features. Shape: (n_samples, input_dim) or (input_dim,).

        Returns
        -------
        np.ndarray
            Latent vectors. Shape: (n_samples, latent_dim) or (latent_dim,).
        """
        single_sample = features.ndim == 1

        if single_sample:
            features = features.reshape(1, -1)

        if features.shape[1] != self._input_dim:
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match "
                f"expected input_dim {self._input_dim}."
            )

        # Subtract bias if present
        if self._bias is not None:
            centered = features - self._bias
        else:
            centered = features

        # Linear projection
        latent = centered @ self._weights

        if single_sample:
            return latent.squeeze(0)

        return latent

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set projection weights.

        Parameters
        ----------
        weights : np.ndarray
            New projection matrix. Shape: (input_dim, latent_dim).
        """
        if weights.shape != (self._input_dim, self._latent_dim):
            raise ValueError(
                f"weights shape must be ({self._input_dim}, {self._latent_dim}). "
                f"Got {weights.shape}."
            )
        self._weights = weights.copy()

    def set_bias(self, bias: Optional[np.ndarray]) -> None:
        """
        Set input bias.

        Parameters
        ----------
        bias : Optional[np.ndarray]
            New bias. Shape: (input_dim,) or None.
        """
        if bias is not None:
            if bias.shape != (self._input_dim,):
                raise ValueError(
                    f"bias shape must be ({self._input_dim},). Got {bias.shape}."
                )
            self._bias = bias.copy()
        else:
            self._bias = None


class PCAEncoder(RepresentationEncoder):
    """
    Principal Component Analysis encoder.

    Projects features onto principal components learned from data.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    latent_dim : int
        Number of principal components (output dimension).

    Attributes
    ----------
    is_fitted : bool
        Whether the encoder has been fitted to data.
    components : Optional[np.ndarray]
        Principal components. Shape: (latent_dim, input_dim).
    mean : Optional[np.ndarray]
        Feature mean from training. Shape: (input_dim,).
    explained_variance : Optional[np.ndarray]
        Variance explained by each component. Shape: (latent_dim,).
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        """Initialize PCA encoder."""
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive. Got {input_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive. Got {latent_dim}.")
        if latent_dim > input_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) cannot exceed input_dim ({input_dim})."
            )

        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._components: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None

    @property
    def input_dim(self) -> int:
        """Expected input feature dimension."""
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        """Output latent vector dimension."""
        return self._latent_dim

    @property
    def is_fitted(self) -> bool:
        """Whether the encoder has been fitted to data."""
        return self._components is not None

    @property
    def components(self) -> Optional[np.ndarray]:
        """Principal components. Shape: (latent_dim, input_dim)."""
        return self._components

    @property
    def mean(self) -> Optional[np.ndarray]:
        """Feature mean from training. Shape: (input_dim,)."""
        return self._mean

    @property
    def explained_variance(self) -> Optional[np.ndarray]:
        """Variance explained by each component. Shape: (latent_dim,)."""
        return self._explained_variance

    def fit(self, features: np.ndarray) -> None:
        """
        Fit PCA to training data.

        Parameters
        ----------
        features : np.ndarray
            Training features. Shape: (n_samples, input_dim).
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D (n_samples, input_dim). "
                f"Got {features.ndim} dimensions."
            )
        if features.shape[1] != self._input_dim:
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match "
                f"expected input_dim {self._input_dim}."
            )
        if features.shape[0] < self._latent_dim:
            raise ValueError(
                f"Need at least {self._latent_dim} samples to fit "
                f"{self._latent_dim} components. Got {features.shape[0]}."
            )

        # Center data
        self._mean = np.mean(features, axis=0)
        centered = features - self._mean

        # SVD for PCA
        _, s, vh = np.linalg.svd(centered, full_matrices=False)

        # Store top components
        self._components = vh[:self._latent_dim]

        # Compute explained variance
        n_samples = features.shape[0]
        self._explained_variance = (s[:self._latent_dim] ** 2) / (n_samples - 1)

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode feature vectors via PCA projection.

        Parameters
        ----------
        features : np.ndarray
            Input features. Shape: (n_samples, input_dim) or (input_dim,).

        Returns
        -------
        np.ndarray
            Latent vectors. Shape: (n_samples, latent_dim) or (latent_dim,).

        Raises
        ------
        RuntimeError
            If encoder has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("PCAEncoder must be fitted before encoding.")

        single_sample = features.ndim == 1

        if single_sample:
            features = features.reshape(1, -1)

        if features.shape[1] != self._input_dim:
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match "
                f"expected input_dim {self._input_dim}."
            )

        # Center and project
        centered = features - self._mean
        latent = centered @ self._components.T

        if single_sample:
            return latent.squeeze(0)

        return latent

    def set_parameters(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set PCA parameters directly.

        Parameters
        ----------
        components : np.ndarray
            Principal components. Shape: (latent_dim, input_dim).
        mean : np.ndarray
            Feature mean. Shape: (input_dim,).
        explained_variance : Optional[np.ndarray], optional
            Variance per component. Shape: (latent_dim,).
        """
        if components.shape != (self._latent_dim, self._input_dim):
            raise ValueError(
                f"components shape must be ({self._latent_dim}, {self._input_dim}). "
                f"Got {components.shape}."
            )
        if mean.shape != (self._input_dim,):
            raise ValueError(
                f"mean shape must be ({self._input_dim},). Got {mean.shape}."
            )

        self._components = components.copy()
        self._mean = mean.copy()

        if explained_variance is not None:
            if explained_variance.shape != (self._latent_dim,):
                raise ValueError(
                    f"explained_variance shape must be ({self._latent_dim},). "
                    f"Got {explained_variance.shape}."
                )
            self._explained_variance = explained_variance.copy()


class MLPEncoder(RepresentationEncoder):
    """
    Multi-layer perceptron encoder using numpy.

    A simple feedforward network with one hidden layer and tanh activation.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    latent_dim : int
        Output latent dimension.
    random_seed : Optional[int], optional
        Seed for reproducible weight initialization.

    Attributes
    ----------
    hidden_dim : int
        Hidden layer dimension.
    weights1 : np.ndarray
        Input to hidden weights. Shape: (input_dim, hidden_dim).
    bias1 : np.ndarray
        Hidden layer bias. Shape: (hidden_dim,).
    weights2 : np.ndarray
        Hidden to output weights. Shape: (hidden_dim, latent_dim).
    bias2 : np.ndarray
        Output layer bias. Shape: (latent_dim,).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize MLP encoder with Xavier initialization."""
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive. Got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive. Got {hidden_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive. Got {latent_dim}.")

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

        rng = np.random.default_rng(random_seed)

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + latent_dim))

        self._weights1 = rng.standard_normal((input_dim, hidden_dim)) * scale1
        self._bias1 = np.zeros(hidden_dim)
        self._weights2 = rng.standard_normal((hidden_dim, latent_dim)) * scale2
        self._bias2 = np.zeros(latent_dim)

    @property
    def input_dim(self) -> int:
        """Expected input feature dimension."""
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        """Output latent vector dimension."""
        return self._latent_dim

    @property
    def hidden_dim(self) -> int:
        """Hidden layer dimension."""
        return self._hidden_dim

    @property
    def weights1(self) -> np.ndarray:
        """Input to hidden weights. Shape: (input_dim, hidden_dim)."""
        return self._weights1

    @property
    def bias1(self) -> np.ndarray:
        """Hidden layer bias. Shape: (hidden_dim,)."""
        return self._bias1

    @property
    def weights2(self) -> np.ndarray:
        """Hidden to output weights. Shape: (hidden_dim, latent_dim)."""
        return self._weights2

    @property
    def bias2(self) -> np.ndarray:
        """Output layer bias. Shape: (latent_dim,)."""
        return self._bias2

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode feature vectors through MLP.

        Parameters
        ----------
        features : np.ndarray
            Input features. Shape: (n_samples, input_dim) or (input_dim,).

        Returns
        -------
        np.ndarray
            Latent vectors. Shape: (n_samples, latent_dim) or (latent_dim,).
        """
        single_sample = features.ndim == 1

        if single_sample:
            features = features.reshape(1, -1)

        if features.shape[1] != self._input_dim:
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match "
                f"expected input_dim {self._input_dim}."
            )

        # Hidden layer with tanh activation
        hidden = np.tanh(features @ self._weights1 + self._bias1)

        # Output layer (linear)
        latent = hidden @ self._weights2 + self._bias2

        if single_sample:
            return latent.squeeze(0)

        return latent

    def set_parameters(
        self,
        weights1: np.ndarray,
        bias1: np.ndarray,
        weights2: np.ndarray,
        bias2: np.ndarray,
    ) -> None:
        """
        Set network parameters.

        Parameters
        ----------
        weights1 : np.ndarray
            Input to hidden weights. Shape: (input_dim, hidden_dim).
        bias1 : np.ndarray
            Hidden bias. Shape: (hidden_dim,).
        weights2 : np.ndarray
            Hidden to output weights. Shape: (hidden_dim, latent_dim).
        bias2 : np.ndarray
            Output bias. Shape: (latent_dim,).
        """
        expected_w1 = (self._input_dim, self._hidden_dim)
        expected_b1 = (self._hidden_dim,)
        expected_w2 = (self._hidden_dim, self._latent_dim)
        expected_b2 = (self._latent_dim,)

        if weights1.shape != expected_w1:
            raise ValueError(f"weights1 shape must be {expected_w1}. Got {weights1.shape}.")
        if bias1.shape != expected_b1:
            raise ValueError(f"bias1 shape must be {expected_b1}. Got {bias1.shape}.")
        if weights2.shape != expected_w2:
            raise ValueError(f"weights2 shape must be {expected_w2}. Got {weights2.shape}.")
        if bias2.shape != expected_b2:
            raise ValueError(f"bias2 shape must be {expected_b2}. Got {bias2.shape}.")

        self._weights1 = weights1.copy()
        self._bias1 = bias1.copy()
        self._weights2 = weights2.copy()
        self._bias2 = bias2.copy()

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get network parameters.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (weights1, bias1, weights2, bias2)
        """
        return (
            self._weights1.copy(),
            self._bias1.copy(),
            self._weights2.copy(),
            self._bias2.copy(),
        )
