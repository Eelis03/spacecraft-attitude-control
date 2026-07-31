"""Array aliases and small conversion helpers shared by every layer."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BoolArray", "ComplexArray", "FloatArray", "as_matrix", "as_vector", "unit"]

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type ComplexArray = NDArray[np.complex128]


def as_vector(values: ArrayLike, size: int) -> FloatArray:
    """Return ``values`` as a contiguous float vector of exactly ``size`` entries."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected {size} entries, got {array.size}")
    return array


def as_matrix(values: ArrayLike, shape: tuple[int, int]) -> FloatArray:
    """Return ``values`` as a float matrix of exactly ``shape``."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return array


def unit(vector: ArrayLike) -> FloatArray:
    """Return ``vector`` scaled to unit length.

    Raises ``ValueError`` for a vector whose norm is zero, because the direction
    is then undefined and silently returning zeros would hide the caller's error.
    """
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return array / norm
