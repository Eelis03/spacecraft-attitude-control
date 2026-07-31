"""Measurement of conversion accuracy between attitude representations.

The residual reported for each conversion path is the principal angle between the
original attitude and the attitude recovered after a round trip. Measuring an
angle rather than a component-wise difference makes the result independent of the
representation and immune to the quaternion sign ambiguity, so a sign flip during
the round trip does not register as an error where none exists.
"""

from __future__ import annotations

import numpy as np

from attitude_control.model.attitude import (
    attitude_error,
    dcm_from_quaternion,
    euler321_from_quaternion,
    mrp_from_quaternion,
    mrp_short_rotation,
    principal_angle,
    quaternion_from_axis_angle,
    quaternion_from_dcm,
    quaternion_from_euler321,
    quaternion_from_mrp,
)

__all__ = ["CONVERSION_PATHS", "round_trip_residuals", "worst_orthonormality_defect"]

CONVERSION_PATHS: tuple[str, ...] = ("DCM", "MRP", "Euler 3-2-1")


def _round_trip(quaternion: np.ndarray) -> dict[str, np.ndarray]:
    parameters = mrp_short_rotation(mrp_from_quaternion(quaternion))
    return {
        "DCM": quaternion_from_dcm(dcm_from_quaternion(quaternion)),
        "MRP": quaternion_from_mrp(parameters),
        "Euler 3-2-1": quaternion_from_euler321(euler321_from_quaternion(quaternion)),
    }


def round_trip_residuals(samples: int, seed: int) -> dict[str, float]:
    """Return the worst round trip residual in radians for each conversion path."""
    generator = np.random.default_rng(seed)
    worst = dict.fromkeys(CONVERSION_PATHS, 0.0)
    for _ in range(samples):
        angle = float(generator.uniform(-np.pi, np.pi))
        quaternion = quaternion_from_axis_angle(generator.normal(size=3), angle)
        for label, recovered in _round_trip(quaternion).items():
            residual = principal_angle(attitude_error(recovered, quaternion))
            worst[label] = max(worst[label], residual)
    return worst


def worst_orthonormality_defect(samples: int, seed: int) -> float:
    """Return the largest departure of a generated attitude matrix from orthonormality."""
    generator = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        angle = float(generator.uniform(-np.pi, np.pi))
        matrix = dcm_from_quaternion(quaternion_from_axis_angle(generator.normal(size=3), angle))
        defect = float(np.max(np.abs(matrix @ matrix.T - np.eye(3))))
        worst = max(worst, defect)
    return worst
