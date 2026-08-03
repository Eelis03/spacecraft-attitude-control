"""Algorithm layer: control laws, wheel allocation, and momentum management.

Nothing here integrates, records, or plots. Every entry point maps a state to a
command, so a controller can be exercised at a single point without running a
simulation.
"""

from __future__ import annotations

from attitude_control.algorithm.allocation import (
    Allocation,
    allocate,
    apply_limits,
    delivered_body_torque,
    minimum_norm_wheel_torque,
    null_space_wheel_torque,
)
from attitude_control.algorithm.controller import (
    AttitudeController,
    ConstantTorque,
    ControlSignal,
    LinearQuadraticRegulator,
    QuaternionFeedbackPD,
    QuaternionFeedbackPID,
    error_state,
    routh_integral_limit,
)
from attitude_control.algorithm.momentum import (
    MagneticDumping,
    controllable_momentum,
    cross_product_dipole,
    uncontrollable_momentum,
)

__all__ = [
    "Allocation",
    "AttitudeController",
    "ConstantTorque",
    "ControlSignal",
    "LinearQuadraticRegulator",
    "MagneticDumping",
    "QuaternionFeedbackPD",
    "QuaternionFeedbackPID",
    "allocate",
    "apply_limits",
    "controllable_momentum",
    "cross_product_dipole",
    "delivered_body_torque",
    "error_state",
    "minimum_norm_wheel_torque",
    "null_space_wheel_torque",
    "routh_integral_limit",
    "uncontrollable_momentum",
]
