"""Distribution of a commanded body torque over a redundant reaction wheel array.

A body torque command ``L`` must be realised by motor torques ``u`` on ``n``
wheels. The body torque delivered by the wheels is ``-W u``, so the allocation
problem is the underdetermined linear system

    W u = -L

For the four wheel pyramid ``W`` is ``3 x 4`` with full row rank, so solutions
exist for every command and form a one dimensional affine set. The minimum norm
solution is ``u = -W^+ L`` with ``W^+`` the Moore-Penrose pseudoinverse, and any
vector in the null space of ``W`` may be added without changing the body torque.

Null space wheel speed management
---------------------------------
That freedom is used to steer the wheel speeds. Adding

    u_null = -k (I - W^+ W) (h_w - h_target)

drives the null space component of the stored wheel momentum towards a target,
usually zero, so the array stays away from its speed limits and away from wheel
speeds near zero where bearing friction is worst. The projector ``I - W^+ W`` is
symmetric and idempotent, so the added term is exactly in the null space and the
delivered body torque is unchanged. This is the standard redundancy resolution
also used for control moment gyroscope arrays; see Markley et al. (2010) for the
resulting torque and momentum envelopes.

Saturation
----------
Two limits are enforced, in this order. Motor torque is clipped to the per wheel
limit. Then any torque that would drive a wheel past its momentum limit in the
direction it is already saturated is set to zero, which models a wheel that has
reached maximum speed and can absorb no more. Both operations break the exact
torque reproduction, so :func:`allocate` reports whether either acted.

References
----------
Markley, F. L., Reynolds, R. G., Liu, F. X. and Lebsock, K. L. (2010). Maximum
torque and momentum envelopes for reaction wheel arrays. *Journal of Guidance,
Control, and Dynamics*, 33(5), 1606-1614. DOI 10.2514/1.47968.

Wie, B. (2008). *Space Vehicle Dynamics and Control*, 2nd edition. AIAA,
chapter 7. DOI 10.2514/4.860119.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.model.inertia import WheelArray
from attitude_control.numeric import FloatArray, as_vector

__all__ = [
    "Allocation",
    "allocate",
    "apply_limits",
    "delivered_body_torque",
    "minimum_norm_wheel_torque",
    "null_space_wheel_torque",
]


@dataclass(frozen=True, slots=True)
class Allocation:
    """The outcome of distributing one body torque command over the wheels."""

    wheel_torque: FloatArray
    delivered_torque: FloatArray
    torque_saturated: bool
    momentum_saturated: bool

    @property
    def saturated(self) -> bool:
        """True when either limit acted on this command."""
        return self.torque_saturated or self.momentum_saturated


def delivered_body_torque(wheels: WheelArray, wheel_torque: ArrayLike) -> FloatArray:
    """Return the body torque ``-W u`` actually produced by motor torques ``u``."""
    return -(wheels.axes @ as_vector(wheel_torque, wheels.count))


def minimum_norm_wheel_torque(wheels: WheelArray, commanded_torque: ArrayLike) -> FloatArray:
    """Return the least squares wheel torque realising a commanded body torque.

    Because the pyramid array has full row rank this is an exact solution, not an
    approximation, and it is the one of smallest Euclidean norm.
    """
    return -(wheels.pseudoinverse @ as_vector(commanded_torque, 3))


def null_space_wheel_torque(
    wheels: WheelArray,
    wheel_momentum: ArrayLike,
    gain: float,
    target_momentum: ArrayLike | None = None,
) -> FloatArray:
    """Return a wheel torque in the null space of ``W`` that steers wheel speeds.

    The result produces no body torque, to machine precision, so it can be added
    to any allocation without disturbing the attitude loop.
    """
    momentum = as_vector(wheel_momentum, wheels.count)
    target = (
        np.zeros(wheels.count)
        if target_momentum is None
        else as_vector(target_momentum, wheels.count)
    )
    if gain < 0.0:
        raise ValueError("null space gain must be non-negative")
    return -gain * (wheels.null_projector @ (momentum - target))


def apply_limits(
    wheels: WheelArray,
    wheel_torque: ArrayLike,
    wheel_momentum: ArrayLike,
) -> tuple[FloatArray, bool, bool]:
    """Clip motor torque and block torque that would exceed the momentum limit.

    Returns the limited torque together with flags for whether the torque limit
    and the momentum limit acted.
    """
    torque = as_vector(wheel_torque, wheels.count)
    momentum = as_vector(wheel_momentum, wheels.count)

    clipped = np.clip(torque, -wheels.max_torque, wheels.max_torque)
    torque_saturated = bool(np.any(clipped != torque))

    # A wheel at its momentum limit can still be slowed down, so only torque that
    # would push it further out is removed.
    at_positive_limit = (momentum >= wheels.max_momentum) & (clipped > 0.0)
    at_negative_limit = (momentum <= -wheels.max_momentum) & (clipped < 0.0)
    blocked = at_positive_limit | at_negative_limit
    limited = np.where(blocked, 0.0, clipped)
    return limited, torque_saturated, bool(np.any(blocked))


def allocate(
    wheels: WheelArray,
    commanded_torque: ArrayLike,
    wheel_momentum: ArrayLike,
    *,
    null_space_gain: float = 0.0,
    target_momentum: ArrayLike | None = None,
    enforce_limits: bool = True,
) -> Allocation:
    """Distribute a commanded body torque over the wheels.

    With ``enforce_limits`` disabled, or with a command inside the achievable set,
    ``delivered_torque`` equals ``commanded_torque`` to machine precision.
    """
    command = as_vector(commanded_torque, 3)
    torque = minimum_norm_wheel_torque(wheels, command)
    if null_space_gain > 0.0:
        torque = torque + null_space_wheel_torque(
            wheels, wheel_momentum, null_space_gain, target_momentum
        )

    torque_saturated = False
    momentum_saturated = False
    if enforce_limits:
        torque, torque_saturated, momentum_saturated = apply_limits(wheels, torque, wheel_momentum)

    return Allocation(
        wheel_torque=torque,
        delivered_torque=delivered_body_torque(wheels, torque),
        torque_saturated=torque_saturated,
        momentum_saturated=momentum_saturated,
    )
