"""Fixed step explicit integration of the attitude state.

Classical fourth order Runge-Kutta is used. For attitude work its relevant
property is that, applied to the quaternion kinematic equation at a constant body
rate, one step multiplies the quaternion norm by a factor that can be written in
closed form. Writing ``x = |w| dt / 2``, the exact update has scalar part
``cos x`` and vector magnitude ``sin x``, while RK4 produces the truncated series
``1 - x^2/2 + x^4/24`` and ``x - x^3/6``. The sum of their squares is

    1 - x^6 / 72 + x^8 / 576

so the norm error after one step is ``-x^6 / 144`` to leading order and after
``N`` steps it is ``-N x^6 / 144``. That expression, not an observed number, is
what the norm drift test compares against.

Normalisation
-------------
The propagator does not renormalise unless asked to. Renormalising every step
would make the norm test vacuous, so the tests run with it off and the scenarios
run with it on. Renormalisation is a projection onto the unit sphere and changes
the attitude by an amount of the same order as the drift it removes, so it does
not improve accuracy; it only stops the drift accumulating over long runs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.numeric import FloatArray

__all__ = ["Derivative", "normalise_quaternion_state", "rk4_step", "step_norm_drift"]

Derivative = Callable[[float, FloatArray], FloatArray]


def rk4_step(derivative: Derivative, time: float, state: ArrayLike, step: float) -> FloatArray:
    """Advance ``state`` by one classical fourth order Runge-Kutta step."""
    y = np.asarray(state, dtype=np.float64)
    k1 = derivative(time, y)
    k2 = derivative(time + 0.5 * step, y + 0.5 * step * k1)
    k3 = derivative(time + 0.5 * step, y + 0.5 * step * k2)
    k4 = derivative(time + step, y + step * k3)
    return y + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def normalise_quaternion_state(state: ArrayLike) -> FloatArray:
    """Return a packed state whose leading four entries have unit norm."""
    y = np.array(state, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(y[:4]))
    if norm == 0.0:
        raise ValueError("the quaternion part of the state collapsed to zero")
    y[:4] /= norm
    return y


def step_norm_drift(rate_magnitude: float, step: float, steps: int) -> float:
    """Return the predicted quaternion norm drift of RK4 at a constant body rate.

    The value is ``N x^6 / 144`` with ``x = |w| dt / 2``, derived in the module
    docstring from the truncated cosine and sine series of the RK4 update. It is
    the quantity the norm invariant test measures against, so that the tolerance
    comes from the integrator rather than from an observed error.
    """
    half_angle = 0.5 * abs(rate_magnitude) * step
    return steps * half_angle**6 / 144.0
