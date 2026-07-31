"""Shared fixtures and tolerance helpers for the test suite.

Every tolerance used in this suite is derived from the measurement it applies to
and the derivation is stated in the docstring of the test that uses it. The two
helpers here cover the cases that recur.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from attitude_control.configuration import reference_spacecraft
from attitude_control.model.dynamics import PlantState, body_rate_derivative
from attitude_control.model.inertia import Spacecraft, WheelArray, pyramid_wheel_axes
from attitude_control.numeric import FloatArray
from attitude_control.pipeline.integrator import rk4_step

EPSILON: float = float(np.finfo(np.float64).eps)


def rounding_bound(steps: int, magnitude: float, safety: float = 10.0) -> float:
    """Return a bound on accumulated floating point rounding over ``steps`` steps.

    Each arithmetic operation on a quantity of size ``magnitude`` commits an error
    of at most ``eps * magnitude``. Successive errors are not correlated, so over
    ``n`` steps they accumulate as ``sqrt(n)`` rather than ``n``. The ``safety``
    factor covers the several operations performed per step. The bound depends
    only on the step count and the size of the quantity, never on an observed
    error, so a test using it cannot end up sitting on its own boundary.
    """
    return float(safety * np.sqrt(steps) * EPSILON * magnitude)


def truncation_bound(frequency: float, step: float, duration: float, magnitude: float) -> float:
    """Return an upper bound on the RK4 global truncation error of a closed loop run.

    A mode oscillating at ``frequency`` integrated with step ``step`` picks up a
    local error of ``(frequency * step)^5 / 120`` relative to its amplitude on
    each step. Over ``duration / step`` steps, assuming no cancellation between
    them, the accumulation is ``duration * frequency^5 * step^4 / 120`` relative,
    which multiplied by ``magnitude`` is the bound returned here.

    The bound is deliberately conservative in two ways: the fastest closed loop
    mode is used even though it is a decaying transient, and the errors are summed
    rather than treated as a random walk. It is nonetheless a property of the
    integrator and the design, not of any measured residual.
    """
    return duration * frequency**5 * step**4 / 120.0 * magnitude


def integrate_body_rate(
    spacecraft: Spacecraft,
    body_rate: FloatArray,
    step: float,
    steps: int,
) -> FloatArray:
    """Integrate the torque free Euler equation and return every sampled body rate."""

    idle = np.zeros(spacecraft.wheels.count)

    def derivative(time: float, state: FloatArray) -> FloatArray:
        del time
        return body_rate_derivative(spacecraft, state, idle, idle)

    history = np.empty((steps + 1, 3), dtype=np.float64)
    history[0] = body_rate
    current = np.asarray(body_rate, dtype=np.float64)
    for index in range(steps):
        current = rk4_step(derivative, index * step, current, step)
        history[index + 1] = current
    return history


def integrate_plant(
    spacecraft: Spacecraft,
    state: PlantState,
    wheel_torque: Callable[[float, PlantState], FloatArray],
    step: float,
    steps: int,
    normalise: bool = False,
) -> FloatArray:
    """Integrate the full plant with a caller supplied wheel torque and no external torque.

    Normalisation of the quaternion is off by default so that the norm drift of
    the integrator stays visible.
    """
    from attitude_control.model.dynamics import state_derivative
    from attitude_control.pipeline.integrator import normalise_quaternion_state

    def derivative(time: float, packed: FloatArray) -> FloatArray:
        current = PlantState.unflatten(packed)
        return state_derivative(spacecraft, current, wheel_torque(time, current))

    packed = state.flatten()
    history = np.empty((steps + 1, packed.size), dtype=np.float64)
    history[0] = packed
    for index in range(steps):
        packed = rk4_step(derivative, index * step, packed, step)
        if normalise:
            packed = normalise_quaternion_state(packed)
        history[index + 1] = packed
    return history


@pytest.fixture(scope="session")
def spacecraft() -> Spacecraft:
    """The reference vehicle used by the examples and the regression run."""
    return reference_spacecraft()


@pytest.fixture(scope="session")
def wheels(spacecraft: Spacecraft) -> WheelArray:
    """The four wheel pyramid of the reference vehicle."""
    return spacecraft.wheels


@pytest.fixture(scope="session")
def axisymmetric() -> Spacecraft:
    """A body that is symmetric about its third axis, with idle wheels."""
    return Spacecraft(
        inertia=np.diag([100.0, 100.0, 150.0]),
        wheels=WheelArray(
            axes=pyramid_wheel_axes(),
            axial_inertia=np.full(4, 0.0064),
            max_torque=0.05,
            max_momentum=4.0,
        ),
    )


@pytest.fixture(scope="session")
def asymmetric() -> Spacecraft:
    """A body with three distinct principal moments, with idle wheels."""
    return Spacecraft(
        inertia=np.diag([70.0, 100.0, 130.0]),
        wheels=WheelArray(
            axes=pyramid_wheel_axes(),
            axial_inertia=np.full(4, 0.0064),
            max_torque=0.05,
            max_momentum=4.0,
        ),
    )


@pytest.fixture
def generator() -> np.random.Generator:
    """A seeded random generator, so every random test is reproducible."""
    return np.random.default_rng(20260731)
