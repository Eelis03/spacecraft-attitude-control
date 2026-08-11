"""Tier one: properties of the redundant wheel torque allocation."""

from __future__ import annotations

import numpy as np
import pytest

from attitude_control.algorithm.allocation import (
    allocate,
    apply_limits,
    delivered_body_torque,
    minimum_norm_wheel_torque,
    null_space_wheel_torque,
)
from attitude_control.model.inertia import WheelArray, pyramid_wheel_axes
from attitude_control.numeric import FloatArray
from tests.conftest import EPSILON

# The allocation is two matrix products with matrices of norm at most 1.16 and a
# condition number of exactly one for the isotropic pyramid. Four accumulations
# per entry give a bound of 4 eps on the relative error; the factor of 64 below
# covers the rest of the chain with room to spare and does not depend on any
# observed residual.
_ALLOCATION_TOLERANCE = 64.0 * EPSILON


def achievable_torques(
    wheels: WheelArray, generator: np.random.Generator, count: int
) -> list[FloatArray]:
    """Return commands whose minimum norm solution fits inside the torque limits."""
    commands: list[FloatArray] = []
    while len(commands) < count:
        candidate = generator.normal(size=3) * 0.05
        if float(np.max(np.abs(wheels.pseudoinverse @ candidate))) <= wheels.max_torque:
            commands.append(candidate)
    return commands


def test_allocation_reproduces_the_commanded_torque(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """Inside the achievable set the delivered torque equals the command exactly.

    The array has full row rank, so ``W W^+ = I`` and the allocation is an exact
    solution rather than a least squares approximation. Tolerance: rounding in two
    small matrix products, as derived at the top of this module.
    """
    for command in achievable_torques(wheels, generator, 200):
        result = allocate(wheels, command, np.zeros(4))
        assert not result.saturated
        assert np.allclose(result.delivered_torque, command, atol=_ALLOCATION_TOLERANCE)


def test_allocation_is_exact_with_null_space_steering(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """Adding the null space term changes the wheel torques but not the body torque."""
    for command in achievable_torques(wheels, generator, 100):
        momentum = generator.normal(size=4) * 0.5
        plain = allocate(wheels, command, momentum, enforce_limits=False)
        steered = allocate(wheels, command, momentum, null_space_gain=0.05, enforce_limits=False)
        assert np.allclose(steered.delivered_torque, command, atol=_ALLOCATION_TOLERANCE)
        assert not np.allclose(steered.wheel_torque, plain.wheel_torque, atol=1e-9)


def test_null_space_torque_produces_no_body_torque(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """The steering term lies in the null space of the distribution matrix."""
    for _ in range(200):
        momentum = generator.normal(size=4)
        torque = null_space_wheel_torque(wheels, momentum, gain=0.1)
        assert np.allclose(
            delivered_body_torque(wheels, torque),
            np.zeros(3),
            atol=_ALLOCATION_TOLERANCE * float(np.linalg.norm(momentum)),
        )


def test_null_space_steering_drives_the_null_component_towards_the_target(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """The steering term reduces the part of the wheel momentum it can act on."""
    projector = wheels.null_projector
    for _ in range(50):
        momentum = generator.normal(size=4)
        torque = null_space_wheel_torque(wheels, momentum, gain=0.1)
        before = float(np.linalg.norm(projector @ momentum))
        after = float(np.linalg.norm(projector @ (momentum + 0.5 * torque)))
        assert after < before


def test_minimum_norm_solution_is_the_smallest(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """No other exact solution has a smaller Euclidean norm."""
    null_direction = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
    assert np.allclose(wheels.axes @ null_direction, np.zeros(3), atol=_ALLOCATION_TOLERANCE)
    for command in achievable_torques(wheels, generator, 50):
        base = minimum_norm_wheel_torque(wheels, command)
        for offset in (-0.5, -0.1, 0.1, 0.5):
            alternative = base + offset * null_direction
            assert np.allclose(
                delivered_body_torque(wheels, alternative), command, atol=_ALLOCATION_TOLERANCE
            )
            assert float(np.linalg.norm(alternative)) > float(np.linalg.norm(base))


def test_torque_limit_clips_and_reports(wheels: WheelArray) -> None:
    """Motor torque beyond the limit is clipped and the flag is raised."""
    excessive = np.array([1.0, -1.0, 0.5, -0.5]) * wheels.max_torque * 2.0
    limited, torque_saturated, momentum_saturated = apply_limits(wheels, excessive, np.zeros(4))
    assert torque_saturated
    assert not momentum_saturated
    assert np.allclose(np.abs(limited), np.abs(np.clip(excessive, -0.05, 0.05)))
    assert float(np.max(np.abs(limited))) <= wheels.max_torque


def test_momentum_limit_blocks_only_the_outward_direction(wheels: WheelArray) -> None:
    """A saturated wheel can still be slowed down but cannot be sped up.

    This is the physical behaviour: the limit is on stored momentum, so torque
    that removes momentum stays available.
    """
    momentum = np.array([wheels.max_momentum, -wheels.max_momentum, 0.0, 0.0])
    outward = np.array([0.01, -0.01, 0.01, 0.01])
    limited, _, momentum_saturated = apply_limits(wheels, outward, momentum)
    assert momentum_saturated
    assert limited[0] == 0.0
    assert limited[1] == 0.0
    assert np.allclose(limited[2:], outward[2:])

    inward = np.array([-0.01, 0.01, 0.0, 0.0])
    restored, _, blocked = apply_limits(wheels, inward, momentum)
    assert not blocked
    assert np.allclose(restored, inward)


def test_saturation_makes_the_delivered_torque_fall_short(wheels: WheelArray) -> None:
    """A command outside the achievable set is not delivered, and that is reported."""
    command = np.array([1.0, 0.0, 0.0])
    result = allocate(wheels, command, np.zeros(4))
    assert result.torque_saturated
    assert float(np.linalg.norm(result.delivered_torque)) < float(np.linalg.norm(command))
    assert float(np.max(np.abs(result.wheel_torque))) <= wheels.max_torque


def test_pseudoinverse_is_a_right_inverse(wheels: WheelArray) -> None:
    """``W W^+`` is the identity, which is the reason allocation can be exact."""
    assert np.allclose(wheels.axes @ wheels.pseudoinverse, np.eye(3), atol=_ALLOCATION_TOLERANCE)
    assert np.allclose(
        wheels.null_projector @ wheels.null_projector,
        wheels.null_projector,
        atol=_ALLOCATION_TOLERANCE,
    )
    assert np.allclose(wheels.null_projector, wheels.null_projector.T, atol=_ALLOCATION_TOLERANCE)
    assert int(np.linalg.matrix_rank(wheels.null_projector)) == 1


def test_wheel_speed_and_momentum_conversions_are_inverse(wheels: WheelArray) -> None:
    """Momentum and speed convert both ways through the axial inertia."""
    speeds = np.array([100.0, -250.0, 50.0, 0.0])
    assert np.allclose(wheels.speed(wheels.momentum(speeds)), speeds, atol=_ALLOCATION_TOLERANCE)
    assert wheels.max_speed == pytest.approx(
        wheels.max_momentum / float(np.min(wheels.axial_inertia)), rel=1e-15
    )


def test_allocation_works_for_a_non_isotropic_array(generator: np.random.Generator) -> None:
    """Exactness does not depend on the array being isotropic, only on its rank."""
    skewed = WheelArray(
        axes=pyramid_wheel_axes(half_angle=np.deg2rad(35.0)),
        axial_inertia=np.array([0.006, 0.007, 0.0065, 0.0064]),
        max_torque=0.2,
        max_momentum=4.0,
    )
    for _ in range(100):
        command = generator.normal(size=3) * 0.02
        result = allocate(skewed, command, np.zeros(4), enforce_limits=False)
        assert np.allclose(result.delivered_torque, command, atol=_ALLOCATION_TOLERANCE)


def test_allocation_rejects_a_negative_null_space_gain(wheels: WheelArray) -> None:
    """A negative steering gain would push wheel speeds outward, so it is refused."""
    with pytest.raises(ValueError, match="non-negative"):
        null_space_wheel_torque(wheels, np.zeros(4), gain=-1.0)
