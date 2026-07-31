"""Tier one: what magnetic momentum dumping can and cannot do.

The central claim is negative: because the magnetic torque is ``m x B`` for some
dipole ``m``, no choice of ``m`` produces a torque with a component along ``B``,
so at any instant the momentum along the field is untouchable. The tests below
assert that at the algebraic level, then at the level of one dumping step, then
over a full closed loop run with the field held fixed in inertial space, where the
conserved quantity is exact rather than approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

from attitude_control.algorithm.momentum import (
    MagneticDumping,
    controllable_momentum,
    cross_product_dipole,
    uncontrollable_momentum,
)
from attitude_control.configuration import (
    DUMPING_GAIN,
    PD_NATURAL_FREQUENCY,
    constant_field_environment,
    controllers,
    dumping_scenario,
    orbiting_field_environment,
)
from attitude_control.model.attitude import dcm_from_quaternion
from attitude_control.model.environment import magnetic_torque
from attitude_control.model.inertia import Spacecraft
from attitude_control.pipeline.scenario import run_scenario
from tests.conftest import EPSILON, rounding_bound, truncation_bound


def random_field(generator: np.random.Generator) -> np.ndarray:
    """Return a field vector of realistic low Earth orbit magnitude."""
    return generator.normal(size=3) * 2.0e-5


def test_cross_product_law_removes_only_the_perpendicular_component(
    generator: np.random.Generator,
) -> None:
    """The law delivers exactly minus the gain times the perpendicular momentum.

    ``m = (k / |B|^2) (h x B)`` gives ``m x B = -k (h - B_hat (B_hat . h))``,
    which is an identity, so the check is against the closed form. Tolerance:
    the expression is a chain of about ten operations on the scaled quantities,
    hence a relative error of a few eps; 64 eps of the torque scale covers it.
    """
    for _ in range(200):
        momentum = generator.normal(size=3) * 2.0
        field = random_field(generator)
        dipole = cross_product_dipole(momentum, field, gain=DUMPING_GAIN)
        torque = magnetic_torque(dipole, field)
        expected = -DUMPING_GAIN * controllable_momentum(momentum, field)
        scale = DUMPING_GAIN * float(np.linalg.norm(momentum))
        assert np.allclose(torque, expected, atol=64.0 * EPSILON * scale)


def test_no_dipole_can_torque_about_the_field(generator: np.random.Generator) -> None:
    """The claim, stated over arbitrary dipoles rather than only the control law."""
    for _ in range(500):
        dipole = generator.normal(size=3) * 50.0
        field = random_field(generator)
        torque = magnetic_torque(dipole, field)
        scale = float(np.linalg.norm(dipole)) * float(np.linalg.norm(field)) ** 2
        assert abs(float(np.dot(torque, field))) <= 64.0 * EPSILON * scale


def test_momentum_splits_into_a_removable_and_an_untouchable_part(
    generator: np.random.Generator,
) -> None:
    """The two projections are orthogonal and sum to the original momentum."""
    for _ in range(200):
        momentum = generator.normal(size=3) * 2.0
        field = random_field(generator)
        along = uncontrollable_momentum(momentum, field)
        across = controllable_momentum(momentum, field)
        scale = float(np.linalg.norm(momentum))
        assert np.allclose(along + across, momentum, atol=64.0 * EPSILON * scale)
        assert abs(float(np.dot(along, across))) <= 64.0 * EPSILON * scale**2
        assert abs(float(np.dot(across, field))) <= (
            64.0 * EPSILON * scale * float(np.linalg.norm(field))
        )


def test_one_dumping_step_leaves_the_along_field_momentum_alone(
    generator: np.random.Generator,
) -> None:
    """Integrating the produced torque for any interval cannot change the along part."""
    law = MagneticDumping(gain=DUMPING_GAIN, max_dipole=30.0)
    for _ in range(200):
        momentum = generator.normal(size=3) * 2.0
        field = random_field(generator)
        direction = field / np.linalg.norm(field)
        updated = momentum + 100.0 * law.torque(momentum, field)
        scale = float(np.linalg.norm(momentum))
        assert float(np.dot(updated, direction)) == pytest.approx(
            float(np.dot(momentum, direction)), abs=64.0 * EPSILON * scale
        )
        assert float(np.linalg.norm(controllable_momentum(updated, field))) < float(
            np.linalg.norm(controllable_momentum(momentum, field))
        )


def test_dipole_saturation_preserves_the_torque_direction(
    generator: np.random.Generator,
) -> None:
    """Scaling rather than clipping keeps the rod command pointing the right way."""
    for _ in range(100):
        momentum = generator.normal(size=3) * 50.0
        field = random_field(generator)
        unlimited = cross_product_dipole(momentum, field, gain=1.0)
        limited = cross_product_dipole(momentum, field, gain=1.0, max_dipole=5.0)
        assert float(np.linalg.norm(limited)) <= 5.0 * (1.0 + 1e-12)
        cosine = float(np.dot(unlimited, limited)) / (
            float(np.linalg.norm(unlimited)) * float(np.linalg.norm(limited))
        )
        assert cosine == pytest.approx(1.0, abs=1e-12)


def test_zero_field_produces_no_dipole() -> None:
    """Without a field there is nothing to push against, so the command is zero."""
    assert np.array_equal(cross_product_dipole(np.ones(3), np.zeros(3), 1.0), np.zeros(3))
    assert np.array_equal(uncontrollable_momentum(np.ones(3), np.zeros(3)), np.zeros(3))


def test_a_fixed_field_conserves_momentum_along_it_over_a_full_run(
    spacecraft: Spacecraft,
) -> None:
    """With the field frozen in inertial space the along-field momentum is exact.

    The only external torque is ``m x B``, which is orthogonal to ``B`` by
    construction, so ``d/dt (h_inertial . B_hat) = 0`` exactly for a field that
    does not move in the inertial frame. Any departure from that is numerical, so
    the bound is the sum of the two numerical error sources: accumulated rounding
    at the size of the momentum, and the RK4 global truncation error for the
    fastest closed loop mode, which is the attitude loop at 0.02 rad/s. Neither
    term uses an observed residual. The truncation term dominates and is
    conservative, because it assumes the attitude transient persists for the whole
    orbit; a violation of the orthogonality would change the projection by a
    finite fraction of 1.29 N m s, which is seven orders above this bound.

    The perpendicular component, which the law can act on, must fall by exactly
    the factor ``exp(-k T)`` the law prescribes, and that is asserted too so the
    test cannot pass by doing nothing at all. The law acts on the wheel momentum
    ``W h_w`` while the conserved quantity is the total ``J w + W h_w``; the
    attitude loop holds the body term far below the stored term, so the total
    follows the same exponential up to that ratio, and one per cent covers it.
    """
    step = 4.0
    config = dumping_scenario(
        spacecraft,
        controllers(spacecraft)[0],
        constant_field_environment(),
        orbits=1.0,
        time_step=step,
        sample_stride=5,
    )
    trace = run_scenario(config)

    axis = dcm_from_quaternion(trace.quaternion[0]).T @ trace.magnetic_field_body[0]
    axis = axis / np.linalg.norm(axis)
    projection = trace.inertial_momentum @ axis
    scale = float(np.linalg.norm(trace.inertial_momentum[0]))
    bound = rounding_bound(config.steps, scale) + truncation_bound(
        PD_NATURAL_FREQUENCY, step, config.duration, scale
    )

    assert float(np.max(np.abs(projection - projection[0]))) < bound
    perpendicular = np.linalg.norm(
        trace.inertial_momentum - np.outer(projection, axis), axis=1
    )
    decay = float(np.exp(-DUMPING_GAIN * config.duration))
    assert perpendicular[-1] == pytest.approx(perpendicular[0] * decay, rel=0.01)


def test_an_orbiting_field_removes_every_component(spacecraft: Spacecraft) -> None:
    """Along the orbit the untouchable direction moves, so the whole vector goes.

    This is the counterpart of the previous test and is what makes magnetic
    dumping usable at all. The threshold is qualitative on purpose: the exact
    fraction removed depends on the orbit and the gain, and is reported in the
    results rather than pinned here.
    """
    config = dumping_scenario(
        spacecraft,
        controllers(spacecraft)[0],
        orbiting_field_environment(),
        orbits=2.0,
        time_step=4.0,
        sample_stride=5,
    )
    trace = run_scenario(config)
    stored = np.linalg.norm(trace.stored_body_momentum, axis=1)
    assert stored[-1] < 0.3 * stored[0]


def test_dumping_rejects_impossible_settings() -> None:
    """A negative gain or a non-positive dipole limit is refused."""
    with pytest.raises(ValueError, match="non-negative"):
        cross_product_dipole(np.ones(3), np.ones(3), gain=-1.0)
    with pytest.raises(ValueError, match="maximum dipole"):
        cross_product_dipole(np.ones(3), np.ones(3), gain=1.0, max_dipole=0.0)
