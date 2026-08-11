"""Tier one: invariants of the rigid body, wheel, and environment models."""

from __future__ import annotations

import numpy as np
import pytest

from attitude_control.algorithm.controller import QuaternionFeedbackPD
from attitude_control.configuration import slew_scenario
from attitude_control.model.attitude import (
    dcm_from_quaternion,
    is_rotation_matrix,
    quaternion_from_axis_angle,
    quaternion_identity,
)
from attitude_control.model.dynamics import (
    PlantState,
    body_angular_momentum,
    body_torque_from_wheel_torque,
    gyroscopic_torque,
    inertial_angular_momentum,
    rotational_kinetic_energy,
)
from attitude_control.model.environment import (
    EARTH_RADIUS,
    GRAVITATIONAL_PARAMETER,
    CircularOrbit,
    TiltedDipoleField,
    gravity_gradient_torque,
    magnetic_torque,
)
from attitude_control.model.inertia import (
    ISOTROPIC_PYRAMID_HALF_ANGLE,
    Spacecraft,
    WheelArray,
    principal_inertia,
    pyramid_wheel_axes,
    symmetric_inertia,
)
from attitude_control.pipeline.integrator import step_norm_drift
from attitude_control.pipeline.scenario import run_scenario
from tests.conftest import EPSILON, integrate_body_rate, integrate_plant, rounding_bound


def test_quaternion_norm_drift_matches_the_integrator_prediction(
    asymmetric: Spacecraft,
) -> None:
    """RK4 norm drift over a long integration equals its closed form value.

    The body spins about a principal axis with no torque, so the body rate is
    exactly constant and one RK4 step multiplies the quaternion norm squared by
    ``1 - x^6/72 + x^8/576`` with ``x = |w| dt / 2``. The drift after ``N`` steps
    is therefore ``N x^6 / 144``, computed by ``step_norm_drift``. The first
    neglected term is smaller than the retained one by ``x^2 / 8``, which is
    7.8e-5 here, so a one per cent tolerance sits two orders above the modelling
    error and is not derived from any observed number.
    """
    rate = np.array([0.0, 0.0, 0.1])
    step, steps = 0.5, 2000
    state = PlantState.create(quaternion_identity(), rate, np.zeros(4))
    history = integrate_plant(
        asymmetric, state, lambda t, s: np.zeros(4), step, steps, normalise=False
    )

    assert np.allclose(history[:, 4:7], rate, atol=rounding_bound(steps, float(np.max(rate))))
    drift = 1.0 - float(np.linalg.norm(history[-1, :4]))
    predicted = step_norm_drift(float(np.linalg.norm(rate)), step, steps)
    assert predicted == pytest.approx(3.39e-9, rel=0.01)
    assert drift == pytest.approx(predicted, rel=0.01)


def test_attitude_matrix_stays_a_proper_rotation_through_a_long_run(
    asymmetric: Spacecraft,
) -> None:
    """Every attitude matrix along a long integration is orthonormal with unit determinant.

    Tolerance: the matrix is quadratic in the quaternion, so for a quaternion of
    norm ``1 - d`` it satisfies ``A A^T = (1 - d)^4 I`` and ``det A = (1 - d)^6``
    exactly. The largest defect is therefore ``6 d``, with ``d`` the norm drift
    predicted by ``step_norm_drift``. The bound below is ten times ``d``, which
    covers the worst of the two defects with a factor of 1.7 to spare and is
    derived entirely from the integrator, not from an observed value.
    """
    step, steps = 0.5, 2000
    rate = np.array([0.0, 0.0, 0.1])
    history = integrate_plant(
        asymmetric,
        PlantState.create(quaternion_identity(), rate, np.zeros(4)),
        lambda t, s: np.zeros(4),
        step,
        steps,
        normalise=False,
    )
    drift = step_norm_drift(float(np.linalg.norm(rate)), step, steps)
    for packed in history[::100]:
        assert is_rotation_matrix(dcm_from_quaternion(packed[:4]), tolerance=10.0 * drift)
    # Renormalising restores a proper rotation to machine precision.
    final = dcm_from_quaternion(history[-1, :4] / np.linalg.norm(history[-1, :4]))
    assert is_rotation_matrix(final, tolerance=64.0 * EPSILON)


def test_inertial_momentum_is_conserved_with_active_wheels(spacecraft: Spacecraft) -> None:
    """Total angular momentum in the inertial frame is constant when nothing pushes.

    The wheels are commanded through a full slew, so the body momentum and the
    wheel momentum both change by a large fraction of their range while their sum,
    transported to the inertial frame, does not. Tolerance: rounding accumulated
    over the run at the size of the momentum involved, from ``rounding_bound``,
    with no reference to the observed drift.
    """
    controller = QuaternionFeedbackPD(spacecraft, natural_frequency=0.02, damping_ratio=0.7)
    config = slew_scenario(spacecraft, controller, duration=400.0, time_step=0.5)
    trace = run_scenario(config)

    stored = np.linalg.norm(trace.stored_body_momentum, axis=1)
    assert float(np.max(stored)) > 0.5, "the wheels must actually be exercised"

    drift = np.linalg.norm(trace.inertial_momentum - trace.inertial_momentum[0], axis=1)
    assert float(np.max(drift)) < rounding_bound(config.steps, float(np.max(stored)))


def test_body_momentum_bookkeeping_matches_the_definition(
    spacecraft: Spacecraft, generator: np.random.Generator
) -> None:
    """``J w + W h_w`` is what the model reports, and the inertial form is its rotation."""
    for _ in range(50):
        rate = generator.normal(size=3) * 0.01
        wheel = generator.normal(size=4) * 0.5
        quaternion = quaternion_from_axis_angle(generator.normal(size=3), 1.1)
        expected = spacecraft.inertia @ rate + spacecraft.wheels.axes @ wheel
        assert np.allclose(
            body_angular_momentum(spacecraft, rate, wheel), expected, atol=64.0 * EPSILON
        )
        assert np.allclose(
            inertial_angular_momentum(spacecraft, quaternion, rate, wheel),
            dcm_from_quaternion(quaternion).T @ expected,
            atol=64.0 * EPSILON,
        )


def test_wheel_torque_pushes_the_body_the_other_way(spacecraft: Spacecraft) -> None:
    """Spinning a wheel up applies the opposite torque to the body."""
    torque = np.array([0.01, 0.0, 0.0, 0.0])
    body = body_torque_from_wheel_torque(spacecraft, torque)
    assert np.allclose(body, -0.01 * spacecraft.wheels.axes[:, 0], atol=64.0 * EPSILON)


def test_gyroscopic_torque_does_no_work(
    spacecraft: Spacecraft, generator: np.random.Generator
) -> None:
    """``w . (w x h)`` vanishes, so the coupling term cannot change energy."""
    for _ in range(50):
        rate = generator.normal(size=3)
        wheel = generator.normal(size=4)
        product = float(np.dot(rate, gyroscopic_torque(spacecraft, rate, wheel)))
        scale = float(np.linalg.norm(rate)) * float(
            np.linalg.norm(body_angular_momentum(spacecraft, rate, wheel))
        )
        assert product == pytest.approx(0.0, abs=64.0 * EPSILON * scale)


def test_torque_free_asymmetric_body_conserves_energy_and_momentum(
    asymmetric: Spacecraft,
) -> None:
    """Both quadratic invariants of Euler's equation are held to integrator accuracy.

    Tolerance: the local defect of RK4 in a conserved quantity is of order
    ``(|w| dt)^5``, here 3.1e-14 relative, and accumulates over 6000 steps to at
    most 1.9e-10 relative in the worst case of no cancellation. The tolerance of
    1e-8 leaves a factor of 50 above that bound.
    """
    rate = np.array([0.3, 0.5, 0.2])
    step, steps = 0.01, 6000
    history = integrate_body_rate(asymmetric, rate, step, steps)

    energy = np.array([rotational_kinetic_energy(asymmetric, w, np.zeros(4)) for w in history])
    momentum = np.linalg.norm(history @ asymmetric.inertia, axis=1)
    assert np.max(np.abs(energy / energy[0] - 1.0)) < 1e-8
    assert np.max(np.abs(momentum / momentum[0] - 1.0)) < 1e-8


def test_axisymmetric_body_precesses_at_the_analytic_rate(axisymmetric: Spacecraft) -> None:
    """Torque free motion of a symmetric body matches the closed form solution.

    For ``J = diag(Jt, Jt, Ja)`` the axial rate is constant and the transverse
    rate rotates in the body frame at ``lambda = w3 (Ja - Jt) / Jt``. Tolerance:
    the RK4 global error for a rotation at rate ``lambda`` over time ``T`` is
    ``T lambda^5 dt^4 / 120`` in phase, here 1.6e-10 rad, giving an amplitude
    error of 7.8e-12; 1e-10 leaves a factor of 12 above that.
    """
    transverse, axial = 0.05, 0.2
    rate = np.array([transverse, 0.0, axial])
    step, steps = 0.05, 6000
    history = integrate_body_rate(axisymmetric, rate, step, steps)
    time = step * np.arange(steps + 1)

    moments = np.diag(axisymmetric.inertia)
    precession = axial * (moments[2] - moments[0]) / moments[0]
    assert precession == pytest.approx(0.1, abs=1e-15)

    assert np.allclose(history[:, 2], axial, atol=rounding_bound(steps, axial))
    assert np.allclose(history[:, 0], transverse * np.cos(precession * time), atol=1e-10)
    assert np.allclose(history[:, 1], transverse * np.sin(precession * time), atol=1e-10)


def test_intermediate_axis_is_unstable_and_the_others_are_not(
    asymmetric: Spacecraft,
) -> None:
    """Spin about the intermediate axis flips; spin about the extreme axes does not.

    The flip is genuinely chaotic, so only the qualitative signature is asserted.
    The flip time and the trajectory after the flip depend on the summation order
    of the arithmetic and are not reproducible across machines, so neither is
    pinned here or in the regression file.
    """
    step, steps = 0.005, 12000
    perturbation = 1.0e-3
    signatures = {}
    for axis in range(3):
        rate = np.full(3, perturbation)
        rate[axis] = 1.0
        history = integrate_body_rate(asymmetric, rate, step, steps)
        magnitude = np.linalg.norm(history, axis=1)
        signatures[axis] = float(np.min(history[:, axis] / magnitude))

    assert signatures[1] < -0.9, "the intermediate axis should flip"
    assert signatures[0] > 0.99, "the minor axis spin should stay put"
    assert signatures[2] > 0.99, "the major axis spin should stay put"


def test_the_unstable_case_still_conserves_energy_and_momentum(
    asymmetric: Spacecraft,
) -> None:
    """Chaos in the trajectory does not licence a drift in the invariants.

    Tolerance: as for the stable case, the local RK4 defect in a conserved
    quantity is of order ``(|w| dt)^5``, here 3.1e-14 relative, and 12000 steps
    bound the accumulation at 3.7e-10; 1e-8 leaves a factor of 27.
    """
    rate = np.array([1.0e-3, 1.0, 1.0e-3])
    step, steps = 0.005, 12000
    history = integrate_body_rate(asymmetric, rate, step, steps)
    energy = np.array([rotational_kinetic_energy(asymmetric, w, np.zeros(4)) for w in history])
    momentum = np.linalg.norm(history @ asymmetric.inertia, axis=1)
    assert np.max(np.abs(energy / energy[0] - 1.0)) < 1e-8
    assert np.max(np.abs(momentum / momentum[0] - 1.0)) < 1e-8


def test_gravity_gradient_torque_is_perpendicular_to_nadir(
    spacecraft: Spacecraft, generator: np.random.Generator
) -> None:
    """The gravity gradient torque has no component along the nadir direction."""
    for _ in range(50):
        nadir = generator.normal(size=3)
        nadir = nadir / np.linalg.norm(nadir)
        torque = gravity_gradient_torque(spacecraft, nadir, 6.9e6)
        assert float(np.dot(torque, nadir)) == pytest.approx(
            0.0, abs=64.0 * EPSILON * float(np.linalg.norm(torque))
        )


def test_gravity_gradient_torque_vanishes_along_a_principal_axis() -> None:
    """A principal axis pointed at nadir gives no gravity gradient torque."""
    body = Spacecraft(
        inertia=symmetric_inertia((90.0, 100.0, 75.0)),
        wheels=WheelArray(pyramid_wheel_axes(), np.full(4, 0.0064), 0.05, 4.0),
    )
    for axis in np.eye(3):
        torque = gravity_gradient_torque(body, axis, 6.9e6)
        assert np.allclose(torque, np.zeros(3), atol=64.0 * EPSILON * 100.0)


def test_gravity_gradient_torque_scales_with_the_inverse_cube_of_radius(
    spacecraft: Spacecraft,
) -> None:
    """Doubling the orbit radius divides the gravity gradient torque by eight."""
    nadir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    near = gravity_gradient_torque(spacecraft, nadir, 7.0e6)
    far = gravity_gradient_torque(spacecraft, nadir, 14.0e6)
    assert np.allclose(far * 8.0, near, rtol=1e-12)


def test_magnetic_torque_is_always_perpendicular_to_the_field(
    generator: np.random.Generator,
) -> None:
    """No dipole can produce a torque with a component along the field."""
    for _ in range(200):
        dipole = generator.normal(size=3) * 10.0
        field = generator.normal(size=3) * 3.0e-5
        torque = magnetic_torque(dipole, field)
        scale = float(np.linalg.norm(dipole)) * float(np.linalg.norm(field)) ** 2
        assert float(np.dot(torque, field)) == pytest.approx(0.0, abs=64.0 * EPSILON * scale)


def test_dipole_field_falls_off_as_the_inverse_cube() -> None:
    """The dipole magnitude scales as one over the cube of the geocentric distance."""
    inner = TiltedDipoleField(orbit=CircularOrbit(radius=7.0e6, inclination=0.9))
    outer = TiltedDipoleField(orbit=CircularOrbit(radius=14.0e6, inclination=0.9))
    ratio = float(np.linalg.norm(inner.field_inertial(0.0))) / float(
        np.linalg.norm(outer.field_inertial(0.0))
    )
    assert ratio == pytest.approx(8.0, rel=1e-12)


def test_dipole_field_is_twice_as_strong_over_the_pole() -> None:
    """A dipole field on axis is twice the field at the same distance on the equator.

    With the axis aligned, ``B = 2 mu / r^3``, and in the equatorial plane
    ``B = mu / r^3``, which is a closed form check on the field expression.
    """
    aligned = TiltedDipoleField(
        orbit=CircularOrbit(radius=7.0e6, inclination=0.5 * np.pi, initial_argument=0.5 * np.pi),
        tilt=0.0,
    )
    equatorial = TiltedDipoleField(orbit=CircularOrbit(radius=7.0e6, inclination=0.0), tilt=0.0)
    polar_strength = float(np.linalg.norm(aligned.field_inertial(0.0)))
    equatorial_strength = float(np.linalg.norm(equatorial.field_inertial(0.0)))
    assert polar_strength == pytest.approx(2.0 * equatorial_strength, rel=1e-12)


def test_circular_orbit_period_matches_kepler() -> None:
    """The mean motion follows from the gravitational parameter and the radius."""
    radius = EARTH_RADIUS + 550.0e3
    orbit = CircularOrbit(radius=radius, inclination=np.deg2rad(51.6))
    expected = 2.0 * np.pi * np.sqrt(radius**3 / GRAVITATIONAL_PARAMETER)
    assert orbit.period == pytest.approx(expected, rel=1e-14)
    assert float(np.linalg.norm(orbit.position(1234.0))) == pytest.approx(radius, rel=1e-14)


def test_pyramid_array_is_isotropic() -> None:
    """At the isotropic half angle the array has the same torque gain about every axis.

    ``W W^T = (4/3) I`` exactly follows from ``tan^2(b) = 2``, so this is a closed
    form check rather than a recorded number.
    """
    axes = pyramid_wheel_axes()
    assert np.allclose(axes @ axes.T, (4.0 / 3.0) * np.eye(3), atol=64.0 * EPSILON)
    assert pytest.approx(np.arctan(np.sqrt(2.0)), rel=1e-15) == ISOTROPIC_PYRAMID_HALF_ANGLE


def test_inertia_validation_rejects_impossible_tensors() -> None:
    """A tensor that no mass distribution can produce is refused."""
    with pytest.raises(ValueError, match="positive definite"):
        symmetric_inertia((1.0, 1.0, -1.0))
    with pytest.raises(ValueError, match="triangle inequality"):
        symmetric_inertia((1.0, 1.0, 5.0))
    moments = principal_inertia(symmetric_inertia((90.0, 100.0, 75.0), (5.0, -3.0, 2.0)))
    assert moments[0] < moments[1] < moments[2]


def test_wheel_array_validation() -> None:
    """The wheel array refuses geometry that cannot control three axes."""
    with pytest.raises(ValueError, match="unit vector"):
        WheelArray(np.eye(3) * 2.0, np.ones(3), 0.05, 4.0)
    with pytest.raises(ValueError, match="three dimensions"):
        WheelArray(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]), np.ones(2), 0.05, 4.0)
    with pytest.raises(ValueError, match="axial inertia"):
        WheelArray(pyramid_wheel_axes(), np.array([1.0, 1.0, 1.0, -1.0]), 0.05, 4.0)
