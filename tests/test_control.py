"""Tier one: closed loop properties of the two controllers."""

from __future__ import annotations

import numpy as np
import pytest

from attitude_control.algorithm.controller import (
    AttitudeController,
    ConstantTorque,
    ControlSignal,
    LinearQuadraticRegulator,
    QuaternionFeedbackPD,
    error_state,
)
from attitude_control.analysis.metrics import ManoeuvreMetrics, signed_error_angle
from attitude_control.configuration import (
    LQR_WEIGHTS,
    PD_DAMPING_RATIO,
    PD_NATURAL_FREQUENCY,
    controllers,
)
from attitude_control.model.attitude import (
    quaternion_from_axis_angle,
    quaternion_identity,
)
from attitude_control.model.inertia import Spacecraft
from attitude_control.pipeline.scenario import ScenarioConfig, run_scenario
from tests.conftest import EPSILON


def make_signal(
    quaternion: np.ndarray,
    body_rate: np.ndarray,
    wheel_momentum: np.ndarray | None = None,
    commanded: np.ndarray | None = None,
) -> ControlSignal:
    """Build a control signal with sensible defaults for the unused channels."""
    return ControlSignal(
        time=0.0,
        quaternion=quaternion,
        body_rate=body_rate,
        wheel_momentum=np.zeros(4) if wheel_momentum is None else wheel_momentum,
        commanded_quaternion=quaternion_identity() if commanded is None else commanded,
        commanded_rate=np.zeros(3),
    )


def test_both_controllers_satisfy_the_protocol(spacecraft: Spacecraft) -> None:
    """The shared protocol is what makes the two designs comparable."""
    for controller in controllers(spacecraft):
        assert isinstance(controller, AttitudeController)
        assert isinstance(controller.name, str) and controller.name
        torque = controller.body_torque(make_signal(quaternion_identity(), np.zeros(3)))
        assert torque.shape == (3,)


def test_both_controllers_command_nothing_at_the_target(spacecraft: Spacecraft) -> None:
    """At rest at the commanded attitude the control effort is exactly zero."""
    for controller in controllers(spacecraft):
        torque = controller.body_torque(make_signal(quaternion_identity(), np.zeros(3)))
        assert np.allclose(torque, np.zeros(3), atol=0.0)


def test_pd_gains_have_the_documented_structure(spacecraft: Spacecraft) -> None:
    """``K = 2 wn^2 J`` and ``P = 2 zeta wn J``, which is what makes wn and zeta meaningful."""
    controller = QuaternionFeedbackPD(spacecraft, PD_NATURAL_FREQUENCY, PD_DAMPING_RATIO)
    assert np.allclose(
        controller.proportional_gain,
        2.0 * PD_NATURAL_FREQUENCY**2 * spacecraft.inertia,
        atol=64.0 * EPSILON,
    )
    assert np.allclose(
        controller.derivative_gain,
        2.0 * PD_DAMPING_RATIO * PD_NATURAL_FREQUENCY * spacecraft.inertia,
        atol=64.0 * EPSILON,
    )


def test_lqr_attitude_gain_has_a_closed_form(spacecraft: Spacecraft) -> None:
    """The LQR attitude gain block is ``sqrt(q_att / r) I`` for any inertia tensor.

    Substituting ``S12 = sqrt(r q_att) J`` into the attitude block of the Riccati
    equation gives ``S12 J^-2 S12 = r q_att I``, which holds identically, and then
    ``K_attitude = r^-1 J^-1 S12 = sqrt(q_att / r) I``. This is a closed form
    result, independent of the inertia and of the solver, so it is checked against
    the formula rather than against a recorded number.
    """
    controller = LinearQuadraticRegulator(
        spacecraft, LQR_WEIGHTS.attitude, LQR_WEIGHTS.rate, LQR_WEIGHTS.torque
    )
    expected = np.sqrt(LQR_WEIGHTS.attitude / LQR_WEIGHTS.torque)
    assert expected == pytest.approx(0.04, rel=1e-15)
    assert np.allclose(controller.gain[:, :3], expected * np.eye(3), atol=1e-12)


def test_lqr_closed_loop_is_stable_and_damped(spacecraft: Spacecraft) -> None:
    """Every closed loop pole is in the left half plane with the expected damping.

    With no rate weight the symmetric root locus of a double integrator places
    every pole at 45 degrees, giving a damping ratio of ``1 / sqrt(2)``. A small
    rate weight moves it very little, so the check is stated as a bound rather
    than an equality.
    """
    controller = LinearQuadraticRegulator(
        spacecraft, LQR_WEIGHTS.attitude, LQR_WEIGHTS.rate, LQR_WEIGHTS.torque
    )
    poles = controller.closed_loop_poles
    assert np.all(poles.real < 0.0)
    damping = -poles.real / np.abs(poles)
    assert np.all(damping >= 1.0 / np.sqrt(2.0) - 1e-9)
    slowest = float(np.min(np.abs(poles)))
    heaviest = float(np.max(np.linalg.eigvalsh(spacecraft.inertia)))
    predicted = np.sqrt(np.sqrt(LQR_WEIGHTS.attitude / LQR_WEIGHTS.torque) / heaviest)
    assert slowest == pytest.approx(predicted, rel=1e-9)


def test_error_state_takes_the_short_way_round(spacecraft: Spacecraft) -> None:
    """A 190 degree error is treated as a 170 degree error the other way.

    Without the canonical sign choice the feedback would drive the long way round
    for every error beyond 180 degrees. The vector part of the error quaternion is
    what changes sign, so the commanded torque flips with it.
    """
    axis = np.array([0.0, 0.0, 1.0])
    long_way = quaternion_from_axis_angle(axis, np.deg2rad(190.0))
    short_way = quaternion_from_axis_angle(axis, np.deg2rad(-170.0))
    for controller in controllers(spacecraft):
        first = controller.body_torque(make_signal(long_way, np.zeros(3)))
        second = controller.body_torque(make_signal(short_way, np.zeros(3)))
        assert np.allclose(first, second, atol=1e-12)
        assert float(np.dot(first, axis)) > 0.0


def test_error_state_transports_the_commanded_rate() -> None:
    """A commanded rate is compared in the body frame, not the commanded frame."""
    commanded = quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.4)
    signal = ControlSignal(
        time=0.0,
        quaternion=commanded,
        body_rate=np.array([0.0, 0.0, 0.02]),
        wheel_momentum=np.zeros(4),
        commanded_quaternion=commanded,
        commanded_rate=np.array([0.0, 0.0, 0.02]),
    )
    vector, rate = error_state(signal)
    assert np.allclose(vector, np.zeros(3), atol=64.0 * EPSILON)
    assert np.allclose(rate, np.zeros(3), atol=64.0 * EPSILON)


@pytest.mark.parametrize("angle_deg", [100.0, -100.0])
def test_both_controllers_null_a_large_slew(spacecraft: Spacecraft, angle_deg: float) -> None:
    """A large slew reaches the commanded attitude with no steady state error.

    Tolerance: the closed loop envelope decays as ``exp(-zeta wn t)``, so after
    900 s at ``wn = 0.02`` and ``zeta = 1/sqrt(2)`` the residual is at most
    ``exp(-12.73)`` of the initial 100 degrees, that is 2.9e-4 degrees. The bound
    below is three times that. Neither controller has an integral term, and no
    disturbance acts, so nothing sets a floor above it. The slew is sized so the
    torque demand stays inside the wheel envelope, which the test asserts, so the
    linear argument for the bound applies without qualification.
    """
    duration = 900.0
    predicted = abs(angle_deg) * np.exp(-PD_DAMPING_RATIO * PD_NATURAL_FREQUENCY * duration)
    for controller in controllers(spacecraft):
        config = ScenarioConfig(
            spacecraft=spacecraft,
            controller=controller,
            duration=duration,
            time_step=0.5,
            initial_quaternion=quaternion_from_axis_angle((1.0, 2.0, 2.0), np.deg2rad(angle_deg)),
            commanded_quaternion=quaternion_identity(),
            sample_stride=10,
        )
        trace = run_scenario(config)
        metrics = ManoeuvreMetrics.evaluate(trace, spacecraft)
        assert metrics.initial_error_deg == pytest.approx(abs(angle_deg), abs=1e-9)
        assert metrics.final_error_deg < 3.0 * predicted
        assert float(np.max(np.abs(trace.body_rate[-1]))) < 1e-6
        assert not trace.saturated.any()


def test_a_saturated_slew_still_reaches_the_target(spacecraft: Spacecraft) -> None:
    """Torque saturation slows the response but does not stop it converging.

    The gain here demands more than the wheels can deliver over the first part of
    the manoeuvre, so the closed loop is genuinely nonlinear and the linear
    residual bound does not apply. The assertion is therefore the qualitative one
    that matters: the array does saturate, and the error still reaches a value far
    below the 0.1 degree settling threshold.
    """
    controller = QuaternionFeedbackPD(spacecraft, 0.05, PD_DAMPING_RATIO)
    config = ScenarioConfig(
        spacecraft=spacecraft,
        controller=controller,
        duration=600.0,
        time_step=0.2,
        initial_quaternion=quaternion_from_axis_angle((1.0, 2.0, 2.0), np.deg2rad(170.0)),
        commanded_quaternion=quaternion_identity(),
        sample_stride=10,
    )
    trace = run_scenario(config)
    assert trace.saturated.any()
    assert float(np.max(np.abs(trace.wheel_torque))) <= spacecraft.wheels.max_torque + 1e-15
    assert ManoeuvreMetrics.evaluate(trace, spacecraft).final_error_deg < 1e-3


def test_small_step_matches_the_linear_prediction(spacecraft: Spacecraft) -> None:
    """The small angle response reproduces the analytic second order step response.

    With the gyroscopic feedforward active the closed loop is exactly
    ``theta_ddot + 2 zeta wn theta_dot + wn^2 theta = 0`` up to the replacement of
    ``2 sin(theta/2)`` by ``theta``, whose relative error is ``theta^2 / 24``.
    Tolerance: with an initial error of 0.5 degrees that modelling error is
    3.2e-6 of the initial amplitude, and the RK4 contribution at a step of 0.05 s
    is smaller by six orders. The bound below is ten times the modelling error,
    which is where the deviation must sit if the implementation is right.
    """
    initial = np.deg2rad(0.5)
    natural, damping = PD_NATURAL_FREQUENCY, PD_DAMPING_RATIO
    controller = QuaternionFeedbackPD(spacecraft, natural, damping)
    config = ScenarioConfig(
        spacecraft=spacecraft,
        controller=controller,
        duration=400.0,
        time_step=0.05,
        initial_quaternion=quaternion_from_axis_angle((1.0, 2.0, 2.0), initial),
        commanded_quaternion=quaternion_identity(),
        sample_stride=20,
    )
    trace = run_scenario(config)

    damped = natural * np.sqrt(1.0 - damping**2)
    time = trace.time
    predicted = (
        initial
        * np.exp(-damping * natural * time)
        * (np.cos(damped * time) + damping / np.sqrt(1.0 - damping**2) * np.sin(damped * time))
    )
    observed = signed_error_angle(trace)
    tolerance = 10.0 * initial * initial**2 / 24.0
    assert float(np.max(np.abs(observed - predicted))) < tolerance


def test_overshoot_matches_the_second_order_value(spacecraft: Spacecraft) -> None:
    """The measured overshoot equals the analytic value for the chosen damping.

    For a second order system released from a displacement with zero rate the
    first undershoot past zero is ``exp(-pi zeta / sqrt(1 - zeta^2))``, which is
    4.32 per cent at ``zeta = 1/sqrt(2)``. Tolerance: the sample stride of 1 s
    against a damped period of 222 s resolves the extremum to
    ``(pi dt / T)^2 / 2 = 1e-4`` of its value, and the small angle modelling error
    adds 3.2e-6, so 0.01 percentage points is a comfortable bound.
    """
    initial = np.deg2rad(0.5)
    controller = QuaternionFeedbackPD(spacecraft, PD_NATURAL_FREQUENCY, PD_DAMPING_RATIO)
    config = ScenarioConfig(
        spacecraft=spacecraft,
        controller=controller,
        duration=400.0,
        time_step=0.1,
        initial_quaternion=quaternion_from_axis_angle((1.0, 2.0, 2.0), initial),
        commanded_quaternion=quaternion_identity(),
        sample_stride=10,
    )
    metrics = ManoeuvreMetrics.evaluate(run_scenario(config), spacecraft)
    analytic = 100.0 * np.exp(
        -np.pi * PD_DAMPING_RATIO / np.sqrt(1.0 - PD_DAMPING_RATIO**2)
    )
    assert analytic == pytest.approx(4.3214, abs=1e-4)
    assert metrics.overshoot_percent == pytest.approx(analytic, abs=0.01)


def test_feedforward_cancels_the_gyroscopic_term(spacecraft: Spacecraft) -> None:
    """With feedforward on, the commanded torque differs from the plain law by ``w x h``."""
    with_feedforward = QuaternionFeedbackPD(spacecraft, 0.02, 0.7, feedforward=True)
    without = QuaternionFeedbackPD(spacecraft, 0.02, 0.7, feedforward=False)
    signal = make_signal(
        quaternion_from_axis_angle((1.0, 0.0, 1.0), 0.6),
        np.array([0.01, -0.02, 0.03]),
        wheel_momentum=np.array([0.4, -0.2, 0.1, 0.3]),
    )
    difference = with_feedforward.body_torque(signal) - without.body_torque(signal)
    from attitude_control.model.dynamics import gyroscopic_torque

    assert np.allclose(
        difference,
        gyroscopic_torque(spacecraft, signal.body_rate, signal.wheel_momentum),
        atol=64.0 * EPSILON,
    )


def test_controller_construction_rejects_bad_parameters(spacecraft: Spacecraft) -> None:
    """Gains and weights that cannot produce a stabilising design are refused."""
    with pytest.raises(ValueError, match="natural frequency"):
        QuaternionFeedbackPD(spacecraft, -1.0, 0.7)
    with pytest.raises(ValueError, match="damping ratio"):
        QuaternionFeedbackPD(spacecraft, 0.02, 0.0)
    with pytest.raises(ValueError, match="torque_weight"):
        LinearQuadraticRegulator(spacecraft, 1.0, 1.0, 0.0)


def test_constant_torque_controller_is_open_loop() -> None:
    """The constant torque law ignores the state, which is what makes it open loop."""
    controller = ConstantTorque.create((0.1, -0.2, 0.3), label="open loop")
    assert controller.name == "open loop"
    first = controller.body_torque(make_signal(quaternion_identity(), np.zeros(3)))
    second = controller.body_torque(
        make_signal(quaternion_from_axis_angle((1.0, 1.0, 0.0), 2.0), np.ones(3))
    )
    assert np.array_equal(first, second)
