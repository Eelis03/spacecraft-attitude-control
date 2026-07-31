"""Attitude controllers behind one shared protocol.

Both controllers produce a *body torque command* in N m from the same
:class:`ControlSignal`, so a scenario can be run twice with only the controller
swapped and the two responses compared without any other difference.

Quaternion feedback PD
----------------------
Wie and Barba (1985) and Wie, Weiss and Arapostathis (1989) give the control law

    L = -K dq_v - P w

where ``dq_v`` is the vector part of the error quaternion in canonical form. With
the Lyapunov function ``V = w^T J w + 2 k (dq_v . dq_v + (1 - dq_0)^2)`` the
derivative along the closed loop trajectories is ``-2 w^T P w``, which is negative
semidefinite for any positive definite ``P``, and LaSalle's theorem then gives
global asymptotic stability for ``K = k I`` and ``P`` positive definite. The proof
does not depend on the inertia tensor, which is why the law needs no model.

The gain structure used here is the inertia-scaled one,

    K = 2 wn^2 J,    P = 2 zeta wn J

Substituting into ``J w_dot = -K dq_v - P w`` with the small angle relations
``dq_v ~ theta / 2`` and ``theta_dot = w`` gives, for every axis independently,

    theta_ddot + 2 zeta wn theta_dot + wn^2 theta = 0

so the two gains are exactly the natural frequency and damping ratio of the
closed loop. That is the property the linear prediction test checks.

Linear quadratic regulator
--------------------------
The attitude dynamics linearised about the commanded attitude and zero rate are

    theta_dot = w,    J w_dot = L

so with the state ``x = (theta, w)`` and input ``L``,

    A = [[0, I], [0, 0]],    B = [[0], [J^-1]]

The pair is controllable for any invertible ``J``. Minimising the quadratic cost
``integral(x^T Q x + L^T R L) dt`` gives ``L = -R^-1 B^T S x`` with ``S`` the
stabilising solution of the continuous algebraic Riccati equation
``A^T S + S A - S B R^-1 B^T S + Q = 0``; see Kalman (1960). Because ``J`` has
products of inertia the resulting gain is not diagonal, so the LQR couples the
axes where the PD law does not.

Both controllers accept an optional feedforward of the gyroscopic term
``w x (J w + W h_w)``. When it is enabled the closed loop is exactly the linear
system above rather than an approximation of it.

References
----------
Wie, B. and Barba, P. M. (1985). Quaternion feedback for spacecraft large angle
manoeuvres. *Journal of Guidance, Control, and Dynamics*, 8(3), 360-365. DOI
10.2514/3.19988.

Wie, B., Weiss, H. and Arapostathis, A. (1989). Quaternion feedback regulator for
spacecraft eigenaxis rotations. *Journal of Guidance, Control, and Dynamics*,
12(3), 375-380. DOI 10.2514/3.20418.

Kalman, R. E. (1960). Contributions to the theory of optimal control. *Boletin de
la Sociedad Matematica Mexicana*, 5, 102-119.
https://liberzon.csl.illinois.edu/teaching/kalman_optimal_control.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_continuous_are

from attitude_control.model.attitude import attitude_error, dcm_from_quaternion
from attitude_control.model.dynamics import gyroscopic_torque
from attitude_control.model.inertia import Spacecraft
from attitude_control.numeric import ComplexArray, FloatArray, as_matrix, as_vector

__all__ = [
    "AttitudeController",
    "ConstantTorque",
    "ControlSignal",
    "LinearQuadraticRegulator",
    "QuaternionFeedbackPD",
    "error_state",
]


@dataclass(frozen=True, slots=True)
class ControlSignal:
    """Everything a controller is allowed to see at one instant."""

    time: float
    quaternion: FloatArray
    body_rate: FloatArray
    wheel_momentum: FloatArray
    commanded_quaternion: FloatArray
    commanded_rate: FloatArray


@runtime_checkable
class AttitudeController(Protocol):
    """A control law mapping the measured and commanded state to a body torque."""

    @property
    def name(self) -> str:
        """Short label used in reports and figures."""
        ...

    def body_torque(self, signal: ControlSignal) -> FloatArray:
        """Return the commanded body torque in N m."""
        ...


def error_state(signal: ControlSignal) -> tuple[FloatArray, FloatArray]:
    """Return the error quaternion vector part and the error body rate.

    The error rate is the body rate minus the commanded rate transported into the
    body frame, which is the correct comparison when the commanded attitude is
    itself moving. The error quaternion is canonical, so the feedback always takes
    the short way round; this is where the quaternion sign ambiguity is resolved.
    """
    error = attitude_error(signal.quaternion, signal.commanded_quaternion)
    transported = dcm_from_quaternion(error) @ signal.commanded_rate
    return error[1:], signal.body_rate - transported


@dataclass(frozen=True, slots=True)
class QuaternionFeedbackPD:
    """Quaternion feedback proportional plus derivative regulator.

    Parameters
    ----------
    spacecraft:
        Supplies the inertia tensor used to scale the gains and, when
        ``feedforward`` is set, the gyroscopic term to cancel.
    natural_frequency:
        Closed loop natural frequency in rad/s.
    damping_ratio:
        Closed loop damping ratio; 1/sqrt(2) gives about 4.3 per cent overshoot.
    feedforward:
        Cancel ``w x (J w + W h_w)`` when True.
    """

    spacecraft: Spacecraft
    natural_frequency: float
    damping_ratio: float
    feedforward: bool = True
    label: str = "quaternion PD"
    proportional_gain: FloatArray = field(init=False)
    derivative_gain: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        if self.natural_frequency <= 0.0:
            raise ValueError("natural frequency must be positive")
        if self.damping_ratio <= 0.0:
            raise ValueError("damping ratio must be positive")
        inertia = self.spacecraft.inertia
        object.__setattr__(
            self, "proportional_gain", 2.0 * self.natural_frequency**2 * inertia
        )
        object.__setattr__(
            self,
            "derivative_gain",
            2.0 * self.damping_ratio * self.natural_frequency * inertia,
        )

    @property
    def name(self) -> str:
        """Short label used in reports and figures."""
        return self.label

    def body_torque(self, signal: ControlSignal) -> FloatArray:
        """Return the commanded body torque in N m."""
        error_vector, error_rate = error_state(signal)
        torque = -(self.proportional_gain @ error_vector) - self.derivative_gain @ error_rate
        if self.feedforward:
            torque = torque + gyroscopic_torque(
                self.spacecraft, signal.body_rate, signal.wheel_momentum
            )
        return torque


@dataclass(frozen=True, slots=True)
class LinearQuadraticRegulator:
    """Infinite horizon LQR on the linearised attitude dynamics.

    Parameters
    ----------
    spacecraft:
        Supplies the inertia tensor that enters the input matrix.
    attitude_weight:
        Weight on the small angle attitude error, per rad^2.
    rate_weight:
        Weight on the body rate error, per (rad/s)^2.
    torque_weight:
        Weight on the commanded torque, per (N m)^2.
    feedforward:
        Cancel ``w x (J w + W h_w)`` when True.

    The attitude error fed to the gain is ``2 dq_v``, which equals the principal
    rotation vector to first order and stays bounded by two for any rotation, so
    the law degrades gracefully rather than diverging outside the linear region.
    """

    spacecraft: Spacecraft
    attitude_weight: float = 1.0
    rate_weight: float = 1.0
    torque_weight: float = 1.0
    feedforward: bool = True
    label: str = "LQR"
    gain: FloatArray = field(init=False)
    riccati_solution: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("attitude_weight", self.attitude_weight),
            ("rate_weight", self.rate_weight),
            ("torque_weight", self.torque_weight),
        ):
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
        dynamics = np.zeros((6, 6), dtype=np.float64)
        dynamics[:3, 3:] = np.eye(3)
        input_matrix = np.zeros((6, 3), dtype=np.float64)
        input_matrix[3:, :] = self.spacecraft.inertia_inverse
        state_cost = np.diag(
            np.concatenate(
                (
                    np.full(3, self.attitude_weight),
                    np.full(3, self.rate_weight),
                )
            )
        )
        input_cost = self.torque_weight * np.eye(3)
        solution = as_matrix(
            solve_continuous_are(dynamics, input_matrix, state_cost, input_cost), (6, 6)
        )
        object.__setattr__(self, "riccati_solution", solution)
        object.__setattr__(
            self, "gain", np.linalg.solve(input_cost, input_matrix.T @ solution)
        )

    @property
    def name(self) -> str:
        """Short label used in reports and figures."""
        return self.label

    @property
    def closed_loop_poles(self) -> ComplexArray:
        """Eigenvalues of the closed loop linearised dynamics, in rad/s."""
        dynamics = np.zeros((6, 6), dtype=np.float64)
        dynamics[:3, 3:] = np.eye(3)
        input_matrix = np.zeros((6, 3), dtype=np.float64)
        input_matrix[3:, :] = self.spacecraft.inertia_inverse
        eigenvalues = np.linalg.eigvals(dynamics - input_matrix @ self.gain)
        return np.asarray(eigenvalues, dtype=np.complex128)

    def body_torque(self, signal: ControlSignal) -> FloatArray:
        """Return the commanded body torque in N m."""
        error_vector, error_rate = error_state(signal)
        state = np.concatenate((2.0 * error_vector, error_rate))
        torque = -(self.gain @ state)
        if self.feedforward:
            torque = torque + gyroscopic_torque(
                self.spacecraft, signal.body_rate, signal.wheel_momentum
            )
        return torque


@dataclass(frozen=True, slots=True)
class ConstantTorque:
    """A controller that always commands the same body torque.

    Present so that open loop plant behaviour, including the torque free cases
    used in the invariant tests, can be driven through the same scenario runner
    as the closed loop cases.
    """

    torque: FloatArray
    label: str = "constant torque"

    @classmethod
    def create(cls, torque: ArrayLike, label: str = "constant torque") -> ConstantTorque:
        """Build a constant torque controller from any array-like torque."""
        return cls(torque=as_vector(torque, 3), label=label)

    @property
    def name(self) -> str:
        """Short label used in reports and figures."""
        return self.label

    def body_torque(self, signal: ControlSignal) -> FloatArray:
        """Return the commanded body torque in N m."""
        del signal
        return self.torque.copy()
