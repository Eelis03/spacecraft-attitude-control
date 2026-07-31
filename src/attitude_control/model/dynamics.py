"""Rigid body rotational dynamics with reaction wheels.

State
-----
The plant state is the attitude quaternion ``q``, the body angular rate ``w`` in
body components, and the stored wheel momentum ``h_w``, one scalar per wheel,
measured about that wheel's spin axis and *relative to the body*.

Momentum bookkeeping
--------------------
With ``J`` the inertia tensor of the whole vehicle including the wheels, ``W`` the
``3 x n`` matrix of wheel spin axes, and ``h_w`` the relative wheel momentum, the
total angular momentum of the system in body components is

    h = J w + W h_w

and in inertial components it is ``A(q)^T h``. Because the wheels are axisymmetric
their spin does not alter ``J``, so no other term appears. Euler's equation
``h_dot_inertial = L_external`` becomes, in the rotating body frame,

    J w_dot + W u + w x (J w + W h_w) = L_external

where ``u = h_w_dot`` is the vector of motor torques applied to the wheels. The
torque delivered *to the body* is therefore ``-W u``: spinning a wheel up pushes
the body the other way. Every sign in this package follows from that line.

References
----------
Hughes, P. C. (2004). *Spacecraft Attitude Dynamics*. Dover. ISBN 978-0486439259.

Markley, F. L. and Crassidis, J. L. (2014). *Fundamentals of Spacecraft Attitude
Determination and Control*. Springer, chapter 7. DOI 10.1007/978-1-4939-0802-8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.model.attitude import dcm_from_quaternion, quaternion_derivative
from attitude_control.model.inertia import Spacecraft
from attitude_control.numeric import FloatArray, as_vector

__all__ = [
    "PlantState",
    "body_angular_momentum",
    "body_rate_derivative",
    "body_torque_from_wheel_torque",
    "gyroscopic_torque",
    "inertial_angular_momentum",
    "rotational_kinetic_energy",
    "state_derivative",
]


@dataclass(frozen=True, slots=True)
class PlantState:
    """Attitude, body rate, and stored wheel momentum at one instant."""

    quaternion: FloatArray
    body_rate: FloatArray
    wheel_momentum: FloatArray

    @classmethod
    def create(
        cls,
        quaternion: ArrayLike,
        body_rate: ArrayLike,
        wheel_momentum: ArrayLike,
    ) -> PlantState:
        """Build a state from any array-like inputs, validating the shapes."""
        wheels = np.asarray(wheel_momentum, dtype=np.float64).reshape(-1)
        return cls(
            quaternion=as_vector(quaternion, 4),
            body_rate=as_vector(body_rate, 3),
            wheel_momentum=wheels,
        )

    def flatten(self) -> FloatArray:
        """Pack the state into one vector for the integrator."""
        return np.concatenate((self.quaternion, self.body_rate, self.wheel_momentum))

    @classmethod
    def unflatten(cls, values: ArrayLike) -> PlantState:
        """Unpack a state vector produced by :meth:`flatten`."""
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size < 7:
            raise ValueError(f"state vector needs at least 7 entries, got {array.size}")
        return cls(quaternion=array[:4], body_rate=array[4:7], wheel_momentum=array[7:])


def body_angular_momentum(
    spacecraft: Spacecraft,
    body_rate: ArrayLike,
    wheel_momentum: ArrayLike,
) -> FloatArray:
    """Return the total system angular momentum in body components, ``J w + W h_w``."""
    rate = as_vector(body_rate, 3)
    wheels = as_vector(wheel_momentum, spacecraft.wheels.count)
    return spacecraft.inertia @ rate + spacecraft.wheels.axes @ wheels


def inertial_angular_momentum(
    spacecraft: Spacecraft,
    quaternion: ArrayLike,
    body_rate: ArrayLike,
    wheel_momentum: ArrayLike,
) -> FloatArray:
    """Return the total system angular momentum in inertial components.

    This quantity is constant whenever no external torque acts, whatever the
    wheels are doing, which is the strongest single check on the wheel model.
    """
    momentum = body_angular_momentum(spacecraft, body_rate, wheel_momentum)
    return dcm_from_quaternion(quaternion).T @ momentum


def rotational_kinetic_energy(
    spacecraft: Spacecraft,
    body_rate: ArrayLike,
    wheel_momentum: ArrayLike,
) -> float:
    """Return the total rotational kinetic energy of body and wheels.

    The wheel term is written in terms of the relative momentum as
    ``h_w . (W^T w) + h_w^2 / (2 I_w)``, which is the cross term plus the energy
    of the relative spin. For a body with no wheel momentum this reduces to the
    familiar ``0.5 w^T J w``.
    """
    rate = as_vector(body_rate, 3)
    wheels = as_vector(wheel_momentum, spacecraft.wheels.count)
    body = 0.5 * float(rate @ spacecraft.inertia @ rate)
    cross = float(wheels @ (spacecraft.wheels.axes.T @ rate))
    spin = 0.5 * float(np.sum(wheels * wheels / spacecraft.wheels.axial_inertia))
    return body + cross + spin


def gyroscopic_torque(
    spacecraft: Spacecraft,
    body_rate: ArrayLike,
    wheel_momentum: ArrayLike,
) -> FloatArray:
    """Return ``w x (J w + W h_w)``, the term that couples the body axes.

    Feeding the negative of this back as a control term cancels the coupling
    exactly and leaves a linear plant, which is why both controllers in this
    package offer it as a feedforward option.
    """
    rate = as_vector(body_rate, 3)
    return np.cross(rate, body_angular_momentum(spacecraft, rate, wheel_momentum))


def body_torque_from_wheel_torque(spacecraft: Spacecraft, wheel_torque: ArrayLike) -> FloatArray:
    """Return the body torque ``-W u`` produced by motor torques ``u`` on the wheels."""
    torque = as_vector(wheel_torque, spacecraft.wheels.count)
    return -(spacecraft.wheels.axes @ torque)


def body_rate_derivative(
    spacecraft: Spacecraft,
    body_rate: ArrayLike,
    wheel_momentum: ArrayLike,
    wheel_torque: ArrayLike,
    external_torque: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Return ``w_dot`` from Euler's equation with wheels and an external torque."""
    rate = as_vector(body_rate, 3)
    external = as_vector(external_torque, 3)
    applied = (
        external
        - gyroscopic_torque(spacecraft, rate, wheel_momentum)
        + body_torque_from_wheel_torque(spacecraft, wheel_torque)
    )
    return spacecraft.inertia_inverse @ applied


def state_derivative(
    spacecraft: Spacecraft,
    state: PlantState,
    wheel_torque: ArrayLike,
    external_torque: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Return the packed derivative of the plant state.

    The quaternion is propagated by its exact kinematic equation with no
    normalisation term added, so that the norm drift of the integrator remains
    visible and can be measured.
    """
    torque = as_vector(wheel_torque, spacecraft.wheels.count)
    return np.concatenate(
        (
            quaternion_derivative(state.quaternion, state.body_rate),
            body_rate_derivative(
                spacecraft,
                state.body_rate,
                state.wheel_momentum,
                torque,
                external_torque,
            ),
            torque,
        )
    )
