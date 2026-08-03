"""The reference vehicle and the three scenarios built on it.

This module is the single place where numbers are chosen. The example scripts and
the regression tests both import from here, so a scenario cannot drift between
what the documentation reports and what the tests pin.

Reference vehicle
-----------------
A 500 kg class small satellite in a 550 km orbit. The inertia tensor has non-zero
products of inertia, which is the interesting case: it makes the gyroscopic
coupling real and it makes the LQR gain matrix full. The wheel array is a four
wheel pyramid at the isotropic half angle, with 4 N m s and 0.05 N m per wheel,
representative of a small commercial wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from attitude_control.algorithm.controller import (
    AttitudeController,
    LinearQuadraticRegulator,
    QuaternionFeedbackPD,
    QuaternionFeedbackPID,
)
from attitude_control.algorithm.momentum import MagneticDumping
from attitude_control.model.attitude import quaternion_from_axis_angle, quaternion_identity
from attitude_control.model.environment import (
    EARTH_RADIUS,
    CircularOrbit,
    ConstantField,
    GravityGradient,
    TiltedDipoleField,
)
from attitude_control.model.inertia import (
    Spacecraft,
    WheelArray,
    pyramid_wheel_axes,
    symmetric_inertia,
)
from attitude_control.numeric import FloatArray
from attitude_control.pipeline.scenario import MagneticEnvironment, ScenarioConfig

__all__ = [
    "AGGRESSIVE_NATURAL_FREQUENCY",
    "DUMPING_GAIN",
    "DUMPING_MAX_DIPOLE",
    "INTEGRAL_FRACTION",
    "LQR_WEIGHTS",
    "NULL_SPACE_GAIN",
    "PD_DAMPING_RATIO",
    "PD_NATURAL_FREQUENCY",
    "SLEW_ANGLE",
    "SLEW_AXIS",
    "LqrWeights",
    "aggressive_controller",
    "constant_field_environment",
    "controllers",
    "disturbance_scenario",
    "dumping_scenario",
    "initial_wheel_momentum",
    "integral_controller",
    "orbiting_field_environment",
    "reference_orbit",
    "reference_spacecraft",
    "slew_scenario",
    "target_quaternion",
]

# Vehicle inertia in kg m^2, about the centre of mass in body axes.
_INERTIA_DIAGONAL: Final[tuple[float, float, float]] = (90.0, 100.0, 75.0)
_INERTIA_PRODUCTS: Final[tuple[float, float, float]] = (5.0, -3.0, 2.0)

# Reaction wheel parameters. 4.0 N m s at 0.0064 kg m^2 is 625 rad/s, or 5968 rpm.
_WHEEL_AXIAL_INERTIA: Final[float] = 0.0064
_WHEEL_MAX_TORQUE: Final[float] = 0.05
_WHEEL_MAX_MOMENTUM: Final[float] = 4.0

# Orbit: 550 km altitude, 51.6 degree inclination.
_ORBIT_ALTITUDE: Final[float] = 550.0e3
_ORBIT_INCLINATION: Final[float] = np.deg2rad(51.6)

# Quaternion PD gains, chosen so the peak commanded torque stays inside the wheel
# array envelope for the reference slew.
PD_NATURAL_FREQUENCY: Final[float] = 0.02
PD_DAMPING_RATIO: Final[float] = float(1.0 / np.sqrt(2.0))

# Reference slew: 60 degrees about an axis that is not a principal axis, so all
# three body axes are exercised and the gyroscopic coupling is active.
SLEW_ANGLE: Final[float] = np.deg2rad(60.0)
SLEW_AXIS: Final[tuple[float, float, float]] = (1.0, 2.0, 2.0)

# Magnetic dumping gain in 1/s, giving a 5000 s time constant on the
# controllable part of the stored momentum, and a 30 A m^2 rod limit.
DUMPING_GAIN: Final[float] = 2.0e-4
DUMPING_MAX_DIPOLE: Final[float] = 30.0

# Null space steering gain in 1/s for the redundant wheel array.
NULL_SPACE_GAIN: Final[float] = 0.01


@dataclass(frozen=True, slots=True)
class LqrWeights:
    """State and input weights for the linear quadratic regulator."""

    attitude: float
    rate: float
    torque: float


# For this plant the LQR attitude gain is exactly sqrt(attitude / torque) times
# the identity and the slowest closed loop mode has natural frequency
# sqrt(sqrt(attitude / torque) / J_max). The weights below put that slowest mode
# at 0.01979 rad/s, within about one per cent of the PD natural frequency, so the
# two designs are compared at matched bandwidth rather than merely differing.
LQR_WEIGHTS: Final[LqrWeights] = LqrWeights(attitude=1.0, rate=1.0, torque=625.0)

# A deliberately over-driven PD design used to exercise wheel torque saturation.
AGGRESSIVE_NATURAL_FREQUENCY: Final[float] = 0.05

# Integral gain of the PID design, as a fraction of the Routh-Hurwitz limit
# 2 zeta wn^3. A quarter of the limit places the integral pole at 0.0146 rad/s,
# a 68 s time constant, which is fast against the 5739 s orbit period, and costs
# damping: the dominant pair moves from zeta = 0.71 to zeta = 0.49. Anything
# closer to the limit rings.
INTEGRAL_FRACTION: Final[float] = 0.25


def reference_spacecraft() -> Spacecraft:
    """Return the reference vehicle with its four wheel pyramid."""
    wheels = WheelArray(
        axes=pyramid_wheel_axes(),
        axial_inertia=np.full(4, _WHEEL_AXIAL_INERTIA),
        max_torque=_WHEEL_MAX_TORQUE,
        max_momentum=_WHEEL_MAX_MOMENTUM,
    )
    return Spacecraft(
        inertia=symmetric_inertia(_INERTIA_DIAGONAL, _INERTIA_PRODUCTS),
        wheels=wheels,
    )


def reference_orbit() -> CircularOrbit:
    """Return the reference circular orbit."""
    return CircularOrbit(
        radius=EARTH_RADIUS + _ORBIT_ALTITUDE,
        inclination=_ORBIT_INCLINATION,
    )


def controllers(spacecraft: Spacecraft) -> tuple[AttitudeController, ...]:
    """Return the two controllers compared throughout this package."""
    return (
        QuaternionFeedbackPD(
            spacecraft=spacecraft,
            natural_frequency=PD_NATURAL_FREQUENCY,
            damping_ratio=PD_DAMPING_RATIO,
        ),
        LinearQuadraticRegulator(
            spacecraft=spacecraft,
            attitude_weight=LQR_WEIGHTS.attitude,
            rate_weight=LQR_WEIGHTS.rate,
            torque_weight=LQR_WEIGHTS.torque,
        ),
    )


def integral_controller(spacecraft: Spacecraft) -> QuaternionFeedbackPID:
    """Return the PD design of :func:`controllers` with integral action added.

    Kept out of :func:`controllers` on purpose. The two laws returned there are
    compared on the slew, where an integral term has nothing to do because no
    disturbance acts; this one exists for the disturbance rejection run, where
    the static offset is the thing being removed.
    """
    return QuaternionFeedbackPID(
        spacecraft=spacecraft,
        natural_frequency=PD_NATURAL_FREQUENCY,
        damping_ratio=PD_DAMPING_RATIO,
        integral_fraction=INTEGRAL_FRACTION,
    )


def aggressive_controller(spacecraft: Spacecraft) -> QuaternionFeedbackPD:
    """Return a PD design whose torque demand exceeds the wheel torque limit.

    Included so that the saturation model is exercised by the reported results
    and not only by the tests.
    """
    return QuaternionFeedbackPD(
        spacecraft=spacecraft,
        natural_frequency=AGGRESSIVE_NATURAL_FREQUENCY,
        damping_ratio=PD_DAMPING_RATIO,
        label="quaternion PD, over-driven",
    )


def target_quaternion() -> FloatArray:
    """Return the attitude the reference slew starts from.

    The manoeuvre regulates to the identity attitude, so the initial state is the
    commanded rotation applied in reverse.
    """
    return quaternion_from_axis_angle(SLEW_AXIS, SLEW_ANGLE)


def initial_wheel_momentum(spacecraft: Spacecraft, stored_body_momentum: FloatArray) -> FloatArray:
    """Return the wheel momentum whose body projection is ``stored_body_momentum``.

    The minimum norm choice is used, so the null space component is zero and the
    array starts as far from its speed limits as the request allows.
    """
    return spacecraft.wheels.pseudoinverse @ stored_body_momentum


def slew_scenario(
    spacecraft: Spacecraft,
    controller: AttitudeController,
    duration: float = 900.0,
    time_step: float = 0.2,
    sample_stride: int = 5,
) -> ScenarioConfig:
    """Return the reference rest to rest slew, regulating to the identity attitude."""
    return ScenarioConfig(
        spacecraft=spacecraft,
        controller=controller,
        duration=duration,
        time_step=time_step,
        initial_quaternion=target_quaternion(),
        initial_body_rate=np.zeros(3),
        commanded_quaternion=quaternion_identity(),
        null_space_gain=NULL_SPACE_GAIN,
        sample_stride=sample_stride,
    )


def disturbance_scenario(
    spacecraft: Spacecraft,
    controller: AttitudeController,
    orbits: float = 2.0,
    time_step: float = 1.0,
    sample_stride: int = 10,
) -> ScenarioConfig:
    """Return an inertial hold disturbed by gravity gradient torque.

    The commanded attitude is fixed in inertial space, so the nadir direction
    sweeps through the body frame once per orbit and the gravity gradient torque
    is periodic with a non-zero mean. The mean is what accumulates in the wheels.
    """
    orbit = reference_orbit()
    return ScenarioConfig(
        spacecraft=spacecraft,
        controller=controller,
        duration=orbits * orbit.period,
        time_step=time_step,
        initial_quaternion=quaternion_identity(),
        commanded_quaternion=quaternion_identity(),
        disturbance=GravityGradient(spacecraft=spacecraft, orbit=orbit),
        null_space_gain=NULL_SPACE_GAIN,
        sample_stride=sample_stride,
    )


def constant_field_environment(time: float = 0.0) -> ConstantField:
    """Return the reference dipole field frozen at ``time``, fixed in inertial space."""
    field = TiltedDipoleField(orbit=reference_orbit())
    return ConstantField(vector=field.field_inertial(time))


def dumping_scenario(
    spacecraft: Spacecraft,
    controller: AttitudeController,
    environment: MagneticEnvironment,
    stored_body_momentum: FloatArray | None = None,
    orbits: float = 3.0,
    time_step: float = 2.0,
    sample_stride: int = 5,
    label: str = "",
) -> ScenarioConfig:
    """Return an inertial hold that dumps stored wheel momentum magnetically.

    No environmental disturbance acts, so the only external torque is the one the
    torque rods produce. That isolates what magnetic actuation can and cannot do.
    """
    orbit = reference_orbit()
    stored = (
        np.array([1.5, -1.0, 2.0], dtype=np.float64)
        if stored_body_momentum is None
        else stored_body_momentum
    )
    return ScenarioConfig(
        spacecraft=spacecraft,
        controller=controller,
        duration=orbits * orbit.period,
        time_step=time_step,
        initial_quaternion=quaternion_identity(),
        initial_wheel_momentum=initial_wheel_momentum(spacecraft, stored),
        commanded_quaternion=quaternion_identity(),
        dumping=MagneticDumping(gain=DUMPING_GAIN, max_dipole=DUMPING_MAX_DIPOLE),
        magnetic_environment=environment,
        null_space_gain=NULL_SPACE_GAIN,
        sample_stride=sample_stride,
        label=label,
    )


def orbiting_field_environment() -> TiltedDipoleField:
    """Return the tilted dipole field sampled along the reference orbit."""
    return TiltedDipoleField(orbit=reference_orbit())
