"""Scenario configuration and the runner that produces a structured trace.

A scenario wires together a plant, a controller, an attitude command, an optional
environmental disturbance, and an optional magnetic momentum dumping law, then
integrates the closed loop and records every signal an analysis might want.

The control law is evaluated inside the derivative function rather than held
constant over a step. The closed loop is therefore a continuous time system
integrated by RK4, which is what makes the small angle response comparable with
the analytic second order prediction. Sampled data effects, computation delay,
and sensor noise are outside the scope of this package; see the design notes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.algorithm.allocation import Allocation, allocate
from attitude_control.algorithm.controller import AttitudeController, ControlSignal
from attitude_control.algorithm.momentum import MagneticDumping
from attitude_control.model.attitude import (
    attitude_error,
    principal_angle,
    quaternion_identity,
    quaternion_normalise,
)
from attitude_control.model.dynamics import (
    PlantState,
    inertial_angular_momentum,
    state_derivative,
)
from attitude_control.model.environment import magnetic_torque
from attitude_control.model.inertia import Spacecraft
from attitude_control.numeric import BoolArray, FloatArray, as_vector
from attitude_control.pipeline.integrator import normalise_quaternion_state, rk4_step

__all__ = [
    "DisturbanceModel",
    "MagneticEnvironment",
    "ScenarioConfig",
    "ScenarioTrace",
    "run_scenario",
]


class DisturbanceModel(Protocol):
    """An external torque acting on the body, in body components."""

    def torque(self, time: float, quaternion: ArrayLike) -> FloatArray:
        """Return the external torque in N m at ``time`` for attitude ``quaternion``."""
        ...


class MagneticEnvironment(Protocol):
    """A magnetic field model that can be sampled in body components."""

    def field_body(self, time: float, quaternion: ArrayLike) -> FloatArray:
        """Return the magnetic flux density in body components, in tesla."""
        ...


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Everything needed to run one closed loop simulation.

    Attributes
    ----------
    spacecraft:
        Plant model, including the wheel array.
    controller:
        Control law producing a body torque command.
    duration:
        Simulated time in seconds.
    time_step:
        Integration step in seconds.
    initial_quaternion, initial_body_rate, initial_wheel_momentum:
        Initial plant state. The wheel momentum defaults to zero on every wheel.
    commanded_quaternion, commanded_rate:
        Constant attitude command. A time varying command would enter here as a
        callable; the scenarios in this package all use a fixed target.
    disturbance:
        Optional environmental torque model.
    dumping, magnetic_environment:
        Optional magnetic momentum unloading. Both must be supplied together.
    null_space_gain, target_wheel_momentum:
        Redundancy resolution for the wheel array.
    enforce_wheel_limits:
        Apply the wheel torque and momentum limits when True.
    sample_stride:
        Record one sample every ``sample_stride`` steps, to keep traces small for
        long runs. The first and last steps are always recorded.
    label:
        Name recorded in the trace. Defaults to the controller name, and is set
        when several runs share a controller but differ in the environment.
    """

    spacecraft: Spacecraft
    controller: AttitudeController
    duration: float
    time_step: float
    initial_quaternion: FloatArray = field(default_factory=quaternion_identity)
    initial_body_rate: FloatArray = field(default_factory=lambda: np.zeros(3))
    initial_wheel_momentum: FloatArray | None = None
    commanded_quaternion: FloatArray = field(default_factory=quaternion_identity)
    commanded_rate: FloatArray = field(default_factory=lambda: np.zeros(3))
    disturbance: DisturbanceModel | None = None
    dumping: MagneticDumping | None = None
    magnetic_environment: MagneticEnvironment | None = None
    null_space_gain: float = 0.0
    target_wheel_momentum: FloatArray | None = None
    enforce_wheel_limits: bool = True
    sample_stride: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if self.time_step <= 0.0:
            raise ValueError("time step must be positive")
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.sample_stride < 1:
            raise ValueError("sample stride must be at least one")
        if (self.dumping is None) != (self.magnetic_environment is None):
            raise ValueError("magnetic dumping needs both a law and a field model")

    @property
    def steps(self) -> int:
        """Number of integration steps, rounded to the nearest whole step."""
        return max(1, round(self.duration / self.time_step))

    def initial_state(self) -> PlantState:
        """Return the packed initial plant state."""
        wheels = self.spacecraft.wheels.count
        momentum = (
            np.zeros(wheels)
            if self.initial_wheel_momentum is None
            else as_vector(self.initial_wheel_momentum, wheels)
        )
        return PlantState.create(
            quaternion_normalise(self.initial_quaternion),
            self.initial_body_rate,
            momentum,
        )


@dataclass(frozen=True, slots=True)
class ScenarioTrace:
    """Recorded signals from one scenario run.

    Every array has the sample count as its first axis. ``wheel_torque`` and the
    torque channels hold the value applied over the step that begins at the
    corresponding time.
    """

    name: str
    time: FloatArray
    quaternion: FloatArray
    body_rate: FloatArray
    wheel_momentum: FloatArray
    wheel_torque: FloatArray
    commanded_torque: FloatArray
    delivered_torque: FloatArray
    external_torque: FloatArray
    magnetic_field_body: FloatArray
    dipole: FloatArray
    stored_body_momentum: FloatArray
    inertial_momentum: FloatArray
    saturated: BoolArray
    commanded_quaternion: FloatArray

    @property
    def samples(self) -> int:
        """Number of recorded samples."""
        return int(self.time.size)

    def error_angle(self) -> FloatArray:
        """Return the principal angle of the attitude error at each sample, in rad."""
        return np.array(
            [
                principal_angle(attitude_error(q, self.commanded_quaternion))
                for q in self.quaternion
            ],
            dtype=np.float64,
        )

    def error_quaternion(self) -> FloatArray:
        """Return the canonical attitude error quaternion at each sample."""
        return np.array(
            [attitude_error(q, self.commanded_quaternion) for q in self.quaternion],
            dtype=np.float64,
        )

    def wheel_speed(self, spacecraft: Spacecraft) -> FloatArray:
        """Return the wheel speeds in rad/s at each sample."""
        return self.wheel_momentum / spacecraft.wheels.axial_inertia


def _stack(rows: Sequence[FloatArray]) -> FloatArray:
    return np.array(rows, dtype=np.float64)


def run_scenario(config: ScenarioConfig) -> ScenarioTrace:
    """Integrate the closed loop and return the recorded trace."""
    spacecraft = config.spacecraft
    wheels = spacecraft.wheels
    commanded_quaternion = quaternion_normalise(config.commanded_quaternion)
    commanded_rate = as_vector(config.commanded_rate, 3)
    step = config.time_step

    def environment(time: float, state: PlantState) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the external torque, the body field, and the commanded dipole."""
        external = np.zeros(3, dtype=np.float64)
        if config.disturbance is not None:
            external = external + config.disturbance.torque(time, state.quaternion)
        field_body = np.zeros(3, dtype=np.float64)
        dipole = np.zeros(3, dtype=np.float64)
        if config.dumping is not None and config.magnetic_environment is not None:
            field_body = config.magnetic_environment.field_body(time, state.quaternion)
            stored = wheels.axes @ state.wheel_momentum
            dipole = config.dumping.dipole(stored, field_body)
            external = external + magnetic_torque(dipole, field_body)
        return external, field_body, dipole

    def control(time: float, state: PlantState) -> tuple[FloatArray, Allocation]:
        """Return the commanded body torque and the wheel allocation that realises it."""
        signal = ControlSignal(
            time=time,
            quaternion=state.quaternion,
            body_rate=state.body_rate,
            wheel_momentum=state.wheel_momentum,
            commanded_quaternion=commanded_quaternion,
            commanded_rate=commanded_rate,
        )
        command = config.controller.body_torque(signal)
        allocation = allocate(
            wheels,
            command,
            state.wheel_momentum,
            null_space_gain=config.null_space_gain,
            target_momentum=config.target_wheel_momentum,
            enforce_limits=config.enforce_wheel_limits,
        )
        return command, allocation

    def derivative(time: float, packed: FloatArray) -> FloatArray:
        state = PlantState.unflatten(packed)
        external, _, _ = environment(time, state)
        _, allocation = control(time, state)
        return state_derivative(spacecraft, state, allocation.wheel_torque, external)

    state = config.initial_state()
    packed = state.flatten()

    times: list[float] = []
    quaternions: list[FloatArray] = []
    rates: list[FloatArray] = []
    wheel_momenta: list[FloatArray] = []
    wheel_torques: list[FloatArray] = []
    commanded_torques: list[FloatArray] = []
    delivered_torques: list[FloatArray] = []
    external_torques: list[FloatArray] = []
    fields: list[FloatArray] = []
    dipoles: list[FloatArray] = []
    stored: list[FloatArray] = []
    inertial: list[FloatArray] = []
    saturation: list[bool] = []

    total_steps = config.steps
    for index in range(total_steps + 1):
        time = index * step
        state = PlantState.unflatten(packed)

        if index % config.sample_stride == 0 or index == total_steps:
            external, field_body, dipole = environment(time, state)
            command, allocation = control(time, state)
            times.append(time)
            quaternions.append(state.quaternion.copy())
            rates.append(state.body_rate.copy())
            wheel_momenta.append(state.wheel_momentum.copy())
            wheel_torques.append(allocation.wheel_torque.copy())
            commanded_torques.append(command)
            delivered_torques.append(allocation.delivered_torque.copy())
            external_torques.append(external.copy())
            fields.append(field_body.copy())
            dipoles.append(dipole.copy())
            stored.append(wheels.axes @ state.wheel_momentum)
            inertial.append(
                inertial_angular_momentum(
                    spacecraft, state.quaternion, state.body_rate, state.wheel_momentum
                )
            )
            saturation.append(allocation.saturated)

        if index == total_steps:
            break
        packed = normalise_quaternion_state(rk4_step(derivative, time, packed, step))

    return ScenarioTrace(
        name=config.label or config.controller.name,
        time=np.array(times, dtype=np.float64),
        quaternion=_stack(quaternions),
        body_rate=_stack(rates),
        wheel_momentum=_stack(wheel_momenta),
        wheel_torque=_stack(wheel_torques),
        commanded_torque=_stack(commanded_torques),
        delivered_torque=_stack(delivered_torques),
        external_torque=_stack(external_torques),
        magnetic_field_body=_stack(fields),
        dipole=_stack(dipoles),
        stored_body_momentum=_stack(stored),
        inertial_momentum=_stack(inertial),
        saturated=np.array(saturation, dtype=bool),
        commanded_quaternion=commanded_quaternion,
    )
