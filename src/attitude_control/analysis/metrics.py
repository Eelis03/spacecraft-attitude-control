"""Performance metrics computed from a scenario trace.

Definitions are stated here because manoeuvre metrics are only comparable when
the definitions match.

* **Settling time** is the earliest sample after which the principal attitude
  error stays below a threshold for the rest of the run. Taking the earliest
  sample of a permanently satisfied condition, rather than the first crossing,
  means a transient dip below the threshold does not count.
* **Overshoot** is measured on the error angle *signed about the initial error
  axis*, which is negative once the body has passed the target. It is reported as
  a percentage of the initial error. The unsigned principal angle cannot express
  overshoot because it is non-negative by construction.
* **Peak wheel speed** and **peak wheel torque** are maxima over wheels and time
  of the absolute values.
* **Stored momentum** is the magnitude of the total body frame angular momentum
  ``J w + W h_w``, which is what has to be held and eventually dumped.
* **Momentum drift** is the largest change in the inertial frame total angular
  momentum over the run. With no external torque it should stay at the level of
  the integrator error, so it doubles as a correctness monitor on every run.
* **Mean error vector** is the time average of the small angle attitude error
  over the tail of a run. It is a vector average, not an average of magnitudes,
  which is the only version that distinguishes a constant pointing offset from a
  zero mean oscillation of the same size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attitude_control.model.attitude import signed_angle_about
from attitude_control.model.inertia import Spacecraft
from attitude_control.numeric import FloatArray
from attitude_control.pipeline.scenario import ScenarioTrace

__all__ = [
    "ManoeuvreMetrics",
    "mean_error_vector",
    "momentum_drift",
    "settling_time",
    "signed_error_angle",
]


def signed_error_angle(trace: ScenarioTrace) -> FloatArray:
    """Return the attitude error signed about the initial error axis, in rad.

    The axis is taken from the first sample. When the initial error is smaller
    than the small-angle threshold the run is a regulation task with no
    meaningful axis, and the unsigned principal angle is returned instead.
    """
    errors = trace.error_quaternion()
    axis_norm = float(np.linalg.norm(errors[0, 1:]))
    if axis_norm < np.sqrt(np.finfo(np.float64).eps):
        return np.array([2.0 * np.linalg.norm(e[1:]) for e in errors], dtype=np.float64)
    axis = errors[0, 1:] / axis_norm
    return np.array([signed_angle_about(e, axis) for e in errors], dtype=np.float64)


def settling_time(trace: ScenarioTrace, threshold: float) -> float:
    """Return the time after which the error angle stays below ``threshold`` rad.

    Returns ``inf`` when the error never settles within the recorded run.
    """
    error = trace.error_angle()
    outside = np.flatnonzero(error > threshold)
    if outside.size == 0:
        return float(trace.time[0])
    last = int(outside[-1])
    if last + 1 >= trace.time.size:
        return float("inf")
    return float(trace.time[last + 1])


def mean_error_vector(trace: ScenarioTrace, tail_fraction: float = 0.5) -> FloatArray:
    """Return the time averaged attitude error over the tail of a run, in rad.

    The quantity averaged is ``2 dq_v``, which equals the principal rotation
    vector to first order. The average runs over the last ``tail_fraction`` of
    the samples, so an initial transient does not contaminate it.

    A proportional plus derivative loop under a disturbance with a non-zero mean
    settles to a constant offset, and this average returns that offset. A loop
    with integral action settles to a zero mean oscillation about the command, and
    this average returns approximately zero while the mean of the error
    *magnitude* stays at the amplitude of the oscillation.
    """
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    errors = trace.error_quaternion()
    start = int(errors.shape[0] * (1.0 - tail_fraction))
    return np.asarray(2.0 * np.mean(errors[start:, 1:], axis=0), dtype=np.float64)


def momentum_drift(trace: ScenarioTrace) -> float:
    """Return the largest deviation of inertial angular momentum from its initial value."""
    deltas = trace.inertial_momentum - trace.inertial_momentum[0]
    return float(np.max(np.linalg.norm(deltas, axis=1)))


@dataclass(frozen=True, slots=True)
class ManoeuvreMetrics:
    """Summary of one slew or regulation run."""

    name: str
    initial_error_deg: float
    settling_time_s: float
    overshoot_percent: float
    final_error_deg: float
    peak_body_rate_deg_s: float
    peak_wheel_speed_rpm: float
    peak_wheel_torque_nm: float
    peak_commanded_torque_nm: float
    peak_stored_momentum_nms: float
    momentum_drift_nms: float
    saturated_fraction: float

    @classmethod
    def evaluate(
        cls,
        trace: ScenarioTrace,
        spacecraft: Spacecraft,
        settling_threshold_deg: float = 0.1,
    ) -> ManoeuvreMetrics:
        """Compute every metric from a recorded trace."""
        error = trace.error_angle()
        signed = signed_error_angle(trace)
        initial = float(signed[0])
        if abs(initial) > 0.0:
            excursion = float(np.min(signed / initial))
            overshoot = max(0.0, -excursion) * 100.0
        else:
            overshoot = 0.0
        speeds = trace.wheel_speed(spacecraft)
        return cls(
            name=trace.name,
            initial_error_deg=float(np.rad2deg(error[0])),
            settling_time_s=settling_time(trace, float(np.deg2rad(settling_threshold_deg))),
            overshoot_percent=overshoot,
            final_error_deg=float(np.rad2deg(error[-1])),
            peak_body_rate_deg_s=float(np.rad2deg(np.max(np.linalg.norm(trace.body_rate, axis=1)))),
            peak_wheel_speed_rpm=float(np.max(np.abs(speeds)) * 60.0 / (2.0 * np.pi)),
            peak_wheel_torque_nm=float(np.max(np.abs(trace.wheel_torque))),
            peak_commanded_torque_nm=float(
                np.max(np.linalg.norm(trace.commanded_torque, axis=1))
            ),
            peak_stored_momentum_nms=float(
                np.max(np.linalg.norm(trace.stored_body_momentum, axis=1))
            ),
            momentum_drift_nms=momentum_drift(trace),
            saturated_fraction=float(np.mean(trace.saturated)),
        )

    def as_row(self) -> tuple[str, ...]:
        """Return the metrics formatted for a fixed width report table."""
        return (
            self.name,
            f"{self.settling_time_s:.1f}",
            f"{self.overshoot_percent:.2f}",
            f"{self.final_error_deg:.3e}",
            f"{self.peak_wheel_speed_rpm:.1f}",
            f"{self.peak_wheel_torque_nm:.3e}",
            f"{self.peak_stored_momentum_nms:.4f}",
        )

    @staticmethod
    def header() -> tuple[str, ...]:
        """Return the column headings matching :meth:`as_row`."""
        return (
            "controller",
            "settle s",
            "over %",
            "final deg",
            "wheel rpm",
            "torque Nm",
            "momentum Nms",
        )
