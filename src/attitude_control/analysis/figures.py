"""Figures summarising scenario traces.

The non-interactive Agg backend is selected before pyplot is imported, so the
example scripts run identically with or without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from attitude_control.algorithm.momentum import (
    controllable_momentum,
    uncontrollable_momentum,
)
from attitude_control.model.inertia import Spacecraft
from attitude_control.pipeline.scenario import ScenarioTrace

__all__ = ["plot_disturbance", "plot_dumping", "plot_slew_comparison"]

_FIGURE_SIZE = (9.0, 7.0)
_DPI = 130


def _finish(figure: Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=_DPI)
    plt.close(figure)
    return destination


def plot_slew_comparison(
    traces: Sequence[ScenarioTrace],
    spacecraft: Spacecraft,
    destination: Path,
) -> Path:
    """Plot error angle, body rate, wheel speed, and wheel torque for each trace."""
    figure, axes = plt.subplots(2, 2, figsize=_FIGURE_SIZE, sharex=True)
    for trace in traces:
        axes[0, 0].plot(trace.time, np.rad2deg(trace.error_angle()), label=trace.name)
        axes[0, 1].plot(
            trace.time, np.rad2deg(np.linalg.norm(trace.body_rate, axis=1)), label=trace.name
        )
        speeds = trace.wheel_speed(spacecraft) * 60.0 / (2.0 * np.pi)
        axes[1, 0].plot(trace.time, np.max(np.abs(speeds), axis=1), label=trace.name)
        axes[1, 1].plot(
            trace.time, np.max(np.abs(trace.wheel_torque), axis=1), label=trace.name
        )

    axes[0, 0].set_ylabel("attitude error [deg]")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_ylabel("body rate [deg/s]")
    axes[1, 0].set_ylabel("largest wheel speed [rpm]")
    axes[1, 1].set_ylabel("largest wheel torque [N m]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("time [s]")
    figure.suptitle("Slew manoeuvre, controller comparison")
    return _finish(figure, destination)


def plot_disturbance(
    trace: ScenarioTrace,
    spacecraft: Spacecraft,
    destination: Path,
) -> Path:
    """Plot pointing error, disturbance torque, and accumulated wheel momentum."""
    figure, axes = plt.subplots(3, 1, figsize=_FIGURE_SIZE, sharex=True)
    hours = trace.time / 3600.0
    axes[0].plot(hours, np.rad2deg(trace.error_angle()) * 3600.0, color="C0")
    axes[0].set_ylabel("pointing error [arcsec]")
    for index, label in enumerate("xyz"):
        axes[1].plot(hours, trace.external_torque[:, index] * 1e6, label=f"body {label}")
    axes[1].set_ylabel("disturbance torque [micro N m]")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        hours, np.linalg.norm(trace.stored_body_momentum, axis=1), color="C3", label="magnitude"
    )
    for index, label in enumerate("xyz"):
        axes[2].plot(hours, trace.stored_body_momentum[:, index], alpha=0.6, label=f"body {label}")
    axes[2].set_ylabel("stored momentum [N m s]")
    axes[2].set_xlabel("time [h]")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    del spacecraft
    figure.suptitle("Gravity gradient disturbance rejection")
    return _finish(figure, destination)


def plot_dumping(
    traces: Sequence[ScenarioTrace],
    destination: Path,
) -> Path:
    """Plot stored momentum and its split along and across the magnetic field."""
    figure, axes = plt.subplots(len(traces), 1, figsize=_FIGURE_SIZE, sharex=True, squeeze=False)
    for row, trace in enumerate(traces):
        axis = axes[row, 0]
        hours = trace.time / 3600.0
        along = np.array(
            [
                np.linalg.norm(uncontrollable_momentum(h, b))
                for h, b in zip(trace.stored_body_momentum, trace.magnetic_field_body, strict=True)
            ]
        )
        across = np.array(
            [
                np.linalg.norm(controllable_momentum(h, b))
                for h, b in zip(trace.stored_body_momentum, trace.magnetic_field_body, strict=True)
            ]
        )
        axis.plot(hours, np.linalg.norm(trace.stored_body_momentum, axis=1), label="total")
        axis.plot(hours, along, label="along field, not controllable")
        axis.plot(hours, across, label="across field, controllable")
        axis.set_ylabel("momentum [N m s]")
        axis.set_title(trace.name, fontsize=10)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("time [h]")
    figure.suptitle("Magnetic momentum dumping")
    return _finish(figure, destination)
