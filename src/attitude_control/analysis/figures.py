"""Figures summarising scenario traces.

The non-interactive Agg backend is selected before pyplot is imported, so the
example scripts run identically with or without a display.

Size and resolution
-------------------
Every figure is two panels at ``(7.4, 4.6)`` inches and 110 dots per inch, which
is 814 by 506 pixels. The choice is deliberate rather than a default. The three
published figures are tracked in the repository and share a 250 kilobyte budget,
so each one has about 80 kilobytes to spend; a line plot of this size lands near
50, while the matplotlib default of 100 dots per inch at a four panel layout
lands near 150 and would need either a compression dependency or a third of the
figures dropped. Nothing here is a raster of data, so the resolution only has to
carry the axis labels legibly.
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

_FIGURE_SIZE = (7.4, 4.6)
_DPI = 110


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
    """Plot the attitude error and the wheel torque against the torque limit.

    The torque panel is what the settling times in the tables cannot show: the
    over-driven design reaches the target sooner only by sitting on the wheel
    torque limit, which appears here as a flat top against the dashed line.
    """
    figure, axes = plt.subplots(1, 2, figsize=_FIGURE_SIZE)
    for trace in traces:
        axes[0].semilogy(trace.time, np.rad2deg(trace.error_angle()), label=trace.name)
        axes[1].plot(trace.time, np.max(np.abs(trace.wheel_torque), axis=1), label=trace.name)

    axes[1].axhline(
        spacecraft.wheels.max_torque,
        color="0.3",
        linestyle="--",
        linewidth=1.0,
        label="wheel torque limit",
    )
    axes[0].set_ylabel("attitude error [deg]")
    axes[1].set_ylabel("largest wheel torque [N m]")
    for axis in axes:
        axis.set_xlabel("time [s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Slew manoeuvre, controller comparison", fontsize=11)
    return _finish(figure, destination)


def plot_disturbance(
    traces: Sequence[ScenarioTrace],
    spacecraft: Spacecraft,
    destination: Path,
) -> Path:
    """Plot the pointing error and the stored momentum for each control law.

    The two panels answer different questions. Integral action moves the error
    curve towards zero; it leaves the momentum curve exactly where it was,
    because the wheels still absorb every newton metre second the environment
    delivers.
    """
    del spacecraft
    figure, axes = plt.subplots(1, 2, figsize=_FIGURE_SIZE)
    for trace in traces:
        hours = trace.time / 3600.0
        axes[0].plot(hours, np.rad2deg(trace.error_angle()) * 3600.0, label=trace.name)
        axes[1].plot(
            hours, np.linalg.norm(trace.stored_body_momentum, axis=1), label=trace.name
        )
    axes[0].set_ylabel("pointing error [arcsec]")
    axes[1].set_ylabel("stored momentum [N m s]")
    for axis in axes:
        axis.set_xlabel("time [h]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Gravity gradient hold, pointing and stored momentum", fontsize=11)
    return _finish(figure, destination)


def plot_dumping(
    traces: Sequence[ScenarioTrace],
    destination: Path,
) -> Path:
    """Plot stored momentum and its split along and across the magnetic field."""
    figure, axes = plt.subplots(1, len(traces), figsize=_FIGURE_SIZE, sharey=True, squeeze=False)
    for column, trace in enumerate(traces):
        axis = axes[0, column]
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
        axis.set_title(trace.name, fontsize=9)
        axis.set_xlabel("time [h]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    axes[0, 0].set_ylabel("momentum [N m s]")
    figure.suptitle("Magnetic momentum dumping", fontsize=11)
    return _finish(figure, destination)
