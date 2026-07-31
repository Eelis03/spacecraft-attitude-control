"""Hold an inertial attitude against gravity gradient torque.

Run with ``--quick`` for the shortened version used by the integration tests.

    uv run python examples/disturbance_rejection.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from attitude_control.analysis.metrics import ManoeuvreMetrics
from attitude_control.analysis.report import format_table
from attitude_control.configuration import (
    controllers,
    disturbance_scenario,
    reference_orbit,
    reference_spacecraft,
)
from attitude_control.pipeline.scenario import run_scenario

_FIGURE = Path("figures") / "disturbance_rejection.png"


def parse_arguments() -> argparse.Namespace:
    """Return the command line options."""
    parser = argparse.ArgumentParser(description="Gravity gradient disturbance rejection")
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    parser.add_argument("--no-figure", action="store_true", help="skip figure generation")
    return parser.parse_args()


def main() -> None:
    """Run the inertial hold with each controller and report the pointing error."""
    options = parse_arguments()
    spacecraft = reference_spacecraft()
    orbit = reference_orbit()
    orbits = 0.1 if options.quick else 2.0
    step = 2.0 if options.quick else 1.0

    print(f"orbit period {orbit.period:.1f} s, mean motion {orbit.mean_motion:.6e} rad/s")

    traces = [
        run_scenario(disturbance_scenario(spacecraft, law, orbits=orbits, time_step=step))
        for law in controllers(spacecraft)
    ]

    rows = []
    for trace in traces:
        error = np.rad2deg(trace.error_angle()) * 3600.0
        torque = np.linalg.norm(trace.external_torque, axis=1)
        stored = np.linalg.norm(trace.stored_body_momentum, axis=1)
        metrics = ManoeuvreMetrics.evaluate(trace, spacecraft)
        rows.append(
            (
                trace.name,
                f"{np.max(torque):.3e}",
                f"{np.mean(error[error.size // 2 :]):.1f}",
                f"{np.max(error):.1f}",
                f"{stored[-1]:.4f}",
                f"{stored[-1] / orbits:.4f}",
                f"{metrics.peak_wheel_speed_rpm:.1f}",
                f"{metrics.momentum_drift_nms:.2e}",
            )
        )

    print()
    print(
        format_table(
            (
                "controller",
                "peak torque Nm",
                "mean error as",
                "peak error as",
                "stored Nms",
                "Nms per orbit",
                "wheel rpm",
                "impulse Nms",
            ),
            rows,
        )
    )
    print(
        "\nThe impulse column is the change in total inertial angular momentum, which"
        "\nequals the integral of the external torque. It matches the stored momentum"
        "\ncolumn because the wheels absorbed the whole disturbance."
    )

    if not options.no_figure:
        from attitude_control.analysis.figures import plot_disturbance

        print(f"\nfigure written to {plot_disturbance(traces[0], spacecraft, _FIGURE)}")


if __name__ == "__main__":
    main()
