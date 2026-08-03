"""Hold an inertial attitude against gravity gradient torque.

Three laws are compared: the two without integral action, whose residual is the
static gain of the loop against the mean disturbance torque, and the PID design,
whose integral term removes that residual and leaves only the periodic part.

Run with ``--quick`` for the shortened version used by the integration tests.

    uv run python examples/disturbance_rejection.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from attitude_control.analysis.metrics import ManoeuvreMetrics, mean_error_vector
from attitude_control.analysis.report import format_table
from attitude_control.configuration import (
    controllers,
    disturbance_scenario,
    integral_controller,
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

    laws = [*controllers(spacecraft), integral_controller(spacecraft)]
    traces = [
        run_scenario(disturbance_scenario(spacecraft, law, orbits=orbits, time_step=step))
        for law in laws
    ]
    print(
        f"peak gravity gradient torque "
        f"{np.max(np.linalg.norm(traces[0].external_torque, axis=1)):.3e} N m"
    )

    rows = []
    for trace in traces:
        error = np.rad2deg(trace.error_angle()) * 3600.0
        offset = np.rad2deg(np.linalg.norm(mean_error_vector(trace))) * 3600.0
        stored = np.linalg.norm(trace.stored_body_momentum, axis=1)
        metrics = ManoeuvreMetrics.evaluate(trace, spacecraft)
        rows.append(
            (
                trace.name,
                f"{np.mean(error[error.size // 2 :]):.1f}",
                f"{np.max(error):.1f}",
                f"{offset:.2f}",
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
                "mean error as",
                "peak error as",
                "mean error vector as",
                "stored Nms",
                "Nms per orbit",
                "wheel rpm",
                "impulse Nms",
            ),
            rows,
        )
    )
    print(
        "\nThe mean error vector column averages the error as a vector over the second"
        "\nhalf of the run, so a constant offset survives it and a zero mean oscillation"
        "\ndoes not. It is the number that separates a loop with integral action from one"
        "\nwithout."
    )
    print(
        "\nThe impulse column is the change in total inertial angular momentum, which"
        "\nequals the integral of the external torque. It matches the stored momentum"
        "\ncolumn because the wheels absorbed the whole disturbance, and it is the same"
        "\nfor all three laws because integral action changes where the vehicle points,"
        "\nnot how much momentum the environment delivers."
    )

    if not options.no_figure:
        from attitude_control.analysis.figures import plot_disturbance

        print(f"\nfigure written to {plot_disturbance(traces, spacecraft, _FIGURE)}")


if __name__ == "__main__":
    main()
