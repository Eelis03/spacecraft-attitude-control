"""Compare quaternion PD and LQR on the same rest to rest slew.

Run with ``--quick`` for the shortened version used by the integration tests.

    uv run python examples/slew_manoeuvre.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from attitude_control.analysis.metrics import ManoeuvreMetrics
from attitude_control.analysis.report import metrics_table
from attitude_control.configuration import (
    aggressive_controller,
    controllers,
    reference_spacecraft,
    slew_scenario,
)
from attitude_control.pipeline.scenario import run_scenario

_FIGURE = Path("figures") / "slew_manoeuvre.png"


def parse_arguments() -> argparse.Namespace:
    """Return the command line options."""
    parser = argparse.ArgumentParser(description="Slew manoeuvre controller comparison")
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    parser.add_argument("--no-figure", action="store_true", help="skip figure generation")
    return parser.parse_args()


def main() -> None:
    """Run the slew with each controller and report the manoeuvre metrics."""
    options = parse_arguments()
    spacecraft = reference_spacecraft()
    duration = 200.0 if options.quick else 900.0
    step = 0.5 if options.quick else 0.2

    laws = [*controllers(spacecraft), aggressive_controller(spacecraft)]
    traces = [
        run_scenario(slew_scenario(spacecraft, law, duration=duration, time_step=step))
        for law in laws
    ]
    metrics = [ManoeuvreMetrics.evaluate(trace, spacecraft) for trace in traces]

    for item in metrics:
        print(
            f"{item.name}: peak commanded torque {item.peak_commanded_torque_nm:.4f} N m, "
            f"torque limited for {item.saturated_fraction * 100.0:.1f} per cent of the run, "
            f"inertial momentum drift {item.momentum_drift_nms:.2e} N m s"
        )

    print()
    print(metrics_table(metrics))

    if not options.no_figure:
        from attitude_control.analysis.figures import plot_slew_comparison

        print(f"\nfigure written to {plot_slew_comparison(traces, spacecraft, _FIGURE)}")


if __name__ == "__main__":
    main()
