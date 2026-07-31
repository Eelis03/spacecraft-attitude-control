"""Dump stored wheel momentum with magnetic torque rods.

Two runs are compared. In the first the magnetic field is frozen in inertial
space, so the direction that cannot be torqued about never moves and the
component of momentum along it survives. In the second the spacecraft flies the
reference orbit, the field direction sweeps, and the whole momentum vector is
removed.

Run with ``--quick`` for the shortened version used by the integration tests.

    uv run python examples/momentum_dumping.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from attitude_control.analysis.report import DUMPING_HEADER, dumping_summary, format_table
from attitude_control.configuration import (
    constant_field_environment,
    controllers,
    dumping_scenario,
    orbiting_field_environment,
    reference_spacecraft,
)
from attitude_control.model.attitude import dcm_from_quaternion
from attitude_control.pipeline.scenario import ScenarioTrace, run_scenario

_FIGURE = Path("figures") / "momentum_dumping.png"


def parse_arguments() -> argparse.Namespace:
    """Return the command line options."""
    parser = argparse.ArgumentParser(description="Magnetic momentum dumping")
    parser.add_argument("--quick", action="store_true", help="short run for smoke testing")
    parser.add_argument("--no-figure", action="store_true", help="skip figure generation")
    return parser.parse_args()


def along_field_report(trace: ScenarioTrace) -> str:
    """Report the inertial momentum component along the initial field direction.

    With a field fixed in the inertial frame this component is exactly conserved,
    because the magnetic torque is orthogonal to the field by construction. The
    number printed here is therefore a direct measurement of the limitation.
    """
    axis = dcm_from_quaternion(trace.quaternion[0]).T @ trace.magnetic_field_body[0]
    axis = axis / np.linalg.norm(axis)
    projection = trace.inertial_momentum @ axis
    change = float(np.max(np.abs(projection - projection[0])))
    return (
        f"{trace.name}: inertial momentum along the initial field direction "
        f"{projection[0]:+.6f} N m s at the start, {projection[-1]:+.6f} N m s at the end, "
        f"largest excursion {change:.2e} N m s"
    )


def main() -> None:
    """Run both dumping cases and report what each removed."""
    options = parse_arguments()
    spacecraft = reference_spacecraft()
    controller = controllers(spacecraft)[0]
    orbits = 0.1 if options.quick else 3.0
    step = 4.0 if options.quick else 2.0

    environments = (
        ("field fixed in inertial space", constant_field_environment()),
        ("field along the reference orbit", orbiting_field_environment()),
    )

    traces = [
        run_scenario(
            dumping_scenario(
                spacecraft,
                controller,
                environment,
                orbits=orbits,
                time_step=step,
                label=label,
            )
        )
        for label, environment in environments
    ]

    for trace in traces:
        print(along_field_report(trace))

    print()
    print(format_table(DUMPING_HEADER, [dumping_summary(trace) for trace in traces]))

    if not options.no_figure:
        from attitude_control.analysis.figures import plot_dumping

        print(f"\nfigure written to {plot_dumping(traces, _FIGURE)}")


if __name__ == "__main__":
    main()
