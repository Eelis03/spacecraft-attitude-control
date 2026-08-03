"""Regenerate the three figures tracked in ``docs/figures``.

This is the one command behind the figures shown in the README:

    uv run python scripts/publish_figures.py

The scenarios are the same ones the example scripts run, at the same settings, so
a figure and the numbers printed beside it always describe the same simulation.
The destination is resolved from this file rather than from the working
directory, so the command writes to the repository wherever it is run from.

The tracked figures share a byte budget, which this script prints and checks. A
repository that carries images has to keep them small enough that cloning it stays
cheap, and the honest way to enforce that is to measure it every time they are
regenerated rather than to hope.
"""

from __future__ import annotations

import sys
from pathlib import Path

from attitude_control.analysis.figures import (
    plot_disturbance,
    plot_dumping,
    plot_slew_comparison,
)
from attitude_control.configuration import (
    aggressive_controller,
    constant_field_environment,
    controllers,
    disturbance_scenario,
    dumping_scenario,
    integral_controller,
    orbiting_field_environment,
    reference_spacecraft,
    slew_scenario,
)
from attitude_control.pipeline.scenario import run_scenario

DESTINATION = Path(__file__).resolve().parents[1] / "docs" / "figures"
BUDGET_BYTES = 250 * 1024


def main() -> int:
    """Write every published figure and report the total against the budget."""
    spacecraft = reference_spacecraft()

    slew = [
        run_scenario(slew_scenario(spacecraft, law))
        for law in (
            *controllers(spacecraft),
            aggressive_controller(spacecraft),
            integral_controller(spacecraft),
        )
    ]
    disturbance = [
        run_scenario(disturbance_scenario(spacecraft, law))
        for law in (*controllers(spacecraft), integral_controller(spacecraft))
    ]
    dumping = [
        run_scenario(
            dumping_scenario(spacecraft, controllers(spacecraft)[0], environment, label=label)
        )
        for label, environment in (
            ("field fixed in inertial space", constant_field_environment()),
            ("field along the reference orbit", orbiting_field_environment()),
        )
    ]

    written = [
        plot_dumping(dumping, DESTINATION / "momentum_dumping.png"),
        plot_disturbance(disturbance, spacecraft, DESTINATION / "disturbance_rejection.png"),
        plot_slew_comparison(slew, spacecraft, DESTINATION / "slew_manoeuvre.png"),
    ]

    total = 0
    for path in written:
        size = path.stat().st_size
        total += size
        print(f"{path.relative_to(DESTINATION.parents[1]).as_posix():40s} {size / 1024:7.1f} KiB")
    print(f"{'total':40s} {total / 1024:7.1f} KiB of {BUDGET_BYTES / 1024:.0f} KiB budget")

    if total > BUDGET_BYTES:
        print("over budget: reduce the figure size or the resolution", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
