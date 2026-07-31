"""Text rendering of metrics, kept out of the example scripts.

Formatting lives here rather than in the examples so that every script prints the
same table for the same numbers, and so the examples stay pure wiring.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from attitude_control.algorithm.momentum import controllable_momentum, uncontrollable_momentum
from attitude_control.analysis.metrics import ManoeuvreMetrics
from attitude_control.numeric import FloatArray
from attitude_control.pipeline.scenario import ScenarioTrace

__all__ = ["DUMPING_HEADER", "dumping_summary", "format_table", "metrics_table"]

DUMPING_HEADER: tuple[str, ...] = (
    "run",
    "|h| start",
    "|h| end",
    "removed %",
    "along start",
    "along end",
    "across start",
    "across end",
    "proj t0 start",
    "proj t0 end",
)


def format_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a fixed width text table."""
    columns = len(header)
    widths = [
        max(len(str(header[index])), *(len(str(row[index])) for row in rows))
        if rows
        else len(str(header[index]))
        for index in range(columns)
    ]
    lines = ["  ".join(str(header[i]).ljust(widths[i]) for i in range(columns))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(str(row[i]).ljust(widths[i]) for i in range(columns)) for row in rows
    )
    return "\n".join(lines)


def metrics_table(metrics: Sequence[ManoeuvreMetrics]) -> str:
    """Return a table of manoeuvre metrics, one row per run."""
    return format_table(ManoeuvreMetrics.header(), [m.as_row() for m in metrics])


def _split(trace: ScenarioTrace) -> tuple[FloatArray, FloatArray]:
    along = np.array(
        [
            np.linalg.norm(uncontrollable_momentum(h, b))
            for h, b in zip(trace.stored_body_momentum, trace.magnetic_field_body, strict=True)
        ],
        dtype=np.float64,
    )
    across = np.array(
        [
            np.linalg.norm(controllable_momentum(h, b))
            for h, b in zip(trace.stored_body_momentum, trace.magnetic_field_body, strict=True)
        ],
        dtype=np.float64,
    )
    return along, across


def dumping_summary(trace: ScenarioTrace) -> tuple[str, ...]:
    """Return one report row describing how much momentum a dumping run removed.

    The row separates the component of stored momentum along the instantaneous
    field, which no dipole can act on, from the component across it, which the
    cross product law removes exponentially.
    """
    along, across = _split(trace)
    total = np.linalg.norm(trace.stored_body_momentum, axis=1)
    initial_axis = trace.magnetic_field_body[0]
    projection = trace.stored_body_momentum @ initial_axis / float(np.linalg.norm(initial_axis))
    return (
        trace.name,
        f"{total[0]:.4f}",
        f"{total[-1]:.4f}",
        f"{100.0 * (1.0 - total[-1] / total[0]):.1f}",
        f"{along[0]:.4f}",
        f"{along[-1]:.4f}",
        f"{across[0]:.4f}",
        f"{across[-1]:.4f}",
        f"{projection[0]:.4f}",
        f"{projection[-1]:.4f}",
    )
