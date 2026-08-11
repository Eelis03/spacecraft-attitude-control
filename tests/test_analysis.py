"""The analysis layer: metrics, text tables, round trip residuals, and figures.

Everything here reads a trace and produces something a human looks at, so the
assertions are about the contract each output has to keep: a table that lines up,
a residual that is small for the right reason, a figure whose pixel size is the
one the published byte budget was calculated from.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from attitude_control.analysis.figures import (
    plot_disturbance,
    plot_dumping,
    plot_slew_comparison,
)
from attitude_control.analysis.metrics import (
    ManoeuvreMetrics,
    mean_error_vector,
    settling_time,
)
from attitude_control.analysis.report import (
    DUMPING_HEADER,
    dumping_summary,
    format_table,
    metrics_table,
)
from attitude_control.analysis.representation import (
    CONVERSION_PATHS,
    round_trip_residuals,
    worst_orthonormality_defect,
)
from attitude_control.configuration import (
    constant_field_environment,
    controllers,
    dumping_scenario,
    integral_controller,
    reference_spacecraft,
    slew_scenario,
)
from attitude_control.model.inertia import Spacecraft
from attitude_control.pipeline.scenario import ScenarioTrace, run_scenario

# The published figures are 7.4 by 4.6 inches at 110 dots per inch. The byte
# budget in the README was computed for exactly this raster, so the numbers are
# pinned here rather than read back from the module they came from.
EXPECTED_PIXELS = (814, 506)


def png_size(path: Path) -> tuple[int, int]:
    """Return the pixel width and height from the PNG header.

    Read directly from the IHDR chunk so that checking the raster size of a
    figure needs no image library beyond the one that wrote it.
    """
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


@pytest.fixture(scope="module")
def slew_traces() -> list[ScenarioTrace]:
    """Two short slews, one per control law, enough to exercise every reader."""
    spacecraft = reference_spacecraft()
    return [
        run_scenario(slew_scenario(spacecraft, law, duration=60.0, time_step=0.5, sample_stride=5))
        for law in controllers(spacecraft)
    ]


@pytest.fixture(scope="module")
def dumping_traces() -> list[ScenarioTrace]:
    """One short magnetic dumping run in a field fixed in inertial space."""
    spacecraft = reference_spacecraft()
    return [
        run_scenario(
            dumping_scenario(
                spacecraft,
                controllers(spacecraft)[0],
                constant_field_environment(),
                orbits=0.02,
                time_step=4.0,
                sample_stride=2,
                label="fixed field",
            )
        )
    ]


def test_metrics_row_matches_its_header(
    slew_traces: list[ScenarioTrace], spacecraft: Spacecraft
) -> None:
    """A report row has exactly one entry per column, which is what keeps the table honest."""
    metrics = ManoeuvreMetrics.evaluate(slew_traces[0], spacecraft)
    assert len(metrics.as_row()) == len(ManoeuvreMetrics.header())
    assert metrics.as_row()[0] == metrics.name


def test_settling_time_is_infinite_when_the_run_never_settles(
    slew_traces: list[ScenarioTrace],
) -> None:
    """A 60 s window cannot settle a 60 degree slew, and the metric says so.

    Returning the end of the run instead would look like a fast settle, which is
    the failure mode this distinguishes.
    """
    assert settling_time(slew_traces[0], np.deg2rad(0.1)) == float("inf")
    assert settling_time(slew_traces[0], np.deg2rad(180.0)) == 0.0


def test_mean_error_vector_averages_over_the_requested_window(
    slew_traces: list[ScenarioTrace],
) -> None:
    """The tail fraction selects a window, and a decaying error shrinks in it.

    The slew error decays monotonically over this short run, so the average over
    the last tenth must be smaller than the average over the whole run. A
    windowing bug that ignored the fraction, or that took the head instead of the
    tail, would fail this.
    """
    trace = slew_traces[0]
    tail = float(np.linalg.norm(mean_error_vector(trace, tail_fraction=0.1)))
    whole = float(np.linalg.norm(mean_error_vector(trace, tail_fraction=1.0)))
    assert 0.0 < tail < whole
    with pytest.raises(ValueError, match="tail fraction"):
        mean_error_vector(trace, tail_fraction=0.0)
    with pytest.raises(ValueError, match="tail fraction"):
        mean_error_vector(trace, tail_fraction=1.5)


def test_format_table_pads_every_line_to_the_same_width() -> None:
    """Columns line up, including when a cell is wider than its heading."""
    text = format_table(("a", "bb"), [("xxxx", "y"), ("z", "wwwww")])
    lines = text.splitlines()
    assert len({len(line) for line in lines}) == 1
    assert set(lines[1]) == {"-", " "}
    assert lines[0].startswith("a   ")


def test_format_table_without_rows_uses_the_header_widths() -> None:
    """An empty table still renders, which is what stops a report crashing on no data."""
    text = format_table(("alpha", "b"), [])
    assert text.splitlines() == ["alpha  b", "-----  -"]


def test_metrics_table_starts_with_the_metric_header(
    slew_traces: list[ScenarioTrace], spacecraft: Spacecraft
) -> None:
    """The rendered table and the metric definition cannot drift apart."""
    metrics = [ManoeuvreMetrics.evaluate(trace, spacecraft) for trace in slew_traces]
    lines = metrics_table(metrics).splitlines()
    assert re.split(r"\s{2,}", lines[0].strip()) == list(ManoeuvreMetrics.header())
    assert len(lines) == len(metrics) + 2


def test_dumping_summary_is_internally_consistent(dumping_traces: list[ScenarioTrace]) -> None:
    """The reported removed fraction agrees with the start and end magnitudes it prints."""
    row = dumping_summary(dumping_traces[0])
    assert len(row) == len(DUMPING_HEADER)
    start, end, removed = float(row[1]), float(row[2]), float(row[3])
    assert removed == pytest.approx(100.0 * (1.0 - end / start), abs=0.05)
    along_start, across_start = float(row[4]), float(row[6])
    total = float(np.hypot(along_start, across_start))
    assert total == pytest.approx(start, abs=1e-3)


def test_round_trip_residuals_are_small_and_reproducible() -> None:
    """Every conversion path round trips to machine precision, the same way every time.

    Tolerance: a round trip performs of order ten arithmetic operations on
    quantities of size one, so the residual angle is a few machine epsilons. The
    Euler path divides by the cosine of the pitch angle and therefore loses more,
    which is why it is bounded separately and more loosely.
    """
    residuals = round_trip_residuals(samples=200, seed=7)
    assert set(residuals) == set(CONVERSION_PATHS)
    assert residuals["DCM"] < 1e-14
    assert residuals["MRP"] < 1e-14
    assert residuals["Euler 3-2-1"] < 1e-12
    assert residuals == round_trip_residuals(samples=200, seed=7)
    assert all(value > 0.0 for value in residuals.values())


def test_orthonormality_defect_stays_at_rounding_level() -> None:
    """Generated attitude matrices are orthonormal to a few machine epsilons."""
    assert worst_orthonormality_defect(samples=200, seed=7) < 1e-14


def test_slew_figure_has_the_published_raster_size(
    slew_traces: list[ScenarioTrace], spacecraft: Spacecraft, tmp_path: Path
) -> None:
    """The figure is written at the size the byte budget was computed for."""
    path = plot_slew_comparison(slew_traces, spacecraft, tmp_path / "nested" / "slew.png")
    assert path.is_file()
    assert png_size(path) == EXPECTED_PIXELS


def test_disturbance_figure_accepts_several_control_laws(
    spacecraft: Spacecraft, tmp_path: Path
) -> None:
    """The pointing panel compares laws, so it has to take more than one trace."""
    traces = [
        run_scenario(slew_scenario(spacecraft, law, duration=40.0, time_step=0.5, sample_stride=5))
        for law in (controllers(spacecraft)[0], integral_controller(spacecraft))
    ]
    path = plot_disturbance(traces, spacecraft, tmp_path / "disturbance.png")
    assert png_size(path) == EXPECTED_PIXELS


def test_dumping_figure_is_written_per_run(
    dumping_traces: list[ScenarioTrace], tmp_path: Path
) -> None:
    """One panel per dumping run, at the published size."""
    path = plot_dumping(dumping_traces, tmp_path / "dumping.png")
    assert png_size(path) == EXPECTED_PIXELS
