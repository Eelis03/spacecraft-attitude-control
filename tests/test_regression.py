"""Tier two: a recorded reference run compared against pinned numbers.

What is pinned and what is not
------------------------------
Only quantities that another machine can reproduce are pinned. Every scenario
recorded here is a stable closed loop: the error dynamics are contracting, so a
difference of one unit in the last place in a LAPACK factorisation cannot grow,
and every recorded aggregate is a smooth functional of a converged trajectory.

Three things are deliberately *not* pinned.

* The inertial momentum drift, which is the accumulated rounding of the run and
  therefore differs with the order a machine sums a dot product. It is checked
  against a derived upper bound instead.
* The final error of the over-driven slew, which reaches 1e-12 degrees. At that
  size the value is rounding noise, not a physical result, so only an upper bound
  is asserted.
* Anything from the intermediate axis instability, which is chaotic. Its
  qualitative signature is tested in ``test_dynamics`` and no trajectory number
  from it appears in this file.

Tolerances
----------
``_TOLERANCES`` states one rule per quantity with the reason. None of them was
chosen by observing a difference; each follows from how the quantity is computed.

Regenerating
------------
Run ``uv run python -m tests.test_regression`` after a deliberate change in
behaviour, and describe the change in the commit that updates the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from attitude_control.algorithm.controller import LinearQuadraticRegulator
from attitude_control.analysis.metrics import ManoeuvreMetrics
from attitude_control.configuration import (
    LQR_WEIGHTS,
    PD_NATURAL_FREQUENCY,
    aggressive_controller,
    constant_field_environment,
    controllers,
    disturbance_scenario,
    dumping_scenario,
    orbiting_field_environment,
    reference_orbit,
    reference_spacecraft,
    slew_scenario,
)
from attitude_control.model.attitude import dcm_from_quaternion
from attitude_control.model.environment import GRAVITATIONAL_PARAMETER
from attitude_control.model.inertia import Spacecraft, pyramid_wheel_axes
from attitude_control.pipeline.scenario import (
    MagneticEnvironment,
    ScenarioTrace,
    run_scenario,
)
from tests.conftest import rounding_bound, truncation_bound

REFERENCE_PATH = Path(__file__).parent / "data" / "reference_run.json"

# Reduced settings so the regression tier stays inside the suite time budget. The
# physics is the same as the example scripts; only the horizon and the step differ.
SLEW_DURATION = 900.0
SLEW_STEP = 0.5
SLEW_STRIDE = 10
ENVIRONMENT_ORBITS = 1.0
ENVIRONMENT_STEP = 4.0
ENVIRONMENT_STRIDE = 10


def _slew_traces(spacecraft: Spacecraft) -> dict[str, ScenarioTrace]:
    laws = [*controllers(spacecraft), aggressive_controller(spacecraft)]
    return {
        law.name: run_scenario(
            slew_scenario(
                spacecraft,
                law,
                duration=SLEW_DURATION,
                time_step=SLEW_STEP,
                sample_stride=SLEW_STRIDE,
            )
        )
        for law in laws
    }


def _disturbance_trace(spacecraft: Spacecraft) -> ScenarioTrace:
    return run_scenario(
        disturbance_scenario(
            spacecraft,
            controllers(spacecraft)[0],
            orbits=ENVIRONMENT_ORBITS,
            time_step=ENVIRONMENT_STEP,
            sample_stride=ENVIRONMENT_STRIDE,
        )
    )


def _dumping_traces(spacecraft: Spacecraft) -> dict[str, ScenarioTrace]:
    environments: dict[str, MagneticEnvironment] = {
        "fixed field": constant_field_environment(),
        "orbiting field": orbiting_field_environment(),
    }
    return {
        label: run_scenario(
            dumping_scenario(
                spacecraft,
                controllers(spacecraft)[0],
                environment,
                orbits=ENVIRONMENT_ORBITS,
                time_step=ENVIRONMENT_STEP,
                sample_stride=ENVIRONMENT_STRIDE,
                label=label,
            )
        )
        for label, environment in environments.items()
    }


def _slew_summary(spacecraft: Spacecraft, trace: ScenarioTrace) -> dict[str, float]:
    metrics = ManoeuvreMetrics.evaluate(trace, spacecraft)
    return {
        "initial_error_deg": metrics.initial_error_deg,
        "settling_time_s": metrics.settling_time_s,
        "overshoot_percent": metrics.overshoot_percent,
        "final_error_deg": metrics.final_error_deg,
        "peak_body_rate_deg_s": metrics.peak_body_rate_deg_s,
        "peak_wheel_speed_rpm": metrics.peak_wheel_speed_rpm,
        "peak_wheel_torque_nm": metrics.peak_wheel_torque_nm,
        "peak_commanded_torque_nm": metrics.peak_commanded_torque_nm,
        "peak_stored_momentum_nms": metrics.peak_stored_momentum_nms,
        "saturated_samples": float(np.count_nonzero(trace.saturated)),
    }


def _dumping_summary(trace: ScenarioTrace) -> dict[str, float]:
    stored = np.linalg.norm(trace.stored_body_momentum, axis=1)
    field = trace.magnetic_field_body[0] / np.linalg.norm(trace.magnetic_field_body[0])
    return {
        "stored_start_nms": float(stored[0]),
        "stored_end_nms": float(stored[-1]),
        "removed_fraction": float(1.0 - stored[-1] / stored[0]),
        "initial_field_projection_start_nms": float(trace.stored_body_momentum[0] @ field),
        "initial_field_projection_end_nms": float(trace.stored_body_momentum[-1] @ field),
        "peak_dipole_am2": float(np.max(np.linalg.norm(trace.dipole, axis=1))),
    }


def build_reference() -> dict[str, Any]:
    """Compute every recorded quantity from the reference scenarios."""
    spacecraft = reference_spacecraft()
    orbit = reference_orbit()
    lqr = LinearQuadraticRegulator(
        spacecraft, LQR_WEIGHTS.attitude, LQR_WEIGHTS.rate, LQR_WEIGHTS.torque
    )
    axes = pyramid_wheel_axes()
    disturbance = _disturbance_trace(spacecraft)

    return {
        "settings": {
            "slew_duration_s": SLEW_DURATION,
            "slew_time_step_s": SLEW_STEP,
            "slew_sample_stride": SLEW_STRIDE,
            "environment_orbits": ENVIRONMENT_ORBITS,
            "environment_time_step_s": ENVIRONMENT_STEP,
            "environment_sample_stride": ENVIRONMENT_STRIDE,
        },
        "closed_form": {
            "orbit_period_s": orbit.period,
            "orbit_mean_motion_rad_s": orbit.mean_motion,
            "kepler_period_s": float(
                2.0 * np.pi * np.sqrt(orbit.radius**3 / GRAVITATIONAL_PARAMETER)
            ),
            "lqr_attitude_gain": float(lqr.gain[0, 0]),
            "lqr_attitude_gain_formula": float(
                np.sqrt(LQR_WEIGHTS.attitude / LQR_WEIGHTS.torque)
            ),
            "lqr_slowest_pole_rad_s": float(np.min(np.abs(lqr.closed_loop_poles))),
            "pyramid_gram_diagonal": float((axes @ axes.T)[0, 0]),
        },
        "slew": {
            name: _slew_summary(spacecraft, trace)
            for name, trace in _slew_traces(spacecraft).items()
        },
        "disturbance": {
            "peak_error_arcsec": float(np.max(np.rad2deg(disturbance.error_angle())) * 3600.0),
            "final_error_arcsec": float(np.rad2deg(disturbance.error_angle()[-1]) * 3600.0),
            "peak_disturbance_torque_nm": float(
                np.max(np.linalg.norm(disturbance.external_torque, axis=1))
            ),
            "stored_momentum_end_nms": float(
                np.linalg.norm(disturbance.stored_body_momentum[-1])
            ),
            "peak_wheel_speed_rpm": ManoeuvreMetrics.evaluate(
                disturbance, spacecraft
            ).peak_wheel_speed_rpm,
        },
        "dumping": {
            name: _dumping_summary(trace)
            for name, trace in _dumping_traces(reference_spacecraft()).items()
        },
    }


# Rules are matched in order, so the specific ones come first. "rel" is a
# relative tolerance and "abs" an absolute one. Each entry states where it comes
# from; none of them was chosen by looking at a difference between two runs.
_TOLERANCES: tuple[tuple[str, str, float], ...] = (
    # Values read out of the algebraic Riccati solution. LAPACK is free to
    # produce a different last digit on a different machine, and the condition
    # number of this 6 by 6 problem is small, so 1e-9 is three orders above the
    # worst backward error the solve can commit.
    ("lqr_", "rel", 1e-9),
    # A threshold crossing on a smooth decaying signal can move by one sample
    # under any perturbation at all, so the tolerance is exactly one sample.
    ("settling_time_s", "abs", SLEW_STEP * SLEW_STRIDE),
    # A count of samples inside the saturated region, movable by one sample for
    # the same reason as the settling time.
    ("saturated_samples", "abs", 1.0),
    # The final error is 3e-6 of the initial state, so a trajectory perturbation
    # of 1e-14 relative becomes 3e-9 when expressed relative to it. 1e-4 leaves
    # five orders of margin.
    ("final_error_deg", "rel", 1e-4),
    # Arithmetic on constants only: an orbit period, a matrix product of exact
    # trigonometric values. Reproducible to the last few digits everywhere.
    ("closed_form", "rel", 1e-12),
    ("initial_error_deg", "rel", 1e-12),
    # Everything else is a smooth aggregate of a contracting trajectory. A one
    # unit in the last place difference inside LAPACK perturbs the trajectory by
    # about 1e-14 relative and cannot grow, so 1e-6 leaves eight orders of
    # margin while still catching any behavioural change worth noticing.
    ("peak", "rel", 1e-6),
    ("overshoot_percent", "rel", 1e-6),
    ("removed_fraction", "rel", 1e-6),
    ("stored", "rel", 1e-6),
    ("projection", "rel", 1e-6),
    ("arcsec", "rel", 1e-6),
)


def _tolerance_for(key: str) -> tuple[str, float]:
    for pattern, kind, tolerance in _TOLERANCES:
        if pattern in key:
            return kind, tolerance
    raise AssertionError(f"no tolerance rule covers {key!r}")


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """Load the recorded reference values."""
    return dict(json.loads(REFERENCE_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def produced() -> dict[str, Any]:
    """Recompute every recorded quantity from the current code."""
    return build_reference()


def test_settings_have_not_drifted(
    reference: dict[str, Any], produced: dict[str, Any]
) -> None:
    """The reference is only meaningful if it was recorded with these settings."""
    assert produced["settings"] == reference["settings"]


@pytest.mark.parametrize("section", ["closed_form", "disturbance"])
def test_flat_sections_match_the_reference(
    reference: dict[str, Any], produced: dict[str, Any], section: str
) -> None:
    """Recorded scalars match, each within the tolerance its rule prescribes."""
    expected = reference[section]
    actual = produced[section]
    assert set(actual) == set(expected)
    for key, value in expected.items():
        kind, tolerance = _tolerance_for(f"{section}.{key}")
        if kind == "rel":
            assert actual[key] == pytest.approx(value, rel=tolerance)
        else:
            assert actual[key] == pytest.approx(value, abs=tolerance)


@pytest.mark.parametrize("section", ["slew", "dumping"])
def test_nested_sections_match_the_reference(
    reference: dict[str, Any], produced: dict[str, Any], section: str
) -> None:
    """Recorded run summaries match, run by run and quantity by quantity."""
    expected = reference[section]
    actual = produced[section]
    assert set(actual) == set(expected)
    for run, quantities in expected.items():
        assert set(actual[run]) == set(quantities)
        for key, value in quantities.items():
            if run == "quaternion PD, over-driven" and key == "final_error_deg":
                # Rounding noise at 1e-12 degrees, so only bounded, never pinned.
                assert actual[run][key] < 1e-6
                continue
            kind, tolerance = _tolerance_for(f"{section}.{key}")
            if kind == "rel":
                assert actual[run][key] == pytest.approx(value, rel=tolerance)
            else:
                assert actual[run][key] == pytest.approx(value, abs=tolerance)


def test_lqr_gain_matches_its_closed_form(produced: dict[str, Any]) -> None:
    """The recorded LQR gain agrees with the formula it is supposed to satisfy."""
    values = produced["closed_form"]
    assert values["lqr_attitude_gain"] == pytest.approx(
        values["lqr_attitude_gain_formula"], rel=1e-12
    )
    assert values["orbit_period_s"] == pytest.approx(values["kepler_period_s"], rel=1e-12)
    assert values["pyramid_gram_diagonal"] == pytest.approx(4.0 / 3.0, rel=1e-12)


def test_momentum_drift_is_bounded_rather_than_pinned() -> None:
    """The drift is accumulated rounding, so it is bounded and never recorded.

    Pinning it would pin the order in which a particular machine sums a dot
    product, which is exactly the mistake this suite is written to avoid.
    """
    spacecraft = reference_spacecraft()
    for trace in _slew_traces(spacecraft).values():
        drift = float(
            np.max(np.linalg.norm(trace.inertial_momentum - trace.inertial_momentum[0], axis=1))
        )
        stored = float(np.max(np.linalg.norm(trace.stored_body_momentum, axis=1)))
        steps = int(SLEW_DURATION / SLEW_STEP)
        assert drift < rounding_bound(steps, stored)


def test_the_fixed_field_run_conserves_the_inertial_projection() -> None:
    """The exactly conserved quantity of the fixed field run is bounded, not pinned.

    The recorded ``initial_field_projection`` values are the projection of the
    *stored wheel* momentum, which changes slightly because the body itself holds
    a little momentum during the transient. The quantity that is exactly conserved
    is the projection of the *total inertial* momentum, and its excursion is
    numerical noise, so it is bounded here rather than recorded.
    """
    traces = _dumping_traces(reference_spacecraft())
    trace = traces["fixed field"]
    axis = dcm_from_quaternion(trace.quaternion[0]).T @ trace.magnetic_field_body[0]
    axis = axis / np.linalg.norm(axis)
    projection = trace.inertial_momentum @ axis
    scale = float(np.linalg.norm(trace.inertial_momentum[0]))
    bound = truncation_bound(
        PD_NATURAL_FREQUENCY,
        ENVIRONMENT_STEP,
        ENVIRONMENT_ORBITS * reference_orbit().period,
        scale,
    )
    assert float(np.max(np.abs(projection - projection[0]))) < bound


def main() -> None:
    """Recompute the reference file and write it to disk."""
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_reference()
    REFERENCE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
