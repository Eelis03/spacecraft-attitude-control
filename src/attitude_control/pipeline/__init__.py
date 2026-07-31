"""Pipeline layer: the integrator and the scenario runner.

This layer owns time. It advances the plant, calls the algorithm layer, and
records a structured trace, without interpreting the result or drawing anything.
"""

from __future__ import annotations

from attitude_control.pipeline.integrator import (
    Derivative,
    normalise_quaternion_state,
    rk4_step,
    step_norm_drift,
)
from attitude_control.pipeline.scenario import (
    DisturbanceModel,
    MagneticEnvironment,
    ScenarioConfig,
    ScenarioTrace,
    run_scenario,
)

__all__ = [
    "Derivative",
    "DisturbanceModel",
    "MagneticEnvironment",
    "ScenarioConfig",
    "ScenarioTrace",
    "normalise_quaternion_state",
    "rk4_step",
    "run_scenario",
    "step_norm_drift",
]
