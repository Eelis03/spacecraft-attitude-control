"""Analysis layer: metrics and figures computed from recorded traces.

Importing :mod:`attitude_control.analysis.figures` pulls in matplotlib, so it is
not re-exported here; example scripts import it directly when they need it.
"""

from __future__ import annotations

from attitude_control.analysis.metrics import (
    ManoeuvreMetrics,
    mean_error_vector,
    momentum_drift,
    settling_time,
    signed_error_angle,
)
from attitude_control.analysis.report import (
    DUMPING_HEADER,
    dumping_summary,
    format_table,
    metrics_table,
)
from attitude_control.analysis.representation import (
    round_trip_residuals,
    worst_orthonormality_defect,
)

__all__ = [
    "DUMPING_HEADER",
    "ManoeuvreMetrics",
    "dumping_summary",
    "format_table",
    "mean_error_vector",
    "metrics_table",
    "momentum_drift",
    "round_trip_residuals",
    "settling_time",
    "signed_error_angle",
    "worst_orthonormality_defect",
]
