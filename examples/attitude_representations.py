"""Round trip one attitude through every representation and report the residuals.

The script also shows the two places where a representation is not unique: the
quaternion sign ambiguity, and the modified Rodrigues parameter shadow set.

    uv run python examples/attitude_representations.py
"""

from __future__ import annotations

import argparse

import numpy as np

from attitude_control.analysis.report import format_table
from attitude_control.analysis.representation import (
    round_trip_residuals,
    worst_orthonormality_defect,
)
from attitude_control.model.attitude import (
    dcm_from_mrp,
    dcm_from_quaternion,
    is_rotation_matrix,
    mrp_from_quaternion,
    mrp_shadow,
    quaternion_from_axis_angle,
)

_SEED = 20260731


def parse_arguments() -> argparse.Namespace:
    """Return the command line options."""
    parser = argparse.ArgumentParser(description="Attitude representation round trips")
    parser.add_argument("--quick", action="store_true", help="fewer sample rotations")
    parser.add_argument("--no-figure", action="store_true", help="accepted for consistency")
    return parser.parse_args()


def main() -> None:
    """Report round trip residuals and the two non-uniqueness cases."""
    options = parse_arguments()
    samples = 200 if options.quick else 20000

    residuals = round_trip_residuals(samples, _SEED)
    defect = worst_orthonormality_defect(samples, _SEED)

    rows = [(name, f"{value:.3e}", f"{np.rad2deg(value):.3e}") for name, value in residuals.items()]
    print(f"{samples} random rotations, worst round trip error as a principal angle")
    print(format_table(("path", "error [rad]", "error [deg]"), rows))
    print(f"\nworst departure from orthonormality: {defect:.3e}")

    quaternion = quaternion_from_axis_angle((1.0, 2.0, 2.0), np.deg2rad(150.0))
    flipped = -quaternion
    matrix = dcm_from_quaternion(quaternion)
    print("\nquaternion sign ambiguity")
    print(f"  q      = {np.array2string(quaternion, precision=6)}")
    print(f"  -q     = {np.array2string(flipped, precision=6)}")
    print(
        "  identical attitude matrix to "
        f"{np.max(np.abs(matrix - dcm_from_quaternion(flipped))):.3e}"
    )
    print(f"  both give a proper rotation: {is_rotation_matrix(dcm_from_quaternion(flipped))}")

    parameters = mrp_from_quaternion(quaternion)
    shadow = mrp_shadow(parameters)
    print("\nmodified Rodrigues parameter shadow set")
    print(
        f"  s      = {np.array2string(parameters, precision=6)}, "
        f"norm {np.linalg.norm(parameters):.6f}"
    )
    print(f"  shadow = {np.array2string(shadow, precision=6)}, norm {np.linalg.norm(shadow):.6f}")
    print(f"  product of the two norms: {np.linalg.norm(parameters) * np.linalg.norm(shadow):.6f}")
    print(
        "  identical attitude matrix to "
        f"{np.max(np.abs(dcm_from_mrp(parameters) - dcm_from_mrp(shadow))):.3e}"
    )


if __name__ == "__main__":
    main()
