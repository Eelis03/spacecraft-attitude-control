"""Quaternion attitude dynamics with reaction wheel LQR and PD control and momentum dumping.

The package is split into five layers, each importing only from the ones below it:

``attitude_control.model``
    Attitude representations, inertia, rigid body and wheel dynamics, and the
    environment models. Pure functions, no state and no input or output.
``attitude_control.algorithm``
    Control laws behind one protocol, wheel torque allocation, and magnetic
    momentum management. No integration and no plotting.
``attitude_control.pipeline``
    The fixed step integrator and the scenario runner that produces a trace.
``attitude_control.analysis``
    Metrics and figures computed from a trace.
``attitude_control.configuration``
    The reference vehicle and the three scenarios the examples and tests share.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
