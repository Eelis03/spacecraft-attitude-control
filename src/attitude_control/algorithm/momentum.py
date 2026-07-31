"""Magnetic momentum dumping with torque rods.

A reaction wheel array absorbs the secular part of every environmental torque, so
its stored momentum grows without bound unless an external torque removes it.
Magnetic torque rods provide that external torque by driving a dipole moment ``m``
against the geomagnetic field ``B``, producing ``L = m x B``.

The cross product law
---------------------
To remove stored momentum ``h`` the desired external torque is ``-k h``. The
choice

    m = (k / |B|^2) (h x B)

gives

    L = m x B = -k (h - B_hat (B_hat . h))

that is, exactly minus ``k`` times the component of ``h`` perpendicular to the
field. This is the cross product unloading law analysed by Camillo and Markley
(1980).

The limitation this makes visible
---------------------------------
``L = m x B`` is orthogonal to ``B`` for every possible ``m``. No dipole can
produce a torque about the field direction, so at any instant the component of
stored momentum along ``B`` cannot be changed at all. Two consequences follow and
both appear in the results of this package rather than being asserted:

* with a field direction that is fixed, the along-field component of momentum is
  exactly conserved and dumping stalls with that component intact;
* along a real orbit the field direction sweeps through a large solid angle, so
  the uncontrollable direction moves and the full momentum vector can be removed
  over a fraction of an orbit. The rate at which that happens is set by how much
  the field direction turns, which is why magnetic dumping is slow in a near
  equatorial orbit and faster in an inclined one.

References
----------
Camillo, P. J. and Markley, F. L. (1980). Orbit-averaged behaviour of magnetic
control laws for momentum unloading. *Journal of Guidance and Control*, 3(6),
563-568. DOI 10.2514/3.56036.

Stickler, A. C. and Alfriend, K. T. (1976). Elementary magnetic attitude control
system. *Journal of Spacecraft and Rockets*, 13(5), 282-287. DOI 10.2514/3.57089.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.model.environment import magnetic_torque
from attitude_control.numeric import FloatArray, as_vector

__all__ = [
    "MagneticDumping",
    "controllable_momentum",
    "cross_product_dipole",
    "uncontrollable_momentum",
]


def cross_product_dipole(
    excess_momentum: ArrayLike,
    field_body: ArrayLike,
    gain: float,
    max_dipole: float | None = None,
) -> FloatArray:
    """Return the dipole moment of the cross product unloading law, in A m^2.

    ``gain`` has units of 1/s and sets the time constant with which the
    controllable part of the momentum decays. When ``max_dipole`` is given the
    result is scaled down, not clipped component by component, so that the torque
    direction is preserved when the rods saturate.
    """
    momentum = as_vector(excess_momentum, 3)
    field = as_vector(field_body, 3)
    squared = float(np.dot(field, field))
    if squared == 0.0:
        return np.zeros(3, dtype=np.float64)
    if gain < 0.0:
        raise ValueError("dumping gain must be non-negative")
    dipole = (gain / squared) * np.cross(momentum, field)
    if max_dipole is not None:
        if max_dipole <= 0.0:
            raise ValueError("maximum dipole must be positive")
        magnitude = float(np.linalg.norm(dipole))
        if magnitude > max_dipole:
            dipole = dipole * (max_dipole / magnitude)
    return dipole


def uncontrollable_momentum(momentum: ArrayLike, field_body: ArrayLike) -> FloatArray:
    """Return the component of ``momentum`` along the field, which cannot be removed."""
    h = as_vector(momentum, 3)
    field = as_vector(field_body, 3)
    squared = float(np.dot(field, field))
    if squared == 0.0:
        return np.zeros(3, dtype=np.float64)
    return field * (float(np.dot(h, field)) / squared)


def controllable_momentum(momentum: ArrayLike, field_body: ArrayLike) -> FloatArray:
    """Return the component of ``momentum`` perpendicular to the field."""
    return as_vector(momentum, 3) - uncontrollable_momentum(momentum, field_body)


@dataclass(frozen=True, slots=True)
class MagneticDumping:
    """Cross product momentum unloading with a torque rod dipole limit.

    Parameters
    ----------
    gain:
        Unloading gain in 1/s. The perpendicular component of stored momentum
        decays with time constant ``1 / gain`` while the rods are unsaturated.
    max_dipole:
        Largest dipole magnitude the rods can produce, in A m^2.
    target_momentum:
        Stored body momentum the law drives towards, in N m s. Usually zero.
    """

    gain: float
    max_dipole: float
    target_momentum: FloatArray | None = None

    def dipole(self, stored_momentum: ArrayLike, field_body: ArrayLike) -> FloatArray:
        """Return the commanded dipole moment in A m^2."""
        excess = as_vector(stored_momentum, 3)
        if self.target_momentum is not None:
            excess = excess - as_vector(self.target_momentum, 3)
        return cross_product_dipole(excess, field_body, self.gain, self.max_dipole)

    def torque(self, stored_momentum: ArrayLike, field_body: ArrayLike) -> FloatArray:
        """Return the external body torque produced by the rods, in N m."""
        return magnetic_torque(self.dipole(stored_momentum, field_body), field_body)
