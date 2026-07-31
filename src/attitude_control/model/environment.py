"""Orbit geometry, gravity gradient torque, and a tilted dipole magnetic field.

The environment models here are the smallest ones that reproduce the two effects
this package needs to demonstrate: a disturbance torque with both a periodic and a
secular part, and a magnetic field whose direction sweeps around the orbit so that
magnetic momentum dumping is possible at all.

Gravity gradient torque
-----------------------
For a rigid body at geocentric distance ``r`` with nadir unit vector ``n``
expressed in body components,

    L_gg = 3 mu / r^3 * (n x J n)

This is the leading term of the expansion of the gravitational force over the
body, exact to first order in the ratio of body size to orbit radius. See Wertz
(1978), section 17.2, or Hughes (2004), section 8.3.

Magnetic field
--------------
A centred tilted dipole,

    B(r) = (B0 R^3 / r^3) * (3 (m . r_hat) r_hat - m)

with ``m`` the unit dipole axis, ``B0`` the mean equatorial surface field and
``R`` the Earth radius. The dipole axis points approximately towards geographic
south, so that the field at the north pole points into the Earth, and is tilted
about 11 degrees from the rotation axis. The model omits the higher order terms
of the real geomagnetic field and its secular variation; the field magnitude is
accurate to roughly 20 per cent and the direction to a few degrees, which is
enough for control law behaviour but not for on-board use.

References
----------
Wertz, J. R., editor (1978). *Spacecraft Attitude Determination and Control*.
Springer. DOI 10.1007/978-94-009-9907-7.

Hughes, P. C. (2004). *Spacecraft Attitude Dynamics*. Dover. ISBN 978-0486439259.

Alken, P. et al. (2021). International Geomagnetic Reference Field: the
thirteenth generation. *Earth, Planets and Space*, 73, 49. DOI
10.1186/s40623-020-01288-x. Source of the dipole tilt and mean field strength
quoted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.model.attitude import dcm_from_quaternion
from attitude_control.model.inertia import Spacecraft
from attitude_control.numeric import FloatArray, as_vector, unit

__all__ = [
    "DIPOLE_TILT",
    "EARTH_MEAN_FIELD",
    "EARTH_RADIUS",
    "EARTH_ROTATION_RATE",
    "GRAVITATIONAL_PARAMETER",
    "CircularOrbit",
    "GravityGradient",
    "TiltedDipoleField",
    "gravity_gradient_torque",
    "magnetic_torque",
]

# WGS 84 / EGM96 gravitational parameter of the Earth, m^3 s^-2.
GRAVITATIONAL_PARAMETER: Final[float] = 3.986004418e14
# WGS 84 equatorial radius, m.
EARTH_RADIUS: Final[float] = 6378137.0
# Mean equatorial surface field of the IGRF dipole term, T.
EARTH_MEAN_FIELD: Final[float] = 3.12e-5
# Tilt of the dipole axis from the rotation axis, rad.
DIPOLE_TILT: Final[float] = np.deg2rad(11.4)
# Sidereal rotation rate of the Earth, rad/s.
EARTH_ROTATION_RATE: Final[float] = 7.292115e-5


@dataclass(frozen=True, slots=True)
class CircularOrbit:
    """A circular Keplerian orbit described by radius, inclination, and node.

    The orbit is expressed in an inertial frame whose third axis is the Earth
    rotation axis. Argument of latitude is measured from the ascending node.
    """

    radius: float
    inclination: float
    raan: float = 0.0
    initial_argument: float = 0.0
    gravitational_parameter: float = GRAVITATIONAL_PARAMETER
    mean_motion: float = field(init=False)

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("orbit radius must be positive")
        rate = float(np.sqrt(self.gravitational_parameter / self.radius**3))
        object.__setattr__(self, "mean_motion", rate)

    @property
    def period(self) -> float:
        """Orbit period in seconds."""
        return float(2.0 * np.pi / self.mean_motion)

    def position(self, time: float) -> FloatArray:
        """Return the inertial position vector in metres at ``time``."""
        argument = self.initial_argument + self.mean_motion * time
        cos_u, sin_u = np.cos(argument), np.sin(argument)
        cos_i, sin_i = np.cos(self.inclination), np.sin(self.inclination)
        cos_o, sin_o = np.cos(self.raan), np.sin(self.raan)
        return self.radius * np.array(
            [
                cos_u * cos_o - sin_u * cos_i * sin_o,
                cos_u * sin_o + sin_u * cos_i * cos_o,
                sin_u * sin_i,
            ],
            dtype=np.float64,
        )


def gravity_gradient_torque(
    spacecraft: Spacecraft,
    nadir_body: ArrayLike,
    radius: float,
    gravitational_parameter: float = GRAVITATIONAL_PARAMETER,
) -> FloatArray:
    """Return the gravity gradient torque in body components.

    ``nadir_body`` is the unit vector from the spacecraft towards the centre of
    the Earth, expressed in body components. Its sign does not matter because the
    expression is quadratic in it.
    """
    nadir = unit(as_vector(nadir_body, 3))
    coefficient = 3.0 * gravitational_parameter / radius**3
    return coefficient * np.cross(nadir, spacecraft.inertia @ nadir)


def magnetic_torque(dipole_moment: ArrayLike, field_body: ArrayLike) -> FloatArray:
    """Return the torque ``m x B`` produced by a magnetic dipole in a field.

    The result is orthogonal to ``field_body`` by construction. No choice of
    ``dipole_moment`` can produce a torque component along the field, which is the
    fundamental limitation of magnetic actuation.
    """
    return np.cross(as_vector(dipole_moment, 3), as_vector(field_body, 3))


@dataclass(frozen=True, slots=True)
class GravityGradient:
    """Gravity gradient torque along a circular orbit."""

    spacecraft: Spacecraft
    orbit: CircularOrbit

    def torque(self, time: float, quaternion: ArrayLike) -> FloatArray:
        """Return the gravity gradient torque in body components at ``time``."""
        position = self.orbit.position(time)
        nadir_body = dcm_from_quaternion(quaternion) @ (-position)
        return gravity_gradient_torque(
            self.spacecraft,
            nadir_body,
            self.orbit.radius,
            self.orbit.gravitational_parameter,
        )


@dataclass(frozen=True, slots=True)
class TiltedDipoleField:
    """A centred tilted dipole geomagnetic field sampled along a circular orbit."""

    orbit: CircularOrbit
    tilt: float = DIPOLE_TILT
    mean_field: float = EARTH_MEAN_FIELD
    reference_radius: float = EARTH_RADIUS
    rotation_rate: float = EARTH_ROTATION_RATE
    initial_longitude: float = 0.0

    def dipole_axis(self, time: float) -> FloatArray:
        """Return the unit dipole axis in inertial components at ``time``.

        The axis points approximately towards geographic south and precesses with
        the Earth's rotation.
        """
        longitude = self.initial_longitude + self.rotation_rate * time
        return np.array(
            [
                -np.sin(self.tilt) * np.cos(longitude),
                -np.sin(self.tilt) * np.sin(longitude),
                -np.cos(self.tilt),
            ],
            dtype=np.float64,
        )

    def field_inertial(self, time: float) -> FloatArray:
        """Return the magnetic flux density in inertial components, in tesla."""
        position = self.orbit.position(time)
        radius = float(np.linalg.norm(position))
        direction = position / radius
        axis = self.dipole_axis(time)
        scale = self.mean_field * (self.reference_radius / radius) ** 3
        return scale * (3.0 * float(np.dot(axis, direction)) * direction - axis)

    def field_body(self, time: float, quaternion: ArrayLike) -> FloatArray:
        """Return the magnetic flux density in body components, in tesla."""
        return dcm_from_quaternion(quaternion) @ self.field_inertial(time)


@dataclass(frozen=True, slots=True)
class ConstantField:
    """A magnetic field fixed in the inertial frame.

    Used to isolate the along-field limitation of magnetic actuation: with the
    field direction frozen, the component of momentum along it can never be
    removed, whereas an orbiting spacecraft sees the direction sweep and can
    remove every component over time.
    """

    vector: FloatArray

    def field_inertial(self, time: float) -> FloatArray:
        """Return the constant field in inertial components, in tesla."""
        del time
        return np.asarray(self.vector, dtype=np.float64)

    def field_body(self, time: float, quaternion: ArrayLike) -> FloatArray:
        """Return the constant field in body components, in tesla."""
        return dcm_from_quaternion(quaternion) @ self.field_inertial(time)
