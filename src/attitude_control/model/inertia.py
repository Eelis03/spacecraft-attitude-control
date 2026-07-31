"""Inertia tensors and reaction wheel array geometry.

The spacecraft inertia tensor used throughout this package is the tensor of the
*whole* vehicle about its centre of mass, including the reaction wheels sitting at
their nominal positions. Because a reaction wheel is axisymmetric about its spin
axis, spinning it does not change that tensor, so the tensor is constant and the
wheel contribution to angular momentum reduces to a single scalar per wheel. That
is the bookkeeping convention of Hughes (2004), chapter 6, and it is what makes
:func:`attitude_control.model.dynamics.body_angular_momentum` correct.

References
----------
Hughes, P. C. (2004). *Spacecraft Attitude Dynamics*. Dover. ISBN 978-0486439259.

Markley, F. L., Reynolds, R. G., Liu, F. X. and Lebsock, K. L. (2010). Maximum
torque and momentum envelopes for reaction wheel arrays. *Journal of Guidance,
Control, and Dynamics*, 33(5), 1606-1614. DOI 10.2514/1.47968.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.numeric import FloatArray, as_matrix, as_vector

__all__ = [
    "ISOTROPIC_PYRAMID_HALF_ANGLE",
    "Spacecraft",
    "WheelArray",
    "principal_inertia",
    "pyramid_wheel_axes",
    "symmetric_inertia",
]

# Four wheels on a cone about the body z axis give an isotropic array, meaning
# W W^T is a multiple of the identity, when 2 sin^2(b) = 4 cos^2(b), that is
# tan^2(b) = 2. The array then has the same torque gain about every body axis.
ISOTROPIC_PYRAMID_HALF_ANGLE: Final[float] = float(np.arctan(np.sqrt(2.0)))


def symmetric_inertia(
    diagonal: ArrayLike,
    products: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Build an inertia tensor from ``(Ixx, Iyy, Izz)`` and ``(Ixy, Ixz, Iyz)``.

    The products of inertia are entered with the sign they carry in the tensor,
    so a tensor entry of ``-2`` is passed as ``-2``. The result is checked for
    positive definiteness, which every physical inertia tensor satisfies.
    """
    d = as_vector(diagonal, 3)
    p = as_vector(products, 3)
    tensor = np.array(
        [[d[0], p[0], p[1]], [p[0], d[1], p[2]], [p[1], p[2], d[2]]],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(tensor)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"inertia tensor is not positive definite: eigenvalues {eigenvalues}")
    # The triangle inequality on principal moments is a necessary condition for a
    # real mass distribution; violating it means the tensor cannot be realised.
    a, b, c = (float(v) for v in eigenvalues)
    if a + b <= c:
        raise ValueError(f"principal moments violate the triangle inequality: {eigenvalues}")
    return tensor


def principal_inertia(tensor: ArrayLike) -> FloatArray:
    """Return the three principal moments of inertia in ascending order."""
    return np.linalg.eigvalsh(as_matrix(tensor, (3, 3)))


def pyramid_wheel_axes(
    half_angle: float = ISOTROPIC_PYRAMID_HALF_ANGLE,
    azimuth_offset: float = 0.25 * np.pi,
) -> FloatArray:
    """Return the ``3 x 4`` spin axis matrix of a four wheel pyramid.

    The wheels sit on a cone of half angle ``half_angle`` about the body z axis at
    azimuths spaced 90 degrees apart. With the default half angle the array is
    isotropic and has one null direction, ``(1, -1, 1, -1) / 2``, along which
    wheel momentum can be redistributed without producing any body torque.
    """
    azimuths = azimuth_offset + 0.5 * np.pi * np.arange(4, dtype=np.float64)
    return np.array(
        [
            np.sin(half_angle) * np.cos(azimuths),
            np.sin(half_angle) * np.sin(azimuths),
            np.full(4, np.cos(half_angle)),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class WheelArray:
    """A set of reaction wheels with fixed spin axes and per wheel limits.

    Attributes
    ----------
    axes:
        ``3 x n`` distribution matrix whose columns are unit spin axes in body
        components.
    axial_inertia:
        ``n`` axial moments of inertia in kg m^2.
    max_torque:
        Motor torque limit per wheel in N m.
    max_momentum:
        Stored momentum limit per wheel in N m s, which is the axial inertia
        times the maximum wheel speed.
    """

    axes: FloatArray
    axial_inertia: FloatArray
    max_torque: float
    max_momentum: float
    pseudoinverse: FloatArray = field(init=False)
    null_projector: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        axes = np.asarray(self.axes, dtype=np.float64)
        if axes.ndim != 2 or axes.shape[0] != 3:
            raise ValueError(f"axes must have shape (3, n), got {axes.shape}")
        norms = np.linalg.norm(axes, axis=0)
        if not np.allclose(norms, 1.0, atol=1e-12):
            raise ValueError("every wheel spin axis must be a unit vector")
        if np.linalg.matrix_rank(axes) != 3:
            raise ValueError("the wheel array must span three dimensions to be three axis capable")
        inertia = as_vector(self.axial_inertia, axes.shape[1])
        if float(np.min(inertia)) <= 0.0:
            raise ValueError("every wheel axial inertia must be positive")
        if self.max_torque <= 0.0 or self.max_momentum <= 0.0:
            raise ValueError("wheel limits must be positive")

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "axial_inertia", inertia)
        # The minimum norm right inverse. Because the array has full row rank,
        # axes @ pseudoinverse is the identity to machine precision, which is why
        # allocation reproduces a commanded torque exactly.
        object.__setattr__(self, "pseudoinverse", np.linalg.pinv(axes))
        object.__setattr__(
            self,
            "null_projector",
            np.eye(axes.shape[1]) - np.linalg.pinv(axes) @ axes,
        )

    @property
    def count(self) -> int:
        """Number of wheels in the array."""
        return int(self.axes.shape[1])

    @property
    def max_speed(self) -> float:
        """Largest wheel speed in rad/s, taken over the wheels."""
        return float(np.max(self.max_momentum / self.axial_inertia))

    def speed(self, wheel_momentum: ArrayLike) -> FloatArray:
        """Convert stored wheel momentum in N m s to wheel speed in rad/s."""
        return as_vector(wheel_momentum, self.count) / self.axial_inertia

    def momentum(self, wheel_speed: ArrayLike) -> FloatArray:
        """Convert wheel speed in rad/s to stored wheel momentum in N m s."""
        return as_vector(wheel_speed, self.count) * self.axial_inertia


@dataclass(frozen=True, slots=True)
class Spacecraft:
    """A rigid body carrying a reaction wheel array."""

    inertia: FloatArray
    wheels: WheelArray
    inertia_inverse: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        tensor = as_matrix(self.inertia, (3, 3))
        if not np.allclose(tensor, tensor.T, atol=1e-12):
            raise ValueError("the inertia tensor must be symmetric")
        if float(np.linalg.eigvalsh(tensor)[0]) <= 0.0:
            raise ValueError("the inertia tensor must be positive definite")
        object.__setattr__(self, "inertia", tensor)
        object.__setattr__(self, "inertia_inverse", np.linalg.inv(tensor))
