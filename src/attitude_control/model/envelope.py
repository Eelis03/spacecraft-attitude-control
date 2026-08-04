"""The torque and momentum envelopes of a redundant reaction wheel array.

The body torques an array can deliver are the image of the per wheel torque box
under the distribution matrix,

    T = { -W u : |u_i| <= tau }

which is the Minkowski sum of the segments ``[-tau w_i, tau w_i]``, that is a
zonotope: a convex polytope symmetric about the origin. Replacing the motor
torque limit by the stored momentum limit gives the momentum envelope, the set of
body momenta ``W h_w`` the array can hold. The two sets differ by that scale
factor and nothing else, so everything below applies to both.

Support and faces
-----------------
The support function of a zonotope has a closed form,

    h(n) = tau sum_i |w_i . n|

and every face of one in three dimensions is spanned by two of its generators, so
the candidate normals are the cross products of the pairs of spin axes. The
envelope is therefore exactly the intersection of the slabs ``|L . n| <= h(n)``
over those pairs, which needs no optimisation and gives three quantities at once:

* the gauge ``max_n |L . n| / h(n)``, which is one on the boundary and larger
  outside, so a single number says how much of the envelope a command uses;
* the reach about a direction, the reciprocal of the gauge of the unit vector
  along it, which is a pure torque about that axis rather than the largest
  available component along it;
* the radius guaranteed about *every* direction, which is that of the inscribed
  sphere and therefore ``min_n h(n)``, because the nearest point of the boundary
  lies on a face.

For the isotropic pyramid the four spin axes are four body diagonals of a cube,
the envelope is a rhombic dodecahedron, and all three are exact: ``4 tau /
sqrt(3)`` about any body axis, ``2 tau`` about any spin axis, and
``2 sqrt(2/3) tau`` about the worst direction, which is a face normal.

What the minimum norm solution reaches
--------------------------------------
The envelope is what *some* distribution of motor torques delivers, which is more
than the minimum norm solution ``-W^+ L`` delivers, because the limits are per
wheel and therefore an infinity norm constraint. About a spin axis that solution
puts three quarters of the demand on one wheel, so it is guaranteed to fit inside
the limits only out to ``4 tau / 3``, short of the guaranteed radius above by a
factor of ``sqrt(3/2)``. Commands in that band are achievable and are still
reported as saturated by the allocation layer, which applies that solution. The
design notes record why the linear program that would close the gap was rejected;
what these functions add is that the gap can be measured rather than guessed.

References
----------
Markley, F. L., Reynolds, R. G., Liu, F. X. and Lebsock, K. L. (2010). Maximum
torque and momentum envelopes for reaction wheel arrays. *Journal of Guidance,
Control, and Dynamics*, 33(5), 1606-1614. DOI 10.2514/1.47968.

Wie, B. (2008). *Space Vehicle Dynamics and Control*, 2nd edition. AIAA,
chapter 7. DOI 10.2514/4.860119.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.model.inertia import WheelArray
from attitude_control.numeric import FloatArray, as_vector, unit

__all__ = [
    "envelope_normals",
    "guaranteed_momentum",
    "guaranteed_torque",
    "maximum_momentum_about",
    "maximum_torque_about",
    "momentum_utilisation",
    "torque_utilisation",
]

# Spin axes closer together in angle than this span no face, and the direction of
# their cross product would be rounding noise rather than geometry.
_PARALLEL_TOLERANCE: Final[float] = 1.0e-9


def envelope_normals(wheels: WheelArray) -> FloatArray:
    """Return the unit normals of the envelope faces, one per row.

    Parallel pairs of spin axes span nothing and are dropped. For an array whose
    axes are not in general position the result is a superset of the true face
    normals, which costs a redundant slab and never an incorrect one, because
    every slab is an exact bound on the achievable set whether it touches it or
    not.
    """
    axes = wheels.axes
    normals: list[FloatArray] = []
    for first in range(wheels.count - 1):
        for second in range(first + 1, wheels.count):
            normal = np.cross(axes[:, first], axes[:, second])
            length = float(np.linalg.norm(normal))
            if length > _PARALLEL_TOLERANCE:
                normals.append(normal / length)
    return np.array(normals, dtype=np.float64)


def _support(wheels: WheelArray, limit: float, normals: FloatArray) -> FloatArray:
    """Return the support function of the envelope on each row of ``normals``."""
    return np.asarray(limit * np.sum(np.abs(normals @ wheels.axes), axis=1), dtype=np.float64)


def _gauge(wheels: WheelArray, limit: float, vector: ArrayLike) -> float:
    """Return the gauge of the envelope at ``vector``.

    The gauge is the factor the envelope would have to be scaled by for ``vector``
    to sit on its boundary: one on the boundary, below one inside, above one
    outside, and zero only at the origin. It is exact because the envelope is the
    intersection of the slabs its faces define.
    """
    normals = envelope_normals(wheels)
    projections = np.abs(normals @ as_vector(vector, 3))
    return float(np.max(projections / _support(wheels, limit, normals)))


def torque_utilisation(wheels: WheelArray, commanded_torque: ArrayLike) -> float:
    """Return the fraction of the torque envelope a commanded body torque uses.

    Above one the command is outside the envelope, so no distribution of motor
    torques realises it and the wheels must fall short whatever the allocation.
    Below one some distribution does realise it, which is a weaker statement than
    the minimum norm one fitting inside the limits.
    """
    return _gauge(wheels, wheels.max_torque, commanded_torque)


def momentum_utilisation(wheels: WheelArray, stored_momentum: ArrayLike) -> float:
    """Return the fraction of the momentum envelope a stored body momentum uses.

    The argument is the body momentum ``W h_w`` rather than the per wheel
    momentum, so this reports how close the array is to being unable to absorb any
    more in the direction it is already loaded.
    """
    return _gauge(wheels, wheels.max_momentum, stored_momentum)


def maximum_torque_about(wheels: WheelArray, direction: ArrayLike) -> float:
    """Return the largest torque the array can deliver about ``direction``, in N m.

    The torque is a pure one about that axis, so this is the distance from the
    origin to the envelope along it and not the largest available component, which
    would come with torque about the other two axes as well.
    """
    return 1.0 / _gauge(wheels, wheels.max_torque, unit(as_vector(direction, 3)))


def maximum_momentum_about(wheels: WheelArray, direction: ArrayLike) -> float:
    """Return the largest momentum the array can hold about ``direction``, in N m s."""
    return 1.0 / _gauge(wheels, wheels.max_momentum, unit(as_vector(direction, 3)))


def guaranteed_torque(wheels: WheelArray) -> float:
    """Return the torque magnitude available about every body direction, in N m.

    This is the radius of the largest sphere inside the envelope, so a command no
    larger than it is realisable whichever way it points. It is the number a
    control design should be sized against, because the direction the demand falls
    in is chosen by the manoeuvre rather than by the array.
    """
    return float(np.min(_support(wheels, wheels.max_torque, envelope_normals(wheels))))


def guaranteed_momentum(wheels: WheelArray) -> float:
    """Return the momentum the array can hold about every body direction, in N m s."""
    return float(np.min(_support(wheels, wheels.max_momentum, envelope_normals(wheels))))
