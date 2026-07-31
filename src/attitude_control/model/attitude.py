"""Attitude representations and the conversions between them.

Four representations are provided, together with conversions in both directions:

* unit quaternions, scalar first, ``q = (q0, q1, q2, q3)``;
* direction cosine matrices (DCM), also called attitude matrices;
* modified Rodrigues parameters (MRP), including the shadow set;
* 3-2-1 (yaw, pitch, roll) Euler angles.

Conventions
-----------
Every DCM in this package is the *attitude matrix* that maps a vector expressed in
the reference (inertial) frame to the same vector expressed in the body frame::

    v_body = dcm_from_quaternion(q) @ v_inertial

The quaternion follows the Hamilton product convention with the scalar part first.
With that convention the attitude matrix is

    A(q) = (q0^2 - qv . qv) I + 2 qv qv^T - 2 q0 [qv x]

and the kinematic equation driven by the body rate ``w`` is

    q_dot = 0.5 * q (x) (0, w)

which expands to ``q0_dot = -0.5 qv . w`` and ``qv_dot = 0.5 (q0 w + qv x w)``.
These are equations (2.88) and (3.20) of Markley and Crassidis (2014) written in
scalar-first order.

Sign ambiguity
--------------
``q`` and ``-q`` describe the same physical attitude, since ``A(q) = A(-q)``. The
ambiguity is real and cannot be removed from the algebra, so it is handled
explicitly: :func:`quaternion_canonical` selects the representative with a
non-negative scalar part, which is the one whose principal rotation angle lies in
``[0, pi]``. Feedback laws must apply that choice to the *error* quaternion, or a
180 degree slew can be commanded where a 180 degree slew the other way is shorter.

Shadow set
----------
The MRP vector ``s = qv / (1 + q0)`` is singular at a principal angle of 360
degrees. The shadow set ``s_shadow = -s / (s . s)`` describes the same attitude
and is singular at 0 degrees instead. Switching to whichever of the two has norm
at most one keeps the parameters bounded by ``tan(pi/4) = 1`` and away from both
singularities; see Schaub and Junkins (2018), section 3.6.

References
----------
Markley, F. L. and Crassidis, J. L. (2014). *Fundamentals of Spacecraft Attitude
Determination and Control*. Springer. DOI 10.1007/978-1-4939-0802-8.

Schaub, H. and Junkins, J. L. (2018). *Analytical Mechanics of Space Systems*,
4th edition. AIAA. DOI 10.2514/4.105210.

Shepperd, S. W. (1978). Quaternion from rotation matrix. *Journal of Guidance and
Control*, 1(3), 223-224. DOI 10.2514/3.55767b.

Shuster, M. D. (1993). A survey of attitude representations. *Journal of the
Astronautical Sciences*, 41(4), 439-517.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from attitude_control.numeric import FloatArray, as_matrix, as_vector, unit

__all__ = [
    "MRP_SHADOW_NORM",
    "attitude_error",
    "dcm_from_euler321",
    "dcm_from_mrp",
    "dcm_from_quaternion",
    "euler321_from_dcm",
    "euler321_from_quaternion",
    "is_rotation_matrix",
    "mrp_derivative",
    "mrp_from_dcm",
    "mrp_from_quaternion",
    "mrp_shadow",
    "mrp_short_rotation",
    "principal_angle",
    "quaternion_canonical",
    "quaternion_conjugate",
    "quaternion_derivative",
    "quaternion_from_axis_angle",
    "quaternion_from_dcm",
    "quaternion_from_euler321",
    "quaternion_from_mrp",
    "quaternion_identity",
    "quaternion_multiply",
    "quaternion_normalise",
    "rotation_vector",
    "signed_angle_about",
    "skew",
]

# An MRP set and its shadow have norms whose product is one, so the two sets meet
# at norm one, which is the principal angle of 180 degrees.
MRP_SHADOW_NORM: float = 1.0

# Below this |sin(theta/2)| the principal axis of a quaternion is numerically
# undefined. The value is the square root of the double-precision epsilon, the
# usual choice for a first-order cancellation threshold.
_SMALL_ANGLE: float = float(np.sqrt(np.finfo(np.float64).eps))

# Euler extraction is ill conditioned when the pitch angle approaches +/- 90
# degrees, where yaw and roll describe the same rotation. The threshold is again
# the square root of the machine epsilon expressed on the cosine of the pitch.
_GIMBAL_LOCK: float = 1.0 - _SMALL_ANGLE


def skew(vector: ArrayLike) -> FloatArray:
    """Return the skew-symmetric matrix ``[v x]`` satisfying ``[v x] u = v x u``."""
    v = as_vector(vector, 3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def quaternion_identity() -> FloatArray:
    """Return the identity quaternion, which is the zero rotation."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quaternion_normalise(quaternion: ArrayLike) -> FloatArray:
    """Return ``quaternion`` scaled to unit norm."""
    return unit(as_vector(quaternion, 4))


def quaternion_canonical(quaternion: ArrayLike) -> FloatArray:
    """Return the representative of ``quaternion`` whose scalar part is non-negative.

    ``q`` and ``-q`` are the same attitude. Choosing ``q0 >= 0`` fixes the
    principal rotation angle to ``[0, pi]``, which is the short way round.
    """
    q = as_vector(quaternion, 4)
    return -q if q[0] < 0.0 else q.copy()


def quaternion_conjugate(quaternion: ArrayLike) -> FloatArray:
    """Return the conjugate, which for a unit quaternion is the inverse rotation."""
    q = as_vector(quaternion, 4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Return the Hamilton product ``left (x) right``.

    With the attitude-matrix convention of this module the product satisfies
    ``A(p (x) q) = A(q) A(p)``, so the *right* operand is applied first when the
    result is read as an attitude.
    """
    p = as_vector(left, 4)
    q = as_vector(right, 4)
    scalar = p[0] * q[0] - float(np.dot(p[1:], q[1:]))
    vector = p[0] * q[1:] + q[0] * p[1:] + np.cross(p[1:], q[1:])
    return np.array([scalar, vector[0], vector[1], vector[2]], dtype=np.float64)


def quaternion_from_axis_angle(axis: ArrayLike, angle: float) -> FloatArray:
    """Return the quaternion for a rotation of ``angle`` radians about ``axis``."""
    direction = unit(as_vector(axis, 3))
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], np.sin(half) * direction)).astype(np.float64)


def principal_angle(quaternion: ArrayLike) -> float:
    """Return the principal rotation angle in ``[0, pi]``.

    The scalar part is taken in absolute value, which selects the shorter of the
    two rotations described by ``q`` and ``-q``. The angle is formed from
    ``atan2`` of the vector norm against the scalar part rather than from
    ``arccos``, because ``arccos`` loses half of the available digits near zero.
    """
    q = as_vector(quaternion, 4)
    return float(2.0 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0])))


def rotation_vector(quaternion: ArrayLike) -> FloatArray:
    """Return the principal rotation vector, that is the axis scaled by the angle.

    The zero vector is returned for angles below the small-angle threshold, where
    the axis is not defined.
    """
    q = quaternion_canonical(quaternion)
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm < _SMALL_ANGLE:
        # Near identity, sin(theta/2) ~ theta/2, so the rotation vector is 2 qv.
        return 2.0 * q[1:]
    angle = float(2.0 * np.arctan2(vector_norm, q[0]))
    return angle * q[1:] / vector_norm


def signed_angle_about(quaternion: ArrayLike, axis: ArrayLike) -> float:
    """Return the rotation angle of ``quaternion`` measured about ``axis``.

    The result is signed and lies in ``(-pi, pi]``. It is used to give the error
    of a slew a sign, so that an overshoot past the target can be distinguished
    from an undershoot; the unsigned principal angle cannot express that.
    """
    q = as_vector(quaternion, 4)
    direction = unit(as_vector(axis, 3))
    return float(2.0 * np.arctan2(float(np.dot(q[1:], direction)), q[0]))


def dcm_from_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Return the attitude matrix mapping inertial components to body components."""
    q = as_vector(quaternion, 4)
    scalar = float(q[0])
    vector = q[1:]
    matrix: FloatArray = (
        (scalar * scalar - float(np.dot(vector, vector))) * np.eye(3)
        + 2.0 * np.outer(vector, vector)
        - 2.0 * scalar * skew(vector)
    )
    return matrix


def quaternion_from_dcm(matrix: ArrayLike) -> FloatArray:
    """Return the canonical quaternion of an attitude matrix.

    Uses Shepperd's method: the four candidate expressions for the quaternion
    components are formed, the largest is selected, and the remaining three
    components are recovered by division. Selecting the largest keeps the divisor
    bounded away from zero, so the conversion is accurate for every rotation
    including the 180 degree case that defeats the naive trace formula.
    """
    c = as_matrix(matrix, (3, 3))
    trace = float(np.trace(c))
    candidates = np.array(
        [
            1.0 + trace,
            1.0 + 2.0 * c[0, 0] - trace,
            1.0 + 2.0 * c[1, 1] - trace,
            1.0 + 2.0 * c[2, 2] - trace,
        ],
        dtype=np.float64,
    )
    index = int(np.argmax(candidates))
    root = float(np.sqrt(max(candidates[index], 0.0)))
    half = 0.5 * root
    scale = 0.0 if root == 0.0 else 0.25 / half

    q = np.empty(4, dtype=np.float64)
    if index == 0:
        q[0] = half
        q[1] = scale * (c[1, 2] - c[2, 1])
        q[2] = scale * (c[2, 0] - c[0, 2])
        q[3] = scale * (c[0, 1] - c[1, 0])
    elif index == 1:
        q[1] = half
        q[0] = scale * (c[1, 2] - c[2, 1])
        q[2] = scale * (c[0, 1] + c[1, 0])
        q[3] = scale * (c[2, 0] + c[0, 2])
    elif index == 2:
        q[2] = half
        q[0] = scale * (c[2, 0] - c[0, 2])
        q[1] = scale * (c[0, 1] + c[1, 0])
        q[3] = scale * (c[1, 2] + c[2, 1])
    else:
        q[3] = half
        q[0] = scale * (c[0, 1] - c[1, 0])
        q[1] = scale * (c[2, 0] + c[0, 2])
        q[2] = scale * (c[1, 2] + c[2, 1])
    return quaternion_canonical(quaternion_normalise(q))


def is_rotation_matrix(matrix: ArrayLike, tolerance: float = 1e-12) -> bool:
    """Return True when ``matrix`` is orthonormal with determinant plus one."""
    c = as_matrix(matrix, (3, 3))
    orthonormal = bool(np.max(np.abs(c @ c.T - np.eye(3))) <= tolerance)
    proper = abs(float(np.linalg.det(c)) - 1.0) <= tolerance
    return orthonormal and proper


def mrp_from_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Return the MRP vector ``qv / (1 + q0)`` of a quaternion.

    The quaternion is put into canonical form first, so the result is the set with
    norm at most one and the singularity at a principal angle of 360 degrees is
    never reached.
    """
    q = quaternion_canonical(quaternion)
    return q[1:] / (1.0 + float(q[0]))


def quaternion_from_mrp(parameters: ArrayLike) -> FloatArray:
    """Return the canonical quaternion of an MRP vector."""
    s = as_vector(parameters, 3)
    squared = float(np.dot(s, s))
    denominator = 1.0 + squared
    q = np.empty(4, dtype=np.float64)
    q[0] = (1.0 - squared) / denominator
    q[1:] = 2.0 * s / denominator
    return quaternion_canonical(q)


def mrp_shadow(parameters: ArrayLike) -> FloatArray:
    """Return the shadow set ``-s / (s . s)``, which is the same attitude.

    The shadow set corresponds to the quaternion of opposite sign. Its norm is the
    reciprocal of the original norm, so the two sets are singular at opposite ends
    of the rotation range.
    """
    s = as_vector(parameters, 3)
    squared = float(np.dot(s, s))
    if squared == 0.0:
        raise ValueError("the zero MRP has no shadow set; its shadow is at infinity")
    return -s / squared


def mrp_short_rotation(parameters: ArrayLike) -> FloatArray:
    """Return whichever of ``s`` and its shadow has norm at most one.

    Applying this after every integration step keeps MRP states bounded and away
    from the 360 degree singularity, at the cost of a discontinuity in the
    parameters at 180 degrees. The attitude itself is continuous across the
    switch, which is what the invariant tests check.
    """
    s = as_vector(parameters, 3)
    squared = float(np.dot(s, s))
    if squared > MRP_SHADOW_NORM:
        return -s / squared
    return s.copy()


def dcm_from_mrp(parameters: ArrayLike) -> FloatArray:
    """Return the attitude matrix of an MRP vector.

    Evaluated in closed form rather than through the quaternion, following Schaub
    and Junkins (2018) equation 3.147, so that the conversion is a genuinely
    independent path and the round-trip tests are not comparing a function with
    itself.
    """
    s = as_vector(parameters, 3)
    squared = float(np.dot(s, s))
    cross = skew(s)
    denominator = (1.0 + squared) ** 2
    return np.eye(3) + (8.0 * (cross @ cross) - 4.0 * (1.0 - squared) * cross) / denominator


def mrp_from_dcm(matrix: ArrayLike) -> FloatArray:
    """Return the MRP vector of an attitude matrix, with norm at most one."""
    return mrp_from_quaternion(quaternion_from_dcm(matrix))


def dcm_from_euler321(angles: ArrayLike) -> FloatArray:
    """Return the attitude matrix of a 3-2-1 Euler sequence ``(yaw, pitch, roll)``.

    The sequence is a rotation of ``yaw`` about the third axis, then ``pitch``
    about the new second axis, then ``roll`` about the new first axis, which is
    the aerospace standard.
    """
    yaw, pitch, roll = as_vector(angles, 3)
    about_yaw = np.array(
        [[np.cos(yaw), np.sin(yaw), 0.0], [-np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    about_pitch = np.array(
        [
            [np.cos(pitch), 0.0, -np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    about_roll = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), np.sin(roll)], [0.0, -np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    return about_roll @ about_pitch @ about_yaw


def euler321_from_dcm(matrix: ArrayLike) -> FloatArray:
    """Return the 3-2-1 Euler angles ``(yaw, pitch, roll)`` of an attitude matrix.

    At gimbal lock, where the pitch angle reaches plus or minus 90 degrees, only
    the sum or difference of yaw and roll is observable. The roll angle is then
    set to zero and the whole rotation is reported as yaw, which is one arbitrary
    but continuous choice among infinitely many.
    """
    c = as_matrix(matrix, (3, 3))
    sine_pitch = -float(np.clip(c[0, 2], -1.0, 1.0))
    if abs(sine_pitch) >= _GIMBAL_LOCK:
        pitch = float(np.copysign(0.5 * np.pi, sine_pitch))
        yaw = float(np.arctan2(-c[1, 0], c[1, 1]))
        roll = 0.0
    else:
        pitch = float(np.arcsin(sine_pitch))
        yaw = float(np.arctan2(c[0, 1], c[0, 0]))
        roll = float(np.arctan2(c[1, 2], c[2, 2]))
    return np.array([yaw, pitch, roll], dtype=np.float64)


def quaternion_from_euler321(angles: ArrayLike) -> FloatArray:
    """Return the canonical quaternion of a 3-2-1 Euler sequence."""
    return quaternion_from_dcm(dcm_from_euler321(angles))


def euler321_from_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Return the 3-2-1 Euler angles of a quaternion."""
    return euler321_from_dcm(dcm_from_quaternion(quaternion))


def quaternion_derivative(quaternion: ArrayLike, body_rate: ArrayLike) -> FloatArray:
    """Return ``q_dot`` for a body angular rate expressed in body components."""
    q = as_vector(quaternion, 4)
    rate = as_vector(body_rate, 3)
    scalar = -0.5 * float(np.dot(q[1:], rate))
    vector = 0.5 * (q[0] * rate + np.cross(q[1:], rate))
    return np.array([scalar, vector[0], vector[1], vector[2]], dtype=np.float64)


def mrp_derivative(parameters: ArrayLike, body_rate: ArrayLike) -> FloatArray:
    """Return ``s_dot`` for a body angular rate expressed in body components.

    Schaub and Junkins (2018) equation 3.154. Provided so that MRP states can be
    propagated directly; the integrators in this package use the quaternion form,
    which has no singularity at all.
    """
    s = as_vector(parameters, 3)
    rate = as_vector(body_rate, 3)
    squared = float(np.dot(s, s))
    matrix = (1.0 - squared) * np.eye(3) + 2.0 * skew(s) + 2.0 * np.outer(s, s)
    return 0.25 * (matrix @ rate)


def attitude_error(quaternion: ArrayLike, commanded: ArrayLike) -> FloatArray:
    """Return the canonical error quaternion of ``quaternion`` relative to ``commanded``.

    The returned quaternion ``dq`` satisfies ``A(dq) = A(q) A(q_cmd)^T``, that is
    it rotates the commanded frame onto the body frame. It is put into canonical
    form, which resolves the sign ambiguity in favour of the shorter rotation and
    is what makes quaternion feedback take the short path around.
    """
    q = as_vector(quaternion, 4)
    reference = as_vector(commanded, 4)
    return quaternion_canonical(quaternion_multiply(quaternion_conjugate(reference), q))
