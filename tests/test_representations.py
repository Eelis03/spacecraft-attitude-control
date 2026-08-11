"""Tier one: invariants of the attitude representations and their conversions."""

from __future__ import annotations

import numpy as np
import pytest

from attitude_control.model.attitude import (
    attitude_error,
    dcm_from_euler321,
    dcm_from_mrp,
    dcm_from_quaternion,
    euler321_from_dcm,
    euler321_from_quaternion,
    is_rotation_matrix,
    mrp_derivative,
    mrp_from_dcm,
    mrp_from_quaternion,
    mrp_shadow,
    mrp_short_rotation,
    principal_angle,
    quaternion_canonical,
    quaternion_conjugate,
    quaternion_derivative,
    quaternion_from_axis_angle,
    quaternion_from_dcm,
    quaternion_from_euler321,
    quaternion_from_mrp,
    quaternion_identity,
    quaternion_multiply,
    rotation_vector,
    signed_angle_about,
    skew,
)
from attitude_control.numeric import FloatArray
from tests.conftest import EPSILON

# A conversion chains at most a few dozen elementary operations on quantities of
# order one, so its result carries an error of order eps. The factor of 64 covers
# the operation count of the longest chain, which is quaternion to matrix to
# quaternion by Shepperd's method. Nothing here is derived from an observed error.
_CONVERSION_TOLERANCE = 64.0 * EPSILON

_SAMPLES = 400


def random_quaternions(generator: np.random.Generator, count: int) -> FloatArray:
    """Return ``count`` quaternions with angles spread over the full range."""
    angles = generator.uniform(-np.pi, np.pi, size=count)
    axes = generator.normal(size=(count, 3))
    return np.array(
        [quaternion_from_axis_angle(axis, angle) for axis, angle in zip(axes, angles, strict=True)],
        dtype=np.float64,
    )


def raw_mrp(quaternion: FloatArray) -> FloatArray:
    """Return the MRP of a quaternion without choosing the canonical sign.

    The library deliberately canonicalises, which always yields the set of norm at
    most one. This helper produces the other set so that the shadow switch can be
    exercised directly.
    """
    return quaternion[1:] / (1.0 + float(quaternion[0]))


def test_skew_matches_the_cross_product(generator: np.random.Generator) -> None:
    """``skew(a) @ b`` reproduces ``a x b`` and ``skew(a)`` is antisymmetric."""
    for _ in range(50):
        a, b = generator.normal(size=3), generator.normal(size=3)
        assert np.allclose(skew(a) @ b, np.cross(a, b), atol=_CONVERSION_TOLERANCE)
        assert np.allclose(skew(a), -skew(a).T, atol=0.0)


def test_dcm_is_orthonormal_with_unit_determinant(generator: np.random.Generator) -> None:
    """Every generated attitude matrix is a proper rotation.

    Tolerance: the matrix entries are quadratic in quaternion components of order
    one, so the departure from orthonormality is a few rounding errors.
    """
    for quaternion in random_quaternions(generator, _SAMPLES):
        matrix = dcm_from_quaternion(quaternion)
        assert is_rotation_matrix(matrix, tolerance=_CONVERSION_TOLERANCE)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=_CONVERSION_TOLERANCE)


def test_quaternion_dcm_round_trip(generator: np.random.Generator) -> None:
    """Quaternion to matrix to quaternion returns the canonical original."""
    for quaternion in random_quaternions(generator, _SAMPLES):
        recovered = quaternion_from_dcm(dcm_from_quaternion(quaternion))
        assert np.allclose(recovered, quaternion_canonical(quaternion), atol=_CONVERSION_TOLERANCE)


def test_sign_flip_is_the_same_attitude(generator: np.random.Generator) -> None:
    """``q`` and ``-q`` give the same matrix, and both round trip to the same set.

    This is the quaternion sign ambiguity. It is a property of the mapping, not a
    defect, so the test asserts that both representatives are accepted and that
    the conversion resolves them to one canonical answer.
    """
    for quaternion in random_quaternions(generator, _SAMPLES):
        flipped = -quaternion
        assert np.allclose(
            dcm_from_quaternion(quaternion),
            dcm_from_quaternion(flipped),
            atol=_CONVERSION_TOLERANCE,
        )
        assert np.allclose(
            quaternion_from_dcm(dcm_from_quaternion(flipped)),
            quaternion_canonical(quaternion),
            atol=_CONVERSION_TOLERANCE,
        )
        assert principal_angle(quaternion) == pytest.approx(
            principal_angle(flipped), abs=_CONVERSION_TOLERANCE
        )


def test_mrp_round_trip_and_closed_form_matrix(generator: np.random.Generator) -> None:
    """MRP conversions agree with the quaternion path in both directions."""
    for quaternion in random_quaternions(generator, _SAMPLES):
        parameters = mrp_from_quaternion(quaternion)
        assert float(np.linalg.norm(parameters)) <= 1.0 + _CONVERSION_TOLERANCE
        assert np.allclose(
            dcm_from_mrp(parameters),
            dcm_from_quaternion(quaternion),
            atol=_CONVERSION_TOLERANCE,
        )
        assert np.allclose(
            quaternion_from_mrp(parameters),
            quaternion_canonical(quaternion),
            atol=_CONVERSION_TOLERANCE,
        )
        assert np.allclose(
            mrp_from_dcm(dcm_from_quaternion(quaternion)),
            parameters,
            atol=_CONVERSION_TOLERANCE,
        )


def test_mrp_shadow_set_is_the_same_attitude(generator: np.random.Generator) -> None:
    """The shadow set describes the same rotation and has the reciprocal norm."""
    for quaternion in random_quaternions(generator, _SAMPLES):
        parameters = mrp_from_quaternion(quaternion)
        if float(np.linalg.norm(parameters)) < 1e-6:
            continue
        shadow = mrp_shadow(parameters)
        assert np.allclose(
            dcm_from_mrp(shadow), dcm_from_mrp(parameters), atol=_CONVERSION_TOLERANCE
        )
        product = float(np.linalg.norm(parameters)) * float(np.linalg.norm(shadow))
        assert product == pytest.approx(1.0, abs=_CONVERSION_TOLERANCE / np.linalg.norm(parameters))
        assert np.allclose(mrp_shadow(shadow), parameters, atol=_CONVERSION_TOLERANCE)


def test_mrp_shadow_switch_preserves_the_attitude(generator: np.random.Generator) -> None:
    """Switching sets when the norm exceeds one leaves the attitude unchanged.

    The rotations used here have principal angles beyond 180 degrees in the raw
    parameterisation, which is exactly where the switch has to act; without it the
    parameters run to infinity at 360 degrees.
    """
    axes = generator.normal(size=(_SAMPLES, 3))
    angles = generator.uniform(np.deg2rad(181.0), np.deg2rad(359.0), size=_SAMPLES)
    for axis, angle in zip(axes, angles, strict=True):
        quaternion = quaternion_from_axis_angle(axis, angle)
        parameters = raw_mrp(quaternion)
        assert float(np.linalg.norm(parameters)) > 1.0
        switched = mrp_short_rotation(parameters)
        assert float(np.linalg.norm(switched)) <= 1.0
        assert np.allclose(
            dcm_from_mrp(switched), dcm_from_mrp(parameters), atol=_CONVERSION_TOLERANCE
        )


def test_mrp_short_rotation_leaves_small_sets_alone(generator: np.random.Generator) -> None:
    """A set already inside the unit ball is returned unchanged."""
    for quaternion in random_quaternions(generator, 50):
        parameters = mrp_from_quaternion(quaternion)
        assert np.array_equal(mrp_short_rotation(parameters), parameters)


def test_zero_mrp_has_no_shadow() -> None:
    """The shadow of the identity attitude is at infinity, so the call is refused."""
    with pytest.raises(ValueError, match="shadow"):
        mrp_shadow(np.zeros(3))


def test_euler_round_trip(generator: np.random.Generator) -> None:
    """Euler angles round trip through the matrix away from gimbal lock."""
    for quaternion in random_quaternions(generator, _SAMPLES):
        matrix = dcm_from_quaternion(quaternion)
        if abs(matrix[0, 2]) > 0.999:
            continue
        angles = euler321_from_dcm(matrix)
        assert np.allclose(dcm_from_euler321(angles), matrix, atol=_CONVERSION_TOLERANCE)
        assert np.allclose(
            quaternion_from_euler321(euler321_from_quaternion(quaternion)),
            quaternion_canonical(quaternion),
            atol=_CONVERSION_TOLERANCE,
        )


@pytest.mark.parametrize("pitch", [0.5 * np.pi, -0.5 * np.pi])
def test_euler_gimbal_lock_recovers_the_matrix(pitch: float) -> None:
    """At gimbal lock the angles are not unique but the matrix is still recovered.

    Only the sum or difference of yaw and roll is observable at a pitch of plus or
    minus 90 degrees, so the test checks the matrix rather than the angles.
    """
    for yaw in np.linspace(-np.pi, np.pi, 9):
        for roll in np.linspace(-np.pi, np.pi, 9):
            matrix = dcm_from_euler321((yaw, pitch, roll))
            recovered = euler321_from_dcm(matrix)
            assert recovered[2] == pytest.approx(0.0, abs=0.0)
            assert np.allclose(
                dcm_from_euler321(recovered), matrix, atol=1e3 * _CONVERSION_TOLERANCE
            )


def test_quaternion_product_composes_attitude_matrices(
    generator: np.random.Generator,
) -> None:
    """``A(p (x) q) = A(q) A(p)``, which fixes the product convention."""
    left = random_quaternions(generator, 100)
    right = random_quaternions(generator, 100)
    for p, q in zip(left, right, strict=True):
        assert np.allclose(
            dcm_from_quaternion(quaternion_multiply(p, q)),
            dcm_from_quaternion(q) @ dcm_from_quaternion(p),
            atol=_CONVERSION_TOLERANCE,
        )


def test_conjugate_is_the_inverse_rotation(generator: np.random.Generator) -> None:
    """The product of a quaternion with its conjugate is the identity."""
    for quaternion in random_quaternions(generator, 100):
        product = quaternion_multiply(quaternion, quaternion_conjugate(quaternion))
        assert np.allclose(product, quaternion_identity(), atol=_CONVERSION_TOLERANCE)


def test_attitude_error_is_the_relative_rotation(generator: np.random.Generator) -> None:
    """``attitude_error`` returns the rotation from the commanded frame to the body."""
    body = random_quaternions(generator, 100)
    commanded = random_quaternions(generator, 100)
    for q, reference in zip(body, commanded, strict=True):
        error = attitude_error(q, reference)
        assert error[0] >= 0.0
        assert np.allclose(
            dcm_from_quaternion(error),
            dcm_from_quaternion(q) @ dcm_from_quaternion(reference).T,
            atol=_CONVERSION_TOLERANCE,
        )


def test_attitude_error_of_a_flipped_command_is_unchanged(
    generator: np.random.Generator,
) -> None:
    """Flipping the sign of either quaternion leaves the canonical error the same."""
    body = random_quaternions(generator, 100)
    commanded = random_quaternions(generator, 100)
    for q, reference in zip(body, commanded, strict=True):
        base = attitude_error(q, reference)
        for signs in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
            flipped = attitude_error(signs[0] * q, signs[1] * reference)
            assert np.allclose(flipped, base, atol=_CONVERSION_TOLERANCE)


def test_rotation_vector_matches_axis_and_angle(generator: np.random.Generator) -> None:
    """The principal rotation vector has the principal angle as its length."""
    axes = generator.normal(size=(200, 3))
    angles = generator.uniform(-np.pi, np.pi, size=200)
    for axis, angle in zip(axes, angles, strict=True):
        quaternion = quaternion_from_axis_angle(axis, angle)
        vector = rotation_vector(quaternion)
        assert float(np.linalg.norm(vector)) == pytest.approx(
            principal_angle(quaternion), abs=_CONVERSION_TOLERANCE
        )


def test_rotation_vector_of_identity_is_zero() -> None:
    """The identity attitude has no axis, and the small angle branch returns zero."""
    assert np.allclose(rotation_vector(quaternion_identity()), np.zeros(3), atol=0.0)


def test_signed_angle_recovers_the_commanded_rotation() -> None:
    """The signed angle about the rotation axis reproduces the angle with its sign."""
    axis = np.array([1.0, 2.0, 2.0])
    for angle in np.linspace(-np.pi + 0.05, np.pi - 0.05, 41):
        quaternion = quaternion_from_axis_angle(axis, angle)
        assert signed_angle_about(quaternion, axis) == pytest.approx(
            angle, abs=_CONVERSION_TOLERANCE
        )


def test_quaternion_and_mrp_kinematics_agree(generator: np.random.Generator) -> None:
    """The two kinematic equations describe the same motion.

    Differentiating ``s = qv / (1 + q0)`` and substituting the quaternion rate
    must reproduce the MRP rate. Tolerance is the conversion tolerance divided by
    the smallest denominator encountered, since that division amplifies error.
    """
    for quaternion in random_quaternions(generator, 200):
        q = quaternion_canonical(quaternion)
        rate = generator.normal(size=3)
        q_dot = quaternion_derivative(q, rate)
        denominator = 1.0 + q[0]
        chain = q_dot[1:] / denominator - q[1:] * q_dot[0] / denominator**2
        expected = mrp_derivative(mrp_from_quaternion(q), rate)
        assert np.allclose(chain, expected, atol=_CONVERSION_TOLERANCE / denominator**2)


def test_quaternion_derivative_is_orthogonal_to_the_quaternion(
    generator: np.random.Generator,
) -> None:
    """``q . q_dot`` vanishes, which is why the exact flow keeps the norm at one."""
    for quaternion in random_quaternions(generator, 200):
        rate = generator.normal(size=3)
        product = float(np.dot(quaternion, quaternion_derivative(quaternion, rate)))
        assert product == pytest.approx(0.0, abs=_CONVERSION_TOLERANCE * np.linalg.norm(rate))


def test_conversions_reject_wrong_shapes() -> None:
    """Shape errors are reported rather than being broadcast into nonsense."""
    with pytest.raises(ValueError, match="expected 4 entries"):
        dcm_from_quaternion([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="expected shape"):
        quaternion_from_dcm(np.eye(4))
    with pytest.raises(ValueError, match="expected 3 entries"):
        skew([1.0, 2.0])
