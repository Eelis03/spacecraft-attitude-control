"""Tier one: the torque and momentum a reaction wheel array can actually deliver.

The achievable set is a zonotope, so every quantity in
:mod:`attitude_control.model.envelope` has a closed form and none of them needs
an optimisation. The tests below check those closed forms against the
geometry of the reference array, then check the geometry itself against a search
over the corners of the per wheel torque box, which is the one description of the
achievable set that owes nothing to the zonotope argument.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from attitude_control.algorithm.allocation import allocate, delivered_body_torque
from attitude_control.configuration import aggressive_controller, controllers, slew_scenario
from attitude_control.model.envelope import (
    envelope_normals,
    guaranteed_momentum,
    guaranteed_torque,
    maximum_momentum_about,
    maximum_torque_about,
    momentum_utilisation,
    torque_utilisation,
)
from attitude_control.model.inertia import Spacecraft, WheelArray
from attitude_control.numeric import FloatArray, unit
from attitude_control.pipeline.scenario import run_scenario
from tests.conftest import EPSILON

# The four spin axes of the isotropic pyramid are four body diagonals of a cube,
# so the envelope is a rhombic dodecahedron and its three characteristic radii are
# exact multiples of the per wheel limit: 4/sqrt(3) about a body axis, 2 about a
# spin axis, and 2 sqrt(2/3) about the worst direction, which is a face normal
# such as (1, 1, 0)/sqrt(2).
_ABOUT_BODY_AXIS = 4.0 / np.sqrt(3.0)
_ABOUT_SPIN_AXIS = 2.0
_GUARANTEED = 2.0 * np.sqrt(2.0 / 3.0)

# Every quantity here is a sum of four products of unit vectors divided by another
# such sum, so the relative error is a few eps. The bound below is derived from
# that count and not from any observed residual.
_RELATIVE_TOLERANCE = 1e-14


def box_corners(count: int) -> FloatArray:
    """Return every corner of the unit per wheel command box, one per row."""
    return np.array(list(itertools.product((-1.0, 1.0), repeat=count)), dtype=np.float64)


def corner_supports(wheels: WheelArray, normals: FloatArray, limit: float) -> FloatArray:
    """Return the largest component along each normal of any realisable command.

    A linear function over a box attains its maximum at a corner, so enumerating
    the corners gives the support function of the achievable set exactly, without
    using the closed form the module under test relies on.
    """
    corners = box_corners(wheels.count) * limit
    reachable = np.array([delivered_body_torque(wheels, corner) for corner in corners])
    return np.asarray(np.max(reachable @ normals.T, axis=0), dtype=np.float64)


def test_the_isotropic_pyramid_envelope_matches_its_closed_form(wheels: WheelArray) -> None:
    """Each radius of the rhombic dodecahedron is an exact multiple of the limit.

    The worst direction is a face normal, where the envelope touches its inscribed
    sphere, so the reach about ``(1, 1, 0)`` is the guaranteed radius itself.
    Tolerance: the rounding bound stated at the top of this module.
    """
    for axis in np.eye(3):
        assert maximum_torque_about(wheels, axis) == pytest.approx(
            _ABOUT_BODY_AXIS * wheels.max_torque, rel=_RELATIVE_TOLERANCE
        )
    for index in range(wheels.count):
        assert maximum_torque_about(wheels, wheels.axes[:, index]) == pytest.approx(
            _ABOUT_SPIN_AXIS * wheels.max_torque, rel=_RELATIVE_TOLERANCE
        )
    assert guaranteed_torque(wheels) == pytest.approx(
        _GUARANTEED * wheels.max_torque, rel=_RELATIVE_TOLERANCE
    )
    assert maximum_torque_about(wheels, (1.0, 1.0, 0.0)) == pytest.approx(
        guaranteed_torque(wheels), rel=_RELATIVE_TOLERANCE
    )


def test_every_face_normal_is_orthogonal_to_the_two_axes_that_span_it(
    wheels: WheelArray,
) -> None:
    """A face of a zonotope is spanned by the generators lying in it.

    Six pairs of spin axes give six normals, and each one is perpendicular to
    exactly the two axes of its own pair, which is what makes the face a genuine
    two dimensional piece of the boundary rather than an edge.
    """
    normals = envelope_normals(wheels)
    assert normals.shape == (6, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=64.0 * EPSILON)
    for normal in normals:
        in_face = np.abs(normal @ wheels.axes) < 64.0 * EPSILON
        assert int(np.count_nonzero(in_face)) == 2


def test_the_envelope_radii_are_attained_by_wheel_torques_inside_the_limits(
    wheels: WheelArray,
) -> None:
    """Each closed form radius is realised by an explicit motor torque command.

    The three commands are read off the geometry: all four wheels driven together
    give the body axis maximum, three against one give the spin axis maximum, and
    an opposed pair with the other two idle gives the guaranteed radius. Every
    one of them sits on the per wheel limit, which is what a point on the boundary
    of the envelope means, so the radii are attained rather than merely bounded.
    """
    limit = wheels.max_torque
    cases = (
        (np.full(4, -limit), np.array([0.0, 0.0, 1.0]), _ABOUT_BODY_AXIS),
        (limit * np.array([-1.0, -1.0, 1.0, -1.0]), wheels.axes[:, 0], _ABOUT_SPIN_AXIS),
        (limit * np.array([-1.0, 0.0, 1.0, 0.0]), unit((1.0, 1.0, 0.0)), _GUARANTEED),
    )
    for wheel_torque, direction, radius in cases:
        assert float(np.max(np.abs(wheel_torque))) <= limit
        delivered = delivered_body_torque(wheels, wheel_torque)
        assert np.allclose(delivered, radius * limit * direction, atol=64.0 * EPSILON * limit)
        assert maximum_torque_about(wheels, direction) == pytest.approx(
            radius * limit, rel=_RELATIVE_TOLERANCE
        )
        assert torque_utilisation(wheels, delivered) == pytest.approx(1.0, rel=_RELATIVE_TOLERANCE)


def test_the_guaranteed_radius_matches_a_search_over_the_corners_of_the_torque_box(
    wheels: WheelArray,
) -> None:
    """The closed form agrees with a maximisation over the sixteen corners.

    The inscribed sphere touches the boundary on a face, so its radius is the
    smallest of the face supports, and each of those is the largest component of
    a realisable torque along that normal. Enumerating the corners computes the
    same numbers without the support formula.
    """
    normals = envelope_normals(wheels)
    supports = corner_supports(wheels, normals, wheels.max_torque)
    assert guaranteed_torque(wheels) == pytest.approx(
        float(np.min(supports)), rel=_RELATIVE_TOLERANCE
    )


def test_every_realisable_torque_lies_inside_the_envelope(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """Motor torques inside the limits never produce a utilisation above one."""
    for _ in range(500):
        wheel_torque = generator.uniform(-1.0, 1.0, size=wheels.count) * wheels.max_torque
        delivered = delivered_body_torque(wheels, wheel_torque)
        assert torque_utilisation(wheels, delivered) <= 1.0 + _RELATIVE_TOLERANCE


def test_a_command_past_the_envelope_cannot_be_realised(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """One part in a thousand outside the reach is separated from every corner.

    A point lies outside a convex set as soon as one direction separates it from
    the whole of the set, so exhibiting a face normal on which the command
    projects further than any corner of the torque box proves the command is
    unreachable by any allocation whatsoever.
    """
    normals = envelope_normals(wheels)
    supports = corner_supports(wheels, normals, wheels.max_torque)
    for _ in range(200):
        direction = unit(generator.normal(size=3))
        command = 1.001 * maximum_torque_about(wheels, direction) * direction
        assert torque_utilisation(wheels, command) == pytest.approx(1.001, rel=_RELATIVE_TOLERANCE)
        assert float(np.max(np.abs(normals @ command) - supports)) > 0.0


def test_the_minimum_norm_allocation_saturates_inside_the_envelope(
    wheels: WheelArray,
) -> None:
    """A command the array can deliver is not one the minimum norm solution reaches.

    About a spin axis the minimum norm solution puts three quarters of the demand
    on that one wheel, so it fits inside the limits only out to 4/3 of the limit,
    while the envelope reaches twice it. The command below sits between the two:
    the envelope reports it at 80 per cent, the wheel torques written out here
    realise it exactly with a fifth of the limit to spare, and the minimum norm
    allocation this package uses saturates on it. That gap is the whole content of
    the linear programming alternative the design notes record as rejected.
    """
    limit = wheels.max_torque
    command = 1.6 * limit * wheels.axes[:, 0]
    assert torque_utilisation(wheels, command) == pytest.approx(0.8, rel=_RELATIVE_TOLERANCE)

    exact = 0.8 * limit * np.array([-1.0, -1.0, 1.0, -1.0])
    assert float(np.max(np.abs(exact))) <= limit
    assert np.allclose(delivered_body_torque(wheels, exact), command, atol=64.0 * EPSILON * limit)
    assert allocate(wheels, command, np.zeros(wheels.count)).torque_saturated


def test_the_momentum_envelope_is_the_torque_envelope_scaled_by_the_limits(
    wheels: WheelArray, generator: np.random.Generator
) -> None:
    """Both sets are the same zonotope, so only the per wheel limit separates them.

    The relation worth stating is the one an operator uses: the utilisation of a
    vector is its magnitude divided by the reach of the envelope in its own
    direction, so a single number says how much is left and in which direction it
    is running out.
    """
    ratio = wheels.max_momentum / wheels.max_torque
    for _ in range(200):
        vector = generator.normal(size=3)
        assert maximum_momentum_about(wheels, vector) == pytest.approx(
            ratio * maximum_torque_about(wheels, vector), rel=_RELATIVE_TOLERANCE
        )
        assert momentum_utilisation(wheels, vector) == pytest.approx(
            float(np.linalg.norm(vector)) / maximum_momentum_about(wheels, vector),
            rel=_RELATIVE_TOLERANCE,
        )
    assert guaranteed_momentum(wheels) == pytest.approx(
        _GUARANTEED * wheels.max_momentum, rel=_RELATIVE_TOLERANCE
    )


def test_orthogonal_arrays_give_a_box_envelope() -> None:
    """Three orthogonal wheels give a cube, and a repeated axis stretches it.

    The second array exercises a parallel pair, which spans no face and therefore
    contributes no slab. Both envelopes are read off directly: the achievable set
    is a box, so the guaranteed radius is the smallest half width and the reach
    about a direction is set by the largest component of that direction measured
    in half widths.
    """
    limit = 0.05
    cube = WheelArray(
        axes=np.eye(3),
        axial_inertia=np.full(3, 0.0064),
        max_torque=limit,
        max_momentum=4.0,
    )
    assert envelope_normals(cube).shape == (3, 3)
    assert guaranteed_torque(cube) == pytest.approx(limit, rel=_RELATIVE_TOLERANCE)
    assert maximum_torque_about(cube, (1.0, 1.0, 1.0)) == pytest.approx(
        limit * np.sqrt(3.0), rel=_RELATIVE_TOLERANCE
    )

    repeated = WheelArray(
        axes=np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        axial_inertia=np.full(4, 0.0064),
        max_torque=limit,
        max_momentum=4.0,
    )
    assert envelope_normals(repeated).shape == (5, 3)
    assert maximum_torque_about(repeated, (1.0, 0.0, 0.0)) == pytest.approx(
        2.0 * limit, rel=_RELATIVE_TOLERANCE
    )
    assert guaranteed_torque(repeated) == pytest.approx(limit, rel=_RELATIVE_TOLERANCE)


def test_the_reference_gains_keep_the_demand_inside_the_envelope(
    spacecraft: Spacecraft,
) -> None:
    """The reason the reference natural frequency was chosen, measured on a run.

    A rest to rest slew asks for its largest torque at the first sample, where the
    attitude error is largest and the rate is still zero, so a minute of the
    manoeuvre is enough to catch the peak. The reference design stays inside the
    envelope and the over-driven design does not, which is why the latter spends
    part of its run against the wheel torque limit.
    """
    wheels = spacecraft.wheels
    reference = run_scenario(slew_scenario(spacecraft, controllers(spacecraft)[0], duration=60.0))
    over_driven = run_scenario(
        slew_scenario(spacecraft, aggressive_controller(spacecraft), duration=60.0)
    )
    inside = max(torque_utilisation(wheels, torque) for torque in reference.commanded_torque)
    outside = max(torque_utilisation(wheels, torque) for torque in over_driven.commanded_torque)
    assert inside < 1.0
    assert outside > 1.0


def test_the_envelope_refuses_a_direction_with_no_length(wheels: WheelArray) -> None:
    """A zero vector has no direction, so there is nothing to report about it.

    A zero command is a different matter: it uses none of the envelope, which is
    what the utilisation returns.
    """
    with pytest.raises(ValueError, match="zero vector"):
        maximum_torque_about(wheels, np.zeros(3))
    with pytest.raises(ValueError, match="zero vector"):
        maximum_momentum_about(wheels, np.zeros(3))
    assert torque_utilisation(wheels, np.zeros(3)) == 0.0
    assert momentum_utilisation(wheels, np.zeros(3)) == 0.0
