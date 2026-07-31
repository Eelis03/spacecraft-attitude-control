# Spacecraft Attitude Control

Quaternion attitude dynamics with reaction wheel LQR and PD control and momentum dumping.

[![CI](https://github.com/Eelis03/spacecraft-attitude-control/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/spacecraft-attitude-control/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library simulates the attitude of a rigid spacecraft carrying a redundant
reaction wheel array and magnetic torque rods, and compares two three axis
control laws on identical manoeuvres. It provides four attitude representations
with conversions in both directions, rigid body dynamics with a full inertia
tensor and correct body plus wheel momentum bookkeeping, quaternion feedback PD
and linear quadratic regulator designs behind one protocol, pseudoinverse wheel
allocation with null space speed management and saturation, and magnetic momentum
unloading. It is aimed at guidance, navigation, and control engineers who need a
readable reference implementation to check a design or a derivation against.

## Problem

A spacecraft that points an instrument has to rotate its body to a commanded
attitude and hold it there while the environment pushes back. Three things make
that harder than a textbook servo problem.

The kinematics are not linear. Attitude lives on a curved space, so every
parameterisation has either a singularity or a redundancy. Quaternions are
redundant, since `q` and `-q` are the same attitude, and a feedback law that
ignores that will happily command a 350 degree slew where a 10 degree slew would
do. Modified Rodrigues parameters are singular at 360 degrees unless the shadow
set is used, and Euler angles are singular at a pitch of 90 degrees.

The actuators are internal. Reaction wheels exchange angular momentum with the
body and cannot change the total. Every torque the environment applies is
therefore stored in the wheels and stays there. A gravity gradient torque of
5.0e-5 N m may sound negligible, but on the orbit used here it deposits 0.14 N m s
into the wheels every orbit, and with 15.1 orbits in a day that is 2.2 N m s
against a 4.0 N m s per wheel limit. Something outside the vehicle has to take
that momentum away.

The one actuator that can do that is fundamentally incomplete. A magnetic torque
rod produces `L = m x B`, which is orthogonal to the field for every possible
dipole `m`. The momentum along the field direction cannot be touched at all at
any instant. Whether the vehicle can still be unloaded depends entirely on how
much the field direction moves as the orbit progresses.

## Approach

Attitude is propagated as a unit quaternion, which has no singularity anywhere,
with the direction cosine matrix, modified Rodrigues parameters, and 3-2-1 Euler
angles provided as conversions for the places where each is the natural choice.
The sign ambiguity is resolved by a canonical form with a non-negative scalar
part, applied to the error quaternion, which is what makes the feedback take the
short way round. The shadow set switch keeps modified Rodrigues parameters inside
the unit ball. Conversions follow Shuster (1993) and Schaub and Junkins (2018),
with the matrix to quaternion direction using Shepperd's method (1978) so that it
stays accurate at a 180 degree rotation.

The plant is Euler's equation with the wheel array folded in as
`J w_dot + W u + w x (J w + W h_w) = L_external`, following Hughes (2004) and
Markley and Crassidis (2014). Two controllers sit behind a single protocol so a
scenario can be re-run with only the control law swapped. The first is quaternion
feedback PD, `L = -K dq_v - P w`, whose global asymptotic stability is proved by
Wie and Barba (1985) and Wie, Weiss and Arapostathis (1989) with a Lyapunov
argument that needs no plant model; the gains are set as `K = 2 wn^2 J` and
`P = 2 zeta wn J`, which makes every axis an identical second order system with
the stated natural frequency and damping ratio. The second is an infinite horizon
LQR (Kalman, 1960) designed on the attitude dynamics linearised about the command,
which couples the axes where the PD law does not. Both offer a feedforward of the
gyroscopic term, which cancels the coupling exactly and leaves a linear closed
loop.

Wheel torque is distributed over a four wheel pyramid by the minimum norm
pseudoinverse, exact rather than approximate because the array has full row rank,
with a null space term steering the wheel speeds and a saturation model on both
motor torque and stored momentum. Momentum is unloaded by the cross product law
`m = (k / |B|^2) (h x B)` of Camillo and Markley (1980) against a tilted dipole
field. See [docs/design-notes.md](docs/design-notes.md) for the alternatives that
were considered and rejected, and for what the model deliberately leaves out.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/spacecraft-attitude-control.git
cd spacecraft-attitude-control
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

Slew a spacecraft 60 degrees and read off what it cost:

```python
from attitude_control.analysis.metrics import ManoeuvreMetrics
from attitude_control.configuration import controllers, reference_spacecraft, slew_scenario
from attitude_control.pipeline.scenario import run_scenario

spacecraft = reference_spacecraft()
controller = controllers(spacecraft)[0]          # quaternion feedback PD
trace = run_scenario(slew_scenario(spacecraft, controller))
metrics = ManoeuvreMetrics.evaluate(trace, spacecraft)

print(f"settling time      {metrics.settling_time_s:.1f} s")
print(f"overshoot          {metrics.overshoot_percent:.2f} per cent")
print(f"peak wheel speed   {metrics.peak_wheel_speed_rpm:.1f} rpm")
print(f"peak wheel torque  {metrics.peak_wheel_torque_nm:.4f} N m")
print(f"momentum drift     {metrics.momentum_drift_nms:.2e} N m s")
```

```text
settling time      470.0 s
overshoot          4.16 per cent
peak wheel speed   912.2 rpm
peak wheel torque  0.0262 N m
momentum drift     1.87e-15 N m s
```

Runnable examples live in `examples/`:

```bash
uv run python examples/attitude_representations.py
uv run python examples/slew_manoeuvre.py
uv run python examples/disturbance_rejection.py
uv run python examples/momentum_dumping.py
```

Each accepts `--quick` for a shortened run and `--no-figure` to skip plotting.

## Results

Every number below is printed by the command shown above it. The vehicle is the
same throughout: inertia `[[90, 5, -3], [5, 100, 2], [-3, 2, 75]]` kg m^2, four
reaction wheels on an isotropic pyramid with 0.0064 kg m^2 axial inertia, a
0.05 N m torque limit and a 4.0 N m s momentum limit per wheel, on a circular
orbit at 550 km altitude and 51.6 degrees inclination. The PD design uses
`wn = 0.02` rad/s and `zeta = 1/sqrt(2)`; the LQR uses weights
`(q_att, q_rate, r) = (1, 1, 625)`, which place its slowest closed loop mode at
0.01979 rad/s, within about one per cent of the PD natural frequency, so the two
are compared at matched bandwidth.

### Representation accuracy

`uv run python examples/attitude_representations.py`, 20000 random rotations,
worst round trip error measured as a principal angle:

| conversion path | error [rad] | error [deg] |
| --- | --- | --- |
| quaternion to matrix to quaternion | 6.305e-16 | 3.612e-14 |
| quaternion to MRP to quaternion | 7.407e-16 | 4.244e-14 |
| quaternion to Euler 3-2-1 to quaternion | 2.264e-14 | 1.297e-12 |

The worst departure from orthonormality over the same set is 1.221e-15. The
Euler path is a factor of 36 worse than the other two because the extraction
divides by the cosine of the pitch angle, which is small near gimbal lock.

### Slew manoeuvre, PD against LQR

`uv run python examples/slew_manoeuvre.py`, a rest to rest 60 degree slew about
the body axis `(1, 2, 2)`, 900 s at a 0.2 s step:

| controller | settling [s] | overshoot [%] | final error [deg] | peak wheel [rpm] | peak wheel torque [N m] | peak stored momentum [N m s] |
| --- | --- | --- | --- | --- | --- | --- |
| quaternion PD | 470.0 | 4.16 | 1.992e-04 | 912.2 | 2.621e-02 | 0.8534 |
| LQR | 379.0 | 3.94 | 1.565e-04 | 946.7 | 2.887e-02 | 0.8839 |
| quaternion PD, over-driven | 192.0 | 3.94 | 1.236e-12 | 1896.2 | 5.000e-02 | 2.0194 |

Peak commanded body torque was 0.0366 N m for PD, 0.0400 N m for LQR, and
0.2287 N m for the over-driven case, which exceeded the wheel torque limit and
was clipped for 2.3 per cent of the run. The total angular momentum in the
inertial frame drifted by 1.87e-15, 1.61e-15, and 1.90e-15 N m s respectively,
against stored momenta of order 1 N m s, so the wheel bookkeeping is exact to
rounding.

At matched bandwidth the LQR settles 19 per cent faster than the PD design and
overshoots slightly less, and it pays for that with 4 per cent more wheel speed
and 10 per cent more wheel torque. The reason is the gain structure. The PD gains
are proportional to the inertia tensor, so the closed loop is the same second
order system on all three axes and the heavy axis receives proportionally more
torque. The LQR weights the commanded torque equally on all three axes instead,
which gives its attitude gain the closed form `sqrt(q_att / r) I`, exactly
0.04 I here for any inertia tensor. The result is a faster response on the light
axes and a slower one on the heavy axis, and the mixture settles sooner overall.
Neither design has an integral term, so both converge to the commanded attitude
with no steady state error when no disturbance acts, which the 1e-4 degree final
errors show. The over-driven case is included to exercise the saturation model:
it reaches the target in 192 s but only by running the wheels at their torque
limit and to twice the stored momentum.

### Disturbance rejection

`uv run python examples/disturbance_rejection.py`, an inertial hold over two
orbits at a 1.0 s step. The orbit period is 5739.0 s and the mean motion is
1.094824e-03 rad/s.

| controller | peak GG torque [N m] | mean error [arcsec] | peak error [arcsec] | stored after 2 orbits [N m s] | per orbit [N m s] | peak wheel [rpm] |
| --- | --- | --- | --- | --- | --- | --- |
| quaternion PD | 5.035e-05 | 190.3 | 292.2 | 0.2865 | 0.1432 | 276.5 |
| LQR | 5.035e-05 | 167.7 | 259.6 | 0.2865 | 0.1433 | 276.5 |

The change in total inertial angular momentum over the run is 2.87e-01 N m s for
both, which matches the stored momentum column, confirming that the wheels
absorbed the whole disturbance impulse and nothing leaked.

Neither controller drives the error to zero here, because neither has integral
action and the gravity gradient torque has a non-zero mean in the body frame. The
residual is the static gain of the loop against that mean torque, so the LQR is
12 per cent better only because its attitude gain happens to be larger on the
axes the disturbance loads. Both are of order 200 arcsec, which is a poor
pointing performance and is the honest consequence of a proportional plus
derivative structure. Momentum accumulates at 0.143 N m s per orbit, that is
2.2 N m s per day against a 4.0 N m s per wheel limit, so the array needs
unloading within a few days even with gravity gradient as the only disturbance.

### Magnetic momentum dumping

`uv run python examples/momentum_dumping.py`, starting from 2.6926 N m s of
stored momentum, three orbits at a 2.0 s step, gain 2.0e-4 per second and a
30 A m^2 rod limit. The two runs differ only in whether the magnetic field is
allowed to move.

| run | stored start [N m s] | stored end [N m s] | removed [%] | along field start | along field end | across field start | across field end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| field fixed in inertial space | 2.6926 | 1.2960 | 51.9 | 1.2938 | 1.2938 | 2.3613 | 0.0755 |
| field along the reference orbit | 2.6926 | 0.2525 | 90.6 | 1.2938 | 0.1361 | 2.3613 | 0.2127 |

This is the limitation made visible rather than described. With the field frozen,
the component of momentum across the field falls from 2.3613 to 0.0755 N m s, a
97 per cent reduction, while the component along the field is 1.2938 N m s at the
start and 1.2938 N m s at the end. Measured on the exactly conserved quantity,
the projection of total inertial momentum on the field direction, the largest
excursion over the whole run is 5.99e-12 N m s, which is integration noise. No
gain, no run time, and no rod sizing changes that number, because `m x B` is
orthogonal to `B` for every `m`. Dumping stalls with 48 per cent of the momentum
still on board.

Along the real orbit the same law removes 90.6 per cent, because the field
direction sweeps through a large solid angle and the uncontrollable direction
moves with it. On that run the projection of inertial momentum on the initial
field direction goes from 1.2938 to 0.0804 N m s: the same direction that was
untouchable in the first run becomes reachable once the field has turned. Peak
rod dipole stayed below the 30 A m^2 limit in both runs.

## Architecture

Five layers, each importing only from the ones above it in this table.

| Module | Responsibility |
| --- | --- |
| `src/attitude_control/numeric.py` | Array type aliases and shape-checked conversion helpers. |
| `src/attitude_control/model/attitude.py` | Quaternion, DCM, MRP, and Euler representations, conversions in both directions, sign ambiguity and shadow set handling, kinematic equations. |
| `src/attitude_control/model/inertia.py` | Validated inertia tensors, wheel array geometry, pyramid distribution matrix, pseudoinverse and null space projector. |
| `src/attitude_control/model/dynamics.py` | Euler's equation with wheels, body and inertial angular momentum, kinetic energy, packed state derivative. |
| `src/attitude_control/model/environment.py` | Circular orbit, gravity gradient torque, tilted dipole magnetic field, constant field. |
| `src/attitude_control/algorithm/controller.py` | The controller protocol, quaternion feedback PD, LQR on the linearised dynamics, gyroscopic feedforward. |
| `src/attitude_control/algorithm/allocation.py` | Minimum norm wheel torque, null space wheel speed steering, torque and momentum saturation. |
| `src/attitude_control/algorithm/momentum.py` | Cross product magnetic unloading and the split of momentum into removable and untouchable parts. |
| `src/attitude_control/pipeline/integrator.py` | Fixed step RK4 and the closed form quaternion norm drift of one step. |
| `src/attitude_control/pipeline/scenario.py` | Scenario configuration and the runner that produces a structured trace. |
| `src/attitude_control/analysis/metrics.py` | Settling time, overshoot, peak wheel speed and torque, stored momentum, momentum drift. |
| `src/attitude_control/analysis/representation.py` | Round trip residuals between representations. |
| `src/attitude_control/analysis/report.py` | Fixed width text tables for the example scripts. |
| `src/attitude_control/analysis/figures.py` | Matplotlib figures for the three scenarios. |
| `src/attitude_control/configuration.py` | The reference vehicle, orbit, gains, and the three scenarios shared by examples and tests. |
| `examples/` | Wiring scripts with no logic of their own. |

The model layer is pure: no state, no input or output, every function testable
against a closed form. The algorithm layer maps a state to a command and never
integrates. The pipeline layer owns time and produces a trace without
interpreting it. The analysis layer only reads traces.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

Tier one covers the mathematics. The quaternion norm drift over a 1000 s
unnormalised integration is compared against its closed form value
`N (|w| dt / 2)^6 / 144`, which follows from the truncated cosine and sine series
of an RK4 step. Every representation round trips, including across a quaternion
sign flip and through the modified Rodrigues shadow set switch at rotations
beyond 180 degrees. Attitude matrices stay orthonormal with determinant plus one.
Total angular momentum in the inertial frame is conserved through a full slew
with the wheels active. A torque free axisymmetric body reproduces the analytic
precession rate `w3 (Ja - Jt) / Jt`, an asymmetric one conserves both energy and
momentum, and the intermediate axis instability appears for a spin about the
middle axis and not for the other two. Both controllers null a large slew with no
steady state error, the small angle step response matches the analytic second
order solution, magnetic dumping is shown to leave the along-field momentum
untouched, and the wheel allocation reproduces any achievable commanded torque
exactly.

Tier two is a recorded reference run in `tests/data/reference_run.json`, pinned
with a tolerance rule per quantity. Only reproducible quantities are recorded:
closed form values, converged aggregates, counts, and classifications. The
inertial momentum drift is checked against a derived bound rather than pinned,
because it is accumulated rounding and depends on the order a machine sums a dot
product. The intermediate axis instability contributes only a qualitative
signature and no trajectory number, because it is genuinely chaotic. Every
tolerance is derived from how the quantity is computed, and the derivation is
stated in the docstring beside it.

Tier three runs each `examples/` script as a subprocess under `--quick`.

## References

### Papers and books

- Wie, B. and Barba, P. M. (1985). Quaternion feedback for spacecraft large angle
  manoeuvres. *Journal of Guidance, Control, and Dynamics*, 8(3), 360-365.
  DOI [10.2514/3.19988](https://doi.org/10.2514/3.19988). Source of the quaternion
  feedback control law and its Lyapunov stability proof.
- Wie, B., Weiss, H. and Arapostathis, A. (1989). Quaternion feedback regulator
  for spacecraft eigenaxis rotations. *Journal of Guidance, Control, and
  Dynamics*, 12(3), 375-380.
  DOI [10.2514/3.20418](https://doi.org/10.2514/3.20418). Source of the global
  asymptotic stability result and the gain structure used here.
- Kalman, R. E. (1960). Contributions to the theory of optimal control.
  *Boletin de la Sociedad Matematica Mexicana*, 5, 102-119.
  [Stable copy](https://liberzon.csl.illinois.edu/teaching/kalman_optimal_control.pdf).
  Source of the algebraic Riccati equation the LQR design solves.
- Shepperd, S. W. (1978). Quaternion from rotation matrix. *Journal of Guidance
  and Control*, 1(3), 223-224.
  DOI [10.2514/3.55767b](https://doi.org/10.2514/3.55767b). Source of the
  numerically stable matrix to quaternion conversion.
- Shuster, M. D. (1993). A survey of attitude representations. *Journal of the
  Astronautical Sciences*, 41(4), 439-517.
  [Stable copy](http://malcolmdshuster.com/Pub_1993h_J_Repsurv_scan.pdf).
  Reference for the conversions and their singularities.
- Schaub, H. and Junkins, J. L. (2018). *Analytical Mechanics of Space Systems*,
  4th edition. AIAA. DOI [10.2514/4.105210](https://doi.org/10.2514/4.105210).
  Source of the modified Rodrigues parameter shadow set and its kinematics.
- Markley, F. L. and Crassidis, J. L. (2014). *Fundamentals of Spacecraft
  Attitude Determination and Control*. Springer.
  DOI [10.1007/978-1-4939-0802-8](https://doi.org/10.1007/978-1-4939-0802-8).
  Source of the quaternion conventions and the wheel dynamics formulation.
- Hughes, P. C. (2004). *Spacecraft Attitude Dynamics*. Dover.
  ISBN 978-0486439259. Source of the body plus wheel momentum bookkeeping and the
  gravity gradient torque expansion.
- Markley, F. L., Reynolds, R. G., Liu, F. X. and Lebsock, K. L. (2010). Maximum
  torque and momentum envelopes for reaction wheel arrays. *Journal of Guidance,
  Control, and Dynamics*, 33(5), 1606-1614.
  DOI [10.2514/1.47968](https://doi.org/10.2514/1.47968). Reference for the
  pyramid array geometry and its achievable torque set.
- Camillo, P. J. and Markley, F. L. (1980). Orbit-averaged behaviour of magnetic
  control laws for momentum unloading. *Journal of Guidance and Control*, 3(6),
  563-568. DOI [10.2514/3.56036](https://doi.org/10.2514/3.56036). Source of the
  cross product unloading law and the analysis of what it can reach.
- Stickler, A. C. and Alfriend, K. T. (1976). Elementary magnetic attitude control
  system. *Journal of Spacecraft and Rockets*, 13(5), 282-287.
  DOI [10.2514/3.57089](https://doi.org/10.2514/3.57089). Reference for magnetic
  actuation and its limitations.
- Wertz, J. R., editor (1978). *Spacecraft Attitude Determination and Control*.
  Springer. DOI [10.1007/978-94-009-9907-7](https://doi.org/10.1007/978-94-009-9907-7).
  Reference for the environmental torque models.
- Wie, B. (2008). *Space Vehicle Dynamics and Control*, 2nd edition. AIAA.
  DOI [10.2514/4.860119](https://doi.org/10.2514/4.860119). Reference for wheel
  allocation and redundancy resolution.
- Alken, P. et al. (2021). International Geomagnetic Reference Field: the
  thirteenth generation. *Earth, Planets and Space*, 73, 49.
  DOI [10.1186/s40623-020-01288-x](https://doi.org/10.1186/s40623-020-01288-x).
  Source of the dipole tilt and mean equatorial field strength.

### Dependencies

- [NumPy](https://numpy.org/) 2.0 or later, BSD 3-Clause. Array arithmetic and
  linear algebra throughout every layer.
- [SciPy](https://scipy.org/) 1.14 or later, BSD 3-Clause. Only
  `solve_continuous_are`, which produces the LQR gain.
- [Matplotlib](https://matplotlib.org/) 3.9 or later, Matplotlib licence, a
  BSD-compatible licence derived from the Python Software Foundation licence.
  Figures in the analysis layer.
- [pytest](https://pytest.org/) 8.3 or later, MIT. Test runner.
- [Ruff](https://docs.astral.sh/ruff/) 0.8 or later, MIT. Linting and import
  ordering.
- [mypy](https://mypy-lang.org/) 1.13 or later, MIT. Static type checking in
  strict mode over the package, the tests, and the examples.

## License

Released under the MIT license. See [LICENSE](LICENSE).
