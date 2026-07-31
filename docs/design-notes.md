# Design notes for Spacecraft Attitude Control

## Method selection

### Attitude parameterisation: unit quaternion as the state

The propagated state is a unit quaternion in scalar-first Hamilton convention.
The four dimensional embedding of a three dimensional rotation group buys a
kinematic equation, `q_dot = 0.5 q (x) (0, w)`, that is bilinear, singularity
free, and cheap. The cost is one redundancy, `q` and `-q` describing the same
attitude, and one constraint, the unit norm, that a general integrator does not
preserve.

Both costs are paid explicitly rather than hidden. The redundancy is resolved by
`quaternion_canonical`, which selects the representative with a non-negative
scalar part. This is applied to the *error* quaternion inside `error_state`, so
the feedback always takes the shorter of the two rotations. The consequence is
easy to state and easy to test: a commanded error of 190 degrees produces exactly
the same torque as a commanded error of minus 170 degrees, and the test
`test_error_state_takes_the_short_way_round` asserts it.

The norm constraint is handled by renormalising once per integration step in the
scenario runner, and deliberately not renormalising inside the integrator. The
reason is that renormalisation makes the drift invisible, and the drift is the
most informative diagnostic the integrator has. Applied to the quaternion
kinematics at a constant body rate, one classical fourth order Runge-Kutta step
multiplies the squared norm by exactly `1 - x^6/72 + x^8/576` with
`x = |w| dt / 2`, because the RK4 update is the fourth order truncation of the
matrix exponential and the quaternion parts are the truncated cosine and sine
series. That gives a closed form prediction for the drift after `N` steps, which
`step_norm_drift` computes and the invariant test compares against to one per
cent. A test that renormalised every step would have asserted nothing.

The other three representations are provided as conversions, each for the place
where it is the natural choice. The direction cosine matrix is what actually
rotates vectors between frames. Modified Rodrigues parameters are the minimal
three parameter set with the mildest singularity, useful when a three dimensional
attitude state is wanted; the shadow set `s -> -s / (s . s)` is implemented, and
`mrp_short_rotation` switches whenever the norm exceeds one, which keeps the
parameters bounded and 180 degrees away from the singularity at all times. Euler
angles are provided because operators read them, and their gimbal lock is handled
by fixing the roll angle at zero and reporting the whole rotation as yaw, which
is one arbitrary but continuous choice among infinitely many.

### Dynamics: total inertia including wheels

The inertia tensor `J` is that of the whole vehicle including the wheels at their
nominal positions. Because a reaction wheel is axisymmetric about its spin axis,
spinning it does not change that tensor, so `J` is constant and the wheel
contribution to angular momentum collapses to one scalar per wheel. Total angular
momentum in body components is then `J w + W h_w` and nothing else. This is the
convention of Hughes (2004) chapter 6 and Markley and Crassidis (2014) chapter 7.

The alternative convention, in which `J` is the inertia of the platform alone and
each wheel contributes its own tensor, is equivalent but produces more terms and
more opportunities to drop one. The chosen form makes the strongest available
correctness test cheap to write: with no external torque, `A(q)^T (J w + W h_w)`
must be exactly constant, whatever the wheels are doing. The reference slew holds
it to 1.9e-15 N m s against a stored momentum of order 1 N m s.

The sign convention is fixed once and followed everywhere. The motor torque `u`
acts on the wheel, so `h_w_dot = u` and the torque delivered to the body is
`-W u`. Every allocation and dynamics sign in the package follows from that one
line, which is stated at the top of `model/dynamics.py`.

### Control: two laws behind one protocol

Quaternion feedback PD was chosen as the baseline because it has a proof and the
proof needs no plant model. Wie, Weiss and Arapostathis (1989) exhibit a Lyapunov
function whose derivative along the closed loop is negative semidefinite for any
positive definite derivative gain, and LaSalle's theorem then gives global
asymptotic stability. Nothing in the argument depends on the inertia tensor being
known, which is a strong practical property.

The gain structure is the inertia-scaled one, `K = 2 wn^2 J` and
`P = 2 zeta wn J`. Substituting into `J w_dot = -K dq_v - P w` and using the small
angle relations gives `theta_ddot + 2 zeta wn theta_dot + wn^2 theta = 0` on every
axis independently. Two consequences follow. The tuning parameters become a
natural frequency and a damping ratio rather than six numbers with no units. And
the closed loop acquires an exact analytic prediction that a test can check, which
`test_small_step_matches_the_linear_prediction` does to within ten times the
`theta^2 / 24` linearisation error.

The LQR was chosen as the comparison because it makes a different trade at the
same bandwidth rather than being a variation on the same one. Designed on
`theta_dot = w`, `J w_dot = L` with `Q = diag(q_att I, q_rate I)` and `R = r I`, it
weights control effort equally across the three body axes where the PD law weights
it proportionally to inertia. The result is a gain matrix that couples the axes
and a response that is faster on the light axes and slower on the heavy one.

The LQR attitude gain has a closed form worth recording. Substituting
`S12 = sqrt(r q_att) J` into the attitude block of the Riccati equation gives
`S12 J^-2 S12 = r q_att I`, which holds identically, and therefore
`K_attitude = r^-1 J^-1 S12 = sqrt(q_att / r) I` for any inertia tensor at all.
That is checked against the formula in `test_lqr_attitude_gain_has_a_closed_form`
rather than against a recorded number, which is the right way to pin a value that
a solver happens to produce.

Both laws offer a feedforward of `w x (J w + W h_w)`. With it enabled the closed
loop is exactly the linear system the designs assume, which is what makes the
linear prediction test meaningful; without it the coupling remains and the
response degrades gracefully. The comparison in the results uses it on both, so
the only difference between the two runs is the gain matrix.

### Allocation: minimum norm with null space steering

The four wheel pyramid at the isotropic half angle, `tan^2(b) = 2`, gives
`W W^T = (4/3) I`, so the array has the same torque gain about every body axis and
one null direction, `(1, -1, 1, -1) / 2`. Because `W` has full row rank the
pseudoinverse solution `u = -W^+ L` is exact rather than least squares, which the
allocation test asserts to 64 machine epsilons over 200 random achievable
commands.

The null space is used for wheel speed management, adding
`-k (I - W^+ W) (h_w - h_target)`. The projector is symmetric and idempotent, so
the added term produces no body torque at all, which is tested directly. The
practical purpose is to keep the array away from its speed limits and away from
zero crossings where bearing friction is worst.

Saturation is modelled as two separate limits in a fixed order: motor torque is
clipped per wheel, then any torque that would push an already saturated wheel
further out is zeroed. The second limit deliberately allows a saturated wheel to
be slowed down, because that is what the hardware does. Both limits break the
exactness of the allocation, so `Allocation` reports which acted, and the
over-driven slew in the results shows the effect.

### Momentum management: cross product law

The unloading law is `m = (k / |B|^2) (h x B)`, which produces
`L = -k (h - B_hat (B_hat . h))`, that is minus the gain times the component of
stored momentum perpendicular to the field. This is the law analysed by Camillo
and Markley (1980). It was chosen over the alternatives below because it is the
one that follows directly from the constraint: the achievable torque set is the
plane perpendicular to `B`, and the law is the orthogonal projection of the
desired torque onto that plane.

Rod saturation scales the dipole rather than clipping it component by component,
so the torque direction is preserved when the rods run out of authority. Clipping
would rotate the torque and could point it the wrong way.

The magnetic field model is a centred tilted dipole, accurate to roughly 20 per
cent in magnitude and a few degrees in direction. That is enough to reproduce the
behaviour of the control law, which depends on the field direction sweeping around
the orbit, and not enough for anything flown.

## Rejected alternatives

### Direction cosine matrix as the propagated state

Propagating the nine components of the matrix directly avoids every
parameterisation question. It was rejected because the constraint set is six
dimensional, orthonormality plus unit determinant, and a general integrator
violates all six. Reprojection requires an orthogonalisation such as a polar
decomposition or a singular value decomposition on every step, which is both
expensive and a much larger correction than the single quaternion normalisation.
The drift also has no clean closed form, so the sharpest invariant test in this
suite would not have been available.

### Modified Rodrigues parameters as the propagated state

Three parameters is the minimum, and with the shadow set switch the
representation is bounded and never within 180 degrees of a singularity. It was
rejected for the integration state because the switch is a discontinuity in the
state vector. A fixed step Runge-Kutta method crossing that discontinuity within a
step commits an error that no step size control can see, and the momentum
conservation test would then have had to tolerate it. The conversions and the
kinematic equation are provided, so a caller who wants a three parameter state can
have one; the package simply does not integrate in it.

### Sliding mode or adaptive attitude control

A sliding mode law would have given a stronger robustness statement against the
inertia uncertainty this package does not model. It was rejected because the
chattering it introduces interacts badly with a wheel torque limit, and because a
robustness claim with no uncertainty model in the simulation would be untestable
here. An adaptive law was rejected for the same reason: with the inertia known
exactly, adaptation has nothing to do, so the comparison would have measured
nothing.

### Nonlinear model predictive attitude control

An MPC formulation would handle the wheel torque and momentum limits explicitly
rather than by clipping after the fact, and would produce a better constrained
slew than the over-driven PD run in the results. It was rejected on scope: it
requires a nonlinear program solver in the loop, which changes the dependency set
and moves the interesting content from attitude dynamics to optimisation. The
saturation model is honest about what it does instead, and the results show the
degradation rather than hiding it.

### B-dot detumbling law

The `-k B_dot` law is the standard first mode after separation and is a natural
companion to the magnetic hardware modelled here. It was rejected because it
solves a different problem, reducing a large initial body rate, and would have
added an actuation mode without adding anything to the reaction wheel and momentum
management story that is the subject of this package.

### Wheel allocation by linear programming

Solving `min |u|_inf` subject to `W u = -L` would use the achievable torque set
better than the minimum norm solution, since the limits are per wheel and
therefore an infinity norm constraint. It was rejected because the improvement is
small for an isotropic array, roughly the ratio between the two norms in four
dimensions, and because the pseudoinverse has a closed form that makes the
exactness of the allocation provable rather than merely observed.

### Higher fidelity magnetic field

The full IGRF expansion would replace the 20 per cent magnitude error of the
tilted dipole with a fraction of a per cent. It was rejected because it needs a
coefficient table that would have to be shipped and kept current, and because
nothing in the momentum dumping results depends on field magnitude to better than
tens of per cent. The one conclusion that matters, that the along-field component
is untouchable, is exact for any field model whatsoever.

### Variable step or geometric integration

An adaptive Runge-Kutta method would spend fewer steps in the quiet parts of the
long orbital runs. It was rejected because a fixed step makes the regression
values reproducible without pinning a step size controller's decisions, and because
the closed form drift analysis that anchors the norm test only exists for a fixed
step. An integrator that preserves the unit norm exactly was rejected for the
reason given above: it would have made the drift invisible.

## Known limitations

### Sensor noise and estimation are out of scope

Every controller receives the true attitude and the true body rate. There is no
star tracker, no gyroscope, no attitude estimator, and no measurement noise. This
hides three things that dominate real pointing budgets. First, the derivative term
of the PD law would amplify rate noise directly, so the achievable damping in
flight is set by the gyroscope noise floor rather than by the desired `zeta`.
Second, an estimator introduces phase lag that eats gain margin, so the natural
frequency that is safe here is optimistic. Third, the 200 arcsec residuals reported
under gravity gradient are a control limitation only; a real budget would add
estimation error on top of them. Adding a multiplicative extended Kalman filter and
a noise model would remove this limitation and would roughly double the size of the
package.

### Flexible modes are out of scope

The vehicle is rigid. Real spacecraft carry solar arrays and antennas whose first
bending modes often sit between 0.5 Hz and 5 Hz, and the standard design rule is to
keep the control bandwidth an order of magnitude below the first mode. The
0.02 rad/s used here is 0.0032 Hz and would clear a 0.5 Hz mode comfortably, so the
reported results would probably survive; the over-driven case at 0.05 rad/s is
0.008 Hz and would also clear it. What is genuinely hidden is the interaction
during the manoeuvre: a rigid model cannot show energy pumped into a flexible mode
by a torque profile with content near its frequency, cannot show a settling time
extended by a lightly damped appendage ringing after the slew, and cannot show the
gain and phase margin lost to a mode inside the loop. A hybrid coordinate model
with modal participation factors, as in Hughes (2004) chapter 5, would be the way to
remove this, and it would change the over-driven result most.

### The controllers have no integral action

Neither law integrates the attitude error, so a constant disturbance torque
produces a constant offset equal to the static loop gain divided into it. The
gravity gradient results show this directly at roughly 200 arcsec. Adding an
integral term, or better a disturbance observer, would remove the offset at the cost
of an extra state and of wind-up behaviour under saturation. This was left out
deliberately: it makes the disturbance rejection result an honest measurement of
what a proportional plus derivative structure achieves, rather than a measurement of
how well an integrator was tuned.

### The control law is continuous time

The controller is evaluated inside the derivative function, so the closed loop is a
continuous time system integrated by Runge-Kutta rather than a sampled data system.
This is what makes the small angle response comparable with the analytic second
order prediction. It hides the phase lag of a zero order hold, which for a sample
period `T` is approximately `wn T / 2` radians, and it hides computation delay
entirely. At a typical 10 Hz control rate and `wn = 0.02` rad/s that lag is 0.001
rad, negligible, so the reported results would not move; at the over-driven
0.05 rad/s it is still small. The conclusion would change for a bandwidth an order of
magnitude higher.

### The wheel model is idealised

Wheels are modelled as an axial inertia, a torque limit, and a momentum limit. There
is no bearing friction, no motor torque ripple, no cogging, no static friction near
zero speed, and no wheel imbalance. Static friction near zero speed is the omission
that matters most: it produces a limit cycle in real systems, and it is the reason
the null space steering here targets zero wheel speed when a real design would target
a non-zero bias. With friction modelled, the results would show a small persistent
oscillation instead of clean convergence to 1e-4 degrees.

### The orbit is Keplerian and circular

There is no oblateness, no drag, and no eccentricity. Over the two or three orbits
simulated, nodal regression from the second zonal harmonic is a few degrees, which
would slightly change how the magnetic field direction sweeps and therefore the
90.6 per cent figure for orbital dumping. The direction of the conclusion, that a
moving field direction makes the full dump possible and a fixed one does not, is
unaffected.

### Environmental torques other than gravity gradient are absent

Aerodynamic drag torque, solar radiation pressure torque, and residual magnetic
dipole torque are not modelled. At 550 km, drag torque on a small satellite is
typically comparable to the gravity gradient torque and residual dipole torque can
exceed both. The momentum accumulation figure of 0.143 N m s per orbit is therefore
a lower bound on what a real vehicle would have to dump, probably by a factor of two
or three.

### The magnetic dumping gain is fixed

The unloading gain is a constant. Real implementations schedule it against field
strength or disable dumping when the geometry is unfavourable, because a fixed gain
wastes rod authority when the momentum is nearly aligned with the field. The fixed
field run in the results is the extreme case of that waste: the rods keep commanding
a dipole that removes nothing at all once the perpendicular component is gone.

### The regression baseline covers three scenarios only

The recorded run in `tests/data/reference_run.json` pins the slew, the gravity
gradient hold, and the two dumping cases at reduced settings. A change that only
affects behaviour outside those, for example a bug in the Euler angle gimbal lock
branch or in a non-isotropic wheel array, is caught by the tier one property tests
but not by the regression tier. That split is deliberate: the property tests are
where behaviour outside the recorded scenarios belongs, because they state what must
be true rather than what happened to be measured.
