Theory overview
===============

This page summarises the phaseless auxiliary-field quantum Monte Carlo
(ph-AFQMC) method as it is implemented in ipie, and connects each ingredient of
the algorithm to the classes and input options that control it. It is intended
as a compact reference rather than a derivation; for a thorough treatment see
the reviews by Motta and Zhang [Motta2018]_ and Lee, Pham and Reichman
[Lee2022]_, and the two ipie release papers [Malone2023]_ [Jiang2024]_.
Practical details of running calculations are covered in :doc:`quickstart`,
:doc:`input_file` and :doc:`python_api`; extensions such as free-projection
and finite-temperature AFQMC are described in :doc:`advanced`.

Hamiltonian and Cholesky decomposition
--------------------------------------

ipie targets *ab initio* Hamiltonians written in second quantisation in an
orthonormal one-particle basis of :math:`M` spatial orbitals,

.. math::

   \hat{H} = E_0 + \sum_{pq}^{M} h_{pq}\, \hat{a}^\dagger_p \hat{a}_q
           + \frac{1}{2}\sum_{pqrs}^{M} (pr|qs)\,
             \hat{a}^\dagger_p \hat{a}^\dagger_q \hat{a}_s \hat{a}_r ,

where :math:`E_0` is a constant (nuclear repulsion plus any frozen-core
contribution), :math:`h_{pq}` are the one-electron integrals, and
:math:`(pr|qs)` are the two-electron repulsion integrals in chemists'
notation. Spin labels are suppressed; every orbital index implicitly carries a
spin, and the operators :math:`\hat{a}^\dagger_p` are spin-orbital creation
operators.

The two-electron integral tensor is positive semi-definite when viewed as an
:math:`M^2 \times M^2` matrix in the composite indices :math:`(pr)` and
:math:`(qs)`, so it admits a (modified, pivoted) Cholesky decomposition

.. math::

   (pr|qs) \approx \sum_{\gamma=1}^{N_\gamma} L^{\gamma}_{pr} L^{\gamma\,*}_{qs},

which is truncated once the largest remaining diagonal element falls below a
threshold (typically :math:`10^{-5}` Hartree). The number of Cholesky vectors
scales as :math:`N_\gamma \sim 5\text{--}10\,M`. In terms of the one-body
operators

.. math::

   \hat{v}_\gamma = \sum_{pr} L^{\gamma}_{pr}\, \hat{a}^\dagger_p \hat{a}_r ,

normal ordering of the two-body term yields the form used throughout AFQMC,

.. math::

   \hat{H} = E_0 + \sum_{pq} h'_{pq}\, \hat{a}^\dagger_p \hat{a}_q
           + \frac{1}{2}\sum_{\gamma} \hat{v}_\gamma^{2},
   \qquad
   h'_{pq} = h_{pq} - \frac{1}{2}\sum_{r}\sum_{\gamma} L^{\gamma}_{pr} L^{\gamma\,*}_{qr}
           = h_{pq} - \frac{1}{2}\sum_{r} (pr|rq).

The modified one-body matrix :math:`h'` is stored as the attribute
``h1e_mod`` of the Hamiltonian classes. For real integrals the Cholesky
vectors are real and symmetric in :math:`(p,r)`, the Hamiltonian is
represented by :class:`ipie.hamiltonians.generic.GenericRealChol`, and the
number of auxiliary fields is :math:`N_\gamma`. For complex integrals
(:class:`ipie.hamiltonians.generic.GenericComplexChol`, e.g. periodic systems
away from the :math:`\Gamma` point) each non-Hermitian :math:`L^\gamma` is
split into Hermitian and anti-Hermitian parts,
:math:`A^\gamma = \tfrac{1}{2}(L^\gamma + L^{\gamma\dagger})` and
:math:`B^\gamma = \tfrac{i}{2}(L^\gamma - L^{\gamma\dagger})`, so that
:math:`\hat{v}_\gamma^2 \to \hat{A}_\gamma^2 + \hat{B}_\gamma^2` and the number
of fields becomes :math:`2N_\gamma`. When the Cholesky tensor is too large for
one node it can be distributed over MPI ranks with
:class:`ipie.hamiltonians.generic_chunked.GenericRealCholChunked`; k-point
symmetry-adapted, ISDF and THC factorisations are described in
:doc:`advanced`.

Imaginary-time projection
-------------------------

Ground-state AFQMC is a projector method. For any initial state
:math:`|\Phi_0\rangle` with non-zero overlap with the exact ground state
:math:`|\Psi_0\rangle`,

.. math::

   |\Psi_0\rangle \propto \lim_{\tau\to\infty} e^{-\tau(\hat{H}-E_0)}\,|\Phi_0\rangle,

because every excited-state component is damped exponentially in the
imaginary time :math:`\tau` relative to the ground state. In practice the
projection is discretised into :math:`n` small steps of length
:math:`\Delta\tau` (input option ``timestep``, default 0.005
:math:`\mathrm{Ha}^{-1}`),

.. math::

   e^{-\tau \hat{H}} = \left(e^{-\Delta\tau \hat{H}}\right)^{n},
   \qquad \tau = n\,\Delta\tau ,

and the initial state is chosen as a Slater determinant (by default the same
determinant used for the trial wavefunction).

Trotter splitting and the Hubbard-Stratonovich transformation
-------------------------------------------------------------

Writing :math:`\hat{H} = \hat{H}_1 + \hat{H}_2` with
:math:`\hat{H}_1 = \sum_{pq} h'_{pq}\hat{a}^\dagger_p\hat{a}_q` and
:math:`\hat{H}_2 = \tfrac{1}{2}\sum_\gamma \hat{v}_\gamma^2`, ipie uses the
symmetric Trotter-Suzuki decomposition

.. math::

   e^{-\Delta\tau \hat{H}} = e^{-\frac{\Delta\tau}{2}\hat{H}_1}\,
                             e^{-\Delta\tau \hat{H}_2}\,
                             e^{-\frac{\Delta\tau}{2}\hat{H}_1}
                             + \mathcal{O}(\Delta\tau^{3}),

so that the accumulated time-step error in the energy is
:math:`\mathcal{O}(\Delta\tau^{2})`. The two-body propagator is then linearised
with the Hubbard-Stratonovich (HS) transformation. For each auxiliary field
:math:`\gamma`,

.. math::

   e^{-\frac{\Delta\tau}{2}\hat{v}_\gamma^{2}}
   = \int_{-\infty}^{\infty} \frac{dx_\gamma}{\sqrt{2\pi}}\, e^{-x_\gamma^{2}/2}\,
     e^{\,i\sqrt{\Delta\tau}\,x_\gamma \hat{v}_\gamma},

which is exact. Collecting all fields into a vector
:math:`\mathbf{x}\in\mathbb{R}^{N_\gamma}` with the standard normal density
:math:`p(\mathbf{x})`,

.. math::

   e^{-\Delta\tau \hat{H}} = \int d\mathbf{x}\, p(\mathbf{x})\, \hat{B}(\mathbf{x}),
   \qquad
   \hat{B}(\mathbf{x}) = e^{-\frac{\Delta\tau}{2}\hat{H}_1}\,
                         e^{\,i\sqrt{\Delta\tau}\,\mathbf{x}\cdot\hat{\mathbf{v}}}\,
                         e^{-\frac{\Delta\tau}{2}\hat{H}_1}.

Every :math:`\hat{B}(\mathbf{x})` is the exponential of a one-body operator,
and by Thouless' theorem it maps a Slater determinant onto another Slater
determinant. The many-body propagator has thus been traded for a
high-dimensional integral over one-body propagators that can be sampled by
Monte Carlo.

Mean-field subtraction
~~~~~~~~~~~~~~~~~~~~~~

The fluctuations of the auxiliary fields are reduced by subtracting the
trial-state expectation value :math:`\bar{v}_\gamma = \langle \hat{v}_\gamma\rangle_T`
from each operator before the HS transformation [Motta2018]_:

.. math::

   \frac{1}{2}\sum_\gamma \hat{v}_\gamma^{2}
   = \frac{1}{2}\sum_\gamma (\hat{v}_\gamma - \bar{v}_\gamma)^{2}
     + \sum_\gamma \bar{v}_\gamma \hat{v}_\gamma
     - \frac{1}{2}\sum_\gamma \bar{v}_\gamma^{2}.

The middle term is absorbed into the one-body propagator and the last term is a
constant. In :class:`ipie.propagation.phaseless_base.PhaselessBase` the shift
is computed by the module-level function ``construct_mean_field_shift`` and
stored as ``mf_shift`` (which holds :math:`i\bar{v}_\gamma`), the shifted
one-body propagator :math:`e^{-\frac{\Delta\tau}{2}\hat{H}'_1}` is built once
by ``construct_one_body_propagator`` (stored as ``expH1``), and the two-body
factor is applied by the ``apply_VHS`` method of
:class:`ipie.propagation.phaseless_generic.PhaselessGeneric`, which forms
:math:`\hat{V}_{\mathrm{HS}} = i\sqrt{\Delta\tau}\sum_\gamma x_\gamma L^\gamma`
as a dense :math:`M\times M` matrix per walker and applies its exponential to
the walker orbitals as a Taylor series truncated at order ``exp_nmax``
(default 6). Because the shifted operators :math:`\hat{v}_\gamma-\bar{v}_\gamma`
are used in the HS transformation, each sampled propagator also carries the
c-number factor :math:`e^{-i\sqrt{\Delta\tau}\,\mathbf{x}\cdot\bar{\mathbf{v}}}`,
which enters the walker weight and phase below.

Walkers, weights and the mixed estimator
----------------------------------------

The propagated state is represented by an ensemble of :math:`N_w` walkers,
each a Slater determinant :math:`|\phi_k\rangle` with a weight :math:`w_k`,

.. math::

   |\Psi(\tau)\rangle \simeq \sum_{k=1}^{N_w} w_k\,
   \frac{|\phi_k\rangle}{\langle\Psi_T|\phi_k\rangle},

where :math:`|\Psi_T\rangle` is the trial wavefunction. A walker stores its
:math:`\alpha` and :math:`\beta` orbital coefficient matrices
(:math:`M\times N_\alpha` and :math:`M\times N_\beta`) in the class
:class:`ipie.walkers.uhf_walkers.UHFWalkers`; the generalised (spin-orbital)
variant :class:`ipie.walkers.ghf_walkers.GHFWalkers` stores a single
:math:`2M\times N` matrix. The number of walkers per MPI rank is set by
``num_walkers`` (JSON key ``nwalkers``); the total population is
``num_walkers`` times the number of ranks.

Ground-state properties are extracted with the *mixed estimator*, which for
the energy reads

.. math::

   E_{\mathrm{mixed}}(\tau)
   = \frac{\langle\Psi_T|\hat{H}|\Psi(\tau)\rangle}{\langle\Psi_T|\Psi(\tau)\rangle}
   \simeq \frac{\sum_k w_k\, E_L[\phi_k]}{\sum_k w_k},
   \qquad
   E_L[\phi] = \frac{\langle\Psi_T|\hat{H}|\phi\rangle}{\langle\Psi_T|\phi\rangle}.

Because :math:`\hat{H}` commutes with the projector, the mixed estimator of
the energy converges to the exact ground-state energy (within the phaseless
approximation) and is not biased by the quality of :math:`|\Psi_T\rangle`
beyond the constraint discussed below. The local energy :math:`E_L` is
evaluated with the generalised Wick theorem from the walker Green's function

.. math::

   G_{pq}[\phi] = \frac{\langle\Psi_T|\hat{a}^\dagger_p \hat{a}_q|\phi\rangle}
                       {\langle\Psi_T|\phi\rangle}
                = \left[\Psi_T^{*}\,(\phi^{T}\Psi_T^{*})^{-1}\phi^{T}\right]_{pq}
                = \left[\phi\,(\Psi_T^\dagger \phi)^{-1}\Psi_T^\dagger\right]_{qp},

.. math::

   E_L[\phi] = E_0 + \sum_{pq} h_{pq} G_{pq}
             + \frac{1}{2}\sum_\gamma\Big[
               \Big(\sum_{pr} L^\gamma_{pr} G_{pr}\Big)^{2}
               - \sum_{prqs} L^\gamma_{pr} L^\gamma_{qs}\, G_{ps} G_{qr}\Big],

where the two terms in brackets are the Coulomb and exchange contributions
(the :math:`\langle\hat{a}^\dagger_p\hat{a}_q\rangle` index convention above is
the one in which ipie stores ``G``, and the contractions are written in that
convention). ipie evaluates these expressions in
:mod:`ipie.estimators.local_energy_sd` (single-determinant trials) using
half-rotated quantities (see below), and in
:mod:`ipie.estimators.local_energy_wicks` for particle-hole multi-determinant
trials (NOCI trials use :mod:`ipie.estimators.local_energy_noci`). The
energy estimator :class:`ipie.estimators.energy.EnergyEstimator` accumulates
the numerator :math:`\sum_k w_k E_L[\phi_k]` and denominator
:math:`\sum_k w_k` (columns ``ENumer`` and ``EDenom`` of the output) and reports
their ratio as ``ETotal``.

Importance sampling and the force bias
--------------------------------------

Sampling :math:`\mathbf{x}` directly from :math:`p(\mathbf{x})` is extremely
inefficient because most sampled determinants have small overlap with the
ground state. Importance sampling shifts the Gaussian by a walker-dependent
*force bias* :math:`\bar{\mathbf{x}}` [Zhang2003]_ [Purwanto2004]_. Inserting
:math:`p(\mathbf{x}) = p(\mathbf{x}-\bar{\mathbf{x}})\,
e^{-\mathbf{x}\cdot\bar{\mathbf{x}} + \bar{\mathbf{x}}^{2}/2}` and changing
variables gives

.. math::

   |\phi'\rangle = \hat{B}(\mathbf{x}-\bar{\mathbf{x}})\,|\phi\rangle,
   \qquad
   w' = w\, I(\mathbf{x},\bar{\mathbf{x}},\phi),
   \qquad
   I(\mathbf{x},\bar{\mathbf{x}},\phi)
   = \frac{\langle\Psi_T|\phi'\rangle}{\langle\Psi_T|\phi\rangle}\,
     e^{\,\mathbf{x}\cdot\bar{\mathbf{x}} - \bar{\mathbf{x}}^{2}/2},

with :math:`\mathbf{x}` still drawn from the standard normal distribution.
The importance function :math:`I` is made as smooth as possible (its
fluctuations cancel to first order in :math:`\sqrt{\Delta\tau}`) by choosing
the optimal force bias

.. math::

   \bar{x}_\gamma = -\,i\sqrt{\Delta\tau}\,
   \big(\langle \hat{v}_\gamma\rangle_{\phi} - \bar{v}_\gamma\big),
   \qquad
   \langle \hat{v}_\gamma\rangle_{\phi} = \sum_{pr} L^\gamma_{pr}\, G_{pr}[\phi].

This is the quantity returned by ``calc_force_bias`` of the trial wavefunction
classes (implemented in :mod:`ipie.propagation.force_bias` for single
determinants). Its cost per walker is :math:`\mathcal{O}(N_\gamma N M)` when
half-rotated Cholesky vectors are used.

The force bias can occasionally become very large, for example when a walker
approaches the nodal surface of the trial state. Such outliers are the main
source of instabilities in ph-AFQMC, so ipie bounds the shift: any component
with :math:`|\bar{x}_\gamma|` larger than ``fb_bound`` (default 1.0) is
rescaled to unit modulus while retaining its phase
(``apply_bound_force_bias``; note that the rescaled modulus is 1, not
``fb_bound``, in the current implementation). The propagator counts how often
the bound is applied in its attribute ``nfb_trig`` (``afqmc.propagator.nfb_trig``
after a run); the count is not printed automatically. A large count is a sign
that the timestep or trial wavefunction is inadequate.

The phaseless approximation
---------------------------

For a general (complex or non-positive) Hamiltonian the importance function
:math:`I` is complex, and the walker weights acquire phases that are
uniformly distributed after long propagation. This is the fermion sign
(phase) problem: the signal-to-noise ratio of every estimator then decays
exponentially with :math:`\tau`. The phaseless approximation of Zhang and
Krakauer [Zhang2003]_, which generalises the constrained-path approximation for
real propagators [Zhang1997]_, removes it by imposing a gauge-like constraint
with respect to the trial state. Writing

.. math::

   \Delta\theta = \arg\!\left[\frac{\langle\Psi_T|\phi'\rangle}{\langle\Psi_T|\phi\rangle}\,
   e^{-i\sqrt{\Delta\tau}\,(\mathbf{x}-\bar{\mathbf{x}})\cdot\bar{\mathbf{v}}}\right],

which includes the c-number phase produced by the mean-field subtraction (in
ipie it is obtained as the imaginary part of :math:`-\Delta\tau E_{\mathrm{hyb}}`
minus the force-bias term, see below), the weight is updated with the modulus
of the importance function and a *cosine projection* of the phase rotation,

.. math::

   w' = w\; \big|I(\mathbf{x},\bar{\mathbf{x}},\phi)\big|\;
   \max\!\big(0,\cos\Delta\theta\big).

Walkers whose overlap with the trial state acquires a phase rotation larger
than :math:`\pi/2` in a single step are eliminated, and the random walk is
confined to the half-plane of positive real overlap with
:math:`|\Psi_T\rangle`. The resulting bias vanishes when :math:`|\Psi_T\rangle`
is exact and is otherwise small and systematically improvable through the
trial wavefunction; the walkers' distribution follows
:math:`\mathrm{Re}\,\langle\Psi_T|\phi\rangle`, which makes the method
variational in neither direction but usually accurate to within a few
:math:`\mathrm{mE_h}` of exact results with good trials [Lee2022]_.

Hybrid versus local-energy weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two ways of evaluating :math:`|I|` exist. In the *local energy* formulation the
overlap ratio is expanded to leading order in :math:`\Delta\tau`, giving
:math:`|I| \approx e^{-\Delta\tau\,\mathrm{Re}\,[E_L(\phi)+E_L(\phi')]/2 + \Delta\tau E_{\mathrm{shift}}}`.
This requires the local energy at every step, which is the most expensive
quantity in the calculation. In the *hybrid* formulation [Purwanto2004]_
[Motta2018]_ one instead defines the *hybrid energy*

.. math::

   E_{\mathrm{hyb}}(\phi\to\phi') = -\frac{1}{\Delta\tau}\ln
   I(\mathbf{x},\bar{\mathbf{x}},\phi),

whose real part plays the role of the local energy and whose imaginary part
yields :math:`\Delta\theta`. ipie always uses the hybrid formulation. In
``PhaselessBase.update_weight`` the hybrid energy is formed from the overlap
ratio together with the force-bias factor and the constant arising from the
mean-field shift, and the weight is updated as

.. math::

   w' = w\; \exp\!\Big[-\Delta\tau\,\mathrm{Re}\Big(\tfrac{1}{2}\big(E_{\mathrm{hyb}}^{\,\mathrm{new}}
        + E_{\mathrm{hyb}}^{\,\mathrm{old}}\big) - E_{\mathrm{shift}}\Big)\Big]
        \; \max(0,\cos\Delta\theta),

where the average over the current and the previous step is the usual
trapezoidal estimate. :math:`E_{\mathrm{shift}}` is a running estimate of the
ground-state energy that keeps the total weight roughly constant. In ipie it is
the weighted average of the hybrid energies over the previous block (the
``HybridEnergy`` column of the output) and is zero during the first block. The
local energy is then needed only when the energy estimator is evaluated, once
every ``num_steps_per_block`` steps, which is what allows ipie to use
comparatively expensive trial wavefunctions.

Energy bound
~~~~~~~~~~~~

Rare walkers with very large :math:`|E_{\mathrm{hyb}}|` would otherwise
dominate the population. Following the standard practice
[Purwanto2009]_ [Lee2022]_, ipie clips the real part of the hybrid energy of
each walker to the window

.. math::

   E_{\mathrm{shift}} - \sqrt{\frac{c}{\Delta\tau}}
   \;\le\; \mathrm{Re}\,E_{\mathrm{hyb}} \;\le\;
   E_{\mathrm{shift}} + \sqrt{\frac{c}{\Delta\tau}},

with :math:`c` given by the option ``ene_bound_const`` (default 2.0). The
bound is not applied while :math:`E_{\mathrm{shift}}` is still zero, i.e.
during the first block. Both ``fb_bound`` and ``ene_bound_const`` are
attributes of :class:`ipie.qmc.options.QMCParams` and arguments of
:meth:`ipie.qmc.afqmc.AFQMC.build`. Only the force-bias bound is counted
(``propagator.nfb_trig``, see above); the hybrid-energy bound is not counted
and neither count is printed.

Population control
------------------

Even with importance sampling and the energy shift the weights of individual
walkers drift apart over time, and after many steps a few walkers would carry
almost all of the weight. Every ``pop_control_freq`` steps (default 5) ipie
therefore performs population control with
:class:`ipie.walkers.pop_controller.PopController`. First the weights are
rescaled by the ratio of the target total weight (equal to the total number of
walkers) to the current total weight; the ``WeightFactor`` column of the output
records the total weight before rescaling and is what makes the energy shift
self-consistent. Then a branching step redistributes the population while
leaving the weighted ensemble statistically unchanged. Three algorithms are
available through ``pop_control_method``:

``pair_branch`` (default)
   Walkers are sorted by weight and paired from the two ends of the list. When
   the lighter walker of a pair falls below ``min_weight`` (0.1) or the heavier
   one exceeds ``max_weight`` (4.0), one of the two is chosen with probability
   proportional to its weight, cloned, and both copies receive the mean weight
   of the pair; the other walker is removed. This introduces the least amount
   of noise and was adopted following the analysis of Zhang and co-workers
   [Zhang1997]_ [Purwanto2004]_.

``comb``
   The comb method of Booth and Gubernatis [Booth2009]_, in which the total
   weight is divided into :math:`N_w` equal intervals and a random offset
   selects the surviving walkers.

``stochastic_reconfiguration``
   Walkers are resampled with probabilities proportional to their weights and
   the survivors are assigned equal weights; this is also the method used for
   correlated sampling, where the same branching decisions can be replayed
   across runs.

The population control step is the only place where walkers are exchanged
between MPI ranks; the ``pop_control_freq`` option therefore also controls the
communication overhead. Population control introduces a small bias that
decreases with the number of walkers and can be checked by increasing
``num_walkers``.

Stabilisation and the timestep
------------------------------

Repeated multiplication by :math:`\hat{B}(\mathbf{x})` causes the columns of a
walker's orbital matrix to collapse onto the lowest-lying one-particle states
and lose linear independence, which destroys the numerical accuracy of the
determinants and inverses needed for the Green's function. Every
``stabilize_freq`` steps (JSON key ``stabilise_freq``; attribute
``num_stblz``, default 5) ipie re-orthogonalises each walker with a QR
decomposition
:math:`\phi = QR` (method ``reortho`` of the walker classes). The orthonormal
factor :math:`Q` replaces :math:`\phi`; because the Slater determinant is
unchanged up to normalisation, the walker weight is unaffected and only the
stored overlap is divided by :math:`\det R` (accumulated in log form to avoid
overflow). Free-projection calculations, which need the norm, keep track of
:math:`\det R` explicitly.

The timestep controls three distinct errors: the Trotter error
(:math:`\mathcal{O}(\Delta\tau^{2})`), the error from truncating the HS
exponential and the phaseless bias, all of which decrease with
:math:`\Delta\tau`, at the cost of needing more steps to reach a given
imaginary time and larger autocorrelation between blocks. Production
calculations typically use :math:`\Delta\tau = 0.005\text{--}0.01\,
\mathrm{Ha}^{-1}` and check convergence with a second timestep or an
extrapolation to :math:`\Delta\tau\to 0`. A related consideration is
equilibration: the projection must run for a few
:math:`\mathrm{Ha}^{-1}` before the walker distribution is stationary, and the
early blocks are discarded during analysis.

Blocks and statistics
~~~~~~~~~~~~~~~~~~~~~

The simulation is organised into ``num_blocks`` blocks of
``num_steps_per_block`` propagation steps each (defaults 100 and 25 in the
Python API; total imaginary time
:math:`\tau_{\max} = \texttt{num\_blocks}\times\texttt{num\_steps\_per\_block}\times\Delta\tau`).
Within a block the walkers are propagated, re-orthogonalised every
``stabilize_freq`` steps and population-controlled every ``pop_control_freq``
steps; at the end of a block the energy estimator (and any other requested
estimator) is evaluated and written to the output file, and the energy shift is
updated. Consecutive block averages are correlated, so the statistical error of
the mean must be estimated by reblocking [Flyvbjerg1989]_ or from the
integrated autocorrelation time [Jonsson2018]_; both are implemented in
:mod:`ipie.analysis` and described in :doc:`analysis`. A separate equilibration
phase with its own timestep and frequencies can be requested with the ``eq_``
options (``num_eq_blocks`` etc.); by default it is not used and equilibration
is handled by discarding blocks in the analysis.

Trial wavefunctions
-------------------

The trial wavefunction enters through the overlaps, Green's functions and
force biases, and it determines the accuracy of the phaseless constraint. ipie
supports the following families, all deriving from
:class:`ipie.trial_wavefunction.wavefunction_base.TrialWavefunctionBase`.

Single Slater determinants
   :class:`ipie.trial_wavefunction.single_det.SingleDet` accepts RHF, ROHF or
   UHF orbitals as a single :math:`M\times(N_\alpha+N_\beta)` coefficient
   matrix; RHF orbitals are simply duplicated for the :math:`\alpha` and
   :math:`\beta` blocks. For closed-shell RHF trials the :math:`\beta` block can
   be skipped by setting ``walkers.rhf = True`` after ``walkers.build(trial)``
   (the kernels contain an RHF fast path), but this is not detected
   automatically. Generalised Hartree-Fock (GHF) trials
   with spin-orbital coefficients are provided by
   :class:`ipie.trial_wavefunction.single_det_ghf.SingleDetGHF` together with
   :class:`ipie.walkers.ghf_walkers.GHFWalkers` [Jiang2024]_.

Particle-hole multi-determinant expansions
   :class:`ipie.trial_wavefunction.particle_hole.ParticleHole` represents a
   CI-type expansion :math:`|\Psi_T\rangle = \sum_I c_I |D_I\rangle` in which all
   determinants share the same orthonormal orbitals and are labelled by
   their excitation (particle-hole) pattern relative to a reference, e.g. the
   output of a selected-CI (heat-bath CI) or CASSCF calculation. Overlaps,
   Green's functions and the local energy are evaluated with the generalised
   Wick theorem algorithm of Mahajan and Sharma [Mahajan2021]_ [Mahajan2022]_,
   whose cost grows only linearly with the number of determinants
   :math:`N_{\mathrm{det}}` rather than :math:`N_{\mathrm{det}} N^2`. The number
   of determinants retained is set by ``num_dets_for_trial``, and the
   expansion can be processed in chunks (``num_det_chunks``) to limit memory
   usage. The Wick-theorem estimators live in
   :mod:`ipie.estimators.local_energy_wicks`.

Non-orthogonal configuration interaction
   :class:`ipie.trial_wavefunction.noci.NOCI` handles linear combinations of
   determinants with different orbital sets,
   :math:`|\Psi_T\rangle = \sum_I c_I |\phi_I\rangle`. Each determinant
   contributes a separate overlap and Green's function, so the cost scales as
   :math:`N_{\mathrm{det}}` times that of a single-determinant trial.

Half-rotated integrals
~~~~~~~~~~~~~~~~~~~~~~

For single-determinant (and particle-hole reference) trials the Green's
function :math:`G = \Psi_T^{*}\,(\phi^{T}\Psi_T^{*})^{-1}\phi^{T}` factorises,
and every trial-dependent contraction can be pre-computed. Defining the
*half-rotated* one-body integrals and Cholesky vectors

.. math::

   \tilde{h}_{iq} = \sum_p \Psi^{*}_{T,pi}\, h_{pq},
   \qquad
   \tilde{L}^{\gamma}_{iq} = \sum_p \Psi^{*}_{T,pi}\, L^{\gamma}_{pq},
   \qquad i = 1,\dots,N_\sigma,

(with the unmodified one-body integrals :math:`h`; the modified :math:`h'`
enters only the one-body propagator) together with the *half* Green's function
:math:`\Theta = (\phi^{T}\Psi_T^{*})^{-1}\phi^{T}` of shape
:math:`N_\sigma\times M`, the force bias becomes
:math:`\langle\hat{v}_\gamma\rangle_\phi = \sum_{iq}\tilde{L}^\gamma_{iq}\Theta_{iq}`
and the exchange energy becomes a sum over :math:`N_\gamma` products of two
:math:`N_\sigma\times N_\sigma` matrices. This reduces the per-walker cost of
the local energy from :math:`\mathcal{O}(N_\gamma M^{2} N)` to
:math:`\mathcal{O}(N_\gamma M N^{2})`, the dominant scaling of ph-AFQMC
[Motta2018]_ [Malone2023]_. The half-rotation is performed by the
``half_rotate`` method of the trial wavefunction (functions in
:mod:`ipie.trial_wavefunction.half_rotate`), which stores the results as
``_rH1a``, ``_rH1b``, ``_rchola`` and ``_rcholb``; these are the arrays used by
the kernels in :mod:`ipie.estimators.local_energy_sd`. The same trick makes
it possible to keep only the half-rotated Cholesky vectors in GPU memory and
to distribute them across ranks for large systems.

Free-projection AFQMC
---------------------

Free-projection (fp-)AFQMC omits both the phaseless constraint and the
force-bias importance sampling: the auxiliary fields are drawn from the bare
Gaussian (only the mean-field shift and a constant energy shift ``ene_0`` are
applied), each walker carries a modulus and a separate phase, and the energy
is estimated from the ratio of complex-weighted sums
:math:`\sum_k w_k E_L[\phi_k]/\sum_k w_k`. The
result is exact in the limit of infinite sampling and vanishing timestep but
suffers from the sign problem, so the statistical error grows exponentially
with imaginary time; it is nonetheless useful as a benchmark for short
projection times from good initial states, in particular when combined with
trials such as CCSD [Jiang2024]_. ipie implements it in
:class:`ipie.addons.free_projection.propagation.free_propagation.FreePropagation`
and the driver
:class:`ipie.addons.free_projection.qmc.fp_afqmc.FPAFQMC`; see :doc:`advanced`.

Finite-temperature AFQMC
------------------------

Finite-temperature AFQMC [Zhang1999]_ [Rubenstein2012]_ samples the grand
canonical partition function :math:`\mathrm{Tr}\, e^{-\beta(\hat{H}-\mu\hat{N})}`
at inverse temperature :math:`\beta` and chemical potential :math:`\mu`. The
imaginary-time interval :math:`[0,\beta]` is discretised into
:math:`\beta/\Delta\tau` slices, the HS transformation is applied to each, and
the trace over Fock space of a product of one-body propagators is evaluated as
a determinant, :math:`\mathrm{Tr}\prod_l e^{\hat{A}_l} = \det(1 + \prod_l e^{A_l})`.
Walkers are thus products of :math:`M\times M` matrices rather than Slater
determinants, stabilised with a propagator stack, and the phaseless constraint
is imposed with respect to a mean-field (one-body) trial density matrix. The
implementation lives in :mod:`ipie.addons.thermal`, with the driver
:class:`ipie.addons.thermal.qmc.thermal_afqmc.ThermalAFQMC` and walkers
:class:`ipie.addons.thermal.walkers.uhf_walkers.UHFThermalWalkers`; see
:doc:`advanced` and [Jiang2024]_.

Further reading
---------------

The symmetry properties of the AFQMC random walk and the importance of
symmetry-preserving trial wavefunctions are discussed by Shi and Zhang
[Shi2013]_; the use of the frozen-core approximation and excited-state
targeting with ph-AFQMC by Purwanto, Zhang and Krakauer [Purwanto2009]_.
Comprehensive benchmarks of ph-AFQMC for main-group chemistry are collected in
[Lee2022]_. The phaseless constraint can also be made self-consistent by
feeding the AFQMC one-body density matrix back into the independent-particle
calculation that generates the trial state [Qin16]_, and for lattice models
the symmetry-broken (spin- and charge-density-wave) character of the
Hartree-Fock trial state [Xu11]_ strongly affects the quality of the
constraint.
