Examples
========

The ``examples/`` directory of the ipie repository contains a set of small,
self-contained scripts that exercise most of the code paths described in
:doc:`quickstart`, :doc:`python_api` and :doc:`advanced`. Each example lives in
a numbered directory (there is no ``09``) and, unless stated otherwise, is run
from inside that directory with

.. code-block:: bash

   python run_afqmc.py

Every script that supports MPI can also be launched with
``mpirun -np N python run_afqmc.py``. Some examples (02, 06, 11, 15 and 16)
divide a fixed total walker population by ``comm.size`` so that the total
number of walkers is independent of the number of ranks; the others specify the
number of walkers *per rank*, so their total population grows with ``N``. The
full example tree is at
https://github.com/JoonhoLee-Group/ipie/tree/develop/examples.

Prerequisites
-------------

Almost all examples generate their integrals and trial wavefunctions with
`PySCF <https://pyscf.org>`_, which is not a hard dependency of ipie and must be
installed separately (``pip install pyscf``). Several examples need further
optional packages; these are listed in the table below and again in each
section. Most scripts that need ``cupy``, ``mpi4py``, ``trexio``, ``fqe`` or
the ``shciscf`` plugin call ``sys.exit(0)`` when the import fails, so that the
example suite still passes on machines without, for example, a GPU;
``18-ad_afqmc`` is the exception and imports ``torch`` unconditionally, so it
fails with an ``ImportError`` when ``torch`` is missing.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Example
     - Extra dependencies
     - Demonstrates
   * - :ref:`01-simple <ex-01>`
     - pyscf
     - Legacy JSON-input workflow with ``pyscf_to_ipie.py`` and ``bin/ipie``
   * - :ref:`02-multi_determinant <ex-02>`
     - pyscf
     - CASSCF multi-Slater (particle-hole) trial wavefunction
   * - :ref:`03-custom_observable <ex-03>`
     - pyscf
     - Writing custom estimators (mixed 1-RDM)
   * - :ref:`04-s2_observable <ex-04>`
     - pyscf
     - Custom :math:`\langle S^2 \rangle` estimator
   * - :ref:`05-frozen_core <ex-05>`
     - pyscf
     - Frozen-core integrals with ``--frozen-core``
   * - :ref:`06-gpu <ex-06>`
     - pyscf, cupy, mpi4py
     - GPU execution and Cholesky vectors distributed over GPUs
   * - :ref:`07-custom_trial <ex-07>`
     - pyscf
     - Subclassing a trial wavefunction and energy estimator
   * - :ref:`08-custom_walker <ex-08>`
     - pyscf
     - Subclassing walkers and assembling the driver by hand
   * - :ref:`10-pyscf_interface <ex-10>`
     - pyscf
     - Minimal PySCF to AFQMC workflow with analysis
   * - :ref:`11-trexio <ex-11>`
     - trexio
     - Reading integrals and a CI wavefunction from a TREXIO file
   * - :ref:`12-shciscf-active-space <ex-12>`
     - pyscf, Dice, shciscf plugin
     - Automatic active-space selection and SHCISCF trial
   * - :ref:`13-free_projection <ex-13>`
     - pyscf
     - Free-projection AFQMC add-on
   * - :ref:`14-fqe-wavefunction <ex-14>`
     - pyscf, fqe
     - Converting wavefunctions between FQE and ipie
   * - :ref:`15-share_mem_cpu <ex-15>`
     - pyscf, mpi4py
     - Integrals in MPI shared memory on a node
   * - :ref:`16-ft_afqmc <ex-16>`
     - none
     - Finite-temperature AFQMC add-on for the UEG
   * - :ref:`17-ghf_afqmc <ex-17>`
     - pyscf
     - GHF trial wavefunctions and GHF walkers
   * - :ref:`18-ad_afqmc <ex-18>`
     - pyscf, torch
     - Automatic-differentiation AFQMC for a dipole moment
   * - :ref:`19-hubbard <ex-19>`
     - pyscf
     - One-dimensional Hubbard model through the generic Hamiltonian
   * - :ref:`20-save_and_restart <ex-20>`
     - pyscf
     - Writing and reading walker checkpoints
   * - :ref:`21-kpt_chol <ex-21>`
     - mpi4py (cupy for the GPU script)
     - Periodic k-point Cholesky Hamiltonian
   * - :ref:`22-kpt_isdf <ex-22>`
     - mpi4py, cupy
     - Periodic k-point ISDF Hamiltonian on GPUs

Running the examples as a test suite
------------------------------------

The examples double as an integration test. The GitHub Actions workflow
(``.github/workflows/ci.yml``, job ``examples``) installs ``requirements.txt``
plus ``pyblock``, ``torch``, ``pyscf`` and ``fqe`` and then runs, from the
repository root,

.. code-block:: bash

   python dev/run_tests.py --examples

``dev/run_tests.py`` (function ``run_examples``) treats two directories as
"legacy" examples: for ``01-simple`` it runs ``scf.py`` followed by
``tools/pyscf/pyscf_to_ipie.py -i scf.chk``, and for ``05-frozen_core`` it does
the same with ``--frozen-core 5``. For every other example directory it simply
executes ``python -u <dir>/run_afqmc.py``. Consequently only the
``run_afqmc.py`` entry point is exercised; auxiliary scripts such as
``06-gpu/run_afqmc_chunked.py``, ``21-kpt_chol/run_afqmc_gpu.py`` and the
notebook in ``13-free_projection`` are not run by CI. Because the runner does
not change directory, intermediate files (``scf.chk``, ``hamiltonian.h5``,
``wavefunction.h5``, ``estimates.0.h5``) are written to the repository root.
Note that ``mpi4py`` is not installed in that CI job, so examples that require
it exit early without testing anything.

An older shell script, ``tools/run_examples.sh``, runs examples 01 to 05, 07, 08
and 10 by changing into each directory in turn (it must be launched from the
repository root). It predates the current layout of ``02-multi_determinant``
(it expects a ``scf.py`` there that no longer exists) and is kept mainly for
reference; ``dev/run_tests.py --examples`` is the maintained entry point. The
other modes of ``dev/run_tests.py`` (``--pytest``, ``--mpi``, ``--integration``,
``--pylint``, ``--black``, ``--flynt``, ``--all``) run the unit and MPI tests
and the code-style checks.

.. _ex-01:

01-simple: the JSON-input workflow
----------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/01-simple>`__

This is the classical two-step workflow described in :doc:`input_file`. The
``scf.py`` script runs a PySCF UHF calculation for a ten-atom hydrogen chain
(STO-6G, 1.6 bohr spacing) and, crucially, sets ``mf.chkfile = "scf.chk"``.
The checkpoint is then converted into ipie's HDF5 inputs with the command-line
tool

.. code-block:: bash

   python scf.py
   python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json

which calls :func:`ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk` to write
``hamiltonian.h5`` (Cholesky-decomposed integrals in the MO basis) and
``wavefunction.h5`` (the UHF determinant), and
:func:`ipie.utils.io.write_json_input_file` to write a template ``input.json``.
After adjusting ``dt``, ``nwalkers`` and ``blocks`` the calculation is run
through the ``bin/ipie`` executable:

.. code-block:: bash

   mpirun -np N python /path/to/ipie/bin/ipie input.json > output.dat
   python /path/to/ipie/tools/reblock.py -b 10 -f output.dat

The README compares the result with the Simons hydrogen-chain benchmark value
of :math:`-5.3819(6)` Ha and recommends at least 1000 walkers in production.
(The README still refers to an ``afqmc.h5`` output file; the converter now
writes ``hamiltonian.h5`` and ``wavefunction.h5`` as shown in its own JSON
snippet.) Dependencies: pyscf; mpi4py optional.

.. _ex-02:

02-multi_determinant: CASSCF trial wavefunction
-----------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/02-multi_determinant>`__

Runs RHF followed by a CASSCF(6,6) calculation on triplet
:math:`\mathrm{N}_2` (cc-pVDZ, 3.0 bohr) with PySCF, extracts the CI
expansion with ``pyscf.fci.addons.large_ci`` and stores the coefficients and
occupation strings under ``mcscf/`` in the checkpoint file. Passing
``mcscf=True`` to :func:`ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk`
then writes a particle-hole wavefunction file. The script reads
``hamiltonian.h5`` and ``wavefunction.h5`` back and assembles the driver
explicitly:

* :func:`ipie.hamiltonians.generic.Generic` to build the Hamiltonian from
  ``hcore``, ``LXmn`` and ``e0``;
* :class:`ipie.trial_wavefunction.particle_hole.ParticleHole` with
  ``num_dets_for_props`` set to the full expansion and
  ``compute_trial_energy = True``;
* :class:`ipie.walkers.uhf_walkers.UHFWalkersParticleHole` initialised from a
  randomly perturbed, re-orthogonalised copy of the reference determinant;
* :meth:`ipie.qmc.afqmc.AFQMC.build` with ``640 // comm.size`` walkers.

The script calls ``config.update_option("use_gpu", False)`` explicitly after
the MPI import and carries a commented-out
``config.update_option("use_gpu", True)`` line next to it, showing where GPU
execution would be switched on (as explained under :ref:`ex-06`, the switch
only takes effect if it runs before ``ipie.utils.backend`` is first imported,
which the import order of this script does not satisfy); see :doc:`advanced`
for the multi-determinant (Wick's theorem) machinery it relies on. Run with
``python run_afqmc.py`` or ``mpirun -np N python run_afqmc.py``. Dependencies:
pyscf.

.. _ex-03:

03-custom_observable: writing your own estimator
------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/03-custom_observable>`__

After building a driver for the H\ :sub:`10` chain with
:func:`ipie.qmc.calc.build_afqmc_driver`, the script defines two estimators by
subclassing :class:`ipie.estimators.estimator_base.EstimatorBase`: ``Diagonal1RDM``
accumulates the diagonal of the mixed-estimate one-particle reduced density
matrix and ``Mixed1RDM`` the full :math:`(2, M, M)` array. The pattern to copy
is: store numerator and denominator buffers in ``self._data``, set
``self._shape`` and ``self.scalar_estimator = False``, and implement
``compute_estimator`` using ``trial.calc_greens_function(walkers,
build_full=True)`` and ``walkers.Ga``/``walkers.Gb``. The estimators are passed
to :meth:`ipie.qmc.afqmc.AFQMC.run` through ``additional_estimators`` and read
back with :func:`ipie.analysis.extraction.extract_observable`, which returns an
array of shape ``(num_blocks + 1, *shape)`` for non-scalar estimators. The
script asserts that the trace of the RDM equals the electron count. This is
also the example that the wheel-publishing workflow runs as a smoke test.
Dependencies: pyscf.

.. _ex-04:

04-s2_observable: a mixed :math:`\langle S^2 \rangle` estimator
---------------------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/04-s2_observable>`__

A stretched :math:`\mathrm{H}_2` molecule (3.2 Angstrom, STO-3G) is converged
to a spin-symmetry-broken UHF solution using PySCF's stability analysis. The
``S2Mixed`` estimator (again an :class:`~ipie.estimators.estimator_base.EstimatorBase`
subclass) evaluates
:math:`\langle S^2\rangle = N_\downarrow + M_s(M_s+1) - \sum_w w\,\mathrm{tr}(G^\alpha_w G^\beta_w)`
from the walker Green's functions obtained with
:func:`ipie.estimators.greens_function.greens_function`, prints it to stdout
(``print_to_stdout = True``) and the script checks the first block against the
UHF value :math:`N_\downarrow + M_s(M_s+1) - \mathrm{tr}(P^\alpha S P^\beta S)`.
It uses only 10 walkers per task and 50 blocks. Dependencies: pyscf.

.. _ex-05:

05-frozen_core: freezing core orbitals
--------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/05-frozen_core>`__

A phosphorus atom (6-31G, quartet UHF) is prepared with ``scf.py``, which also
runs a frozen-core UCCSD(T) reference. The converter is invoked with the
``--frozen-core`` flag

.. code-block:: bash

   python scf.py
   python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json --frozen-core 5

which freezes the 1s, 2s and 2p orbitals via
:func:`ipie.utils.from_pyscf.freeze_core`, leaving a CAS(5,8) problem with
``"nup": 4, "ndown": 1`` in the generated ``input.json``. The README lists
reference AFQMC energies for UHF and ROHF trials (about :math:`-340.7047` Ha)
alongside the all-electron numbers, illustrating the size of the frozen-core
error for this system. Run through ``bin/ipie`` as in ``01-simple``.
Dependencies: pyscf.

.. _ex-06:

06-gpu: running on GPUs
-----------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/06-gpu>`__

Three scripts cover GPU execution for a small H\ :sub:`4` chain.

``run_afqmc.py`` is the single-GPU case. It requires ``cupy`` (and exits
quietly otherwise) and shows the essential ordering constraint: ipie's array
backend is selected when :mod:`ipie.utils.backend` is first imported, so

.. code-block:: python

   from ipie.config import config
   config.update_option("use_gpu", True)

must run before any other ipie module is imported (or, equivalently, set
``IPIE_USE_GPU=1`` in the environment). After that the standard
:func:`ipie.qmc.calc.build_afqmc_driver` path is used unchanged. Note that the
script prints ``afqmc.qmc``; the driver exposes its options as
``afqmc.params`` (see :class:`ipie.qmc.options.QMCParams`), so this line raises
an ``AttributeError`` in the current code.

``chunked_chol.py`` prepares input for the multi-GPU case: it runs the SCF,
writes ``hamiltonian.h5`` and calls
:func:`ipie.utils.chunk_large_chol.split_cholesky` to split the Cholesky
vectors into ``chol_0.h5`` ... ``chol_3.h5``.

``run_afqmc_chunked.py`` distributes those chunks over four GPUs that
collectively hold one copy of the Hamiltonian. It builds
:class:`ipie.utils.mpi.MPIHandler` with ``nmembers=4``, binds each rank to a
device with ``xp.cuda.Device(rank % gpu_number_per_node).use()``, has each rank
read its own ``chol_{srank}.h5`` chunk, packs it with
:func:`ipie.utils.pack_numba.pack_cholesky`, and constructs a
:class:`ipie.hamiltonians.generic_chunked.GenericRealCholChunked` Hamiltonian,
a :class:`ipie.trial_wavefunction.single_det.SingleDet` trial and
:class:`ipie.walkers.uhf_walkers.UHFWalkers`, all carrying the handler. Run it
with a rank count that is a multiple of ``nmembers``:

.. code-block:: bash

   python chunked_chol.py
   IPIE_USE_GPU=1 mpirun -np 4 python run_afqmc_chunked.py

Unlike ``run_afqmc.py``, this script imports ``ipie.utils.backend`` (directly,
and through ``GenericRealCholChunked``) before it reaches its own
``config.update_option("use_gpu", True)`` call, so that call comes too late to
select cupy and ``xp.cuda`` would fail with an ``AttributeError``; exporting
``IPIE_USE_GPU=1`` in the environment before launching, as shown above, sets
the option before the first import. (The chunked script also does
``from chunked_chol import *``, so the SCF and splitting are repeated on
import.) Dependencies: pyscf, cupy, mpi4py. See :doc:`advanced` for how
chunking works.

.. _ex-07:

07-custom_trial: subclassing a trial wavefunction
-------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/07-custom_trial>`__

Demonstrates the minimal surface a trial wavefunction must implement.
``NoisySingleDet`` inherits from :class:`ipie.trial_wavefunction.single_det.SingleDet`
and overrides ``calc_overlap``, ``calc_greens_function`` and
``calc_force_bias`` to multiply the results by Gaussian noise. The script also
defines a ``NoisyEnergyEstimator`` derived from
:class:`ipie.estimators.energy.EnergyEstimator` that calls ``local_energy_batch``
directly, and injects it under the key ``"energy"`` in ``additional_estimators``,
replacing the default. This shows how to swap the energy estimator; it is not
strictly required here, because the default estimator's ``plum``-dispatched
``local_energy`` also accepts a ``SingleDet`` subclass (the comment in the
script about missing multiple dispatch predates this). The Hamiltonian and trial are built directly from PySCF
objects with :func:`ipie.utils.from_pyscf.generate_hamiltonian` and
:func:`ipie.utils.from_pyscf.generate_wavefunction_from_mo_coeff` (no
intermediate HDF5 files), and the run is analysed with
:func:`ipie.analysis.autocorr.reblock_by_autocorr`. Dependencies: pyscf.

.. _ex-08:

08-custom_walker: subclassing walkers and assembling the driver
---------------------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/08-custom_walker>`__

Extends the previous example to a custom walker class: ``CustomUHFWalkers``
derives from :class:`ipie.walkers.uhf_walkers.UHFWalkers` and overrides
``reortho``. A custom walker object could simply be passed to
:meth:`~ipie.qmc.afqmc.AFQMC.build` through its ``walkers`` argument (as
``02-multi_determinant`` does); this example instead shows the fully manual
route and builds every component by hand:
:class:`ipie.systems.generic.Generic`, the Hamiltonian, the trial, the walkers,
a :class:`ipie.qmc.options.QMCParams` dataclass, and a
:class:`ipie.propagation.phaseless_generic.PhaselessGeneric` propagator whose
``build`` method takes the Hamiltonian, trial, walkers and
:class:`~ipie.utils.mpi.MPIHandler`. These are passed to the
:class:`ipie.qmc.afqmc.AFQMC` constructor directly. This is the lowest-level
way of driving ipie and is the pattern to follow when none of the factory
methods fit. Dependencies: pyscf.

.. _ex-10:

10-pyscf_interface: PySCF to AFQMC in one script
------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/10-pyscf_interface>`__

The recommended minimal workflow for molecules. Rank 0 runs UHF and a
UCCSD(T) reference on the H\ :sub:`10` chain and calls
:func:`ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk`; after a barrier
all ranks construct the driver with :func:`ipie.qmc.calc.build_afqmc_driver`
(100 walkers per task, fixed seed), shorten the run to 10 blocks by editing
``afqmc.params.num_blocks``, and rank 0 post-processes the ``ETotal`` column
with :func:`ipie.analysis.extraction.extract_observable` and
:func:`ipie.analysis.autocorr.reblock_by_autocorr`. Run serially or with
``mpirun -np N python run_afqmc.py``. Dependencies: pyscf.

.. _ex-11:

11-trexio: reading a TREXIO file
--------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/11-trexio>`__

Shows the TREXIO interface. :func:`ipie.utils.from_trexio.gen_ipie_from_trexio`
reads the bundled ``h2o_dz.h5`` (water, double-zeta basis; the script prints
the reference Hartree-Fock and CI energies stored with it) and returns a
dictionary with electron counts, the core Hamiltonian, the Cholesky vectors,
the nuclear repulsion and the CI determinants (``ci_coeffs``, ``occa``,
``occb``). Rank 0 reads and broadcasts these, then a
:func:`~ipie.hamiltonians.generic.Generic` Hamiltonian and a
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHole` trial are built and
walkers are created through the ``UHFWalkersTrial`` dispatcher in
:mod:`ipie.walkers.walkers_dispatch`. The script exits if the ``trexio`` Python
package is missing (``pip install trexio``).

The final driver construction in this script uses an outdated calling
convention (``AFQMC(comm, system=..., nwalkers=...)`` and ``run(comm=comm)``)
that no longer matches :class:`ipie.qmc.afqmc.AFQMC`, and the dispatcher call
omits the required ``mpi_handler`` argument; use :meth:`ipie.qmc.afqmc.AFQMC.build`
as in ``02-multi_determinant`` when adapting it. Dependencies: trexio (numpy
only otherwise).

.. _ex-12:

12-shciscf-active-space: SHCI-selected active spaces
----------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/12-shciscf-active-space>`__

A semi-automatic active-space workflow built on the Dice SHCI code via PySCF's
``shciscf`` plugin. :func:`ipie.utils.from_dice.build_driver_from_shciscf` runs
a coarse SHCI calculation on :math:`\mathrm{N}_2` (cc-pVDZ) in a large
active space that excludes only the two core orbitals, computes natural-orbital occupation numbers
from the SHCI 1-RDM, selects an active space with a NOON threshold
(``noons_thresh=0.05`` here), rotates the orbitals, reruns a tight SHCISCF and
finally builds a :class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`
trial in the *full* orbital space together with a Cholesky Hamiltonian
(``chol_cut=1e-8``). The script checks that the trial energy computed by ipie
matches the SHCISCF total energy; the individual steps are available as
separate functions in :mod:`ipie.utils.from_dice`. The docstring warns that
active-space selection is not black-box and the defaults may need tuning.
Dependencies: pyscf, Dice (which needs MPI, ``num_proc=8`` processes here) and
the ``pyscf.shciscf`` plugin; the script exits early without them.

.. _ex-13:

13-free_projection: free-projection AFQMC
-----------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/13-free_projection>`__

Introduces the :mod:`ipie.addons.free_projection` add-on (see
:doc:`advanced`). ``run_afqmc.py`` prepares the H\ :sub:`10` inputs as in
``10-pyscf_interface`` and then calls
:func:`ipie.addons.free_projection.qmc.calc.build_fpafqmc_driver` with

.. code-block:: python

   qmc_options = {
       "num_iterations_fp": 100,
       "num_blocks": 5,
       "num_steps": 20,
       "num_walkers": 10,
       "dt": 0.05,
   }

so that energies are sampled at five imaginary times :math:`\tau = 1, 2, ..., 5`
and each estimate is averaged over 100 independent repetitions. Because the
estimates are ratios of noisy numerators and denominators, the analysis uses
:func:`ipie.addons.free_projection.analysis.jackknife.jackknife_ratios` on the
``ENumer``/``EDenom`` and phase columns returned by
:func:`ipie.addons.free_projection.analysis.extraction.extract_observable`.

The accompanying notebook ``fp_afqmc.ipynb`` goes further: it compares FCI,
free-projection and phaseless AFQMC energies for the chain and then shows a
manual construction with a CASSCF
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHole` trial and a CCSD
initial state, using :class:`ipie.addons.free_projection.qmc.fp_afqmc.FPAFQMC`,
:class:`ipie.addons.free_projection.qmc.options.QMCParamsFP`,
:class:`ipie.addons.free_projection.propagation.free_propagation.FreePropagation`,
:class:`ipie.addons.free_projection.propagation.CCSD.CCSD` and
:class:`ipie.addons.free_projection.walkers.uhf_walkers.UHFWalkersParticleHoleFP`.
Dependencies: pyscf (and Jupyter for the notebook).

.. _ex-14:

14-fqe-wavefunction: exchanging wavefunctions with FQE
------------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/14-fqe-wavefunction>`__

This example does not run AFQMC; it shows how to move a CI wavefunction between
OpenFermion-FQE and ipie. Starting from a CASSCF(6,6) calculation on
:math:`\mathrm{N}_2` (STO-3G), ``build_ipie_wavefunction_from_pyscf`` creates a
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHole` trial in the full
orbital space (ipie inserts "melting core" orbitals for the doubly occupied
inactive space; ``strip_melting_cores`` removes them again). Helper functions
convert between the ipie ``(coeffs, occa, occb)`` tuple and an
``fqe.Wavefunction`` sector by sector, and the script asserts that the CASSCF
energy, the ipie variational energy (``trial.calculate_energy``) and the FQE
expectation value agree to :math:`10^{-8}` Ha, including after a round trip.
Thresholds on the CI coefficients and on the Cholesky cutoff can be varied to
see their effect. Dependencies: pyscf and ``fqe``; exits early if either is
missing.

.. _ex-15:

15-share_mem_cpu: integrals in shared memory
--------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/15-share_mem_cpu>`__

Intended to illustrate the node-local shared-memory path used by the JSON
driver: :func:`ipie.utils.mpi.get_shared_comm` splits ``MPI.COMM_WORLD`` into
communicators of ranks that share memory, and
:func:`ipie.hamiltonians.utils.get_hamiltonian` (with ``pack_chol=True``) reads
``hamiltonian.h5`` once per node into an MPI shared-memory window
(:func:`ipie.utils.mpi.get_shared_array`) so that the Cholesky vectors are not
replicated per rank. The rest of the script builds a
:class:`~ipie.trial_wavefunction.single_det.SingleDet` trial,
:class:`~ipie.walkers.uhf_walkers.UHFWalkers` and calls
:meth:`~ipie.qmc.afqmc.AFQMC.build` with ``1224 // comm.size`` walkers.

As committed, the script is incomplete: it uses ``scf``, ``comm``,
``get_shared_comm``, ``dir`` and ``num_walkers`` before importing or defining
them, so it only "passes" in CI because it exits when ``mpi4py`` is absent.
Treat it as a sketch and see the shared-memory section of :doc:`advanced` for a
working description. Dependencies: pyscf, mpi4py.

.. _ex-16:

16-ft_afqmc: finite-temperature AFQMC
-------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/16-ft_afqmc>`__

Uses the :mod:`ipie.addons.thermal` add-on on a two-electron uniform electron
gas. The local ``ueg.py`` module (a copy of
:class:`ipie.addons.thermal.utils.ueg.UEG`) builds the plane-wave basis for
``rs=3``, ``ecut=0.5`` and writes ``ueg_integrals.h5`` with
:func:`ipie.utils.io.write_qmcpack_sparse`. The driver is created with
:func:`ipie.addons.thermal.qmc.calc.build_thermal_afqmc_driver` from an options
dictionary specifying a ``"one_body"`` trial density matrix
(:class:`ipie.addons.thermal.trial.one_body.OneBody`), chemical potential
``mu=0.133579``, inverse temperature ``beta=10``, ``timestep=0.5``,
``stack_size=10`` and 20 blocks. After ``afqmc.run()`` both the energy and the
average particle number (``"nav"``, from
:class:`ipie.addons.thermal.estimators.particle_number.ThermalNumberEstimator`)
are extracted and reblocked. No PySCF is needed; run with
``python run_afqmc.py`` or under ``mpirun``.

.. _ex-17:

17-ghf_afqmc: GHF trials and walkers
------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/17-ghf_afqmc>`__

Stretched :math:`\mathrm{H}_2` (4.2 bohr, STO-6G) in the RHF orbital basis.
Integrals are generated with :func:`ipie.utils.from_pyscf.generate_integrals`
and wrapped in :class:`ipie.hamiltonians.generic.GenericRealChol`. A
symmetry-broken UHF determinant is used both as a
:class:`~ipie.trial_wavefunction.single_det.SingleDet` trial and, after
embedding into a :math:`2M \times N` spinor matrix and applying an SU(2)
spin-axis rotation, as a :class:`ipie.trial_wavefunction.single_det_ghf.SingleDetGHF`
trial with :class:`ipie.walkers.ghf_walkers.GHFWalkers`. The script asserts
that the GHF trial energy is invariant under the rotation and runs a short
AFQMC calculation with :meth:`~ipie.qmc.afqmc.AFQMC.build`. The second half
shows the alternative constructors ``SingleDetGHF(trial_uhf)`` and
``GHFWalkers(walkers_uhf)`` that promote existing UHF objects. Dependencies:
pyscf.

.. _ex-18:

18-ad_afqmc: automatic differentiation for properties
-----------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/18-ad_afqmc>`__

Computes the dipole moment of CO (cc-pVDZ, RHF, two frozen core orbitals) with
the PyTorch-based :mod:`ipie.addons.adafqmc` add-on. The dipole operator is
transformed to the MO basis and its frozen-core contribution folded into a
constant; :func:`ipie.addons.adafqmc.utils.miscellaneous.generate_hamiltonian_from_pyscf`
builds a Hamiltonian object that carries the observable coupled with a
strength ``coupling``. The trial wavefunction and its derivative with respect
to the coupling are obtained by forward-mode AD
(``torch.func.jvp``) through
:func:`ipie.addons.adafqmc.utils.miscellaneous.get_hf_wgradient`, and the
driver is built and run as

.. code-block:: python

   adafqmc = ADAFQMC.build(comm, trial_tangent, **options)
   energy, obs = adafqmc.run(hamobs, trial_detached, tg)

with :class:`ipie.addons.adafqmc.qmc.adafqmc.ADAFQMC` and a
:class:`ipie.qmc.comm.FakeComm` communicator. The options in the script (10
walkers, 10 AD blocks of 10 steps) are deliberately tiny. Dependencies: pyscf
and ``torch``; install the latter with ``pip install -e .[torch]``.

.. _ex-19:

19-hubbard: a lattice model through the generic Hamiltonian
-----------------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/19-hubbard>`__

The modern code has no dedicated Hubbard class (one exists only in
``ipie.legacy``), but any model whose two-body term can be written as a sum of
squares fits :class:`ipie.hamiltonians.generic.GenericRealChol`. The script
builds a two-site periodic Hubbard model with :math:`t = 1`, :math:`U = 4`:
the hopping matrix and the on-site ``eri`` tensor are handed to a PySCF
``UHF`` object via ``get_hcore``, ``get_ovlp`` and ``_eri`` overrides (with a
spin-polarised initial guess), and the "Cholesky" vectors are obtained from an
SVD of the :math:`n^2 \times n^2` interaction matrix. A
:class:`~ipie.trial_wavefunction.single_det.SingleDet` trial from the UHF
orbitals and :meth:`~ipie.qmc.afqmc.AFQMC.build` complete the calculation.
Dependencies: pyscf (for the mean-field only).

.. _ex-20:

20-save_and_restart: walker checkpoints
---------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/20-save_and_restart>`__

Shows the checkpoint hooks on the walker classes. ``EnergyEstimatorDumpWalkers``
subclasses :class:`ipie.estimators.energy.EnergyEstimator` and, at chosen block
indices, calls :meth:`ipie.walkers.base_walkers.BaseWalkers.write_walkers_batch`,
which appends three datasets per checkpoint, ``walker_timeslice_<n>`` (the
alpha and beta orbitals stacked along the last axis), ``walker_weight_<n>`` and
``walker_hybrid_energy_<n>``, to ``walkers_<rank>.h5``. After the run the
script restores the population with
:meth:`ipie.walkers.base_walkers.BaseWalkers.read_walkers_batch` (which reads
the last time slice and recomputes the overlaps against the trial) and asserts
that the energy recomputed from the restored walkers equals the last block's
``ETotal``. The example runs serially with :class:`ipie.qmc.comm.FakeComm`;
under MPI each rank writes its own file. Dependencies: pyscf.

.. _ex-21:

21-kpt_chol: periodic k-point Cholesky Hamiltonian
--------------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/21-kpt_chol>`__

Runs AFQMC for a periodic carbon system on a :math:`3 \times 1 \times 1`
k-mesh with a double-zeta basis, using the bundled ``afqmc_C_311_dz_chol.h5``.
The comments in the script document the file layout: ``hcore``
``(nk, M, M)``, ``chol`` ``(nchol, nk, M, nunique_k, M)``, ``e0`` and
``kpoints`` ``(nk, 3)`` in fractional coordinates, where the "unique" k-points
are ordered as the self-inverse set followed by the :math:`Q_+` set as returned
by :func:`ipie.utils.kpt_conv.find_self_inverse_set` and
:func:`ipie.utils.kpt_conv.find_Qplus`.

``run_afqmc.py`` (CPU, requires mpi4py) reads the Hamiltonian with
:func:`ipie.hamiltonians.utils.get_kpt_hamiltonian`, which returns a
:class:`ipie.hamiltonians.kpt_hamiltonian.KptComplexCholSymm`, builds a
:class:`ipie.trial_wavefunction.single_det_kpt.KptSingleDet` trial occupying
the lowest four orbitals of each spin at every k-point, and creates
:class:`~ipie.walkers.uhf_walkers.UHFWalkers` of dimension ``nk * M`` by
``nk * N``. ``run_afqmc_gpu.py`` additionally requires cupy, switches the
backend on with ``config.update_option("use_gpu", True)``, and constructs a
:class:`ipie.hamiltonians.kpt_chunked.KptComplexCholChunked` from the raw
``chol`` array (``NMEMBERS = 1``, so the whole tensor sits on each GPU) with
``walkers.rhf = True``. Run with ``mpirun -np N python run_afqmc.py``. The
GPU script imports ``ipie.utils.backend`` before it calls
``config.update_option``, so the in-script switch has no effect; export the
option in the environment instead:
``IPIE_USE_GPU=1 mpirun -np N python run_afqmc_gpu.py``. Dependencies: mpi4py;
cupy for the GPU script.

.. _ex-22:

22-kpt_isdf: periodic k-point ISDF Hamiltonian
----------------------------------------------

`Source <https://github.com/JoonhoLee-Group/ipie/tree/develop/examples/22-kpt_isdf>`__

The same carbon system with the electron-repulsion integrals compressed by
interpolative separable density fitting. ``afqmc_C_311_dz_isdf.h5`` stores
``hcore``, ``MPQ`` ``(nunique_k, nisdf, nisdf)``, the Bloch orbitals on the
ISDF grid ``cgto`` ``(nk, nisdf, M)``, ``e0`` and ``kpoints``. The script forms
the Cholesky factor of ``MPQ`` with ``numpy.linalg.cholesky`` and builds
:class:`ipie.hamiltonians.kpt_isdf_hamiltonian.KptISDF`; trial, walkers and
driver are set up exactly as in ``21-kpt_chol``. The script requires cupy and
mpi4py, enables the GPU backend and assumes four GPUs per node
(``GPU_NUMBER_PER_NODE = 4``). As in ``21-kpt_chol/run_afqmc_gpu.py``, the
backend is imported before ``use_gpu`` is set inside the script, so run it with
``IPIE_USE_GPU=1 mpirun -np N python run_afqmc.py``.
