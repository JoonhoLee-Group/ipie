Advanced features
=================

This page covers the capabilities of ipie that go beyond a single-node,
CPU-only, single-determinant phaseless calculation. Every section is grounded in
the current code: it names the classes involved (see the :doc:`api/index`), the
example that exercises them (see :doc:`examples`) and, where relevant, the
limitations of the implementation. Features that only exist in the
``ipie.legacy`` package are marked as such; that package is not covered by the
API reference and is not maintained at the same level as the core code.

MPI parallelism and distributed Hamiltonians
--------------------------------------------

Walker-level parallelism
~~~~~~~~~~~~~~~~~~~~~~~~

ipie parallelises over walkers with MPI. If ``mpi4py`` is importable,
:mod:`ipie.config` exposes it as ``ipie.config.MPI``; otherwise a serial
stand-in, :class:`ipie.qmc.comm.FakeComm`, is substituted so that the same
scripts run unmodified in serial. Each rank owns ``num_walkers`` walkers, the
total population is ``num_walkers * comm.size``
(:class:`ipie.qmc.options.QMCParams`), and population control
(:class:`ipie.walkers.pop_controller.PopController`) exchanges walkers between
ranks. Estimators are reduced across ``comm`` at the end of each block, and a
single ``estimates.0.h5`` is written by rank 0.

All communicator bookkeeping is held by :class:`ipie.utils.mpi.MPIHandler`.
Its constructor takes ``nmembers`` (plus an optional ``verbose`` flag) and splits
``MPI.COMM_WORLD`` into ``ngroups = size // nmembers`` groups; ``nmembers`` must
divide the world size. Within a group the ranks share one copy of the
Hamiltonian (see below); across groups the usual walker parallelism applies. The
default ``nmembers=1`` makes every rank its own group, i.e. plain walker
parallelism. :meth:`~ipie.utils.mpi.MPIHandler.scatter_group` and
:meth:`~ipie.utils.mpi.MPIHandler.allreduce_group` operate on the group
communicator ``handler.scomm``.

.. code-block:: python

   from ipie.utils.mpi import MPIHandler

   handler = MPIHandler(nmembers=4)   # 4 ranks share one Hamiltonian
   handler.comm, handler.scomm        # world and group communicators
   handler.rank, handler.srank        # world rank and rank within the group

With the JSON input file the same thing is controlled by the ``nmembers`` key of
the ``qmc`` section, which :func:`ipie.qmc.calc.get_driver` forwards to
``MPIHandler(nmembers=qmc_opts.get("nmembers", 1))``.

Shared-memory integrals on a node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the driver is built from an HDF5 file (:func:`ipie.qmc.calc.get_driver`,
:func:`ipie.qmc.calc.build_afqmc_driver`, :meth:`ipie.qmc.afqmc.AFQMC.build_from_hdf5`)
the integrals are read by :func:`ipie.hamiltonians.utils.get_hamiltonian`,
which shares them over whatever communicator it is given: if MPI-3 shared
memory is available on that communicator
(:func:`ipie.utils.mpi.have_shared_mem`), its rank 0 reads ``hcore``, the
Cholesky tensor and the packed (upper-triangular) Cholesky tensor into windows
allocated with :func:`ipie.utils.mpi.get_shared_array`, and the other ranks of
the communicator see the same memory. The driver factories pass
``mpi_handler.scomm``, the ``nmembers``-sized *group* communicator, not the
node-wide one. Integrals are therefore shared only among the ranks of a group
(which must sit on one node for this to work), and with the default
``nmembers=1`` every rank still holds its own copy of the resulting
:class:`ipie.hamiltonians.generic.GenericRealChol`. To share one copy per node
from a Python script, call
``get_hamiltonian(filename, get_shared_comm(comm))`` yourself;
:func:`ipie.utils.mpi.get_shared_comm` wraps
``comm.Split_type(MPI.COMM_TYPE_SHARED)`` (``MPIHandler.shared_comm`` holds the
same communicator but is not used by the factories). ``examples/15-share_mem_cpu``
and ``examples/21-kpt_chol`` do exactly this, but note the caveats listed in
:doc:`examples`.

The packed tensor is stored *in addition to* the full Cholesky tensor, so a
real Hamiltonian costs about 1.5 times the memory of the full tensor alone (the
verbose output prints both contributions and their sum). ``pack_chol`` (JSON
key ``symmetry``, aliases ``pack_chol``/``pack_cholesky``, default ``True``)
only skips filling the packed array in the shared-memory path of
``get_hamiltonian``; when it is ``False`` the Hamiltonian is built without the
``shmem`` flag and :class:`~ipie.hamiltonians.generic.GenericRealChol` packs
the tensor itself unconditionally, so the option does not reduce memory.

Chunked Hamiltonians across ranks or GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For systems whose Cholesky vectors do not fit in the memory of one device, the
vectors can be split along the auxiliary index over the ``nmembers`` ranks of a
group. This is the mechanism behind the "distributed Hamiltonian" support.

* :func:`ipie.utils.chunk_large_chol.split_cholesky` reads ``LXmn`` from the
  molecular ``hamiltonian.h5``; :func:`~ipie.utils.chunk_large_chol.split_cholesky_kpt`
  reads the ``chol`` dataset of a k-point file and, note, **removes that
  dataset from the source file** after splitting (keep a backup). Both write
  ``chol_0.h5``, ``chol_1.h5``, ... in the current directory using the same
  partition as :func:`ipie.utils.mpi.make_splits_displacements`.
* :class:`ipie.hamiltonians.generic_chunked.GenericRealCholChunked` is the
  Hamiltonian class. It accepts either the full ``chol`` array (and chunks it
  itself with ``scatter_group``) or a pre-split ``chol_chunk`` together with
  its packed form ``chol_packed_chunk``; the one-body correction
  ``h1e_mod`` is assembled with an ``allreduce`` over the group. It sets
  ``chunked = True`` so that the propagator and estimators pick the chunked
  kernels (:class:`ipie.propagation.phaseless_generic.PhaselessGenericChunked`,
  :mod:`ipie.estimators.local_energy_sd_chunked`). Only real, 8-fold symmetric
  integrals are supported; the ISDF and k-point analogues are
  :class:`ipie.hamiltonians.chunked_isdf.GenericRealISDFChunked` and
  :class:`ipie.hamiltonians.kpt_chunked.KptComplexCholChunked`. The mapping
  from Hamiltonian class to propagator lives in the ``Propagator`` dictionary
  of :mod:`ipie.propagation.propagator`.
* The trial's half-rotated Cholesky tensors are chunked too:
  :class:`~ipie.trial_wavefunction.single_det.SingleDet` takes the handler as
  a constructor argument and its ``half_rotate`` overload for
  ``GenericRealCholChunked`` produces per-rank half-rotated chunks
  (``_rchola_chunk``/``_rcholb_chunk``). The helper
  :meth:`ipie.qmc.afqmc.AFQMCBase.distribute_hamiltonian`, which would call
  ``trial.chunk(handler)`` when ``nmembers > 1``, is currently not invoked by
  :meth:`~ipie.qmc.afqmc.AFQMC.run` (only by the free-projection driver).

``examples/06-gpu/run_afqmc_chunked.py`` is the reference implementation:

.. code-block:: python

   handler = MPIHandler(nmembers=4)
   with h5py.File(f"chol_{handler.srank}.h5") as fa:
       chol_chunk = fa["chol"][()]
   chol_packed_chunk = ...   # pack_cholesky on the chunk
   ham = GenericRealCholChunked(
       numpy.array([hcore, hcore]), None, chol_chunk, chol_packed_chunk, e0, handler
   )
   trial = SingleDet(numpy.hstack([psiT, psiT]), nelec, num_basis, handler)
   walkers = UHFWalkers(phi0, nup, ndown, nbasis, num_walkers, mpi_handler=handler)
   afqmc = AFQMC.build(nelec, ham, trial, walkers, num_walkers, ..., mpi_handler=handler)

The chunked MPI paths are tested by ``dev/run_tests.py --mpi`` (six-rank runs
of ``ipie/estimators/tests/test_generic_chunked.py`` and
``ipie/propagation/tests/test_generic_chunked.py``). Chunking is not yet
available through the JSON input file: :func:`~ipie.qmc.calc.get_driver`
creates the handler but always builds an unchunked Hamiltonian, so use the
Python API as above.

GPU execution
-------------

Selecting the backend
~~~~~~~~~~~~~~~~~~~~~

ipie uses a single array namespace, ``arraylib`` in :mod:`ipie.utils.backend`,
that is bound to either ``numpy`` or ``cupy`` when the module is first
imported. The choice is read from the ``use_gpu`` option of the global
``config`` object in :mod:`ipie.config`, which is initialised from the
environment variable ``IPIE_USE_GPU`` (``0``/``1``). The option can also be set
programmatically, but only *before* the first import of any module that pulls in
:mod:`ipie.utils.backend`:

.. code-block:: python

   from ipie.config import config
   config.update_option("use_gpu", True)
   # only now import the rest of ipie
   from ipie.qmc.afqmc import AFQMC

Code paths import ``from ipie.utils.backend import arraylib as xp`` and write
``xp.`` code that runs on either device; helpers such as ``to_host``,
``synchronize``, ``qr`` and ``cast_to_device`` in the same module hide the
remaining differences. :meth:`ipie.qmc.afqmc.AFQMCBase.copy_to_gpu`, called at
the start of :meth:`~ipie.qmc.afqmc.AFQMC.run`, selects device
``comm.rank % ngpus`` and moves the propagator, Hamiltonian, trial and walkers
to it via their ``cast_to_cupy`` methods
(:class:`~ipie.trial_wavefunction.particle_hole.ParticleHole` trials are
skipped by ``copy_to_gpu`` and stay on the host; the GPU Wick's kernels copy
the slices they need with ``xp.asarray`` at evaluation time). The intended layout is therefore **one MPI
rank per GPU**; a warning is printed if there are more GPUs than ranks. Examples
that need more control bind the device explicitly with
``xp.cuda.Device(rank % gpus_per_node).use()`` before building the Hamiltonian
(``examples/06-gpu``, ``21-kpt_chol/run_afqmc_gpu.py``, ``22-kpt_isdf``).
Combined with ``nmembers`` this gives Hamiltonians distributed over several
GPUs; MPI communication of device arrays requires a CUDA-aware MPI build.

Kernels and mixed precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most GPU work is ordinary CuPy linear algebra, but the hot spots have custom
kernels under ``ipie/*/kernels/gpu``: ``ipie/estimators/kernels/gpu/exchange.py``
(Numba CUDA reductions for the exchange energy),
``exchange_kpt.py`` and ``ipie/propagation/kernels/gpu/vhs.py`` (CuPy
``RawKernel`` code for the k-point exchange energy and the k-point
Hubbard-Stratonovich potential), and ``wicks_gpu.py`` (the multi-determinant
Wick's-theorem kernels). The corresponding CPU code lives in
``ipie/estimators/kernels/cpu/wicks.py`` (Numba). Memory limits for the GPU
paths are configurable through ``config`` (``max_memory_for_wicks`` and
``max_memory_sd_energy_gpu``, both 2 GB by default).

The ``mixed_precision`` option (environment variable
``IPIE_USE_MIXED_PRECISION``) exists in :mod:`ipie.config`, but in the current
core code it is only consulted through a ``trial.mixed_precision`` attribute
in :mod:`ipie.estimators.generic` that no trial class sets; the mixed-precision
propagation is implemented only in ``ipie.legacy``. Treat it as unsupported in
the modern code path.

cuTensorNet contractions for k-point Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The periodic (k-point) propagators and estimators optionally use NVIDIA
cuQuantum's cuTensorNet for large tensor contractions.
:mod:`ipie.utils.cuquantum_backend` tries to import ``cuquantum`` (version 26.1
or later) and exports two flavours of the contraction API: the ``*_optional``
objects fall back to ``xp.einsum`` when cuQuantum is missing, while the
``*_required`` objects raise an informative ``ImportError``.
:mod:`ipie.propagation.phaseless_kpt` switches to ``construct_VHS_cuquantum``
when the number of k-point pairs exceeds the CUDA grid limit (``nk**2 >
65536``), and :mod:`ipie.estimators.local_energy_kpt_sd_isdf` and
:mod:`ipie.utils.contract_gf_cgto` use it for the ISDF energy. Install
``cuquantum-python`` to enable it; nothing else changes in user scripts.

Multi-determinant trial wavefunctions
-------------------------------------

Particle-hole (CI-type) expansions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Orthogonal multi-Slater trials :math:`|\Psi_T\rangle = \sum_I c_I |D_I\rangle`,
where every determinant is an excitation out of a common reference, are
represented by :class:`ipie.trial_wavefunction.particle_hole.ParticleHole`. The
wavefunction is passed as a tuple ``(coeffs, occa, occb)`` of CI coefficients
and alpha/beta occupied-orbital index arrays, which is exactly what
:func:`ipie.utils.io.write_wavefunction` stores when
:func:`~ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk` is called with
``mcscf=True`` (``examples/02-multi_determinant``), what the TREXIO and Dice
readers return, and what :func:`~ipie.trial_wavefunction.utils.get_trial_wavefunction`
reads for ``particle_hole`` (and QMCPACK ``phmsd``) files. Determinants defined
in an active space are expanded to the full orbital space by inserting
"melting core" orbitals (``use_active_space=True``).

Overlaps, Green's functions and local energies are evaluated with the
generalised Wick's theorem algorithms of Mahajan, Lee and Sharma
(:doc:`bibliography`), implemented in :mod:`ipie.estimators.local_energy_wicks`
(``local_energy_multi_det_trial_wicks_batch_opt_chunked`` and its GPU
counterpart), :mod:`ipie.estimators.greens_function_multi_det` and
:mod:`ipie.propagation.overlap`. Determinants are grouped by excitation level
and processed in batches. Three parameters control cost versus accuracy:

``num_dets_for_trial`` (JSON ``ndets``)
   Number of determinants kept in the trial (``-1`` for all). The expansion
   should be sorted by decreasing :math:`|c_I|` first.
``num_dets_for_props`` (JSON ``ndets_props``)
   Number of determinants used when evaluating trial-only properties such as
   the one-body density matrix for the mean-field shift and the trial energy.
``num_det_chunks`` (JSON ``ndet_chunks``)
   Splits the excitation tables into chunks so that the intermediate buffers
   fit in memory. :func:`~ipie.trial_wavefunction.utils.get_trial_wavefunction`
   instantiates :class:`~ipie.trial_wavefunction.particle_hole.ParticleHoleNonChunked`
   when this is 1 and :class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`
   otherwise; :meth:`~ipie.qmc.afqmc.AFQMC.build_from_hdf5` exposes two of
   these knobs as ``num_dets_chunk`` and ``num_dets_for_trial_props``; the
   number of determinants in the trial cannot be set there (all determinants
   in the file are used).

.. code-block:: python

   from ipie.trial_wavefunction.particle_hole import ParticleHole

   trial = ParticleHole((coeffs, occa, occb), nelec, nbasis,
                        num_dets_for_trial=1000, num_dets_for_props=100,
                        num_det_chunks=4)
   trial.build()
   trial.half_rotate(ham)

The walker class for these trials is
:class:`ipie.walkers.uhf_walkers.UHFWalkersParticleHole`, selected
automatically by the ``UHFWalkersTrial`` dispatcher in
:mod:`ipie.walkers.walkers_dispatch`. The slower reference implementations
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHoleSlow` and
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHoleNaive` are kept for
testing.

The Wick's helper C library
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computing the trial one-body density matrix of a long CI expansion in pure
Python is slow. ``ipie/lib/wicks`` contains a small C library
(``determinant_utils.c``, ``density_matrix.c``) wrapped with ``ctypes`` in
``ipie/lib/wicks/wicks_helper.py``. It encodes determinants as bit strings
(``encode_dets``), fixes phases (``convert_phase``) and computes the 1-RDM
(``compute_opdm``). It is optional: :class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`
tries to import it and falls back to ``compute_1rdm`` in Python if the shared
library is missing. To build it:

.. code-block:: bash

   cd ipie/lib/wicks
   cmake . && make          # produces libwicks_helper.so next to the sources

The library is not built by ``pip install`` and is not part of the API
reference.

Non-orthogonal CI (NOCI) trials
-------------------------------

:class:`ipie.trial_wavefunction.noci.NOCI` handles linear combinations of
determinants built from *different* orbital sets,
:math:`|\Psi_T\rangle = \sum_I c_I |\phi_I\rangle`. It takes a tuple
``(coeffs, psi)`` where ``psi`` has shape ``(ndets, nbasis, nalpha + nbeta)``,
computes the trial 1-RDM from the pairwise overlaps, and evaluates overlaps and
force biases by looping over determinants
(:func:`ipie.propagation.overlap.calc_overlap_multi_det`,
:func:`ipie.propagation.force_bias.construct_force_bias_batch_multi_det_trial`,
:mod:`ipie.estimators.local_energy_noci`). The matching walkers are
:class:`ipie.walkers.uhf_walkers.UHFWalkersNOCI`. NOCI wavefunctions are read
from HDF5 with :func:`ipie.utils.io.read_noci_wavefunction` (written by
:func:`ipie.utils.io.write_noci_wavefunction`) or from QMCPACK ``nomsd``
files. There is no shipped example; the cost scales linearly with the number
of determinants, so NOCI is intended for expansions of tens to hundreds of
determinants.

GHF trials and walkers
----------------------

Generalised Hartree-Fock (spinor) trial wavefunctions, in which alpha and beta
components mix, are supported through
:class:`ipie.trial_wavefunction.single_det_ghf.SingleDetGHF` and
:class:`ipie.walkers.ghf_walkers.GHFWalkers`. The trial is a
:math:`2M \times N` matrix whose upper and lower blocks are the alpha and beta
components. Both classes have two constructors (multiple dispatch via
``plum``): from a raw coefficient matrix, or from an existing UHF object

.. code-block:: python

   trial = SingleDetGHF(psi_ghf, nelec, num_basis)      # from a 2M x N matrix
   trial = SingleDetGHF(trial_uhf)                       # promote a SingleDet
   walkers = GHFWalkers(psi_ghf, nup, ndown, num_basis, nwalkers, MPIHandler())
   walkers = GHFWalkers(walkers_uhf)                     # promote UHFWalkers

Overlaps, Green's functions and energies use dedicated GHF routines
(:func:`ipie.propagation.overlap.calc_overlap_single_det_ghf`,
:func:`ipie.estimators.greens_function_single_det.greens_function_single_det_ghf`,
:func:`ipie.estimators.local_energy_sd.local_energy_single_det_ghf_batch`,
:func:`ipie.estimators.generic.cholesky_jk_ghf`). Only single-determinant GHF
trials with real or complex Cholesky Hamiltonians are implemented.
``examples/17-ghf_afqmc`` verifies that the energy of a spin-rotated UHF
determinant is invariant, then runs AFQMC with GHF walkers.

Frozen core
-----------

Core orbitals can be removed from the AFQMC problem at the integral level.
:func:`ipie.utils.from_pyscf.freeze_core` takes the MO-basis one-body
Hamiltonian, the Cholesky vectors, the core energy and the basis-change
matrix, computes the energy and the effective one-body potential of the
frozen determinant, adds that potential to ``h1e``, and slices both ``h1e`` and
the Cholesky vectors to the active orbitals. The number of electrons is reduced
accordingly. This is wired into :func:`~ipie.utils.from_pyscf.generate_hamiltonian`
and :func:`~ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk` through
``num_frozen_core``, and into the command-line converter as

.. code-block:: bash

   python tools/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json --frozen-core 5

which also writes the reduced ``nup``/``ndown`` into ``input.json``
(``examples/05-frozen_core``). Frozen core presumes the MO basis
(``ortho_ao=False``) and that the lowest ``num_frozen_core`` orbitals of each
spin are the ones to freeze. The AD-AFQMC add-on has its own ``num_frozen``
argument in :func:`ipie.addons.adafqmc.utils.miscellaneous.generate_hamiltonian_from_pyscf`.

Add-ons
-------

The ``ipie/addons`` directory holds algorithms that build on the core library
but are maintained somewhat independently and have their own drivers. Each
mirrors the layout of the main package (``qmc``, ``propagation``, ``walkers``,
``estimators``, ...).

Free-projection AFQMC
~~~~~~~~~~~~~~~~~~~~~

:mod:`ipie.addons.free_projection` implements free projection, i.e. AFQMC
without the phaseless constraint. It evaluates

.. math::

   E(\tau) = \frac{\langle \Psi_T | H e^{-\tau H} | \Phi_0 \rangle}
                  {\langle \Psi_T | e^{-\tau H} | \Phi_0 \rangle}

by sampling the propagator without importance sampling, so that the estimate is
exact but its variance grows exponentially with :math:`\tau`. The driver
:class:`ipie.addons.free_projection.qmc.fp_afqmc.FPAFQMC` subclasses
:class:`~ipie.qmc.afqmc.AFQMC`; it repeats the projection
``num_iterations_fp`` times (:class:`~ipie.addons.free_projection.qmc.options.QMCParamsFP`),
each time restarting from the initial state, and writes one estimator file per
block so that :math:`E(\tau)` is available at every block boundary. The
propagator is :class:`~ipie.addons.free_projection.propagation.free_propagation.FreePropagation`
(``ene_0`` supplies an energy shift), the walkers are
:class:`~ipie.addons.free_projection.walkers.uhf_walkers.UHFWalkersFP` /
:class:`~ipie.addons.free_projection.walkers.uhf_walkers.UHFWalkersParticleHoleFP`,
and the initial state can be a CCSD wavefunction sampled through
:class:`~ipie.addons.free_projection.propagation.CCSD.CCSD`. Population control
is disabled. Ratio estimates should be analysed with
:func:`ipie.addons.free_projection.analysis.jackknife.jackknife_ratios`; a
phase estimator (:class:`~ipie.addons.free_projection.estimators.phase.PhaseEstimatorFP`)
monitors the average walker phase. The convenience builder is
:func:`ipie.addons.free_projection.qmc.calc.build_fpafqmc_driver`
(``examples/13-free_projection``).

Finite-temperature AFQMC
~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`ipie.addons.thermal` implements grand-canonical, finite-temperature
AFQMC with :class:`ipie.addons.thermal.qmc.thermal_afqmc.ThermalAFQMC`. The
imaginary-time interval :math:`\beta` is divided into ``beta / timestep``
slices, walkers are products of one-body propagators stored in a
:class:`~ipie.addons.thermal.walkers.stack.PropagatorStack` (grouped
``stack_size`` slices at a time and stabilised by QR), and the density matrix
is obtained with :func:`ipie.addons.thermal.estimators.greens_function.greens_function_qr_strat`.
The trial is a one-body density matrix,
:class:`~ipie.addons.thermal.trial.one_body.OneBody` (free-fermion) or
:class:`~ipie.addons.thermal.trial.mean_field.MeanField` (thermal Hartree-Fock),
whose chemical potential is fitted to the target particle number with
:func:`ipie.addons.thermal.trial.chem_pot.find_chemical_potential`. The
``qmc`` options carry ``mu`` and ``beta`` in addition to the usual keys
(:class:`~ipie.addons.thermal.qmc.options.ThermalQMCParams`); only one step
per block is allowed, and ``lowrank`` walkers are declared but rejected by the
driver builder. Estimators include the energy and the average particle number
(:class:`~ipie.addons.thermal.estimators.particle_number.ThermalNumberEstimator`).
Build a driver with :func:`ipie.addons.thermal.qmc.calc.build_thermal_afqmc_driver`
from an options dictionary as in ``examples/16-ft_afqmc``, which ships its own
uniform-electron-gas generator (``examples/16-ft_afqmc/ueg.py``; an equivalent
class lives in :class:`ipie.addons.thermal.utils.ueg.UEG`).

Automatic-differentiation AFQMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`ipie.addons.adafqmc` is a separate, PyTorch-based implementation of
phaseless AFQMC for RHF trials whose purpose is to compute observables as
derivatives of the energy with respect to a coupling :math:`\lambda` in
:math:`H + \lambda O`. The driver :class:`ipie.addons.adafqmc.qmc.adafqmc.ADAFQMC`
equilibrates the walkers, then runs ``num_ad_blocks`` blocks of
``ad_block_size`` steps under ``torch`` autograd and returns the energy and the
observable, with optional gradient checkpointing (``grad_checkpointing``,
``chkpt_size``). The Hamiltonian-plus-observable object is created from a PySCF
mean-field with :func:`ipie.addons.adafqmc.utils.miscellaneous.generate_hamiltonian_from_pyscf`
(supporting ``num_frozen`` core orbitals), and the trial together with its
response to :math:`\lambda` is obtained by forward-mode AD of a Hartree-Fock
solve (:func:`~ipie.addons.adafqmc.utils.miscellaneous.get_hf_wgradient`,
:func:`~ipie.addons.adafqmc.utils.miscellaneous.trial_tangent`). The add-on
uses its own walker, propagator and estimator classes rather than the core
ones. Install the extra with ``pip install -e .[torch]``; see
``examples/18-ad_afqmc`` for a dipole-moment calculation on CO.

Isometric THC
~~~~~~~~~~~~~

:mod:`ipie.addons.ithc` implements AFQMC with an isometric tensor
hypercontraction (iTHC) factorisation of the two-electron integrals. The
Hamiltonian :class:`ipie.addons.ithc.hamiltonians.generic_ithc.GenericITHC` takes
the one-body matrix, an isometry ``U`` of shape ``(nbasis, nbasis_extended)``
and the symmetric THC core matrix ``W`` in the extended basis, and decomposes
the Hubbard-Stratonovich fields over pairs of extended-basis indices. It comes
with its own :class:`~ipie.addons.ithc.trial_wavefunction.single_det.SingleDet`
trial (with an iTHC half-rotation), propagator
(:class:`~ipie.addons.ithc.propagation.phaseless_ithc.PhaselessITHC`), energy
estimator and a driver :class:`ipie.addons.ithc.qmc.afqmc.AFQMC` that mirrors
the core one. The example scripts live inside the add-on rather than under
``examples/``: ``ipie/addons/ithc/example_ithc/run_afqmc.py`` runs an
H\ :sub:`10` chain from the bundled ``hamiltonian.h5`` (fields ``hkin``,
``hnuc``, ``eri``, ``u``, ``w``, ``enuc``, ``nelectron``) and compares with
FCI, and ``gpu_benchmark.py`` times random iTHC instances. This add-on is
recent and should be considered experimental; note that the package directory
contains ``_init_.py`` rather than ``__init__.py`` (it is imported as a
namespace package) and that ``gpu_benchmark.py`` imports from
``ipie.hamiltonians.generic_ithc``, a module path that does not exist.

Periodic systems: k-point Cholesky and ISDF
-------------------------------------------

Solids are treated with explicit crystal-momentum symmetry. Orbitals carry a
k-index, the Cholesky vectors carry a momentum-transfer index, and inversion
symmetry (:math:`\mathbf{q} \leftrightarrow -\mathbf{q}`) is used to halve the
storage. The helpers in :mod:`ipie.utils.kpt_conv` build the required index
maps: :func:`~ipie.utils.kpt_conv.find_self_inverse_set` returns the k-points
that are their own inverse, :func:`~ipie.utils.kpt_conv.find_Qplus` one member
of each remaining :math:`\pm\mathbf{q}` pair, and the Hamiltonian classes
construct the ``k + q`` and ``k - q`` lookup tables from them.

Hamiltonians
~~~~~~~~~~~~

* :class:`ipie.hamiltonians.kpt_hamiltonian.KptComplexCholSymm` stores
  Cholesky vectors of shape ``(nchol, nk, M, nunique_k, M)`` using the
  symmetry above; :class:`~ipie.hamiltonians.kpt_hamiltonian.KptComplexChol`
  is the unsymmetrised variant with shape ``(nchol, nk, M, nk, M)``.
  :func:`ipie.hamiltonians.utils.get_kpt_hamiltonian` reads the HDF5 layout
  (``hcore``, ``chol``, ``e0``, ``kpoints``; see
  :func:`ipie.utils.io.read_kpt_hamiltonian`) into shared memory and returns
  the symmetric class.
* :class:`ipie.hamiltonians.kpt_chunked.KptComplexCholChunked` distributes the
  Cholesky vectors over the ``nmembers`` of an
  :class:`~ipie.utils.mpi.MPIHandler`, exactly as for molecules; chunks are
  prepared with :func:`ipie.utils.chunk_large_chol.split_cholesky_kpt`.
* :class:`ipie.hamiltonians.kpt_isdf_hamiltonian.KptISDF` represents the
  two-electron integrals by interpolative separable density fitting: it takes
  the ISDF core tensor ``MPQ`` of shape ``(nunique_k, nisdf, nisdf)``, its
  Cholesky factor, and the Bloch orbitals evaluated on the interpolation grid
  ``cgto`` of shape ``(nk, nisdf, M)``. The molecular ISDF analogues are
  :class:`ipie.hamiltonians.isdf.GenericRealISDF` and
  :class:`~ipie.hamiltonians.isdf.GenericComplexISDF`.

Propagators (:class:`~ipie.propagation.phaseless_kpt.PhaselessKptChol`,
:class:`~ipie.propagation.phaseless_kpt.PhaselessKptCholChunked`,
:class:`~ipie.propagation.phaseless_kpt.PhaselessKptISDF`) and the k-point
energy estimators (:mod:`ipie.estimators.local_energy_kpt_sd`,
:mod:`ipie.estimators.local_energy_kpt_sd_chunked`,
:mod:`ipie.estimators.local_energy_kpt_sd_isdf`) are selected through the
``Propagator`` dictionary and the walker type; the GPU kernels in
``ipie/propagation/kernels/gpu/vhs.py`` and
``ipie/estimators/kernels/gpu/exchange_kpt.py`` and the optional cuTensorNet
backend described above exist for these paths.

Trials and walkers
~~~~~~~~~~~~~~~~~~

The trial is :class:`ipie.trial_wavefunction.single_det_kpt.KptSingleDet`, a
single determinant given per k-point as an array of shape
``(nk, M, nalpha + nbeta)``; ``noccas``/``noccbs`` allow different occupations
at different k-points. Walkers are ordinary
:class:`~ipie.walkers.uhf_walkers.UHFWalkers` over the supercell basis, with
``nk * M`` orbitals and ``nk * n`` electrons per spin, and are constructed
either directly or via the ``KptSingleDet`` overload of ``UHFWalkersTrial``.

.. code-block:: python

   ham = get_kpt_hamiltonian("afqmc_C_311_dz_chol.h5", get_shared_comm(comm))
   trial = KptSingleDet(psi, ham.nk, (neleca, nelecb), ham.nbasis)
   trial.build(); trial.half_rotate(ham, scomm)
   walkers = UHFWalkers(phi, ham.nk * neleca, ham.nk * nelecb,
                        ham.nk * ham.nbasis, nwalkers, mpi_handler=handler)

``examples/21-kpt_chol`` and ``examples/22-kpt_isdf`` provide ready-made
integral files for a small carbon system; ``21-kpt_chol`` has a CPU script
(``run_afqmc.py``) and a GPU script (``run_afqmc_gpu.py``), while
``22-kpt_isdf`` ships a single GPU-only script. The
k-point code is comparatively new (it was added to the ``develop`` branch in
2026); the JSON driver does not yet know about it, and generating the input
files from a periodic PySCF calculation is left to the user's own scripts.

Interfaces to other codes
-------------------------

TREXIO
~~~~~~

:func:`ipie.utils.from_trexio.gen_ipie_from_trexio` reads a TREXIO file
(``pip install trexio``) and returns a dictionary with the MO-basis core
Hamiltonian (``hcore``), the Cholesky-decomposed two-electron integrals
(``chol``, from ``mo_2e_int_eri_cholesky``), the electron counts
(``nup``/``ndn``), the nuclear repulsion (``e0``) and the determinant list
(``ci_coeffs`` with the decoded ``occa``/``occb`` index arrays). These map directly
onto :func:`~ipie.hamiltonians.generic.Generic` and
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`
(``examples/11-trexio``). The whole Cholesky tensor is assembled in memory, so
this is aimed at small to medium systems.

Dice / SHCI
~~~~~~~~~~~

:mod:`ipie.utils.from_dice` contains two layers. The low-level functions
:func:`~ipie.utils.from_dice.read_dice_wavefunction` (parses Dice's binary
``dets.bin``) and :func:`~ipie.utils.from_dice.convert_phase` (reorders
spin-orbital strings from ``abab`` to ``aabb`` ordering and fixes the sign)
are exposed on the command line by ``tools/extract_dice.py``:

.. code-block:: bash

   python tools/extract_dice.py --dice-wfn dets.bin --filename wfn.h5 --sort --ndets 1000

which writes an ipie particle-hole wavefunction with
:func:`ipie.utils.io.write_wavefunction`. Note that ``tools/extract_dice.py``
imports :mod:`ipie.utils.from_dice`, which requires the ``pyscf.shciscf``
plugin to be importable even for this offline conversion; install it first or
the script fails with ``ImportError``. The high-level functions
:func:`~ipie.utils.from_dice.run_shci_coarse`, :func:`~ipie.utils.from_dice.run_shciscf`,
:func:`~ipie.utils.from_dice.build_trial_from_shciscf` and
:func:`~ipie.utils.from_dice.build_driver_from_shciscf` drive Dice through the
``pyscf.shciscf`` plugin to select an active space from natural-orbital
occupations and return a ready :class:`~ipie.qmc.afqmc.AFQMC` driver
(``examples/12-shciscf-active-space``). The module raises ``ImportError`` at
import time if the plugin is missing.

FQE wavefunctions
~~~~~~~~~~~~~~~~~

There is no dedicated module for OpenFermion-FQE; instead
``examples/14-fqe-wavefunction/run_afqmc.py`` contains reusable helper functions
(``get_occa_occb_coeff_from_fqe_wfn``, ``get_fqe_wfn_from_occ_coeff``,
``strip_melting_cores``) that translate between an ``fqe.Wavefunction`` and the
``(coeffs, occa, occb)`` tuple expected by
:class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`, and verify the
conversion by comparing variational energies.

FCIDUMP
~~~~~~~

``tools/fcidump_to_afqmc.py`` converts a Molpro-style FCIDUMP into an ipie
Hamiltonian file:

.. code-block:: bash

   python tools/fcidump_to_afqmc.py -i FCIDUMP -o fcidump.h5 -t 1e-5 -s 8

It reads the file with :func:`ipie.utils.hamiltonian_converter.read_fcidump`
(``-s`` gives the permutational symmetry, 1, 4 or 8), performs a modified
Cholesky decomposition with :func:`ipie.utils.linalg.modified_cholesky` to the
threshold ``-t``, and writes the result with
:func:`ipie.utils.io.write_qmcpack_dense`, switching to complex output
automatically if the integrals are complex (or with ``--write-complex``). The
output is read by :func:`~ipie.hamiltonians.utils.get_hamiltonian` like any
other Hamiltonian file; a trial wavefunction still has to be supplied
separately.

Save and restart
----------------

Checkpointing is implemented at the walker level rather than in the driver.
:class:`ipie.walkers.base_walkers.BaseWalkers` (and therefore
:class:`~ipie.walkers.uhf_walkers.UHFWalkers`) accepts ``write_filepath``,
``write_restart``, ``write_freq`` and ``write_time`` constructor arguments.
When ``write_restart`` is true, the main loop in :meth:`ipie.qmc.afqmc.AFQMC.run`
calls :meth:`~ipie.walkers.base_walkers.BaseWalkers.write_walkers_batch` every
``write_freq`` steps (or once, at step ``write_time``). Each call appends a new
time slice, ``walker_timeslice_<n>``, ``walker_weight_<n>`` and
``walker_hybrid_energy_<n>``, to ``walkers_<rank>.h5`` in ``write_filepath``;
device arrays are copied to the host first.

To restart, build the driver as usual, then call
:meth:`~ipie.walkers.base_walkers.BaseWalkers.read_walkers_batch` on its
walkers before ``run()``:

.. code-block:: python

   afqmc = AFQMC.build(nelec, ham, trial, num_walkers=num_walkers, ...)
   afqmc.walkers.read_walkers_batch(trial, comm)   # loads the last time slice
   afqmc.run()

The reader restores ``phia``/``phib``, the weights and hybrid energies, and
recomputes the trial overlaps. The number of walkers per rank and the rank
count must match those used when writing, since files are per rank. Nothing
else (random-number state, estimator accumulators, the energy shift) is
checkpointed, so a restarted run is a statistically independent continuation
rather than a bitwise resumption; the equilibration blocks can, however, be
skipped. ``examples/20-save_and_restart`` demonstrates the mechanism by
triggering the write from a custom energy estimator and checking that the
restored population reproduces the last block's energy. The alternative
factory :meth:`ipie.qmc.afqmc.AFQMC.build_from_hdf5` rebuilds the Hamiltonian
and trial from ``hamiltonian.h5``/``wavefunction.h5`` and is convenient for
restart scripts.

Model Hamiltonians: the Hubbard model
-------------------------------------

Dedicated ``Hubbard`` Hamiltonian, propagator and trial classes exist only in
``ipie.legacy`` (``ipie/legacy/hamiltonians/hubbard.py`` and related modules),
which is not exercised by the modern driver. In the current code, lattice
models are handled by the generic machinery: any Hamiltonian whose two-body
term is expressed as a sum of squares can be passed to
:class:`ipie.hamiltonians.generic.GenericRealChol` (or the complex variant). For
the on-site Hubbard interaction the "Cholesky" vectors are trivial to construct
by hand, and ``examples/19-hubbard`` does exactly this for a two-site periodic
chain:

.. code-block:: python

   eri = numpy.zeros((n, n, n, n))
   for i in range(n):
       eri[i, i, i, i] = U
   u, s, vdag = scipy.linalg.svd(eri.reshape(n**2, n**2))
   chol = (u @ numpy.diag(s ** 0.5))[:, :n]          # (n*n, nchol)
   ham = GenericRealChol(numpy.array([h1, h1]), chol, 0.0)

The trial (a UHF determinant from PySCF with a spin-density-wave initial guess)
and the driver are then the standard :class:`~ipie.trial_wavefunction.single_det.SingleDet`
and :meth:`~ipie.qmc.afqmc.AFQMC.build`. This route uses the continuous
(Gaussian) Hubbard-Stratonovich transformation; the discrete
spin-decomposition propagators of the legacy code are not available in the
modern driver.
