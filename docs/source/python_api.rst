Driving ipie from Python
========================

The JSON input file described in :doc:`input_file` is convenient for routine
calculations, but everything it does is a thin layer over a small set of Python
objects. Building those objects yourself gives you full control: you can
construct integrals in memory rather than on disk, swap in a custom trial
wavefunction or walker class, add your own observables, and post-process the
results in the same script. This page describes that object model and the
different ways of assembling a calculation from Python. All of the snippets are
distilled from the scripts in the ``examples/`` directory of the repository; see
:doc:`examples` for a map of those.

.. note::

   For the impatient: the shortest path from a PySCF calculation to an AFQMC
   energy is :func:`ipie.qmc.calc.build_afqmc_driver`, shown in
   :ref:`python-api-build-afqmc-driver`. The most flexible path is
   :meth:`ipie.qmc.afqmc.AFQMC.build`, shown in :ref:`python-api-afqmc-build`.

The object model
----------------

An AFQMC run in ipie is assembled from the following pieces. The names are the
attribute names on the driver object, so once you have an ``afqmc`` object you
can inspect ``afqmc.hamiltonian``, ``afqmc.trial`` and so on.

``system``
   :class:`ipie.systems.generic.Generic`. Holds only the number of alpha and
   beta electrons (``nup``, ``ndown``, ``nelec``). Constructed as
   ``Generic(nelec=(nalpha, nbeta))``.

``hamiltonian``
   A container for the one-body integrals, the Cholesky-decomposed two-body
   integrals and the constant (nuclear plus frozen-core) energy. For molecules
   this is :class:`ipie.hamiltonians.generic.GenericRealChol` (real integrals)
   or :class:`ipie.hamiltonians.generic.GenericComplexChol` (complex
   integrals); the factory function :func:`ipie.hamiltonians.generic.Generic`
   picks one based on the dtype of the Cholesky vectors. See
   :ref:`python-api-hamiltonian` for the expected array layout.

``trial``
   The trial wavefunction, a subclass of
   :class:`ipie.trial_wavefunction.wavefunction_base.TrialWavefunctionBase`:
   :class:`ipie.trial_wavefunction.single_det.SingleDet` (UHF/RHF/ROHF
   determinant), :class:`ipie.trial_wavefunction.particle_hole.ParticleHole`
   (multi-determinant CI-type expansion),
   :class:`ipie.trial_wavefunction.noci.NOCI` (non-orthogonal CI) and
   :class:`ipie.trial_wavefunction.single_det_ghf.SingleDetGHF` (generalised
   Hartree--Fock). Every trial must be prepared with ``trial.build()`` and
   ``trial.half_rotate(hamiltonian)`` before it can be used; the second call
   pre-contracts the integrals with the trial orbitals, which is what makes
   the local energy and force bias cheap.

``walkers``
   The population of Slater determinants on this MPI task, for example
   :class:`ipie.walkers.uhf_walkers.UHFWalkers`. Different trials need
   different walker classes (multi-determinant trials keep extra
   intermediates); the dispatch function
   :func:`ipie.walkers.walkers_dispatch.UHFWalkersTrial` selects the right one
   for a given trial, and
   :func:`ipie.walkers.walkers_dispatch.get_initial_walker` provides a
   sensible starting determinant. Walkers must be prepared with
   ``walkers.build(trial)``.

``propagator``
   Implements one imaginary-time step, including the force bias and the
   phaseless constraint. For molecular Hamiltonians this is
   :class:`ipie.propagation.phaseless_generic.PhaselessGeneric`. The module
   :mod:`ipie.propagation.propagator` exposes a dictionary ``Propagator`` that
   maps each Hamiltonian class to the propagator that handles it, so
   ``Propagator[type(hamiltonian)](timestep)`` always gives the right class.
   The propagator must be prepared with
   ``propagator.build(hamiltonian, trial, walkers, mpi_handler)``.

``params``
   :class:`ipie.qmc.options.QMCParams`, a dataclass holding the numerical
   parameters of the run (number of walkers, timestep, block structure,
   stabilisation and population-control frequencies, random seed, and so on).
   Only the block-structure fields (``num_blocks``, ``num_steps_per_block``,
   ``num_stblz``, ``pop_control_freq``, ``pop_control_method``,
   ``num_eq_blocks`` and their ``eq_`` counterparts) are read by ``run()`` and
   can therefore be changed after the driver has been built; ``timestep`` and
   ``rng_seed`` are consumed when the propagator and driver are constructed, and
   the walker count is fixed by the walker container, so set those when
   building the driver.

``mpi_handler``
   :class:`ipie.utils.mpi.MPIHandler`. Wraps ``MPI.COMM_WORLD`` and, when the
   Hamiltonian is distributed over several ranks (``nmembers > 1``), the
   split communicator for each group. If ``mpi4py`` is not installed a serial
   stand-in communicator is used automatically, so all of the code on this
   page runs unchanged in serial.

``estimators``
   An :class:`ipie.estimators.handler.EstimatorHandler` that owns the
   observables measured at the end of each block. It is created inside
   ``run()``; the energy estimator is always present and you can add more, see
   :ref:`python-api-custom-observables`.

The driver itself is :class:`ipie.qmc.afqmc.AFQMC`. Its constructor simply
stores the objects above::

   AFQMC(system, hamiltonian, trial, walkers, propagator, mpi_handler, params,
         eq_propagator=None, verbose=0)

Most of the time you will not call the constructor directly but one of the
factory routines described next.

Building blocks
---------------

.. _python-api-hamiltonian:

Integrals and the Hamiltonian object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ipie expects the two-electron integrals in Cholesky-decomposed form,
:math:`(ik|jl) \approx \sum_\gamma L^\gamma_{ik} L^\gamma_{jl}`. The
Hamiltonian constructors take

* ``h1e``: a real or complex array of shape ``(2, nbasis, nbasis)`` holding the
  one-body Hamiltonian for the alpha and beta spin channels (usually the same
  matrix twice), and
* ``chol``: the Cholesky vectors as a two-dimensional array of shape
  ``(nbasis * nbasis, nchol)``, i.e. the pair index :math:`(i,k)` is flattened
  into the first dimension and the Cholesky index is last,
* ``ecore``: a float holding the nuclear repulsion (plus any frozen-core
  contribution).

The HDF5 file written by the PySCF converters uses the *other* layout,
``LXmn`` with shape ``(nchol, nbasis, nbasis)``, so a transpose and reshape is
needed when you read one back:

.. code-block:: python

   import h5py
   import numpy
   from ipie.hamiltonians.generic import Generic as HamGeneric

   with h5py.File("hamiltonian.h5", "r") as fh5:
       LXmn = fh5["LXmn"][()]   # (nchol, nbasis, nbasis)
       hcore = fh5["hcore"][()] # (nbasis, nbasis)
       e0 = fh5["e0"][()]

   nchol, nbasis, _ = LXmn.shape
   chol = LXmn.transpose((1, 2, 0)).reshape((nbasis * nbasis, nchol))
   ham = HamGeneric(numpy.array([hcore, hcore]), chol, e0)

The helpers :func:`ipie.utils.from_pyscf.copy_LXmn_to_LPX` and
:func:`ipie.utils.from_pyscf.copy_LPX_to_LXmn` perform exactly this
conversion in each direction. If all you want is to load a file, the
convenience function :func:`ipie.hamiltonians.utils.get_hamiltonian` reads
``hamiltonian.h5``, places the integrals in MPI shared memory when available,
and returns a ready-made object::

   from ipie.utils.mpi import MPIHandler
   from ipie.hamiltonians.utils import get_hamiltonian

   mpi_handler = MPIHandler()
   ham = get_hamiltonian("hamiltonian.h5", mpi_handler.scomm, pack_chol=True, verbose=True)

Generating integrals from PySCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`ipie.utils.from_pyscf` provides both file-based and in-memory routes from
a PySCF calculation. The file-based route writes ``hamiltonian.h5`` and
``wavefunction.h5`` from a PySCF checkpoint file and is what the
``tools/pyscf/pyscf_to_ipie.py`` script calls internally:

.. code-block:: python

   from pyscf import gto, scf
   from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

   mol = gto.M(atom=[("H", 1.6 * i, 0, 0) for i in range(10)],
               basis="sto-6g", unit="Bohr")
   mf = scf.UHF(mol)
   mf.chkfile = "scf.chk"
   mf.kernel()

   gen_ipie_input_from_pyscf_chk(
       mf.chkfile,
       hamil_file="hamiltonian.h5",
       wfn_file="wavefunction.h5",
       chol_cut=1e-5,          # Cholesky truncation threshold
       ortho_ao=False,         # work in the MO basis (True: orthogonalised AO basis)
       mcscf=False,            # True to read a CASSCF/CI expansion from the chkfile
       num_frozen_core=0,
   )

The in-memory route skips the files entirely. :func:`ipie.utils.from_pyscf.generate_hamiltonian`
returns a :class:`~ipie.hamiltonians.generic.GenericRealChol` and
:func:`ipie.utils.from_pyscf.generate_wavefunction_from_mo_coeff` returns the
occupied orbitals expressed in the chosen basis:

.. code-block:: python

   import numpy as np
   from ipie.utils.from_pyscf import generate_hamiltonian, generate_wavefunction_from_mo_coeff
   from ipie.trial_wavefunction.single_det import SingleDet

   mf = scf.RHF(mol).run()

   ham = generate_hamiltonian(mol, mf.mo_coeff, mf.get_hcore(), mf.mo_coeff)

   orbs = generate_wavefunction_from_mo_coeff(mf.mo_coeff, mf.mo_occ, mf.mo_coeff, mol.nelec)
   num_basis = mf.mo_coeff.shape[-1]
   # For an RHF reference `orbs` is a single (nbasis, nalpha) block; SingleDet
   # wants alpha and beta orbitals side by side.
   trial = SingleDet(np.hstack([orbs, orbs]), mol.nelec, num_basis)
   trial.build()
   trial.half_rotate(ham)

For a UHF reference ``generate_wavefunction_from_mo_coeff`` returns a list
``[orbs_alpha, orbs_beta]`` and the trial is ``SingleDet(np.hstack(orbs), ...)``.
If you would rather do the basis change yourself,
:func:`ipie.utils.from_pyscf.generate_integrals` returns the raw
``(h1e, chol, enuc)`` triple with ``chol`` in ``(nchol, nbasis, nbasis)`` layout
(see ``examples/17-ghf_afqmc/run_afqmc.py``), and
:func:`ipie.utils.from_pyscf.generate_hamiltonian_from_chk` builds the
Hamiltonian object directly from a checkpoint file.

Trial wavefunctions
~~~~~~~~~~~~~~~~~~~

All trial classes take the number of electrons ``(nalpha, nbeta)`` and the
number of basis functions and must be prepared with ``build()`` and
``half_rotate(hamiltonian)``:

* ``SingleDet(wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False)``
  where ``wavefunction`` is an array of shape ``(nbasis, nalpha + nbeta)`` holding
  the alpha orbitals followed by the beta orbitals. In the MO basis of an RHF
  reference this is just the first columns of the identity matrix.
* ``ParticleHole(wfn, nelec, nbasis, num_dets_for_props=100, num_dets_for_trial=-1, num_det_chunks=1, use_active_space=True, verbose=False)``
  where ``wfn = (coeffs, occ_alpha, occ_beta)`` is a CI expansion given as an
  array of coefficients and the lists of occupied orbital indices for each
  determinant. See :ref:`python-api-msd` below.
* ``NOCI(wavefunction, num_elec, num_basis, verbose=False)`` where
  ``wavefunction = (coeffs, psi)`` and ``psi`` has shape
  ``(ndet, nbasis, nalpha + nbeta)``.
* ``SingleDetGHF(wavefunction, num_elec, num_basis, verbose=False)`` where
  ``wavefunction`` has shape ``(2 * nbasis, nalpha + nbeta)`` in the spin-orbital
  basis. A GHF trial can also be built from an existing UHF trial,
  ``SingleDetGHF(uhf_trial)``.

Setting ``trial.compute_trial_energy = True`` before handing the trial to the
driver makes ``AFQMC.build`` evaluate the variational energy of the trial
(``trial.calculate_energy(system, hamiltonian)``), a cheap and very useful
sanity check. The result is available afterwards as ``trial.energy`` (with
``trial.e1b`` and ``trial.e2b``) and is printed only if the trial was created
with ``verbose=True``.

When the trial lives in a ``wavefunction.h5`` file, the factory
:func:`ipie.trial_wavefunction.utils.get_trial_wavefunction` detects the
format (``psi_T_alpha``/``psi_T_beta`` datasets for a single determinant,
``ci_coeffs``/``occ_alpha``/``occ_beta`` for a particle-hole expansion, or
``ci_coeffs``/``psi_T_alpha``/``psi_T_beta`` for NOCI) and returns the
corresponding object.

Walkers
~~~~~~~

Walker containers hold ``nwalkers`` determinants *for the current MPI task* as
the arrays ``walkers.phia`` (``(nwalkers, nbasis, nalpha)``) and
``walkers.phib``, together with the weights ``walkers.weight``, the overlaps
with the trial and, after a Green's function evaluation, ``walkers.Ga`` and
``walkers.Gb``. The constructor is::

   UHFWalkers(initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose=False)

where ``initial_walker`` is a single ``(nbasis, nalpha + nbeta)`` matrix that is
copied into every walker. Multi-determinant trials use
:class:`ipie.walkers.uhf_walkers.UHFWalkersParticleHole` or
:class:`ipie.walkers.uhf_walkers.UHFWalkersNOCI`, and GHF trials use
:class:`ipie.walkers.ghf_walkers.GHFWalkers`. Rather than remembering the
pairing, let ipie choose:

.. code-block:: python

   from ipie.walkers.walkers_dispatch import UHFWalkersTrial, get_initial_walker

   _, initial_walker = get_initial_walker(trial)
   walkers = UHFWalkersTrial(trial, initial_walker, nup, ndown, ham.nbasis, num_walkers, mpi_handler)
   walkers.build(trial)

This is exactly what ``AFQMC.build`` does when you do not pass walkers
yourself.

.. _python-api-afqmc-build:

Route 1: ``AFQMC.build``
------------------------

:meth:`ipie.qmc.afqmc.AFQMC.build` is the recommended entry point when you
already hold a Hamiltonian object and a prepared trial. It creates the
system, the walkers (unless you provide them), the propagator and the
:class:`~ipie.qmc.options.QMCParams`, and returns a driver ready to run. Its
full signature and defaults are::

   AFQMC.build(
       num_elec,                      # (nalpha, nbeta)
       hamiltonian,
       trial_wavefunction,
       walkers=None,                  # built from the trial if None
       num_walkers=100,               # walkers PER MPI TASK
       seed=None,                     # random seed (None: drawn on rank 0 and broadcast)
       num_steps_per_block=25,
       num_blocks=100,
       timestep=0.005,
       stabilize_freq=5,              # QR re-orthogonalisation frequency (steps)
       eq_stabilize_freq=2,           # same, during the optional equilibration phase
       pop_control_method="pair_branch",   # or "comb", "stochastic_reconfiguration"
       pop_control_freq=5,            # population control frequency (steps)
       eq_pop_control_freq=2,
       eq_timestep=None,              # timestep for the equilibration phase (None: timestep)
       eq_num_steps_per_block=None,   # (None: num_steps_per_block)
       num_eq_blocks=0,               # blocks of explicit equilibration before production
       ene_bound_const=2.0,           # local-energy bound is sqrt(ene_bound_const / timestep)
       fb_bound=1.0,                  # force-bias bound
       correlated_samp=False,
       reference_run=False,
       walkermap_filepath=None,
       verbose=True,
       mpi_handler=None,              # a fresh MPIHandler() if None
   )

A complete single-determinant example, adapted from
``examples/07-custom_trial/run_afqmc.py``:

.. code-block:: python

   import numpy as np
   from pyscf import gto, scf

   from ipie.qmc.afqmc import AFQMC
   from ipie.trial_wavefunction.single_det import SingleDet
   from ipie.utils.from_pyscf import generate_hamiltonian, generate_wavefunction_from_mo_coeff

   mol = gto.M(atom=[("H", 1.6 * i, 0, 0) for i in range(10)],
               basis="sto-6g", unit="Bohr")
   mf = scf.RHF(mol).run()

   ham = generate_hamiltonian(mol, mf.mo_coeff, mf.get_hcore(), mf.mo_coeff)
   orbs = generate_wavefunction_from_mo_coeff(mf.mo_coeff, mf.mo_occ, mf.mo_coeff, mol.nelec)
   trial = SingleDet(np.hstack([orbs, orbs]), mol.nelec, mf.mo_coeff.shape[-1])
   trial.compute_trial_energy = True
   trial.build()
   trial.half_rotate(ham)

   afqmc = AFQMC.build(
       mol.nelec, ham, trial,
       num_walkers=100,
       num_steps_per_block=25,
       num_blocks=100,
       timestep=0.005,
       seed=59306159,
   )
   afqmc.run()
   afqmc.finalise(verbose=True)

The walkers are re-orthogonalised every ``stabilize_freq`` steps, population
control happens every ``pop_control_freq`` steps, and observables are measured
and written once every ``num_steps_per_block`` steps. The total imaginary time
projected is ``num_blocks * num_steps_per_block * timestep``. Parameters live
in ``afqmc.params``; the block-structure fields may be changed before calling
``run()``::

   afqmc.params.num_blocks = 20
   print(afqmc.params)

``timestep`` and ``seed``, on the other hand, are baked into the propagator and
the random-number generator when the driver is built, and ``num_walkers`` fixes
the size of the walker container, so editing those fields of ``afqmc.params``
afterwards has no effect on the propagation; rebuild the driver instead.

Route 2: ``AFQMC.build_from_hdf5``
----------------------------------

If the integrals and trial are already on disk in ipie's HDF5 formats,
:meth:`ipie.qmc.afqmc.AFQMC.build_from_hdf5` reads them (through
:func:`~ipie.hamiltonians.utils.get_hamiltonian` and
:func:`~ipie.trial_wavefunction.utils.get_trial_wavefunction`), half-rotates
the trial and calls ``AFQMC.build``::

   AFQMC.build_from_hdf5(
       num_elec,
       ham_file,                        # e.g. "hamiltonian.h5"
       wfn_file,                        # e.g. "wavefunction.h5"
       num_walkers=100,
       seed=None,
       num_steps_per_block=25,
       num_blocks=100,
       timestep=0.005,
       stabilize_freq=5,
       pop_control_freq=5,
       num_dets_chunk=1,                # multi-determinant trials: chunks of determinants
       num_dets_for_trial_props=100,    # multi-determinant trials: dets used for trial properties
       pack_cholesky=True,              # store only the upper triangle of each L^gamma
       verbose=True,
   )

For example:

.. code-block:: python

   from ipie.qmc.afqmc import AFQMC

   afqmc = AFQMC.build_from_hdf5((5, 5), "hamiltonian.h5", "wavefunction.h5",
                                 num_walkers=50, num_blocks=100, seed=7)
   afqmc.run()
   afqmc.finalise(verbose=True)

This is the closest Python equivalent of running ``ipie input.json``. Note
that it only exposes the most common options; use ``AFQMC.build`` when you need
the equilibration-phase or bound parameters.

.. _python-api-build-afqmc-driver:

Route 3: ``build_afqmc_driver`` and the input dictionary
--------------------------------------------------------

:mod:`ipie.qmc.calc` contains the machinery behind the command-line
executable. :func:`ipie.qmc.calc.build_afqmc_driver` is the smallest possible
wrapper around it::

   build_afqmc_driver(
       comm,                                 # MPI communicator, e.g. MPI.COMM_WORLD
       nelec,                                # (nalpha, nbeta)
       wavefunction_file="wavefunction.h5",
       hamiltonian_file="hamiltonian.h5",
       num_walkers_per_task=10,
       estimator_filename="estimates.0.h5",   # currently ignored; pass estimator_filename to run() instead
       seed=None,
       verbosity=0,
   )

Together with :func:`~ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk`
this gives the shortest complete script (``examples/10-pyscf_interface/run_afqmc.py``):

.. code-block:: python

   from pyscf import gto, scf
   from ipie.config import MPI
   from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
   from ipie.qmc.calc import build_afqmc_driver

   comm = MPI.COMM_WORLD
   mol = gto.M(atom=[("H", 1.6 * i, 0, 0) for i in range(10)],
               basis="sto-6g", unit="Bohr")
   if comm.rank == 0:
       mf = scf.UHF(mol)
       mf.chkfile = "scf.chk"
       mf.kernel()
       gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)
   comm.barrier()

   afqmc = build_afqmc_driver(comm, nelec=mol.nelec, num_walkers_per_task=100, seed=41100801)
   afqmc.params.num_blocks = 100   # override any default before running
   afqmc.run()

Note that the QMC parameters other than the walker count and seed take the
defaults of the input-file parser (timestep 0.005, 25 steps per block, 1000
blocks, stabilisation and population control every 5 steps), so you will
usually want to adjust the block structure through ``afqmc.params`` as above;
the timestep can only be changed through ``get_driver`` (``"qmc": {"dt": ...}``,
see below). The trial options also take the input-file defaults, in particular
``ndets`` and ``ndets_props`` default to 1, so a multi-determinant
``wavefunction.h5`` is silently truncated to its first determinant. For such a
trial use ``get_driver`` with ``"trial": {"filename": ..., "ndets": -1,
"ndets_props": N}`` or ``AFQMC.build_from_hdf5``, which keeps all
determinants.

``build_afqmc_driver`` is itself a call to :func:`ipie.qmc.calc.get_driver`
with a dictionary that has the same structure as the JSON input file. You can
call ``get_driver`` directly with any dictionary that :doc:`input_file`
accepts, which is handy for generating parameter sweeps programmatically:

.. code-block:: python

   from ipie.qmc.calc import get_driver

   options = {
       "system": {"nup": 5, "ndown": 5},
       "hamiltonian": {"name": "Generic", "integrals": "hamiltonian.h5"},
       "trial": {"filename": "wavefunction.h5"},
       "qmc": {"dt": 0.005, "nwalkers": 100, "nsteps": 25, "blocks": 200, "rng_seed": 7},
   }
   afqmc = get_driver(options, comm)
   afqmc.run()

Finally, :func:`ipie.qmc.calc.setup_calculation` accepts either such a
dictionary or the path of a JSON file and returns ``(afqmc, comm)``; it is what
the ``ipie`` executable calls.

Route 4: assembling the driver by hand
--------------------------------------

When you need to replace one of the components with your own class (see
:ref:`python-api-custom-classes`) you can skip the factories and wire the
objects together yourself, following ``examples/08-custom_walker/run_afqmc.py``:

.. code-block:: python

   from ipie.config import MPI
   from ipie.qmc.afqmc import AFQMC
   from ipie.qmc.options import QMCParams
   from ipie.systems.generic import Generic
   from ipie.propagation.propagator import Propagator
   from ipie.utils.mpi import MPIHandler
   from ipie.walkers.uhf_walkers import UHFWalkers

   comm = MPI.COMM_WORLD
   mpi_handler = MPIHandler()
   num_walkers = 100

   system = Generic(nelec=mol.nelec)
   # ham and trial built and half-rotated as above

   walkers = UHFWalkers(np.hstack([orbs, orbs]), system.nup, system.ndown,
                        ham.nbasis, num_walkers, mpi_handler)
   walkers.build(trial)

   params = QMCParams(
       num_walkers=num_walkers,
       total_num_walkers=num_walkers * comm.size,
       num_blocks=10,
       num_steps_per_block=25,
       timestep=0.005,
       rng_seed=7,
   )

   propagator = Propagator[type(ham)](params.timestep)
   propagator.build(ham, trial, walkers, mpi_handler)

   afqmc = AFQMC(system, ham, trial, walkers, propagator, mpi_handler, params)
   afqmc.run()

The remaining :class:`~ipie.qmc.options.QMCParams` fields (``num_stblz``,
``pop_control_method``, ``pop_control_freq``, ``num_eq_blocks``,
``eq_timestep``, ``fb_bound``, ``ene_bound_const``, ...) have the same
defaults as the keyword arguments of ``AFQMC.build``.

Running, output and clean-up
----------------------------

:meth:`ipie.qmc.afqmc.AFQMC.run` performs the random walk::

   afqmc.run(
       walkers=None,                 # replace the driver's walkers before starting
       estimator_filename=None,      # HDF5 output file; None -> "estimates.0.h5"
       verbose=True,
       discard_weights_aftereq=False,   # reset weights to 1 after the equilibration phase
       additional_estimators=None,   # dict name -> EstimatorBase, see below
   )

During the run one line per block is printed to stdout on rank 0 and the same
numbers are written to the HDF5 file, whose name is available afterwards as
``afqmc.estimators.filename``. The file is overwritten if it already exists,
so pass a distinct ``estimator_filename`` when running several calculations
from one script. :meth:`AFQMC.finalise() <ipie.qmc.afqmc.AFQMCBase.finalise>` prints wall-clock
timings broken down by propagation, estimators, orthogonalisation and
population control::

   afqmc.finalise(verbose=True)

Reading the output back into pandas and turning it into an energy with an
error bar is covered in :doc:`analysis`; the one-liner is

.. code-block:: python

   from ipie.analysis.extraction import extract_observable
   from ipie.analysis.autocorr import reblock_by_autocorr

   data = extract_observable(afqmc.estimators.filename, "energy")
   print(reblock_by_autocorr(data["ETotal"].values[10:]))   # discard 10 blocks

.. _python-api-msd:

Multi-determinant trials
------------------------

Multi-determinant (particle--hole) trials are described by CI coefficients and,
for each determinant, the indices of the occupied alpha and beta orbitals.
``examples/02-multi_determinant/run_afqmc.py`` obtains them from a PySCF
CASSCF calculation, stores them in the checkpoint file and lets
``gen_ipie_input_from_pyscf_chk(..., mcscf=True)`` write ``wavefunction.h5``
with the datasets ``ci_coeffs``, ``occ_alpha`` and ``occ_beta``:

.. code-block:: python

   import h5py
   from pyscf import fci, gto, mcscf, scf
   from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

   mol = gto.M(atom=[("N", 0, 0, 0), ("N", (0, 0, 3.0))], basis="ccpvdz",
               spin=2, unit="Bohr")
   mf = scf.RHF(mol)
   mf.chkfile = "scf.chk"
   mf.kernel()
   mc = mcscf.CASSCF(mf, 6, 6)
   mc.chkfile = "scf.chk"
   e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
   coeff, occa, occb = zip(*fci.addons.large_ci(fcivec, 6, (4, 2), tol=1e-8, return_strs=False))
   with h5py.File("scf.chk", "r+") as fh5:
       fh5["mcscf/ci_coeffs"] = coeff
       fh5["mcscf/occs_alpha"] = occa
       fh5["mcscf/occs_beta"] = occb
   gen_ipie_input_from_pyscf_chk("scf.chk", mcscf=True)

The trial is then a :class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`
built from the ``(coeffs, occ_alpha, occ_beta)`` tuple. The orbital indices
refer to the active space; ipie detects that the determinants contain fewer
electrons than ``nelec`` and inserts the doubly occupied core orbitals itself.
``use_active_space=True`` (the default) additionally restricts the
Wick's-theorem intermediates to the active orbitals actually referenced by the
expansion. The walkers must be
:class:`~ipie.walkers.uhf_walkers.UHFWalkersParticleHole` (which is what
``UHFWalkersTrial`` returns for this trial):

.. code-block:: python

   import numpy
   from ipie.qmc.afqmc import AFQMC
   from ipie.trial_wavefunction.particle_hole import ParticleHole
   from ipie.utils.mpi import MPIHandler
   from ipie.walkers.uhf_walkers import UHFWalkersParticleHole

   # `ham` is built from hamiltonian.h5 exactly as in the Hamiltonian section above.
   with h5py.File("wavefunction.h5", "r") as fh5:
       wavefunction = (fh5["ci_coeffs"][:], fh5["occ_alpha"][:], fh5["occ_beta"][:])

   nelec = (8, 6)
   num_walkers = 100
   trial = ParticleHole(
       wavefunction, nelec, ham.nbasis,
       num_dets_for_props=len(wavefunction[0]),   # dets used for trial energy / 1-RDM
       verbose=True,
   )
   trial.compute_trial_energy = True
   trial.build()
   trial.half_rotate(ham)

   # Start the walkers from a slightly perturbed reference determinant.
   initial_walker = numpy.hstack([trial.psi0a, trial.psi0b])
   numpy.random.seed(123456789)
   initial_walker, _ = numpy.linalg.qr(initial_walker + numpy.random.random(initial_walker.shape))
   walkers = UHFWalkersParticleHole(initial_walker, nelec[0], nelec[1], ham.nbasis,
                                    num_walkers, MPIHandler())
   walkers.build(trial)

   afqmc = AFQMC.build(nelec, ham, trial, walkers=walkers, num_walkers=num_walkers,
                       num_blocks=100, seed=96264512)
   afqmc.run()

The most important knobs are:

``num_dets_for_trial``
   How many determinants (in the order given) to keep in the trial; ``-1``
   (default) keeps all of them.

``num_dets_for_props``
   How many determinants to use when evaluating properties of the trial
   itself, such as its variational energy. Defaults to 100.

``num_det_chunks``
   The Wick's-theorem local-energy code processes the determinant list in
   chunks to bound memory; increase this for very long expansions. The chunk
   granularity is set only by this parameter (the ``max_memory_for_wicks``
   option of ``ipie.config.config`` is defined but not consulted by the code).
   When reading the trial from a file with ``build_from_hdf5`` these two
   parameters are called ``num_dets_chunk`` and ``num_dets_for_trial_props``.

Three sibling classes in :mod:`ipie.trial_wavefunction.particle_hole`
implement the same trial. ``ParticleHoleNonChunked`` is the class actually
instantiated by :func:`~ipie.trial_wavefunction.utils.get_trial_wavefunction`
(hence by ``build_from_hdf5`` and the JSON input) when ``num_dets_chunk == 1``,
the default; ``ParticleHoleSlow`` and ``ParticleHoleNaive`` are progressively
simpler and slower reference implementations used mainly for testing. Note
that the ``ParticleHole`` classes are not
moved to the GPU when GPU mode is enabled; only the Hamiltonian, propagator and
walkers are.

GHF trials and walkers
~~~~~~~~~~~~~~~~~~~~~~

Generalised Hartree--Fock trials, in which alpha and beta orbitals are mixed,
use :class:`~ipie.trial_wavefunction.single_det_ghf.SingleDetGHF` together
with :class:`~ipie.walkers.ghf_walkers.GHFWalkers`; both take a
``(2 * nbasis, nalpha + nbeta)`` spin-orbital coefficient matrix and both can
alternatively be constructed from their UHF counterparts
(``SingleDetGHF(uhf_trial)``, ``GHFWalkers(uhf_walkers)``). Since
``AFQMC.build`` only knows how to create UHF-style walkers automatically, you
must build the GHF walkers yourself and pass them through the ``walkers``
argument; see ``examples/17-ghf_afqmc/run_afqmc.py`` for a complete script
that checks the GHF energy against the UHF one.

.. _python-api-custom-observables:

Custom observables
------------------

Every observable is a subclass of
:class:`ipie.estimators.estimator_base.EstimatorBase`. The
:class:`~ipie.estimators.handler.EstimatorHandler` calls
``compute_estimator(system, walkers, hamiltonian, trial)`` on each estimator
at the end of every block, sums the resulting buffers over MPI tasks, and
writes them to the HDF5 file. To add one you need to

1. define ``self._data``, an ordered dictionary of complex-valued buffers.
   Because averages in AFQMC are ratios of weighted sums, you should
   accumulate the weighted numerator(s) and the denominator (the sum of walker
   weights) separately, **with the denominator as the last entry**; the
   analysis routine divides everything else by it;
2. define ``self._shape``, the shape of the final observable;
3. set ``self.scalar_estimator`` to ``False`` for array-valued quantities
   (``True`` makes the entries appear as named columns, as the energy
   estimator does);
4. implement ``compute_estimator``.

The following is the essential part of ``Diagonal1RDM`` from
``examples/03-custom_observable/run_afqmc.py``, which measures the diagonal of
the mixed-estimate one-body density matrix:

.. code-block:: python

   import numpy as np
   from ipie.estimators.estimator_base import EstimatorBase

   class Diagonal1RDM(EstimatorBase):
       def __init__(self, ham):
           super().__init__()
           self._data = {
               "DiagGNumer": np.zeros((ham.nbasis), dtype=np.complex128),
               "DiagGDenom": np.zeros((1), dtype=np.complex128),
           }
           self._shape = (ham.nbasis,)
           self.scalar_estimator = False   # array valued
           self.print_to_stdout = False    # do not add columns to the stdout table
           self.ascii_filename = None      # optionally also write to a text file

       def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
           trial.calc_greens_function(walkers, build_full=True)   # fills walkers.Ga, walkers.Gb
           self["DiagGNumer"] = np.einsum("w,wii->i", walkers.weight, walkers.Ga + walkers.Gb)
           self["DiagGDenom"] = sum(walkers.weight)

Estimators are attached by name through the ``additional_estimators`` argument
of ``run()`` and read back with the same name:

.. code-block:: python

   afqmc.run(additional_estimators={"diagG": Diagonal1RDM(ham=afqmc.hamiltonian)})

   from ipie.analysis.extraction import extract_observable
   diag = extract_observable(afqmc.estimators.filename, "diagG")   # shape (nblocks + 1, nbasis)
   print(diag[-1].sum().real)   # should equal the number of electrons

Multi-dimensional observables have to be flattened into a one-dimensional
buffer (``self["GNumer"] = numer.ravel()`` with ``self._shape = (2, nbasis, nbasis)``);
``extract_observable`` reshapes them back. The same example file contains a
full ``Mixed1RDM`` class, and ``examples/04-s2_observable/run_afqmc.py``
measures :math:`\langle S^2 \rangle` with a scalar-like array estimator that
also prints to stdout.

Using the key ``"energy"`` in ``additional_estimators`` replaces the default
:class:`~ipie.estimators.energy.EnergyEstimator`. This is how the custom-trial
and save/restart examples inject their own local-energy evaluation while
keeping the standard ``ETotal``/``E1Body``/``E2Body`` columns.

.. warning::

   These are *mixed* estimates, :math:`\langle\Psi_T|O|\Phi\rangle /
   \langle\Psi_T|\Phi\rangle`, which are exact only for operators that commute
   with the Hamiltonian. For other observables the mixed estimate carries a
   bias that is first order in the error of the trial wavefunction. See
   :doc:`theory` before drawing conclusions from them.

.. _python-api-custom-classes:

Custom trial wavefunctions and walkers
--------------------------------------

Because the driver only relies on a handful of methods, you can subclass any
of the components. ``examples/07-custom_trial/run_afqmc.py`` derives
``NoisySingleDet`` from :class:`~ipie.trial_wavefunction.single_det.SingleDet`
and overrides ``calc_overlap``, ``calc_greens_function`` and
``calc_force_bias``, the three methods the propagator calls. For trials that
are not small variations on an existing class, inherit from
:class:`~ipie.trial_wavefunction.wavefunction_base.TrialWavefunctionBase` and
implement ``build``, ``half_rotate``, ``calc_overlap``,
``calc_greens_function`` and ``calc_force_bias``. The local energy of the
built-in :class:`~ipie.estimators.energy.EnergyEstimator` is selected by
multiple dispatch on the ``(system, hamiltonian, walkers, trial)`` types, so a
new trial type also needs an energy estimator; the example simply subclasses
``EnergyEstimator`` and calls
:func:`ipie.estimators.local_energy_batch.local_energy_batch` explicitly, then
passes it as ``additional_estimators={"energy": ...}``.

``examples/08-custom_walker/run_afqmc.py`` does the same for walkers,
subclassing :class:`~ipie.walkers.uhf_walkers.UHFWalkers` and overriding
``reortho``. Custom walkers cannot be created by ``AFQMC.build`` (it only
knows the dispatch table in :mod:`ipie.walkers.walkers_dispatch`), so either
pass them via ``walkers=`` or assemble the driver by hand as in Route 4.

Running with MPI
----------------

Every script on this page can be launched with ``mpirun``:

.. code-block:: bash

   mpirun -np 8 python run_afqmc.py > output.dat

Points to keep in mind:

* ``num_walkers`` (and ``num_walkers_per_task`` in ``build_afqmc_driver``) is
  the number of walkers **per MPI task**; the total population, which is what
  determines the population-control bias and the statistical error, is
  ``num_walkers * comm.size`` and is stored in ``afqmc.params.total_num_walkers``.
  Some examples (e.g. ``examples/02-multi_determinant``) therefore write
  ``num_walkers = 640 // comm.size`` to keep the total fixed regardless of the
  number of ranks.
* The random seed given on input is offset by the rank on each process, so
  the walkers on different tasks follow different streams. If ``seed`` is
  ``None`` a seed is drawn on rank 0 and broadcast, and it is recorded in the
  output.
* Only rank 0 prints and writes the estimator file. Anything that should
  happen once (running PySCF, writing ``hamiltonian.h5``) should be guarded
  with ``if comm.rank == 0:`` followed by ``comm.barrier()`` as in the
  ``build_afqmc_driver`` example above. ``from ipie.config import MPI`` gives
  you ``mpi4py.MPI`` when it is installed and a serial replacement otherwise.
* When the integrals are read with :func:`~ipie.hamiltonians.utils.get_hamiltonian`
  (as ``build_from_hdf5`` and ``get_driver`` do), the Cholesky vectors are
  placed in MPI shared memory so that ranks on the same node do not hold
  separate copies. When you construct the Hamiltonian from arrays yourself,
  each rank holds its own copy.
* For Hamiltonians too large to fit on one rank or GPU, the Cholesky vectors
  can be split across the ``nmembers`` ranks of an ``MPIHandler(nmembers=...)``
  group using :class:`ipie.hamiltonians.generic_chunked.GenericRealCholChunked`.
  See ``examples/06-gpu/run_afqmc_chunked.py`` and :doc:`advanced`.

Enabling the GPU
----------------

GPU execution requires CuPy and is switched on through the global
configuration object *before* the ipie modules that do numerical work are
imported, because the array library (NumPy or CuPy) is chosen at import time:

.. code-block:: python

   from ipie.config import config
   config.update_option("use_gpu", True)

   from ipie.qmc.afqmc import AFQMC     # import the rest of ipie after this point
   ...

Equivalently, set the environment variable ``IPIE_USE_GPU=1`` when launching
the script (``IPIE_USE_MIXED_PRECISION=1`` additionally enables mixed
precision). When GPU mode is active, ``run()`` assigns each MPI rank the
device ``rank % number_of_gpus`` and copies the propagator, Hamiltonian, trial
and walkers to it, so the natural configuration is one MPI task per GPU. See
``examples/06-gpu/`` for complete scripts.

Checkpointing walkers and restarting
------------------------------------

ipie can dump the walker population to disk and read it back. Each task writes
its own file, ``walkers_<rank>.h5`` (in the directory given by
``write_filepath``, by default the working directory), containing the
orbitals, weights and hybrid energies of every walker at the time of writing;
successive dumps are appended as ``walker_timeslice_N`` and so on. There are
two ways to trigger a dump:

* construct the walkers with ``write_restart=True`` and either
  ``write_freq=N`` (dump every ``N`` steps) or ``write_time=step`` (dump once
  at a given step)::

     walkers = UHFWalkers(initial_walker, nup, ndown, nbasis, num_walkers, mpi_handler,
                          write_restart=True, write_freq=250)

* or call ``walkers.write_walkers_batch(comm)`` yourself, for instance from
  inside a custom energy estimator, which is what
  ``examples/20-save_and_restart/run_afqmc.py`` does in order to control
  exactly which blocks are saved.

To resume from the last dump, build a new driver with the same Hamiltonian and
trial, then overwrite its freshly initialised walkers before running:

.. code-block:: python

   afqmc = AFQMC.build(mol.nelec, ham, trial, num_walkers=num_walkers, num_blocks=100)
   afqmc.walkers.read_walkers_batch(trial, comm)   # loads walkers_<rank>.h5
   afqmc.run(estimator_filename="estimates.1.h5")

``read_walkers_batch`` restores the orbitals, weights and hybrid energies of
the most recent time slice and recomputes the overlaps with the trial. It must
be called with the same number of MPI tasks and walkers per task as the run
that produced the files. Be aware of what is *not* restored: the random-number
state, the energy shift used in the propagator and the population-control
bookkeeping all start afresh, and the new estimator file starts again from
block 0. A restarted run is therefore statistically equivalent to, but not
bitwise identical with, an uninterrupted one, and you should treat the first
block or two after a restart as re-equilibration when analysing the combined
data (see :doc:`analysis`).
