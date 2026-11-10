Quickstart
==========

This page walks through the canonical ipie workflow on a small test case, a
chain of ten hydrogen atoms at a bond length of 1.6 bohr in the STO-6G basis
(the same system as ``examples/01-simple`` in the repository):

1. run a mean-field calculation with `PySCF <https://pyscf.org>`_ and save a
   checkpoint file,
2. convert the checkpoint into ipie's HDF5 integral and trial-wavefunction
   files plus a JSON input file,
3. run AFQMC from the command line, in serial or under MPI,
4. analyse the output.

A second section shows the same calculation driven entirely from Python.
Both routes need ``pyscf`` in addition to ipie (see :doc:`installation`).

Command-line workflow
---------------------

1. Mean-field calculation
~~~~~~~~~~~~~~~~~~~~~~~~~

Save the following as ``scf.py`` (it is ``examples/01-simple/scf.py``). The
only ipie-specific requirement is that a ``chkfile`` is written, since the
converter reads the molecule, the core Hamiltonian and the molecular-orbital
coefficients and occupations from it.

.. code-block:: python

   from pyscf import gto, scf

   atom = gto.M(
       atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
       basis="sto-6g",
       verbose=4,
       unit="Bohr",
   )
   mf = scf.UHF(atom)
   mf.chkfile = "scf.chk"
   mf.kernel()

.. code-block:: bash

   python scf.py

2. Generate the AFQMC input
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pyscf_to_ipie.py`` (``tools/pyscf/pyscf_to_ipie.py`` in the repository; it
is also installed on your ``PATH`` by ``setup.py``) turns the checkpoint into
everything ipie needs:

.. code-block:: bash

   pyscf_to_ipie.py -i scf.chk -j input.json
   # equivalently: python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json

Three files are written to the current directory:

``hamiltonian.h5``
   The one-body integrals, the Cholesky-decomposed two-electron integrals and
   the constant (nuclear-repulsion) energy in the molecular-orbital basis. For
   a UHF reference the alpha orbitals define the basis unless ``--ortho-ao``
   is given, in which case an orthogonalised AO basis is used instead.
``wavefunction.h5``
   The trial wavefunction, here the single UHF determinant expressed in the
   same basis. With ``--mcscf`` a multi-determinant (particle-hole) trial is
   extracted from a PySCF MCSCF checkpoint instead.
``input.json``
   A ready-to-run ipie input file pointing at the two HDF5 files.

The generated ``input.json`` looks like this:

.. code-block:: json

   {
       "system": {
           "nup": 5,
           "ndown": 5
       },
       "hamiltonian": {
           "name": "Generic",
           "integrals": "hamiltonian.h5"
       },
       "qmc": {
           "dt": 0.005,
           "nwalkers": 10,
           "nsteps": 25,
           "blocks": 100,
           "batched": true,
           "pop_control_freq": 5,
           "stabilise_freq": 5
       },
       "trial": {
           "filename": "wavefunction.h5"
       },
       "estimators": {
           "filename": "estimates.0.h5"
       }
   }

The ``qmc`` block should be reviewed before running. ``dt`` is the imaginary
time step, ``nsteps`` the number of steps between estimator evaluations and
``blocks`` the number of such blocks. ``nwalkers`` is the number of walkers
**per MPI task**; the total walker population is ``nwalkers`` times the number
of tasks, and a total of roughly 1000 or more walkers is recommended to keep
the population-control bias small. For a four-process run of this example,
``"nwalkers": 160`` is a reasonable choice. Every option, its default and its
aliases are documented in :doc:`input_file`.

The most useful converter options are

=========================== ===============================================
Option                      Meaning (default)
=========================== ===============================================
``-i, --input``             PySCF checkpoint file (required)
``-j, --json-input``        name of the JSON input file (``input.json``)
``--hamiltonian``           name of the integral file (``hamiltonian.h5``)
``--wavefunction``          name of the trial file (``wavefunction.h5``)
``-e, --estimates``         estimator output file (``estimates.0.h5``)
``-t, --thresh``            Cholesky convergence threshold (``1e-5``)
``--frozen-core N``         freeze ``N`` core orbitals
``--mcscf``                 build a multi-determinant trial from an MCSCF chk
``-o, --ortho-ao``          use an orthogonalised AO basis
``--lin-dep``               linear-dependency cutoff for that basis (``0``)
``-v, --verbose``           verbose output
=========================== ===============================================

Run ``pyscf_to_ipie.py --help`` for the full list.

3. Run AFQMC
~~~~~~~~~~~~

The ``ipie`` launcher (``bin/ipie``, copied onto your ``PATH`` by the
``scripts`` list in ``setup.py``; it is not a ``console_scripts`` entry
point) takes the JSON file as its only positional argument. Under MPI each
process propagates its own ``nwalkers`` walkers:

.. code-block:: bash

   mpirun -np 4 ipie input.json > output.dat

If the ``ipie`` script is not on your ``PATH`` (for example when running from
a source checkout without installing), call it through Python instead:

.. code-block:: bash

   mpirun -np 4 python /path/to/ipie/bin/ipie input.json > output.dat

Without ``mpirun`` the same command runs on a single process. Note that
``mpi4py`` (``pip install ipie[mpi]``, see :doc:`installation`) is required
for MPI runs: without it ipie falls back to a serial fake communicator, so
``mpirun -np 4 ipie ...`` starts four independent serial calculations that each
believe they are rank 0 and overwrite each other's ``estimates.0.h5``.
``ipie --gpu input.json`` switches on the CuPy backend; see
:doc:`installation`.

The launcher calls :func:`ipie.qmc.calc.setup_calculation`, which reads the
input on rank 0, broadcasts it, builds an :class:`ipie.qmc.afqmc.AFQMC`
driver and returns it together with the communicator; the launcher then calls
``afqmc.run(verbose=True)`` followed by ``afqmc.finalise(verbose=True)``.
``output.dat`` starts with a summary of the setup (each
line prefixed with ``#``), followed by a table with one row per block. The
energy estimator always contributes the ``ETotal``, ``E1Body`` and ``E2Body``
columns, next to the walker-population diagnostics. The same data, along with
any additional estimators, is written to ``estimates.0.h5``.

4. Analyse the results
~~~~~~~~~~~~~~~~~~~~~~

``reblock.py`` (``tools/reblock.py``, also installed on ``PATH``) estimates
the mean and statistical error of ``ETotal`` from the block data, using the
autocorrelation time to choose the block size. The first blocks belong to the
equilibration phase and must be discarded; ``-b`` gives the number of blocks
to skip:

.. code-block:: bash

   reblock.py -b 10 -f output.dat
   # or read the HDF5 file directly
   reblock.py -b 10 -f estimates.0.h5

It prints a one-row table with the columns ``ETotal_ac`` (mean),
``ETotal_error_ac`` (one-sigma standard error), ``ETotal_nsamp_ac`` (number of
independent blocks), ``ac`` (block size in units of the original blocks) and
``filename`` (the file the row was computed from; one row per file when
several are given).
Deciding how many blocks to discard is not automatic: inspect the ``ETotal``
column of ``output.dat`` and choose ``-b`` such that the discarded part covers
the initial transient.

The same analysis is available from Python:

.. code-block:: python

   from ipie.analysis.extraction import extract_observable
   from ipie.analysis.autocorr import reblock_by_autocorr

   # pandas DataFrame with one row per block; the 'energy' estimator is always present
   data = extract_observable("estimates.0.h5", "energy")
   print(data.head())

   # discard the first 10 blocks and reblock the total energy
   result = reblock_by_autocorr(data["ETotal"].values[10:])
   print(result)

The total energy should agree, within error bars, with the AFQMC value
of -5.3819(6) Hartree from the `Simons hydrogen chain benchmark
<https://github.com/simonsfoundation/hydrogen-benchmark-PRX/blob/master/N_10_OBC/R_1.6/AFQMC_basis-STO>`_
(N = 10, open boundaries, R = 1.6 bohr, STO basis). See :doc:`analysis` for
more on error estimation and extracting other observables.

Python workflow
---------------

The whole calculation can also be scripted, which is convenient for scanning
parameters or adding custom estimators (this is what
``examples/03-custom_observable/run_afqmc.py`` does). The two helpers are
:func:`ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk`, which plays the
role of the converter script,

.. code-block:: python

   def gen_ipie_input_from_pyscf_chk(
       pyscf_chkfile: str,
       hamil_file: str = "hamiltonian.h5",
       wfn_file: str = "wavefunction.h5",
       verbose: bool = True,
       chol_cut: float = 1e-5,
       ortho_ao: bool = False,
       mcscf: bool = False,
       linear_dep_thresh: float = 1e-8,
       num_frozen_core: int = 0,
   ) -> None

and :func:`ipie.qmc.calc.build_afqmc_driver`, which assembles the driver from
those files and a small set of options (everything else takes the input-file
defaults: ``dt`` 0.005, 25 steps per block, **1000 blocks**, stabilisation and
population control every 5 steps):

.. code-block:: python

   def build_afqmc_driver(
       comm,
       nelec: tuple,
       wavefunction_file: str = "wavefunction.h5",
       hamiltonian_file: str = "hamiltonian.h5",
       num_walkers_per_task: int = 10,
       estimator_filename: str = "estimates.0.h5",
       seed: int = None,
       verbosity: int = 0,
   ) -> AFQMC

Putting them together:

.. code-block:: python

   from pyscf import gto, scf

   from ipie.config import MPI
   from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
   from ipie.qmc.calc import build_afqmc_driver
   from ipie.analysis.extraction import extract_observable

   mol = gto.M(
       atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
       basis="sto-6g",
       verbose=4,
       unit="Bohr",
   )
   mf = scf.UHF(mol)
   mf.chkfile = "scf.chk"
   mf.kernel()

   # Write hamiltonian.h5 and wavefunction.h5 from the checkpoint file.
   gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)

   # MPI.COMM_WORLD is a serial FakeComm when mpi4py is not installed.
   comm = MPI.COMM_WORLD

   afqmc = build_afqmc_driver(comm, nelec=mol.nelec, num_walkers_per_task=100)
   print(afqmc.params)            # the QMCParams dataclass with the run settings
   afqmc.params.num_blocks = 100  # the default is 1000 blocks; match the command-line run above
   afqmc.run()
   afqmc.finalise(verbose=True)

   data = extract_observable(afqmc.estimators.filename, "energy")
   print(data[["ETotal", "E1Body", "E2Body"]].tail())

``afqmc.params`` is a :class:`ipie.qmc.options.QMCParams` instance. The
number of blocks (``num_blocks``), the steps per block
(``num_steps_per_block``), the stabilisation frequency (``num_stblz``) and the
population-control frequency (``pop_control_freq``) are read from it inside
``run()``, so they can be changed on the driver before ``run()`` is called as
in the example above. The random seed and the time step, however, are fixed at
construction: the seed is applied in ``AFQMC.__init__`` and the propagator is
built with ``params.timestep`` in :func:`ipie.qmc.calc.get_driver` and never
rebuilt, so editing ``params.rng_seed`` or ``params.timestep`` afterwards has
no effect. Pass the seed through the ``seed`` argument of
:func:`ipie.qmc.calc.build_afqmc_driver`; to change the time step hand a full
options dictionary to :func:`ipie.qmc.calc.get_driver` (``{"qmc": {"dt":
0.01, ...}}``) or use :meth:`ipie.qmc.afqmc.AFQMC.build`, which takes
``timestep`` and ``seed`` keyword arguments. ``run()`` also accepts an ``additional_estimators``
dictionary of :class:`ipie.estimators.estimator_base.EstimatorBase` objects,
which is how ``examples/03-custom_observable`` adds a one-body reduced density
matrix estimator; see :doc:`advanced`.

.. note::

   When such a script is launched with ``mpirun``, every rank executes the
   PySCF calculation and the conversion. That is harmless for a system this
   small, but for production runs generate ``hamiltonian.h5`` and
   ``wavefunction.h5`` once beforehand (or guard that part with
   ``if comm.rank == 0:`` followed by ``comm.barrier()``) and only call
   :func:`ipie.qmc.calc.build_afqmc_driver` on all ranks.

Where to go next
----------------

- :doc:`input_file` lists every JSON option, its default and aliases.
- :doc:`python_api` explains how to build the system, Hamiltonian, trial
  wavefunction, walkers and propagator objects by hand instead of going
  through :func:`ipie.qmc.calc.build_afqmc_driver`.
- :doc:`analysis` covers the analysis tools in more depth.
- :doc:`examples` describes the scripts in the ``examples/`` directory, which
  include multi-determinant trials, frozen-core calculations, GPU runs,
  free-projection and finite-temperature AFQMC.
